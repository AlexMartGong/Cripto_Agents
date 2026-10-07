"""Gasto con pago por uso: la cuota declarada no frena un rol remoto, y el tope en USD sí.

Dos reglas del bloque T7, las dos en `ModelRouter`:

- **`_choose`**: con `billing = payg` el router no le pregunta al contador por un rol remoto. Zen
  no publica límite de peticiones, así que la cifra que quede en `quota_per_window` es un límite
  que el proveedor no impone. Con la suscripción nada cambia, y eso es el control.
- **`_admit`**: lo que frena es `SpendGuard`, el mismo guarda de los sondeos, que se mira antes de
  abrir cada invocación de pago. Un acierto de caché no pregunta y el reintento de una invocación
  ya abierta no se niega.

Cada guarda lleva su mutación: el método de producción reescrito con el fallo dentro, y la misma
comprobación exigida a fallar.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, cast

import pytest

from crypto_agents.cache import InMemoryResponseCache, cache_key
from crypto_agents.llm import (
    Completion,
    InvalidModelOutputError,
    ModelRouter,
    SpendCapReachedError,
    TokenUsage,
    prompt_digest,
)
from crypto_agents.quota import QuotaExhaustedError, QuotaLedger
from crypto_agents.settings import (
    DEFAULT_PRICING,
    ConfigError,
    ModelChoice,
    RoleConfig,
    Settings,
    load_settings,
)
from crypto_agents.spend import SpendGuard, spend_guard, unpriced_roles
from crypto_agents.state import (
    AgentRole,
    Backend,
    Billing,
    Dimension,
    FailureKind,
    StructuredOutputMode,
    TechnicalVerdict,
)
from tests.conftest import role_map, verdict_payload
from tests.test_upstream import mutated_method

if TYPE_CHECKING:
    from collections.abc import Sequence

    from crypto_agents.cache import ResponseCache

NOW = datetime(2026, 10, 7, 12, 0, tzinfo=UTC)

PRICED = "glm-5.3-flash"
"""Un id con fila de pago por uso: 0.15 USD por millón de tokens de entrada."""

DECIDER_MODEL = "glm-5.2"

TOKENS = 1_000_000
CALL_USD = 0.15
"""Lo que cuesta cada llamada del backend falso: un millón de tokens de entrada."""

VERDICT = verdict_payload(Dimension.STRUCTURE)
BROKEN = '{"dimension": "structure"}'


def remote(model: str, quota: int) -> ModelChoice:
    return ModelChoice(
        backend=Backend.OPENAI,
        model=model,
        family="zhipu",
        structured_output=StructuredOutputMode.JSON_SCHEMA,
        quota_per_window=quota,
    )


def local(quota: int = 10_000) -> ModelChoice:
    return ModelChoice(
        backend=Backend.OLLAMA,
        model="qwen3:8b",
        family="qwen",
        structured_output=StructuredOutputMode.JSON_SCHEMA,
        quota_per_window=quota,
    )


def settings_for(billing: Billing, quota: int = 1, structure: str = PRICED) -> Settings:
    """Tres roles que importan, los tres con la cuota mínima.

    `structure` es remoto con respaldo local, `decider` remoto sin respaldo —`Settings` no le
    admite ninguno— y `volume` tiene el primario local, como en un brazo `local_*`.
    """
    roles = role_map()
    roles[AgentRole.STRUCTURE] = RoleConfig(primary=remote(structure, quota), fallback=local())
    roles[AgentRole.DECIDER] = RoleConfig(primary=remote(DECIDER_MODEL, quota))
    roles[AgentRole.VOLUME] = RoleConfig(primary=local(quota))
    return load_settings(
        roles=roles,
        openai={"api_key": "sk-test"},
        ollama={"host": "http://localhost:11434"},
        billing=billing,
    )


class Clock:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now


class Scripted:
    """Backend que contesta lo que se le dicte y declara siempre el mismo uso."""

    def __init__(self, texts: Sequence[str] = ()) -> None:
        self._texts = list(texts)
        self.asked: list[tuple[str, str]] = []

    async def complete(self, choice: ModelChoice, prompt: str, schema: type) -> Completion:
        del schema
        self.asked.append((choice.model, prompt))
        text = self._texts.pop(0) if self._texts else VERDICT
        return Completion(
            text=text,
            usage=TokenUsage(prompt_tokens=TOKENS, cached_tokens=0, completion_tokens=0),
        )


def router_for(
    settings: Settings,
    backend: Scripted,
    guard: SpendGuard | None = None,
    cache: ResponseCache | None = None,
    router: type[ModelRouter] = ModelRouter,
) -> tuple[ModelRouter, QuotaLedger]:
    clock = Clock()
    ledger = QuotaLedger(settings.quota_window, clock)
    built = router(
        settings,
        ledger,
        {Backend.OPENAI: backend, Backend.OLLAMA: backend},
        clock,
        cache,
        guard=guard,
    )
    return built, ledger


def mutant(method: str, old: str, new: str) -> type[ModelRouter]:
    """`ModelRouter` con ese método reescrito: el código de producción con el fallo dentro."""
    rewritten = mutated_method(getattr(ModelRouter, method), old, new, {})
    return cast("type[ModelRouter]", type("Mutante", (ModelRouter,), {method: rewritten}))


# ─────────────────────────── La cuota declarada con pago por uso ──────────────────────────────────


async def assert_payg_never_brakes_a_remote_role(router: type[ModelRouter] = ModelRouter) -> None:
    """Tres llamadas con cuota 1: el rol remoto sigue en su primario y el decisor no aborta."""
    backend = Scripted()
    built, ledger = router_for(settings_for(Billing.PAYG), backend, router=router)

    for prompt in ("uno", "dos", "tres"):
        _, calls = await built.invoke(AgentRole.STRUCTURE, prompt, TechnicalVerdict)
        assert [call.backend for call in calls] == [Backend.OPENAI], (
            "con pago por uso un rol remoto degradó al respaldo local por una cuota que Zen "
            "no publica"
        )
    for prompt in ("uno", "dos", "tres"):
        _, calls = await built.invoke(AgentRole.DECIDER, prompt, TechnicalVerdict)
        assert [call.model for call in calls] == [DECIDER_MODEL]

    assert [model for model, _ in backend.asked] == [PRICED] * 3 + [DECIDER_MODEL] * 3
    assert ledger.used(AgentRole.STRUCTURE, PRICED) == 3.0, "el contador dejó de registrar"


@pytest.mark.asyncio
async def test_under_payg_a_remote_role_out_of_quota_is_neither_degraded_nor_aborted() -> None:
    await assert_payg_never_brakes_a_remote_role()


@pytest.mark.asyncio
async def test_mutation_applying_the_quota_under_payg_is_caught() -> None:
    """La mutación: `_choose` le pregunta al contador también con pago por uso."""
    applying = mutant(
        "_choose",
        "if self._settings.billing is Billing.PAYG and primary.backend not in LOCAL_BACKENDS:",
        "if False:",
    )
    with pytest.raises(AssertionError, match="degradó al respaldo local"):
        await assert_payg_never_brakes_a_remote_role(applying)


@pytest.mark.asyncio
async def test_under_the_subscription_the_same_quotas_still_degrade_and_abort() -> None:
    """El control: con `go` nada cambia. La segunda llamada ya no cabe en la ventana."""
    backend = Scripted()
    built, _ = router_for(settings_for(Billing.GO), backend)

    _, first = await built.invoke(AgentRole.STRUCTURE, "uno", TechnicalVerdict)
    _, second = await built.invoke(AgentRole.STRUCTURE, "dos", TechnicalVerdict)
    assert [call.backend for call in first] == [Backend.OPENAI]
    assert [call.backend for call in second] == [Backend.OLLAMA]

    await built.invoke(AgentRole.DECIDER, "uno", TechnicalVerdict)
    with pytest.raises(QuotaExhaustedError):
        await built.invoke(AgentRole.DECIDER, "dos", TechnicalVerdict)


@pytest.mark.asyncio
async def test_under_payg_a_local_primary_still_goes_through_the_ledger() -> None:
    """Lo que deja de frenar es la cuota de un proveedor remoto, no el contador entero.

    Un brazo local pone el respaldo de primario: su cuota es un centinela nuestro y sigue
    contando igual con cualquier forma de pago.
    """
    built, _ = router_for(settings_for(Billing.PAYG), Scripted())

    await built.invoke(AgentRole.VOLUME, "uno", TechnicalVerdict)
    with pytest.raises(QuotaExhaustedError):
        await built.invoke(AgentRole.VOLUME, "dos", TechnicalVerdict)


# ───────────────────────────────────── El tope de gasto ───────────────────────────────────────────


async def assert_the_cap_stops_new_invocations(router: type[ModelRouter] = ModelRouter) -> None:
    """Con 0.25 USD de tope y 0.15 por llamada: dos invocaciones se abren y la tercera no.

    La segunda se abre con 0.15 gastados y termina en 0.30: es la que cruza el tope, y termina.
    No hay cota antes de llamar, así que «no pasarse» no se puede prometer; lo que se promete es
    que con el tope alcanzado no se abre ninguna más.
    """
    backend = Scripted()
    guard = SpendGuard(0.25)
    built, _ = router_for(settings_for(Billing.PAYG), backend, guard, router=router)

    await built.invoke(AgentRole.STRUCTURE, "uno", TechnicalVerdict)
    assert guard.spent_usd == pytest.approx(CALL_USD)
    await built.invoke(AgentRole.STRUCTURE, "dos", TechnicalVerdict)
    assert guard.spent_usd == pytest.approx(2 * CALL_USD)

    try:
        await built.invoke(AgentRole.STRUCTURE, "tres", TechnicalVerdict)
    except SpendCapReachedError as error:
        assert error.calls == (), "no se llamó a nadie: no hay intento que traer"
        assert error.cap_usd == 0.25
        assert error.spent_usd == pytest.approx(2 * CALL_USD)
    else:
        raise AssertionError("se abrió una invocación con el tope de gasto ya alcanzado")

    assert len(backend.asked) == 2, "el proveedor vio una petición que el tope debía parar"
    assert guard.refused == ["structure"]
    assert guard.spent_usd == pytest.approx(2 * CALL_USD), "lo negado no gasta"


@pytest.mark.asyncio
async def test_with_the_cap_reached_no_new_invocation_is_opened() -> None:
    await assert_the_cap_stops_new_invocations()


@pytest.mark.asyncio
async def test_mutation_a_router_that_does_not_ask_the_guard_is_caught() -> None:
    """La mutación: `_admit` deja pasar siempre."""
    deaf = mutant("_admit", "if self._guard is None or self._guard.allow(role.value):", "if True:")
    with pytest.raises(AssertionError, match="con el tope de gasto ya alcanzado"):
        await assert_the_cap_stops_new_invocations(deaf)


async def assert_a_cache_hit_passes_with_the_cap_reached(
    router: type[ModelRouter] = ModelRouter,
) -> None:
    """Lo ya pagado se vuelve a leer aunque no quede tope: es lo que hace posible reanudar."""
    cache = InMemoryResponseCache()
    settings = settings_for(Billing.PAYG)
    filling, _ = router_for(settings, Scripted(), cache=cache)
    await filling.invoke(AgentRole.STRUCTURE, "ya pagado", TechnicalVerdict)

    backend = Scripted()
    guard = SpendGuard(0.01)
    guard.add((await filling.invoke(AgentRole.DECIDER, "otra", TechnicalVerdict))[1])
    assert guard.spent_usd >= guard.cap_usd
    built, _ = router_for(settings, backend, guard, cache, router)

    try:
        _, calls = await built.invoke(AgentRole.STRUCTURE, "ya pagado", TechnicalVerdict)
    except SpendCapReachedError as error:
        raise AssertionError("el tope cortó un acierto de caché, que no gasta") from error

    assert [call.cache_hit for call in calls] == [True]
    assert backend.asked == []
    assert guard.refused == [], "un acierto de caché no le pregunta al tope"
    with pytest.raises(SpendCapReachedError):
        await built.invoke(AgentRole.STRUCTURE, "sin pagar", TechnicalVerdict)


@pytest.mark.asyncio
async def test_a_cache_hit_passes_with_the_cap_reached() -> None:
    await assert_a_cache_hit_passes_with_the_cap_reached()


@pytest.mark.asyncio
async def test_mutation_asking_the_guard_before_reading_the_cache_is_caught() -> None:
    """La mutación: el tope se mira antes de la caché, y una reanudación caliente se corta."""
    eager = mutant(
        "invoke",
        "        replayed = self._replay_attempt(choice, digest, schema, check)\n"
        "        if replayed is None:",
        "        self._admit(role, calls)\n"
        "        replayed = self._replay_attempt(choice, digest, schema, check)\n"
        "        if replayed is None:",
    )
    with pytest.raises(AssertionError, match="cortó un acierto de caché"):
        await assert_a_cache_hit_passes_with_the_cap_reached(eager)


@pytest.mark.asyncio
async def test_the_retry_of_an_open_invocation_is_not_refused() -> None:
    """El intento inválido ya se pagó; negarle la corrección es perder lo pagado para nada."""
    backend = Scripted([BROKEN, VERDICT])
    guard = SpendGuard(0.10)
    built, _ = router_for(settings_for(Billing.PAYG), backend, guard)

    _, calls = await built.invoke(AgentRole.STRUCTURE, "uno", TechnicalVerdict)

    assert [call.valid for call in calls] == [False, True]
    assert guard.spent_usd == pytest.approx(2 * CALL_USD), "los dos intentos entran en el gasto"
    assert guard.refused == []


@pytest.mark.asyncio
async def test_a_cut_invocation_carries_the_attempts_it_replayed_from_the_cache() -> None:
    """Un intento inválido reproducido de la caché y, al ir a llamar, el tope: la fila no se pierde.

    Es por lo que la excepción hereda de `ModelInvocationError`: el nodo copia sus intentos al
    estado, y sin ellos el journal de la evaluación cortada no diría qué se había reproducido.
    """
    cache = InMemoryResponseCache()
    settings = settings_for(Billing.PAYG)
    first, _ = router_for(settings, Scripted([BROKEN, BROKEN]), cache=cache)
    with pytest.raises(InvalidModelOutputError):
        await first.invoke(AgentRole.STRUCTURE, "uno", TechnicalVerdict)

    # Solo el primer intento: el del reintento no está, así que hay que ir a llamar.
    key = cache_key(
        Backend.OPENAI,
        PRICED,
        prompt_digest("uno"),
        TechnicalVerdict,
        StructuredOutputMode.JSON_SCHEMA,
    )
    stored = cache.get(key)
    assert stored is not None
    only_the_first = InMemoryResponseCache()
    only_the_first.set(key, stored)

    guard = SpendGuard(0.01)
    spent, _ = router_for(settings, Scripted(), cache=InMemoryResponseCache())
    guard.add((await spent.invoke(AgentRole.DECIDER, "otra", TechnicalVerdict))[1])
    backend = Scripted()
    built, _ = router_for(settings, backend, guard, only_the_first)

    with pytest.raises(SpendCapReachedError) as cut:
        await built.invoke(AgentRole.STRUCTURE, "uno", TechnicalVerdict)

    (replayed,) = cut.value.calls
    assert replayed.cache_hit
    assert replayed.failure_kind is FailureKind.SCHEMA
    assert backend.asked == []


@pytest.mark.asyncio
async def test_a_local_call_never_asks_the_guard() -> None:
    """Una llamada local es gratis por regla: con el tope alcanzado sigue pasando."""
    guard = SpendGuard(0.01)
    settings = settings_for(Billing.PAYG, quota=10)
    paying, _ = router_for(settings, Scripted())
    guard.add((await paying.invoke(AgentRole.DECIDER, "otra", TechnicalVerdict))[1])
    built, _ = router_for(settings, Scripted(), guard)

    _, calls = await built.invoke(AgentRole.VOLUME, "uno", TechnicalVerdict)

    assert [call.backend for call in calls] == [Backend.OLLAMA]
    assert guard.refused == []


@pytest.mark.asyncio
async def test_without_a_guard_nothing_is_capped() -> None:
    backend = Scripted()
    built, _ = router_for(settings_for(Billing.PAYG), backend)

    for prompt in ("uno", "dos", "tres"):
        await built.invoke(AgentRole.STRUCTURE, prompt, TechnicalVerdict)

    assert len(backend.asked) == 3


# ─────────────────────────────── Quién puede construir el guarda ──────────────────────────────────


def test_a_role_whose_model_has_no_payg_price_is_named() -> None:
    """`mimo-v2.5` es un id de Go: en pago por uso no existe y no tiene fila de precio."""
    settings = settings_for(Billing.PAYG, structure="mimo-v2.5")
    assert DEFAULT_PRICING.price_for("mimo-v2.5", Billing.PAYG, NOW) is None

    assert unpriced_roles(settings, NOW) == ((AgentRole.STRUCTURE, "mimo-v2.5"),)
    assert unpriced_roles(settings, NOW, [AgentRole.DECIDER, AgentRole.VOLUME]) == ()


def test_a_local_primary_is_never_unpriced() -> None:
    """Un primario local no gasta: que su id no tenga precio no deja ciego al tope."""
    assert unpriced_roles(settings_for(Billing.PAYG), NOW) == ()


def test_no_cap_declared_is_no_guard() -> None:
    assert spend_guard(settings_for(Billing.GO), None, ()) is None


def test_the_guard_carries_the_cap_it_was_given() -> None:
    guard = spend_guard(settings_for(Billing.PAYG), 2.5, ())
    assert guard is not None
    assert guard.cap_usd == 2.5
    assert guard.spent_usd == 0.0


def test_a_cap_in_dollars_is_refused_under_the_subscription() -> None:
    with pytest.raises(ConfigError, match="CA_BILLING=go"):
        spend_guard(settings_for(Billing.GO), 1.0, ())


def test_a_cap_that_cannot_price_a_role_is_refused_naming_it() -> None:
    settings = settings_for(Billing.PAYG, structure="mimo-v2.5")
    with pytest.raises(ConfigError, match=re.escape("structure → mimo-v2.5")):
        spend_guard(settings, 1.0, unpriced_roles(settings, NOW))


@pytest.mark.parametrize("cap", [0.0, -1.0])
def test_a_cap_that_is_not_positive_is_refused(cap: float) -> None:
    with pytest.raises(ConfigError, match="positivo"):
        spend_guard(settings_for(Billing.PAYG), cap, ())
