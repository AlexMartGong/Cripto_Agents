"""Pruebas del router de modelos.

Verifican lo que hace intercambiable el backend y medible el presupuesto: el
router resuelve por cuota, consulta la caché, valida la salida, reintenta con el
error adjunto y deja rastro de cada intento.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from langchain_core.messages import AIMessage

from crypto_agents.cache import InMemoryResponseCache, cache_key
from crypto_agents.llm import (
    BackendNotCalledError,
    InvalidModelOutputError,
    ModelCallError,
    ModelRouter,
    build_backends,
    prompt_digest,
    raw_text,
)
from crypto_agents.quota import QuotaExhaustedError, QuotaLedger
from crypto_agents.settings import Backend, ModelChoice, RoleConfig, Settings, load_settings
from crypto_agents.state import (
    AgentRole,
    Bias,
    Dimension,
    FailureKind,
    Observation,
    TechnicalVerdict,
)
from tests.conftest import CHEAP, role_map

START = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)
SCARCE = ModelChoice(
    backend=Backend.OPENAI, model="gpt-x", family="gpt", quota_weight=2.0, quota_per_window=2
)


def verdict_payload(dimension: Dimension = Dimension.STRUCTURE) -> str:
    """JSON válido para `TechnicalVerdict`, como lo devolvería un modelo."""
    return TechnicalVerdict(
        dimension=dimension,
        bias=Bias.BULLISH,
        confidence=0.6,
        observations=[
            Observation(
                id=f"{dimension.value}-1",
                text="El precio sostiene el soporte previo y marca un maximo superior.",
                cites=["EMA_50"],
                supports=Bias.BULLISH,
            )
        ],
        invalidation="Pierde el soporte de 63000.",
    ).model_dump_json()


class FakeClock:
    """Reloj controlado por la prueba."""

    def __init__(self, start: datetime = START) -> None:
        self.now = start

    def __call__(self) -> datetime:
        """Instante actual según la prueba."""
        return self.now

    def advance(self, delta: timedelta) -> None:
        """Mueve el reloj hacia adelante."""
        self.now += delta


class ScriptedBackend:
    """Devuelve payloads fijos en orden y anota qué prompt recibió."""

    def __init__(self, *payloads: str) -> None:
        self.payloads = list(payloads)
        self.seen: list[tuple[str, str]] = []

    async def complete(self, choice: ModelChoice, prompt: str, schema: type) -> str:
        """Entrega el siguiente payload del guion, repitiendo el último si se agota."""
        self.seen.append((choice.model, prompt))
        index = min(len(self.seen) - 1, len(self.payloads) - 1)
        return self.payloads[index]


class FailingBackend:
    """Falla como falla un proveedor: antes de que exista contenido que validar."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.seen: list[tuple[str, str]] = []

    async def complete(self, choice: ModelChoice, prompt: str, schema: type) -> str:
        """Registra el intento y lanza."""
        self.seen.append((choice.model, prompt))
        raise self.error


def make_settings(primary: ModelChoice, fallback: ModelChoice | None = None) -> Settings:
    """Configuración con el par primario/respaldo indicado."""
    return load_settings(
        roles=role_map(primary, fallback),
        openai={"api_key": "sk-test"},
        ollama={"host": "http://localhost:11434"},
    )


def make_router(
    settings: Settings,
    clock: FakeClock,
    *,
    payloads: tuple[str, ...] = (),
    cache: InMemoryResponseCache | None = None,
    max_attempts: int = 2,
) -> tuple[ModelRouter, QuotaLedger, dict[Backend, ScriptedBackend]]:
    """Router con backends falsos para ambos proveedores."""
    scripted = payloads or (verdict_payload(),)
    backends = {
        Backend.OPENAI: ScriptedBackend(*scripted),
        Backend.OLLAMA: ScriptedBackend(*scripted),
    }
    ledger = QuotaLedger(settings, clock)
    router = ModelRouter(settings, ledger, backends, clock, cache, max_attempts)
    return router, ledger, backends


# ──────────────────────────────────────── Camino feliz ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_invoke_validates_and_records_the_call() -> None:
    """Toda llamada deja un `LLMCall`: no hay forma de gastar cuota sin rastro."""
    clock = FakeClock()
    router, ledger, _ = make_router(make_settings(SCARCE), clock)

    verdict, calls = await router.invoke(AgentRole.STRUCTURE, "analiza", TechnicalVerdict)

    assert verdict.dimension is Dimension.STRUCTURE
    assert len(calls) == 1
    assert calls[0].model == "gpt-x"
    assert calls[0].quota_weight == 2.0
    assert calls[0].prompt_digest == prompt_digest("analiza")
    assert calls[0].at == START
    assert ledger.used(AgentRole.STRUCTURE, "gpt-x") == 2.0


@pytest.mark.asyncio
async def test_router_dispatches_to_the_backend_of_the_resolved_model() -> None:
    """El rol no elige proveedor: lo elige el modelo que resolvió el presupuesto."""
    clock = FakeClock()
    router, _, backends = make_router(make_settings(CHEAP), clock)

    await router.invoke(AgentRole.BULL, "argumenta", TechnicalVerdict)

    assert [model for model, _ in backends[Backend.OLLAMA].seen] == ["qwen3:8b"]
    assert backends[Backend.OPENAI].seen == []


# ────────────────────────────────────────── Cuota ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_router_degrades_to_fallback_backend_when_quota_runs_out() -> None:
    """Agotado el primario, la siguiente llamada sale por el proveedor del respaldo.

    Sobre un rol técnico: el decisor no tiene respaldo que pueda atenderle.
    """
    clock = FakeClock()
    router, _, backends = make_router(make_settings(SCARCE, CHEAP), clock)

    await router.invoke(AgentRole.STRUCTURE, "primera", TechnicalVerdict)
    await router.invoke(AgentRole.STRUCTURE, "segunda", TechnicalVerdict)

    assert [model for model, _ in backends[Backend.OPENAI].seen] == ["gpt-x"]
    assert [model for model, _ in backends[Backend.OLLAMA].seen] == ["qwen3:8b"]


@pytest.mark.asyncio
async def test_router_raises_before_spending_when_quota_is_exhausted() -> None:
    """`QuotaExhaustedError` se lanza antes de tocar el backend: no se gasta nada."""
    clock = FakeClock()
    router, _, backends = make_router(make_settings(SCARCE), clock)

    await router.invoke(AgentRole.DECIDER, "primera", TechnicalVerdict)
    with pytest.raises(QuotaExhaustedError):
        await router.invoke(AgentRole.DECIDER, "segunda", TechnicalVerdict)

    assert len(backends[Backend.OPENAI].seen) == 1


@pytest.mark.asyncio
async def test_router_reports_a_missing_backend_by_name() -> None:
    """Un backend declarado pero no construido debe fallar señalando el rol."""
    clock = FakeClock()
    settings = make_settings(CHEAP)
    router = ModelRouter(settings, QuotaLedger(settings, clock), {}, clock)

    with pytest.raises(LookupError, match="ollama"):
        await router.invoke(AgentRole.VOLUME, "analiza", TechnicalVerdict)


# ─────────────────────────────────────────── Reintento ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_router_retries_after_a_validation_failure() -> None:
    """Una salida que no valida se reintenta una vez y la segunda se acepta."""
    clock = FakeClock()
    router, _, backends = make_router(
        make_settings(CHEAP), clock, payloads=('{"dimension": "structure"}', verdict_payload())
    )

    verdict, calls = await router.invoke(AgentRole.STRUCTURE, "analiza", TechnicalVerdict)

    assert verdict.confidence == 0.6
    assert len(calls) == 2
    assert len(backends[Backend.OLLAMA].seen) == 2


@pytest.mark.asyncio
async def test_retry_prompt_carries_the_validation_error() -> None:
    """El reintento adjunta el error; si no, el digest sería el mismo y la caché lo repetiría."""
    clock = FakeClock()
    router, _, backends = make_router(
        make_settings(CHEAP), clock, payloads=('{"dimension": "structure"}', verdict_payload())
    )

    await router.invoke(AgentRole.STRUCTURE, "analiza", TechnicalVerdict)

    first_prompt, second_prompt = (prompt for _, prompt in backends[Backend.OLLAMA].seen)
    assert first_prompt != second_prompt
    assert "no pasó la validación" in second_prompt
    assert "confidence" in second_prompt


@pytest.mark.asyncio
async def test_every_attempt_consumes_quota() -> None:
    """Un intento fallido ya se pagó en el proveedor: cuenta igual."""
    clock = FakeClock()
    router, ledger, _ = make_router(
        make_settings(CHEAP), clock, payloads=('{"dimension": "structure"}', verdict_payload())
    )

    _, calls = await router.invoke(AgentRole.STRUCTURE, "analiza", TechnicalVerdict)

    assert [call.cache_hit for call in calls] == [False, False]
    assert ledger.used(AgentRole.STRUCTURE, "qwen3:8b") == 2.0


@pytest.mark.asyncio
async def test_each_attempt_records_its_backend_and_whether_it_validated() -> None:
    """Un veredicto puede empezar remoto y acabar local: cada intento se firma solo.

    El primer intento agota el presupuesto del primario y no valida; el reintento
    lo atiende ya el respaldo. Sin `backend` y `valid` por intento, el journal
    guardaría dos llamadas indistinguibles y no habría forma de ver que la salida
    buena la produjo el modelo pequeño.
    """
    clock = FakeClock()
    settings = make_settings(SCARCE, CHEAP)
    backends = {
        Backend.OPENAI: ScriptedBackend('{"dimension": "structure"}'),  # remoto: no valida
        Backend.OLLAMA: ScriptedBackend(verdict_payload()),  # respaldo local: sí valida
    }
    router = ModelRouter(settings, QuotaLedger(settings, clock), backends, clock)

    _, calls = await router.invoke(AgentRole.STRUCTURE, "analiza", TechnicalVerdict)

    assert [call.backend for call in calls] == [Backend.OPENAI, Backend.OLLAMA]
    assert [call.valid for call in calls] == [False, True]
    assert [call.model for call in calls] == ["gpt-x", "qwen3:8b"]


@pytest.mark.asyncio
async def test_router_gives_up_after_the_attempt_budget() -> None:
    """Agotados los intentos falla explícitamente en vez de devolver basura."""
    clock = FakeClock()
    router, _, _ = make_router(make_settings(CHEAP), clock, payloads=("{}",))

    with pytest.raises(InvalidModelOutputError) as excinfo:
        await router.invoke(AgentRole.MOMENTUM, "analiza", TechnicalVerdict)
    assert excinfo.value.role is AgentRole.MOMENTUM
    assert excinfo.value.attempts == 2


@pytest.mark.asyncio
async def test_single_attempt_configuration_does_not_retry() -> None:
    """`max_attempts=1` significa una llamada y punto."""
    clock = FakeClock()
    router, _, backends = make_router(make_settings(CHEAP), clock, payloads=("{}",), max_attempts=1)

    with pytest.raises(InvalidModelOutputError):
        await router.invoke(AgentRole.VOLUME, "analiza", TechnicalVerdict)
    assert len(backends[Backend.OLLAMA].seen) == 1


# ─────────────────────────────────────────── Transporte ──────────────────────────────────────────
# Un 400 del gateway, un DNS que no resuelve o una credencial caducada fallan
# antes de que exista salida que validar. Es el caso en el que el registro más
# falta y el que menos se ejerció: hasta ahora la llamada iba fuera del `try`, así
# que el proveedor cobraba y no quedaba fila.


@pytest.mark.asyncio
async def test_a_transport_failure_still_records_its_call_with_the_cause() -> None:
    """La petición salió: hay `LLMCall`, marcada como transporte y con el error dentro.

    La regla 4 no admite excepción por que el proveedor se porte mal — es
    exactamente cuando hace falta el registro para saber a quién reclamar.
    """
    clock = FakeClock()
    settings = make_settings(CHEAP)
    backend = FailingBackend(RuntimeError("400 model not found: glm-9.9"))
    ledger = QuotaLedger(settings, clock)
    router = ModelRouter(settings, ledger, {Backend.OLLAMA: backend}, clock)

    with pytest.raises(ModelCallError) as excinfo:
        await router.invoke(AgentRole.STRUCTURE, "analiza", TechnicalVerdict)

    (call,) = excinfo.value.calls
    assert call.valid is False
    assert call.failure is not None
    assert call.failure.kind is FailureKind.TRANSPORT
    assert "400 model not found" in call.failure.message
    assert call.prompt_digest == prompt_digest("analiza")
    assert ledger.used(AgentRole.STRUCTURE, "qwen3:8b") == 1.0


@pytest.mark.asyncio
async def test_a_transport_failure_is_not_retried() -> None:
    """No hay salida que corregir, así que reintentar solo gasta otra llamada.

    La plantilla de reintento adjunta un error de validación. Contra un 400 el
    segundo intento es idéntico al primero con texto de más.
    """
    clock = FakeClock()
    settings = make_settings(CHEAP)
    backend = FailingBackend(RuntimeError("connection reset"))
    router = ModelRouter(settings, QuotaLedger(settings, clock), {Backend.OLLAMA: backend}, clock)

    with pytest.raises(ModelCallError):
        await router.invoke(AgentRole.STRUCTURE, "analiza", TechnicalVerdict)

    assert len(backend.seen) == 1


@pytest.mark.asyncio
async def test_a_transport_failure_names_the_role_and_the_model() -> None:
    """El mensaje debe decir qué variable cambiar, no dejar un traceback del cliente."""
    clock = FakeClock()
    settings = make_settings(CHEAP)
    backend = FailingBackend(RuntimeError("401 unauthorized"))
    router = ModelRouter(settings, QuotaLedger(settings, clock), {Backend.OLLAMA: backend}, clock)

    with pytest.raises(ModelCallError) as excinfo:
        await router.invoke(AgentRole.VOLUME, "analiza", TechnicalVerdict)

    assert excinfo.value.role is AgentRole.VOLUME
    assert excinfo.value.model == "qwen3:8b"
    assert "volume" in str(excinfo.value)
    assert "401 unauthorized" in str(excinfo.value)


@pytest.mark.asyncio
async def test_a_backend_that_never_called_is_not_disguised_as_transport() -> None:
    """Un hueco de caché en un replay no es un fallo del proveedor.

    `CacheOnlyBackend` levanta `BackendNotCalledError` para decir que se negó a
    llamar. Envolverla en `ModelCallError` mandaría a revisar la red cuando lo
    que falta es rellenar la caché, y borraría el mensaje que nombra el modelo,
    el esquema y el prompt. Tampoco deja `LLMCall`: no hubo llamada.
    """
    clock = FakeClock()
    settings = make_settings(CHEAP)
    backend = FailingBackend(BackendNotCalledError("falta en caché: modelo qwen3:8b"))
    ledger = QuotaLedger(settings, clock)
    router = ModelRouter(settings, ledger, {Backend.OLLAMA: backend}, clock)

    with pytest.raises(BackendNotCalledError):
        await router.invoke(AgentRole.STRUCTURE, "analiza", TechnicalVerdict)

    assert ledger.used(AgentRole.STRUCTURE, "qwen3:8b") == 0.0


@pytest.mark.asyncio
async def test_exhausted_attempts_carry_the_calls_they_paid_for() -> None:
    """Agotar los intentos no puede perder las filas: el nodo las escribe al journal.

    El contador de cuota ya las tiene, pero vive en memoria. Si no viajan dentro
    de la excepción, la evaluación que se cayó por un modelo que no valida queda
    en el archivo con `calls` vacío, indistinguible de una que no llamó a nadie.
    """
    clock = FakeClock()
    router, _, _ = make_router(make_settings(CHEAP), clock, payloads=("{}",))

    with pytest.raises(InvalidModelOutputError) as excinfo:
        await router.invoke(AgentRole.MOMENTUM, "analiza", TechnicalVerdict)

    calls = excinfo.value.calls
    assert len(calls) == 2
    assert [call.valid for call in calls] == [False, False]
    assert all(
        call.failure is not None and call.failure.kind is FailureKind.VALIDATION for call in calls
    )


# ──────────────────────────────────── Texto crudo del adaptador ───────────────────────────────────
# `with_structured_output` usa `function_calling` por defecto, y con ese método el
# mensaje llega con `content` vacío: lo que el modelo dijo está en la llamada a
# herramienta. Leer solo `content` devolvía cadena vacía y el router reintentaba
# contra un error que no describía nada de lo ocurrido.


def test_raw_text_prefers_the_message_content_when_there_is_any() -> None:
    """Si el modelo respondió en texto, eso es lo más crudo que hay."""
    result = {"raw": AIMessage(content='{"dimension": "structure"}'), "parsed": None}
    assert raw_text(result) == '{"dimension": "structure"}'


def test_raw_text_falls_back_to_the_tool_call_arguments() -> None:
    """El caso real: `content` vacío y la respuesta dentro del tool call."""
    message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "TechnicalVerdict",
                "args": {"dimension": "structure", "bias": "bullish"},
                "id": "call_1",
                "type": "tool_call",
            }
        ],
    )

    recovered = raw_text({"raw": message, "parsed": None})

    assert "structure" in recovered
    assert "bullish" in recovered


def test_raw_text_recovers_arguments_that_were_not_even_json() -> None:
    """Cuando ni los argumentos parsean, el texto sigue siendo lo que hay que enseñar."""
    message = AIMessage(
        content="",
        invalid_tool_calls=[
            {
                "name": "TechnicalVerdict",
                "args": '{"dimension": "struc',
                "id": "call_1",
                "error": "unterminated string",
                "type": "invalid_tool_call",
            }
        ],
    )

    assert raw_text({"raw": message, "parsed": None}) == '{"dimension": "struc'


def test_raw_text_gives_up_with_an_empty_string() -> None:
    """Sin nada que rescatar devuelve vacío, que el router trata como salida inválida."""
    assert raw_text({"raw": AIMessage(content=""), "parsed": None}) == ""
    assert raw_text(None) == ""


# ──────────────────────────────────────────── Caché ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_second_identical_call_hits_the_cache() -> None:
    """El mismo input no gasta cuota dos veces."""
    clock = FakeClock()
    cache = InMemoryResponseCache()
    router, ledger, backends = make_router(make_settings(CHEAP), clock, cache=cache)

    await router.invoke(AgentRole.STRUCTURE, "analiza", TechnicalVerdict)
    _, calls = await router.invoke(AgentRole.STRUCTURE, "analiza", TechnicalVerdict)

    assert calls[0].cache_hit is True
    assert len(backends[Backend.OLLAMA].seen) == 1
    assert ledger.used(AgentRole.STRUCTURE, "qwen3:8b") == 1.0


@pytest.mark.asyncio
async def test_cache_hit_still_leaves_a_record() -> None:
    """Un acierto de caché no consume cuota, pero sí queda registrado."""
    clock = FakeClock()
    router, _, _ = make_router(make_settings(CHEAP), clock, cache=InMemoryResponseCache())

    await router.invoke(AgentRole.STRUCTURE, "analiza", TechnicalVerdict)
    _, calls = await router.invoke(AgentRole.STRUCTURE, "analiza", TechnicalVerdict)

    assert len(calls) == 1
    assert calls[0].latency_ms == 0.0


@pytest.mark.asyncio
async def test_a_cached_answer_survives_an_exhausted_quota() -> None:
    """Con la caché caliente no se consulta el presupuesto: no se va a llamar a nadie.

    `resolve()` lanza cuando no queda cuota, así que mirar la caché después de
    resolver haría fallar un replay sobre respuestas ya guardadas. Es justo lo que
    necesita una ablación: reejecutar sin volver a pagar.
    """
    clock = FakeClock()
    cache = InMemoryResponseCache()
    router, ledger, backends = make_router(make_settings(SCARCE), clock, cache=cache)

    await router.invoke(AgentRole.STRUCTURE, "analiza", TechnicalVerdict)
    assert ledger.remaining(AgentRole.STRUCTURE, SCARCE) < SCARCE.quota_weight

    verdict, calls = await router.invoke(AgentRole.STRUCTURE, "analiza", TechnicalVerdict)

    assert verdict.confidence == 0.6
    assert calls[0].cache_hit is True
    assert len(backends[Backend.OPENAI].seen) == 1


@pytest.mark.asyncio
async def test_a_different_prompt_misses_the_cache() -> None:
    """La clave depende del prompt: otra pregunta es otra llamada."""
    clock = FakeClock()
    router, _, backends = make_router(make_settings(CHEAP), clock, cache=InMemoryResponseCache())

    await router.invoke(AgentRole.STRUCTURE, "analiza", TechnicalVerdict)
    await router.invoke(AgentRole.STRUCTURE, "analiza de nuevo", TechnicalVerdict)

    assert len(backends[Backend.OLLAMA].seen) == 2


@pytest.mark.asyncio
async def test_stale_cache_entry_is_discarded_instead_of_used() -> None:
    """Una entrada que ya no valida contra el esquema se trata como fallo de caché."""
    clock = FakeClock()
    cache = InMemoryResponseCache()
    cache.set(cache_key("qwen3:8b", prompt_digest("analiza"), TechnicalVerdict), '{"roto": 1}')
    router, _, backends = make_router(make_settings(CHEAP), clock, cache=cache)

    verdict, calls = await router.invoke(AgentRole.STRUCTURE, "analiza", TechnicalVerdict)

    assert verdict.confidence == 0.6
    assert calls[0].cache_hit is False
    assert len(backends[Backend.OLLAMA].seen) == 1


# ─────────────────────────────────────────── Construcción ─────────────────────────────────────────


def test_build_backends_only_creates_what_is_configured() -> None:
    """Sin credenciales de OpenAI no se instancia su cliente."""
    settings = load_settings(
        roles={role: RoleConfig(primary=CHEAP) for role in AgentRole}
        | {
            AgentRole.BEAR: RoleConfig(primary=CHEAP.model_copy(update={"family": "llama"})),
        },
        ollama={"host": "http://localhost:11434"},
    )
    assert set(build_backends(settings)) == {Backend.OLLAMA}


def test_router_rejects_a_zero_attempt_budget() -> None:
    """Cero intentos no es una configuración válida."""
    clock = FakeClock()
    settings = make_settings(CHEAP)
    with pytest.raises(ValueError, match="max_attempts"):
        ModelRouter(settings, QuotaLedger(settings, clock), {}, clock, max_attempts=0)


def test_prompt_digest_is_stable_and_hex() -> None:
    """El digest identifica el prompt para caché y replay."""
    digest = prompt_digest("analiza")
    assert digest == prompt_digest("analiza")
    assert len(digest) == 64
    assert digest != prompt_digest("analiza ")
