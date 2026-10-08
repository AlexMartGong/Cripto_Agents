"""Bloque T: lo que cambia al pasar la ablación de la suscripción Go al pago por uso de Zen.

Dos criterios de aceptación con su mutación, que viven juntos porque comparten la forma: un aserto
sobre el código real, y la prueba de que ese aserto se rompe si se deshace la protección.

- El conteo previo sobre el manifiesto real muestra que cada rol remoto de `full` cabe con margen
  para el decisor a 1.2 intentos, y que el contador no degradaría ninguno a su respaldo local.
  Mutación: dejar la cuota del decisor en la cifra de Go (880), menor que su conteo con reintentos.
- `meta.json` guarda la `base_url` sin credenciales. Mutación: guardarla tal cual.
"""

from __future__ import annotations

import asyncio
import inspect
import math
import textwrap
from datetime import UTC, datetime
from functools import cache
from typing import TYPE_CHECKING, Any

import pytest

from crypto_agents import ablation, audit
from crypto_agents import settings as settings_module
from crypto_agents.ablation import (
    ARMS,
    DECIDER_ATTEMPTS,
    DryRunReport,
    dry_run,
    plan_from_manifest,
    render_dry_run,
)
from crypto_agents.audit import RunMeta, read_meta, write_meta
from crypto_agents.cache import InMemoryResponseCache
from crypto_agents.llm import ModelRouter
from crypto_agents.quota import QuotaExhaustedError, QuotaLedger
from crypto_agents.selection import HISTORY_DIR, load_manifest, load_selection_histories
from crypto_agents.settings import (
    ZEN_UNPUBLISHED_QUOTA,
    ModelChoice,
    RoleConfig,
    Settings,
    load_settings,
    public_url,
)
from crypto_agents.state import (
    AgentRole,
    Backend,
    Billing,
    LLMCall,
    StructuredOutputMode,
)
from tests.test_audit import meta as audit_meta

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

MANIFEST = "data/ablation_selection.json"
NOW = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)
GO_DECIDER_QUOTA = 880
"""La cuota del decisor en Go: el techo que CLAUDE.md derivaba de la página de la suscripción."""

MARGIN = 10
"""Veces que la cota con reintentos tiene que caber en la cuota declarada (el centinela da ~100)."""


# ──────────────────────────────────── Declaraciones payg propuestas ───────────────────────────────


def zen(model: str, family: str, mode: StructuredOutputMode, quota: int) -> ModelChoice:
    """Un rol remoto de Zen tal como lo propone el bloque: centinela de cuota y peso 1.0."""
    return ModelChoice(
        backend=Backend.OPENAI,
        model=model,
        family=family,
        structured_output=mode,
        quota_weight=1.0,
        quota_per_window=quota,
    )


LOCAL = ModelChoice(
    backend=Backend.OLLAMA,
    model="qwen3:8b",
    family="qwen",
    structured_output=StructuredOutputMode.JSON_SCHEMA,
    quota_per_window=10_000,
)
"""El respaldo local, que el bloque no toca."""


def payg_settings(decider_quota: int = ZEN_UNPUBLISHED_QUOTA) -> Settings:
    """El mapa de roles con las declaraciones payg propuestas para los seis roles remotos."""
    schema, json_mode, tools = (
        StructuredOutputMode.JSON_SCHEMA,
        StructuredOutputMode.JSON_MODE,
        StructuredOutputMode.FUNCTION_CALLING,
    )
    quota = ZEN_UNPUBLISHED_QUOTA
    roles = {
        AgentRole.STRUCTURE: RoleConfig(
            primary=zen("deepseek-v4.1-flash", "deepseek", schema, quota), fallback=LOCAL
        ),
        AgentRole.MOMENTUM: RoleConfig(
            primary=zen("deepseek-v4-flash", "deepseek", json_mode, quota), fallback=LOCAL
        ),
        AgentRole.VOLUME: RoleConfig(
            primary=zen("glm-5.3-flash", "zhipu", schema, quota), fallback=LOCAL
        ),
        AgentRole.BULL: RoleConfig(
            primary=zen("kimi-k2.6", "moonshot", schema, quota), fallback=LOCAL
        ),
        AgentRole.BEAR: RoleConfig(primary=zen("minimax-m3", "minimax", json_mode, quota)),
        AgentRole.DECIDER: RoleConfig(primary=zen("glm-5.2", "zhipu", tools, decider_quota)),
    }
    return load_settings(
        roles=roles,
        openai={"api_key": "no-se-usa", "base_url": "https://opencode.ai/zen/v1"},
        ollama={"host": "http://localhost:11434"},
        billing=Billing.PAYG,
    )


@cache
def manifest_report(decider_quota: int) -> DryRunReport:
    """El conteo previo sobre el manifiesto real y su histórico, con la caché vacía."""
    manifest = load_manifest(MANIFEST)
    plan = plan_from_manifest(manifest, load_selection_histories(manifest, HISTORY_DIR))
    return asyncio.run(
        dry_run(
            ARMS,
            plan,
            payg_settings(decider_quota),
            InMemoryResponseCache(),
            lambda: NOW,
        )
    )


def assert_every_remote_role_fits(report: DryRunReport) -> None:
    """Cada rol remoto de `full` cabe con margen para los reintentos del decisor."""
    remote = [line for line in report.quota if line.backend is Backend.OPENAI]
    assert {line.role for line in remote} == set(AgentRole), "falta un rol remoto en el conteo"
    for line in remote:
        assert line.fits_with_retries, (
            f"{line.role.value}: {line.quota_with_retries:.0f} con reintentos "
            f"no cabe en {line.per_window}"
        )
        assert line.quota_with_retries * MARGIN <= line.per_window, (
            f"{line.role.value} cabe justo ({line.quota_with_retries:.0f} de {line.per_window})"
        )


def call_of(role: AgentRole, choice: ModelChoice) -> LLMCall:
    return LLMCall(
        role=role,
        backend=choice.backend,
        model=choice.model,
        structured_output=choice.structured_output,
        quota_weight=choice.quota_weight,
        prompt_digest="0" * 64,
        valid=True,
        latency_ms=1.0,
        at=NOW,
    )


def test_the_manifest_dry_run_counts_the_decider_the_way_claude_md_says() -> None:
    """Sin esto la prueba de margen mediría otro conteo: 840 llamadas, 1 008 con reintentos."""
    report = manifest_report(ZEN_UNPUBLISHED_QUOTA)
    decider = next(line for line in report.quota if line.role is AgentRole.DECIDER)

    assert report.activations == 140
    assert decider.calls == 840
    assert decider.quota_with_retries == pytest.approx(840 * DECIDER_ATTEMPTS)
    assert DECIDER_ATTEMPTS == 1.2


def test_with_the_payg_declarations_every_remote_role_fits_with_room_for_retries() -> None:
    assert_every_remote_role_fits(manifest_report(ZEN_UNPUBLISHED_QUOTA))


STAGE_ONE = ("full", "solo", "always_buy", "always_sell", "random_uniform", "rule_trend")
"""La etapa 1 de la enmienda 3: `full`, `solo` y las cuatro líneas base."""


@cache
def stage_one_report() -> DryRunReport:
    """El conteo previo de la etapa 1 sobre el manifiesto real, con la caché vacía."""
    manifest = load_manifest(MANIFEST)
    plan = plan_from_manifest(manifest, load_selection_histories(manifest, HISTORY_DIR))
    arms = [arm for arm in ARMS if arm.name in STAGE_ONE]
    assert len(arms) == len(STAGE_ONE)
    return asyncio.run(dry_run(arms, plan, payg_settings(), InMemoryResponseCache(), lambda: NOW))


def test_stage_one_of_amendment_three_asks_only_for_full_and_solo() -> None:
    """140 activaciones: siete nodos que llaman, y ninguno de un brazo que la etapa no corre.

    Las cuatro líneas base no tienen filas —no llaman a nadie—, y `no_debate`, `bull_only` y los
    dos brazos locales no aparecen porque no se pidieron: la etapa 1 no gasta en ellos.
    """
    report = stage_one_report()

    assert report.evaluations == report.activations == 140
    assert report.prepare_failures == 0
    assert {(row.arm, row.node) for row in report.rows} == {
        ("full", "structure"),
        ("full", "momentum"),
        ("full", "volume"),
        ("full", "bull"),
        ("full", "bear"),
        ("full", "decide"),
        ("solo", "decide_solo"),
    }
    assert all(row.calls == 140 for row in report.rows)
    assert all(row.backend is Backend.OPENAI for row in report.rows), "ningún rol va en local"


def test_stage_one_pays_each_exact_prompt_once_and_bounds_the_rest() -> None:
    """Exactas: los tres técnicos y `decide_solo`, 140 cada uno. Cota: las mesas y `decide`."""
    by_node = {(row.arm, row.node): row for row in stage_one_report().rows}

    for key in (
        ("full", "structure"),
        ("full", "momentum"),
        ("full", "volume"),
        ("solo", "decide_solo"),
    ):
        assert by_node[key].exact
        assert (by_node[key].cached, by_node[key].to_pay) == (0, 140)
    for key in (("full", "bull"), ("full", "bear"), ("full", "decide")):
        assert not by_node[key].exact
        assert by_node[key].to_pay is None


def test_stage_one_asks_the_decider_for_at_most_280_calls() -> None:
    """140 de `full` (cota) y 140 de `solo` (exactas): un tercio de las 840 de los seis brazos."""
    decider = next(line for line in stage_one_report().quota if line.role is AgentRole.DECIDER)

    assert (decider.exact_calls, decider.bound_calls) == (140, 140)
    assert decider.calls == 280
    assert not decider.limits(Billing.PAYG)
    rendered = render_dry_run(stage_one_report())
    assert "**NO**" not in rendered
    for absent in ("no_debate", "bull_only", "local_technicals", "local_bull", "always_buy"):
        assert f"`{absent}`" not in rendered


def spent_router(settings: Settings, report: DryRunReport) -> tuple[ModelRouter, QuotaLedger]:
    """Un router cuyo contador ya anotó, de cada rol remoto, toda su cota con reintentos."""
    ledger = QuotaLedger(settings.quota_window, lambda: NOW)
    for line in (line for line in report.quota if line.backend is Backend.OPENAI):
        primary = settings.role_choices(line.role)[0]
        for _ in range(math.ceil(line.quota_with_retries)):
            ledger.record(call_of(line.role, primary))
    return ModelRouter(settings, ledger, {}, lambda: NOW), ledger


def test_with_the_payg_declarations_the_ledger_never_degrades_a_remote_role() -> None:
    """Gastar toda la cota con reintentos de un rol no lo manda a su respaldo local.

    Hasta el bloque T7 esto dependía de que cada rol remoto declarara el centinela. Sigue siendo
    cierto con él —la cifra cabe—, y ya no depende de él: lo decide el router (la prueba de abajo).
    """
    settings = payg_settings()
    router, ledger = spent_router(settings, manifest_report(ZEN_UNPUBLISHED_QUOTA))
    for role in AgentRole:
        choices = settings.role_choices(role)
        assert ledger.resolve(role, choices) == choices[0], f"{role.value} se degradaría"
        assert router._choose(role, choices) == choices[0]


def test_the_decider_never_has_a_local_fallback_to_degrade_to() -> None:
    assert payg_settings().role_config(AgentRole.DECIDER).fallback is None


def test_the_go_figure_left_on_the_decider_still_does_not_fit_the_count() -> None:
    """880 es menor que 840 x 1.2: la aritmética del conteo no cambió con el bloque T7.

    Se rehace el conteo con esa cuota —no se edita la línea ya calculada— para recorrer el mismo
    camino que quien se deje el número viejo en `.env`. Lo que cambió es a quién frena esa cifra:
    con la suscripción, al decisor; con pago por uso, a nadie (las dos pruebas de abajo).
    """
    report = manifest_report(GO_DECIDER_QUOTA)
    decider = next(line for line in report.quota if line.role is AgentRole.DECIDER)
    assert decider.per_window == GO_DECIDER_QUOTA
    assert decider.fits, "a un intento sí cabe: es justo lo que la columna destapa"
    with pytest.raises(AssertionError, match="decider"):
        assert_every_remote_role_fits(report)
    assert decider.limits(Billing.GO)
    assert not decider.limits(Billing.PAYG), "con pago por uso esa cifra no frena al decisor"


def test_under_payg_the_go_figure_left_on_the_decider_no_longer_exhausts_it() -> None:
    """Era el caso que T dejó escrito como defecto: 880 en `.env` con `CA_BILLING=payg`.

    1 008 llamadas del decisor anotadas en la ventana y una cuota de 880: el contador, preguntado,
    diría que no cabe. El router ya no le pregunta por un rol remoto con pago por uso —Zen no
    publica ese límite—, así que el decisor sigue en su modelo y la evaluación no aborta.
    """
    settings = payg_settings(GO_DECIDER_QUOTA)
    router, ledger = spent_router(settings, manifest_report(GO_DECIDER_QUOTA))
    choices = settings.role_choices(AgentRole.DECIDER)

    assert not ledger.fits(AgentRole.DECIDER, choices[0]), "el contador sigue registrando"
    assert router._choose(AgentRole.DECIDER, choices) == choices[0]


def test_under_the_subscription_the_same_figure_does_exhaust_the_decider() -> None:
    """El control: con `go` nada cambió, y el decisor sin cuota aborta registrado."""
    settings = payg_settings(GO_DECIDER_QUOTA).model_copy(update={"billing": Billing.GO})
    router, _ = spent_router(settings, manifest_report(GO_DECIDER_QUOTA))

    with pytest.raises(QuotaExhaustedError):
        router._choose(AgentRole.DECIDER, settings.role_choices(AgentRole.DECIDER))


# ───────────────────────────────────────── base_url sin credenciales ──────────────────────────────

LEAKY = "https://user-xyz:hunter2@opencode.ai/zen/v1?token=tokenvalue-xyz#fragment-xyz"
SECRETS = ("user-xyz", "hunter2", "tokenvalue-xyz", "fragment-xyz")


def assert_meta_hides_the_credentials(directory: Path) -> None:
    """Un meta construido con la URL completa, escrito y releído, no la lleva."""
    original = audit_meta()
    meta = RunMeta.model_validate({**original.model_dump(mode="json"), "base_url": LEAKY})
    path = write_meta(directory, meta)
    text = path.read_text("utf-8")
    for secret in SECRETS:
        assert secret not in text, f"meta.json filtra {secret!r}"
    assert read_meta(directory).base_url == "https://opencode.ai/zen/v1"


def mutated(function: Callable[..., Any], old: str, new: str) -> Callable[..., Any]:
    """La función con un fragmento cambiado, compilada en el espacio del módulo de `settings`."""
    source = textwrap.dedent(inspect.getsource(function))
    assert source.count(old) == 1, f"el fragmento {old!r} ya no está una sola vez"
    namespace: dict[str, Any] = dict(vars(settings_module))
    exec(compile(source.replace(old, new), "<mutante>", "exec"), namespace)
    return namespace[function.__name__]  # type: ignore[no-any-return]


def test_the_meta_never_carries_the_credentials_of_the_url(tmp_path: Path) -> None:
    assert_meta_hides_the_credentials(tmp_path)


def test_mutation_storing_the_url_as_it_is_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mutant = mutated(
        public_url,
        'return f"{parts.scheme}://{authority}{parts.path}"',
        "return url",
    )
    monkeypatch.setattr(audit, "public_url", mutant)
    with pytest.raises(AssertionError):
        assert_meta_hides_the_credentials(tmp_path)


def test_the_ablation_writes_the_url_through_the_same_filter() -> None:
    """Si el meta dejara de pasar por `public_url`, el validador seguiría salvándolo."""
    assert "public_url(" in inspect.getsource(ablation.build_run_meta)
    assert "build_run_meta(" in inspect.getsource(ablation._run)
