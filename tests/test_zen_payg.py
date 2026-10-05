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
from crypto_agents.ablation import ARMS, DECIDER_ATTEMPTS, DryRunReport, dry_run, plan_from_manifest
from crypto_agents.audit import RunMeta, read_meta, write_meta
from crypto_agents.cache import InMemoryResponseCache
from crypto_agents.quota import QuotaLedger
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


def test_with_the_payg_declarations_the_ledger_never_degrades_a_remote_role() -> None:
    """Gastar toda la cota con reintentos de un rol no lo manda a su respaldo local."""
    settings = payg_settings()
    report = manifest_report(ZEN_UNPUBLISHED_QUOTA)
    for line in (line for line in report.quota if line.backend is Backend.OPENAI):
        choices = settings.role_choices(line.role)
        ledger = QuotaLedger(settings.quota_window, lambda: NOW)
        for _ in range(math.ceil(line.quota_with_retries)):
            ledger.record(call_of(line.role, choices[0]))

        assert ledger.resolve(line.role, choices) == choices[0], (
            f"{line.role.value} se degradaría tras {math.ceil(line.quota_with_retries)} llamadas"
        )


def test_the_decider_never_has_a_local_fallback_to_degrade_to() -> None:
    assert payg_settings().role_config(AgentRole.DECIDER).fallback is None


def test_mutation_the_decider_left_at_the_go_figure_is_caught() -> None:
    """880 es menor que 840 x 1.2: con la cifra de Go, el decisor agotaría la ventana.

    Se rehace el conteo con esa cuota —no se edita la línea ya calculada— para que la prueba
    recorra el mismo camino que recorrería quien se deje el número viejo en `.env`.
    """
    report = manifest_report(GO_DECIDER_QUOTA)
    decider = next(line for line in report.quota if line.role is AgentRole.DECIDER)
    assert decider.per_window == GO_DECIDER_QUOTA
    assert decider.fits, "a un intento sí cabe: es justo lo que la columna nueva destapa"
    with pytest.raises(AssertionError, match="decider"):
        assert_every_remote_role_fits(report)


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
    """Si `_run` dejara de pasar por `public_url`, el validador seguiría salvando el meta."""
    assert "public_url(" in inspect.getsource(ablation._run)
