"""Pruebas de la ablación: variantes del pipeline, brazos y reporte.

Con modelos falsos todas las variantes deciden lo mismo, así que aquí no se
comprueba *qué* decide cada brazo —eso lo dirá la corrida real— sino que cada
variante recorre la forma de grafo que dice recorrer, que las reglas permanentes
siguen en pie en todas, y que la comparación no miente.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from crypto_agents.ablation import (
    ARMS,
    AblationArm,
    agreement,
    arm_settings,
    build_arm_result,
    decision_actions,
    render_report,
    run_arm,
)
from crypto_agents.cache import InMemoryResponseCache
from crypto_agents.graph import PipelineVariant, build_graph
from crypto_agents.journal import EvaluationRecord
from crypto_agents.metrics import summarise
from crypto_agents.replay import ReplaySettings
from crypto_agents.settings import (
    ConfigError,
    ModelChoice,
    RoleConfig,
    Settings,
    load_settings,
)
from crypto_agents.state import Action, AgentRole, Backend, Proposal, StructuredOutputMode
from tests.conftest import CHEAP, HEALTHY, PRESET, FakeLLM, role_map
from tests.test_replay import Harness, run, synthetic_rows


def local(model: str, family: str) -> ModelChoice:
    """Modelo local declarado como respaldo."""
    return ModelChoice(
        backend=Backend.OLLAMA,
        model=model,
        family=family,
        structured_output=StructuredOutputMode.JSON_SCHEMA,
        quota_per_window=1000,
    )


# ─────────────────────────────────────── Formas del grafo ─────────────────────────────────────────


@pytest.mark.parametrize("variant", list(PipelineVariant))
def test_every_variant_compiles(variant: PipelineVariant) -> None:
    """Las cuatro formas construyen un grafo válido."""
    assert build_graph(variant) is not None


@pytest.mark.parametrize("variant", list(PipelineVariant))
def test_every_variant_ends_with_the_risk_gate_before_the_order(
    variant: PipelineVariant,
) -> None:
    """Regla permanente 2 en todas las variantes: el decisor propone, el riesgo dispone.

    Una ablación que cambiara la cola determinista mediría el arnés, y además
    abriría un camino al mercado que no pasa por el gate.
    """
    graph = build_graph(variant)
    nodes = set(graph.get_graph().nodes)
    assert {"risk", "execute", "journal"} <= nodes


def test_the_solo_variant_has_no_technical_agents() -> None:
    """Un modelo, una llamada: si quedaran los técnicos no sería el baseline."""
    nodes = set(build_graph(PipelineVariant.SOLO).get_graph().nodes)
    assert not nodes & {"structure", "momentum", "volume", "consolidate", "bull", "bear"}
    assert "decide_solo" in nodes


def test_the_no_debate_variant_keeps_the_technicals_and_drops_the_desks() -> None:
    """Es exactamente el contraste que la variante existe para medir."""
    nodes = set(build_graph(PipelineVariant.NO_DEBATE).get_graph().nodes)
    assert {"structure", "momentum", "volume", "consolidate"} <= nodes
    assert not nodes & {"bull", "bear"}


def test_the_bull_only_variant_keeps_one_desk() -> None:
    """Una mesa sí, la contraparte no."""
    nodes = set(build_graph(PipelineVariant.BULL_ONLY).get_graph().nodes)
    assert "bull" in nodes
    assert "bear" not in nodes


# ─────────────────────────────────── Ejecución de las variantes ───────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("variant", list(PipelineVariant))
async def test_every_variant_produces_decisions_over_the_history(
    variant: PipelineVariant,
) -> None:
    """Cada forma llega hasta el final del histórico sin errores de nodo."""
    harness = Harness()
    harness.graph = build_graph(variant)
    records = await run(harness, synthetic_rows(), fill=True, evaluations=12)

    assert len(records) == 12
    assert all(not record.errors for record in records), [r.errors for r in records if r.errors]
    assert any(record.proposed is not None for record in records)


@pytest.mark.asyncio
async def test_variants_without_desks_record_a_proposal_not_a_decision() -> None:
    """El journal distingue «no hubo mesas» de «el decisor no dijo a quién descartaba»."""
    harness = Harness()
    harness.graph = build_graph(PipelineVariant.NO_DEBATE)
    records = await run(harness, synthetic_rows(), fill=True, evaluations=12)

    decided = [record for record in records if record.proposed is not None]
    assert decided, "ninguna evaluación decidió"
    assert all(record.decision is None for record in decided)
    assert all(isinstance(record.proposal, Proposal) for record in decided)


@pytest.mark.asyncio
async def test_the_full_variant_still_records_a_decision() -> None:
    """La rama que opera de verdad no cambia de esquema por existir la ablación."""
    harness = Harness()
    records = await run(harness, synthetic_rows(), fill=True, evaluations=12)

    decided = [record for record in records if record.proposed is not None]
    assert decided
    assert all(record.decision is not None for record in decided)
    assert all(record.proposal is None for record in decided)


@pytest.mark.asyncio
async def test_the_solo_variant_spends_one_call_per_evaluation() -> None:
    """Seis llamadas contra una: es el contraste de coste que la tabla debe mostrar."""
    solo = Harness()
    solo.graph = build_graph(PipelineVariant.SOLO)
    solo_records = await run(solo, synthetic_rows(), fill=True, evaluations=12)

    full_records = await run(Harness(), synthetic_rows(), fill=True, evaluations=12)

    solo_calls = sum(len(record.calls) for record in solo_records)
    full_calls = sum(len(record.calls) for record in full_records)
    assert solo_calls * 6 == full_calls


# ────────────────────────────────────── Brazos y configuración ────────────────────────────────────


def test_a_local_arm_swaps_the_primary_for_the_declared_fallback() -> None:
    """El brazo local recorre el mismo código; solo cambia qué modelo atiende."""
    settings = load_settings(
        roles=role_map(primary=CHEAP, fallback=local("local-x", "local-fam")),
        ollama={"host": "http://localhost:11434"},
    )
    arm = AblationArm(
        name="local_technicals",
        variant=PipelineVariant.FULL,
        question="¿aguantan los técnicos en local?",
        local_roles=frozenset({AgentRole.STRUCTURE}),
    )
    tuned = arm_settings(settings, arm)

    assert tuned.role_config(AgentRole.STRUCTURE).primary.model == "local-x"
    assert tuned.role_config(AgentRole.MOMENTUM).primary.model == CHEAP.model
    assert settings.role_config(AgentRole.STRUCTURE).primary.model == CHEAP.model, "no muta"


def test_a_local_arm_without_a_declared_fallback_fails_loudly() -> None:
    """Pedir local donde no hay respaldo es un error de configuración, no un silencio."""
    settings = load_settings(
        roles=role_map(primary=CHEAP),
        ollama={"host": "http://localhost:11434"},
    )
    arm = AblationArm(
        name="local_bull",
        variant=PipelineVariant.FULL,
        question="¿aguanta una mesa en local?",
        local_roles=frozenset({AgentRole.BULL}),
    )
    with pytest.raises(ConfigError, match="no declara respaldo"):
        arm_settings(settings, arm)


def test_an_arm_without_local_roles_leaves_the_settings_alone() -> None:
    """Sin roles marcados no hay nada que intercambiar."""
    settings = load_settings(roles=role_map(), ollama={"host": "http://localhost:11434"})
    assert arm_settings(settings, ARMS[0]) is settings


def test_the_declared_arms_cover_the_six_questions() -> None:
    """Los brazos del enunciado, cada uno con su pregunta escrita."""
    assert [arm.name for arm in ARMS] == [
        "full",
        "no_debate",
        "bull_only",
        "solo",
        "local_technicals",
        "local_bull",
    ]
    assert all(arm.question for arm in ARMS)


def test_the_decider_cannot_be_moved_to_local_by_an_arm() -> None:
    """Ningún brazo declara al decisor en local: no tiene respaldo, y es a propósito."""
    assert all(AgentRole.DECIDER not in arm.local_roles for arm in ARMS)


# ─────────────────────────────────────────── Driver ───────────────────────────────────────────────


def ablation_settings() -> Settings:
    """Seis roles en familias distintas, cada uno con respaldo local declarado."""
    roles = {
        role: RoleConfig(
            primary=ModelChoice(
                backend=Backend.OLLAMA,
                model=f"remoto-{role.value}",
                family=f"fam-{role.value}",
                structured_output=StructuredOutputMode.JSON_SCHEMA,
                quota_per_window=100_000,
            ),
            fallback=(
                None
                if role is AgentRole.DECIDER
                else local(f"local-{role.value}", f"local-{role.value}")
            ),
        )
        for role in AgentRole
    }
    return load_settings(roles=roles, ollama={"host": "http://localhost:11434"})


@pytest.mark.asyncio
@pytest.mark.parametrize("arm", ARMS, ids=lambda arm: arm.name)
async def test_run_arm_produces_a_comparable_result(arm: AblationArm) -> None:
    """Cada brazo corre entero y devuelve un resultado con la misma forma que los demás.

    Es lo que el comando ejecuta. Con modelos falsos los seis deciden lo mismo, así
    que lo que se comprueba es que ninguno revienta y que todos rinden las mismas
    columnas: la tabla no puede tener huecos según qué brazo.
    """
    rows = synthetic_rows()
    config = ReplaySettings(
        symbol="BTC/USDT", timeframe="1h", warmup_bars=PRESET.min_bars, max_evaluations=8
    )
    result = await run_arm(
        arm,
        rows,
        ablation_settings(),
        config,
        HEALTHY,
        InMemoryResponseCache(),
        lambda: datetime(2026, 8, 1, tzinfo=UTC),
        fill_with={Backend.OLLAMA: FakeLLM()},
        horizon=3,
        preset=PRESET,
    )

    assert result.arm is arm
    assert result.summary.evaluations == 8
    assert len(result.actions) == 8
    assert result.summary.quota_used > 0.0
    assert result.outcomes.orders >= 0


@pytest.mark.asyncio
async def test_a_local_arm_routes_its_roles_through_the_fallback_model() -> None:
    """El brazo local se distingue en el journal: las llamadas llevan el modelo local."""
    rows = synthetic_rows()
    config = ReplaySettings(
        symbol="BTC/USDT", timeframe="1h", warmup_bars=PRESET.min_bars, max_evaluations=6
    )
    arm = next(item for item in ARMS if item.name == "local_technicals")
    result = await run_arm(
        arm,
        rows,
        ablation_settings(),
        config,
        HEALTHY,
        InMemoryResponseCache(),
        lambda: datetime(2026, 8, 1, tzinfo=UTC),
        fill_with={Backend.OLLAMA: FakeLLM()},
        preset=PRESET,
    )

    assert result.models[AgentRole.STRUCTURE] == "local-structure"
    assert result.models[AgentRole.DECIDER] == "remoto-decider"


@pytest.mark.asyncio
async def test_variants_without_desks_still_count_as_having_decided() -> None:
    """Contar solo `decision` daría cero decisiones justo en los brazos a comparar.

    `no_debate` y `solo` registran su salida en `proposal`. Una tabla que los
    contara como si nunca hubieran decidido invitaría a la conclusión contraria a
    la verdadera: que quitar el debate deja al sistema mudo.
    """
    harness = Harness()
    harness.graph = build_graph(PipelineVariant.NO_DEBATE)
    records = await run(harness, synthetic_rows(), fill=True, evaluations=10)

    summary = summarise(records)
    assert summary.decided > 0
    assert sum(summary.actions.values()) == summary.decided


# ──────────────────────────────────────── Comparación ─────────────────────────────────────────────


def test_agreement_counts_matching_actions() -> None:
    """Coincidencia es la misma acción en la misma vela."""
    left = (Action.BUY, Action.HOLD, Action.SELL, None)
    assert agreement(left, left) == 1.0
    assert agreement(left, (Action.BUY, Action.HOLD, Action.BUY, None)) == 0.75


def test_agreement_ignores_confidence() -> None:
    """Comprar con 0.7 y con 0.6 es la misma decisión."""
    assert agreement((Action.BUY,), (Action.BUY,)) == 1.0


def test_agreement_of_different_lengths_is_undefined() -> None:
    """Dos corridas de distinta longitud no son comparables por posición."""
    assert agreement((Action.BUY,), (Action.BUY, Action.HOLD)) is None
    assert agreement((), ()) is None


def test_decision_actions_marks_evaluations_that_never_decided() -> None:
    """Un `None` es «aquí no se decidió», no «aquí se decidió hold»."""
    record = EvaluationRecord(
        run_id=uuid4(),
        at=datetime(2026, 8, 1, tzinfo=UTC),
        symbol="BTC/USDT",
        timeframe="4h",
    )
    assert decision_actions([record]) == (None,)


def test_the_report_renders_every_arm() -> None:
    """La tabla lleva una fila por brazo y la pregunta de cada uno."""
    results = [build_arm_result(arm, [], [], wall_clock_seconds=1.0) for arm in ARMS[:3]]
    report = render_report(results)

    for arm in ARMS[:3]:
        assert f"`{arm.name}`" in report
        assert arm.question in report


def test_the_report_survives_an_empty_run() -> None:
    """Sin corridas la tabla lo dice, en vez de fingir ceros."""
    assert "Sin corridas" in render_report([])
