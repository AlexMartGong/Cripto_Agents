"""Pruebas de la ablación: variantes del pipeline, brazos y reporte.

Con modelos falsos todas las variantes deciden lo mismo, así que aquí no se
comprueba *qué* decide cada brazo —eso lo dirá la corrida real— sino que cada
variante recorre la forma de grafo que dice recorrer, que las reglas permanentes
siguen en pie en todas, y que la comparación no miente.
"""

from __future__ import annotations

import math
import re
import subprocess
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

from crypto_agents import ablation
from crypto_agents.ablation import (
    ARMS,
    DETERMINISTIC_NODES,
    AblationArm,
    ArmResult,
    DryRunReport,
    DryRunRow,
    QuotaLine,
    ReplayPlan,
    agreement,
    agreement_counts,
    agreement_decided,
    arm_settings,
    build_arm_result,
    decision_actions,
    default_journal_dir,
    dry_run,
    git_state,
    llm_nodes,
    main,
    open_run_directory,
    plan_from_history,
    plan_from_manifest,
    render_dry_run,
    render_report,
    run_arm,
    run_arms,
    seed_from_previous,
)
from crypto_agents.activation import ActivationConfig
from crypto_agents.audit import PlanKind, RunMeta, arm_journal_path, read_meta, read_run
from crypto_agents.baselines import trend_action
from crypto_agents.cache import InMemoryResponseCache
from crypto_agents.graph import PipelineVariant, build_graph
from crypto_agents.journal import EvaluationRecord, InMemoryJournal, JsonlJournal
from crypto_agents.metrics import QuotaSplit, summarise
from crypto_agents.outcomes import Scoring, score_run
from crypto_agents.quota import QuotaLedger
from crypto_agents.replay import ReplaySettings
from crypto_agents.selection import (
    DEFAULT_SEED,
    SelectionError,
    SelectionManifest,
    select_activations,
)
from crypto_agents.settings import (
    DEFAULT_PRICING,
    ConfigError,
    ModelChoice,
    RoleConfig,
    Settings,
    load_settings,
)
from crypto_agents.state import (
    Action,
    AgentRole,
    Backend,
    Billing,
    ExecutionMode,
    LLMCall,
    OrderIntent,
    Proposal,
    RiskVerdict,
    StructuredOutputMode,
)
from crypto_agents.stops import COMMON_STOP_DESCRIPTION
from tests.conftest import CHEAP, HEALTHY, PRESET, REAL_HISTORY, FakeLLM, raw_ohlcv, role_map
from tests.test_metrics import CLOSED_GATE, failed, proposal, timed, unbalanced_calls
from tests.test_metrics import record as metric_record
from tests.test_outcomes import position_at, record_at
from tests.test_outcomes import rows as candle_rows
from tests.test_replay import Harness, run, synthetic_rows


def fixed_clock() -> datetime:
    """Reloj del arnés. El instante da igual; que sea el mismo en dos corridas no."""
    return datetime(2026, 8, 1, tzinfo=UTC)


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


def test_the_declared_arms_are_the_six_questions_then_the_four_baselines() -> None:
    """Los seis brazos del enunciado primero, y las cuatro líneas base al final.

    Van detrás para que los índices de los seis no cambien, y cada una con su pregunta.
    """
    assert [arm.name for arm in ARMS[:6]] == [
        "full",
        "no_debate",
        "bull_only",
        "solo",
        "local_technicals",
        "local_bull",
    ]
    assert [(arm.name, arm.variant) for arm in ARMS[6:]] == [
        ("always_buy", PipelineVariant.ALWAYS_BUY),
        ("always_sell", PipelineVariant.ALWAYS_SELL),
        ("random_uniform", PipelineVariant.RANDOM_UNIFORM),
        ("rule_trend", PipelineVariant.RULE_TREND),
    ]
    assert all(arm.question for arm in ARMS)
    assert all(not arm.local_roles for arm in ARMS[6:])


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
    settings = ablation_settings()
    result = await run_arm(
        arm,
        plan_from_history(rows, config),
        settings,
        HEALTHY,
        InMemoryResponseCache(),
        fixed_clock,
        QuotaLedger(settings.quota_window, fixed_clock),
        fill_with={Backend.OLLAMA: FakeLLM()},
        horizon=3,
        preset=PRESET,
        journal=InMemoryJournal(),
    )

    assert result.arm is arm
    assert result.summary.evaluations == 8
    assert len(result.actions) == 8
    if arm.variant in BASELINE_VARIANTS:
        assert result.summary.quota_used == 0.0, "una línea base no llama a nadie"
    else:
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
    settings = ablation_settings()
    result = await run_arm(
        arm,
        plan_from_history(rows, config),
        settings,
        HEALTHY,
        InMemoryResponseCache(),
        fixed_clock,
        QuotaLedger(settings.quota_window, fixed_clock),
        fill_with={Backend.OLLAMA: FakeLLM()},
        preset=PRESET,
        journal=InMemoryJournal(),
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


# ────────────────────────────── La caché deduplica entre brazos ───────────────────────────────────
# Seis brazos por 25 evaluaciones a ~90 s son cerca de cuatro horas. Lo que decide
# si son cuatro o menos es cuántas etapas se pagan una sola vez, así que aquí se
# comprueba dónde reutiliza de verdad y —igual de importante— dónde no.


TECHNICAL_ROLES = frozenset({AgentRole.STRUCTURE, AgentRole.MOMENTUM, AgentRole.VOLUME})


def calls_of(records: Sequence[EvaluationRecord], roles: frozenset[AgentRole]) -> list[LLMCall]:
    """Intentos registrados de esos roles, en orden."""
    return [call for record in records for call in record.calls if call.role in roles]


@pytest.mark.asyncio
async def test_the_cache_reuses_a_shared_stage_between_arms_by_prompt_digest() -> None:
    """Las etapas comunes se pagan una vez, y se demuestra por el digest del prompt.

    El prompt técnico es función del snapshot, los indicadores y los disparadores:
    nada de eso depende de qué forma tenga el grafo aguas abajo, así que `full` y
    `no_debate` hacen literalmente la misma pregunta a los mismos tres modelos. Si
    no se reutilizara, la ablación pagaría cuatro veces los mismos tres veredictos.
    """
    rows = synthetic_rows()
    cache = InMemoryResponseCache()

    first = Harness(cache)
    first_records = await run(first, rows, fill=True, evaluations=8)

    second = Harness(cache)
    second.backend = first.backend
    second.graph = build_graph(PipelineVariant.NO_DEBATE)
    second_records = await run(second, rows, fill=True, evaluations=8)

    paid = calls_of(first_records, TECHNICAL_ROLES)
    reused = calls_of(second_records, TECHNICAL_ROLES)

    assert paid, "el primer brazo no llegó a llamar a ningún técnico"
    assert all(not call.cache_hit for call in paid)
    assert len(reused) == len(paid)
    assert all(call.cache_hit for call in reused), "el segundo brazo volvió a pagar los técnicos"
    assert {call.prompt_digest for call in reused} == {call.prompt_digest for call in paid}


@pytest.mark.asyncio
async def test_the_decider_is_not_reused_between_arms_that_feed_it_differently() -> None:
    """La contraprueba: lo que cambia de prompt no puede compartir entrada.

    El prompt del decisor lleva los alegatos dentro. Sin mesas es otro texto, otro
    digest y otra llamada — y tiene que serlo, porque la respuesta a una pregunta
    distinta no vale como respuesta a esta.
    """
    rows = synthetic_rows()
    cache = InMemoryResponseCache()

    first = Harness(cache)
    await run(first, rows, fill=True, evaluations=8)

    second = Harness(cache)
    second.backend = first.backend
    second.graph = build_graph(PipelineVariant.NO_DEBATE)
    second_records = await run(second, rows, fill=True, evaluations=8)

    decider = calls_of(second_records, frozenset({AgentRole.DECIDER}))
    assert decider
    assert all(not call.cache_hit for call in decider)


@pytest.mark.asyncio
async def test_a_remote_arm_cannot_be_served_what_a_local_arm_cached() -> None:
    """El orden inverso, que era el que fallaba: primero el brazo local, después `full`.

    `full` declara un primario remoto y un respaldo local por rol técnico. La
    lectura de caché recorría los dos, así que con la entrada del respaldo ya
    escrita por `local_technicals` el brazo de referencia la recibía como acierto:
    dejaba de medir sus modelos sin que la tabla lo dijera. Ahora paga sus tres
    lecturas, y cada llamada lleva el modelo remoto.
    """
    rows = synthetic_rows()
    config = ReplaySettings(
        symbol="BTC/USDT", timeframe="1h", warmup_bars=PRESET.min_bars, max_evaluations=6
    )
    settings = ablation_settings()
    cache = InMemoryResponseCache()
    backend = FakeLLM()
    by_name = {arm.name: arm for arm in ARMS}
    ledger = QuotaLedger(settings.quota_window, fixed_clock)
    plan = plan_from_history(rows, config)
    journal = InMemoryJournal()

    for name, destination in (("local_technicals", InMemoryJournal()), ("full", journal)):
        await run_arm(
            by_name[name],
            plan,
            settings,
            HEALTHY,
            cache,
            fixed_clock,
            ledger,
            destination,
            fill_with={Backend.OLLAMA: backend},
            preset=PRESET,
        )

    technical = [
        call
        for record in journal.records
        for call in record.calls
        if call.role in (AgentRole.STRUCTURE, AgentRole.MOMENTUM, AgentRole.VOLUME)
    ]
    assert technical, "el brazo de referencia no llegó a los técnicos"
    assert {call.model for call in technical} == {
        "remoto-structure",
        "remoto-momentum",
        "remoto-volume",
    }
    assert all(not call.cache_hit for call in technical)


@pytest.mark.asyncio
async def test_a_local_arm_cannot_reuse_what_the_remote_arm_paid_for() -> None:
    """Mismo prompt, otro modelo: otra clave. Es correcto, y hay que contarlo aparte.

    La clave de caché lleva el modelo dentro porque la respuesta de un 8B local no
    es la de un modelo remoto grande. `local_technicals` vuelve a pagar sus tres
    lecturas, y quien lea el presupuesto tiene que saberlo antes y no después.
    """
    rows = synthetic_rows()
    config = ReplaySettings(
        symbol="BTC/USDT", timeframe="1h", warmup_bars=PRESET.min_bars, max_evaluations=6
    )
    settings = ablation_settings()
    cache = InMemoryResponseCache()
    backend = FakeLLM()
    clock = fixed_clock

    by_name = {arm.name: arm for arm in ARMS}
    ledger = QuotaLedger(settings.quota_window, clock)
    plan = plan_from_history(rows, config)
    remote = await run_arm(
        by_name["full"],
        plan,
        settings,
        HEALTHY,
        cache,
        clock,
        ledger,
        fill_with={Backend.OLLAMA: backend},
        preset=PRESET,
        journal=InMemoryJournal(),
    )
    local = await run_arm(
        by_name["local_technicals"],
        plan,
        settings,
        HEALTHY,
        cache,
        clock,
        ledger,
        fill_with={Backend.OLLAMA: backend},
        preset=PRESET,
        journal=InMemoryJournal(),
    )

    assert remote.summary.quota_by_role[AgentRole.STRUCTURE] > 0.0
    assert local.summary.quota_by_role[AgentRole.STRUCTURE] > 0.0, "reutilizó otro modelo"
    assert local.models[AgentRole.STRUCTURE] != remote.models[AgentRole.STRUCTURE]


@pytest.mark.asyncio
async def test_a_local_arm_still_reuses_the_roles_it_did_not_move() -> None:
    """`local_bull` mueve una mesa; los tres técnicos siguen siendo los mismos."""
    rows = synthetic_rows()
    config = ReplaySettings(
        symbol="BTC/USDT", timeframe="1h", warmup_bars=PRESET.min_bars, max_evaluations=6
    )
    settings = ablation_settings()
    cache = InMemoryResponseCache()
    backend = FakeLLM()
    clock = fixed_clock

    by_name = {arm.name: arm for arm in ARMS}
    ledger = QuotaLedger(settings.quota_window, clock)
    for name in ("full", "local_bull"):
        result = await run_arm(
            by_name[name],
            plan_from_history(rows, config),
            settings,
            HEALTHY,
            cache,
            clock,
            ledger,
            fill_with={Backend.OLLAMA: backend},
            preset=PRESET,
            journal=InMemoryJournal(),
        )
        if name == "local_bull":
            assert AgentRole.STRUCTURE not in result.summary.quota_by_role
            assert result.summary.quota_by_role[AgentRole.BULL] > 0.0


# ────────────────────────────── Un solo contador para los seis brazos ─────────────────────────────
# Un `QuotaLedger` por brazo son seis contadores creyéndose cada uno dentro del
# presupuesto mientras el proveedor ve la suma. Es el mismo defecto que el runner
# tenía entre símbolos, y volvió por otra puerta: la construcción dentro de
# `run_arm`. Ahora entra por la firma, y esto lo fija por comportamiento.

BUDGETED_ARMS = 3
"""Cuántos brazos caben en el presupuesto que la prueba concede al decisor."""


def with_decider_budget(settings: Settings, per_window: int) -> Settings:
    """La misma configuración con la ventana del decisor recortada."""
    decider = settings.role_config(AgentRole.DECIDER).primary
    roles = dict(settings.roles) | {
        AgentRole.DECIDER: RoleConfig(
            primary=decider.model_copy(update={"quota_per_window": per_window})
        )
    }
    return settings.model_copy(update={"roles": roles})


def decided_in(journal: InMemoryJournal) -> int:
    """Evaluaciones de ese brazo en las que el decisor llegó a responder."""
    return sum(1 for record in journal.records if record.proposed is not None)


def quota_aborts(journal: InMemoryJournal) -> list[EvaluationRecord]:
    """Registros que abortaron por cuota agotada, con su error dentro."""
    return [
        record
        for record in journal.records
        if any("cuota agotada" in error.message for error in record.errors)
    ]


@pytest.mark.asyncio
async def test_the_arms_share_one_ledger_and_the_late_ones_find_the_window_empty() -> None:
    """Presupuesto para tres brazos y seis brazos pidiendo: el cuarto se lo encuentra vacío.

    Las cuatro líneas base corren detrás con la ventana ya agotada y deciden igual: no
    llaman a nadie, así que no tienen cuota de la que quedarse sin.

    Con un contador por brazo los seis creerían tener la ventana entera y los seis
    decidirían, gastando el doble de lo declarado sin que nada lo dijera. Con uno
    solo, el gasto agregado no puede pasar del presupuesto y el brazo que se queda
    fuera **aborta registrando**: su evaluación queda en el journal con el error,
    que es lo que distingue «no había cuota» de «este brazo no decidió nunca».

    Cada brazo lleva su propia caché a propósito. Con una compartida, cuántas
    llamadas paga cada uno depende de si los modelos falsos devuelven lo mismo, y lo
    que se prueba aquí es el contador, no la deduplicación.
    """
    rows = synthetic_rows()
    config = ReplaySettings(
        symbol="BTC/USDT", timeframe="1h", warmup_bars=PRESET.min_bars, max_evaluations=8
    )
    settings = ablation_settings()
    counted = await dry_run(
        [ARMS[0]],
        plan_from_history(rows, config),
        settings,
        InMemoryResponseCache(),
        fixed_clock,
        preset=PRESET,
    )
    assert counted.activations > 0, "sin activaciones no hay decisor al que agotar"

    budget = counted.activations * BUDGETED_ARMS
    tight = with_decider_budget(settings, budget)
    ledger = QuotaLedger(tight.quota_window, fixed_clock)

    journals: list[tuple[AblationArm, InMemoryJournal]] = []
    for arm in ARMS:
        journal = InMemoryJournal()
        await run_arm(
            arm,
            plan_from_history(rows, config),
            tight,
            HEALTHY,
            InMemoryResponseCache(),
            fixed_clock,
            ledger,
            fill_with={Backend.OLLAMA: FakeLLM()},
            preset=PRESET,
            journal=journal,
        )
        journals.append((arm, journal))

    modelled = [(arm, journal) for arm, journal in journals if arm.variant not in BASELINE_VARIANTS]
    decided = [arm.name for arm, journal in modelled if decided_in(journal) > 0]
    starved = [(arm, journal) for arm, journal in journals if quota_aborts(journal)]

    assert decided == [arm.name for arm in ARMS[:BUDGETED_ARMS]], (
        "el presupuesto alcanza para tres brazos: si deciden más, cada uno tiene el suyo"
    )
    for arm, journal in journals:
        if arm.variant in BASELINE_VARIANTS:
            assert decided_in(journal) > 0, f"{arm.name} no decidió con la ventana agotada"
            assert not quota_aborts(journal), f"{arm.name} dependió de la cuota"
    assert starved, "ningún brazo se encontró la ventana agotada"
    for arm, journal in starved:
        aborted = quota_aborts(journal)
        assert all(record.proposed is None for record in aborted), f"{arm.name} decidió sin cuota"
        assert all(record.activation is not None for record in aborted)
        assert len(journal.records) == 8, f"{arm.name} dejó evaluaciones sin registrar"

    model = tight.role_config(AgentRole.DECIDER).primary.model
    assert ledger.used(AgentRole.DECIDER, model) <= budget


# ──────────────────────────── El plan que sale de un manifiesto ───────────────────────────────────
# El histórico comprometido son 500 velas: 100 evaluables y 15 activaciones. Seis
# formas de pipeline sobre 15 decisiones no se separan. El manifiesto sustituye ese
# recorrido contiguo por activaciones repartidas entre símbolos y tramos, que es lo
# que permite preguntar si la ventaja de un brazo sobrevive al cambio de régimen.


def two_symbol_manifest() -> tuple[SelectionManifest, dict[str, list[list[float]]]]:
    """Manifiesto pequeño sobre dos series sintéticas distintas."""
    data = {
        "BTC/USDT": synthetic_rows(300),
        "ETH/USDT": raw_ohlcv(
            [110.0 + 6.0 * math.sin(index / 5.0) + 0.04 * index for index in range(300)]
        ),
    }
    manifest = select_activations(
        data,
        timeframe="1h",
        target=8,
        strata=2,
        seed=11,
        horizon=3,
        config=ActivationConfig(preset=PRESET),
        preset=PRESET,
        now=datetime(2026, 8, 15, tzinfo=UTC),
    )
    return manifest, data


@pytest.mark.asyncio
async def test_two_dry_runs_over_a_manifest_ask_for_exactly_the_same_prompts() -> None:
    """Si el conteo no es reproducible, el presupuesto no vale para decidir nada.

    Los digests son las claves de caché: que dos conteos coincidan es lo que dice
    que la segunda corrida encontrará pagado lo que la primera pagó, en vez de
    volver a pasar por caja con un prompt que cambió por debajo.
    """
    manifest, data = two_symbol_manifest()
    settings = ablation_settings()

    first = await dry_run(
        ARMS,
        plan_from_manifest(manifest, data),
        settings,
        InMemoryResponseCache(),
        fixed_clock,
        preset=PRESET,
    )
    second = await dry_run(
        ARMS,
        plan_from_manifest(manifest, data),
        settings,
        InMemoryResponseCache(),
        fixed_clock,
        preset=PRESET,
    )

    assert first.prompts == second.prompts
    assert first.prompts["structure"], "el conteo no calculó ningún prompt exacto"
    assert first.evaluations == len(manifest.entries)
    assert first.activations == len(manifest.entries), "el manifiesto trae activaciones confirmadas"


@pytest.mark.asyncio
async def test_a_manifest_run_sees_the_window_the_manifest_recorded() -> None:
    """El digest de velas de cada evaluación es el que la selección escribió.

    Es lo que hace reproducible una corrida aunque el histórico se vuelva a
    descargar: si el exchange revisa una vela, esto deja de coincidir y se ve al
    empezar en vez de en una tabla que ya no compara con la anterior.
    """
    manifest, data = two_symbol_manifest()
    settings = ablation_settings()
    backend = FakeLLM()
    full = next(arm for arm in ARMS if arm.name == "full")

    async def once() -> list[EvaluationRecord]:
        journal = InMemoryJournal()
        await run_arm(
            full,
            plan_from_manifest(manifest, data),
            settings,
            HEALTHY,
            InMemoryResponseCache(),
            fixed_clock,
            QuotaLedger(settings.quota_window, fixed_clock),
            fill_with={Backend.OLLAMA: backend},
            horizon=3,
            preset=PRESET,
            journal=journal,
        )
        return journal.records

    first, second = await once(), await once()
    expected = [entry.candles_digest for entry in manifest.entries]

    assert [record.snapshot.candles_digest for record in first if record.snapshot] == expected
    assert [record.snapshot.candles_digest for record in second if record.snapshot] == expected


@pytest.mark.asyncio
async def test_a_manifest_run_covers_every_symbol_it_declares() -> None:
    """Una corrida que se quedara en un símbolo mediría un régimen y no la arquitectura."""
    manifest, data = two_symbol_manifest()
    settings = ablation_settings()
    journal = InMemoryJournal()

    await run_arm(
        next(arm for arm in ARMS if arm.name == "solo"),
        plan_from_manifest(manifest, data),
        settings,
        HEALTHY,
        InMemoryResponseCache(),
        fixed_clock,
        QuotaLedger(settings.quota_window, fixed_clock),
        fill_with={Backend.OLLAMA: FakeLLM()},
        horizon=3,
        preset=PRESET,
        journal=journal,
    )

    assert {record.symbol for record in journal.records} == set(manifest.by_symbol)


@pytest.mark.asyncio
async def test_a_manifest_whose_history_moved_refuses_to_run() -> None:
    """Verificar al construir el plan es verificar antes de la primera llamada."""
    manifest, data = two_symbol_manifest()
    touched = manifest.entries[0]
    rows = [list(row) for row in data[touched.symbol]]
    rows[touched.index][4] += 5.0
    data[touched.symbol] = rows

    with pytest.raises(SelectionError, match="la ventana cambió"):
        plan_from_manifest(manifest, data)


# ───────────────────────────────── Conteo previo (dry-run) ────────────────────────────────────────


def dry_run_config(evaluations: int = 8) -> ReplaySettings:
    """Ventana corta del histórico sintético, la misma para conteo y corrida."""
    return ReplaySettings(
        symbol="BTC/USDT",
        timeframe="1h",
        warmup_bars=PRESET.min_bars,
        max_evaluations=evaluations,
    )


@pytest.mark.asyncio
async def test_the_dry_run_counts_without_calling_anyone() -> None:
    """Cuenta la factura sin abrirla.

    La prueba de que no llama es la caché vacía: el router del conteo lleva
    `CacheOnlyBackend` en todas las ranuras, así que una sola llamada levantaría
    `ReplayCacheMissError` y esto fallaría en vez de pasar en silencio.
    """
    cache = InMemoryResponseCache()
    report = await dry_run(
        ARMS,
        plan_from_history(synthetic_rows(), dry_run_config()),
        ablation_settings(),
        cache,
        fixed_clock,
        preset=PRESET,
    )

    assert report.evaluations == 8
    assert report.activations > 0
    assert len(cache) == 0, "el conteo escribió en la caché"
    assert report.rows


@pytest.mark.asyncio
async def test_the_dry_run_agrees_with_what_the_run_actually_spends() -> None:
    """El número exacto tiene que ser el número, no una estimación cercana.

    Es lo que justifica el conteo: si el `structure` previsto y el gastado no
    coinciden, la tabla de presupuesto no sirve para decidir si se lanza la
    ablación.
    """
    rows = synthetic_rows()
    settings = ablation_settings()
    config = dry_run_config()
    clock = fixed_clock
    full = next(arm for arm in ARMS if arm.name == "full")

    report = await dry_run(
        [full],
        plan_from_history(rows, config),
        settings,
        InMemoryResponseCache(),
        clock,
        preset=PRESET,
    )
    result = await run_arm(
        full,
        plan_from_history(rows, config),
        settings,
        HEALTHY,
        InMemoryResponseCache(),
        clock,
        QuotaLedger(settings.quota_window, clock),
        fill_with={Backend.OLLAMA: FakeLLM()},
        preset=PRESET,
        journal=InMemoryJournal(),
    )

    assert report.activations == result.summary.activated
    for node, role in (
        ("structure", AgentRole.STRUCTURE),
        ("momentum", AgentRole.MOMENTUM),
        ("volume", AgentRole.VOLUME),
    ):
        row = next(item for item in report.rows if item.node == node)
        assert row.exact
        assert row.calls == result.summary.calls[(role, Backend.OLLAMA)].attempts


@pytest.mark.asyncio
async def test_the_dry_run_shows_what_a_later_arm_will_not_have_to_pay() -> None:
    """Si el conteo ignorase la caché compartida, cuadruplicaría la etapa técnica."""
    report = await dry_run(
        [arm for arm in ARMS if arm.name in ("full", "no_debate")],
        plan_from_history(synthetic_rows(), dry_run_config()),
        ablation_settings(),
        InMemoryResponseCache(),
        fixed_clock,
        preset=PRESET,
    )

    first = next(row for row in report.rows if row.arm == "full" and row.node == "structure")
    second = next(row for row in report.rows if row.arm == "no_debate" and row.node == "structure")

    assert first.to_pay == first.calls
    assert first.cached == 0
    assert second.to_pay == 0
    assert second.cached == second.calls


@pytest.mark.asyncio
async def test_the_dry_run_reads_a_cache_that_is_already_warm() -> None:
    """Reejecutar sobre una caché llena tiene que contar cero a pagar, no todo otra vez."""
    rows = synthetic_rows()
    settings = ablation_settings()
    config = dry_run_config()
    clock = fixed_clock
    cache = InMemoryResponseCache()
    full = next(arm for arm in ARMS if arm.name == "full")

    await run_arm(
        full,
        plan_from_history(rows, config),
        settings,
        HEALTHY,
        cache,
        clock,
        QuotaLedger(settings.quota_window, clock),
        fill_with={Backend.OLLAMA: FakeLLM()},
        preset=PRESET,
        journal=InMemoryJournal(),
    )
    report = await dry_run(
        [full], plan_from_history(rows, config), settings, cache, clock, preset=PRESET
    )

    row = next(item for item in report.rows if item.node == "structure")
    assert row.to_pay == 0
    assert row.cached == row.calls


@pytest.mark.asyncio
async def test_the_dry_run_adds_up_the_quota_the_shared_ledger_will_see() -> None:
    """El agregado no es informativo: es la precondición para lanzar la corrida.

    Los seis brazos comparten un `QuotaLedger`, así que esta suma es exactamente lo
    que ese contador contará y lo que el proveedor cobrará. Un desglose por brazo
    diría seis veces «cabe» sobre un presupuesto que solo existe una vez, y el sexto
    brazo se encontraría la ventana agotada a mitad de una corrida de horas.
    """
    tight = with_decider_budget(ablation_settings(), 3)
    report = await dry_run(
        ARMS,
        plan_from_history(synthetic_rows(), dry_run_config()),
        tight,
        InMemoryResponseCache(),
        fixed_clock,
        preset=PRESET,
    )

    decider = next(line for line in report.quota if line.role is AgentRole.DECIDER)
    modelled = [arm for arm in ARMS if arm.variant not in BASELINE_VARIANTS]
    assert len(modelled) == 6, "las cuatro líneas base no llaman al decisor y no suman"
    assert decider.calls == report.activations * len(modelled)
    assert not decider.fits, "seis brazos contra tres llamadas por ventana tienen que no caber"


@pytest.mark.parametrize("variant", list(PipelineVariant))
def test_the_count_knows_every_node_that_spends_quota(variant: PipelineVariant) -> None:
    """Los nodos se leen del grafo compilado, así que uno nuevo no puede pasar inadvertido.

    Las cifras son la razón de ser de la ablación: seis llamadas contra una. Si el
    conteo las tomara de una tabla escrita a mano, añadir un nodo con modelo a una
    variante subestimaría la factura sin que nada avisara.
    """
    expected = {
        PipelineVariant.FULL: 6,
        PipelineVariant.NO_DEBATE: 4,
        PipelineVariant.BULL_ONLY: 5,
        PipelineVariant.SOLO: 1,
        **dict.fromkeys(BASELINE_VARIANTS, 0),
    }
    nodes = llm_nodes(variant)

    assert len(nodes) == expected[variant]
    assert set(nodes) == set(build_graph(variant).get_graph().nodes) - DETERMINISTIC_NODES


def test_the_rendered_count_separates_the_exact_from_the_upper_bound() -> None:
    """Sumar las dos como si fueran lo mismo invitaría a leer la suma como la factura."""
    report = DryRunReport(
        evaluations=10,
        activations=4,
        prepare_failures=0,
        rows=(
            DryRunRow(
                arm="full",
                node="structure",
                role=AgentRole.STRUCTURE,
                backend=Backend.OLLAMA,
                model="remoto-structure",
                calls=4,
                exact=True,
                cached=1,
                to_pay=3,
            ),
            DryRunRow(
                arm="full",
                node="decide",
                role=AgentRole.DECIDER,
                backend=Backend.OLLAMA,
                model="remoto-decider",
                calls=4,
                exact=False,
            ),
        ),
    )
    rendered = render_dry_run(report)

    assert "| 4 | 1 | 3 |" in rendered
    assert "| ≤ 4 | — | — |" in rendered


def test_the_run_directory_records_the_billing_the_settings_declare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Sin la forma de pago escrita, el consumo de la corrida no se puede expresar después."""
    paying = ablation_settings().model_copy(update={"billing": Billing.PAYG})
    monkeypatch.setattr(ablation, "load_settings", lambda _path: paying)
    monkeypatch.setattr(ablation, "build_backends", lambda _settings: {Backend.OLLAMA: FakeLLM()})
    directory = tmp_path / "corrida"

    code = main(
        [
            "--history",
            str(REAL_HISTORY),
            "--evaluations",
            "3",
            "--arms",
            "solo",
            "--fill",
            "--cache",
            str(tmp_path / "cache"),
            "--out",
            str(tmp_path / "tabla.md"),
            "--journal-dir",
            str(directory),
        ]
    )

    assert code == 0, capsys.readouterr().err
    assert read_meta(directory).billing is Billing.PAYG


# ───────────────────────── Estimación del pool en el conteo previo ────────────────────────────────


PAGE_LABEL = "estimación de la página, no medida"


def quota_report(billing: Billing = Billing.GO) -> DryRunReport:
    """Dos pares con estimado de la página y uno sin él."""
    return DryRunReport(
        evaluations=140,
        activations=140,
        prepare_failures=0,
        billing=billing,
        quota=(
            QuotaLine(
                role=AgentRole.DECIDER,
                model="glm-5.2",
                calls=840,
                quota=840.0,
                per_window=880,
                page_estimate=880,
            ),
            QuotaLine(
                role=AgentRole.MOMENTUM,
                model="deepseek-v4-flash",
                calls=420,
                quota=840.0,
                per_window=63_300,
                page_estimate=13_000,
            ),
        ),
    )


def test_the_dry_run_prices_the_pool_with_the_page_estimate_and_says_so() -> None:
    """840 / 880 = 95.45% de una ventana de 5 h para el decisor; 420 / 13 000 = 3.23% momentum.

    Se suman porque el pool es uno: 98.69%. La etiqueta no es decoración: sin tokens medidos
    esto es lo que dice la página, y presentarlo sin esa palabra lo haría pasar por una medida.
    """
    text = render_dry_run(quota_report())
    assert PAGE_LABEL in text
    section = text.split(PAGE_LABEL)[1].splitlines()
    decider = next(line for line in section if line.startswith("| decider | `glm-5.2`"))
    assert "95.45%" in decider
    momentum = next(line for line in section if "deepseek-v4-flash" in line)
    assert "3.23%" in momentum
    total = next(line for line in section if line.startswith("| **total**"))
    assert "≤ 98.69%" in total


def test_a_pair_the_page_does_not_estimate_leaves_the_total_undetermined() -> None:
    report = quota_report()
    extra = QuotaLine(
        role=AgentRole.BULL, model="modelo-raro", calls=10, quota=10.0, per_window=100
    )
    text = render_dry_run(report.model_copy(update={"quota": (*report.quota, extra)}))
    section = text.split(PAGE_LABEL)[1].splitlines()
    assert "sin estimado" in next(line for line in section if "modelo-raro" in line)
    total = next(line for line in section if line.startswith("| **total**"))
    assert "no determinado" in total


def test_under_pay_as_you_go_the_estimate_says_there_is_no_pool_and_no_dollars_yet() -> None:
    text = render_dry_run(quota_report(Billing.PAYG))
    assert PAGE_LABEL not in text.split("Cuota agregada")[0]
    assert "sin pool" in text
    assert "no determinado" in text


@pytest.mark.asyncio
async def test_the_page_estimate_per_pair_is_filled_from_the_settings_table() -> None:
    """El dry-run no copia las cifras: las lee de `settings.pricing`."""
    prices = DEFAULT_PRICING.model_copy(update={"page_estimates": {"remoto-decider": 200}})
    tuned = ablation_settings().model_copy(update={"pricing": prices})
    report = await dry_run(
        ARMS,
        plan_from_history(synthetic_rows(), dry_run_config()),
        tuned,
        InMemoryResponseCache(),
        fixed_clock,
        preset=PRESET,
    )
    estimates = {line.role: line.page_estimate for line in report.quota}
    assert estimates[AgentRole.DECIDER] == 200
    assert estimates[AgentRole.STRUCTURE] is None
    assert report.billing is Billing.GO


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
    results = [build_arm_result(arm, [], {}, wall_clock_seconds=1.0) for arm in ARMS[:3]]
    report = render_report(results)

    for arm in ARMS[:3]:
        assert f"`{arm.name}`" in report
        assert arm.question in report


def test_the_table_publishes_the_weighted_rate_and_names_the_worst_pair() -> None:
    """La columna que la primera tabla llamaba «fallo validación» era el máximo de un par.

    Un brazo con 1 inválida de 10 respondidas salía como 50% porque su peor par
    tenía 1 de 2. Ahora la tasa es la del brazo, con sus dos números, y el peor par
    va aparte con su nombre. La latencia es la mediana de las 12 llamadas vivas
    —todas a 10 ms— y no el reloj de pared entre intentos, que aquí daría 5000 ms.
    """
    records = [metric_record(calls=unbalanced_calls())]
    report = render_report([build_arm_result(ARMS[0], records, {}, wall_clock_seconds=60.0)])

    assert "1/10 (10%)" in report
    assert "bull/ollama 1/2 (50%)" in report
    assert "| 12, 10 ms |" in report
    assert "| 10.0 | 2.0 |" in report, "8 válidas y 2 rechazos remotos; 2 locales"


def test_an_arm_served_from_cache_reports_no_latency_and_no_rate() -> None:
    """Sin llamadas vivas no hay latencia ni tasa: ni cero ni el reloj de pared."""
    report = render_report([build_arm_result(ARMS[0], [], {}, wall_clock_seconds=60.0)])

    assert "no determinado: sin llamadas vivas" in report
    assert "no determinado: 0 respondidas" in report


def test_the_report_survives_an_empty_run() -> None:
    """Sin corridas la tabla lo dice, en vez de fingir ceros."""
    assert "Sin corridas" in render_report([])


# ───────────────────────────────────── Columnas de la comparación ─────────────────────────────────
# Las columnas nuevas de la tabla no calculan nada: leen las funciones puras de `metrics`. Lo que
# estas pruebas vigilan es que cada cifra salga del conjunto correcto —los aciertos de caché
# cuentan como llamadas y no como cuota— y que ninguna tasa salga sin los dos números que la hacen.

CALLS = "llamadas"
CACHE_HITS = "aciertos de caché"
LIVE_CALLS = "vivas"
QUOTA_REMOTE = "cuota remota"
QUOTA_LOCAL = "cuota local"
LATENCY_MEDIAN = "latencia viva (n, mediana)"
LATENCY_MEAN = "latencia media"
LATENCY_P95 = "latencia p95"
VALIDATION = "fallo validación (inválidas/respondidas)"
DECIDED = "decididas"
ACTIONABLE = "accionables"
VETOED = "vetadas"
VETOES_BY_RULE = "vetos por regla"
ORDERS = "órdenes"
UNEXECUTED = "aprobadas sin orden"
LOST_CAUSE = "causa"
LOST_COUNT = "evaluaciones"
RESOLVED = "órdenes resueltas"
INVALIDATED = "invalidadas"
WINS = "aciertos (IC95% Wilson)"
RETURN = "retorno medio ± EE (n)"


def agreement_all(baseline: str = "full") -> str:
    """Cabecera de la coincidencia sobre todas las evaluaciones."""
    return f"coincidencia con `{baseline}` (todas las evals)"


def agreement_both(baseline: str = "full") -> str:
    """Cabecera de la coincidencia solo donde ambos brazos decidieron."""
    return f"coincidencia con `{baseline}` (ambos decidieron)"


def parse_tables(report: str) -> list[tuple[list[str], list[list[str]]]]:
    """Cada bloque de líneas `|` del informe: su cabecera y sus filas, ya partidas en celdas."""
    tables: list[tuple[list[str], list[list[str]]]] = []
    block: list[list[str]] = []
    for line in [*report.splitlines(), ""]:
        if line.startswith("|"):
            block.append([cell.strip() for cell in line.strip("|").split("|")])
        elif block:
            tables.append((block[0], block[2:]))
            block = []
    return tables


def cells(report: str, column: str, arm: str = "full") -> list[str]:
    """Celdas de un brazo en la primera tabla que lleva esa columna."""
    for header, body in parse_tables(report):
        if column in header:
            index = header.index(column)
            return [row[index] for row in body if row[0] == f"`{arm}`"]
    raise AssertionError(f"ninguna tabla lleva la columna {column!r}")


def cell(report: str, column: str, arm: str = "full") -> str:
    """La única celda de un brazo en esa columna."""
    [value] = cells(report, column, arm)
    return value


def hits_and_live_calls() -> tuple[LLMCall, ...]:
    """Siete intentos: cuatro vivos y tres aciertos de caché, remotos y locales.

    Vivos: 100, 200 y 600 ms remotos de peso 2, y 300 ms local de peso 1. Los tres
    aciertos de caché (dos remotos de peso 2, uno local de peso 1) registran 0 ms.
    Contados como cuota darían 10.0 remota y 2.0 local; contados en la latencia,
    una media de 171 ms con n = 7. Ninguna de las dos es lo que costó la corrida.
    """
    return (
        timed(100.0, weight=2.0),
        timed(200.0, weight=2.0),
        timed(600.0, weight=2.0),
        timed(0.0, weight=2.0, cache_hit=True),
        timed(0.0, weight=2.0, cache_hit=True),
        timed(300.0, backend=Backend.OLLAMA),
        timed(0.0, backend=Backend.OLLAMA, cache_hit=True),
    )


def test_cache_hits_count_as_calls_but_not_as_quota() -> None:
    """Corrida sintética: los aciertos de caché están en las llamadas y fuera de la cuota."""
    result = build_arm_result(
        ARMS[0], [metric_record(calls=hits_and_live_calls())], {}, wall_clock_seconds=1.0
    )
    report = render_report([result])

    assert (result.attempts.total, result.attempts.cache_hits, result.attempts.live) == (7, 3, 4)
    assert cell(report, CALLS) == "7"
    assert cell(report, CACHE_HITS) == "3/7 (43%)"
    assert cell(report, LIVE_CALLS) == "4/7 (57%)"
    assert cell(report, QUOTA_REMOTE) == "6.0", "tres vivas de peso 2; los hits no suman"
    assert cell(report, QUOTA_LOCAL) == "1.0", "una viva de peso 1; el hit no suma"


def test_latency_ignores_cache_hits() -> None:
    """Las cuatro vivas dan n = 4, media 300, mediana 250 y p95 600 ms; los hits no entran."""
    result = build_arm_result(
        ARMS[0], [metric_record(calls=hits_and_live_calls())], {}, wall_clock_seconds=1.0
    )
    report = render_report([result])

    assert cell(report, LATENCY_MEDIAN) == "4, 250 ms"
    assert cell(report, LATENCY_MEAN) == "300 ms"
    assert cell(report, LATENCY_P95) == "600 ms"


def test_an_arm_without_live_calls_has_no_mean_and_no_p95_instead_of_zero() -> None:
    """Todo desde la caché: la latencia no es 0 ms, es que no hay medida."""
    records = [metric_record(calls=(timed(0.0, cache_hit=True),))]
    report = render_report([build_arm_result(ARMS[0], records, {}, wall_clock_seconds=1.0)])

    assert cell(report, LATENCY_MEAN) == "no determinado: sin llamadas vivas"
    assert cell(report, LATENCY_P95) == "no determinado: sin llamadas vivas"


def traded_order() -> OrderIntent:
    """Orden de papel que el ejecutor sí emitió."""
    return OrderIntent(
        symbol="BTC/USDT",
        side=Action.BUY,
        size_fraction=0.1,
        reference_price=100.0,
        invalidation_price=95.0,
        mode=ExecutionMode.PAPER,
    )


def funnel_records() -> list[EvaluationRecord]:
    """Cinco evaluaciones: una orden, un veto, un hold, una sin cuota y una con el gate cerrado."""
    approved = RiskVerdict(approved=True, final_size_fraction=0.1)
    vetoed = RiskVerdict(
        approved=False, final_size_fraction=0.0, veto_rule="cooldown", veto_reason="quedan 30 min"
    )
    return [
        metric_record(proposal=proposal(Action.BUY, 95.0), risk=approved, order=traded_order()),
        metric_record(proposal=proposal(Action.SELL, 105.0), risk=vetoed),
        metric_record(proposal=proposal(Action.HOLD)),
        metric_record(errors=failed("decide", "cuota agotada: nada cabe en la ventana")),
        metric_record(activation=CLOSED_GATE),
    ]


def test_the_funnel_columns_carry_the_denominator_of_their_stage() -> None:
    """5 evals, 3 decididas, 2 accionables, 1 vetada y 1 orden: cada etapa sobre la anterior."""
    report = render_report(
        [build_arm_result(ARMS[0], funnel_records(), {}, wall_clock_seconds=1.0)]
    )

    assert cell(report, DECIDED) == "3/5 (60%)"
    assert cell(report, ACTIONABLE) == "2/3 (67%)"
    assert cell(report, VETOED) == "1/2 (50%)"
    assert cell(report, VETOES_BY_RULE) == "cooldown 1"
    assert cell(report, ORDERS) == "1/2 (50%)"
    assert cell(report, UNEXECUTED) == "0/2 (0%)"


def test_lost_evaluations_are_listed_by_node_and_cause_over_all_evaluations() -> None:
    """La que se quedó sin cuota y la del gate cerrado, cada una con su causa y sobre las 5."""
    report = render_report(
        [build_arm_result(ARMS[0], funnel_records(), {}, wall_clock_seconds=1.0)]
    )

    (header, body) = next(table for table in parse_tables(report) if LOST_CAUSE in table[0])
    lost = {(row[1], row[2]): row[3] for row in body if row[0] == "`full`"}

    assert header[:2] == ["brazo", "nodo"]
    assert lost == {
        ("decide", "quota"): "1/5 (20%)",
        ("sin error", "gate_closed"): "1/5 (20%)",
    }


def test_an_arm_that_lost_nothing_says_so_with_its_denominator() -> None:
    """Cero de cinco es una medida; una fila ausente obligaría a adivinar si se midió."""
    records = [metric_record(proposal=proposal(Action.HOLD)) for _ in range(5)]
    report = render_report([build_arm_result(ARMS[0], records, {}, wall_clock_seconds=1.0)])

    assert cells(report, LOST_COUNT) == ["0/5 (0%)"]


def decided(*actions: Action | None) -> list[EvaluationRecord]:
    """Una evaluación por acción; `None` es una que no llegó a decidir."""
    return [
        metric_record(proposal=None if action is None else proposal(action, 95.0))
        for action in actions
    ]


def test_agreement_counts_publishes_the_numbers_behind_agreement() -> None:
    """`agreement` no cambia: es el mismo cociente, ahora con sus dos números."""
    left = (Action.BUY, Action.HOLD, None, Action.SELL)
    right = (Action.BUY, Action.SELL, None, Action.SELL)

    counts = agreement_counts(left, right)

    assert counts is not None
    assert (counts.numerator, counts.denominator) == (3, 4)
    assert counts.rate == agreement(left, right)
    assert agreement_counts(left, right[:3]) is None
    assert agreement_counts((), ()) is None


def test_agreement_among_decided_leaves_out_where_either_did_not_decide() -> None:
    """Cinco evaluaciones; en dos alguien no decidió. De las tres comunes, coinciden dos.

    Contar el «no decidí» de ambos como acuerdo da 3 de 5, y es un acuerdo real en
    la tabla de todas las evaluaciones; aquí la pregunta es qué hacen los decisores
    cuando los dos deciden.
    """
    left = (Action.BUY, None, Action.HOLD, None, Action.SELL)
    right = (Action.BUY, None, Action.SELL, Action.HOLD, Action.SELL)

    both = agreement_decided(left, right)

    assert both is not None
    assert (both.numerator, both.denominator) == (2, 3)
    assert agreement_decided(left, right[:4]) is None


def test_agreement_among_decided_without_common_decisions_has_no_rate() -> None:
    """Ninguna evaluación con ambos decidiendo: 0 de 0 no es 0%."""
    both = agreement_decided((Action.BUY, None), (None, Action.BUY))

    assert both is not None
    assert both.denominator == 0
    assert both.rate is None


def test_the_table_publishes_both_agreements_with_their_own_denominators() -> None:
    """`full` decide (buy, hold, nada) y `solo` (buy, sell, nada): 2/3 en total, 1/2 decididas."""
    full = build_arm_result(
        ARMS[0], decided(Action.BUY, Action.HOLD, None), {}, wall_clock_seconds=1.0
    )
    solo = build_arm_result(
        ARMS[3], decided(Action.BUY, Action.SELL, None), {}, wall_clock_seconds=1.0
    )
    report = render_report([full, solo])

    assert cell(report, agreement_all(), "full") == "3/3 (100%)"
    assert cell(report, agreement_both(), "full") == "2/2 (100%)"
    assert cell(report, agreement_all(), "solo") == "2/3 (67%)"
    assert cell(report, agreement_both(), "solo") == "1/2 (50%)"


def outcome_records_and_history() -> tuple[list[EvaluationRecord], list[list[float]]]:
    """Tres órdenes a horizonte 3: largo +0.4%, largo +0.2% y corto -0.4%. Ninguna toca su stop.

    El retorno medio es +0.0667% y el error estándar 0.240%. Redondeado a entero el
    primero sale «0%» —ningún lector sabría si la media es positiva— y con dos
    decimales sale +0.07%.
    """
    history = candle_rows(
        [
            (100.5, 99.9, 100.0),
            (100.5, 99.9, 100.0),
            (100.5, 99.9, 100.0),
            (100.5, 99.9, 100.4),
            (100.5, 99.9, 100.2),
        ]
    )
    records = [
        record_at(0, Action.BUY, 100.0, 95.0),
        record_at(1, Action.BUY, 100.0, 95.0),
        record_at(0, Action.SELL, 100.0, 105.0),
    ]
    return records, history


def test_wins_publish_their_count_and_the_wilson_interval() -> None:
    """2 aciertos de 3 resueltas: 67%, con un intervalo al 95% de 21% a 94%."""
    records, history = outcome_records_and_history()
    report = render_report(
        [
            build_arm_result(
                ARMS[0], records, {"BTC/USDT": history}, wall_clock_seconds=1.0, horizon=3
            )
        ]
    )

    assert cell(report, RESOLVED) == "3/3 (100%)"
    assert cell(report, INVALIDATED) == "0/3 (0%)"
    assert cell(report, WINS) == "2/3 (67%) IC95 [21%, 94%]"


def test_the_mean_return_keeps_its_decimals_and_shows_its_error_and_n() -> None:
    """+0.07% ± 0.24% con n = 3: ni redondeado a entero ni sin error ni sin tamaño de muestra."""
    records, history = outcome_records_and_history()
    result = build_arm_result(
        ARMS[0], records, {"BTC/USDT": history}, wall_clock_seconds=1.0, horizon=3
    )
    report = render_report([result])

    assert cell(report, RETURN) == "+0.07% ± 0.24% (EE), n=3"
    assert result.returns is not None
    assert result.returns.mean == pytest.approx(result.outcomes.mean_return)
    assert result.returns.n == result.outcomes.resolved


def test_a_single_resolved_order_has_a_mean_and_says_why_it_has_no_error() -> None:
    """Con n = 1 no hay dispersión que estimar: se dice, no se publica un cero."""
    records, history = outcome_records_and_history()
    report = render_report(
        [
            build_arm_result(
                ARMS[0], records[:1], {"BTC/USDT": history}, wall_clock_seconds=1.0, horizon=3
            )
        ]
    )

    assert cell(report, RETURN) == "+0.40% ± no determinado (EE: una sola orden), n=1"


def test_without_orders_there_are_no_wins_and_no_return() -> None:
    """Sin órdenes: ni 0% de aciertos ni retorno 0, que serían medidas que nadie hizo."""
    report = render_report([build_arm_result(ARMS[0], [], {}, wall_clock_seconds=1.0)])

    assert cell(report, WINS).startswith("no determinado")
    assert cell(report, RETURN).startswith("no determinado")
    assert cell(report, RESOLVED).startswith("no determinado")


def test_no_rate_in_the_table_comes_without_its_denominator() -> None:
    """Cada tasa trae `k/n`, o dice `no determinado`; nunca un porcentaje a solas.

    Se recorre la tabla de tres corridas —una con embudo, una con resultados y una
    vacía— y cada celda de las columnas que son una fracción de algo se comprueba
    por su forma. Quitar el denominador de cualquiera hace fallar la que lo pierda.
    """
    records, history = outcome_records_and_history()
    results = [
        build_arm_result(ARMS[0], funnel_records(), {}, wall_clock_seconds=1.0),
        build_arm_result(
            ARMS[1], records, {"BTC/USDT": history}, wall_clock_seconds=1.0, horizon=3
        ),
        build_arm_result(ARMS[3], [], {}, wall_clock_seconds=1.0),
    ]
    report = render_report(results)

    fraction_columns = [
        CACHE_HITS,
        LIVE_CALLS,
        VALIDATION,
        DECIDED,
        ACTIONABLE,
        VETOED,
        ORDERS,
        UNEXECUTED,
        agreement_all(),
        agreement_both(),
        LOST_COUNT,
        RESOLVED,
        INVALIDATED,
        WINS,
        POSITIONS,
    ]
    for column in fraction_columns:
        for result in results:
            for value in cells(report, column, result.arm.name):
                assert re.search(r"\d+/\d+", value) or value.startswith("no determinado"), (
                    f"{column!r} de {result.arm.name}: {value!r}"
                )


@pytest.mark.asyncio
async def test_the_columns_add_up_over_a_real_replay() -> None:
    """Sobre el grafo real: lo que cuenta una columna es lo que cuenta la otra.

    Las órdenes de `risk_flow` son las de `summarise`, accionables más holds son las
    decididas, y las perdidas son las evaluaciones que no decidieron. Tres
    identidades que valen mientras ninguna etapa cuente algo que otra no ve.
    """
    harness = Harness()
    records = await run(harness, synthetic_rows(), fill=True, evaluations=12)
    result = build_arm_result(ARMS[0], records, {}, wall_clock_seconds=1.0)

    assert result.summary.decided > 0
    assert result.risk.orders == result.summary.traded
    assert result.risk.actionable + result.risk.holds == result.summary.decided
    assert sum(result.undecided.values()) == result.summary.evaluations - result.summary.decided
    assert result.attempts.total == sum(len(record.calls) for record in records)


# ─────────────────────────────────────────── Líneas base ──────────────────────────────────────────
# Cuatro brazos que no llaman a nadie: comprar siempre, vender siempre, un azar uniforme y una
# regla de tendencia fija. Todos emiten un `Proposal` y recorren la misma cola determinista que
# los demás, con el stop común. Lo que estas pruebas vigilan es que cuesten cero por construcción
# y que lo único que los distinga de un brazo con modelos sea quién decide.

BASELINE_VARIANTS = (
    PipelineVariant.ALWAYS_BUY,
    PipelineVariant.ALWAYS_SELL,
    PipelineVariant.RANDOM_UNIFORM,
    PipelineVariant.RULE_TREND,
)
BASELINE_ARMS = [arm for arm in ARMS if arm.variant in BASELINE_VARIANTS]
TAIL = {("risk", "execute"), ("execute", "journal"), ("journal", "__end__")}


def baseline_plan(evaluations: int = 40) -> ReplayPlan:
    """Ventana sintética de 40 evaluaciones, la misma para todas las pruebas de líneas base."""
    return plan_from_history(synthetic_rows(), dry_run_config(evaluations))


async def run_baseline(
    arm: AblationArm, plan: ReplayPlan | None = None
) -> tuple[ArmResult, list[EvaluationRecord], QuotaLedger]:
    """Corre una línea base en modo solo-caché: sin `fill_with` y con la caché vacía.

    Una sola llamada al router levantaría `ReplayCacheMissError`, así que si esto
    termina es que el brazo no llamó a nadie.
    """
    settings = ablation_settings()
    ledger = QuotaLedger(settings.quota_window, fixed_clock)
    journal = InMemoryJournal()
    result = await run_arm(
        arm,
        plan if plan is not None else baseline_plan(),
        settings,
        HEALTHY,
        InMemoryResponseCache(),
        fixed_clock,
        ledger,
        journal,
        horizon=3,
        preset=PRESET,
    )
    return result, journal.records, ledger


@pytest.mark.parametrize("variant", BASELINE_VARIANTS)
def test_a_baseline_has_one_decision_node_and_no_model_node(variant: PipelineVariant) -> None:
    """Preparar, decidir sin modelo y la cola: nada más. Y el conteo previo no le ve coste."""
    nodes = set(build_graph(variant).get_graph().nodes)

    assert nodes == {
        "__start__",
        "__end__",
        "prepare",
        f"decide_{variant.value}",
        "risk",
        "execute",
        "journal",
    }
    assert llm_nodes(variant) == ()


@pytest.mark.parametrize("variant", list(PipelineVariant))
def test_the_deterministic_tail_is_the_same_in_every_variant(variant: PipelineVariant) -> None:
    """Riesgo → ejecución → journal es idéntico en las diez, y a `risk` solo llega un decisor.

    Regla permanente 2 con otra cara: una línea base que saltara el gate abriría un
    camino al mercado que no pasa por él, y la comparación mediría el arnés.
    """
    edges = {(edge.source, edge.target) for edge in build_graph(variant).get_graph().edges}

    assert edges >= TAIL
    assert {target for source, target in edges if source == "risk"} == {"execute"}
    assert {source for source, target in edges if target == "execute"} == {"risk"}
    assert {source for source, target in edges if target == "risk"} <= {
        "decide",
        "decide_single_desk",
        "decide_without_debate",
        "decide_solo",
        *(f"decide_{item.value}" for item in BASELINE_VARIANTS),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("arm", BASELINE_ARMS, ids=lambda arm: arm.name)
async def test_a_baseline_spends_nothing_and_never_reaches_the_cache(arm: AblationArm) -> None:
    """`calls == []` en cada registro, cuota cero y el contador compartido sin tocar."""
    result, records, ledger = await run_baseline(arm)
    settings = ablation_settings()

    assert result.summary.decided > 0, "el gate abrió y la línea base decidió"
    assert all(record.calls == () for record in records)
    assert result.attempts.total == 0
    assert result.summary.quota_used == 0.0
    assert result.quota == QuotaSplit(remote=0.0, local=0.0)
    assert result.summary.calls == {}
    assert all(
        ledger.used(role, settings.role_config(role).primary.model) == 0.0 for role in AgentRole
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "side"), [("always_buy", Action.BUY), ("always_sell", Action.SELL)]
)
async def test_the_fixed_lines_take_their_side_at_every_activation(name: str, side: Action) -> None:
    """Todo lo que deciden es su lado, y deciden en cada activación y solo en ellas."""
    arm = next(item for item in ARMS if item.name == name)

    result, records, _ = await run_baseline(arm)

    activated = [r for r in records if r.activation is not None and r.activation.should_run]
    assert activated, "sin activaciones la prueba no mide nada"
    assert {action for action in result.actions if action is not None} == {side}
    assert sum(action is not None for action in result.actions) == len(activated)


@pytest.mark.asyncio
async def test_the_random_line_follows_the_seed_of_the_plan() -> None:
    """Misma semilla, mismas decisiones; otra semilla, otras. Y actúa con más de una acción."""
    arm = next(item for item in ARMS if item.name == "random_uniform")
    plan = baseline_plan()

    first, _, _ = await run_baseline(arm, plan)
    again, _, _ = await run_baseline(arm, plan)
    other, _, _ = await run_baseline(arm, plan._replace(seed=plan.seed + 1))

    assert first.actions == again.actions
    assert first.actions != other.actions
    assert len({action for action in first.actions if action is not None}) >= 2


def test_the_plan_carries_the_seed_of_its_manifest() -> None:
    """La semilla del manifiesto llega al plan; sin manifiesto, la de la selección por defecto."""
    manifest, data = two_symbol_manifest()

    assert plan_from_manifest(manifest, data).seed == manifest.seed
    assert plan_from_history(synthetic_rows(), dry_run_config()).seed == DEFAULT_SEED


@pytest.mark.asyncio
async def test_the_trend_line_decides_what_the_rule_says_about_what_the_record_kept() -> None:
    """Rehacer la regla con el cierre y los indicadores del registro da la misma acción.

    Es lo que permite auditar la línea base desde el journal: lo guardado alcanza
    para reproducir cada decisión, sin volver a calcular nada.
    """
    arm = next(item for item in ARMS if item.name == "rule_trend")

    _, records, _ = await run_baseline(arm)

    decided = [record for record in records if record.proposed is not None]
    assert decided
    for record in decided:
        assert record.snapshot is not None
        assert record.indicators is not None
        assert record.proposed is not None
        assert record.proposed.action is trend_action(
            record.indicators, record.snapshot.close, PRESET
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("arm", BASELINE_ARMS, ids=lambda arm: arm.name)
async def test_a_baseline_scores_the_same_with_its_own_stop_and_the_common_one(
    arm: AblationArm,
) -> None:
    """Las líneas base declaran el stop común, así que `OWN_STOP` y `COMMON_STOP` coinciden.

    Es la prueba de que hay una sola fórmula: si la línea base y la puntuación
    calcularan el stop por separado, esta igualdad se rompería en cuanto una de las
    dos cambiara.
    """
    plan = baseline_plan()
    _, records, _ = await run_baseline(arm, plan)

    own = score_run(records, plan.histories, 3, Scoring.OWN_STOP)
    common = score_run(records, plan.histories, 3, Scoring.COMMON_STOP)

    assert own.positions > 0, "sin posiciones la igualdad no dice nada"
    assert own.per_position == common.per_position
    assert own.per_evaluation == common.per_evaluation
    assert (own.positions, own.resolved) == (common.positions, common.resolved)


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["solo", "always_buy"])
async def test_every_arm_leaves_the_indicators_in_its_journal(name: str) -> None:
    """También los que no consolidan evidencia: sin esto no hay stop común que reconstruir."""
    arm = next(item for item in ARMS if item.name == name)
    settings = ablation_settings()
    journal = InMemoryJournal()
    await run_arm(
        arm,
        baseline_plan(12),
        settings,
        HEALTHY,
        InMemoryResponseCache(),
        fixed_clock,
        QuotaLedger(settings.quota_window, fixed_clock),
        journal,
        fill_with={Backend.OLLAMA: FakeLLM()},
        horizon=3,
        preset=PRESET,
    )

    assert journal.records
    assert all(record.indicators is not None for record in journal.records)


@pytest.mark.asyncio
async def test_the_dry_run_prices_the_baselines_at_zero() -> None:
    """El conteo previo no les encuentra ni un nodo que gaste: cero filas, cero cuota."""
    report = await dry_run(
        BASELINE_ARMS,
        baseline_plan(),
        ablation_settings(),
        InMemoryResponseCache(),
        preset=PRESET,
    )

    assert report.rows == ()
    assert report.quota == ()
    assert report.known_to_pay == 0


def test_the_command_runs_the_baselines_by_default() -> None:
    """`--arms` sin tocar incluye los diez: cuestan cero, así que no hay motivo para omitirlos."""
    arms = ablation._parse_args(["--dry-run"]).arms.split(",")

    assert arms == [arm.name for arm in ARMS]
    assert {"always_buy", "always_sell", "random_uniform", "rule_trend"} <= set(arms)


# ───────────────────────────────── Retorno según el stop (tabla) ──────────────────────────────────

POSITIONS = "posiciones resueltas"
SCORING = "puntuación"
PER_POSITION = "retorno por posición (media ± EE, n)"
PER_EVALUATION = "retorno por evaluación (media ± EE, n)"
PAIRED = "Δ por evaluación vs `solo` (IC95%)"
STOP_HISTORY: list[tuple[float, float, float]] = [
    (100, 100, 100),
    (101, 99, 101),
    (102, 100, 102),
    (103, 101, 103),
]


def buying(count: int, idle: int = 0) -> list[EvaluationRecord]:
    """`count` compras de +3% al cierre del horizonte y `idle` evaluaciones sin propuesta."""
    quiet = position_at(0, Action.BUY).model_copy(update={"order": None, "proposal": None})
    return [position_at(0, Action.BUY) for _ in range(count)] + [quiet] * idle


def scored_pair(full: list[EvaluationRecord], solo: list[EvaluationRecord]) -> str:
    """El informe de dos brazos que recorrieron el mismo número de evaluaciones."""
    histories = {"BTC/USDT": candle_rows(STOP_HISTORY)}
    return render_report(
        [
            build_arm_result(ARMS[0], full, histories, 1.0, horizon=3),
            build_arm_result(ARMS[3], solo, histories, 1.0, horizon=3),
        ]
    )


def test_each_arm_is_scored_three_ways_with_a_return_per_position_and_per_evaluation() -> None:
    """Tres filas por brazo; por posición promedia solo las compras, por evaluación también el 0."""
    report = scored_pair(buying(15, idle=15), buying(30))

    assert cells(report, SCORING, "full") == [
        "stop propio",
        f"stop común ({COMMON_STOP_DESCRIPTION})",
        "cierre del horizonte",
    ]
    assert cells(report, POSITIONS, "full") == ["15/15 (100%)"] * 3
    assert cells(report, PER_POSITION, "full") == ["+3.00% ± 0.00% (EE), n=15"] * 3
    assert cells(report, PER_EVALUATION, "full") == ["+1.50% ± 0.28% (EE), n=30"] * 3
    assert cells(report, PER_EVALUATION, "solo") == ["+3.00% ± 0.00% (EE), n=30"] * 3


def test_the_paired_difference_against_solo_comes_with_its_interval_and_n() -> None:
    """`full` ganó 3% en 15 de 30 evaluaciones y `solo` en las 30: Δ = -1.50% con su IC95%."""
    report = scored_pair(buying(15, idle=15), buying(30))

    assert cells(report, PAIRED, "full") == ["-1.50% [-2.05%, -0.95%], n=30"] * 3
    assert cells(report, PAIRED, "solo") == ["— (referencia)"] * 3


def test_the_difference_with_few_evaluations_says_why_it_has_no_interval() -> None:
    """Veinte evaluaciones no alcanzan para el 1.96: se dice cuántas hay y cuántas hacen falta."""
    report = scored_pair(buying(10, idle=10), buying(20))

    assert cells(report, PAIRED, "full")[0] == "no determinado: n=20 < 30"


def test_without_the_reference_arm_the_difference_says_so() -> None:
    """Sin `solo` en la corrida no hay contra qué emparejar."""
    histories = {"BTC/USDT": candle_rows(STOP_HISTORY)}
    report = render_report([build_arm_result(ARMS[0], buying(30), histories, 1.0, horizon=3)])

    assert cells(report, PAIRED, "full")[0] == "no determinado: no hay brazo `solo`"


def test_an_arm_without_positions_has_no_return_instead_of_zero() -> None:
    """Sin posiciones no hay retorno por posición; por evaluación sí: son 30 ceros."""
    report = scored_pair(buying(0, idle=30), buying(30))

    assert cells(report, POSITIONS, "full")[0].startswith("no determinado")
    assert cells(report, PER_POSITION, "full")[0].startswith("no determinado")
    assert cells(report, PER_EVALUATION, "full")[0] == "+0.00% ± 0.00% (EE), n=30"


def test_the_results_carry_the_three_scorings() -> None:
    """`ArmResult.scored` trae las tres, también en un brazo sin evaluaciones."""
    result = build_arm_result(ARMS[0], [], {}, wall_clock_seconds=1.0)

    assert set(result.scored) == set(Scoring)


# ─────────────────────────────── Lo que la corrida deja en disco ──────────────────────────────────
# La primera ablación completa dejó una tabla y ningún registro: los journals eran
# en memoria. Nada de lo que sigue comprueba qué decide un brazo, sino que lo que
# pasó en cada evaluación —sobre todo en las que no decidieron— sigue ahí después.

PLAN_SHA = "c" * 64


def run_meta(resumed_from: Path | None = None, plan_sha256: str = PLAN_SHA) -> RunMeta:
    """Metadatos de una corrida de prueba con los brazos que se le pidan."""
    return RunMeta(
        plan_kind=PlanKind.HISTORY,
        plan_path="tests/data/btcusdt_4h.csv",
        plan_sha256=plan_sha256,
        argv=(),
        fill=True,
        arms=("solo",),
        started_at=fixed_clock(),
        resumed_from=None if resumed_from is None else str(resumed_from),
    )


@pytest.mark.asyncio
async def test_an_aborted_evaluation_reaches_the_file_with_its_calls_and_errors(
    tmp_path: Path,
) -> None:
    """Un decisor que nunca valida: la evaluación aborta y el archivo lo cuenta todo.

    Es el caso que la tabla perdida no pudo contestar. Lo que tiene que sobrevivir
    al proceso no es que hubo un aborto, sino en qué nodo, con qué mensaje y qué
    intentos se habían pagado ya: los dos del decisor, inválidos, y los cinco
    anteriores, válidos.
    """
    settings = ablation_settings()
    path = tmp_path / "full.jsonl"
    await run_arm(
        ARMS[0],
        plan_from_history(synthetic_rows(), dry_run_config()),
        settings,
        HEALTHY,
        InMemoryResponseCache(),
        fixed_clock,
        QuotaLedger(settings.quota_window, fixed_clock),
        JsonlJournal(path),
        fill_with={Backend.OLLAMA: FakeLLM({"decider": "{}"})},
        preset=PRESET,
    )

    records = JsonlJournal(path).read_all()
    aborted = [record for record in records if record.errors]

    assert len(records) == 8, "una evaluación no llegó al archivo"
    assert aborted, "ninguna evaluación abortó: la prueba no ejerce la rama que vigila"
    for record in aborted:
        assert record.proposed is None
        assert record.errors[0].node == "decide"
        assert "sin salida válida" in record.errors[0].message
        invalid = [call for call in record.calls if not call.valid]
        assert [call.role for call in invalid] == [AgentRole.DECIDER, AgentRole.DECIDER]
        assert len(record.calls) == 7


@pytest.mark.asyncio
async def test_every_arm_writes_its_own_journal_in_the_run_directory(tmp_path: Path) -> None:
    """Un archivo por brazo, y lo que hay dentro es lo que el brazo resumió.

    Con un archivo para todos, el `run_id` —un UUID5 del instante, idéntico entre
    brazos— dejaría seis líneas indistinguibles por evaluación.
    """
    settings = ablation_settings()
    arms = [arm for arm in ARMS if arm.name in ("full", "solo")]
    results = await run_arms(
        arms,
        plan_from_history(synthetic_rows(), dry_run_config()),
        settings,
        InMemoryResponseCache(),
        fixed_clock,
        QuotaLedger(settings.quota_window, fixed_clock),
        tmp_path,
        fill_with={Backend.OLLAMA: FakeLLM()},
        preset=PRESET,
    )

    assert sorted(path.name for path in tmp_path.iterdir()) == ["full.jsonl", "solo.jsonl"]
    for arm, result in zip(arms, results, strict=True):
        written = JsonlJournal(arm_journal_path(tmp_path, arm.name)).read_all()
        assert len(written) == 8
        assert summarise(written) == result.summary


def test_a_run_directory_is_named_after_the_instant_it_started() -> None:
    """Uno por invocación: dos corridas no pueden caer en el mismo sitio."""
    started = datetime(2026, 10, 1, 17, 5, 9, tzinfo=UTC)
    assert default_journal_dir(started).as_posix() == "var/ablation/20261001T170509Z"


def test_a_run_directory_is_never_reused(tmp_path: Path) -> None:
    """Escribir sobre una corrida anterior mezclaría dos pasadas en los mismos archivos."""
    open_run_directory(tmp_path / "corrida", run_meta())

    assert read_meta(tmp_path / "corrida") == run_meta()
    with pytest.raises(ConfigError, match="ya contiene"):
        open_run_directory(tmp_path / "corrida", run_meta())


def remote_decider_settings(per_window: int) -> Settings:
    """El decisor en un backend remoto, con la ventana que la prueba le concede.

    Hace falta uno remoto de verdad: la siembra ignora el backend local, que es
    donde viven todos los modelos de `ablation_settings()`.
    """
    base = ablation_settings()
    roles = dict(base.roles) | {
        AgentRole.DECIDER: RoleConfig(
            primary=ModelChoice(
                backend=Backend.OPENAI,
                model="remoto-decider",
                family="fam-decider",
                structured_output=StructuredOutputMode.JSON_SCHEMA,
                quota_per_window=per_window,
            )
        )
    }
    return load_settings(
        roles=roles, ollama={"host": "http://localhost:11434"}, openai={"api_key": "sk-test"}
    )


@pytest.mark.asyncio
async def test_resuming_seeds_the_ledger_with_what_the_previous_run_spent(tmp_path: Path) -> None:
    """La primera pasada agota la ventana del decisor; la reanudación se la encuentra agotada.

    Con un contador limpio, la segunda pasada vuelve a decidir todo: el proveedor
    ve el doble de lo declarado y nadie lo sabe. Sembrado desde el directorio
    previo, el brazo aborta registrando `cuota agotada`, que es lo que pasaría de
    verdad contra el gateway.

    La caché es nueva en cada pasada a propósito: con la misma, la reanudación
    saldría entera de la caché y no habría cuota que comprobar.
    """
    solo = [arm for arm in ARMS if arm.name == "solo"]
    plan = plan_from_history(synthetic_rows(), dry_run_config())
    counted = await dry_run(
        solo, plan, remote_decider_settings(1), InMemoryResponseCache(), fixed_clock, preset=PRESET
    )
    settings = remote_decider_settings(counted.activations)
    backends = {Backend.OPENAI: FakeLLM(), Backend.OLLAMA: FakeLLM()}

    async def one_pass(directory: Path, previous: Path | None) -> list[EvaluationRecord]:
        ledger = QuotaLedger(settings.quota_window, fixed_clock)
        if previous is not None:
            assert seed_from_previous(ledger, previous, PLAN_SHA) == counted.activations
        open_run_directory(directory, run_meta(resumed_from=previous))
        await run_arms(
            solo,
            plan,
            settings,
            InMemoryResponseCache(),
            fixed_clock,
            ledger,
            directory,
            fill_with=backends,
            preset=PRESET,
        )
        return list(read_run(directory).arms[0].records)

    first = await one_pass(tmp_path / "a", None)
    resumed = await one_pass(tmp_path / "b", tmp_path / "a")

    assert sum(1 for record in first if record.proposed is not None) == counted.activations
    assert all(record.proposed is None for record in resumed)
    starved = [record for record in resumed if record.errors]
    assert len(starved) == counted.activations
    assert all("cuota agotada" in record.errors[0].message for record in starved)


def test_resuming_a_different_plan_is_refused(tmp_path: Path) -> None:
    """Otro manifiesto son otras evaluaciones: no hay nada que reanudar."""
    open_run_directory(tmp_path / "a", run_meta(plan_sha256="d" * 64))
    ledger = QuotaLedger(ablation_settings().quota_window, fixed_clock)

    with pytest.raises(ConfigError, match="no es una reanudación"):
        seed_from_previous(ledger, tmp_path / "a", PLAN_SHA)


def test_resuming_from_something_that_is_not_a_run_is_refused(tmp_path: Path) -> None:
    """Sin poder leer lo gastado no se arranca dándolo por cero."""
    ledger = QuotaLedger(ablation_settings().quota_window, fixed_clock)

    with pytest.raises(ConfigError, match="no se puede reanudar"):
        seed_from_previous(ledger, tmp_path / "no-existe", PLAN_SHA)


def git(directory: Path, *arguments: str) -> None:
    """Un comando de git dentro del repositorio de la prueba."""
    subprocess.run(
        ["git", "-c", "user.name=prueba", "-c", "user.email=prueba@example.com", *arguments],
        cwd=directory,
        check=True,
        capture_output=True,
    )


def test_the_git_state_names_the_commit_and_whether_the_tree_matches_it(tmp_path: Path) -> None:
    """Fuera de un repositorio no se inventa nada; dentro, commit y si hay cambios.

    Un commit con cambios sin comprometer nombra un código que no es el que
    corrió, así que el segundo valor es tan parte de la respuesta como el primero.
    """
    assert git_state(tmp_path) == (None, None)

    git(tmp_path, "init", "--quiet")
    git(tmp_path, "commit", "--quiet", "--allow-empty", "-m", "inicio")
    commit, dirty = git_state(tmp_path)
    assert commit is not None
    assert len(commit) == 40
    assert dirty is False

    (tmp_path / "nuevo.py").write_text("x = 1\n", encoding="utf-8")
    assert git_state(tmp_path) == (commit, True)


def test_the_command_leaves_a_run_directory_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """De `main()` al disco: metadatos y un journal por brazo, sin pedirlo aparte.

    Es el cableado que faltaba. `run_arm` ya aceptaba un journal y el comando no le
    pasaba ninguno, así que cada pieza estaba probada y la corrida real no dejó
    nada. Con el preset de producción los modelos falsos citan un indicador que no
    existe y las evaluaciones abortan: da igual, lo que se comprueba es que llegan.
    """
    monkeypatch.setattr(ablation, "load_settings", lambda _path: ablation_settings())
    monkeypatch.setattr(ablation, "build_backends", lambda _settings: {Backend.OLLAMA: FakeLLM()})
    directory = tmp_path / "corrida"
    argv = [
        "--history",
        str(REAL_HISTORY),
        "--evaluations",
        "3",
        "--arms",
        "full,solo",
        "--fill",
        "--cache",
        str(tmp_path / "cache"),
        "--out",
        str(tmp_path / "tabla.md"),
        "--journal-dir",
        str(directory),
    ]

    assert main(argv) == 0, capsys.readouterr().err

    run = read_run(directory)
    assert run.meta.arms == ("full", "solo")
    assert run.meta.fill is True
    assert run.meta.argv == tuple(argv)
    assert run.meta.plan_kind is PlanKind.HISTORY
    assert run.meta.resumed_from is None
    assert run.meta.billing is Billing.GO
    assert [len(arm.records) for arm in run.arms] == [3, 3]
    assert all(arm.sha256 is not None for arm in run.arms)


def test_a_dry_run_writes_no_run_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """El conteo no es una corrida: no deja journals ni metadatos que alguien pueda auditar."""
    monkeypatch.setattr(ablation, "load_settings", lambda _path: ablation_settings())
    directory = tmp_path / "corrida"

    code = main(
        [
            "--history",
            str(REAL_HISTORY),
            "--evaluations",
            "3",
            "--dry-run",
            "--cache",
            str(tmp_path / "cache"),
            "--journal-dir",
            str(directory),
        ]
    )

    assert code == 0, capsys.readouterr().err
    assert not directory.exists()
