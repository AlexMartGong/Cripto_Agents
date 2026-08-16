"""Pruebas de la ablación: variantes del pipeline, brazos y reporte.

Con modelos falsos todas las variantes deciden lo mismo, así que aquí no se
comprueba *qué* decide cada brazo —eso lo dirá la corrida real— sino que cada
variante recorre la forma de grafo que dice recorrer, que las reglas permanentes
siguen en pie en todas, y que la comparación no miente.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

if TYPE_CHECKING:
    from collections.abc import Sequence

from crypto_agents.ablation import (
    ARMS,
    DETERMINISTIC_NODES,
    AblationArm,
    DryRunReport,
    DryRunRow,
    agreement,
    arm_settings,
    build_arm_result,
    decision_actions,
    dry_run,
    llm_nodes,
    plan_from_history,
    plan_from_manifest,
    render_dry_run,
    render_report,
    run_arm,
)
from crypto_agents.activation import ActivationConfig
from crypto_agents.cache import InMemoryResponseCache
from crypto_agents.graph import PipelineVariant, build_graph
from crypto_agents.journal import EvaluationRecord, InMemoryJournal
from crypto_agents.metrics import summarise
from crypto_agents.quota import QuotaLedger
from crypto_agents.replay import ReplaySettings
from crypto_agents.selection import SelectionError, SelectionManifest, select_activations
from crypto_agents.settings import (
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
    LLMCall,
    Proposal,
    StructuredOutputMode,
)
from tests.conftest import CHEAP, HEALTHY, PRESET, FakeLLM, raw_ohlcv, role_map
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

    decided = [arm.name for arm, journal in journals if decided_in(journal) > 0]
    starved = [(arm, journal) for arm, journal in journals if quota_aborts(journal)]

    assert decided == [arm.name for arm in ARMS[:BUDGETED_ARMS]], (
        "el presupuesto alcanza para tres brazos: si deciden más, cada uno tiene el suyo"
    )
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
    assert decider.calls == report.activations * len(ARMS)
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


def test_the_report_survives_an_empty_run() -> None:
    """Sin corridas la tabla lo dice, en vez de fingir ceros."""
    assert "Sin corridas" in render_report([])
