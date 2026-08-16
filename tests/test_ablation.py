"""Pruebas de la ablación: variantes del pipeline, brazos y reporte.

Con modelos falsos todas las variantes deciden lo mismo, así que aquí no se
comprueba *qué* decide cada brazo —eso lo dirá la corrida real— sino que cada
variante recorre la forma de grafo que dice recorrer, que las reglas permanentes
siguen en pie en todas, y que la comparación no miente.
"""

from __future__ import annotations

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
    render_dry_run,
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
from crypto_agents.state import (
    Action,
    AgentRole,
    Backend,
    LLMCall,
    Proposal,
    StructuredOutputMode,
)
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
    clock = lambda: datetime(2026, 8, 1, tzinfo=UTC)  # noqa: E731

    by_name = {arm.name: arm for arm in ARMS}
    remote = await run_arm(
        by_name["full"],
        rows,
        settings,
        config,
        HEALTHY,
        cache,
        clock,
        fill_with={Backend.OLLAMA: backend},
        preset=PRESET,
    )
    local = await run_arm(
        by_name["local_technicals"],
        rows,
        settings,
        config,
        HEALTHY,
        cache,
        clock,
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
    clock = lambda: datetime(2026, 8, 1, tzinfo=UTC)  # noqa: E731

    by_name = {arm.name: arm for arm in ARMS}
    for name in ("full", "local_bull"):
        result = await run_arm(
            by_name[name],
            rows,
            settings,
            config,
            HEALTHY,
            cache,
            clock,
            fill_with={Backend.OLLAMA: backend},
            preset=PRESET,
        )
        if name == "local_bull":
            assert AgentRole.STRUCTURE not in result.summary.quota_by_role
            assert result.summary.quota_by_role[AgentRole.BULL] > 0.0


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
        synthetic_rows(),
        ablation_settings(),
        dry_run_config(),
        cache,
        lambda: datetime(2026, 8, 1, tzinfo=UTC),
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
    clock = lambda: datetime(2026, 8, 1, tzinfo=UTC)  # noqa: E731
    full = next(arm for arm in ARMS if arm.name == "full")

    report = await dry_run(
        [full], rows, settings, config, InMemoryResponseCache(), clock, preset=PRESET
    )
    result = await run_arm(
        full,
        rows,
        settings,
        config,
        HEALTHY,
        InMemoryResponseCache(),
        clock,
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
        synthetic_rows(),
        ablation_settings(),
        dry_run_config(),
        InMemoryResponseCache(),
        lambda: datetime(2026, 8, 1, tzinfo=UTC),
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
    clock = lambda: datetime(2026, 8, 1, tzinfo=UTC)  # noqa: E731
    cache = InMemoryResponseCache()
    full = next(arm for arm in ARMS if arm.name == "full")

    await run_arm(
        full,
        rows,
        settings,
        config,
        HEALTHY,
        cache,
        clock,
        fill_with={Backend.OLLAMA: FakeLLM()},
        preset=PRESET,
    )
    report = await dry_run([full], rows, settings, config, cache, clock, preset=PRESET)

    row = next(item for item in report.rows if item.node == "structure")
    assert row.to_pay == 0
    assert row.cached == row.calls


@pytest.mark.asyncio
async def test_the_dry_run_adds_up_the_quota_no_single_ledger_can_see() -> None:
    """`run_arm` construye un contador por brazo: seis contadores, un solo proveedor.

    Cada uno se cree dentro del presupuesto mientras el gasto agregado es la suma
    de los seis. Es el mismo fallo que el runner evita con un `QuotaLedger` único
    para todos los símbolos, y aquí no se puede evitar igual —los brazos tienen que
    tener presupuestos independientes para ser comparables—, así que se reporta.
    """
    settings = ablation_settings()
    tight = settings.model_copy(
        update={
            "roles": dict(settings.roles)
            | {
                AgentRole.DECIDER: RoleConfig(
                    primary=settings.role_config(AgentRole.DECIDER).primary.model_copy(
                        update={"quota_per_window": 3}
                    )
                )
            }
        }
    )
    report = await dry_run(
        ARMS,
        synthetic_rows(),
        tight,
        dry_run_config(),
        InMemoryResponseCache(),
        lambda: datetime(2026, 8, 1, tzinfo=UTC),
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
                quota=4.0,
            ),
            DryRunRow(
                arm="full",
                node="decide",
                role=AgentRole.DECIDER,
                backend=Backend.OLLAMA,
                model="remoto-decider",
                calls=4,
                exact=False,
                quota=4.0,
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
    results = [build_arm_result(arm, [], [], wall_clock_seconds=1.0) for arm in ARMS[:3]]
    report = render_report(results)

    for arm in ARMS[:3]:
        assert f"`{arm.name}`" in report
        assert arm.question in report


def test_the_report_survives_an_empty_run() -> None:
    """Sin corridas la tabla lo dice, en vez de fingir ceros."""
    assert "Sin corridas" in render_report([])
