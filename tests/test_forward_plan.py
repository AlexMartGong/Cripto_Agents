"""Pruebas de la planificación del estudio hacia adelante.

Cinco cosas tienen que ser ciertas para que las cifras de `docs/forward.md` sirvan, y cada una
tiene su mutación:

- **Las probabilidades.** Calculadas a mano con la tabla de la normal. Usar el IC 95 % en el
  empate, quitar el suelo del efecto de diseño o volver al δ de la ablación tiene que romperlas.
- **La regla del efecto de diseño.** El máximo de las seis series sustituto: ni la mediana ni una
  sola serie.
- **El recuento por bloque.** Bloques completos, vacíos incluidos y bordes fuera.
- **La fecha de inicio.** El primer jueves estrictamente posterior al merge, borde incluido.
- **El cierre.** 50 bloques no vacíos, con el 52 como máximo: un bloque vacío no suma.

Y tres comprobaciones de lo que el estudio repite en cada lote, también con su mutación: el tope de
gasto proporcional al manifiesto, los recuentos del pool y los retornos publicados de la etapa 1,
que la atadura con el journal no ve.
"""

from __future__ import annotations

import inspect
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from crypto_agents import criteria
from crypto_agents import design_effect as design_module
from crypto_agents import forward_plan as module
from crypto_agents.audit import RunKind, file_sha256
from crypto_agents.criteria import paired_returns
from crypto_agents.design_effect import SURROGATE_PAIRS, Planning, PoolBar, SeriesError, pool_run_id
from crypto_agents.forward_plan import (
    BATCH_DEADLINE,
    DELTA,
    EARLIEST_START,
    FORWARD_ARMS,
    MAX_BLOCKS,
    POOL_COUNTS,
    PRIMARY,
    PUBLISHED_RETURNS,
    PUBLISHED_RETURNS_SHA256,
    REFERENCE_DELTAS,
    SPEND_CAP_MULTIPLE,
    TARGET_BLOCKS,
    USD_PER_EVALUATION,
    Z_EQUIVALENCE,
    BlockCounts,
    ForwardPlan,
    ForwardPlanError,
    OutsideBlock,
    PoolCounts,
    batch_spend_cap,
    block_counts,
    build_report,
    calendar,
    closing_block,
    first_block,
    main,
    pool_counts,
    pool_planning,
    published_returns,
    render_report,
    spend_cap,
    study_costs,
)
from crypto_agents.metrics import EPOCH, ClusteredError, block_of
from crypto_agents.paired_dispersion import Sources
from crypto_agents.selection import stratify
from crypto_agents.spend import SpendGuard
from crypto_agents.state import (
    Action,
    AgentRole,
    Backend,
    FrozenModel,
    LLMCall,
    StructuredOutputMode,
)
from tests.test_criteria import HORIZON, quota_lost
from tests.test_design_effect import (
    BARS,
    DAILY,
    DAY_ZERO,
    STRATA,
    UNCONFIRMED,
    Scenario,
    daily_manifest,
    decided,
    swept,
    write_scenario,
)
from tests.test_design_effect import scenario as design_scenario
from tests.test_net_outcomes import mutated
from tests.test_paired_dispersion import snapshot as tree_snapshot

if TYPE_CHECKING:
    from uuid import UUID

    from crypto_agents.selection import SelectionManifest

ROOT = Path(__file__).resolve().parents[1]
WEEK = timedelta(days=7)

# ─────────────────────────────── Lo fijado antes de los números ───────────────────────────────────


def test_the_constants_are_the_ones_fixed_before_the_numbers() -> None:
    """δ es el del estudio y no el de la ablación; el bloque es el clúster de la enmienda 4."""
    assert DELTA == 0.004
    assert DELTA != criteria.DELTA == 0.002
    assert REFERENCE_DELTAS == (0.002, 0.003, 0.005)
    assert DELTA not in REFERENCE_DELTAS
    assert TARGET_BLOCKS == 50
    assert MAX_BLOCKS == 52
    assert Z_EQUIVALENCE == 1.645
    assert SPEND_CAP_MULTIPLE == 3
    assert USD_PER_EVALUATION == 0.01509
    assert timedelta(days=14) == BATCH_DEADLINE
    assert PRIMARY == ("solo", "rule_trend")
    assert ("solo", *criteria.BASELINE_ARMS) == FORWARD_ARMS
    assert module.CLUSTER_DAYS is criteria.CLUSTER_DAYS
    assert module.used_deff is design_module.used_deff
    assert module.planned_power is design_module.planned_power
    assert module.block_of is block_of


def test_the_registered_counts_and_the_published_returns_are_the_published_ones() -> None:
    """Los recuentos de `docs/ablation.md` y el archivo de `tests/data`, con su sha-256. De las
    4 740 del barrido, 112 no se confirman y 4 626 sí con el horizonte por delante: las 2 que
    faltan se confirman sin él («Recálculo de dispersión»)."""
    assert (
        PoolCounts(activations=4740, unconfirmed=112, candidates=4737, confirmed=4626)
        == POOL_COUNTS
    )
    assert POOL_COUNTS.activations - POOL_COUNTS.unconfirmed - POOL_COUNTS.confirmed == 2
    assert Path("tests/data/stage1_paired_returns.jsonl") == PUBLISHED_RETURNS
    assert file_sha256(ROOT / PUBLISHED_RETURNS) == PUBLISHED_RETURNS_SHA256

    published = published_returns(ROOT / PUBLISHED_RETURNS, PUBLISHED_RETURNS_SHA256)

    assert len(published) == 140
    assert all(set(criteria.BASELINE_ARMS) <= set(row) for row in published.values())


def test_a_published_file_with_another_digest_is_refused(tmp_path: Path) -> None:
    edited = tmp_path / "stage1_paired_returns.jsonl"
    edited.write_bytes((ROOT / PUBLISHED_RETURNS).read_bytes().replace(b"0.0", b"0.1", 1))

    with pytest.raises(ForwardPlanError, match="no es el archivo publicado"):
        published_returns(edited, PUBLISHED_RETURNS_SHA256)


def test_the_earliest_start_opens_a_block_on_a_thursday() -> None:
    assert datetime(2026, 10, 22, tzinfo=UTC) == EARLIEST_START
    assert EARLIEST_START.weekday() == 3
    assert timedelta(0) == (EARLIEST_START - EPOCH) % WEEK
    assert EPOCH.weekday() == 3, "los bloques alineados al epoch empiezan en jueves"


def test_the_command_takes_no_delta() -> None:
    """δ es una constante del estudio: no hay argumento con el que pasar otro."""
    for flag in ("--delta", "--exploratory-delta"):
        with pytest.raises(SystemExit):
            main(["var/ablation/corrida", flag, "0.002"])


# ──────────────────────────────── Las probabilidades, a mano ──────────────────────────────────────

SIGMA = 0.02
N = 400
"""Con `sigma = 0.02` y `n = 400`, `EE = 0.02·√(deff/400) = 0.001·√deff`."""

TOLERANCE = 1e-4


def plan_with(deff: float) -> Planning:
    return Planning(n=N, deff=deff, series="sintética", days=7)


def assert_hand_computed_probabilities() -> None:
    """Valores de la tabla de la normal, a cuatro decimales.

    - deff 1, δ = 0.003: EE = 0.001 y δ/EE = 3. Supera: Φ(3 - 1.96) = Φ(1.04) = 0.8508. Empate:
      `b = min(0.003 - 0.001645, 0.00196) = 0.001355`, simétrico, así que 2Φ(1.355) - 1 = 0.8246.
    - El δ registrado, 0.004, con deff 1: δ/EE = 4. Supera: Φ(2.04) = 0.9793. Empate:
      `min(4 - 1.645, 1.96) = 1.96` —manda que el IC 95 % incluya el 0—, 2Φ(1.96) - 1 = 0.9500.
    - deff 4: EE = 0.002. Con δ = 0.004, δ/EE = 2: supera Φ(0.04) = 0.5160 y empate
      2Φ(0.355) - 1 = 0.2774. Con δ = 0.003, δ/EE = 1.5 < 1.645: ningún IC 90 % cabe en (-δ, δ).
    - deff bruto 0.25: el suelo lo lleva a 1, y las cifras son las del primer caso.
    """
    one, four, below = plan_with(1.0), plan_with(4.0), plan_with(0.25)

    assert module.planned_power(SIGMA, one, 0.003) == pytest.approx(0.8508, abs=TOLERANCE)
    assert module.tie_probability(SIGMA, one, 0.003) == pytest.approx(0.8246, abs=TOLERANCE)
    assert module.planned_power(SIGMA, four, 0.004) == pytest.approx(0.5160, abs=TOLERANCE)
    assert module.tie_probability(SIGMA, four, 0.004) == pytest.approx(0.2774, abs=TOLERANCE)
    assert module.tie_probability(SIGMA, four, 0.003) == 0.0
    assert module.planned_power(SIGMA, below, 0.003) == pytest.approx(0.8508, abs=TOLERANCE)
    assert module.tie_probability(SIGMA, below, 0.003) == pytest.approx(0.8246, abs=TOLERANCE)

    found = module.probabilities(SIGMA, one)
    registered = [item for item in found if item.registered]
    assert len(registered) == 1
    assert registered[0].delta == 0.004
    assert registered[0].above == pytest.approx(0.9793, abs=TOLERANCE)
    assert registered[0].tie == pytest.approx(0.9500, abs=TOLERANCE)
    assert [item.delta for item in found] == [0.002, 0.003, 0.004, 0.005]


def test_the_probabilities_hold_for_the_real_code() -> None:
    """El control de las mutaciones: sin error puesto, la comprobación pasa."""
    assert_hand_computed_probabilities()


def test_mutation_a_tie_judged_with_the_95_interval_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Con 1.96 en lugar de 1.645, el empate del primer caso sería 2Φ(1.04) - 1 = 0.7017."""
    mutant = mutated(
        module.tie_probability,
        "margin = Z_EQUIVALENCE * error",
        "margin = Z_INTERVAL * error",
        module,
    )
    monkeypatch.setattr(module, "tie_probability", mutant)
    with pytest.raises(AssertionError):
        assert_hand_computed_probabilities()


@pytest.mark.parametrize("where", ["los dos", "solo el empate"])
def test_mutation_a_design_effect_without_its_floor_is_caught(
    where: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sin el suelo, el deff bruto 0.25 deja EE = 0.0005: supera Φ(4.04) y empate 0.95."""
    mutant = mutated(
        design_module.used_deff, "return max(1.0, plan.deff)", "return plan.deff", design_module
    )
    monkeypatch.setattr(module, "used_deff", mutant)
    if where == "los dos":
        monkeypatch.setattr(design_module, "used_deff", mutant)
    with pytest.raises(AssertionError):
        assert_hand_computed_probabilities()


def test_mutation_the_delta_of_the_ablation_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """Con δ = 0.002 la fila registrada sería Φ(0.04) y 2Φ(0.355) - 1, no Φ(2.04) y 0.95."""
    monkeypatch.setattr(module, "DELTA", criteria.DELTA)
    with pytest.raises(AssertionError):
        assert_hand_computed_probabilities()


def test_the_tie_probability_is_symmetric_and_needs_a_design_effect() -> None:
    one = plan_with(1.0)

    shifted_up = module.tie_probability(SIGMA, one, 0.003, true_difference=0.0005)
    shifted_down = module.tie_probability(SIGMA, one, 0.003, true_difference=-0.0005)

    assert shifted_up == pytest.approx(shifted_down)
    assert shifted_up is not None
    assert shifted_up < 0.8246
    assert module.tie_probability(SIGMA, Planning(n=N), 0.003) is None
    assert module.tie_probability(0.0, one, 0.003) == 1.0
    with pytest.raises(ValueError):
        module.tie_probability(SIGMA, one, 0.0)


# ───────────────────────────── El efecto de diseño de planificación ───────────────────────────────

SURROGATE_NAMES = tuple(f"{arm} - {reference}" for arm, reference in SURROGATE_PAIRS)
SURROGATE_DEFFS = (3.0, 9.0, 2.0, 5.0, 4.0, 6.0)
"""El máximo, 9, no es la primera serie (3) ni la mediana por rango (5)."""


def series_error(name: str, deff: float, days: int = 7, surrogate: bool = True) -> SeriesError:
    return SeriesError(
        series=name,
        surrogate=surrogate,
        days=days,
        error=ClusteredError(n=100, clusters=60, naive=0.001, clustered=0.001 * math.sqrt(deff)),
    )


def planning_rows() -> list[SeriesError]:
    """Las seis series con L = 7, más una línea base sola y una serie con L = 1 que no entran."""
    return [
        *(
            series_error(name, deff)
            for name, deff in zip(SURROGATE_NAMES, SURROGATE_DEFFS, strict=True)
        ),
        series_error("always_buy", 12.0, surrogate=False),
        series_error(SURROGATE_NAMES[0], 20.0, days=1),
    ]


def assert_the_planning_deff_is_the_largest_of_the_six() -> None:
    plan = module.pool_planning(planning_rows(), 2449)

    assert plan.deff == pytest.approx(9.0)
    assert plan.series == SURROGATE_NAMES[1]
    assert plan.days == 7
    assert plan.n == 2449


def test_the_planning_deff_is_the_largest_of_the_six_surrogates() -> None:
    assert_the_planning_deff_is_the_largest_of_the_six()


@pytest.mark.parametrize(
    "new",
    [
        "worst = sorted(surrogates, key=_deff_of)[len(surrogates) // 2]",
        "worst = surrogates[0]",
    ],
    ids=["la mediana", "una sola serie"],
)
def test_mutation_a_planning_deff_that_is_not_the_maximum_is_caught(
    new: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    mutant = mutated(pool_planning, "worst = max(surrogates, key=_deff_of)", new, module)
    monkeypatch.setattr(module, "pool_planning", mutant)
    with pytest.raises(AssertionError):
        assert_the_planning_deff_is_the_largest_of_the_six()


def test_the_planning_deff_needs_the_six_surrogates() -> None:
    rows = planning_rows()
    with pytest.raises(ForwardPlanError, match="las 6 series sustituto"):
        pool_planning(rows[1:], 100)
    with pytest.raises(ForwardPlanError, match="las 6 series sustituto"):
        pool_planning([*rows[:5], series_error(SURROGATE_NAMES[0], 7.0)], 100)
    flat = SeriesError(
        series=SURROGATE_NAMES[1],
        surrogate=True,
        days=7,
        error=ClusteredError(n=100, clusters=60, naive=0.0, clustered=0.0),
    )
    with pytest.raises(ForwardPlanError, match="sin efecto de diseño"):
        pool_planning([rows[0], flat, *rows[2:6]], 100)


# ─────────────────────────────────────── La fecha de inicio ───────────────────────────────────────

STARTS = [
    (datetime(2026, 10, 15, 18, tzinfo=UTC), datetime(2026, 10, 22, tzinfo=UTC)),
    (datetime(2026, 10, 15, tzinfo=UTC), datetime(2026, 10, 22, tzinfo=UTC)),
    (datetime(2026, 10, 21, 23, 59, 59, 999999, tzinfo=UTC), datetime(2026, 10, 22, tzinfo=UTC)),
    (datetime(2026, 10, 22, tzinfo=UTC), datetime(2026, 10, 29, tzinfo=UTC)),
    (datetime(2026, 10, 22, 0, 0, 0, 1, tzinfo=UTC), datetime(2026, 10, 29, tzinfo=UTC)),
    (datetime(2026, 10, 29, tzinfo=UTC), datetime(2026, 11, 5, tzinfo=UTC)),
    (datetime(2026, 11, 3, 12, tzinfo=UTC), datetime(2026, 11, 5, tzinfo=UTC)),
]
"""(instante del merge, primer bloque). En el borde exacto de un bloque el estudio empieza en el
siguiente: «después del merge» es estrictamente después."""


def assert_the_start_rule_holds() -> None:
    for merged, expected in STARTS:
        assert module.first_block(merged) == expected, merged.isoformat()


def test_the_first_block_is_the_first_thursday_strictly_after_the_merge() -> None:
    assert_the_start_rule_holds()
    for _, start in STARTS:
        assert start.weekday() == 3
        assert (start - EPOCH) % WEEK == timedelta(0)


def test_mutation_a_block_that_starts_at_the_merge_instant_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """«El primero que empieza en o después del merge»: solo difiere en el borde exacto."""
    mutant = mutated(
        first_block,
        "block_of(merged_at, CLUSTER_DAYS) + 1",
        "block_of(merged_at - timedelta(microseconds=1), CLUSTER_DAYS) + 1",
        module,
    )
    monkeypatch.setattr(module, "first_block", mutant)
    with pytest.raises(AssertionError):
        assert_the_start_rule_holds()


def test_a_merge_instant_without_a_time_zone_is_refused() -> None:
    with pytest.raises(ValueError, match="sin zona horaria"):
        first_block(datetime(2026, 10, 16, 12))


def test_the_calendar_of_the_earliest_start() -> None:
    """Desde el 2026-10-22: el fin esperado en el bloque 50 y el más tardío en el 52; cada lote,
    al cierre de su bloque más 6 velas de 4 h, con 14 días de plazo."""
    dates = calendar(EARLIEST_START, "4h", 6)

    assert dates.first == datetime(2026, 10, 22, tzinfo=UTC)
    assert dates.last == datetime(2027, 9, 30, tzinfo=UTC)
    assert dates.end == datetime(2027, 10, 7, tzinfo=UTC)
    assert dates.end - dates.first == TARGET_BLOCKS * WEEK
    assert dates.first_ready == datetime(2026, 10, 30, tzinfo=UTC)
    assert dates.first_deadline == datetime(2026, 11, 12, tzinfo=UTC)
    assert dates.last_ready == datetime(2027, 10, 8, tzinfo=UTC)
    assert dates.last_deadline == datetime(2027, 10, 21, tzinfo=UTC)
    assert dates.latest_last == datetime(2027, 10, 14, tzinfo=UTC)
    assert dates.latest_end == datetime(2027, 10, 21, tzinfo=UTC)
    assert dates.latest_end - dates.first == MAX_BLOCKS * WEEK
    assert dates.latest_ready == datetime(2027, 10, 22, tzinfo=UTC)
    assert dates.latest_deadline == datetime(2027, 11, 4, tzinfo=UTC)
    with pytest.raises(ValueError, match="no es el inicio de un bloque"):
        calendar(EARLIEST_START + timedelta(days=1), "4h", 6)


# ─────────────────────────────────────────── El cierre ────────────────────────────────────────────


def weeks_with_empty(*empty: int) -> list[int]:
    """52 bloques con activaciones salvo los de las posiciones `empty`, contadas desde 1."""
    return [0 if position in empty else 7 for position in range(1, MAX_BLOCKS + 1)]


def assert_the_study_closes_as_written() -> None:
    assert module.closing_block(weeks_with_empty()) == 50
    assert module.closing_block(weeks_with_empty(10)) == 51
    assert module.closing_block(weeks_with_empty(1, 30)) == 52
    assert module.closing_block(weeks_with_empty(1, 2, 3)) is None
    assert module.closing_block(weeks_with_empty(3, 40, 51)) is None
    assert module.closing_block(weeks_with_empty(10, 52)) == 51, "lo que viene después no cuenta"
    assert module.closing_block(weeks_with_empty(51, 52)) == 50


def test_the_study_closes_at_the_fiftieth_block_with_activations() -> None:
    """Un vacío cierra en el 51, dos en el 52 y tres dejan el estudio «no determinado»."""
    assert_the_study_closes_as_written()


def test_mutation_empty_blocks_counted_towards_the_fifty_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mutant = mutated(closing_block, "if count > 0:", "if count >= 0:", module)
    monkeypatch.setattr(module, "closing_block", mutant)
    with pytest.raises(AssertionError):
        assert_the_study_closes_as_written()


def test_a_study_still_open_or_a_negative_count_is_refused() -> None:
    with pytest.raises(ValueError, match="sigue abierto"):
        closing_block([5] * 49)
    with pytest.raises(ValueError, match="sigue abierto"):
        closing_block([0, *[5] * 49])
    with pytest.raises(ValueError, match="negativas"):
        closing_block([-1, *[5] * 51])


# ───────────────────────────────────── El recuento por bloque ─────────────────────────────────────

COUNTS = (3, 0, 5, 2, 4, 6, 1, 7, 2, 5)
"""Activaciones de los diez bloques completos del manifiesto diario. Ordenadas: 0 1 2 2 3 4 5 5 6 7,
así que la media es 3.5, el p10 (rango 1) es 0 y el p90 (rango 9) es 6."""

EDGES = ((0, 2), (11, 1))
"""(bloque, activaciones) de los dos bloques parciales: el primero y el último."""


def counted_bars() -> list[PoolBar]:
    """Velas diarias del manifiesto sintético, repartidas por bloque según `COUNTS` y `EDGES`.

    El manifiesto diario tiene 84 velas desde el 2026-01-01 —un jueves, inicio de bloque—,
    warm-up 1 y horizonte 3: la primera evaluación cierra el día 2 y la última el día 81. Los
    bloques completos empiezan los días 7, 14 … 70; el del día 0 y el del día 77 son parciales.
    """
    bars: list[PoolBar] = []
    for block, count in [*((index + 1, count) for index, count in enumerate(COUNTS)), *EDGES]:
        first_day = 3 if block == 0 else 1
        for offset in range(count):
            moment = DAY_ZERO + block * WEEK + timedelta(days=first_day + offset % 5)
            symbol = f"S{offset % 2}"
            bars.append(
                PoolBar(symbol, block * 7 + offset, 0, moment, pool_run_id(symbol, "1d", moment))
            )
    return bars


def assert_the_blocks_are_counted_by_hand() -> None:
    found = module.block_counts(counted_bars(), daily_manifest())

    assert found.counts == COUNTS
    assert found.first == DAY_ZERO + WEEK
    assert found.last == DAY_ZERO + 10 * WEEK
    assert found.outside == (
        OutsideBlock(start=DAY_ZERO, evaluations=2),
        OutsideBlock(start=DAY_ZERO + 11 * WEEK, evaluations=1),
    )
    assert found.mean_per_block == pytest.approx(3.5)
    assert found.mean_per_filled_block == pytest.approx(35 / 9)
    assert found.p10 == 0
    assert found.p90 == 6
    assert found.empty == 1


def test_the_blocks_are_counted_by_hand() -> None:
    assert_the_blocks_are_counted_by_hand()


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (
            "counts = tuple(tally.get(block, 0) for block in blocks)",
            "counts = tuple(tally[block] for block in blocks if block in tally)",
        ),
        ("blocks = range(first, last + 1)", "blocks = range(first - 1, last + 2)"),
    ],
    ids=["sin los bloques vacíos", "con los bordes parciales"],
)
def test_mutation_blocks_counted_another_way_are_caught(
    old: str, new: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "block_counts", mutated(block_counts, old, new, module))
    with pytest.raises(AssertionError):
        assert_the_blocks_are_counted_by_hand()


def test_a_pool_whose_blocks_are_all_empty_has_no_mean_per_filled_block() -> None:
    blocks = BlockCounts(days=7, first=DAY_ZERO, last=DAY_ZERO + WEEK, counts=(0, 0))

    assert blocks.mean_per_block == 0
    assert blocks.mean_per_filled_block is None
    assert blocks.empty == 2


def test_a_series_with_a_gap_is_refused() -> None:
    manifest = daily_manifest()
    gapped = manifest.series[0].model_copy(update={"bars": manifest.series[0].bars - 1})
    broken = manifest.model_copy(update={"series": (gapped, *manifest.series[1:])})

    with pytest.raises(ForwardPlanError, match="tiene huecos"):
        block_counts(counted_bars(), broken)


def test_a_pool_without_a_complete_block_is_refused() -> None:
    manifest = daily_manifest()
    short = tuple(
        ref.model_copy(update={"bars": 9, "last": ref.first + timedelta(days=8)})
        for ref in manifest.series
    )
    with pytest.raises(ForwardPlanError, match="ningún bloque completo"):
        block_counts([], manifest.model_copy(update={"series": short}))


# ──────────────────────────────────────── El tope de gasto ────────────────────────────────────────

AT = datetime(2026, 10, 29, tzinfo=UTC)


def decider_call() -> LLMCall:
    """Una llamada del decisor que cuesta lo registrado por evaluación, al décimo de milésima:
    (1 150 · 1.40 + 3 064 · 4.40) / 1e6 = 0.0150916 USD con los precios de `glm-5.2`."""
    return LLMCall(
        role=AgentRole.DECIDER,
        backend=Backend.OPENAI,
        model="glm-5.2",
        structured_output=StructuredOutputMode.FUNCTION_CALLING,
        quota_weight=1.0,
        prompt_digest="a" * 64,
        valid=True,
        latency_ms=100.0,
        at=AT,
        prompt_tokens=1150,
        cached_tokens=0,
        completion_tokens=3064,
    )


def manifest_of(evaluations: int) -> SelectionManifest:
    """Un manifiesto sintético con ese número de entradas: el tope solo las cuenta."""
    manifest = daily_manifest()
    entries = [*manifest.entries] * (evaluations // len(manifest.entries) + 1)
    return manifest.model_copy(update={"entries": tuple(entries[:evaluations])})


def lost_to_the_cap(cap: float, evaluations: int) -> int:
    """Las evaluaciones de un lote que el guarda de gasto de producción deja sin abrir con ese
    tope: cada una es una invocación del decisor que cuesta lo registrado."""
    guard = SpendGuard(cap)
    for position in range(evaluations):
        if guard.allow(f"evaluación {position}"):
            guard.add([decider_call()])
    return len(guard.refused)


def assert_two_hundred_lose_nothing() -> None:
    """Un manifiesto de 200 evaluaciones recibe 3 x 200 x 0.01509 = 9.054 USD y no pierde
    ninguna. Con el tope fijo de 2.22 USD, el guarda abre ⌈2.22 / 0.0150916⌉ = 148 y deja 52."""
    manifest = manifest_of(200)
    assert len(manifest.entries) == 200

    cap = batch_spend_cap(manifest)

    assert cap == pytest.approx(3 * 200 * 0.01509)
    assert lost_to_the_cap(cap, 200) == 0


def test_the_call_costs_what_is_registered_per_evaluation() -> None:
    guard = SpendGuard(1.0)
    guard.add([decider_call()])
    assert guard.spent_usd == pytest.approx(USD_PER_EVALUATION, rel=1e-3)
    assert lost_to_the_cap(2.22, 200) == 52


def test_a_batch_of_two_hundred_gets_a_cap_in_proportion_and_loses_nothing() -> None:
    assert_two_hundred_lose_nothing()


def test_mutation_a_fixed_cap_per_batch_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    mutant = mutated(
        spend_cap, "SPEND_CAP_MULTIPLE * evaluations * USD_PER_EVALUATION", "2.22", module
    )
    monkeypatch.setattr(module, "spend_cap", mutant)
    with pytest.raises(AssertionError):
        assert_two_hundred_lose_nothing()


def test_an_empty_block_has_no_cap_and_a_negative_one_is_refused() -> None:
    assert spend_cap(0) == 0.0
    with pytest.raises(ValueError, match="negativas"):
        spend_cap(-1)


def test_the_study_bound_is_the_sum_of_its_caps() -> None:
    """3 x 2 449 x 0.01509 = 110.866 USD: la cota del estudio esperado. No depende de que la
    etapa 1 tenga coste medido."""
    costs = study_costs(0.01, 36.0, 2449)

    assert costs.study_bound_usd == pytest.approx(110.86623)
    assert costs.usd(100) == pytest.approx(1.0)
    assert costs.hours(100) == pytest.approx(1.0)

    unmeasured = study_costs(None, None, 2449)

    assert unmeasured.study_bound_usd == costs.study_bound_usd
    assert unmeasured.usd(10) is None
    assert unmeasured.hours(10) is None


# ─────────────────────────────── El reporte, sobre el escenario diario ────────────────────────────

SOURCES = Sources(
    directory="var/ablation/corrida",
    command="python -m crypto_agents.forward_plan var/ablation/corrida",
    commit="`abc1234` (árbol limpio)",
    plan="`manifest.json` (manifest), sha-256 `" + "c" * 64 + "`",
    journals=(("solo.jsonl", "e" * 64),),
    histories=(("S0", "history/s0_1d.csv", "f" * 64),),
)


def scenario(monkeypatch: pytest.MonkeyPatch) -> Scenario:
    """El escenario diario de `design_effect`, con su barrido y su confirmación escritos a mano
    también aquí: `pool_counts` los llama por su cuenta."""
    found = design_scenario(monkeypatch)
    for name in ("activated_bars", "confirm_activation"):
        monkeypatch.setattr(module, name, getattr(design_module, name))
    return found


def published_of(found: Scenario) -> dict[UUID, dict[str, float]]:
    """Lo que la etapa sintética habría publicado: su puntuación de hoy, derivada como
    `tests/data/stage1_paired_returns.jsonl`."""
    published: dict[UUID, dict[str, float]] = {}
    for arm in ("solo", *criteria.BASELINE_ARMS):
        reference = "rule_trend" if arm == "solo" else "solo"
        aligned = paired_returns(arm, reference, found.stage, DAILY, HORIZON)
        assert not isinstance(aligned, str)
        for record, value in zip(aligned.arm_records, aligned.arm_returns, strict=True):
            published.setdefault(record.run_id, {})[arm] = value
    return published


def counts_of(found: Scenario) -> PoolCounts:
    """Por el módulo, para que una mutación de `pool_counts` se vea."""
    return module.pool_counts(found.pool, found.manifest, DAILY)


def report_of(found: Scenario) -> ForwardPlan:
    return build_report(
        found.stage,
        DAILY,
        HORIZON,
        found.manifest,
        found.pool,
        found.recalculation,
        published_of(found),
        counts_of(found),
    )


def test_the_pool_counts_follow_their_definition(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sin `pool_records`: todas las del barrido, las que la ventana no confirma de entre todas,
    las que `stratify` reparte y, de ellas, las confirmadas. En el escenario las dos sin confirmar
    tienen horizonte."""
    found = scenario(monkeypatch)
    every = {symbol: swept(symbol) for symbol in DAILY}
    stratified = {
        (symbol, bar.index)
        for symbol in DAILY
        for cell in stratify(DAILY[symbol], every[symbol], STRATA, HORIZON)
        for bar in cell
    }

    counts = counts_of(found)

    assert counts.activations == sum(len(bars) for bars in every.values()) == 2 * BARS
    assert counts.unconfirmed == len(UNCONFIRMED) == 2
    assert counts.candidates == len(stratified)
    assert counts.confirmed == len(stratified - UNCONFIRMED) == len(found.pool.bars)


def assert_an_unconfirmed_bar_without_horizon_counts(found: Scenario) -> None:
    """Con la última vela de S0 sin confirmar, que no tiene horizonte y ningún tramo la reparte,
    son tres sin confirmar, no dos."""
    assert counts_of(found).unconfirmed == 3


def without_the_last_bar(monkeypatch: pytest.MonkeyPatch) -> None:
    confirm = vars(module)["confirm_activation"]

    def refusing(rows: list[list[float]], index: int, *rest: object) -> tuple[str, ...] | None:
        if rows == DAILY["S0"] and index == BARS - 1:
            return None
        return confirm(rows, index, *rest)  # type: ignore[no-any-return]

    monkeypatch.setattr(module, "confirm_activation", refusing)


def test_an_unconfirmed_bar_without_horizon_is_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    found = scenario(monkeypatch)
    without_the_last_bar(monkeypatch)
    assert_an_unconfirmed_bar_without_horizon_counts(found)


def test_mutation_counting_unconfirmed_only_among_the_stratified_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    found = scenario(monkeypatch)
    without_the_last_bar(monkeypatch)
    mutant = mutated(
        pool_counts,
        "unconfirmed=candidates - len(pool.bars) + outside,",
        "unconfirmed=candidates - len(pool.bars),",
        module,
    )
    monkeypatch.setattr(module, "pool_counts", mutant)
    with pytest.raises(AssertionError):
        assert_an_unconfirmed_bar_without_horizon_counts(found)


def test_the_report_over_the_daily_scenario(monkeypatch: pytest.MonkeyPatch) -> None:
    found = scenario(monkeypatch)
    report = report_of(found)

    assert report.consistent
    assert report.counts == report.registered_counts == counts_of(found)
    assert [tie.arm for tie in report.ties] == list(criteria.BASELINE_ARMS)
    assert [tie.arm for tie in report.published] == list(criteria.BASELINE_ARMS)
    assert all(tie.exact for tie in report.published)
    assert {tie.published for tie in report.published} == {len(found.manifest.entries)}
    assert [row.series for row in report.pool] == list(SURROGATE_NAMES)
    assert {row.days for row in report.pool} == {7}
    deffs = [row.error.deff for row in report.pool]
    assert all(deff is not None for deff in deffs)
    assert report.planning.deff == max(deff for deff in deffs if deff is not None)
    filled = report.blocks.mean_per_filled_block
    assert filled is not None
    assert report.planning.n == round(TARGET_BLOCKS * filled)
    assert (report.target_blocks, report.max_blocks) == (TARGET_BLOCKS, MAX_BLOCKS)
    assert report.confirmed == len(found.pool.bars)
    assert report.stage_n == len(found.manifest.entries)
    assert sum(report.blocks.counts) + sum(item.evaluations for item in report.blocks.outside) == (
        len(found.pool.bars)
    )
    assert len(report.rows) == 3 * len(criteria.BASELINE_ARMS)
    assert [row.reference for row in report.rows[:3]] == ["rule_trend"] * 3
    assert all(row.primary for row in report.rows[:3])
    assert not any(row.primary for row in report.rows[3:])
    assert report.calendar.first == EARLIEST_START


def test_no_model_in_the_module_has_a_field_for_a_mean() -> None:
    """Lo que no está en el modelo no se puede imprimir por descuido."""
    models = [
        item
        for _, item in inspect.getmembers(module, inspect.isclass)
        if issubclass(item, FrozenModel) and item.__module__ == module.__name__
    ]
    assert ForwardPlan in models

    banned = ("mean", "avg", "average", "expected", "total", "centre")
    offenders = [
        f"{model.__name__}.{name}"
        for model in models
        for name in model.model_fields
        if any(word in name for word in banned)
    ]
    assert offenders == []


def test_the_report_cites_its_sources_and_recommends_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = render_report(report_of(scenario(monkeypatch)), SOURCES)

    for digest in ("c" * 64, "e" * 64, "f" * 64):
        assert f"`{digest}`" in text
    assert "`history/s0_1d.csv`" in text
    assert "δ registrado del estudio = 0.004 (0.400% de retorno bruto por evaluación)" in text
    assert "(registrado)" in text
    assert "Ninguna cifra de este reporte es una media de retorno." in text
    assert "Fin esperado, si ningún bloque sale vacío: bloque 50" in text
    assert "Fin más tardío posible: bloque 52" in text
    assert "bloques a la media" not in text, "no hay fila de un gasto que no puede ocurrir"
    assert (
        "**tope de gasto de un lote**: 3 x las evaluaciones de su manifiesto x 0.01509 USD" in text
    )
    assert "**cota del gasto del estudio que dan las reglas**: la suma de los topes" in text
    assert "al centavo" not in text, "el tope ya no es una cifra fija redondeada"
    assert text.count("Reproduce: `python -m crypto_agents.forward_plan") == 8
    assert "--delta" not in text
    lowered = text.lower()
    for word in ("conviene", "recom", "debería", "aconsej", "sugier", "vale la pena", "hay que"):
        assert word not in lowered


# ─────────────────────────────────── El comando, de un directorio ─────────────────────────────────


def run_command(directory: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> int:
    """El comando con menos remuestreos: el bootstrap no es lo que se prueba aquí."""
    monkeypatch.setattr(module, "DEFAULT_RESAMPLES", 200)
    return main([str(directory), "--history-dir", str(tmp_path / "history")])


def publish(
    found: Scenario,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    published: dict[UUID, dict[str, float]] | None = None,
) -> None:
    """Lo registrado de la etapa sintética, en el lugar del de la etapa 1: sus recuentos y sus
    retornos publicados, en un archivo con su sha-256."""
    path = tmp_path / "publicado.jsonl"
    rows = published_of(found) if published is None else published
    path.write_text(
        "".join(
            json.dumps({"run_id": str(run_id), "returns": returns}) + "\n"
            for run_id, returns in rows.items()
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "PUBLISHED_RETURNS", path)
    monkeypatch.setattr(module, "PUBLISHED_RETURNS_SHA256", file_sha256(path))
    monkeypatch.setattr(module, "POOL_COUNTS", counts_of(found))


def test_the_command_reads_a_run_cites_its_files_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    found = scenario(monkeypatch)
    directory = write_scenario(tmp_path, found)
    publish(found, tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    before = tree_snapshot(tmp_path)

    assert run_command(directory, tmp_path, monkeypatch) == 0

    assert tree_snapshot(tmp_path) == before
    out = capsys.readouterr().out
    for name in ("solo.jsonl", "always_buy.jsonl", "rule_trend.jsonl"):
        assert f"sha-256 `{file_sha256(directory / name)}`" in out
    assert f"sha-256 `{file_sha256(tmp_path / 'manifest.json')}`" in out
    for symbol in ("s0", "s1"):
        assert f"sha-256 `{file_sha256(tmp_path / 'history' / f'{symbol}_1d.csv')}`" in out
    assert (
        f"`python -m crypto_agents.forward_plan {directory} --history-dir {tmp_path / 'history'}`"
    ) in out
    assert "**NO**" not in out


def test_a_baseline_journal_that_disagrees_with_the_pool_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    found = scenario(monkeypatch)
    stage = dict(found.stage)
    stage["always_sell"] = [
        decided(stage["always_sell"][3], Action.HOLD),
        *stage["always_sell"][:3],
        *stage["always_sell"][4:],
    ]
    directory = write_scenario(tmp_path, found, stage)
    publish(found, tmp_path, monkeypatch)

    assert run_command(directory, tmp_path, monkeypatch) == 1
    assert "| always_sell | 40 | 40 | 1 | 1 | 1 | **NO** |" in capsys.readouterr().out


def test_an_invalid_run_exits_one_and_prints_no_figure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    found = scenario(monkeypatch)
    stage = dict(found.stage)
    stage["full"] = [*stage["full"], quota_lost()]
    directory = write_scenario(tmp_path, found, stage)
    publish(found, tmp_path, monkeypatch)

    assert run_command(directory, tmp_path, monkeypatch) == 1
    out = capsys.readouterr().out
    assert out.startswith("CORRIDA INVÁLIDA")
    assert "deff" not in out


def test_a_probe_and_a_missing_directory_exit_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    found = scenario(monkeypatch)
    probe = write_scenario(tmp_path, found, kind=RunKind.PROBE)
    publish(found, tmp_path, monkeypatch)

    assert run_command(probe, tmp_path, monkeypatch) == 2
    assert "es un sondeo" in capsys.readouterr().err
    assert run_command(tmp_path / "no-existe", tmp_path, monkeypatch) == 2
    assert "no es una corrida" in capsys.readouterr().err


def test_pool_counts_other_than_the_registered_exit_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    found = scenario(monkeypatch)
    directory = write_scenario(tmp_path, found)
    publish(found, tmp_path, monkeypatch)
    counts = counts_of(found)
    monkeypatch.setattr(
        module, "POOL_COUNTS", counts.model_copy(update={"unconfirmed": counts.unconfirmed + 1})
    )

    assert run_command(directory, tmp_path, monkeypatch) == 1
    assert "- coinciden: **NO**" in capsys.readouterr().out


def test_a_return_other_than_the_published_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    found = scenario(monkeypatch)
    directory = write_scenario(tmp_path, found)
    published = published_of(found)
    first = next(iter(published))
    published[first]["always_buy"] += 0.001
    publish(found, tmp_path, monkeypatch, published)

    assert run_command(directory, tmp_path, monkeypatch) == 1
    out = capsys.readouterr().out
    assert "| always_buy | 40 | 40 | 1 | **NO** |" in out
    assert "| always_sell | 40 | 40 | 0 | sí |" in out


def test_a_change_in_the_scoring_passes_the_journal_and_not_the_published_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Puntuar con una vela menos de horizonte cambia los dos lados de la atadura con el
    journal, que sigue diciendo que coinciden. Lo publicado no se mueve, y el comando sale con 1."""
    found = scenario(monkeypatch)
    directory = write_scenario(tmp_path, found)
    publish(found, tmp_path, monkeypatch)
    score_run = vars(criteria)["score_run"]

    def shorter(records: object, histories: object, horizon: int, *rest: object) -> object:
        return score_run(records, histories, horizon - 1, *rest)

    monkeypatch.setattr(criteria, "score_run", shorter)

    assert run_command(directory, tmp_path, monkeypatch) == 1
    out = capsys.readouterr().out
    for arm in criteria.BASELINE_ARMS:
        assert f"| {arm} | 40 | 40 | 0 | 0 | 0 | sí |" in out
    assert "| **NO** |" in out.split("Las líneas base contra los retornos publicados")[1]


def test_a_published_file_that_is_not_the_published_one_exits_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    found = scenario(monkeypatch)
    directory = write_scenario(tmp_path, found)
    publish(found, tmp_path, monkeypatch)
    monkeypatch.setattr(module, "PUBLISHED_RETURNS_SHA256", "0" * 64)

    assert run_command(directory, tmp_path, monkeypatch) == 2
    assert "no es el archivo publicado" in capsys.readouterr().err
