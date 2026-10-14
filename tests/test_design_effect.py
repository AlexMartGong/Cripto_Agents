"""Pruebas del efecto de diseño del pool.

Cuatro cosas tienen que ser ciertas para que el efecto de diseño sirva, y cada una tiene su
mutación:

- **La aritmética.** El error estándar agrupado sobre un vector con clústeres hechos a mano y
  valores calculados a mano. Quitar `G/(G - 1)`, agrupar por símbolo en lugar de por tiempo o
  centrar con la media del clúster tiene que romper la comprobación.
- **El universo.** Las series del pool son las de la ablación: mismo `run_id`, mismo instante,
  mismas propuestas que deja el arnés, y la puntuación de `criteria`.
- **Las submuestras.** Son las de `selection.py`: misma semilla, mismas velas.
- **La regla.** Percentil alto y no mediana, clústeres mínimos y no medianos, solo las series
  sustituto, y nada que no sea elegible.

El escenario de un directorio entero usa velas diarias: con bloques de un día hacen falta más de
cincuenta días para que un esquema llegue a ser elegible.
"""

from __future__ import annotations

import inspect
import math
import statistics
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, NamedTuple

import pytest

from crypto_agents import design_effect as module
from crypto_agents import metrics as metrics_module
from crypto_agents import paired_dispersion as recalculation_module
from crypto_agents import replay
from crypto_agents.ablation import ARMS, plan_from_manifest
from crypto_agents.activation_sweep import ActivatedBar
from crypto_agents.audit import RunKind, file_sha256
from crypto_agents.baselines import random_action
from crypto_agents.criteria import BASELINE_ARMS, paired_returns
from crypto_agents.design_effect import (
    BLOCK_DAYS,
    PLANNING_PERCENT,
    SUBSAMPLE_SIZES,
    SURROGATE_PAIRS,
    ClusteredError,
    Coactivation,
    DesignEffect,
    DesignEffectError,
    Overlap,
    Planning,
    PoolBar,
    PoolRecords,
    Series,
    SizeDeff,
    build_report,
    clustered_error,
    clusters_of,
    coactivation,
    deff_by_size,
    main,
    overlap_by_symbol,
    planning_deff,
    pool_run_id,
    pool_series,
    recomputed_n,
    render_report,
    stage_check,
    stage_ties,
    subsample,
)
from crypto_agents.indicators import DEFAULT_PRESET
from crypto_agents.market import write_ohlcv_csv
from crypto_agents.metrics import nearest_rank, paired_differences
from crypto_agents.paired_dispersion import (
    PAIRS,
    Dispersion,
    PairDispersion,
    Recalculation,
    SampleSize,
    SigmaRange,
    Sources,
    n_for_power,
    pool_cells,
    power_at,
)
from crypto_agents.selection import (
    PlannedEvaluation,
    SelectionManifest,
    SeriesRef,
    select_activations,
    select_bars,
    stratify,
    window_digest,
    write_manifest,
)
from crypto_agents.settings import DEFAULT_PRICING
from crypto_agents.state import Action, Billing, FrozenModel, IndicatorSet, MarketSnapshot, Proposal
from tests.conftest import PRESET, raw_ohlcv
from tests.test_ablation import run_baseline
from tests.test_criteria import DELTA, HORIZON, quota_lost, write_run
from tests.test_net_outcomes import mutated
from tests.test_paired_dispersion import snapshot as tree_snapshot
from tests.test_paired_dispersion import with_calls
from tests.test_selection import GATE
from tests.test_selection import histories as hourly_histories

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
    from uuid import UUID

    from crypto_agents.journal import EvaluationRecord

# ───────────────────────── El error agrupado es el de `metrics`, sin copia ────────────────────────
# Las pruebas del CR1 y de la asignación a bloques están con la función, en `tests/test_metrics.py`:
# desde la enmienda 4 la usa también `criteria`, que este módulo importa, y por eso vive en
# `metrics.py`.

DAY_ZERO = datetime(2026, 1, 1, tzinfo=UTC)

SHIFT = 0.05
"""Una constante sumada a todo: si la función leyera la media, el resultado se movería."""


def test_the_clustered_error_is_the_one_of_metrics_and_not_a_copy() -> None:
    """`criteria` juzga con este mismo objeto. Dos funciones podrían divergir; una, no."""
    assert module.clustered_errors is metrics_module.clustered_errors
    assert module.clustered_error is metrics_module.clustered_error
    assert module.clusters_of is metrics_module.clusters_of
    assert module.block_of is metrics_module.block_of
    assert module.ClusteredError is metrics_module.ClusteredError
    assert module.MIN_CLUSTERS is metrics_module.MIN_CLUSTERS == 50


def test_the_cluster_lengths_are_the_ones_fixed_before_the_numbers() -> None:
    assert BLOCK_DAYS == (1, 2, 3, 7)


# ──────────────────────── El pool sintético de la selección (velas de 1 h) ────────────────────────


WINDOW = 60
"""Velas que ve cada evaluación del pool sintético. Menos que la serie, a propósito: así la
ventana del replay no es la serie entera, y calcular los indicadores sobre una u otra se nota."""


def selected(target: int = 30, seed: int = 7) -> SelectionManifest:
    """Una selección sobre las series sintéticas de `test_selection`, con esa ventana."""
    return select_activations(
        hourly_histories(),
        timeframe="1h",
        target=target,
        strata=3,
        seed=seed,
        horizon=3,
        candle_limit=WINDOW,
        config=GATE,
        preset=PRESET,
        now=datetime(2026, 8, 15, tzinfo=UTC),
    )


def hourly_pool() -> tuple[SelectionManifest, dict[str, list[list[float]]], PoolRecords]:
    """El pool de las series sintéticas de `test_selection`, con el gate y el preset reales."""
    manifest = selected()
    series = hourly_histories()
    return manifest, series, module.pool_records(manifest, series, preset=PRESET, config=GATE)


def test_the_run_id_is_the_one_the_replay_gives() -> None:
    moment = datetime(2024, 10, 23, 12, tzinfo=UTC)

    assert module.REPLAY_NAMESPACE == replay.REPLAY_NAMESPACE
    assert pool_run_id("ADA/USDT", "4h", moment) == replay.replay_run_id("ADA/USDT", "4h", moment)
    assert pool_run_id("ADA/USDT", "4h", moment) != pool_run_id("ADA/USDT", "1h", moment)


def test_the_pool_is_every_confirmed_activation_evaluated_by_the_four_baselines() -> None:
    manifest, series, pool = hourly_pool()
    candidates = [bar for cells in pool.cells.values() for cell in cells for bar in cell]

    assert tuple(pool.arms) == BASELINE_ARMS
    assert len(pool.bars) == len(pool.confirmed) <= len(candidates)
    assert pool.activations == sum(ref.activations for ref in manifest.series)
    assert {(bar.symbol, bar.candle) for bar in pool.bars} == set(pool.confirmed)
    assert {(entry.symbol, entry.index) for entry in manifest.entries} <= set(pool.confirmed)
    for name in BASELINE_ARMS:
        assert [record.run_id for record in pool.arms[name]] == [bar.run_id for bar in pool.bars]

    hour = timedelta(hours=1)
    for bar, record in zip(pool.bars, pool.arms["random_uniform"], strict=True):
        opened = datetime.fromtimestamp(series[bar.symbol][bar.candle][0] / 1000, UTC)
        assert record.snapshot is not None
        assert record.snapshot.timestamp == opened
        assert record.at == bar.at == opened + hour, "el instante es el cierre de la vela"
        assert record.run_id == replay.replay_run_id(bar.symbol, "1h", bar.at)
        assert record.proposed is not None
        assert record.proposed.action is random_action(manifest.seed, record.run_id)
    assert {r.proposed.action for r in pool.arms["always_buy"] if r.proposed} == {Action.BUY}
    assert {r.proposed.action for r in pool.arms["always_sell"] if r.proposed} == {Action.SELL}


async def harness_proposals(
    manifest: SelectionManifest, series: dict[str, list[list[float]]]
) -> dict[str, dict[UUID, EvaluationRecord]]:
    """Lo que deja el arnés de la ablación, solo-caché, para cada línea base sobre el manifiesto."""
    plan = plan_from_manifest(manifest, series)
    journaled: dict[str, dict[UUID, EvaluationRecord]] = {}
    for name in BASELINE_ARMS:
        arm = next(item for item in ARMS if item.name == name)
        _, records, _ = await run_baseline(arm, plan)
        assert all(record.calls == () for record in records)
        journaled[name] = {record.run_id: record for record in records}
    return journaled


async def assert_the_direct_records_are_the_ones_the_harness_leaves() -> None:
    manifest, series, pool = hourly_pool()
    journaled = await harness_proposals(manifest, series)
    wanted = {(entry.symbol, entry.index) for entry in manifest.entries}
    ids = {bar.run_id for bar in pool.bars if (bar.symbol, bar.candle) in wanted}

    assert len(ids) == len(manifest.entries)
    for name in BASELINE_ARMS:
        assert set(journaled[name]) == ids, f"{name}: el arnés y el pool no evalúan lo mismo"
        ours = {record.run_id: record for record in pool.arms[name]}
        for run_id, theirs in journaled[name].items():
            mine = ours[run_id]
            assert mine.proposed == theirs.proposed
            assert mine.at == theirs.at
            assert mine.indicators == theirs.indicators
            assert mine.snapshot is not None
            assert theirs.snapshot is not None
            assert mine.snapshot.timestamp == theirs.snapshot.timestamp
            assert mine.snapshot.close == theirs.snapshot.close
            assert mine.snapshot.candles_digest == theirs.snapshot.candles_digest


@pytest.mark.asyncio
async def test_the_direct_records_are_the_ones_the_ablation_harness_leaves() -> None:
    """Las series no pasan por el arnés. Donde los dos caminos evalúan la misma vela, el registro
    es el mismo: id, instante, indicadores y propuesta, stop incluido."""
    await assert_the_direct_records_are_the_ones_the_harness_leaves()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("function", "old", "new"),
    [
        ("pool_records", "moment = bar.at + period", "moment = bar.at"),
        (
            "baseline_proposals",
            '"always_sell": always_sell_proposal(snapshot, indicators)',
            '"always_sell": always_buy_proposal(snapshot, indicators)',
        ),
        (
            "baseline_proposals",
            "random_proposal(snapshot, indicators, seed, run_id)",
            "random_proposal(snapshot, indicators, seed + 1, run_id)",
        ),
        (
            "evaluated",
            "to_dataframe(replay_window(rows, bar.index, candle_limit))",
            "to_dataframe(rows[: bar.index + 1])",
        ),
    ],
    ids=["la apertura como instante", "otra línea base", "otra semilla", "la serie entera"],
)
async def test_mutation_records_that_are_not_the_harness_ones_are_caught(
    function: str, old: str, new: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    mutant = mutated(getattr(module, function), old, new, module)
    monkeypatch.setattr(module, function, mutant)
    with pytest.raises(AssertionError):
        await assert_the_direct_records_are_the_ones_the_harness_leaves()


# ─────────────────────────── Las submuestras son las de la selección ──────────────────────────────

SELECTIONS = [(12, 7), (30, 7), (30, 99)]
"""(tamaño, semilla) de las selecciones que se comparan."""


def assert_the_subsamples_are_the_selections() -> None:
    _, _, pool = hourly_pool()
    for target, seed in SELECTIONS:
        chosen = selected(target=target, seed=seed)

        drawn = module.subsample(pool.cells, pool.confirmed, target, seed)

        assert drawn == {(entry.symbol, entry.index) for entry in chosen.entries}
        assert len(drawn) == target


def test_a_subsample_is_the_selection_of_that_size_and_seed() -> None:
    """Misma semilla, mismas submuestras: `select_activations` elige exactamente esas velas."""
    assert_the_subsamples_are_the_selections()


def test_mutation_a_subsample_drawn_with_another_seed_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mutant = mutated(
        subsample, "select_bars(cells, n, seed, ", "select_bars(cells, n, seed + 1, ", module
    )
    monkeypatch.setattr(module, "subsample", mutant)
    with pytest.raises(AssertionError):
        assert_the_subsamples_are_the_selections()


def test_the_subsamples_of_one_seed_are_nested_and_those_of_another_differ() -> None:
    _, _, pool = hourly_pool()

    small = subsample(pool.cells, pool.confirmed, 12, 7)
    large = subsample(pool.cells, pool.confirmed, 30, 7)
    other = subsample(pool.cells, pool.confirmed, 30, 99)

    assert small < large
    assert other != large


def test_drawing_a_subsample_leaves_the_cells_as_they_were() -> None:
    """El barajado es sobre copias: cada submuestra sale del mismo pool, no de sus restos."""
    _, _, pool = hourly_pool()
    before = {symbol: [list(cell) for cell in cells] for symbol, cells in pool.cells.items()}

    first = subsample(pool.cells, pool.confirmed, 30, 7)
    again = subsample(pool.cells, pool.confirmed, 30, 7)

    assert first == again
    assert pool.cells == before


def test_a_subsample_never_takes_an_unconfirmed_candle() -> None:
    cells = {"A": [[hand_bar("A", index) for index in range(8)]]}
    confirmed = {("A", index): ("range_breakout",) for index in (1, 3, 5, 7)}

    assert subsample(cells, confirmed, 4, 3) == set(confirmed)


# ───────────────────────────── Estructura de dependencia, a mano ──────────────────────────────────


def hand_bar(symbol: str, index: int) -> ActivatedBar:
    return ActivatedBar(
        symbol=symbol,
        timeframe="1d",
        index=index,
        at=DAY_ZERO + timedelta(days=index),
        triggers=("range_breakout",),
    )


def pool_bar(symbol: str, candle: int, at: datetime | None = None) -> PoolBar:
    moment = at if at is not None else DAY_ZERO + timedelta(days=candle + 1)
    return PoolBar(symbol, candle, 0, moment, pool_run_id(symbol, "1d", moment))


def assert_the_overlap_is_counted_by_hand() -> None:
    """Horizonte 6. En A: 10 y 12 distan 2; 30 y 36 distan 6, que ya no comparte vela, y ninguna
    de las dos tiene otra vecina cerca; 50 y 51 distan 1. Se solapan 10, 12, 50 y 51: cuatro de
    seis. En B, 10 y 100: ninguna."""
    bars = [pool_bar("A", candle) for candle in (36, 10, 51, 30, 50, 12)]
    bars += [pool_bar("B", candle) for candle in (10, 100)]

    found = {item.symbol: item for item in module.overlap_by_symbol(bars, 6)}

    assert found["A"] == Overlap(symbol="A", evaluations=6, overlapping=4)
    assert found["B"] == Overlap(symbol="B", evaluations=2, overlapping=0)


def test_the_overlap_counts_evaluations_that_share_a_candle_of_their_horizon() -> None:
    assert_the_overlap_is_counted_by_hand()


def test_mutation_an_overlap_that_reaches_one_candle_further_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mutant = mutated(
        overlap_by_symbol, "later - earlier < horizon", "later - earlier <= horizon", module
    )
    monkeypatch.setattr(module, "overlap_by_symbol", mutant)
    with pytest.raises(AssertionError):
        assert_the_overlap_is_counted_by_hand()


def test_the_coactivation_counts_candles_by_how_many_symbols_they_activate() -> None:
    first, second, third = (DAY_ZERO + timedelta(days=day) for day in (1, 2, 3))
    bars = [
        pool_bar("A", 0, first),
        pool_bar("B", 0, first),
        pool_bar("C", 0, first),
        pool_bar("A", 1, second),
        pool_bar("B", 1, second),
        pool_bar("A", 2, third),
    ]

    found = coactivation(bars)

    assert found == (
        Coactivation(symbols=1, candles=1),
        Coactivation(symbols=2, candles=1),
        Coactivation(symbols=3, candles=1),
    )
    assert [item.evaluations for item in found] == [1, 2, 3]


# ─────────────────────────── El escenario de un directorio (velas de 1 d) ─────────────────────────

SYMBOLS = ("S0", "S1")
BARS = 84
TIMEFRAME = "1d"
STRATA = 5
CANDLE_LIMIT = 5
TARGET = 40
MANIFEST_SEED = 11
UNCONFIRMED = {("S0", 2), ("S1", 41)}
"""Activaciones del barrido que la ventana del replay no confirma."""

SIZES = (40, 100, 140)
SUBSAMPLES = 6
SUBSAMPLE_SEED = 5
SCHEMES = (1, 2)
"""Con 81 días de pool, los bloques de un día pasan de 50 clústeres en n = 100 y n = 140, y los
de dos días no llegan nunca."""

RESAMPLES = 200
ONE_DAY = timedelta(days=1)


def closes(position: int) -> list[float]:
    """Cierres suaves y con memoria: el retorno de una vela se parece al de su vecina."""
    return [
        100.0 + 1.5 * math.sin(index / 3.0 + position) + 0.4 * math.sin(index * 1.7 + position)
        for index in range(BARS)
    ]


DAILY = {
    symbol: raw_ohlcv(closes(position), start=DAY_ZERO, step=ONE_DAY)
    for position, symbol in enumerate(SYMBOLS)
}


def swept(symbol: str) -> tuple[ActivatedBar, ...]:
    """El barrido sintético: el gate abre en todas las velas."""
    return tuple(
        ActivatedBar(
            symbol=symbol,
            timeframe=TIMEFRAME,
            index=index,
            at=DAY_ZERO + index * ONE_DAY,
            triggers=("range_breakout",),
        )
        for index in range(BARS)
    )


def confirmation(bar: ActivatedBar) -> tuple[str, ...] | None:
    return None if (bar.symbol, bar.index) in UNCONFIRMED else ("range_breakout",)


def indicators_at(index: int) -> IndicatorSet:
    """ATR fijo, pila de medias alcista y un ADX que no llega a 25 una vela de cada tres."""
    values = dict.fromkeys(DEFAULT_PRESET.column_names(), 50.0)
    values.update(
        {
            "ATRr_14": 2.0,
            "EMA_20": 99.0,
            "EMA_50": 98.0,
            "EMA_200": 97.0,
            "ADX_14": 30.0 if index % 3 else 10.0,
        }
    )
    return IndicatorSet(values=values)


def fake_world(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sustituye el barrido, la confirmación y los indicadores por los escritos a mano.

    El gate y el preset tienen sus pruebas, y con 84 velas el preset de producción no llega ni al
    warm-up. Lo que se prueba aquí es lo que el módulo hace con un pool, no cómo se abre el gate.
    """

    def bars(_rows: object, symbol: str, _timeframe: str, *_: object) -> tuple[ActivatedBar, ...]:
        return swept(symbol)

    def confirm(rows: list[list[float]], index: int, *_: object) -> tuple[str, ...] | None:
        symbol = next(name for name, series in DAILY.items() if series == rows)
        return None if (symbol, index) in UNCONFIRMED else ("range_breakout",)

    def evaluated(
        rows: list[list[float]], bar: ActivatedBar, run_id: UUID, *_: object
    ) -> tuple[MarketSnapshot, IndicatorSet]:
        snapshot = MarketSnapshot(
            run_id=run_id,
            exchange="binance",
            symbol=bar.symbol,
            timeframe=bar.timeframe,
            timestamp=bar.at,
            close=rows[bar.index][4],
            candles_digest="a" * 64,
            candles_count=CANDLE_LIMIT,
        )
        return snapshot, indicators_at(bar.index)

    for target in (module, recalculation_module):
        monkeypatch.setattr(target, "activated_bars", bars)
        monkeypatch.setattr(target, "confirm_activation", confirm)
    monkeypatch.setattr(module, "evaluated", evaluated)


def daily_manifest(seed: int = MANIFEST_SEED) -> SelectionManifest:
    """El manifiesto que la selección escribiría sobre el pool sintético con esa semilla."""
    cells = {symbol: stratify(DAILY[symbol], swept(symbol), STRATA, HORIZON) for symbol in SYMBOLS}
    picks = select_bars(cells, TARGET, seed, confirmation)
    entries = sorted(
        (
            PlannedEvaluation(
                symbol=pick.bar.symbol,
                timeframe=TIMEFRAME,
                index=pick.bar.index,
                at=pick.bar.at,
                candles_digest=window_digest(DAILY[pick.bar.symbol], pick.bar.index, CANDLE_LIMIT),
                triggers=pick.triggers,
                stratum=pick.stratum,
            )
            for pick in picks
        ),
        key=lambda entry: (entry.symbol, entry.index),
    )
    return SelectionManifest(
        created_at=DAY_ZERO,
        seed=seed,
        target=TARGET,
        strata=STRATA,
        timeframe=TIMEFRAME,
        warmup_bars=1,
        candle_limit=CANDLE_LIMIT,
        horizon=HORIZON,
        series=tuple(
            SeriesRef(
                symbol=symbol,
                timeframe=TIMEFRAME,
                bars=BARS,
                first=DAY_ZERO,
                last=DAY_ZERO + (BARS - 1) * ONE_DAY,
                digest="b" * 64,
                activations=BARS,
            )
            for symbol in SYMBOLS
        ),
        entries=tuple(entries),
    )


def decided(record: EvaluationRecord, action: Action) -> EvaluationRecord:
    """La misma evaluación, decidida por un brazo con modelo."""
    stop = {Action.BUY: 90.0, Action.SELL: 110.0, Action.HOLD: None}[action]
    proposal = Proposal(
        action=action,
        confidence=0.5,
        size_fraction=0.0 if action is Action.HOLD else 0.1,
        invalidation_price=stop,
        rationale="Decisión sintética de un brazo con modelo.",
    )
    return record.model_copy(update={"proposal": proposal})


def stage_of(pool: PoolRecords, manifest: SelectionManifest) -> dict[str, list[EvaluationRecord]]:
    """La corrida sintética: los dos brazos con modelo y las cuatro líneas base del manifiesto.

    Las líneas base son los registros del pool en las velas del manifiesto, así que la atadura
    se cumple salvo que la prueba la rompa. `solo` rota entre comprar, vender y esperar; `full`
    compra una de cada dos.
    """
    wanted = {(entry.symbol, entry.index) for entry in manifest.entries}
    keep = [
        position for position, bar in enumerate(pool.bars) if (bar.symbol, bar.candle) in wanted
    ]
    base = [pool.arms["always_buy"][position] for position in keep]
    rotation = (Action.BUY, Action.SELL, Action.HOLD)
    return {
        "full": with_calls(
            [
                decided(record, Action.BUY if index % 2 == 0 else Action.HOLD)
                for index, record in enumerate(base)
            ],
            step=2.0,
        ),
        "solo": with_calls(
            [decided(record, rotation[index % 3]) for index, record in enumerate(base)], step=1.0
        ),
        **{name: [pool.arms[name][position] for position in keep] for name in BASELINE_ARMS},
    }


class Scenario(NamedTuple):
    manifest: SelectionManifest
    pool: PoolRecords
    stage: dict[str, list[EvaluationRecord]]
    recalculation: Recalculation


def scenario(monkeypatch: pytest.MonkeyPatch) -> Scenario:
    fake_world(monkeypatch)
    manifest = daily_manifest()
    pool = module.pool_records(manifest, DAILY)
    stage = stage_of(pool, manifest)
    calls = {
        name: [call for record in records for call in record.calls]
        for name, records in stage.items()
    }
    recalculation = recalculation_module.build_report(
        stage,
        DAILY,
        HORIZON,
        DELTA,
        manifest,
        pool_cells(manifest, DAILY),
        calls,
        DEFAULT_PRICING,
        Billing.PAYG,
        RESAMPLES,
        seed=1,
    )
    return Scenario(manifest, pool, stage, recalculation)


def report_of(found: Scenario, **changes: object) -> DesignEffect:
    arguments: dict[str, object] = {
        "stage": found.stage,
        "manifest": found.manifest,
        "pool": found.pool,
        "recalculation": found.recalculation,
        **changes,
    }
    return build_report(
        arguments["stage"],  # type: ignore[arg-type]
        DAILY,
        HORIZON,
        DELTA,
        arguments["manifest"],  # type: ignore[arg-type]
        arguments["pool"],  # type: ignore[arg-type]
        arguments["recalculation"],  # type: ignore[arg-type]
        SIZES,
        SUBSAMPLES,
        SUBSAMPLE_SEED,
        SCHEMES,
    )


def test_the_scenario_is_a_valid_recalculation(monkeypatch: pytest.MonkeyPatch) -> None:
    """El control del escenario: `paired_dispersion` lo da por bueno antes de medir nada más."""
    found = scenario(monkeypatch)

    assert len(found.manifest.entries) == TARGET
    assert len(found.pool.bars) == 2 * 81 - len(UNCONFIRMED)
    assert len(found.recalculation.pairs) == len(PAIRS)
    assert found.recalculation.consistent


# ─────────────────────────────── Las series y su puntuación ───────────────────────────────────────


def test_the_series_are_the_four_baselines_and_their_six_differences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    found = scenario(monkeypatch)

    series = pool_series(found.pool, DAILY, HORIZON)

    assert len(SURROGATE_PAIRS) == 6
    assert [item.name for item in series if not item.surrogate] == list(BASELINE_ARMS)
    assert [item.name for item in series if item.surrogate] == [
        f"{arm} - {reference}" for arm, reference in SURROGATE_PAIRS
    ]
    assert {len(item.values) for item in series} == {len(found.pool.bars)}


def assert_the_series_are_scored_as_criteria_scores(found: Scenario) -> None:
    """Cada serie, contra lo que `criteria.paired_returns` da para ese par, por `run_id`."""
    by_name = {item.name: item.values for item in module.pool_series(found.pool, DAILY, HORIZON)}
    order = [bar.run_id for bar in found.pool.bars]
    for arm, reference in SURROGATE_PAIRS:
        aligned = paired_returns(arm, reference, found.pool.arms, DAILY, HORIZON)
        assert not isinstance(aligned, str)
        ids = [record.run_id for record in aligned.arm_records]
        differences = dict(
            zip(
                ids,
                paired_differences(aligned.arm_returns, aligned.reference_returns),
                strict=True,
            )
        )
        own = dict(zip(ids, aligned.arm_returns, strict=True))

        assert by_name[f"{arm} - {reference}"] == tuple(differences[run_id] for run_id in order)
        assert by_name[arm] == tuple(own[run_id] for run_id in order)
    assert any(value != 0.0 for value in by_name["always_buy - always_sell"])


def test_the_series_are_the_vectors_criteria_would_subtract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert_the_series_are_scored_as_criteria_scores(scenario(monkeypatch))


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (
            "paired_differences(aligned.arm_returns, aligned.reference_returns)",
            "paired_differences(aligned.reference_returns, aligned.arm_returns)",
        ),
        ("differences[run_id] for run_id in order", "differences[run_id] for run_id in ids"),
    ],
    ids=["la resta al revés", "el orden de criteria y no el del pool"],
)
def test_mutation_a_series_that_is_not_the_one_of_criteria_is_caught(
    old: str, new: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    found = scenario(monkeypatch)
    monkeypatch.setattr(module, "pool_series", mutated(pool_series, old, new, module))
    with pytest.raises(AssertionError):
        assert_the_series_are_scored_as_criteria_scores(found)


def test_the_module_scores_through_criteria_and_nothing_else(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Un espía en `paired_returns`: los seis pares del pool pasan por ahí, con el horizonte."""
    found = scenario(monkeypatch)
    seen: list[tuple[str, str, int]] = []
    real = paired_returns

    def spy(arm: str, reference: str, arms: object, histories: object, horizon: int) -> object:
        seen.append((arm, reference, horizon))
        return real(arm, reference, arms, histories, horizon)  # type: ignore[arg-type]

    monkeypatch.setattr(module, "paired_returns", spy)
    pool_series(found.pool, DAILY, HORIZON)

    assert seen == [(arm, reference, HORIZON) for arm, reference in SURROGATE_PAIRS]


# ─────────────────────────────── La atadura con la corrida ────────────────────────────────────────


def test_the_pool_and_the_run_agree_on_every_evaluation_they_share(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    found = scenario(monkeypatch)

    ties = stage_ties(found.pool, found.stage, DAILY, HORIZON)

    assert [tie.arm for tie in ties] == list(BASELINE_ARMS)
    for tie in ties:
        assert (tie.evaluations, tie.found) == (TARGET, TARGET)
        assert (tie.actions, tie.stops, tie.returns) == (0, 0, 0)
        assert tie.exact


def test_a_baseline_that_decided_otherwise_in_the_run_breaks_the_tie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Una sola evaluación de `always_buy` que en el journal vendió: acción, stop y retorno."""
    found = scenario(monkeypatch)
    stage = dict(found.stage)
    stage["always_buy"] = [decided(stage["always_buy"][0], Action.SELL), *stage["always_buy"][1:]]

    ties = {tie.arm: tie for tie in stage_ties(found.pool, stage, DAILY, HORIZON)}

    assert (ties["always_buy"].actions, ties["always_buy"].stops) == (1, 1)
    assert ties["always_buy"].returns == 1
    assert not ties["always_buy"].exact
    assert ties["always_sell"].exact
    assert not report_of(found, stage=stage).consistent


def test_an_evaluation_of_the_run_that_the_pool_does_not_hold_breaks_the_tie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Con otro `run_id` el pool no encuentra las evaluaciones de la corrida: no hay atadura."""
    found = scenario(monkeypatch)
    stage = dict(found.stage)
    stray = stage["rule_trend"][0].model_copy(update={"run_id": pool_run_id("S0", "4h", DAY_ZERO)})
    stage["rule_trend"] = [stray, *stage["rule_trend"][1:]]

    ties = {tie.arm: tie for tie in stage_ties(found.pool, stage, DAILY, HORIZON)}

    assert (ties["rule_trend"].evaluations, ties["rule_trend"].found) == (TARGET, TARGET - 1)
    assert not ties["rule_trend"].exact


def test_a_run_without_a_baseline_cannot_be_tied(monkeypatch: pytest.MonkeyPatch) -> None:
    found = scenario(monkeypatch)
    stage = {name: records for name, records in found.stage.items() if name != "rule_trend"}

    with pytest.raises(DesignEffectError, match="no trae la línea base rule_trend"):
        stage_ties(found.pool, stage, DAILY, HORIZON)


def test_a_run_without_evaluations_of_a_baseline_is_not_an_exact_tie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    found = scenario(monkeypatch)

    ties = stage_ties(found.pool, {**found.stage, "always_buy": []}, DAILY, HORIZON)

    assert not ties[0].exact


def test_the_subsample_with_the_manifest_seed_is_the_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    found = scenario(monkeypatch)
    other = daily_manifest(seed=MANIFEST_SEED + 1)

    assert report_of(found).manifest_reproduced
    assert not report_of(
        found, manifest=other.model_copy(update={"seed": MANIFEST_SEED})
    ).consistent


def test_the_confirmed_cells_are_the_ones_the_recalculation_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Las confirmadas por celda salen de dos pasadas distintas y tienen que ser las mismas."""
    found = scenario(monkeypatch)
    shorter = found.pool._replace(bars=found.pool.bars[1:])

    assert report_of(found).cells_agree
    assert not module.cells_agree(shorter, found.recalculation.pool)


# ─────────────────────────────── Efecto de diseño en función de n ─────────────────────────────────


def expected_rows(
    found: Scenario, series: Sequence[Series]
) -> dict[tuple[int, str, int], SizeDeff]:
    """Lo que `deff_by_size` tiene que dar, recalculado aquí submuestra a submuestra."""
    place = {(bar.symbol, bar.candle): position for position, bar in enumerate(found.pool.bars)}
    rows: dict[tuple[int, str, int], SizeDeff] = {}
    for n in SIZES:
        for days in SCHEMES:
            groups: list[int] = []
            deffs: dict[str, list[float]] = {item.name: [] for item in series}
            for k in range(SUBSAMPLES):
                drawn = subsample(found.pool.cells, found.pool.confirmed, n, SUBSAMPLE_SEED + k)
                chosen = sorted(place[key] for key in drawn)
                clusters = clusters_of([found.pool.bars[position] for position in chosen], days)
                groups.append(len(set(clusters)))
                for item in series:
                    error = clustered_error(
                        [item.values[position] for position in chosen], clusters
                    )
                    if error.deff is not None:
                        deffs[item.name].append(error.deff)
            for item in series:
                ordered = sorted(deffs[item.name])
                rows[(n, item.name, days)] = SizeDeff(
                    n=n,
                    series=item.name,
                    surrogate=item.surrogate,
                    days=days,
                    subsamples=SUBSAMPLES,
                    clusters_min=min(groups),
                    clusters_median=sorted(groups)[SUBSAMPLES // 2 - 1],
                    undetermined=SUBSAMPLES - len(ordered),
                    deff_median=ordered[len(ordered) // 2 - 1] if ordered else None,
                    deff_high=ordered[-1] if ordered else None,
                )
    return rows


def assert_the_deff_by_size_is_the_one_recomputed_here(found: Scenario) -> None:
    """Con seis submuestras el percentil 90 por rango más cercano es la mayor y la mediana la
    tercera: `ceil(0.9·6) = 6` y `ceil(0.5·6) = 3`."""
    series = pool_series(found.pool, DAILY, HORIZON)
    wanted = expected_rows(found, series)

    rows = module.deff_by_size(series, found.pool, SIZES, SUBSAMPLES, SUBSAMPLE_SEED, SCHEMES)

    assert PLANNING_PERCENT == 90
    assert len(rows) == len(SIZES) * len(series) * len(SCHEMES) == len(wanted)
    for row in rows:
        expected = wanted[(row.n, row.series, row.days)]
        assert row.model_dump(exclude={"deff_median", "deff_high"}) == expected.model_dump(
            exclude={"deff_median", "deff_high"}
        )
        assert row.deff_median == pytest.approx(expected.deff_median)
        assert row.deff_high == pytest.approx(expected.deff_high)
    spread = [row for row in rows if row.clusters_min != row.clusters_median]
    assert spread, "si G no cambia entre submuestras, mínimo y mediana no se distinguen"
    assert any(row.deff_high != row.deff_median for row in rows if row.deff_high is not None), (
        "si deff no cambia entre submuestras, percentil y mediana no se distinguen"
    )


def test_the_deff_by_size_is_the_median_and_the_high_percentile_across_subsamples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert_the_deff_by_size_is_the_one_recomputed_here(scenario(monkeypatch))


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("nearest_rank(found, PLANNING_PERCENT)", "nearest_rank(found, 50)"),
        ("clusters_min=min(groups[days])", "clusters_min=max(groups[days])"),
        ("n, seed + k)", "n, seed)"),
        ("groups[days].append(len(set(clusters)))", "groups[days].append(len(clusters))"),
    ],
    ids=["la mediana por percentil", "el G mayor por el menor", "una sola semilla", "G = n"],
)
def test_mutation_a_deff_by_size_computed_another_way_is_caught(
    old: str, new: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    found = scenario(monkeypatch)
    monkeypatch.setattr(module, "deff_by_size", mutated(deff_by_size, old, new, module))
    with pytest.raises(AssertionError):
        assert_the_deff_by_size_is_the_one_recomputed_here(found)


def test_every_size_draws_the_same_seeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """K semillas consecutivas desde la declarada, las mismas para cada n."""
    found = scenario(monkeypatch)
    series = pool_series(found.pool, DAILY, HORIZON)
    seen: list[tuple[int, int]] = []
    real = module.subsample

    def spy(cells: object, confirmed: object, n: int, seed: int) -> frozenset[tuple[str, int]]:
        seen.append((n, seed))
        return real(cells, confirmed, n, seed)  # type: ignore[arg-type]

    monkeypatch.setattr(module, "subsample", spy)
    deff_by_size(series, found.pool, SIZES, SUBSAMPLES, SUBSAMPLE_SEED, SCHEMES)

    assert seen == [(n, SUBSAMPLE_SEED + k) for n in SIZES for k in range(SUBSAMPLES)]


def test_the_grid_and_the_number_of_subsamples_are_the_declared_ones() -> None:
    assert SUBSAMPLE_SIZES == (140, 280, 525, 980, 1960, 3290, 3605)
    assert module.DEFAULT_SUBSAMPLES == 200
    assert module.DEFAULT_SUBSAMPLE_SEED == 20261012
    assert all(n % 35 == 0 for n in SUBSAMPLE_SIZES)


# ──────────────────────────────────── La regla de planificación ───────────────────────────────────


def size_row(
    n: int,
    series: str,
    days: int,
    high: float | None,
    median: float | None = 1.0,
    surrogate: bool = True,
    clusters_min: int = 120,
    clusters_median: int = 125,
) -> SizeDeff:
    return SizeDeff(
        n=n,
        series=series,
        surrogate=surrogate,
        days=days,
        subsamples=200,
        clusters_min=clusters_min,
        clusters_median=clusters_median,
        undetermined=0 if high is not None else 200,
        deff_median=median,
        deff_high=high,
    )


RULE_ROWS = [
    size_row(140, "a - b", 1, high=1.30, median=1.10),
    size_row(140, "a - b", 7, high=2.50, clusters_min=40, clusters_median=55),
    size_row(140, "a - c", 2, high=1.45, median=1.00),
    size_row(140, "a", 1, high=3.00, surrogate=False),
    size_row(140, "b - c", 3, high=None, median=None),
    size_row(280, "a - b", 7, high=2.00, clusters_min=49),
    size_row(525, "a - b", 1, high=0.90, median=0.80),
]
"""En n = 140 la regla da 1.45, de `a - c` con L = 2: la línea base sola (3.00) no entra, el
esquema con un G mínimo de 40 (2.50) tampoco aunque su G mediano sea 55, y una serie sin deff no
cuenta. En n = 280 no hay ninguno elegible. En n = 525 el único vale 0.90 y así se queda."""


def assert_the_planning_rule_gives_the_hand_picked_rows() -> None:
    assert module.planning_deff(RULE_ROWS) == (
        Planning(n=140, deff=1.45, series="a - c", days=2),
        Planning(n=280),
        Planning(n=525, deff=0.90, series="a - b", days=1),
    )


def test_the_planning_deff_is_the_largest_high_percentile_among_what_is_eligible() -> None:
    """Sin suelo en 1: la regla no lo pone, y un 0.90 se queda en 0.90."""
    assert_the_planning_rule_gives_the_hand_picked_rows()


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("row.surrogate and ", ""),
        ("row.eligible and ", ""),
        ("key=lambda row: row.deff_high or 0.0", "key=lambda row: row.deff_median or 0.0"),
        ("deff=worst.deff_high", "deff=worst.deff_median"),
        ("worst = max(admitted", "worst = min(admitted"),
    ],
    ids=[
        "entran las líneas base",
        "entra lo no elegible",
        "ordena por mediana",
        "da la mediana",
        "el menor",
    ],
)
def test_mutation_a_planning_rule_that_is_not_the_declared_one_is_caught(
    old: str, new: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "planning_deff", mutated(planning_deff, old, new, module))
    with pytest.raises(AssertionError):
        assert_the_planning_rule_gives_the_hand_picked_rows()


def test_mutation_eligibility_by_the_median_cluster_count_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Elegible es que *ninguna* submuestra baje de 50 clústeres, no que la típica no baje."""
    declared = inspect.getattr_static(SizeDeff, "eligible")
    assert isinstance(declared, property)
    getter = declared.fget
    assert getter is not None
    mutant = mutated(
        getter,
        "self.clusters_min >= MIN_CLUSTERS",
        "self.clusters_median >= MIN_CLUSTERS",
        module,
    )
    monkeypatch.setattr(SizeDeff, "eligible", mutant)
    with pytest.raises(AssertionError):
        assert_the_planning_rule_gives_the_hand_picked_rows()


def test_the_scenario_has_sizes_with_and_without_an_eligible_scheme(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Con 40 evaluaciones no puede haber 50 clústeres; con 100 y 140, los bloques de un día sí."""
    report = report_of(scenario(monkeypatch))
    by_n = {plan.n: plan for plan in report.planning}

    assert by_n[40] == Planning(n=40)
    for n in (100, 140):
        assert by_n[n].days == 1
        assert by_n[n].series in {f"{arm} - {reference}" for arm, reference in SURROGATE_PAIRS}
        wanted = max(
            row.deff_high
            for row in report.by_size
            if row.n == n and row.surrogate and row.days == 1 and row.deff_high is not None
        )
        assert by_n[n].deff == wanted
    assert not any(row.eligible for row in report.by_size if row.days == 2)


# ───────────────────────────────── La comprobación sobre la corrida ───────────────────────────────


def test_the_stage_check_clusters_the_differences_criteria_subtracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    found = scenario(monkeypatch)

    checks, missing = stage_check(found.stage, DAILY, HORIZON, SCHEMES)

    assert missing == ()
    assert [(row.arm, row.reference, row.days) for row in checks] == [
        (arm, reference, days) for arm, reference in PAIRS for days in SCHEMES
    ]
    for row in checks:
        aligned = paired_returns(row.arm, row.reference, found.stage, DAILY, HORIZON)
        assert not isinstance(aligned, str)
        differences = paired_differences(aligned.arm_returns, aligned.reference_returns)
        expected = clustered_error(differences, clusters_of(aligned.arm_records, row.days))
        assert row.error == expected
        assert row.error.n == TARGET
        assert row.error.naive == pytest.approx(statistics.stdev(differences) / math.sqrt(TARGET))
        assert not row.error.eligible, "40 evaluaciones no dan 50 clústeres"


def test_the_naive_error_of_the_check_is_the_one_the_recalculation_reproduces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """El EE ingenuo es `sigma_D/√n`, el que `paired_dispersion` ya ató al de `criteria`."""
    found = scenario(monkeypatch)
    published = {(pair.arm, pair.reference): pair for pair in found.recalculation.pairs}

    checks, _ = stage_check(found.stage, DAILY, HORIZON, SCHEMES)

    for row in checks:
        pair = published[(row.arm, row.reference)]
        assert row.error.naive == pytest.approx(pair.dispersion.stderr)
        assert row.error.naive == pytest.approx(pair.criteria_stderr)


def test_a_pair_the_run_does_not_have_is_missing_and_not_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    found = scenario(monkeypatch)
    stage = {name: records for name, records in found.stage.items() if name != "full"}

    checks, missing = stage_check(stage, DAILY, HORIZON, SCHEMES)

    assert {(item.arm, item.reference) for item in missing} == {
        pair for pair in PAIRS if "full" in pair
    }
    assert {row.arm for row in checks} == {"solo"}


# ──────────────────────────────────────── El n recalculado ────────────────────────────────────────

HAND_SIGMA = 0.04 / 3
HAND_POWER_N = 348.848
"""`((1.96 + 0.8416)·s/δ)²` con δ = 0.002 y s = 0.04/3: el de `tests/test_paired_dispersion.py`."""


def pair_with(
    reference: str, low: float, point: float, high: float, arm: str = "solo"
) -> PairDispersion:
    sizes = (
        SampleSize(sigma=low, exact=0.0, rounded=0),
        SampleSize(sigma=point, exact=0.0, rounded=0),
        SampleSize(sigma=high, exact=0.0, rounded=0),
    )
    return PairDispersion(
        arm=arm,
        reference=reference,
        agreed=0,
        nonzero=0,
        nonzero_agreed=0,
        dispersion=Dispersion(
            n=140,
            sigma=point,
            bootstrap=SigmaRange(low=low, high=high),
            half_width_n=sizes,
            power_n=sizes,
        ),
        criteria_stderr=point / math.sqrt(140),
    )


HAND_PLANNING = (
    Planning(n=280, deff=1.0, series="a - b", days=1),
    Planning(n=525, deff=1.6, series="a - b", days=1),
    Planning(n=980, deff=2.5, series="a - c", days=3),
    Planning(n=1960, deff=2.6, series="a - c", days=3),
)
"""Con 348.848: 280 no llega ni con deff 1; 525 pide 558.2; 980 pide 872.1 y cumple."""


def assert_the_recomputed_n_is_the_hand_computed_one() -> None:
    pairs = [pair_with("always_buy", HAND_SIGMA / 2, HAND_SIGMA, 4 * HAND_SIGMA)]

    low, point, high = module.recomputed_n(pairs, HAND_PLANNING, DELTA)

    assert point.independent == pytest.approx(HAND_POWER_N, rel=1e-5)
    assert point.independent == pytest.approx(n_for_power(HAND_SIGMA, DELTA))
    assert (point.n, point.deff) == (980, 2.5)
    assert point.required == pytest.approx(HAND_POWER_N * 2.5, rel=1e-5)
    assert (point.label, point.sigma) == ("puntual", HAND_SIGMA)

    assert low.independent == pytest.approx(HAND_POWER_N / 4, rel=1e-5)
    assert (low.label, low.n, low.deff) == ("extremo bajo", 280, 1.0)

    assert high.independent == pytest.approx(HAND_POWER_N * 16, rel=1e-5)
    assert (high.label, high.n, high.deff, high.required) == ("extremo alto", None, None, None)


def test_the_recomputed_n_is_the_smallest_of_the_grid_that_satisfies_the_inequality() -> None:
    assert_the_recomputed_n_is_the_hand_computed_one()


def assert_the_planned_power_is_the_hand_computed_one() -> None:
    """La pregunta al revés, sobre la misma rejilla. Con sigma = 0.04/3 y δ = 0.002:

    - en n = 980 con deff 2.5 cuentan 392: EE = sigma/√392 = 0.00067344, δ/EE = 2.9698,
      y la potencia es Φ(2.9698 - 1.96) = Φ(1.0098) = 0.8437;
    - en n = 525 con deff 1.6 cuentan 328.1: Φ(2.7171 - 1.96) = Φ(0.7571) = 0.7755, por debajo
      de 80%, que es por lo que 525 no cumplía arriba;
    - donde ningún esquema es elegible no hay efecto de diseño ni potencia: no se le supone 1.
    """
    by_n = {plan.n: plan for plan in HAND_PLANNING}

    assert module.planned_power(HAND_SIGMA, by_n[980], DELTA) == pytest.approx(0.8437, abs=5e-5)
    assert module.planned_power(HAND_SIGMA, by_n[525], DELTA) == pytest.approx(0.7755, abs=5e-5)
    assert module.planned_power(HAND_SIGMA, by_n[280], DELTA) == pytest.approx(
        power_at(HAND_SIGMA, 280, DELTA)
    ), "con deff 1 las evaluaciones cuentan como independientes"
    assert module.planned_power(HAND_SIGMA, Planning(n=140), DELTA) is None


def test_the_planned_power_is_the_power_of_that_n_with_its_planning_deff() -> None:
    assert_the_planned_power_is_the_hand_computed_one()


def test_mutation_a_planned_power_that_ignores_the_design_effect_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mutant = mutated(module.planned_power, "delta, deff)", "delta)", module)
    monkeypatch.setattr(module, "planned_power", mutant)
    with pytest.raises(AssertionError):
        assert_the_planned_power_is_the_hand_computed_one()


# ─────────────────────── El suelo de la enmienda 4, solo en la potencia ───────────────────────────

NARROWED = Planning(n=400, deff=0.5, series="a - b", days=7)
"""Un n cuyo deff de planificación baja de 1: el agrupado saldría menor que el ingenuo."""


def assert_the_power_never_counts_more_than_n_evaluations() -> None:
    """`criteria` juzga con `max(EE ingenuo, EE agrupado)`, que en deff es `max(1, deff)`.

    A mano: con deff 0.5 las 400 evaluaciones contarían como 800; con el suelo cuentan como
    400, y la potencia es la de 400 independientes. Con deff 2.5 el suelo no actúa: 980 cuentan
    como 392, lo de antes. Sin deff no hay nada que usar.
    """
    by_n = {plan.n: plan for plan in HAND_PLANNING}

    assert module.used_deff(NARROWED) == 1.0
    assert module.effective_n(NARROWED) == 400.0
    power = module.planned_power(HAND_SIGMA, NARROWED, DELTA)
    assert power is not None
    assert power == pytest.approx(power_at(HAND_SIGMA, 400, DELTA))
    assert power < power_at(HAND_SIGMA, 400, DELTA, 0.5), (
        "sin suelo la potencia sería mayor que la de 400 independientes"
    )

    assert module.used_deff(by_n[980]) == 2.5
    assert module.effective_n(by_n[980]) == pytest.approx(392.0)
    assert module.effective_n(by_n[525]) == pytest.approx(328.125)
    assert module.effective_n(by_n[280]) == 280.0

    assert module.used_deff(Planning(n=140)) is None
    assert module.effective_n(Planning(n=140)) is None


def test_the_power_uses_the_planning_deff_with_a_floor_at_one() -> None:
    assert_the_power_never_counts_more_than_n_evaluations()


def test_mutation_a_power_without_the_floor_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    mutant = mutated(module.used_deff, "return max(1.0, plan.deff)", "return plan.deff", module)
    monkeypatch.setattr(module, "used_deff", mutant)
    with pytest.raises(AssertionError):
        assert_the_power_never_counts_more_than_n_evaluations()


def test_the_rule_fixed_before_the_numbers_has_no_floor() -> None:
    """La sección 7 es la regla previa a la enmienda: con deff 0.5 pide la mitad, sin suelo.

    El suelo es de la potencia de la sección 8. Si entrara aquí, las secciones 1 a 7 dejarían de
    ser las que «Efecto de diseño» publicó.
    """
    assert module.required_at(300.0, NARROWED) == 150.0
    assert module.satisfies(300.0, NARROWED)
    (plan,) = planning_deff([size_row(400, "a - b", 7, high=0.5, clusters_min=60)])
    assert plan.deff == 0.5, "la regla de planificación tampoco lo pone"


def effective_rows(report: DesignEffect) -> list[list[str]]:
    section = render_report(report, SOURCES).split("\n## 8. ")[1].split("\n## ")[0]
    table = section.split("| n | deff de planificación | deff usado | n efectivo |")[1]
    rows = []
    for line in table.splitlines()[2:]:
        if not line.startswith("| "):
            break
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    return rows


def test_the_effective_n_table_is_n_over_the_used_deff_of_each_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cada fila es un n de la rejilla: su deff de planificación, el usado y `n / deff usado`."""
    report = report_of(scenario(monkeypatch))
    plans = sorted(report.planning, key=lambda plan: plan.n)
    rows = effective_rows(report)

    assert len(rows) == len(plans)
    for cells, plan in zip(rows, plans, strict=True):
        assert cells[0] == str(plan.n)
        if plan.deff is None:
            assert cells[1:] == ["no determinado: ningún esquema elegible", "—", "—"]
            continue
        used = max(1.0, plan.deff)
        assert cells[1:] == [f"{plan.deff:.3f}", f"{used:.3f}", f"{plan.n / used:.1f}"]
    assert any(plan.deff is None for plan in plans), "el escenario tiene un n sin esquema elegible"


def test_the_report_says_where_the_floor_acts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Un n con deff de planificación menor que 1 se nombra; sin ninguno, se dice que no actúa."""
    found = scenario(monkeypatch)
    report = report_of(found)
    judged = [plan for plan in report.planning if plan.deff is not None]
    below = sorted(plan.n for plan in judged if plan.deff is not None and plan.deff < 1.0)
    text = render_report(report, SOURCES)

    assert "`max(EE ingenuo, EE agrupado)`" in text
    assert "La sección 7 es la regla fijada antes de los números y no lo lleva." in text
    if below:
        sizes = ", ".join(str(n) for n in below)
        assert f"el suelo actúa en {len(below)} de {len(judged)} n: {sizes}." in text
    else:
        assert f"el suelo no actúa en ninguno de los {len(judged)} n con deff." in text

    narrowed = report.model_copy(
        update={"planning": (*report.planning, NARROWED.model_copy(update={"n": 7}))}
    )
    forced = render_report(narrowed, SOURCES)
    assert f"el suelo actúa en {len(below) + 1} de {len(judged) + 1} n: " in forced
    row = next(cells for cells in effective_rows(narrowed) if cells[0] == "7")
    assert row == ["7", "0.500", "1.000", "7.0"]


def test_the_power_table_is_the_planned_power_of_each_row_and_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cada celda es `planned_power` de esa fila en ese n, a un decimal de porcentaje, y un n sin
    efecto de diseño de planificación dice «no determinado» en lugar de una potencia."""
    report = report_of(scenario(monkeypatch))
    section = render_report(report, SOURCES).split("\n## 8. ")[1].split("\n## ")[0]
    plans = sorted(report.planning, key=lambda plan: plan.n)
    rows = [line for line in section.splitlines() if line.startswith("| solo |")]

    assert "Potencia de `solo` contra cada línea base" in section
    assert len(rows) == len(report.recomputed)
    for line, row in zip(rows, report.recomputed, strict=True):
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        assert cells[:3] == [row.arm, row.reference, row.label]
        expected = [module.planned_power(row.sigma, plan, DELTA) for plan in plans]
        assert cells[4:] == [
            "no determinado" if power is None else f"{power:.1%}" for power in expected
        ]
    assert any(plan.deff is None for plan in plans), "el escenario tiene un n sin esquema elegible"


@pytest.mark.parametrize(
    ("function", "old", "new"),
    [
        ("required_at", "return independent * plan.deff", "return independent"),
        (
            "required_at",
            "return independent * plan.deff",
            "return independent * math.sqrt(plan.deff)",
        ),
        ("satisfies", "needed <= plan.n", "needed <= plan.n * 2"),
        (
            "recomputed_n",
            "n_for_power(sigma, delta, power)",
            "n_for_power(sigma, delta * 2, power)",
        ),
        (
            "recomputed_n",
            "sorted(planning, key=lambda plan: plan.n)",
            "sorted(planning, key=lambda plan: -plan.n)",
        ),
    ],
    ids=["sin efecto de diseño", "con su raíz", "con holgura", "otro delta", "el mayor n"],
)
def test_mutation_an_n_recomputed_another_way_is_caught(
    function: str, old: str, new: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, function, mutated(getattr(module, function), old, new, module))
    with pytest.raises(AssertionError):
        assert_the_recomputed_n_is_the_hand_computed_one()


def test_a_size_without_a_planning_deff_cannot_be_the_answer() -> None:
    """Sin esquema elegible un n no se puede juzgar: se salta, aunque el siguiente cueste más."""
    planning = (Planning(n=980), Planning(n=1960, deff=5.0, series="a - b", days=1))

    _, point, _ = recomputed_n([pair_with("rule_trend", 0.0, HAND_SIGMA, 1.0)], planning, DELTA)

    assert (point.n, point.deff) == (1960, 5.0)


def test_only_the_target_arm_against_each_baseline_is_recomputed() -> None:
    pairs = [
        pair_with("solo", HAND_SIGMA, HAND_SIGMA, HAND_SIGMA, arm="full"),
        pair_with("always_buy", HAND_SIGMA, HAND_SIGMA, HAND_SIGMA, arm="full"),
        *(pair_with(name, HAND_SIGMA, HAND_SIGMA, HAND_SIGMA) for name in BASELINE_ARMS),
    ]

    rows = recomputed_n(pairs, HAND_PLANNING, DELTA)

    assert module.TARGET_ARM == "solo"
    assert [(row.arm, row.reference) for row in rows] == [
        ("solo", name) for name in BASELINE_ARMS for _ in range(3)
    ]


def test_no_observed_mean_reaches_the_n(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sumar una constante al retorno de un brazo mueve su media y nada de lo que da el n."""
    found = scenario(monkeypatch)
    aligned = paired_returns("solo", "always_buy", found.stage, DAILY, HORIZON)
    assert not isinstance(aligned, str)
    differences = paired_differences(aligned.arm_returns, aligned.reference_returns)
    clusters = clusters_of(aligned.arm_records, 1)

    plain = clustered_error(differences, clusters)
    shifted = clustered_error([value + SHIFT for value in differences], clusters)

    assert statistics.fmean(differences) != pytest.approx(statistics.fmean(differences) + SHIFT)
    assert shifted.naive == pytest.approx(plain.naive)
    assert shifted.clustered == pytest.approx(plain.clustered)
    assert shifted.deff == pytest.approx(plain.deff)


# ─────────────────────────────────────────── El reporte ───────────────────────────────────────────

SOURCES = Sources(
    directory="var/ablation/corrida",
    command="python -m crypto_agents.design_effect var/ablation/corrida --delta 0.002",
    commit="`abc1234` (árbol limpio)",
    plan="`manifest.json` (manifest), sha-256 `" + "c" * 64 + "`",
    journals=(("full.jsonl", "d" * 64), ("solo.jsonl", "e" * 64)),
    histories=(("S0", "history/s0_1d.csv", "f" * 64),),
)


def test_no_model_in_the_module_has_a_field_for_a_mean() -> None:
    """Lo que no está en el modelo no se puede imprimir por descuido."""
    models = [
        item
        for _, item in inspect.getmembers(module, inspect.isclass)
        if issubclass(item, FrozenModel) and item.__module__ == module.__name__
    ]
    assert DesignEffect in models
    models.append(ClusteredError)  # vive en `metrics` desde la enmienda 4, y el reporte lo imprime

    banned = ("mean", "avg", "average", "expected", "total", "centre")
    offenders = [
        f"{model.__name__}.{name}"
        for model in models
        for name in model.model_fields
        if any(word in name for word in banned)
    ]
    assert offenders == []


def test_the_report_prints_no_mean(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ni la media de una serie del pool ni la de una diferencia de la corrida aparecen. Sus
    errores estándar sí."""
    found = scenario(monkeypatch)
    report = report_of(found)
    text = render_report(report, SOURCES)

    series = {item.name: item.values for item in pool_series(found.pool, DAILY, HORIZON)}
    for name in ("always_sell", "always_buy - rule_trend"):
        mean = f"{statistics.fmean(series[name]):.3%}"
        assert mean not in {"0.000%", "-0.000%"}
        assert mean not in text
    whole = next(row for row in report.pool if row.series == "always_sell" and row.days == 1)
    assert f"{whole.error.naive:.3%}" in text
    assert "media de retorno" in text


def test_the_report_recommends_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Da el efecto de diseño, n, coste y viabilidad. Qué etapa correr no le toca."""
    text = render_report(report_of(scenario(monkeypatch)), SOURCES).lower()

    for word in ("conviene", "recom", "debería", "aconsej", "sugier", "vale la pena", "hay que"):
        assert word not in text
    for verdict in ("justifica su coste", "derrota", "es mejor", "es peor"):
        assert verdict not in text


def test_the_report_cites_every_digest_and_says_what_it_fixed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = render_report(report_of(scenario(monkeypatch)), SOURCES)

    for digest in ("c" * 64, "d" * 64, "e" * 64, "f" * 64):
        assert f"`{digest}`" in text
    assert "`history/s0_1d.csv`" in text
    assert "`abc1234`" in text
    assert "δ = 0.002 (0.200% de retorno por evaluación)" in text
    assert f"semilla del manifiesto de la corrida, {MANIFEST_SEED}" in text
    assert f"las semillas {SUBSAMPLE_SEED} a {SUBSAMPLE_SEED + SUBSAMPLES - 1}" in text
    assert "L ∈ {1, 2}" in text
    assert "n ∈ {40, 100, 140}" in text
    assert "G < 50 se reporta y no es elegible" in text
    assert "retorno bruto" in text


def test_every_table_says_which_command_reproduces_it(monkeypatch: pytest.MonkeyPatch) -> None:
    text = render_report(report_of(scenario(monkeypatch)), SOURCES)

    sections = text.split("\n## ")[1:]
    numbered = [section for section in sections if section[0].isdigit()]
    assert len(numbered) == 8
    for section in numbered:
        assert "| --- |" in section
        assert f"Reproduce: `{SOURCES.command}`" in section
    assert "python -m crypto_agents.criteria var/ablation/corrida --delta 0.002" in text
    assert "python -m crypto_agents.paired_dispersion var/ablation/corrida --delta 0.002" in text


def test_the_report_has_the_rows_it_promises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Diez series por dos esquemas sobre el pool, por tres n entre submuestras; nueve pares por
    dos esquemas sobre la corrida; cuatro líneas base por tres valores de sigma_D."""
    report = report_of(scenario(monkeypatch))
    text = render_report(report, SOURCES)
    sections = {section.split("\n", 1)[0][:2]: section for section in text.split("\n## ")[1:]}

    def rows(key: str, start: str) -> list[str]:
        return [line for line in sections[key].splitlines() if line.startswith(start)]

    assert len(rows("1.", "| always_")) == 2
    assert len(rows("2.", "| S")) == 2
    assert len(rows("2.", "| todos |")) == 1
    assert len(rows("3.", "| ")) - 2 == 10 * len(SCHEMES)
    assert sections["3."].count("| sustituto |") == 6 * len(SCHEMES)
    assert len(rows("4.", "| ")) - 2 == len(SIZES) * 10 * len(SCHEMES)
    assert [line.split("|")[1].strip() for line in rows("5.", "| ")[2:]] == ["40", "100", "140"]
    assert "| 40 | no determinado: ningún esquema elegible | — | — |" in sections["5."]
    assert len(rows("6.", "| full |")) + len(rows("6.", "| solo |")) == len(PAIRS) * len(SCHEMES)
    assert [line.split("|")[1:3] for line in rows("6.", "| ")[2 : 2 + len(SCHEMES)]] == [
        [f" {days} ", f" {len(PAIRS)} "] for days in SCHEMES
    ]
    assert len(rows("7.", "| solo |")) == 2 * len(BASELINE_ARMS) * 3, "el n y lo que pide cada n"
    assert "| en n = 40 | en n = 100 | en n = 140 |" in sections["7."]
    assert len(rows("8.", "| solo |")) == len(BASELINE_ARMS) * 3, "la potencia en cada n"
    assert "| en n = 40 | en n = 100 | en n = 140 |" in sections["8."]
    assert "**NO**" not in text


def test_a_tie_that_fails_is_marked_in_the_report(monkeypatch: pytest.MonkeyPatch) -> None:
    found = scenario(monkeypatch)
    stage = dict(found.stage)
    stage["always_buy"] = [decided(stage["always_buy"][0], Action.SELL), *stage["always_buy"][1:]]

    text = render_report(report_of(found, stage=stage), SOURCES)

    assert "| always_buy | 40 | 40 | 1 | 1 | 1 | **NO** |" in text
    assert "| always_sell | 40 | 40 | 0 | 0 | 0 | sí |" in text


def test_the_recomputed_rows_carry_the_room_and_the_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    """Un n que cumple va con su celda, si cabe, lo adicional, el coste y la pared de `solo`."""
    found = scenario(monkeypatch)
    report = report_of(found).model_copy(
        update={
            "recomputed": (
                module.RecomputedN(
                    arm="solo",
                    reference="always_buy",
                    label="puntual",
                    sigma=0.01,
                    independent=90.0,
                    n=100,
                    deff=1.1,
                    required=99.0,
                ),
                module.RecomputedN(
                    arm="solo",
                    reference="always_sell",
                    label="puntual",
                    sigma=0.05,
                    independent=900.0,
                ),
            )
        }
    )
    costs = {cost.arm: cost for cost in found.recalculation.costs}
    clocks = {clock.arm: clock for clock in found.recalculation.clocks}
    each, pace = costs["solo"].per_evaluation, clocks["solo"].per_evaluation
    assert each is not None
    assert pace is not None

    section = render_report(report, SOURCES).split("\n## 7.")[1]
    row = next(line for line in section.splitlines() if line.startswith("| solo | always_buy "))
    cells = [cell.strip() for cell in row.split("|")[1:-1]]

    assert cells[:8] == ["solo", "always_buy", "puntual", "1.000%", "90.0", "100", "1.100", "99.0"]
    assert cells[8] == "10", "100 entre las 10 celdas"
    assert cells[9].startswith("cabe (10 ≤ ")
    assert cells[11] == "60"
    assert cells[12] == f"{each * 100:.2f}"
    assert cells[13] == f"{each * 60:.2f}"
    assert cells[14] == f"{pace * 100 / 3600:.1f}"
    assert "| solo | always_sell | puntual | 5.000% | 900.0 | ninguno de la rejilla |" in section

    plans = {plan.n: plan for plan in report.planning}
    needs = [line for line in section.splitlines() if line.startswith("| solo | always_buy ")][1]
    asked = [cell.strip() for cell in needs.split("|")[4:-1]]
    assert asked[0] == "no determinado", "n = 40 no tiene efecto de diseño de planificación"
    for cell, n in zip(asked[1:], (100, 140), strict=True):
        deff = plans[n].deff
        assert deff is not None
        assert cell == f"{90.0 * deff:.1f} {'≤' if 90.0 * deff <= n else '>'} {n}"


# ─────────────────────────────────── El comando, de un directorio ─────────────────────────────────


def write_scenario(
    tmp_path: Path,
    found: Scenario,
    stage: dict[str, list[EvaluationRecord]] | None = None,
    **meta: object,
) -> Path:
    """Históricos, manifiesto y corrida en disco, como los lee el comando."""
    history_dir = tmp_path / "history"
    for symbol, rows in DAILY.items():
        write_ohlcv_csv(history_dir / f"{symbol.lower()}_{TIMEFRAME}.csv", rows)
    plan = tmp_path / "manifest.json"
    if not plan.exists():
        write_manifest(plan, found.manifest)
    return write_run(
        tmp_path,
        found.stage if stage is None else stage,
        plan=plan,
        billing=Billing.PAYG,
        **meta,  # type: ignore[arg-type]
    )


def run_command(directory: Path, tmp_path: Path, *extra: str) -> int:
    return main(
        [
            str(directory),
            "--delta",
            str(DELTA),
            "--history-dir",
            str(tmp_path / "history"),
            "--resamples",
            str(RESAMPLES),
            "--subsamples",
            str(SUBSAMPLES),
            "--sizes",
            ",".join(str(n) for n in SIZES),
            *extra,
        ]
    )


def test_the_command_reads_a_run_cites_its_files_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    found = scenario(monkeypatch)
    directory = write_scenario(tmp_path, found)
    monkeypatch.chdir(tmp_path)
    before = tree_snapshot(tmp_path)

    assert run_command(directory, tmp_path) == 0

    assert tree_snapshot(tmp_path) == before
    out = capsys.readouterr().out
    for name in ("full.jsonl", "solo.jsonl", "always_buy.jsonl", "rule_trend.jsonl"):
        assert f"sha-256 `{file_sha256(directory / name)}`" in out
    assert f"sha-256 `{file_sha256(tmp_path / 'manifest.json')}`" in out
    for symbol in ("s0", "s1"):
        assert f"sha-256 `{file_sha256(tmp_path / 'history' / f'{symbol}_1d.csv')}`" in out
    assert (
        f"{directory} --delta 0.002 --history-dir {tmp_path / 'history'} --resamples 200 "
        "--subsamples 6 --sizes 40,100,140"
    ) in out
    assert "| rule_trend | 40 | 40 | 0 | 0 | 0 | sí |" in out
    assert "es el manifiesto de la corrida: sí" in out
    assert "**NO**" not in out


def test_the_command_is_reproducible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    found = scenario(monkeypatch)
    directory = write_scenario(tmp_path, found)

    assert run_command(directory, tmp_path) == 0
    first = capsys.readouterr().out
    assert run_command(directory, tmp_path) == 0
    assert capsys.readouterr().out == first
    assert run_command(directory, tmp_path, "--subsample-seed", "77") == 0
    assert capsys.readouterr().out != first


def sizing_flags(tmp_path: Path) -> list[str]:
    return [
        "--history-dir",
        str(tmp_path / "history"),
        "--resamples",
        str(RESAMPLES),
        "--subsamples",
        str(SUBSAMPLES),
        "--sizes",
        ",".join(str(n) for n in SIZES),
    ]


def test_without_the_argument_the_command_uses_the_registered_delta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """δ es el de la enmienda 2 se escriba o no: la línea «Reproduce» es la misma."""
    directory = write_scenario(tmp_path, scenario(monkeypatch))

    assert run_command(directory, tmp_path) == 0
    with_flag = capsys.readouterr().out
    assert main([str(directory), *sizing_flags(tmp_path)]) == 0

    assert capsys.readouterr().out == with_flag
    assert "δ no registrado" not in with_flag


def test_a_delta_that_is_not_the_registered_one_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as raised:
        main([str(tmp_path), "--delta", "0.2"])

    assert raised.value.code == 2
    assert "--delta 0.2 es δ = 20.000%" in capsys.readouterr().err


def test_an_exploratory_delta_is_marked_and_cannot_be_reproduced_as_registered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """El n depende de δ: con otro, el reporte lo dice en la cabecera y en cada «Reproduce»."""
    directory = write_scenario(tmp_path, scenario(monkeypatch))

    code = main([str(directory), "--exploratory-delta", "0.004", *sizing_flags(tmp_path)])
    out = capsys.readouterr().out

    assert code == 0
    assert "**δ no registrado**: 0.400%. El δ de la enmienda 2 es 0.200%" in out
    assert f"{directory} --exploratory-delta 0.004 --history-dir" in out
    assert "--delta 0.004" not in out
    assert "--delta 0.002" not in out


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

    assert run_command(directory, tmp_path) == 1
    assert "| always_sell | 40 | 40 | 1 | 1 | 1 | **NO** |" in capsys.readouterr().out


def test_mutation_a_random_line_drawn_with_another_seed_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """La semilla es la del manifiesto. Con otra, la línea al azar del pool no es la del journal."""
    found = scenario(monkeypatch)
    directory = write_scenario(tmp_path, found)
    mutant = mutated(
        module.baseline_proposals,
        "random_proposal(snapshot, indicators, seed, run_id)",
        "random_proposal(snapshot, indicators, seed + 1, run_id)",
        module,
    )
    monkeypatch.setattr(module, "baseline_proposals", mutant)

    assert run_command(directory, tmp_path) == 1
    out = capsys.readouterr().out
    row = next(line for line in out.splitlines() if line.startswith("| random_uniform | 40 | 40 "))
    assert row.endswith("| **NO** |")
    assert "| always_buy | 40 | 40 | 0 | 0 | 0 | sí |" in out


def test_an_invalid_run_exits_one_and_prints_no_figure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Una corrida que el criterio 6 invalida no se interpreta, tampoco para medir su pool."""
    found = scenario(monkeypatch)
    stage = dict(found.stage)
    stage["full"] = [*stage["full"], quota_lost()]
    directory = write_scenario(tmp_path, found, stage)

    assert run_command(directory, tmp_path) == 1

    out = capsys.readouterr().out
    assert out.startswith("CORRIDA INVÁLIDA")
    assert "deff" not in out


def test_a_probe_and_a_missing_directory_exit_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    found = scenario(monkeypatch)
    probe = write_scenario(tmp_path, found, kind=RunKind.PROBE)

    assert run_command(probe, tmp_path) == 2
    assert "es un sondeo" in capsys.readouterr().err
    assert run_command(tmp_path / "no-existe", tmp_path) == 2
    assert "no es una corrida" in capsys.readouterr().err


def test_a_size_the_pool_cannot_deliver_exits_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    found = scenario(monkeypatch)
    directory = write_scenario(tmp_path, found)

    assert run_command(directory, tmp_path, "--sizes", "40,400") == 2
    assert (
        "solo se pudieron confirmar 160 activaciones de las 400 pedidas" in capsys.readouterr().err
    )


def test_the_arguments_must_make_sense(tmp_path: Path) -> None:
    for extra in (
        ["--delta", "0"],
        ["--delta", "0.2"],
        ["--exploratory-delta", "0"],
        ["--delta", "0.002", "--exploratory-delta", "0.004"],
        ["--delta", "0.002", "--resamples", "0"],
        ["--delta", "0.002", "--subsamples", "0"],
        ["--delta", "0.002", "--sizes", "140,x"],
        ["--delta", "0.002", "--sizes", "1"],
    ):
        with pytest.raises(SystemExit):
            main([str(tmp_path), *extra])


def test_the_percentile_is_a_value_that_occurred() -> None:
    """Rango más cercano, sin interpolar: con 200 submuestras el percentil 90 es la 180ª."""
    values = [float(index) for index in range(1, 201)]

    assert nearest_rank(values, PLANNING_PERCENT) == 180.0
    assert nearest_rank(values, 50) == 100.0
