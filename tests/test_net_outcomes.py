"""Retorno neto de una posición: comisión, deslizamiento y funding, calculados a mano.

El neto es `bruto - 2·(taker + slippage) - lado · Σ tasas`, con la suma sobre las liquidaciones
de `(entrada, salida]`. Todo se escribe a mano para poder poner cada liquidación exactamente
donde la prueba la quiere: en el instante de la entrada, en el de la salida, un instante después.

Las velas son de 4 h y las liquidaciones de 8 h, como las reales. La orden se emite en la vela 1,
que abre a las 04:00 y cierra a las 08:00: **su entrada cae justo en una liquidación**, y es esa
la que no se paga. Con el horizonte de 6 velas la salida es el cierre de la vela 7, a las 32:00
—otra liquidación—, y esa sí.

    hora  0h      8h      16h     24h     32h     40h
    tasa  +0.0009 +0.0005 +0.0001 +0.0003 -0.0002 +0.0007
          antes   entrada  ─── pagadas ───  salida  después
          (8h lleva 13 ms de jitter, 24h lleva 7)

Suma pagada: 0.0001 + 0.0003 - 0.0002 = 0.0002. Con taker 0.001 y deslizamiento 0.0005 el viaje de
ida y vuelta cuesta 0.003.

Las tres mutaciones que el neto no puede sobrevivir se prueban de verdad: se reescribe el código
de producción con el error puesto y se exige que la comprobación a mano **falle**. Una mutación
que ninguna prueba nota no es una prueba, es un adorno.
"""

from __future__ import annotations

import inspect
import textwrap
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest

from crypto_agents import funding as funding_module
from crypto_agents import outcomes as outcomes_module
from crypto_agents.funding import FundingSeries, snap_to_settlement
from crypto_agents.outcomes import (
    NetInputs,
    Outcome,
    OutcomeError,
    ScoredRun,
    Scoring,
    TradeOutcome,
    net_return,
    score_position,
    score_record,
    score_run,
)
from crypto_agents.perp_probe import FundingRow
from crypto_agents.settings import DEFAULT_COSTS, CostModel
from crypto_agents.state import Action
from tests.test_outcomes import position_at, record_at

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from crypto_agents.journal import EvaluationRecord

START = datetime(2026, 8, 1, tzinfo=UTC)
HOUR_MS = 3_600_000
BAR_MS = 4 * HOUR_MS
ORIGIN_MS = int(START.timestamp() * 1000)
COSTS = CostModel(taker_fee=0.001, slippage=0.0005)
"""Viaje de ida y vuelta: 2 · (0.001 + 0.0005) = 0.003."""

ENTRY_INDEX = 1
HORIZON = 6
STOP_LONG, STOP_SHORT = 90.0, 120.0

# (hora desde START, milisegundos de jitter, tasa)
SETTLEMENTS = (
    (0, 0, 0.0009),
    (8, 13, 0.0005),
    (16, 0, 0.0001),
    (24, 7, 0.0003),
    (32, 0, -0.0002),
    (40, 0, 0.0007),
)
PAID = 0.0001 + 0.0003 - 0.0002


def series(rows: tuple[tuple[int, int, float], ...] = SETTLEMENTS) -> FundingSeries:
    return FundingSeries.from_rows(
        [
            FundingRow(timestamp_ms=ORIGIN_MS + hour * HOUR_MS + jitter, rate=rate)
            for hour, jitter, rate in rows
        ]
    )


def candles(
    touch_at: int | None = None, touch_low: float = 85.0, touch_high: float = 130.0
) -> list[list[float]]:
    """Velas de 4 h: la 1 es la de entrada (cierre 100) y la 7 cierra en 110.

    Con `touch_at`, esa vela se ensancha hasta `touch_low` y `touch_high`: el stop largo en 90 y
    el corto en 120 saltan ahí, cada uno por su lado, y ninguna otra vela los toca.
    """
    bars = [
        (100, 100, 100),
        (101, 99, 100),
        (103, 99, 102),
        (106, 101, 105),
        (108, 104, 107),
        (109, 105, 108),
        (111, 107, 109),
        (112, 108, 110),
        (115, 109, 114),
    ]
    out: list[list[float]] = []
    for index, (high, low, close) in enumerate(bars):
        out.append(
            [
                float(ORIGIN_MS + index * BAR_MS),
                float(close),
                float(touch_high) if index == touch_at else float(high),
                float(touch_low) if index == touch_at else float(low),
                float(close),
                1000.0,
            ]
        )
    return out


def order(side: Action, history: list[list[float]]) -> TradeOutcome:
    """La posición resuelta de una orden en la vela de entrada, con el stop declarado."""
    stop = STOP_LONG if side is Action.BUY else STOP_SHORT
    moment = START + timedelta(hours=4 * ENTRY_INDEX)
    record = record_at(ENTRY_INDEX, side, 100.0, stop)
    assert record.snapshot is not None
    record = record.model_copy(
        update={"snapshot": record.snapshot.model_copy(update={"timestamp": moment})}
    )
    outcome = score_record(record, history, HORIZON)
    assert outcome is not None
    return outcome


def net(
    side: Action, touch_at: int | None = None, funding: FundingSeries | None = None
) -> float | None:
    """El neto de la orden de prueba. Llama a `net_return` por el módulo: las mutaciones lo ven."""
    history = candles(touch_at)
    return outcomes_module.net_return(
        order(side, history), history, series() if funding is None else funding, COSTS
    )


# ───────────────────────────────── Cifras calculadas a mano ───────────────────────────────────


def assert_hand_computed_figures() -> None:
    """Todo lo que el neto tiene que dar. Las mutaciones se miden contra esto."""
    # Largo que llega al horizonte: bruto +0.10, costes 0.003, paga 0.0002 de funding.
    assert net(Action.BUY) == pytest.approx(0.0968, abs=1e-12)
    # Corto que llega al horizonte: bruto -0.10, costes 0.003, COBRA 0.0002.
    assert net(Action.SELL) == pytest.approx(-0.1028, abs=1e-12)
    # Largo con el stop tocado en la vela 4 (abre a las 16:00): bruto -0.10, paga la de las 16:00.
    assert net(Action.BUY, touch_at=4) == pytest.approx(-0.1031, abs=1e-12)
    # Tocado en la vela 3 (abre a las 12:00): la de las 16:00 es el cierre de la vela, no se paga.
    assert net(Action.BUY, touch_at=3) == pytest.approx(-0.103, abs=1e-12)
    # Tocado en la propia vela siguiente a la entrada: la ventana queda vacía, y la liquidación
    # del instante de entrada (08:00 con 13 ms de jitter) tampoco se paga.
    assert net(Action.BUY, touch_at=2) == pytest.approx(-0.103, abs=1e-12)
    # Corto con el stop tocado en la vela 3 (abre a las 12:00, cierra a las 16:00): bruto -0.20
    # (sale en 120), y la liquidación POSITIVA de las 16:00 —que un corto cobraría— no entra.
    assert net(Action.SELL, touch_at=3) == pytest.approx(-0.203, abs=1e-12)
    # Tocado en la 4 (abre a las 16:00): la de las 16:00 es la de la apertura y sí entra.
    assert net(Action.SELL, touch_at=4) == pytest.approx(-0.2029, abs=1e-12)
    # Sin la liquidación de las 24:00 el hueco de 16 h no es de la rejilla: no determinado.
    holed = series(tuple(row for row in SETTLEMENTS if row[0] != 24))
    assert net(Action.BUY, funding=holed) is None


def test_the_net_matches_the_figures_computed_by_hand() -> None:
    assert_hand_computed_figures()


def test_gross_is_untouched_and_the_costs_are_the_settings_model() -> None:
    history = candles()
    outcome = order(Action.BUY, history)
    assert outcome.outcome is Outcome.HELD
    assert outcome.gross_return == pytest.approx(0.10)
    assert COSTS.round_trip == pytest.approx(0.003)
    assert DEFAULT_COSTS.round_trip == pytest.approx(2 * (0.0005 + 0.0002))
    assert net_return(outcome, history, series(), COSTS) == pytest.approx(
        outcome.gross_return - COSTS.round_trip - PAID
    )


def test_a_short_collects_the_funding_a_long_pays() -> None:
    """Misma ventana, mismas tasas: el largo resta Σ y el corto la suma."""
    long_costs = 0.10 - COSTS.round_trip
    short_costs = -0.10 - COSTS.round_trip
    assert net(Action.BUY) == pytest.approx(long_costs - PAID)
    assert net(Action.SELL) == pytest.approx(short_costs + PAID)


def test_a_negative_funding_pays_the_long_and_costs_the_short() -> None:
    negative = series(tuple((h, j, -abs(r)) for h, j, r in SETTLEMENTS))
    assert net(Action.BUY, funding=negative) > 0.10 - COSTS.round_trip  # type: ignore[operator]
    assert net(Action.SELL, funding=negative) < -0.10 - COSTS.round_trip  # type: ignore[operator]


# ───────────────────────── Salida por stop: la ventana acaba en la apertura de la vela ─────────

# Orden en la vela 1 (entrada a las 08:00). La liquidación de las 16:00 es POSITIVA y grande, y
# un corto la cobraría. Es el cierre de la vela 3 (12:00-16:00) y la apertura de la 4.
POSITIVE_AT_16 = ((0, 0, 0.0009), (8, 13, 0.0005), (16, 0, 0.0004), (24, 0, 0.0003), (32, 0, 0.0))
SHORT_STOP_GROSS = -0.20  # entra a 100, el stop en 120 salta: -(120 - 100) / 100


def assert_the_short_stop_window_ends_at_the_open_of_the_candle() -> None:
    funding = series(POSITIVE_AT_16)
    # Stop dentro de [12:00, 16:00): la ventana es (08:00, 12:00], vacía. La liquidación de las
    # 16:00 —el cierre de la vela, t + 4 h— queda fuera, y con ella el cobro del corto.
    inside = net(Action.SELL, touch_at=3, funding=funding)
    assert inside == pytest.approx(SHORT_STOP_GROSS - COSTS.round_trip, abs=1e-12)
    # La misma orden, si el stop salta en la vela que ABRE a las 16:00: ahora sí es la apertura,
    # la ventana es (08:00, 16:00] y el corto cobra 0.0004.
    on_open = net(Action.SELL, touch_at=4, funding=funding)
    assert on_open == pytest.approx(SHORT_STOP_GROSS - COSTS.round_trip + 0.0004, abs=1e-12)


def test_a_short_stopped_inside_a_candle_does_not_collect_the_settlement_at_its_close() -> None:
    """Corto, liquidación positiva en t + 4 h, stop dentro de [t, t + 4 h): el neto no entra."""
    assert_the_short_stop_window_ends_at_the_open_of_the_candle()


@pytest.mark.parametrize("rate_at_close", [-0.01, 0.0, 0.0004, 0.05])
def test_the_rate_at_the_close_of_the_stop_candle_does_not_move_the_net(
    rate_at_close: float,
) -> None:
    """Sea cual sea la tasa de las 16:00, el corto parado dentro de la vela 3 da lo mismo."""
    rows = tuple((h, j, rate_at_close if h == 16 else r) for h, j, r in POSITIVE_AT_16)
    for side in (Action.SELL, Action.BUY):
        base = net(side, touch_at=3, funding=series(POSITIVE_AT_16))
        assert net(side, touch_at=3, funding=series(rows)) == base


def test_a_stop_candle_needs_its_close_in_the_series_to_know_nothing_fell_inside() -> None:
    """Para descartar una liquidación a mitad de vela la serie tiene que llegar a su cierre."""
    reaching_only_the_open = series(tuple(row for row in POSITIVE_AT_16 if row[0] <= 8))
    assert net(Action.SELL, touch_at=3, funding=reaching_only_the_open) is None


def test_a_long_stopped_inside_the_same_candle_does_not_pay_it_either() -> None:
    funding = series(POSITIVE_AT_16)
    assert net(Action.BUY, touch_at=3, funding=funding) == pytest.approx(
        -0.10 - COSTS.round_trip, abs=1e-12
    )


def test_mutation_the_short_stop_exit_at_the_candle_close_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Con el cierre como fin de la ventana el corto cobraría 0.0004 que el stop no llegó a ver."""
    assert_the_short_stop_window_ends_at_the_open_of_the_candle()  # control: el código real pasa
    mutant = mutated(
        net_return,
        "paid = funding.over(entry_ms, exit_ms, exit_before_ms=exit_ms + bar_ms)",
        "paid = funding.over(entry_ms, exit_ms + bar_ms)",
        outcomes_module,
    )
    monkeypatch.setattr(outcomes_module, "net_return", mutant)
    with pytest.raises(AssertionError):
        assert_the_short_stop_window_ends_at_the_open_of_the_candle()
    # Y el neto que daría la mutación es exactamente el que incluye la liquidación cobrada.
    leaked = net(Action.SELL, touch_at=3, funding=series(POSITIVE_AT_16))
    assert leaked == pytest.approx(SHORT_STOP_GROSS - COSTS.round_trip + 0.0004, abs=1e-12)


# ──────────────────────────────── La ventana es (entrada, salida] ─────────────────────────────


def test_the_settlement_at_the_entry_instant_is_not_paid_even_with_jitter() -> None:
    """08:00:00.013 es la liquidación de las 08:00, la del instante de entrada: no se paga."""
    entry_ms = ORIGIN_MS + 8 * HOUR_MS
    jittered = ORIGIN_MS + 8 * HOUR_MS + 13
    assert jittered > entry_ms, "sin ajustar, la marca sería «estrictamente posterior»"
    assert snap_to_settlement(jittered) == entry_ms
    assert series().over(entry_ms, entry_ms + 24 * HOUR_MS) == pytest.approx(PAID)


def test_the_settlement_at_the_exit_instant_is_paid() -> None:
    entry_ms, exit_ms = ORIGIN_MS + 8 * HOUR_MS, ORIGIN_MS + 32 * HOUR_MS
    with_exit = series().over(entry_ms, exit_ms)
    before_exit = series().over(entry_ms, exit_ms - 1)
    assert with_exit == pytest.approx(0.0002)
    assert before_exit == pytest.approx(0.0004), "sin la de las 32:00 quedan 0.0001 + 0.0003"


def test_a_jittered_settlement_before_the_exit_still_counts() -> None:
    """07:59:59.990 de la salida es la liquidación de las 08:00 de esa salida."""
    early = series(((8, 0, 0.0), (16, -10, 0.0004), (24, 0, 0.0)))
    assert early.over(ORIGIN_MS + 8 * HOUR_MS, ORIGIN_MS + 16 * HOUR_MS) == pytest.approx(0.0004)


def test_nothing_after_the_exit_is_read() -> None:
    """Sin look-ahead: cambiar lo posterior a la salida por barbaridades no mueve el neto."""
    tampered = series(
        tuple((h, j, 1e9 if h > 32 else r) for h, j, r in SETTLEMENTS),
    )
    assert net(Action.BUY, funding=tampered) == pytest.approx(0.0968, abs=1e-12)
    assert net(Action.SELL, funding=tampered) == pytest.approx(-0.1028, abs=1e-12)


def test_nothing_before_the_entry_is_read() -> None:
    tampered = series(tuple((h, j, 1e9 if h < 8 else r) for h, j, r in SETTLEMENTS))
    assert net(Action.BUY, funding=tampered) == pytest.approx(0.0968, abs=1e-12)


def test_the_series_may_end_exactly_at_the_exit_and_the_result_does_not_change() -> None:
    """Cortar la serie justo en la salida —lo único que un sistema vivo vería— da lo mismo."""
    truncated = series(tuple(row for row in SETTLEMENTS if row[0] <= 32))
    assert net(Action.BUY, funding=truncated) == net(Action.BUY)


def test_the_bars_after_the_exit_are_not_read() -> None:
    history = candles()
    outcome = order(Action.BUY, history)
    tampered = [
        row if index <= 7 else [row[0], 1e9, 1e9, 1e-9, 1e9, 1.0]
        for index, row in enumerate(history)
    ]
    assert net_return(outcome, tampered, series(), COSTS) == net_return(
        outcome, history, series(), COSTS
    )


# ───────────────────────────── Lo que falta no es cero: no determinado ────────────────────────


def test_a_missing_settlement_inside_the_window_leaves_the_order_undetermined() -> None:
    holed = series(tuple(row for row in SETTLEMENTS if row[0] != 16))
    assert net(Action.BUY, funding=holed) is None
    assert net(Action.SELL, funding=holed) is None


@pytest.mark.parametrize(
    ("kept", "why"),
    [
        (lambda h: h >= 16, "la serie empieza después de la entrada: no hay fila anterior"),
        (lambda h: h <= 24, "la serie acaba antes de la salida: no hay fila posterior"),
        (lambda h: h in (8, 32), "un hueco de 24 h entre la entrada y la salida"),
        (lambda h: False, "una serie vacía"),
    ],
    ids=["empieza-tarde", "acaba-pronto", "hueco", "vacía"],
)
def test_a_series_that_does_not_cover_the_window_is_undetermined(
    kept: Callable[[int], bool], why: str
) -> None:
    partial = series(tuple(row for row in SETTLEMENTS if kept(row[0])))
    assert net(Action.BUY, funding=partial) is None, why


def test_a_settlement_in_the_middle_of_a_stop_candle_is_undetermined() -> None:
    """Con liquidaciones cada hora y velas de 4 h, un stop dentro de la vela no dice cuántas."""
    hourly = series(tuple((h, 0, 0.0001) for h in range(0, 48)))
    assert net(Action.BUY, touch_at=3, funding=hourly) is None
    # El mismo stop, pero con el horizonte, que sí tiene un instante de salida: determinado.
    assert net(Action.BUY, funding=hourly) is not None


def test_an_order_without_an_exit_has_no_net_to_compute() -> None:
    history = candles()[:4]
    unresolved = order(Action.BUY, history)
    assert unresolved.outcome is Outcome.UNRESOLVED
    with pytest.raises(OutcomeError, match="sin resolver"):
        net_return(unresolved, history, series(), COSTS)


def test_candles_with_a_hole_are_a_bug_not_a_missing_funding() -> None:
    history = candles()
    outcome = order(Action.BUY, history)
    gapped = [
        row if index < 5 else [row[0] + BAR_MS, *row[1:]] for index, row in enumerate(history)
    ]
    with pytest.raises(OutcomeError, match="huecos"):
        net_return(outcome, gapped, series(), COSTS)


# ───────────────────────────────── Sobre una corrida entera ───────────────────────────────────


def run_records() -> list[EvaluationRecord]:
    """Una compra resuelta, un hold, una sin futuro y una de un símbolo sin funding."""
    moment = START + timedelta(hours=4 * ENTRY_INDEX)
    buy = position_at(ENTRY_INDEX, Action.BUY, declared=STOP_LONG, moment=moment)
    hold = buy.model_copy(update={"order": None, "proposal": None})
    late = position_at(
        ENTRY_INDEX, Action.BUY, declared=STOP_LONG, moment=START + timedelta(hours=4 * 7)
    )
    elsewhere = buy.model_copy(update={"symbol": "ETH/USDT"})
    return [buy, hold, late, elsewhere]


def scored(net_inputs: NetInputs | None, scoring: Scoring = Scoring.OWN_STOP) -> ScoredRun:
    history = candles()
    return outcomes_module.score_run(
        run_records(), {"BTC/USDT": history, "ETH/USDT": history}, HORIZON, scoring, net_inputs
    )


def test_the_net_vector_is_aligned_and_never_fills_a_missing_funding_with_zero() -> None:
    inputs = NetInputs(funding={"BTC/USDT": series()}, costs=COSTS)  # ETH sin funding
    run = scored(inputs)
    assert run.per_evaluation_net is not None
    assert len(run.per_evaluation_net) == 4
    assert run.per_evaluation_net[0] == pytest.approx(0.0968, abs=1e-12)
    assert run.per_evaluation_net[1] == 0.0, "un hold no opera: no paga costes"
    assert run.per_evaluation_net[2] == 0.0, "sin futuro suficiente no se resuelve, como el bruto"
    assert run.per_evaluation_net[3] is None, "ETH no tiene funding: no determinado, no cero"
    assert run.per_position_net is not None
    assert len(run.per_position_net) == 2
    assert run.per_position_net[0] == pytest.approx(0.0968, abs=1e-12)
    assert run.per_position_net[1] is None
    assert run.net_undetermined == 1


def test_without_net_inputs_there_is_no_net() -> None:
    run = scored(None)
    assert run.per_evaluation_net is None
    assert run.per_position_net is None
    assert run.net_undetermined == 0


@pytest.mark.parametrize("scoring", list(Scoring))
def test_the_gross_figures_are_the_same_with_or_without_costs(scoring: Scoring) -> None:
    """Los criterios leen el bruto: ni los costes ni el funding pueden moverlo."""
    without = scored(None, scoring)
    cheap = scored(
        NetInputs(funding={"BTC/USDT": series()}, costs=CostModel(taker_fee=0.0, slippage=0.0)),
        scoring,
    )
    dear = scored(
        NetInputs(funding={"BTC/USDT": series()}, costs=CostModel(taker_fee=0.01, slippage=0.02)),
        scoring,
    )
    for other in (cheap, dear):
        assert other.per_evaluation == without.per_evaluation
        assert other.per_position == without.per_position
        assert (other.positions, other.resolved, other.unscorable) == (
            without.positions,
            without.resolved,
            without.unscorable,
        )


@pytest.mark.parametrize("scoring", list(Scoring))
def test_the_net_is_computed_for_each_of_the_three_scorings(scoring: Scoring) -> None:
    """Entrada, salida y lado salen de la propia posición, la puntúe quien la puntúe."""
    history = candles()
    record = run_records()[0]
    position = score_position(record, history, HORIZON, scoring)
    assert position is not None
    run = scored(NetInputs(funding={"BTC/USDT": series()}, costs=COSTS), scoring)
    assert run.per_position_net is not None
    expected = net_return(position, history, series(), COSTS)
    assert run.per_position_net[0] == expected


# ───────────────────────────────────────── Mutaciones ─────────────────────────────────────────


def mutated(
    function: Callable[..., Any], old: str, new: str, module: ModuleType
) -> Callable[..., Any]:
    """La función con un fragmento de su código cambiado, compilada en el espacio de `module`."""
    source = textwrap.dedent(inspect.getsource(function))
    assert source.count(old) == 1, f"el fragmento {old!r} ya no está una sola vez en el código"
    namespace: dict[str, Any] = dict(vars(module))
    exec(compile(source.replace(old, new), "<mutante>", "exec"), namespace)
    return namespace[function.__name__]  # type: ignore[no-any-return]


def test_the_hand_computed_figures_hold_for_the_real_code() -> None:
    """El control de las mutaciones: sin error puesto, la comprobación pasa."""
    assert_hand_computed_figures()


def test_mutation_the_short_sign_not_inverted_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    mutant = mutated(
        net_return,
        "direction = 1.0 if outcome.side is Action.BUY else -1.0",
        "direction = 1.0",
        outcomes_module,
    )
    monkeypatch.setattr(outcomes_module, "net_return", mutant)
    with pytest.raises(AssertionError):
        assert_hand_computed_figures()


def test_mutation_the_settlement_at_the_entry_instant_included_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mutant = mutated(
        FundingSeries.over,
        "first = bisect_right(times, entry_ms)",
        "first = bisect_left(times, entry_ms)",
        funding_module,
    )
    monkeypatch.setattr(FundingSeries, "over", mutant)
    with pytest.raises(AssertionError):
        assert_hand_computed_figures()


def test_mutation_the_missing_funding_assumed_zero_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mutant = mutated(
        FundingSeries.over,
        "if any(interval_hours(times[k], times[k + 1]) is None for k in range(lower, upper)):\n"
        "        return None",
        "if any(interval_hours(times[k], times[k + 1]) is None for k in range(lower, upper)):\n"
        "        return 0.0",
        funding_module,
    )
    monkeypatch.setattr(FundingSeries, "over", mutant)
    with pytest.raises(AssertionError):
        assert_hand_computed_figures()


def assert_a_symbol_without_funding_is_undetermined() -> None:
    inputs = NetInputs(funding={"BTC/USDT": series()}, costs=COSTS)
    result = scored(inputs)
    assert result.per_evaluation_net is not None
    assert result.per_evaluation_net[3] is None


def test_mutation_the_missing_funding_in_the_run_assumed_zero_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """También en la corrida: un símbolo sin serie dejaría de ser `None` si se rellenara con 0."""
    assert_a_symbol_without_funding_is_undetermined()  # control: el código real lo cumple
    mutant = mutated(
        score_run,
        "None\n                if series is None or rows is None\n"
        "                else net_return(outcome, rows, series, net.costs)",
        "0.0\n                if series is None or rows is None\n"
        "                else net_return(outcome, rows, series, net.costs)",
        outcomes_module,
    )
    monkeypatch.setattr(outcomes_module, "score_run", mutant)
    with pytest.raises(AssertionError):
        assert_a_symbol_without_funding_is_undetermined()


def test_mutation_the_stop_exit_at_the_candle_close_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Salir al cierre de la vela del stop cobraría una liquidación que no llegó a pagarse."""
    mutant = mutated(
        net_return,
        "paid = funding.over(entry_ms, exit_ms, exit_before_ms=exit_ms + bar_ms)",
        "paid = funding.over(entry_ms, exit_ms + bar_ms)",
        outcomes_module,
    )
    monkeypatch.setattr(outcomes_module, "net_return", mutant)
    with pytest.raises(AssertionError):
        assert_hand_computed_figures()
