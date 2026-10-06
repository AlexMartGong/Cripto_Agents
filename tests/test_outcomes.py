"""Pruebas del puntuador de resultados.

Las velas se escriben a mano: para comprobar que una invalidación se detecta hay
que poder poner el mínimo exactamente donde la prueba quiere.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from crypto_agents.journal import EvaluationRecord
from crypto_agents.outcomes import (
    Outcome,
    OutcomeError,
    Scoring,
    UnscorableError,
    resolved_returns,
    score_outcomes,
    score_position,
    score_record,
    score_run,
)
from crypto_agents.state import (
    Action,
    ExecutionMode,
    IndicatorSet,
    MarketSnapshot,
    OrderIntent,
    Proposal,
)
from crypto_agents.stops import common_stop
from tests.conftest import real_rows

START = datetime(2026, 8, 1, tzinfo=UTC)
STEP_MS = 3_600_000


def rows(bars: list[tuple[float, float, float]]) -> list[list[float]]:
    """Velas `[ts, open, high, low, close, volume]` desde `(high, low, close)`."""
    return [
        [float(START.timestamp() * 1000 + index * STEP_MS), close, high, low, close, 1000.0]
        for index, (high, low, close) in enumerate(bars)
    ]


def record_at(index: int, side: Action, entry: float, invalidation: float) -> EvaluationRecord:
    """Registro con una orden emitida en la vela `index`."""
    moment = datetime.fromtimestamp(START.timestamp() + index * 3600, tz=UTC)
    return EvaluationRecord(
        run_id=uuid4(),
        at=moment,
        symbol="BTC/USDT",
        timeframe="1h",
        snapshot=MarketSnapshot(
            run_id=uuid4(),
            exchange="binance",
            symbol="BTC/USDT",
            timeframe="1h",
            timestamp=moment,
            close=entry,
            candles_digest="a" * 64,
            candles_count=100,
        ),
        order=OrderIntent(
            symbol="BTC/USDT",
            side=side,
            size_fraction=0.05,
            reference_price=entry,
            invalidation_price=invalidation,
            mode=ExecutionMode.PAPER,
        ),
    )


def test_a_long_stopped_out_is_scored_at_its_invalidation() -> None:
    """El mínimo de la vela toca el nivel: la tesis se rompió, aunque cerrara arriba."""
    history = rows([(100, 100, 100), (101, 94, 101), (102, 101, 102), (103, 102, 103)])
    outcome = score_record(record_at(0, Action.BUY, 100.0, 95.0), history, horizon=3)

    assert outcome is not None
    assert outcome.outcome is Outcome.INVALIDATED
    assert outcome.exit_price == 95.0
    assert outcome.gross_return < 0.0
    assert outcome.bars_held == 1


def test_invalidation_is_checked_against_the_range_not_the_close() -> None:
    """Un stop se toca intradía; puntuar por cierres regalaría operaciones perdidas."""
    history = rows([(100, 100, 100), (101, 90, 101), (102, 101, 102)])
    outcome = score_record(record_at(0, Action.BUY, 100.0, 95.0), history, horizon=2)

    assert outcome is not None
    assert outcome.outcome is Outcome.INVALIDATED, "cerró en 101 pero pasó por 90"


def test_a_long_that_survives_is_scored_at_the_horizon_close() -> None:
    """Sin tocar la invalidación, el resultado es el cierre del horizonte."""
    history = rows([(100, 100, 100), (106, 99, 105), (112, 104, 110), (115, 109, 114)])
    outcome = score_record(record_at(0, Action.BUY, 100.0, 95.0), history, horizon=3)

    assert outcome is not None
    assert outcome.outcome is Outcome.HELD
    assert outcome.exit_price == 114.0
    assert outcome.gross_return == 0.14


def test_a_short_inverts_the_sign_of_the_return() -> None:
    """Vender y que el precio baje es ganar."""
    history = rows([(100, 100, 100), (99, 94, 95), (96, 89, 90), (91, 84, 85)])
    outcome = score_record(record_at(0, Action.SELL, 100.0, 110.0), history, horizon=3)

    assert outcome is not None
    assert outcome.outcome is Outcome.HELD
    assert outcome.gross_return == 0.15


def test_a_short_is_invalidated_by_the_high() -> None:
    """En corto, el nivel se toca por arriba."""
    history = rows([(100, 100, 100), (112, 99, 101), (102, 100, 101)])
    outcome = score_record(record_at(0, Action.SELL, 100.0, 110.0), history, horizon=2)

    assert outcome is not None
    assert outcome.outcome is Outcome.INVALIDATED


def test_an_order_without_enough_future_is_unresolved() -> None:
    """El histórico se acabó antes del horizonte: no cuenta, ni a favor ni en contra."""
    history = rows([(100, 100, 100), (101, 99, 101)])
    outcome = score_record(record_at(0, Action.BUY, 100.0, 95.0), history, horizon=5)

    assert outcome is not None
    assert outcome.outcome is Outcome.UNRESOLVED


def test_an_evaluation_without_an_order_is_not_scored() -> None:
    """Una evaluación que no operó no tiene resultado que medir."""
    record = EvaluationRecord(run_id=uuid4(), at=START, symbol="BTC/USDT", timeframe="1h")
    assert score_record(record, rows([(100, 100, 100)]), horizon=3) is None


def test_stats_keep_the_denominator_next_to_the_rate() -> None:
    """Una tasa de acierto sobre dos operaciones no es una tasa de acierto."""
    history = rows([(100, 100, 100), (110, 99, 108), (112, 107, 111), (113, 110, 112)])
    stats = score_outcomes(
        [record_at(0, Action.BUY, 100.0, 95.0), record_at(0, Action.SELL, 100.0, 115.0)],
        {"BTC/USDT": history},
        horizon=3,
    )

    assert stats.orders == 2
    assert stats.resolved == 2
    assert stats.wins == 1
    assert stats.win_rate == 0.5


def test_an_order_is_scored_against_its_own_symbol() -> None:
    """Con siete series en juego, puntuar contra la serie equivocada sería ruido.

    La selección de la ablación salta entre símbolos, así que las series entran
    por símbolo: una orden cuyo histórico no está declarado no se puntúa, en vez
    de puntuarse contra el primero que hubiera a mano.
    """
    history = rows([(100, 100, 100), (110, 99, 108), (112, 107, 111), (113, 110, 112)])
    order = record_at(0, Action.BUY, 100.0, 95.0)

    assert score_outcomes([order], {"BTC/USDT": history}, horizon=3).orders == 1
    assert score_outcomes([order], {"ETH/USDT": history}, horizon=3).orders == 0


def test_no_orders_reports_no_rate_instead_of_zero() -> None:
    """Sin operaciones no hay tasa: cero sería una afirmación que nadie midió."""
    stats = score_outcomes([], {"BTC/USDT": rows([(100, 100, 100)])}, horizon=3)
    assert stats.win_rate is None
    assert stats.mean_return is None


# ──────────────────────────────── Órdenes que no deberían existir ─────────────────────────────────


@pytest.mark.parametrize(
    ("side", "invalidation"),
    [(Action.BUY, 105.0), (Action.SELL, 95.0), (Action.BUY, 100.0)],
    ids=["buy por encima", "sell por debajo", "en la entrada"],
)
def test_an_order_with_its_stop_on_the_wrong_side_is_not_scored(
    side: Action, invalidation: float
) -> None:
    """Un largo «invalidado» por encima de su entrada saldría con ganancia en la vela siguiente.

    La primera ablación puntuó órdenes así. Hoy el gate las veta y `OrderIntent`
    no se deja construir con ese stop, así que la orden se fabrica aquí saltándose
    la validación: si alguna llega a puntuarse es un bug, y la respuesta es
    decirlo, no devolver un resultado que parece una medida.
    """
    good = record_at(0, side, entry=100.0, invalidation=95.0 if side is Action.BUY else 105.0)
    assert good.order is not None
    broken = good.model_copy(
        update={
            "order": OrderIntent.model_construct(
                **(good.order.model_dump() | {"invalidation_price": invalidation})
            )
        }
    )
    history = rows([(101.0, 99.0, 100.0), (106.0, 94.0, 104.0), (107.0, 103.0, 106.0)])

    with pytest.raises(OutcomeError, match="lado equivocado"):
        score_record(broken, history, horizon=2)
    with pytest.raises(OutcomeError):
        score_outcomes([broken], {"BTC/USDT": history}, horizon=2)
    assert score_record(good, history, horizon=2) is not None


# ───────────────────────────────── Retornos individuales ──────────────────────────────────────────
# `OutcomeStats` guarda el total y no los retornos, así que no puede dar una dispersión.
# `resolved_returns` los entrega uno a uno y repite el filtro de `score_outcomes` sin tocarlo:
# lo que ata las dos copias es esta prueba.


def mixed_orders() -> tuple[list[EvaluationRecord], list[list[float]]]:
    """Cuatro órdenes: largo que aguanta, corto que aguanta, largo invalidado y uno sin futuro.

    A horizonte 3: +14%, +15% y -5% resueltas; la cuarta empieza en la última vela y
    no tiene horizonte, así que queda sin resolver.
    """
    long_run = rows([(100, 100, 100), (106, 99, 105), (112, 104, 110), (115, 109, 114)])
    short_run = rows([(100, 100, 100), (99, 94, 95), (96, 89, 90), (91, 84, 85)])
    stopped = rows([(100, 100, 100), (101, 94, 101), (102, 101, 102), (103, 102, 103)])
    # Un símbolo, una sola serie: los tres recorridos se encadenan desplazando sus
    # timestamps, y cada orden se siembra en el índice donde empieza el suyo.
    history = [
        *long_run,
        *[[row[0] + 4 * STEP_MS, *row[1:]] for row in short_run],
        *[[row[0] + 8 * STEP_MS, *row[1:]] for row in stopped],
    ]
    records = [
        record_at(0, Action.BUY, 100.0, 95.0),
        record_at(4, Action.SELL, 100.0, 110.0),
        record_at(8, Action.BUY, 100.0, 95.0),
        record_at(11, Action.BUY, 103.0, 95.0),
    ]
    return records, history


def test_resolved_returns_agree_with_the_scored_stats() -> None:
    """Mismo conjunto, mismo total, mismos aciertos: dos caminos a una misma cifra.

    Si una de las dos copias del filtro cambia sin la otra, el retorno medio de la
    tabla dejaría de ser el que `score_outcomes` publica, con ambos pareciendo
    correctos. La orden sin horizonte no entra en ninguna.
    """
    records, history = mixed_orders()

    stats = score_outcomes(records, {"BTC/USDT": history}, horizon=3)
    returns = resolved_returns(records, {"BTC/USDT": history}, horizon=3)

    assert stats.orders == 4
    assert stats.unresolved == 1
    assert len(returns) == stats.resolved == 3
    assert sum(returns) == pytest.approx(stats.total_return)
    assert sum(1 for value in returns if value > 0.0) == stats.wins
    assert sorted(returns) == pytest.approx([-0.05, 0.14, 0.15])


def test_resolved_returns_ignore_orders_whose_symbol_has_no_history() -> None:
    """Igual que `score_outcomes`: sin su serie, la orden no se puntúa contra la de otro."""
    records, history = mixed_orders()

    assert resolved_returns(records, {"ETH/USDT": history}, horizon=3) == ()
    assert resolved_returns([], {"BTC/USDT": history}, horizon=3) == ()


# ───────────────────────────── Puntuación común: stop común y cierre ──────────────────────────────
# Tres formas de salir de la misma posición: con el stop que declaró el decisor (la de siempre),
# con uno común a todos los brazos y sin stop. Las dos últimas puntúan la propuesta, no la orden:
# un stop declarado en el lado equivocado vetaba la orden y borraba la dirección de la medida.


def position_at(
    index: int,
    side: Action,
    entry: float = 100.0,
    atr: float = 2.0,
    declared: float | None = None,
    moment: datetime | None = None,
) -> EvaluationRecord:
    """Registro con propuesta, orden e indicadores: lo que cualquier brazo deja en el journal."""
    stop = declared if declared is not None else entry * (0.95 if side is Action.BUY else 1.05)
    base = record_at(index, side, entry, stop)
    assert base.snapshot is not None
    snapshot = base.snapshot
    if moment is not None:
        snapshot = snapshot.model_copy(update={"timestamp": moment})
    return base.model_copy(
        update={
            "snapshot": snapshot,
            "proposal": Proposal(
                action=side,
                confidence=0.5,
                size_fraction=0.1,
                invalidation_price=stop,
                rationale="Posición de prueba para la puntuación común.",
            ),
            "indicators": IndicatorSet(values={"ATRr_14": atr}),
        }
    )


def vetoed(record: EvaluationRecord, stop: float) -> EvaluationRecord:
    """La misma propuesta con el stop declarado en el lado equivocado: sin orden."""
    assert record.proposal is not None
    return record.model_copy(
        update={
            "order": None,
            "proposal": record.proposal.model_copy(update={"invalidation_price": stop}),
        }
    )


@pytest.mark.parametrize(
    ("side", "bars", "stop"),
    [
        (Action.BUY, [(100, 100, 100), (101, 95, 101), (102, 101, 102), (103, 102, 103)], 96.0),
        (Action.SELL, [(100, 100, 100), (105, 99, 99), (99, 97, 98), (98, 96, 97)], 104.0),
    ],
    ids=["compra: el mínimo toca, el cierre no", "venta: el máximo toca, el cierre no"],
)
def test_the_common_stop_is_checked_against_the_range_not_the_close(
    side: Action, bars: list[tuple[float, float, float]], stop: float
) -> None:
    """Entrada 100 y ATR 2: stop en 96 o 104. La vela 1 lo cruza intradía y cierra del otro lado.

    Puntuar por cierres daría a la línea base un crédito que el mercado no le habría
    dado: el stop se toca dentro de la vela.
    """
    outcome = score_position(position_at(0, side), rows(bars), 3, Scoring.COMMON_STOP)

    assert outcome is not None
    assert outcome.outcome is Outcome.INVALIDATED
    assert outcome.exit_price == stop
    assert outcome.bars_held == 1


def test_the_common_stop_ignores_the_one_the_decider_declared() -> None:
    """Un stop declarado en 99.9 se toca en la vela 1; el común, en 96, no.

    Con el propio sale invalidado con -0.1%; con el común aguanta hasta el cierre del
    horizonte. La orden y la propuesta conservan el stop que declaró el decisor.
    """
    history = rows([(100, 100, 100), (101, 99, 101), (102, 100, 102), (103, 101, 103)])
    record = position_at(0, Action.BUY, declared=99.9)

    own = score_position(record, history, 3, Scoring.OWN_STOP)
    common = score_position(record, history, 3, Scoring.COMMON_STOP)

    assert own == score_record(record, history, 3)
    assert own is not None
    assert own.outcome is Outcome.INVALIDATED
    assert own.exit_price == 99.9
    assert common is not None
    assert common.outcome is Outcome.HELD
    assert common.exit_price == 103.0
    assert record.order is not None
    assert record.order.invalidation_price == 99.9
    assert record.proposal is not None
    assert record.proposal.invalidation_price == 99.9


@pytest.mark.parametrize(
    ("side", "expected"), [(Action.BUY, 0.03), (Action.SELL, -0.03)], ids=["compra", "venta"]
)
def test_scoring_at_the_horizon_close_has_no_stop_at_all(side: Action, expected: float) -> None:
    """La vela 1 cae a 50 y recupera: sin stop, la salida es el cierre de la vela 3, 103."""
    history = rows([(100, 100, 100), (101, 50, 101), (102, 100, 102), (103, 101, 103)])

    outcome = score_position(position_at(0, side), history, 3, Scoring.HORIZON_CLOSE)

    assert outcome is not None
    assert outcome.outcome is Outcome.HELD
    assert outcome.exit_price == 103.0
    assert outcome.gross_return == pytest.approx(expected)


@pytest.mark.parametrize("scoring", [Scoring.COMMON_STOP, Scoring.HORIZON_CLOSE])
def test_without_enough_future_the_position_is_unresolved(scoring: Scoring) -> None:
    """Dos velas y horizonte 3: no cuenta, ni a favor ni en contra."""
    history = rows([(100, 100, 100), (101, 99, 100)])

    outcome = score_position(position_at(0, Action.BUY), history, 3, scoring)

    assert outcome is not None
    assert outcome.outcome is Outcome.UNRESOLVED


def test_a_proposal_vetoed_for_its_stop_is_still_scored_by_direction() -> None:
    """Compra con el stop declarado por encima de la entrada: sin orden, pero con dirección.

    El gate la veta con razón, y por eso `OWN_STOP` no la puntúa. Para las otras dos
    el stop declarado es irrelevante, y dejarla fuera haría que el stop del decisor
    filtrara qué direcciones se miden, que es lo que esta puntuación existe para evitar.
    """
    history = rows([(100, 100, 100), (101, 99, 101), (102, 100, 102), (103, 101, 103)])
    record = vetoed(position_at(0, Action.BUY), stop=105.0)

    assert score_position(record, history, 3, Scoring.OWN_STOP) is None
    common = score_position(record, history, 3, Scoring.COMMON_STOP)
    closing = score_position(record, history, 3, Scoring.HORIZON_CLOSE)
    assert common is not None
    assert closing is not None
    assert common.gross_return == pytest.approx(0.03)
    assert closing.gross_return == pytest.approx(0.03)


@pytest.mark.parametrize("scoring", list(Scoring))
def test_a_hold_has_no_position_under_any_scoring(scoring: Scoring) -> None:
    """No operar no tiene resultado que medir, se puntúe como se puntúe."""
    history = rows([(100, 100, 100), (101, 99, 101), (102, 100, 102), (103, 101, 103)])
    hold = position_at(0, Action.BUY).model_copy(
        update={
            "order": None,
            "proposal": Proposal(
                action=Action.HOLD,
                confidence=0.5,
                size_fraction=0.0,
                invalidation_price=None,
                rationale="Sin lectura que justifique abrir una posición.",
            ),
        }
    )

    assert score_position(hold, history, 3, scoring) is None


def test_the_common_stop_needs_the_indicators_the_record_carries() -> None:
    """Un journal sin indicadores no puede reconstruir el stop común: se dice, no se inventa.

    La puntuación por cierre no los necesita, y la del stop propio tampoco.
    """
    history = rows([(100, 100, 100), (101, 99, 101), (102, 100, 102), (103, 101, 103)])
    bare = position_at(0, Action.BUY).model_copy(update={"indicators": None})

    with pytest.raises(UnscorableError, match="indicadores"):
        score_position(bare, history, 3, Scoring.COMMON_STOP)
    assert score_position(bare, history, 3, Scoring.HORIZON_CLOSE) is not None
    assert score_position(bare, history, 3, Scoring.OWN_STOP) is not None


@pytest.mark.parametrize(
    ("side", "bars"),
    [
        (Action.BUY, [(100, 100, 100), (101, 98, 100), (100, 95, 97), (99, 97, 98)]),
        (Action.BUY, [(100, 100, 100), (103, 99, 102), (106, 101, 105), (108, 104, 107)]),
        (Action.SELL, [(100, 100, 100), (103, 99, 101), (105, 100, 104), (104, 101, 103)]),
        (Action.SELL, [(100, 100, 100), (101, 97, 98), (99, 94, 95), (97, 92, 93)]),
        (Action.BUY, [(100, 100, 100), (101, 99, 100)]),
    ],
    ids=["compra invalidada", "compra aguanta", "venta invalidada", "venta aguanta", "sin futuro"],
)
def test_declaring_the_common_stop_scores_the_same_under_both_stops(
    side: Action, bars: list[tuple[float, float, float]]
) -> None:
    """Si el decisor declara exactamente el stop común, las dos puntuaciones son idénticas.

    Es lo que ata la copia del recorrido al original: `score_record` es el código de
    siempre y no se toca, así que cualquier deriva de la copia aparece aquí. También es
    la propiedad de las líneas base, que declaran ese stop.
    """
    record = position_at(0, side, declared=common_stop(side, 100.0, 2.0))
    history = rows(bars)

    own = score_position(record, history, 3, Scoring.OWN_STOP)
    common = score_position(record, history, 3, Scoring.COMMON_STOP)

    assert own is not None
    assert own == common


def test_a_run_scored_with_the_declared_stop_is_the_scoring_that_already_existed() -> None:
    """`OWN_STOP` sobre una corrida da el mismo recuento, los mismos retornos y el mismo total."""
    records, history = mixed_orders()
    histories = {"BTC/USDT": history}

    stats = score_outcomes(records, histories, horizon=3)
    run = score_run(records, histories, 3, Scoring.OWN_STOP)

    assert run.scoring is Scoring.OWN_STOP
    assert run.positions == stats.orders == 4
    assert run.resolved == stats.resolved == 3
    assert run.unscorable == 0
    assert run.per_position == resolved_returns(records, histories, horizon=3)
    assert sum(run.per_position) == pytest.approx(stats.total_return)
    assert run.per_evaluation == (*run.per_position, 0.0), "la cuarta orden no se resolvió"


def test_the_return_per_evaluation_is_zero_where_there_is_no_position() -> None:
    """Un hold, un símbolo sin serie y una compra: el vector es (0, 0, r), alineado."""
    history = rows([(100, 100, 100), (101, 99, 101), (102, 100, 102), (103, 101, 103)])
    hold = position_at(0, Action.BUY).model_copy(update={"order": None, "proposal": None})
    elsewhere = position_at(0, Action.BUY).model_copy(update={"symbol": "ETH/USDT"})
    buy = position_at(0, Action.BUY)

    run = score_run([hold, elsewhere, buy], {"BTC/USDT": history}, 3, Scoring.HORIZON_CLOSE)

    assert run.per_evaluation == pytest.approx((0.0, 0.0, 0.03))
    assert run.per_position == pytest.approx((0.03,))
    assert run.positions == run.resolved == 1


def test_a_record_that_cannot_be_scored_is_counted_and_does_not_stop_the_run() -> None:
    """Dos compras con stop común, una sin indicadores: una puntuada y otra sin puntuar."""
    history = rows([(100, 100, 100), (101, 99, 101), (102, 100, 102), (103, 101, 103)])
    bare = position_at(0, Action.BUY).model_copy(update={"indicators": None})

    run = score_run(
        [position_at(0, Action.BUY), bare], {"BTC/USDT": history}, 3, Scoring.COMMON_STOP
    )

    assert run.unscorable == 1
    assert run.positions == run.resolved == 1
    assert run.per_evaluation == pytest.approx((0.03, 0.0))


# ─────────────────────────────────── Sin mirar hacia adelante ─────────────────────────────────────


def tampered_future(history: list[list[float]], after: int) -> list[list[float]]:
    """La misma serie con las velas posteriores a `after` reventadas: mínimo 1 y máximo 1e9."""
    return [
        row if index <= after else [row[0], row[4] * 2.0, 1e9, 1.0, row[4] * 2.0, row[5]]
        for index, row in enumerate(history)
    ]


@pytest.mark.parametrize("scoring", list(Scoring))
@pytest.mark.parametrize("side", [Action.BUY, Action.SELL])
@pytest.mark.parametrize("cut", [150, 300, 420])
def test_no_scoring_reads_candles_past_its_horizon(
    scoring: Scoring, side: Action, cut: int
) -> None:
    """El resultado de una posición en la vela `i` no cambia si se cortan o se rompen las velas
    posteriores a `i + horizonte`.

    Mismo método que `test_lookahead`, aplicado a la puntuación: truncar por la derecha
    justo después del horizonte, y además sustituir lo que viene detrás por velas
    imposibles. Una puntuación que mirase una vela de más vería el mínimo en 1 o el
    máximo en 1e9 y cambiaría de resultado.
    """
    history = real_rows()
    horizon = 6
    close = history[cut][4]
    moment = datetime.fromtimestamp(history[cut][0] / 1000.0, tz=UTC)
    record = position_at(cut, side, entry=close, atr=close * 0.01, moment=moment)

    full = score_position(record, history, horizon, scoring)
    truncated = score_position(record, history[: cut + horizon + 1], horizon, scoring)
    broken = score_position(record, tampered_future(history, cut + horizon), horizon, scoring)

    assert full is not None
    assert full.outcome is not Outcome.UNRESOLVED
    assert full == truncated == broken
