"""Pruebas del puntuador de resultados.

Las velas se escriben a mano: para comprobar que una invalidación se detecta hay
que poder poner el mínimo exactamente donde la prueba quiere.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from crypto_agents.journal import EvaluationRecord
from crypto_agents.outcomes import Outcome, score_outcomes, score_record
from crypto_agents.state import Action, ExecutionMode, MarketSnapshot, OrderIntent

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
