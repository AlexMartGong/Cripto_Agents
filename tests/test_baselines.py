"""Pruebas de las líneas base de la ablación.

Ninguna llama a un modelo ni lee más que los indicadores y el cierre de la vela
evaluada. Lo que estas pruebas vigilan es que sean lo que dicen ser: constantes
congeladas antes de ver resultados, una regla de tendencia sin parámetros ocultos,
un azar reproducible, y ninguna mirada hacia adelante.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from uuid import UUID, uuid5

import pytest

from crypto_agents.baselines import (
    BASELINE_CONFIDENCE,
    BASELINE_SIZE,
    TREND_ADX_MIN,
    always_buy_proposal,
    always_sell_proposal,
    baseline_seed,
    random_action,
    random_proposal,
    trend_action,
    trend_proposal,
)
from crypto_agents.indicators import DEFAULT_PRESET, IndicatorPreset, enrich, to_indicator_set
from crypto_agents.risk import RiskLimits
from crypto_agents.state import Action, IndicatorSet, MarketSnapshot, stop_on_wrong_side
from crypto_agents.stops import atr_of, common_stop
from tests.conftest import PRESET, real_candles
from tests.test_lookahead import CUTS

NAMESPACE = UUID("00000000-0000-0000-0000-00000000b45e")


def run_id(index: int) -> UUID:
    """Identificador reproducible de una evaluación ficticia."""
    return uuid5(NAMESPACE, str(index))


def snapshot(close: float = 100.0) -> MarketSnapshot:
    """Momento evaluado con el cierre que la prueba necesita."""
    from datetime import UTC, datetime

    return MarketSnapshot(
        run_id=run_id(0),
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="4h",
        timestamp=datetime(2026, 8, 1, tzinfo=UTC),
        close=close,
        candles_digest="a" * 64,
        candles_count=500,
    )


def indicators(
    fast: float = 108.0,
    slow: float = 105.0,
    trend: float = 100.0,
    adx: float = 30.0,
    atr: float = 2.0,
) -> IndicatorSet:
    """Los cinco valores que las líneas base pueden leer, con los nombres del preset corto."""
    return IndicatorSet(
        values={
            f"EMA_{PRESET.ema_fast}": fast,
            f"EMA_{PRESET.ema_slow}": slow,
            f"EMA_{PRESET.ema_trend}": trend,
            f"ADX_{PRESET.adx}": adx,
            f"ATRr_{PRESET.atr}": atr,
        }
    )


# ── Lo que se congeló ────────────────────────────────────────────────────────


def test_every_parameter_was_frozen_before_any_result_existed() -> None:
    """Cuatro números y ninguno se buscó: convenciones, no resultados.

    25 es el umbral de Wilder y el mismo que ya usa el gate de activación. Si una
    edición obliga a tocar esta prueba, la edición tiene que defenderse como una
    decisión, no como un ajuste.
    """
    assert TREND_ADX_MIN == 25.0
    assert BASELINE_SIZE == 0.05
    assert BASELINE_CONFIDENCE == 0.5


def test_the_baseline_size_never_meets_the_position_cap() -> None:
    """El gate no recorta a las líneas base: su tamaño cabe bajo el tope por posición."""
    assert RiskLimits().max_position_fraction >= BASELINE_SIZE


# ── La regla de tendencia ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("close", "fast", "slow", "trend", "adx", "expected"),
    [
        (110.0, 108.0, 105.0, 100.0, 30.0, Action.BUY),
        (90.0, 92.0, 95.0, 100.0, 30.0, Action.SELL),
        (110.0, 108.0, 105.0, 100.0, 25.0, Action.BUY),
        (90.0, 92.0, 95.0, 100.0, 25.0, Action.SELL),
        (110.0, 108.0, 105.0, 100.0, 24.99, Action.HOLD),
        (90.0, 92.0, 95.0, 100.0, 24.99, Action.HOLD),
        (110.0, 104.0, 105.0, 100.0, 30.0, Action.HOLD),
        (107.0, 108.0, 105.0, 100.0, 30.0, Action.HOLD),
        (110.0, 105.0, 105.0, 100.0, 30.0, Action.HOLD),
        (108.0, 108.0, 105.0, 100.0, 30.0, Action.HOLD),
        (110.0, 105.0, 108.0, 100.0, 30.0, Action.HOLD),
        (110.0, 108.0, 105.0, 105.0, 30.0, Action.HOLD),
    ],
    ids=[
        "pila alcista",
        "pila bajista",
        "adx justo en el umbral, compra",
        "adx justo en el umbral, venta",
        "adx bajo, alcista",
        "adx bajo, bajista",
        "rápida bajo la lenta",
        "cierre bajo la rápida",
        "rápida igual a la lenta",
        "cierre igual a la rápida",
        "medias cruzadas",
        "lenta igual a la de fondo",
    ],
)
def test_the_trend_rule_needs_a_strict_stack_and_a_trending_market(
    close: float, fast: float, slow: float, trend: float, adx: float, expected: Action
) -> None:
    """`close > rápida > lenta > fondo` con ADX ≥ 25 compra; el espejo vende; todo lo demás, hold.

    Las desigualdades son estrictas y el ADX no: un empate no es una pila, y el
    umbral de Wilder incluye el 25.
    """
    values = indicators(fast=fast, slow=slow, trend=trend, adx=adx)

    assert trend_action(values, close, PRESET) is expected


@pytest.mark.parametrize("preset", [PRESET, DEFAULT_PRESET], ids=["corto", "producción"])
def test_the_trend_rule_reads_only_columns_the_preset_produces(preset: IndicatorPreset) -> None:
    """Ninguna columna inventada: con exactamente las del preset, la regla no se rompe."""
    values = IndicatorSet(
        values={name: 100.0 + index for index, name in enumerate(preset.column_names())}
    )

    assert trend_action(values, 100.0, preset) in set(Action)


# ── El azar ──────────────────────────────────────────────────────────────────


def test_the_seed_derivation_is_pinned_and_independent_of_the_process() -> None:
    """sha-256 de `semilla|run_id`, no `hash()`: dos procesos darían dos azares distintos."""
    seed, ident = 20260815, run_id(7)
    expected = int.from_bytes(hashlib.sha256(f"{seed}|{ident}".encode()).digest()[:8], "big")

    assert baseline_seed(seed, ident) == expected


def test_the_same_seed_and_run_give_the_same_draw() -> None:
    """Reproducible: dos corridas del mismo plan toman las mismas decisiones al azar."""
    draws = [random_action(20260815, run_id(index)) for index in range(50)]

    assert draws == [random_action(20260815, run_id(index)) for index in range(50)]


def test_the_draw_depends_on_the_run_id() -> None:
    """Ignorar el `run_id` daría la misma acción en las 140 evaluaciones."""
    draws = {random_action(20260815, run_id(index)) for index in range(60)}

    assert draws == {Action.BUY, Action.SELL, Action.HOLD}


def test_the_draw_depends_on_the_seed() -> None:
    """Otra semilla del manifiesto es otro azar sobre las mismas evaluaciones."""
    first = [random_action(1, run_id(index)) for index in range(60)]
    second = [random_action(2, run_id(index)) for index in range(60)]

    assert first != second


def test_the_draw_is_uniform_over_the_three_actions() -> None:
    """Tres mil evaluaciones: cada acción cae entre 29% y 38%. Es sha-256, así que es estable."""
    counts = Counter(random_action(20260815, run_id(index)) for index in range(3000))

    for action in Action:
        assert 0.29 < counts[action] / 3000 < 0.38, counts


# ── Las propuestas ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("build", "side"),
    [(always_buy_proposal, Action.BUY), (always_sell_proposal, Action.SELL)],
    ids=["always_buy", "always_sell"],
)
def test_the_fixed_lines_declare_the_common_stop(build: object, side: Action) -> None:
    """Compra o vende siempre, con el stop de `common_stop` y el tamaño fijo."""
    values, shot = indicators(atr=2.0), snapshot(100.0)

    proposal = build(shot, values)  # type: ignore[operator]

    assert proposal.action is side
    assert proposal.invalidation_price == common_stop(side, 100.0, atr_of(values))
    assert proposal.size_fraction == BASELINE_SIZE
    assert proposal.confidence == BASELINE_CONFIDENCE
    assert not stop_on_wrong_side(side, proposal.invalidation_price, 100.0)


def test_a_random_hold_opens_nothing() -> None:
    """Un hold lleva tamaño cero y ningún stop: no hay posición que invalidar."""
    ident = next(run_id(i) for i in range(100) if random_action(1, run_id(i)) is Action.HOLD)

    proposal = random_proposal(snapshot(), indicators(), 1, ident)

    assert proposal.action is Action.HOLD
    assert proposal.size_fraction == 0.0
    assert proposal.invalidation_price is None


def test_the_random_line_acts_with_the_common_stop_too() -> None:
    """Una compra o venta al azar usa el mismo stop que las demás líneas base."""
    ident = next(run_id(i) for i in range(100) if random_action(1, run_id(i)) is Action.SELL)

    proposal = random_proposal(snapshot(100.0), indicators(atr=2.0), 1, ident)

    assert proposal.action is Action.SELL
    assert proposal.invalidation_price == common_stop(Action.SELL, 100.0, 2.0)


def test_the_trend_line_declares_the_common_stop_when_it_acts() -> None:
    """La línea de tendencia compra con pila alcista y su stop es el común."""
    proposal = trend_proposal(snapshot(110.0), indicators(atr=2.0), PRESET)

    assert proposal.action is Action.BUY
    assert proposal.invalidation_price == common_stop(Action.BUY, 110.0, 2.0)

    flat = trend_proposal(snapshot(110.0), indicators(adx=10.0), PRESET)
    assert flat.action is Action.HOLD
    assert flat.invalidation_price is None


# ── Sin mirar hacia adelante ─────────────────────────────────────────────────


@pytest.mark.parametrize("cut", CUTS)
def test_the_trend_rule_and_the_stop_read_no_future_candles(cut: int) -> None:
    """La decisión y el stop en la vela `i` son los mismos con y sin futuro delante.

    Mismo método que `test_lookahead`, aplicado a lo que las líneas base derivan de
    los indicadores: no basta que las diez columnas sean estables si una regla las
    leyera por otra posición.
    """
    candles = real_candles()
    complete = to_indicator_set(enrich(candles, PRESET).iloc[: cut + 1], PRESET)
    truncated = to_indicator_set(enrich(candles.iloc[: cut + 1], PRESET), PRESET)
    close = float(candles["close"].iloc[cut])

    assert trend_action(truncated, close, PRESET) is trend_action(complete, close, PRESET)
    for side in (Action.BUY, Action.SELL):
        assert common_stop(side, close, atr_of(truncated)) == common_stop(
            side, close, atr_of(complete)
        )


def test_the_trend_rule_is_not_vacuous_over_the_real_history() -> None:
    """La prueba anterior no vale si la regla siempre dice hold: sobre el histórico real actúa."""
    candles = real_candles()
    enriched = enrich(candles, PRESET)
    actions = {
        trend_action(
            to_indicator_set(enriched.iloc[: cut + 1], PRESET),
            float(candles["close"].iloc[cut]),
            PRESET,
        )
        for cut in range(60, len(candles))
    }

    assert actions == {Action.BUY, Action.SELL, Action.HOLD}
