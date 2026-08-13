"""Pruebas del gate de activación.

Cada regla tiene su caso que dispara y su caso que calla. La línea base es una
serie plana: sobre ella las cuatro reglas guardan silencio, así que cualquier
disparo viene de la forma que la prueba añadió al final y de nada más.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from crypto_agents.activation import (
    ActivationConfig,
    evaluate_activation,
    ma_cross,
    range_breakout,
    regime_change,
    volatility_jump,
)
from crypto_agents.indicators import IndicatorPreset, enrich
from tests.conftest import candles, flat_closes

if TYPE_CHECKING:
    import pandas as pd

PRESET = IndicatorPreset(
    rsi=5, ema_fast=3, ema_slow=8, ema_trend=21, atr=5, adx=5, bbands=5, volume_ma=5
)
CONFIG = ActivationConfig(preset=PRESET)
BASE = flat_closes(PRESET.min_bars)


def series(*extra: float) -> pd.DataFrame:
    """Serie plana con las velas indicadas añadidas al final, ya enriquecida."""
    return enrich(candles([*BASE, *extra]), PRESET)


# ─────────────────────────────────────────── Cruce de medias ──────────────────────────────────────


def test_ma_cross_fires_upward() -> None:
    """La media rápida cruza la lenta al alza en la última vela."""
    assert ma_cross(series(110.0), CONFIG) == "ma_cross_bullish"


def test_ma_cross_fires_downward() -> None:
    """Y también a la baja."""
    assert ma_cross(series(90.0), CONFIG) == "ma_cross_bearish"


def test_ma_cross_is_quiet_without_a_cross() -> None:
    """Sobre una serie plana las medias coinciden y no hay cruce."""
    assert ma_cross(series(), CONFIG) is None


def test_ma_cross_does_not_refire_while_separated() -> None:
    """Ya cruzadas, seguir separándose no es un cruce nuevo.

    Sin esto el gate dispararía en cada vela de una tendencia y gastaría cuota
    repitiendo la misma señal.
    """
    assert ma_cross(series(110.0, 111.0), CONFIG) is None


# ────────────────────────────────────────── Ruptura de rango ──────────────────────────────────────


def test_range_breakout_fires_upward() -> None:
    """El cierre supera el máximo de las velas previas."""
    assert range_breakout(series(110.0), CONFIG) == "range_breakout_up"


def test_range_breakout_fires_downward() -> None:
    """Y por debajo del mínimo."""
    assert range_breakout(series(90.0), CONFIG) == "range_breakout_down"


def test_range_breakout_is_quiet_inside_the_range() -> None:
    """Un movimiento dentro del rango previo no es ruptura."""
    assert range_breakout(series(100.2), CONFIG) is None


def test_range_breakout_needs_to_exceed_not_match() -> None:
    """Tocar el máximo previo no basta: la comparación es estricta."""
    enriched = series(100.2)
    window_high = float(enriched["high"].iloc[-CONFIG.breakout_lookback - 1 : -1].max())
    enriched.loc[enriched.index[-1], "close"] = window_high
    assert range_breakout(enriched, CONFIG) is None


# ──────────────────────────────────────── Salto de volatilidad ────────────────────────────────────


def test_volatility_jump_fires_on_a_wide_candle() -> None:
    """Una vela de rango muy superior a la media reciente dispara."""
    assert volatility_jump(series(110.0), CONFIG) == "volatility_jump"


def test_volatility_jump_is_quiet_on_constant_range() -> None:
    """Con rango constante el ATR no salta."""
    assert volatility_jump(series(), CONFIG) is None


def test_volatility_jump_is_quiet_below_the_ratio() -> None:
    """Un repunte pequeño queda por debajo del umbral configurado."""
    assert volatility_jump(series(100.2), CONFIG) is None


def test_volatility_jump_respects_a_stricter_ratio() -> None:
    """El umbral es un dato: subirlo apaga un disparo que antes ocurría."""
    strict = ActivationConfig(preset=PRESET, atr_jump_ratio=10.0)
    assert volatility_jump(series(110.0), strict) is None


# ───────────────────────────────────────── Cambio de régimen ──────────────────────────────────────


def test_regime_change_fires_crossing_above_the_trend() -> None:
    """El precio cruza la media larga hacia arriba."""
    assert regime_change(series(110.0), CONFIG) == "regime_change_above_trend"


def test_regime_change_fires_crossing_below_the_trend() -> None:
    """Y hacia abajo."""
    assert regime_change(series(90.0), CONFIG) == "regime_change_below_trend"


def test_regime_change_is_quiet_without_a_cross() -> None:
    """Serie plana: ni cruce de tendencia ni ADX definido."""
    assert regime_change(series(), CONFIG) is None


def test_regime_change_fires_when_adx_crosses_into_trend() -> None:
    """El carácter del mercado cambia aunque el precio no cruce la media larga.

    El ADX se fija directamente sobre el DataFrame: la regla es pura sobre las
    columnas enriquecidas, y así se prueba esa rama sin fabricar una serie de
    precios que produzca el cruce por accidente.
    """
    enriched = series()
    enriched.loc[enriched.index[-2], f"ADX_{PRESET.adx}"] = 20.0
    enriched.loc[enriched.index[-1], f"ADX_{PRESET.adx}"] = 30.0
    assert regime_change(enriched, CONFIG) == "regime_change_trending"


def test_regime_change_fires_when_adx_falls_into_range() -> None:
    """Y al revés: el mercado deja de tener tendencia."""
    enriched = series()
    enriched.loc[enriched.index[-2], f"ADX_{PRESET.adx}"] = 30.0
    enriched.loc[enriched.index[-1], f"ADX_{PRESET.adx}"] = 20.0
    assert regime_change(enriched, CONFIG) == "regime_change_ranging"


def test_regime_change_is_quiet_while_adx_stays_on_one_side() -> None:
    """Un ADX alto y estable no es un cambio de régimen."""
    enriched = series()
    enriched.loc[enriched.index[-2], f"ADX_{PRESET.adx}"] = 30.0
    enriched.loc[enriched.index[-1], f"ADX_{PRESET.adx}"] = 35.0
    assert regime_change(enriched, CONFIG) is None


# ──────────────────────────────────────────── El gate ─────────────────────────────────────────────


def test_gate_stays_closed_when_nothing_happens() -> None:
    """Sin cambio material el grafo termina aquí y no se gasta cuota."""
    check = evaluate_activation(series(), CONFIG)
    assert check.should_run is False
    assert check.triggers == []
    assert "ninguna regla" in check.reason


def test_gate_opens_and_lists_every_trigger() -> None:
    """Una vela que rompe todo debe registrar las cuatro reglas."""
    check = evaluate_activation(series(110.0), CONFIG)
    assert check.should_run is True
    assert check.triggers == [
        "ma_cross_bullish",
        "range_breakout_up",
        "volatility_jump",
        "regime_change_above_trend",
    ]


def test_gate_opens_with_a_single_trigger() -> None:
    """Basta una regla para justificar el gasto."""
    check = evaluate_activation(series(100.2), CONFIG)
    assert check.should_run is True
    assert check.triggers == ["ma_cross_bullish", "regime_change_above_trend"]


def test_gate_refuses_to_run_without_two_candles() -> None:
    """Ninguna regla puede comparar la última vela con la anterior si no hay anterior."""
    check = evaluate_activation(candles([100.0]), CONFIG)
    assert check.should_run is False
    assert "dos velas" in check.reason


def test_gate_accepts_a_custom_rule_set() -> None:
    """Las reglas se inyectan: un subconjunto permite aislar comportamiento."""
    check = evaluate_activation(series(110.0), CONFIG, rules=(volatility_jump,))
    assert check.triggers == ["volatility_jump"]
