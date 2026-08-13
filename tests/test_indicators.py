"""Pruebas del cálculo de indicadores."""

from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from crypto_agents.indicators import (
    DEFAULT_PRESET,
    IndicatorError,
    IndicatorPreset,
    compute_indicators,
    enrich,
    to_indicator_set,
)
from tests.conftest import candles, drifting_closes

SMALL_PRESET = IndicatorPreset(
    rsi=5, ema_fast=3, ema_slow=8, ema_trend=21, atr=5, adx=5, bbands=5, volume_ma=5
)


def test_preset_rejects_unordered_emas() -> None:
    """Un cruce de medias no significa nada si la rápida no es más rápida."""
    with pytest.raises(ValidationError, match="ema_fast < ema_slow < ema_trend"):
        IndicatorPreset(ema_fast=50, ema_slow=20)


def test_min_bars_doubles_the_longest_period() -> None:
    """Warm-up con margen: las EMA arrastran NaN más allá de su periodo nominal."""
    assert DEFAULT_PRESET.min_bars == 400


def test_insufficient_history_fails_with_a_countable_message() -> None:
    """El mensaje dice cuántas velas hacen falta y cuántas hay."""
    with pytest.raises(IndicatorError, match=r"al menos 400 velas.*hay 100"):
        enrich(candles(drifting_closes(100)))


def test_enrich_adds_every_declared_column() -> None:
    """Los nombres son exactamente lo que un agente podrá citar."""
    enriched = enrich(candles(drifting_closes(SMALL_PRESET.min_bars)), SMALL_PRESET)
    for name in SMALL_PRESET.column_names():
        assert name in enriched.columns


def test_enrich_preserves_the_original_candles() -> None:
    """El DataFrame de entrada no se toca."""
    original = candles(drifting_closes(SMALL_PRESET.min_bars))
    before = original.copy()
    enrich(original, SMALL_PRESET)
    pd.testing.assert_frame_equal(original, before)


def test_bollinger_columns_are_renamed() -> None:
    """pandas-ta emite `BBL_5_2.0_2.0`; ese nombre acabaría en un prompt."""
    enriched = enrich(candles(drifting_closes(SMALL_PRESET.min_bars)), SMALL_PRESET)
    assert "BBL_5" in enriched.columns
    assert not any(column.startswith("BBL_5_") for column in enriched.columns)


def test_indicator_set_has_exactly_the_preset_names() -> None:
    """Nada de más ni de menos: `unknown_indicators()` compara contra este conjunto."""
    indicators = compute_indicators(candles(drifting_closes(SMALL_PRESET.min_bars)), SMALL_PRESET)
    assert indicators.names == frozenset(SMALL_PRESET.column_names())


def test_indicator_set_takes_the_last_row() -> None:
    """Solo la última fila viaja en el estado."""
    frame = candles(drifting_closes(SMALL_PRESET.min_bars))
    enriched = enrich(frame, SMALL_PRESET)
    indicators = to_indicator_set(enriched, SMALL_PRESET)
    assert indicators.values["EMA_3"] == pytest.approx(float(enriched["EMA_3"].iloc[-1]))


def test_default_preset_produces_finite_values() -> None:
    """Con el warm-up del preset por defecto no queda ningún NaN."""
    indicators = compute_indicators(candles(drifting_closes(DEFAULT_PRESET.min_bars)))
    assert len(indicators.values) == len(DEFAULT_PRESET.column_names())


def test_missing_columns_are_reported() -> None:
    """Extraer de un DataFrame sin enriquecer debe decir qué falta."""
    with pytest.raises(IndicatorError, match="faltan columnas"):
        to_indicator_set(candles(drifting_closes(50)), SMALL_PRESET)


def test_nan_in_last_row_is_rejected_by_the_contract() -> None:
    """Warm-up insuficiente llega hasta `IndicatorSet`, que lo rechaza."""
    enriched = enrich(candles(drifting_closes(SMALL_PRESET.min_bars)), SMALL_PRESET)
    enriched.loc[enriched.index[-1], "RSI_5"] = float("nan")
    with pytest.raises(ValidationError, match="no finitos"):
        to_indicator_set(enriched, SMALL_PRESET)
