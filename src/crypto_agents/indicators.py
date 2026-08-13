"""Cálculo de indicadores con pandas-ta.

Los agentes técnicos interpretan estos valores; nunca los calculan. Por eso los
nombres de columna importan: son exactamente lo que un agente puede citar en
`Observation.cites`, y `unknown_indicators()` los contrasta contra este conjunto.

`enrich()` devuelve la serie completa (la necesita el gate de activación, que
compara la última vela con las anteriores) y `to_indicator_set()` extrae la
última fila, que es lo único que viaja en el estado.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

import pandas_ta as ta  # registra el accesor `df.ta`
from pydantic import Field, model_validator

from crypto_agents.state import FrozenModel, IndicatorSet

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

__all__ = [
    "DEFAULT_PRESET",
    "IndicatorError",
    "IndicatorPreset",
    "compute_indicators",
    "enrich",
    "to_indicator_set",
]


class IndicatorError(RuntimeError):
    """No hay historial suficiente para calcular el preset sin NaN."""


class IndicatorPreset(FrozenModel):
    """Periodos del preset técnico.

    Vive en el módulo y no en `Settings`: los periodos son parte del método de
    análisis, no de la configuración de despliegue.
    """

    rsi: int = Field(default=14, gt=1)
    ema_fast: int = Field(default=20, gt=1)
    ema_slow: int = Field(default=50, gt=1)
    ema_trend: int = Field(default=200, gt=1)
    atr: int = Field(default=14, gt=1)
    adx: int = Field(default=14, gt=1)
    bbands: int = Field(default=20, gt=1)
    volume_ma: int = Field(default=20, gt=1)

    @model_validator(mode="after")
    def _emas_are_ordered(self) -> Self:
        """Un cruce de medias no significa nada si la rápida no es más rápida."""
        if not self.ema_fast < self.ema_slow < self.ema_trend:
            raise ValueError("se exige ema_fast < ema_slow < ema_trend")
        return self

    @property
    def periods(self) -> tuple[int, ...]:
        """Todos los periodos declarados."""
        return (
            self.rsi,
            self.ema_fast,
            self.ema_slow,
            self.ema_trend,
            self.atr,
            self.adx,
            self.bbands,
            self.volume_ma,
        )

    @property
    def min_bars(self) -> int:
        """Velas mínimas para un warm-up razonable.

        Heurística de dos veces el periodo más largo: las medias exponenciales y
        el ADX arrastran NaN bastante más allá de su periodo nominal. La garantía
        real la da `IndicatorSet`, que rechaza cualquier NaN residual.
        """
        return 2 * max(self.periods)

    def column_names(self) -> tuple[str, ...]:
        """Nombres que producirá `enrich()`, en orden estable."""
        return (
            f"RSI_{self.rsi}",
            f"EMA_{self.ema_fast}",
            f"EMA_{self.ema_slow}",
            f"EMA_{self.ema_trend}",
            f"ATRr_{self.atr}",
            f"ADX_{self.adx}",
            f"BBL_{self.bbands}",
            f"BBU_{self.bbands}",
            f"BBP_{self.bbands}",
            f"VOL_SMA_{self.volume_ma}",
        )


DEFAULT_PRESET = IndicatorPreset()


def enrich(candles: pd.DataFrame, preset: IndicatorPreset = DEFAULT_PRESET) -> pd.DataFrame:
    """Añade las columnas del preset a las velas, sin tocar las originales.

    Las bandas de Bollinger se renombran: pandas-ta las emite como
    `BBL_20_2.0_2.0`, y ese nombre acabaría dentro de un prompt y de una cita de
    agente. Los nombres cortos son estables y legibles.
    """
    if len(candles) < preset.min_bars:
        raise IndicatorError(
            f"se requieren al menos {preset.min_bars} velas para el preset, hay {len(candles)}"
        )

    enriched = candles.copy()
    enriched[f"RSI_{preset.rsi}"] = candles.ta.rsi(length=preset.rsi)
    for length in (preset.ema_fast, preset.ema_slow, preset.ema_trend):
        enriched[f"EMA_{length}"] = candles.ta.ema(length=length)
    enriched[f"ATRr_{preset.atr}"] = candles.ta.atr(length=preset.atr)
    enriched[f"ADX_{preset.adx}"] = candles.ta.adx(length=preset.adx)[f"ADX_{preset.adx}"]

    bands = candles.ta.bbands(length=preset.bbands)
    suffix = f"_{preset.bbands}_2.0_2.0"
    enriched[f"BBL_{preset.bbands}"] = bands[f"BBL{suffix}"]
    enriched[f"BBU_{preset.bbands}"] = bands[f"BBU{suffix}"]
    enriched[f"BBP_{preset.bbands}"] = bands[f"BBP{suffix}"]

    enriched[f"VOL_SMA_{preset.volume_ma}"] = ta.sma(candles["volume"], length=preset.volume_ma)
    return enriched


def to_indicator_set(
    enriched: pd.DataFrame, preset: IndicatorPreset = DEFAULT_PRESET
) -> IndicatorSet:
    """Extrae la última fila. `IndicatorSet` rechaza cualquier NaN residual."""
    names: Sequence[str] = preset.column_names()
    missing = [name for name in names if name not in enriched.columns]
    if missing:
        raise IndicatorError(f"faltan columnas en el DataFrame: {', '.join(missing)}")
    last = enriched.loc[:, list(names)].iloc[-1]
    return IndicatorSet(values={name: float(last[name]) for name in names})


def compute_indicators(
    candles: pd.DataFrame, preset: IndicatorPreset = DEFAULT_PRESET
) -> IndicatorSet:
    """Atajo: enriquece y extrae la última fila."""
    return to_indicator_set(enrich(candles, preset), preset)
