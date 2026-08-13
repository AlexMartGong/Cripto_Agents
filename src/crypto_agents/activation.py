"""Gate de activación: decide si vale la pena gastar llamadas a modelos.

Cada regla es una función pura sobre el DataFrame enriquecido y devuelve el
trigger que disparó, o `None`. Sin estado entre corridas: el cruce ocurrió en la
última vela cerrada, la ruptura es contra el rango de las N velas previas, el
salto de volatilidad es el ATR contra su propia media. Eso las hace testeables
con velas sintéticas y sin persistencia.

Si ninguna regla dispara, el grafo termina aquí y no se gasta cuota.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from crypto_agents.indicators import DEFAULT_PRESET
from crypto_agents.state import ActivationCheck

if TYPE_CHECKING:
    import pandas as pd

    from crypto_agents.indicators import IndicatorPreset

__all__ = [
    "DEFAULT_CONFIG",
    "RULES",
    "ActivationConfig",
    "ActivationRule",
    "evaluate_activation",
    "ma_cross",
    "range_breakout",
    "regime_change",
    "volatility_jump",
]


@dataclass(frozen=True, slots=True)
class ActivationConfig:
    """Umbrales del gate. Congelado: no se ajusta a mitad de una corrida."""

    breakout_lookback: int = 20
    atr_jump_ratio: float = 1.5
    atr_jump_lookback: int = 20
    adx_trend_threshold: float = 25.0
    preset: IndicatorPreset = DEFAULT_PRESET


DEFAULT_CONFIG = ActivationConfig()


class ActivationRule(Protocol):
    """Regla del gate: mira las velas enriquecidas y devuelve un trigger o nada."""

    def __call__(self, enriched: pd.DataFrame, config: ActivationConfig) -> str | None:
        """Nombre del trigger si dispara, `None` si no."""
        ...


def _crossed(series_a: pd.Series, series_b: pd.Series) -> int:
    """Signo del cruce entre las dos últimas velas: 1 al alza, -1 a la baja, 0 sin cruce.

    Compara el signo de la diferencia, no los valores: un toque exacto seguido de
    separación no cuenta como cruce en la dirección contraria.
    """
    previous = float(series_a.iloc[-2]) - float(series_b.iloc[-2])
    current = float(series_a.iloc[-1]) - float(series_b.iloc[-1])
    if previous <= 0.0 < current:
        return 1
    if previous >= 0.0 > current:
        return -1
    return 0


def ma_cross(enriched: pd.DataFrame, config: ActivationConfig) -> str | None:
    """La media rápida cruzó la lenta en la última vela cerrada."""
    fast = enriched[f"EMA_{config.preset.ema_fast}"]
    slow = enriched[f"EMA_{config.preset.ema_slow}"]
    direction = _crossed(fast, slow)
    if direction > 0:
        return "ma_cross_bullish"
    if direction < 0:
        return "ma_cross_bearish"
    return None


def range_breakout(enriched: pd.DataFrame, config: ActivationConfig) -> str | None:
    """El cierre rompió el rango de las N velas previas.

    El rango excluye la vela actual: compararla consigo misma haría que cualquier
    máximo local pareciera una ruptura.
    """
    window = enriched.iloc[-config.breakout_lookback - 1 : -1]
    if window.empty:
        return None
    close = float(enriched["close"].iloc[-1])
    if close > float(window["high"].max()):
        return "range_breakout_up"
    if close < float(window["low"].min()):
        return "range_breakout_down"
    return None


def volatility_jump(enriched: pd.DataFrame, config: ActivationConfig) -> str | None:
    """El ATR actual se disparó respecto a su propia media reciente."""
    atr = enriched[f"ATRr_{config.preset.atr}"]
    baseline = atr.iloc[-config.atr_jump_lookback - 1 : -1].mean()
    if not baseline > 0.0:
        return None
    if float(atr.iloc[-1]) / float(baseline) >= config.atr_jump_ratio:
        return "volatility_jump"
    return None


def regime_change(enriched: pd.DataFrame, config: ActivationConfig) -> str | None:
    """Cambió el régimen de fondo: el precio cruzó la media larga, o el ADX cruzó el umbral.

    Son las dos formas en que cambia el régimen: dirección (por encima o por
    debajo de la tendencia) y carácter (con tendencia o en rango).
    """
    close = enriched["close"]
    trend = enriched[f"EMA_{config.preset.ema_trend}"]
    direction = _crossed(close, trend)
    if direction > 0:
        return "regime_change_above_trend"
    if direction < 0:
        return "regime_change_below_trend"

    adx = enriched[f"ADX_{config.preset.adx}"]
    previous, current = float(adx.iloc[-2]), float(adx.iloc[-1])
    threshold = config.adx_trend_threshold
    if previous < threshold <= current:
        return "regime_change_trending"
    if previous >= threshold > current:
        return "regime_change_ranging"
    return None


RULES: tuple[ActivationRule, ...] = (ma_cross, range_breakout, volatility_jump, regime_change)


def evaluate_activation(
    enriched: pd.DataFrame,
    config: ActivationConfig = DEFAULT_CONFIG,
    rules: tuple[ActivationRule, ...] = RULES,
) -> ActivationCheck:
    """Aplica todas las reglas. Con al menos un trigger, el grafo sigue."""
    if len(enriched) < 2:
        return ActivationCheck(
            should_run=False, triggers=[], reason="hacen falta al menos dos velas cerradas"
        )

    triggers = [trigger for rule in rules if (trigger := rule(enriched, config)) is not None]
    if triggers:
        return ActivationCheck(
            should_run=True,
            triggers=triggers,
            reason=f"{len(triggers)} regla(s) dispararon: {', '.join(triggers)}",
        )
    return ActivationCheck(
        should_run=False, triggers=[], reason="ninguna regla de activación disparó"
    )
