"""Líneas base de la ablación: lo que decide quien no tiene un modelo.

Un brazo que no se compara con nada no demuestra nada: que `full` gane dinero en
una ventana alcista no dice si decide o si el mercado subía. Estas cuatro políticas
fijan el suelo contra el que se lee cada resultado, y cuestan cero llamadas.

- `always_buy` y `always_sell`: la dirección más simple, en cada activación del gate.
- `random_uniform`: buy, sell o hold con igual probabilidad, reproducible.
- `rule_trend`: una regla de tendencia clásica sobre las columnas del preset.

Todas emiten un `Proposal` y recorren la misma cola que los brazos con modelos
—decisor, riesgo, ejecución, journal—, y todas declaran el stop común de `stops.py`.
Así lo único que las distingue de un brazo con modelos es quién decide la dirección.

Trabajan detrás del gate de activación, como los demás brazos: miden si un modelo
aporta algo *donde el gate decide mirar*, no si gate más modelo bate a comprar y
mantener.

Nada aquí importa el router, el grafo ni el replay: es lo que impide, por
construcción y no por disciplina, que una línea base cueste una llamada.

Los números de este módulo se congelaron antes de ver ningún resultado. Son
convenciones, no hallazgos: ninguna salida de la ablación existía cuando se
escribieron, y `tests/test_baselines.py` falla si alguien los toca.
"""

from __future__ import annotations

import hashlib
import random
from typing import TYPE_CHECKING

from crypto_agents.state import Action, Proposal
from crypto_agents.stops import atr_of, common_stop

if TYPE_CHECKING:
    from uuid import UUID

    from crypto_agents.indicators import IndicatorPreset
    from crypto_agents.state import IndicatorSet, MarketSnapshot

__all__ = [
    "BASELINE_CONFIDENCE",
    "BASELINE_SIZE",
    "TREND_ADX_MIN",
    "always_buy_proposal",
    "always_sell_proposal",
    "baseline_seed",
    "random_action",
    "random_proposal",
    "trend_action",
    "trend_proposal",
]

TREND_ADX_MIN = 25.0
"""ADX mínimo para que `rule_trend` actúe. El umbral de Wilder, y el que ya usa el gate."""

BASELINE_SIZE = 0.05
"""Tamaño que piden las líneas base. Bajo el tope por posición, así que el gate no lo recorta.

Ninguna métrica de retorno lo pondera: es el relleno que `Proposal` exige.
"""

BASELINE_CONFIDENCE = 0.5
"""Una línea base no tiene confianza que declarar. Relleno que ninguna métrica lee."""

_UNIFORM_ACTIONS = (Action.BUY, Action.SELL, Action.HOLD)


def baseline_seed(seed: int, run_id: UUID) -> int:
    """Semilla de una evaluación: sha-256 de `semilla|run_id`, no `hash()`.

    El `hash()` de una cadena cambia entre procesos; con él, dos corridas del mismo
    plan tomarían decisiones al azar distintas y la tabla no se podría reproducir.
    """
    digest = hashlib.sha256(f"{seed}|{run_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def random_action(seed: int, run_id: UUID) -> Action:
    """Buy, sell o hold con igual probabilidad, función solo de la semilla y del `run_id`.

    El `run_id` de un replay es un UUID5 de símbolo, timeframe e instante: la misma
    evaluación saca la misma acción en cualquier corrida, y no depende del orden en
    que se recorran.
    """
    return random.Random(baseline_seed(seed, run_id)).choice(_UNIFORM_ACTIONS)


def trend_action(indicators: IndicatorSet, close: float, preset: IndicatorPreset) -> Action:
    """Regla de tendencia: pila de medias estricta y mercado con tendencia.

    Compra si `close > EMA_rápida > EMA_lenta > EMA_de_fondo` y `ADX ≥ 25`; vende
    con la pila contraria; en cualquier otro caso, hold. Los periodos son los del
    preset. Las desigualdades son estrictas —un empate no es una pila— y el umbral
    de ADX no, porque el de Wilder incluye el 25.

    Lee solo el cierre y columnas del preset, en la última vela: no hay dónde mirar
    hacia adelante, y `tests/test_baselines.py` lo comprueba sobre el histórico real.
    """
    values = indicators.values
    fast = values[f"EMA_{preset.ema_fast}"]
    slow = values[f"EMA_{preset.ema_slow}"]
    trend = values[f"EMA_{preset.ema_trend}"]
    if values[f"ADX_{preset.adx}"] < TREND_ADX_MIN:
        return Action.HOLD
    if close > fast > slow > trend:
        return Action.BUY
    if close < fast < slow < trend:
        return Action.SELL
    return Action.HOLD


def _proposal(
    action: Action, snapshot: MarketSnapshot, indicators: IndicatorSet, rationale: str
) -> Proposal:
    """Propuesta de una línea base: el stop es siempre el común, y un hold no lleva ninguno."""
    if action is Action.HOLD:
        return Proposal(
            action=action,
            confidence=BASELINE_CONFIDENCE,
            size_fraction=0.0,
            rationale=rationale,
        )
    return Proposal(
        action=action,
        confidence=BASELINE_CONFIDENCE,
        size_fraction=BASELINE_SIZE,
        invalidation_price=common_stop(action, snapshot.close, atr_of(indicators)),
        rationale=rationale,
    )


def always_buy_proposal(snapshot: MarketSnapshot, indicators: IndicatorSet) -> Proposal:
    """Compra en cada activación."""
    return _proposal(
        Action.BUY, snapshot, indicators, "Línea base: compra en cada activación del gate."
    )


def always_sell_proposal(snapshot: MarketSnapshot, indicators: IndicatorSet) -> Proposal:
    """Vende en cada activación."""
    return _proposal(
        Action.SELL, snapshot, indicators, "Línea base: vende en cada activación del gate."
    )


def random_proposal(
    snapshot: MarketSnapshot, indicators: IndicatorSet, seed: int, run_id: UUID
) -> Proposal:
    """Buy, sell o hold al azar uniforme, reproducible desde la semilla del plan."""
    return _proposal(
        random_action(seed, run_id),
        snapshot,
        indicators,
        "Línea base: acción al azar uniforme entre buy, sell y hold.",
    )


def trend_proposal(
    snapshot: MarketSnapshot, indicators: IndicatorSet, preset: IndicatorPreset
) -> Proposal:
    """La regla de tendencia sobre la vela evaluada."""
    return _proposal(
        trend_action(indicators, snapshot.close, preset),
        snapshot,
        indicators,
        "Línea base: regla de tendencia con pila de medias estricta y ADX de Wilder.",
    )
