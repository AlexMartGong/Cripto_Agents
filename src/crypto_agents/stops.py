"""El stop común: una distancia en ATR que no la elige ningún decisor.

Dos preguntas distintas se mezclan en el resultado de un brazo: si acertó la
dirección y si puso bien el nivel donde su tesis se rompe. Un decisor que compra
con el stop pegado al precio sale invalidado por ruido aunque tuviera razón, y uno
que lo pone lejos aguanta lo que otro no. Para separar las dos hace falta un stop
que sea el mismo para todos.

Esta es la única función que lo fabrica. Las líneas base lo declaran en su
`Proposal` y la puntuación común lo reconstruye sobre cualquier brazo; con dos
fórmulas, la diferencia entre un brazo y una línea base mediría la diferencia entre
las dos. `tests/test_architecture.py` exige que ningún otro módulo cite el múltiplo.

Va en ATR y no en porcentaje fijo porque la corrida salta entre siete símbolos: un
2% es ancho en BTC a 4h y estrecho en SOL, y un stop «común» en la regla pero no en
el riesgo mediría qué símbolo se parece más a ese porcentaje. El ATR iguala la
distancia en unidades de volatilidad. Es lo que obliga a que el registro lleve los
indicadores de la vela evaluada.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from crypto_agents.state import Action

if TYPE_CHECKING:
    from crypto_agents.state import IndicatorSet

__all__ = [
    "ATR_PREFIX",
    "COMMON_STOP_ATR_MULTIPLE",
    "COMMON_STOP_DESCRIPTION",
    "atr_of",
    "common_stop",
]

COMMON_STOP_ATR_MULTIPLE = 2.0
"""Distancia del stop a la entrada, en ATR. Congelada antes de ver ningún resultado.

Dos ATR es el valor convencional, no uno buscado: ninguna salida de la ablación
existía cuando se escribió, y `tests/test_stops.py` falla si alguien lo toca.
"""

COMMON_STOP_DESCRIPTION = f"{COMMON_STOP_ATR_MULTIPLE:g}×ATR"  # noqa: RUF001
"""El stop común como se nombra en un reporte: lo derivan del múltiplo, no lo repiten."""

ATR_PREFIX = "ATRr_"
"""Prefijo de la columna de ATR del preset; el periodo depende del preset."""


def atr_of(indicators: IndicatorSet) -> float:
    """El ATR de la vela evaluada, sea cual sea el periodo del preset.

    Exige exactamente una columna: dos candidatas son un preset mal formado, y
    escoger una en silencio daría un stop que depende del orden del diccionario.
    """
    names = sorted(name for name in indicators.values if name.startswith(ATR_PREFIX))
    if len(names) != 1:
        raise ValueError(
            f"se esperaba una sola columna {ATR_PREFIX}*, hay {len(names)}: {', '.join(names)}"
        )
    return indicators.values[names[0]]


def common_stop(side: Action, entry: float, atr: float) -> float:
    """Nivel de invalidación común: `entry - 2·ATR` en una compra, `entry + 2·ATR` en una venta.

    Un ATR de cero deja el stop en la entrada. No se le inventa una distancia mínima:
    es un dato —velas planas— y el gate lo vetará por estar del lado equivocado en
    vez de que un suelo arbitrario lo esconda.
    """
    if side is Action.HOLD:
        raise ValueError("un hold no abre posición: no tiene stop")
    if atr < 0.0:
        raise ValueError(f"el ATR no puede ser negativo: {atr}")
    distance = COMMON_STOP_ATR_MULTIPLE * atr
    stop = entry - distance if side is Action.BUY else entry + distance
    if stop <= 0.0:
        raise ValueError(f"el stop común no es un precio positivo: {stop}")
    return stop
