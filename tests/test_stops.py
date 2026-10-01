"""Pruebas del stop común.

Es la única función que fabrica un stop que no declaró un decisor: las líneas base
lo ponen en su `Proposal` y la puntuación común lo reconstruye sobre cualquier
brazo. Si tuvieran dos fórmulas, la comparación entre un brazo y una línea base
mediría la diferencia entre las dos y no la del decisor.
"""

from __future__ import annotations

import pytest

from crypto_agents.state import Action, IndicatorSet, stop_on_wrong_side
from crypto_agents.stops import COMMON_STOP_ATR_MULTIPLE, atr_of, common_stop


def test_the_multiple_was_frozen_before_any_result_existed() -> None:
    """2 ATR, el valor convencional, fijado antes de ver una sola salida de la ablación.

    Si esta prueba obliga a editar el número, la edición es una decisión que hay
    que defender en el commit y no un ajuste a lo que salió.
    """
    assert COMMON_STOP_ATR_MULTIPLE == 2.0


def test_a_buy_stops_two_atr_below_the_entry() -> None:
    """Entrada 100 y ATR 1.5: el stop de una compra está en 100 - 3 = 97."""
    assert common_stop(Action.BUY, 100.0, 1.5) == pytest.approx(97.0)


def test_a_sell_stops_two_atr_above_the_entry() -> None:
    """La venta es el espejo: 100 + 3 = 103."""
    assert common_stop(Action.SELL, 100.0, 1.5) == pytest.approx(103.0)


@pytest.mark.parametrize("side", [Action.BUY, Action.SELL])
def test_the_stop_is_always_on_the_side_that_invalidates(side: Action) -> None:
    """Con ATR positivo el stop nunca queda del lado equivocado: el veto no lo toca."""
    stop = common_stop(side, 100.0, 0.01)

    assert not stop_on_wrong_side(side, stop, 100.0)


@pytest.mark.parametrize("side", [Action.BUY, Action.SELL])
def test_a_zero_atr_leaves_the_stop_on_the_entry(side: Action) -> None:
    """Sin recorrido no hay stop útil: queda en la entrada y el gate lo vetará.

    No se inventa una distancia mínima. Una serie de velas planas es un dato, y
    esconderlo con un suelo arbitrario sería ajustar el stop a lo que conviene.
    """
    assert common_stop(side, 100.0, 0.0) == 100.0
    assert stop_on_wrong_side(side, 100.0, 100.0)


def test_a_hold_has_no_stop() -> None:
    """Un hold no abre posición: pedirle un stop es un bug de quien llama."""
    with pytest.raises(ValueError, match="hold"):
        common_stop(Action.HOLD, 100.0, 1.0)


def test_a_negative_atr_is_refused() -> None:
    """Un ATR negativo no es una medida de rango."""
    with pytest.raises(ValueError, match="ATR"):
        common_stop(Action.BUY, 100.0, -1.0)


def test_a_stop_that_is_not_a_price_is_refused() -> None:
    """Una compra a 1.0 con ATR 0.6 pondría el stop en -0.2: eso no es un precio."""
    with pytest.raises(ValueError, match="positivo"):
        common_stop(Action.BUY, 1.0, 0.6)


def test_the_atr_is_found_whatever_the_period_of_the_preset() -> None:
    """`ATRr_14` en producción y `ATRr_5` en el preset corto de las pruebas."""
    assert atr_of(IndicatorSet(values={"ATRr_14": 2.5, "EMA_20": 100.0})) == 2.5
    assert atr_of(IndicatorSet(values={"ATRr_5": 1.25, "EMA_3": 100.0})) == 1.25


def test_without_an_atr_there_is_no_stop() -> None:
    """Sin columna de ATR, mejor decirlo que devolver cero."""
    with pytest.raises(ValueError, match="ATRr_"):
        atr_of(IndicatorSet(values={"EMA_20": 100.0}))


def test_two_atr_columns_are_ambiguous() -> None:
    """Dos candidatos son un preset mal formado: se rechaza en vez de escoger uno."""
    with pytest.raises(ValueError, match="ATRr_"):
        atr_of(IndicatorSet(values={"ATRr_5": 1.0, "ATRr_14": 2.0}))
