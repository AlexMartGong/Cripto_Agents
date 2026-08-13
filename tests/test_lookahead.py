"""Pruebas de que ningún cálculo mira hacia adelante.

El método es el mismo para todo: calcular sobre la serie completa y sobre la
serie truncada justo después de la vela `i`, y exigir que lo que se lee en `i`
sea idéntico. Si algo usara datos posteriores, truncar por la derecha lo
cambiaría.

Se trunca solo por la derecha, nunca por la izquierda: EMA y ADX son recursivos y
arrastran todo el historial previo, así que recortar el principio sí cambia el
valor en `i` y eso es correcto, no un look-ahead.

Sobre datos reales, no sintéticos: una serie construida a mano puede tener una
forma que esconda el error justo en la ventana probada.
"""

from __future__ import annotations

import pytest

from crypto_agents.activation import DEFAULT_CONFIG, RULES, ActivationConfig, evaluate_activation
from crypto_agents.indicators import IndicatorPreset, enrich
from tests.conftest import PRESET, real_candles

CUTS = [120, 200, 305, 410, 499]
"""Cortes repartidos por el histórico, incluido el último."""


@pytest.mark.parametrize("cut", CUTS)
def test_no_indicator_of_the_preset_uses_future_candles(cut: int) -> None:
    """Los diez valores del preset en la vela `i` no cambian al cortar la serie ahí.

    Es la prueba que separa un backtest de una ilusión: si un indicador leyera una
    vela posterior, el replay compraría sabiendo lo que iba a pasar y las métricas
    de la corrida no significarían nada.
    """
    candles = real_candles()
    complete = enrich(candles, PRESET)
    truncated = enrich(candles.iloc[: cut + 1], PRESET)

    for name in PRESET.column_names():
        expected = float(complete[name].iloc[cut])
        actual = float(truncated[name].iloc[cut])
        assert actual == expected, f"{name} en la vela {cut} cambia al truncar la serie"


@pytest.mark.parametrize("cut", CUTS)
def test_the_activation_gate_reads_no_future_candles(cut: int) -> None:
    """El veredicto del gate sobre la vela `i` es el mismo con y sin futuro delante."""
    candles = real_candles()
    complete = enrich(candles, PRESET)
    truncated = enrich(candles.iloc[: cut + 1], PRESET)
    config = ActivationConfig(preset=PRESET)

    expected = evaluate_activation(complete.iloc[: cut + 1], config)
    actual = evaluate_activation(truncated, config)
    assert actual.triggers == expected.triggers
    assert actual.should_run == expected.should_run


@pytest.mark.parametrize("rule", RULES, ids=lambda rule: rule.__name__)
@pytest.mark.parametrize("cut", [200, 410])
def test_each_activation_rule_is_stable_under_truncation(rule: object, cut: int) -> None:
    """Regla por regla, para que un fallo señale cuál mira adelante."""
    candles = real_candles()
    complete = enrich(candles, PRESET)
    truncated = enrich(candles.iloc[: cut + 1], PRESET)
    config = ActivationConfig(preset=PRESET)

    assert rule(truncated, config) == rule(complete.iloc[: cut + 1], config)  # type: ignore[operator]


def test_the_default_preset_is_also_clean() -> None:
    """El preset corto de las pruebas no es el que corre en producción."""
    candles = real_candles()
    preset = IndicatorPreset()
    cut = len(candles) - 1

    complete = enrich(candles, preset)
    truncated = enrich(candles.iloc[: cut + 1], preset)
    for name in preset.column_names():
        assert float(truncated[name].iloc[cut]) == float(complete[name].iloc[cut])


def test_the_default_activation_config_is_also_clean() -> None:
    """Y con los umbrales de producción, no con los de las pruebas."""
    candles = real_candles()
    complete = enrich(candles, IndicatorPreset())
    cut = 400
    truncated = enrich(candles.iloc[: cut + 1], IndicatorPreset())

    expected = evaluate_activation(complete.iloc[: cut + 1], DEFAULT_CONFIG)
    assert evaluate_activation(truncated, DEFAULT_CONFIG).triggers == expected.triggers
