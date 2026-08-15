"""Pruebas del barrido del gate sobre histórico.

Todo en frío: las series salen del histórico versionado o de cierres escritos a
mano. Lo que se comprueba es que el barrido cuenta lo que el gate dice, no que el
gate acierte — eso ya lo cubre `tests/test_activation.py`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from crypto_agents.activation import DEFAULT_CONFIG, evaluate_activation
from crypto_agents.activation_sweep import (
    DEFAULT_SYMBOLS,
    SeriesSweep,
    history_path,
    load_series,
    render_report,
    sweep_series,
)
from crypto_agents.indicators import DEFAULT_PRESET, enrich
from crypto_agents.market import MarketDataError, to_dataframe, write_ohlcv_csv
from tests.conftest import drifting_closes, raw_ohlcv, real_rows

if TYPE_CHECKING:
    from pathlib import Path


def test_the_sweep_counts_exactly_what_the_gate_says() -> None:
    """El barrido no puede tener su propia opinión sobre cuándo dispara una regla.

    Se recorre el mismo histórico a mano llamando a `evaluate_activation` y los
    dos totales deben coincidir. Si el barrido reimplementara las reglas, la
    tabla mediría el barrido.
    """
    rows = real_rows()
    result = sweep_series(rows, "BTC/USDT", "4h")

    enriched = enrich(to_dataframe(rows), DEFAULT_PRESET)
    expected = sum(
        1
        for index in range(DEFAULT_PRESET.min_bars, len(enriched))
        if evaluate_activation(enriched.iloc[: index + 1], DEFAULT_CONFIG).should_run
    )

    assert result.activated == expected
    assert result.evaluated == len(enriched) - DEFAULT_PRESET.min_bars


def test_warm_up_bars_are_not_counted_as_evaluations() -> None:
    """Las velas de warm-up no llegan al gate en producción, y aquí tampoco.

    Contarlas inflaría el número que dimensiona la ablación con evaluaciones que
    abortarían en `prepare_market_data`.
    """
    rows = real_rows()
    result = sweep_series(rows, "BTC/USDT", "4h")

    assert result.bars == len(rows)
    assert result.evaluated == result.bars - DEFAULT_PRESET.min_bars


def test_rule_totals_are_consistent_with_trigger_totals() -> None:
    """Cada trigger cuenta en su regla: las dos tablas son la misma cuenta agregada distinto."""
    result = sweep_series(real_rows(), "BTC/USDT", "4h")

    assert sum(result.by_trigger.values()) == sum(result.by_rule.values())
    assert set(result.by_rule) <= {"ma_cross", "range_breakout", "volatility_jump", "regime_change"}


def test_every_trigger_of_a_candle_is_counted_not_just_the_first() -> None:
    """Una vela puede disparar varias reglas a la vez, y las cuatro cuentan.

    Quedarse con el primer trigger subcontaría todas las reglas menos la que va
    primero en `RULES`, y la conclusión «esta regla apenas dispara» sería un
    artefacto del orden de una tupla. Se ve porque hay más triggers que
    activaciones: si fueran iguales, cada vela habría disparado exactamente una.
    """
    result = sweep_series(real_rows(), "BTC/USDT", "4h")

    assert result.activated > 0
    assert sum(result.by_trigger.values()) > result.activated


def test_a_trigger_without_a_known_rule_is_loud() -> None:
    """Una regla nueva mal agregada haría creer que no dispara nunca.

    Ese sería un artefacto del agregador presentado como hallazgo, así que el
    barrido se niega antes que contar mal.
    """
    from crypto_agents.activation_sweep import _rule_of

    with pytest.raises(MarketDataError, match="trigger sin regla"):
        _rule_of("regla_inventada_up")


def test_a_quiet_series_activates_nothing() -> None:
    """Sobre una serie sin forma, el gate no abre y el barrido lo dice sin fallar."""
    rows = raw_ohlcv(drifting_closes(DEFAULT_PRESET.min_bars + 20))

    result = sweep_series(rows, "TEST/USDT", "4h")

    assert result.evaluated == 20
    assert result.activated == 0
    assert result.rate == 0.0
    assert result.by_rule == {}


def test_the_digest_pins_which_candles_were_swept() -> None:
    """Sin las velas versionadas, el digest es lo que hace comparables dos tablas."""
    rows = real_rows()
    first = sweep_series(rows, "BTC/USDT", "4h")
    altered = [list(row) for row in rows]
    altered[-1][4] += 0.01

    assert sweep_series(altered, "BTC/USDT", "4h").digest != first.digest


@pytest.mark.asyncio
async def test_a_cached_series_is_not_downloaded_again(tmp_path: Path) -> None:
    """El caché existe para que repetir la tabla no vuelva a pedir 160 páginas.

    Si `load_series` tocara la red aquí, la prueba fallaría por intentar salir:
    ninguna prueba de esta suite tiene permiso para hacerlo.
    """
    rows = raw_ohlcv(drifting_closes(10))
    write_ohlcv_csv(history_path("BTC/USDT", "4h", tmp_path), rows)

    loaded = await load_series(
        "BTC/USDT",
        "4h",
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 1, tzinfo=UTC),
        directory=tmp_path,
    )

    assert len(loaded) == len(rows)


def test_the_report_names_every_series_with_its_provenance() -> None:
    """La tabla sin procedencia no se puede comparar con la siguiente."""
    result = sweep_series(real_rows(), "BTC/USDT", "4h")

    report = render_report([result])

    assert "BTC/USDT" in report
    assert result.digest in report
    assert "Procedencia" in report
    assert str(DEFAULT_PRESET.min_bars) in report


def test_the_report_names_the_triggers_that_never_fired() -> None:
    """«Esta regla no dispara nunca» es una conclusión, y tiene que estar escrita.

    Es una de las tres preguntas que el barrido existe para responder, así que no
    puede quedar como un hueco que el lector tenga que notar.
    """
    quiet = SeriesSweep(
        symbol="TEST/USDT",
        timeframe="4h",
        bars=100,
        evaluated=10,
        activated=1,
        by_trigger={"volatility_jump": 1},
        by_rule={"volatility_jump": 1},
        first=datetime(2024, 1, 1, tzinfo=UTC),
        last=datetime(2024, 2, 1, tzinfo=UTC),
        digest="a" * 64,
    )

    report = render_report([quiet])

    assert "no dispararon ni una vez" in report
    assert "`ma_cross_bullish`" in report


def test_the_seven_symbols_are_declared() -> None:
    """La lista es el eje de la tabla: cambiarla cambia lo que la tabla dice."""
    assert len(DEFAULT_SYMBOLS) == 7
    assert DEFAULT_SYMBOLS[0] == "BTC/USDT"
