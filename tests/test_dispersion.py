"""Pruebas de la dispersión del retorno y del efecto detectable.

Todo sobre series sintéticas con la dispersión escrita a mano: ninguna prueba calcula la media
de un retorno del pool ni de una línea base sobre datos reales, que es justo lo que el módulo
se niega a producir. Las medias que aparecen aquí son de series inventadas y sirven para
comprobar que no salen en el reporte.
"""

from __future__ import annotations

import inspect
import statistics
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from crypto_agents import dispersion
from crypto_agents.activation import ActivationConfig
from crypto_agents.activation_sweep import ActivatedBar
from crypto_agents.dispersion import (
    DEFAULT_HOURS_PER_140,
    DEFAULT_SAMPLE_SIZES,
    PoolReport,
    ReturnDispersion,
    SeriesProvenance,
    build_report,
    detectable_difference,
    hours_for,
    main,
    pool_outcomes,
    render_report,
    summarise,
)
from crypto_agents.indicators import compute_indicators
from crypto_agents.market import candles_digest, to_dataframe, write_ohlcv_csv
from crypto_agents.metrics import paired_difference
from crypto_agents.outcomes import Outcome, TradeOutcome
from crypto_agents.selection import replay_window
from crypto_agents.state import Action, FrozenModel
from crypto_agents.stops import atr_of, common_stop
from tests.conftest import PRESET, raw_ohlcv, real_rows

if TYPE_CHECKING:
    from pathlib import Path

CONFIG = ActivationConfig(preset=PRESET)
HORIZON = 6
STEP = timedelta(hours=4)
SYMBOL = "SYN/USDT"


def trade(
    side: Action, entry: float, exit_: float, outcome: Outcome = Outcome.HELD
) -> TradeOutcome:
    """Una posición con la entrada y la salida escritas a mano."""
    return TradeOutcome(
        at=0, side=side, outcome=outcome, entry_price=entry, exit_price=exit_, bars_held=HORIZON
    )


def zigzag_rows(count: int = 80) -> list[list[float]]:
    """Una subida de una unidad por vela con un zigzag de ±0.3.

    El zigzag evita el RSI de una serie sin pérdidas, que es 100 sin más y puede salir NaN.
    La vela `i` y la `i + 6` comparten paridad, así que su diferencia de cierre es exactamente
    seis: el retorno de una compra a seis velas es `6 / cierre_i`, escrito sin pasar por el
    módulo.
    """
    closes = [100.0 + index + (0.3 if index % 2 else -0.3) for index in range(count)]
    return raw_ohlcv(closes, step=STEP, spread=0.5)


def bars_for(rows: list[list[float]], indices: range) -> list[ActivatedBar]:
    """Activaciones inyectadas: el gate no interviene, que es lo que se quiere aislar."""
    return [
        ActivatedBar(
            symbol=SYMBOL,
            timeframe="4h",
            index=index,
            at=datetime.fromtimestamp(rows[index][0] / 1000.0, tz=UTC),
            triggers=("range_breakout_up",),
        )
        for index in indices
    ]


# ───────────────────────────────────────── Desviación ─────────────────────────────────────────────


def test_the_deviation_is_the_sample_one() -> None:
    """Cuatro retornos de ±2%: la muestral es 0.02·√(4/3); la poblacional, 0.02 a secas.

    Dividir entre n en lugar de n - 1 da 0.02 y subestima la dispersión justo cuando n es
    pequeña, que es el caso que esta cifra existe para juzgar.
    """
    outcomes = [
        trade(Action.BUY, 100.0, 102.0),
        trade(Action.BUY, 100.0, 98.0),
        trade(Action.BUY, 100.0, 102.0),
        trade(Action.BUY, 100.0, 98.0),
    ]

    result = summarise("global", Action.BUY, outcomes)

    assert result.stdev == pytest.approx(0.02 * (4 / 3) ** 0.5, rel=1e-12)
    assert result.stdev != pytest.approx(0.02, rel=1e-3)


def test_a_sell_is_measured_in_its_own_direction() -> None:
    """Vender y que baje es +2%: el signo del lado entra en el retorno que se dispersa."""
    outcomes = [trade(Action.SELL, 100.0, 98.0), trade(Action.SELL, 100.0, 102.0)]

    result = summarise("global", Action.SELL, outcomes)

    assert result.stdev == pytest.approx(statistics.stdev([0.02, -0.02]), rel=1e-12)


def test_unresolved_positions_are_counted_apart_and_stay_out_of_the_deviation() -> None:
    """Una posición sin horizonte por delante no es un retorno de cero: no entra."""
    outcomes = [
        trade(Action.BUY, 100.0, 103.0),
        trade(Action.BUY, 100.0, 97.0),
        trade(Action.BUY, 100.0, 100.0, Outcome.UNRESOLVED),
    ]

    result = summarise("global", Action.BUY, outcomes)

    assert (result.resolved, result.unresolved) == (2, 1)
    assert result.stdev == pytest.approx(statistics.stdev([0.03, -0.03]), rel=1e-12)


def test_invalidated_positions_are_counted_over_the_resolved() -> None:
    outcomes = [
        trade(Action.BUY, 100.0, 96.0, Outcome.INVALIDATED),
        trade(Action.BUY, 100.0, 101.0),
        trade(Action.BUY, 100.0, 100.0, Outcome.UNRESOLVED),
    ]

    result = summarise("global", Action.BUY, outcomes)

    assert (result.invalidated, result.resolved) == (1, 2)


@pytest.mark.parametrize("count", [0, 1])
def test_fewer_than_two_returns_have_no_deviation(count: int) -> None:
    """Una sola posición no tiene dispersión que estimar: se dice, no se da cero."""
    result = summarise("global", Action.BUY, [trade(Action.BUY, 100.0, 101.0)][:count])

    assert result.stdev is None


# ───────────────────────────────────────── El pool ────────────────────────────────────────────────


def test_buys_in_the_pool_exit_at_the_horizon_close() -> None:
    """Sin que el stop común los toque, cada compra sale al cierre de la vela `i + 6`.

    Los retornos esperados salen de las velas, no del módulo, y la desviación esperada de
    la función estándar: con la poblacional el módulo no coincide.
    """
    rows = zigzag_rows()
    indices = range(50, 60)

    result = pool_outcomes(rows, bars_for(rows, indices), HORIZON, preset=PRESET, config=CONFIG)

    assert all(item.outcome is Outcome.HELD for item in result.buy)
    expected = [(rows[i + HORIZON][4] - rows[i][4]) / rows[i][4] for i in indices]
    assert [item.gross_return for item in result.buy] == pytest.approx(expected, rel=1e-12)
    assert summarise("global", Action.BUY, result.buy).stdev == pytest.approx(
        statistics.stdev(expected), rel=1e-12
    )


def test_sells_in_the_pool_leave_through_the_common_stop() -> None:
    """La subida alcanza el stop de las ventas, y sale exactamente en `entry + 2·ATR`.

    El stop esperado se reconstruye con `common_stop` y el ATR de la ventana del replay: si el
    módulo puntuara con su propia distancia, la dispersión sería la de ese stop y no la de la
    tabla de la ablación.
    """
    rows = zigzag_rows()
    indices = range(50, 60)

    result = pool_outcomes(rows, bars_for(rows, indices), HORIZON, preset=PRESET, config=CONFIG)

    assert all(item.outcome is Outcome.INVALIDATED for item in result.sell)
    for index, item in zip(indices, result.sell, strict=True):
        atr = atr_of(compute_indicators(to_dataframe(replay_window(rows, index)), PRESET))
        assert item.exit_price == pytest.approx(common_stop(Action.SELL, rows[index][4], atr))


def test_the_atr_comes_from_the_window_the_arm_would_see() -> None:
    """El ATR es recursivo: sobre la serie entera daría otro stop que sobre la ventana del replay.

    Con `candle_limit` más corto que la serie las dos ventanas difieren, y el stop de las
    ventas tiene que ser el de la ventana, no el de `rows[: i + 1]`.
    """
    rows = zigzag_rows(120)
    indices = range(90, 100)
    limit = 50

    result = pool_outcomes(
        rows, bars_for(rows, indices), HORIZON, preset=PRESET, config=CONFIG, candle_limit=limit
    )

    for index, item in zip(indices, result.sell, strict=True):
        window = to_dataframe(replay_window(rows, index, limit))
        atr = atr_of(compute_indicators(window, PRESET))
        assert item.exit_price == pytest.approx(
            common_stop(Action.SELL, rows[index][4], atr), rel=1e-12
        )


def test_activations_without_a_horizon_ahead_are_unresolved() -> None:
    """Las del final de la serie no tienen seis velas por delante: se cuentan y no entran."""
    rows = zigzag_rows()
    indices = range(len(rows) - 3, len(rows) - 1)

    result = pool_outcomes(rows, bars_for(rows, indices), HORIZON, preset=PRESET, config=CONFIG)

    assert all(item.outcome is Outcome.UNRESOLVED for item in (*result.buy, *result.sell))
    assert summarise("global", Action.BUY, result.buy).resolved == 0


def test_activations_the_replay_window_does_not_confirm_are_kept() -> None:
    """El pool es el del barrido: contar las no confirmadas no las quita."""
    rows = zigzag_rows()
    bars = bars_for(rows, range(50, 60))

    result = pool_outcomes(rows, bars, HORIZON, preset=PRESET, config=CONFIG)

    assert len(result.buy) == len(result.sell) == len(bars)
    assert 0 <= result.unconfirmed <= len(bars)


# ───────────────────────────────────── Efecto detectable ──────────────────────────────────────────


def test_the_detectable_difference_matches_the_hand_computation() -> None:
    """sigma = 1, n = 100: el semiancho es 1.96·√(2(1 - rho))/10 y el detectable suma z_0.8."""
    independent = detectable_difference(1.0, 100, 0.0)
    correlated = detectable_difference(1.0, 100, 0.5)

    assert independent.sigma_difference == pytest.approx(2**0.5, rel=1e-12)
    assert independent.half_width == pytest.approx(1.96 * 2**0.5 / 10, rel=1e-12)
    assert correlated.sigma_difference == pytest.approx(1.0, rel=1e-12)
    assert correlated.half_width == pytest.approx(0.196, rel=1e-12)
    assert correlated.detectable == pytest.approx((1.96 + 0.8416212335729143) / 10, rel=1e-9)


def test_the_half_width_is_the_one_the_published_interval_has() -> None:
    """Con rho = -1 la diferencia de `x` y `-x` es `2x`: el intervalo de `paired_difference`
    y el semiancho de aquí tienen que ser el mismo número.

    Ata este módulo al intervalo que la ablación publica: si uno cambiara de z, el efecto
    detectable hablaría de otro intervalo.
    """
    values = [(-1) ** index * 0.01 * (1 + index % 3) for index in range(40)]

    published = paired_difference(values, [-value for value in values])
    effect = detectable_difference(statistics.stdev(values), 40, -1.0)

    assert published is not None
    assert effect.half_width == pytest.approx(published.high - published.mean, rel=1e-12)


def test_more_evaluations_or_more_correlation_shrink_the_detectable_effect() -> None:
    base = detectable_difference(0.05, 140, 0.0)

    assert detectable_difference(0.05, 560, 0.0).half_width == pytest.approx(
        base.half_width / 2, rel=1e-12
    )
    assert detectable_difference(0.05, 140, 0.5).detectable < base.detectable
    assert detectable_difference(0.05, 140, 1.0).detectable == 0.0


@pytest.mark.parametrize(
    "arguments",
    [
        {"sigma": -0.1, "n": 100, "rho": 0.0},
        {"sigma": float("nan"), "n": 100, "rho": 0.0},
        {"sigma": 0.1, "n": 1, "rho": 0.0},
        {"sigma": 0.1, "n": 100, "rho": 1.5},
        {"sigma": 0.1, "n": 100, "rho": 0.0, "power": 1.0},
        {"sigma": 0.1, "n": 100, "rho": 0.0, "power": 0.0},
    ],
)
def test_arguments_outside_their_domain_are_refused(arguments: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        detectable_difference(**arguments)  # type: ignore[arg-type]


def test_time_scales_linearly_from_the_hours_given() -> None:
    assert hours_for(140) == DEFAULT_HOURS_PER_140 == 36.0
    assert [hours_for(n) for n in DEFAULT_SAMPLE_SIZES] == [36.0, 72.0, 108.0]
    assert hours_for(280, 10.0) == 20.0


# ──────────────────────────────────────── Sin medias ──────────────────────────────────────────────


def test_no_model_in_the_module_has_a_field_for_a_mean() -> None:
    """Lo que no está en el modelo no se puede imprimir por descuido."""
    models = [
        item
        for _, item in inspect.getmembers(dispersion, inspect.isclass)
        if issubclass(item, FrozenModel) and item.__module__ == dispersion.__name__
    ]
    assert ReturnDispersion in models

    banned = ("mean", "avg", "average", "expected", "total")
    offenders = [
        f"{model.__name__}.{name}"
        for model in models
        for name in model.model_fields
        if any(word in name for word in banned)
    ]
    assert offenders == []


def synthetic_report(monkeypatch: pytest.MonkeyPatch) -> tuple[PoolReport, float]:
    """El informe de la serie sintética, y la media de sus compras para buscarla en el texto.

    El barrido se sustituye por las activaciones escritas a mano: lo que se prueba es el
    reporte, y el gate ya tiene sus propias pruebas.
    """
    rows = zigzag_rows()
    indices = range(50, 60)
    bars = tuple(bars_for(rows, indices))
    monkeypatch.setattr(dispersion, "activated_bars", lambda *args, **kwargs: bars)
    report = build_report({SYMBOL: rows}, HORIZON, preset=PRESET, config=CONFIG)
    mean = statistics.fmean((rows[i + HORIZON][4] - rows[i][4]) / rows[i][4] for i in indices)
    return report, mean


def test_the_report_does_not_print_the_mean_return(monkeypatch: pytest.MonkeyPatch) -> None:
    """La media de las compras sintéticas es 3.8%: no aparece en ninguna de sus formas."""
    report, mean = synthetic_report(monkeypatch)

    text = render_report(report)

    assert 0.03 < mean < 0.05
    for rendering in (f"{mean:.2%}", f"{mean:.4f}", f"{mean:.1%}", f"{mean * 100:.2f}"):
        assert rendering not in text
    assert "desviación estándar" in text


def test_the_report_cites_the_digest_of_every_series(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = zigzag_rows()
    report, _ = synthetic_report(monkeypatch)

    text = render_report(report)

    assert candles_digest(to_dataframe(rows)) in text
    assert SYMBOL in text


def test_the_report_proposes_no_equivalence_margin(monkeypatch: pytest.MonkeyPatch) -> None:
    """El margen lo fija quien lee la tabla, no la tabla."""
    report, _ = synthetic_report(monkeypatch)

    text = render_report(report).lower()

    assert "margen" not in text
    assert "δ" not in text
    assert "recomend" not in text


def test_the_report_has_a_row_for_every_size_and_correlation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report, _ = synthetic_report(monkeypatch)

    text = render_report(report)

    for n in DEFAULT_SAMPLE_SIZES:
        assert f"| {n} | 0.0 |" in text
        assert f"| {n} | 0.5 |" in text


def hand_made_report(buy: float | None, sell: float | None) -> PoolReport:
    """Un informe con las dos desviaciones globales escritas a mano."""
    return PoolReport(
        horizon=HORIZON,
        series=(
            SeriesProvenance(
                symbol=SYMBOL,
                bars=2,
                first=datetime(2026, 8, 1, tzinfo=UTC),
                last=datetime(2026, 8, 2, tzinfo=UTC),
                digest="0" * 64,
            ),
        ),
        activations=10,
        unconfirmed=0,
        dispersions=(
            ReturnDispersion(
                label="global", side=Action.BUY, resolved=10, unresolved=0, invalidated=0, stdev=buy
            ),
            ReturnDispersion(
                label="global",
                side=Action.SELL,
                resolved=10,
                unresolved=0,
                invalidated=0,
                stdev=sell,
            ),
        ),
    )


def test_the_effect_is_computed_from_the_larger_global_deviation() -> None:
    """Un brazo que a veces hace hold tiene menos dispersión que cualquiera de las dos líneas
    base, así que se usa la mayor: la cota alta, que no promete más potencia de la que hay."""
    text = render_report(hand_made_report(buy=0.02, sell=0.05))

    larger = detectable_difference(0.05, 140, 0.0)
    smaller = detectable_difference(0.02, 140, 0.0)
    assert f"{larger.half_width:.2%}" in text
    assert f"{smaller.half_width:.2%}" not in text


def test_a_report_without_two_resolved_positions_says_so() -> None:
    """Sin desviación no hay efecto detectable, y el reporte no lo inventa."""
    empty = ReturnDispersion(
        label="global", side=Action.BUY, resolved=1, unresolved=0, invalidated=0, stdev=None
    )
    report = PoolReport(
        horizon=HORIZON,
        series=(
            SeriesProvenance(
                symbol=SYMBOL,
                bars=2,
                first=datetime(2026, 8, 1, tzinfo=UTC),
                last=datetime(2026, 8, 2, tzinfo=UTC),
                digest="0" * 64,
            ),
        ),
        activations=1,
        unconfirmed=0,
        dispersions=(empty,),
    )

    text = render_report(report)

    assert "No determinado" in text
    assert "| 140 |" not in text.split("## Efecto detectable")[1].split("## Tiempo")[0]


# ──────────────────────────────────────────── Comando ─────────────────────────────────────────────


def test_the_command_is_reproducible_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Dos ejecuciones dan el mismo texto, con el digest de la serie, y no dejan ningún archivo."""
    rows = real_rows()
    history = tmp_path / "history"
    history.mkdir()
    write_ohlcv_csv(history / "btcusdt_4h.csv", rows)
    workdir = tmp_path / "cwd"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    before = sorted(path.name for path in tmp_path.rglob("*"))
    arguments = ["--history-dir", str(history), "--symbols", "BTC/USDT"]

    assert main(arguments) == 0
    first = capsys.readouterr().out
    assert main(arguments) == 0
    second = capsys.readouterr().out

    assert first == second
    assert candles_digest(to_dataframe(rows)) in first
    assert sorted(path.name for path in tmp_path.rglob("*")) == before


def test_a_missing_series_is_named_and_nothing_is_downloaded(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["--history-dir", str(tmp_path), "--symbols", "BTC/USDT"])

    captured = capsys.readouterr()
    assert code == 1
    assert "btcusdt_4h.csv" in captured.err
    assert captured.out == ""
