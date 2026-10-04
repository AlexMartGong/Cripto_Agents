"""Las series de funding versionadas, y el lector que las consulta.

Dos mitades. La primera es sobre los **archivos**: que `data/funding/` sigue siendo lo que su
README dice —mismos bytes, mismo rango, misma rejilla— y que cubre el histórico de velas contra
el que se puntúa. La segunda, sobre `funding.py`: el ajuste de las marcas, la ventana
`(entrada, salida]` y lo que no es cero sino «no determinado».

Ninguna toca la red: el sondeo que bajó las series ya corrió.
"""

from __future__ import annotations

import csv
import hashlib
import re
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import pytest

from crypto_agents.activation_sweep import DEFAULT_SYMBOLS, history_path
from crypto_agents.funding import (
    FUNDING_DIR,
    SETTLEMENT_SNAP_MS,
    FundingError,
    FundingSeries,
    funding_digests,
    funding_path,
    load_funding,
    read_funding_csv,
    snap_to_settlement,
)
from crypto_agents.market import read_ohlcv_csv
from crypto_agents.perp_probe import CSV_HEADER, HOUR_MS, FundingRow, interval_hours, series_csv
from crypto_agents.selection import HISTORY_DIR

ROOT = Path(__file__).parents[1]
README = ROOT / FUNDING_DIR / "README.md"
EIGHT_HOURS = 8 * HOUR_MS


def ms(moment: datetime) -> int:
    return int(moment.timestamp() * 1000)


# ───────────────────────────────── Los archivos versionados ───────────────────────────────────


def readme_digests() -> dict[str, str]:
    """`archivo → sha-256` de la sección Digests del README, tal como la imprime `sha256sum`."""
    found = re.findall(r"^([0-9a-f]{64})  (\S+\.csv)$", README.read_text("utf-8"), re.MULTILINE)
    return {name: digest for digest, name in found}


def test_every_symbol_has_its_file_and_the_readme_digest_matches_its_bytes() -> None:
    declared = readme_digests()
    expected = {funding_path(symbol).name for symbol in DEFAULT_SYMBOLS}
    assert set(declared) == expected, "el README y el universo del proyecto no coinciden"
    for name, digest in declared.items():
        actual = hashlib.sha256((ROOT / FUNDING_DIR / name).read_bytes()).hexdigest()
        assert actual == digest, f"{name}: el archivo ya no es el que el README cita"


def test_the_directory_holds_nothing_the_readme_does_not_declare() -> None:
    files = {path.name for path in (ROOT / FUNDING_DIR).glob("*.csv")}
    assert files == set(readme_digests())


def test_the_digests_function_agrees_with_the_readme() -> None:
    declared = readme_digests()
    computed = funding_digests(DEFAULT_SYMBOLS, ROOT / FUNDING_DIR)
    assert {funding_path(symbol).name: digest for symbol, digest in computed.items()} == declared


@pytest.mark.parametrize("symbol", DEFAULT_SYMBOLS)
def test_a_series_is_what_the_probe_writes_and_loads_clean(symbol: str) -> None:
    path = ROOT / funding_path(symbol)
    assert path.read_text("utf-8").startswith(CSV_HEADER)
    loaded = read_funding_csv(path)
    assert len(loaded.times) == len(loaded.rates) == 2191


@pytest.mark.parametrize("symbol", DEFAULT_SYMBOLS)
def test_a_series_covers_the_candle_history_from_the_first_open_to_the_last_close(
    symbol: str,
) -> None:
    """Entrada, la primera apertura; salida, el cierre de la última vela: ambas cubiertas."""
    candles = read_ohlcv_csv(ROOT / history_path(symbol, "4h", HISTORY_DIR))
    bar_ms = int(candles[1][0] - candles[0][0])
    first_open, last_close = int(candles[0][0]), int(candles[-1][0]) + bar_ms
    loaded = read_funding_csv(ROOT / funding_path(symbol))

    assert loaded.times[0] <= first_open
    assert loaded.times[-1] >= last_close
    assert loaded.over(first_open + bar_ms, last_close) is not None
    assert loaded.times[0] == ms(datetime(2024, 8, 16, tzinfo=UTC))
    assert loaded.times[-1] == ms(datetime(2026, 8, 16, tzinfo=UTC))


@pytest.mark.parametrize("symbol", DEFAULT_SYMBOLS)
def test_every_gap_is_on_the_eight_hour_grid(symbol: str) -> None:
    """Sin un solo hueco ni tramo corto: el neto de cualquier ventana es determinable."""
    loaded = read_funding_csv(ROOT / funding_path(symbol))
    assert {b - a for a, b in pairwise(loaded.times)} == {EIGHT_HOURS}
    assert all(interval_hours(a, b) == 8 for a, b in pairwise(loaded.times))


def test_the_jitter_the_readme_warns_about_is_really_in_the_files() -> None:
    """Sin el ajuste, una parte de las liquidaciones caería en el lado equivocado de la entrada."""
    raw = [
        int(row[0])
        for symbol in DEFAULT_SYMBOLS
        for row in list(csv.reader((ROOT / funding_path(symbol)).read_text("utf-8").splitlines()))[
            1:
        ]
    ]
    off_grid = [t for t in raw if t % EIGHT_HOURS != 0]
    assert off_grid, (
        "si el jitter desapareciera, el README y el ajuste habrían dejado de hacer falta"
    )
    assert all(0 < t % EIGHT_HOURS < SETTLEMENT_SNAP_MS // 2 for t in off_grid)
    assert all(snap_to_settlement(t) % EIGHT_HOURS == 0 for t in raw)


# ──────────────────────────────────────── Ajuste de marcas ────────────────────────────────────


@pytest.mark.parametrize(
    ("offset_ms", "snapped"),
    [
        (0, 0),
        (13, 0),
        (26, 0),
        (-10, 0),
        (29_999, 0),
        (30_000, 60_000),
        (-29_999, 0),
        (45_000, 60_000),
    ],
)
def test_a_timestamp_snaps_to_the_nearest_minute(offset_ms: int, snapped: int) -> None:
    base = 8 * HOUR_MS
    assert snap_to_settlement(base + offset_ms) == base + snapped


def test_two_rows_in_the_same_minute_are_one_settlement_read_twice() -> None:
    rows = [FundingRow(8 * HOUR_MS, 0.0001), FundingRow(8 * HOUR_MS + 13, 0.0001)]
    with pytest.raises(FundingError, match="repetidas"):
        FundingSeries.from_rows(rows)


def test_an_unsorted_series_is_refused() -> None:
    rows = [FundingRow(16 * HOUR_MS, 0.0), FundingRow(8 * HOUR_MS, 0.0)]
    with pytest.raises(FundingError, match="desordenadas"):
        FundingSeries.from_rows(rows)


# ──────────────────────────────────────────── Ventana ─────────────────────────────────────────


def eight_hourly(rates: list[float], first_hour: int = 0) -> FundingSeries:
    return FundingSeries.from_rows(
        [FundingRow((first_hour + 8 * k) * HOUR_MS, rate) for k, rate in enumerate(rates)]
    )


def test_the_window_is_open_at_the_entry_and_closed_at_the_exit() -> None:
    funding = eight_hourly([0.1, 0.2, 0.3, 0.4])  # 0h, 8h, 16h, 24h
    assert funding.over(8 * HOUR_MS, 24 * HOUR_MS) == pytest.approx(0.3 + 0.4)
    assert funding.over(0, 16 * HOUR_MS) == pytest.approx(0.2 + 0.3)
    assert funding.over(0, 24 * HOUR_MS) == pytest.approx(0.9)


def test_a_window_with_no_settlement_inside_is_zero_when_it_is_covered() -> None:
    """Entre dos liquidaciones consecutivas no se paga nada, y eso es una cifra, no un hueco."""
    funding = eight_hourly([0.1, 0.2])
    assert funding.over(2 * HOUR_MS, 6 * HOUR_MS) == 0.0
    assert funding.over(8 * HOUR_MS, 8 * HOUR_MS) == 0.0


def test_a_window_outside_the_series_is_undetermined_not_zero() -> None:
    funding = eight_hourly([0.1, 0.2])  # 0h, 8h
    assert funding.over(-HOUR_MS, 8 * HOUR_MS) is None
    assert funding.over(0, 9 * HOUR_MS) is None
    assert FundingSeries.from_rows([]).over(0, 0) is None


def test_an_off_grid_gap_inside_the_window_is_undetermined() -> None:
    """16 h entre dos filas: falta una liquidación, y no se sabe cuánto valía."""
    funding = FundingSeries.from_rows(
        [FundingRow(0, 0.1), FundingRow(8 * HOUR_MS, 0.2), FundingRow(24 * HOUR_MS, 0.3)]
    )
    assert funding.over(0, 8 * HOUR_MS) == pytest.approx(0.2)
    assert funding.over(0, 24 * HOUR_MS) is None
    assert funding.over(9 * HOUR_MS, 24 * HOUR_MS) is None, "el hueco está en el tramo que la cubre"


def test_a_gap_outside_what_the_window_needs_does_not_matter() -> None:
    funding = FundingSeries.from_rows(
        [FundingRow(0, 0.1), FundingRow(8 * HOUR_MS, 0.2), FundingRow(40 * HOUR_MS, 0.3)]
    )
    assert funding.over(0, 8 * HOUR_MS) == pytest.approx(0.2)


def test_an_exit_inside_a_candle_pays_up_to_its_open_and_never_its_close() -> None:
    funding = eight_hourly([0.1, 0.2, 0.3, 0.4])
    open_, close = 16 * HOUR_MS, 20 * HOUR_MS
    assert funding.over(8 * HOUR_MS, open_, exit_before_ms=close) == pytest.approx(0.3)
    assert funding.over(8 * HOUR_MS, 12 * HOUR_MS, exit_before_ms=16 * HOUR_MS) == 0.0


def test_a_settlement_strictly_inside_the_exit_stretch_cannot_be_decided() -> None:
    hourly = FundingSeries.from_rows([FundingRow(h * HOUR_MS, 0.001) for h in range(0, 30)])
    assert hourly.over(0, 4 * HOUR_MS, exit_before_ms=8 * HOUR_MS) is None


def test_the_exit_stretch_needs_the_series_to_reach_its_end() -> None:
    funding = eight_hourly([0.1, 0.2])
    assert funding.over(0, 4 * HOUR_MS, exit_before_ms=12 * HOUR_MS) is None


def test_a_backwards_or_empty_window_is_a_bug() -> None:
    funding = eight_hourly([0.1, 0.2, 0.3])
    with pytest.raises(FundingError):
        funding.over(8 * HOUR_MS, 0)
    with pytest.raises(FundingError):
        funding.over(0, 8 * HOUR_MS, exit_before_ms=8 * HOUR_MS)


def test_the_sum_does_not_accumulate_rounding_error() -> None:
    funding = FundingSeries.from_rows([FundingRow(h * 8 * HOUR_MS, 0.1) for h in range(0, 11)])
    assert funding.over(0, 10 * 8 * HOUR_MS) == 1.0


# ──────────────────────────────────────────── Lectura ─────────────────────────────────────────


def test_what_the_probe_writes_is_what_the_reader_reads(tmp_path: Path) -> None:
    rows = [FundingRow(8 * HOUR_MS * k + 13 * (k % 2), 0.00012345 * (k - 2)) for k in range(5)]
    path = tmp_path / "btcusdt.csv"
    path.write_bytes(series_csv(rows))
    loaded = read_funding_csv(path)
    assert loaded.rates == tuple(row.rate for row in rows), "ni un bit se pierde"
    assert loaded.times == tuple(8 * HOUR_MS * k for k in range(5))


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("ts,rate\n0,0.1\n", "cabecera"),
        (CSV_HEADER + "0,abc\n", "ilegible"),
        (CSV_HEADER + "0\n", "ilegible"),
        (CSV_HEADER + "0,0.1,9\n", "ilegible"),
        ("", "cabecera"),
    ],
)
def test_a_malformed_series_is_refused_and_named(
    tmp_path: Path, content: str, message: str
) -> None:
    path = tmp_path / "x.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(FundingError, match=message):
        read_funding_csv(path)


def test_a_missing_file_is_an_error_not_an_empty_series(tmp_path: Path) -> None:
    with pytest.raises(FundingError, match=r"btcusdt\.csv"):
        load_funding(["BTC/USDT"], tmp_path)
    with pytest.raises(FundingError, match=r"btcusdt\.csv"):
        funding_digests(["BTC/USDT"], tmp_path)


def test_the_file_name_follows_the_history_convention() -> None:
    assert funding_path("BTC/USDT", Path("d")) == Path("d/btcusdt.csv")
    assert funding_path("DOGE/USDT", Path("d")).name == "dogeusdt.csv"
