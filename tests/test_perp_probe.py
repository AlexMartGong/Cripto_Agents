"""Pruebas del sondeo público de perpetuos. Ninguna toca la red.

Dos orígenes de datos, y se distinguen a propósito:

- `tests/data/perp/*.json`: respuestas **grabadas** de los dos exchanges (mercados, tickers y una
  página de funding de BTC). Dan la forma real de lo que devuelve ccxt.
- Lo construido en línea: series con intervalo de 4 h o de 1 h, huecos, páginas llenas, cabeceras
  y códigos de error. Ningún exchange tenía hoy un tramo así, y son **sintéticos** por eso.

El corazón es `normalise_24h`: la prueba lleva el esperado calculado a mano y otra que verifica
que un normalizador con el intervalo fijo en 8 h daría una cifra distinta, que es lo que haría
fallar la mutación.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, cast

import ccxt
import numpy as np
import pytest

from crypto_agents import perp_probe
from crypto_agents.activation_sweep import DEFAULT_SYMBOLS
from crypto_agents.perp_probe import (
    CAPABILITIES,
    EXCHANGES,
    FUNDING_PAGE_LIMIT,
    MAX_REQUESTS_PER_SERIES,
    UNIVERSE_BASES,
    Absent,
    ContractSpec,
    EndpointLog,
    Environment,
    FundingRow,
    ProbeError,
    describe_market,
    fetch_funding_series,
    instrument,
    interval_hours,
    normalise_24h,
    parse_funding,
    perp_symbol,
    render_report,
    run_probe,
    series_csv,
    series_digest,
    summarise_funding,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence

FIXTURES = Path(__file__).parent / "data" / "perp"
FIXTURE_DIGESTS = {
    "binanceusdm.json": "638e36b5fceab8b86ba11ac35c7e1a8868863924fbd0273ac378e89c0f88d3b7",
    "bybit.json": "a8e3cacfe710f17fc711a01fd68919f76388e701aaa8e8c8e7e5d9f1a3a90bb2",
}
HOUR = 3_600_000
DAY = 24 * HOUR
SENTINEL_RESPONSE_HEADER = "SENTINEL-RESPONSE-HEADER"
SENTINEL_REQUEST_HEADER = "SENTINEL-REQUEST-HEADER"


# ─────────────────────────────── Fixtures grabados ───────────────────────────────────────


@dataclass(frozen=True)
class Recorded:
    """Lo capturado de un exchange, en la forma en que ccxt lo devolvió."""

    exchange_id: str
    precision_mode: int
    markets: dict[str, dict[str, object]]
    tickers: dict[str, dict[str, object]]
    funding_btc: list[dict[str, object]]
    since_ms: int
    until_ms: int


def recorded(exchange_id: str) -> Recorded:
    payload = cast(
        "dict[str, object]", json.loads((FIXTURES / f"{exchange_id}.json").read_text("utf-8"))
    )
    window = cast("dict[str, str]", payload["window"])
    return Recorded(
        exchange_id=exchange_id,
        precision_mode=cast("int", payload["precisionMode"]),
        markets=cast("dict[str, dict[str, object]]", payload["markets"]),
        tickers=cast("dict[str, dict[str, object]]", payload["tickers"]),
        funding_btc=cast("list[dict[str, object]]", payload["funding_btc"]),
        since_ms=int(datetime.fromisoformat(window["since"]).timestamp() * 1000),
        until_ms=int(datetime.fromisoformat(window["until"]).timestamp() * 1000),
    )


def raw_row(symbol: str, timestamp: int, rate: float) -> dict[str, object]:
    """Una fila de funding con la forma que ccxt entrega."""
    return {"symbol": symbol, "fundingRate": rate, "timestamp": timestamp, "info": {}}


def rows_of(points: Sequence[tuple[int, float]]) -> list[FundingRow]:
    return [FundingRow(timestamp_ms=ts, rate=rate) for ts, rate in points]


def test_the_recorded_fixtures_are_the_ones_that_were_captured() -> None:
    """Editar un fixture a mano lo haría incomparable con la captura sin que nada avise."""
    for name, digest in FIXTURE_DIGESTS.items():
        assert hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest() == digest


def test_the_universe_matches_the_symbols_the_rest_of_the_project_evaluates() -> None:
    """El módulo no importa `activation_sweep`; este test es lo que impide que diverjan."""
    assert tuple(f"{base}/USDT" for base in UNIVERSE_BASES) == DEFAULT_SYMBOLS


def test_the_tick_size_constant_is_ccxts() -> None:
    assert perp_probe.TICK_SIZE == ccxt.TICK_SIZE


# ─────────────────────────────── Mercados: qué existe ────────────────────────────────────


@pytest.mark.parametrize("exchange_id", EXCHANGES)
def test_every_symbol_of_the_universe_is_described_from_the_recorded_markets(
    exchange_id: str,
) -> None:
    rec = recorded(exchange_id)
    for base in UNIVERSE_BASES:
        spec = describe_market(base, rec.markets, rec.precision_mode)
        market = rec.markets[perp_symbol(base)]
        limits = cast("dict[str, dict[str, object]]", market["limits"])
        precision = cast("dict[str, object]", market["precision"])
        assert isinstance(spec, ContractSpec)
        assert spec.symbol == f"{base}/USDT:USDT"
        assert spec.contract_size == market["contractSize"]
        assert spec.tick_size == precision["price"]
        assert spec.min_amount == limits["amount"]["min"]
        assert spec.min_cost == limits["cost"]["min"]


def test_btc_figures_are_pinned_by_hand_for_both_exchanges() -> None:
    """Cifras escritas a mano, no leídas del mismo diccionario que el código bajo prueba."""
    binance = describe_market("BTC", recorded("binanceusdm").markets, ccxt.TICK_SIZE)
    bybit = describe_market("BTC", recorded("bybit").markets, ccxt.TICK_SIZE)
    assert binance == ContractSpec("BTC/USDT:USDT", 1.0, 0.1, 0.001, 50.0)
    # Bybit deja `limits.cost.min` en None: el mínimo nominal no está donde se pregunta.
    assert bybit == ContractSpec("BTC/USDT:USDT", 1.0, 0.1, 0.001, None)


def test_a_symbol_missing_from_the_catalogue_does_not_exist() -> None:
    markets = dict(recorded("bybit").markets)
    del markets["ADA/USDT:USDT"]
    result = describe_market("ADA", markets, ccxt.TICK_SIZE)
    assert isinstance(result, Absent)
    assert result.symbol == "ADA/USDT:USDT"
    assert "ausente" in result.reason


def test_the_usdc_perpetual_and_the_dated_future_are_not_the_usdt_swap() -> None:
    """Los dos señuelos son reales: existen en el catálogo y no son lo que se busca."""
    rec = recorded("binanceusdm")
    decoys = {s: m for s, m in rec.markets.items() if s == "BTC/USDC:USDC" or "-" in s}
    assert len(decoys) == 2
    assert isinstance(describe_market("BTC", decoys, ccxt.TICK_SIZE), Absent)

    # Aunque estén bajo la clave que se busca, los campos mandan y no la clave.
    usdc = decoys["BTC/USDC:USDC"]
    disguised = {"BTC/USDT:USDT": usdc}
    result = describe_market("BTC", disguised, ccxt.TICK_SIZE)
    assert isinstance(result, Absent)
    assert "USDT" in result.reason


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"swap": False, "future": True}, "swap"),
        ({"linear": False, "inverse": True}, "lineal"),
        ({"active": False}, "inactivo"),
    ],
)
def test_a_market_that_is_not_an_active_linear_swap_does_not_exist(
    change: dict[str, object], reason: str
) -> None:
    market = dict(recorded("bybit").markets["BTC/USDT:USDT"], **change)
    result = describe_market("BTC", {"BTC/USDT:USDT": market}, ccxt.TICK_SIZE)
    assert isinstance(result, Absent)
    assert reason in result.reason


def test_the_tick_is_unknown_when_the_precision_mode_is_not_tick_size() -> None:
    """Con otro modo `precision.price` son decimales o cifras, no un tick."""
    spec = describe_market("BTC", recorded("bybit").markets, ccxt.DECIMAL_PLACES)
    assert isinstance(spec, ContractSpec)
    assert spec.tick_size is None
    assert spec.contract_size == 1.0


# ─────────────────────────── Funding: filas y normalización ──────────────────────────────


def test_funding_rows_are_parsed_from_the_recorded_page() -> None:
    rec = recorded("binanceusdm")
    rows = parse_funding(rec.funding_btc)
    assert len(rows) == 25
    assert rows[0] == FundingRow(1756684800000, 4.66e-05)
    assert [r.timestamp_ms for r in rows] == sorted(r.timestamp_ms for r in rows)


@pytest.mark.parametrize(
    "broken", [{"fundingRate": None}, {"timestamp": None}, {"fundingRate": "x"}]
)
def test_a_funding_row_without_a_number_is_an_error_not_a_zero(broken: dict[str, object]) -> None:
    with pytest.raises(ProbeError):
        parse_funding([dict(raw_row("BTC/USDT:USDT", 0, 0.0001), **broken)])


@pytest.mark.parametrize(
    ("gap_ms", "expected"),
    [
        (8 * HOUR, 8),
        (8 * HOUR + 5, 8),  # el jitter de milisegundos de binance
        (8 * HOUR - 1, 8),
        (4 * HOUR, 4),
        (2 * HOUR, 2),
        (HOUR, 1),
        (7 * HOUR, None),
        (16 * HOUR, None),  # un registro faltante parece un intervalo de 16 h
        (8 * HOUR + 120_000, None),  # media hora fuera de rejilla no se redondea a 8
        (0, None),
        (-8 * HOUR, None),
    ],
)
def test_the_interval_is_inferred_from_the_gap_and_only_if_it_is_on_the_grid(
    gap_ms: int, expected: int | None
) -> None:
    assert interval_hours(1_000_000, 1_000_000 + gap_ms) == expected


MIXED = rows_of(
    [
        (0, 0.0001),  # primera fila: no tiene periodo anterior
        (8 * HOUR + 5, 0.0003),  # 8 h (con jitter)
        (16 * HOUR, -0.0001),  # 8 h
        (20 * HOUR, 0.0001),  # 4 h
        (24 * HOUR + 2, 0.0002),  # 4 h
        (25 * HOUR, 0.00005),  # 1 h
        (41 * HOUR, 0.0004),  # 16 h: fuera de rejilla, no atribuible
        (49 * HOUR, 0.0001),  # 8 h
    ]
)
MIXED_24H = [0.0009, -0.0003, 0.0006, 0.0012, 0.0012, 0.0003]
"""Esperado a mano: tasa * 24 / intervalo. 0.0003*3, -0.0001*3, 0.0001*6, 0.0002*6, 0.00005*24,
0.0001*3. La fila de 16 h y la primera no entran."""


def test_each_period_is_normalised_by_its_own_interval() -> None:
    result = normalise_24h(MIXED)
    assert list(result.values) == pytest.approx(MIXED_24H)
    assert result.unattributable == 2
    assert dict(result.intervals) == {8: 3, 4: 2, 1: 1}


def test_a_normaliser_with_the_interval_fixed_at_eight_hours_would_be_caught() -> None:
    """La mutación que pide el criterio: con 8 h fijo, el esperado a mano no coincide."""
    wrong = [cur.rate * 24 / 8 for _, cur in pairwise(MIXED)]
    assert wrong != pytest.approx(MIXED_24H)
    assert wrong[2] == pytest.approx(0.0003)  # el periodo de 4 h valdría la mitad de lo cierto
    assert MIXED_24H[2] == pytest.approx(0.0006)


def test_the_real_series_normalises_by_eight_hours_and_drops_only_the_first_row() -> None:
    rows = parse_funding(recorded("binanceusdm").funding_btc)
    result = normalise_24h(rows)
    assert result.unattributable == 1
    assert dict(result.intervals) == {8: 24}
    assert list(result.values) == pytest.approx([row.rate * 3 for row in rows[1:]])


def test_an_empty_or_single_row_series_has_nothing_to_normalise() -> None:
    assert normalise_24h([]).values == ()
    assert normalise_24h([]).unattributable == 0
    single = normalise_24h(rows_of([(0, 0.0001)]))
    assert single.values == ()
    assert single.unattributable == 1


# ─────────────────────────────────── Resumen ─────────────────────────────────────────────


def test_the_summary_matches_the_hand_computed_figures() -> None:
    summary = summarise_funding(MIXED)
    assert summary.periods == 8
    assert summary.normalised == 6
    assert summary.unattributable == 2
    assert summary.negative == 1  # solo -0.0001; el signo no depende del intervalo
    assert summary.mean_24h == pytest.approx(0.00065)
    assert summary.median_24h == pytest.approx(0.00075)
    # posiciones (n-1)·0.95 = 4.75 entre 0.0012 y 0.0012
    assert summary.p95_24h == pytest.approx(0.0012)


def test_the_p95_interpolates_linearly_like_numpy() -> None:
    rng = np.random.default_rng(7)
    rates = rng.normal(0.0001, 0.0002, size=200)
    rows = rows_of([(i * 8 * HOUR, float(rate)) for i, rate in enumerate(rates)])
    summary = summarise_funding(rows)
    values = [float(r) * 3 for r in rates[1:]]
    assert summary.p95_24h == pytest.approx(float(np.percentile(values, 95)))
    assert summary.median_24h == pytest.approx(statistics.median(values))
    assert summary.mean_24h == pytest.approx(statistics.mean(values))


def test_a_zero_rate_is_not_negative() -> None:
    rows = rows_of([(0, 0.0), (8 * HOUR, 0.0), (16 * HOUR, -1e-9)])
    assert summarise_funding(rows).negative == 1


def test_a_summary_with_too_little_data_says_so_instead_of_inventing_a_figure() -> None:
    assert summarise_funding([]).mean_24h is None
    one = summarise_funding(rows_of([(0, 0.0001), (8 * HOUR, 0.0002)]))
    assert one.mean_24h == pytest.approx(0.0006)
    assert one.p95_24h is None  # un solo periodo no define un percentil


# ───────────────────────────── Serie guardada y digest ───────────────────────────────────


def test_the_csv_is_canonical_and_its_digest_is_the_sha256_of_those_bytes() -> None:
    rows = rows_of([(8 * HOUR, 0.0003), (0, 0.0001)])  # desordenadas a propósito
    content = series_csv(rows)
    assert content == b"timestamp_ms,funding_rate\n0,0.0001\n28800000,0.0003\n"
    assert series_digest(content) == hashlib.sha256(content).hexdigest()
    assert series_csv(list(reversed(rows))) == content


def test_a_float_survives_the_csv_round_trip_bit_for_bit() -> None:
    rows = parse_funding(recorded("binanceusdm").funding_btc)
    lines = series_csv(rows).decode().splitlines()[1:]
    assert [float(line.split(",")[1]) for line in lines] == [r.rate for r in rows]


# ─────────────────────────────────── Paginación ──────────────────────────────────────────


class FundingOnly:
    """Un exchange que solo sabe contestar `fetch_funding_rate_history`, con la semántica elegida.

    - `newest_first`: lo que hace Bybit con `until` — las más recientes de la ventana.
    - `upper_exclusive`: el extremo superior no entra.
    - `always_full`: ignora los límites y devuelve siempre una página llena.
    """

    def __init__(
        self,
        timestamps: Sequence[int],
        *,
        newest_first: bool = False,
        upper_exclusive: bool = False,
        always_full: bool = False,
    ) -> None:
        self.timestamps = sorted(timestamps)
        self.newest_first = newest_first
        self.upper_exclusive = upper_exclusive
        self.always_full = always_full
        self.calls = 0

    async def fetch_funding_rate_history(
        self,
        symbol: str,
        since: int | None = None,
        limit: int | None = None,
        params: dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        self.calls += 1
        assert since is not None
        assert limit is not None
        assert params is not None
        until = cast("int", params["until"])
        if self.always_full:
            return [raw_row(symbol, since + i, 0.0) for i in range(limit)]
        low = bisect_left(self.timestamps, since)
        high = (bisect_left if self.upper_exclusive else bisect_right)(self.timestamps, until)
        if self.newest_first:
            chosen = self.timestamps[max(low, high - limit) : high]
        else:
            chosen = self.timestamps[low : min(high, low + limit)]
        return [raw_row(symbol, t, (t // HOUR) * 1e-8) for t in chosen]


def regular(start: int, count: int, interval_hours: int = 8) -> list[int]:
    return [start + i * interval_hours * HOUR for i in range(count)]


SEMANTICS = [
    pytest.param({}, id="oldest-first-closed"),
    pytest.param({"newest_first": True}, id="newest-first-closed"),
    pytest.param({"upper_exclusive": True}, id="oldest-first-half-open"),
    pytest.param({"newest_first": True, "upper_exclusive": True}, id="newest-first-half-open"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("semantics", SEMANTICS)
@pytest.mark.parametrize("interval_hours", [8, 4, 1])
async def test_the_whole_series_comes_back_whatever_the_exchange_does_with_its_bounds(
    semantics: dict[str, bool], interval_hours: int
) -> None:
    """Paginar hacia delante contra un exchange que da lo más reciente se saltaría datos."""
    count = 20 * 24 // interval_hours  # 20 días
    stamps = regular(0, count, interval_hours)
    exchange = FundingOnly(stamps, **semantics)
    # `end` pasa un milisegundo de la última fila: con extremo superior exclusivo, la que cae
    # justo en `end` no la devuelve ninguna ventana, ni de este diseño ni de otro.
    rows = await fetch_funding_series(exchange, "BTC/USDT:USDT", 0, stamps[-1] + 1, page_limit=10)
    assert [r.timestamp_ms for r in rows] == stamps
    assert [r.rate for r in rows] == [(t // HOUR) * 1e-8 for t in stamps]


@pytest.mark.asyncio
async def test_a_symbol_listed_in_the_middle_of_the_window_is_not_cut_short() -> None:
    """Las ventanas anteriores al listado vuelven vacías, y eso no es el final de la serie."""
    stamps = regular(40 * DAY, 90)
    exchange = FundingOnly(stamps)
    rows = await fetch_funding_series(exchange, "X/USDT:USDT", 0, 100 * DAY, page_limit=10)
    assert [r.timestamp_ms for r in rows] == stamps


@pytest.mark.asyncio
async def test_an_exchange_with_no_funding_gives_an_empty_series() -> None:
    rows = await fetch_funding_series(FundingOnly([]), "X/USDT:USDT", 0, 30 * DAY, page_limit=10)
    assert rows == []


@pytest.mark.asyncio
async def test_a_full_page_is_never_accepted_as_complete() -> None:
    """Una página con exactamente `limit` filas puede estar truncada: se parte la ventana."""
    stamps = regular(0, 40, interval_hours=1)  # 40 filas en una sola ventana inicial
    exchange = FundingOnly(stamps)
    rows = await fetch_funding_series(exchange, "X/USDT:USDT", 0, stamps[-1] + 1, page_limit=40)
    assert len(rows) == 40
    assert exchange.calls > 1


@pytest.mark.asyncio
async def test_a_regular_series_does_not_split_needlessly() -> None:
    """Medido contra Bybit: una ventana de `limit` periodos cerrada tiene `limit + 1` filas."""
    stamps = regular(0, 2190)  # dos años a 8 h
    exchange = FundingOnly(stamps)
    rows = await fetch_funding_series(exchange, "X/USDT:USDT", 0, stamps[-1] + 1, page_limit=200)
    assert len(rows) == 2190
    assert exchange.calls <= 12  # 2190 periodos entre ventanas de 198: once, sin una sola partida


@pytest.mark.asyncio
async def test_a_series_that_needs_endless_splitting_fails_loudly_and_cheaply() -> None:
    """Sin tope de peticiones, partir sin fin sería un martilleo al exchange."""
    dense = [i * 60_000 for i in range(30 * 24 * 60)]  # una fila por minuto durante 30 días
    exchange = FundingOnly(dense)
    with pytest.raises(ProbeError, match="peticiones"):
        await fetch_funding_series(exchange, "X/USDT:USDT", 0, 30 * DAY, page_limit=10)
    assert exchange.calls <= MAX_REQUESTS_PER_SERIES


@pytest.mark.asyncio
async def test_a_window_too_small_to_split_is_reported() -> None:
    exchange = FundingOnly([], always_full=True)
    with pytest.raises(ProbeError, match="ventana"):
        await fetch_funding_series(exchange, "X/USDT:USDT", 0, 30_000, page_limit=10)


@pytest.mark.asyncio
@pytest.mark.parametrize("exchange_id", EXCHANGES)
async def test_the_declared_page_limits_hold_the_recorded_window_in_one_request(
    exchange_id: str,
) -> None:
    rec = recorded(exchange_id)
    stamps = [cast("int", row["timestamp"]) for row in rec.funding_btc]
    exchange = FundingOnly(stamps)
    rows = await fetch_funding_series(
        exchange, "BTC/USDT:USDT", rec.since_ms, rec.until_ms, FUNDING_PAGE_LIMIT[exchange_id]
    )
    assert [r.timestamp_ms for r in rows] == stamps
    assert exchange.calls == 1


# ─────────────────────────────── Conectividad ────────────────────────────────────────────


class FakeHttpError(Exception):
    """Lo que ccxt levanta ante un 4xx o 5xx: el mensaje lleva el status, no las cabeceras."""


class Probe:
    """Un exchange mínimo para `instrument`: `fetch` y el gancho que ccxt llama por respuesta."""

    fetch: Callable[..., Awaitable[object]]
    on_rest_response: Callable[..., object]

    def __init__(self, behaviour: Callable[[Probe, str], object]) -> None:
        self.behaviour = behaviour
        self.fetch = self._fetch
        self.on_rest_response = self._hook
        self.hook_returns: list[object] = []

    def _hook(self, *args: object) -> object:
        self.hook_returns.append(args[5])
        return args[5]

    async def _fetch(
        self, url: str, method: str = "GET", headers: object = None, body: object = None
    ) -> object:
        return self.behaviour(self, url)


def respond(exchange: Probe, url: str, status: int, *, raises: bool = False) -> object:
    exchange.on_rest_response(
        status,
        "reason",
        url,
        "GET",
        {"X-Leak": SENTINEL_RESPONSE_HEADER},
        "{}",
        {"X-Leak": SENTINEL_REQUEST_HEADER},
        None,
    )
    if raises:
        raise FakeHttpError(f"probe GET {url} {status}")
    return {"ok": True}


def stepping_timer(step: float = 0.25) -> Callable[[], float]:
    now = [0.0]

    def timer() -> float:
        now[0] += step
        return now[0]

    return timer


@pytest.mark.asyncio
async def test_a_successful_request_is_logged_with_status_latency_and_no_query() -> None:
    exchange = Probe(lambda e, url: respond(e, url, 200))
    log = EndpointLog()
    instrument(exchange, log, timer=stepping_timer(0.25))
    result = await exchange.fetch(
        "https://fapi.example.com/fapi/v1/fundingRate?symbol=BTCUSDT&signature=abc123"
    )
    assert result == {"ok": True}
    (call,) = log.calls
    assert (call.method, call.host, call.path) == (
        "GET",
        "fapi.example.com",
        "/fapi/v1/fundingRate",
    )
    assert call.status == 200
    assert call.error is None
    assert call.latency_ms == pytest.approx(250.0)
    assert "signature" not in repr(call)


@pytest.mark.asyncio
async def test_a_rejected_request_keeps_its_status_and_still_raises() -> None:
    exchange = Probe(lambda e, url: respond(e, url, 451, raises=True))
    log = EndpointLog()
    instrument(exchange, log, timer=stepping_timer())
    with pytest.raises(FakeHttpError):
        await exchange.fetch("https://fapi.example.com/x")
    (call,) = log.calls
    assert call.status == 451
    assert call.error == "FakeHttpError"


@pytest.mark.asyncio
async def test_a_request_with_no_response_has_no_status_and_no_stale_one() -> None:
    """Un timeout justo después de un 451 no hereda el 451."""

    def behaviour(exchange: Probe, url: str) -> object:
        if "slow" in url:
            raise TimeoutError
        return respond(exchange, url, 451, raises=True)

    exchange = Probe(behaviour)
    log = EndpointLog()
    instrument(exchange, log, timer=stepping_timer())
    fetch = exchange.fetch
    with pytest.raises(FakeHttpError):
        await fetch("https://h.example.com/first")
    with pytest.raises(TimeoutError):
        await fetch("https://h.example.com/slow")
    assert [(c.status, c.error) for c in log.calls] == [
        (451, "FakeHttpError"),
        (None, "TimeoutError"),
    ]


@pytest.mark.asyncio
async def test_the_original_response_hook_still_sees_everything() -> None:
    exchange = Probe(lambda e, url: respond(e, url, 200))
    instrument(exchange, EndpointLog(), timer=stepping_timer())
    await exchange.fetch("https://h.example.com/x")
    assert exchange.hook_returns == ["{}"]


@pytest.mark.asyncio
async def test_no_header_ever_reaches_the_log() -> None:
    exchange = Probe(lambda e, url: respond(e, url, 200))
    log = EndpointLog()
    instrument(exchange, log, timer=stepping_timer())
    await exchange.fetch(
        "https://h.example.com/x", "GET", {"X-Leak": SENTINEL_REQUEST_HEADER}, None
    )
    dump = repr(log.calls)
    assert SENTINEL_REQUEST_HEADER not in dump
    assert SENTINEL_RESPONSE_HEADER not in dump


# ───────────────────────── El sondeo entero, con un exchange falso ───────────────────────────


class FakeExchange:
    """La superficie de ccxt que usa el módulo, con la captura grabada detrás.

    Cada método pasa por `self.fetch` como lo haría ccxt, así que `instrument` mide lo mismo que
    mediría en producción. El fixture solo trae funding de BTC: los demás símbolos reciben una
    serie sintética de 8 h, rotulada así porque no es una grabación.
    """

    fetch: Callable[..., Awaitable[object]]
    on_rest_response: Callable[..., object]

    def __init__(
        self,
        exchange_id: str,
        rec: Recorded,
        *,
        has: dict[str, object] | None = None,
        outcomes: Mapping[str, int | Exception] | None = None,
        markets: dict[str, dict[str, object]] | None = None,
        on_load_markets: Callable[[], None] | None = None,
    ) -> None:
        self.id = exchange_id
        self.rec = rec
        self.has: dict[str, object] = has if has is not None else {}
        self.precisionMode = rec.precision_mode
        self.outcomes = dict(outcomes or {})
        self.markets = markets if markets is not None else rec.markets
        self.on_load_markets = on_load_markets
        self.urls: list[str] = []
        self.history_calls: list[str] = []
        self.closed = False
        self.fetch = self._fetch
        self.on_rest_response = self._hook

    def _hook(self, *args: object) -> object:
        return args[5]

    async def _fetch(
        self, url: str, method: str = "GET", headers: object = None, body: object = None
    ) -> object:
        self.urls.append(url)
        for needle, outcome in self.outcomes.items():
            if needle in url:
                if isinstance(outcome, int):
                    self.on_rest_response(
                        outcome,
                        "reason",
                        url,
                        method,
                        {"X-Leak": SENTINEL_RESPONSE_HEADER},
                        "",
                        {"X-Leak": SENTINEL_REQUEST_HEADER},
                        None,
                    )
                    raise FakeHttpError(f"{self.id} {method} {url} {outcome}")
                raise outcome
        self.on_rest_response(
            200,
            "OK",
            url,
            method,
            {"X-Leak": SENTINEL_RESPONSE_HEADER},
            "{}",
            {"X-Leak": SENTINEL_REQUEST_HEADER},
            None,
        )
        return {}

    async def load_markets(self) -> dict[str, dict[str, object]]:
        if self.on_load_markets is not None:
            self.on_load_markets()
        await self.fetch(f"https://{self.id}.example.com/api/markets")
        return self.markets

    async def fetch_ticker(self, symbol: str) -> dict[str, object]:
        await self.fetch(f"https://{self.id}.example.com/api/ticker?symbol={symbol}")
        return self.rec.tickers[symbol]

    async def fetch_funding_rate_history(
        self,
        symbol: str,
        since: int | None = None,
        limit: int | None = None,
        params: dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        self.history_calls.append(symbol)
        await self.fetch(f"https://{self.id}.example.com/api/fundingRate?symbol={symbol}")
        assert since is not None
        assert limit is not None
        assert params is not None
        until = cast("int", params["until"])
        if symbol == "BTC/USDT:USDT":
            rows = self.rec.funding_btc
        else:
            rows = [
                raw_row(symbol, t, 0.0001)
                for t in regular(self.rec.since_ms, 25)  # sintético: 8 h, 25 filas
            ]
        return [r for r in rows if since <= cast("int", r["timestamp"]) <= until][:limit]

    async def close(self) -> None:
        self.closed = True


FIXED_END = datetime(2025, 9, 9, tzinfo=UTC)  # = `until` de los fixtures
PROBE_DAYS = 8  # → `since` de los fixtures: la ventana pedida es exactamente la grabada

HAS_BY_EXCHANGE: dict[str, dict[str, object]] = {
    "binanceusdm": {
        "fetchFundingRateHistory": True,
        "setLeverage": True,
        "setMarginMode": True,
        "createOrder": True,
        "createReduceOnlyOrder": True,
        "createStopOrder": "emulated",
        "createStopLossOrder": True,
        "createTakeProfitOrder": True,
        "createTriggerOrder": None,
    },
    "bybit": {
        "fetchFundingRateHistory": True,
        "setLeverage": True,
        "setMarginMode": None,
        "createOrder": True,
        "createReduceOnlyOrder": True,
        "createStopOrder": True,
        "createStopLossOrder": True,
        "createTakeProfitOrder": True,
        "createTriggerOrder": True,
    },
}


def make_env(
    *,
    outcomes: Mapping[str, int | Exception] | None = None,
    markets: dict[str, dict[str, object]] | None = None,
    on_load_markets: Callable[[], None] | None = None,
) -> tuple[Environment, dict[str, FakeExchange]]:
    """Un entorno determinista; el diccionario devuelto se llena con lo que la fábrica crea."""
    created: dict[str, FakeExchange] = {}

    def factory(exchange_id: str) -> FakeExchange:
        created[exchange_id] = FakeExchange(
            exchange_id,
            recorded(exchange_id),
            has=HAS_BY_EXCHANGE[exchange_id],
            outcomes=outcomes,
            markets=markets,
            on_load_markets=on_load_markets,
        )
        return created[exchange_id]

    env = Environment(
        clock=lambda: FIXED_END,
        hostname="test-host",
        git_state=lambda: ("deadbeef", False),
        exchange_factory=factory,
        timer=stepping_timer(0.01),
    )
    return env, created


def section(text: str, title: str) -> list[str]:
    """Las líneas de la sección cuyo encabezado empieza por `-- {title}`."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"-- {title}"))
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("-- ")), len(lines))
    return lines[start + 1 : end]


@pytest.mark.asyncio
@pytest.mark.parametrize("exchange_id", EXCHANGES)
async def test_a_full_probe_writes_one_series_per_symbol_and_cites_its_digest(
    tmp_path: Path, exchange_id: str
) -> None:
    env, fakes = make_env()
    report = await run_probe(exchange_id, out_dir=tmp_path, env=env, days=PROBE_DAYS)

    assert report.complete
    assert report.run_dir == tmp_path / exchange_id / "20250909T000000Z"
    assert fakes[exchange_id].closed
    assert len(report.results) == len(UNIVERSE_BASES)

    text = render_report(report)
    funding = [line for line in section(text, "Funding") if "/USDT:USDT" in line]
    assert len(funding) == len(UNIVERSE_BASES)
    for result in report.results:
        assert result.series is not None
        csv_path = report.run_dir / f"{result.base}_USDT_USDT_funding.csv"
        assert result.series.path == csv_path
        digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        assert result.series.digest == digest
        (line,) = (ln for ln in funding if ln.startswith(result.symbol))
        assert line.split()[-1] == digest[:16]
        assert digest in "\n".join(section(text, "Procedencia"))


@pytest.mark.asyncio
async def test_the_btc_funding_figures_in_the_report_are_computed_from_the_recorded_rows(
    tmp_path: Path,
) -> None:
    rec = recorded("binanceusdm")
    env, _ = make_env()
    report = await run_probe("binanceusdm", out_dir=tmp_path, env=env, days=PROBE_DAYS)
    line = next(
        ln for ln in section(render_report(report), "Funding") if ln.startswith("BTC/USDT:USDT")
    )
    rates = [cast("float", row["fundingRate"]) for row in rec.funding_btc]
    daily = [rate * 3 for rate in rates[1:]]
    negative = sum(1 for rate in rates if rate < 0)
    assert f"{statistics.mean(daily) * 100:.5f}%" in line
    assert f"{statistics.median(daily) * 100:.5f}%" in line
    assert f"{negative}/25 ({100 * negative / 25:.1f}%)" in line
    assert "8h:24" in line


@pytest.mark.asyncio
@pytest.mark.parametrize("exchange_id", EXCHANGES)
async def test_every_contract_row_cites_the_digest_of_its_own_entry(
    tmp_path: Path, exchange_id: str
) -> None:
    env, _ = make_env()
    report = await run_probe(exchange_id, out_dir=tmp_path, env=env, days=PROBE_DAYS)
    spec_file = cast(
        "dict[str, dict[str, object]]",
        json.loads((report.run_dir / "spec.json").read_text("utf-8")),
    )
    rows = [ln for ln in section(render_report(report), "Contratos") if "/USDT:USDT" in ln]
    assert len(rows) == len(UNIVERSE_BASES)
    for line in rows:
        symbol = line.split()[0]
        canonical = json.dumps(spec_file[symbol], sort_keys=True, separators=(",", ":"))
        assert line.split()[-1] == hashlib.sha256(canonical.encode()).hexdigest()[:16]


@pytest.mark.asyncio
async def test_a_missing_symbol_is_reported_as_not_existing_and_costs_no_funding_request(
    tmp_path: Path,
) -> None:
    rec = recorded("bybit")
    markets = {s: m for s, m in rec.markets.items() if s != "ADA/USDT:USDT"}
    env, fakes = make_env(markets=markets)
    report = await run_probe("bybit", out_dir=tmp_path, env=env, days=PROBE_DAYS)
    assert report.complete  # que un símbolo no exista es un hallazgo, no un fallo del sondeo
    line = next(ln for ln in section(render_report(report), "Contratos") if "ADA" in ln)
    assert "no existe" in line
    assert "ADA/USDT:USDT" not in fakes["bybit"].history_calls
    assert not (report.run_dir / "ADA_USDT_USDT_funding.csv").exists()


@pytest.mark.asyncio
async def test_the_meta_file_exists_before_the_first_request(tmp_path: Path) -> None:
    """La corrida que más hay que poder identificar es la que se interrumpe."""
    seen: list[bool] = []
    run_dir = tmp_path / "bybit" / "20250909T000000Z"
    env, _ = make_env(on_load_markets=lambda: seen.append((run_dir / "meta.json").exists()))
    await run_probe("bybit", out_dir=tmp_path, env=env, days=PROBE_DAYS)
    assert seen == [True]


@pytest.mark.asyncio
async def test_the_meta_file_records_the_run_and_nothing_sensitive(tmp_path: Path) -> None:
    env, _ = make_env()
    report = await run_probe("binanceusdm", out_dir=tmp_path, env=env, days=PROBE_DAYS)
    meta = cast("dict[str, object]", json.loads((report.run_dir / "meta.json").read_text("utf-8")))
    assert meta["hostname"] == "test-host"
    assert meta["exchange"] == "binanceusdm"
    assert meta["git_commit"] == "deadbeef"
    assert meta["git_dirty"] is False
    assert meta["days"] == PROBE_DAYS
    series = cast("dict[str, str]", meta["series_sha256"])
    assert (
        series["BTC/USDT:USDT"]
        == hashlib.sha256((report.run_dir / "BTC_USDT_USDT_funding.csv").read_bytes()).hexdigest()
    )


@pytest.mark.asyncio
async def test_a_run_directory_is_never_reused(tmp_path: Path) -> None:
    env, _ = make_env()
    await run_probe("bybit", out_dir=tmp_path, env=env, days=PROBE_DAYS)
    env_again, _ = make_env()
    with pytest.raises(FileExistsError):
        await run_probe("bybit", out_dir=tmp_path, env=env_again, days=PROBE_DAYS)


@pytest.mark.asyncio
@pytest.mark.parametrize("exchange_id", EXCHANGES)
async def test_no_header_reaches_the_report_or_any_file(tmp_path: Path, exchange_id: str) -> None:
    env, _ = make_env()
    report = await run_probe(exchange_id, out_dir=tmp_path, env=env, days=PROBE_DAYS)
    blobs = [render_report(report)] + [
        path.read_text("utf-8") for path in report.run_dir.iterdir() if path.is_file()
    ]
    for blob in blobs:
        assert SENTINEL_REQUEST_HEADER not in blob
        assert SENTINEL_RESPONSE_HEADER not in blob


@pytest.mark.asyncio
async def test_the_connectivity_section_lists_endpoints_codes_latency_and_the_machine(
    tmp_path: Path,
) -> None:
    env, _ = make_env()
    report = await run_probe("bybit", out_dir=tmp_path, env=env, days=PROBE_DAYS)
    text = render_report(report)
    assert "test-host" in text
    lines = section(text, "Conectividad")
    funding = next(ln for ln in lines if "/api/fundingRate" in ln)
    assert "200:7" in funding  # un 200 por símbolo
    assert "10.0" in funding  # cada petición dura 0.01 s con el reloj de pasos
    assert any("/api/markets" in ln and "200:1" in ln for ln in lines)


# ───────────────────────── Región, rate limit y fallos parciales ─────────────────────────────


@pytest.mark.asyncio
async def test_a_rate_limit_stops_the_whole_probe_and_says_so(tmp_path: Path) -> None:
    """Con 429 o 418 no se sigue golpeando: binance banea la IP si se insiste."""
    env, fakes = make_env(outcomes={"fundingRate?symbol=ETH": 429})
    report = await run_probe("binanceusdm", out_dir=tmp_path, env=env, days=PROBE_DAYS)
    assert not report.complete
    assert report.aborted is not None
    assert "429" in report.aborted
    assert fakes["binanceusdm"].history_calls == ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    text = render_report(report)
    assert "ABORTADO" in text
    assert "429" in text
    assert fakes["binanceusdm"].closed


@pytest.mark.asyncio
async def test_a_region_block_on_one_symbol_is_reported_and_the_others_continue(
    tmp_path: Path,
) -> None:
    env, fakes = make_env(outcomes={"fundingRate?symbol=SOL": 451})
    report = await run_probe("binanceusdm", out_dir=tmp_path, env=env, days=PROBE_DAYS)
    assert not report.complete
    assert report.aborted is None
    sol = next(r for r in report.results if r.base == "SOL")
    assert sol.series is None
    assert sol.series_error is not None
    assert "451" in sol.series_error
    assert len(fakes["binanceusdm"].history_calls) == len(
        UNIVERSE_BASES
    )  # una por símbolo, sin reintento
    line = next(ln for ln in section(render_report(report), "Funding") if ln.startswith("SOL"))
    assert "no determinado" in line
    assert "451" in line
    assert any(r.series is not None for r in report.results if r.base != "SOL")


@pytest.mark.asyncio
async def test_a_failed_ticker_loses_the_volume_but_not_the_funding(tmp_path: Path) -> None:
    env, _ = make_env(outcomes={"ticker?symbol=XRP": TimeoutError()})
    report = await run_probe("bybit", out_dir=tmp_path, env=env, days=PROBE_DAYS)
    xrp = next(r for r in report.results if r.base == "XRP")
    assert xrp.ticker is None
    assert xrp.ticker_error is not None
    assert xrp.series is not None
    assert not report.complete
    line = next(ln for ln in section(render_report(report), "Contratos") if ln.startswith("XRP"))
    assert "no determinado" in line


@pytest.mark.asyncio
async def test_a_probe_with_no_markets_reports_it_and_closes_the_session(tmp_path: Path) -> None:
    env, fakes = make_env(outcomes={"/api/markets": 451})
    report = await run_probe("bybit", out_dir=tmp_path, env=env, days=PROBE_DAYS)
    assert not report.complete
    assert report.load_error is not None
    assert "451" in report.load_error
    assert report.results == ()
    assert fakes["bybit"].closed
    text = render_report(report)
    assert "451" in text
    assert "/api/markets" in "\n".join(section(text, "Conectividad"))


# ────────────────────────────────── Capacidades ──────────────────────────────────────────


def test_the_capability_list_covers_what_the_executor_will_use() -> None:
    assert {
        "fetchFundingRateHistory",
        "setLeverage",
        "setMarginMode",
        "createOrder",
        "createReduceOnlyOrder",
        "createStopOrder",
        "createStopLossOrder",
        "createTakeProfitOrder",
        "createTriggerOrder",
    } == set(CAPABILITIES)


@pytest.mark.asyncio
async def test_capabilities_of_both_exchanges_are_printed_raw_whichever_one_runs(
    tmp_path: Path,
) -> None:
    env, fakes = make_env()
    report = await run_probe("bybit", out_dir=tmp_path, env=env, days=PROBE_DAYS)
    assert set(fakes) == set(EXCHANGES)
    assert fakes["binanceusdm"].closed  # el otro exchange no queda con una sesión abierta
    lines = section(render_report(report), "Capacidades")
    stop = next(ln for ln in lines if ln.startswith("createStopOrder"))
    assert "emulated" in stop  # el valor crudo, sin reinterpretar
    margin = next(ln for ln in lines if ln.startswith("setMarginMode"))
    assert "None" in margin
    reduce_only = next(ln for ln in lines if ln.startswith("createReduceOnlyOrder"))
    assert reduce_only.count("True") == 2
    header = "\n".join(lines)
    assert "binanceusdm" in header
    assert "bybit" in header


# ────────────────────────────────────── CLI ──────────────────────────────────────────────


def test_the_cli_rejects_an_exchange_it_does_not_know() -> None:
    with pytest.raises(SystemExit) as raised:
        perp_probe.main(["--exchange", "kraken"])
    assert raised.value.code == 2


def test_the_cli_requires_an_exchange() -> None:
    with pytest.raises(SystemExit) as raised:
        perp_probe.main([])
    assert raised.value.code == 2


def test_the_cli_prints_the_report_and_exits_zero_when_everything_answered(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env, _ = make_env()
    code = perp_probe.main(
        ["--exchange", "bybit", "--out-dir", str(tmp_path), "--days", str(PROBE_DAYS)], env=env
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "-- Conectividad" in out
    assert "-- Procedencia" in out


def test_the_cli_exits_one_when_something_failed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env, _ = make_env(outcomes={"fundingRate?symbol=SOL": 451})
    code = perp_probe.main(
        ["--exchange", "bybit", "--out-dir", str(tmp_path), "--days", str(PROBE_DAYS)], env=env
    )
    assert code == 1
    assert "451" in capsys.readouterr().out


def test_the_default_output_directory_is_ignored_by_git() -> None:
    """`var/` no se versiona: las series son salida de ejecución, no fuente."""
    assert Path("var", "perp") == perp_probe.OUT_DIR
    gitignore = (Path(__file__).parents[1] / ".gitignore").read_text("utf-8").splitlines()
    assert "var/" in gitignore
