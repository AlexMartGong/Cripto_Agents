"""Pruebas del cliente de mercado. Cero red: el origen de velas se inyecta."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pandas as pd
import pytest
from pydantic import SecretStr

from crypto_agents.market import (
    CcxtMarketClient,
    CcxtTradingClient,
    MarketDataError,
    build_snapshot,
    candles_digest,
    download_history,
    drop_forming_candle,
    load_candles,
    read_ohlcv_csv,
    timeframe_to_timedelta,
    to_dataframe,
    write_ohlcv_csv,
)
from crypto_agents.settings import ExchangeSettings
from tests.conftest import START, FakeMarketClient, candles, drifting_closes, raw_ohlcv

# ────────────────────────────────────────────── Timeframes ────────────────────────────────────────


@pytest.mark.parametrize(
    ("timeframe", "expected"),
    [
        ("1m", timedelta(minutes=1)),
        ("15m", timedelta(minutes=15)),
        ("4h", timedelta(hours=4)),
        ("1d", timedelta(days=1)),
        ("1w", timedelta(weeks=1)),
    ],
)
def test_timeframe_is_parsed(timeframe: str, expected: timedelta) -> None:
    """Los timeframes de ccxt se traducen a duración."""
    assert timeframe_to_timedelta(timeframe) == expected


@pytest.mark.parametrize("timeframe", ["", "h", "1y", "xh", "1", "1H"])
def test_invalid_timeframe_is_rejected(timeframe: str) -> None:
    """Un timeframe que no se entiende debe fallar, no adivinarse."""
    with pytest.raises(MarketDataError):
        timeframe_to_timedelta(timeframe)


# ────────────────────────────────────────────── Normalización ─────────────────────────────────────


def test_to_dataframe_builds_utc_index() -> None:
    """El índice queda en UTC y las columnas en el orden esperado."""
    frame = to_dataframe(raw_ohlcv([100.0, 101.0, 102.0]))
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert str(frame.index.tz) == "UTC"
    assert frame.index[0] == pd.Timestamp(START)
    assert frame["close"].iloc[-1] == 102.0


def test_empty_response_is_rejected() -> None:
    """Sin velas no hay nada que analizar."""
    with pytest.raises(MarketDataError, match="no devolvió velas"):
        to_dataframe([])


def test_duplicate_timestamps_are_rejected() -> None:
    """Timestamps repetidos romperían cualquier cálculo de ventana."""
    rows = raw_ohlcv([100.0, 101.0])
    rows[1][0] = rows[0][0]
    with pytest.raises(MarketDataError, match="duplicados"):
        to_dataframe(rows)


def test_unsorted_candles_are_rejected() -> None:
    """Las velas fuera de orden invalidan los indicadores en silencio."""
    rows = raw_ohlcv([100.0, 101.0, 102.0])
    rows.reverse()
    with pytest.raises(MarketDataError, match="ordenadas"):
        to_dataframe(rows)


def test_gaps_are_rejected() -> None:
    """Un hueco produciría NaN que se arrastran hasta el IndicatorSet."""
    rows = raw_ohlcv([100.0, 101.0])
    rows[1][4] = float("nan")
    with pytest.raises(MarketDataError, match="huecos"):
        to_dataframe(rows)


# ───────────────────────────────────────── Vela en formación ──────────────────────────────────────


def test_forming_candle_is_dropped() -> None:
    """La última vela todavía no cerró: incluirla haría parpadear los triggers."""
    frame = candles([100.0, 101.0, 102.0])
    now = START + timedelta(hours=2, minutes=30)
    trimmed = drop_forming_candle(frame, "1h", now)
    assert len(trimmed) == 2
    assert trimmed["close"].iloc[-1] == 101.0


def test_closed_candle_is_kept() -> None:
    """Si el periodo ya cerró, la vela es válida."""
    frame = candles([100.0, 101.0, 102.0])
    now = START + timedelta(hours=3, minutes=1)
    assert len(drop_forming_candle(frame, "1h", now)) == 3


def test_candle_closing_exactly_now_is_kept() -> None:
    """En el instante exacto de cierre la vela ya está completa."""
    frame = candles([100.0, 101.0, 102.0])
    now = START + timedelta(hours=3)
    assert len(drop_forming_candle(frame, "1h", now)) == 3


# ────────────────────────────────────────────── Digest ────────────────────────────────────────────


def test_digest_is_reproducible() -> None:
    """Las mismas velas producen el mismo digest en dos construcciones distintas."""
    first = candles([100.0, 101.0, 102.0])
    second = candles([100.0, 101.0, 102.0])
    assert candles_digest(first) == candles_digest(second)


def test_digest_changes_with_a_single_price() -> None:
    """Un cambio de precio debe cambiar el digest."""
    base = candles([100.0, 101.0, 102.0])
    altered = candles([100.0, 101.0, 102.01])
    assert candles_digest(base) != candles_digest(altered)


def test_digest_changes_with_timestamps() -> None:
    """Los mismos precios en otro momento son otras velas."""
    base = candles([100.0, 101.0, 102.0])
    shifted = candles([100.0, 101.0, 102.0], start=START + timedelta(hours=1))
    assert candles_digest(base) != candles_digest(shifted)


def test_digest_changes_with_volume() -> None:
    """El volumen entra en el digest: dos series con distinto volumen no son iguales."""
    base = candles([100.0, 101.0, 102.0])
    other = candles([100.0, 101.0, 102.0], volumes=[1000.0, 1000.0, 2000.0])
    assert candles_digest(base) != candles_digest(other)


def test_digest_is_hex_of_expected_length() -> None:
    """`MarketSnapshot` exige 64 caracteres hex."""
    digest = candles_digest(candles([100.0, 101.0]))
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")


# ──────────────────────────────────────────── Orquestación ────────────────────────────────────────


@pytest.mark.asyncio
async def test_load_candles_drops_the_forming_one() -> None:
    """El cliente falso entrega tres velas y queda una fuera por estar en formación."""
    client = FakeMarketClient(raw_ohlcv([100.0, 101.0, 102.0]))
    frame = await load_candles(
        client, "BTC/USDT", "1h", limit=500, now=START + timedelta(hours=2, minutes=30)
    )
    assert len(frame) == 2
    assert client.calls == [("BTC/USDT", "1h", 500)]


@pytest.mark.asyncio
async def test_load_candles_fails_when_nothing_is_closed() -> None:
    """Una sola vela todavía en formación deja el conjunto vacío."""
    client = FakeMarketClient(raw_ohlcv([100.0]))
    with pytest.raises(MarketDataError, match="no quedaron velas cerradas"):
        await load_candles(client, "BTC/USDT", "1h", limit=500, now=START)


# ──────────────────────────────── Descarga paginada ──────────────────────────────────────────────


class PagingClient:
    """Sirve como sirve el exchange: como mucho `page` velas por petición."""

    def __init__(self, rows: list[list[float]], page: int = 1000) -> None:
        self.rows = rows
        self.page = page
        self.calls: list[int] = []

    async def fetch_ohlcv_since(
        self, symbol: str, timeframe: str, since: int, limit: int
    ) -> list[list[float]]:
        """Página que empieza en el primer timestamp mayor o igual a `since`."""
        del symbol, timeframe
        self.calls.append(since)
        matching = [row for row in self.rows if row[0] >= since]
        return matching[: min(limit, self.page)]


@pytest.mark.asyncio
async def test_download_history_pages_past_the_exchange_limit() -> None:
    """Dos años en 1h son 17 520 velas y el exchange sirve 1000: hay que paginar.

    Sin paginar, pedir dos años devuelve las últimas mil velas sin decir que
    faltan las demás — un histórico truncado en silencio, que sostiene una tabla
    entera sobre datos que no son los que dice.
    """
    rows = raw_ohlcv([100.0 + index * 0.01 for index in range(2500)], step=timedelta(hours=1))
    client = PagingClient(rows)

    downloaded = await download_history(
        client,  # type: ignore[arg-type]
        "BTC/USDT",
        "1h",
        START,
        START + timedelta(hours=2500),
    )

    assert len(downloaded) == 2500
    assert len(client.calls) == 3
    assert [row[0] for row in downloaded] == sorted(row[0] for row in downloaded)


@pytest.mark.asyncio
async def test_download_history_stops_when_a_page_adds_nothing() -> None:
    """Un exchange que devuelve siempre lo mismo no puede convertir esto en un bucle."""
    rows = raw_ohlcv([100.0, 101.0], step=timedelta(hours=1))
    client = PagingClient(rows, page=2)

    downloaded = await download_history(
        client,  # type: ignore[arg-type]
        "BTC/USDT",
        "1h",
        START,
        START + timedelta(days=365),
    )

    assert len(downloaded) == 2


@pytest.mark.asyncio
async def test_download_history_excludes_candles_past_the_end() -> None:
    """El rango pedido es el rango devuelto: una vela de más cambia el digest."""
    rows = raw_ohlcv([100.0] * 10, step=timedelta(hours=1))
    client = PagingClient(rows)

    downloaded = await download_history(
        client,  # type: ignore[arg-type]
        "BTC/USDT",
        "1h",
        START,
        START + timedelta(hours=4),
    )

    assert len(downloaded) == 4


def test_csv_round_trip_keeps_the_digest() -> None:
    """El caché de `var/history/` tiene que devolver exactamente lo que guardó.

    El timestamp se escribe entero por esto: en notación científica el ida y
    vuelta por `float` pierde milisegundos y el digest de la serie deja de
    coincidir consigo mismo, que es justo lo que el reporte publica.
    """
    import tempfile
    from pathlib import Path as _Path

    rows = raw_ohlcv(drifting_closes(50))
    with tempfile.TemporaryDirectory() as directory:
        path = _Path(directory) / "serie.csv"
        write_ohlcv_csv(path, rows)
        assert candles_digest(to_dataframe(read_ohlcv_csv(path))) == candles_digest(
            to_dataframe(rows)
        )


def test_unknown_exchange_is_reported_at_construction() -> None:
    """Un exchange inexistente debe fallar al construir, no en la primera descarga.

    Construye el cliente real: ccxt se importa pero no abre ninguna conexión.
    """
    with pytest.raises(MarketDataError, match="exchange desconocido"):
        CcxtMarketClient("no_existe")


# ──────────────────────────── Lectura y órdenes son dos clientes ─────────────────────────────────
# Un solo cliente mezclaba dos papeles de los que solo uno quiere credenciales, y
# las dos consecuencias eran medibles: con las claves puestas ccxt firma también
# los endpoints públicos y producción responde `-2008 Invalid Api-Key ID`; y con
# `sandbox=true` la fuente de datos pasaba a ser testnet, que da 58 velas de 4h
# contra un preset que exige 400.


def credentialed_settings() -> ExchangeSettings:
    """Configuración como la de operación: claves de testnet y sandbox activo."""
    return ExchangeSettings(
        exchange_id="binance",
        api_key=SecretStr("clave-de-testnet"),
        api_secret=SecretStr("secreto-de-testnet"),
        sandbox=True,
    )


def test_the_market_client_never_carries_credentials() -> None:
    """Aunque la configuración las tenga, el cliente de lectura no puede recibirlas.

    La garantía es la firma: `CcxtMarketClient` toma un `exchange_id`, así que no
    hay por dónde pasarle una clave. Aceptar el `ExchangeSettings` e ignorar tres
    de sus campos sería una configuración que dice una cosa y hace otra.
    """
    settings = credentialed_settings()
    client = CcxtMarketClient(settings.exchange_id)

    assert not client._exchange.apiKey
    assert not client._exchange.secret


def test_the_market_client_loads_only_spot_markets() -> None:
    """`load_markets()` corre antes de la primera vela y puede tumbarla entera.

    En binance carga spot, futuros lineales y futuros inversos en paralelo; basta
    que uno no conteste. `dapi.binance.com` dio `RequestTimeout` dos veces
    seguidas mientras spot respondía, y en el runner eso es una evaluación
    perdida por ciclo por un mercado que no se usa.
    """
    client = CcxtMarketClient("binance")

    assert client._exchange.options["fetchMarkets"] == ["spot"]


def test_the_market_client_reads_production_even_with_sandbox_on() -> None:
    """`sandbox` dejó de gobernar de dónde se leen las velas, y tiene que notarse.

    Testnet contesta perfectamente y devuelve histórico insuficiente, así que
    apuntar ahí la fuente de datos hace que ninguna evaluación pueda completarse.
    """
    settings = credentialed_settings()
    reader = CcxtMarketClient(settings.exchange_id)
    trader = CcxtTradingClient(settings)

    assert reader.api_base_url != trader.api_base_url


def test_the_trading_client_keeps_the_credentials_and_the_sandbox() -> None:
    """El otro lado del reparto: lo que firma órdenes sí lleva claves y sí va a testnet."""
    client = CcxtTradingClient(credentialed_settings())

    assert client._exchange.apiKey == "clave-de-testnet"
    assert client._exchange.secret == "secreto-de-testnet"


@pytest.mark.asyncio
async def test_verifying_credentials_that_do_not_exist_says_so() -> None:
    """Sin claves declaradas no hay nada que comprobar, y decirlo es la respuesta."""
    client = CcxtTradingClient(ExchangeSettings(exchange_id="binance"))

    with pytest.raises(MarketDataError, match="no hay credenciales"):
        await client.verify_credentials()


def test_build_snapshot_carries_identity_not_data() -> None:
    """El snapshot lleva la identidad del momento, nunca el DataFrame."""
    frame = candles([100.0, 101.0, 102.0])
    run_id = uuid4()
    snapshot = build_snapshot(frame, run_id, "binance", "BTC/USDT", "1h")

    assert snapshot.run_id == run_id
    assert snapshot.close == 102.0
    assert snapshot.candles_count == 3
    assert snapshot.candles_digest == candles_digest(frame)
    assert snapshot.timestamp == datetime(2026, 8, 1, 2, 0, tzinfo=UTC)
