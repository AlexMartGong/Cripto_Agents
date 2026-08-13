"""Acceso a datos de mercado: descarga, normalización y digest reproducible.

Todo aquí es determinista y ocurre antes de gastar una sola llamada a modelo.

Dos decisiones que afectan la corrección del resto del pipeline:

- Se descarta la vela en formación. ccxt devuelve la vela del periodo actual, que
  todavía cambia. Un indicador calculado sobre ella enciende y apaga triggers
  varias veces dentro del mismo periodo, y el gate gastaría cuota por ruido.
- El digest se calcula sobre bytes canónicos big-endian, no con
  `pd.util.hash_pandas_object`, que no garantiza estabilidad entre versiones de
  pandas. Dos ejecuciones sobre las mismas velas deben producir el mismo digest.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Protocol

import pandas as pd

from crypto_agents.state import MarketSnapshot

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from crypto_agents.settings import ExchangeSettings

__all__ = [
    "OHLCV_COLUMNS",
    "CcxtMarketClient",
    "MarketClient",
    "MarketDataError",
    "build_snapshot",
    "candles_digest",
    "drop_forming_candle",
    "load_candles",
    "timeframe_to_timedelta",
    "to_dataframe",
]

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
_DIGEST_VERSION = b"ohlcv-v1"
_TIMEFRAME_UNITS = {"m": "minutes", "h": "hours", "d": "days", "w": "weeks"}


class MarketDataError(RuntimeError):
    """Los datos recibidos del exchange no sirven para calcular indicadores."""


class MarketClient(Protocol):
    """Contrato mínimo de un origen de velas. Las pruebas inyectan uno falso."""

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> list[list[float]]:
        """Devuelve filas `[timestamp_ms, open, high, low, close, volume]`."""
        ...


class CcxtMarketClient:
    """Cliente sobre `ccxt.async_support`.

    Async por coherencia con `ModelRouter.invoke`: un solo modelo de concurrencia
    en todo el grafo. El exchange mantiene una sesión que hay que cerrar.
    """

    def __init__(self, settings: ExchangeSettings) -> None:
        import ccxt.async_support as ccxt_async  # importación diferida: ccxt es pesado

        exchange_class = getattr(ccxt_async, settings.exchange_id, None)
        if exchange_class is None:
            raise MarketDataError(f"exchange desconocido para ccxt: {settings.exchange_id}")

        credentials: dict[str, object] = {"enableRateLimit": True}
        if settings.api_key is not None and settings.api_secret is not None:
            credentials["apiKey"] = settings.api_key.get_secret_value()
            credentials["secret"] = settings.api_secret.get_secret_value()

        self._exchange = exchange_class(credentials)
        if settings.sandbox:
            self._exchange.set_sandbox_mode(True)

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> list[list[float]]:
        """Descarga velas crudas del exchange."""
        result: list[list[float]] = await self._exchange.fetch_ohlcv(
            symbol, timeframe=timeframe, limit=limit
        )
        return result

    async def close(self) -> None:
        """Cierra la sesión HTTP. Sin esto, el event loop queda con conexiones abiertas."""
        await self._exchange.close()


def timeframe_to_timedelta(timeframe: str) -> timedelta:
    """Traduce un timeframe de ccxt (`1m`, `4h`, `1d`) a duración."""
    if len(timeframe) < 2:
        raise MarketDataError(f"timeframe inválido: {timeframe!r}")
    amount, unit = timeframe[:-1], timeframe[-1]
    if unit not in _TIMEFRAME_UNITS or not amount.isdigit():
        raise MarketDataError(f"timeframe inválido: {timeframe!r}")
    return timedelta(**{_TIMEFRAME_UNITS[unit]: int(amount)})


def to_dataframe(raw: Sequence[Sequence[float]]) -> pd.DataFrame:
    """Normaliza las filas de ccxt a un DataFrame con índice temporal UTC."""
    if not raw:
        raise MarketDataError("el exchange no devolvió velas")

    frame = pd.DataFrame(list(raw), columns=["timestamp", *OHLCV_COLUMNS])
    frame.index = pd.to_datetime(frame.pop("timestamp"), unit="ms", utc=True)
    frame.index.name = "timestamp"
    frame = frame.astype("float64")

    if frame.index.has_duplicates:
        raise MarketDataError("el exchange devolvió timestamps duplicados")
    if not frame.index.is_monotonic_increasing:
        raise MarketDataError("las velas no vienen ordenadas cronológicamente")
    if frame.isna().to_numpy().any():
        raise MarketDataError("hay huecos en las velas recibidas")
    return frame


def drop_forming_candle(candles: pd.DataFrame, timeframe: str, now: datetime) -> pd.DataFrame:
    """Descarta la última vela si su periodo todavía no ha cerrado.

    `now` se inyecta por la misma razón que el reloj del contador de cuota: si no,
    el corte dependería del reloj real y sería improbable de probar.
    """
    if candles.empty:
        return candles
    closes_at = candles.index[-1] + timeframe_to_timedelta(timeframe)
    if closes_at > pd.Timestamp(now):
        return candles.iloc[:-1]
    return candles


def candles_digest(candles: pd.DataFrame) -> str:
    """Huella reproducible de un conjunto de velas.

    Big-endian explícito para que el digest no dependa de la arquitectura, y con
    etiqueta de versión para poder cambiar el formato sin confundir digests
    viejos con nuevos.
    """
    digest = hashlib.sha256()
    digest.update(_DIGEST_VERSION)
    digest.update(candles.index.to_numpy("datetime64[ns]").astype(">i8").tobytes())
    digest.update(candles.loc[:, list(OHLCV_COLUMNS)].to_numpy().astype(">f8").tobytes())
    return digest.hexdigest()


async def load_candles(
    client: MarketClient,
    symbol: str,
    timeframe: str,
    limit: int,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Descarga, normaliza y deja solo velas cerradas."""
    raw = await client.fetch_ohlcv(symbol, timeframe, limit)
    candles = drop_forming_candle(to_dataframe(raw), timeframe, now or datetime.now(UTC))
    if candles.empty:
        raise MarketDataError("no quedaron velas cerradas tras descartar la vela en formación")
    return candles


def build_snapshot(
    candles: pd.DataFrame, run_id: UUID, exchange: str, symbol: str, timeframe: str
) -> MarketSnapshot:
    """Identidad del momento evaluado. El DataFrame no viaja: solo su digest."""
    last = candles.index[-1]
    return MarketSnapshot(
        run_id=run_id,
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        timestamp=last.to_pydatetime(),
        close=float(candles["close"].iloc[-1]),
        candles_digest=candles_digest(candles),
        candles_count=len(candles),
    )
