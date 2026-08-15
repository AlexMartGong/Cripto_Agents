"""Acceso a datos de mercado: descarga, normalización y digest reproducible.

Todo aquí es determinista y ocurre antes de gastar una sola llamada a modelo.

Tres decisiones que afectan la corrección del resto del pipeline:

- Se descarta la vela en formación. ccxt devuelve la vela del periodo actual, que
  todavía cambia. Un indicador calculado sobre ella enciende y apaga triggers
  varias veces dentro del mismo periodo, y el gate gastaría cuota por ruido.
- El digest se calcula sobre bytes canónicos big-endian, no con
  `pd.util.hash_pandas_object`, que no garantiza estabilidad entre versiones de
  pandas. Dos ejecuciones sobre las mismas velas deben producir el mismo digest.
- **Leer mercado y operar son dos clientes distintos.** Uno solo mezclaba dos
  papeles de los que únicamente uno quiere credenciales, y las consecuencias eran
  medibles: con las claves puestas ccxt firma también los endpoints públicos y
  producción responde `-2008 Invalid Api-Key ID`; y con `sandbox=true` la fuente
  de datos pasaba a ser testnet, que devuelve 58 velas de 4h contra un preset que
  exige 400. Ninguna evaluación podía completarse.
"""

from __future__ import annotations

import csv
import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import pandas as pd

from crypto_agents.state import MarketSnapshot

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from crypto_agents.settings import ExchangeSettings

__all__ = [
    "OHLCV_COLUMNS",
    "SPOT_ONLY",
    "CcxtMarketClient",
    "CcxtTradingClient",
    "MarketClient",
    "MarketDataError",
    "build_snapshot",
    "candles_digest",
    "download_history",
    "drop_forming_candle",
    "load_candles",
    "read_ohlcv_csv",
    "timeframe_to_timedelta",
    "to_dataframe",
    "write_ohlcv_csv",
]

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")

SPOT_ONLY = {"fetchMarkets": ["spot"]}
"""Restringe la carga de mercados a spot en el cliente de lectura.

`load_markets()` corre antes de la primera vela, y en binance carga tres
universos en paralelo: spot, futuros lineales y futuros inversos. Basta que uno
no conteste para que la lectura falle entera — `dapi.binance.com` dio
`RequestTimeout` dos veces seguidas desde el escritorio mientras spot respondía
sin problema. Pedir velas de spot no necesita los otros dos, y cargarlos triplica
la superficie de fallo de una lectura pública que en el runner cuesta una
evaluación perdida por ciclo.

Es una clave que ccxt define por exchange; los que no la usan la ignoran.
"""

_PAGE_LIMIT = 1000
"""Techo por petición del exchange. Binance no sirve más aunque se le pida."""

_DIGEST_VERSION = b"ohlcv-v1"
_TIMEFRAME_UNITS = {"m": "minutes", "h": "hours", "d": "days", "w": "weeks"}


class MarketDataError(RuntimeError):
    """Los datos recibidos del exchange no sirven para calcular indicadores."""


class MarketClient(Protocol):
    """Contrato mínimo de un origen de velas. Las pruebas inyectan uno falso."""

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> list[list[float]]:
        """Devuelve filas `[timestamp_ms, open, high, low, close, volume]`."""
        ...


class _CcxtExchange(Protocol):
    """La superficie de ccxt de la que depende este módulo, y solo esa.

    ccxt no trae anotaciones, así que sin esto cada uso necesitaría silenciar al
    comprobador. Declararla como protocolo deja escrito qué se usa de verdad: son
    siete miembros, y cualquier cosa que crezca aquí es una dependencia nueva que
    se ve en el diff.
    """

    urls: dict[str, object]
    options: dict[str, object]
    apiKey: str | None  # noqa: N815  # el nombre lo pone ccxt
    secret: str | None

    def set_sandbox_mode(self, enabled: bool) -> None:
        """Cambia las URLs a las de pruebas."""
        ...

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int, since: int | None = None
    ) -> list[list[float]]:
        """Velas crudas."""
        ...

    async def fetch_balance(self) -> object:
        """Llamada privada: exige credenciales válidas."""
        ...

    async def close(self) -> None:
        """Cierra la sesión HTTP."""
        ...


def _build_exchange(exchange_id: str, options: dict[str, object]) -> _CcxtExchange:
    """Instancia el exchange de ccxt, o falla nombrando el id."""
    import ccxt.async_support as ccxt_async  # importación diferida: ccxt es pesado

    exchange_class = getattr(ccxt_async, exchange_id, None)
    if exchange_class is None:
        raise MarketDataError(f"exchange desconocido para ccxt: {exchange_id}")
    built: _CcxtExchange = exchange_class({"enableRateLimit": True, **options})
    return built


def _api_base_url(exchange: _CcxtExchange) -> str:
    """A dónde apunta ccxt de verdad tras aplicar el modo sandbox.

    Se lee del cliente y no de la configuración: `sandbox=true` es una intención,
    y esto es la consecuencia. Es la diferencia entre creer que se opera contra
    testnet y comprobarlo.
    """
    urls = exchange.urls.get("api")
    if isinstance(urls, dict):
        for key in ("public", "spot", "rest"):
            value = urls.get(key)
            if isinstance(value, str):
                return value
        return next((value for value in urls.values() if isinstance(value, str)), "")
    return urls if isinstance(urls, str) else ""


class CcxtMarketClient:
    """Origen de velas. No puede llevar credenciales ni apuntar a sandbox.

    Recibe un `exchange_id` y no un `ExchangeSettings`, y eso es lo que hace la
    garantía estructural en vez de una promesa: por la firma no entra una clave.
    Aceptar el objeto de configuración e ignorar tres de sus campos diría una cosa
    en `.env` y haría otra.

    Las dos razones son consecuencias medidas, no preferencias:

    - **Con credenciales, ccxt firma también los endpoints públicos**, y binance
      responde `-2008 Invalid Api-Key ID` a una lectura de velas que sin claves
      devuelve 500 sin problema. Leer mercado no necesita autenticarse.
    - **Testnet no sirve como fuente de datos**: devuelve 58 velas de 4h contra un
      preset que exige 400. Con `sandbox` gobernando la lectura, ninguna
      evaluación podía completarse jamás.

    Async por coherencia con `ModelRouter.invoke`: un solo modelo de concurrencia
    en todo el grafo. El exchange mantiene una sesión que hay que cerrar.
    """

    def __init__(self, exchange_id: str) -> None:
        self._exchange = _build_exchange(exchange_id, {"options": dict(SPOT_ONLY)})

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> list[list[float]]:
        """Descarga las velas más recientes. Es el método del protocolo que usa el grafo."""
        result: list[list[float]] = await self._exchange.fetch_ohlcv(
            symbol, timeframe=timeframe, limit=limit
        )
        return result

    async def fetch_ohlcv_since(
        self, symbol: str, timeframe: str, since: int, limit: int
    ) -> list[list[float]]:
        """Una página que empieza en `since`, en milisegundos.

        Método aparte y no un argumento opcional de `fetch_ohlcv`: el protocolo
        `MarketClient` que consumen los nodos pide las últimas N velas y nada
        más, y ensancharlo dejaría a un nodo la posibilidad de elegir desde
        cuándo mira el mercado.
        """
        result: list[list[float]] = await self._exchange.fetch_ohlcv(
            symbol, timeframe=timeframe, since=since, limit=limit
        )
        return result

    @property
    def api_base_url(self) -> str:
        """URL efectiva. Siempre la de producción: aquí no hay sandbox que aplicar."""
        return _api_base_url(self._exchange)

    async def close(self) -> None:
        """Cierra la sesión HTTP. Sin esto, el event loop queda con conexiones abiertas."""
        await self._exchange.close()


class CcxtTradingClient:
    """Cliente autenticado. El único que firma, y solo para lo que hay que firmar.

    Es el que honra `CA_EXCHANGE__SANDBOX`, porque tras separar los papeles esa
    variable ya no gobierna de dónde se leen las velas: gobierna a dónde van las
    órdenes. Hoy su único uso es comprobar credenciales en `doctor`, ya que el
    envío real exige además `CA_EXECUTION__MODE=live`.
    """

    def __init__(self, settings: ExchangeSettings) -> None:
        credentials: dict[str, object] = {}
        if settings.api_key is not None and settings.api_secret is not None:
            credentials["apiKey"] = settings.api_key.get_secret_value()
            credentials["secret"] = settings.api_secret.get_secret_value()

        self._exchange = _build_exchange(settings.exchange_id, credentials)
        if settings.sandbox:
            self._exchange.set_sandbox_mode(True)

    @property
    def api_base_url(self) -> str:
        """URL efectiva tras aplicar el modo sandbox."""
        return _api_base_url(self._exchange)

    async def verify_credentials(self) -> None:
        """Llamada privada de lectura: prueba que las claves sirven.

        Leer velas ya no las usa —ni siquiera las tiene—, así que sin esto unas
        claves mal copiadas no darían señal hasta la primera orden, cuando ya hay
        una decisión tomada detrás.
        """
        if not self._exchange.apiKey:
            raise MarketDataError("no hay credenciales declaradas que comprobar")
        try:
            await self._exchange.fetch_balance()
        except Exception as error:  # ccxt levanta su propia jerarquía
            raise MarketDataError(f"{type(error).__name__}: {error}") from error

    async def close(self) -> None:
        """Cierra la sesión HTTP."""
        await self._exchange.close()


async def download_history(
    client: CcxtMarketClient, symbol: str, timeframe: str, start: datetime, end: datetime
) -> list[list[float]]:
    """Descarga un rango largo paginando, porque el exchange corta la respuesta.

    Binance devuelve como mucho 1000 velas por petición: dos años en 4h son cinco
    páginas y en 1h son dieciocho. Sin paginar, pedir dos años devuelve las
    últimas 1000 velas sin decir que faltan las demás — un histórico truncado en
    silencio, que es peor que ninguno.

    El avance se hace desde el último timestamp recibido y no sumando el periodo:
    un hueco en los datos del exchange desalinearía el segundo cálculo y la
    descarga se quedaría pidiendo velas que no existen. Si una página no aporta
    filas nuevas se corta, para que un exchange que devuelve siempre lo mismo no
    convierta esto en un bucle infinito.
    """
    period = timeframe_to_timedelta(timeframe)
    end_ms = int(end.timestamp() * 1000)
    cursor = int(start.timestamp() * 1000)
    rows: list[list[float]] = []
    seen: set[float] = set()

    while cursor < end_ms:
        page = await client.fetch_ohlcv_since(symbol, timeframe, cursor, _PAGE_LIMIT)
        fresh = [row for row in page if row[0] not in seen and row[0] < end_ms]
        if not fresh:
            break
        seen.update(row[0] for row in fresh)
        rows.extend(fresh)
        cursor = int(max(row[0] for row in fresh)) + int(period.total_seconds() * 1000)

    rows.sort(key=lambda row: row[0])
    if not rows:
        raise MarketDataError(f"{symbol} {timeframe}: el exchange no devolvió velas en el rango")
    return rows


def read_ohlcv_csv(path: Path | str) -> list[list[float]]:
    """Lee un histórico en CSV con cabecera `timestamp,open,high,low,close,volume`.

    Devuelve filas crudas, en el mismo formato que entrega ccxt, para que nada
    aguas arriba distinga entre un histórico de archivo y uno recién descargado.
    """
    rows: list[list[float]] = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise MarketDataError(f"histórico vacío: {path}")
        for line in reader:
            rows.append([float(value) for value in line])
    if not rows:
        raise MarketDataError(f"histórico sin velas: {path}")
    return rows


def write_ohlcv_csv(path: Path | str, rows: Sequence[Sequence[float]]) -> None:
    """Guarda filas crudas en el formato que lee `read_ohlcv_csv`.

    El timestamp se escribe como entero: en notación científica el ida y vuelta
    por `float` pierde milisegundos y el digest de la serie deja de coincidir
    consigo mismo entre una descarga y su relectura.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", *OHLCV_COLUMNS])
        for row in rows:
            writer.writerow([int(row[0]), *row[1:]])


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
