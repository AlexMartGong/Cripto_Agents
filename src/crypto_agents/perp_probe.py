"""Sondeo público de perpetuos USDT: qué existe, cuánto cuesta el funding y desde dónde se alcanza.

Solo lee endpoints públicos de ccxt. Sin claves, sin órdenes, sin modelos. Informa y no elige:
decidir exchange es otra pregunta, con otros datos.

    python -m crypto_agents.perp_probe --exchange binanceusdm
    python -m crypto_agents.perp_probe --exchange bybit

Cinco decisiones que lo mantienen honesto, cada una con prueba:

- **El intervalo del funding se infiere, no se supone.** Cada periodo cubre `(ts[k-1], ts[k]]`, y
  su intervalo es el hueco con el anterior, aceptado solo si cae en {1, 2, 4, 8} h. Binance y
  Bybit cambian el intervalo por símbolo, así que fijar 8 h daría cifras dos u ocho veces
  equivocadas sin que nada fallara. La primera fila de una serie y los huecos fuera de rejilla se
  excluyen de la normalización y se cuentan. Límite conocido: un registro faltante dentro de un
  tramo de 4 h parece un hueco válido de 8 h y no se puede detectar.
- **La paginación no depende de la semántica de cada exchange.** Binance devuelve las filas más
  antiguas desde `since`; Bybit, con `until`, las más recientes de la ventana. Paginar hacia
  delante contra el segundo se saltaría datos en silencio. Cada petición pide una ventana cerrada
  por los dos lados, una página llena se considera ambigua y se parte, y las ventanas se solapan
  en el borde para no perder la fila que cae justo en él.
- **Una petición por ventana, sin reintentos, con tope.** Un 429 o un 418 aborta todo el sondeo:
  binance banea la IP si se insiste. Otro fallo deja ese símbolo en `no determinado` y se sigue.
- **Cada cifra cita el digest de su origen.** El de la serie de funding es el sha-256 de los bytes
  del CSV guardado, así que se verifica con `sha256sum` sin este código. Contrato y volumen citan el
  sha-256 del JSON canónico de la entrada.
- **No guarda ni imprime cabeceras.** `instrument()` mide cada petición HTTP sin leer una sola
  cabecera, y de la URL descarta la query.

Este módulo no importa nada del paquete: ni el router, ni `execution`, ni `settings`, ni
`market`. No lee ninguna variable de entorno y construye ccxt sin credenciales, sin proxy y sin
sandbox. Un test de arquitectura lo fija. Nada de lo que imprime es un retorno de una estrategia.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import socket
import statistics
import subprocess
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Protocol
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence

__all__ = [
    "ACCEPTED_INTERVAL_HOURS",
    "CAPABILITIES",
    "EXCHANGES",
    "FUNDING_PAGE_LIMIT",
    "HISTORY_DAYS",
    "MAX_REQUESTS_PER_SERIES",
    "OUT_DIR",
    "RATE_LIMIT_STATUSES",
    "TICK_SIZE",
    "UNIVERSE_BASES",
    "Absent",
    "ContractSpec",
    "EndpointCall",
    "EndpointLog",
    "Environment",
    "FundingRow",
    "FundingSummary",
    "Normalised",
    "ProbeError",
    "ProbeReport",
    "SeriesResult",
    "SymbolResult",
    "Ticker24h",
    "build_exchange",
    "capabilities",
    "describe_market",
    "fetch_funding_series",
    "instrument",
    "interval_hours",
    "main",
    "normalise_24h",
    "parse_funding",
    "perp_symbol",
    "render_report",
    "run_probe",
    "series_csv",
    "series_digest",
    "summarise_funding",
]

EXCHANGES = ("binanceusdm", "bybit")
"""Los dos exchanges que se comparan. Ambos exponen perpetuos lineales USDT por ccxt."""

UNIVERSE_BASES = ("BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE")
"""Las bases del universo del resto del proyecto. Un test las ata a `DEFAULT_SYMBOLS`."""

QUOTE = "USDT"
HISTORY_DAYS = 730

FUNDING_PAGE_LIMIT = {"binanceusdm": 1000, "bybit": 200}
"""Techo de filas por petición de cada exchange. Pedir más no devuelve más."""

OUT_DIR = Path("var", "perp")

CAPABILITIES = (
    "fetchFundingRateHistory",
    "setLeverage",
    "setMarginMode",
    "createOrder",
    "createReduceOnlyOrder",
    "createStopOrder",
    "createStopLossOrder",
    "createTakeProfitOrder",
    "createTriggerOrder",
)
"""Lo que `exchange.has` dice de lo que usaría un ejecutor.

`createOrder` con `reduceOnly` no tiene bandera propia: `createReduceOnlyOrder` es lo más cercano.
`has` describe lo que ccxt implementa, no lo que una cuenta o una región pueden usar.
"""

TICK_SIZE = 4
"""`ccxt.TICK_SIZE`. Un test lo ata a la constante real sin importar ccxt en el nivel de módulo."""

ACCEPTED_INTERVAL_HOURS = (1, 2, 4, 8)
INTERVAL_TOLERANCE_MS = 60_000
"""Cuánto puede separarse un hueco de un número entero de horas. Cubre el jitter de milisegundos
de las marcas de binance sin aceptar como 8 h un hueco de 8 h y media."""

HOURS_PER_DAY = 24
HOUR_MS = 3_600_000
WINDOW_INTERVAL_HOURS = 8
WINDOW_SLACK_ROWS = 2
"""La ventana inicial cubre `limit - 2` periodos de 8 h. Cerrada por los dos lados, una ventana de
`limit` periodos contiene `limit + 1` filas y saldría siempre llena: se partiría siempre, y la
primera corrida real lo midió (217 peticiones a Bybit en vez de unas 80). Un símbolo con
intervalo menor sí llena la página, y entonces la ventana se parte."""

MIN_WINDOW_MS = 60_000
MAX_REQUESTS_PER_SERIES = 600
"""Tope de peticiones por serie. El peor caso real, 1 h de intervalo durante dos años en Bybit, son
unas 350; pasar de aquí es un exchange que no respeta las ventanas y no hay que golpearlo más."""

RATE_LIMIT_STATUSES = frozenset({418, 429})
ERROR_TEXT_LIMIT = 200


class ProbeError(RuntimeError):
    """Algo que el sondeo no puede dar por bueno: datos malformados o un exchange que no cumple."""


# ──────────────────────────────── Superficie de ccxt ─────────────────────────────────────


class _Instrumentable(Protocol):
    """Los dos puntos de ccxt que `instrument` sustituye. Atributos asignables, no métodos."""

    fetch: Callable[..., Awaitable[object]]
    on_rest_response: Callable[..., object]


class _FundingSource(Protocol):
    """Lo único que la paginación le pide a un exchange."""

    async def fetch_funding_rate_history(
        self,
        symbol: str,
        since: int | None = None,
        limit: int | None = None,
        params: dict[str, object] | None = None,
    ) -> Sequence[Mapping[str, object]]:
        """Historial de funding."""
        ...


class _PerpExchange(_Instrumentable, _FundingSource, Protocol):
    """La superficie de ccxt de la que depende este módulo, y solo esa.

    Las propiedades son de solo lectura: lo que el módulo necesita de ellas es leerlas.
    """

    @property
    def id(self) -> str:
        """Identificador de ccxt."""
        ...

    @property
    def has(self) -> Mapping[str, object]:
        """Capacidades que ccxt implementa."""
        ...

    @property
    def precisionMode(self) -> int:  # noqa: N802  # el nombre lo pone ccxt
        """Cómo interpretar `precision`."""
        ...

    async def load_markets(self) -> Mapping[str, Mapping[str, object]]:
        """Catálogo de mercados."""
        ...

    async def fetch_ticker(self, symbol: str) -> Mapping[str, object]:
        """Ticker de 24 h."""
        ...

    async def close(self) -> None:
        """Cierra la sesión HTTP."""
        ...


def build_exchange(exchange_id: str) -> _PerpExchange:
    """Instancia ccxt sin credenciales, sin proxy y sin sandbox.

    Solo `enableRateLimit` y, en Bybit, restringir la carga de mercados a los lineales: la misma
    lección que `SPOT_ONLY` en `market.py`, que una lectura pública no debe depender de que
    contesten tres universos cuando solo se pide uno. Binance USDⓈ-M ya carga solo `linear`.
    """
    if exchange_id not in EXCHANGES:
        raise ProbeError(f"exchange no soportado: {exchange_id}")
    import ccxt.async_support as ccxt_async  # importación diferida: ccxt es pesado

    options: dict[str, object] = {}
    if exchange_id == "bybit":
        options["fetchMarkets"] = {"types": ["linear"]}
    built: _PerpExchange = getattr(ccxt_async, exchange_id)(
        {"enableRateLimit": True, "options": options}
    )
    return built


# ───────────────────────────── Mercados: qué existe y con qué límites ─────────────────────────


def perp_symbol(base: str) -> str:
    """Símbolo unificado de ccxt para el perpetuo lineal USDT de `base`."""
    return f"{base}/{QUOTE}:{QUOTE}"


@dataclass(frozen=True, slots=True)
class ContractSpec:
    """Lo que `market` dice de un contrato. `None` es «el exchange no lo informa», no cero."""

    symbol: str
    contract_size: float | None
    tick_size: float | None
    min_amount: float | None
    min_cost: float | None


@dataclass(frozen=True, slots=True)
class Absent:
    """El símbolo no existe como perpetuo lineal USDT activo, y por qué."""

    symbol: str
    reason: str


@dataclass(frozen=True, slots=True)
class Ticker24h:
    """Volumen de 24 h de un ticker. `exchange_ms` es `None` en Bybit, que no lo informa."""

    quote_volume: float | None
    base_volume: float | None
    exchange_ms: int | None


def _number(value: object) -> float | None:
    """Un número real; ni un bool ni un texto cuentan."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _child(mapping: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = mapping.get(key)
    return value if isinstance(value, dict) else {}


def describe_market(
    base: str, markets: Mapping[str, Mapping[str, object]], precision_mode: int
) -> ContractSpec | Absent:
    """El contrato de `base`, o por qué no existe. Manda lo que dicen los campos, no la clave.

    Un perpetuo es `swap`, `linear`, liquidado y cotizado en USDT y activo. Un futuro con
    vencimiento o un perpetuo liquidado en USDC comparten base y se parecen, pero no son este.
    """
    symbol = perp_symbol(base)
    market = markets.get(symbol)
    if market is None:
        return Absent(symbol, "ausente del catálogo")
    if market.get("swap") is not True:
        return Absent(symbol, "no es swap")
    if market.get("linear") is not True:
        return Absent(symbol, "no es lineal")
    if market.get("settle") != QUOTE or market.get("quote") != QUOTE:
        return Absent(symbol, f"no es USDT (liquida en {market.get('settle')})")
    if market.get("active") is False:
        return Absent(symbol, "inactivo")

    limits = _child(market, "limits")
    precision = _child(market, "precision")
    return ContractSpec(
        symbol=symbol,
        contract_size=_number(market.get("contractSize")),
        # Con otro modo `precision.price` son decimales o cifras significativas, no un tick.
        tick_size=_number(precision.get("price")) if precision_mode == TICK_SIZE else None,
        min_amount=_number(_child(limits, "amount").get("min")),
        min_cost=_number(_child(limits, "cost").get("min")),
    )


def parse_ticker(raw: Mapping[str, object]) -> Ticker24h:
    """Volumen de 24 h de un ticker de ccxt."""
    exchange_ms = _number(raw.get("timestamp"))
    return Ticker24h(
        quote_volume=_number(raw.get("quoteVolume")),
        base_volume=_number(raw.get("baseVolume")),
        exchange_ms=None if exchange_ms is None else int(exchange_ms),
    )


# ─────────────────────────────────── Funding: filas ──────────────────────────────────────────


@dataclass(frozen=True, slots=True, order=True)
class FundingRow:
    """Una liquidación de funding: la marca de tiempo y la tasa de ese periodo, como fracción."""

    timestamp_ms: int
    rate: float


def parse_funding(raw: Sequence[Mapping[str, object]]) -> list[FundingRow]:
    """Filas de ccxt a `FundingRow`. Una fila sin número es un error, no un cero."""
    rows: list[FundingRow] = []
    for entry in raw:
        timestamp = _number(entry.get("timestamp"))
        rate = _number(entry.get("fundingRate"))
        if timestamp is None or rate is None:
            raise ProbeError(f"fila de funding sin marca o sin tasa numérica: {dict(entry)!r}")
        rows.append(FundingRow(timestamp_ms=int(timestamp), rate=rate))
    return rows


def _windows(start_ms: int, end_ms: int, step_ms: int) -> deque[tuple[int, int]]:
    """Ventanas cerradas que cubren `[start, end]`, solapadas en el borde."""
    pending: deque[tuple[int, int]] = deque()
    cursor = start_ms
    while cursor < end_ms:
        upper = min(cursor + step_ms, end_ms)
        pending.append((cursor, upper))
        cursor = upper
    return pending


async def fetch_funding_series(
    exchange: _FundingSource, symbol: str, start_ms: int, end_ms: int, page_limit: int
) -> list[FundingRow]:
    """El historial de funding de `symbol` en `[start_ms, end_ms]`, completo o con un error claro.

    Una petición por ventana, cerrada por los dos lados con `since` y `until`. Una página con
    `page_limit` filas o más puede estar truncada, así que no se acepta: se parte la ventana por
    la mitad. Una ventana vacía no corta nada, porque un símbolo listado tarde las tiene.
    Las filas fuera de la ventana se descartan y las repetidas del borde se fusionan.
    """
    step_ms = (page_limit - WINDOW_SLACK_ROWS) * WINDOW_INTERVAL_HOURS * HOUR_MS
    pending = _windows(start_ms, end_ms, step_ms)
    found: dict[int, float] = {}
    requests = 0
    while pending:
        lower, upper = pending.popleft()
        if requests >= MAX_REQUESTS_PER_SERIES:
            raise ProbeError(
                f"{symbol}: más de {MAX_REQUESTS_PER_SERIES} peticiones de funding; "
                "el exchange no respeta las ventanas y no se sigue"
            )
        requests += 1
        raw = await exchange.fetch_funding_rate_history(
            symbol, since=lower, limit=page_limit, params={"until": upper}
        )
        if len(raw) >= page_limit:
            if upper - lower <= MIN_WINDOW_MS:
                raise ProbeError(
                    f"{symbol}: ventana de {(upper - lower) / 1000:.0f} s con la página llena; "
                    "no se puede partir más"
                )
            middle = (lower + upper) // 2
            pending.appendleft((middle, upper))
            pending.appendleft((lower, middle))
            continue
        for row in parse_funding(raw):
            if lower <= row.timestamp_ms <= upper:
                found[row.timestamp_ms] = row.rate
    return [FundingRow(timestamp_ms=ts, rate=found[ts]) for ts in sorted(found)]


# ─────────────────────────── Funding: normalización y resumen ────────────────────────────────


def interval_hours(previous_ms: int, current_ms: int) -> int | None:
    """Horas del periodo que termina en `current_ms`, o `None` si el hueco no es de rejilla."""
    gap = current_ms - previous_ms
    hours = round(gap / HOUR_MS)
    if hours in ACCEPTED_INTERVAL_HOURS and abs(gap - hours * HOUR_MS) <= INTERVAL_TOLERANCE_MS:
        return hours
    return None


@dataclass(frozen=True, slots=True)
class Normalised:
    """Tasas equivalentes a 24 h, con lo que se dejó fuera y los intervalos que se vieron."""

    values: tuple[float, ...]
    intervals: tuple[tuple[int, int], ...]
    unattributable: int


def normalise_24h(rows: Sequence[FundingRow]) -> Normalised:
    """Cada tasa por `24 / intervalo`, con el intervalo propio de su periodo.

    La primera fila no tiene periodo anterior y un hueco fuera de rejilla no tiene intervalo
    creíble: ninguna de las dos se normaliza, y las dos se cuentan.
    """
    values: list[float] = []
    seen: Counter[int] = Counter()
    unattributable = 1 if rows else 0
    for previous, current in pairwise(rows):
        hours = interval_hours(previous.timestamp_ms, current.timestamp_ms)
        if hours is None:
            unattributable += 1
            continue
        seen[hours] += 1
        values.append(current.rate * HOURS_PER_DAY / hours)
    return Normalised(
        values=tuple(values),
        intervals=tuple(sorted(seen.items(), reverse=True)),
        unattributable=unattributable,
    )


@dataclass(frozen=True, slots=True)
class FundingSummary:
    """Resumen de una serie. Las cifras de 24 h son fracciones por 24 h, no porcentajes.

    `negative` cuenta sobre todos los `periods` crudos: el signo no depende del intervalo, y
    un periodo con tasa cero no es negativo.
    """

    periods: int
    normalised: int
    unattributable: int
    intervals: tuple[tuple[int, int], ...]
    mean_24h: float | None
    median_24h: float | None
    p95_24h: float | None
    negative: int


def summarise_funding(rows: Sequence[FundingRow]) -> FundingSummary:
    """Media, mediana y p95 del equivalente a 24 h, y cuántos periodos fueron negativos.

    El p95 interpola linealmente, como el `percentile` por omisión de numpy, y pide al menos dos
    periodos: con uno solo no hay percentil que calcular.
    """
    normalised = normalise_24h(rows)
    values = list(normalised.values)
    return FundingSummary(
        periods=len(rows),
        normalised=len(values),
        unattributable=normalised.unattributable,
        intervals=normalised.intervals,
        mean_24h=statistics.fmean(values) if values else None,
        median_24h=statistics.median(values) if values else None,
        p95_24h=(
            statistics.quantiles(values, n=100, method="inclusive")[94]
            if len(values) >= 2
            else None
        ),
        negative=sum(1 for row in rows if row.rate < 0),
    )


# ────────────────────────────── Serie guardada y su digest ───────────────────────────────────


CSV_HEADER = "timestamp_ms,funding_rate\n"


def series_csv(rows: Sequence[FundingRow]) -> bytes:
    """La serie como CSV canónico: ordenada, con `repr` de cada tasa para no perder un bit."""
    body = "".join(f"{row.timestamp_ms},{row.rate!r}\n" for row in sorted(rows))
    return (CSV_HEADER + body).encode("utf-8")


def series_digest(content: bytes) -> str:
    """El sha-256 de los bytes del archivo: se verifica con `sha256sum`, sin este código."""
    return hashlib.sha256(content).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def entry_digest(entry: Mapping[str, object]) -> str:
    """El sha-256 del JSON canónico de una entrada de contrato."""
    return hashlib.sha256(_canonical_json(dict(entry)).encode("utf-8")).hexdigest()


# ───────────────────────────────────── Conectividad ──────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class EndpointCall:
    """Una petición HTTP: a dónde, qué contestó y cuánto tardó. Nada de cabeceras ni de query."""

    method: str
    host: str
    path: str
    status: int | None
    latency_ms: float
    error: str | None


@dataclass(slots=True)
class EndpointLog:
    """Las peticiones que hizo un exchange, en orden."""

    calls: list[EndpointCall] = field(default_factory=list)

    @property
    def last_status(self) -> int | None:
        """El código de la última petición, o `None` si no llegó a haber respuesta."""
        return self.calls[-1].status if self.calls else None


def _error_text(error: BaseException) -> str:
    return f"{type(error).__name__}: {str(error)[:ERROR_TEXT_LIMIT]}"


def instrument(
    exchange: _Instrumentable, log: EndpointLog, timer: Callable[[], float] = time.perf_counter
) -> None:
    """Hace que cada petición HTTP del exchange quede en `log`, sin leer ninguna cabecera.

    ccxt no guarda el código de una respuesta que salió bien. Se obtiene de dos puntos que ccxt
    ofrece: `on_rest_response`, que corre con el código antes de que ccxt decida si levanta, y
    `fetch`, que se envuelve para medir el tiempo. Ninguno de los dos argumentos de cabeceras se
    toca: se reenvían tal cual al original. El código se reinicia en cada petición para que un
    timeout no herede el 451 de la anterior.
    """
    original_fetch = exchange.fetch
    original_hook = exchange.on_rest_response
    last: list[int | None] = [None]

    def on_rest_response(status_code: int, *args: object) -> object:
        last[0] = status_code
        return original_hook(status_code, *args)

    async def fetch(
        url: str, method: str = "GET", headers: object = None, body: object = None
    ) -> object:
        parts = urlsplit(url)
        last[0] = None
        started = timer()
        error: str | None = None
        try:
            return await original_fetch(url, method, headers, body)
        except Exception as raised:  # ccxt levanta su propia jerarquía
            error = type(raised).__name__
            raise
        finally:
            log.calls.append(
                EndpointCall(
                    method=method,
                    host=parts.netloc,
                    path=parts.path,
                    status=last[0],
                    latency_ms=(timer() - started) * 1000,
                    error=error,
                )
            )

    exchange.on_rest_response = on_rest_response
    exchange.fetch = fetch


def capabilities(exchange: _PerpExchange) -> dict[str, object]:
    """El valor crudo de `exchange.has` para cada capacidad: `True`, `'emulated'`, `None`."""
    return {name: exchange.has.get(name) for name in CAPABILITIES}


# ──────────────────────────────────────── Sondeo ─────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Environment:
    """Todo lo que el sondeo toma del mundo, para que las pruebas lo fijen.

    El reloj, el nombre de la máquina, el estado de git, la fábrica de exchanges y el cronómetro
    entran por aquí: ninguna función de lógica de este módulo llama a `datetime.now()`.
    """

    clock: Callable[[], datetime]
    hostname: str
    git_state: Callable[[], tuple[str | None, bool | None]]
    exchange_factory: Callable[[str], _PerpExchange]
    timer: Callable[[], float] = time.perf_counter


def _git_state() -> tuple[str | None, bool | None]:
    """El commit y si hay cambios sin confirmar; con cambios, el commit no es lo que corrió."""
    here = Path(__file__).parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=here,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=here,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None
    return commit, bool(status.strip())


def default_environment() -> Environment:
    """El entorno real: reloj de pared, esta máquina y ccxt de verdad."""
    return Environment(
        clock=lambda: datetime.now(UTC),
        hostname=socket.gethostname(),
        git_state=_git_state,
        exchange_factory=build_exchange,
    )


@dataclass(frozen=True, slots=True)
class SeriesResult:
    """Una serie de funding guardada, con su digest y su resumen."""

    path: Path
    digest: str
    rows: int
    first_ms: int | None
    last_ms: int | None
    summary: FundingSummary


@dataclass(frozen=True, slots=True)
class SymbolResult:
    """Todo lo que se averiguó de un símbolo, incluido lo que falló y por qué."""

    base: str
    symbol: str
    spec: ContractSpec | Absent
    ticker: Ticker24h | None
    ticker_error: str | None
    entry: Mapping[str, object] | None
    series: SeriesResult | None
    series_error: str | None


@dataclass(frozen=True, slots=True)
class ProbeReport:
    """Lo que el sondeo averiguó y de dónde salió."""

    exchange_id: str
    hostname: str
    started: datetime
    window_start: datetime
    window_end: datetime
    run_dir: Path
    precision_mode: int | None
    results: tuple[SymbolResult, ...]
    endpoints: tuple[EndpointCall, ...]
    capabilities: Mapping[str, Mapping[str, object]]
    load_error: str | None
    aborted: str | None

    @property
    def complete(self) -> bool:
        """Verdadero si no falló nada. Un símbolo que no existe es un hallazgo, no un fallo."""
        if self.load_error is not None or self.aborted is not None:
            return False
        return all(
            result.ticker_error is None and result.series_error is None for result in self.results
        )


def _stamp(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def _file_name(symbol: str) -> str:
    return symbol.replace(":", "_").replace("/", "_") + "_funding.csv"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


async def run_probe(
    exchange_id: str, *, out_dir: Path, env: Environment, days: int = HISTORY_DAYS
) -> ProbeReport:
    """Sondea un exchange y escribe `out_dir/<exchange>/<UTC inicio>/`. Nunca reusa un directorio.

    `meta.json` se escribe antes de la primera petición: la corrida que más hay que poder
    identificar es la que se interrumpe. Una sola petición por ventana y sin reintentos; con 429
    o 418 se aborta todo, con cualquier otro fallo solo ese símbolo queda sin determinar.
    """
    started = env.clock()
    window_end = started
    window_start = started - timedelta(days=days)
    run_dir = out_dir / exchange_id / _stamp(started)
    run_dir.mkdir(parents=True, exist_ok=False)
    commit, dirty = env.git_state()

    exchange = env.exchange_factory(exchange_id)
    log = EndpointLog()
    instrument(exchange, log, env.timer)
    meta: dict[str, object] = {
        "exchange": exchange_id,
        "hostname": env.hostname,
        "started": started.isoformat(),
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "days": days,
        "argv": sys.argv,
        "git_commit": commit,
        "git_dirty": dirty,
        "status": "running",
    }
    _write_json(run_dir / "meta.json", meta)

    results: list[SymbolResult] = []
    load_error: str | None = None
    aborted: str | None = None
    precision_mode: int | None = None
    all_capabilities: dict[str, Mapping[str, object]] = {}

    try:
        for other_id in EXCHANGES:
            other = exchange if other_id == exchange_id else env.exchange_factory(other_id)
            all_capabilities[other_id] = capabilities(other)
            if other is not exchange:
                await other.close()
        precision_mode = exchange.precisionMode
        try:
            markets = await exchange.load_markets()
        except Exception as error:  # ccxt levanta su propia jerarquía
            load_error = _error_text(error)
            markets = {}
            if log.last_status in RATE_LIMIT_STATUSES:
                aborted = f"{log.last_status} al cargar el catálogo de mercados"

        plan = _Plan(
            markets=markets,
            precision_mode=precision_mode,
            start_ms=int(window_start.timestamp() * 1000),
            end_ms=int(window_end.timestamp() * 1000),
            page_limit=FUNDING_PAGE_LIMIT[exchange_id],
            run_dir=run_dir,
            clock=env.clock,
        )
        for base in UNIVERSE_BASES if load_error is None else ():
            result, aborted = await _probe_symbol(exchange, log, base, plan)
            results.append(result)
            if aborted is not None:
                break
    finally:
        await exchange.close()

    series_digests = {
        result.symbol: result.series.digest for result in results if result.series is not None
    }
    entries = {result.symbol: dict(result.entry) for result in results if result.entry is not None}
    _write_json(run_dir / "spec.json", entries)
    meta.update(
        {
            "status": "complete" if load_error is None and aborted is None else "incomplete",
            "finished": env.clock().isoformat(),
            "series_sha256": series_digests,
            "load_error": load_error,
            "aborted": aborted,
            "endpoints": [
                {
                    "method": call.method,
                    "host": call.host,
                    "path": call.path,
                    "status": call.status,
                    "latency_ms": round(call.latency_ms, 3),
                    "error": call.error,
                }
                for call in log.calls
            ],
        }
    )
    _write_json(run_dir / "meta.json", meta)

    return ProbeReport(
        exchange_id=exchange_id,
        hostname=env.hostname,
        started=started,
        window_start=window_start,
        window_end=window_end,
        run_dir=run_dir,
        precision_mode=precision_mode,
        results=tuple(results),
        endpoints=tuple(log.calls),
        capabilities=all_capabilities,
        load_error=load_error,
        aborted=aborted,
    )


@dataclass(frozen=True, slots=True)
class _Plan:
    """Lo que cada símbolo necesita de la corrida: el catálogo, la ventana y dónde escribir."""

    markets: Mapping[str, Mapping[str, object]]
    precision_mode: int
    start_ms: int
    end_ms: int
    page_limit: int
    run_dir: Path
    clock: Callable[[], datetime]


async def _probe_symbol(
    exchange: _PerpExchange, log: EndpointLog, base: str, plan: _Plan
) -> tuple[SymbolResult, str | None]:
    """Un símbolo: contrato, volumen y funding. Devuelve también el motivo de abortar, si lo hay."""
    spec = describe_market(base, plan.markets, plan.precision_mode)
    symbol = perp_symbol(base)
    if isinstance(spec, Absent):
        return SymbolResult(base, symbol, spec, None, None, None, None, None), None

    ticker: Ticker24h | None = None
    ticker_error: str | None = None
    try:
        ticker = parse_ticker(await exchange.fetch_ticker(symbol))
    except Exception as error:  # ccxt levanta su propia jerarquía
        ticker_error = _error_text(error)
        if log.last_status in RATE_LIMIT_STATUSES:
            return _unfinished(base, symbol, spec, ticker_error), _abort_reason(log, symbol)

    entry: dict[str, object] = {
        "symbol": symbol,
        "contract_size": spec.contract_size,
        "tick_size": spec.tick_size,
        "min_amount": spec.min_amount,
        "min_cost": spec.min_cost,
        "quote_volume_24h": None if ticker is None else ticker.quote_volume,
        "base_volume_24h": None if ticker is None else ticker.base_volume,
        "ticker_ms": None if ticker is None else ticker.exchange_ms,
        "fetched_at": plan.clock().isoformat(),
    }

    series: SeriesResult | None = None
    series_error: str | None = None
    try:
        rows = await fetch_funding_series(
            exchange, symbol, plan.start_ms, plan.end_ms, plan.page_limit
        )
        content = series_csv(rows)
        path = plan.run_dir / _file_name(symbol)
        path.write_bytes(content)
        series = SeriesResult(
            path=path,
            digest=series_digest(content),
            rows=len(rows),
            first_ms=rows[0].timestamp_ms if rows else None,
            last_ms=rows[-1].timestamp_ms if rows else None,
            summary=summarise_funding(rows),
        )
    except Exception as error:  # ccxt levanta su propia jerarquía, y ProbeError la nuestra
        series_error = _error_text(error)
        if log.last_status in RATE_LIMIT_STATUSES:
            result = SymbolResult(
                base, symbol, spec, ticker, ticker_error, entry, None, series_error
            )
            return result, _abort_reason(log, symbol)

    result = SymbolResult(base, symbol, spec, ticker, ticker_error, entry, series, series_error)
    return result, None


def _unfinished(base: str, symbol: str, spec: ContractSpec, error: str) -> SymbolResult:
    return SymbolResult(base, symbol, spec, None, error, None, None, "no se llegó a pedir")


def _abort_reason(log: EndpointLog, symbol: str) -> str:
    return (
        f"HTTP {log.last_status} en {symbol}: no se sigue golpeando el exchange "
        "(binance banea la IP si se insiste)"
    )


# ───────────────────────────────────────── Informe ───────────────────────────────────────────


_UNDETERMINED = "no determinado"


def _percent(value: float | None, why: str) -> str:
    return f"{value * 100:.5f}%" if value is not None else f"{_UNDETERMINED}: {why}"


def _plain(value: float | None) -> str:
    return "no informado" if value is None else f"{value:g}"


def _volume(value: float | None) -> str:
    return "no informado" if value is None else f"{value:,.0f}"


def _fraction(part: int, whole: int) -> str:
    return f"{part}/{whole} ({100 * part / whole:.1f}%)" if whole else f"{_UNDETERMINED}: sin filas"


def _moment(milliseconds: int | None) -> str:
    if milliseconds is None:
        return "-"
    return datetime.fromtimestamp(milliseconds / 1000, UTC).strftime("%Y-%m-%d")


def _connectivity(calls: Sequence[EndpointCall]) -> list[str]:
    grouped: dict[tuple[str, str], list[EndpointCall]] = {}
    for call in calls:
        grouped.setdefault((call.host, call.path), []).append(call)
    lines = [
        f"{'endpoint':<52}{'peticiones':>11}  {'códigos':<18}{'ms med':>9}{'ms máx':>9}  errores"
    ]
    for (host, path), group in grouped.items():
        codes = Counter("sin respuesta" if c.status is None else str(c.status) for c in group)
        shown = " ".join(f"{code}:{count}" for code, count in sorted(codes.items()))
        errors = Counter(c.error for c in group if c.error)
        latencies = [c.latency_ms for c in group]
        lines.append(
            f"{host + path:<52}{len(group):>11}  {shown:<18}"
            f"{statistics.median(latencies):>9.1f}{max(latencies):>9.1f}  "
            f"{', '.join(f'{name}:{n}' for name, n in errors.items()) or '-'}"
        )
    return lines


def _contract_line(result: SymbolResult) -> str:
    spec = result.spec
    if isinstance(spec, Absent):
        return f"{result.symbol:<16}no existe: {spec.reason}"
    ticker = result.ticker
    undetermined = f"{_UNDETERMINED}: {result.ticker_error}"
    volume_base = _volume(ticker.base_volume) if ticker else undetermined
    volume_quote = _volume(ticker.quote_volume) if ticker else undetermined
    digest = entry_digest(result.entry)[:16] if result.entry is not None else "-"
    return (
        f"{result.symbol:<16}{_plain(spec.contract_size):>10}{_plain(spec.tick_size):>14}"
        f"{_plain(spec.min_amount):>12}{_plain(spec.min_cost):>14}  "
        f"{volume_base:>18}{volume_quote:>20}  {digest}"
    )


def _funding_line(result: SymbolResult) -> str:
    if isinstance(result.spec, Absent):
        return f"{result.symbol:<16}no existe: sin serie que resumir"
    series = result.series
    if series is None:
        return f"{result.symbol:<16}{_UNDETERMINED}: {result.series_error}"
    summary = series.summary
    intervals = " ".join(f"{hours}h:{count}" for hours, count in summary.intervals) or "-"
    p95_why = "menos de dos periodos"
    return (
        f"{result.symbol:<16}{series.rows:>6}{summary.normalised:>7}{summary.unattributable:>6}  "
        f"{intervals:<12}"
        f"{_percent(summary.mean_24h, 'sin periodos'):>12}"
        f"{_percent(summary.median_24h, 'sin periodos'):>12}"
        f"{_percent(summary.p95_24h, p95_why):>12}  "
        f"{_fraction(summary.negative, summary.periods):<19}"
        f"{_moment(series.first_ms)}..{_moment(series.last_ms)}  {series.digest[:16]}"
    )


def render_report(report: ProbeReport) -> str:
    """El informe como texto. Cada fila con cifras termina en el digest de su origen."""
    lines = [
        f"== Sondeo de perpetuos USDT: {report.exchange_id} ==",
        f"máquina: {report.hostname}    inicio: {report.started.isoformat()}",
        f"ventana de funding: {report.window_start.date()} → {report.window_end.date()}",
        f"salida: {report.run_dir}",
        "",
    ]
    if report.aborted is not None:
        lines += [f"ABORTADO: {report.aborted}", ""]
    if report.load_error is not None:
        lines += [f"CATÁLOGO NO CARGADO: {report.load_error}", ""]

    lines += ["-- Conectividad (una petición HTTP por fila de origen; sin cabeceras ni query)"]
    lines += _connectivity(report.endpoints) or ["(ninguna petición)"]

    mode = "TICK_SIZE" if report.precision_mode == TICK_SIZE else f"modo {report.precision_mode}"
    lines += [
        "",
        f"-- Contratos (precisionMode={mode}; de market['limits'] y del ticker de 24 h)",
        f"{'símbolo':<16}{'tamaño':>10}{'tick':>14}{'cant mín':>12}{'nominal mín':>14}  "
        f"{'vol24h base':>18}{'vol24h USDT':>20}  digest",
    ]
    lines += [_contract_line(result) for result in report.results] or ["(sin contratos)"]

    lines += [
        "",
        "-- Funding normalizado a 24 h (tasa * 24 / intervalo del periodo, en % por 24 h)",
        f"{'símbolo':<16}{'filas':>6}{'norm':>7}{'excl':>6}  {'intervalos':<12}"
        f"{'media':>12}{'mediana':>12}{'p95':>12}  {'negativos':<19}{'rango':<24}digest",
    ]
    lines += [_funding_line(result) for result in report.results] or ["(sin series)"]

    names = list(report.capabilities)
    lines += [
        "",
        "-- Capacidades (exchange.has, valor crudo de ccxt; no prueba permisos de cuenta)",
    ]
    lines.append(f"{'capacidad':<26}" + "".join(f"{name:<16}" for name in names))
    for capability in CAPABILITIES:
        cells = "".join(f"{report.capabilities[name].get(capability)!s:<16}" for name in names)
        lines.append(f"{capability:<26}{cells}")

    lines += ["", "-- Procedencia (sha-256 de cada archivo guardado; verificable con sha256sum)"]
    for result in report.results:
        if result.series is not None:
            lines.append(f"{result.symbol:<16}{result.series.digest}  {result.series.path.name}")
    lines.append(f"{'meta.json':<16}se escribe antes de la primera petición y al final")
    return "\n".join(lines) + "\n"


# ────────────────────────────────────────── CLI ──────────────────────────────────────────────


def main(argv: Sequence[str] | None = None, env: Environment | None = None) -> int:
    """Punto de entrada. Sale con 0 si todo contestó y con 1 si algo falló o se abortó."""
    parser = argparse.ArgumentParser(
        description="Sondeo público de perpetuos USDT (solo lectura, sin claves)"
    )
    parser.add_argument("--exchange", required=True, choices=EXCHANGES)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--days", type=int, default=HISTORY_DAYS)
    args = parser.parse_args(argv)

    report = asyncio.run(
        run_probe(
            args.exchange,
            out_dir=args.out_dir,
            env=env if env is not None else default_environment(),
            days=args.days,
        )
    )
    print(render_report(report), end="")
    return 0 if report.complete else 1


if __name__ == "__main__":
    sys.exit(main())
