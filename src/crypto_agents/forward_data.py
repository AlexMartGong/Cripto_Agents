"""Los datos del estudio hacia adelante: la etapa 1 versionada, las series extendidas, la población.

`docs/forward.md` fija de dónde salen los datos de cada lote, y este módulo los prepara sin llamar a
ningún modelo:

- **La etapa 1, versionada.** `data/stage1/` es la copia, byte a byte, de los seis archivos de
  `var/ablation/20261009T045033Z` que leen las comprobaciones de cada lote (apartado 10, punto 9).
  `verify_stage1` comprueba sus sha-256 antes de leer nada.
- **Las series extendidas.** El histórico de velas y el funding crecen solo por la derecha
  (apartado 4). Lo guardado nunca se reescribe: se añaden filas al final de sus bytes. Cada
  extensión vuelve a descargar la serie entera desde 2024-08-16 y la compara fila a fila con lo
  guardado.
  - Una vela revisada detiene la extensión hasta que un flag la nombra, y entonces la serie
    conserva el valor guardado.
  - Una vela nueva que falta también detiene la extensión; si sigue faltando 24 h después, el
    lote declara el hueco persistente.
  - Una liquidación de funding revisada detiene la extensión sin más.
- **La población de un bloque** (apartado 2). Es la definición del pool sobre el que se planificó,
  con las mismas funciones:
  - el barrido (`activation_sweep.activated_bars`);
  - la regla del horizonte (`selection.stratify`);
  - el bloque del cierre de la vela (`metrics.block_of`);
  - la confirmación con la ventana del replay (`selection.confirm_activation`).

  El manifiesto del bloque es un `SelectionManifest`, así que lo leen sin cambios las puertas de
  la ablación: `verify_histories`, `plan_from_manifest`, `audit.load_plan` y
  `forward_plan.batch_spend_cap`.

Este módulo no importa el router ni nada que lo traiga. Dos tests de arquitectura lo fijan: uno mira
sus imports y otro lo importa en un proceso nuevo.
"""

from __future__ import annotations

import csv
import io
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from pydantic import AwareDatetime, Field

from crypto_agents.activation import DEFAULT_CONFIG, ActivationConfig
from crypto_agents.activation_sweep import (
    DEFAULT_SYMBOLS,
    activated_bars,
    history_path,
    load_series,
)
from crypto_agents.audit import file_sha256
from crypto_agents.criteria import CLUSTER_DAYS
from crypto_agents.funding import FUNDING_DIR, funding_path
from crypto_agents.indicators import DEFAULT_PRESET, IndicatorPreset
from crypto_agents.market import read_ohlcv_csv, timeframe_to_timedelta
from crypto_agents.metrics import EPOCH, block_of
from crypto_agents.perp_probe import default_environment, run_probe
from crypto_agents.selection import (
    DEFAULT_CANDLE_LIMIT,
    DEFAULT_HORIZON,
    DEFAULT_SEED,
    DEFAULT_TIMEFRAME,
    HISTORY_DIR,
    PlannedEvaluation,
    SelectionManifest,
    confirm_activation,
    load_manifest,
    series_ref,
    stratify,
    window_digest,
    write_manifest,
)
from crypto_agents.state import FrozenModel

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence

    from crypto_agents.activation_sweep import ActivatedBar

__all__ = [
    "BATCH_DIR",
    "FORWARD_DIR",
    "FORWARD_FUNDING_DIR",
    "FORWARD_HISTORY_DIR",
    "MANIFEST_DIR",
    "PUBLISHED_FUNDING_SHA256",
    "PUBLISHED_HISTORY_SHA256",
    "REVISIONS_FILE",
    "SERIES_START",
    "STAGE1_DIR",
    "STAGE1_FILES",
    "AcceptedRevision",
    "CandleFetch",
    "CandleRef",
    "ForwardDataError",
    "FundingFetch",
    "HistoryExtension",
    "MissingCandle",
    "MissingCandlesError",
    "RevisedCandle",
    "RevisedCandlesError",
    "RevisionLedger",
    "SeriesFile",
    "StudyPopulation",
    "block_manifest",
    "block_start_of",
    "extend_funding",
    "extend_history",
    "fetch_perp_funding",
    "fetch_spot_candles",
    "manifest_path",
    "ohlcv_lines",
    "read_histories",
    "read_revisions",
    "same_population",
    "study_population",
    "verify_stage1",
    "write_block_manifest",
]

STAGE1_DIR = Path("data/stage1")
"""La copia versionada de la etapa 1: lo que leen las comprobaciones de cada lote."""

STAGE1_FILES: Mapping[str, str] = {
    "meta.json": "a9d4b2c3ff84a5c895eae2afa366e58bc89838758a0a9471cf5b2f9214aae06c",
    "solo.jsonl": "1180554e2c19aab20d623f2112547aae8f930ad934d3fc17ca65fd5c70cc4b17",
    "always_buy.jsonl": "77fe118e5efb5714384c3d3485baf31a9c80bdbded2c34f2b775c9667fddd1ff",
    "always_sell.jsonl": "ba1643fb2ce45114f728418e8894eb46b55f125fe2cef1fe0211e273746bc1d9",
    "random_uniform.jsonl": "607b738ea2ee764d4ee8433adc0cac748f738761dfcea418f1a6690cd6ff6198",
    "rule_trend.jsonl": "3b8d58f9ce1b22699a7f0d92b2dd47e91235c3db9f0ef3b53d3bed215b19ab93",
}
"""Los seis archivos y sus sha-256 publicados: `docs/forward.md`, apartado 10, punto 9."""

FORWARD_DIR = Path("data/forward")
FORWARD_HISTORY_DIR = FORWARD_DIR / "history"
"""Las velas extendidas, con el nombre de `history_path`: las lee `load_selection_histories`."""

FORWARD_FUNDING_DIR = FORWARD_DIR / "funding"
"""El funding extendido, con el nombre de `funding_path`: lo lee `load_funding`."""

MANIFEST_DIR = FORWARD_DIR / "manifests"
BATCH_DIR = FORWARD_DIR / "batches"

DOWNLOAD_DIR = Path("var/forward/download")
"""Donde `load_series` deja su copia de cada descarga. No se versiona y nada la lee."""

PERP_DIR = Path("var/forward/perp")
"""Donde `perp_probe` deja cada descarga de funding, con su `meta.json`. No se versiona."""

SERIES_START = datetime(2024, 8, 16, tzinfo=UTC)
"""El primer instante de `data/history/` y de `data/funding/`: las extensiones empiezan aquí."""

PUBLISHED_HISTORY_SHA256: Mapping[str, str] = {
    "ADA/USDT": "93b6462816a9f606450e5b5a1440b1b2cac46ef0ede9a56b1700d26c1b58255d",
    "BNB/USDT": "65ede8a9b4500197986bc5e1f922c5f59a6f33f39b359e0550dc6be080cb8375",
    "BTC/USDT": "d1b7e107de43199c7b80e51afd66c660f1f89dc98c2d0f639c580f325b225ad4",
    "DOGE/USDT": "5c264c914a0db15d5563e7dc47f3db9fb4fbbde8268364ea11baf319ec4ecfff",
    "ETH/USDT": "5ea175a55e90112f6f10510e4fe67d328dfd2795f6a9571e49e6f29ac34c5404",
    "SOL/USDT": "56517823a5b88b279ed3c15f2d78d491f15ddfa0fcf1280f17e19e436f1dfc3e",
    "XRP/USDT": "e907bb3e44bc5587b0736e6cc4604f15fae36b3dc46cc185bf3566cb1662348a",
}
"""Los sha-256 de `data/history/`, publicados en `docs/forward.md`, apartado 9."""

REVISIONS_FILE = FORWARD_DIR / "revisions.json"
"""Las velas revisadas que el operador aceptó (`docs/forward.md`, apartado 4, «Vela revisada»).

Cada una lleva el valor guardado y el devuelto. Se versiona con los datos del estudio. Cada
extensión lo lee, y una vela que figura aquí se acepta sola mientras el exchange devuelva
exactamente el valor registrado: si la cambia otra vez, es una revisión nueva y la extensión se
detiene.
"""

PUBLISHED_FUNDING_SHA256: Mapping[str, str] = {
    "ADA/USDT": "c0a49ea47cad58ac40cb5e19a2efd15381e2e6e8070e597763f487db9e7733e8",
    "BNB/USDT": "510211daa2d56b2279b8533f0632d73cab019734689f93c0d1f54533f9173fbb",
    "BTC/USDT": "3ae0e01844e4b9f17281da6ab8205804ff064c44107558615a0739be48cc84ac",
    "DOGE/USDT": "f50ec881651eac0eead303ad34fe03eb92667614eebd2ba3ad0852b8ffd22c08",
    "ETH/USDT": "02f6e0c5cbccfc6b5eb2e3e35e88d2beebad43553ec210950ff56bca6b07f888",
    "SOL/USDT": "bd5b061ec461882f58679769417fb6c9da6f953a474c3bd219061e5190268fc0",
    "XRP/USDT": "d0972913d4bdfa49eb3809605221dadede1bb815cf7d0b6a8a1cf68dcbb33329",
}
"""Los sha-256 de `data/funding/`, publicados en su README."""

FUNDING_EXCHANGE = "binanceusdm"

type CandleFetch = Callable[[str, str, datetime, datetime], Awaitable[list[list[float]]]]
"""Velas de `[inicio, fin)` de un símbolo y una temporalidad, como las da `download_history`."""

type FundingFetch = Callable[[datetime, datetime], Awaitable[Mapping[str, bytes]]]
"""El funding de `[inicio, fin]`, con los dos extremos incluidos: los bytes del CSV de cada símbolo,
tal como los escribe `perp_probe`."""


class ForwardDataError(RuntimeError):
    """Los datos del estudio no son los que tienen que ser: un archivo publicado que cambió, una
    vela revisada, una vela que falta o una serie que no cubre lo que se le pide."""


class SeriesFile(FrozenModel):
    """Un archivo extendido después de una extensión: lo que el `meta` del lote deja escrito."""

    symbol: str = Field(min_length=1)
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rows: int = Field(gt=0)
    first: AwareDatetime
    last: AwareDatetime
    """La apertura de la última vela, o la marca de la última liquidación."""


class CandleRef(NamedTuple):
    """Una vela por su símbolo y su apertura: lo que nombra el flag que acepta una revisión."""

    symbol: str
    opens: datetime


class RevisedCandle(FrozenModel):
    """Una vela guardada que el exchange devuelve distinta, o que ya no devuelve."""

    symbol: str = Field(min_length=1)
    opens: AwareDatetime
    stored: tuple[float, ...]
    """Apertura, máximo, mínimo, cierre y volumen guardados."""

    returned: tuple[float, ...] | None
    """Lo que devolvió el exchange en esta descarga. `None` si ya no la devuelve."""


class AcceptedRevision(RevisedCandle):
    """Una vela revisada que el operador aceptó con el flag: la serie conserva `stored`."""

    accepted_at: AwareDatetime
    block: AwareDatetime
    """El inicio del bloque cuyo lote la aceptó."""


class RevisionLedger(FrozenModel):
    """Lo que guarda `REVISIONS_FILE`."""

    accepted: tuple[AcceptedRevision, ...] = ()


class MissingCandle(FrozenModel):
    """Una vela nueva que la descarga no trae."""

    symbol: str = Field(min_length=1)
    opens: AwareDatetime


class RevisedCandlesError(ForwardDataError):
    """Velas guardadas que el exchange devuelve distintas y nadie aceptó. La extensión se detiene
    sin escribir nada; el operador las anota en el registro y vuelve con el flag."""

    def __init__(self, revised: Sequence[RevisedCandle]) -> None:
        self.revised = tuple(revised)
        names = ", ".join(f"{item.symbol} {item.opens.isoformat()}" for item in self.revised)
        super().__init__(
            f"el exchange devuelve distintas {len(self.revised)} velas guardadas: {names}"
        )


class MissingCandlesError(ForwardDataError):
    """Velas nuevas que la descarga no trae. La extensión se detiene sin escribir nada; si siguen
    faltando al volver a descargar 24 h después, el hueco es persistente."""

    def __init__(self, missing: Sequence[MissingCandle]) -> None:
        self.missing = tuple(missing)
        names = ", ".join(f"{item.symbol} {item.opens.isoformat()}" for item in self.missing)
        super().__init__(f"faltan {len(self.missing)} velas nuevas: {names}")


class HistoryExtension(NamedTuple):
    """Lo que deja una extensión de las velas."""

    files: tuple[SeriesFile, ...]
    accepted: tuple[AcceptedRevision, ...]
    """Las revisiones que esta extensión aceptó por primera vez, por el flag."""


# ───────────────────────────────────────── La etapa 1 ─────────────────────────────────────────────


def verify_stage1(directory: Path = STAGE1_DIR) -> None:
    """Comprueba que los seis archivos de la etapa 1 son los publicados, antes de leer ninguno."""
    for name, expected in STAGE1_FILES.items():
        path = directory / name
        if not path.is_file():
            raise ForwardDataError(f"falta {path}: la copia de la etapa 1 no está completa")
        actual = file_sha256(path)
        if actual != expected:
            raise ForwardDataError(
                f"{path}: sha-256 {actual[:12]}…, y el publicado es {expected[:12]}…; no es la "
                "etapa 1"
            )


# ──────────────────────────────────────── Las velas ───────────────────────────────────────────────


async def fetch_spot_candles(
    symbol: str, timeframe: str, start: datetime, end: datetime
) -> list[list[float]]:
    """Las velas de Binance spot por el camino de `activation_sweep.load_series`.

    Ese camino es `CcxtMarketClient("binance")` y `download_history`: lectura pública, sin
    credenciales y solo con los mercados spot. Con `refresh=True` siempre descarga. Deja una copia
    en `DOWNLOAD_DIR`, que no se versiona; de aquí solo salen las filas que devuelve.
    """
    return await load_series(symbol, timeframe, start, end, directory=DOWNLOAD_DIR, refresh=True)


def ohlcv_lines(rows: Sequence[Sequence[float]]) -> bytes:
    """Filas de velas con el formato byte a byte de `data/history/`: `\\n` como fin de línea, el
    timestamp entero y el `repr` de cada valor.

    `market.write_ohlcv_csv` escribe `\\r\\n`: con ella, añadir una fila reescribiría las
    publicadas con otros bytes.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    for row in rows:
        writer.writerow([int(row[0]), *(float(value) for value in row[1:])])
    return buffer.getvalue().encode("utf-8")


def _published_bytes(path: Path, expected: str) -> bytes:
    """Los bytes de un archivo publicado, tras comprobar que siguen siendo los publicados."""
    if not path.is_file():
        raise ForwardDataError(f"falta el archivo publicado {path}")
    content = path.read_bytes()
    actual = file_sha256(path)
    if actual != expected:
        raise ForwardDataError(
            f"{path}: sha-256 {actual[:12]}…, y el publicado es {expected[:12]}…; el histórico "
            "publicado no se toca"
        )
    return content


def _base_bytes(current: Path, published: bytes) -> bytes:
    """Lo guardado: la serie extendida si existe, o la publicada si todavía no se extendió."""
    if not current.is_file():
        return published
    content = current.read_bytes()
    if not content.startswith(published):
        raise ForwardDataError(
            f"{current} no empieza por los bytes publicados: la serie extendida se editó"
        )
    return content


def _moment(milliseconds: float) -> datetime:
    return datetime.fromtimestamp(int(milliseconds) / 1000, UTC)


def _write_atomically(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _ms(moment: datetime) -> int:
    return int(moment.timestamp() * 1000)


def read_revisions(path: Path = REVISIONS_FILE) -> RevisionLedger:
    """Las revisiones aceptadas hasta ahora; ninguna si el archivo no existe todavía."""
    if not path.is_file():
        return RevisionLedger()
    return RevisionLedger.model_validate_json(path.read_text(encoding="utf-8"))


async def extend_history(
    until: datetime,
    fetch: CandleFetch = fetch_spot_candles,
    *,
    now: datetime,
    accept: Collection[CandleRef] = (),
    accepted_at: datetime | None = None,
    block: datetime | None = None,
    symbols: Sequence[str] = DEFAULT_SYMBOLS,
    timeframe: str = DEFAULT_TIMEFRAME,
    directory: Path = FORWARD_HISTORY_DIR,
    published: Path = HISTORY_DIR,
    published_sha256: Mapping[str, str] = PUBLISHED_HISTORY_SHA256,
    revisions: Path = REVISIONS_FILE,
) -> HistoryExtension:
    """Extiende las velas de cada símbolo hasta la que cierra en `until`, solo por la derecha.

    La serie se vuelve a descargar entera, desde `SERIES_START`, y cada vela guardada se compara con
    la descargada. Lo guardado no se reescribe nunca: las filas nuevas se añaden al final de sus
    bytes, así que las primeras 4 380 son siempre las de `data/history/`. Si el archivo ya llega
    más lejos (un lote de un bloque posterior corrió antes), se verifica hasta donde llega y no se
    recorta.

    Las dos reglas del apartado 4:

    - **Vela revisada.** Una vela guardada que el exchange devuelve distinta, o ya no devuelve,
      detiene la extensión (`RevisedCandlesError`). Si `accept` la nombra, se acepta: la serie
      conserva el valor guardado y la aceptación queda en `revisions`, con el valor guardado y el
      devuelto. Una vela ya aceptada se acepta sola mientras el exchange devuelva el valor
      registrado. Un flag que nombra una vela que no está revisada se rechaza.
    - **Velas que faltan.** Una vela nueva que no viene detiene la extensión
      (`MissingCandlesError`). Decidir si el hueco es persistente es del lote, que guarda cuándo lo
      vio.

    **Solo velas cerradas.** `now` es el instante actual, inyectado como en
    `market.drop_forming_candle`. Con `until` posterior a `now`, la vela que cierra en `until` está
    todavía en formación y el exchange la devuelve igual: así se guardó la fila 4 380 de
    `data/history/`. Se niega antes de descargar, con un error y no como vela que falta, porque no
    es un hueco del exchange y no debe contar para el hueco persistente.

    Nada se escribe hasta que las siete series pasaron.
    """
    if until > now:
        raise ForwardDataError(
            f"la vela que cierra en {until.isoformat()} sigue en formación a las "
            f"{now.isoformat()}: la extensión solo escribe velas cerradas"
        )
    if accept and (accepted_at is None or block is None):
        raise ValueError("aceptar una revisión exige el instante y el bloque que la aceptan")
    period = timeframe_to_timedelta(timeframe)
    step_ms = int(period.total_seconds() * 1000)
    ledger = read_revisions(revisions)
    known = {(item.symbol, _ms(item.opens)): item for item in ledger.accepted}
    wanted = {(ref.symbol, _ms(ref.opens)) for ref in accept}
    named: set[tuple[str, int]] = set()
    fresh_acceptances: list[AcceptedRevision] = []
    revised: list[RevisedCandle] = []
    missing: list[MissingCandle] = []
    verified: list[tuple[str, Path, bytes, bytes, int, int, int]] = []

    for symbol in sorted(symbols):
        source = history_path(symbol, timeframe, published)
        if symbol not in published_sha256:
            raise ForwardDataError(f"{symbol}: no hay sha-256 publicado para su histórico")
        base = _published_bytes(source, published_sha256[symbol])
        target = history_path(symbol, timeframe, directory)
        stored_bytes = _base_bytes(target, base)
        stored = read_ohlcv_csv(target if target.is_file() else source)

        end = max(until, _moment(stored[-1][0]) + period)
        downloaded = await fetch(symbol, timeframe, SERIES_START, end)
        fresh = {
            int(row[0]): [int(row[0]), *(float(value) for value in row[1:6])] for row in downloaded
        }
        kept_times = [int(row[0]) for row in stored]
        new_times = list(range(kept_times[-1] + step_ms, _ms(end), step_ms))
        stray = sorted(set(fresh) - set(kept_times) - set(new_times))
        if stray:
            raise ForwardDataError(
                f"{symbol}: la descarga trae velas fuera de la rejilla de {timeframe}: "
                + ", ".join(_moment(moment).isoformat() for moment in stray[:3])
            )

        for kept in stored:
            moment = int(kept[0])
            row = fresh.get(moment)
            if row == [moment, *kept[1:]]:
                continue
            candle = RevisedCandle(
                symbol=symbol,
                opens=_moment(moment),
                stored=tuple(kept[1:]),
                returned=None if row is None else tuple(row[1:]),
            )
            key = (symbol, moment)
            previous = known.get(key)
            if (
                previous is not None
                and previous.stored == candle.stored
                and previous.returned == candle.returned
            ):
                named.add(key)
                continue
            if key in wanted:
                assert accepted_at is not None and block is not None
                named.add(key)
                fresh_acceptances.append(
                    AcceptedRevision(**candle.model_dump(), accepted_at=accepted_at, block=block)
                )
                continue
            revised.append(candle)

        appended: list[list[float]] = []
        for moment in new_times:
            row = fresh.get(moment)
            if row is None:
                missing.append(MissingCandle(symbol=symbol, opens=_moment(moment)))
            else:
                appended.append(row)
        content = stored_bytes + ohlcv_lines(appended)
        if not content.startswith(base):
            raise ForwardDataError(f"{symbol}: la serie extendida no empieza por la publicada")
        last = new_times[-1] if new_times else kept_times[-1]
        verified.append(
            (
                symbol,
                target,
                stored_bytes,
                content,
                len(stored) + len(appended),
                kept_times[0],
                last,
            )
        )

    if revised:
        raise RevisedCandlesError(revised)
    unknown = sorted(wanted - named)
    if unknown:
        raise ForwardDataError(
            "el flag nombra velas que el exchange no devuelve distintas: "
            + ", ".join(f"{symbol} {_moment(moment).isoformat()}" for symbol, moment in unknown)
        )
    if missing:
        raise MissingCandlesError(missing)

    # Se escribe cuando las siete pasaron: una serie revisada no deja a las demás extendidas.
    files: list[SeriesFile] = []
    for symbol, target, stored_bytes, content, rows, first, last in verified:
        if not target.is_file() or content != stored_bytes:
            _write_atomically(target, content)
        files.append(
            SeriesFile(
                symbol=symbol,
                path=str(target),
                sha256=file_sha256(target),
                rows=rows,
                first=_moment(first),
                last=_moment(last),
            )
        )
    if fresh_acceptances:
        updated = RevisionLedger(accepted=(*ledger.accepted, *fresh_acceptances))
        _write_atomically(revisions, (updated.model_dump_json(indent=2) + "\n").encode("utf-8"))
    return HistoryExtension(files=tuple(files), accepted=tuple(fresh_acceptances))


def read_histories(
    directory: Path = FORWARD_HISTORY_DIR,
    symbols: Sequence[str] = DEFAULT_SYMBOLS,
    timeframe: str = DEFAULT_TIMEFRAME,
) -> dict[str, list[list[float]]]:
    """Las velas extendidas de cada símbolo, por el mismo lector que el resto del paquete."""
    histories: dict[str, list[list[float]]] = {}
    for symbol in sorted(symbols):
        path = history_path(symbol, timeframe, directory)
        if not path.is_file():
            raise ForwardDataError(f"falta el histórico extendido de {symbol}: {path}")
        histories[symbol] = read_ohlcv_csv(path)
    return histories


# ──────────────────────────────────────── El funding ──────────────────────────────────────────────


async def fetch_perp_funding(start: datetime, end: datetime) -> dict[str, bytes]:
    """El funding de los siete perpetuos por `perp_probe`, la descarga de `data/funding/`.

    Es `perp_probe --exchange binanceusdm --start … --end …`, con los dos extremos incluidos:
    lectura pública, sin claves y sin órdenes. Escribe su directorio en `PERP_DIR`, que no se
    versiona. Una serie que no llegó es un error, no una serie vacía.
    """
    report = await run_probe(
        FUNDING_EXCHANGE, out_dir=PERP_DIR, env=default_environment(), start=start, end=end
    )
    if report.aborted is not None or report.load_error is not None:
        raise ForwardDataError(
            f"el funding no se pudo descargar: {report.aborted or report.load_error}"
        )
    series: dict[str, bytes] = {}
    for result in report.results:
        if result.series is None:
            raise ForwardDataError(
                f"{result.symbol}: sin serie de funding ({result.series_error or 'sin motivo'})"
            )
        series[f"{result.base}/USDT"] = result.series.path.read_bytes()
    return series


def _first_difference(stored: bytes, fresh: bytes) -> str:
    """La primera línea en que la descarga no repite lo guardado, para nombrarla en el error."""
    for number, (kept, new) in enumerate(
        zip(stored.decode().splitlines(), fresh.decode().splitlines(), strict=False), start=1
    ):
        if kept != new:
            return f"línea {number}: guardada {kept!r}, ahora {new!r}"
    return "la descarga es más corta que lo guardado"


def _funding_rows(content: bytes) -> list[tuple[int, str]]:
    lines = content.decode("utf-8").splitlines()[1:]
    return [(int(line.split(",", 1)[0]), line) for line in lines if line]


async def extend_funding(
    until: datetime,
    fetch: FundingFetch = fetch_perp_funding,
    *,
    symbols: Sequence[str] = DEFAULT_SYMBOLS,
    directory: Path = FORWARD_FUNDING_DIR,
    published: Path = FUNDING_DIR,
    published_sha256: Mapping[str, str] = PUBLISHED_FUNDING_SHA256,
) -> tuple[SeriesFile, ...]:
    """Extiende el funding de cada símbolo hasta `until`, incluido, solo por la derecha.

    Se descarga la serie entera desde `SERIES_START` y sus bytes tienen que empezar por los
    guardados: los de `data/funding/` y los de la extensión anterior. `perp_probe` escribe un CSV
    canónico, ordenado y con el `repr` de cada tasa, así que una liquidación revisada, una que falta
    o una marca con otro jitter rompen el prefijo y detienen la extensión. Si lo guardado ya llega
    más lejos que `until`, se verifica hasta ahí y no se recorta.
    """
    bases: dict[str, bytes] = {}
    stored: dict[str, bytes] = {}
    last = until
    for symbol in sorted(symbols):
        if symbol not in published_sha256:
            raise ForwardDataError(f"{symbol}: no hay sha-256 publicado para su funding")
        base = _published_bytes(funding_path(symbol, published), published_sha256[symbol])
        current = _base_bytes(funding_path(symbol, directory), base)
        bases[symbol], stored[symbol] = base, current
        rows = _funding_rows(current)
        if rows:
            last = max(last, datetime.fromtimestamp(rows[-1][0] / 1000, UTC))

    downloaded = await fetch(SERIES_START, last)
    for symbol in sorted(symbols):
        content = downloaded.get(symbol)
        if content is None:
            raise ForwardDataError(f"{symbol}: la descarga de funding no trae su serie")
        if not content.startswith(stored[symbol]):
            raise ForwardDataError(
                f"{symbol}: el funding descargado no repite lo guardado; "
                f"{_first_difference(stored[symbol], content)}"
            )

    # Como las velas: se escribe cuando las siete pasaron.
    files: list[SeriesFile] = []
    for symbol in sorted(symbols):
        content = downloaded[symbol]
        target = funding_path(symbol, directory)
        if not target.is_file() or content != stored[symbol]:
            _write_atomically(target, content)
        rows = _funding_rows(content)
        files.append(
            SeriesFile(
                symbol=symbol,
                path=str(target),
                sha256=file_sha256(target),
                rows=len(rows),
                first=datetime.fromtimestamp(rows[0][0] / 1000, UTC),
                last=datetime.fromtimestamp(rows[-1][0] / 1000, UTC),
            )
        )
    return tuple(files)


# ─────────────────────────────────────── La población ─────────────────────────────────────────────


def block_start_of(moment: datetime) -> datetime:
    """El inicio del bloque de 7 días, alineado al epoch, en que cae ese instante."""
    return EPOCH + block_of(moment, CLUSTER_DAYS) * timedelta(days=CLUSTER_DAYS)


class StudyPopulation(NamedTuple):
    """La población de varios bloques, y lo que el barrido encontró en cada serie entera."""

    blocks: dict[datetime, tuple[PlannedEvaluation, ...]]
    """Por inicio de bloque, sus activaciones confirmadas, por símbolo y vela."""

    activations: dict[str, int]
    """Activaciones del barrido sobre cada serie entera: las que lleva `SeriesRef`."""


def study_population(
    histories: Mapping[str, Sequence[Sequence[float]]],
    starts: Sequence[datetime],
    *,
    timeframe: str = DEFAULT_TIMEFRAME,
    horizon: int = DEFAULT_HORIZON,
    candle_limit: int = DEFAULT_CANDLE_LIMIT,
    config: ActivationConfig = DEFAULT_CONFIG,
    preset: IndicatorPreset = DEFAULT_PRESET,
) -> StudyPopulation:
    """La población de cada bloque que empieza en `starts`, según el apartado 2.

    Entran las activaciones del barrido sobre el histórico extendido que cumplen tres cosas:
    - tienen el horizonte por delante (`selection.stratify` con un solo tramo, que es su regla de
      elegibilidad sin repartir);
    - su instante evaluado, el cierre de la vela, cae en el bloque;
    - el gate las confirma con la ventana del replay (`selection.confirm_activation`).

    Es lo que hace `design_effect.pool_records`, la población de `forward_plan`, sin decidir las
    líneas base. Sin muestreo. Un bloque cuyo horizonte el histórico no cubre se rechaza en vez de
    devolverse corto: le faltarían sus últimas activaciones.
    """
    period = timeframe_to_timedelta(timeframe)
    step = timedelta(days=CLUSTER_DAYS)
    wanted: dict[int, datetime] = {}
    for start in starts:
        if start.tzinfo is None or (start - EPOCH) % step:
            raise ForwardDataError(f"{start.isoformat()} no es el inicio de un bloque")
        wanted[block_of(start, CLUSTER_DAYS)] = start

    found_by_symbol: dict[str, tuple[ActivatedBar, ...]] = {}
    for symbol in sorted(histories):
        rows = histories[symbol]
        covered = _moment(rows[-1][0]) + period
        latest = max(wanted.values(), default=None)
        if latest is not None and covered < latest + step + horizon * period:
            raise ForwardDataError(
                f"{symbol}: las velas llegan a {covered.isoformat()} y el bloque de "
                f"{latest.isoformat()} necesita hasta "
                f"{(latest + step + horizon * period).isoformat()}"
            )
        found_by_symbol[symbol] = activated_bars(rows, symbol, timeframe, config, preset)

    blocks: dict[datetime, list[PlannedEvaluation]] = {start: [] for start in starts}
    for symbol, found in found_by_symbol.items():
        rows = histories[symbol]
        (eligible,) = stratify(rows, found, 1, horizon)
        for bar in eligible:
            block = wanted.get(block_of(bar.at + period, CLUSTER_DAYS))
            if block is None:
                continue
            triggers = confirm_activation(rows, bar.index, candle_limit, config, preset)
            if triggers is None:
                continue
            blocks[block].append(
                PlannedEvaluation(
                    symbol=symbol,
                    timeframe=timeframe,
                    index=bar.index,
                    at=bar.at,
                    candles_digest=window_digest(rows, bar.index, candle_limit),
                    triggers=triggers,
                    stratum=0,
                )
            )
    return StudyPopulation(
        blocks={
            start: tuple(sorted(entries, key=lambda entry: (entry.symbol, entry.index)))
            for start, entries in blocks.items()
        },
        activations={symbol: len(found) for symbol, found in found_by_symbol.items()},
    )


def block_manifest(
    histories: Mapping[str, Sequence[Sequence[float]]],
    population: StudyPopulation,
    start: datetime,
    created_at: datetime,
    *,
    timeframe: str = DEFAULT_TIMEFRAME,
    horizon: int = DEFAULT_HORIZON,
    candle_limit: int = DEFAULT_CANDLE_LIMIT,
    seed: int = DEFAULT_SEED,
    preset: IndicatorPreset = DEFAULT_PRESET,
) -> SelectionManifest | None:
    """El manifiesto del bloque: toda su población, o `None` si está vacío.

    Los parámetros salen de las constantes del código, que son las de la ablación. El lote los
    compara después con `data/ablation_selection.json`: si alguien cambiara una constante, esa
    comparación lo detendría antes de llamar. Un tramo y `target` igual al número de entradas: aquí
    no se elige, se toma todo. Un bloque vacío no tiene manifiesto, porque `SelectionManifest` exige
    al menos una entrada, y lo registra el `meta` de su lote.
    """
    entries = population.blocks[start]
    if not entries:
        return None
    return SelectionManifest(
        created_at=created_at,
        seed=seed,
        target=len(entries),
        strata=1,
        timeframe=timeframe,
        warmup_bars=preset.min_bars,
        candle_limit=candle_limit,
        horizon=horizon,
        series=tuple(
            series_ref(symbol, timeframe, histories[symbol], population.activations[symbol])
            for symbol in sorted(histories)
        ),
        entries=entries,
    )


def manifest_path(start: datetime, directory: Path = MANIFEST_DIR) -> Path:
    """`data/forward/manifests/<YYYY-MM-DD>.json`, por el inicio del bloque."""
    return directory / f"{start:%Y-%m-%d}.json"


def same_population(saved: SelectionManifest, recomputed: SelectionManifest) -> bool:
    """Si un manifiesto guardado es la población recalculada: las mismas entradas con los mismos
    parámetros. La fecha y las referencias de las series sí pueden diferir: la serie sigue
    creciendo con los lotes siguientes. La usan el lote al reanudar y el análisis."""
    return (
        saved.entries == recomputed.entries
        and saved.seed == recomputed.seed
        and saved.timeframe == recomputed.timeframe
        and saved.warmup_bars == recomputed.warmup_bars
        and saved.candle_limit == recomputed.candle_limit
        and saved.horizon == recomputed.horizon
    )


def write_block_manifest(manifest: SelectionManifest, path: Path) -> SelectionManifest:
    """Escribe el manifiesto, o, si ya existe, comprueba que es el mismo y devuelve el guardado.

    Al reanudar, la población se vuelve a calcular sobre el histórico, que solo creció por la
    derecha, y tiene que dar las mismas entradas con los mismos parámetros (`same_population`).
    Si no, algo cambió debajo del lote, y no se sigue.
    """
    if not path.is_file():
        write_manifest(path, manifest)
        return manifest
    saved = load_manifest(path)
    if not same_population(saved, manifest):
        raise ForwardDataError(
            f"{path} ya existe y la población recalculada no es la suya: el histórico o el código "
            "cambiaron bajo el lote"
        )
    return saved
