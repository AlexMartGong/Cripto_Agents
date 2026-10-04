"""Funding de perpetuos: qué liquidaciones paga o cobra una posición entre su entrada y su salida.

Lee las series versionadas de `data/funding/` —las que bajó `perp_probe`, no hay otra descarga— y
contesta una sola pregunta: la suma de las tasas de las liquidaciones dentro de una ventana, o
`None` si no puede asegurar que sean todas. No llama a nadie, no usa credenciales y no toca la red.

**La ventana es `(entrada, salida]`.** Una liquidación en el instante exacto de la entrada no se
paga —la posición todavía no existía cuando se cobró— y una en el de la salida sí. Es la única
forma de no mirar hacia delante: nada posterior a la salida entra en la suma.

**Las marcas de Binance traen jitter.** Entre 0 y 26 ms después de la hora exacta, en cerca de la
mitad de las filas (`…08:00:00.013`). Comparada tal cual con una entrada a las 08:00:00.000, esa
liquidación sería «estrictamente posterior» y se contaría en la entrada, que es justo la que no se
paga. Cada marca se ajusta al minuto más cercano al cargarla; el jitter es de milisegundos y la
rejilla de liquidación, de horas, así que no hay ambigüedad posible.

**Lo que falta no es cero.** Para dar una suma la serie tiene que cubrir la ventana: una fila en o
antes de la entrada y otra en o después de la salida, y ningún hueco entre ellas fuera de la
rejilla de 1, 2, 4 y 8 h de `perp_probe`. Si no, `None`. Suponer cero con una liquidación ausente
sería inventar un retorno. El límite que se conoce: un registro perdido dentro de un tramo de 4 h
se ve como un hueco legítimo de 8 h y no se puede detectar.
"""

from __future__ import annotations

import csv
import hashlib
import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

from crypto_agents.perp_probe import CSV_HEADER, FundingRow, interval_hours

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

__all__ = [
    "FUNDING_DIR",
    "SETTLEMENT_SNAP_MS",
    "FundingError",
    "FundingSeries",
    "funding_digests",
    "funding_path",
    "load_funding",
    "read_funding_csv",
    "snap_to_settlement",
]

FUNDING_DIR = Path("data/funding")
"""Dónde viven las series versionadas, con su procedencia y sus digests en un README."""

SETTLEMENT_SNAP_MS = 60_000
"""Cada marca se ajusta al minuto más cercano. El jitter observado es de decenas de milisegundos."""


class FundingError(RuntimeError):
    """Una serie que no se puede leer, o una ventana que no tiene sentido."""


def snap_to_settlement(timestamp_ms: int) -> int:
    """La marca ajustada al minuto más cercano: `08:00:00.013` y `07:59:59.990` son las 08:00."""
    return (timestamp_ms + SETTLEMENT_SNAP_MS // 2) // SETTLEMENT_SNAP_MS * SETTLEMENT_SNAP_MS


@dataclass(frozen=True, slots=True)
class FundingSeries:
    """Las liquidaciones de un símbolo: marcas ajustadas, estrictamente crecientes, y su tasa."""

    times: tuple[int, ...]
    rates: tuple[float, ...]

    @classmethod
    def from_rows(cls, rows: Sequence[FundingRow]) -> FundingSeries:
        """Ajusta las marcas y exige que sigan siendo estrictamente crecientes.

        Dos filas que caen en el mismo minuto, o una serie desordenada, no son una serie de
        liquidaciones: son dos lecturas de lo mismo o un archivo roto, y sumarlas dos veces
        inventaría un coste.
        """
        times = tuple(snap_to_settlement(row.timestamp_ms) for row in rows)
        for previous, current in pairwise(times):
            if current <= previous:
                raise FundingError(
                    f"marcas repetidas o desordenadas tras ajustar: {previous} → {current}"
                )
        return cls(times=times, rates=tuple(row.rate for row in rows))

    def over(
        self, entry_ms: int, exit_ms: int, *, exit_before_ms: int | None = None
    ) -> float | None:
        """Suma de las tasas con marca en `(entry_ms, exit_ms]`, o `None` si no está todo.

        Con `exit_before_ms` la salida no es un instante sino «algún momento en `[exit_ms,
        exit_before_ms)`» —el de un stop que se toca dentro de una vela—. Las liquidaciones
        hasta `exit_ms` se pagan seguro; la de `exit_before_ms`, el cierre de la vela, no, porque
        el stop saltó antes; y una **estrictamente dentro** del tramo incierto no se puede decir
        si se pagó: `None`.
        """
        if exit_ms < entry_ms:
            raise FundingError(f"salida {exit_ms} anterior a la entrada {entry_ms}")
        if exit_before_ms is not None and exit_before_ms <= exit_ms:
            raise FundingError(f"el tramo de salida [{exit_ms}, {exit_before_ms}) está vacío")
        covered_until = exit_ms if exit_before_ms is None else exit_before_ms

        times = self.times
        lower = bisect_right(times, entry_ms) - 1
        upper = bisect_left(times, covered_until)
        if lower < 0 or upper >= len(times):
            return None
        if any(interval_hours(times[k], times[k + 1]) is None for k in range(lower, upper)):
            return None
        if exit_before_ms is not None and bisect_left(times, exit_before_ms) > bisect_right(
            times, exit_ms
        ):
            return None

        first = bisect_right(times, entry_ms)
        last = bisect_right(times, exit_ms)
        return math.fsum(self.rates[first:last])


def funding_path(symbol: str, directory: Path = FUNDING_DIR) -> Path:
    """`BTC/USDT` → `data/funding/btcusdt.csv`, como `data/history/btcusdt_4h.csv` sin la barra."""
    return directory / f"{symbol.replace('/', '').lower()}.csv"


def read_funding_csv(path: Path) -> FundingSeries:
    """Lee una serie de `perp_probe`: `timestamp_ms,funding_rate`, una liquidación por línea."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise FundingError(f"no se puede leer la serie de funding {path}: {error}") from error
    reader = csv.reader(text.splitlines())
    header = next(reader, None)
    if header is None or ",".join(header) != CSV_HEADER.strip():
        raise FundingError(f"{path}: la cabecera no es {CSV_HEADER.strip()!r}")
    rows: list[FundingRow] = []
    for number, record in enumerate(reader, start=2):
        try:
            timestamp, rate = record
            rows.append(FundingRow(timestamp_ms=int(timestamp), rate=float(rate)))
        except ValueError as error:
            raise FundingError(f"{path}:{number}: fila de funding ilegible: {record!r}") from error
    try:
        return FundingSeries.from_rows(rows)
    except FundingError as error:
        raise FundingError(f"{path}: {error}") from error


def load_funding(symbols: Iterable[str], directory: Path = FUNDING_DIR) -> dict[str, FundingSeries]:
    """Las series de los símbolos pedidos. Una que falta es un error: no se asume sin funding."""
    return {symbol: read_funding_csv(funding_path(symbol, directory)) for symbol in symbols}


def funding_digests(symbols: Iterable[str], directory: Path = FUNDING_DIR) -> Mapping[str, str]:
    """El sha-256 del archivo de cada símbolo, para citarlo al lado de la cifra que sale de él."""
    digests: dict[str, str] = {}
    for symbol in symbols:
        path = funding_path(symbol, directory)
        try:
            digests[symbol] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            raise FundingError(f"no se puede leer la serie de funding {path}: {error}") from error
    return digests
