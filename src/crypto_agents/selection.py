"""Selección estratificada de activaciones para la ablación. Cero llamadas a modelo.

Dos años de 4h por siete símbolos son ~4 400 velas cada uno y unas 630
activaciones por símbolo: material sobra. Lo que no sobra es cuota — seis brazos
llegan al decisor y ninguno reutiliza su caché, porque el prompt lleva dentro los
alegatos que cada forma del pipeline produce— así que hay que elegir un
subconjunto, y **cómo** se elige decide qué mide la tabla.

150 activaciones contiguas de un solo símbolo son un régimen de mercado. Una
ablación sobre eso compara seis pipelines bajo una sola condición y llama
«ventaja» a lo que puede ser afinidad con un tramo de dos meses. Repartiendo
entre los siete símbolos y entre tramos separados en el tiempo, lo que se mide es
si la ventaja de un brazo sobrevive al cambio de régimen, que es la única ventaja
que sirve para operar.

Tres propiedades que el manifiesto tiene que dar, y de dónde salen:

- **Reproducible sin volver a elegir.** La selección se escribe entera —símbolo,
  vela y sello— y se versiona. La semilla queda dentro, pero lo que reproduce una
  corrida es la lista, no la semilla: si mañana cambia el histórico, la semilla
  daría otra selección y la lista falla ruidosamente, que es lo correcto.
- **Reproducible aunque el histórico se vuelva a descargar.** Cada entrada lleva
  el digest de la ventana exacta que verá la evaluación. `verify_histories()` lo
  comprueba antes de gastar nada: una vela revisada por el exchange se ve al
  empezar y no tres horas después, en una tabla que ya no compara con nada.
- **Confirmada contra la ventana real.** El barrido calcula indicadores sobre la
  serie entera y corta; la evaluación los calcula sobre las 500 velas que el
  exchange devuelve. EMA y ADX son recursivos, así que las dos cifras no tienen
  por qué coincidir en la vela `i`. Cada candidato se vuelve a pasar por el gate
  con la ventana que verá el replay, y solo entra si también abre ahí.

Este módulo no importa `crypto_agents.llm` ni `crypto_agents.replay`, así que
construir un manifiesto no puede costar dinero. Un test de arquitectura lo fija.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import AwareDatetime, Field, model_validator

from crypto_agents.activation import DEFAULT_CONFIG, ActivationConfig, evaluate_activation
from crypto_agents.activation_sweep import (
    DEFAULT_SYMBOLS,
    DEFAULT_YEARS,
    ActivatedBar,
    activated_bars,
    history_path,
    load_series,
)
from crypto_agents.indicators import DEFAULT_PRESET, IndicatorPreset, enrich
from crypto_agents.market import (
    MarketDataError,
    candles_digest,
    read_ohlcv_csv,
    to_dataframe,
)
from crypto_agents.state import FrozenModel

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Self

__all__ = [
    "DEFAULT_CANDLE_LIMIT",
    "DEFAULT_HORIZON",
    "DEFAULT_MANIFEST",
    "DEFAULT_SEED",
    "DEFAULT_STRATA",
    "DEFAULT_TARGET",
    "HISTORY_DIR",
    "PlannedEvaluation",
    "SelectionError",
    "SelectionManifest",
    "SeriesRef",
    "confirm_activation",
    "load_manifest",
    "load_selection_histories",
    "main",
    "refuse_to_overwrite",
    "replay_window",
    "select_activations",
    "verify_histories",
    "window_digest",
    "write_manifest",
]

DEFAULT_TARGET = 140
"""Activaciones que barre la ablación completa.

Sale de la cuota del decisor, que es el único rol sin respaldo: 880 llamadas por
ventana entre los seis brazos —los seis deciden, incluidos los dos locales, y
ninguno reutiliza la caché del decisor— son 146 activaciones. 140 deja 40
llamadas de margen para reintentos, que también consumen cuota.
"""

DEFAULT_STRATA = 5
"""Tramos por símbolo. Con 7 símbolos y 140 activaciones, 4 por tramo, exacto."""

DEFAULT_SEED = 20260815
"""Semilla del muestreo dentro de cada celda (símbolo y tramo).

Declarada y escrita en el manifiesto, no escondida en una llamada a `random`. Lo
que la semilla elige es *cuál* de las activaciones de una celda entra; el reparto
entre celdas no depende de ella.
"""

DEFAULT_HORIZON = 6
"""Velas que se caminan para puntuar una orden. Una activación sin esas velas por
delante no se puede puntuar, así que no entra en la selección."""

DEFAULT_CANDLE_LIMIT = 500
"""Velas que la evaluación pide al exchange. Decide la ventana que confirma el gate."""

DEFAULT_TIMEFRAME = "4h"
DEFAULT_MANIFEST = Path("data/ablation_selection.json")
HISTORY_DIR = Path("data/history")
"""Dónde vive el histórico de la ablación, versionado.

Distinto de `var/history`, que es caché del barrido y está en .gitignore. Estas
siete series sí se comprometen: son la entrada de una tabla que se publica, y un
manifiesto que apunta a un archivo que nadie más tiene no reproduce nada.
"""


class SelectionError(RuntimeError):
    """La selección no se puede construir o el histórico ya no es el mismo."""


class SeriesRef(FrozenModel):
    """Una serie del histórico, con lo que hace falta para reconocerla."""

    symbol: str = Field(min_length=1)
    timeframe: str = Field(min_length=2)
    bars: int = Field(gt=0)
    first: AwareDatetime
    last: AwareDatetime
    digest: str = Field(min_length=64, max_length=64)
    activations: int = Field(ge=0)
    """Activaciones que el gate encontró en la serie entera, antes de seleccionar."""


class PlannedEvaluation(FrozenModel):
    """Una vela a evaluar: dónde está, cuándo es y qué ventana verá.

    `candles_digest` es el de la ventana ya sin la vela en formación, es decir
    exactamente el que `MarketSnapshot.candles_digest` llevará cuando el replay
    evalúe esta entrada. Comparar los dos es lo que convierte «el histórico
    cambió» en un error al empezar en vez de en una tabla incomparable.
    """

    symbol: str = Field(min_length=1)
    timeframe: str = Field(min_length=2)
    index: int = Field(ge=0)
    at: AwareDatetime
    candles_digest: str = Field(min_length=64, max_length=64)
    triggers: tuple[str, ...] = ()
    stratum: int = Field(default=0, ge=0)
    """Tramo temporal del que salió. Se guarda para poder auditar el reparto."""


class SelectionManifest(FrozenModel):
    """La selección completa: qué se evalúa, sobre qué histórico y con qué reglas."""

    version: int = Field(default=1, ge=1)
    created_at: AwareDatetime
    seed: int
    target: int = Field(gt=0)
    strata: int = Field(gt=0)
    timeframe: str = Field(min_length=2)
    warmup_bars: int = Field(gt=0)
    candle_limit: int = Field(gt=0)
    horizon: int = Field(gt=0)
    series: tuple[SeriesRef, ...] = Field(min_length=1)
    entries: tuple[PlannedEvaluation, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _entries_belong_to_declared_series(self) -> Self:
        """Una entrada de un símbolo que no está en `series` no se puede resolver."""
        known = {(item.symbol, item.timeframe) for item in self.series}
        unknown = sorted(
            {
                f"{entry.symbol} {entry.timeframe}"
                for entry in self.entries
                if (entry.symbol, entry.timeframe) not in known
            }
        )
        if unknown:
            raise ValueError(f"entradas sin serie declarada: {', '.join(unknown)}")
        return self

    @property
    def by_symbol(self) -> dict[str, int]:
        """Cuántas entradas aporta cada símbolo."""
        counts: dict[str, int] = {}
        for entry in self.entries:
            counts[entry.symbol] = counts.get(entry.symbol, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def by_stratum(self) -> dict[int, int]:
        """Cuántas entradas aporta cada tramo temporal, sumando símbolos."""
        counts: dict[int, int] = {}
        for entry in self.entries:
            counts[entry.stratum] = counts.get(entry.stratum, 0) + 1
        return dict(sorted(counts.items()))


# ─────────────────────────────── La ventana que verá el replay ────────────────────────────────────


def replay_window(
    rows: Sequence[Sequence[float]], index: int, candle_limit: int = DEFAULT_CANDLE_LIMIT
) -> list[list[float]]:
    """Velas cerradas que la evaluación de `index` tendrá delante.

    Reproduce lo que hacen `HistoricalMarketClient` y `drop_forming_candle`
    juntos: el exchange entrega la ventana hasta la vela evaluada *más la
    siguiente*, todavía en formación, y el pipeline descarta esa última. De ahí
    que el corte superior sea `index + 2` y el resultado termine en `index`.

    Se replica en vez de importarse porque `replay.py` importa el router, y este
    módulo no puede tener a mano un proveedor al que llamar. `tests/test_selection.py`
    fija que las dos ventanas son la misma.
    """
    if index < 0 or index >= len(rows):
        raise SelectionError(f"vela {index} fuera de la serie de {len(rows)} filas")
    end = min(index + 2, len(rows))
    start = max(0, end - candle_limit)
    return [list(row) for row in rows[start : index + 1]]


def window_digest(
    rows: Sequence[Sequence[float]], index: int, candle_limit: int = DEFAULT_CANDLE_LIMIT
) -> str:
    """Digest de esa ventana, en el mismo formato que `MarketSnapshot.candles_digest`."""
    return candles_digest(to_dataframe(replay_window(rows, index, candle_limit)))


def confirm_activation(
    rows: Sequence[Sequence[float]],
    index: int,
    candle_limit: int = DEFAULT_CANDLE_LIMIT,
    config: ActivationConfig = DEFAULT_CONFIG,
    preset: IndicatorPreset = DEFAULT_PRESET,
) -> tuple[str, ...] | None:
    """Disparadores que el gate encuentra con la ventana del replay, o `None`.

    El barrido calcula el preset sobre la serie entera y corta por la derecha; la
    evaluación lo calcula sobre 500 velas. EMA y ADX son recursivos, así que una
    vela puede activar en el barrido y no en la corrida. Meterla igual dejaría en
    el manifiesto entradas que no llaman a nadie, y la selección diría 140 donde
    la tabla mediría menos.
    """
    window = replay_window(rows, index, candle_limit)
    if len(window) <= preset.min_bars:
        return None
    check = evaluate_activation(enrich(to_dataframe(window), preset), config)
    if not check.should_run:
        return None
    return tuple(check.triggers)


# ─────────────────────────────────────── Selección ────────────────────────────────────────────────


def select_activations(
    histories: Mapping[str, Sequence[Sequence[float]]],
    timeframe: str = DEFAULT_TIMEFRAME,
    target: int = DEFAULT_TARGET,
    strata: int = DEFAULT_STRATA,
    seed: int = DEFAULT_SEED,
    horizon: int = DEFAULT_HORIZON,
    candle_limit: int = DEFAULT_CANDLE_LIMIT,
    config: ActivationConfig = DEFAULT_CONFIG,
    preset: IndicatorPreset = DEFAULT_PRESET,
    now: datetime | None = None,
) -> SelectionManifest:
    """Reparte `target` activaciones entre símbolos y tramos, y las confirma.

    El reparto es determinista y no depende de la semilla: cada símbolo aporta su
    cuota y dentro de cada símbolo cada tramo aporta la suya. La semilla solo
    decide *cuál* de las activaciones de una celda entra, que es donde una
    elección arbitraria es inevitable.

    Una celda corta no se rellena con la celda vecina en silencio: se completa con
    lo que quede del mismo símbolo, en orden de tramo, y el manifiesto guarda el
    tramo de cada entrada para que el desequilibrio se pueda leer después.

    **Las selecciones están anidadas.** El barajado de cada celda consume del
    generador según cuántas activaciones tiene la celda, no según `target`, y de
    cada celda se toma un prefijo de ese orden. Con la misma semilla y las mismas
    series, pedir más activaciones añade a las ya elegidas y no cambia ninguna: se
    puede crecer de 140 a 280 conservando todo lo ya medido. El relleno de una
    celda corta sigue tomando prefijos, así que no rompe la contención; lo que
    rompe es el reparto por tramo.
    """
    if not histories:
        raise SelectionError("no hay ninguna serie de la que seleccionar")

    rng = random.Random(seed)
    symbols = sorted(histories)
    quotas = _share(target, len(symbols))
    entries: list[PlannedEvaluation] = []
    refs: list[SeriesRef] = []

    for symbol, quota in zip(symbols, quotas, strict=True):
        rows = histories[symbol]
        found = activated_bars(rows, symbol, timeframe, config, preset)
        frame = to_dataframe(rows)
        refs.append(
            SeriesRef(
                symbol=symbol,
                timeframe=timeframe,
                bars=len(rows),
                first=frame.index[0].to_pydatetime(),
                last=frame.index[-1].to_pydatetime(),
                digest=candles_digest(frame),
                activations=len(found),
            )
        )
        entries.extend(
            _select_from_series(
                rows, found, quota, strata, rng, horizon, candle_limit, config, preset
            )
        )

    if len(entries) < target:
        raise SelectionError(
            f"solo se pudieron confirmar {len(entries)} activaciones de las {target} pedidas"
        )

    entries.sort(key=lambda entry: (entry.symbol, entry.index))
    return SelectionManifest(
        created_at=now if now is not None else datetime.now(UTC),
        seed=seed,
        target=target,
        strata=strata,
        timeframe=timeframe,
        warmup_bars=preset.min_bars,
        candle_limit=candle_limit,
        horizon=horizon,
        series=tuple(refs),
        entries=tuple(entries),
    )


def _select_from_series(
    rows: Sequence[Sequence[float]],
    found: Sequence[ActivatedBar],
    quota: int,
    strata: int,
    rng: random.Random,
    horizon: int,
    candle_limit: int,
    config: ActivationConfig,
    preset: IndicatorPreset,
) -> list[PlannedEvaluation]:
    """Cuota de un símbolo, repartida entre tramos y confirmada vela a vela."""
    eligible = [bar for bar in found if bar.index + horizon <= len(rows) - 1]
    if not eligible:
        raise SelectionError(f"{found[0].symbol if found else '?'}: ninguna activación puntuable")

    first, last = eligible[0].index, eligible[-1].index
    span = max(1, last - first + 1)
    buckets: list[list[ActivatedBar]] = [[] for _ in range(strata)]
    for bar in eligible:
        position = min(strata - 1, (bar.index - first) * strata // span)
        buckets[position].append(bar)

    empty = [index for index, bucket in enumerate(buckets) if not bucket]
    if empty:
        raise SelectionError(
            f"{eligible[0].symbol}: tramos sin ninguna activación: {empty}. "
            "Con un tramo vacío la selección deja de estar repartida en el tiempo"
        )

    for bucket in buckets:
        rng.shuffle(bucket)

    picked: list[PlannedEvaluation] = []
    for stratum, (bucket, share) in enumerate(zip(buckets, _share(quota, strata), strict=True)):
        picked.extend(_confirm_up_to(rows, bucket, share, stratum, candle_limit, config, preset))

    stratum = 0
    while len(picked) < quota and stratum < strata:
        taken = {entry.index for entry in picked}
        remaining = [bar for bar in buckets[stratum] if bar.index not in taken]
        picked.extend(
            _confirm_up_to(
                rows, remaining, quota - len(picked), stratum, candle_limit, config, preset
            )
        )
        stratum += 1
    return picked


def _confirm_up_to(
    rows: Sequence[Sequence[float]],
    candidates: Sequence[ActivatedBar],
    wanted: int,
    stratum: int,
    candle_limit: int,
    config: ActivationConfig,
    preset: IndicatorPreset,
) -> list[PlannedEvaluation]:
    """Toma candidatos hasta llegar a `wanted`, descartando los que no confirman."""
    chosen: list[PlannedEvaluation] = []
    for bar in candidates:
        if len(chosen) >= wanted:
            break
        triggers = confirm_activation(rows, bar.index, candle_limit, config, preset)
        if triggers is None:
            continue
        chosen.append(
            PlannedEvaluation(
                symbol=bar.symbol,
                timeframe=bar.timeframe,
                index=bar.index,
                at=bar.at,
                candles_digest=window_digest(rows, bar.index, candle_limit),
                triggers=triggers,
                stratum=stratum,
            )
        )
    return chosen


def _share(total: int, parts: int) -> list[int]:
    """Reparte `total` en `parts` cuotas que difieren como mucho en uno."""
    if parts <= 0:
        raise SelectionError("no se puede repartir entre cero partes")
    base, extra = divmod(total, parts)
    return [base + (1 if index < extra else 0) for index in range(parts)]


# ────────────────────────────── Persistencia y verificación ───────────────────────────────────────


def write_manifest(path: Path, manifest: SelectionManifest) -> None:
    """Escribe el manifiesto en JSON legible en un diff."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.model_dump(mode="json")
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_manifest(path: Path | str) -> SelectionManifest:
    """Lee un manifiesto versionado."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SelectionError(f"no se pudo leer el manifiesto {path}: {error}") from error
    return SelectionManifest.model_validate(payload)


def load_selection_histories(
    manifest: SelectionManifest, directory: Path = HISTORY_DIR
) -> dict[str, list[list[float]]]:
    """Series que el manifiesto declara, leídas del histórico versionado."""
    histories: dict[str, list[list[float]]] = {}
    for ref in manifest.series:
        path = history_path(ref.symbol, ref.timeframe, directory)
        if not path.is_file():
            raise SelectionError(f"falta el histórico de {ref.symbol} {ref.timeframe}: {path}")
        histories[ref.symbol] = read_ohlcv_csv(path)
    return histories


def verify_histories(
    manifest: SelectionManifest, histories: Mapping[str, Sequence[Sequence[float]]]
) -> None:
    """Comprueba que cada entrada verá exactamente la ventana que se seleccionó.

    Se hace antes de correr y no al final: una serie revisada por el exchange
    convierte la tabla en incomparable con la anterior, y enterarse al empezar
    cuesta un segundo contra las horas que cuesta enterarse al terminar.
    """
    missing = sorted({entry.symbol for entry in manifest.entries} - set(histories))
    if missing:
        raise SelectionError(f"faltan históricos para: {', '.join(missing)}")

    for entry in manifest.entries:
        rows = histories[entry.symbol]
        if entry.index >= len(rows):
            raise SelectionError(
                f"{entry.symbol}: la vela {entry.index} no existe en un histórico "
                f"de {len(rows)} filas"
            )
        actual = window_digest(rows, entry.index, manifest.candle_limit)
        if actual != entry.candles_digest:
            raise SelectionError(
                f"{entry.symbol} {entry.at:%Y-%m-%d %H:%M}: la ventana cambió "
                f"(manifiesto {entry.candles_digest[:12]}…, histórico {actual[:12]}…)"
            )


# ─────────────────────────────────────────── Comando ──────────────────────────────────────────────


def render_selection(manifest: SelectionManifest) -> str:
    """Resumen legible del manifiesto: reparto por símbolo, por tramo y procedencia."""
    lines = [
        f"Selección: {len(manifest.entries)} activaciones · semilla {manifest.seed} "
        f"· {manifest.strata} tramos · horizonte {manifest.horizon}",
        "",
        "| símbolo | velas | activaciones | seleccionadas | desde | hasta | digest |",
        "| --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    counts = manifest.by_symbol
    for ref in manifest.series:
        lines.append(
            f"| {ref.symbol} | {ref.bars} | {ref.activations} | {counts.get(ref.symbol, 0)} | "
            f"{ref.first:%Y-%m-%d} | {ref.last:%Y-%m-%d} | `{ref.digest[:16]}…` |"
        )
    lines.extend(
        [
            "",
            "| tramo | entradas |",
            "| --- | ---: |",
            *(f"| {stratum} | {count} |" for stratum, count in manifest.by_stratum.items()),
        ]
    )
    return "\n".join(lines) + "\n"


async def _run(args: argparse.Namespace) -> SelectionManifest:
    """Descarga o lee las series pedidas y construye el manifiesto."""
    symbols = tuple(item.strip() for item in args.symbols.split(",") if item.strip())
    end = datetime.now(UTC)
    start = end - timedelta(days=365 * args.years)

    histories: dict[str, list[list[float]]] = {}
    for symbol in symbols:
        rows = await load_series(
            symbol, args.timeframe, start, end, directory=args.history, refresh=args.refresh
        )
        histories[symbol] = rows
        print(f"{symbol:10} {args.timeframe:3} {len(rows):>6} velas", file=sys.stderr)

    return select_activations(
        histories,
        timeframe=args.timeframe,
        target=args.target,
        strata=args.strata,
        seed=args.seed,
        horizon=args.horizon,
        candle_limit=args.candle_limit,
    )


def refuse_to_overwrite(out: Path | None) -> str | None:
    """Por qué no se puede escribir ahí, o `None` si se puede. Se pregunta antes de calcular.

    `--out` valía por defecto el manifiesto versionado: volver a lanzar el comando —con otro
    `--target`, por ejemplo, para ver una selección mayor— sustituía el plan sobre el que corre
    la ablación, y con él los `prompt_digest` de todo lo ya medido. Lo que reproduce una corrida
    es esa lista, así que escribirla no puede ser lo que pasa cuando no se dice nada.
    """
    if out is None:
        return (
            "falta --out: el manifiesto versionado no se escribe por defecto. Di dónde va la "
            "selección nueva"
        )
    if out.exists():
        return f"{out} ya existe y no se pisa: un manifiesto escrito es un plan que algo ya usó"
    return None


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada: `python -m crypto_agents.selection`."""
    parser = argparse.ArgumentParser(
        prog="python -m crypto_agents.selection",
        description="Selecciona activaciones repartidas por símbolo y por tramo temporal.",
    )
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME)
    parser.add_argument("--years", type=int, default=DEFAULT_YEARS)
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--strata", type=int, default=DEFAULT_STRATA)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--candle-limit", type=int, default=DEFAULT_CANDLE_LIMIT)
    parser.add_argument("--history", type=Path, default=HISTORY_DIR)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "dónde escribir el manifiesto. Obligatorio, y un archivo que ya existe no se pisa: "
            f"`{DEFAULT_MANIFEST}` es el plan versionado de la ablación"
        ),
    )
    parser.add_argument(
        "--refresh", action="store_true", help="vuelve a descargar aunque haya histórico"
    )
    args = parser.parse_args(argv)

    refusal = refuse_to_overwrite(args.out)
    if refusal is not None:
        print(f"error: {refusal}", file=sys.stderr)
        return 1

    try:
        manifest = asyncio.run(_run(args))
    except (SelectionError, MarketDataError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    write_manifest(args.out, manifest)
    print(render_selection(manifest))
    print(f"manifiesto escrito en {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
