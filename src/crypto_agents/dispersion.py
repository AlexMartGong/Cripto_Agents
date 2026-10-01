"""Dispersión del retorno por evaluación sobre el pool de activaciones. Cero llamadas a modelo.

Responde una pregunta que hay que contestar *antes* de fijar el margen de equivalencia de
la ablación: con la dispersión que tiene el retorno, ¿puede una diferencia pareada de
n = 140 evaluaciones separarse del cero? Sin la desviación estándar no hay forma de saber
si el criterio puede cumplirse o fallar por construcción.

Tres decisiones que lo mantienen honesto:

- **Mide sobre el pool, no sobre la selección.** Las 4 740 activaciones de 4h son todo lo que
  el gate deja pasar; las 140 del manifiesto son una muestra de ellas. Medir la dispersión
  sobre las 140 sería mirar el resultado del que se quiere fijar el criterio.
- **No calcula, imprime ni guarda ninguna media de retorno.** Ni del pool ni de las líneas
  base. Es estructural y no una promesa: `ReturnDispersion` no tiene campo de media, el
  reporte solo recibe esos modelos, y el comando no escribe archivos. Un criterio fijado
  después de ver cuánto rinde `always_buy` en la ventana no sería un criterio.
- **Puntúa por el camino de la ablación.** Cada activación se convierte en un
  `EvaluationRecord` con la propuesta de `always_buy_proposal` o `always_sell_proposal` y se
  puntúa con `score_position(..., Scoring.COMMON_STOP)`. Una copia de esa aritmética
  mediría este archivo y no el stop que la tabla usa.

Este módulo no importa el router, el grafo ni el replay, y tampoco descarga: si falta una
serie lo dice, no la baja. Un test de arquitectura lo fija.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import AwareDatetime, Field

from crypto_agents.activation import DEFAULT_CONFIG, ActivationConfig, evaluate_activation
from crypto_agents.activation_sweep import DEFAULT_SYMBOLS, activated_bars, history_path
from crypto_agents.baselines import always_buy_proposal, always_sell_proposal
from crypto_agents.indicators import DEFAULT_PRESET, IndicatorPreset, enrich, to_indicator_set
from crypto_agents.journal import EvaluationRecord
from crypto_agents.market import (
    MarketDataError,
    build_snapshot,
    candles_digest,
    read_ohlcv_csv,
    to_dataframe,
)
from crypto_agents.outcomes import Outcome, Scoring, TradeOutcome, score_position
from crypto_agents.selection import (
    DEFAULT_CANDLE_LIMIT,
    DEFAULT_TIMEFRAME,
    HISTORY_DIR,
    replay_window,
)
from crypto_agents.state import Action, FrozenModel

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from crypto_agents.activation_sweep import ActivatedBar

__all__ = [
    "DEFAULT_CORRELATIONS",
    "DEFAULT_HORIZON",
    "DEFAULT_HOURS_PER_140",
    "DEFAULT_POWER",
    "DEFAULT_SAMPLE_SIZES",
    "DetectableEffect",
    "DispersionError",
    "PoolOutcomes",
    "PoolReport",
    "ReturnDispersion",
    "SeriesProvenance",
    "build_report",
    "detectable_difference",
    "hours_for",
    "load_histories",
    "main",
    "pool_outcomes",
    "render_report",
    "summarise",
]

DEFAULT_HORIZON = 6
"""Velas hasta la salida. Es el `--horizon` de la ablación; no se importa de allí porque
`ablation.py` trae el router."""

DEFAULT_SAMPLE_SIZES = (140, 280, 420)
"""Tamaños del plan que se comparan: el del manifiesto y sus dos primeros múltiplos."""

DEFAULT_CORRELATIONS = (0.0, 0.5)
"""Correlación pareada entre dos brazos: ninguna, y moderada. No se estimó: se barre."""

DEFAULT_POWER = 0.8
"""Potencia para el efecto detectable. Convención, no un valor buscado."""

Z_INTERVAL = 1.96
"""El del intervalo que publica `metrics.paired_difference`: el semiancho de aquí es el suyo."""

DEFAULT_HOURS_PER_140 = 36.0
"""Pared de la primera corrida para 140 evaluaciones. Dato aportado, no leído de ningún journal."""

EXCHANGE = "binance"


class DispersionError(RuntimeError):
    """Algo que impide medir sin inventar: una serie que falta, una posición que no se sitúa."""


# ─────────────────────────────────────────── Modelos ──────────────────────────────────────────────


class ReturnDispersion(FrozenModel):
    """Dispersión del retorno de una línea base sobre un conjunto de activaciones.

    **No tiene campo de media, a propósito.** Lo que no está en el modelo no se puede
    imprimir por descuido, y el reporte solo conoce modelos de este módulo.
    """

    label: str = Field(min_length=1)
    """`global` o el símbolo."""

    side: Action
    resolved: int = Field(ge=0)
    """Posiciones con horizonte por delante: las que entran en la desviación."""

    unresolved: int = Field(ge=0)
    """Posiciones a las que el histórico no les alcanza para el horizonte. Quedan fuera."""

    invalidated: int = Field(ge=0)
    """De las resueltas, las que salieron por el stop común antes del horizonte."""

    stdev: float | None = Field(default=None, ge=0.0)
    """Desviación estándar muestral (n - 1). `None` con menos de dos posiciones resueltas."""


class SeriesProvenance(FrozenModel):
    """De qué serie sale cada cifra: lo que permite saber si dos reportes miraron lo mismo."""

    symbol: str = Field(min_length=1)
    bars: int = Field(gt=0)
    first: AwareDatetime
    last: AwareDatetime
    digest: str = Field(min_length=64, max_length=64)


class PoolOutcomes(FrozenModel):
    """Lo que la puntuación común dio sobre las activaciones de una serie, por lado."""

    buy: tuple[TradeOutcome, ...]
    sell: tuple[TradeOutcome, ...]
    unconfirmed: int = Field(ge=0)
    """Activaciones del barrido que el gate no abre con la ventana del replay. No se filtran."""


class DetectableEffect(FrozenModel):
    """Cuánto tiene que valer una diferencia pareada para distinguirse del cero con n."""

    n: int = Field(ge=2)
    rho: float = Field(ge=-1.0, le=1.0)
    sigma_difference: float = Field(ge=0.0)
    half_width: float = Field(ge=0.0)
    """Semiancho del IC95%: la diferencia observada más pequeña cuyo intervalo excluye el cero."""

    detectable: float = Field(ge=0.0)
    """Diferencia verdadera más pequeña que el intervalo excluye del cero con la potencia dada."""


class PoolReport(FrozenModel):
    """Todo lo que el reporte sabe. Ninguna media entre estos campos."""

    horizon: int = Field(gt=0)
    series: tuple[SeriesProvenance, ...]
    activations: int = Field(ge=0)
    unconfirmed: int = Field(ge=0)
    dispersions: tuple[ReturnDispersion, ...]
    """Global primero, comprador y vendedor; después cada símbolo en el mismo orden."""


# ─────────────────────────────────────────── Cálculo ──────────────────────────────────────────────


def _run_id(bar: ActivatedBar) -> UUID:
    """Un id por activación, derivado de ella. Solo existe porque el snapshot lo exige."""
    return uuid5(NAMESPACE_URL, f"dispersion|{bar.symbol}|{bar.timeframe}|{bar.at.isoformat()}")


def pool_outcomes(
    rows: Sequence[Sequence[float]],
    bars: Iterable[ActivatedBar],
    horizon: int = DEFAULT_HORIZON,
    preset: IndicatorPreset = DEFAULT_PRESET,
    config: ActivationConfig = DEFAULT_CONFIG,
    candle_limit: int = DEFAULT_CANDLE_LIMIT,
) -> PoolOutcomes:
    """Compra y venta en cada activación, puntuadas con el stop común.

    El ATR sale de la ventana que el brazo vería —`replay_window`—, no de la serie entera:
    es recursivo, y sobre la serie completa daría un stop algo distinto del de la ablación.
    Las activaciones que esa ventana no confirma se cuentan y se mantienen: el pool es el
    del barrido, y quitarlas sería elegir cuáles pesan.
    """
    buys: list[TradeOutcome] = []
    sells: list[TradeOutcome] = []
    unconfirmed = 0
    for bar in bars:
        window = to_dataframe(replay_window(rows, bar.index, candle_limit))
        enriched = enrich(window, preset)
        if not evaluate_activation(enriched, config).should_run:
            unconfirmed += 1
        snapshot = build_snapshot(window, _run_id(bar), EXCHANGE, bar.symbol, bar.timeframe)
        indicators = to_indicator_set(enriched, preset)
        for make, sink in ((always_buy_proposal, buys), (always_sell_proposal, sells)):
            record = EvaluationRecord(
                run_id=snapshot.run_id,
                at=snapshot.timestamp,
                symbol=bar.symbol,
                timeframe=bar.timeframe,
                snapshot=snapshot,
                indicators=indicators,
                proposal=make(snapshot, indicators),
            )
            outcome = score_position(record, rows, horizon, Scoring.COMMON_STOP)
            if outcome is None:
                raise DispersionError(
                    f"{bar.symbol} vela {bar.index}: la posición no se sitúa en la serie"
                )
            sink.append(outcome)
    return PoolOutcomes(buy=tuple(buys), sell=tuple(sells), unconfirmed=unconfirmed)


def summarise(label: str, side: Action, outcomes: Iterable[TradeOutcome]) -> ReturnDispersion:
    """Desviación estándar muestral y recuentos; la media se calcula dentro de `stdev` y no sale.

    Muestral (entre n - 1) y no poblacional: es una muestra de la que se quiere decir algo
    sobre las demás, y dividir entre n subestimaría la dispersión justo cuando n es pequeña,
    que es el caso que esta cifra existe para juzgar.
    """
    items = list(outcomes)
    resolved = [item for item in items if item.outcome is not Outcome.UNRESOLVED]
    returns = [item.gross_return for item in resolved]
    return ReturnDispersion(
        label=label,
        side=side,
        resolved=len(resolved),
        unresolved=len(items) - len(resolved),
        invalidated=sum(1 for item in resolved if item.outcome is Outcome.INVALIDATED),
        stdev=statistics.stdev(returns) if len(returns) >= 2 else None,
    )


def detectable_difference(
    sigma: float, n: int, rho: float, power: float = DEFAULT_POWER
) -> DetectableEffect:
    """Efecto detectable de la diferencia pareada de dos brazos con la misma dispersión.

    Con la misma `sigma` en los dos brazos y correlación `rho`, la desviación de la
    diferencia es `sigma_dif = sigma·√(2(1 - rho))`. El intervalo normal al 95% excluye el
    cero cuando la diferencia observada pasa de `1.96·sigma_dif/√n` (el semiancho), y esa
    diferencia es detectable con probabilidad `power` cuando la verdadera vale al menos
    `(1.96 + z_power)·sigma_dif/√n`.

    Es la aproximación normal de `metrics.paired_difference`, sin t ni bootstrap. Supone
    evaluaciones independientes; el solape de horizontes y la correlación entre símbolos la
    hacen optimista.
    """
    if sigma < 0.0 or not math.isfinite(sigma):
        raise ValueError(f"la desviación tiene que ser finita y no negativa: {sigma}")
    if n < 2:
        raise ValueError(f"hacen falta al menos dos evaluaciones: {n}")
    if not -1.0 <= rho <= 1.0:
        raise ValueError(f"la correlación está en [-1, 1]: {rho}")
    if not 0.0 < power < 1.0:
        raise ValueError(f"la potencia está en (0, 1): {power}")
    sigma_difference = sigma * math.sqrt(2.0 * (1.0 - rho))
    error = sigma_difference / math.sqrt(n)
    z_power = statistics.NormalDist().inv_cdf(power)
    return DetectableEffect(
        n=n,
        rho=rho,
        sigma_difference=sigma_difference,
        half_width=Z_INTERVAL * error,
        detectable=(Z_INTERVAL + z_power) * error,
    )


def hours_for(n: int, hours_per_140: float = DEFAULT_HOURS_PER_140) -> float:
    """Horas de pared para `n` evaluaciones, lineal desde las de 140. Solo un orden de magnitud."""
    return hours_per_140 * n / 140.0


# ─────────────────────────────────────────── Reporte ──────────────────────────────────────────────


def build_report(
    histories: dict[str, list[list[float]]],
    horizon: int = DEFAULT_HORIZON,
    timeframe: str = DEFAULT_TIMEFRAME,
    preset: IndicatorPreset = DEFAULT_PRESET,
    config: ActivationConfig = DEFAULT_CONFIG,
    candle_limit: int = DEFAULT_CANDLE_LIMIT,
    progress: bool = False,
) -> PoolReport:
    """Mide el pool de todas las series y devuelve solo dispersiones y procedencia."""
    series: list[SeriesProvenance] = []
    per_symbol: list[ReturnDispersion] = []
    all_buy: list[TradeOutcome] = []
    all_sell: list[TradeOutcome] = []
    activations = unconfirmed = 0

    for symbol, rows in histories.items():
        candles = to_dataframe(rows)
        series.append(
            SeriesProvenance(
                symbol=symbol,
                bars=len(candles),
                first=candles.index[0].to_pydatetime(),
                last=candles.index[-1].to_pydatetime(),
                digest=candles_digest(candles),
            )
        )
        bars = activated_bars(rows, symbol, timeframe, config, preset)
        outcomes = pool_outcomes(rows, bars, horizon, preset, config, candle_limit)
        activations += len(bars)
        unconfirmed += outcomes.unconfirmed
        all_buy.extend(outcomes.buy)
        all_sell.extend(outcomes.sell)
        per_symbol.append(summarise(symbol, Action.BUY, outcomes.buy))
        per_symbol.append(summarise(symbol, Action.SELL, outcomes.sell))
        if progress:
            print(f"{symbol:10} {timeframe:3} {len(bars):>5} activaciones", file=sys.stderr)

    return PoolReport(
        horizon=horizon,
        series=tuple(series),
        activations=activations,
        unconfirmed=unconfirmed,
        dispersions=(
            summarise("global", Action.BUY, all_buy),
            summarise("global", Action.SELL, all_sell),
            *per_symbol,
        ),
    )


def _percent(value: float | None) -> str:
    """Dos decimales de porcentaje, como el resto de las tablas; sin dato lo dice."""
    return "no determinado" if value is None else f"{value:.2%}"


def _fraction(numerator: int, denominator: int) -> str:
    """Una fracción con sus dos números delante."""
    if denominator == 0:
        return "no determinado: sin posiciones resueltas"
    return f"{numerator}/{denominator} ({numerator / denominator:.1%})"


def _conservative_sigma(report: PoolReport) -> float | None:
    """La mayor de las dos desviaciones globales: cota alta de la de un brazo real."""
    stdevs = [
        item.stdev
        for item in report.dispersions
        if item.label == "global" and item.stdev is not None
    ]
    return max(stdevs) if stdevs else None


def render_report(
    report: PoolReport,
    sample_sizes: Sequence[int] = DEFAULT_SAMPLE_SIZES,
    correlations: Sequence[float] = DEFAULT_CORRELATIONS,
    power: float = DEFAULT_POWER,
    hours_per_140: float = DEFAULT_HOURS_PER_140,
) -> str:
    """Tablas en Markdown. Sin medias, sin veredicto y sin margen de equivalencia propuesto."""
    lines = [
        "# Dispersión del retorno por evaluación — pool de activaciones",
        "",
        f"Pool: {report.activations} activaciones sobre {len(report.series)} series, "
        f"horizonte de {report.horizon} velas, stop común. Compra y venta en cada activación "
        "(`always_buy`, `always_sell`). No se calcula, imprime ni guarda ninguna media de retorno.",
        "",
        f"De las activaciones del barrido, {report.unconfirmed} no abren el gate con la ventana "
        "del replay; se mantienen en el pool.",
        "",
        "## Series",
        "",
        "| símbolo | velas | desde | hasta | digest sha-256 |",
        "| --- | ---: | --- | --- | --- |",
        *(
            f"| {item.symbol} | {item.bars} | {item.first:%Y-%m-%d} | {item.last:%Y-%m-%d} "
            f"| `{item.digest}` |"
            for item in report.series
        ),
        "",
        "## Dispersión",
        "",
        "Retorno bruto por unidad nocional. En el pool toda activación abre posición, así que "
        "el retorno por evaluación y el de la posición son el mismo. Las posiciones sin "
        "horizonte por delante quedan fuera de la desviación.",
        "",
        "| conjunto | lado | resueltas | sin resolver | invalidadas | desviación estándar |",
        "| --- | --- | ---: | ---: | --- | ---: |",
        *(
            f"| {item.label} | {item.side.value} | {item.resolved} | {item.unresolved} "
            f"| {_fraction(item.invalidated, item.resolved)} | {_percent(item.stdev)} |"
            for item in report.dispersions
        ),
        "",
        "## Efecto detectable de la diferencia pareada",
        "",
    ]

    sigma = _conservative_sigma(report)
    if sigma is None:
        lines.append("No determinado: sin dos posiciones resueltas no hay desviación.")
    else:
        lines += [
            f"sigma = {_percent(sigma)}: la mayor de las dos desviaciones globales. Se supone la "
            "misma en los dos brazos, así que sigma_dif = sigma·√(2(1 - rho)). Un brazo que a "
            "veces hace hold mete ceros y baja la suya: esta es una cota alta.",
            "",
            "- **Semiancho del IC95%**: la diferencia observada más pequeña que excluye el cero "
            "(1.96·sigma_dif/√n).",
            "- **Detectable**: la diferencia verdadera más pequeña que el intervalo excluye del "
            f"cero con potencia {power:.0%}.",
            "",
            "| n | rho | sigma_dif | semiancho IC95% | detectable |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
        for n in sample_sizes:
            for rho in correlations:
                effect = detectable_difference(sigma, n, rho, power)
                lines.append(
                    f"| {n} | {rho:.1f} | {_percent(effect.sigma_difference)} "
                    f"| {_percent(effect.half_width)} | {_percent(effect.detectable)} |"
                )
        lines += [
            "",
            "Supone evaluaciones independientes y el intervalo normal de "
            "`metrics.paired_difference`. Horizontes que se solapan y símbolos que se mueven "
            "juntos reducen el n efectivo: los números de arriba son optimistas en eso.",
        ]

    lines += [
        "",
        "## Tiempo, solo como orden de magnitud",
        "",
        f"{hours_per_140:g} h para 140 evaluaciones: dato de la primera corrida (descartada), "
        "aportado a mano y no leído de ningún journal. Se escala en línea recta con n.",
        "",
        "| n | horas | días |",
        "| ---: | ---: | ---: |",
        *(
            f"| {n} | {hours_for(n, hours_per_140):.0f} | {hours_for(n, hours_per_140) / 24:.1f} |"
            for n in sample_sizes
        ),
        "",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────── Comando ──────────────────────────────────────────────


def load_histories(
    symbols: Sequence[str], timeframe: str, directory: Path
) -> dict[str, list[list[float]]]:
    """Las series del directorio, sin descargar nada: si falta una, se dice cuál."""
    histories: dict[str, list[list[float]]] = {}
    for symbol in symbols:
        path = history_path(symbol, timeframe, directory)
        if not path.is_file():
            raise DispersionError(f"falta el histórico de {symbol} {timeframe}: {path}")
        histories[symbol] = read_ohlcv_csv(path)
    return histories


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada: `python -m crypto_agents.dispersion`."""
    parser = argparse.ArgumentParser(
        description="Dispersión del retorno por evaluación y efecto detectable, sin modelos"
    )
    parser.add_argument("--history-dir", type=Path, default=HISTORY_DIR)
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--power", type=float, default=DEFAULT_POWER)
    parser.add_argument(
        "--hours-per-140",
        type=float,
        default=DEFAULT_HOURS_PER_140,
        help="horas de pared de la primera corrida para 140 evaluaciones (dato aportado)",
    )
    args = parser.parse_args(argv)

    symbols = tuple(item.strip() for item in args.symbols.split(",") if item.strip())
    try:
        histories = load_histories(symbols, args.timeframe, args.history_dir)
        report = build_report(histories, args.horizon, args.timeframe, progress=True)
    except (DispersionError, MarketDataError) as error:
        print(f"dispersion: {error}", file=sys.stderr)
        return 1
    print(render_report(report, power=args.power, hours_per_140=args.hours_per_140))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
