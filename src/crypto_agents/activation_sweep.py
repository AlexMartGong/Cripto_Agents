"""Barrido del gate de activación sobre histórico. Cero llamadas a modelo.

Responde tres preguntas que hasta ahora se contestaban esperando a que el mercado
cooperara: cuántas evaluaciones rinde una ventana histórica, si alguna de las
cuatro reglas no dispara nunca, y si 4h da material suficiente o solo lo hay en
1h. Sin eso, dimensionar la ablación es adivinar.

Dos decisiones que hacen que la tabla mida el gate y no el barrido:

- **Se llama a `evaluate_activation`, no a una copia de las reglas.** Una
  reimplementación mediría este archivo. Es el mismo argumento por el que ningún
  brazo de la ablación toca la cola determinista.
- **Los indicadores se calculan una vez por serie y después se corta.** Lo
  autoriza `tests/test_lookahead.py`, que fija que truncar por la derecha no
  cambia la barra `i`: si eso no fuera cierto habría que recalcular el preset en
  cada vela, y el barrido pasaría de un minuto a horas. Como es cierto, cortar es
  exacto y barato.

Este módulo no importa `crypto_agents.llm` y no puede gastar cuota, lo cual es
justo lo que permite repetir la tabla cuantas veces haga falta. Un test de
arquitectura lo fija.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import Field

from crypto_agents.activation import DEFAULT_CONFIG, ActivationConfig, evaluate_activation
from crypto_agents.indicators import DEFAULT_PRESET, IndicatorPreset, enrich
from crypto_agents.market import (
    CcxtMarketClient,
    MarketDataError,
    candles_digest,
    download_history,
    read_ohlcv_csv,
    to_dataframe,
    write_ohlcv_csv,
)
from crypto_agents.state import FrozenModel

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "DEFAULT_SYMBOLS",
    "DEFAULT_TIMEFRAMES",
    "SeriesSweep",
    "history_path",
    "load_series",
    "main",
    "render_report",
    "sweep_series",
]

DEFAULT_SYMBOLS = (
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "ADA/USDT",
    "DOGE/USDT",
)
"""Los siete con más profundidad de libro y continuidad histórica en binance spot.

La profundidad importa para esta tabla en concreto: `range_breakout` y
`volatility_jump` leen forma del precio, y en un par sin liquidez esa forma la
dibuja el spread, no el mercado.
"""

DEFAULT_TIMEFRAMES = ("4h", "1h")
DEFAULT_YEARS = 2

_RULE_BY_PREFIX = {
    "ma_cross": "ma_cross",
    "range_breakout": "range_breakout",
    "volatility_jump": "volatility_jump",
    "regime_change": "regime_change",
}
"""Traduce un trigger a la regla que lo emitió, para agregar por las dos vías."""

HISTORY_DIR = Path("var/history")
"""Dónde se cachean las velas. Está en .gitignore: son 10 MB de salida, no fuente.

Lo que se versiona es `docs/activation.md`, con el rango y el digest de cada
serie. Con eso, dos tablas se pueden comparar sabiendo si miraron lo mismo.
"""


class SeriesSweep(FrozenModel):
    """Resultado del barrido sobre una serie: un símbolo en un timeframe."""

    symbol: str = Field(min_length=1)
    timeframe: str = Field(min_length=2)
    bars: int = Field(ge=0)
    """Velas descargadas, incluidas las de warm-up."""

    evaluated: int = Field(ge=0)
    """Velas sobre las que el gate llegó a pronunciarse.

    Es `bars` menos el warm-up del preset. Es el número que dimensiona la
    ablación: no hay más evaluaciones posibles que estas.
    """

    activated: int = Field(ge=0)
    by_trigger: dict[str, int] = Field(default_factory=dict)
    by_rule: dict[str, int] = Field(default_factory=dict)
    first: datetime
    last: datetime
    digest: str = Field(min_length=64, max_length=64)

    @property
    def rate(self) -> float:
        """Fracción de velas evaluadas en las que el gate abrió."""
        return self.activated / self.evaluated if self.evaluated else 0.0


def history_path(symbol: str, timeframe: str, directory: Path = HISTORY_DIR) -> Path:
    """Ruta del CSV cacheado de una serie."""
    return directory / f"{symbol.replace('/', '').lower()}_{timeframe}.csv"


async def load_series(
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    directory: Path = HISTORY_DIR,
    refresh: bool = False,
) -> list[list[float]]:
    """Velas de la serie, del caché si están y del exchange si no.

    El caché existe porque el barrido se repite y la descarga son ~160 peticiones
    públicas. No se versiona: son datos de salida, y lo que hace comparables dos
    tablas es el digest que queda escrito en el reporte, no el CSV.
    """
    path = history_path(symbol, timeframe, directory)
    if path.is_file() and not refresh:
        return read_ohlcv_csv(path)

    client = CcxtMarketClient("binance")
    try:
        rows = await download_history(client, symbol, timeframe, start, end)
    finally:
        await client.close()
    write_ohlcv_csv(path, rows)
    return rows


def sweep_series(
    rows: Sequence[Sequence[float]],
    symbol: str,
    timeframe: str,
    config: ActivationConfig = DEFAULT_CONFIG,
    preset: IndicatorPreset = DEFAULT_PRESET,
) -> SeriesSweep:
    """Corre el gate sobre cada vela cerrada de la serie y cuenta lo que disparó.

    Recibe filas crudas y no un DataFrame: normalizar es de `market.py`, y con la
    firma en filas este módulo no necesita nombrar pandas — que es lo que
    `test_architecture.py` acota a los tres módulos que sí calculan sobre él.

    El recorrido empieza en `preset.min_bars` porque antes de eso los indicadores
    arrastran NaN y `IndicatorSet` los rechazaría: son velas que en producción
    tampoco llegarían al gate.
    """
    candles = to_dataframe(rows)
    enriched = enrich(candles, preset)
    by_trigger: dict[str, int] = {}
    by_rule: dict[str, int] = {}
    activated = 0
    evaluated = 0

    for index in range(preset.min_bars, len(enriched)):
        check = evaluate_activation(enriched.iloc[: index + 1], config)
        evaluated += 1
        if not check.should_run:
            continue
        activated += 1
        for trigger in check.triggers:
            by_trigger[trigger] = by_trigger.get(trigger, 0) + 1
            rule = _rule_of(trigger)
            by_rule[rule] = by_rule.get(rule, 0) + 1

    return SeriesSweep(
        symbol=symbol,
        timeframe=timeframe,
        bars=len(candles),
        evaluated=evaluated,
        activated=activated,
        by_trigger=dict(sorted(by_trigger.items())),
        by_rule=dict(sorted(by_rule.items())),
        first=candles.index[0].to_pydatetime(),
        last=candles.index[-1].to_pydatetime(),
        digest=candles_digest(candles),
    )


def _rule_of(trigger: str) -> str:
    """Regla que emitió un trigger. Falla ruidosamente ante uno desconocido.

    Una regla nueva cuyo trigger no encaje aquí quedaría fuera de la tabla por
    regla sin que nadie lo notara, y la conclusión «esta regla no dispara nunca»
    sería un artefacto del agregador.
    """
    for prefix, rule in _RULE_BY_PREFIX.items():
        if trigger.startswith(prefix):
            return rule
    raise MarketDataError(f"trigger sin regla conocida: {trigger}")


def render_report(results: Sequence[SeriesSweep], preset: IndicatorPreset = DEFAULT_PRESET) -> str:
    """Las dos tablas más la procedencia, en el formato de `docs/activation.md`."""
    rule_names = list(_RULE_BY_PREFIX.values())
    triggers = sorted({trigger for item in results for trigger in item.by_trigger})

    lines: list[str] = []
    lines.append("## Activaciones por símbolo y timeframe")
    lines.append("")
    lines.append(
        "| símbolo | tf | velas | evaluadas | activaciones | tasa | "
        + " | ".join(f"`{name}`" for name in rule_names)
        + " |"
    )
    lines.append(
        "| --- | --- | ---: | ---: | ---: | ---: | " + " | ".join("---:" for _ in rule_names) + " |"
    )
    for item in results:
        counts = " | ".join(str(item.by_rule.get(name, 0)) for name in rule_names)
        lines.append(
            f"| {item.symbol} | {item.timeframe} | {item.bars} | {item.evaluated} | "
            f"{item.activated} | {item.rate:.1%} | {counts} |"
        )

    lines.append("")
    lines.append("## Activaciones por trigger")
    lines.append("")
    lines.append("| trigger | " + " | ".join(tf for tf in DEFAULT_TIMEFRAMES) + " | total |")
    lines.append("| --- | " + " | ".join("---:" for _ in DEFAULT_TIMEFRAMES) + " | ---: |")
    for trigger in triggers:
        per_tf = [
            sum(item.by_trigger.get(trigger, 0) for item in results if item.timeframe == tf)
            for tf in DEFAULT_TIMEFRAMES
        ]
        lines.append(
            f"| `{trigger}` | " + " | ".join(str(value) for value in per_tf) + f" | {sum(per_tf)} |"
        )

    never = [trigger for trigger in _expected_triggers() if trigger not in triggers]
    lines.append("")
    if never:
        lines.append(
            f"Triggers que no dispararon ni una vez: {', '.join(f'`{t}`' for t in never)}."
        )
    else:
        lines.append("Los nueve triggers dispararon al menos una vez.")

    lines.append("")
    lines.append("## Procedencia")
    lines.append("")
    lines.append(
        f"Warm-up del preset: {preset.min_bars} velas por serie, descontadas de `evaluadas`."
    )
    lines.append("")
    lines.append("| símbolo | tf | desde | hasta | velas | digest sha-256 |")
    lines.append("| --- | --- | --- | --- | ---: | --- |")
    for item in results:
        lines.append(
            f"| {item.symbol} | {item.timeframe} | {item.first:%Y-%m-%d} | {item.last:%Y-%m-%d} | "
            f"{item.bars} | `{item.digest}` |"
        )
    return "\n".join(lines)


def _expected_triggers() -> tuple[str, ...]:
    """Los nueve triggers que las cuatro reglas pueden emitir.

    Dos por `ma_cross`, dos por `range_breakout`, uno por `volatility_jump` y
    cuatro por `regime_change`, que distingue dirección de carácter.
    """
    return (
        "ma_cross_bearish",
        "ma_cross_bullish",
        "range_breakout_down",
        "range_breakout_up",
        "regime_change_above_trend",
        "regime_change_below_trend",
        "regime_change_ranging",
        "regime_change_trending",
        "volatility_jump",
    )


async def _run(
    symbols: Sequence[str], timeframes: Sequence[str], years: int, refresh: bool
) -> list[SeriesSweep]:
    """Descarga —o lee del caché— y barre cada serie."""
    end = datetime.now(UTC)
    start = end - timedelta(days=365 * years)
    results: list[SeriesSweep] = []
    for timeframe in timeframes:
        for symbol in symbols:
            rows = await load_series(symbol, timeframe, start, end, refresh=refresh)
            item = sweep_series(rows, symbol, timeframe)
            results.append(item)
            print(
                f"{symbol:10} {timeframe:3} {item.bars:>6} velas  "
                f"{item.activated:>5}/{item.evaluated:<6} ({item.rate:.1%})",
                file=sys.stderr,
            )
    return results


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada: `python -m crypto_agents.activation_sweep`."""
    parser = argparse.ArgumentParser(description="Barrido del gate de activación sobre histórico")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--timeframes", default=",".join(DEFAULT_TIMEFRAMES))
    parser.add_argument("--years", type=int, default=DEFAULT_YEARS)
    parser.add_argument(
        "--refresh", action="store_true", help="vuelve a descargar aunque haya caché"
    )
    args = parser.parse_args(argv)

    symbols = tuple(item.strip() for item in args.symbols.split(",") if item.strip())
    timeframes = tuple(item.strip() for item in args.timeframes.split(",") if item.strip())
    results = asyncio.run(_run(symbols, timeframes, args.years, args.refresh))
    print(render_report(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
