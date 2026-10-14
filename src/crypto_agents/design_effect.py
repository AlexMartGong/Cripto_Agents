"""Efecto de diseño del pool: cuánto infla el error estándar la dependencia entre evaluaciones.

`paired_dispersion` da el n de una etapa posterior con `sigma_D/√n`, el mismo error estándar que
el intervalo de `criteria`. Los dos suponen evaluaciones independientes. A n = 140 sobre 4 626
activaciones confirmadas el supuesto era tolerable; a n ≈ 3 600, que es el 78% del pool, no lo es:
el gate se dispara en rachas, así que muchas evaluaciones del mismo símbolo comparten parte de su
horizonte, y un movimiento de mercado activa varios símbolos en la misma vela. Este módulo mide
cuánto infla eso el error estándar y recalcula el n. No llama a nadie.

    python -m crypto_agents.design_effect <directorio> --delta 0.002

Seis decisiones que lo mantienen honesto:

- **No puntúa, no resta y no elige velas por su cuenta.** Las decisiones de las líneas base son
  las de `baselines.py`, los vectores salen de `criteria.paired_returns`, la resta de
  `metrics.paired_differences` y cada submuestra de `selection.select_bars`, que es la elección
  de `select_activations`. `sigma_D`, la viabilidad, el coste y la pared son los de
  `paired_dispersion`.
- **Las series del pool se atan a la corrida.** En las evaluaciones que la etapa 1 ya corrió,
  cada línea base tiene que proponer aquí la misma acción, el mismo stop y rendir lo mismo que en
  su journal. Si no, el comando sale con 1: las series no serían las de la ablación.
- **Las diferencias entre líneas base son un sustituto, y se rotulan así.** `solo` menos una
  línea base no se puede calcular sobre el pool sin pagar. Lo que sí se puede es la diferencia
  entre dos líneas base, que tiene la misma forma: dos políticas sobre las mismas velas.
- **El clúster es un bloque de tiempo, no un símbolo.** Bloques UTC de L días alineados al epoch
  que juntan todos los símbolos: la dependencia que se quiere atrapar es la de lo que pasa a la
  vez. El error estándar agrupado es CR1 para la media.
- **La regla se fijó antes de ver los números.** Para cada n, el efecto de diseño de
  planificación es el máximo, sobre las series sustituto y los esquemas elegibles, del percentil
  90 entre submuestras. Un esquema con menos de 50 clústeres se reporta y no es elegible. El
  reporte dice qué serie y qué L lo dieron, y no elige nada más.
- **Ninguna media.** Ningún modelo de este módulo tiene campo de media y el reporte no imprime
  ninguna: el n depende de `sigma_D`, de δ y del efecto de diseño, no de una diferencia observada.

Este módulo no importa el router ni nada que lo traiga. Dos tests de arquitectura lo fijan: uno
mira sus imports y otro lo importa en un proceso nuevo y comprueba que el router no se cargó.
"""

from __future__ import annotations

import argparse
import math
import sys
from itertools import combinations, pairwise
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import Field

from crypto_agents.activation import DEFAULT_CONFIG, ActivationConfig
from crypto_agents.activation_sweep import activated_bars
from crypto_agents.audit import AuditError
from crypto_agents.baselines import (
    always_buy_proposal,
    always_sell_proposal,
    random_proposal,
    trend_proposal,
)
from crypto_agents.criteria import (
    BASELINE_ARMS,
    PERCENT_DIGITS,
    CriteriaError,
    add_delta_arguments,
    delta_flag,
    paired_returns,
    render_invalid,
    resolve_delta,
    unregistered_delta,
)
from crypto_agents.dispersion import DEFAULT_POWER
from crypto_agents.indicators import DEFAULT_PRESET, IndicatorPreset, enrich, to_indicator_set
from crypto_agents.journal import EvaluationRecord
from crypto_agents.market import build_snapshot, timeframe_to_timedelta, to_dataframe
from crypto_agents.metrics import (
    MIN_CLUSTERS,
    ClusteredError,
    block_of,
    clustered_error,
    clustered_errors,
    clusters_of,
    nearest_rank,
    paired_differences,
)
from crypto_agents.paired_dispersion import (
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    PAIRS,
    InvalidRunError,
    MissingPair,
    Recalculation,
    RecalculationError,
    Sources,
    commit_label,
    fit,
    n_for_power,
    paid_arms,
    pool_cells,
    power_at,
    read_stage,
    stage_hours,
    stage_usd,
)
from crypto_agents.paired_dispersion import build_report as build_recalculation
from crypto_agents.selection import (
    HISTORY_DIR,
    SelectionError,
    confirm_activation,
    replay_window,
    select_bars,
    stratify,
)
from crypto_agents.settings import DEFAULT_PRICING
from crypto_agents.state import FrozenModel

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import datetime

    from crypto_agents.activation_sweep import ActivatedBar
    from crypto_agents.paired_dispersion import PairDispersion, Pool
    from crypto_agents.selection import SelectionManifest
    from crypto_agents.state import IndicatorSet, MarketSnapshot, Proposal

__all__ = [
    "BLOCK_DAYS",
    "DEFAULT_SUBSAMPLES",
    "DEFAULT_SUBSAMPLE_SEED",
    "MIN_CLUSTERS",
    "PLANNING_PERCENT",
    "SUBSAMPLE_SIZES",
    "SURROGATE_PAIRS",
    "TARGET_ARM",
    "ClusteredError",
    "Coactivation",
    "DesignEffect",
    "DesignEffectError",
    "Overlap",
    "PairCheck",
    "Planning",
    "PoolBar",
    "PoolRecords",
    "RecomputedN",
    "Series",
    "SeriesError",
    "SizeDeff",
    "StageTie",
    "baseline_proposals",
    "block_of",
    "build_report",
    "cells_agree",
    "clustered_error",
    "clustered_errors",
    "clusters_of",
    "coactivation",
    "deff_by_size",
    "effective_n",
    "evaluated",
    "main",
    "overlap_by_symbol",
    "planned_power",
    "planning_deff",
    "pool_errors",
    "pool_records",
    "pool_run_id",
    "pool_series",
    "recomputed_n",
    "render_report",
    "required_at",
    "satisfies",
    "stage_check",
    "stage_ties",
    "subsample",
    "used_deff",
]

BLOCK_DAYS = (1, 2, 3, 7)
"""Los L de los esquemas de clúster, en días. Fijados antes de ver ningún resultado."""

SUBSAMPLE_SIZES = (140, 280, 525, 980, 1960, 3290, 3605)
"""La rejilla de n: el de la etapa 1, su doble y los n que `paired_dispersion` dio, hasta 35 por
la celda más pequeña del pool."""

DEFAULT_SUBSAMPLES = 200
"""K: submuestras estratificadas por cada n."""

DEFAULT_SUBSAMPLE_SEED = 20261012
"""Semilla de las submuestras: la fecha del bloque. La submuestra k usa `semilla + k`, la misma
para todos los n, así que las de una semilla están anidadas a lo largo de la rejilla."""

PLANNING_PERCENT = 90
"""El percentil entre submuestras que la regla toma de cada serie y esquema."""

SURROGATE_PAIRS: tuple[tuple[str, str], ...] = tuple(combinations(BASELINE_ARMS, 2))
"""Las seis diferencias pareadas entre líneas base: el sustituto de `solo` menos una línea base."""

TARGET_ARM = "solo"
"""El brazo cuyo n contra cada línea base se recalcula."""

EXCHANGE = "binance"

REPLAY_NAMESPACE = uuid5(NAMESPACE_URL, "https://crypto-agents/replay")
"""El espacio de nombres de `replay.REPLAY_NAMESPACE`, repetido: `replay.py` importa el router y
este módulo no puede. `tests/test_design_effect.py` fija que los dos son el mismo."""

_SIGMA_LABELS = ("extremo bajo", "puntual", "extremo alto")
_SEPARATOR = " - "


class DesignEffectError(RuntimeError):
    """El efecto de diseño no se puede medir sin inventar: falta un brazo, o una serie no cuadra."""


# ─────────────────────────────────────────── Modelos ──────────────────────────────────────────────


class PoolBar(NamedTuple):
    """Una activación confirmada del pool, con lo que la identifica en la corrida."""

    symbol: str
    candle: int
    stratum: int
    at: datetime
    """El cierre de la vela evaluada, que es el instante de entrada y `EvaluationRecord.at`."""

    run_id: UUID


class PoolRecords(NamedTuple):
    """El pool evaluado por las cuatro líneas base.

    `NamedTuple` y no `FrozenModel` por lo mismo que `ReplayPlan`: son miles de registros ya
    validados al construirse, y revalidarlos al agruparlos es tiempo pagado por nada.
    """

    bars: tuple[PoolBar, ...]
    """Las confirmadas, por símbolo y, dentro de cada uno, por vela."""

    arms: dict[str, tuple[EvaluationRecord, ...]]
    """Los registros de cada línea base, en el orden de `bars`."""

    cells: dict[str, list[list[ActivatedBar]]]
    """Por símbolo, las celdas de `selection.stratify`: todas las candidatas."""

    confirmed: dict[tuple[str, int], tuple[str, ...]]
    """(símbolo, vela) de cada confirmada, con lo que la disparó en la ventana del replay."""

    activations: int
    """Activaciones del barrido, tengan o no horizonte por delante."""


class Series(NamedTuple):
    """El retorno por evaluación de una línea base, o la diferencia pareada entre dos."""

    name: str
    surrogate: bool
    """Si es una de las seis diferencias que hacen de sustituto."""

    values: tuple[float, ...]
    """Un valor por activación del pool, en el orden de `PoolRecords.bars`."""


class StageTie(FrozenModel):
    """Una línea base sobre el pool contra la misma línea base en la corrida."""

    arm: str
    evaluations: int = Field(ge=0)
    """Evaluaciones de la línea base en la corrida."""

    found: int = Field(ge=0)
    """De esas, las que están en el pool por `run_id`."""

    actions: int = Field(ge=0)
    """Evaluaciones compartidas con otra acción."""

    stops: int = Field(ge=0)
    """Evaluaciones compartidas con otro stop declarado."""

    returns: int = Field(ge=0)
    """Evaluaciones compartidas con otro retorno."""

    @property
    def exact(self) -> bool:
        """Si el pool y la corrida dicen lo mismo de cada evaluación que comparten."""
        return (
            self.evaluations > 0
            and self.found == self.evaluations
            and self.actions == self.stops == self.returns == 0
        )


class Overlap(FrozenModel):
    """Cuántas evaluaciones de un símbolo comparten horizonte con otra del mismo símbolo."""

    symbol: str
    evaluations: int = Field(ge=0)
    overlapping: int = Field(ge=0)


class Coactivation(FrozenModel):
    """Cuántas velas activan a la vez a ese número de símbolos."""

    symbols: int = Field(gt=0)
    candles: int = Field(ge=0)

    @property
    def evaluations(self) -> int:
        """Las evaluaciones que caen en esas velas."""
        return self.symbols * self.candles


class SeriesError(FrozenModel):
    """Una serie del pool entero bajo un esquema de clúster."""

    series: str
    surrogate: bool
    days: int = Field(gt=0)
    error: ClusteredError


class SizeDeff(FrozenModel):
    """El efecto de diseño de una serie, un esquema y un n, entre submuestras."""

    n: int = Field(ge=2)
    series: str
    surrogate: bool
    days: int = Field(gt=0)
    subsamples: int = Field(gt=0)
    clusters_min: int = Field(ge=1)
    clusters_median: int = Field(ge=1)
    undetermined: int = Field(ge=0)
    """Submuestras sin efecto de diseño: una serie constante no tiene error que comparar."""

    deff_median: float | None = Field(default=None, ge=0.0)
    deff_high: float | None = Field(default=None, ge=0.0)
    """El percentil `PLANNING_PERCENT` entre las submuestras que sí lo tienen."""

    @property
    def eligible(self) -> bool:
        """Si ninguna submuestra baja de `MIN_CLUSTERS` clústeres con este esquema."""
        return self.clusters_min >= MIN_CLUSTERS


class Planning(FrozenModel):
    """El efecto de diseño de planificación de un n, con quién lo dio."""

    n: int = Field(ge=2)
    deff: float | None = Field(default=None, ge=0.0)
    """`None` si ningún esquema es elegible en ese n: entonces el n no se puede juzgar."""

    series: str | None = None
    days: int | None = None


class PairCheck(FrozenModel):
    """Un par de la corrida bajo un esquema de clúster."""

    arm: str
    reference: str
    days: int = Field(gt=0)
    error: ClusteredError


class RecomputedN(FrozenModel):
    """El n que pide un par con el efecto de diseño de planificación, para una `sigma_D`."""

    arm: str
    reference: str
    label: str
    sigma: float = Field(ge=0.0)
    independent: float = Field(ge=0.0)
    """El n de `paired_dispersion.n_for_power`: el que supone evaluaciones independientes."""

    n: int | None = None
    """El menor n de la rejilla que cumple, o `None` si ninguno cumple."""

    deff: float | None = None
    """El de planificación en ese n."""

    required: float | None = None
    """`independent · deff`: lo que el par pide en ese n. Cumple porque no pasa de n."""


class DesignEffect(FrozenModel):
    """Todo lo que el reporte sabe. Ninguna media entre estos campos."""

    delta: float = Field(gt=0.0)
    power: float = Field(gt=0.0, lt=1.0)
    horizon: int = Field(gt=0)
    timeframe: str
    baseline_seed: int
    """Semilla de `random_uniform`: la del manifiesto de la corrida."""

    subsamples: int = Field(gt=0)
    subsample_seed: int
    sizes: tuple[int, ...]
    block_days: tuple[int, ...]
    activations: int = Field(ge=0)
    candidates: int = Field(ge=0)
    confirmed: int = Field(ge=0)
    ties: tuple[StageTie, ...]
    manifest_reproduced: bool
    """Si la submuestra del tamaño y la semilla del manifiesto es el manifiesto."""

    cells_agree: bool
    """Si las confirmadas por celda son las que cuenta `paired_dispersion.pool_cells`."""

    overlaps: tuple[Overlap, ...]
    coactivations: tuple[Coactivation, ...]
    pool: tuple[SeriesError, ...]
    by_size: tuple[SizeDeff, ...]
    planning: tuple[Planning, ...]
    checks: tuple[PairCheck, ...]
    missing: tuple[MissingPair, ...] = ()
    recomputed: tuple[RecomputedN, ...]
    recalculation: Recalculation

    @property
    def consistent(self) -> bool:
        """Si las series del pool son las de la ablación y lo heredado de la corrida cuadra."""
        return (
            all(tie.exact for tie in self.ties)
            and self.manifest_reproduced
            and self.cells_agree
            and self.recalculation.consistent
        )


# ─────────────────────────────────────── El pool evaluado ─────────────────────────────────────────


def pool_run_id(symbol: str, timeframe: str, moment: datetime) -> UUID:
    """El `run_id` que el replay da a esa evaluación: lo mismo que `replay.replay_run_id`.

    Se replica porque `replay.py` importa el router. Hace falta el mismo id, no uno cualquiera:
    `random_uniform` saca su acción de la semilla y del `run_id`, y con otro id la línea al azar
    del pool no sería la de la corrida en las evaluaciones que comparten.
    """
    return uuid5(REPLAY_NAMESPACE, f"{symbol}|{timeframe}|{moment.isoformat()}")


def evaluated(
    rows: Sequence[Sequence[float]],
    bar: ActivatedBar,
    run_id: UUID,
    candle_limit: int,
    preset: IndicatorPreset,
) -> tuple[MarketSnapshot, IndicatorSet]:
    """El snapshot y los indicadores que el brazo vería en esa vela.

    Sobre `selection.replay_window`, no sobre la serie entera: EMA, ADX y ATR son recursivos, y
    calculados sobre la serie completa darían otra regla de tendencia y otro stop común que los
    de la ablación.
    """
    window = to_dataframe(replay_window(rows, bar.index, candle_limit))
    snapshot = build_snapshot(window, run_id, EXCHANGE, bar.symbol, bar.timeframe)
    return snapshot, to_indicator_set(enrich(window, preset), preset)


def baseline_proposals(
    snapshot: MarketSnapshot,
    indicators: IndicatorSet,
    seed: int,
    run_id: UUID,
    preset: IndicatorPreset,
) -> dict[str, Proposal]:
    """Lo que propone cada línea base en una evaluación, con las funciones de `baselines.py`.

    Las mismas cuatro llamadas que hacen los nodos `decide_*` del grafo, con los mismos
    argumentos: la semilla del plan y el `run_id` para la línea al azar, el preset para la regla.
    """
    return {
        "always_buy": always_buy_proposal(snapshot, indicators),
        "always_sell": always_sell_proposal(snapshot, indicators),
        "random_uniform": random_proposal(snapshot, indicators, seed, run_id),
        "rule_trend": trend_proposal(snapshot, indicators, preset),
    }


def pool_records(
    manifest: SelectionManifest,
    histories: Mapping[str, Sequence[Sequence[float]]],
    preset: IndicatorPreset = DEFAULT_PRESET,
    config: ActivationConfig = DEFAULT_CONFIG,
    progress: bool = False,
) -> PoolRecords:
    """Cada activación confirmada del pool, evaluada por las cuatro líneas base.

    El universo es el de la selección: las activaciones del barrido con el horizonte por delante
    (`selection.stratify`) que el gate abre también con la ventana del replay
    (`selection.confirm_activation`). Cada registro lleva el `run_id` y el instante que le daría
    el replay —el cierre de la vela evaluada—, así que las evaluaciones que la corrida ya hizo
    tienen aquí el mismo id.
    """
    period = timeframe_to_timedelta(manifest.timeframe)
    bars: list[PoolBar] = []
    arms: dict[str, list[EvaluationRecord]] = {name: [] for name in BASELINE_ARMS}
    cells: dict[str, list[list[ActivatedBar]]] = {}
    confirmed: dict[tuple[str, int], tuple[str, ...]] = {}
    activations = 0

    for ref in manifest.series:
        rows = histories[ref.symbol]
        found = activated_bars(rows, ref.symbol, manifest.timeframe, config, preset)
        activations += len(found)
        cells[ref.symbol] = stratify(rows, found, manifest.strata, manifest.horizon)
        for stratum, cell in enumerate(cells[ref.symbol]):
            for bar in cell:
                triggers = confirm_activation(
                    rows, bar.index, manifest.candle_limit, config, preset
                )
                if triggers is None:
                    continue
                confirmed[(bar.symbol, bar.index)] = triggers
                moment = bar.at + period
                run_id = pool_run_id(bar.symbol, bar.timeframe, moment)
                snapshot, indicators = evaluated(rows, bar, run_id, manifest.candle_limit, preset)
                proposals = baseline_proposals(snapshot, indicators, manifest.seed, run_id, preset)
                if tuple(proposals) != BASELINE_ARMS:
                    raise DesignEffectError(
                        f"las líneas base no son las de criteria: {', '.join(proposals)}"
                    )
                bars.append(PoolBar(bar.symbol, bar.index, stratum, moment, run_id))
                for name, proposal in proposals.items():
                    arms[name].append(
                        EvaluationRecord(
                            run_id=run_id,
                            at=moment,
                            symbol=bar.symbol,
                            timeframe=bar.timeframe,
                            snapshot=snapshot,
                            indicators=indicators,
                            proposal=proposal,
                        )
                    )
        if progress:
            print(f"{ref.symbol:10} {len(found):>5} activaciones", file=sys.stderr)

    return PoolRecords(
        bars=tuple(bars),
        arms={name: tuple(records) for name, records in arms.items()},
        cells=cells,
        confirmed=confirmed,
        activations=activations,
    )


def pool_series(
    pool: PoolRecords, histories: Mapping[str, Sequence[Sequence[float]]], horizon: int
) -> tuple[Series, ...]:
    """Las cuatro series de retorno por evaluación y las seis diferencias pareadas entre ellas.

    Cada par pasa por `criteria.paired_returns`, que empareja por `run_id` y puntúa con el stop
    común, y la resta es `metrics.paired_differences`: lo mismo que haría `criteria` si las
    líneas base fueran dos brazos de una corrida. El vector de una línea base tiene que salir
    igual de cada par en que aparece; si no, la puntuación dependería de con quién se empareja.
    """
    order = [bar.run_id for bar in pool.bars]
    singles: dict[str, dict[UUID, float]] = {}
    surrogates: list[Series] = []
    for arm, reference in SURROGATE_PAIRS:
        aligned = paired_returns(arm, reference, pool.arms, histories, horizon)
        if isinstance(aligned, str):
            raise DesignEffectError(f"{arm} contra {reference} sobre el pool: {aligned}")
        ids = [record.run_id for record in aligned.arm_records]
        if aligned.unpaired or set(ids) != set(order):
            raise DesignEffectError(
                f"{arm} contra {reference}: las dos líneas base no cubren el mismo pool"
            )
        differences = dict(
            zip(
                ids,
                paired_differences(aligned.arm_returns, aligned.reference_returns),
                strict=True,
            )
        )
        for name, returns in ((arm, aligned.arm_returns), (reference, aligned.reference_returns)):
            scored = dict(zip(ids, returns, strict=True))
            if singles.setdefault(name, scored) != scored:
                raise DesignEffectError(f"{name} no rinde lo mismo en todos los pares en que entra")
        surrogates.append(
            Series(
                name=f"{arm}{_SEPARATOR}{reference}",
                surrogate=True,
                values=tuple(differences[run_id] for run_id in order),
            )
        )
    return (
        *(
            Series(name, False, tuple(singles[name][run_id] for run_id in order))
            for name in BASELINE_ARMS
        ),
        *surrogates,
    )


def stage_ties(
    pool: PoolRecords,
    stage: Mapping[str, Sequence[EvaluationRecord]],
    histories: Mapping[str, Sequence[Sequence[float]]],
    horizon: int,
) -> tuple[StageTie, ...]:
    """Cada línea base del pool contra su journal de la corrida, evaluación por evaluación.

    Es lo que ata las series de este módulo al arnés de la ablación sin pasar por él: donde los
    dos caminos evaluaron la misma vela, tienen que haber propuesto y rendido lo mismo. El
    retorno se compara como lo compara `criteria`, emparejando por `run_id`.
    """
    ties: list[StageTie] = []
    for name in BASELINE_ARMS:
        if name not in stage:
            raise DesignEffectError(
                f"la corrida no trae la línea base {name}: no hay con qué atar las series del pool"
            )
        ours = {record.run_id: record for record in pool.arms[name]}
        theirs = [record for record in stage[name] if record.run_id in ours]
        mine = [ours[record.run_id] for record in theirs]
        differing = 0
        if theirs:
            aligned = paired_returns(
                "pool", "corrida", {"pool": mine, "corrida": theirs}, histories, horizon
            )
            if isinstance(aligned, str):
                raise DesignEffectError(f"{name} del pool contra la corrida: {aligned}")
            differing = sum(
                1
                for value in paired_differences(aligned.arm_returns, aligned.reference_returns)
                if value != 0.0
            )
        ties.append(
            StageTie(
                arm=name,
                evaluations=len(stage[name]),
                found=len(theirs),
                actions=sum(
                    1 for a, b in zip(mine, theirs, strict=True) if _action(a) != _action(b)
                ),
                stops=sum(1 for a, b in zip(mine, theirs, strict=True) if _stop(a) != _stop(b)),
                returns=differing,
            )
        )
    return tuple(ties)


def _action(record: EvaluationRecord) -> str | None:
    proposed = record.proposed
    return None if proposed is None else proposed.action.value


def _stop(record: EvaluationRecord) -> float | None:
    proposed = record.proposed
    return None if proposed is None else proposed.invalidation_price


# ───────────────────────────────── Estructura de dependencia ──────────────────────────────────────


def overlap_by_symbol(bars: Sequence[PoolBar], horizon: int) -> tuple[Overlap, ...]:
    """Por símbolo, cuántas evaluaciones comparten alguna vela de su horizonte con otra.

    La evaluación de la vela `i` entra a su cierre y recorre las velas `i + 1 … i + horizonte`.
    Dos del mismo símbolo comparten vela cuando sus posiciones distan menos de `horizonte`. Es la
    ventana de diseño: una posición que salió antes por el stop ya no estaba abierta, pero su
    retorno se decidió dentro de esa ventana igual.
    """
    if horizon <= 0:
        raise ValueError(f"el horizonte tiene que ser positivo: {horizon}")
    indices: dict[str, list[int]] = {}
    for bar in bars:
        indices.setdefault(bar.symbol, []).append(bar.candle)
    found: list[Overlap] = []
    for symbol, positions in indices.items():
        ordered = sorted(positions)
        shared = [later - earlier < horizon for earlier, later in pairwise(ordered)]
        overlapping = sum(
            1
            for place in range(len(ordered))
            if (place > 0 and shared[place - 1]) or (place < len(shared) and shared[place])
        )
        found.append(Overlap(symbol=symbol, evaluations=len(ordered), overlapping=overlapping))
    return tuple(found)


def coactivation(bars: Sequence[PoolBar]) -> tuple[Coactivation, ...]:
    """Cuántas velas activan a un símbolo, a dos, a tres… a la vez."""
    together: dict[datetime, set[str]] = {}
    for bar in bars:
        together.setdefault(bar.at, set()).add(bar.symbol)
    counts: dict[int, int] = {}
    for symbols in together.values():
        counts[len(symbols)] = counts.get(len(symbols), 0) + 1
    return tuple(Coactivation(symbols=size, candles=counts[size]) for size in sorted(counts))


# ───────────────────────────────── Error estándar agrupado ────────────────────────────────────────


def pool_errors(
    series: Sequence[Series], bars: Sequence[PoolBar], block_days: Sequence[int] = BLOCK_DAYS
) -> tuple[SeriesError, ...]:
    """Cada serie sobre el pool entero, bajo cada esquema de clúster."""
    schemes = {days: clusters_of(bars, days) for days in block_days}
    return tuple(
        SeriesError(series=item.name, surrogate=item.surrogate, days=days, error=error)
        for item in series
        for days, error in clustered_errors(item.values, schemes).items()
    )


# ─────────────────────────────── Efecto de diseño en función de n ─────────────────────────────────


def subsample(
    cells: Mapping[str, Sequence[Sequence[ActivatedBar]]],
    confirmed: Mapping[tuple[str, int], tuple[str, ...]],
    n: int,
    seed: int,
) -> frozenset[tuple[str, int]]:
    """Las velas de la selección estratificada de tamaño `n` con esa semilla.

    Es `selection.select_bars`, la elección de `select_activations`, con la confirmación leída
    de una tabla en lugar de recalculada: con la misma semilla y el mismo pool, la submuestra es
    el manifiesto que `python -m crypto_agents.selection` escribiría.
    """
    picks = select_bars(cells, n, seed, lambda bar: confirmed.get((bar.symbol, bar.index)))
    return frozenset((pick.bar.symbol, pick.bar.index) for pick in picks)


def deff_by_size(
    series: Sequence[Series],
    pool: PoolRecords,
    sizes: Sequence[int] = SUBSAMPLE_SIZES,
    subsamples: int = DEFAULT_SUBSAMPLES,
    seed: int = DEFAULT_SUBSAMPLE_SEED,
    block_days: Sequence[int] = BLOCK_DAYS,
    progress: bool = False,
) -> tuple[SizeDeff, ...]:
    """Mediana y percentil alto del efecto de diseño, por n, serie y esquema.

    Para cada n se sacan `subsamples` selecciones estratificadas, la k-ésima con `seed + k`, y
    sobre cada una se calcula el error agrupado de cada serie bajo cada esquema. Los clústeres
    de cada evaluación se asignan una vez, sobre el pool, y cada submuestra se queda con los
    suyos: el bloque de un instante no depende de con quién se muestrea.
    """
    if subsamples <= 0:
        raise ValueError(f"hace falta al menos una submuestra: {subsamples}")
    place = {(bar.symbol, bar.candle): position for position, bar in enumerate(pool.bars)}
    blocks = {days: clusters_of(pool.bars, days) for days in block_days}
    rows: list[SizeDeff] = []
    for n in sizes:
        deffs: dict[tuple[str, int], list[float]] = {}
        groups: dict[int, list[int]] = {days: [] for days in block_days}
        for k in range(subsamples):
            chosen = sorted(
                place[key] for key in subsample(pool.cells, pool.confirmed, n, seed + k)
            )
            schemes = {days: [blocks[days][position] for position in chosen] for days in block_days}
            for item in series:
                errors = clustered_errors([item.values[position] for position in chosen], schemes)
                for days, error in errors.items():
                    if error.deff is not None:
                        deffs.setdefault((item.name, days), []).append(error.deff)
            for days, clusters in schemes.items():
                groups[days].append(len(set(clusters)))
        for item in series:
            for days in block_days:
                found = deffs.get((item.name, days), [])
                rows.append(
                    SizeDeff(
                        n=n,
                        series=item.name,
                        surrogate=item.surrogate,
                        days=days,
                        subsamples=subsamples,
                        clusters_min=min(groups[days]),
                        clusters_median=round(nearest_rank(groups[days], 50)),
                        undetermined=subsamples - len(found),
                        deff_median=nearest_rank(found, 50) if found else None,
                        deff_high=nearest_rank(found, PLANNING_PERCENT) if found else None,
                    )
                )
        if progress:
            print(f"n = {n:>5}: {subsamples} submuestras", file=sys.stderr)
    return tuple(rows)


def planning_deff(rows: Sequence[SizeDeff]) -> tuple[Planning, ...]:
    """Para cada n, el mayor percentil alto entre las series sustituto y los esquemas elegibles.

    La regla del bloque, fijada antes de ver los números. Las series de una sola línea base se
    reportan y no entran: lo que se dimensiona es una diferencia pareada. Un esquema que en ese
    n deja alguna submuestra con menos de `MIN_CLUSTERS` clústeres tampoco entra, aunque dé el
    valor más alto. Sin ninguno elegible el n queda sin efecto de diseño, que no es lo mismo que
    tener 1.
    """
    plans: list[Planning] = []
    for n in sorted({row.n for row in rows}):
        admitted = [
            row
            for row in rows
            if row.n == n and row.surrogate and row.eligible and row.deff_high is not None
        ]
        if not admitted:
            plans.append(Planning(n=n))
            continue
        worst = max(admitted, key=lambda row: row.deff_high or 0.0)
        plans.append(Planning(n=n, deff=worst.deff_high, series=worst.series, days=worst.days))
    return tuple(plans)


# ─────────────────────────────── Comprobación sobre la corrida y n ────────────────────────────────


def stage_check(
    arms: Mapping[str, Sequence[EvaluationRecord]],
    histories: Mapping[str, Sequence[Sequence[float]]],
    horizon: int,
    block_days: Sequence[int] = BLOCK_DAYS,
) -> tuple[tuple[PairCheck, ...], tuple[MissingPair, ...]]:
    """El error agrupado de cada par de la corrida, sobre los vectores que resta `criteria`.

    Los nueve pares de `paired_dispersion.PAIRS`. Con una muestra tan repartida en el tiempo
    como la de la etapa 1 casi todos los bloques llevan una sola evaluación, y entonces el error
    agrupado es el ingenuo.
    """
    checks: list[PairCheck] = []
    missing: list[MissingPair] = []
    for arm, reference in PAIRS:
        aligned = paired_returns(arm, reference, arms, histories, horizon)
        if isinstance(aligned, str):
            missing.append(MissingPair(arm=arm, reference=reference, why=aligned))
            continue
        differences = paired_differences(aligned.arm_returns, aligned.reference_returns)
        if len(differences) < 2:
            missing.append(
                MissingPair(arm=arm, reference=reference, why="menos de dos evaluaciones")
            )
            continue
        schemes = {days: clusters_of(aligned.arm_records, days) for days in block_days}
        checks.extend(
            PairCheck(arm=arm, reference=reference, days=days, error=error)
            for days, error in clustered_errors(differences, schemes).items()
        )
    return tuple(checks), tuple(missing)


def required_at(independent: float, plan: Planning) -> float | None:
    """Lo que un par pide en ese n: su n de evaluaciones independientes por el deff de ese n.

    `((1.96 + z)·sigma_D·√deff(n)/δ)²` es `((1.96 + z)·sigma_D/δ)² · deff(n)`: el efecto de
    diseño multiplica el n, no la desviación. `None` si ese n no tiene efecto de diseño de
    planificación.
    """
    if plan.deff is None:
        return None
    return independent * plan.deff


def satisfies(independent: float, plan: Planning) -> bool:
    """Si lo que el par pide en ese n no pasa de n. Un n que no se puede juzgar no cumple."""
    needed = required_at(independent, plan)
    return needed is not None and needed <= plan.n


def used_deff(plan: Planning) -> float | None:
    """El efecto de diseño con el que se calcula la potencia: el de planificación, con suelo en 1.

    Es el suelo de la enmienda 4 visto desde aquí. `criteria` juzga con `max(EE ingenuo, EE
    agrupado)`, y como `deff = (EE agrupado / EE ingenuo)²`, eso es `max(1, deff)`: la potencia
    es la del intervalo que se va a construir, y ese intervalo nunca es más estrecho que el
    ingenuo. La regla de planificación (`planning_deff`) y el n de la sección 7 (`required_at`)
    no llevan suelo: se fijaron antes de los números y siguen como estaban. `None` si ese n no
    tiene efecto de diseño de planificación.
    """
    if plan.deff is None:
        return None
    return max(1.0, plan.deff)


def effective_n(plan: Planning) -> float | None:
    """Cuántas evaluaciones independientes valen las n de ese tamaño: `n / deff usado`.

    Es con lo que se compara el n que `paired_dispersion` pide con evaluaciones independientes.
    Con el suelo nunca pasa de n: una muestra dependiente no vale más que una independiente del
    mismo tamaño. `None` sin efecto de diseño de planificación.
    """
    deff = used_deff(plan)
    if deff is None:
        return None
    return plan.n / deff


def planned_power(sigma: float, plan: Planning, delta: float) -> float | None:
    """La potencia de un par en ese n, con el efecto de diseño usado de ese n.

    La pregunta de `required_at` al revés: no qué n hace falta para una potencia, sino qué
    potencia da un n. `paired_dispersion.power_at` con el error estándar inflado por
    `used_deff`, el de planificación con el suelo de la enmienda 4; donde ese deff no baja de 1
    —toda la rejilla del pool— es la inversa exacta de `required_at`. `None` si ese n no tiene
    efecto de diseño de planificación: no se puede juzgar, que no es lo mismo que suponerle 1.
    """
    deff = used_deff(plan)
    if deff is None:
        return None
    return power_at(sigma, plan.n, delta, deff)


def recomputed_n(
    pairs: Sequence[PairDispersion],
    planning: Sequence[Planning],
    delta: float,
    power: float = DEFAULT_POWER,
    arm: str = TARGET_ARM,
) -> tuple[RecomputedN, ...]:
    """El menor n de la rejilla que cumple, para `arm` contra cada línea base.

    Un n cumple cuando `((1.96 + z)·sigma_D·√deff(n)/δ)² ≤ n`, es decir cuando el n que
    `paired_dispersion` pide con evaluaciones independientes, multiplicado por el efecto de
    diseño de planificación de ese n, no pasa de él. `sigma_D` va con su valor puntual y con los
    dos extremos de su rango bootstrap. Un n sin efecto de diseño de planificación no se puede
    juzgar y se salta.
    """
    ordered = sorted(planning, key=lambda plan: plan.n)
    found: list[RecomputedN] = []
    for pair in pairs:
        if pair.arm != arm or pair.reference not in BASELINE_ARMS:
            continue
        spread = pair.dispersion
        sigmas = (spread.bootstrap.low, spread.sigma, spread.bootstrap.high)
        for label, sigma in zip(_SIGMA_LABELS, sigmas, strict=True):
            independent = n_for_power(sigma, delta, power)
            enough = next((plan for plan in ordered if satisfies(independent, plan)), None)
            found.append(
                RecomputedN(
                    arm=pair.arm,
                    reference=pair.reference,
                    label=label,
                    sigma=sigma,
                    independent=independent,
                    n=None if enough is None else enough.n,
                    deff=None if enough is None else enough.deff,
                    required=None if enough is None else required_at(independent, enough),
                )
            )
    return tuple(found)


def cells_agree(pool: PoolRecords, counted: Pool) -> bool:
    """Si las confirmadas de cada celda son las que cuenta `paired_dispersion.pool_cells`.

    Las dos cuentas salen de pasadas distintas sobre el mismo histórico. La viabilidad de un n
    se juzga con la de `pool_cells` y las series con la de aquí: si no coinciden, el n que
    «cabe» y el n sobre el que se midió el efecto de diseño no son del mismo pool.
    """
    theirs = {(cell.symbol, cell.stratum): cell.confirmed for cell in counted.cells}
    ours: dict[tuple[str, int], int] = {
        (symbol, stratum): 0
        for symbol, cells in pool.cells.items()
        for stratum in range(len(cells))
    }
    for bar in pool.bars:
        ours[(bar.symbol, bar.stratum)] += 1
    return ours == theirs


def build_report(
    stage: Mapping[str, Sequence[EvaluationRecord]],
    histories: Mapping[str, Sequence[Sequence[float]]],
    horizon: int,
    delta: float,
    manifest: SelectionManifest,
    pool: PoolRecords,
    recalculation: Recalculation,
    sizes: Sequence[int] = SUBSAMPLE_SIZES,
    subsamples: int = DEFAULT_SUBSAMPLES,
    subsample_seed: int = DEFAULT_SUBSAMPLE_SEED,
    block_days: Sequence[int] = BLOCK_DAYS,
    power: float = DEFAULT_POWER,
    progress: bool = False,
) -> DesignEffect:
    """El efecto de diseño de un pool ya evaluado y una corrida ya leída. Pura sobre sus argumentos.

    `pool` y `recalculation` llegan calculados: son lo lento, y no dependen de nada de aquí.
    """
    series = pool_series(pool, histories, horizon)
    by_size = deff_by_size(series, pool, sizes, subsamples, subsample_seed, block_days, progress)
    planning = planning_deff(by_size)
    checks, missing = stage_check(stage, histories, horizon, block_days)

    wanted = frozenset((entry.symbol, entry.index) for entry in manifest.entries)
    try:
        reproduced = (
            subsample(pool.cells, pool.confirmed, len(manifest.entries), manifest.seed) == wanted
        )
    except SelectionError:
        reproduced = False

    return DesignEffect(
        delta=delta,
        power=power,
        horizon=horizon,
        timeframe=manifest.timeframe,
        baseline_seed=manifest.seed,
        subsamples=subsamples,
        subsample_seed=subsample_seed,
        sizes=tuple(sizes),
        block_days=tuple(block_days),
        activations=pool.activations,
        candidates=sum(len(cell) for cells in pool.cells.values() for cell in cells),
        confirmed=len(pool.bars),
        ties=stage_ties(pool, stage, histories, horizon),
        manifest_reproduced=reproduced,
        cells_agree=cells_agree(pool, recalculation.pool),
        overlaps=overlap_by_symbol(pool.bars, horizon),
        coactivations=coactivation(pool.bars),
        pool=pool_errors(series, pool.bars, block_days),
        by_size=by_size,
        planning=planning,
        checks=checks,
        missing=missing,
        recomputed=recomputed_n(recalculation.pairs, planning, delta, power),
        recalculation=recalculation,
    )


# ─────────────────────────────────────────── Reporte ──────────────────────────────────────────────


def _pct(value: float, digits: int = PERCENT_DIGITS) -> str:
    """Porcentaje con los decimales de `criteria`, que es con lo que se compara."""
    return f"{value:.{digits}%}"


def _ratio(value: float | None) -> str:
    return "no determinado" if value is None else f"{value:.3f}"


def _usd(value: float | None) -> str:
    return "no determinado" if value is None else f"{value:.2f}"


def _hours(value: float | None) -> str:
    return "no determinado" if value is None else f"{value:.1f}"


def _yes(flag: bool) -> str:
    return "sí" if flag else "no"


def _kind(surrogate: bool) -> str:
    return "sustituto" if surrogate else "línea base"


def _reproduce(command: str, also: str = "") -> list[str]:
    return ["", f"Reproduce: `{command}`{also}", ""]


def _error_cells(error: ClusteredError) -> str:
    clustered = "no determinado" if error.clustered is None else _pct(error.clustered)
    return (
        f"{error.n} | {error.clusters} | {_pct(error.naive)} | {clustered} "
        f"| {_ratio(error.deff)} | {_yes(error.eligible)}"
    )


def _tie_table(report: DesignEffect) -> list[str]:
    lines = [
        "| línea base | evaluaciones en la corrida | de ellas, en el pool | acción distinta "
        "| stop distinto | retorno distinto | coincide |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {tie.arm} | {tie.evaluations} | {tie.found} | {tie.actions} | {tie.stops} "
        f"| {tie.returns} | {'sí' if tie.exact else '**NO**'} |"
        for tie in report.ties
    ]
    return lines


def _overlap_table(report: DesignEffect) -> list[str]:
    lines = [
        "| símbolo | evaluaciones | con horizonte solapado | fracción |",
        "| --- | ---: | ---: | ---: |",
    ]
    rows = [(item.symbol, item.evaluations, item.overlapping) for item in report.overlaps]
    rows.append(
        (
            "todos",
            sum(item.evaluations for item in report.overlaps),
            sum(item.overlapping for item in report.overlaps),
        )
    )
    for symbol, evaluations, overlapping in rows:
        share = "no determinado" if evaluations == 0 else f"{overlapping / evaluations:.1%}"
        lines.append(f"| {symbol} | {evaluations} | {overlapping} | {share} |")
    return lines


def _coactivation_table(report: DesignEffect) -> list[str]:
    candles = sum(item.candles for item in report.coactivations)
    evaluations = sum(item.evaluations for item in report.coactivations)
    lines = [
        "| símbolos activados en la misma vela | velas | fracción de las velas | evaluaciones "
        "| fracción de las evaluaciones |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    lines += [
        f"| {item.symbols} | {item.candles} | {item.candles}/{candles} "
        f"({item.candles / candles:.1%}) | {item.evaluations} "
        f"| {item.evaluations}/{evaluations} ({item.evaluations / evaluations:.1%}) |"
        for item in report.coactivations
    ]
    return lines


def _pool_table(report: DesignEffect) -> list[str]:
    lines = [
        "| serie | tipo | L (días) | n | G | EE ingenuo | EE agrupado | deff | elegible |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    lines += [
        f"| {row.series} | {_kind(row.surrogate)} | {row.days} | {_error_cells(row.error)} |"
        for row in report.pool
    ]
    return lines


def _size_table(report: DesignEffect) -> list[str]:
    lines = [
        "| n | serie | tipo | L (días) | G mínimo | G mediano | deff mediana "
        f"| deff p{PLANNING_PERCENT} | submuestras sin deff | elegible |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    lines += [
        f"| {row.n} | {row.series} | {_kind(row.surrogate)} | {row.days} | {row.clusters_min} "
        f"| {row.clusters_median} | {_ratio(row.deff_median)} | {_ratio(row.deff_high)} "
        f"| {row.undetermined} | {_yes(row.eligible)} |"
        for row in report.by_size
    ]
    return lines


def _planning_table(report: DesignEffect) -> list[str]:
    lines = [
        "| n | deff de planificación | serie que lo dio | L que lo dio (días) |",
        "| ---: | ---: | --- | ---: |",
    ]
    for plan in report.planning:
        if plan.deff is None:
            lines.append(f"| {plan.n} | no determinado: ningún esquema elegible | — | — |")
        else:
            lines.append(f"| {plan.n} | {_ratio(plan.deff)} | {plan.series} | {plan.days} |")
    return lines


def _check_table(report: DesignEffect) -> list[str]:
    lines = [
        "| brazo | referencia | L (días) | n | G | EE ingenuo | EE agrupado | deff | elegible |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    lines += [
        f"| {row.arm} | {row.reference} | {row.days} | {_error_cells(row.error)} |"
        for row in report.checks
    ]
    lines += [
        f"| {item.arm} | {item.reference} | — | — | — | — | — | — | no determinado: {item.why} |"
        for item in report.missing
    ]
    return lines


def _check_summary(report: DesignEffect) -> list[str]:
    lines = [
        "| L (días) | pares | G | deff menor | deff mayor |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for days in report.block_days:
        rows = [row for row in report.checks if row.days == days]
        deffs = [row.error.deff for row in rows if row.error.deff is not None]
        groups = sorted({row.error.clusters for row in rows})
        if not deffs:
            lines.append(f"| {days} | {len(rows)} | — | no determinado | no determinado |")
            continue
        clusters = str(groups[0]) if len(groups) == 1 else f"{groups[0]} a {groups[-1]}"
        lines.append(
            f"| {days} | {len(rows)} | {clusters} | {_ratio(min(deffs))} | {_ratio(max(deffs))} |"
        )
    return lines


def _requirement_table(report: DesignEffect) -> list[str]:
    plans = sorted(report.planning, key=lambda plan: plan.n)
    lines = [
        "| brazo | referencia | sigma_D | "
        + " | ".join(f"en n = {plan.n}" for plan in plans)
        + " |",
        "| --- | --- | --- |" + " ---: |" * len(plans),
    ]
    for row in report.recomputed:
        cells: list[str] = []
        for plan in plans:
            needed = required_at(row.independent, plan)
            if needed is None:
                cells.append("no determinado")
            else:
                sign = "≤" if satisfies(row.independent, plan) else ">"
                cells.append(f"{needed:.1f} {sign} {plan.n}")
        lines.append(f"| {row.arm} | {row.reference} | {row.label} | {' | '.join(cells)} |")
    return lines


def _effective_table(report: DesignEffect) -> list[str]:
    lines = [
        "| n | deff de planificación | deff usado | n efectivo |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for plan in sorted(report.planning, key=lambda plan: plan.n):
        used = used_deff(plan)
        effective = effective_n(plan)
        if used is None or effective is None:
            lines.append(f"| {plan.n} | no determinado: ningún esquema elegible | — | — |")
        else:
            lines.append(f"| {plan.n} | {_ratio(plan.deff)} | {_ratio(used)} | {effective:.1f} |")
    return lines


def _floor_note(report: DesignEffect) -> str:
    """En cuántos n de la rejilla actúa el suelo: los que tienen un deff de planificación < 1."""
    judged = sum(1 for plan in report.planning if plan.deff is not None)
    floored = sorted(
        plan.n for plan in report.planning if plan.deff is not None and plan.deff < 1.0
    )
    if not floored:
        return f"En esta rejilla el suelo no actúa en ninguno de los {judged} n con deff."
    sizes = ", ".join(str(n) for n in floored)
    return f"En esta rejilla el suelo actúa en {len(floored)} de {judged} n: {sizes}."


def _power_table(report: DesignEffect) -> list[str]:
    plans = sorted(report.planning, key=lambda plan: plan.n)
    lines = [
        "| brazo | referencia | sigma_D | valor | "
        + " | ".join(f"en n = {plan.n}" for plan in plans)
        + " |",
        "| --- | --- | --- | --- |" + " ---: |" * len(plans),
    ]
    for row in report.recomputed:
        cells = [
            "no determinado" if power is None else f"{power:.1%}"
            for power in (planned_power(row.sigma, plan, report.delta) for plan in plans)
        ]
        lines.append(
            f"| {row.arm} | {row.reference} | {row.label} | {_pct(row.sigma)} "
            f"| {' | '.join(cells)} |"
        )
    return lines


def _recomputed_table(report: DesignEffect) -> list[str]:
    found = report.recalculation
    unit = found.unit
    reusing = found.pool.capacity(fresh=False)
    fresh = found.pool.capacity(fresh=True)
    lines = [
        "| brazo | referencia | sigma_D | valor | n con evaluaciones independientes "
        "| menor n de la rejilla que cumple | deff de ese n | n que pide con ese deff "
        f"| por celda | reutilizando las {found.stage_n} | sin ellas | n adicional | USD del n "
        "| USD del adicional | horas de pared del n |",
        "|" + " --- |" * 15,
    ]
    for row in report.recomputed:
        head = (
            f"| {row.arm} | {row.reference} | {row.label} | {_pct(row.sigma)} "
            f"| {row.independent:.1f} "
        )
        if row.n is None or row.required is None:
            lines.append(head + "| ninguno de la rejilla | — | — | — | — | — | — | — | — | — |")
            continue
        paid = paid_arms(row.arm, row.reference)
        per_cell = math.ceil(row.n / unit)
        extra = max(0, row.n - found.stage_n)
        lines.append(
            head + f"| {row.n} | {_ratio(row.deff)} | {row.required:.1f} | {per_cell} "
            f"| {fit(per_cell, reusing)} | {fit(per_cell, fresh)} | {extra} "
            f"| {_usd(stage_usd(found.costs, paid, row.n))} "
            f"| {_usd(stage_usd(found.costs, paid, extra))} "
            f"| {_hours(stage_hours(found.clocks, paid, row.n))} |"
        )
    return lines


def render_report(report: DesignEffect, sources: Sources) -> str:
    """El efecto de diseño en Markdown. Sin medias, sin veredicto y sin decir qué etapa correr."""
    command = sources.command
    found = report.recalculation
    smallest = found.pool.smallest(fresh=False)
    smallest_fresh = found.pool.smallest(fresh=True)
    sizes = ", ".join(str(n) for n in report.sizes)
    schemes = ", ".join(str(days) for days in report.block_days)
    last_seed = report.subsample_seed + report.subsamples - 1
    recalculation = (
        f"python -m crypto_agents.paired_dispersion {sources.directory} {delta_flag(report.delta)}"
    )
    criteria = f"python -m crypto_agents.criteria {sources.directory} {delta_flag(report.delta)}"
    unregistered = unregistered_delta(report.delta)

    lines = [
        f"# Efecto de diseño del pool, con la corrida `{sources.directory}`",
        "",
        f"Reproduce: `{command}`",
        "",
        f"- commit de la corrida: {sources.commit}",
        f"- plan: {sources.plan}",
        *(f"- `{name}` sha-256 `{digest}`" for name, digest in sources.journals),
        *(
            f"- histórico `{symbol}`: `{path}`, sha-256 `{digest}`"
            for symbol, path, digest in sources.histories
        ),
        f"- δ = {report.delta:g} ({report.delta:.3%} de retorno por evaluación) · horizonte "
        f"{report.horizon} velas de {report.timeframe} · puntuación: stop común, retorno bruto, "
        "por `criteria.paired_returns`",
        *([] if unregistered is None else [f"- {unregistered}"]),
        f"- `random_uniform` con la semilla del manifiesto de la corrida, {report.baseline_seed}",
        f"- clústeres: bloques UTC de L días alineados al epoch, L ∈ {{{schemes}}}, que juntan "
        "todos los símbolos; a cada evaluación la asigna el cierre de su vela, que es su "
        "entrada; un instante en el borde exacto abre el bloque nuevo",
        f"- submuestras: {report.subsamples} selecciones estratificadas por cada n ∈ {{{sizes}}}, "
        f"con `selection.select_bars` y las semillas {report.subsample_seed} a {last_seed}, las "
        "mismas para todos los n",
        f"- un esquema con G < {MIN_CLUSTERS} se reporta y no es elegible; entre submuestras "
        "cuenta el G mínimo",
        "",
        "## 1. Universo y atadura con la corrida",
        "",
        f"{report.activations} activaciones del barrido sobre {found.symbols} series; "
        f"{report.candidates} con el horizonte por delante; {report.confirmed} que el gate abre "
        "también con la ventana del replay (`selection.confirm_activation`). Esas son el "
        "universo: las que una selección puede entregar.",
        "",
        "Sobre cada una deciden las cuatro líneas base con las funciones de `baselines.py`, y "
        "cada registro lleva el `run_id` que le daría el replay. La tabla compara, en las "
        "evaluaciones que la corrida ya hizo, lo que este comando calcula con lo que dejó "
        "escrito el journal de esa línea base. Tiene que coincidir todo: si no, las series del "
        "pool no son las de la ablación y el comando sale con 1.",
        "",
        *_tie_table(report),
        "",
        f"- la submuestra de {found.stage_n} con la semilla del manifiesto "
        f"({report.baseline_seed}) es el manifiesto de la corrida: "
        f"{'sí' if report.manifest_reproduced else '**NO**'}",
        "- las confirmadas por celda son las que cuenta `paired_dispersion.pool_cells`: "
        f"{'sí' if report.cells_agree else '**NO**'}",
        *_reproduce(command),
        "## 2. Estructura de dependencia",
        "",
        f"**Horizonte solapado.** La evaluación de la vela `i` recorre las velas `i + 1` a "
        f"`i + {report.horizon}`. Dos del mismo símbolo se solapan cuando sus velas distan menos "
        f"de {report.horizon}. Es la ventana de diseño, no la salida real por stop.",
        "",
        *_overlap_table(report),
        "",
        "**Símbolos activados en la misma vela.** Una vela cuenta una vez, con cuántos símbolos "
        "confirmaron activación en ella.",
        "",
        *_coactivation_table(report),
        *_reproduce(command),
        f"## 3. Error estándar agrupado sobre las {report.confirmed} del pool",
        "",
        "- **EE ingenuo**: `stdev(n - 1)/√n`, el que publica `criteria`.",
        "- **EE agrupado**: CR1 para la media, `EE² = G/(G - 1) · Σ_g (Σ_{i∈g} (x_i - x̄))² / n²`, "
        "con G los bloques que tienen alguna evaluación.",
        "- **deff**: `(EE agrupado / EE ingenuo)²`.",
        "- Las seis diferencias entre líneas base hacen de **sustituto** de `solo` menos una "
        "línea base, que no se puede calcular sobre el pool sin pagar.",
        "",
        *_pool_table(report),
        *_reproduce(command),
        "## 4. Efecto de diseño en función de n",
        "",
        f"Mediana y percentil {PLANNING_PERCENT} de deff entre las {report.subsamples} "
        "submuestras de cada n, por rango más cercano. G cambia con la submuestra: van el "
        f"mínimo y el mediano, y el esquema es elegible si el mínimo no baja de {MIN_CLUSTERS}.",
        "",
        *_size_table(report),
        *_reproduce(command),
        "## 5. Efecto de diseño de planificación",
        "",
        f"La regla, fijada antes de ver los números: para cada n, el máximo del percentil "
        f"{PLANNING_PERCENT} de deff sobre las series sustituto y los esquemas elegibles. Las "
        "series de una sola línea base no entran. La tabla dice qué serie y qué L lo dieron.",
        "",
        *_planning_table(report),
        *_reproduce(command),
        "## 6. Comprobación sobre la corrida",
        "",
        f"Los nueve pares de la corrida, sobre los vectores que resta `criteria`, con sus "
        f"{found.stage_n} evaluaciones. Con una evaluación por bloque el EE agrupado es "
        "exactamente el ingenuo y deff vale 1; cuántos bloques tienen más de una lo dice G "
        "frente a n. La primera tabla resume, por L, entre qué valores cae deff en los pares.",
        "",
        *_check_summary(report),
        "",
        *_check_table(report),
        *_reproduce(command, f" (los EE ingenuos: `{criteria}`)"),
        f"## 7. n recalculado para `{TARGET_ARM}` contra cada línea base",
        "",
        f"- **n con evaluaciones independientes**: `((1.96 + z)·sigma_D/δ)²`, con z el cuantil "
        f"normal de {report.power:.0%}. Es el de `paired_dispersion`, sin redondear.",
        "- **menor n de la rejilla que cumple**: el primer n con "
        "`((1.96 + z)·sigma_D·√deff(n)/δ)² ≤ n`, con deff(n) el de planificación de la sección 5.",
        "- **n que pide con ese deff**: el de evaluaciones independientes por deff(n).",
        "- `sigma_D` es la de `paired_dispersion`: puntual y los dos extremos de su rango "
        f"bootstrap ({found.resamples} remuestreos, semilla {found.seed}).",
        f"- **por celda**: n entre las {found.unit} celdas. **reutilizando las "
        f"{found.stage_n}**: contra la celda más pequeña del pool, {smallest.symbol} tramo "
        f"{smallest.stratum}, con {smallest.confirmed} confirmadas. **sin ellas**: contra "
        f"{smallest_fresh.symbol} tramo {smallest_fresh.stratum}, con {smallest_fresh.fresh}.",
        f"- **USD** y **horas**: n por el coste y la pared por evaluación de `{TARGET_ARM}` en "
        "la corrida; las líneas base no llaman a ningún modelo.",
        "",
        *_recomputed_table(report),
        "",
        "Lo que pide cada par en cada n de la rejilla —el n de evaluaciones independientes por "
        "el deff de planificación de ese n—, frente a ese n. Cumple donde no lo pasa.",
        "",
        *_requirement_table(report),
        *_reproduce(command, f" (sigma_D, coste y pared: `{recalculation}`)"),
        f"## 8. Potencia de `{TARGET_ARM}` contra cada línea base en cada n de la rejilla",
        "",
        "La pregunta de la sección 7 al revés: no qué n hace falta para una potencia, sino qué "
        "potencia da cada n. `Φ(δ·√(n/deff(n))/sigma_D - 1.96)`: la probabilidad de que una "
        "diferencia real de δ deje el IC95% entero sobre 0, con deff(n) el usado de la tabla "
        "de abajo y `sigma_D` la de la sección 7. Es la misma aproximación normal, y en ella "
        "no entra ninguna diferencia observada.",
        "",
        "**El suelo de la enmienda 4.** `criteria` juzga con `max(EE ingenuo, EE agrupado)`, "
        "que en deff es `max(1, deff)`: el **deff usado** es el de planificación de la sección "
        "5 con ese suelo, y la potencia es la del intervalo que se va a construir. "
        f"{_floor_note(report)} La sección 7 es la regla fijada antes de los números y no lo "
        "lleva. **n efectivo** es `n / deff usado`: cuántas evaluaciones independientes valen "
        "las n, que es con lo que se compara el «n con evaluaciones independientes» de la "
        "sección 7.",
        "",
        *_effective_table(report),
        "",
        *_power_table(report),
        *_reproduce(command, f" (sigma_D: `{recalculation}`)"),
        "## Lo que estas cifras suponen",
        "",
        "- El sustituto es una diferencia entre dos políticas sin modelo. `solo` discrepa de una "
        "línea base donde su decisor quiere, no donde una regla fija; que su diferencia dependa "
        "en el tiempo como la de dos líneas base es un supuesto, no una medida.",
        "- CR1 corrige por el número de clústeres y no por su tamaño desigual. Con pocos "
        f"clústeres subestima; por eso un esquema con G < {MIN_CLUSTERS} no entra.",
        "- Un bloque corta la dependencia en su borde: dos evaluaciones a cada lado de él "
        "cuentan como independientes. Los L largos atrapan más y dejan menos clústeres.",
        f"- Las {report.subsamples} submuestras de un n salen del mismo pool y se pisan más "
        "cuanto mayor es n: con el n más alto casi todas comparten la mayoría de sus "
        "evaluaciones, y su percentil dice poco más que su mediana.",
        "- `sigma_D` sale de la corrida y su rango es el de un bootstrap, no un intervalo de "
        "confianza. El coste y la pared son los de esa corrida, escalados en línea recta.",
        f"- La rejilla tiene {len(report.sizes)} valores. «El menor n que cumple» es el menor de "
        "ellos, no el menor n posible.",
        "- Ninguna cifra de este reporte es una media de retorno.",
        "",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────── Comando ──────────────────────────────────────────────


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada: `python -m crypto_agents.design_effect <directorio> --delta X`.

    Sale con 0 si las series del pool coinciden con la corrida donde se pueden comparar, con 1
    si algo no cuadra o la corrida es inválida, y con 2 si el directorio no se pudo leer.
    """
    parser = argparse.ArgumentParser(
        prog="python -m crypto_agents.design_effect",
        description="Efecto de diseño del pool de activaciones, y el n que pide con él.",
    )
    parser.add_argument("directory", type=Path, help="directorio de la corrida, con meta.json")
    add_delta_arguments(parser)
    parser.add_argument("--plan", type=Path, default=None, help="manifiesto del plan")
    parser.add_argument("--history-dir", type=Path, default=HISTORY_DIR)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="semilla del bootstrap")
    parser.add_argument("--subsamples", type=int, default=DEFAULT_SUBSAMPLES)
    parser.add_argument("--subsample-seed", type=int, default=DEFAULT_SUBSAMPLE_SEED)
    parser.add_argument(
        "--sizes",
        default=",".join(str(n) for n in SUBSAMPLE_SIZES),
        help="la rejilla de n, separada por comas",
    )
    args = parser.parse_args(argv)
    delta = resolve_delta(parser, args)
    if args.resamples <= 0:
        parser.error("--resamples debe ser positivo")
    if args.subsamples <= 0:
        parser.error("--subsamples debe ser positivo")
    try:
        sizes = tuple(int(item) for item in args.sizes.split(",") if item.strip())
    except ValueError:
        parser.error("--sizes son enteros separados por comas")
    if not sizes or min(sizes) < 2:
        parser.error("--sizes necesita al menos un n de dos o más")

    try:
        stage = read_stage(args.directory, args.plan, args.history_dir, args.horizon)
        run = stage.chain[0]
        meta = run.meta
        recalculation = build_recalculation(
            stage.arms,
            stage.histories,
            stage.horizon,
            delta,
            stage.manifest,
            pool_cells(stage.manifest, stage.histories, progress=True),
            stage.calls,
            DEFAULT_PRICING,
            meta.billing,
            args.resamples,
            args.seed,
            resumed=len(stage.chain) > 1,
        )
        report = build_report(
            stage.arms,
            stage.histories,
            stage.horizon,
            delta,
            stage.manifest,
            pool_records(stage.manifest, stage.histories, progress=True),
            recalculation,
            sizes,
            args.subsamples,
            args.subsample_seed,
            progress=True,
        )
    except InvalidRunError as error:
        print(render_invalid(error.validity, str(args.directory)))
        return 1
    except (
        AuditError,
        SelectionError,
        CriteriaError,
        RecalculationError,
        DesignEffectError,
        OSError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    flags = [delta_flag(delta)]
    if args.plan is not None:
        flags.append(f"--plan {args.plan}")
    if args.history_dir != HISTORY_DIR:
        flags.append(f"--history-dir {args.history_dir}")
    if args.horizon is not None:
        flags.append(f"--horizon {args.horizon}")
    if args.resamples != DEFAULT_RESAMPLES:
        flags.append(f"--resamples {args.resamples}")
    if args.seed != DEFAULT_SEED:
        flags.append(f"--seed {args.seed}")
    if args.subsamples != DEFAULT_SUBSAMPLES:
        flags.append(f"--subsamples {args.subsamples}")
    if args.subsample_seed != DEFAULT_SUBSAMPLE_SEED:
        flags.append(f"--subsample-seed {args.subsample_seed}")
    if sizes != SUBSAMPLE_SIZES:
        flags.append(f"--sizes {','.join(str(n) for n in sizes)}")
    sources = Sources(
        directory=str(args.directory),
        command=f"python -m crypto_agents.design_effect {args.directory} {' '.join(flags)}",
        commit=commit_label(meta.git_commit, meta.git_dirty),
        plan=f"`{stage.plan_path}` ({meta.plan_kind.value}), sha-256 `{meta.plan_sha256}`",
        journals=tuple((arm.path.name, arm.sha256) for arm in run.arms if arm.sha256 is not None),
        histories=stage.history_files,
    )
    print(render_report(report, sources))
    return 0 if report.consistent else 1


if __name__ == "__main__":
    raise SystemExit(main())
