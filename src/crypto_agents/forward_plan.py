"""Planificación del estudio hacia adelante de `solo`: lo que `docs/forward.md` prerregistra.

El estudio corre, bloque a bloque, las activaciones nuevas de cada semana por `solo` y las cuatro
líneas base, y compara `solo` con `rule_trend` una sola vez, al cerrar: con 50 bloques no vacíos,
o al terminar el 52 sin ellos. Antes de que exista ningún dato hay que saber qué puede decir:
cuántas evaluaciones da una semana, cuánto valen con la dependencia que mide el efecto de diseño,
con qué probabilidad sale cada veredicto y cuánto cuesta. Este módulo lo calcula sobre el pool
histórico y la etapa 1, y no llama a nadie.

    python -m crypto_agents.forward_plan <directorio de la etapa 1>

Cinco decisiones que lo mantienen honesto:

- **No mide nada por su cuenta.** El pool, sus series y su error agrupado son los de
  `design_effect`; `sigma_D`, el coste y la pared, los de `paired_dispersion`; el bloque es el
  clúster de la enmienda 4 (`metrics.block_of` con `criteria.CLUSTER_DAYS`), y el suelo del efecto
  de diseño, `design_effect.used_deff`. Lo único nuevo es la probabilidad de «empate» con la regla
  de equivalencia del estudio, el recuento por bloque, los cuatro recuentos del pool y la
  comparación de las líneas base con los retornos que publicó la etapa 1.
- **La regla del efecto de diseño se fijó antes de los números**: el máximo, sobre las seis
  diferencias entre líneas base, del deff del pool entero con L = 7, con suelo en 1.
- **δ es una constante.** `DELTA` es el δ registrado del estudio, 0.40 %, y no es el de la
  ablación (`criteria.DELTA`, 0.20 %), que este módulo no nombra. Los otros tres valores van de
  referencia y el reporte lo dice. No hay argumento con el que cambiarlo.
- **Ninguna media de retorno.** Las probabilidades salen de `sigma_D`, del efecto de diseño y de
  δ, no de una diferencia observada.
- **No elige nada.** Da cifras; el diseño lo fijó el operador antes de verlas.

Este módulo no importa el router ni nada que lo traiga. Dos tests de arquitectura lo fijan: uno
mira sus imports y otro lo importa en un proceso nuevo y comprueba que el router no se cargó.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import AwareDatetime, Field

from crypto_agents.activation_sweep import activated_bars
from crypto_agents.audit import AuditError, file_sha256
from crypto_agents.criteria import (
    BASELINE_ARMS,
    CLUSTER_DAYS,
    PERCENT_DIGITS,
    CriteriaError,
    paired_returns,
    render_invalid,
)
from crypto_agents.design_effect import (
    SURROGATE_PAIRS,
    DesignEffectError,
    Planning,
    SeriesError,
    StageTie,
    effective_n,
    planned_power,
    pool_errors,
    pool_records,
    pool_series,
    stage_ties,
    used_deff,
)
from crypto_agents.dispersion import Z_INTERVAL
from crypto_agents.market import timeframe_to_timedelta
from crypto_agents.metrics import EPOCH, block_of, nearest_rank
from crypto_agents.paired_dispersion import (
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    InvalidRunError,
    MissingPair,
    PairDispersion,
    Recalculation,
    RecalculationError,
    Sources,
    commit_label,
    pool_cells,
    read_stage,
    stage_hours,
    stage_usd,
)
from crypto_agents.paired_dispersion import build_report as build_recalculation
from crypto_agents.selection import HISTORY_DIR, SelectionError, confirm_activation
from crypto_agents.settings import DEFAULT_PRICING
from crypto_agents.state import FrozenModel

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from crypto_agents.design_effect import PoolBar, PoolRecords
    from crypto_agents.journal import EvaluationRecord
    from crypto_agents.selection import SelectionManifest

__all__ = [
    "BATCH_DEADLINE",
    "CLUSTER_DAYS",
    "DELTA",
    "EARLIEST_START",
    "FORWARD_ARMS",
    "MAX_BLOCKS",
    "POOL_COUNTS",
    "PRIMARY",
    "PUBLISHED_RETURNS",
    "PUBLISHED_RETURNS_SHA256",
    "REFERENCE_DELTAS",
    "SPEND_CAP_MULTIPLE",
    "TARGET_BLOCKS",
    "USD_PER_EVALUATION",
    "Z_EQUIVALENCE",
    "BlockCounts",
    "Calendar",
    "DeltaProbability",
    "ForwardPlan",
    "ForwardPlanError",
    "OutsideBlock",
    "PoolCounts",
    "ProbabilityRow",
    "PublishedTie",
    "StudyCosts",
    "batch_spend_cap",
    "block_counts",
    "block_of",
    "build_report",
    "calendar",
    "closing_block",
    "first_block",
    "main",
    "planned_power",
    "pool_counts",
    "pool_planning",
    "probabilities",
    "probability_rows",
    "published_returns",
    "published_ties",
    "render_report",
    "spend_cap",
    "study_costs",
    "tie_probability",
    "tie_region",
    "used_deff",
]

DELTA = 0.004
"""δ registrado del estudio: 0.40 % de retorno bruto por evaluación, como fracción, con el stop
común y el horizonte de 6 velas. Lo fijó el operador antes de este cálculo. No es el de la
ablación, que sigue siendo `criteria.DELTA` y no se toca."""

REFERENCE_DELTAS = (0.002, 0.003, 0.005)
"""Otros δ que el reporte imprime al lado, como referencia. Ninguno es el registrado."""

TARGET_BLOCKS = 50
"""Bloques no vacíos con los que cierra el estudio: los que tienen al menos una activación de la
población. El análisis es uno solo, al cerrar."""

MAX_BLOCKS = 52
"""El último bloque que el estudio puede correr. Si al terminar este no lleva `TARGET_BLOCKS` no
vacíos, el estudio queda «no determinado»."""

EARLIEST_START = datetime(2026, 10, 22, tzinfo=UTC)
"""El primer bloque si `docs/forward.md` está en master antes de este instante. Es un jueves a
las 00:00 UTC, el inicio de un bloque de 7 días alineado al epoch."""

Z_EQUIVALENCE = 1.645
"""El cuantil del IC 90 % de la regla de empate, redondeado como el 1.96 del IC 95 %."""

SPEND_CAP_MULTIPLE = 3
"""El tope de gasto de un lote, en veces lo que costarían las evaluaciones de su manifiesto a
`USD_PER_EVALUATION`."""

USD_PER_EVALUATION = 0.01509
"""USD por evaluación con que se calcula el tope: la cota superior de `solo` en la etapa 1
(`paired_dispersion`, sección 6), a cinco decimales, como se publicó. Es una cifra registrada: el
tope no cambia si alguien vuelve a medir."""

PUBLISHED_RETURNS = Path("tests/data/stage1_paired_returns.jsonl")
"""El retorno bruto por evaluación de los seis brazos de la etapa 1, con el stop común, tal como
lo puntuó el código de entonces. Su procedencia está en `tests/data/README.md`."""

PUBLISHED_RETURNS_SHA256 = "b521801e0545d3965d3f96c0fba32debd92b30988af6d31e3cfa1d034e55841f"

BATCH_DEADLINE = timedelta(days=14)
"""Plazo para completar el lote de un bloque, contado desde su cierre."""

PRIMARY = ("solo", "rule_trend")
"""La comparación principal. Las otras tres líneas base son descriptivas."""

FORWARD_ARMS = ("solo", *BASELINE_ARMS)
"""Los brazos de cada lote: `solo` y las cuatro líneas base, que no llaman a ningún modelo."""

TARGET_ARM = PRIMARY[0]

_SIGMA_LABELS = ("extremo bajo", "puntual", "extremo alto")
_COMMAND = "python -m crypto_agents.forward_plan"


class ForwardPlanError(RuntimeError):
    """El cálculo no se puede hacer sin inventar: una serie con huecos, un pool sin bloques
    completos o una serie sustituto que falta."""


# ─────────────────────────────────────────── Modelos ──────────────────────────────────────────────


class OutsideBlock(FrozenModel):
    """Un bloque del pool que no cae entero dentro del tramo evaluable: no entra en el recuento."""

    start: AwareDatetime
    evaluations: int = Field(ge=0)


class BlockCounts(FrozenModel):
    """Activaciones confirmadas del pool por bloque completo de 7 días, vacíos incluidos."""

    days: int = Field(gt=0)
    first: AwareDatetime
    """Inicio del primer bloque completo."""

    last: AwareDatetime
    """Inicio del último bloque completo."""

    counts: tuple[int, ...] = Field(min_length=1)
    """Uno por bloque completo, en orden de tiempo. Un bloque sin activaciones es un 0."""

    outside: tuple[OutsideBlock, ...] = ()
    """Los bloques de los bordes, parciales, con lo que tenían."""

    @property
    def mean_per_block(self) -> float:
        """Media sobre todos los bloques completos, los vacíos como 0: la semana esperada."""
        return statistics.fmean(self.counts)

    @property
    def mean_per_filled_block(self) -> float | None:
        """Media sobre los bloques con alguna activación: los que cuentan para cerrar el
        estudio. `None` si todos están vacíos."""
        filled = [count for count in self.counts if count > 0]
        return statistics.fmean(filled) if filled else None

    @property
    def p10(self) -> float:
        return nearest_rank(self.counts, 10)

    @property
    def p90(self) -> float:
        return nearest_rank(self.counts, 90)

    @property
    def empty(self) -> int:
        """Bloques completos sin ninguna activación confirmada."""
        return sum(1 for count in self.counts if count == 0)


class DeltaProbability(FrozenModel):
    """Las dos probabilidades de un par para un δ."""

    delta: float = Field(gt=0.0)
    registered: bool
    above: float | None = Field(default=None, ge=0.0, le=1.0)
    """P(«supera») si la diferencia real es δ: el IC 95 % entero por encima de 0."""

    tie: float | None = Field(default=None, ge=0.0, le=1.0)
    """P(«empate») si la diferencia real es 0: el IC 90 % dentro de (-δ, +δ) y el IC 95 %
    incluye el 0."""


class ProbabilityRow(FrozenModel):
    """`solo` contra una línea base, con un valor de `sigma_D`."""

    arm: str
    reference: str
    primary: bool
    label: str
    sigma: float = Field(ge=0.0)
    values: tuple[DeltaProbability, ...]


class PoolCounts(FrozenModel):
    """Los cuatro recuentos del pool sobre unos históricos: lo que dan el barrido y el gate."""

    activations: int = Field(ge=0)
    """Activaciones del barrido (`activation_sweep.activated_bars`), tengan o no horizonte."""

    unconfirmed: int = Field(ge=0)
    """De esas, las que el gate no abre con la ventana del replay (`selection.confirm_activation`),
    como las cuenta `dispersion`."""

    candidates: int = Field(ge=0)
    """Las que tienen el horizonte por delante: las que `selection.stratify` reparte en celdas."""

    confirmed: int = Field(ge=0)
    """De esas, las que el gate abre también con la ventana del replay: el pool."""


POOL_COUNTS = PoolCounts(activations=4740, unconfirmed=112, candidates=4737, confirmed=4626)
"""Los recuentos registrados sobre `data/history/`, el histórico de la etapa 1: los de
`docs/ablation.md`, «Recálculo de dispersión» y «Efecto de diseño», sección 1."""


class PublishedTie(FrozenModel):
    """Una línea base sobre el pool contra el retorno que publicó la etapa 1."""

    arm: str
    published: int = Field(ge=0)
    """Evaluaciones de la línea base en el archivo publicado."""

    found: int = Field(ge=0)
    """De esas, las que están en el pool por `run_id`."""

    returns: int = Field(ge=0)
    """Evaluaciones encontradas con otro retorno que el publicado."""

    @property
    def exact(self) -> bool:
        """Si la puntuación de hoy da, evaluación por evaluación, el retorno publicado."""
        return self.published > 0 and self.found == self.published and self.returns == 0


class StudyCosts(FrozenModel):
    """Coste y pared por evaluación medidos en la etapa 1, y la cota del gasto del estudio."""

    per_evaluation_usd: float | None = Field(default=None, ge=0.0)
    """La cota superior de `solo` por evaluación; las líneas base valen 0."""

    per_evaluation_seconds: float | None = Field(default=None, ge=0.0)
    study_bound_usd: float = Field(ge=0.0)
    """La cota del gasto del estudio que dan las reglas: la suma de los topes de sus lotes, que es
    `spend_cap` de las evaluaciones del estudio. Solo los bloques no vacíos llaman a algún modelo y
    una reanudación comparte el tope de su lote."""

    def usd(self, evaluations: float) -> float | None:
        if self.per_evaluation_usd is None:
            return None
        return evaluations * self.per_evaluation_usd

    def hours(self, evaluations: float) -> float | None:
        if self.per_evaluation_seconds is None:
            return None
        return evaluations * self.per_evaluation_seconds / 3600.0


class Calendar(FrozenModel):
    """Las fechas del estudio para un primer bloque: el cierre esperado, si ningún bloque sale
    vacío, y el más tardío posible, en el bloque `MAX_BLOCKS`."""

    first: AwareDatetime
    last: AwareDatetime
    """Inicio del bloque 50."""

    end: AwareDatetime
    """Cierre del bloque 50."""

    first_ready: AwareDatetime
    """Desde cuándo se puede correr el lote del primer bloque: su cierre más el horizonte."""

    first_deadline: AwareDatetime
    last_ready: AwareDatetime
    last_deadline: AwareDatetime
    """Plazo del lote del bloque 50: el análisis, si ningún bloque sale vacío."""

    latest_last: AwareDatetime
    """Inicio del bloque 52."""

    latest_end: AwareDatetime
    """Cierre del bloque 52."""

    latest_ready: AwareDatetime
    latest_deadline: AwareDatetime
    """Plazo del lote del bloque 52. El análisis no puede esperar más allá."""


class ForwardPlan(FrozenModel):
    """Todo lo que el reporte sabe. Ninguna media de retorno entre estos campos."""

    delta: float = Field(gt=0.0)
    reference_deltas: tuple[float, ...]
    target_blocks: int = Field(gt=0)
    max_blocks: int = Field(gt=0)
    horizon: int = Field(gt=0)
    timeframe: str
    baseline_seed: int
    resamples: int = Field(gt=0)
    bootstrap_seed: int
    stage_n: int = Field(gt=0)
    """Evaluaciones de la corrida de la que sale `sigma_D`."""

    confirmed: int = Field(ge=0)
    """Activaciones confirmadas del pool."""

    counts: PoolCounts
    registered_counts: PoolCounts
    """Los recuentos con que se comparan `counts`: en el comando, `POOL_COUNTS`."""

    ties: tuple[StageTie, ...]
    published: tuple[PublishedTie, ...]
    pool: tuple[SeriesError, ...]
    """Las seis series sustituto sobre el pool entero, con L = 7."""

    planning: Planning
    """El deff de planificación y el n esperado a los 50 bloques no vacíos."""

    blocks: BlockCounts
    rows: tuple[ProbabilityRow, ...]
    missing: tuple[MissingPair, ...] = ()
    costs: StudyCosts
    calendar: Calendar
    recalculation_consistent: bool

    @property
    def consistent(self) -> bool:
        """Si el pool tiene los recuentos registrados, sus series son las de la ablación y su
        puntuación la publicada, y lo heredado de la corrida cuadra."""
        return (
            self.counts == self.registered_counts
            and all(tie.exact for tie in self.ties)
            and all(tie.exact for tie in self.published)
            and self.recalculation_consistent
        )


# ─────────────────────────────────────────── Cálculo ──────────────────────────────────────────────


def first_block(merged_at: datetime) -> datetime:
    """Inicio del primer bloque del estudio si `docs/forward.md` entró en master en `merged_at`.

    El primer bloque de 7 días alineado al epoch —un jueves a las 00:00 UTC— que empieza
    estrictamente después del merge, y nunca antes de `EARLIEST_START`. Un merge justo en el
    borde de un bloque no lo abre: el estudio empieza en el siguiente.
    """
    step = timedelta(days=CLUSTER_DAYS)
    following = EPOCH + (block_of(merged_at, CLUSTER_DAYS) + 1) * step
    return max(EARLIEST_START, following)


def calendar(first: datetime, timeframe: str, horizon: int) -> Calendar:
    """Las fechas del estudio que empieza en el bloque `first`.

    El lote de un bloque se puede correr cuando todas sus activaciones tienen el horizonte por
    delante: al cierre del bloque más `horizon` velas.
    """
    step = timedelta(days=CLUSTER_DAYS)
    if (first - EPOCH) % step:
        raise ValueError(f"{first.isoformat()} no es el inicio de un bloque de {CLUSTER_DAYS} días")
    wait = horizon * timeframe_to_timedelta(timeframe)
    end = first + TARGET_BLOCKS * step
    latest = first + MAX_BLOCKS * step
    return Calendar(
        first=first,
        last=end - step,
        end=end,
        first_ready=first + step + wait,
        first_deadline=first + step + BATCH_DEADLINE,
        last_ready=end + wait,
        last_deadline=end + BATCH_DEADLINE,
        latest_last=latest - step,
        latest_end=latest,
        latest_ready=latest + wait,
        latest_deadline=latest + BATCH_DEADLINE,
    )


def closing_block(counts: Sequence[int]) -> int | None:
    """El bloque, contado desde 1, en que cierra un estudio con estas activaciones por bloque.

    El estudio cierra en el bloque en que acumula `TARGET_BLOCKS` bloques con al menos una
    activación de la población. Un bloque vacío se cuenta y se reporta, pero no suma. Si al
    terminar el bloque `MAX_BLOCKS` no los acumula, devuelve `None`: el estudio es «no
    determinado». La regla solo lee cuántas activaciones tuvo cada bloque, que no depende de nada
    que haga `solo`. Una secuencia más corta que `MAX_BLOCKS` que todavía no cerró es un estudio
    abierto, y se rechaza.
    """
    if any(count < 0 for count in counts):
        raise ValueError("un bloque no puede tener activaciones negativas")
    filled = 0
    for position, count in enumerate(counts[:MAX_BLOCKS], start=1):
        if count > 0:
            filled += 1
        if filled == TARGET_BLOCKS:
            return position
    if len(counts) < MAX_BLOCKS:
        raise ValueError(f"{len(counts)} bloques y {filled} no vacíos: el estudio sigue abierto")
    return None


def block_counts(bars: Sequence[PoolBar], manifest: SelectionManifest) -> BlockCounts:
    """Activaciones confirmadas por bloque completo de 7 días.

    Cada evaluación va al bloque de su instante, el cierre de la vela evaluada, como en la
    enmienda 4. Un bloque es **completo** si cae entero dentro del tramo en que las siete series
    pueden dar una evaluación: desde el cierre de la primera vela tras el warm-up hasta el de la
    última con el horizonte por delante. Los bloques completos sin ninguna activación cuentan 0,
    porque en el estudio también se cuentan y se reportan, aunque no sumen para cerrar; los de los
    bordes, parciales, se dejan fuera y se reportan.
    """
    period = timeframe_to_timedelta(manifest.timeframe)
    step = timedelta(days=CLUSTER_DAYS)
    opens: list[datetime] = []
    closes: list[datetime] = []
    for ref in manifest.series:
        if ref.last - ref.first != (ref.bars - 1) * period:
            raise ForwardPlanError(
                f"{ref.symbol}: {ref.bars} velas de {manifest.timeframe} no van de "
                f"{ref.first.isoformat()} a {ref.last.isoformat()}; la serie tiene huecos"
            )
        opens.append(ref.first + (manifest.warmup_bars + 1) * period)
        closes.append(ref.last - (manifest.horizon - 1) * period)
    reach_from, reach_to = max(opens), min(closes)

    first = block_of(reach_from, CLUSTER_DAYS)
    if EPOCH + first * step < reach_from:
        first += 1
    last = block_of(reach_to + period - step, CLUSTER_DAYS)
    if last < first:
        raise ForwardPlanError("el pool no cubre ningún bloque completo")

    tally: dict[int, int] = {}
    for bar in bars:
        block = block_of(bar.at, CLUSTER_DAYS)
        tally[block] = tally.get(block, 0) + 1
    blocks = range(first, last + 1)
    counts = tuple(tally.get(block, 0) for block in blocks)
    return BlockCounts(
        days=CLUSTER_DAYS,
        first=EPOCH + first * step,
        last=EPOCH + last * step,
        counts=counts,
        outside=tuple(
            OutsideBlock(start=EPOCH + block * step, evaluations=tally[block])
            for block in sorted(tally)
            if block not in blocks
        ),
    )


def pool_counts(
    pool: PoolRecords,
    manifest: SelectionManifest,
    histories: Mapping[str, Sequence[Sequence[float]]],
) -> PoolCounts:
    """Los cuatro recuentos del pool, con el barrido y la confirmación que usa `pool_records`.

    `pool_records` confirma solo las activaciones con el horizonte por delante. Las pocas que no
    lo tienen se confirman aquí, para contar las que el gate no abre sobre todas las del barrido,
    como las cuenta `dispersion`.
    """
    candidates = sum(len(cell) for cells in pool.cells.values() for cell in cells)
    swept = outside = 0
    for ref in manifest.series:
        rows = histories[ref.symbol]
        stratified = {bar.index for cell in pool.cells[ref.symbol] for bar in cell}
        found = activated_bars(rows, ref.symbol, manifest.timeframe)
        swept += len(found)
        outside += sum(
            1
            for bar in found
            if bar.index not in stratified
            and confirm_activation(rows, bar.index, manifest.candle_limit) is None
        )
    if swept != pool.activations:
        raise ForwardPlanError(
            f"el barrido da {swept} activaciones y el pool contó {pool.activations}: no son los "
            "mismos históricos"
        )
    return PoolCounts(
        activations=pool.activations,
        unconfirmed=candidates - len(pool.bars) + outside,
        candidates=candidates,
        confirmed=len(pool.bars),
    )


def published_returns(path: Path, sha256: str) -> dict[UUID, dict[str, float]]:
    """El retorno publicado de cada evaluación de la etapa 1, por `run_id` y brazo.

    Comprueba el sha-256 del archivo antes de leerlo: uno editado no es el publicado.
    """
    digest = file_sha256(path)
    if digest != sha256:
        raise ForwardPlanError(
            f"{path}: sha-256 {digest[:12]}…, y el publicado es {sha256[:12]}…; no es el archivo "
            "publicado"
        )
    published: dict[UUID, dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        published[UUID(row["run_id"])] = {
            arm: float(value) for arm, value in row["returns"].items()
        }
    return published


def published_ties(
    pool: PoolRecords,
    published: Mapping[UUID, Mapping[str, float]],
    histories: Mapping[str, Sequence[Sequence[float]]],
    horizon: int,
) -> tuple[PublishedTie, ...]:
    """Cada línea base del pool contra el retorno que publicó la etapa 1, evaluación por evaluación.

    `design_effect.stage_ties` puntúa el pool y el journal con la misma función, así que un cambio
    en la puntuación le pasa inadvertido: cambia los dos lados a la vez. Aquí el otro lado es lo
    que la puntuación de la etapa 1 dejó escrito, así que la puntuación queda fijada por su
    salida. Puntúa con `criteria.paired_returns`, la puerta de los criterios: empareja dos
    brazos, y aquí los dos son el mismo registro del pool, de modo que solo se lee uno.
    """
    ties: list[PublishedTie] = []
    for name in BASELINE_ARMS:
        expected = {
            run_id: returns[name] for run_id, returns in published.items() if name in returns
        }
        ours = {record.run_id: record for record in pool.arms[name]}
        shared = [ours[run_id] for run_id in expected if run_id in ours]
        differing = 0
        if shared:
            aligned = paired_returns(
                name, "publicado", {name: shared, "publicado": shared}, histories, horizon
            )
            if isinstance(aligned, str):
                raise ForwardPlanError(f"{name} del pool contra lo publicado: {aligned}")
            differing = sum(
                1
                for record, value in zip(aligned.arm_records, aligned.arm_returns, strict=True)
                if value != expected[record.run_id]
            )
        ties.append(
            PublishedTie(arm=name, published=len(expected), found=len(shared), returns=differing)
        )
    return tuple(ties)


def _deff_of(row: SeriesError) -> float:
    deff = row.error.deff
    if deff is None:
        raise ForwardPlanError(f"{row.series}: sin efecto de diseño sobre el pool")
    return deff


def pool_planning(errors: Sequence[SeriesError], n: int) -> Planning:
    """El efecto de diseño de planificación del estudio, con `n` evaluaciones esperadas.

    La regla, fijada antes de los números: el máximo, sobre las seis diferencias entre líneas
    base, del deff del pool entero con bloques de `CLUSTER_DAYS` días. Las líneas base solas no
    entran: lo que se dimensiona es una diferencia pareada. El suelo `max(1, deff)` no se aplica
    aquí sino donde se usa, en `design_effect.used_deff`, el mismo que lleva la potencia de la
    enmienda 4.
    """
    surrogates = [row for row in errors if row.surrogate and row.days == CLUSTER_DAYS]
    names = {row.series for row in surrogates}
    if len(surrogates) != len(SURROGATE_PAIRS) or len(names) != len(surrogates):
        raise ForwardPlanError(
            f"hacen falta las {len(SURROGATE_PAIRS)} series sustituto con L = {CLUSTER_DAYS}, "
            f"hay {len(surrogates)}"
        )
    worst = max(surrogates, key=_deff_of)
    return Planning(n=n, deff=_deff_of(worst), series=worst.series, days=worst.days)


def tie_region(error: float, delta: float) -> tuple[float, float] | None:
    """Las medias observadas con las que sale «empate» con ese error estándar: `(a, b)`, abierto.

    El IC 90 % entero dentro de (-δ, +δ) y el IC 95 % incluyendo el 0 es `a < m < b` con
    `a = max(-δ + 1.645·EE, -1.96·EE)` y `b = min(δ - 1.645·EE, 1.96·EE)`. `None` si `a ≥ b`: con
    ese EE no hay empate posible. Es la región de `tie_probability`, y la regla de veredicto del
    análisis (`forward_analysis.forward_verdict`) tiene que dar empate exactamente en ella.
    """
    margin = Z_EQUIVALENCE * error
    low = max(-delta + margin, -Z_INTERVAL * error)
    high = min(delta - margin, Z_INTERVAL * error)
    if low >= high:
        return None
    return low, high


def tie_probability(
    sigma: float, plan: Planning, delta: float, true_difference: float = 0.0
) -> float | None:
    """Probabilidad de «empate» si la diferencia real es `true_difference`.

    «Empate» es la prueba de equivalencia del estudio: el IC 90 % entero dentro de (-δ, +δ) y el
    IC 95 % incluyendo el 0. Con la media observada `m ~ N(μ, EE²)` y
    `EE = sigma·√(deff usado / n)`, eso es `a < m < b`, la región de `tie_region`; si no la hay,
    no hay empate posible. El deff usado lleva el suelo de la enmienda 4. `None` si el n no tiene
    efecto de diseño de planificación.
    """
    if sigma < 0.0 or not math.isfinite(sigma):
        raise ValueError(f"la desviación tiene que ser finita y no negativa: {sigma}")
    if delta <= 0.0 or not math.isfinite(delta):
        raise ValueError(f"delta tiene que ser positivo: {delta}")
    deff = used_deff(plan)
    if deff is None:
        return None
    error = sigma * math.sqrt(deff / plan.n)
    if error == 0.0:
        return 1.0 if true_difference == 0.0 else 0.0
    region = tie_region(error, delta)
    if region is None:
        return 0.0
    low, high = region
    observed = statistics.NormalDist(true_difference, error)
    return observed.cdf(high) - observed.cdf(low)


def probabilities(sigma: float, plan: Planning) -> tuple[DeltaProbability, ...]:
    """P(«supera» | Δ = δ) y P(«empate» | Δ = 0) para el δ registrado y los de referencia.

    «Supera» es `design_effect.planned_power`: la potencia del IC 95 % con el deff usado, la
    misma que publicó la enmienda 4. Lee `DELTA` al llamarse, no al definirse.
    """
    registered = DELTA
    return tuple(
        DeltaProbability(
            delta=delta,
            registered=delta == registered,
            above=planned_power(sigma, plan, delta),
            tie=tie_probability(sigma, plan, delta),
        )
        for delta in sorted({registered, *REFERENCE_DELTAS})
    )


def probability_rows(pairs: Sequence[PairDispersion], plan: Planning) -> tuple[ProbabilityRow, ...]:
    """`solo` contra cada línea base, con `sigma_D` en sus dos extremos y su valor puntual."""
    rows: list[ProbabilityRow] = []
    for pair in pairs:
        if pair.arm != TARGET_ARM or pair.reference not in BASELINE_ARMS:
            continue
        spread = pair.dispersion
        sigmas = (spread.bootstrap.low, spread.sigma, spread.bootstrap.high)
        rows.extend(
            ProbabilityRow(
                arm=pair.arm,
                reference=pair.reference,
                primary=(pair.arm, pair.reference) == PRIMARY,
                label=label,
                sigma=sigma,
                values=probabilities(sigma, plan),
            )
            for label, sigma in zip(_SIGMA_LABELS, sigmas, strict=True)
        )
    return tuple(sorted(rows, key=lambda row: not row.primary))


def spend_cap(evaluations: float) -> float:
    """El tope de gasto, en USD, de un lote de `evaluations` evaluaciones de `solo`.

    `SPEND_CAP_MULTIPLE` veces lo que esas evaluaciones costarían a `USD_PER_EVALUATION`. Es
    proporcional al lote y no fijo: con un tope fijo, un lote grande se quedaría sin gasto antes
    de terminar, y las semanas con más activaciones son las más volátiles, de modo que sus
    faltantes no serían al azar. Lee las dos constantes al llamarse.
    """
    if evaluations < 0:
        raise ValueError(f"un lote no puede tener evaluaciones negativas: {evaluations}")
    return SPEND_CAP_MULTIPLE * evaluations * USD_PER_EVALUATION


def batch_spend_cap(manifest: SelectionManifest) -> float:
    """El tope del lote de un bloque: una evaluación de `solo` por entrada de su manifiesto.

    Se calcula antes de la primera llamada del lote y se acumula entre sus reanudaciones. Un
    bloque vacío tiene tope 0 y no llama a nadie.
    """
    return spend_cap(len(manifest.entries))


def study_costs(
    per_evaluation_usd: float | None, per_evaluation_seconds: float | None, evaluations: int
) -> StudyCosts:
    """El coste y la pared medidos por evaluación, y la cota del gasto de un estudio de
    `evaluations` evaluaciones que dan las reglas: la suma de los topes de sus lotes."""
    return StudyCosts(
        per_evaluation_usd=per_evaluation_usd,
        per_evaluation_seconds=per_evaluation_seconds,
        study_bound_usd=spend_cap(evaluations),
    )


def build_report(
    stage: Mapping[str, Sequence[EvaluationRecord]],
    histories: Mapping[str, Sequence[Sequence[float]]],
    horizon: int,
    manifest: SelectionManifest,
    pool: PoolRecords,
    recalculation: Recalculation,
    published: Mapping[UUID, Mapping[str, float]],
    registered: PoolCounts,
) -> ForwardPlan:
    """La planificación de un pool ya evaluado y una corrida ya recalculada.

    `pool` y `recalculation` llegan calculados: son lo lento, y no dependen de nada de aquí.
    `published` son los retornos publicados de la etapa 1 y `registered`, los recuentos con que
    se comparan los del pool. Lo único que lee aparte es el barrido de `pool_counts`.
    """
    series = pool_series(pool, histories, horizon)
    errors = pool_errors(series, pool.bars, (CLUSTER_DAYS,))
    blocks = block_counts(pool.bars, manifest)
    filled = blocks.mean_per_filled_block
    if filled is None:
        raise ForwardPlanError("ningún bloque completo del pool tiene activaciones")
    plan = pool_planning(errors, round(TARGET_BLOCKS * filled))
    hours = stage_hours(recalculation.clocks, FORWARD_ARMS, 1)
    return ForwardPlan(
        delta=DELTA,
        reference_deltas=REFERENCE_DELTAS,
        target_blocks=TARGET_BLOCKS,
        max_blocks=MAX_BLOCKS,
        horizon=horizon,
        timeframe=manifest.timeframe,
        baseline_seed=manifest.seed,
        resamples=recalculation.resamples,
        bootstrap_seed=recalculation.seed,
        stage_n=recalculation.stage_n,
        confirmed=len(pool.bars),
        counts=pool_counts(pool, manifest, histories),
        registered_counts=registered,
        ties=stage_ties(pool, stage, histories, horizon),
        published=published_ties(pool, published, histories, horizon),
        pool=tuple(row for row in errors if row.surrogate),
        planning=plan,
        blocks=blocks,
        rows=probability_rows(recalculation.pairs, plan),
        missing=tuple(
            item
            for item in recalculation.missing
            if item.arm == TARGET_ARM and item.reference in BASELINE_ARMS
        ),
        costs=study_costs(
            stage_usd(recalculation.costs, FORWARD_ARMS, 1),
            None if hours is None else hours * 3600.0,
            plan.n,
        ),
        calendar=calendar(EARLIEST_START, manifest.timeframe, horizon),
        recalculation_consistent=recalculation.consistent,
    )


# ─────────────────────────────────────────── Reporte ──────────────────────────────────────────────


def _pct(value: float, digits: int = PERCENT_DIGITS) -> str:
    return f"{value:.{digits}%}"


def _ratio(value: float | None) -> str:
    return "no determinado" if value is None else f"{value:.3f}"


def _chance(value: float | None) -> str:
    return "no determinado" if value is None else f"{value:.1%}"


def _usd(value: float | None) -> str:
    return "no determinado" if value is None else f"{value:.2f}"


def _hundredths(value: float | None) -> str:
    return "no determinado" if value is None else f"{value:.2f}"


def _tenths(value: float | None) -> str:
    return "no determinado" if value is None else f"{value:.1f}"


def _instant(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%MZ")


def _reproduce(command: str, also: str = "") -> list[str]:
    return ["", f"Reproduce: `{command}`{also}", ""]


def _yes(value: bool) -> str:
    return "sí" if value else "**NO**"


def _counts_table(report: ForwardPlan) -> list[str]:
    lines = [
        "| | activaciones del barrido | sin confirmar | con el horizonte por delante "
        "| confirmadas |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    lines += [
        f"| {label} | {counts.activations} | {counts.unconfirmed} | {counts.candidates} "
        f"| {counts.confirmed} |"
        for label, counts in (
            ("estos históricos", report.counts),
            ("registrados", report.registered_counts),
        )
    ]
    return [*lines, "", f"- coinciden: {_yes(report.counts == report.registered_counts)}"]


def _published_table(report: ForwardPlan) -> list[str]:
    lines = [
        "| línea base | evaluaciones publicadas | de ellas, en el pool | retorno distinto "
        "| coincide |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {tie.arm} | {tie.published} | {tie.found} | {tie.returns} | {_yes(tie.exact)} |"
        for tie in report.published
    ]
    return lines


def _tie_table(report: ForwardPlan) -> list[str]:
    lines = [
        "| línea base | evaluaciones en la corrida | de ellas, en el pool | acción distinta "
        "| stop distinto | retorno distinto | coincide |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {tie.arm} | {tie.evaluations} | {tie.found} | {tie.actions} | {tie.stops} "
        f"| {tie.returns} | {_yes(tie.exact)} |"
        for tie in report.ties
    ]
    return lines


def _pool_table(report: ForwardPlan) -> list[str]:
    lines = [
        "| serie | L (días) | n | G | EE ingenuo | EE agrupado | deff |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report.pool:
        error = row.error
        clustered = "no determinado" if error.clustered is None else _pct(error.clustered)
        lines.append(
            f"| {row.series} | {row.days} | {error.n} | {error.clusters} | {_pct(error.naive)} "
            f"| {clustered} | {_ratio(error.deff)} |"
        )
    return lines


def _block_table(report: ForwardPlan) -> list[str]:
    blocks = report.blocks
    filled = blocks.mean_per_filled_block
    return [
        "| bloques completos | vacíos | activaciones | media por bloque "
        "| media por bloque no vacío | p10 | p90 | mínimo | máximo |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {len(blocks.counts)} | {blocks.empty} | {sum(blocks.counts)} "
        f"| {blocks.mean_per_block:.2f} | {_hundredths(filled)} | {blocks.p10:g} "
        f"| {blocks.p90:g} | {min(blocks.counts)} | {max(blocks.counts)} |",
    ]


def _effective_table(report: ForwardPlan) -> list[str]:
    plan = report.planning
    used = used_deff(plan)
    whole = effective_n(plan)
    filled = report.blocks.mean_per_filled_block
    each = None if used is None or filled is None else filled / used
    return [
        "| | evaluaciones | deff usado | n efectivo |",
        "| --- | ---: | ---: | ---: |",
        f"| por bloque no vacío | {_hundredths(filled)} | {_ratio(used)} | {_tenths(each)} |",
        f"| a los {report.target_blocks} bloques no vacíos | {plan.n} | {_ratio(used)} "
        f"| {_tenths(whole)} |",
    ]


def _delta_header(item: DeltaProbability) -> str:
    mark = " (registrado)" if item.registered else ""
    return f"δ = {_pct(item.delta)}{mark}"


def _probability_table(report: ForwardPlan) -> list[str]:
    if not report.rows:
        return ["Ningún par de `solo` contra una línea base tiene `sigma_D`."]
    deltas = report.rows[0].values
    lines = [
        "| comparación | brazo | referencia | sigma_D | valor | "
        + " | ".join(
            f"supera si Δ = δ, {_delta_header(item)} | empate si Δ = 0, {_delta_header(item)}"
            for item in deltas
        )
        + " |",
        "| --- | --- | --- | --- | ---: |" + " ---: | ---: |" * len(deltas),
    ]
    for row in report.rows:
        kind = "principal" if row.primary else "descriptiva"
        cells = " | ".join(f"{_chance(item.above)} | {_chance(item.tie)}" for item in row.values)
        lines.append(
            f"| {kind} | {row.arm} | {row.reference} | {row.label} | {_pct(row.sigma)} | {cells} |"
        )
    lines += [
        f"| — | {item.arm} | {item.reference} | — | no determinado: {item.why} |"
        + " — | — |" * len(deltas)
        for item in report.missing
    ]
    return lines


def _cost_table(report: ForwardPlan) -> list[str]:
    costs = report.costs
    blocks = report.blocks
    rows = (
        ("lote medio", blocks.mean_per_block, f"{blocks.mean_per_block:.2f}"),
        ("lote p90", blocks.p90, f"{blocks.p90:g}"),
        ("lote máximo del pool", float(max(blocks.counts)), f"{max(blocks.counts)}"),
        (
            f"el estudio esperado, {report.target_blocks} bloques no vacíos",
            float(report.planning.n),
            f"{report.planning.n}",
        ),
    )
    lines = [
        "| | evaluaciones | USD | tope | horas de pared |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    lines += [
        f"| {label} | {shown} | {_usd(costs.usd(count))} | {_usd(spend_cap(count))} "
        f"| {_tenths(costs.hours(count))} |"
        for label, count, shown in rows
    ]
    return lines


def _cost_lines(report: ForwardPlan) -> list[str]:
    costs = report.costs
    seconds = costs.per_evaluation_seconds
    per_second = "no determinado" if seconds is None else f"{seconds:.1f}"
    per_usd = (
        "no determinado" if costs.per_evaluation_usd is None else f"{costs.per_evaluation_usd:.5f}"
    )
    formula = f"{SPEND_CAP_MULTIPLE} x las evaluaciones de su manifiesto x {USD_PER_EVALUATION} USD"
    return [
        f"- USD por evaluación medido: {per_usd}, la cota superior de `{TARGET_ARM}` en la "
        "corrida (`paired_dispersion`, sección 6); las cuatro líneas base valen 0.",
        f"- USD por evaluación registrado para el tope: {USD_PER_EVALUATION} "
        "(`forward_plan.USD_PER_EVALUATION`).",
        f"- segundos de pared por evaluación: {per_second}, los de `{TARGET_ARM}` en la corrida "
        "(`paired_dispersion`, sección 7), en la máquina que la corrió.",
        f"- **tope de gasto de un lote**: {formula}, calculado antes de su primera llamada y "
        "acumulado entre sus reanudaciones (`forward_plan.batch_spend_cap`). Un bloque vacío no "
        "llama a nadie.",
        f"- **cota del gasto del estudio que dan las reglas**: la suma de los topes de sus lotes, "
        f"{SPEND_CAP_MULTIPLE} x las evaluaciones del estudio x {USD_PER_EVALUATION} USD; con "
        f"las {report.planning.n} esperadas, {_usd(costs.study_bound_usd)} USD. Lo único que el "
        "tope no corta es la invocación que lo cruza, que termina.",
        "- la tabla da, para el lote medio, el p90 y el máximo de los bloques completos del pool, "
        f"y para el estudio esperado ({report.target_blocks} bloques no vacíos a su media), el "
        "coste medido y el tope. El tope de la última fila es la cota del estudio.",
    ]


def _calendar_lines(report: ForwardPlan) -> list[str]:
    dates = report.calendar
    target, most = report.target_blocks, report.max_blocks
    return [
        f"- regla: el primer bloque es el que empieza estrictamente después del merge de "
        f"`docs/forward.md` en master, y nunca antes de {_instant(EARLIEST_START)}. Un merge "
        "justo en el borde de un bloque abre el siguiente.",
        f"- cierre: el estudio cierra en el bloque en que acumula {target} bloques no vacíos; si "
        f"al terminar el bloque {most} no los acumula, queda «no determinado».",
        f"- con el merge antes de {_instant(EARLIEST_START)}: bloque 1 = "
        f"[{_instant(dates.first)}, {_instant(dates.first + timedelta(days=CLUSTER_DAYS))}). "
        f"Fin esperado, si ningún bloque sale vacío: bloque {target} = "
        f"[{_instant(dates.last)}, {_instant(dates.end)}). Fin más tardío posible: bloque {most} "
        f"= [{_instant(dates.latest_last)}, {_instant(dates.latest_end)}).",
        f"- el lote de un bloque se corre desde su cierre más {report.horizon} velas de "
        f"{report.timeframe}: el del bloque 1, desde {_instant(dates.first_ready)}; el del bloque "
        f"{target}, desde {_instant(dates.last_ready)}; el del bloque {most}, desde "
        f"{_instant(dates.latest_ready)}.",
        f"- plazo de un lote: {BATCH_DEADLINE.days} días tras el cierre de su bloque. El del "
        f"bloque 1 vence el {_instant(dates.first_deadline)}; el del bloque {target}, el "
        f"{_instant(dates.last_deadline)}; el del bloque {most}, el "
        f"{_instant(dates.latest_deadline)}.",
    ]


def render_report(report: ForwardPlan, sources: Sources) -> str:
    """La planificación en Markdown. Sin medias de retorno, sin veredicto y sin recomendar."""
    command = sources.command
    plan = report.planning
    blocks = report.blocks
    references = ", ".join(_pct(value) for value in report.reference_deltas)
    recalculation = f"python -m crypto_agents.paired_dispersion {sources.directory}"
    outside = ", ".join(f"{_instant(item.start)} ({item.evaluations})" for item in blocks.outside)

    lines = [
        f"# Planificación del estudio hacia adelante, con la corrida `{sources.directory}`",
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
        f"- δ registrado del estudio = {report.delta:g} ({_pct(report.delta)} de retorno bruto "
        f"por evaluación); de referencia, {references}. Horizonte {report.horizon} velas de "
        f"{report.timeframe}, stop común, por `criteria.paired_returns`.",
        f"- bloques: {CLUSTER_DAYS} días UTC alineados al epoch, que empiezan los jueves a las "
        "00:00 UTC; son los clústeres de la enmienda 4 (`criteria.CLUSTER_DAYS`). A cada "
        "evaluación la asigna el cierre de su vela; un instante en el borde abre el bloque nuevo.",
        f"- `random_uniform` con la semilla del manifiesto de la corrida, {report.baseline_seed}",
        f"- `sigma_D` de `paired_dispersion`: bootstrap de {report.resamples} remuestreos, "
        f"semilla {report.bootstrap_seed}",
        "",
        "## 1. Atadura con la corrida y con lo publicado",
        "",
        "Tres comprobaciones del mercado y de la puntuación, sin modelos. Si una no cuadra, el "
        "comando sale con 1.",
        "",
        "**Los recuentos del pool** sobre los históricos de arriba: las activaciones del barrido "
        "(`activation_sweep.activated_bars`), las que el gate no abre con la ventana del replay "
        "(`selection.confirm_activation`), las que tienen el horizonte por delante "
        "(`selection.stratify`) y, de estas, las confirmadas. Tienen que ser las registradas "
        "(`forward_plan.POOL_COUNTS`).",
        "",
        *_counts_table(report),
        "",
        "**Las líneas base contra su journal.** Las cuatro deciden sobre cada activación "
        "confirmada del pool con las funciones de `baselines.py` (`design_effect.pool_records`). "
        "En las evaluaciones que la corrida ya hizo tienen que dar la acción, el stop y el "
        "retorno de su journal (`design_effect.stage_ties`). El retorno de los dos lados lo "
        "calcula la puntuación de hoy.",
        "",
        *_tie_table(report),
        "",
        "**Las líneas base contra los retornos publicados.** La tabla anterior no ve un cambio en "
        "la puntuación, que mueve los dos lados a la vez. Esta puntúa el pool con "
        "`criteria.paired_returns` y lo compara, evaluación por evaluación, con el retorno bruto "
        f"que publicó la etapa 1: `{PUBLISHED_RETURNS}`, sha-256 `{PUBLISHED_RETURNS_SHA256}`.",
        "",
        *_published_table(report),
        *_reproduce(command),
        f"## 2. Efecto de diseño del pool entero con L = {CLUSTER_DAYS}",
        "",
        "La regla, fijada antes de los números: el deff de planificación es el **máximo** del "
        f"deff de las seis diferencias entre líneas base sobre las {report.confirmed} "
        "activaciones confirmadas del pool, con CR1 y un clúster por bloque de "
        f"{CLUSTER_DAYS} días; el **deff usado** es `max(1, deff)`, el suelo de la enmienda 4 "
        "(`design_effect.used_deff`). Las diferencias entre líneas base son un sustituto de "
        f"`{TARGET_ARM}` menos una línea base, que no se puede calcular sobre el pool sin pagar.",
        "",
        *_pool_table(report),
        "",
        f"- deff de planificación: {_ratio(plan.deff)}, de `{plan.series}`",
        f"- deff usado: {_ratio(used_deff(plan))}",
        *_reproduce(command),
        "## 3. Activaciones confirmadas por bloque",
        "",
        f"Bloques completos: los de {CLUSTER_DAYS} días que caen enteros en el tramo en que las "
        "siete series pueden dar una evaluación —desde el cierre de la primera vela tras el "
        "warm-up hasta el de la última con el horizonte por delante—, del "
        f"{_instant(blocks.first)} al que empieza el {_instant(blocks.last)}. Un bloque completo "
        "sin activaciones cuenta 0 en la media por bloque y queda fuera de la media por bloque "
        "no vacío. Fuera quedan los bloques parciales de los bordes, con sus activaciones: "
        f"{outside or 'ninguno'}. p10 y p90 por rango más cercano.",
        "",
        *_block_table(report),
        *_reproduce(command),
        "## 4. n efectivo",
        "",
        f"n efectivo = evaluaciones / deff usado. El estudio cierra con {report.target_blocks} "
        "bloques no vacíos, así que las evaluaciones esperadas son "
        f"{report.target_blocks} x la media por bloque no vacío, redondeado al entero: es el n de "
        "la sección 5.",
        "",
        *_effective_table(report),
        *_reproduce(command),
        f"## 5. Probabilidad de cada veredicto para `{TARGET_ARM}` contra cada línea base",
        "",
        "Con `EE = sigma_D·√(deff usado / n)` y la media observada normal alrededor de la "
        "diferencia real Δ:",
        "",
        "- **supera si Δ = δ**: P(IC 95 % entero por encima de 0) = `Φ(δ/EE - 1.96)`, "
        "`design_effect.planned_power`.",
        "- **empate si Δ = 0**: P(IC 90 % dentro de (-δ, +δ) y IC 95 % con el 0) = "
        f"P(a < media < b), con `a = max(-δ + {Z_EQUIVALENCE}·EE, -1.96·EE)` y "
        f"`b = min(δ - {Z_EQUIVALENCE}·EE, 1.96·EE)`; 0 si `a ≥ b`.",
        f"- `sigma_D` es la de `paired_dispersion`: puntual y los dos extremos de su rango "
        f"bootstrap. n = {plan.n} y el deff usado de la sección 2.",
        f"- La comparación principal es `{PRIMARY[0]}` - `{PRIMARY[1]}`; las demás son "
        "descriptivas.",
        "",
        *_probability_table(report),
        *_reproduce(command, f" (sigma_D: `{recalculation}`)"),
        "## 6. Evaluaciones, coste y pared",
        "",
        *_cost_lines(report),
        "",
        *_cost_table(report),
        *_reproduce(command, f" (coste y pared: `{recalculation}`)"),
        "## 7. Calendario",
        "",
        *_calendar_lines(report),
        *_reproduce(command),
        "## Lo que estas cifras suponen",
        "",
        "- El deff es el de un sustituto: diferencias entre dos políticas sin modelo. Que "
        f"`{TARGET_ARM}` menos una línea base dependa en el tiempo igual es un supuesto.",
        f"- `sigma_D` sale de las {report.stage_n} evaluaciones de la etapa 1 y su rango es el "
        "de un bootstrap, no un intervalo de confianza.",
        "- Las activaciones por bloque, el deff y `sigma_D` son del periodo del histórico; el "
        "régimen de mercado del estudio puede ser otro.",
        "- Las probabilidades tratan el error estándar como conocido y usan la aproximación "
        "normal; en el estudio el EE agrupado se estima con sus 50 clústeres.",
        "- El coste y la pared son los de la etapa 1 por evaluación, escalados en línea recta.",
        "- Ninguna cifra de este reporte es una media de retorno.",
        "",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────── Comando ──────────────────────────────────────────────


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada: `python -m crypto_agents.forward_plan <directorio>`.

    Sale con 0 si los recuentos del pool son los registrados, sus líneas base coinciden con la
    corrida y con los retornos publicados y lo heredado cuadra; con 1 si algo no cuadra o la
    corrida es inválida, y con 2 si el directorio o el archivo publicado no se pudieron leer.
    """
    parser = argparse.ArgumentParser(
        prog=_COMMAND,
        description="Planificación del estudio hacia adelante de solo, sin llamar a ningún modelo.",
    )
    parser.add_argument("directory", type=Path, help="directorio de la etapa 1, con meta.json")
    parser.add_argument("--plan", type=Path, default=None, help="manifiesto del plan")
    parser.add_argument("--history-dir", type=Path, default=HISTORY_DIR)
    args = parser.parse_args(argv)

    try:
        published = published_returns(PUBLISHED_RETURNS, PUBLISHED_RETURNS_SHA256)
        stage = read_stage(args.directory, args.plan, args.history_dir, None)
        run = stage.chain[0]
        meta = run.meta
        recalculation = build_recalculation(
            stage.arms,
            stage.histories,
            stage.horizon,
            DELTA,
            stage.manifest,
            pool_cells(stage.manifest, stage.histories, progress=True),
            stage.calls,
            DEFAULT_PRICING,
            meta.billing,
            DEFAULT_RESAMPLES,
            DEFAULT_SEED,
            resumed=len(stage.chain) > 1,
        )
        report = build_report(
            stage.arms,
            stage.histories,
            stage.horizon,
            stage.manifest,
            pool_records(stage.manifest, stage.histories, progress=True),
            recalculation,
            published,
            POOL_COUNTS,
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
        ForwardPlanError,
        OSError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    flags: list[str] = []
    if args.plan is not None:
        flags.append(f"--plan {args.plan}")
    if args.history_dir != HISTORY_DIR:
        flags.append(f"--history-dir {args.history_dir}")
    sources = Sources(
        directory=str(args.directory),
        command=" ".join([_COMMAND, str(args.directory), *flags]),
        commit=commit_label(meta.git_commit, meta.git_dirty),
        plan=f"`{stage.plan_path}` ({meta.plan_kind.value}), sha-256 `{meta.plan_sha256}`",
        journals=tuple((arm.path.name, arm.sha256) for arm in run.arms if arm.sha256 is not None),
        histories=stage.history_files,
    )
    print(render_report(report, sources))
    return 0 if report.consistent else 1


if __name__ == "__main__":
    raise SystemExit(main())
