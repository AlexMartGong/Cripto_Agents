"""El análisis del estudio hacia adelante de `solo`: lo que `docs/forward.md` pide en su apartado 6.

    python -m crypto_agents.forward analyze <directorio del estudio>
    python -m crypto_agents.forward rehearse-analysis

Hay un único análisis, cuando el estudio cierra: `solo` contra `rule_trend`, sobre el retorno bruto
con el stop común, con el intervalo de la enmienda 4 y la regla de veredicto del apartado 6. Antes
de puntuar una sola evaluación, el comando se niega o da «no determinado», en este orden:

1. **El estudio terminó antes de cerrar.** Lo dicen los `forward.json` de todas las pasadas, también
   las sustituidas, porque sustituir una pasada no deshace lo que llamó:
   - el id del decisor retirado del catálogo;
   - un hueco persistente;
   - una pasada que llamó con el camino medido distinto del registrado. La configuración del
     decisor distinta no termina el estudio: invalida el lote (apartado 5).

   Va primero porque un estudio terminado nunca llega a 50 bloques: con el cierre delante, el
   comando se negaría para siempre en vez de decir «no determinado».
2. **El cierre** (apartado 2), con `forward_plan.closing_block` sobre las poblaciones de los
   bloques que cubre el histórico extendido. Si el estudio sigue abierto, se niega. Si termina el
   bloque 52 sin 50 bloques no vacíos, el veredicto es «no determinado».
3. **Los lotes completos.** Cada lote tiene que ser válido y estar cerrado, o tener el plazo
   vencido. Si no, se niega.
4. **La integridad.** Si algo de esto falla, se niega:
   - el manifiesto es la población recalculada;
   - cada pasada que corrió, sustituida o no, dejó su `forward.json` entero, y una pasada de
     líneas base no trae ni `solo` ni ninguna llamada;
   - ningún brazo tiene un `run_id` que no sea del manifiesto;
   - ningún `run_id` tiene registro completo en dos pasadas no sustituidas.
5. **Las líneas base completas.** Con el plazo vencido, cada `run_id` del manifiesto tiene que
   tener registro de las cuatro. Si no, se niega y nombra el comando que las completa sin llamar a
   ningún modelo (`forward baselines`).
6. **Las causas de «no determinado» que se leen sin puntuar**: un lote inválido al vencer su
   plazo, y las faltantes por encima del 5 %. Al vencer el plazo, toda evaluación de `solo` sin
   decisión que no sea una respuesta inválida del modelo es faltante: la del proveedor, el 402 o el
   tope, la de una pasada interrumpida y todas las de un lote que no corrió (apartado 5).

Ningún argumento se salta una negativa.

`rehearse-analysis` es el ensayo sobre la etapa 1. Las mismas funciones arman las cuatro filas de
`solo` contra cada línea base, y cada cifra se compara con la que publicó `docs/ablation.md`. El
ensayo no juzga nada.

Tres decisiones lo mantienen dentro de lo registrado:

- **δ, el 1.645 y el cierre son los de `forward_plan`**, sin copias. No se importan
  `criteria.DELTA` ni `criteria.classify`, que son los de la ablación.
- **La regla de veredicto es una función propia y pura**, `forward_verdict`. Su empate es
  exactamente la región `(a, b)` de `forward_plan.tie_region`, la que usó la planificación, y un
  test lo fija hasta el último ulp.
- **Se puntúa por las puertas de los criterios.** El vector sale de `criteria.paired_returns`. La
  resta y el intervalo salen de `metrics.clustered_paired_difference`, con un clúster por bloque de
  7 días del instante evaluado. Una faltante sin registro de `solo` entra como un registro sin
  decisión hecho en memoria (`missing_record`), que nunca se escribe: así vale 0 frente al retorno
  de `rule_trend` por la misma puerta, sin una segunda forma de alinear.

Este módulo no importa el router ni nada que lo traiga. Dos tests de arquitectura lo fijan.
"""

from __future__ import annotations

import statistics
from collections import Counter
from datetime import UTC, datetime
from enum import StrEnum
from fractions import Fraction
from typing import TYPE_CHECKING, NamedTuple

from pydantic import AwareDatetime, Field, ValidationError

from crypto_agents.activation_sweep import DEFAULT_SYMBOLS, history_path
from crypto_agents.audit import AuditError, arm_journal_path, file_sha256, read_meta, read_run
from crypto_agents.criteria import (
    BASELINE_ARMS,
    CLUSTER_DAYS,
    PERCENT_DIGITS,
    SCORING,
    CriteriaError,
    half_width,
    paired_returns,
)
from crypto_agents.design_effect import pool_run_id
from crypto_agents.forward_data import (
    BATCH_DIR,
    FORWARD_DIR,
    FORWARD_FUNDING_DIR,
    FORWARD_HISTORY_DIR,
    MANIFEST_DIR,
    REVISIONS_FILE,
    STAGE1_DIR,
    STAGE1_FILES,
    ForwardDataError,
    block_manifest,
    manifest_path,
    read_histories,
    same_population,
    study_population,
    verify_stage1,
)
from crypto_agents.forward_plan import (
    DELTA,
    FORWARD_ARMS,
    MAX_BLOCKS,
    PRIMARY,
    TARGET_BLOCKS,
    Z_EQUIVALENCE,
    closing_block,
)
from crypto_agents.forward_record import (
    FIRST_BLOCK,
    FORWARD_META_FILE,
    REGISTERED_PATH,
    STEP,
    SUPERSEDED_FILE,
    BatchCounts,
    BatchMeta,
    BatchOutcome,
    BlockDates,
    DeciderConfig,
    DuplicateRecordError,
    ForwardError,
    batch_counts,
    batch_directory,
    batch_passes,
    batch_state,
    block_dates,
    decider_mismatches,
    decider_rows,
    pass_backends,
    registered_decider,
    served_pairs,
)
from crypto_agents.funding import FUNDING_DIR, FundingError, funding_path, load_funding
from crypto_agents.journal import EvaluationRecord
from crypto_agents.market import timeframe_to_timedelta
from crypto_agents.metrics import (
    MIN_PAIRED_N,
    NO_HEADER,
    AbortKind,
    ClusteredPair,
    NetPairs,
    PairedDifference,
    clustered_paired_difference,
    clusters_of,
    paired_differences,
    paired_net_difference,
    undecided_causes,
)
from crypto_agents.outcomes import NetInputs, Scoring, score_run
from crypto_agents.paired_dispersion import RecalculationError, commit_label, read_stage
from crypto_agents.selection import (
    DEFAULT_TIMEFRAME,
    HISTORY_DIR,
    SelectionError,
    SelectionManifest,
    load_manifest,
)
from crypto_agents.settings import DEFAULT_COSTS
from crypto_agents.state import AgentRole, FrozenModel, NodeError

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Mapping, Sequence
    from pathlib import Path
    from uuid import UUID

    from crypto_agents.criteria import PairedReturns
    from crypto_agents.forward_data import StudyPopulation
    from crypto_agents.selection import PlannedEvaluation

__all__ = [
    "DESCRIPTIVE_REFERENCES",
    "FIGURES",
    "INVALID_RESPONSE",
    "MISSING_CAUSES",
    "MISSING_LIMIT",
    "MISSING_NODE",
    "NO_CALL",
    "PUBLISHED_SOURCE",
    "PUBLISHED_STAGE1",
    "VERDICT_LABEL",
    "AnalysisRefusedError",
    "AnalysisResult",
    "BlockRead",
    "Closing",
    "Described",
    "PairComparison",
    "Scored",
    "StudyPaths",
    "Termination",
    "Verdict",
    "align",
    "analyze",
    "by_upstream",
    "compare_aligned",
    "covered_blocks",
    "decider_upstream",
    "describe",
    "forward_verdict",
    "interval_90",
    "invalid_at_deadline",
    "missing_by_cause",
    "missing_exceeds",
    "missing_record",
    "own_stop",
    "pass_metas",
    "primary_verdict",
    "read_block",
    "rehearse_analysis",
    "score_study",
    "stage1_figures",
    "study_closing",
    "study_paths",
    "termination",
]

MISSING_LIMIT = Fraction(5, 100)
"""Las faltantes, sobre las evaluaciones del estudio, por encima de las cuales el estudio es «no
determinado» (`docs/forward.md`, apartado 5). La comparación es exacta: un 5.00 % no lo supera."""

ARM, REFERENCE = PRIMARY
DESCRIPTIVE_REFERENCES = tuple(arm for arm in BASELINE_ARMS if arm != REFERENCE)
"""Las otras tres líneas base: descriptivas, sin palabra de veredicto."""

NO_CALL = "sin llamada"
"""El grupo, en el reparto por upstream, de las faltantes: no hubo llamada válida del decisor."""

INVALID_RESPONSE = "respuesta inválida"
"""El grupo de las respuestas inválidas tras los dos intentos: el decisor contestó, mal."""

MISSING_NODE = "análisis"
"""El nodo del error de un registro hecho en memoria para una faltante sin registro de `solo`."""

MISSING_CAUSES = ("402", "proveedor", "tope", "otra causa", "sin registro")
"""Las causas de una faltante al vencer el plazo, en el orden en que se imprimen. «sin registro» es
el `run_id` del manifiesto que ninguna pasada evaluó: una pasada interrumpida o un lote que no
corrió."""

_COMMAND = "python -m crypto_agents.forward"


class AnalysisRefusedError(RuntimeError):
    """El análisis se niega: el estudio no ha cerrado o su registro no es íntegro. No hay
    veredicto ni se puntúa nada."""


class AnalysisResult(NamedTuple):
    """Lo que el comando imprime y con qué código sale."""

    code: int
    lines: tuple[str, ...]


# ───────────────────────────────────────── La regla de veredicto ──────────────────────────────────


class Verdict(StrEnum):
    """Los cinco veredictos de la comparación principal (apartado 6)."""

    ABOVE = "above"
    """El IC 95 % queda entero por encima de 0: `solo` supera a `rule_trend`."""

    BELOW = "below"
    """El IC 95 % queda entero por debajo de 0."""

    TIE = "tie"
    """El IC 90 % queda entero dentro de (-δ, +δ) y el IC 95 % contiene el 0."""

    INCONCLUSIVE = "inconclusive"
    """Cualquier otro caso."""

    UNDETERMINED = "undetermined"
    """No hay intervalo, o el estudio no llegó a producir uno que se pueda leer."""


VERDICT_LABEL = {
    Verdict.ABOVE: "supera",
    Verdict.BELOW: "peor",
    Verdict.TIE: "empate",
    Verdict.INCONCLUSIVE: "no concluyente",
    Verdict.UNDETERMINED: "no determinado",
}


def forward_verdict(interval: PairedDifference | None, delta: float) -> Verdict:
    """El veredicto del apartado 6 sobre el intervalo de la enmienda 4, en este orden:

    1. no determinado: no hay intervalo;
    2. supera: el IC 95 % queda entero por encima de 0;
    3. peor: el IC 95 % queda entero por debajo de 0;
    4. empate: el IC 90 % queda entero dentro de (-δ, +δ), con bordes estrictos, y el IC 95 %
       contiene el 0, también estricto;
    5. no concluyente: cualquier otro caso.

    El IC 95 % es el del intervalo (`media ± 1.96 · EE usado`); el IC 90 %, `media ± 1.645 · EE
    usado`. Se decide con la precisión completa. El IC 90 % se compara escribiendo la media contra
    sus dos límites, `-δ + 1.645·EE < media < δ - 1.645·EE`, y no cada extremo contra ±δ: son las
    mismas operaciones de coma flotante que `forward_plan.tie_region`, así que la región de empate
    es exactamente la `(a, b)` con que se planificó. La cláusula del IC 95 % deja supera y empate
    excluyentes.
    """
    if delta <= 0.0:
        raise ValueError(f"delta tiene que ser positivo: {delta}")
    if interval is None:
        return Verdict.UNDETERMINED
    if interval.low > 0.0:
        return Verdict.ABOVE
    if interval.high < 0.0:
        return Verdict.BELOW
    margin = Z_EQUIVALENCE * interval.stderr
    inside = -delta + margin < interval.mean < delta - margin
    if inside and interval.low < 0.0 < interval.high:
        return Verdict.TIE
    return Verdict.INCONCLUSIVE


def interval_90(interval: PairedDifference) -> tuple[float, float]:
    """El IC 90 % que se imprime: `media ± 1.645 · EE usado`. El veredicto no lo lee de aquí."""
    margin = Z_EQUIVALENCE * interval.stderr
    return interval.mean - margin, interval.mean + margin


def primary_verdict(pair: ClusteredPair | None) -> Verdict:
    """El veredicto de la comparación principal, con el δ registrado del estudio leído al llamar."""
    return forward_verdict(None if pair is None else pair.interval, DELTA)


def missing_exceeds(missing: int, evaluations: int) -> bool:
    """Si las faltantes superan `MISSING_LIMIT` de las evaluaciones del estudio. Exacto."""
    if evaluations <= 0:
        return False
    return Fraction(missing, evaluations) > MISSING_LIMIT


# ──────────────────────────────────────────── Comparar ────────────────────────────────────────────


class PairComparison(FrozenModel):
    """Un brazo contra otro sobre las evaluaciones que comparten, sin veredicto."""

    arm: str
    reference: str
    n: int = Field(ge=0)
    unpaired: int = Field(ge=0)
    pair: ClusteredPair | None = None
    """La diferencia con sus dos errores y el intervalo del mayor. `None` con menos de
    `MIN_PAIRED_N` parejas."""

    net: NetPairs | None = None
    """La misma diferencia sobre el retorno neto, o `None` si no se pidió."""


def _clusters(aligned: PairedReturns) -> list[int]:
    """Un clúster por bloque de 7 días del instante evaluado: cada bloque del estudio es uno."""
    return clusters_of(aligned.arm_records, CLUSTER_DAYS)


def align(
    arm: str,
    reference: str,
    arms: Mapping[str, Sequence[EvaluationRecord]],
    histories: Mapping[str, Sequence[Sequence[float]]],
    horizon: int,
) -> PairedReturns:
    """Los dos vectores de retorno bruto que se restan, por `criteria.paired_returns`, o un error si
    no se pueden emparejar: sin ellos no hay nada que comparar."""
    aligned = paired_returns(arm, reference, arms, histories, horizon)
    if isinstance(aligned, str):
        raise CriteriaError(f"`{arm}` contra `{reference}`: {aligned}")
    return aligned


def compare_aligned(
    arm: str,
    reference: str,
    aligned: PairedReturns,
    histories: Mapping[str, Sequence[Sequence[float]]],
    horizon: int,
    net: NetInputs | None = None,
) -> PairComparison:
    """La diferencia pareada bruta de dos brazos ya alineados, con el intervalo de la enmienda 4, y
    la neta si se pasa `net`.

    La resta y el intervalo son los de `metrics.clustered_paired_difference`. El neto puntúa los
    mismos registros alineados con la misma puntuación y el funding, por `outcomes.net_return`; las
    parejas sin funding completo se cuentan y se quitan (`metrics.paired_net_difference`).
    """
    clusters = _clusters(aligned)
    pair = clustered_paired_difference(aligned.arm_returns, aligned.reference_returns, clusters)
    net_pairs: NetPairs | None = None
    if net is not None:
        ours = score_run(aligned.arm_records, histories, horizon, SCORING, net)
        theirs = score_run(aligned.reference_records, histories, horizon, SCORING, net)
        if ours.per_evaluation_net is None or theirs.per_evaluation_net is None:
            raise ValueError("la puntuación con funding no devolvió el neto")
        net_pairs = paired_net_difference(
            ours.per_evaluation_net, theirs.per_evaluation_net, clusters
        )
    return PairComparison(
        arm=arm,
        reference=reference,
        n=len(aligned.arm_returns),
        unpaired=aligned.unpaired,
        pair=pair,
        net=net_pairs,
    )


class Described(FrozenModel):
    """Una diferencia descriptiva: su media y su n siempre, su intervalo cuando lo hay."""

    label: str
    n: int = Field(ge=0)
    mean: float | None = None
    pair: ClusteredPair | None = None

    @property
    def why(self) -> str | None:
        """Por qué no hay intervalo, o `None` si lo hay."""
        if self.pair is None:
            return f"menos de {MIN_PAIRED_N} parejas"
        return self.pair.why


def describe(
    label: str,
    arm_values: Sequence[float],
    reference_values: Sequence[float],
    clusters: Sequence[Hashable],
) -> Described:
    """La diferencia `arm - reference` de un subconjunto, con la resta de `metrics`."""
    differences = paired_differences(arm_values, reference_values)
    return Described(
        label=label,
        n=len(differences),
        mean=statistics.fmean(differences) if differences else None,
        pair=(
            clustered_paired_difference(arm_values, reference_values, clusters)
            if differences
            else None
        ),
    )


def own_stop(
    aligned: PairedReturns, histories: Mapping[str, Sequence[Sequence[float]]], horizon: int
) -> Described:
    """La comparación principal puntuada con el stop que declara cada brazo. Descriptiva."""
    ours = score_run(aligned.arm_records, histories, horizon, Scoring.OWN_STOP)
    theirs = score_run(aligned.reference_records, histories, horizon, Scoring.OWN_STOP)
    return describe("stop propio", ours.per_evaluation, theirs.per_evaluation, _clusters(aligned))


def decider_upstream(record: EvaluationRecord) -> str:
    """El grupo de una evaluación de `solo` en el reparto por upstream.

    Es quien sirvió la llamada válida del decisor. Sin llamada válida, una respuesta inválida tras
    los dos intentos va a su propio grupo, y lo demás —las faltantes— a «sin llamada».
    """
    valid = [call for call in record.calls if call.role is AgentRole.DECIDER and call.valid]
    if valid:
        call = valid[-1]
        if call.upstream_model is None:
            return NO_HEADER
        if call.upstream_endpoint is None:
            return f"`{call.upstream_model}`, sin endpoint"
        return f"`{call.upstream_model}` por `{call.upstream_endpoint}`"
    if any(kind is AbortKind.VALIDATION for _, kind in undecided_causes([record])):
        return INVALID_RESPONSE
    return NO_CALL


def by_upstream(aligned: PairedReturns) -> tuple[Described, ...]:
    """La comparación principal separada por el upstream de la llamada válida del decisor."""
    clusters = _clusters(aligned)
    groups: dict[str, list[int]] = {}
    for position, record in enumerate(aligned.arm_records):
        groups.setdefault(decider_upstream(record), []).append(position)
    return tuple(
        describe(
            label,
            [aligned.arm_returns[i] for i in positions],
            [aligned.reference_returns[i] for i in positions],
            [clusters[i] for i in positions],
        )
        for label, positions in sorted(groups.items())
    )


# ──────────────────────────────────────── El estudio en disco ─────────────────────────────────────


class StudyPaths(NamedTuple):
    """Dónde lee el análisis, dentro del directorio del estudio: lo que escribe el lote."""

    root: Path
    history: Path
    funding: Path
    manifests: Path
    batches: Path
    revisions: Path


def study_paths(directory: Path) -> StudyPaths:
    """Las rutas del estudio bajo `directory`, con los mismos nombres que `data/forward/`."""

    def inside(path: Path) -> Path:
        return directory / path.relative_to(FORWARD_DIR)

    return StudyPaths(
        root=directory,
        history=inside(FORWARD_HISTORY_DIR),
        funding=inside(FORWARD_FUNDING_DIR),
        manifests=inside(MANIFEST_DIR),
        batches=inside(BATCH_DIR),
        revisions=inside(REVISIONS_FILE),
    )


class Termination(FrozenModel):
    """Por qué y dónde terminó el estudio antes de cerrar (apartado 5)."""

    block: int = Field(ge=1)
    start: AwareDatetime
    where: str
    """La pasada que lo dice."""

    cause: str


def pass_metas(batches: Path) -> list[tuple[Path, BatchMeta]]:
    """El `forward.json` de cada pasada de cada lote, también las que no corrieron y las
    sustituidas: sustituir una pasada no deshace lo que termina el estudio (apartado 5)."""
    found: list[tuple[Path, BatchMeta]] = []
    if not batches.is_dir():
        return found
    for block_dir in sorted(batches.iterdir()):
        if not block_dir.is_dir():
            continue
        for directory in sorted(block_dir.iterdir()):
            record = directory / FORWARD_META_FILE
            if record.is_file():
                meta = BatchMeta.model_validate_json(record.read_text(encoding="utf-8"))
                found.append((directory, meta))
    return found


def termination(metas: Iterable[tuple[Path, BatchMeta]]) -> Termination | None:
    """La primera terminación del estudio que registran las pasadas, o `None`.

    Tres causas: el decisor registrado salió del catálogo, una vela nueva siguió faltando 24 h
    después, o una pasada que corrió llamó a un modelo con el camino medido distinto del registrado
    (apartado 3). Cuenta también la pasada sustituida. Una pasada que se negó antes de llamar no
    termina nada, y una con la configuración del decisor distinta invalida su lote sin terminar el
    estudio (`decider_mismatches`).
    """
    found: list[Termination] = []
    for directory, meta in metas:
        cause: str | None = None
        if meta.outcome is BatchOutcome.TERMINATED:
            cause = "el decisor registrado ya no está en el catálogo de la pasarela"
        elif meta.outcome is BatchOutcome.GAP_PERSISTENT:
            cause = "una vela nueva siguió faltando 24 h después: hueco persistente"
        elif meta.outcome is BatchOutcome.RAN and meta.measured_path is not None:
            changed = meta.measured_path.differences(REGISTERED_PATH)
            if changed:
                cause = (
                    "una pasada llamó a un modelo con el camino medido distinto del registrado: "
                    f"{', '.join(changed)}"
                )
        if cause is not None:
            found.append(
                Termination(
                    block=meta.block.number,
                    start=meta.block.start,
                    where=str(directory),
                    cause=cause,
                )
            )
    return min(found, key=lambda item: (item.start, item.where)) if found else None


def covered_blocks(histories: Mapping[str, Sequence[Sequence[float]]]) -> int:
    """Cuántos bloques del estudio, desde el 1 y hasta `MAX_BLOCKS`, cubre el histórico: los que
    tienen en las siete series sus velas hasta el cierre más el horizonte."""
    if not histories or any(not rows for rows in histories.values()):
        return 0
    period = timeframe_to_timedelta(DEFAULT_TIMEFRAME)
    covered = min(
        datetime.fromtimestamp(rows[-1][0] / 1000, UTC) + period for rows in histories.values()
    )
    count = 0
    while count < MAX_BLOCKS and block_dates(FIRST_BLOCK + count * STEP).ready <= covered:
        count += 1
    return count


class Closing(NamedTuple):
    """Los bloques que cubre el histórico, su población y el bloque en que cierra el estudio."""

    starts: tuple[datetime, ...]
    counts: tuple[int, ...]
    population: StudyPopulation
    block: int | None
    """`None`: terminó el bloque 52 sin 50 bloques no vacíos."""


def study_closing(histories: Mapping[str, Sequence[Sequence[float]]]) -> Closing:
    """La regla de cierre del apartado 2 sobre la población de cada bloque que cubre el histórico.

    Solo lee cuántas activaciones tuvo cada bloque. Un estudio que sigue abierto es una negativa.
    """
    covered = covered_blocks(histories)
    starts = tuple(FIRST_BLOCK + position * STEP for position in range(covered))
    population = study_population(histories, starts)
    counts = tuple(len(population.blocks[start]) for start in starts)
    try:
        block = closing_block(counts)
    except ValueError:
        filled = sum(1 for count in counts if count)
        raise AnalysisRefusedError(
            f"el estudio sigue abierto: el histórico extendido cubre {covered} bloques, {filled} "
            f"no vacíos, y el estudio cierra con {TARGET_BLOCKS} no vacíos (apartado 2)"
        ) from None
    return Closing(starts, counts, population, block)


class BlockRead(NamedTuple):
    """Lo que dejó el lote de un bloque, leído de sus pasadas. Sin velas: no puntúa."""

    dates: BlockDates
    evaluations: int
    """Las activaciones de su población."""

    passes: tuple[Path, ...]
    """Las pasadas no sustituidas que corrieron, con journals."""

    superseded: tuple[Path, ...]
    """Las pasadas sustituidas: fuera del lote (apartado 5)."""

    records: dict[str, tuple[EvaluationRecord, ...]]
    """Por brazo, uno por `run_id`: el completo, o el último incompleto."""

    counts: BatchCounts | None
    """Los contadores del lote, o `None` si no corrió ninguna pasada."""


def read_block(
    paths: StudyPaths, start: datetime, evaluations: int, registered: DeciderConfig
) -> BlockRead:
    """El lote del bloque que empieza en `start`, juntando sus pasadas no sustituidas como el lote.

    Un `run_id` completo en dos pasadas no sustituidas levanta `DuplicateRecordError`: el análisis
    se niega. Una pasada que corrió `solo` con la configuración del decisor distinta de la
    registrada deja el lote inválido (`decider_mismatches`).
    """
    dates = block_dates(start)
    directory = batch_directory(paths.batches, start)
    passes = tuple(batch_passes(directory))
    superseded = tuple(
        path for path in batch_passes(directory, include_superseded=True) if path not in passes
    )
    if not passes:
        return BlockRead(dates, evaluations, (), superseded, {}, None)
    state = batch_state(passes)
    records = {arm: tuple(by_id.values()) for arm, by_id in state.records.items()}
    records.setdefault(ARM, ())
    counts = batch_counts(
        evaluations, records, pass_backends(passes), decider_mismatches(passes, registered)
    )
    return BlockRead(dates, evaluations, passes, superseded, records, counts)


def _refuse_incomplete(blocks: Sequence[BlockRead], now: datetime) -> None:
    """Cada bloque no vacío tiene su lote completo: válido y cerrado, o con el plazo vencido."""
    for block in blocks:
        if block.evaluations == 0 or now >= block.dates.deadline:
            continue
        if block.counts is None or not block.counts.complete:
            raise AnalysisRefusedError(
                f"el lote del bloque {block.dates.number} no está completo y su plazo vence el "
                f"{block.dates.deadline.isoformat()}: hay que completarlo antes (apartado 5)"
            )


def _checked_manifest(
    path: Path,
    histories: Mapping[str, Sequence[Sequence[float]]],
    population: StudyPopulation,
    dates: BlockDates,
) -> SelectionManifest:
    saved = load_manifest(path)
    recomputed = block_manifest(histories, population, dates.start, saved.created_at)
    if recomputed is None or not same_population(saved, recomputed):
        raise AnalysisRefusedError(
            f"el manifiesto del bloque {dates.number} ({path}) no es la población recalculada "
            "desde el histórico extendido"
        )
    return saved


def _entry_run_id(entry: PlannedEvaluation) -> UUID:
    """El `run_id` que el replay da a una entrada: el de su instante evaluado."""
    return pool_run_id(
        entry.symbol, entry.timeframe, entry.at + timeframe_to_timedelta(entry.timeframe)
    )


def _check_pass(directory: Path, digest: str) -> None:
    """El `forward.json` de una pasada que corrió, sustituida o no, está entero y es coherente."""
    record = directory / FORWARD_META_FILE
    if not record.is_file():
        raise AnalysisRefusedError(f"la pasada {directory} corrió y no dejó su {FORWARD_META_FILE}")
    meta = BatchMeta.model_validate_json(record.read_text(encoding="utf-8"))
    if meta.outcome is BatchOutcome.RAN:
        if meta.measured_path is None or meta.decider is None:
            raise AnalysisRefusedError(
                f"la pasada {directory} corrió sin registrar el camino medido o el decisor"
            )
    elif meta.outcome is BatchOutcome.BASELINES:
        run = read_run(directory)
        if any(arm.arm == ARM for arm in run.arms) or arm_journal_path(directory, ARM).is_file():
            raise AnalysisRefusedError(
                f"la pasada de líneas base {directory} trae `{ARM}`: solo corre las cuatro líneas "
                "base"
            )
        calls = sum(len(record.calls) for arm in run.arms for record in arm.records)
        if calls:
            raise AnalysisRefusedError(
                f"la pasada de líneas base {directory} trae {calls} llamadas a un modelo"
            )
    else:
        raise AnalysisRefusedError(
            f"la pasada {directory} tiene journals y su {FORWARD_META_FILE} dice "
            f"«{meta.outcome.value}»"
        )
    if meta.manifest_sha256 != digest:
        raise AnalysisRefusedError(
            f"la pasada {directory} corrió con otro manifiesto: registra {meta.manifest_sha256}, "
            f"y el manifiesto del bloque tiene {digest}"
        )


def _check_integrity(
    paths: StudyPaths,
    block: BlockRead,
    histories: Mapping[str, Sequence[Sequence[float]]],
    population: StudyPopulation,
) -> SelectionManifest | None:
    """La integridad de un bloque hasta el de cierre. Devuelve su manifiesto, si lo tiene.

    Se niega por cuatro cosas: un manifiesto que no es la población, un `forward.json` que falta o
    está incompleto en una pasada que corrió —sustituida o no—, un `run_id` que no es del manifiesto
    en algún brazo, o una pasada de líneas base con `solo` o con llamadas. Los duplicados entre
    pasadas no sustituidas ya los paró `read_block`.

    Un `run_id` del manifiesto sin registro no es un fallo de integridad: con el plazo vencido es
    una faltante si es de `solo` (`missing_by_cause`) y una línea base por completar si es de una
    línea base (`_refuse_without_baselines`).
    """
    dates = block.dates
    path = manifest_path(dates.start, paths.manifests)
    ran = (*block.passes, *block.superseded)
    if block.evaluations == 0:
        if path.is_file():
            raise AnalysisRefusedError(
                f"el bloque {dates.number} está vacío y tiene manifiesto: {path}"
            )
        if ran:
            raise AnalysisRefusedError(
                f"el bloque {dates.number} está vacío y tiene journals en {ran[0]}"
            )
        return None
    if not path.is_file():
        if ran:
            raise AnalysisRefusedError(
                f"el lote del bloque {dates.number} corrió sin manifiesto: {path}"
            )
        return None
    saved = _checked_manifest(path, histories, population, dates)
    digest = file_sha256(path)
    for directory in ran:
        _check_pass(directory, digest)
    expected = {_entry_run_id(entry) for entry in saved.entries}
    for arm, records in block.records.items():
        extra = {record.run_id for record in records} - expected
        if extra:
            raise AnalysisRefusedError(
                f"bloque {dates.number}, `{arm}`: {len(extra)} run_id que no son del manifiesto"
            )
    return saved


def _refuse_without_baselines(
    blocks: Sequence[BlockRead],
    manifests: Sequence[SelectionManifest | None],
    population: StudyPopulation,
    now: datetime,
) -> None:
    """Con el plazo vencido, cada `run_id` del manifiesto tiene registro de las cuatro líneas base.

    Sin ellas, una faltante de `solo` no tendría con qué emparejarse. Se pueden completar después
    del plazo sin llamar a ningún modelo (apartado 5), así que es una negativa y no «no
    determinado». Antes del plazo, un lote sin ellas ya está incompleto.
    """
    for block, manifest in zip(blocks, manifests, strict=True):
        if block.evaluations == 0 or now < block.dates.deadline:
            continue
        entries = population.blocks[block.dates.start] if manifest is None else manifest.entries
        expected = {_entry_run_id(entry) for entry in entries}
        short = [
            f"`{arm}` {len(expected - {record.run_id for record in block.records.get(arm, ())})}"
            for arm in BASELINE_ARMS
            if expected - {record.run_id for record in block.records.get(arm, ())}
        ]
        if short:
            raise AnalysisRefusedError(
                f"bloque {block.dates.number}: run_id del manifiesto sin registro de las líneas "
                f"base ({', '.join(short)}). Se completan sin llamar a ningún modelo con "
                f"`{_COMMAND} baselines --block {block.dates.start:%Y-%m-%d}` (apartado 5)"
            )


def _invalid_reasons(block: BlockRead) -> list[str]:
    """Lo que deja inválido el lote de un bloque no vacío al vencer su plazo (apartado 5).

    Un lote que no corrió no es inválido: sus evaluaciones son faltantes. Tampoco lo es una
    evaluación de `solo` sin decisión por otra causa, que también es faltante.
    """
    counts = block.counts
    if counts is None:
        return []
    reasons = list(counts.invalidating)
    undecided = sorted(
        f"{arm} {sum(1 for record in block.records.get(arm, ()) if record.proposed is None)}"
        for arm in BASELINE_ARMS
        if any(record.proposed is None for record in block.records.get(arm, ()))
    )
    if undecided:
        reasons.append(f"líneas base con registros sin decisión: {', '.join(undecided)}")
    return reasons


def invalid_at_deadline(blocks: Sequence[BlockRead], now: datetime) -> list[str]:
    """Los bloques no vacíos con el plazo vencido cuyo lote sigue inválido, con el porqué."""
    found: list[str] = []
    for block in blocks:
        if block.evaluations == 0 or now < block.dates.deadline:
            continue
        reasons = _invalid_reasons(block)
        if reasons:
            found.append(f"bloque {block.dates.number} ({'; '.join(reasons)})")
    return found


def missing_by_cause(block: BlockRead) -> dict[str, int]:
    """Las faltantes de `solo` de un bloque al vencer su plazo, por causa (`MISSING_CAUSES`).

    Toda evaluación sin decisión que no sea una respuesta inválida del modelo: la que perdió el
    proveedor, un 402 o el tope; la que no tiene decisión por otra causa; y la que no tiene
    registro, de una pasada interrumpida o de un lote que no corrió (apartado 5). Su suma es
    `BatchCounts.missing_at_deadline`, o todas las evaluaciones si el lote no corrió.
    """
    counts = block.counts
    if counts is None:
        return {**dict.fromkeys(MISSING_CAUSES, 0), "sin registro": block.evaluations}
    return {
        **counts.missing_by_cause,
        "otra causa": counts.other_undecided,
        "sin registro": counts.unrecorded,
    }


def _missing(block: BlockRead) -> int:
    """Cuántas faltantes deja el bloque al vencer su plazo."""
    if block.counts is None:
        return block.evaluations
    return block.counts.missing_at_deadline


def missing_record(entry: PlannedEvaluation) -> EvaluationRecord:
    """Una faltante sin registro de `solo`, hecha en memoria para puntuarla: sin decisión, vale 0.

    No se escribe nunca. Lleva el `run_id`, el instante, el símbolo y la temporalidad de la entrada
    del manifiesto, que es lo que `criteria.paired_returns` empareja con `rule_trend`, y un error
    del nodo `MISSING_NODE` que dice qué es. En el reparto por upstream cae en «sin llamada».
    """
    at = entry.at + timeframe_to_timedelta(entry.timeframe)
    return EvaluationRecord(
        run_id=_entry_run_id(entry),
        at=at,
        symbol=entry.symbol,
        timeframe=entry.timeframe,
        errors=(
            NodeError(
                node=MISSING_NODE,
                message=f"faltante: `{ARM}` sin registro al vencer el plazo del lote",
                at=at,
            ),
        ),
    )


# ────────────────────────────────────────────── El análisis ───────────────────────────────────────


class Scored(NamedTuple):
    """Todo lo que se puntúa, una vez pasadas las negativas."""

    horizon: int
    primary: PairComparison
    verdict: Verdict
    descriptive: tuple[PairComparison, ...]
    own: Described
    upstreams: tuple[Described, ...]
    net_lines: tuple[str, ...]


def score_study(
    blocks: Sequence[BlockRead],
    manifests: Sequence[SelectionManifest | None],
    horizon: int,
    histories: Mapping[str, Sequence[Sequence[float]]],
    funding: Path,
) -> Scored:
    """La comparación principal y los descriptivos, sobre los registros de los bloques 1 al de
    cierre. Solo se llama cuando ninguna negativa ni ninguna causa de «no determinado» saltó.

    Cada `run_id` del manifiesto sin registro de `solo` entra como `missing_record`: una faltante
    vale 0 frente al retorno de `rule_trend`, por la misma puerta que las demás.
    """
    arms = {
        arm: [record for block in blocks for record in block.records.get(arm, ())]
        for arm in FORWARD_ARMS
    }
    for block, manifest in zip(blocks, manifests, strict=True):
        if manifest is None:
            continue
        recorded = {record.run_id for record in block.records.get(ARM, ())}
        arms[ARM] += [
            missing_record(entry)
            for entry in manifest.entries
            if _entry_run_id(entry) not in recorded
        ]
    symbols = sorted({record.symbol for records in arms.values() for record in records})
    net, net_lines = _net_inputs(symbols, funding)
    aligned = align(ARM, REFERENCE, arms, histories, horizon)
    primary = compare_aligned(ARM, REFERENCE, aligned, histories, horizon, net)
    descriptive = tuple(
        compare_aligned(
            ARM, reference, align(ARM, reference, arms, histories, horizon), histories, horizon
        )
        for reference in DESCRIPTIVE_REFERENCES
    )
    return Scored(
        horizon=horizon,
        primary=primary,
        verdict=primary_verdict(primary.pair),
        descriptive=descriptive,
        own=own_stop(aligned, histories, horizon),
        upstreams=by_upstream(aligned),
        net_lines=tuple(net_lines),
    )


def _net_inputs(symbols: Sequence[str], directory: Path) -> tuple[NetInputs | None, list[str]]:
    """El funding y los costes del neto descriptivo, con lo que hay que citar de ellos."""
    costs = DEFAULT_COSTS
    lines = [
        f"- costes del neto: comisión taker {costs.taker_fee:.4%} por lado, deslizamiento "
        f"{costs.slippage:.4%} por lado (supuesto sin medir), ida y vuelta "
        f"{costs.round_trip:.4%} (`settings.DEFAULT_COSTS`, {costs.as_of.isoformat()}); funding "
        f"de `{directory}`"
    ]
    try:
        return NetInputs(funding=load_funding(symbols, directory), costs=costs), lines
    except FundingError as error:
        return None, [*lines, f"- funding: sin cifra: {error}"]


def analyze(
    directory: Path,
    now: datetime,
    git: tuple[str | None, bool | None],
    stage1: Path = STAGE1_DIR,
) -> AnalysisResult:
    """El análisis del estudio en `directory`: el de `data/forward/`.

    Sale con 0 si da veredicto —«no determinado» incluido—, con 1 si se niega y con 2 si no pudo
    leer lo que necesita. `now` decide qué plazos vencieron; `git` es el commit del análisis.
    """
    head = [
        "# Análisis del estudio hacia adelante de `solo`",
        "",
        f"Reproduce: `{_COMMAND} analyze {directory}`",
        "",
        f"- commit del análisis: {commit_label(*git)}",
        f"- estudio: `{directory}`; instante del análisis: {now.astimezone(UTC).isoformat()}",
    ]
    try:
        body = _analysis(study_paths(directory), now, stage1)
    except AnalysisRefusedError as refusal:
        return AnalysisResult(1, (*head, "", *_refusal(str(refusal))))
    except DuplicateRecordError as error:
        return AnalysisResult(
            1, (*head, "", *_refusal(f"un run_id con decisión en dos pasadas: {error}"))
        )
    except (
        AuditError,
        CriteriaError,
        ForwardDataError,
        ForwardError,
        SelectionError,
        ValidationError,
        OSError,
    ) as error:
        return AnalysisResult(2, (f"error: {error}",))
    return AnalysisResult(0, (*head, *body))


def _refusal(reason: str) -> list[str]:
    return [
        f"**El análisis se niega**: {reason}.",
        "",
        "No se puntuó ninguna evaluación y no hay veredicto.",
    ]


def _analysis(paths: StudyPaths, now: datetime, stage1: Path) -> list[str]:
    """Las negativas en su orden, las causas de «no determinado» y, si nada salta, la puntuación."""
    if not paths.root.is_dir():
        raise ForwardDataError(f"no existe el directorio del estudio {paths.root}")
    verify_stage1(stage1)
    registered = registered_decider(read_meta(stage1))
    stage1_served = served_pairs(decider_rows(_stage1_solo(stage1)))

    metas = pass_metas(paths.batches)
    ended = termination(metas)
    if ended is not None:
        starts = sorted({meta.block.start for _, meta in metas})
        blocks = [
            read_block(paths, start, _manifest_size(paths, start), registered) for start in starts
        ]
        cause = (
            f"el estudio terminó antes de cerrar, en el bloque {ended.block}: {ended.cause} "
            f"(`{ended.where}`)"
        )
        return _undetermined(cause, blocks, paths, stage1_served, now)

    histories = read_histories(paths.history)
    closing = study_closing(histories)
    if closing.block is None:
        blocks = [
            read_block(paths, start, count, registered)
            for start, count in zip(closing.starts, closing.counts, strict=True)
        ]
        filled = sum(1 for count in closing.counts if count)
        cause = (
            f"al terminar el bloque {MAX_BLOCKS} hay {filled} bloques no vacíos, menos de "
            f"{TARGET_BLOCKS} (apartado 2)"
        )
        return _undetermined(cause, blocks, paths, stage1_served, now)

    last = closing.block
    blocks = [
        read_block(paths, start, count, registered)
        for start, count in zip(closing.starts[:last], closing.counts[:last], strict=True)
    ]
    _refuse_incomplete(blocks, now)
    manifests = [_check_integrity(paths, block, histories, closing.population) for block in blocks]
    _refuse_without_baselines(blocks, manifests, closing.population, now)

    causes: list[str] = []
    invalid = invalid_at_deadline(blocks, now)
    if invalid:
        causes.append(f"un lote inválido al vencer su plazo: {', '.join(invalid)}")
    evaluations = sum(block.evaluations for block in blocks)
    missing = sum(_missing(block) for block in blocks)
    if missing_exceeds(missing, evaluations):
        causes.append(
            f"las faltantes son {missing} de {evaluations} evaluaciones "
            f"({missing / evaluations:.2%}), más del {float(MISSING_LIMIT):.0%} (apartado 5)"
        )
    if causes:
        return _undetermined("; ".join(causes), blocks, paths, stage1_served, now)

    horizon = next(manifest.horizon for manifest in manifests if manifest is not None)
    scored = score_study(blocks, manifests, horizon, histories, paths.funding)
    return _scored_report(scored, blocks, paths, stage1_served, now)


def _stage1_solo(stage1: Path) -> tuple[EvaluationRecord, ...]:
    """Los registros de `solo` en la etapa 1: de donde parte el recuento de cambios de upstream."""
    run = read_run(stage1)
    return next((arm.records for arm in run.arms if arm.arm == ARM), ())


def _manifest_size(paths: StudyPaths, start: datetime) -> int:
    path = manifest_path(start, paths.manifests)
    return len(load_manifest(path).entries) if path.is_file() else 0


# ────────────────────────────────────────────── El reporte ────────────────────────────────────────


def _pct(value: float) -> str:
    return f"{value:+.{PERCENT_DIGITS}%}"


def _error(value: float | None) -> str:
    return "—" if value is None else f"{value:.{PERCENT_DIGITS}%}"


def _deff(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _bounds(low: float, high: float) -> str:
    return f"[{_pct(low)}, {_pct(high)}]"


def _primary_table(primary: PairComparison) -> list[str]:
    header = [
        "| n | sin pareja | diferencia media | EE ingenuo | EE agrupado | EE usado | G | deff "
        "| IC 95% | IC 90% |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    pair = primary.pair
    if pair is None:
        cells = "— | — | — | — | — | — | — | —"
        return [*header, f"| {primary.n} | {primary.unpaired} | {cells} |"]
    error, interval = pair.error, pair.interval
    figures = (
        f"| {primary.n} | {primary.unpaired} | {_pct(pair.naive.mean)} | {_error(error.naive)} "
        f"| {_error(error.clustered)} | {_error(None if interval is None else interval.stderr)} "
        f"| {error.clusters} | {_deff(error.deff)} "
    )
    if interval is None:
        return [*header, f"{figures}| — | — |"]
    return [
        *header,
        f"{figures}| {_bounds(interval.low, interval.high)} | {_bounds(*interval_90(interval))} |",
    ]


def _rule_lines() -> list[str]:
    bound = f"{DELTA:.{PERCENT_DIGITS}%}"
    return [
        f"La regla de `docs/forward.md`, apartado 6, con δ = {bound} de retorno bruto por "
        f"evaluación (`forward_plan.DELTA`, fracción {DELTA:g}). Se aplica en este orden, y decide "
        "la primera que se cumple:",
        "",
        f"1. **{VERDICT_LABEL[Verdict.UNDETERMINED]}**: no hay intervalo;",
        f"2. **{VERDICT_LABEL[Verdict.ABOVE]}**: el IC 95% queda entero por encima de 0;",
        f"3. **{VERDICT_LABEL[Verdict.BELOW]}**: el IC 95% queda entero por debajo de 0;",
        f"4. **{VERDICT_LABEL[Verdict.TIE]}**: el IC 90% queda entero dentro de (-{bound}, "
        f"+{bound}), con bordes estrictos, y el IC 95% contiene el 0;",
        f"5. **{VERDICT_LABEL[Verdict.INCONCLUSIVE]}**: cualquier otro caso.",
        "",
        "- El intervalo es el de la enmienda 4. El EE usado es `max(EE ingenuo, EE agrupado)`, con "
        f"el CR1 y un clúster por bloque de {CLUSTER_DAYS} días del instante evaluado, todos los "
        "símbolos juntos.",
        f"- IC 95% = media ± 1.96 · EE usado; IC 90% = media ± {Z_EQUIVALENCE} · EE usado.",
        "- Con menos de 50 clústeres no hay intervalo.",
        "- Se decide con la precisión completa; las cifras se redondean solo para imprimirlas.",
    ]


def _described_row(item: Described) -> str:
    pair = item.pair
    clusters = "—" if pair is None else str(pair.error.clusters)
    mean = "—" if item.mean is None else _pct(item.mean)
    if pair is None or pair.interval is None:
        return f"| {item.label} | {item.n} | {mean} | sin intervalo: {item.why} | {clusters} |"
    interval = pair.interval
    return (
        f"| {item.label} | {item.n} | {_pct(interval.mean)} | "
        f"{_bounds(interval.low, interval.high)} | {clusters} |"
    )


def _comparison_described(comparison: PairComparison) -> Described:
    pair = comparison.pair
    return Described(
        label=f"`{comparison.reference}`",
        n=comparison.n,
        mean=None if pair is None else pair.naive.mean,
        pair=pair,
    )


def _net_row(net: NetPairs | None) -> str:
    if net is None:
        return "| — | — | — | sin cifra: sin funding | — |"
    pair = net.pair
    if pair is None:
        return (
            f"| {net.n} | {net.dropped} | — | sin intervalo: menos de {MIN_PAIRED_N} parejas | — |"
        )
    if pair.interval is None:
        return (
            f"| {net.n} | {net.dropped} | {_pct(pair.naive.mean)} | sin intervalo: {pair.why} "
            f"| {pair.error.clusters} |"
        )
    interval = pair.interval
    return (
        f"| {net.n} | {net.dropped} | {_pct(interval.mean)} | "
        f"{_bounds(interval.low, interval.high)} | {pair.error.clusters} |"
    )


def _descriptive_lines(scored: Scored) -> list[str]:
    table = ["| | n | diferencia media | IC 95% | G |", "| --- | ---: | ---: | --- | ---: |"]
    return [
        "## Descriptivos",
        "",
        "Ninguno entra en el veredicto. Cada uno con su diferencia, su IC 95% de la enmienda 4 y "
        f"su n; con menos de 50 clústeres o menos de {MIN_PAIRED_N} parejas, solo la media y el n.",
        "",
        "### `solo` frente a las otras tres líneas base",
        "",
        *table,
        *(_described_row(_comparison_described(item)) for item in scored.descriptive),
        "",
        "### El neto de la comparación principal",
        "",
        "`outcomes.net_return` sobre los mismos registros, con los costes y el funding extendido; "
        "las parejas sin funding completo se cuentan y se quitan.",
        "",
        *scored.net_lines,
        "",
        "| parejas con funding | quitadas sin funding | diferencia media | IC 95% | G |",
        "| ---: | ---: | ---: | --- | ---: |",
        _net_row(scored.primary.net),
        "",
        "### La comparación principal con el stop que declara cada brazo",
        "",
        *table,
        _described_row(scored.own),
        "",
        "### La comparación principal por upstream de la llamada válida del decisor",
        "",
        f"Las faltantes van en «{NO_CALL}» y las respuestas inválidas en «{INVALID_RESPONSE}».",
        "",
        *table,
        *(_described_row(item) for item in scored.upstreams),
        "",
    ]


def _served_text(pairs: frozenset[tuple[str, str | None]]) -> str:
    return (
        ", ".join(
            f"`{model}` por `{endpoint}`" if endpoint is not None else f"`{model}`"
            for model, endpoint in sorted(pairs, key=str)
        )
        or "ninguno"
    )


def _missing_cell(block: BlockRead, now: datetime) -> str:
    """Las faltantes de un bloque con su desglose, o por qué todavía no son faltantes."""
    if block.evaluations == 0:
        return "—"
    complete = block.counts is not None and block.counts.complete
    if now < block.dates.deadline and not complete:
        return "— (plazo abierto)"
    causes = missing_by_cause(block)
    return f"{_missing(block)} ({', '.join(str(causes[cause]) for cause in MISSING_CAUSES)})"


def _counted(block: BlockRead, now: datetime) -> bool:
    """Si las faltantes del bloque ya cuentan: su plazo venció o su lote está completo."""
    if block.evaluations == 0:
        return False
    return now >= block.dates.deadline or (block.counts is not None and block.counts.complete)


def _counter_lines(
    blocks: Sequence[BlockRead], stage1_served: frozenset[tuple[str, str | None]], now: datetime
) -> list[str]:
    """Los contadores del apartado 6, que se publican con o sin veredicto. Ningún retorno.

    Las faltantes de un bloque cuentan cuando su plazo venció o su lote está completo. Antes, lo
    que falta es una reanudación pendiente, no una faltante.
    """
    lines = [
        "## Contadores",
        "",
        "| bloque | inicio | evaluaciones | pasadas | decisiones válidas | respuestas inválidas "
        f"| faltantes ({', '.join(MISSING_CAUSES)}) | plazo del lote | upstreams del decisor |",
        "| ---: | --- | ---: | --- | ---: | ---: | --- | --- | --- |",
    ]
    previous = stage1_served
    changes: list[str] = []
    actions: Counter[str] = Counter()
    for block in blocks:
        counts = block.counts
        solo = block.records.get(ARM, ())
        served = served_pairs(decider_rows(solo))
        decided = "— | —" if counts is None else f"{counts.decided} | {counts.invalid}"
        passes = str(len(block.passes))
        if block.superseded:
            passes += f" (+{len(block.superseded)} sustituidas)"
        lines.append(
            f"| {block.dates.number} | {block.dates.start:%Y-%m-%d} | {block.evaluations} | "
            f"{passes} | {decided} | {_missing_cell(block, now)} | "
            f"{block.dates.deadline:%Y-%m-%d} | {_served_text(served) if solo else '—'} |"
        )
        if served and served != previous:
            changes.append(
                f"bloque {block.dates.number}: antes {_served_text(previous)}, ahora "
                f"{_served_text(served)}"
            )
            previous = served
        actions.update(
            "sin decisión" if record.proposed is None else record.proposed.action.value
            for record in solo
        )
    filled = [block for block in blocks if block.evaluations]
    empty = [block.dates.number for block in blocks if not block.evaluations]
    evaluations = sum(block.evaluations for block in blocks)
    missing = sum(_missing(block) for block in blocks if _counted(block, now))
    invalid = sum(block.counts.invalid for block in blocks if block.counts is not None)
    share = f" ({missing / evaluations:.2%})" if evaluations else ""
    mix = ", ".join(f"{name} {count}" for name, count in sorted(actions.items())) or "ninguna"
    lines += [
        "",
        f"- bloques: {len(blocks)}, {len(filled)} no vacíos y {len(empty)} vacíos"
        + (f" (los bloques {', '.join(str(number) for number in empty)})" if empty else ""),
        f"- evaluaciones: {evaluations}",
        f"- faltantes: {missing} de {evaluations}{share}; el límite es el "
        f"{float(MISSING_LIMIT):.0%} (apartado 5). Solo cuentan los bloques con el plazo vencido o "
        "el lote completo",
        f"- respuestas inválidas de `solo`: {invalid}",
        f"- acciones de `solo`: {mix}",
        "- cambios de upstream del decisor, desde la etapa 1 "
        f"({_served_text(stage1_served)}): "
        + ("; ".join(changes) if changes else "ninguno")
        + ". Un bloque en que el decisor no respondió no cuenta como cambio: no dice quién sirve.",
        "",
    ]
    return lines


def _source_lines(blocks: Sequence[BlockRead], paths: StudyPaths) -> list[str]:
    """El sha-256 de cada archivo del que sale una cifra: manifiestos, journals y series."""
    lines = ["## Fuentes", ""]
    for block in blocks:
        manifest = manifest_path(block.dates.start, paths.manifests)
        if manifest.is_file():
            lines.append(
                f"- manifiesto del bloque {block.dates.number}: `{manifest}`, sha-256 "
                f"`{file_sha256(manifest)}`"
            )
        for directory in block.passes:
            for arm in FORWARD_ARMS:
                journal = arm_journal_path(directory, arm)
                if journal.is_file():
                    lines.append(f"- `{journal}` sha-256 `{file_sha256(journal)}`")
        for directory in block.superseded:
            mark = directory / SUPERSEDED_FILE
            lines.append(
                f"- pasada sustituida, fuera del lote: `{directory}`; su marca `{mark}` sha-256 "
                f"`{file_sha256(mark)}`"
            )
    for directory, name, path_of in (
        (paths.history, "histórico extendido", _history_file),
        (paths.funding, "funding extendido", funding_path),
    ):
        for symbol in sorted(DEFAULT_SYMBOLS):
            path = path_of(symbol, directory)
            if path.is_file():
                lines.append(f"- {name} `{symbol}`: `{path}`, sha-256 `{file_sha256(path)}`")
            else:
                lines.append(f"- {name} `{symbol}`: `{path}` no está")
    if paths.revisions.is_file():
        lines.append(f"- `{paths.revisions}` sha-256 `{file_sha256(paths.revisions)}`")
    return [*lines, ""]


def _history_file(symbol: str, directory: Path) -> Path:
    return history_path(symbol, DEFAULT_TIMEFRAME, directory)


def _undetermined(
    cause: str,
    blocks: Sequence[BlockRead],
    paths: StudyPaths,
    stage1_served: frozenset[tuple[str, str | None]],
    now: datetime,
) -> list[str]:
    """«No determinado» antes de puntuar: la causa, los contadores y las fuentes. Ningún retorno."""
    return [
        "",
        f"## Veredicto de la comparación principal: `{ARM}` - `{REFERENCE}`",
        "",
        f"**{VERDICT_LABEL[Verdict.UNDETERMINED]}**: {cause}.",
        "",
        "No se puntuó ninguna evaluación: de este estudio solo se publican sus contadores de "
        "validez y su número de evaluaciones (apartados 5 y 6).",
        "",
        *_counter_lines(blocks, stage1_served, now),
        *_source_lines(blocks, paths),
    ]


def _scored_report(
    scored: Scored,
    blocks: Sequence[BlockRead],
    paths: StudyPaths,
    stage1_served: frozenset[tuple[str, str | None]],
    now: datetime,
) -> list[str]:
    primary = scored.primary
    why = ""
    if scored.verdict is Verdict.UNDETERMINED:
        reason = f"menos de {MIN_PAIRED_N} parejas" if primary.pair is None else primary.pair.why
        why = f": {reason}"
    return [
        "",
        f"## Veredicto de la comparación principal: `{ARM}` - `{REFERENCE}`",
        "",
        f"**{VERDICT_LABEL[scored.verdict]}**{why}",
        "",
        "Retorno bruto por evaluación con el stop común y un horizonte de "
        f"{scored.horizon} velas de {DEFAULT_TIMEFRAME}, emparejado por `run_id` "
        "(`criteria.paired_returns`). Las faltantes y las respuestas inválidas de `solo` valen 0.",
        "",
        *_primary_table(primary),
        "",
        *_rule_lines(),
        "",
        *_descriptive_lines(scored),
        *_counter_lines(blocks, stage1_served, now),
        *_source_lines(blocks, paths),
    ]


# ───────────────────────────────────────── El ensayo sobre la etapa 1 ─────────────────────────────

FIGURES = (
    "n",
    "diferencia media",
    "EE ingenuo",
    "EE agrupado",
    "EE usado",
    "G",
    "deff",
    "IC 95%",
    "semiancho",
    "neto: n",
    "neto: diferencia media [IC 95%]",
)
"""Las cifras de una fila de `criteria`, en su orden y sin la columna del veredicto."""

PUBLISHED_STAGE1: Mapping[str, tuple[str, ...]] = {
    "rule_trend": (
        "140", "-0.046%", "0.186%", "0.193%", "0.193%", "71", "1.076",
        "[-0.424%, +0.333%]", "0.378%", "140", "-0.106% [-0.488%, +0.275%]",
    ),
    "always_buy": (
        "140", "-0.449%", "0.345%", "0.278%", "0.345%", "71", "0.649",
        "[-1.126%, +0.228%]", "0.677%", "140", "-0.395% [-1.071%, +0.282%]",
    ),
    "always_sell": (
        "140", "+0.284%", "0.375%", "0.351%", "0.375%", "71", "0.877",
        "[-0.451%, +1.020%]", "0.735%", "140", "+0.322% [-0.413%, +1.057%]",
    ),
    "random_uniform": (
        "140", "-0.061%", "0.364%", "0.353%", "0.364%", "71", "0.939",
        "[-0.775%, +0.654%]", "0.714%", "140", "-0.058% [-0.771%, +0.655%]",
    ),
}  # fmt: skip
"""Las cuatro filas `solo` contra cada línea base que publicó `docs/ablation.md`, «La etapa 1
releída con el intervalo de la enmienda 4», copiadas tal como se imprimieron. La primera es la
comparación principal del estudio."""

PUBLISHED_SOURCE = "`docs/ablation.md`, «La etapa 1 releída con el intervalo de la enmienda 4»"

_MISSING = "sin cifra"


def stage1_figures(comparison: PairComparison) -> tuple[str, ...]:
    """Las cifras de una fila con el formato de `criteria`, para compararlas con lo publicado."""
    pair, net = comparison.pair, comparison.net
    if pair is None:
        return (str(comparison.n), *(_MISSING for _ in FIGURES[1:]))
    error, interval = pair.error, pair.interval
    net_n = _MISSING
    net_diff = _MISSING
    if net is not None:
        net_n = f"{net.n}" + (f" (quitadas sin funding: {net.dropped})" if net.dropped else "")
        if net.diff is not None:
            net_diff = f"{_pct(net.diff.mean)} {_bounds(net.diff.low, net.diff.high)}"
    return (
        str(comparison.n),
        _pct(pair.naive.mean),
        _error(error.naive),
        _error(error.clustered),
        _error(None if interval is None else interval.stderr),
        str(error.clusters),
        _deff(error.deff),
        _MISSING if interval is None else _bounds(interval.low, interval.high),
        _MISSING if interval is None else f"{half_width(interval):.{PERCENT_DIGITS}%}",
        net_n,
        net_diff,
    )


def rehearse_analysis(
    git: tuple[str | None, bool | None],
    stage1: Path = STAGE1_DIR,
    history: Path = HISTORY_DIR,
    funding: Path = FUNDING_DIR,
) -> AnalysisResult:
    """El ensayo del análisis sobre la etapa 1: las cuatro filas contra las publicadas.

    Lee solo `data/stage1/`, después de comprobar sus sha-256, y no tiene argumento con el que
    apuntar a otro sitio. Arma cada fila con `criteria.paired_returns` y `compare_aligned`, las
    mismas funciones del análisis, con el neto de `data/funding/`. No aplica ninguna regla de
    veredicto: compara cifras. Sale con 0 si todas coinciden y con 1 si alguna no.
    """
    try:
        verify_stage1(stage1)
        stage = read_stage(stage1, None, history, None)
    except (
        AuditError,
        CriteriaError,
        ForwardDataError,
        RecalculationError,
        SelectionError,
    ) as error:
        return AnalysisResult(1, (f"error: {error}",))
    symbols = sorted({record.symbol for records in stage.arms.values() for record in records})
    net, net_lines = _net_inputs(symbols, funding)
    rows: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
    for reference, published in PUBLISHED_STAGE1.items():
        aligned = paired_returns(ARM, reference, stage.arms, stage.histories, stage.horizon)
        ours = (
            (_MISSING,) * len(FIGURES)
            if isinstance(aligned, str)
            else stage1_figures(
                compare_aligned(ARM, reference, aligned, stage.histories, stage.horizon, net)
            )
        )
        rows.append((reference, ours, published))
    total = sum(len(published) for _, _, published in rows)
    matching = sum(
        1 for _, ours, published in rows for a, b in zip(ours, published, strict=True) if a == b
    )
    lines = [
        "# Ensayo del análisis sobre la etapa 1",
        "",
        f"Reproduce: `{_COMMAND} rehearse-analysis`",
        "",
        f"- commit del ensayo: {commit_label(*git)}",
        f"- etapa 1: `{stage1}`, con los sha-256 publicados:",
        *(f"  - `{name}` `{digest}`" for name, digest in STAGE1_FILES.items()),
        *(
            f"- histórico `{symbol}`: `{path}`, sha-256 `{digest}`"
            for symbol, path, digest in stage.history_files
        ),
        *(
            f"- funding `{symbol}`: `{funding_path(symbol, funding)}`, sha-256 "
            f"`{file_sha256(funding_path(symbol, funding))}`"
            for symbol in symbols
            if funding_path(symbol, funding).is_file()
        ),
        *net_lines,
        f"- horizonte {stage.horizon} velas, stop común, emparejado por `run_id` "
        "(`criteria.paired_returns`); intervalo de `metrics.clustered_paired_difference` con un "
        f"clúster por bloque de {CLUSTER_DAYS} días del instante evaluado",
        f"- comparado con {PUBLISHED_SOURCE}. Solo se comparan cifras: el ensayo no juzga.",
        "",
        "| brazo | referencia | cifra | ensayo | publicada | coincide |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for reference, ours, published in rows:
        lines += [
            f"| {ARM} | {reference} | {name} | {a} | {b} | {'sí' if a == b else '**NO**'} |"
            for name, a, b in zip(FIGURES, ours, published, strict=True)
        ]
    lines += ["", f"- coinciden {matching} de {total} cifras"]
    return AnalysisResult(0 if matching == total else 1, tuple(lines))
