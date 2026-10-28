"""Los lotes que corre el temporizador: lo que decide `python -m crypto_agents.forward auto`.

Desde el bloque 2 (`AUTO_FIRST_BLOCK`), un temporizador diario lanza `auto`, que corre los lotes
pendientes con el mismo código que `batch --block`. Las decisiones que `docs/forward.md` reserva al
operador siguen siendo suyas, y `auto` no tiene por dónde tomarlas:

- aceptar una vela revisada (apartado 4);
- repetir entero un lote inválido (apartado 5);
- completar las líneas base después del plazo;
- el análisis (apartado 6).

Aquí vive lo que se decide sin red y sin modelos, sobre lo que el estudio tiene en disco y con el
reloj inyectado:

- qué bloques están pendientes (`study_state`);
- qué se hace con el código de salida de cada lote (`ACTIONS`);
- qué hay que avisar al operador.

El lote mismo lo recibe `run_auto` como un ejecutor, que `forward.batch_runner` ata a
`forward.run_batch`: este módulo no puede importar `forward`, que trae el router.

Lee el estudio con las funciones del análisis (`termination`, `pass_metas`, `read_block`) sin
copiarlas ni tocarlas: el código del análisis está congelado (registro, 2026-10-19). No llama al
análisis. Dos tests de arquitectura lo fijan: uno mira lo que nombra y otro lo importa en un
proceso nuevo sin el router.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, NamedTuple, Protocol

from pydantic import AwareDatetime, Field

from crypto_agents.audit import read_meta
from crypto_agents.forward_analysis import StudyPaths, pass_metas, read_block, termination
from crypto_agents.forward_data import manifest_path
from crypto_agents.forward_plan import MAX_BLOCKS, closing_block
from crypto_agents.forward_record import (
    FIRST_BLOCK,
    STEP,
    BatchOutcome,
    BlockDates,
    DuplicateRecordError,
    batch_directory,
    batch_passes,
    block_dates,
    registered_decider,
)
from crypto_agents.selection import load_manifest
from crypto_agents.state import FrozenModel

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence
    from pathlib import Path

    from crypto_agents.forward_record import BatchCounts, BatchMeta, DeciderConfig

__all__ = [
    "ACTIONS",
    "AUTO_FIRST_BLOCK",
    "UPSTREAM_FILE",
    "WARN_BEFORE",
    "Action",
    "AutoReport",
    "BlockReading",
    "BlockRun",
    "BlockStatus",
    "StudyState",
    "action_for",
    "block_status",
    "render_report",
    "render_state",
    "run_auto",
    "study_state",
]

AUTO_FIRST_BLOCK = FIRST_BLOCK + STEP
"""El primer bloque que corre el temporizador: el 2, que empieza el 2026-10-29. El del bloque 1 lo
corre el operador, y `auto` no lo toca ni para reanudarlo ni para avisar de su plazo."""

WARN_BEFORE = timedelta(days=3)
"""Un bloque pendiente al que le queda esto o menos de plazo se avisa siempre."""

UPSTREAM_FILE = "upstream.md"
"""El asiento preparado para el registro cuando cambia el upstream del decisor. No se escribe solo
en `docs/forward.md`."""


# ──────────────────────────────────────── El estado de un bloque ──────────────────────────────────


class BlockStatus(StrEnum):
    """Lo que el estudio tiene en disco de un bloque."""

    NOT_PREPARED = "sin preparar"
    """Ni manifiesto ni registro de bloque vacío: su lote no llegó a preparar el bloque."""

    INCOMPLETE = "incompleto"
    INVALID = "inválido"
    """Lo invalida algo de sus registros o de sus pasadas: se repite entero, a mano."""

    DUPLICATE = "con duplicados"
    """Un `run_id` completo en dos pasadas no sustituidas: el lote se detiene."""

    EMPTY = "vacío"
    COMPLETE = "completo"


class BlockReading(NamedTuple):
    """Un bloque leído del disco, con los contadores de su lote si corrió alguna pasada."""

    dates: BlockDates
    status: BlockStatus
    evaluations: int
    counts: BatchCounts | None


def block_status(
    paths: StudyPaths,
    start: datetime,
    registered: Callable[[], DeciderConfig],
    metas: Sequence[tuple[Path, BatchMeta]],
) -> BlockReading:
    """El estado del bloque que empieza en `start`, leído como lo lee el análisis.

    Con manifiesto, el lote se junta con `read_block`, que cuenta sus pasadas no sustituidas como
    el lote y el análisis. Sin manifiesto, el bloque está vacío si alguna pasada lo registró así,
    y sin preparar si no. `registered` da la configuración registrada del decisor; solo se pide
    cuando hay un lote que leer.
    """
    dates = block_dates(start)
    manifest = manifest_path(start, paths.manifests)
    if not manifest.is_file():
        empty = any(
            meta.outcome is BatchOutcome.EMPTY and meta.block.start == start for _, meta in metas
        )
        status = BlockStatus.EMPTY if empty else BlockStatus.NOT_PREPARED
        return BlockReading(dates, status, 0, None)
    evaluations = len(load_manifest(manifest).entries)
    try:
        block = read_block(paths, start, evaluations, registered())
    except DuplicateRecordError:
        return BlockReading(dates, BlockStatus.DUPLICATE, evaluations, None)
    counts = block.counts
    if counts is None:
        status = BlockStatus.INCOMPLETE
    elif counts.complete:
        status = BlockStatus.COMPLETE
    elif counts.invalidating:
        status = BlockStatus.INVALID
    else:
        status = BlockStatus.INCOMPLETE
    return BlockReading(dates, status, evaluations, counts)


# ─────────────────────────────────────── Los bloques pendientes ───────────────────────────────────


RUNNABLE = frozenset({BlockStatus.NOT_PREPARED, BlockStatus.INCOMPLETE})
"""Los estados en que el temporizador corre el lote."""

BLOCKED = frozenset({BlockStatus.INVALID, BlockStatus.DUPLICATE})
"""Los estados que necesitan al operador: no se corren, se avisan."""


class StudyState(NamedTuple):
    """Lo que el temporizador haría ahora, leído del disco sin red."""

    now: datetime
    pending: tuple[BlockReading, ...]
    """Los bloques cuyo lote se corre, en orden cronológico."""

    blocked: tuple[BlockReading, ...]
    """Los bloques dentro de su plazo que no se corren: inválidos o con duplicados."""

    terminated: str | None
    """Dónde y por qué terminó el estudio, si terminó: entonces no hay nada pendiente."""

    closed_at: int | None
    """El bloque en que cerró el estudio, si ya cerró: los posteriores no son del estudio."""

    upcoming: BlockDates | None
    """El primer bloque del temporizador cuyo lote todavía no se abre."""


def study_state(paths: StudyPaths, stage1: Path, now: datetime) -> StudyState:
    """Los bloques pendientes en `now`, en orden cronológico, y los bloqueados.

    Un bloque está pendiente si cumple todo esto:

    - es del temporizador: desde `AUTO_FIRST_BLOCK` y como mucho el bloque 52;
    - ya pasó su cierre más el horizonte y no venció su plazo, con los bordes de `date_refusal`:
      desde `ready` exacto, y en `deadline` exacto ya no;
    - su lote no está completo: está sin preparar o incompleto;
    - el estudio no terminó (`termination`, la regla que lee el análisis) ni cerró antes de él.

    El cierre se calcula con `closing_block` sobre los bloques anteriores. Un bloque con manifiesto
    no está vacío, porque un bloque vacío no lo tiene; uno sin manifiesto cuenta como vacío. Un
    bloque anterior que nunca se preparó cuenta así como vacío aunque no lo fuera. Si por eso el
    estudio parece abierto y ya cerró, el lote se niega con 2 (`StudyClosedError`) y se avisa.
    """
    metas = pass_metas(paths.batches)
    ended = termination(metas)
    if ended is not None:
        terminated = f"en el bloque {ended.block}: {ended.cause} ({ended.where})"
        return StudyState(now, (), (), terminated, None, None)

    decider: list[DeciderConfig] = []

    def registered() -> DeciderConfig:
        if not decider:
            decider.append(registered_decider(read_meta(stage1)))
        return decider[0]

    filled: list[int] = []
    pending: list[BlockReading] = []
    blocked: list[BlockReading] = []
    for position in range(MAX_BLOCKS):
        start = FIRST_BLOCK + position * STEP
        try:
            closed = closing_block(filled)
        except ValueError:
            closed = None
        if closed is not None:
            return StudyState(now, tuple(pending), tuple(blocked), None, closed, None)
        filled.append(1 if manifest_path(start, paths.manifests).is_file() else 0)
        if start < AUTO_FIRST_BLOCK:
            continue
        dates = block_dates(start)
        if now < dates.ready:
            return StudyState(now, tuple(pending), tuple(blocked), None, None, dates)
        if now >= dates.deadline:
            continue
        reading = block_status(paths, start, registered, metas)
        if reading.status in RUNNABLE:
            pending.append(reading)
        elif reading.status in BLOCKED:
            blocked.append(reading)
    return StudyState(now, tuple(pending), tuple(blocked), None, None, None)


def near_deadline(dates: BlockDates, now: datetime) -> bool:
    """Si al lote le quedan `WARN_BEFORE` o menos de plazo."""
    return dates.deadline - now <= WARN_BEFORE


# ─────────────────────────────────────── Los códigos de salida ────────────────────────────────────


class Action(StrEnum):
    """Lo que hace el temporizador con el código de salida de un lote."""

    NEXT = "completo: se pasa al siguiente"
    RETRY = "se deja para la ejecución siguiente"
    STOP = "se detiene todo y se avisa"
    ENDS = "el estudio termina: se detiene todo y se avisa"
    FAILURE = "fallo: se avisa"


ACTIONS: Mapping[int, Action] = {
    0: Action.NEXT,
    3: Action.RETRY,
    6: Action.RETRY,
    1: Action.STOP,
    5: Action.STOP,
    4: Action.ENDS,
    7: Action.ENDS,
    2: Action.FAILURE,
}
"""Los códigos de `batch` (ver `forward.main`) y lo que hace el temporizador con cada uno:

- 0, el lote está completo, o el bloque vacío;
- 3, quedó incompleto, y 6, falta una vela nueva: los dos se reintentan al día siguiente;
- 1, una negativa sin llamar a nadie, y 5, una vela revisada: los dos necesitan al operador;
- 4, el decisor salió del catálogo, y 7, un hueco persistente: terminan el estudio;
- 2, la fecha o el cierre: con los mismos bordes que el lote no debería pasar, y es un fallo.
"""


def action_for(code: int) -> Action:
    """La acción de un código de salida. Un código que la tabla no conoce es un fallo."""
    return ACTIONS.get(code, Action.FAILURE)


# ─────────────────────────────────────────── La corrida ───────────────────────────────────────────


class RunOutcome(Protocol):
    """Lo que devuelve el lote de un bloque: `forward.BatchResult`."""

    @property
    def code(self) -> int: ...

    @property
    def lines(self) -> tuple[str, ...]: ...

    @property
    def files(self) -> tuple[str, ...]: ...


class BlockRun(FrozenModel):
    """Un lote que corrió el temporizador: su código, lo que se hizo con él y sus contadores."""

    start: AwareDatetime
    number: int = Field(ge=1)
    code: int
    action: Action
    headline: str
    """La primera línea de lo que imprimió el lote."""

    files: tuple[str, ...] = ()
    """Lo que el lote listó para versionar: lo único que la envoltura añade al commit."""

    status: BlockStatus
    """El estado del bloque releído del disco después del lote."""

    evaluations: int = Field(ge=0)
    decided: int | None = None
    invalid: int | None = None
    missing: int | None = None
    upstream_entries: tuple[str, ...] = ()
    """Los `upstream.md` que dejaron las pasadas de esta corrida: el upstream del decisor cambió."""


class AutoReport(FrozenModel):
    """Lo que hizo `auto`: lo lee la envoltura para versionar y avisar. Ningún retorno."""

    at: AwareDatetime
    runs: tuple[BlockRun, ...] = ()
    alerts: tuple[str, ...] = ()
    """Lo que necesita al operador, una línea por caso."""

    retry: bool = False
    """Si algún lote quedó para la ejecución siguiente."""

    terminated: str | None = None
    closed_at: int | None = None

    @property
    def needs_operator(self) -> bool:
        return bool(self.alerts)


def _passes(paths: StudyPaths, start: datetime) -> set[Path]:
    return set(batch_passes(batch_directory(paths.batches, start), include_superseded=True))


def _headline(lines: Sequence[str]) -> str:
    return next((line.strip() for line in lines if line.strip()), "sin salida")


async def run_auto(
    paths: StudyPaths,
    stage1: Path,
    clock: Callable[[], datetime],
    runner: Callable[[datetime], Awaitable[RunOutcome]],
    echo: Callable[[Sequence[str]], None] = lambda lines: None,
) -> AutoReport:
    """Corre los lotes pendientes en orden y aplica `ACTIONS` a cada código.

    `runner` es el lote de un bloque y recibe solo su inicio: `forward.batch_runner` lo ata a
    `run_batch` sin velas aceptadas ni sustitución. STOP y ENDS cortan los bloques que quedan;
    FAILURE avisa y sigue con el siguiente. Tras cada lote se relee el bloque del disco:

    - un lote que quedó incompleto (3) y resulta inválido se trata como STOP, porque repetirlo
      entero es del operador;
    - una pasada nueva que dejó `upstream.md` se avisa: el asiento va al registro a mano.

    Además se avisa de cada bloque bloqueado (inválido o con duplicados), que no se corre, y de
    cada bloque al que le quedan `WARN_BEFORE` o menos de plazo y no queda completo. Sin nada
    pendiente no llama a `runner` ni escribe nada.
    """
    now = clock()
    state = study_state(paths, stage1, now)
    alerts = [_blocked_alert(reading) for reading in state.blocked]

    def reread(start: datetime) -> BlockReading:
        return block_status(
            paths, start, lambda: registered_decider(read_meta(stage1)), pass_metas(paths.batches)
        )

    runs: list[BlockRun] = []
    unfinished: list[BlockReading] = list(state.blocked)
    retry = False
    for position, reading in enumerate(state.pending):
        dates = reading.dates
        before = _passes(paths, dates.start)
        result = await runner(dates.start)
        echo(result.lines)
        action = action_for(result.code)
        after = reread(dates.start)
        new = sorted(_passes(paths, dates.start) - before)
        entries = tuple(
            str(path / UPSTREAM_FILE) for path in new if (path / UPSTREAM_FILE).is_file()
        )
        headline = _headline(result.lines)
        label = f"bloque {dates.number} ({dates.start:%Y-%m-%d})"
        if action is Action.RETRY and after.status is BlockStatus.INVALID:
            action = Action.STOP
            alerts.append(
                f"{label}: el lote salió con {result.code} y es inválido. Repetirlo entero es del "
                "operador (apartado 5)."
            )
        elif action in (Action.STOP, Action.ENDS):
            alerts.append(f"{label}: el lote salió con {result.code}: {headline}")
        elif action is Action.FAILURE:
            alerts.append(f"{label}: fallo, el lote salió con {result.code}: {headline}")
        alerts.extend(
            f"{label}: cambió el upstream del decisor; el asiento preparado está en {entry} y va "
            "al registro a mano."
            for entry in entries
        )
        counts = after.counts
        runs.append(
            BlockRun(
                start=dates.start,
                number=dates.number,
                code=result.code,
                action=action,
                headline=headline,
                files=tuple(result.files),
                status=after.status,
                evaluations=after.evaluations,
                decided=None if counts is None else counts.decided,
                invalid=None if counts is None else counts.invalid,
                missing=None if counts is None else counts.missing,
                upstream_entries=entries,
            )
        )
        if after.status not in (BlockStatus.COMPLETE, BlockStatus.EMPTY):
            unfinished.append(after)
        retry = retry or action is Action.RETRY
        if action in (Action.STOP, Action.ENDS):
            unfinished.extend(state.pending[position + 1 :])
            break

    alerts.extend(
        f"bloque {reading.dates.number} ({reading.dates.start:%Y-%m-%d}): su lote no está "
        f"completo y le quedan {_left(reading.dates, now)} de plazo; vence el "
        f"{reading.dates.deadline.isoformat()}."
        for reading in unfinished
        if near_deadline(reading.dates, now)
    )
    return AutoReport(
        at=now,
        runs=tuple(runs),
        alerts=tuple(alerts),
        retry=retry,
        terminated=state.terminated,
        closed_at=state.closed_at,
    )


def _blocked_alert(reading: BlockReading) -> str:
    """El aviso de un bloque que no se corre."""
    dates = reading.dates
    label = f"bloque {dates.number} ({dates.start:%Y-%m-%d})"
    if reading.status is BlockStatus.DUPLICATE:
        return (
            f"{label}: un run_id tiene registro completo en dos pasadas. No se corre: resolverlo "
            "es del operador."
        )
    reasons = "; ".join(reading.counts.invalidating) if reading.counts is not None else ""
    return (
        f"{label}: el lote es inválido ({reasons}). No se corre: repetirlo entero es del operador "
        "(apartado 5)."
    )


def _left(dates: BlockDates, now: datetime) -> str:
    hours = (dates.deadline - now) / timedelta(hours=1)
    return f"{hours:.1f} h"


# ───────────────────────────────────────────── Salida ─────────────────────────────────────────────


def _block(dates: BlockDates) -> str:
    return f"el lote del bloque {dates.number} ({dates.start:%Y-%m-%d})"


def render_state(state: StudyState) -> list[str]:
    """Lo que `auto --dry-run` dice que haría ahora, sin correr ni escribir nada."""
    lines = [
        f"## Temporizador del estudio, a {state.now.isoformat(timespec='seconds')}, sin correr "
        "nada",
        "",
        f"- corre desde el bloque {block_dates(AUTO_FIRST_BLOCK).number} "
        f"({AUTO_FIRST_BLOCK:%Y-%m-%d}); el lote del bloque 1 lo corre el operador.",
    ]
    if state.terminated is not None:
        return [*lines, f"- el estudio terminó {state.terminated}: no hay nada que correr."]
    if state.closed_at is not None:
        lines.append(f"- el estudio cerró en el bloque {state.closed_at}.")
    for reading in state.pending:
        dates = reading.dates
        lines.append(
            f"- correría {_block(dates)}, {reading.status.value}; vence el "
            f"{dates.deadline.isoformat()}, quedan {_left(dates, state.now)}"
            + (" (se avisaría)." if near_deadline(dates, state.now) else ".")
        )
    for reading in state.blocked:
        lines.append(
            f"- no correría {_block(reading.dates)}: está {reading.status.value}, y se avisaría "
            "al operador."
        )
    if not state.pending:
        lines.append("- no hay ningún lote pendiente.")
    if state.upcoming is not None:
        lines.append(
            f"- el siguiente: {_block(state.upcoming)} se abre el "
            f"{state.upcoming.ready.isoformat()} y vence el {state.upcoming.deadline.isoformat()}."
        )
    return lines


def render_report(report: AutoReport) -> list[str]:
    """El resumen de `auto` tras correr: un renglón por lote y los avisos. Solo contadores."""
    lines = [f"## Temporizador del estudio, {report.at.isoformat(timespec='seconds')}", ""]
    if report.terminated is not None:
        lines.append(f"- el estudio terminó {report.terminated}: no hay nada que correr.")
    if report.closed_at is not None:
        lines.append(f"- el estudio cerró en el bloque {report.closed_at}.")
    if not report.runs:
        lines.append("- no corrió ningún lote.")
    for run in report.runs:
        lines.append(
            f"- bloque {run.number} ({run.start:%Y-%m-%d}): código {run.code}, "
            f"{run.action.value}; {run.status.value}, {run.evaluations} evaluaciones"
        )
    if report.alerts:
        lines += ["", "Avisos para el operador:", *(f"- {alert}" for alert in report.alerts)]
    return lines
