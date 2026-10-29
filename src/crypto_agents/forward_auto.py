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
- qué bloques esperan al operador y no se corren: inválidos, con duplicados, o retenidos porque su
  última pasada se negó o se detuvo por una vela revisada;
- qué bloques vencieron con el lote incompleto y sus líneas base sin completar;
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
from crypto_agents.criteria import BASELINE_ARMS
from crypto_agents.design_effect import pool_run_id
from crypto_agents.forward_analysis import StudyPaths, pass_metas, read_block, termination
from crypto_agents.forward_data import manifest_path
from crypto_agents.forward_plan import MAX_BLOCKS, TARGET_BLOCKS, closing_block
from crypto_agents.forward_record import (
    FIRST_BLOCK,
    STEP,
    BatchOutcome,
    BlockDates,
    DuplicateRecordError,
    ForwardError,
    batch_directory,
    batch_passes,
    batch_state,
    block_dates,
    registered_decider,
    superseded,
)
from crypto_agents.market import timeframe_to_timedelta
from crypto_agents.selection import load_manifest
from crypto_agents.state import FrozenModel

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence
    from pathlib import Path
    from uuid import UUID

    from crypto_agents.forward_record import BatchCounts, BatchMeta, DeciderConfig
    from crypto_agents.selection import PlannedEvaluation

__all__ = [
    "ACTIONS",
    "AUTO_FIRST_BLOCK",
    "HELD_OUTCOMES",
    "UPSTREAM_FILE",
    "WARN_BEFORE",
    "Action",
    "AutoReport",
    "BlockReading",
    "BlockRun",
    "BlockStatus",
    "StudyState",
    "action_for",
    "baselines_complete",
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
"""El asiento que deja una pasada cuando cambia el upstream del decisor. Es el mismo nombre que
escribe `forward.py`, que lo define aparte: este módulo no puede importarlo. Un test exige que los
dos valores sean iguales."""

HELD_OUTCOMES = frozenset({BatchOutcome.REVISED, BatchOutcome.REFUSED})
"""Los resultados de pasada que retienen un bloque si son los de su última pasada no sustituida."""


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

    HELD = "retenido"
    """Su última pasada no sustituida se negó o se detuvo por una vela revisada: no se corre hasta
    que haya una pasada posterior, y esa la corre el operador."""

    EMPTY = "vacío"
    COMPLETE = "completo"


class BlockReading(NamedTuple):
    """Un bloque leído del disco, con los contadores de su lote si corrió alguna pasada."""

    dates: BlockDates
    status: BlockStatus
    evaluations: int
    counts: BatchCounts | None
    held_by: str | None = None
    """La pasada que retiene el bloque y su resultado, si está retenido."""


def latest_pass(
    metas: Sequence[tuple[Path, BatchMeta]], start: datetime
) -> tuple[Path, BatchMeta] | None:
    """La última pasada no sustituida del bloque que empieza en `start`, haya corrido o no.

    Las pasadas se llaman por el instante UTC en que empezaron, así que la última es la de mayor
    nombre. Una pasada que se negó o que se detuvo por las velas no corre y no tiene `meta.json`,
    así que se leen los `forward.json` de todas (`pass_metas`), no solo las que corrieron.
    """
    found = [
        (path, meta)
        for path, meta in metas
        if meta.block.start == start and superseded(path) is None
    ]
    return max(found, key=lambda item: item[0].name, default=None)


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

    Un bloque que se correría (sin preparar o incompleto) queda retenido si su última pasada no
    sustituida es una negativa o una vela revisada (`HELD_OUTCOMES`). Un código 1 que no dejó
    pasada, como un error de red al descargar, no retiene nada: el bloque se reintenta.
    """
    dates = block_dates(start)
    manifest = manifest_path(start, paths.manifests)
    if not manifest.is_file():
        empty = any(
            meta.outcome is BatchOutcome.EMPTY and meta.block.start == start for _, meta in metas
        )
        status = BlockStatus.EMPTY if empty else BlockStatus.NOT_PREPARED
        return _held(BlockReading(dates, status, 0, None), metas)
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
    return _held(BlockReading(dates, status, evaluations, counts), metas)


def _held(reading: BlockReading, metas: Sequence[tuple[Path, BatchMeta]]) -> BlockReading:
    """El bloque retenido si se correría y su última pasada no sustituida lo retiene."""
    if reading.status not in RUNNABLE:
        return reading
    latest = latest_pass(metas, reading.dates.start)
    if latest is not None and latest[1].outcome in HELD_OUTCOMES:
        path, meta = latest
        return reading._replace(
            status=BlockStatus.HELD, held_by=f"{path.name}, {meta.outcome.value}"
        )
    return reading


def _entry_run_id(entry: PlannedEvaluation) -> UUID:
    """El `run_id` que el replay da a una entrada: el de su instante evaluado, como en el
    análisis."""
    return pool_run_id(
        entry.symbol, entry.timeframe, entry.at + timeframe_to_timedelta(entry.timeframe)
    )


def baselines_complete(paths: StudyPaths, start: datetime) -> bool:
    """Si las cuatro líneas base del bloque tienen registro completo de cada `run_id` de su
    manifiesto, en las pasadas no sustituidas: la regla con la que `forward baselines` da el
    bloque por completo.

    Sin manifiesto no lo están: un bloque sin activaciones queda registrado como vacío, también
    cuando se completan sus líneas base, y entonces no llega aquí. Con duplicados el disco no
    permite saberlo, y tampoco lo están.
    """
    manifest = manifest_path(start, paths.manifests)
    if not manifest.is_file():
        return False
    expected = {_entry_run_id(entry) for entry in load_manifest(manifest).entries}
    try:
        state = batch_state(batch_passes(batch_directory(paths.batches, start)))
    except DuplicateRecordError:
        return False
    return all(expected <= state.done.get(arm, set()) for arm in BASELINE_ARMS)


# ─────────────────────────────────────── Los bloques pendientes ───────────────────────────────────


RUNNABLE = frozenset({BlockStatus.NOT_PREPARED, BlockStatus.INCOMPLETE})
"""Los estados en que el temporizador corre el lote."""

BLOCKED = frozenset({BlockStatus.INVALID, BlockStatus.DUPLICATE, BlockStatus.HELD})
"""Los estados que necesitan al operador: no se corren, se avisan."""

DONE = frozenset({BlockStatus.COMPLETE, BlockStatus.EMPTY})
"""Los estados en que al bloque no le falta nada."""


class StudyState(NamedTuple):
    """Lo que el temporizador haría ahora, leído del disco sin red."""

    now: datetime
    pending: tuple[BlockReading, ...]
    """Los bloques cuyo lote se corre, en orden cronológico."""

    blocked: tuple[BlockReading, ...]
    """Los bloques dentro de su plazo que no se corren: inválidos, con duplicados o retenidos."""

    expired: tuple[BlockReading, ...]
    """Los bloques con el plazo vencido, el lote incompleto y sus líneas base sin completar."""

    terminated: str | None
    """Dónde y por qué terminó el estudio, si terminó: entonces no hay nada pendiente."""

    closed_at: int | None
    """El bloque en que cerró el estudio, si ya cerró: los posteriores no son del estudio."""

    upcoming: BlockDates | None
    """El primer bloque del temporizador cuyo lote todavía no se abre."""

    non_empty: int
    """Los bloques no vacíos que lleva el estudio, contados por su manifiesto: hasta el de cierre
    si cerró."""

    exhausted: bool
    """Si el bloque 52 terminó sin 50 bloques no vacíos: el estudio ya no cierra (apartado 2)."""


def study_state(
    paths: StudyPaths, stage1: Path, now: datetime, *, first: datetime = AUTO_FIRST_BLOCK
) -> StudyState:
    """Los bloques pendientes en `now`, en orden cronológico, y los bloqueados.

    Un bloque está pendiente si cumple todo esto:

    - es del temporizador: desde `first` y como mucho el bloque 52. El temporizador no lo pasa y
      empieza en `AUTO_FIRST_BLOCK`; la comprobación semanal (`forward_watch`) pasa `FIRST_BLOCK`
      para mirar también el bloque 1;
    - ya pasó su cierre más el horizonte y no venció su plazo, con los bordes de `date_refusal`:
      desde `ready` exacto, y en `deadline` exacto ya no;
    - su lote no está completo: está sin preparar o incompleto, y no está retenido;
    - el estudio no terminó (`termination`, la regla que lee el análisis) ni cerró antes de él.

    Un bloque del temporizador con el plazo vencido no se corre. Si su lote no quedó completo ni
    vacío y sus líneas base no están completas (`baselines_complete`), va a `expired`: se avisa
    cada día hasta que lo estén.

    El cierre se calcula con `closing_block` sobre los bloques anteriores a cada uno y, después del
    último, sobre los 52. Un bloque con manifiesto no está vacío, porque un bloque vacío no lo
    tiene; uno sin manifiesto cuenta como vacío. Un bloque anterior que nunca se preparó cuenta así
    como vacío aunque no lo fuera. Si por eso el estudio parece abierto y ya cerró, el lote se niega
    con 2 (`StudyClosedError`) y se avisa.

    Antes de cada bloque, `closing_block` dice con un `ValueError` que el estudio sigue abierto, y
    daría el mismo ante una lista más corta de la que toca. Por eso se comprueba antes que la lista
    tenga un elemento por cada bloque anterior: si no, es un `ForwardError`, que no se lee como
    abierto, y `auto` lo deja en su informe y `forward_watch` falla con él.

    El bloque 52 entra en el cierre cuando ya se sabe si está vacío: tiene manifiesto, quedó
    registrado vacío o venció su plazo. Mientras su lote se puede correr sin manifiesto, contarlo
    vacío daría por terminado sin 50 no vacíos un estudio que todavía puede cerrar en él.
    """
    metas = pass_metas(paths.batches)
    ended = termination(metas)
    if ended is not None:
        terminated = f"en el bloque {ended.block}: {ended.cause} ({ended.where})"
        return StudyState(now, (), (), (), terminated, None, None, _manifests(paths), False)

    decider: list[DeciderConfig] = []

    def registered() -> DeciderConfig:
        if not decider:
            decider.append(registered_decider(read_meta(stage1)))
        return decider[0]

    filled: list[int] = []
    pending: list[BlockReading] = []
    blocked: list[BlockReading] = []
    expired: list[BlockReading] = []

    def found(
        closed: int | None, upcoming: BlockDates | None, exhausted: bool = False
    ) -> StudyState:
        return StudyState(
            now,
            tuple(pending),
            tuple(blocked),
            tuple(expired),
            None,
            closed,
            upcoming,
            sum(filled),
            exhausted,
        )

    settled = False
    for position in range(MAX_BLOCKS):
        start = FIRST_BLOCK + position * STEP
        if len(filled) != position:
            raise ForwardError(
                f"la lista del cierre mide {len(filled)} antes del bloque {position + 1}, y debe "
                f"medir {position}: un error al contar, no un estudio abierto"
            )
        try:
            closed = closing_block(filled)
        except ValueError:
            closed = None
        if closed is not None:
            return found(closed, None)
        filled.append(1 if manifest_path(start, paths.manifests).is_file() else 0)
        if start < first:
            continue
        dates = block_dates(start)
        if now < dates.ready:
            return found(None, dates)
        reading = block_status(paths, start, registered, metas)
        settled = bool(filled[-1]) or reading.status is BlockStatus.EMPTY or now >= dates.deadline
        if now >= dates.deadline:
            if reading.status not in DONE and not baselines_complete(paths, start):
                expired.append(reading)
            continue
        if reading.status in RUNNABLE:
            pending.append(reading)
        elif reading.status in BLOCKED:
            blocked.append(reading)
    if not settled:
        return found(None, None)
    closed = closing_block(filled)
    return found(closed, None, exhausted=closed is None)


def _manifests(paths: StudyPaths) -> int:
    """Los bloques del estudio con manifiesto: los no vacíos, contados como los cuenta el cierre."""
    return sum(
        manifest_path(FIRST_BLOCK + position * STEP, paths.manifests).is_file()
        for position in range(MAX_BLOCKS)
    )


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

    closing: str | None = None
    """El aviso del estudio cerrado, o del bloque 52 terminado sin 50 bloques no vacíos. No es un
    fallo ni necesita al operador: no cuenta en `needs_operator`, así que `auto` sale con 0 y la
    envoltura lo avisa como normal. Se repite en cada ejecución, leído del disco."""

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

    Además se avisa de cada bloque bloqueado (inválido, con duplicados o retenido, este con su
    plazo), que no se corre; de cada bloque vencido con el lote incompleto y sus líneas base sin
    completar; y de cada bloque al que le quedan `WARN_BEFORE` o menos de plazo y no queda
    completo. Mientras el estudio esté cerrado, o haya terminado el bloque 52 sin cerrar, cada
    ejecución lo dice en `closing` con el bloque y los no vacíos, leídos del disco: no se guarda que
    ya se avisó. No es un aviso para el operador. Sin nada pendiente no llama a `runner` ni escribe
    nada.
    """
    now = clock()
    state = study_state(paths, stage1, now)
    alerts = [_blocked_alert(reading, now) for reading in state.blocked]
    alerts += [_expired_alert(reading) for reading in state.expired]

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
        closing=_closing_alert(state),
    )


def _closing_alert(state: StudyState) -> str | None:
    """El aviso del estudio cerrado, o del bloque 52 terminado sin 50 bloques no vacíos."""
    if state.closed_at is not None:
        return (
            f"el estudio cerró en el bloque {state.closed_at}, con {state.non_empty} bloques no "
            "vacíos: los bloques posteriores no son del estudio (apartado 2)."
        )
    if state.exhausted:
        return (
            f"el bloque {MAX_BLOCKS}, el último del estudio, terminó con {state.non_empty} bloques "
            f"no vacíos, menos de {TARGET_BLOCKS}: el estudio ya no cierra (apartado 2)."
        )
    return None


def _blocked_alert(reading: BlockReading, now: datetime) -> str:
    """El aviso de un bloque que no se corre."""
    dates = reading.dates
    label = f"bloque {dates.number} ({dates.start:%Y-%m-%d})"
    if reading.status is BlockStatus.HELD:
        return (
            f"{label}: retenido, su última pasada ({reading.held_by}) no llamó a nadie. No se "
            "corre hasta que haya una pasada posterior, que corre el operador. Vence el "
            f"{dates.deadline.isoformat()}; quedan {_left(dates, now)}."
        )
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


def _expired_alert(reading: BlockReading) -> str:
    """El aviso de un bloque vencido al que le faltan las líneas base."""
    dates = reading.dates
    return (
        f"bloque {dates.number} ({dates.start:%Y-%m-%d}): su plazo venció el "
        f"{dates.deadline.isoformat()} con el lote {reading.status.value} y sus líneas base sin "
        "completar. Completarlas, sin llamar a ningún modelo, es del operador (apartado 5)."
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
        f"- bloques no vacíos: {state.non_empty}; el estudio cierra con {TARGET_BLOCKS}.",
    ]
    if state.terminated is not None:
        return [*lines, f"- el estudio terminó {state.terminated}: no hay nada que correr."]
    if state.closed_at is not None:
        lines.append(f"- el estudio cerró en el bloque {state.closed_at}: se avisaría.")
    if state.exhausted:
        lines.append(
            f"- el bloque {MAX_BLOCKS} terminó sin {TARGET_BLOCKS} bloques no vacíos: el "
            "estudio ya no cierra, y se avisaría."
        )
    for reading in state.pending:
        dates = reading.dates
        lines.append(
            f"- correría {_block(dates)}, {reading.status.value}; vence el "
            f"{dates.deadline.isoformat()}, quedan {_left(dates, state.now)}"
            + (" (se avisaría)." if near_deadline(dates, state.now) else ".")
        )
    for reading in state.blocked:
        held = f" ({reading.held_by})" if reading.held_by is not None else ""
        lines.append(
            f"- no correría {_block(reading.dates)}: está {reading.status.value}{held}, y se "
            "avisaría al operador."
        )
    for reading in state.expired:
        lines.append(
            f"- {_block(reading.dates)} venció el {reading.dates.deadline.isoformat()} "
            f"{reading.status.value} y con sus líneas base sin completar: se avisaría."
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
    if report.closing is not None:
        lines.append(f"- {report.closing}")
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
