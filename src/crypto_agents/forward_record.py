"""Lo registrado del estudio hacia adelante y lo que deja escrito cada lote, sin el router.

`docs/forward.md` registra el instante del merge, el bloque 1, la configuración del decisor y los
cuatro sha-256 del camino medido. Cada pasada de un lote deja un `forward.json` y, si corrió, un
journal por brazo. Lo leen dos comandos que no pueden compartir sus módulos:

- el lote (`forward.py`), que construye backends y trae el router;
- el análisis (`forward_analysis.py`), que no puede cargarlo.

Por eso viven aquí, una sola vez, las piezas que los dos necesitan: las fechas de un bloque, la
configuración registrada del decisor, el camino medido registrado, el `meta` de una pasada, el
estado de un lote leído de sus journals, las pasadas sustituidas y sus contadores. `forward.py` y
`forward_render.py` las importan y las reexportan con su nombre de siempre.

Este módulo no importa el router ni nada que lo traiga. Dos tests de arquitectura lo fijan: uno mira
sus imports y otro lo importa en un proceso nuevo.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, NamedTuple

from pydantic import AwareDatetime, Field

from crypto_agents.audit import META_FILE, read_meta, read_run
from crypto_agents.criteria import CLUSTER_DAYS, check_validity
from crypto_agents.forward_data import AcceptedRevision, MissingCandle, RevisedCandle, SeriesFile
from crypto_agents.forward_plan import BATCH_DEADLINE, first_block
from crypto_agents.market import timeframe_to_timedelta
from crypto_agents.metrics import NO_RESPONSE, AbortKind, undecided_causes, upstream_distribution
from crypto_agents.selection import DEFAULT_HORIZON, DEFAULT_TIMEFRAME
from crypto_agents.state import AgentRole, Backend, FrozenModel, StructuredOutputMode

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path
    from uuid import UUID

    from crypto_agents.audit import RunMeta
    from crypto_agents.journal import EvaluationRecord
    from crypto_agents.metrics import UpstreamRow

__all__ = [
    "FIRST_BLOCK",
    "FORWARD_META_FILE",
    "MERGED_AT",
    "PROPOSAL_SCHEMA_SHA256",
    "REGISTERED_MAX_ATTEMPTS",
    "REGISTERED_PATH",
    "RETRY_TEMPLATE_SHA256",
    "SOLO_REQUEST_ENVELOPE_SHA256",
    "SOLO_TEMPLATE_SHA256",
    "STEP",
    "SUPERSEDED_FILE",
    "TARGET_ARM",
    "WAIT",
    "BatchCounts",
    "BatchMeta",
    "BatchOutcome",
    "BatchState",
    "BlockDates",
    "Check",
    "CheckOutcome",
    "DeciderConfig",
    "DuplicateRecordError",
    "ForwardError",
    "MeasuredPath",
    "SupersededMark",
    "batch_counts",
    "batch_directory",
    "batch_passes",
    "batch_records",
    "batch_state",
    "block_dates",
    "complete",
    "decider_mismatches",
    "decider_rows",
    "expected_backends",
    "pass_backends",
    "registered_decider",
    "served_pairs",
    "superseded",
]

MERGED_AT = datetime(2026, 10, 17, 17, 12, 44, tzinfo=UTC)
"""El instante del merge de `docs/forward.md` en master: la hora del push que muestra GitHub para
`ce3afe2`, según el primer asiento del «Registro del estudio». La fecha de committer cae en el
mismo bloque."""

FIRST_BLOCK = first_block(MERGED_AT)
"""El bloque 1 del estudio, por la regla del apartado 2: 2026-10-22T00:00Z."""

REGISTERED_MAX_ATTEMPTS = 2
"""Intentos por llamada del decisor que registra `docs/forward.md`, apartado 3. El `meta.json` de la
etapa 1 no lo lleva: es el valor por defecto de `ModelRouter`, el que `replay_router` no cambia."""

FORWARD_META_FILE = "forward.json"
"""El `meta` de cada pasada del lote, junto al `meta.json` de la corrida cuando la hay."""

SUPERSEDED_FILE = "superseded.json"
"""La marca de una pasada sustituida (`docs/forward.md`, apartado 5, «Un lote inválido»). Se añade
al directorio de la pasada sin reescribir nada de lo que ya tenía."""

TARGET_ARM = "solo"
STEP = timedelta(days=CLUSTER_DAYS)
WAIT = DEFAULT_HORIZON * timeframe_to_timedelta(DEFAULT_TIMEFRAME)
"""Lo que hay que esperar tras el cierre de un bloque: el horizonte, 6 velas de 4h."""

SOLO_REQUEST_ENVELOPE_SHA256 = "f0727ca0e3c51a242b7351e215a0c6596d1f9e7b777f2efda9d402f7ba3a6928"
"""sha-256 registrado de la envoltura de la petición del decisor de `solo` (`envelope_digest`).

Es la de la configuración de la etapa 1 —`glm-5.2`, `function_calling`, temperatura 0.0— con el
esquema `Proposal`. Lo fija `tests/test_forward_render.py` y lo cita `docs/forward.md`, apartado 3.
"""

SOLO_TEMPLATE_SHA256 = "fe622bc357f853db3e5b61bc06ff6312f47d1dfff4cc2eff06db15d20822f2de"
"""sha-256 registrado de los bytes de la plantilla (`docs/forward.md`, apartado 3, «El prompt»)."""

PROPOSAL_SCHEMA_SHA256 = "4970a9ec09cfc93e89e3058561325e55368f5b87f3f154d5f4e686a1f21e5bb7"
"""sha-256 registrado del esquema de `Proposal` en JSON canónico (apartado 3, «El decisor»)."""

RETRY_TEMPLATE_SHA256 = "588bf07e71d7a92a4d098a9659a3a9c4e05dc3347743ccc1f077e634a8e39890"
"""sha-256 registrado del texto de `llm._RETRY_TEMPLATE` en UTF-8 (apartado 3, «El decisor»)."""


class ForwardError(RuntimeError):
    """El lote no puede seguir: la etapa 1 no se puede leer como hace falta, o falta una pieza."""


class DuplicateRecordError(ForwardError):
    """Un `run_id` con registro completo en dos pasadas del mismo lote: el lote se detiene."""


# ───────────────────────────────────────────── Fechas ─────────────────────────────────────────────


class BlockDates(FrozenModel):
    """Un bloque del estudio: número, extremos, y desde cuándo y hasta cuándo se corre su lote."""

    number: int = Field(ge=1)
    start: AwareDatetime
    end: AwareDatetime
    ready: AwareDatetime
    """El cierre más el horizonte: desde aquí todas sus activaciones tienen sus 6 velas."""

    deadline: AwareDatetime
    """El cierre más 14 días: en este instante el lote ya venció."""


def block_dates(start: datetime) -> BlockDates:
    """Las fechas del bloque que empieza en `start`, contado desde `FIRST_BLOCK`."""
    end = start + STEP
    return BlockDates(
        number=(start - FIRST_BLOCK) // STEP + 1,
        start=start,
        end=end,
        ready=end + WAIT,
        deadline=end + BATCH_DEADLINE,
    )


# ─────────────────────────────────────── Lo registrado ────────────────────────────────────────────


class DeciderConfig(FrozenModel):
    """Los campos de la tabla «El decisor» de `docs/forward.md`, apartado 3."""

    model: str
    backend: Backend
    base_url: str | None
    structured_output: StructuredOutputMode | None
    temperature: float
    timeout_seconds: float | None
    max_attempts: int | None

    def differences(self, other: DeciderConfig) -> tuple[str, ...]:
        """Los campos en que `self` y `other` no coinciden."""
        return tuple(
            name for name in type(self).model_fields if getattr(self, name) != getattr(other, name)
        )


def registered_decider(meta: RunMeta) -> DeciderConfig:
    """La configuración registrada: la del `meta.json` de la etapa 1, más `max_attempts`."""
    roles = meta.arm_roles
    if roles is None or TARGET_ARM not in roles or AgentRole.DECIDER not in roles[TARGET_ARM]:
        raise ForwardError("el meta.json de la etapa 1 no registra el decisor de solo")
    decider = roles[TARGET_ARM][AgentRole.DECIDER]
    return DeciderConfig(
        model=decider.model,
        backend=decider.backend,
        base_url=meta.base_url,
        structured_output=decider.structured_output,
        temperature=decider.temperature,
        timeout_seconds=meta.openai_timeout_seconds,
        max_attempts=REGISTERED_MAX_ATTEMPTS,
    )


class MeasuredPath(FrozenModel):
    """Los cuatro sha-256 que fijan lo que viaja al modelo en la llamada del decisor de `solo`."""

    template_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    retry_template_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    envelope_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def differences(self, registered: MeasuredPath) -> tuple[str, ...]:
        """Los campos que no son los registrados, por su nombre."""
        return tuple(
            name
            for name in type(self).model_fields
            if getattr(self, name) != getattr(registered, name)
        )


REGISTERED_PATH = MeasuredPath(
    template_sha256=SOLO_TEMPLATE_SHA256,
    schema_sha256=PROPOSAL_SCHEMA_SHA256,
    retry_template_sha256=RETRY_TEMPLATE_SHA256,
    envelope_sha256=SOLO_REQUEST_ENVELOPE_SHA256,
)
"""El camino medido tal como lo registra `docs/forward.md`, apartado 3."""


# ─────────────────────────────────────────── El `meta` ────────────────────────────────────────────


class Check(StrEnum):
    """Las comprobaciones previas a la primera llamada, en el orden en que se informan."""

    STAGE1 = "etapa 1"
    MEASURED_PATH = "camino medido"
    RENDER = "renderizado"
    TIES = "atadura"
    POOL = "recuentos del pool"
    DECIDER = "decisor"
    MANIFEST = "manifiesto"
    CATALOG = "catálogo"


class CheckOutcome(FrozenModel):
    """El resultado de una comprobación previa."""

    check: Check
    passed: bool
    detail: str
    ends_study: bool = False
    """Solo el catálogo: `glm-5.2` ya no está, y el estudio termina en este bloque (apartado 5)."""


class BatchOutcome(StrEnum):
    """Cómo terminó una pasada del lote."""

    EMPTY = "vacío"
    """El bloque no tiene activaciones: no hay manifiesto ni corrida."""

    REFUSED = "negado"
    """Una comprobación previa falló: no se llamó a nadie."""

    TERMINATED = "terminado"
    """`glm-5.2` ya no está en el catálogo: el estudio termina en este bloque."""

    CAP_EXHAUSTED = "sin tope"
    """Las pasadas anteriores ya gastaron el tope: no se llamó a nadie."""

    REVISED = "vela revisada"
    """El exchange devuelve distinta una vela guardada que nadie aceptó: no se llamó a nadie."""

    GAP = "hueco"
    """Falta una vela nueva: hay que volver a descargar 24 h después."""

    GAP_PERSISTENT = "hueco persistente"
    """Una vela nueva sigue faltando 24 h después: el estudio termina en este bloque."""

    RAN = "corrió"

    BASELINES = "líneas base"
    """Una corrida solo de las cuatro líneas base, después del plazo, sin llamar a ningún modelo
    (apartado 5): completa sus registros."""


class BatchMeta(FrozenModel):
    """El `meta` de una pasada del lote (`forward.json`), escrito antes de su primera llamada."""

    block: BlockDates
    outcome: BatchOutcome
    invoked_at: AwareDatetime
    hostname: str
    git_commit: str | None
    git_dirty: bool | None
    uv_lock_sha256: str | None
    python_version: str
    measured_path: MeasuredPath | None = None
    """Los cuatro sha-256 del camino medido que se calcularon. `None` si no se pudieron calcular."""

    decider: DeciderConfig | None = None
    """La configuración del decisor que se leyó, `max_attempts` incluido."""

    evaluations: int = Field(ge=0)
    manifest_path: str | None = None
    manifest_sha256: str | None = None
    spend_cap_usd: float = Field(ge=0.0)
    spent_before_usd: float = Field(ge=0.0)
    history: tuple[SeriesFile, ...] = ()
    funding: tuple[SeriesFile, ...] = ()
    checks: tuple[CheckOutcome, ...] = ()
    resumed_from: str | None = None
    accepted: tuple[AcceptedRevision, ...] = ()
    """Las velas revisadas que esta pasada aceptó por el flag."""

    revised: tuple[RevisedCandle, ...] = ()
    """Las velas revisadas que detuvieron la pasada."""

    missing: tuple[MissingCandle, ...] = ()
    """Las velas nuevas que faltaban en la descarga de esta pasada."""


class SupersededMark(FrozenModel):
    """La marca que deja sustituida una pasada de un lote inválido (apartado 5).

    La pasada sustituida queda fuera del lote: ni sus registros ni sus duplicados cuentan. Lo que
    termina el estudio sí se sigue leyendo de ella (apartado 3), y su gasto sigue contando para el
    tope del lote.
    """

    at: AwareDatetime
    by: str
    """El nombre del directorio de la pasada que la sustituye: la repetición entera."""

    reasons: tuple[str, ...] = Field(min_length=1)
    """Por qué el lote era inválido cuando se sustituyó."""


def superseded(directory: Path) -> SupersededMark | None:
    """La marca de sustitución de una pasada, o `None` si no está sustituida."""
    path = directory / SUPERSEDED_FILE
    if not path.is_file():
        return None
    return SupersededMark.model_validate_json(path.read_text(encoding="utf-8"))


# ──────────────────────────────────────── Lo que deja un lote ─────────────────────────────────────


def complete(record: EvaluationRecord) -> bool:
    """Si la evaluación ya no se repite: tiene decisión, o el modelo agotó sus intentos sin una
    salida válida. Lo que perdió por el proveedor, un 402 o el tope se vuelve a evaluar."""
    if record.proposed is not None:
        return True
    return any(kind is AbortKind.VALIDATION for _, kind in undecided_causes([record]))


class BatchState(NamedTuple):
    """Lo que dejaron las pasadas de un lote, leído de sus journals."""

    records: dict[str, dict[UUID, EvaluationRecord]]
    """Por brazo y `run_id`, el registro completo, o el último incompleto si no lo hay."""

    done: dict[str, set[UUID]]
    """Por brazo, los `run_id` con registro completo: los que una pasada nueva se salta."""


def batch_state(passes: Sequence[Path]) -> BatchState:
    """El estado del lote a partir de los journals de sus pasadas, de la más antigua a la última.

    Una pasada se salta lo que ya está completo, así que cada `run_id` aparece completo en una sola
    pasada. Si aparece en otra después, algo corrió dos veces lo mismo, y el lote se detiene
    (`DuplicateRecordError`) en vez de elegir uno de los dos.
    """
    records: dict[str, dict[UUID, EvaluationRecord]] = {}
    done: dict[str, set[UUID]] = {}
    where: dict[tuple[str, UUID], Path] = {}
    for path in passes:
        for arm in read_run(path).arms:
            arm_records = records.setdefault(arm.arm, {})
            arm_done = done.setdefault(arm.arm, set())
            for record in arm.records:
                if record.run_id in arm_done:
                    raise DuplicateRecordError(
                        f"{arm.arm}: el run_id {record.run_id} ya tenía registro completo en "
                        f"{where[(arm.arm, record.run_id)]} y aparece otra vez en {path}"
                    )
                arm_records[record.run_id] = record
                if complete(record):
                    arm_done.add(record.run_id)
                    where[(arm.arm, record.run_id)] = path
    return BatchState(records=records, done=done)


def batch_directory(batches: Path, start: datetime) -> Path:
    """`data/forward/batches/<YYYY-MM-DD>/`: el directorio de las pasadas del lote de un bloque."""
    return batches / f"{start:%Y-%m-%d}"


def batch_passes(directory: Path, *, include_superseded: bool = False) -> list[Path]:
    """Las pasadas que corrieron de un lote, de la más antigua a la más reciente.

    Sin las sustituidas, salvo que se pidan: el estado del lote, sus registros y la búsqueda de
    duplicados solo leen las pasadas no sustituidas (apartado 5). Con ellas, la cadena del gasto.
    """
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if (path / META_FILE).is_file() and (include_superseded or superseded(path) is None)
    )


def batch_records(directory: Path) -> dict[str, tuple[EvaluationRecord, ...]]:
    """Los registros de un lote por brazo, uno por `run_id`, juntando sus pasadas: lo que se cuenta
    del lote y lo que leerá el análisis. `directory` es `data/forward/batches/<inicio>/`."""
    state = batch_state(batch_passes(directory))
    return {arm: tuple(by_id.values()) for arm, by_id in state.records.items()}


def expected_backends(meta: RunMeta) -> dict[str, dict[AgentRole, Backend]]:
    """El backend que el `meta.json` de una pasada declara para cada rol de cada brazo: lo que el
    criterio 6 compara con el de cada llamada."""
    roles = meta.arm_roles or {}
    return {
        name: {role: entry.backend for role, entry in arm.items()} for name, arm in roles.items()
    }


def pass_backends(passes: Sequence[Path]) -> dict[str, dict[AgentRole, Backend]]:
    """Los backends declarados para cada brazo, juntando las pasadas del lote: un brazo lo describe
    la última pasada que lo corrió. Una pasada de líneas base no describe `solo`, y un brazo que no
    corrió ninguna pasada no tiene registros que comprobar."""
    found: dict[str, dict[AgentRole, Backend]] = {}
    for path in passes:
        found.update(expected_backends(read_meta(path)))
    found.setdefault(TARGET_ARM, {})
    return found


def decider_mismatches(passes: Sequence[Path], registered: DeciderConfig) -> tuple[str, ...]:
    """Las pasadas que corrieron `solo` con la configuración del decisor distinta de la registrada.

    Invalidan el lote (apartado 5) y no terminan el estudio: lo termina el camino medido cambiado,
    no la configuración (apartado 3). Una pasada sin `forward.json` no se puede juzgar aquí; la
    integridad del análisis la rechaza.
    """
    found: list[str] = []
    for path in passes:
        record = path / FORWARD_META_FILE
        if not record.is_file():
            continue
        meta = BatchMeta.model_validate_json(record.read_text(encoding="utf-8"))
        if meta.outcome is not BatchOutcome.RAN or meta.decider is None:
            continue
        different = meta.decider.differences(registered)
        if different:
            found.append(
                f"la pasada {path.name} corrió con la configuración del decisor distinta de la "
                f"registrada: {', '.join(different)}"
            )
    return tuple(found)


def decider_rows(records: Iterable[EvaluationRecord]) -> tuple[UpstreamRow, ...]:
    """Quién sirvió las llamadas del decisor de esos registros."""
    return tuple(row for row in upstream_distribution(records) if row.role is AgentRole.DECIDER)


def served_pairs(rows: Iterable[UpstreamRow]) -> frozenset[tuple[str, str | None]]:
    """Los pares (modelo, endpoint) que contestaron. Sin respuesta no se sabe quién sirve."""
    return frozenset(
        (row.upstream_model, row.upstream_endpoint)
        for row in rows
        if row.upstream_model != NO_RESPONSE
    )


class BatchCounts(FrozenModel):
    """Los contadores de validez de un lote y de sus evaluaciones. Ningún retorno.

    Los lee el resumen de cada lote y el análisis, que tiene que contar las faltantes y los lotes
    inválidos exactamente como los contó el lote.
    """

    evaluations: int = Field(ge=0)
    """Entradas del manifiesto del bloque."""

    recorded: int = Field(ge=0)
    """Registros de `solo` en la última pasada."""

    decided: int = Field(ge=0)
    """Decisiones válidas de `solo`."""

    invalid: int = Field(ge=0)
    """Respuestas inválidas tras los dos intentos: son del modelo, valen 0 y no son faltantes."""

    missing: int = Field(ge=0)
    """Faltantes: perdidas por un 402, por el proveedor (transporte o plazo) o por el tope."""

    missing_by_cause: dict[str, int] = Field(default_factory=dict)
    other_undecided: int = Field(ge=0)
    """Sin decisión por otra causa, que no es ninguna de las anteriores."""

    baselines_decided: dict[str, int] = Field(default_factory=dict)
    invalidating: tuple[str, ...] = ()
    """Lo que invalida el lote en el estudio: cuota del decisor, caché de otro backend, vetos
    distintos de `invalid_stop_side` y una pasada con la configuración del decisor distinta de la
    registrada (apartado 5)."""

    @property
    def unrecorded(self) -> int:
        """Evaluaciones del manifiesto sin registro de `solo`: una pasada interrumpida o un lote
        que no corrió."""
        return self.evaluations - self.recorded

    @property
    def missing_at_deadline(self) -> int:
        """Las faltantes del apartado 5 al vencer el plazo: toda evaluación de `solo` sin decisión
        que no sea una respuesta inválida del modelo, con registro o sin él."""
        return self.evaluations - self.decided - self.invalid

    @property
    def complete(self) -> bool:
        """El lote es válido y está cerrado: cada activación tiene su decisión o una respuesta
        inválida, y las cuatro líneas base decidieron todas."""
        return (
            not self.invalidating
            and self.missing == 0
            and self.other_undecided == 0
            and self.recorded == self.evaluations
            and self.decided + self.invalid == self.evaluations
            and all(count == self.evaluations for count in self.baselines_decided.values())
        )


def batch_counts(
    evaluations: int,
    records: Mapping[str, Sequence[EvaluationRecord]],
    expected_backends: Mapping[str, Mapping[AgentRole, Backend]],
    passes_invalidating: Sequence[str] = (),
) -> BatchCounts:
    """Los contadores de un lote sobre los registros de sus brazos. Sin velas: no puntúa.

    Las faltantes son las tres causas que en el estudio reanudan el lote en vez de invalidarlo; lo
    que lo invalida son las otras tres condiciones del criterio 6 (apartado 5), más lo que traiga
    `passes_invalidating`: lo que se lee de las pasadas y no de sus registros, como
    `decider_mismatches`.
    """
    validity = check_validity(records, expected_backends)
    target = next(entry for entry in validity.arms if entry.arm == TARGET_ARM)
    solo = records.get(TARGET_ARM, ())
    causes = undecided_causes(solo)
    missing_by_cause = {
        "402": target.funds_lost,
        "proveedor": sum(target.provider_lost.values()),
        "tope": target.spend_cap_lost,
    }
    invalid = sum(count for (_, kind), count in causes.items() if kind is AbortKind.VALIDATION)
    decided = sum(1 for record in solo if record.proposed is not None)
    missing = sum(missing_by_cause.values())
    invalidating = [
        f"{entry.arm}: {reason}"
        for entry in validity.arms
        for reason, count in (
            ("el decisor perdió evaluaciones por cuota", entry.decider_quota_lost),
            ("aciertos de caché con otro backend", entry.cache_backend_mismatch),
            ("vetos distintos de invalid_stop_side", sum(entry.foreign_vetoes.values())),
        )
        if count
    ]
    invalidating += passes_invalidating
    return BatchCounts(
        evaluations=evaluations,
        recorded=len(solo),
        decided=decided,
        invalid=invalid,
        missing=missing,
        missing_by_cause=missing_by_cause,
        other_undecided=len(solo) - decided - invalid - missing,
        baselines_decided={
            arm: sum(1 for record in arm_records if record.proposed is not None)
            for arm, arm_records in records.items()
            if arm != TARGET_ARM
        },
        invalidating=tuple(invalidating),
    )
