"""Los lotes del estudio hacia adelante de `solo`: lo que `docs/forward.md` pide en su apartado 10.

    python -m crypto_agents.forward batch --block 2026-10-22 [--supersede]
    python -m crypto_agents.forward baselines --block 2026-10-22
    python -m crypto_agents.forward rehearse --cache var/ablation-cache --out <directorio nuevo>
    python -m crypto_agents.forward analyze <directorio del estudio>
    python -m crypto_agents.forward rehearse-analysis
    python -m crypto_agents.forward auto [--dry-run | --report <informe>]
    python -m crypto_agents.forward fingerprint [--stage1-cache var/ablation-cache]

Un lote es el bloque de una semana pasado por el arnés de la ablación como un replay: `solo` y las
cuatro líneas base, con `--fill`, la caché del estudio y el tope del lote. El runner en vivo no se
usa. El orden es el del apartado 10, punto 1, y nada llama a un modelo antes del último paso:

1. Las negativas por fecha. Sin escribir nada.
2. La extensión del histórico y del funding (`forward_data`), con las dos reglas del apartado 4:
   - una vela revisada detiene el lote (código 5) hasta que `--accept-revised` la nombra;
   - una vela nueva que falta lo detiene (código 6), y si sigue faltando 24 h después el estudio
     termina en ese bloque (código 7).
3. La población del bloque y su manifiesto.
4. Las comprobaciones previas, todas, sobre archivos versionados. Una que falla deja el lote sin
   llamar a nadie, y dice cuál.
5. La corrida, con el tope acumulado entre las pasadas del lote.
6. Un resumen de contadores y la lista de archivos que hay que versionar: todas las pasadas del
   lote, no solo la última. Este comando no hace commits.

Cada pasada que llamó al decisor deja al final su huella en `data/forward/fingerprints/` (el tok/s
y el orden de las claves de los argumentos, `forward_fingerprint`), y el resumen la cita en una
línea. La huella entra en los archivos que hay que versionar y nunca cambia el resultado ni el
código de salida del lote. `fingerprint` escribe la de la etapa 1 y la de las pasadas anteriores a
este código.

Tres decisiones que lo mantienen dentro de lo registrado:

- **No puntúa nada del estudio.** Corre por `ablation.replay_arm`, que es `run_arm` sin la
  consolidación que calcula retornos. `summarise` no recibe velas: con lo que tiene no puede
  puntuar. Los únicos retornos que se calculan son los de las líneas base de la etapa 1, en la
  comprobación de la atadura que el apartado 10 pide.
- **Es el único módulo del estudio que construye backends**, y los construye después de las
  comprobaciones que no necesitan red. El catálogo de la pasarela es lo único que pide un backend
  antes de correr, y no es una llamada a un modelo.
- **Cada pasada es un directorio nuevo** bajo `data/forward/batches/<inicio>/`, encadenado a la
  anterior por `resumed_from`, y la reanudación es por journal. Una pasada se salta los `run_id`
  que ya tienen registro completo en las anteriores (`batch_state`), y la caché solo sirve los
  reintentos de lo que se vuelve a evaluar. Un `run_id` completo en dos pasadas detiene el lote.
  El tope se siembra con lo que gastaron las pasadas anteriores, leído de sus journals.

Dos caminos más, los del apartado 5:

- **`batch --supersede`** repite entero un lote inválido. Solo con un lote inválido: uno que solo
  está incompleto se reanuda, y repetir uno válido sería volver a tirar los dados hasta que salga
  la decisión que se quiere. Justo antes de la primera llamada marca sustituidas todas las pasadas
  del lote (`superseded.json`), y la repetición corre con una caché propia: la clave de la caché no
  lleva la temperatura, el timeout ni los intentos, así que con la compartida la repetición podría
  devolver las respuestas de la pasada inválida. El gasto de las sustituidas sigue contando para
  el tope del lote.
- **`baselines`** completa, después del plazo, los registros de las cuatro líneas base de un bloque:
  el de un lote interrumpido o que no corrió. No construye backends, corre con un router solo de
  caché sobre una caché vacía y comprueba al terminar que ningún registro trae una llamada.

`rehearse` es el ensayo sin pagar. Pasa el manifiesto de la etapa 1 por las mismas comprobaciones y
la misma función de corrida, solo con caché, y compara decisiones, stops y `prompt_digest` con
`data/stage1/`.

`analyze` y `rehearse-analysis` son el análisis del apartado 6 y su ensayo. Viven en
`forward_analysis`, que no trae el router; aquí solo se despachan, sin construir backends y sin
ningún argumento con el que saltarse una negativa.

`auto` es lo que lanza el temporizador desde el bloque 2. Lee de `forward_auto` qué bloques están
pendientes y qué hacer con cada código de salida, y corre cada lote con `batch_runner`: el mismo
`run_batch` que `batch --block`, sin velas aceptadas y sin sustitución. No hace commits; los hace
la envoltura del servicio (`forward_service`) con la lista de archivos que devuelve cada lote.
`batch`, `baselines`, `auto` y `fingerprint` toman el mismo cerrojo (`batch_lock`): dos pasadas a
la vez sobre un lote dejarían duplicados, y una huella escrita con un lote en marcha sería la de una
pasada a medias. `batch` y `baselines` lo toman después de la negativa por fecha, así que una fecha
negada no escribe nada, ni el cerrojo.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import platform
import socket
import subprocess
import sys
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple
from uuid import UUID

from pydantic import AwareDatetime, Field

from crypto_agents.ablation import (
    ARMS,
    NOTIONAL_ACCOUNT,
    build_run_meta,
    git_state,
    guard_for,
    open_run_directory,
    plan_from_manifest,
    replay_arm,
    seed_from_previous,
)
from crypto_agents.audit import (
    AuditError,
    PlanKind,
    RunDirectory,
    arm_journal_path,
    chain_calls,
    file_sha256,
    read_meta,
    read_run,
    run_chain,
)
from crypto_agents.cache import InMemoryResponseCache, JsonFileResponseCache
from crypto_agents.context import utc_now
from crypto_agents.criteria import CriteriaError
from crypto_agents.design_effect import DesignEffectError, pool_records, stage_ties
from crypto_agents.forward_analysis import StudyPaths, analyze, rehearse_analysis
from crypto_agents.forward_auto import (
    AutoReport,
    UpstreamStatus,
    render_report,
    render_state,
    run_auto,
    study_state,
)
from crypto_agents.forward_data import (
    BATCH_DIR,
    FORWARD_FUNDING_DIR,
    FORWARD_HISTORY_DIR,
    MANIFEST_DIR,
    PUBLISHED_FUNDING_SHA256,
    PUBLISHED_HISTORY_SHA256,
    REVISIONS_FILE,
    STAGE1_DIR,
    AcceptedRevision,
    CandleFetch,
    CandleRef,
    ForwardDataError,
    FundingFetch,
    MissingCandle,
    MissingCandlesError,
    RevisedCandle,
    RevisedCandlesError,
    SeriesFile,
    block_manifest,
    extend_funding,
    extend_history,
    fetch_perp_funding,
    fetch_spot_candles,
    manifest_path,
    read_histories,
    study_population,
    verify_stage1,
    write_block_manifest,
)
from crypto_agents.forward_fingerprint import (
    FINGERPRINT_DIR,
    STAGE1_CACHE,
    block_fingerprints,
    decider_called,
    failed_fingerprint,
    fingerprint,
    fingerprint_line,
    pass_fingerprint_path,
    retroactive,
    write_fingerprint,
)
from crypto_agents.forward_plan import (
    FORWARD_ARMS,
    MAX_BLOCKS,
    POOL_COUNTS,
    PUBLISHED_RETURNS,
    PUBLISHED_RETURNS_SHA256,
    TARGET_BLOCKS,
    ForwardPlanError,
    PoolCounts,
    PublishedTie,
    batch_spend_cap,
    closing_block,
    pool_counts,
    published_returns,
    published_ties,
)
from crypto_agents.forward_record import (
    FIRST_BLOCK,
    FORWARD_META_FILE,
    MERGED_AT,
    REGISTERED_MAX_ATTEMPTS,
    REGISTERED_PATH,
    STEP,
    SUPERSEDED_FILE,
    TARGET_ARM,
    BatchCounts,
    BatchMeta,
    BatchOutcome,
    BlockDates,
    Check,
    CheckOutcome,
    DeciderConfig,
    DuplicateRecordError,
    ForwardError,
    MeasuredPath,
    SupersededMark,
    batch_counts,
    batch_directory,
    batch_passes,
    batch_records,
    batch_state,
    block_dates,
    decider_mismatches,
    decider_rows,
    pass_backends,
    registered_decider,
    served_pairs,
    superseded,
)
from crypto_agents.forward_render import ForwardRenderError, check_run, measured_path
from crypto_agents.forward_service import changed_paths, inside_study
from crypto_agents.funding import FUNDING_DIR
from crypto_agents.journal import JsonlJournal
from crypto_agents.llm import OpenAIBackend, build_backends
from crypto_agents.market import MarketDataError, timeframe_to_timedelta
from crypto_agents.metrics import (
    EPOCH,
    NO_HEADER,
    NO_RESPONSE,
    UpstreamRow,
)
from crypto_agents.paired_dispersion import RecalculationError, read_stage
from crypto_agents.perp_probe import ProbeError
from crypto_agents.quota import QuotaLedger
from crypto_agents.replay import ReplayCacheMissError, replay_router, replay_run_id
from crypto_agents.selection import (
    DEFAULT_MANIFEST,
    HISTORY_DIR,
    SelectionError,
    SelectionManifest,
    load_manifest,
    load_selection_histories,
)
from crypto_agents.settings import DEFAULT_ENV_FILE, ConfigError, load_settings, public_url
from crypto_agents.state import (
    AgentRole,
    Backend,
    Billing,
    FailureKind,
    FrozenModel,
)

if TYPE_CHECKING:
    from collections.abc import (
        Awaitable,
        Callable,
        Collection,
        Iterable,
        Iterator,
        Mapping,
        Sequence,
    )

    from crypto_agents.ablation import AblationArm, ReplayPlan
    from crypto_agents.cache import ResponseCache
    from crypto_agents.design_effect import StageTie
    from crypto_agents.journal import EvaluationRecord
    from crypto_agents.llm import ChatBackend, ModelCatalog, ModelRouter
    from crypto_agents.paired_dispersion import StageRun
    from crypto_agents.quota import Clock
    from crypto_agents.selection import PlannedEvaluation
    from crypto_agents.settings import Settings
    from crypto_agents.spend import SpendGuard
    from crypto_agents.state import LLMCall

__all__ = [
    "COMMAND_ERRORS",
    "FIRST_BLOCK",
    "FORWARD_META_FILE",
    "LOCK_FILE",
    "MERGED_AT",
    "REGISTERED_MAX_ATTEMPTS",
    "ArmComparison",
    "BatchEnvironment",
    "BatchMeta",
    "BatchOutcome",
    "BatchResult",
    "BatchSummary",
    "BlockDates",
    "Check",
    "CheckOutcome",
    "DeciderConfig",
    "DuplicateRecordError",
    "FirstCall",
    "ForwardError",
    "ForwardPaths",
    "LockHeldError",
    "PreparedBlock",
    "SupersededMark",
    "auto_command",
    "baseline_arms",
    "baseline_checks",
    "baselines_date_refusal",
    "baselines_pass",
    "batch_lock",
    "batch_records",
    "batch_runner",
    "batch_state",
    "block_dates",
    "catalog_check",
    "compare_arms",
    "date_refusal",
    "decider_check",
    "default_environment",
    "fingerprint_root",
    "first_showing",
    "main",
    "manifest_check",
    "offline_checks",
    "pass_cache",
    "pool_check",
    "prepare_block",
    "read_decider",
    "registered_decider",
    "rehearse",
    "replay_forward_arms",
    "router_attempts",
    "run_baselines",
    "run_batch",
    "run_pass",
    "served_label",
    "spend_state",
    "study_git_state",
    "study_paths",
    "summarise",
    "ties_check",
    "upstream_entry",
    "upstream_status",
    "write_auto_report",
    "write_superseded",
]

UPSTREAM_FILE = "upstream.md"
"""El asiento preparado para el registro cuando cambia el upstream del decisor. No se escribe solo
en `docs/forward.md`."""

CACHE_DIR = Path("var/forward/cache")
"""La caché propia del estudio. No se versiona: los journals guardan cada llamada."""

RERUN_CACHE = "reruns"
"""Bajo la caché del estudio, la de cada repetición de un lote sustituido:
`reruns/<AAAA-MM-DD>/<pasada que sustituye>/`."""

FUNDING_TAIL = timedelta(days=1)
"""El funding se extiende hasta el cierre del bloque más un día, incluido (apartado 4)."""

GAP_WAIT = timedelta(hours=24)
"""Lo que se espera para volver a descargar una vela que falta antes de declarar el hueco
persistente (apartado 4, «Hueco persistente»)."""

LOCK_FILE = Path("var/forward/lote.lock")
"""El cerrojo que comparten `batch`, `baselines` y `auto`: dos pasadas a la vez sobre un mismo lote
evaluarían los mismos `run_id` y dejarían duplicados."""

_COMMAND = "python -m crypto_agents.forward"


class StudyClosedError(ForwardError):
    """El estudio ya cerró antes de este bloque: el bloque no es parte del estudio."""


class LockHeldError(ForwardError):
    """Otro `batch`, `baselines` o `auto` tiene el cerrojo: este no hace nada."""


@contextmanager
def batch_lock(path: Path) -> Iterator[None]:
    """El cerrojo exclusivo de los comandos que escriben lotes, sin esperar.

    Si otro proceso lo tiene, levanta `LockHeldError` antes de tocar nada del estudio. Lo suelta el
    sistema si el proceso muere, así que un lote cortado no deja el cerrojo tomado.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise LockHeldError(
                f"otro lote está corriendo ({path}): no se hace nada y no se llamó a ningún modelo"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


# ───────────────────────────────────────────── Fechas ─────────────────────────────────────────────


def _block_refusal(start: datetime) -> str | None:
    """Por qué `start` no es el inicio de un bloque del estudio, o `None` si lo es."""
    if start.tzinfo is None or (start - EPOCH) % STEP:
        return (
            f"{start.isoformat()} no es el inicio de un bloque: los bloques empiezan los jueves "
            "a las 00:00Z"
        )
    if start < FIRST_BLOCK:
        return (
            f"el bloque de {start:%Y-%m-%d} es anterior al bloque 1 registrado "
            f"({FIRST_BLOCK:%Y-%m-%d}): no se evalúa nada antes"
        )
    dates = block_dates(start)
    if dates.number > MAX_BLOCKS:
        return (
            f"el bloque de {start:%Y-%m-%d} sería el {dates.number}: el estudio acaba en el "
            f"{MAX_BLOCKS}"
        )
    return None


def date_refusal(start: datetime, now: datetime) -> str | None:
    """Por qué el lote de `start` no se puede correr en `now`, o `None` si se puede.

    Cinco negativas: el inicio no es un jueves a las 00:00Z, es anterior al bloque 1, es posterior
    al bloque 52, todavía no pasó el cierre más el horizonte, o ya venció el plazo de 14 días. En
    los bordes, el lote se puede correr desde `ready` exacto y en `deadline` exacto ya no.
    """
    refusal = _block_refusal(start)
    if refusal is not None:
        return refusal
    dates = block_dates(start)
    if now < dates.ready:
        return (
            f"todavía no: el lote del bloque {dates.number} se corre desde "
            f"{dates.ready.isoformat()}, cuando todas sus activaciones tienen el horizonte delante"
        )
    if now >= dates.deadline:
        return f"el plazo del lote del bloque {dates.number} venció el {dates.deadline.isoformat()}"
    return None


def baselines_date_refusal(start: datetime, now: datetime) -> str | None:
    """Por qué las líneas base de `start` no se pueden completar en `now`, o `None` si se puede.

    Las mismas negativas del bloque que el lote, y una de fecha: solo desde el plazo exacto. Antes
    del plazo las completa el lote, que corre los cinco brazos.
    """
    refusal = _block_refusal(start)
    if refusal is not None:
        return refusal
    dates = block_dates(start)
    if now < dates.deadline:
        return (
            f"todavía no: las líneas base del bloque {dates.number} se completan aparte desde que "
            f"vence su plazo, el {dates.deadline.isoformat()}; antes las completa el lote"
        )
    return None


# ───────────────────────────────────────────── El decisor ─────────────────────────────────────────


def read_decider(settings: Settings, max_attempts: int) -> DeciderConfig:
    """La configuración que el lote leyó de `.env`, con los intentos del router que va a llamar."""
    choice = settings.role_choices(AgentRole.DECIDER)[0]
    openai = settings.openai
    return DeciderConfig(
        model=choice.model,
        backend=choice.backend,
        base_url=public_url(None if openai is None else openai.base_url),
        structured_output=choice.structured_output,
        temperature=choice.temperature,
        timeout_seconds=None if openai is None else openai.timeout_seconds,
        max_attempts=max_attempts,
    )


def router_attempts(router: ModelRouter) -> int:
    """Los intentos por llamada del router. `ModelRouter` no los publica: se lee el atributo que
    fija su constructor, del mismo `replay_router` que construye `replay_arm`."""
    return router._max_attempts


# ────────────────────────────────────── Comprobaciones previas ────────────────────────────────────


def _skipped(check: Check, why: str) -> CheckOutcome:
    return CheckOutcome(check=check, passed=False, detail=f"no se pudo hacer: {why}")


def stage_check(directory: Path) -> CheckOutcome:
    """Los seis archivos de `data/stage1/` son los publicados."""
    try:
        verify_stage1(directory)
    except ForwardDataError as error:
        return CheckOutcome(check=Check.STAGE1, passed=False, detail=str(error))
    return CheckOutcome(
        check=Check.STAGE1,
        passed=True,
        detail=f"los seis sha-256 de {directory} son los publicados",
    )


async def current_path(settings: Settings) -> MeasuredPath | str:
    """Los cuatro sha-256 del camino medido de hoy, con el decisor de `settings`, o por qué no se
    pudieron calcular."""
    try:
        return await measured_path(settings.role_choices(AgentRole.DECIDER)[0])
    except (ForwardRenderError, OSError) as error:
        return str(error)


def measured_check(measured: MeasuredPath | str) -> CheckOutcome:
    """Plantilla, esquema, plantilla de reintento y envoltura: los cuatro sha-256 registrados."""
    if isinstance(measured, str):
        return _skipped(Check.MEASURED_PATH, measured)
    different = measured.differences(REGISTERED_PATH)
    if different:
        return CheckOutcome(
            check=Check.MEASURED_PATH,
            passed=False,
            detail=f"el camino medido cambió: {', '.join(different)} no es el registrado",
        )
    return CheckOutcome(
        check=Check.MEASURED_PATH,
        passed=True,
        detail="plantilla, esquema de Proposal, reintento y envoltura: los registrados",
    )


def decider_check(registered: DeciderConfig, read: DeciderConfig, billing: Billing) -> CheckOutcome:
    """La configuración que se leyó es la registrada, y se paga por uso."""
    problems = [
        f"{name}: registrado {getattr(registered, name)!r}, leído {getattr(read, name)!r}"
        for name in read.differences(registered)
    ]
    if billing is not Billing.PAYG:
        problems.append(f"billing {billing.value}: el estudio paga por uso")
    if problems:
        return CheckOutcome(check=Check.DECIDER, passed=False, detail="; ".join(problems))
    return CheckOutcome(
        check=Check.DECIDER,
        passed=True,
        detail=f"`{read.model}` como en la etapa 1, {read.max_attempts} intentos, pago por uso",
    )


def manifest_check(manifest: SelectionManifest, reference: Path) -> CheckOutcome:
    """Los parámetros del manifiesto del bloque son los de `data/ablation_selection.json`."""
    try:
        expected = load_manifest(reference)
    except SelectionError as error:
        return _skipped(Check.MANIFEST, str(error))
    pairs = {
        "símbolos": (
            sorted(ref.symbol for ref in manifest.series),
            sorted(ref.symbol for ref in expected.series),
        ),
        "temporalidad": (manifest.timeframe, expected.timeframe),
        "candle_limit": (manifest.candle_limit, expected.candle_limit),
        "horizonte": (manifest.horizon, expected.horizon),
        "warm-up": (manifest.warmup_bars, expected.warmup_bars),
        "semilla": (manifest.seed, expected.seed),
    }
    different = [
        f"{name}: {ours!r} y en {reference} {theirs!r}"
        for name, (ours, theirs) in pairs.items()
        if ours != theirs
    ]
    if different:
        return CheckOutcome(check=Check.MANIFEST, passed=False, detail="; ".join(different))
    return CheckOutcome(
        check=Check.MANIFEST,
        passed=True,
        detail=f"símbolos, temporalidad, candle_limit, horizonte, warm-up y semilla de {reference}",
    )


def ties_check(ties: Sequence[StageTie], published: Sequence[PublishedTie]) -> CheckOutcome:
    """Las líneas base sobre el pool dan lo de sus journals y lo publicado, evaluación por
    evaluación."""
    journal_ok = sum(tie.evaluations for tie in ties if tie.exact)
    journal_all = sum(tie.evaluations for tie in ties)
    published_ok = sum(tie.published for tie in published if tie.exact)
    published_all = sum(tie.published for tie in published)
    passed = all(tie.exact for tie in ties) and all(tie.exact for tie in published)
    failing = [tie.arm for tie in ties if not tie.exact] + [
        f"{tie.arm} (publicado)" for tie in published if not tie.exact
    ]
    detail = (
        f"contra los journals {journal_ok} de {journal_all}; contra lo publicado {published_ok} "
        f"de {published_all}"
    )
    if failing:
        detail += f"; no coinciden: {', '.join(failing)}"
    return CheckOutcome(check=Check.TIES, passed=passed, detail=detail)


def pool_check(counts: PoolCounts, registered: PoolCounts) -> CheckOutcome:
    """Los cuatro recuentos del pool sobre `data/history/` son los registrados."""
    figures = (
        f"{counts.activations} / {counts.unconfirmed} / {counts.candidates} / {counts.confirmed}"
    )
    if counts == registered:
        return CheckOutcome(check=Check.POOL, passed=True, detail=f"{figures}, los registrados")
    expected = (
        f"{registered.activations} / {registered.unconfirmed} / {registered.candidates} / "
        f"{registered.confirmed}"
    )
    return CheckOutcome(
        check=Check.POOL, passed=False, detail=f"{figures}, y los registrados son {expected}"
    )


class ForwardPaths(NamedTuple):
    """Dónde lee y escribe el estudio. Las pruebas lo apuntan a un directorio temporal."""

    history: Path = FORWARD_HISTORY_DIR
    funding: Path = FORWARD_FUNDING_DIR
    manifests: Path = MANIFEST_DIR
    batches: Path = BATCH_DIR
    cache: Path = CACHE_DIR
    stage1: Path = STAGE1_DIR
    published_history: Path = HISTORY_DIR
    published_funding: Path = FUNDING_DIR
    reference_manifest: Path = DEFAULT_MANIFEST
    revisions: Path = REVISIONS_FILE
    lock: Path = LOCK_FILE


def study_paths(paths: ForwardPaths) -> StudyPaths:
    """Las rutas del estudio con los nombres con que las lee el análisis, para `auto`."""
    return StudyPaths(
        root=paths.batches.parent,
        history=paths.history,
        funding=paths.funding,
        manifests=paths.manifests,
        batches=paths.batches,
        revisions=paths.revisions,
    )


def fingerprint_root(paths: ForwardPaths) -> Path:
    """`data/forward/fingerprints/`: junto a `batches/`, con la raíz del estudio de `study_paths`.

    No es un campo de `ForwardPaths`: una prueba que apunta el estudio a un directorio temporal
    dejaría ese campo apuntando al repositorio, y escribiría ahí.
    """
    return paths.batches.parent / FINGERPRINT_DIR


_STAGE_ERRORS = (
    AuditError,
    CriteriaError,
    DesignEffectError,
    ForwardPlanError,
    ForwardRenderError,
    RecalculationError,
    SelectionError,
    OSError,
)


async def offline_checks(
    settings: Settings,
    max_attempts: int,
    manifest: SelectionManifest,
    paths: ForwardPaths,
    measured: MeasuredPath | str,
) -> tuple[CheckOutcome, ...]:
    """Las comprobaciones previas que no necesitan red, todas, sobre archivos versionados.

    Las de la etapa 1 leen `data/stage1/` después de comprobar sus sha-256. Si no son los
    publicados, no se hacen y lo dicen. La atadura y los recuentos comparten un solo cálculo del
    pool, sobre `data/history/`: el original, no el extendido. `measured` es el camino medido ya
    calculado, el mismo que queda en el `meta` de la pasada.
    """
    outcomes: dict[Check, CheckOutcome] = {
        Check.MEASURED_PATH: measured_check(measured),
        Check.MANIFEST: manifest_check(manifest, paths.reference_manifest),
    }
    stage1 = stage_check(paths.stage1)
    outcomes[Check.STAGE1] = stage1
    dependent = (Check.RENDER, Check.TIES, Check.POOL, Check.DECIDER)
    if not stage1.passed:
        outcomes.update(
            (check, _skipped(check, "la etapa 1 no es la publicada")) for check in dependent
        )
        return tuple(outcomes[check] for check in Check if check in outcomes)

    try:
        registered = registered_decider(read_meta(paths.stage1))
    except (AuditError, ForwardError) as error:
        outcomes[Check.DECIDER] = _skipped(Check.DECIDER, str(error))
    else:
        outcomes[Check.DECIDER] = decider_check(
            registered, read_decider(settings, max_attempts), settings.billing
        )

    try:
        stage = read_stage(paths.stage1, None, paths.published_history, None)
    except _STAGE_ERRORS as error:
        outcomes.update((check, _skipped(check, str(error))) for check in dependent[:3])
        return tuple(outcomes[check] for check in Check if check in outcomes)

    try:
        plan = plan_from_manifest(stage.manifest, stage.histories)
        render, envelope = await check_run(plan, settings, stage.arms.get(TARGET_ARM, ()))
    except _STAGE_ERRORS as error:
        outcomes[Check.RENDER] = _skipped(Check.RENDER, str(error))
    else:
        outcomes[Check.RENDER] = CheckOutcome(
            check=Check.RENDER,
            passed=render.passed and envelope.passed,
            detail=(
                f"{render.matching} de {render.expected} prompts como en el journal; envoltura "
                f"{'la registrada' if envelope.envelope_matches else envelope.digest[:12] + '…'}"
            ),
        )

    outcomes.update(_pool_outcomes(stage))
    return tuple(outcomes[check] for check in Check if check in outcomes)


def _pool_outcomes(stage: StageRun) -> dict[Check, CheckOutcome]:
    """La atadura y los recuentos, con un solo cálculo del pool sobre `data/history/`."""
    try:
        pool = pool_records(stage.manifest, stage.histories)
        ties = stage_ties(pool, stage.arms, stage.histories, stage.horizon)
        published = published_ties(
            pool,
            published_returns(PUBLISHED_RETURNS, PUBLISHED_RETURNS_SHA256),
            stage.histories,
            stage.horizon,
        )
        counts = pool_counts(pool, stage.manifest, stage.histories)
    except _STAGE_ERRORS as error:
        return {
            Check.TIES: _skipped(Check.TIES, str(error)),
            Check.POOL: _skipped(Check.POOL, str(error)),
        }
    return {Check.TIES: ties_check(ties, published), Check.POOL: pool_check(counts, POOL_COUNTS)}


def baseline_checks(manifest: SelectionManifest, paths: ForwardPaths) -> tuple[CheckOutcome, ...]:
    """Las comprobaciones de una corrida solo de líneas base: las que fijan su código.

    La etapa 1, la atadura, los recuentos del pool y los parámetros del manifiesto. No las del
    decisor —camino medido, renderizado, configuración y catálogo—, porque no se le llama.
    """
    outcomes: dict[Check, CheckOutcome] = {
        Check.MANIFEST: manifest_check(manifest, paths.reference_manifest)
    }
    stage1 = stage_check(paths.stage1)
    outcomes[Check.STAGE1] = stage1
    if not stage1.passed:
        outcomes.update(
            (check, _skipped(check, "la etapa 1 no es la publicada"))
            for check in (Check.TIES, Check.POOL)
        )
    else:
        try:
            stage = read_stage(paths.stage1, None, paths.published_history, None)
        except _STAGE_ERRORS as error:
            outcomes.update(
                (check, _skipped(check, str(error))) for check in (Check.TIES, Check.POOL)
            )
        else:
            outcomes.update(_pool_outcomes(stage))
    return tuple(outcomes[check] for check in Check if check in outcomes)


async def catalog_check(catalog: ModelCatalog, model: str) -> CheckOutcome:
    """Si el decisor registrado sigue en el catálogo de la pasarela: `GET /models`, gratis.

    Un catálogo que no se puede leer no dice nada del id: el lote se niega y se puede repetir. Uno
    que se lee y no trae el id termina el estudio en este bloque (apartado 5).
    """
    try:
        available = await catalog.available_models()
    except Exception as error:  # httpx, openai y la pasarela tienen jerarquías propias
        return CheckOutcome(
            check=Check.CATALOG,
            passed=False,
            detail=f"catálogo inaccesible: {type(error).__name__}: {error}",
        )
    if model in available:
        return CheckOutcome(
            check=Check.CATALOG,
            passed=True,
            detail=f"`{model}` está entre los {len(available)} ids de la pasarela",
        )
    return CheckOutcome(
        check=Check.CATALOG,
        passed=False,
        detail=f"`{model}` no está en el catálogo de la pasarela: el estudio termina aquí",
        ends_study=True,
    )


# ────────────────────────────────────────── El entorno ────────────────────────────────────────────


def openai_catalog(backends: Mapping[Backend, ChatBackend]) -> ModelCatalog:
    """El catálogo de la pasarela: el backend remoto ya construido."""
    backend = backends.get(Backend.OPENAI)
    if not isinstance(backend, OpenAIBackend):
        raise ConfigError("sin backend openai no hay catálogo de la pasarela que preguntar")
    return backend


def env_settings() -> Settings:
    """Los ajustes de `.env`, de donde el lote lee el decisor (apartado 3)."""
    return load_settings(DEFAULT_ENV_FILE)


class BatchEnvironment(NamedTuple):
    """Todo lo que el lote toma del mundo. Las pruebas lo sustituyen entero."""

    clock: Clock = utc_now
    settings: Callable[[], Settings] = env_settings
    backends: Callable[[Settings], Mapping[Backend, ChatBackend]] = build_backends
    catalog: Callable[[Mapping[Backend, ChatBackend]], ModelCatalog] = openai_catalog
    fetch_candles: CandleFetch = fetch_spot_candles
    fetch_funding: FundingFetch = fetch_perp_funding
    paths: ForwardPaths = ForwardPaths()
    published_history_sha256: Mapping[str, str] = PUBLISHED_HISTORY_SHA256
    published_funding_sha256: Mapping[str, str] = PUBLISHED_FUNDING_SHA256


def default_environment() -> BatchEnvironment:
    """El entorno real: reloj de pared, `.env`, la pasarela y los exchanges de verdad."""
    return BatchEnvironment()


async def gateway_check(
    env: BatchEnvironment, backends: Mapping[Backend, ChatBackend], settings: Settings
) -> CheckOutcome:
    """El catálogo con el backend ya construido, o por qué no se pudo preguntar."""
    try:
        catalog = env.catalog(backends)
    except ConfigError as error:
        return _skipped(Check.CATALOG, str(error))
    return await catalog_check(catalog, settings.role_choices(AgentRole.DECIDER)[0].model)


# ──────────────────────────────────────────── La corrida ──────────────────────────────────────────


def entry_run_id(entry: PlannedEvaluation) -> UUID:
    """El `run_id` que el replay da a una entrada del manifiesto: el de su instante evaluado."""
    return replay_run_id(
        entry.symbol, entry.timeframe, entry.at + timeframe_to_timedelta(entry.timeframe)
    )


def forward_arms() -> tuple[AblationArm, ...]:
    """`solo` y las cuatro líneas base, tal como las define la ablación."""
    by_name = {arm.name: arm for arm in ARMS}
    return tuple(by_name[name] for name in FORWARD_ARMS)


def baseline_arms() -> tuple[AblationArm, ...]:
    """Las cuatro líneas base, sin `solo`: lo que corre `baselines`."""
    return tuple(arm for arm in forward_arms() if arm.name != TARGET_ARM)


async def replay_forward_arms(
    plan: ReplayPlan,
    settings: Settings,
    cache: ResponseCache,
    clock: Clock,
    ledger: QuotaLedger,
    directory: Path,
    guard: SpendGuard | None,
    fill_with: Mapping[Backend, ChatBackend] | None,
    skip: Mapping[str, Collection[UUID]] | None = None,
    arms: Sequence[AblationArm] | None = None,
) -> dict[str, tuple[EvaluationRecord, ...]]:
    """Los cinco brazos sobre el plan, cada uno contra su journal en `directory`. Sin puntuar.

    Es la función de corrida del lote, del ensayo y de `baselines`, que pasa en `arms` solo las
    cuatro líneas base. Un solo contador y un solo tope para todos, como en la ablación. Las líneas
    base no llaman a nadie. `skip` son, por brazo, los `run_id` que ya tienen registro completo en
    una pasada anterior del lote: no se vuelven a evaluar, y un brazo sin nada que evaluar no
    escribe journal.
    """
    records: dict[str, tuple[EvaluationRecord, ...]] = {}
    for arm in forward_arms() if arms is None else arms:
        done = skip.get(arm.name, frozenset()) if skip is not None else frozenset()
        entries = tuple(entry for entry in plan.entries if entry_run_id(entry) not in done)
        if not entries:
            print(f"· {arm.name}: nada que evaluar en esta pasada", file=sys.stderr)
            records[arm.name] = ()
            continue
        print(f"· {arm.name}: {arm.question}", file=sys.stderr)
        records[arm.name] = tuple(
            await replay_arm(
                arm,
                plan._replace(entries=entries),
                settings,
                NOTIONAL_ACCOUNT,
                cache,
                clock,
                ledger,
                JsonlJournal(arm_journal_path(directory, arm.name)),
                guard,
                fill_with,
            )
        )
    return records


class SpendState(NamedTuple):
    """El tope del lote y lo que ya gastaron sus pasadas anteriores."""

    guard: SpendGuard
    cap_usd: float
    spent_before_usd: float


def spend_state(
    settings: Settings,
    manifest: SelectionManifest,
    previous: Iterable[LLMCall],
    at: datetime,
) -> SpendState:
    """El tope proporcional del lote, acumulado entre sus pasadas.

    El tope es `forward_plan.batch_spend_cap` de su manifiesto: 3 x evaluaciones x 0.01509 USD. El
    guarda sale por la puerta de la ablación (`guard_for` → `spend_guard`) y se siembra con las
    llamadas de las pasadas anteriores, de modo que lo que le queda es el tope menos lo gastado.
    """
    cap = batch_spend_cap(manifest)
    guard = guard_for(settings, forward_arms(), True, cap, at)
    if guard is None:
        raise ConfigError("el lote necesita tope de gasto y la configuración no lo dio")
    guard.add(previous)
    return SpendState(guard=guard, cap_usd=cap, spent_before_usd=guard.spent_usd)


# ──────────────────────────────────────────── El resumen ──────────────────────────────────────────


class FirstCall(FrozenModel):
    """La primera llamada viva del decisor que mostró un upstream."""

    run_id: UUID
    symbol: str
    evaluated_at: AwareDatetime
    called_at: AwareDatetime
    prompt_digest: str
    journal: str


class BatchSummary(BatchCounts):
    """Lo que el lote imprime al terminar: contadores de validez y de evaluaciones.

    No hay ningún campo de retorno, ni de resultado de una orden, y un test lo fija: durante el
    estudio no se calcula ningún retorno de `solo` (apartado 4). Los contadores y lo que es un lote
    completo son los de `BatchCounts`, los mismos que lee el análisis.
    """

    block_start: AwareDatetime
    block_number: int = Field(ge=1)
    spent_usd: float = Field(ge=0.0)
    """Gasto acumulado del lote, en el extremo alto de lo que el proveedor declaró."""

    cap_usd: float = Field(ge=0.0)
    unpriced_calls: int = Field(ge=0)
    """Llamadas respondidas sin tokens: se facturan y no suman al tope."""

    upstreams: tuple[UpstreamRow, ...] = ()
    """Quién sirvió las llamadas del decisor en la última pasada."""

    previous_label: str
    previous_upstreams: tuple[UpstreamRow, ...] = ()
    upstream_status: UpstreamStatus
    """Respecto a `previous_upstreams`: sin cambios, cambió o no verificable (`upstream_status`)."""

    filled_blocks: int = Field(ge=0)
    """Bloques del estudio con al menos una activación, hasta este incluido."""

    empty_blocks: int = Field(ge=0)
    closes_here: bool
    """Si con este bloque el estudio llega a los 50 bloques no vacíos."""

    @property
    def upstream_changed(self) -> bool:
        """Si el lote deja asiento: un par servido apareció o dejó de aparecer, y se sabe."""
        return self.upstream_status is UpstreamStatus.CHANGED


def summarise(
    block: BlockDates,
    evaluations: int,
    records: Mapping[str, Sequence[EvaluationRecord]],
    expected_backends: Mapping[str, Mapping[AgentRole, Backend]],
    guard: SpendGuard,
    previous_label: str,
    previous_upstreams: Sequence[UpstreamRow],
    counts: Sequence[int],
    passes_invalidating: Sequence[str] = (),
) -> BatchSummary:
    """Los contadores del lote, sobre los registros de su última pasada. Sin velas: no puntúa.

    `counts` son las activaciones de cada bloque del estudio hasta este, de la población: lo único
    que lee la regla de cierre. `passes_invalidating` es lo que invalida el lote y se lee de sus
    pasadas, no de sus registros (`decider_mismatches`).
    """
    tally = batch_counts(evaluations, records, expected_backends, passes_invalidating)
    upstreams = decider_rows(records.get(TARGET_ARM, ()))
    try:
        closing = closing_block(counts)
    except ValueError:
        closing = None
    return BatchSummary(
        **tally.model_dump(),
        block_start=block.start,
        block_number=block.number,
        spent_usd=guard.spent_usd,
        cap_usd=guard.cap_usd,
        unpriced_calls=guard.unpriced,
        upstreams=upstreams,
        previous_label=previous_label,
        previous_upstreams=tuple(previous_upstreams),
        upstream_status=upstream_status(upstreams, previous_upstreams),
        filled_blocks=sum(1 for count in counts if count > 0),
        empty_blocks=sum(1 for count in counts if count == 0),
        closes_here=closing == len(counts),
    )


def upstream_status(
    upstreams: Iterable[UpstreamRow], previous: Iterable[UpstreamRow]
) -> UpstreamStatus:
    """El upstream del decisor de este lote respecto al anterior.

    Una respuesta sin cabecera no dice quién la sirvió. Si el par «sin cabecera» está en los dos
    lotes y no aparece ningún par identificable nuevo, «sin cambios» afirmaría lo que no se ve: es
    no verificable, y no se prepara asiento. Un par identificable que aparece es un cambio, también
    cuando antes no había cabecera; y que deje de haberla, cuando antes la había, también.
    """
    now, before = served_pairs(upstreams), served_pairs(previous)
    headerless = any(model == NO_HEADER for model, _ in now) and any(
        model == NO_HEADER for model, _ in before
    )
    if headerless and all(model == NO_HEADER for model, _ in now - before):
        return UpstreamStatus.UNVERIFIABLE
    return UpstreamStatus.CHANGED if now != before else UpstreamStatus.UNCHANGED


def served_label(call: LLMCall) -> str:
    """La etiqueta con que `metrics.upstream_distribution` cuenta la llamada.

    Quién contestó según la cabecera; `NO_RESPONSE` si el proveedor no llegó a contestar; y
    `NO_HEADER` si contestó sin decirlo. Es la regla de ese módulo, que está congelado y no la
    expone; un test fija que las dos dan lo mismo. Sin ella, una llamada sin cabecera no se
    reconocía en el par `NO_HEADER` que sale de la tabla.
    """
    if call.failure_kind in (FailureKind.TRANSPORT, FailureKind.TIMEOUT):
        return NO_RESPONSE
    return call.upstream_model or NO_HEADER


def first_showing(chain: Sequence[RunDirectory], pair: tuple[str, str | None]) -> FirstCall | None:
    """La primera llamada viva del decisor de `solo`, en toda la cadena del lote, servida por ese
    par (modelo, endpoint), con el modelo etiquetado como en `upstream_distribution`. Una entrada
    de caché dice quién contestó antes, no cuándo."""
    found: list[tuple[datetime, FirstCall]] = []
    for run in chain:
        for arm in run.arms:
            if arm.arm != TARGET_ARM:
                continue
            for record in arm.records:
                for call in record.calls:
                    if call.role is not AgentRole.DECIDER or call.cache_hit:
                        continue
                    if (served_label(call), call.upstream_endpoint) != pair:
                        continue
                    found.append(
                        (
                            call.at,
                            FirstCall(
                                run_id=record.run_id,
                                symbol=record.symbol,
                                evaluated_at=record.at,
                                called_at=call.at,
                                prompt_digest=call.prompt_digest,
                                journal=str(arm.path),
                            ),
                        )
                    )
    return min(found, key=lambda item: item[0])[1] if found else None


def _pair_text(pair: tuple[str, str | None]) -> str:
    model, endpoint = pair
    return f"`{model}` por `{endpoint}`" if endpoint is not None else f"`{model}`, sin endpoint"


def _pair_calls(rows: Iterable[UpstreamRow], pair: tuple[str, str | None]) -> str:
    """Cuántas llamadas del lote sirvió ese par: las vivas y los aciertos de caché, contados."""
    served = [row for row in rows if (row.upstream_model, row.upstream_endpoint) == pair]
    hits = sum(row.cache_hits for row in served)
    live = sum(row.attempts for row in served) - hits
    return f"llamadas del lote: vivas {live}, aciertos de caché {hits}"


def upstream_entry(
    summary: BatchSummary, chain: Sequence[RunDirectory], today: datetime
) -> str | None:
    """El asiento preparado para el «Registro del estudio» si el upstream del decisor cambió.

    Lleva el bloque y, de cada par nuevo, cuántas de sus llamadas fueron vivas y cuántas aciertos
    de caché, y la primera viva que lo muestra. Un lote no verificable no deja asiento. No se
    escribe solo en `docs/forward.md`: lo copia el operador.
    """
    if summary.upstream_status is not UpstreamStatus.CHANGED:
        return None
    now = served_pairs(summary.upstreams)
    before = served_pairs(summary.previous_upstreams)
    lines = [
        f"### {today:%Y-%m-%d} — Cambio de upstream del decisor en el bloque "
        f"{summary.block_number}",
        "",
        f"- Bloque {summary.block_number}, que empieza el {summary.block_start:%Y-%m-%dT%H:%MZ}.",
        f"- Antes, en {summary.previous_label}: "
        + (", ".join(_pair_text(pair) for pair in sorted(before, key=str)) or "ninguno"),
        "- Ahora: " + (", ".join(_pair_text(pair) for pair in sorted(now, key=str)) or "ninguno"),
    ]
    for pair in sorted(now - before, key=str):
        counted = _pair_calls(summary.upstreams, pair)
        first = first_showing(chain, pair)
        if first is None:
            lines.append(f"- {_pair_text(pair)}: {counted}; ninguna viva lo muestra.")
            continue
        lines.append(
            f"- Primera llamada que muestra {_pair_text(pair)}: `run_id` `{first.run_id}`, "
            f"{first.symbol}, instante evaluado {first.evaluated_at:%Y-%m-%dT%H:%MZ}, "
            f"`LLMCall.at` {first.called_at.isoformat()}, en `{first.journal}`; {counted}."
        )
    for pair in sorted(before - now, key=str):
        lines.append(f"- Ya no aparece {_pair_text(pair)}.")
    lines.append(
        "- El estudio continúa (apartado 5), y el resultado se reportará también separado por "
        "upstream."
    )
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────── El `meta` ────────────────────────────────────────────


def _stamp(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def write_forward_meta(directory: Path, meta: BatchMeta) -> Path:
    """Escribe `forward.json` en la pasada. Se crea el directorio si es una pasada que se niega."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / FORWARD_META_FILE
    path.write_text(meta.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def study_git_state(cwd: Path | None = None) -> tuple[str | None, bool | None]:
    """El commit y si el árbol tiene cambios fuera de `data/forward/`, versionados o no.

    Es el `git_dirty` de `forward.json`. Entre una pasada y la siguiente del mismo lote cambian
    bajo `data/forward/` las pasadas anteriores, las series extendidas y el manifiesto, sin que
    cambie el código que corre: contarlos marcó sucia la segunda pasada del bloque 1. Un cambio
    fuera sí cuenta, esté versionado o no.

    `ablation.git_state` no cambia: el despacho congelado del análisis lo llama, y el `meta.json`
    de la pasada sigue registrando el árbol entero, que es lo que imprime `audit`.
    """
    commit, _ = git_state(cwd)
    if commit is None:
        return None, None
    try:
        done = subprocess.run(
            ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return commit, None
    if done.returncode != 0:
        return commit, None
    return commit, any(not inside_study(path) for path in changed_paths(done.stdout))


def base_meta(
    block: BlockDates,
    now: datetime,
    history: Sequence[SeriesFile] = (),
    funding: Sequence[SeriesFile] = (),
    accepted: Sequence[AcceptedRevision] = (),
) -> BatchMeta:
    """Lo que toda pasada registra de sí misma: máquina, commit, dependencias y series.

    Sale como un bloque vacío; cada camino del lote lo completa con `model_copy`. `git_dirty` no
    cuenta lo que cambia bajo `data/forward/` (`study_git_state`).
    """
    commit, dirty = study_git_state()
    lock = Path("uv.lock")
    return BatchMeta(
        block=block,
        outcome=BatchOutcome.EMPTY,
        invoked_at=now,
        hostname=socket.gethostname(),
        git_commit=commit,
        git_dirty=dirty,
        uv_lock_sha256=file_sha256(lock) if lock.is_file() else None,
        python_version=platform.python_version(),
        evaluations=0,
        spend_cap_usd=0.0,
        spent_before_usd=0.0,
        history=tuple(history),
        funding=tuple(funding),
        accepted=tuple(accepted),
    )


# ──────────────────────────────────────────── El lote ─────────────────────────────────────────────


class PreparedBlock(NamedTuple):
    """El bloque con sus series extendidas, su población y su manifiesto, sin comprobar nada."""

    block: BlockDates
    history: tuple[SeriesFile, ...]
    funding: tuple[SeriesFile, ...]
    histories: dict[str, list[list[float]]]
    counts: tuple[int, ...]
    """Activaciones de cada bloque del estudio, del 1 a este."""

    manifest: SelectionManifest | None
    manifest_file: Path | None
    accepted: tuple[AcceptedRevision, ...] = ()
    """Las velas revisadas que esta extensión aceptó por el flag."""


class BatchResult(NamedTuple):
    """Lo que el comando imprime y con qué código sale."""

    code: int
    lines: tuple[str, ...]
    files: tuple[str, ...] = ()
    """Los archivos que hay que versionar, los mismos que imprime `lines`: lo único que la
    envoltura del temporizador añade a un commit (`forward_service`)."""

    upstream: UpstreamStatus | None = None
    """El upstream del decisor respecto al lote anterior, si la pasada corrió el decisor. `auto`
    avisa como normal de uno no verificable."""


async def prepare_block(
    start: datetime, env: BatchEnvironment, accept: Collection[CandleRef] = ()
) -> PreparedBlock:
    """Extiende las series, calcula la población de los bloques 1 a este y escribe el manifiesto.

    Nada de aquí llama a un modelo. El manifiesto se escribe la primera vez y se comprueba en cada
    reanudación. Un bloque vacío no lo tiene. Si el estudio ya cerró en un bloque anterior, se niega
    antes de escribir el manifiesto (`StudyClosedError`). `accept` son las velas revisadas que el
    operador acepta con el flag; una revisada sin aceptar levanta `RevisedCandlesError`, y una vela
    nueva que falta, `MissingCandlesError`.
    """
    block = block_dates(start)
    paths = env.paths
    now = env.clock()
    extension = await extend_history(
        block.ready,
        env.fetch_candles,
        now=now,
        accept=accept,
        accepted_at=now,
        block=start,
        directory=paths.history,
        published=paths.published_history,
        published_sha256=env.published_history_sha256,
        revisions=paths.revisions,
    )
    history = extension.files
    funding = await extend_funding(
        block.end + FUNDING_TAIL,
        env.fetch_funding,
        directory=paths.funding,
        published=paths.published_funding,
        published_sha256=env.published_funding_sha256,
    )
    histories = read_histories(paths.history)
    starts = [FIRST_BLOCK + position * STEP for position in range(block.number)]
    population = study_population(histories, starts)
    counts = tuple(len(population.blocks[moment]) for moment in starts)
    try:
        closed = closing_block(counts[:-1])
    except ValueError:
        closed = None
    if closed is not None:
        raise StudyClosedError(
            f"el estudio ya cerró en el bloque {closed}, con {TARGET_BLOCKS} bloques no vacíos: "
            "este bloque no es parte del estudio"
        )
    manifest = block_manifest(histories, population, start, env.clock())
    if manifest is None:
        return PreparedBlock(
            block, history, funding, histories, counts, None, None, extension.accepted
        )
    path = manifest_path(start, paths.manifests)
    written = write_block_manifest(manifest, path)
    return PreparedBlock(
        block, history, funding, histories, counts, written, path, extension.accepted
    )


def previous_upstreams(start: datetime, paths: ForwardPaths) -> tuple[str, tuple[UpstreamRow, ...]]:
    """Los upstreams del decisor del lote anterior: el del bloque anterior más reciente que corrió,
    juntando sus pasadas. Para el bloque 1, o sin lotes antes, la etapa 1."""
    if paths.batches.is_dir():
        earlier = sorted(
            (path for path in paths.batches.iterdir() if path.name < f"{start:%Y-%m-%d}"),
            reverse=True,
        )
        for block_dir in earlier:
            if batch_passes(block_dir):
                solo = batch_records(block_dir).get(TARGET_ARM, ())
                return f"el lote del bloque que empieza el {block_dir.name}", decider_rows(solo)
    solo_path = arm_journal_path(paths.stage1, TARGET_ARM)
    verify_stage1(paths.stage1)
    run = read_run(paths.stage1)
    solo = next((arm.records for arm in run.arms if arm.arm == TARGET_ARM), ())
    return f"la etapa 1 (`{solo_path}`)", decider_rows(solo)


def files_to_version(
    prepared: PreparedBlock,
    directory: Path | None,
    revisions: Path = REVISIONS_FILE,
    fingerprints: Path | None = None,
) -> list[str]:
    """Lo que hay que versionar tras la pasada `directory`: series, revisiones aceptadas,
    manifiesto, todas las pasadas de su lote y, con `fingerprints`, todas sus huellas.

    Todas, y no solo esta: una pasada que se quedó incompleta, que se detuvo por las velas o que se
    negó es parte del lote, y el lote completo se versiona entero. Lo ya versionado y sin cambios
    no añade nada a un commit.
    """
    files = [item.path for item in (*prepared.history, *prepared.funding)]
    if prepared.accepted:
        files.append(str(revisions))
    if prepared.manifest_file is not None:
        files.append(str(prepared.manifest_file))
    if directory is not None and directory.is_dir():
        files.extend(str(path) for path in sorted(directory.parent.rglob("*")) if path.is_file())
    if fingerprints is not None:
        block = f"{prepared.block.start:%Y-%m-%d}"
        files.extend(str(path) for path in block_fingerprints(fingerprints, block))
    return files


def _check_lines(checks: Sequence[CheckOutcome]) -> list[str]:
    return [
        f"- {check.check.value}: {'sí' if check.passed else '**NO**'} — {check.detail}"
        for check in checks
    ]


def render_summary(summary: BatchSummary) -> list[str]:
    """Los contadores del lote, en Markdown. Ningún retorno."""
    upstreams = ", ".join(
        f"{row.upstream_model} / {row.upstream_endpoint or '—'} ({row.attempts} intentos, "
        f"{row.cache_hits} de caché)"
        for row in summary.upstreams
    )
    causes = ", ".join(f"{cause} {count}" for cause, count in summary.missing_by_cause.items())
    baselines = ", ".join(f"{arm} {count}" for arm, count in summary.baselines_decided.items())
    closing = (
        "con este bloque llega a los 50 no vacíos: el estudio cierra aquí"
        if summary.closes_here
        else f"faltan {TARGET_BLOCKS - summary.filled_blocks} bloques no vacíos para cerrar"
    )
    lines = [
        f"## Lote del bloque {summary.block_number} ({summary.block_start:%Y-%m-%d})",
        "",
        f"- evaluaciones del manifiesto: {summary.evaluations}",
        f"- `solo`: {summary.decided} decisiones válidas, {summary.invalid} respuestas inválidas, "
        f"{summary.missing} faltantes ({causes}), {summary.other_undecided} sin decisión por otra "
        f"causa; {summary.recorded} registros",
        f"- líneas base que decidieron: {baselines}",
        "- lo que invalida el lote: " + ("; ".join(summary.invalidating) or "nada"),
        f"- gasto acumulado del lote: {summary.spent_usd:.4f} USD (extremo alto) de un tope de "
        f"{summary.cap_usd:.4f} USD; {summary.unpriced_calls} llamadas sin tokens",
        f"- upstreams del decisor: {upstreams or 'ninguno'}",
        f"- respecto a {summary.previous_label}: " + _upstream_text(summary.upstream_status),
        "- lote completo: " + _completeness(summary),
        f"- bloques del estudio hasta este: {summary.filled_blocks} no vacíos, "
        f"{summary.empty_blocks} vacíos; {closing}",
    ]
    return lines


def _upstream_text(status: UpstreamStatus) -> str:
    if status is UpstreamStatus.CHANGED:
        return "**cambió**"
    if status is UpstreamStatus.UNVERIFIABLE:
        return (
            "**no verificable**: el decisor contestó sin cabecera en los dos lotes y no aparece "
            "ningún upstream identificable nuevo; no se prepara asiento"
        )
    return "sin cambios"


def _completeness(summary: BatchSummary) -> str:
    if summary.complete:
        return "sí"
    if summary.invalidating:
        return (
            "**NO**: el lote es inválido; se repite entero con `--supersede` antes del plazo "
            "(apartado 5)"
        )
    return "**NO**: hay que reanudar antes del plazo"


def _empty_recorded(directory: Path) -> Path | None:
    """La pasada que ya registró el bloque como vacío, si la hay."""
    if not directory.is_dir():
        return None
    for path in sorted(directory.iterdir()):
        record = path / FORWARD_META_FILE
        if record.is_file():
            meta = BatchMeta.model_validate_json(record.read_text(encoding="utf-8"))
            if meta.outcome is BatchOutcome.EMPTY:
                return path
    return None


def _version_lines(files: Sequence[str]) -> list[str]:
    return [
        "",
        "Archivos que hay que versionar (este comando no hace commits):",
        *(f"- {path}" for path in files),
    ]


def pass_cache(root: Path, batch_dir: Path, rerun: str | None = None) -> Path:
    """La caché de una pasada: la del estudio, salvo en la repetición de un lote sustituido.

    La repetición tiene la suya, `reruns/<bloque>/<pasada que sustituye>/`, nueva y vacía. La clave
    de la caché no lleva la temperatura, el timeout ni los intentos, así que con la compartida una
    repetición podría servir las respuestas de la pasada inválida. `rerun` es la pasada que va a
    sustituir; sin él, se usa la de la última marca del lote, que es la de una repetición que se
    reanuda.
    """
    if rerun is None:
        marks = [
            mark
            for path in batch_passes(batch_dir, include_superseded=True)
            if (mark := superseded(path)) is not None
        ]
        if not marks:
            return root
        rerun = max(marks, key=lambda mark: (mark.at, mark.by)).by
    return root / RERUN_CACHE / batch_dir.name / rerun


def write_superseded(passes: Sequence[Path], mark: SupersededMark) -> list[Path]:
    """Deja sustituidas esas pasadas: un `superseded.json` en cada una, sin tocar lo demás."""
    written: list[Path] = []
    for path in passes:
        target = path / SUPERSEDED_FILE
        if target.exists():
            raise ForwardError(f"{path} ya estaba sustituida")
        target.write_text(mark.model_dump_json(indent=2) + "\n", encoding="utf-8")
        written.append(target)
    return written


async def run_pass(
    prepared: PreparedBlock, env: BatchEnvironment, argv: Sequence[str], supersede: bool = False
) -> BatchResult:
    """Una pasada sobre un bloque ya preparado: comprobaciones, tope, corrida y resumen.

    La reanudación es por journal: la pasada se salta los `run_id` que ya tienen registro completo
    en las pasadas no sustituidas del lote (`batch_state`), y la caché solo sirve para los
    reintentos de lo que se vuelve a evaluar. Si el lote ya estaba completo, se imprime su resumen
    y no se abre otra pasada. Toda pasada que llega a comprobar deja su `forward.json`, también la
    que se niega: el registro anota la fecha y la causa de una negativa.

    Con `supersede`, la pasada repite entero un lote inválido (apartado 5). Se niega si el lote no
    es inválido. Si las comprobaciones pasan y queda tope, marca sustituidas todas sus pasadas justo
    antes de la primera llamada, y corre las cinco ramas desde cero con una caché propia
    (`pass_cache`). El tope sigue sumando lo que gastaron las sustituidas.
    """
    block, paths = prepared.block, env.paths
    now = env.clock()
    batch_dir = batch_directory(paths.batches, block.start)
    directory = batch_dir / _stamp(now)
    meta = base_meta(block, now, prepared.history, prepared.funding, prepared.accepted)
    fingerprints = fingerprint_root(paths)

    def version(where: Path | None) -> tuple[str, ...]:
        return tuple(files_to_version(prepared, where, paths.revisions, fingerprints))

    if prepared.manifest is None or prepared.manifest_file is None:
        recorded = _empty_recorded(batch_dir)
        if recorded is None:
            write_forward_meta(directory, meta)
            recorded = directory
        files = version(recorded)
        lines = [
            f"El bloque {block.number} no tiene activaciones: no hay manifiesto ni corrida "
            f"({recorded / FORWARD_META_FILE}).",
            f"- bloques del estudio hasta este: {sum(1 for c in prepared.counts if c)} no vacíos, "
            f"{sum(1 for c in prepared.counts if not c)} vacíos",
            *_version_lines(files),
        ]
        return BatchResult(0, tuple(lines), files)

    manifest, manifest_file = prepared.manifest, prepared.manifest_file
    manifest_sha = file_sha256(manifest_file)
    settings = env.settings()
    passes = batch_passes(batch_dir)
    chain = batch_passes(batch_dir, include_superseded=True)
    latest = chain[-1] if chain else None
    try:
        state = batch_state(passes)
    except DuplicateRecordError as error:
        return BatchResult(1, (f"error: {error}. El lote se detiene sin llamar a nadie.",))
    previous_calls = chain_calls(latest) if latest is not None else []
    arms = forward_arms()

    if passes:
        spent = spend_state(settings, manifest, previous_calls, now)
        label, before = previous_upstreams(block.start, paths)
        summary = summarise(
            block,
            len(manifest.entries),
            {arm: tuple(by_id.values()) for arm, by_id in state.records.items()},
            pass_backends(passes),
            spent.guard,
            label,
            before,
            prepared.counts,
            decider_mismatches(passes, registered_decider(read_meta(paths.stage1))),
        )
        if supersede and not summary.invalidating:
            lines = [
                f"El lote del bloque {block.number} no es inválido: no se sustituye. Un lote "
                "incompleto se reanuda sin `--supersede`, y repetir uno válido sería volver a "
                "pedir sus decisiones (apartado 5). No se llamó a ningún modelo.",
                "",
                *render_summary(summary),
            ]
            return BatchResult(1, tuple(lines))
        if not supersede and summary.complete:
            lines = [
                f"El lote del bloque {block.number} ya está completo en {batch_dir}: no se abre "
                "otra pasada.",
                "",
                *render_summary(summary),
            ]
            return BatchResult(0, tuple(lines))
        reasons = summary.invalidating
    elif supersede:
        return BatchResult(
            1,
            (
                f"error: el lote del bloque {block.number} no tiene pasadas sin sustituir: se "
                "corre sin `--supersede`. No se llamó a ningún modelo.",
            ),
        )
    else:
        reasons = ()
    if supersede:
        state = state._replace(records={}, done={})
    cache_dir = pass_cache(paths.cache, batch_dir, directory.name if supersede else None)
    cache = JsonFileResponseCache(cache_dir)

    max_attempts = router_attempts(
        replay_router(settings, QuotaLedger(settings.quota_window, env.clock), cache, env.clock)
    )
    measured = await current_path(settings)
    offline = await offline_checks(settings, max_attempts, manifest, paths, measured)
    backends = env.backends(settings)
    checks = (*offline, await gateway_check(env, backends, settings))
    meta = meta.model_copy(
        update={
            "measured_path": None if isinstance(measured, str) else measured,
            "decider": read_decider(settings, max_attempts),
            "evaluations": len(manifest.entries),
            "manifest_path": str(manifest_file),
            "manifest_sha256": manifest_sha,
            "spend_cap_usd": batch_spend_cap(manifest),
            "checks": checks,
            "resumed_from": None if latest is None else str(latest),
        }
    )
    failed = [check for check in checks if not check.passed]
    if failed:
        ends = any(check.ends_study for check in failed)
        outcome = BatchOutcome.TERMINATED if ends else BatchOutcome.REFUSED
        write_forward_meta(directory, meta.model_copy(update={"outcome": outcome}))
        headline = (
            f"El estudio termina en el bloque {block.number}: el decisor registrado ya no está en "
            "el catálogo de la pasarela (apartado 5). No se llamó a ningún modelo."
            if ends
            else f"El lote del bloque {block.number} se niega a correr: falló "
            + ", ".join(f"«{check.check.value}»" for check in failed)
            + ". No se llamó a ningún modelo. Si el camino registrado se restaura, el lote "
            "conserva su plazo (apartado 3)."
        )
        if supersede and not ends:
            headline += " Ninguna pasada quedó sustituida."
        files = version(directory)
        lines = [
            headline,
            "",
            *_check_lines(checks),
            "",
            f"Registro de la pasada: {directory / FORWARD_META_FILE}",
            *_version_lines(files),
        ]
        return BatchResult(4 if ends else 1, tuple(lines), files)

    spent = spend_state(settings, manifest, previous_calls, now)
    meta = meta.model_copy(update={"spent_before_usd": spent.spent_before_usd})
    if spent.spent_before_usd >= spent.cap_usd:
        write_forward_meta(
            directory, meta.model_copy(update={"outcome": BatchOutcome.CAP_EXHAUSTED})
        )
        files = version(directory)
        lines = [
            f"Las pasadas anteriores ya gastaron {spent.spent_before_usd:.4f} USD de un tope de "
            f"{spent.cap_usd:.4f}: no se llama a nadie. Lo que siga sin decisión al vencer el "
            "plazo cuenta como faltante.",
            "",
            *_check_lines(checks),
            *_version_lines(files),
        ]
        return BatchResult(3, tuple(lines), files)

    marks: list[Path] = []
    if supersede:
        mark = SupersededMark(at=now, by=directory.name, reasons=reasons)
        marks = write_superseded(passes, mark)
    ledger = QuotaLedger(settings.quota_window, env.clock)
    if latest is not None:
        seed_from_previous(ledger, latest, manifest_sha)
    run_meta = build_run_meta(
        settings,
        PlanKind.MANIFEST,
        manifest_file,
        manifest_sha,
        argv,
        True,
        arms,
        now,
        latest,
        manifest.horizon,
        spent.cap_usd,
    )
    open_run_directory(directory, run_meta)
    write_forward_meta(directory, meta.model_copy(update={"outcome": BatchOutcome.RAN}))
    plan = plan_from_manifest(manifest, prepared.histories)
    records = await replay_forward_arms(
        plan,
        settings,
        cache,
        env.clock,
        ledger,
        directory,
        spent.guard,
        backends,
        skip=state.done,
    )
    merged = {
        arm.name: {
            **state.records.get(arm.name, {}),
            **{record.run_id: record for record in records.get(arm.name, ())},
        }
        for arm in arms
    }
    current = batch_passes(batch_dir)
    label, before = previous_upstreams(block.start, paths)
    summary = summarise(
        block,
        len(manifest.entries),
        {arm: tuple(by_id.values()) for arm, by_id in merged.items()},
        pass_backends(current),
        spent.guard,
        label,
        before,
        prepared.counts,
        decider_mismatches(current, registered_decider(read_meta(paths.stage1))),
    )
    entry = upstream_entry(summary, run_chain(directory), now)
    if entry is not None:
        (directory / UPSTREAM_FILE).write_text(entry, encoding="utf-8")
    printed = (
        [_pass_fingerprint(directory, cache_dir, fingerprints)]
        if decider_called(records.get(TARGET_ARM, ()))
        else []
    )

    lines = [*render_summary(summary), *printed, "", *_check_lines(checks)]
    if marks:
        lines += [
            "",
            f"Pasadas sustituidas por esta, que repite entero el lote ({len(marks)}):",
            *(f"- {path}" for path in marks),
        ]
    if entry is not None:
        lines += [
            "",
            "Asiento preparado para el «Registro del estudio» (no se escribe solo; está también en "
            f"{directory / UPSTREAM_FILE}):",
            "",
            entry,
        ]
    files = version(directory)
    lines += _version_lines(files)
    return BatchResult(0 if summary.complete else 3, tuple(lines), files, summary.upstream_status)


def _pass_fingerprint(directory: Path, cache: Path, root: Path) -> str:
    """Escribe la huella del decisor de la pasada en `fingerprints/` y da su línea del resumen.

    Nunca cambia el resultado ni el código del lote: un fallo al calcularla queda escrito en el
    archivo, con su motivo, y uno al escribirlo queda en la línea.
    """
    path = pass_fingerprint_path(root, directory)
    journal = arm_journal_path(directory, TARGET_ARM)
    try:
        found = fingerprint(journal, cache)
    except Exception as error:  # la huella describe la pasada: no puede cambiar cómo termina
        found = failed_fingerprint(journal, cache, f"{type(error).__name__}: {error}")
    try:
        write_fingerprint(path, found)
    except OSError as error:
        return f"- huella del decisor: no se pudo escribir {path}: {error}"
    return fingerprint_line(found, path)


async def baselines_pass(
    prepared: PreparedBlock, env: BatchEnvironment, argv: Sequence[str]
) -> BatchResult:
    """Completa, después del plazo, los registros de las cuatro líneas base de un bloque.

    Es el apartado 5: una pasada interrumpida o un lote que no corrió dejan `run_id` del manifiesto
    sin registro de las líneas base, y el análisis los necesita para puntuar las faltantes de
    `solo` contra `rule_trend`. La pasada:

    - corre solo las líneas base que faltan, saltándose las que ya tienen registro completo;
    - repite antes las comprobaciones que fijan su código (`baseline_checks`);
    - no construye backends ni lee el catálogo, y corre con un router solo de caché sobre una
      caché vacía en memoria: una llamada levantaría `ReplayCacheMissError`;
    - comprueba al terminar que ningún registro trae una llamada, y sale con 1 si alguno la trae.

    Un bloque sin activaciones no tiene líneas base que completar. Queda registrado como vacío una
    sola vez, como en el lote, para que el temporizador lea en el disco que no le falta nada.
    """
    block, paths = prepared.block, env.paths
    now = env.clock()
    batch_dir = batch_directory(paths.batches, block.start)
    directory = batch_dir / _stamp(now)
    if prepared.manifest is None or prepared.manifest_file is None:
        recorded = _empty_recorded(batch_dir)
        if recorded is None:
            meta = base_meta(block, now, prepared.history, prepared.funding, prepared.accepted)
            write_forward_meta(directory, meta)
            recorded = directory
        files = tuple(
            files_to_version(prepared, recorded, paths.revisions, fingerprint_root(paths))
        )
        lines = [
            f"El bloque {block.number} no tiene activaciones: no hay líneas base que completar.",
            f"Registro del bloque vacío: {recorded / FORWARD_META_FILE}",
            *_version_lines(files),
        ]
        return BatchResult(0, tuple(lines), files)
    manifest, manifest_file = prepared.manifest, prepared.manifest_file
    chain = batch_passes(batch_dir, include_superseded=True)
    try:
        state = batch_state(batch_passes(batch_dir))
    except DuplicateRecordError as error:
        return BatchResult(1, (f"error: {error}. No se completa nada.",))
    arms = baseline_arms()
    expected = {entry_run_id(entry) for entry in manifest.entries}
    pending = {arm.name: expected - state.done.get(arm.name, set()) for arm in arms}
    if not any(pending.values()):
        return BatchResult(
            0,
            (
                f"Las cuatro líneas base del bloque {block.number} ya tienen registro completo de "
                f"sus {len(expected)} evaluaciones: no se abre otra pasada.",
            ),
        )

    def version(where: Path | None) -> tuple[str, ...]:
        return tuple(files_to_version(prepared, where, paths.revisions, fingerprint_root(paths)))

    checks = baseline_checks(manifest, paths)
    meta = base_meta(block, now, prepared.history, prepared.funding, prepared.accepted).model_copy(
        update={
            "evaluations": len(manifest.entries),
            "manifest_path": str(manifest_file),
            "manifest_sha256": file_sha256(manifest_file),
            "checks": checks,
            "resumed_from": None if not chain else str(chain[-1]),
        }
    )
    failed = [check for check in checks if not check.passed]
    if failed:
        write_forward_meta(directory, meta.model_copy(update={"outcome": BatchOutcome.REFUSED}))
        files = version(directory)
        lines = [
            f"Las líneas base del bloque {block.number} no se completan: falló "
            + ", ".join(f"«{check.check.value}»" for check in failed)
            + ". No se escribió ningún registro.",
            "",
            *_check_lines(checks),
            "",
            f"Registro de la pasada: {directory / FORWARD_META_FILE}",
            *_version_lines(files),
        ]
        return BatchResult(1, tuple(lines), files)

    settings = env.settings()
    open_run_directory(
        directory,
        build_run_meta(
            settings,
            PlanKind.MANIFEST,
            manifest_file,
            file_sha256(manifest_file),
            argv,
            False,
            arms,
            now,
            chain[-1] if chain else None,
            manifest.horizon,
            None,
        ),
    )
    write_forward_meta(directory, meta.model_copy(update={"outcome": BatchOutcome.BASELINES}))
    clock = env.clock
    records = await replay_forward_arms(
        plan_from_manifest(manifest, prepared.histories),
        settings,
        InMemoryResponseCache(),
        clock,
        QuotaLedger(settings.quota_window, clock),
        directory,
        None,
        None,
        skip=state.done,
        arms=arms,
    )
    calls = sum(len(record.calls) for items in records.values() for record in items)
    whole = batch_state(batch_passes(batch_dir))
    rows = [
        f"- {arm.name}: {len(records.get(arm.name, ()))} registros nuevos; con registro completo "
        f"{len(whole.done.get(arm.name, set()) & expected)} de {len(expected)}"
        for arm in arms
    ]
    done = all(expected <= whole.done.get(arm.name, set()) for arm in arms)
    files = version(directory)
    lines = [
        f"## Líneas base del bloque {block.number} ({block.start:%Y-%m-%d}), después del plazo",
        "",
        *rows,
        f"- llamadas a un modelo: {calls}",
        "",
        *_check_lines(checks),
        *_version_lines(files),
    ]
    if calls:
        lines.insert(
            0,
            f"**Una línea base llamó a un modelo**: {calls} llamadas en {directory}. El análisis "
            "rechazará esta pasada.\n",
        )
        return BatchResult(1, tuple(lines), files)
    return BatchResult(0 if done else 3, tuple(lines), files)


def _values(values: tuple[float, ...] | None) -> str:
    if values is None:
        return "ya no la devuelve"
    names = ("apertura", "máximo", "mínimo", "cierre", "volumen")
    return ", ".join(f"{name} {value!r}" for name, value in zip(names, values, strict=True))


def _flag(item: RevisedCandle | MissingCandle) -> str:
    return f"{item.symbol}@{item.opens.astimezone(UTC):%Y-%m-%dT%H:%MZ}"


def revision_entry(block: BlockDates, revised: Sequence[RevisedCandle], today: datetime) -> str:
    """El asiento preparado para el registro: símbolo, apertura, valor guardado y devuelto de cada
    vela revisada (apartado 4, «Vela revisada»). Lo copia el operador."""
    lines = [
        f"### {today:%Y-%m-%d} — Velas revisadas por el exchange, lote del bloque {block.number}",
        "",
        "La extensión se detuvo. Se continúa con el valor guardado, con `--accept-revised` para "
        "cada vela; la serie del estudio es la que se guardó.",
        "",
    ]
    for item in revised:
        lines += [
            f"- **{item.symbol}**, vela que abre el {item.opens.astimezone(UTC):%Y-%m-%dT%H:%MZ}:",
            f"  - guardado: {_values(item.stored)};",
            f"  - devuelto: {_values(item.returned)}.",
        ]
    return "\n".join(lines) + "\n"


def _gap_sightings(directory: Path) -> dict[tuple[str, datetime], datetime]:
    """Cuándo se vio faltar por primera vez cada vela, según las pasadas del lote que lo
    registraron."""
    first: dict[tuple[str, datetime], datetime] = {}
    if not directory.is_dir():
        return first
    for path in sorted(directory.iterdir()):
        record = path / FORWARD_META_FILE
        if not record.is_file():
            continue
        meta = BatchMeta.model_validate_json(record.read_text(encoding="utf-8"))
        for item in meta.missing:
            key = (item.symbol, item.opens)
            first[key] = min(first.get(key, meta.invoked_at), meta.invoked_at)
    return first


def revised_result(
    block: BlockDates, env: BatchEnvironment, error: RevisedCandlesError
) -> BatchResult:
    """La pasada que se detiene por velas revisadas: su registro, el asiento preparado y el flag."""
    now = env.clock()
    directory = batch_directory(env.paths.batches, block.start) / _stamp(now)
    meta = base_meta(block, now).model_copy(
        update={"outcome": BatchOutcome.REVISED, "revised": error.revised}
    )
    write_forward_meta(directory, meta)
    flags = " ".join(f"--accept-revised {_flag(item)}" for item in error.revised)
    lines = [
        f"El lote del bloque {block.number} se detiene: el exchange devuelve distintas "
        f"{len(error.revised)} velas ya guardadas (apartado 4, «Vela revisada»). No se escribió "
        "ninguna serie y no se llamó a ningún modelo.",
        "",
        "Asiento preparado para el «Registro del estudio» (no se escribe solo):",
        "",
        revision_entry(block, error.revised, now),
        "Con el asiento en el registro, se continúa con el valor guardado:",
        "",
        f"    {_COMMAND} batch --block {block.start:%Y-%m-%d} {flags}",
        "",
        f"Registro de la pasada: {directory / FORWARD_META_FILE}",
    ]
    return BatchResult(5, tuple(lines))


def gap_result(block: BlockDates, env: BatchEnvironment, error: MissingCandlesError) -> BatchResult:
    """La pasada que encuentra velas nuevas que faltan (apartado 4, «Hueco persistente»).

    La primera vez se registra cuándo faltaban y hay que volver a descargar 24 h después (código
    6). Si alguna sigue faltando cuando ya pasaron 24 h desde que se vio faltar por primera vez, el
    hueco es persistente y el estudio termina en este bloque (código 7).
    """
    now = env.clock()
    batch_dir = batch_directory(env.paths.batches, block.start)
    directory = batch_dir / _stamp(now)
    seen = _gap_sightings(batch_dir)
    first = {
        (item.symbol, item.opens): seen.get((item.symbol, item.opens), now)
        for item in error.missing
    }
    persistent = [
        item for item in error.missing if now - first[(item.symbol, item.opens)] >= GAP_WAIT
    ]
    outcome = BatchOutcome.GAP_PERSISTENT if persistent else BatchOutcome.GAP
    meta = base_meta(block, now).model_copy(update={"outcome": outcome, "missing": error.missing})
    write_forward_meta(directory, meta)
    listed = [
        f"- {_flag(item)}, falta desde {first[(item.symbol, item.opens)].isoformat()}"
        for item in error.missing
    ]
    if persistent:
        lines = [
            f"El estudio termina en el bloque {block.number}: {len(persistent)} velas nuevas "
            "siguen faltando al volver a descargar 24 h después (apartado 4, «Hueco "
            "persistente», y apartado 5). No se escribió ninguna serie y no se llamó a ningún "
            "modelo.",
            "",
            *listed,
            "",
            f"Registro de la pasada: {directory / FORWARD_META_FILE}",
        ]
        return BatchResult(7, tuple(lines))
    retry = min(first.values()) + GAP_WAIT
    late = (
        f" Ojo: el plazo del lote vence el {block.deadline.isoformat()}, antes de poder volver a "
        "descargar."
        if retry >= block.deadline
        else ""
    )
    lines = [
        f"Faltan {len(error.missing)} velas nuevas en la descarga del bloque {block.number}. No se "
        "escribió ninguna serie y no se llamó a ningún modelo. Vuelve a correr el lote desde "
        f"{retry.isoformat()}: si siguen faltando, el estudio termina en este bloque.{late}",
        "",
        *listed,
        "",
        f"Registro de la pasada: {directory / FORWARD_META_FILE}",
    ]
    return BatchResult(6, tuple(lines))


async def run_batch(
    start: datetime,
    env: BatchEnvironment,
    argv: Sequence[str],
    accept: Collection[CandleRef] = (),
    supersede: bool = False,
) -> BatchResult:
    """El lote del bloque que empieza en `start`: fechas, datos, comprobaciones y corrida.

    `accept` son las velas revisadas que el operador acepta con `--accept-revised`; `supersede`
    repite entero un lote inválido (`run_pass`).
    """
    refusal = date_refusal(start, env.clock())
    if refusal is not None:
        return BatchResult(2, (f"error: {refusal}",))
    prepared = await _prepared(start, env, accept)
    if isinstance(prepared, BatchResult):
        return prepared
    return await run_pass(prepared, env, argv, supersede)


async def run_baselines(
    start: datetime,
    env: BatchEnvironment,
    argv: Sequence[str],
    accept: Collection[CandleRef] = (),
) -> BatchResult:
    """Las líneas base del bloque que empieza en `start`, después de su plazo (`baselines_pass`).

    Prepara el bloque como el lote —extensión, población y manifiesto, sin modelos— porque un lote
    que no corrió puede no tener manifiesto, ni velas si era el último.
    """
    refusal = baselines_date_refusal(start, env.clock())
    if refusal is not None:
        return BatchResult(2, (f"error: {refusal}",))
    prepared = await _prepared(start, env, accept)
    if isinstance(prepared, BatchResult):
        return prepared
    return await baselines_pass(prepared, env, argv)


COMMAND_ERRORS = (
    AuditError,
    ConfigError,
    ForwardDataError,
    ForwardError,
    MarketDataError,
    ProbeError,
    ReplayCacheMissError,
    SelectionError,
)
"""Lo que el comando convierte en «error: …» y código 1: `main` y el lote de cada bloque de
`auto`, que así sale con el mismo código que `batch --block`."""


def batch_runner(
    env: BatchEnvironment, argv: Sequence[str]
) -> Callable[[datetime], Awaitable[BatchResult]]:
    """El lote de cada bloque pendiente de `auto`: `run_batch` con su entorno y su argv, y nada más.

    Sin velas aceptadas y sin sustituir pasadas: las dos decisiones son del operador (apartados 4
    y 5), y aquí no hay por dónde pasarlas. Un test de arquitectura fija que la llamada lleva
    exactamente esos tres argumentos.
    """

    async def run(start: datetime) -> BatchResult:
        try:
            return await run_batch(start, env, argv)
        except COMMAND_ERRORS as error:
            return BatchResult(1, (f"error: {error}",))

    return run


async def _prepared(
    start: datetime, env: BatchEnvironment, accept: Collection[CandleRef]
) -> PreparedBlock | BatchResult:
    """El bloque preparado, o lo que se imprime si las velas o el cierre lo detienen."""
    block = block_dates(start)
    try:
        return await prepare_block(start, env, accept)
    except StudyClosedError as error:
        return BatchResult(2, (f"error: {error}",))
    except RevisedCandlesError as error:
        return revised_result(block, env, error)
    except MissingCandlesError as error:
        return gap_result(block, env, error)


# ───────────────────────────────────────────── El ensayo ──────────────────────────────────────────


class ArmComparison(FrozenModel):
    """Un brazo del ensayo contra el mismo brazo de la etapa 1."""

    arm: str
    ours: int = Field(ge=0)
    theirs: int = Field(ge=0)
    shared: int = Field(ge=0)
    actions: int = Field(ge=0)
    """Evaluaciones compartidas con la misma acción."""

    stops: int = Field(ge=0)
    """Evaluaciones compartidas con el mismo stop declarado."""

    digests: int = Field(ge=0)
    """Evaluaciones compartidas con los mismos `prompt_digest`, rol a rol y en orden."""

    live_calls: int = Field(ge=0)
    """Llamadas del ensayo que no salieron de la caché. Tiene que ser 0."""

    @property
    def exact(self) -> bool:
        return (
            self.ours == self.theirs == self.shared
            and self.actions == self.stops == self.digests == self.shared
            and self.live_calls == 0
        )


def _action(record: EvaluationRecord) -> str | None:
    proposed = record.proposed
    return None if proposed is None else proposed.action.value


def _stop(record: EvaluationRecord) -> float | None:
    proposed = record.proposed
    return None if proposed is None else proposed.invalidation_price


def _digests(record: EvaluationRecord) -> list[tuple[AgentRole, str]]:
    return [(call.role, call.prompt_digest) for call in record.calls]


def compare_arms(
    ours: Mapping[str, Sequence[EvaluationRecord]],
    theirs: Mapping[str, Sequence[EvaluationRecord]],
) -> tuple[ArmComparison, ...]:
    """Cada brazo del ensayo contra la etapa 1, por `run_id`: acción, stop y `prompt_digest`."""
    comparisons: list[ArmComparison] = []
    for arm in FORWARD_ARMS:
        mine = {record.run_id: record for record in ours.get(arm, ())}
        stage = {record.run_id: record for record in theirs.get(arm, ())}
        shared = mine.keys() & stage.keys()
        comparisons.append(
            ArmComparison(
                arm=arm,
                ours=len(mine),
                theirs=len(stage),
                shared=len(shared),
                actions=sum(1 for key in shared if _action(mine[key]) == _action(stage[key])),
                stops=sum(1 for key in shared if _stop(mine[key]) == _stop(stage[key])),
                digests=sum(1 for key in shared if _digests(mine[key]) == _digests(stage[key])),
                live_calls=sum(
                    1 for record in mine.values() for call in record.calls if not call.cache_hit
                ),
            )
        )
    return tuple(comparisons)


async def rehearse(
    env: BatchEnvironment, cache_dir: Path, out: Path, argv: Sequence[str]
) -> BatchResult:
    """El ensayo sin pagar: la etapa 1 por el código del lote, solo con caché.

    Las mismas comprobaciones previas, catálogo incluido, y la misma función de corrida, sin
    `fill_with`: cada ranura es `CacheOnlyBackend` y la caché se lee sin tocarla. Un hueco de caché
    detiene el ensayo nombrando lo que falta. Al final, decisiones, stops y `prompt_digest` contra
    `data/stage1/`, y las llamadas vivas, que tienen que ser cero.
    """
    paths = env.paths
    if out.exists() and any(out.iterdir()):
        return BatchResult(2, (f"error: {out} ya contiene algo: el ensayo escribe en uno nuevo",))
    if not cache_dir.is_dir():
        return BatchResult(2, (f"error: no existe la caché {cache_dir}",))
    settings = env.settings()
    reference = load_manifest(paths.reference_manifest)
    histories = load_selection_histories(reference, paths.published_history)
    cache = JsonFileResponseCache(cache_dir)
    clock = env.clock
    max_attempts = router_attempts(
        replay_router(settings, QuotaLedger(settings.quota_window, clock), cache, clock)
    )
    measured = await current_path(settings)
    offline = await offline_checks(settings, max_attempts, reference, paths, measured)
    checks = (*offline, await gateway_check(env, env.backends(settings), settings))

    open_run_directory(
        out,
        build_run_meta(
            settings,
            PlanKind.MANIFEST,
            paths.reference_manifest,
            file_sha256(paths.reference_manifest),
            argv,
            False,
            forward_arms(),
            clock(),
            None,
            reference.horizon,
            None,
        ),
    )
    plan = plan_from_manifest(reference, histories)
    try:
        records = await replay_forward_arms(
            plan, settings, cache, clock, QuotaLedger(settings.quota_window, clock), out, None, None
        )
    except ReplayCacheMissError as error:
        return BatchResult(1, (*_check_lines(checks), f"error: {error}"))
    stage = read_run(paths.stage1)
    comparisons = compare_arms(records, {arm.arm: arm.records for arm in stage.arms})
    lines = [
        "## Ensayo del lote sobre la etapa 1, solo con caché",
        "",
        f"- caché: `{cache_dir}`; corrida en `{out}`",
        "",
        *_check_lines(checks),
        "",
        "| brazo | ensayo | etapa 1 | en los dos | misma acción | mismo stop | mismos digests "
        "| llamadas vivas | coincide |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        *(
            f"| {item.arm} | {item.ours} | {item.theirs} | {item.shared} | {item.actions} | "
            f"{item.stops} | {item.digests} | {item.live_calls} | "
            f"{'sí' if item.exact else '**NO**'} |"
            for item in comparisons
        ),
    ]
    passed = all(item.exact for item in comparisons) and all(check.passed for check in checks)
    return BatchResult(0 if passed else 1, tuple(lines))


# ──────────────────────────────────────────── El comando ──────────────────────────────────────────


def _echo(lines: Sequence[str]) -> None:
    print("\n".join(lines), flush=True)


def write_auto_report(path: Path, report: AutoReport) -> Path:
    """El informe de `auto` en JSON, que la envoltura del servicio lee para versionar y avisar.

    Sin avisos normales no lleva la clave `notices`: la envoltura se cargó antes de su `git pull`,
    y si su modelo es anterior a ese campo, una clave que no conoce le dejaría sin informe.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    empty = {"notices"} if not report.notices else None
    path.write_text(report.model_dump_json(indent=2, exclude=empty) + "\n", encoding="utf-8")
    return path


def auto_command(
    env: BatchEnvironment, argv: Sequence[str], dry_run: bool, report: Path | None
) -> int:
    """`auto`: los lotes pendientes desde el bloque 2, como los lanza el temporizador.

    Con `dry_run` dice qué haría ahora y nada más. Lee el disco con el reloj del entorno: sin red,
    sin backends, sin escribir y sin tomar el cerrojo. Sin `dry_run`, toma el cerrojo y corre cada
    bloque pendiente con `batch_runner` (`forward_auto.run_auto`). Un error que impide empezar,
    como el cerrojo tomado o un journal que no se puede leer, queda en el informe como un aviso;
    el ensayo lee el estudio dentro del mismo `try` y avisa igual. Sale con 0 si nada necesita al
    operador y con 1 si algo sí. El estudio cerrado, o el bloque 52 terminado sin cerrar, no lo
    necesita: va en `AutoReport.closing`, no en los avisos, y con él solo sale con 0.
    """
    paths = study_paths(env.paths)
    try:
        if dry_run:
            print("\n".join(render_state(study_state(paths, env.paths.stage1, env.clock()))))
            return 0
        with batch_lock(env.paths.lock):
            found = asyncio.run(
                run_auto(paths, env.paths.stage1, env.clock, batch_runner(env, argv), _echo)
            )
    except COMMAND_ERRORS as error:
        print(f"error: {error}", file=sys.stderr)
        found = AutoReport(at=env.clock(), alerts=(f"auto no corrió ningún lote: {error}",))
    if report is not None:
        write_auto_report(report, found)
    print("\n".join(render_report(found)))
    return 1 if found.needs_operator else 0


def _block_start(text: str) -> datetime:
    """Un inicio de bloque en ISO 8601. Sin zona horaria se entiende UTC."""
    try:
        moment = datetime.fromisoformat(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"no es una fecha ISO 8601: {text!r}") from error
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)


def _candle_ref(text: str) -> CandleRef:
    """`SÍMBOLO@APERTURA`, con la apertura en ISO 8601; sin zona horaria se entiende UTC."""
    symbol, separator, moment = text.partition("@")
    if not separator or not symbol:
        raise argparse.ArgumentTypeError(f"se esperaba SÍMBOLO@APERTURA: {text!r}")
    return CandleRef(symbol=symbol, opens=_block_start(moment.replace("Z", "+00:00")))


def main(argv: Sequence[str] | None = None, env: BatchEnvironment | None = None) -> int:
    """Punto de entrada: `python -m crypto_agents.forward
    batch|baselines|rehearse|analyze|rehearse-analysis|auto|fingerprint`.

    `batch` sale con:
    - 0 si el lote está completo, o el bloque vacío;
    - 1 si se negó por una comprobación o por los datos, sin llamar a nadie;
    - 2 por la fecha o el uso;
    - 3 si corrió y quedó incompleto;
    - 4 si el decisor ya no está en el catálogo;
    - 5 si el exchange devuelve distinta una vela guardada que nadie aceptó;
    - 6 si falta una vela nueva y hay que volver a descargar 24 h después;
    - 7 si sigue faltando 24 h después: el estudio termina en ese bloque.

    `batch --supersede` se niega con 1 si el lote no es inválido.

    `baselines` sale con 0 si las cuatro líneas base quedan con registro completo, o no había nada
    que completar; con 1 si se negó por una comprobación o una línea base llamó a un modelo; con 2
    por la fecha —antes del plazo— o el uso; con 3 si alguna quedó sin registro completo; y con 5,
    6 o 7 por las velas, como el lote.

    `rehearse` sale con 0 si coinciden las cinco ramas con cero llamadas vivas y pasan las
    comprobaciones.

    `analyze` sale con 0 si da veredicto, «no determinado» incluido, con 1 si se niega y con 2 si
    no pudo leer el estudio. `rehearse-analysis` sale con 0 si las cifras de la etapa 1 son las
    publicadas y con 1 si no. Ninguno de los dos construye backends.

    `auto` sale con 0 si nada necesita al operador, con 1 si algo sí y con 2 por el uso
    (`auto_command`). `batch`, `baselines` y `auto` comparten un cerrojo: con otro corriendo,
    salen con 1 sin tocar nada. `batch` y `baselines` lo toman después de la negativa por fecha.

    `fingerprint` sale con 0, y con 1 si alguna huella que escribió no se pudo calcular o tiene
    llamadas sin orden de claves. No construye backends ni lee `.env`. Toma el cerrojo de los lotes:
    con un lote en marcha escribiría la huella de una pasada a medias, y después el lote no podría
    escribir la suya, porque una huella nunca se sobrescribe. Con el cerrojo tomado sale con 1 sin
    escribir nada.
    """
    parser = argparse.ArgumentParser(
        prog=_COMMAND, description="Los lotes del estudio hacia adelante de solo."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    batch = commands.add_parser("batch", help="corre el lote de un bloque")
    batch.add_argument("--block", type=_block_start, required=True, help="inicio del bloque")
    batch.add_argument(
        "--accept-revised",
        type=_candle_ref,
        action="append",
        default=[],
        metavar="SÍMBOLO@APERTURA",
        help=(
            "acepta una vela revisada, ya anotada en el registro, y continúa con su valor "
            "guardado (por ejemplo ADA/USDT@2026-08-15T20:00Z); se repite por vela"
        ),
    )
    batch.add_argument(
        "--supersede",
        action="store_true",
        help=(
            "repite entero un lote inválido: marca sustituidas sus pasadas y corre una nueva "
            "(apartado 5); se niega si el lote no es inválido"
        ),
    )
    baselines = commands.add_parser(
        "baselines",
        help="completa después del plazo las líneas base de un bloque, sin llamar a ningún modelo",
    )
    baselines.add_argument("--block", type=_block_start, required=True, help="inicio del bloque")
    baselines.add_argument(
        "--accept-revised",
        type=_candle_ref,
        action="append",
        default=[],
        metavar="SÍMBOLO@APERTURA",
        help="acepta una vela revisada, como en el lote",
    )
    rehearsal = commands.add_parser("rehearse", help="ensayo sin pagar sobre la etapa 1")
    rehearsal.add_argument("--cache", type=Path, required=True, help="caché de la etapa 1")
    rehearsal.add_argument("--out", type=Path, required=True, help="directorio nuevo")
    analysis = commands.add_parser(
        "analyze", help="el análisis del estudio, una sola vez, cuando cierra (apartado 6)"
    )
    analysis.add_argument("directory", type=Path, help="directorio del estudio: data/forward")
    commands.add_parser(
        "rehearse-analysis", help="ensayo del análisis sobre data/stage1/, sin veredicto"
    )
    automatic = commands.add_parser(
        "auto", help="corre los lotes pendientes, como los lanza el temporizador, desde el bloque 2"
    )
    mode = automatic.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true", help="dice qué haría ahora, sin correr ni escribir nada"
    )
    mode.add_argument(
        "--report", type=Path, help="escribe lo que hizo en JSON, para la envoltura del servicio"
    )
    prints = commands.add_parser(
        "fingerprint",
        help="la huella del decisor de la etapa 1 y de las pasadas que todavía no la tienen",
    )
    prints.add_argument(
        "--stage1-cache", type=Path, default=STAGE1_CACHE, help="caché de la etapa 1"
    )
    args = parser.parse_args(argv)
    raw = tuple(argv) if argv is not None else tuple(sys.argv[1:])
    if args.command in ("analyze", "rehearse-analysis"):
        clock = env.clock if env is not None else utc_now
        found = (
            analyze(args.directory, clock(), git_state())
            if args.command == "analyze"
            else rehearse_analysis(git_state())
        )
        print("\n".join(found.lines), file=sys.stderr if found.code == 2 else sys.stdout)
        return found.code
    environment = env if env is not None else default_environment()
    if args.command == "auto":
        return auto_command(environment, raw, args.dry_run, args.report)

    try:
        if args.command == "batch":
            refusal = date_refusal(args.block, environment.clock())
            if refusal is not None:
                result = BatchResult(2, (f"error: {refusal}",))
            else:
                with batch_lock(environment.paths.lock):
                    result = asyncio.run(
                        run_batch(args.block, environment, raw, args.accept_revised, args.supersede)
                    )
        elif args.command == "baselines":
            refusal = baselines_date_refusal(args.block, environment.clock())
            if refusal is not None:
                result = BatchResult(2, (f"error: {refusal}",))
            else:
                with batch_lock(environment.paths.lock):
                    result = asyncio.run(
                        run_baselines(args.block, environment, raw, args.accept_revised)
                    )
        elif args.command == "fingerprint":
            paths = environment.paths
            with batch_lock(paths.lock):
                backfill = retroactive(
                    fingerprint_root(paths),
                    paths.stage1,
                    args.stage1_cache,
                    paths.batches,
                    paths.cache,
                )
            result = BatchResult(backfill.code, backfill.lines)
        else:
            result = asyncio.run(rehearse(environment, args.cache, args.out, raw))
    except COMMAND_ERRORS as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    stream = sys.stderr if result.code == 2 else sys.stdout
    print("\n".join(result.lines), file=stream)
    return result.code


if __name__ == "__main__":
    raise SystemExit(main())
