"""Los lotes del estudio hacia adelante de `solo`: lo que `docs/forward.md` pide en su apartado 10.

    python -m crypto_agents.forward batch --block 2026-10-22
    python -m crypto_agents.forward rehearse --cache var/ablation-cache --out <directorio nuevo>

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
6. Un resumen de contadores y la lista de archivos que hay que versionar. Este comando no hace
   commits.

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

`rehearse` es el ensayo sin pagar. Pasa el manifiesto de la etapa 1 por las mismas comprobaciones y
la misma función de corrida, solo con caché, y compara decisiones, stops y `prompt_digest` con
`data/stage1/`.
"""

from __future__ import annotations

import argparse
import asyncio
import platform
import socket
import sys
from datetime import UTC, datetime, timedelta
from enum import StrEnum
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
    META_FILE,
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
from crypto_agents.cache import JsonFileResponseCache
from crypto_agents.context import utc_now
from crypto_agents.criteria import CLUSTER_DAYS, CriteriaError, check_validity
from crypto_agents.design_effect import DesignEffectError, pool_records, stage_ties
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
from crypto_agents.forward_plan import (
    BATCH_DEADLINE,
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
    first_block,
    pool_counts,
    published_returns,
    published_ties,
)
from crypto_agents.forward_render import (
    REGISTERED_PATH,
    ForwardRenderError,
    MeasuredPath,
    check_run,
    measured_path,
)
from crypto_agents.funding import FUNDING_DIR
from crypto_agents.journal import JsonlJournal
from crypto_agents.llm import OpenAIBackend, build_backends
from crypto_agents.market import MarketDataError, timeframe_to_timedelta
from crypto_agents.metrics import (
    EPOCH,
    NO_RESPONSE,
    AbortKind,
    UpstreamRow,
    undecided_causes,
    upstream_distribution,
)
from crypto_agents.paired_dispersion import RecalculationError, read_stage
from crypto_agents.perp_probe import ProbeError
from crypto_agents.quota import QuotaLedger
from crypto_agents.replay import ReplayCacheMissError, replay_router, replay_run_id
from crypto_agents.selection import (
    DEFAULT_HORIZON,
    DEFAULT_MANIFEST,
    DEFAULT_TIMEFRAME,
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
    FrozenModel,
    StructuredOutputMode,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Mapping, Sequence

    from crypto_agents.ablation import AblationArm, ReplayPlan
    from crypto_agents.audit import RunMeta
    from crypto_agents.cache import ResponseCache
    from crypto_agents.design_effect import StageTie
    from crypto_agents.journal import EvaluationRecord
    from crypto_agents.llm import ChatBackend, ModelCatalog, ModelRouter
    from crypto_agents.quota import Clock
    from crypto_agents.selection import PlannedEvaluation
    from crypto_agents.settings import Settings
    from crypto_agents.spend import SpendGuard
    from crypto_agents.state import LLMCall

__all__ = [
    "FIRST_BLOCK",
    "FORWARD_META_FILE",
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
    "FirstCall",
    "ForwardPaths",
    "PreparedBlock",
    "block_dates",
    "catalog_check",
    "compare_arms",
    "date_refusal",
    "decider_check",
    "default_environment",
    "first_showing",
    "main",
    "manifest_check",
    "offline_checks",
    "pool_check",
    "prepare_block",
    "read_decider",
    "registered_decider",
    "rehearse",
    "replay_forward_arms",
    "router_attempts",
    "run_batch",
    "run_pass",
    "spend_state",
    "summarise",
    "ties_check",
    "upstream_entry",
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

UPSTREAM_FILE = "upstream.md"
"""El asiento preparado para el registro cuando cambia el upstream del decisor. No se escribe solo
en `docs/forward.md`."""

CACHE_DIR = Path("var/forward/cache")
"""La caché propia del estudio. No se versiona: los journals guardan cada llamada."""

TARGET_ARM = "solo"
STEP = timedelta(days=CLUSTER_DAYS)
WAIT = DEFAULT_HORIZON * timeframe_to_timedelta(DEFAULT_TIMEFRAME)
"""Lo que hay que esperar tras el cierre de un bloque: el horizonte, 6 velas de 4h."""

FUNDING_TAIL = timedelta(days=1)
"""El funding se extiende hasta el cierre del bloque más un día, incluido (apartado 4)."""

GAP_WAIT = timedelta(hours=24)
"""Lo que se espera para volver a descargar una vela que falta antes de declarar el hueco
persistente (apartado 4, «Hueco persistente»)."""

_COMMAND = "python -m crypto_agents.forward"


class ForwardError(RuntimeError):
    """El lote no puede seguir: la etapa 1 no se puede leer como hace falta, o falta una pieza."""


class StudyClosedError(ForwardError):
    """El estudio ya cerró antes de este bloque: el bloque no es parte del estudio."""


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


def date_refusal(start: datetime, now: datetime) -> str | None:
    """Por qué el lote de `start` no se puede correr en `now`, o `None` si se puede.

    Cinco negativas: el inicio no es un jueves a las 00:00Z, es anterior al bloque 1, es posterior
    al bloque 52, todavía no pasó el cierre más el horizonte, o ya venció el plazo de 14 días. En
    los bordes, el lote se puede correr desde `ready` exacto y en `deadline` exacto ya no.
    """
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
    if now < dates.ready:
        return (
            f"todavía no: el lote del bloque {dates.number} se corre desde "
            f"{dates.ready.isoformat()}, cuando todas sus activaciones tienen el horizonte delante"
        )
    if now >= dates.deadline:
        return f"el plazo del lote del bloque {dates.number} venció el {dates.deadline.isoformat()}"
    return None


# ───────────────────────────────────────────── El decisor ─────────────────────────────────────────


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
        outcomes[Check.TIES] = _skipped(Check.TIES, str(error))
        outcomes[Check.POOL] = _skipped(Check.POOL, str(error))
    else:
        outcomes[Check.TIES] = ties_check(ties, published)
        outcomes[Check.POOL] = pool_check(counts, POOL_COUNTS)
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


def batch_records(directory: Path) -> dict[str, tuple[EvaluationRecord, ...]]:
    """Los registros de un lote por brazo, uno por `run_id`, juntando sus pasadas: lo que se cuenta
    del lote y lo que leerá el análisis. `directory` es `data/forward/batches/<inicio>/`."""
    state = batch_state(_passes(directory))
    return {arm: tuple(by_id.values()) for arm, by_id in state.records.items()}


def forward_arms() -> tuple[AblationArm, ...]:
    """`solo` y las cuatro líneas base, tal como las define la ablación."""
    by_name = {arm.name: arm for arm in ARMS}
    return tuple(by_name[name] for name in FORWARD_ARMS)


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
) -> dict[str, tuple[EvaluationRecord, ...]]:
    """Los cinco brazos sobre el plan, cada uno contra su journal en `directory`. Sin puntuar.

    Es la función de corrida del lote y del ensayo. Un solo contador y un solo tope para los cinco,
    como en la ablación. Las líneas base no llaman a nadie. `skip` son, por brazo, los `run_id` que
    ya tienen registro completo en una pasada anterior del lote: no se vuelven a evaluar, y un brazo
    sin nada que evaluar no escribe journal.
    """
    records: dict[str, tuple[EvaluationRecord, ...]] = {}
    for arm in forward_arms():
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


class BatchSummary(FrozenModel):
    """Lo que el lote imprime al terminar: contadores de validez y de evaluaciones.

    No hay ningún campo de retorno, ni de resultado de una orden, y un test lo fija: durante el
    estudio no se calcula ningún retorno de `solo` (apartado 4).
    """

    block_start: AwareDatetime
    block_number: int = Field(ge=1)
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
    """Lo que invalida el lote en el estudio: cuota del decisor, caché de otro backend y vetos
    distintos de `invalid_stop_side` (apartado 5)."""

    spent_usd: float = Field(ge=0.0)
    """Gasto acumulado del lote, en el extremo alto de lo que el proveedor declaró."""

    cap_usd: float = Field(ge=0.0)
    unpriced_calls: int = Field(ge=0)
    """Llamadas respondidas sin tokens: se facturan y no suman al tope."""

    upstreams: tuple[UpstreamRow, ...] = ()
    """Quién sirvió las llamadas del decisor en la última pasada."""

    previous_label: str
    previous_upstreams: tuple[UpstreamRow, ...] = ()
    upstream_changed: bool
    filled_blocks: int = Field(ge=0)
    """Bloques del estudio con al menos una activación, hasta este incluido."""

    empty_blocks: int = Field(ge=0)
    closes_here: bool
    """Si con este bloque el estudio llega a los 50 bloques no vacíos."""

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


def _decider_rows(records: Iterable[EvaluationRecord]) -> tuple[UpstreamRow, ...]:
    return tuple(row for row in upstream_distribution(records) if row.role is AgentRole.DECIDER)


def _served(rows: Iterable[UpstreamRow]) -> frozenset[tuple[str, str | None]]:
    """Los pares (modelo, endpoint) que contestaron. Sin respuesta no se sabe quién sirve."""
    return frozenset(
        (row.upstream_model, row.upstream_endpoint)
        for row in rows
        if row.upstream_model != NO_RESPONSE
    )


def summarise(
    block: BlockDates,
    evaluations: int,
    records: Mapping[str, Sequence[EvaluationRecord]],
    expected_backends: Mapping[str, Mapping[AgentRole, Backend]],
    guard: SpendGuard,
    previous_label: str,
    previous_upstreams: Sequence[UpstreamRow],
    counts: Sequence[int],
) -> BatchSummary:
    """Los contadores del lote, sobre los registros de su última pasada. Sin velas: no puntúa.

    `counts` son las activaciones de cada bloque del estudio hasta este, de la población: lo único
    que lee la regla de cierre.
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
    upstreams = _decider_rows(solo)
    try:
        closing = closing_block(counts)
    except ValueError:
        closing = None
    return BatchSummary(
        block_start=block.start,
        block_number=block.number,
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
        spent_usd=guard.spent_usd,
        cap_usd=guard.cap_usd,
        unpriced_calls=guard.unpriced,
        upstreams=upstreams,
        previous_label=previous_label,
        previous_upstreams=tuple(previous_upstreams),
        upstream_changed=_served(upstreams) != _served(previous_upstreams),
        filled_blocks=sum(1 for count in counts if count > 0),
        empty_blocks=sum(1 for count in counts if count == 0),
        closes_here=closing == len(counts),
    )


def first_showing(chain: Sequence[RunDirectory], pair: tuple[str, str | None]) -> FirstCall | None:
    """La primera llamada viva del decisor de `solo`, en toda la cadena del lote, servida por ese
    par (modelo, endpoint). Una entrada de caché dice quién contestó antes, no cuándo."""
    found: list[tuple[datetime, FirstCall]] = []
    for run in chain:
        for arm in run.arms:
            if arm.arm != TARGET_ARM:
                continue
            for record in arm.records:
                for call in record.calls:
                    if call.role is not AgentRole.DECIDER or call.cache_hit:
                        continue
                    if (call.upstream_model, call.upstream_endpoint) != pair:
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


def upstream_entry(
    summary: BatchSummary, chain: Sequence[RunDirectory], today: datetime
) -> str | None:
    """El asiento preparado para el «Registro del estudio» si el upstream del decisor cambió.

    Lleva el bloque y la primera llamada que muestra cada par nuevo. No se escribe solo en
    `docs/forward.md`: lo copia el operador.
    """
    if not summary.upstream_changed:
        return None
    now = _served(summary.upstreams)
    before = _served(summary.previous_upstreams)
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
        first = first_showing(chain, pair)
        if first is None:
            lines.append(f"- {_pair_text(pair)}: solo en aciertos de caché, sin llamada viva")
            continue
        lines.append(
            f"- Primera llamada que muestra {_pair_text(pair)}: `run_id` `{first.run_id}`, "
            f"{first.symbol}, instante evaluado {first.evaluated_at:%Y-%m-%dT%H:%MZ}, "
            f"`LLMCall.at` {first.called_at.isoformat()}, en `{first.journal}`."
        )
    for pair in sorted(before - now, key=str):
        lines.append(f"- Ya no aparece {_pair_text(pair)}.")
    lines.append(
        "- El estudio continúa (apartado 5), y el resultado se reportará también separado por "
        "upstream."
    )
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────── El `meta` ────────────────────────────────────────────


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


def _stamp(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def write_forward_meta(directory: Path, meta: BatchMeta) -> Path:
    """Escribe `forward.json` en la pasada. Se crea el directorio si es una pasada que se niega."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / FORWARD_META_FILE
    path.write_text(meta.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def base_meta(
    block: BlockDates,
    now: datetime,
    history: Sequence[SeriesFile] = (),
    funding: Sequence[SeriesFile] = (),
    accepted: Sequence[AcceptedRevision] = (),
) -> BatchMeta:
    """Lo que toda pasada registra de sí misma: máquina, commit, dependencias y series.

    Sale como un bloque vacío; cada camino del lote lo completa con `model_copy`.
    """
    commit, dirty = git_state()
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


def _passes(directory: Path) -> list[Path]:
    """Las pasadas que corrieron de un lote, de la más antigua a la más reciente."""
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir() if (path / META_FILE).is_file())


def _expected_backends(meta: RunMeta) -> dict[str, dict[AgentRole, Backend]]:
    roles = meta.arm_roles or {}
    return {
        name: {role: entry.backend for role, entry in arm.items()} for name, arm in roles.items()
    }


def previous_upstreams(start: datetime, paths: ForwardPaths) -> tuple[str, tuple[UpstreamRow, ...]]:
    """Los upstreams del decisor del lote anterior: el del bloque anterior más reciente que corrió,
    juntando sus pasadas. Para el bloque 1, o sin lotes antes, la etapa 1."""
    if paths.batches.is_dir():
        earlier = sorted(
            (path for path in paths.batches.iterdir() if path.name < f"{start:%Y-%m-%d}"),
            reverse=True,
        )
        for block_dir in earlier:
            if _passes(block_dir):
                solo = batch_records(block_dir).get(TARGET_ARM, ())
                return f"el lote del bloque que empieza el {block_dir.name}", _decider_rows(solo)
    solo_path = arm_journal_path(paths.stage1, TARGET_ARM)
    verify_stage1(paths.stage1)
    run = read_run(paths.stage1)
    solo = next((arm.records for arm in run.arms if arm.arm == TARGET_ARM), ())
    return f"la etapa 1 (`{solo_path}`)", _decider_rows(solo)


def files_to_version(
    prepared: PreparedBlock, directory: Path | None, revisions: Path = REVISIONS_FILE
) -> list[str]:
    """Lo que hay que versionar tras la pasada: series, revisiones aceptadas, manifiesto y la pasada
    entera."""
    files = [item.path for item in (*prepared.history, *prepared.funding)]
    if prepared.accepted:
        files.append(str(revisions))
    if prepared.manifest_file is not None:
        files.append(str(prepared.manifest_file))
    if directory is not None and directory.is_dir():
        files.extend(str(path) for path in sorted(directory.iterdir()) if path.is_file())
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
        f"- respecto a {summary.previous_label}: "
        + ("**cambió**" if summary.upstream_changed else "sin cambios"),
        "- lote completo: "
        + ("sí" if summary.complete else "**NO**: hay que reanudar antes del plazo"),
        f"- bloques del estudio hasta este: {summary.filled_blocks} no vacíos, "
        f"{summary.empty_blocks} vacíos; {closing}",
    ]
    return lines


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


def _version_lines(
    prepared: PreparedBlock, directory: Path | None, revisions: Path = REVISIONS_FILE
) -> list[str]:
    return [
        "",
        "Archivos que hay que versionar (este comando no hace commits):",
        *(f"- {path}" for path in files_to_version(prepared, directory, revisions)),
    ]


async def run_pass(
    prepared: PreparedBlock, env: BatchEnvironment, argv: Sequence[str]
) -> BatchResult:
    """Una pasada sobre un bloque ya preparado: comprobaciones, tope, corrida y resumen.

    La reanudación es por journal: la pasada se salta los `run_id` que ya tienen registro completo
    en las pasadas anteriores del lote (`batch_state`), y la caché solo sirve para los reintentos de
    lo que se vuelve a evaluar. Si el lote ya estaba completo, se imprime su resumen y no se abre
    otra pasada. Toda pasada que llega a comprobar deja su `forward.json`, también la que se niega:
    el registro anota la fecha y la causa de una negativa.
    """
    block, paths = prepared.block, env.paths
    now = env.clock()
    batch_dir = paths.batches / f"{block.start:%Y-%m-%d}"
    directory = batch_dir / _stamp(now)
    meta = base_meta(block, now, prepared.history, prepared.funding, prepared.accepted)

    def version(where: Path | None) -> list[str]:
        return _version_lines(prepared, where, paths.revisions)

    if prepared.manifest is None or prepared.manifest_file is None:
        recorded = _empty_recorded(batch_dir)
        if recorded is None:
            write_forward_meta(directory, meta)
            recorded = directory
        lines = [
            f"El bloque {block.number} no tiene activaciones: no hay manifiesto ni corrida "
            f"({recorded / FORWARD_META_FILE}).",
            f"- bloques del estudio hasta este: {sum(1 for c in prepared.counts if c)} no vacíos, "
            f"{sum(1 for c in prepared.counts if not c)} vacíos",
            *version(recorded),
        ]
        return BatchResult(0, tuple(lines))

    manifest, manifest_file = prepared.manifest, prepared.manifest_file
    manifest_sha = file_sha256(manifest_file)
    settings = env.settings()
    cache = JsonFileResponseCache(paths.cache)
    passes = _passes(batch_dir)
    latest = passes[-1] if passes else None
    try:
        state = batch_state(passes)
    except DuplicateRecordError as error:
        return BatchResult(1, (f"error: {error}. El lote se detiene sin llamar a nadie.",))
    previous_calls = chain_calls(latest) if latest is not None else []
    arms = forward_arms()

    if latest is not None:
        spent = spend_state(settings, manifest, previous_calls, now)
        label, before = previous_upstreams(block.start, paths)
        summary = summarise(
            block,
            len(manifest.entries),
            {arm: tuple(by_id.values()) for arm, by_id in state.records.items()},
            _expected_backends(read_run(latest).meta),
            spent.guard,
            label,
            before,
            prepared.counts,
        )
        if summary.complete:
            lines = [
                f"El lote del bloque {block.number} ya está completo en {batch_dir}: no se abre "
                "otra pasada.",
                "",
                *render_summary(summary),
            ]
            return BatchResult(0, tuple(lines))

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
        lines = [
            headline,
            "",
            *_check_lines(checks),
            "",
            f"Registro de la pasada: {directory / FORWARD_META_FILE}",
            *version(directory),
        ]
        return BatchResult(4 if ends else 1, tuple(lines))

    spent = spend_state(settings, manifest, previous_calls, now)
    meta = meta.model_copy(update={"spent_before_usd": spent.spent_before_usd})
    if spent.spent_before_usd >= spent.cap_usd:
        write_forward_meta(
            directory, meta.model_copy(update={"outcome": BatchOutcome.CAP_EXHAUSTED})
        )
        lines = [
            f"Las pasadas anteriores ya gastaron {spent.spent_before_usd:.4f} USD de un tope de "
            f"{spent.cap_usd:.4f}: no se llama a nadie. Lo que siga sin decisión al vencer el "
            "plazo cuenta como faltante.",
            "",
            *_check_lines(checks),
            *version(directory),
        ]
        return BatchResult(3, tuple(lines))

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
    label, before = previous_upstreams(block.start, paths)
    summary = summarise(
        block,
        len(manifest.entries),
        {arm: tuple(by_id.values()) for arm, by_id in merged.items()},
        _expected_backends(run_meta),
        spent.guard,
        label,
        before,
        prepared.counts,
    )
    entry = upstream_entry(summary, run_chain(directory), now)
    if entry is not None:
        (directory / UPSTREAM_FILE).write_text(entry, encoding="utf-8")

    lines = [*render_summary(summary), "", *_check_lines(checks)]
    if entry is not None:
        lines += [
            "",
            "Asiento preparado para el «Registro del estudio» (no se escribe solo; está también en "
            f"{directory / UPSTREAM_FILE}):",
            "",
            entry,
        ]
    lines += version(directory)
    return BatchResult(0 if summary.complete else 3, tuple(lines))


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
    directory = env.paths.batches / f"{block.start:%Y-%m-%d}" / _stamp(now)
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
    batch_dir = env.paths.batches / f"{block.start:%Y-%m-%d}"
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
) -> BatchResult:
    """El lote del bloque que empieza en `start`: fechas, datos, comprobaciones y corrida.

    `accept` son las velas revisadas que el operador acepta con `--accept-revised`.
    """
    refusal = date_refusal(start, env.clock())
    if refusal is not None:
        return BatchResult(2, (f"error: {refusal}",))
    block = block_dates(start)
    try:
        prepared = await prepare_block(start, env, accept)
    except StudyClosedError as error:
        return BatchResult(2, (f"error: {error}",))
    except RevisedCandlesError as error:
        return revised_result(block, env, error)
    except MissingCandlesError as error:
        return gap_result(block, env, error)
    return await run_pass(prepared, env, argv)


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
    """Punto de entrada: `python -m crypto_agents.forward batch|rehearse`.

    `batch` sale con:
    - 0 si el lote está completo, o el bloque vacío;
    - 1 si se negó por una comprobación o por los datos, sin llamar a nadie;
    - 2 por la fecha o el uso;
    - 3 si corrió y quedó incompleto;
    - 4 si el decisor ya no está en el catálogo;
    - 5 si el exchange devuelve distinta una vela guardada que nadie aceptó;
    - 6 si falta una vela nueva y hay que volver a descargar 24 h después;
    - 7 si sigue faltando 24 h después: el estudio termina en ese bloque.

    `rehearse` sale con 0 si coinciden las cinco ramas con cero llamadas vivas y pasan las
    comprobaciones.
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
    rehearsal = commands.add_parser("rehearse", help="ensayo sin pagar sobre la etapa 1")
    rehearsal.add_argument("--cache", type=Path, required=True, help="caché de la etapa 1")
    rehearsal.add_argument("--out", type=Path, required=True, help="directorio nuevo")
    args = parser.parse_args(argv)
    raw = tuple(argv) if argv is not None else tuple(sys.argv[1:])
    environment = env if env is not None else default_environment()

    try:
        if args.command == "batch":
            result = asyncio.run(run_batch(args.block, environment, raw, args.accept_revised))
        else:
            result = asyncio.run(rehearse(environment, args.cache, args.out, raw))
    except (
        AuditError,
        ConfigError,
        ForwardDataError,
        ForwardError,
        MarketDataError,
        ProbeError,
        ReplayCacheMissError,
        SelectionError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    stream = sys.stderr if result.code == 2 else sys.stdout
    print("\n".join(result.lines), file=stream)
    return result.code


if __name__ == "__main__":
    raise SystemExit(main())
