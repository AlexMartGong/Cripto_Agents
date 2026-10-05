"""Sondeo de OpenCode Zen: qué modelos responden, en qué modo, con qué fallos y a qué coste.

La ablación por lotes va por pago por uso (Zen) y no por la suscripción Go. Antes de elegir los
modelos de `structure` y `volume` hay que saber cuáles de la lista responden de verdad, y eso no
lo dice el catálogo: en Go el catálogo listaba los seis ids mientras el chat respondía
`400 MissingSessionID`. Este módulo pregunta por el chat.

    python -m crypto_agents.zen_probe --machine desktop [--dry-run]

Escribe `var/zen-probe/<inicio UTC>/` (no versionado, nunca reutilizado) con el mismo formato que
una corrida de la ablación: `meta.json` y un `<modelo>@<rol>.jsonl` por brazo, un
`EvaluationRecord` por veredicto con todos sus intentos. Así `audit` y
`python -m crypto_agents.consumption <directorio>` lo leen sin saber que es un sondeo, y cada cifra
del informe se puede volver a calcular desde el archivo del que sale.

Cinco etapas, todas por `ModelRouter` (regla 4: cada intento deja su `LLMCall`), sin caché y sin
respaldo —un respaldo local contestaría por el remoto y el sondeo diría lo contrario de la verdad—:

1. **Catálogo.** `GET /models`. Dice qué ids lista Zen; no decide nada: un id ausente se sondea
   igual, porque la ausencia en el listado no prueba que el chat no lo sirva (ni la presencia que
   lo haga).
2. **Modos.** Un `Ping` por id con el modo declarado y, si falla, los otros dos (la misma máquina
   que `doctor`). Los cuatro presentes parten del modo que declara `.env`; los candidatos, de
   `json_schema`.
3. **Cabecera.** Un `Ping` con y sin `x-opencode-session`. El catálogo no vale para esto.
4. **Veredictos.** 12 por (candidato, dimensión) con el prompt técnico real y `TechnicalVerdict`,
   más 12 de `momentum` con el modelo presente, con la política de producción
   (`max_attempts=2`).
5. **Encadenado.** Con esos veredictos, las dos mesas y el decisor con sus prompts reales, para
   medir sus tokens. Las salidas se descartan: no hay riesgo, ni orden, ni caché, ni `--fill`.

No elige modelos. Se niega a correr con `billing` distinto de `payg` o contra una `base_url` de Go:
nada de este sondeo se cobra a la suscripción.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple
from urllib.parse import urlsplit

from pydantic import Field

from crypto_agents.ablation import (
    NOTIONAL_ACCOUNT,
    ReplayPlan,
    git_state,
    open_run_directory,
    plan_from_manifest,
)
from crypto_agents.activation import ActivationConfig
from crypto_agents.audit import PlanKind, RunDirectory, RunMeta, arm_journal_path, read_run
from crypto_agents.audit import file_sha256 as sha256_of
from crypto_agents.cache import InMemoryResponseCache
from crypto_agents.consumption import consume, render_consumption
from crypto_agents.context import AgentContext, utc_now
from crypto_agents.doctor import PROBE_PROMPT, Ping, RoleReport, probe_role, probe_router
from crypto_agents.doctor import probe_settings as single_model_settings
from crypto_agents.indicators import DEFAULT_PRESET, IndicatorPreset
from crypto_agents.journal import EvaluationRecord, JsonlJournal
from crypto_agents.llm import (
    ModelInvocationError,
    ModelRouter,
    OpenAIBackend,
    build_backends,
)
from crypto_agents.market import timeframe_to_timedelta
from crypto_agents.metrics import attempt_counts, failure_counts, live_latency
from crypto_agents.nodes import (
    PreparedEvaluation,
    debate_context,
    prepare_evaluation,
    technical_context,
)
from crypto_agents.prompts import debate_prompt, decision_prompt, technical_prompt
from crypto_agents.quota import QuotaLedger
from crypto_agents.replay import HistoricalMarketClient, replay_router, replay_run_id
from crypto_agents.selection import (
    HISTORY_DIR,
    PlannedEvaluation,
    SelectionError,
    load_manifest,
    load_selection_histories,
)
from crypto_agents.settings import (
    DEFAULT_ENV_FILE,
    DEFAULT_PRICING,
    ZEN_UNPUBLISHED_QUOTA,
    ConfigError,
    ModelChoice,
    load_settings,
    public_url,
)
from crypto_agents.state import (
    AgentRole,
    Backend,
    Billing,
    DebateBrief,
    Decision,
    Dimension,
    FailureKind,
    FrozenModel,
    NodeError,
    Side,
    StructuredOutputMode,
    TechnicalVerdict,
)

if TYPE_CHECKING:
    from collections.abc import Coroutine, Mapping, Sequence
    from datetime import datetime

    from crypto_agents.llm import ChatBackend, ModelCatalog
    from crypto_agents.quota import Clock
    from crypto_agents.settings import Settings
    from crypto_agents.state import LLMCall

__all__ = [
    "CANDIDATES",
    "PRESENT",
    "PROBE_DIR",
    "VERDICTS",
    "Candidate",
    "ModelFinding",
    "ProbeFindings",
    "build_meta",
    "main",
    "refuse_unless_zen",
    "render_report",
    "run_probe",
]

PROBE_DIR = Path("var/zen-probe")
"""Dónde van los directorios del sondeo. No se versiona (`var/` está en `.gitignore`)."""

DEFAULT_MANIFEST = Path("data/ablation_selection.json")

VERDICTS = 12
"""Veredictos por (candidato, dimensión): los mismos 12 del re-sondeo del 2026-08-15."""

TECHNICAL_ATTEMPTS = 2
"""Intentos por veredicto: los de producción (`ModelRouter` por defecto), no los del `Ping`."""

SESSION_NOTE = "x-opencode-session"

HEADER_MODEL = "deepseek-v4.1-flash"
"""El modelo del `Ping` con y sin cabecera: un candidato barato que Zen lista en `/models`.

Basta con que la pasarela conteste algo distinto según la cabecera: Go rechazaba con
`400 MissingSessionID` antes de generar nada, así que un `402` sin fondos o un `403` sin acceso al
modelo ya distinguen «la cabecera sobra» de «la cabecera falta». Por eso no se exige un modo
confirmado, y el informe dice si lo estaba.
"""


class Candidate(NamedTuple):
    """Un modelo de pago de Zen, servido por `/chat/completions`, que podría llevar un rol."""

    model: str
    family: str
    """Declarada a mano, como `ModelChoice.family`: deducirla del nombre es lo que se evita."""


CANDIDATES: Final = (
    Candidate("deepseek-v4.1-flash", "deepseek"),
    Candidate("deepseek-v4-pro", "deepseek"),
    Candidate("glm-5.3-flash", "zhipu"),
    Candidate("minimax-m2.7", "minimax"),
    Candidate("kimi-k2.7-code", "moonshot"),
    Candidate("qwen3.8-max", "qwen"),
)

PRESENT: Final = (
    ("glm-5.2", AgentRole.DECIDER),
    ("kimi-k2.6", AgentRole.BULL),
    ("minimax-m3", AgentRole.BEAR),
    ("deepseek-v4-flash", AgentRole.MOMENTUM),
)
"""Los cuatro ids que el mapa de roles ya usa y que Zen debería servir, con el rol que los lleva."""

TECHNICAL_DIMENSIONS: Final = (Dimension.STRUCTURE, Dimension.VOLUME)
"""Las dimensiones para las que se buscan candidatos. `momentum` tiene ya su modelo."""

_ROLE_OF: Final = {
    Dimension.STRUCTURE: AgentRole.STRUCTURE,
    Dimension.MOMENTUM: AgentRole.MOMENTUM,
    Dimension.VOLUME: AgentRole.VOLUME,
}


def refuse_unless_zen(settings: Settings) -> None:
    """Se niega a sondear fuera de Zen o fuera del pago por uso, antes de construir un backend.

    El sondeo cuesta dinero de verdad, y la orden es que no se cobre a la suscripción. Se mira la
    configuración y no se confía en recordar qué `.env` hay puesto: es la comprobación que `doctor`
    no hace antes de gastar.
    """
    if settings.openai is None:
        raise ConfigError("falta CA_OPENAI__API_KEY: no hay con qué sondear")
    if settings.billing is not Billing.PAYG:
        raise ConfigError(
            f"el sondeo es de pago por uso: CA_BILLING={settings.billing.value}, debe ser payg"
        )
    shown = public_url(settings.openai.base_url)
    path = "" if shown is None else urlsplit(shown).path
    if shown is None or path.rstrip("/").startswith("/zen/go"):
        raise ConfigError(
            f"la base_url apunta a OpenCode Go ({shown or 'sin base_url'}): "
            "nada de este sondeo se cobra a la suscripción"
        )


class _NoSessionBackend(OpenAIBackend):
    """El mismo adaptador sin `x-opencode-session`, para ver si Zen la exige.

    Es una subclase y no un parámetro de producción: ningún camino real debe poder mandar una
    petición sin esa cabecera, y quien la quita aquí es el sondeo que quiere saber qué pasa.
    """

    def _headers(self) -> dict[str, str]:
        return {}


# ───────────────────────────────────────── Lo que se sondea ───────────────────────────────────────


class Subject(NamedTuple):
    """Un id a sondear, con el rol bajo el que se le pregunta y la elección que lo describe."""

    model: str
    family: str
    role: AgentRole
    choice: ModelChoice
    present: bool


def _unmetered(choice: ModelChoice) -> ModelChoice:
    """La elección sin la cuota de Go: Zen no publica ninguna y el sondeo no se limita."""
    return choice.model_copy(
        update={"quota_per_window": ZEN_UNPUBLISHED_QUOTA, "quota_weight": 1.0}
    )


def subjects_of(settings: Settings) -> tuple[Subject, ...]:
    """Los cuatro presentes (con el modo que declara `.env`) y los seis candidatos."""
    result: list[Subject] = []
    for model, role in PRESENT:
        declared = settings.role_config(role).primary
        if declared.model != model:
            raise ConfigError(
                f"{role.value} ya no declara {model} (declara {declared.model}): "
                "el sondeo de los presentes parte del mapa de roles de .env"
            )
        result.append(Subject(model, declared.family, role, _unmetered(declared), True))
    template = settings.role_config(AgentRole.STRUCTURE).primary
    for candidate in CANDIDATES:
        choice = ModelChoice(
            backend=Backend.OPENAI,
            model=candidate.model,
            family=candidate.family,
            structured_output=StructuredOutputMode.JSON_SCHEMA,
            quota_weight=1.0,
            quota_per_window=ZEN_UNPUBLISHED_QUOTA,
            temperature=template.temperature,
        )
        result.append(
            Subject(candidate.model, candidate.family, AgentRole.STRUCTURE, choice, False)
        )
    return tuple(result)


class Activation(NamedTuple):
    """Una activación del manifiesto con su capa determinista ya calculada."""

    entry: PlannedEvaluation
    prepared: PreparedEvaluation


def pick_evenly(entries: Sequence[PlannedEvaluation], count: int) -> list[PlannedEvaluation]:
    """`count` entradas equiespaciadas: símbolos y tramos distintos, no las primeras."""
    count = min(count, len(entries))
    return [entries[position * len(entries) // count] for position in range(count)]


async def prepare_activations(
    plan: ReplayPlan,
    settings: Settings,
    clock: Clock,
    count: int,
    preset: IndicatorPreset = DEFAULT_PRESET,
    activation: ActivationConfig | None = None,
) -> list[Activation]:
    """Las capas deterministas de `count` activaciones del plan, por el mismo camino que `dry_run`.

    El router que lleva el contexto es el del replay sin `fill_with`: solo caché, vacía. Esto no
    puede llamar a nadie, y es lo que garantiza que los prompts del sondeo son los que la corrida
    construirá: salen de `prepare_evaluation`, no de una copia.
    """
    router = replay_router(
        settings, QuotaLedger(settings.quota_window, clock), InMemoryResponseCache(), clock
    )
    result: list[Activation] = []
    for entry in pick_evenly(plan.entries, count):
        moment = entry.at + timeframe_to_timedelta(entry.timeframe)
        context = AgentContext(
            settings=settings,
            router=router,
            market=HistoricalMarketClient(
                plan.histories[entry.symbol], entry.index, plan.candle_limit
            ),
            account=NOTIONAL_ACCOUNT,
            run_id=replay_run_id(entry.symbol, entry.timeframe, moment),
            symbol=entry.symbol,
            timeframe=entry.timeframe,
            candle_limit=plan.candle_limit,
            preset=preset,
            activation=activation if activation is not None else ActivationConfig(preset=preset),
            clock=lambda moment=moment: moment,  # type: ignore[misc]
            now=moment,
        )
        try:
            result.append(Activation(entry, await prepare_evaluation(context)))
        except Exception as error:  # una vela del manifiesto que no prepara es un manifiesto roto
            raise ConfigError(
                f"no se pudo preparar {entry.symbol} {entry.at.isoformat()}: "
                f"{type(error).__name__}: {error}"
            ) from error
    return result


# ───────────────────────────────────────────── Hallazgos ──────────────────────────────────────────


class ModelFinding(FrozenModel):
    """Qué se averiguó de un id: si está listado y en qué modo contesta."""

    model: str = Field(min_length=1)
    family: str = Field(min_length=1)
    role: AgentRole
    present: bool
    listed: bool | None
    """Si `GET /models` lo lista. `None` si el catálogo no se pudo leer."""

    declared: StructuredOutputMode
    mode: StructuredOutputMode | None
    """El modo en que dio salida válida, o `None` si en ninguno."""

    note: str


class HeaderFinding(FrozenModel):
    """El `Ping` con y sin `x-opencode-session`, tal como contestó Zen."""

    model: str
    mode: StructuredOutputMode
    mode_confirmed: bool
    """Si el modo del `Ping` lo había confirmado antes el sondeo, o es solo el declarado."""

    with_header: str
    without_header: str


class ProbeFindings(FrozenModel):
    """Lo que el sondeo averiguó y que no cabe en un `LLMCall`. Se guarda como `findings.json`."""

    machine: str = Field(min_length=1)
    catalog_size: int | None = None
    catalog_error: str | None = None
    models: tuple[ModelFinding, ...] = ()
    header: HeaderFinding | None = None
    skipped: tuple[str, ...] = ()
    """Brazos que no corrieron y por qué: un brazo sin journal no es uno con cero veredictos."""


def confirmed_mode(report: RoleReport) -> StructuredOutputMode | None:
    """El modo que produjo salida válida: el declarado, o el alternativo que `doctor` encontró."""
    return report.declared if report.verified else report.working


def _note(report: RoleReport) -> str:
    if report.verified:
        return "responde en el modo declarado"
    if report.exhausted is not None:
        return f"sin cuota para sondear: {report.exhausted}"
    if report.working is not None:
        return f"el declarado no dio salida válida; {report.working.value} sí"
    if report.transport:
        reason = report.calls[-1].failure_message if report.calls else None
        return "rechazado por el proveedor antes de producir contenido" + (
            f": {reason[:200]}" if reason else ""
        )
    return "ningún modo dio salida válida"


# ───────────────────────────────────────── Escritura de brazos ────────────────────────────────────


def arm_name(model: str, label: str) -> str:
    """`<modelo>@<etiqueta>`: el nombre del brazo y de su `.jsonl`."""
    return f"{model}@{label}"


def _write(
    directory: Path,
    arm: str,
    entry: PlannedEvaluation | None,
    calls: Sequence[LLMCall],
    node: str,
    failure: str | None,
    clock: Clock,
) -> None:
    """Un `EvaluationRecord` por veredicto, con todos sus intentos y, si falló, por qué.

    Un `Ping` no viene de ninguna evaluación: lleva el símbolo `probe` y el reloj. Un veredicto
    lleva el símbolo y el instante de la activación de la que salió su prompt.
    """
    if entry is None:
        symbol, timeframe, moment = "probe", "n/a", clock()
        run_id = replay_run_id(symbol, timeframe, moment)
    else:
        symbol, timeframe = entry.symbol, entry.timeframe
        moment = entry.at + timeframe_to_timedelta(entry.timeframe)
        run_id = replay_run_id(symbol, timeframe, moment)
    errors = () if failure is None else (NodeError(node=node, message=failure, at=moment),)
    JsonlJournal(arm_journal_path(directory, arm)).write(
        EvaluationRecord(
            run_id=run_id,
            at=moment,
            symbol=symbol,
            timeframe=timeframe,
            calls=tuple(calls),
            errors=errors,
        )
    )


def _router(
    settings: Settings,
    role: AgentRole,
    choice: ModelChoice,
    backends: Mapping[Backend, ChatBackend],
    clock: Clock,
) -> ModelRouter:
    """Router de producción para un solo modelo: sin caché, sin respaldo, dos intentos."""
    single = single_model_settings(settings, {role: choice})
    ledger = QuotaLedger(single.quota_window, clock)
    return ModelRouter(single, ledger, backends, clock, cache=None, max_attempts=TECHNICAL_ATTEMPTS)


# ────────────────────────────────────────────── Etapas ────────────────────────────────────────────


async def check_catalog(catalog: ModelCatalog) -> tuple[frozenset[str] | None, str | None]:
    """Los ids que lista el gateway, o por qué no se pudo leer. No gasta tokens."""
    try:
        return await catalog.available_models(), None
    except Exception as error:  # httpx, openai y el gateway tienen jerarquías propias
        return None, f"{type(error).__name__}: {error}"


async def ping_subjects(
    settings: Settings,
    subjects: Sequence[Subject],
    backends: Mapping[Backend, ChatBackend],
    clock: Clock,
    directory: Path,
) -> dict[str, RoleReport]:
    """Un `Ping` por id, todos a la vez: son peticiones independientes contra el mismo gateway."""
    reports = await asyncio.gather(
        *(probe_role(settings, s.role, s.choice, backends, clock) for s in subjects)
    )
    by_model: dict[str, RoleReport] = {}
    for subject, report in zip(subjects, reports, strict=True):
        by_model[subject.model] = report
        _write(
            directory,
            arm_name(subject.model, "ping"),
            None,
            report.calls,
            "ping",
            None if confirmed_mode(report) is not None else _note(report),
            clock,
        )
    return by_model


async def header_test(
    settings: Settings,
    subject: Subject,
    mode: StructuredOutputMode,
    confirmed: bool,
    backend: ChatBackend,
    headerless: ChatBackend,
    clock: Clock,
    directory: Path,
) -> HeaderFinding:
    """Un `Ping` al mismo modelo, una vez con `x-opencode-session` y otra sin ella."""
    choice = subject.choice.model_copy(update={"structured_output": mode})
    outcomes: dict[str, str] = {}
    for label, chosen in (("with", backend), ("without", headerless)):
        router = probe_router(
            single_model_settings(settings, {subject.role: choice}),
            QuotaLedger(settings.quota_window, clock),
            {Backend.OPENAI: chosen},
            clock,
        )
        failure: str | None = None
        try:
            _, calls = await router.invoke(subject.role, PROBE_PROMPT, Ping)
            outcomes[label] = "responde"
        except ModelInvocationError as error:
            calls = list(error.calls)
            failure = calls[-1].failure_message if calls else str(error)
            outcomes[label] = f"rechazado: {failure}"
        _write(
            directory,
            arm_name(subject.model, f"header-{label}"),
            None,
            calls,
            "ping",
            failure,
            clock,
        )
    return HeaderFinding(
        model=subject.model,
        mode=mode,
        mode_confirmed=confirmed,
        with_header=outcomes["with"],
        without_header=outcomes["without"],
    )


async def technical_arm(
    settings: Settings,
    subject: Subject,
    mode: StructuredOutputMode,
    dimension: Dimension,
    activations: Sequence[Activation],
    backends: Mapping[Backend, ChatBackend],
    clock: Clock,
    directory: Path,
    verdicts: dict[tuple[str, Dimension, int], TechnicalVerdict],
) -> None:
    """Un veredicto por activación con el prompt técnico real, en serie, escribiendo cada uno.

    Las llamadas de un mismo modelo van una detrás de otra, como las hace el grafo; lo que corre a
    la vez son los candidatos entre sí. Los veredictos válidos se guardan para el encadenado.
    """
    role = _ROLE_OF[dimension]
    choice = subject.choice.model_copy(update={"structured_output": mode})
    router = _router(settings, role, choice, backends, clock)
    arm = arm_name(subject.model, dimension.value)
    for position, activation in enumerate(activations):
        prepared = activation.prepared
        prompt = technical_prompt(
            dimension, prepared.snapshot, prepared.indicators, prepared.activation.triggers
        )
        check = technical_context(dimension, prepared.indicators)
        try:
            verdict, calls = await router.invoke(role, prompt, TechnicalVerdict, check)
        except ModelInvocationError as error:
            _write(
                directory, arm, activation.entry, error.calls, dimension.value, str(error), clock
            )
            continue
        verdicts[(subject.model, dimension, position)] = verdict
        _write(directory, arm, activation.entry, calls, dimension.value, None, clock)


def producers(
    verdicts: Mapping[tuple[str, Dimension, int], TechnicalVerdict],
    dimension: Dimension,
    candidates: Sequence[str],
    at: datetime,
) -> list[str]:
    """Quién produce los veredictos de una dimensión para el encadenado, el mejor primero.

    Gana el que más veredictos válidos dio y, a igualdad, el más barato por token de salida: el
    encadenado solo necesita entradas verosímiles para las mesas, no una opinión sobre quién gana.
    """

    def price(model: str) -> float:
        row = DEFAULT_PRICING.price_for(model, Billing.PAYG, at)
        return row.output_per_mtok if row is not None else float("inf")

    valid = {
        model: sum(1 for (m, d, _), _v in verdicts.items() if m == model and d is dimension)
        for model in candidates
    }
    return sorted((m for m in candidates if valid[m] > 0), key=lambda m: (-valid[m], price(m), m))


async def _desk(
    side: Side,
    routers: Mapping[AgentRole, ModelRouter],
    models: Mapping[AgentRole, str],
    activation: Activation,
    evidence: tuple[TechnicalVerdict, ...],
    directory: Path,
    clock: Clock,
) -> DebateBrief | None:
    """Una mesa argumenta sobre la evidencia común, con su prompt y su validación de contexto."""
    role = AgentRole.BULL if side is Side.BULL else AgentRole.BEAR
    prepared = activation.prepared
    prompt = debate_prompt(side, prepared.snapshot, prepared.indicators, evidence)
    arm = arm_name(models[role], role.value)
    try:
        brief, calls = await routers[role].invoke(
            role, prompt, DebateBrief, debate_context(side, evidence)
        )
    except ModelInvocationError as error:
        _write(directory, arm, activation.entry, error.calls, role.value, str(error), clock)
        return None
    _write(directory, arm, activation.entry, calls, role.value, None, clock)
    return brief


async def chain_stage(
    settings: Settings,
    activations: Sequence[Activation],
    verdicts: Mapping[tuple[str, Dimension, int], TechnicalVerdict],
    backends: Mapping[Backend, ChatBackend],
    clock: Clock,
    directory: Path,
    modes: Mapping[str, StructuredOutputMode],
) -> tuple[str, ...]:
    """Mesas y decisor con sus prompts reales sobre veredictos reales. Devuelve lo que se saltó.

    No es la ablación: no hay riesgo, ni orden, ni caché, ni journal de evaluaciones. Solo se
    pregunta lo que la corrida preguntará, para medir cuántos tokens cuesta cada pregunta. Las
    salidas se descartan.
    """
    skipped: list[str] = []
    routers: dict[AgentRole, ModelRouter] = {}
    models: dict[AgentRole, str] = {}
    for model, role in PRESENT:
        if role not in (AgentRole.BULL, AgentRole.BEAR, AgentRole.DECIDER):
            continue
        mode = modes.get(model)
        if mode is None:
            skipped.append(f"{arm_name(model, role.value)}: sin modo que responda")
            continue
        declared = settings.role_config(role).primary
        choice = _unmetered(declared).model_copy(update={"structured_output": mode})
        routers[role] = _router(settings, role, choice, backends, clock)
        models[role] = model
    if len(routers) < 3:
        return (*skipped, "encadenado: falta un modelo presente")

    by_dimension = {
        dimension: producers(
            verdicts,
            dimension,
            [c.model for c in CANDIDATES]
            if dimension in TECHNICAL_DIMENSIONS
            else [PRESENT[-1][0]],
            clock(),
        )
        for dimension in Dimension
    }
    for position, activation in enumerate(activations):
        picked: list[TechnicalVerdict] = []
        for dimension in Dimension:
            found = next(
                (
                    verdicts[(model, dimension, position)]
                    for model in by_dimension[dimension]
                    if (model, dimension, position) in verdicts
                ),
                None,
            )
            if found is None:
                break
            picked.append(found)
        if len(picked) < len(Dimension):
            skipped.append(f"encadenado #{position}: faltan veredictos técnicos válidos")
            continue
        evidence = tuple(sorted(picked, key=lambda item: item.dimension.value))
        bull, bear = await asyncio.gather(
            *(
                _desk(side, routers, models, activation, evidence, directory, clock)
                for side in (Side.BULL, Side.BEAR)
            )
        )
        if bull is None or bear is None:
            skipped.append(f"encadenado #{position}: una mesa falló, no hay decisor que medir")
            continue
        prepared = activation.prepared
        prompt = decision_prompt(prepared.snapshot, prepared.indicators, evidence, bull, bear)
        arm = arm_name(models[AgentRole.DECIDER], AgentRole.DECIDER.value)
        try:
            _, calls = await routers[AgentRole.DECIDER].invoke(AgentRole.DECIDER, prompt, Decision)
        except ModelInvocationError as error:
            _write(directory, arm, activation.entry, error.calls, "decider", str(error), clock)
            continue
        _write(directory, arm, activation.entry, calls, "decider", None, clock)
    return tuple(skipped)


def all_arms(settings: Settings) -> tuple[str, ...]:
    """Todos los brazos que el sondeo puede escribir, declarados antes de la primera llamada."""
    arms: list[str] = []
    for subject in subjects_of(settings):
        arms.append(arm_name(subject.model, "ping"))
    arms += [arm_name(HEADER_MODEL, "header-with"), arm_name(HEADER_MODEL, "header-without")]
    for candidate in CANDIDATES:
        arms += [arm_name(candidate.model, d.value) for d in TECHNICAL_DIMENSIONS]
    arms.append(arm_name("deepseek-v4-flash", Dimension.MOMENTUM.value))
    arms += [
        arm_name(model, role.value)
        for model, role in PRESENT
        if role in (AgentRole.BULL, AgentRole.BEAR, AgentRole.DECIDER)
    ]
    return tuple(arms)


def build_meta(
    settings: Settings,
    manifest: Path,
    argv: Sequence[str],
    started: datetime,
    commit: str | None,
    dirty: bool | None,
) -> RunMeta:
    """El `meta.json` del sondeo: una corrida de verdad, con su plan, su pasarela y su pago.

    El plan es el manifiesto del que salen los prompts. `base_url` pasa por `public_url`: la
    pasarela queda escrita, la clave y cualquier credencial de la URL no.
    """
    return RunMeta(
        plan_kind=PlanKind.MANIFEST,
        plan_path=str(manifest),
        plan_sha256=sha256_of(manifest),
        argv=tuple(argv),
        fill=True,
        arms=all_arms(settings),
        started_at=started,
        git_commit=commit,
        git_dirty=dirty,
        billing=settings.billing,
        base_url=public_url(None if settings.openai is None else settings.openai.base_url),
        quota_window=settings.quota_window,
    )


async def run_probe(
    settings: Settings,
    plan: ReplayPlan,
    directory: Path,
    machine: str,
    backends: Mapping[Backend, ChatBackend],
    headerless: ChatBackend,
    catalog: ModelCatalog,
    clock: Clock = utc_now,
    verdicts: int = VERDICTS,
    preset: IndicatorPreset = DEFAULT_PRESET,
) -> ProbeFindings:
    """Las cinco etapas, escribiendo en `directory` (que ya tiene su `meta.json`)."""
    subjects = subjects_of(settings)
    activations = await prepare_activations(plan, settings, clock, verdicts, preset)

    listed, catalog_error = await check_catalog(catalog)
    reports = await ping_subjects(settings, subjects, backends, clock, directory)
    modes = {m: mode for m, r in reports.items() if (mode := confirmed_mode(r)) is not None}

    findings = [
        ModelFinding(
            model=s.model,
            family=s.family,
            role=s.role,
            present=s.present,
            listed=None if listed is None else s.model in listed,
            declared=s.choice.structured_output,
            mode=modes.get(s.model),
            note=_note(reports[s.model]),
        )
        for s in subjects
    ]

    skipped: list[str] = []
    header_subject = next(s for s in subjects if s.model == HEADER_MODEL)
    header = await header_test(
        settings,
        header_subject,
        modes.get(HEADER_MODEL, header_subject.choice.structured_output),
        HEADER_MODEL in modes,
        backends[Backend.OPENAI],
        headerless,
        clock,
        directory,
    )

    collected: dict[tuple[str, Dimension, int], TechnicalVerdict] = {}
    jobs: list[Coroutine[object, object, None]] = []

    async def candidate_job(subject: Subject) -> None:
        for dimension in TECHNICAL_DIMENSIONS:
            await technical_arm(
                settings,
                subject,
                modes[subject.model],
                dimension,
                activations,
                backends,
                clock,
                directory,
                collected,
            )

    async def momentum_job(subject: Subject) -> None:
        await technical_arm(
            settings,
            subject,
            modes[subject.model],
            Dimension.MOMENTUM,
            activations,
            backends,
            clock,
            directory,
            collected,
        )

    for subject in subjects:
        if subject.model not in modes:
            skipped.append(f"{subject.model}: sin modo que responda, no se sondean veredictos")
            continue
        if not subject.present:
            jobs.append(candidate_job(subject))
        elif subject.role is AgentRole.MOMENTUM:
            jobs.append(momentum_job(subject))
    await asyncio.gather(*jobs)

    skipped += await chain_stage(
        settings, activations, collected, backends, clock, directory, modes
    )

    result = ProbeFindings(
        machine=machine,
        catalog_size=None if listed is None else len(listed),
        catalog_error=catalog_error,
        models=tuple(findings),
        header=header,
        skipped=tuple(skipped),
    )
    (directory / "findings.json").write_text(result.model_dump_json(indent=2) + "\n", "utf-8")
    return result


# ───────────────────────────────────────────── Informe ────────────────────────────────────────────


def _mean(values: Sequence[int]) -> str:
    return "—" if not values else f"{sum(values) / len(values):.0f}"


def _tokens(calls: Sequence[LLMCall]) -> str:
    """Medias por llamada viva y respondida de prompt / caché / salida, según el proveedor."""
    live = [
        c
        for c in calls
        if not c.cache_hit and c.failure_kind not in (FailureKind.TRANSPORT, FailureKind.TIMEOUT)
    ]
    prompt = [c.prompt_tokens for c in live if c.prompt_tokens is not None]
    cached = [c.cached_tokens for c in live if c.cached_tokens is not None]
    output = [c.completion_tokens for c in live if c.completion_tokens is not None]
    return f"{_mean(prompt)} / {_mean(cached)} / {_mean(output)}"


def _usd(calls: Sequence[LLMCall]) -> str:
    total = consume(calls, DEFAULT_PRICING, Billing.PAYG)
    text = f"{total.cost_usd:.4f}"
    if total.unmeasured == 0:
        return text
    ceiling = total.ceiling_usd
    bound = "sin cota" if ceiling is None else f"cota {ceiling:.4f}"
    return f"≥ {text} ({total.unmeasured} sin medir; {bound})"


def _arm_row(run: RunDirectory, arm_label: str, machine: str) -> str | None:
    arm = next((a for a in run.arms if a.arm == arm_label), None)
    if arm is None or arm.sha256 is None:
        return None
    records = arm.records
    calls = [call for record in records for call in record.calls]
    verdicts = len(records)
    valid = sum(1 for record in records if not record.errors)
    counts = failure_counts(records)
    attempts = attempt_counts(records).total
    latency = live_latency(records)
    retries = "—" if verdicts == 0 else f"{(attempts - verdicts) / verdicts:.2f}"
    mode = ", ".join(sorted({c.structured_output.value for c in calls})) or "—"
    speed = "—" if latency is None else f"{latency.mean_ms / 1000:.1f} s"
    return (
        f"| `{arm_label}` | {mode} | {valid}/{verdicts} | "
        f"{counts[FailureKind.SCHEMA]} | {counts[FailureKind.CONTEXT]} | "
        f"{counts[FailureKind.TIMEOUT]} | {counts[FailureKind.TRANSPORT]} | "
        f"{attempts} | {retries} | {speed} | {_tokens(calls)} | {_usd(calls)} |"
    )


def render_report(run: RunDirectory, findings: ProbeFindings) -> str:
    """Las tablas del sondeo, todas leídas del directorio que las respalda."""
    machine = findings.machine
    meta = run.meta
    lines = [
        f"# Sondeo de Zen — `{run.path}`",
        "",
        f"- máquina donde se midió la latencia: **{machine}**",
        f"- base_url: `{meta.base_url or 'no determinado'}` · facturación: "
        f"{meta.billing.value if meta.billing else 'no determinado'} · "
        f"precios de Zen al {DEFAULT_PRICING.as_of(Billing.PAYG)}",
        f"- plan de donde salen los prompts: `{meta.plan_path}` sha-256 `{meta.plan_sha256}`",
        f"- reproducir las cifras de coste: `python -m crypto_agents.consumption {run.path}`",
        "",
        "## Catálogo y modos",
        "",
    ]
    if findings.catalog_error is not None:
        lines += [f"El catálogo no se pudo leer: `{findings.catalog_error}`.", ""]
    elif findings.catalog_size is not None:
        lines += [f"`GET /models` lista {findings.catalog_size} ids.", ""]
    lines += [
        "| id | familia | rol | presente | en `/models` | declarado | responde en | nota |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in findings.models:
        listed = "no determinado" if item.listed is None else ("sí" if item.listed else "**NO**")
        lines.append(
            f"| `{item.model}` | {item.family} | {item.role.value} | "
            f"{'sí' if item.present else 'candidato'} | {listed} | {item.declared.value} | "
            f"{item.mode.value if item.mode else '**ninguno**'} | {item.note} |"
        )
    lines.append("")

    lines += ["## Cabecera `" + SESSION_NOTE + "`", ""]
    if findings.header is None:
        lines += ["No se pudo probar.", ""]
    else:
        shown = findings.header
        lines += [
            f"- modelo del `Ping`: `{shown.model}`, modo {shown.mode.value} "
            f"({'confirmado' if shown.mode_confirmed else 'solo declarado, sin confirmar'})",
            f"- con la cabecera: {shown.with_header}",
            f"- sin la cabecera: {shown.without_header}",
            "- probada: solo esta, que es la única propia del adaptador; `Authorization` y las "
            "del SDK van siempre.",
            "",
        ]

    header = (
        f"| brazo | modo | válidos | schema | context | timeout | transport | intentos | "
        f"reintentos/veredicto | latencia media ({machine}) | prompt / caché / salida | "
        "USD medido |"
    )
    separator = "|" + " --- |" * 12
    lines += ["## Veredictos y encadenado", "", header, separator]
    skipped_arms: list[str] = []
    for label in (
        a for a in meta.arms if not a.endswith(("@ping", "@header-with", "@header-without"))
    ):
        row = _arm_row(run, label, machine)
        if row is None:
            skipped_arms.append(label)
        else:
            lines.append(row)
    lines.append("")
    if skipped_arms or findings.skipped:
        lines += ["### No corrieron", ""]
        lines += [f"- `{arm}`: el brazo no escribió journal" for arm in skipped_arms]
        lines += [f"- {reason}" for reason in findings.skipped]
        lines.append("")

    arms = {arm.arm: [call for record in arm.records for call in record.calls] for arm in run.arms}
    inputs = [(arm.path.name, arm.sha256) for arm in run.arms if arm.sha256 is not None]
    lines.append(render_consumption(arms, DEFAULT_PRICING, Billing.PAYG, str(run.path), inputs))
    return "\n".join(lines)


# ───────────────────────────────────────── --dry-run y comando ────────────────────────────────────


def render_dry_run(subjects: Sequence[Subject], verdicts: int) -> str:
    """Cuántas llamadas haría el sondeo, como cota. Sin tokens aún no hay dólares que dar."""
    models = len(subjects)
    technical = len(CANDIDATES) * len(TECHNICAL_DIMENSIONS) * verdicts + verdicts
    chain = 3 * verdicts
    pings = (models, 3 * models)
    upper = pings[1] + 2 + TECHNICAL_ATTEMPTS * (technical + chain)
    return "\n".join(
        [
            "Sondeo de Zen — conteo previo (no llama a nadie)",
            "",
            f"- pings de modo: {pings[0]} a {pings[1]} intentos ({models} ids, hasta 3 modos)",
            "- cabecera: 2 intentos",
            f"- veredictos técnicos: {technical} (≤ {TECHNICAL_ATTEMPTS * technical} intentos)",
            f"- encadenado: {chain} veredictos de mesas y decisor (≤ {TECHNICAL_ATTEMPTS * chain})",
            f"- total: ≤ {upper} intentos. Dólares: no determinado, aún no hay tokens medidos.",
            "",
        ]
    )


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m crypto_agents.zen_probe",
        description="Sondea OpenCode Zen (pago por uso). Cuesta dinero salvo con --dry-run.",
    )
    parser.add_argument("--machine", required=True, choices=["desktop", "laptop"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verdicts", type=int, default=VERDICTS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--history-dir", type=Path, default=HISTORY_DIR)
    parser.add_argument("--out", type=Path, default=PROBE_DIR)
    args = parser.parse_args(argv)
    if args.verdicts < 1:
        parser.error("--verdicts debe ser al menos 1")
    args.raw_argv = tuple(argv if argv is not None else sys.argv[1:])
    return args


async def _main(args: argparse.Namespace) -> str:
    settings = load_settings(DEFAULT_ENV_FILE)
    refuse_unless_zen(settings)
    assert settings.openai is not None  # lo garantiza `refuse_unless_zen`
    subjects = subjects_of(settings)
    if args.dry_run:
        return render_dry_run(subjects, args.verdicts)

    manifest = load_manifest(args.manifest)
    plan = plan_from_manifest(manifest, load_selection_histories(manifest, args.history_dir))
    started = utc_now()
    directory = args.out / started.strftime("%Y%m%dT%H%M%SZ")
    commit, dirty = git_state()
    open_run_directory(
        directory,
        build_meta(settings, args.manifest, args.raw_argv, started, commit, dirty),
    )
    print(f"sondeo en {directory}", file=sys.stderr)

    backends = build_backends(settings)
    openai = backends[Backend.OPENAI]
    assert isinstance(openai, OpenAIBackend)
    headerless = _NoSessionBackend(
        settings.openai.api_key.get_secret_value(),
        settings.openai.base_url,
        settings.openai.timeout_seconds,
    )
    findings = await run_probe(
        settings,
        plan,
        directory,
        args.machine,
        backends,
        headerless,
        openai,
        utc_now,
        args.verdicts,
    )
    report = render_report(read_run(directory), findings)
    (directory / "report.md").write_text(report, "utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada: `python -m crypto_agents.zen_probe`."""
    args = _parse(argv)
    try:
        report = asyncio.run(_main(args))
    except (ConfigError, SelectionError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
