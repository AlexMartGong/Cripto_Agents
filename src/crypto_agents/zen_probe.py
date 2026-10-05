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
import re
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
from crypto_agents.audit import (
    PlanKind,
    RunDirectory,
    RunKind,
    RunMeta,
    arm_journal_path,
    read_run,
)
from crypto_agents.audit import file_sha256 as sha256_of
from crypto_agents.cache import InMemoryResponseCache
from crypto_agents.consumption import consume, format_cost, render_consumption
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
from crypto_agents.metrics import attempt_counts, failure_counts, live_latency, provider_rejections
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
    from collections.abc import Coroutine, Iterable, Mapping, Sequence
    from datetime import datetime

    from crypto_agents.llm import ChatBackend, ModelCatalog
    from crypto_agents.quota import Clock
    from crypto_agents.settings import Settings
    from crypto_agents.state import LLMCall

__all__ = [
    "BULL_CANDIDATES",
    "CANDIDATES",
    "MOMENTUM_PRODUCERS",
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


def _present(settings: Settings, model: str, role: AgentRole) -> Subject:
    """Un id que el mapa de roles ya lleva, con el modo que declara `.env`."""
    declared = settings.role_config(role).primary
    if declared.model != model:
        raise ConfigError(
            f"{role.value} ya no declara {model} (declara {declared.model}): "
            "el sondeo de los presentes parte del mapa de roles de .env"
        )
    return Subject(model, declared.family, role, _unmetered(declared), True)


def subjects_of(settings: Settings) -> tuple[Subject, ...]:
    """Los cuatro presentes (con el modo que declara `.env`) y los seis candidatos."""
    result: list[Subject] = [_present(settings, model, role) for model, role in PRESENT]
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


# ────────────────────────────────────────── Tope de gasto ─────────────────────────────────────────


class SpendGuard:
    """Tope duro de un sondeo: no abre una invocación nueva cuando lo gastado lo alcanza.

    No existe una cota de coste rigurosa *antes* de llamar: el repo no fija `max_tokens`, de modo
    que la salida —el razonamiento incluido— no tiene techo, y el prompt de las mesas depende de
    veredictos que aún no existen. Estimar tokens rompe la regla del repo. Lo que sí se puede es
    declarar un tope y pararse en él: el gasto es el **extremo alto** de lo que el proveedor
    declaró (`Consumption.cost_upper_usd` más la cota de las llamadas con `cached_tokens` en
    `null`), y se mira antes de cada invocación.

    Dos límites que el informe repite: las invocaciones que ya estaban en vuelo al alcanzar el tope
    terminan y se suman después (el exceso es, como mucho, una invocación de hasta dos intentos por
    trabajo concurrente), y las llamadas sin tokens no tienen cifra que sumar, así que no cuentan
    para el tope aunque se facturen.
    """

    def __init__(self, cap_usd: float) -> None:
        if cap_usd <= 0:
            raise ValueError("el tope de gasto debe ser positivo")
        self.cap_usd = cap_usd
        self.refused: list[str] = []
        self._calls: list[LLMCall] = []

    def add(self, calls: Iterable[LLMCall]) -> None:
        """Anota lo que el proveedor acaba de ver: se llama con cada intento, válido o no."""
        self._calls.extend(calls)

    @property
    def spent_usd(self) -> float:
        """Lo gastado hasta ahora, en el extremo alto."""
        total = consume(self._calls, DEFAULT_PRICING, Billing.PAYG)
        return total.cost_upper_usd + total.unreported_ceiling_usd

    @property
    def unpriced(self) -> int:
        """Llamadas respondidas sin tokens: se facturaron y no suman al gasto del tope."""
        return consume(self._calls, DEFAULT_PRICING, Billing.PAYG).no_tokens

    def allow(self, label: str) -> bool:
        """Si cabe abrir otra invocación; si no, deja anotado qué se dejó sin hacer."""
        if self.spent_usd < self.cap_usd:
            return True
        self.refused.append(label)
        return False


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

    technicals_from: str | None = None
    """Sondeo técnico del que salió la regla de los productores de evidencia (solo `--desks`)."""

    producers: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    """Dimensión → quién puede producir la evidencia común de las mesas, por orden (`--desks`)."""

    activations: tuple[str, ...] = ()
    """Las activaciones sondeadas, en orden: símbolo e instante de la vela (solo `--desks`)."""

    evidence_by: dict[str, tuple[str | None, ...]] = Field(default_factory=dict)
    """Dimensión → por activación, el productor cuyo veredicto leyeron las mesas, o `None`."""

    spend_cap_usd: float | None = None
    spent_upper_usd: float | None = None
    """Lo gastado en el extremo alto, tal como lo veía el tope. Las llamadas sin tokens no suman."""

    unpriced_calls: int = 0
    refused: tuple[str, ...] = ()
    """Invocaciones que el tope impidió: lo que se dejó sin medir por haberlo alcanzado."""


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
    guard: SpendGuard | None = None,
) -> None:
    """Un veredicto por activación con el prompt técnico real, en serie, escribiendo cada uno.

    Las llamadas de un mismo modelo van una detrás de otra, como las hace el grafo; lo que corre a
    la vez son los candidatos entre sí. Los veredictos válidos se guardan para el encadenado.
    Con `guard`, una activación que encuentra el tope alcanzado no se pregunta.
    """
    choice = subject.choice.model_copy(update={"structured_output": mode})
    router = _router(settings, _ROLE_OF[dimension], choice, backends, clock)
    arm = arm_name(subject.model, dimension.value)
    for position, activation in enumerate(activations):
        if guard is not None and not guard.allow(f"{arm} #{position}"):
            continue
        verdict = await _verdict(router, arm, dimension, activation, directory, clock, guard)
        if verdict is not None:
            verdicts[(subject.model, dimension, position)] = verdict


async def _verdict(
    router: ModelRouter,
    arm: str,
    dimension: Dimension,
    activation: Activation,
    directory: Path,
    clock: Clock,
    guard: SpendGuard | None,
) -> TechnicalVerdict | None:
    """Una invocación con el prompt técnico real: deja su registro y devuelve el veredicto."""
    prepared = activation.prepared
    prompt = technical_prompt(
        dimension, prepared.snapshot, prepared.indicators, prepared.activation.triggers
    )
    check = technical_context(dimension, prepared.indicators)
    try:
        verdict, calls = await router.invoke(_ROLE_OF[dimension], prompt, TechnicalVerdict, check)
    except ModelInvocationError as error:
        if guard is not None:
            guard.add(error.calls)
        _write(directory, arm, activation.entry, error.calls, dimension.value, str(error), clock)
        return None
    if guard is not None:
        guard.add(calls)
    _write(directory, arm, activation.entry, calls, dimension.value, None, clock)
    return verdict


def producer_mode(report: RoleReport) -> StructuredOutputMode | None:
    """El modo en que se le pide evidencia a un productor, o `None` si el `Ping` lo descartó.

    Un rechazo del proveedor antes de generar nada no desmiente el modo declarado —es el criterio
    de `doctor`—, y una ruta que va y viene puede fallar en el `Ping` y contestar un minuto después:
    ese productor se intenta igual en cada activación. Un rechazo no se factura. Quedan fuera el que
    contestó y no validó en ningún modo, y el que agotó el plazo: preguntarle doce veces más es
    retener el plazo entero doce veces.
    """
    confirmed = confirmed_mode(report)
    if confirmed is not None:
        return confirmed
    timed_out = any(call.failure_kind is FailureKind.TIMEOUT for call in report.calls)
    return report.declared if report.transport and not timed_out else None


async def evidence_arm(
    settings: Settings,
    producers: Sequence[tuple[Subject, StructuredOutputMode]],
    dimension: Dimension,
    activations: Sequence[Activation],
    backends: Mapping[Backend, ChatBackend],
    clock: Clock,
    directory: Path,
    guard: SpendGuard | None = None,
) -> dict[int, tuple[str, TechnicalVerdict]]:
    """La evidencia de una dimensión: por activación, el primer productor con un veredicto válido.

    `producers` va por orden de preferencia. Cada activación empieza por el primero, así que un
    productor que falla en una no pierde las siguientes, y el suplente solo se pregunta donde hizo
    falta. Cada intento queda en el brazo `<modelo>@<dimensión>` de quien lo hizo. Devuelve, por
    posición, quién produjo el veredicto y el veredicto: es lo único que las mesas van a leer.
    """
    role = _ROLE_OF[dimension]
    routers = [
        (
            subject,
            _router(
                settings,
                role,
                subject.choice.model_copy(update={"structured_output": mode}),
                backends,
                clock,
            ),
        )
        for subject, mode in producers
    ]
    chosen: dict[int, tuple[str, TechnicalVerdict]] = {}
    for position, activation in enumerate(activations):
        for subject, router in routers:
            arm = arm_name(subject.model, dimension.value)
            if guard is not None and not guard.allow(f"{arm} #{position}"):
                break
            verdict = await _verdict(router, arm, dimension, activation, directory, clock, guard)
            if verdict is not None:
                chosen[position] = (subject.model, verdict)
                break
    return chosen


def _rank(valid: Mapping[str, int], candidates: Sequence[str], at: datetime) -> list[str]:
    """Quién produce los veredictos, el mejor primero: más válidos y, a igualdad, más barato.

    El precio es el de salida del modelo en pago por uso. El encadenado solo necesita entradas
    verosímiles para las mesas, no una opinión sobre quién gana.
    """

    def price(model: str) -> float:
        row = DEFAULT_PRICING.price_for(model, Billing.PAYG, at)
        return row.output_per_mtok if row is not None else float("inf")

    return sorted(
        (m for m in candidates if valid.get(m, 0) > 0), key=lambda m: (-valid[m], price(m), m)
    )


def producers(
    verdicts: Mapping[tuple[str, Dimension, int], TechnicalVerdict],
    dimension: Dimension,
    candidates: Sequence[str],
    at: datetime,
) -> list[str]:
    """Quién produce los veredictos de una dimensión para el encadenado, el mejor primero.

    Gana el que más veredictos válidos dio y, a igualdad, el más barato por token de salida.
    """
    valid = {
        model: sum(1 for (m, d, _), _v in verdicts.items() if m == model and d is dimension)
        for model in candidates
    }
    return _rank(valid, candidates, at)


def producers_from_run(
    run: RunDirectory, dimension: Dimension, candidates: Sequence[str], at: datetime
) -> list[str]:
    """La misma regla de `producers`, leyendo los veredictos válidos del journal de un sondeo.

    Un registro sin error es un veredicto válido del brazo `<modelo>@<dimensión>`. Los veredictos
    en sí no se guardan —el journal lleva `LLMCall`, no contenido—, y por eso las mesas los piden
    de nuevo, pero solo al productor que esta regla elige.
    """
    valid: dict[str, int] = {}
    for arm in run.arms:
        model, _, label = arm.arm.rpartition("@")
        if label == dimension.value and model in candidates:
            valid[model] = sum(1 for record in arm.records if not record.errors)
    return _rank(valid, candidates, at)


async def _desk(
    side: Side,
    router: ModelRouter,
    model: str,
    activation: Activation,
    evidence: tuple[TechnicalVerdict, ...],
    directory: Path,
    clock: Clock,
    guard: SpendGuard | None = None,
) -> DebateBrief | None:
    """Una mesa argumenta sobre la evidencia común, con su prompt y su validación de contexto.

    `evidence` es el mismo objeto para todas las mesas de una activación: el prompt de una mesa
    es función de ella y de nada más, así que dos candidatos al mismo rol reciben el mismo texto.
    """
    role = AgentRole.BULL if side is Side.BULL else AgentRole.BEAR
    prepared = activation.prepared
    prompt = debate_prompt(side, prepared.snapshot, prepared.indicators, evidence)
    arm = arm_name(model, role.value)
    if guard is not None and not guard.allow(f"{arm} {activation.entry.symbol}"):
        return None
    try:
        brief, calls = await router.invoke(
            role, prompt, DebateBrief, debate_context(side, evidence)
        )
    except ModelInvocationError as error:
        if guard is not None:
            guard.add(error.calls)
        _write(directory, arm, activation.entry, error.calls, role.value, str(error), clock)
        return None
    if guard is not None:
        guard.add(calls)
    _write(directory, arm, activation.entry, calls, role.value, None, clock)
    return brief


async def _decide(
    router: ModelRouter,
    model: str,
    label: str,
    activation: Activation,
    evidence: tuple[TechnicalVerdict, ...],
    bull: DebateBrief,
    bear: DebateBrief,
    directory: Path,
    clock: Clock,
    guard: SpendGuard | None = None,
) -> bool:
    """El decisor con su prompt real sobre esas dos mesas. Devuelve si dio una decisión válida.

    La salida se descarta: solo se mide lo que cuesta preguntar. `label` nombra el brazo, y con él
    de qué bull depende el coste (`decider+<bull>`).
    """
    prepared = activation.prepared
    prompt = decision_prompt(prepared.snapshot, prepared.indicators, evidence, bull, bear)
    arm = arm_name(model, label)
    if guard is not None and not guard.allow(f"{arm} {activation.entry.symbol}"):
        return False
    try:
        _, calls = await router.invoke(AgentRole.DECIDER, prompt, Decision)
    except ModelInvocationError as error:
        if guard is not None:
            guard.add(error.calls)
        _write(directory, arm, activation.entry, error.calls, "decider", str(error), clock)
        return False
    if guard is not None:
        guard.add(calls)
    _write(directory, arm, activation.entry, calls, "decider", None, clock)
    return True


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
                _desk(
                    side,
                    routers[AgentRole.BULL if side is Side.BULL else AgentRole.BEAR],
                    models[AgentRole.BULL if side is Side.BULL else AgentRole.BEAR],
                    activation,
                    evidence,
                    directory,
                    clock,
                )
                for side in (Side.BULL, Side.BEAR)
            )
        )
        if bull is None or bear is None:
            skipped.append(f"encadenado #{position}: una mesa falló, no hay decisor que medir")
            continue
        await _decide(
            routers[AgentRole.DECIDER],
            models[AgentRole.DECIDER],
            AgentRole.DECIDER.value,
            activation,
            evidence,
            bull,
            bear,
            directory,
            clock,
        )
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
    arms: Sequence[str] | None = None,
) -> RunMeta:
    """El `meta.json` del sondeo: con su plan, su pasarela y su pago, y `kind=probe`.

    Es un directorio que `audit` y `consumption` leen como el de una corrida, pero no lo es:
    `criteria` lo rechaza por el `kind`.

    El plan es el manifiesto del que salen los prompts. `base_url` pasa por `public_url`: la
    pasarela queda escrita, la clave y cualquier credencial de la URL no.
    """
    return RunMeta(
        kind=RunKind.PROBE,
        plan_kind=PlanKind.MANIFEST,
        plan_path=str(manifest),
        plan_sha256=sha256_of(manifest),
        argv=tuple(argv),
        fill=True,
        arms=tuple(arms) if arms is not None else all_arms(settings),
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


# ───────────────────────────────────────────── Sondeo de mesas ────────────────────────────────────

BULL_CANDIDATES: Final = (
    Candidate("kimi-k3", "moonshot"),
    Candidate("qwen3.8-max", "qwen"),
    Candidate("deepseek-v4-pro", "deepseek"),
)
"""Los modelos que podrían llevar `bull`. Familia declarada a mano, como en `ModelChoice`.

`kimi-k2.6`, el que `.env` declara hoy, no está: Zen lo lista y contesta 410 de forma sostenida, y
sondearlo otra vez es pagar la misma respuesta. Vuelve si soporte dice que no fue retirado."""

MOMENTUM_PRODUCERS: Final = ("deepseek-v4-flash", "deepseek-v4-pro")
"""Quién produce el veredicto de `momentum` que leen las mesas, por orden de preferencia.

El primero es el modelo del mapa de roles; el segundo lo suple en la activación donde aquel no da
un veredicto válido. El orden lo fija quien encarga el sondeo, no la regla de `producers`: no
elige el modelo del rol, solo evita que una ruta caída deje a todas las mesas sin evidencia."""


class TechnicalSource(NamedTuple):
    """De dónde salen los productores de la evidencia común de las mesas."""

    path: Path
    producers: dict[Dimension, str]
    """Structure y volume: el candidato con más veredictos válidos en ese sondeo."""

    sources: tuple[tuple[str, str], ...]
    """Los journals de los que se contaron los válidos, con su sha-256."""


def technical_source(directory: Path, at: datetime) -> TechnicalSource:
    """Aplica la regla de `producers` a los journals de un sondeo técnico ya hecho.

    El sondeo de las mesas no repite los seis candidatos de structure y volume —serían 144
    llamadas— sino que reutiliza la regla sobre lo que aquel ya midió: más veredictos válidos y, a
    igualdad, el más barato por token de salida. No elige el modelo de ningún rol: elige quién
    produce la evidencia que todas las mesas van a leer.
    """
    run = read_run(directory)
    candidates = [c.model for c in CANDIDATES]
    chosen: dict[Dimension, str] = {}
    for dimension in TECHNICAL_DIMENSIONS:
        ranked = producers_from_run(run, dimension, candidates, at)
        if not ranked:
            raise ConfigError(
                f"{directory}: ningún candidato dio un veredicto válido de {dimension.value}"
            )
        chosen[dimension] = ranked[0]
    labels = {d.value for d in TECHNICAL_DIMENSIONS}
    consulted = tuple(
        (f"{run.path.name}/{arm.path.name}", arm.sha256)
        for arm in run.arms
        if arm.sha256 is not None
        and arm.arm.rpartition("@")[2] in labels
        and arm.arm.rpartition("@")[0] in candidates
    )
    return TechnicalSource(directory, chosen, consulted)


class DeskSubjects(NamedTuple):
    """Quién se sondea en la etapa de mesas y bajo qué rol."""

    bulls: tuple[Subject, ...]
    bear: Subject
    decider: Subject
    technicals: dict[Dimension, tuple[Subject, ...]]
    """Los productores de la evidencia común: por dimensión, una lista por orden de preferencia."""

    @property
    def producers(self) -> tuple[Subject, ...]:
        """Todos los productores, de todas las dimensiones."""
        return tuple(subject for group in self.technicals.values() for subject in group)

    @property
    def pings(self) -> tuple[Subject, ...]:
        """Un `Ping` por id aunque lleve varios roles: el modo es del modelo, no del rol."""
        seen: dict[str, Subject] = {}
        for subject in (*self.bulls, self.bear, self.decider, *self.producers):
            seen.setdefault(subject.model, subject)
        return tuple(seen.values())


def _candidate(settings: Settings, model: str, family: str, role: AgentRole) -> Subject:
    """Un id que el mapa de roles no lleva en ese rol: parte de `json_schema`, sin cuota de Go."""
    choice = ModelChoice(
        backend=Backend.OPENAI,
        model=model,
        family=family,
        structured_output=StructuredOutputMode.JSON_SCHEMA,
        quota_weight=1.0,
        quota_per_window=ZEN_UNPUBLISHED_QUOTA,
        temperature=settings.role_config(role).primary.temperature,
    )
    return Subject(model, family, role, choice, False)


def desk_subjects(settings: Settings, source: TechnicalSource) -> DeskSubjects:
    """Los candidatos a bull, el bear y el decisor del mapa de roles, y los productores.

    Structure y volume llevan un productor cada una, el que la regla saca del sondeo técnico.
    Momentum lleva la lista de `MOMENTUM_PRODUCERS`: el primero con el modo que declara `.env`, el
    suplente como cualquier candidato.
    """
    bulls = tuple(_candidate(settings, c.model, c.family, AgentRole.BULL) for c in BULL_CANDIDATES)
    families = {c.model: c.family for c in CANDIDATES}
    technicals: dict[Dimension, tuple[Subject, ...]] = {
        dimension: (
            _candidate(
                settings,
                source.producers[dimension],
                families[source.producers[dimension]],
                _ROLE_OF[dimension],
            ),
        )
        for dimension in TECHNICAL_DIMENSIONS
    }
    first, *substitutes = MOMENTUM_PRODUCERS
    technicals[Dimension.MOMENTUM] = (
        _present(settings, first, AgentRole.MOMENTUM),
        *(
            _candidate(settings, model, families[model], AgentRole.MOMENTUM)
            for model in substitutes
        ),
    )
    return DeskSubjects(
        bulls,
        _present(settings, "minimax-m3", AgentRole.BEAR),
        _present(settings, "glm-5.2", AgentRole.DECIDER),
        technicals,
    )


def decider_label(bull: str) -> str:
    """El brazo del decisor condicionado a un bull: el prompt lleva su alegato."""
    return f"{AgentRole.DECIDER.value}+{bull}"


def all_desk_arms(subjects: DeskSubjects) -> tuple[str, ...]:
    """Todos los brazos del sondeo de mesas, declarados antes de la primera llamada."""
    arms = [arm_name(s.model, "ping") for s in subjects.pings]
    arms += [arm_name(s.model, d.value) for d, group in subjects.technicals.items() for s in group]
    arms += [arm_name(s.model, AgentRole.BULL.value) for s in subjects.bulls]
    arms.append(arm_name(subjects.bear.model, AgentRole.BEAR.value))
    arms += [arm_name(subjects.decider.model, decider_label(s.model)) for s in subjects.bulls]
    return tuple(arms)


async def run_desks_probe(
    settings: Settings,
    plan: ReplayPlan,
    directory: Path,
    machine: str,
    backends: Mapping[Backend, ChatBackend],
    source: TechnicalSource,
    guard: SpendGuard,
    clock: Clock = utc_now,
    activations_count: int = VERDICTS,
    preset: IndicatorPreset = DEFAULT_PRESET,
) -> ProbeFindings:
    """Mesas y decisor sobre la misma evidencia, con `max_attempts=2`, sin caché y sin respaldo.

    La evidencia se produce **una vez** por activación y es el único insumo de todas las mesas: dos
    candidatos a bull reciben el mismo prompt, byte a byte, y lo único que cambia entre sus filas
    es el modelo. Quién la produjo puede cambiar de una activación a otra (`evidence_arm`); que sea
    la misma para todas las mesas de esa activación, no. El bear no depende de ningún bull, así que
    se mide aunque todos fallen. El decisor corre una vez por (activación, bull con alegato válido)
    con el bear de esa activación: su coste queda condicionado a cada bull y no se promedia.
    """
    subjects = desk_subjects(settings, source)
    activations = await prepare_activations(plan, settings, clock, activations_count, preset)

    pings = subjects.pings
    reports = await ping_subjects(settings, pings, backends, clock, directory)
    for report in reports.values():
        guard.add(report.calls)
    modes = {m: mode for m, r in reports.items() if (mode := confirmed_mode(r)) is not None}
    findings = [
        ModelFinding(
            model=s.model,
            family=s.family,
            role=s.role,
            present=s.present,
            listed=None,
            declared=s.choice.structured_output,
            mode=modes.get(s.model),
            note=_note(reports[s.model]),
        )
        for s in pings
    ]

    skipped: list[str] = []
    produced: dict[Dimension, dict[int, tuple[str, TechnicalVerdict]]] = {}

    async def evidence_job(dimension: Dimension, group: Sequence[Subject]) -> None:
        ready: list[tuple[Subject, StructuredOutputMode]] = []
        for subject in group:
            mode = producer_mode(reports[subject.model])
            if mode is None:
                skipped.append(
                    f"{arm_name(subject.model, dimension.value)}: el `Ping` lo descartó "
                    f"({_note(reports[subject.model])}); no produce evidencia"
                )
                continue
            ready.append((subject, mode))
        produced[dimension] = await evidence_arm(
            settings, ready, dimension, activations, backends, clock, directory, guard
        )

    await asyncio.gather(*(evidence_job(d, group) for d, group in subjects.technicals.items()))

    evidence: dict[int, tuple[TechnicalVerdict, ...]] = {}
    for position in range(len(activations)):
        valid = [produced[d][position][1] for d in Dimension if position in produced[d]]
        if len(valid) < len(Dimension):
            skipped.append(f"mesas #{position}: faltan veredictos técnicos válidos")
            continue
        evidence[position] = tuple(sorted(valid, key=lambda item: item.dimension.value))

    def router_for(subject: Subject, role: AgentRole) -> ModelRouter | None:
        mode = modes.get(subject.model)
        if mode is None:
            skipped.append(f"{arm_name(subject.model, role.value)}: sin modo que responda")
            return None
        choice = subject.choice.model_copy(update={"structured_output": mode})
        return _router(settings, role, choice, backends, clock)

    briefs: dict[tuple[str, int], DebateBrief] = {}
    bears: dict[int, DebateBrief] = {}

    async def bull_job(subject: Subject, router: ModelRouter) -> None:
        for position, shared in evidence.items():
            brief = await _desk(
                Side.BULL,
                router,
                subject.model,
                activations[position],
                shared,
                directory,
                clock,
                guard,
            )
            if brief is not None:
                briefs[(subject.model, position)] = brief

    async def bear_job(router: ModelRouter) -> None:
        for position, shared in evidence.items():
            brief = await _desk(
                Side.BEAR,
                router,
                subjects.bear.model,
                activations[position],
                shared,
                directory,
                clock,
                guard,
            )
            if brief is not None:
                bears[position] = brief

    desk_jobs: list[Coroutine[object, object, None]] = []
    for subject in subjects.bulls:
        if (bull_router := router_for(subject, AgentRole.BULL)) is not None:
            desk_jobs.append(bull_job(subject, bull_router))
    if (bear_router := router_for(subjects.bear, AgentRole.BEAR)) is not None:
        desk_jobs.append(bear_job(bear_router))
    await asyncio.gather(*desk_jobs)

    if (decider_router := router_for(subjects.decider, AgentRole.DECIDER)) is not None:

        async def decider_job(subject: Subject, router: ModelRouter) -> None:
            for position, shared in evidence.items():
                bull, bear = briefs.get((subject.model, position)), bears.get(position)
                if bull is None or bear is None:
                    continue
                await _decide(
                    router,
                    subjects.decider.model,
                    decider_label(subject.model),
                    activations[position],
                    shared,
                    bull,
                    bear,
                    directory,
                    clock,
                    guard,
                )

        await asyncio.gather(*(decider_job(s, decider_router) for s in subjects.bulls))
        skipped += [
            f"decisor #{position}: sin bear válido; no se mide para ningún bull"
            for position in evidence
            if position not in bears
        ]

    if guard.refused:
        skipped.append(
            f"tope de {guard.cap_usd:.2f} USD alcanzado: {len(guard.refused)} invocación(es) "
            "sin hacer"
        )
    result = ProbeFindings(
        machine=machine,
        models=tuple(findings),
        skipped=tuple(skipped),
        technicals_from=str(source.path),
        producers={
            d.value: tuple(s.model for s in group) for d, group in subjects.technicals.items()
        },
        activations=tuple(f"{a.entry.symbol} {a.entry.at:%Y-%m-%dT%H:%MZ}" for a in activations),
        evidence_by={
            d.value: tuple(
                produced[d][position][0] if position in produced[d] else None
                for position in range(len(activations))
            )
            for d in Dimension
        },
        spend_cap_usd=guard.cap_usd,
        spent_upper_usd=guard.spent_usd,
        unpriced_calls=guard.unpriced,
        refused=tuple(guard.refused),
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
    text = format_cost(total)
    if total.unmeasured == 0:
        return text
    ceiling = total.ceiling_usd
    bound = "sin cota" if ceiling is None else f"cota {ceiling:.4f}"
    return f"{text} ({total.unmeasured} sin medir; {bound})"


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


# ───────────────────────────────────────── Informe de las mesas ───────────────────────────────────

_SECRET = re.compile(
    r"(?i)(bearer\s+\S+|authorization\S*\s*[:=]\s*\S+|(?:api[_-]?key|token|secret)\S*\s*[:=]\s*\S+"
    r"|\bsk-[A-Za-z0-9_\-]{6,}|(?<=://)[^/\s:@]+:[^@\s]+(?=@))"
)
"""Lo que no puede salir en un informe: una clave, una cabecera, credenciales en una URL."""


def redact(text: str) -> str:
    """El cuerpo de un error del proveedor, sin nada que parezca una credencial.

    Un proveedor puede devolver en un 401 un trozo de la clave o la cabecera que recibió. El
    informe cita el cuerpo para distinguir un 410 de un 503; no tiene por qué citar eso.
    """
    return _SECRET.sub("[redactado]", text)


def _cell(text: str, width: int = 240) -> str:
    """Un cuerpo de error apto para una celda de tabla: sin credenciales, sin `|`, acotado."""
    clean = redact(text).replace("|", "\\|").replace("\n", " ")
    return clean if len(clean) <= width else clean[: width - 1] + "…"


def render_rejections(run: RunDirectory) -> list[str]:
    """Cada rechazo del proveedor con su código y su cuerpo, en filas separadas por código."""
    found = provider_rejections(record for arm in run.arms for record in arm.records)
    lines = [
        "## Rechazos del proveedor",
        "",
        "Un código y un cuerpo distintos son filas distintas: un 410 («endpoint unavailable») y un "
        "503 no se suman. Es lo que el proveedor contestó, no lo que el modelo hizo.",
        "",
    ]
    if not found:
        return [*lines, "Ninguno.", ""]
    lines += ["| modelo | rol | código | cuerpo | intentos |", "| --- | --- | --- | --- | --- |"]
    for (model, role, code, body), count in found.items():
        lines.append(f"| `{model}` | {role.value} | {code} | {_cell(body)} | {count} |")
    return [*lines, ""]


def render_families(subjects: DeskSubjects) -> list[str]:
    """Cada candidato a bull frente a las familias del bear y del decisor."""
    bear, decider = subjects.bear, subjects.decider
    lines = [
        "## Familias: bull frente a bear y al decisor",
        "",
        f"bear: `{bear.model}` ({bear.family}) · decisor: `{decider.model}` ({decider.family}). "
        "La restricción dura del repo es `bull != bear`.",
        "",
        "| candidato a bull | familia | frente a bear | frente al decisor |",
        "| --- | --- | --- | --- |",
    ]
    for bull in subjects.bulls:
        versus = [
            "distinta" if bull.family != other.family else "**IGUAL**" for other in (bear, decider)
        ]
        lines.append(f"| `{bull.model}` | {bull.family} | {versus[0]} | {versus[1]} |")
    momentum = subjects.technicals[Dimension.MOMENTUM]
    for producer in momentum:
        shared = [b.model for b in subjects.bulls if b.family == producer.family]
        if shared:
            lines += [
                "",
                f"Fuera de esa restricción: {', '.join(f'`{m}`' for m in shared)} comparte familia "
                f"con `momentum` (`{producer.model}`, {producer.family}).",
            ]
    own = [b.model for b in subjects.bulls if b.model in {p.model for p in momentum}]
    if own:
        lines += [
            "",
            f"{', '.join(f'`{m}`' for m in own)} es además productor de `momentum`: en las "
            "activaciones donde produjo la evidencia, ese candidato a bull lee un veredicto de su "
            "mismo modelo (ver «Evidencia común»).",
        ]
    return [*lines, ""]


def render_sources(run: RunDirectory) -> list[str]:
    """De qué filas y de qué archivo sale cada cifra: el sha-256 completo de cada journal."""
    lines = [
        "## Fuentes",
        "",
        "`python -m crypto_agents.consumption <directorio>` recalcula cada USD de estos archivos.",
        "",
        "| brazo | archivo | filas `LLMCall` | sha-256 |",
        "| --- | --- | --- | --- |",
    ]
    for arm in run.arms:
        if arm.sha256 is None:
            continue
        rows = sum(len(record.calls) for record in arm.records)
        lines.append(f"| `{arm.arm}` | `{arm.path.name}` | {rows} | `{arm.sha256}` |")
    return [*lines, ""]


def render_evidence(findings: ProbeFindings) -> list[str]:
    """Quién produjo la evidencia de cada activación: lo mismo para todas las mesas de esa fila."""
    lines = [
        "## Evidencia común",
        "",
        "La evidencia se produjo **una vez** por activación y es la misma para todas las mesas. "
        "Por dimensión hay una lista de productores por orden de preferencia, y cada activación "
        "usa el primero que da un veredicto válido. Structure y volume salen de la regla de "
        "`producers` (más válidos; a igualdad, el más barato por token de salida) aplicada a "
        f"`{findings.technicals_from}`; el orden de momentum lo fija `MOMENTUM_PRODUCERS`.",
        "",
    ]
    for dimension, models in findings.producers.items():
        used = findings.evidence_by.get(dimension, ())
        counts = ", ".join(f"`{m}` {sum(1 for u in used if u == m)}/{len(used)}" for m in models)
        lines.append(f"- {dimension}: {' → '.join(f'`{m}`' for m in models)} · usó: {counts}")
    dimensions = list(findings.producers)
    lines += [
        "",
        "| # | activación | " + " | ".join(dimensions) + " | mesas |",
        "| --- | --- |" + " --- |" * (len(dimensions) + 1),
    ]
    for position, label in enumerate(findings.activations):
        used_here = [findings.evidence_by[d][position] for d in dimensions]
        cells = ["**ninguno**" if model is None else f"`{model}`" for model in used_here]
        desks = "sin evidencia: no se midieron" if None in used_here else "sí"
        lines.append(f"| {position} | {label} | " + " | ".join(cells) + f" | {desks} |")
    return [*lines, ""]


def render_cap_cut(run: RunDirectory, findings: ProbeFindings, subjects: DeskSubjects) -> list[str]:
    """Con el tope alcanzado, cuántas activaciones llegó a medir cada candidato a bull.

    «Medidas» es registros escritos: la invocación se hizo y dejó sus filas, validara o no. El
    denominador son las activaciones sondeadas; a una sin evidencia común tampoco se le preguntó.
    """
    if not findings.refused:
        return []
    by_arm = {arm.arm: arm.records for arm in run.arms}
    total = len(findings.activations)

    def measured(arm: str) -> str:
        return f"{len(by_arm.get(arm, ()))}/{total}"

    bear = arm_name(subjects.bear.model, AgentRole.BEAR.value)
    lines = [
        f"El tope cortó la corrida. Activaciones medidas de {total} (bear `{subjects.bear.model}`: "
        f"{measured(bear)}):",
        "",
        "| candidato a bull | bull medidas | decisor condicionado medidas |",
        "| --- | --- | --- |",
    ]
    for bull in subjects.bulls:
        lines.append(
            f"| `{bull.model}` | {measured(arm_name(bull.model, AgentRole.BULL.value))} | "
            f"{measured(arm_name(subjects.decider.model, decider_label(bull.model)))} |"
        )
    return [*lines, ""]


def render_desks_report(run: RunDirectory, findings: ProbeFindings, subjects: DeskSubjects) -> str:
    """Las tablas del sondeo de mesas, todas leídas del directorio que las respalda."""
    machine = findings.machine
    meta = run.meta
    lines = [
        f"# Sondeo de mesas de Zen — `{run.path}`",
        "",
        f"- máquina donde se midió la latencia: **{machine}**",
        f"- base_url: `{meta.base_url or 'no determinado'}` · facturación: "
        f"{meta.billing.value if meta.billing else 'no determinado'} · "
        f"precios de Zen al {DEFAULT_PRICING.as_of(Billing.PAYG)}",
        f"- plan de donde salen los prompts: `{meta.plan_path}` sha-256 `{meta.plan_sha256}`",
        f"- reproducir las cifras de coste: `python -m crypto_agents.consumption {run.path}`",
        "",
    ]
    lines += render_evidence(findings)
    if findings.spend_cap_usd is not None:
        spent = findings.spent_upper_usd or 0.0
        lines += [
            "## Tope de gasto",
            "",
            f"- tope declarado: {findings.spend_cap_usd:.2f} USD · gastado, en el extremo alto: "
            f"{spent:.4f} USD · invocaciones que el tope impidió: {len(findings.refused)}",
            f"- {findings.unpriced_calls} llamada(s) respondieron sin tokens: se facturaron y no "
            "suman al gasto que ve el tope.",
            "- El exceso posible sobre el tope son las invocaciones que ya estaban en vuelo.",
            "",
        ]
        lines += render_cap_cut(run, findings, subjects)
    lines += [
        "## Modos",
        "",
        "| id | familia | rol | declarado | responde en | nota |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in findings.models:
        lines.append(
            f"| `{item.model}` | {item.family} | {item.role.value} | {item.declared.value} | "
            f"{item.mode.value if item.mode else '**ninguno**'} | {_cell(item.note)} |"
        )
    lines.append("")

    header = (
        f"| brazo | modo | válidos | schema | context | timeout | transport | intentos | "
        f"reintentos/veredicto | latencia media ({machine}) | prompt / caché / salida | "
        "USD medido |"
    )
    lines += [
        "## Por (modelo, rol)",
        "",
        "`context` es la mesa contestando como la otra, o citando una evidencia que nadie emitió: "
        "lo mide el modelo, no el proveedor. `válidos` es registros sin error sobre activaciones "
        "preguntadas.",
        "",
        header,
        "|" + " --- |" * 12,
    ]
    skipped_arms: list[str] = []
    for label in (a for a in meta.arms if not a.endswith("@ping")):
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
    lines += render_rejections(run)
    lines += render_families(subjects)
    lines += render_sources(run)
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


def render_desks_dry_run(
    subjects: DeskSubjects, source: TechnicalSource, activations: int, cap_usd: float | None
) -> str:
    """Cuántas llamadas haría el sondeo de mesas, sus precios y el tope. No llama a nadie.

    Los dólares no se estiman: no hay `max_tokens` ni los prompts de las mesas existen hasta que
    los técnicos contestan. Lo que acota el gasto es el tope duro de la corrida real.
    """
    pings = len(subjects.pings)
    technical = len(Dimension) * activations
    substitutes = (len(subjects.producers) - len(Dimension)) * activations
    bulls = len(subjects.bulls) * activations
    bear = activations
    decider = len(subjects.bulls) * activations
    attempts = TECHNICAL_ATTEMPTS
    upper = 3 * pings + attempts * (technical + substitutes + bulls + bear + decider)
    lines = [
        "Sondeo de mesas de Zen — conteo previo (no llama a nadie)",
        "",
        f"- activaciones: {activations} (`pick_evenly` sobre el manifiesto, las mismas de siempre)",
        f"- pings de modo: {pings} a {3 * pings} intentos ({pings} ids, hasta 3 modos cada uno)",
        f"- técnicos, una vez por activación: {technical} (≤ {attempts * technical} intentos) — "
        + ", ".join(
            f"{d.value} " + " → ".join(f"`{s.model}`" for s in group)
            for d, group in subjects.technicals.items()
        ),
        f"  · structure y volume según `{source.path}`; cada suplente solo se pregunta en la "
        f"activación donde el anterior no validó: ≤ {substitutes} más "
        f"(≤ {attempts * substitutes} intentos)",
        f"- bull: {len(subjects.bulls)} candidatos x {activations} = {bulls} "
        f"(≤ {attempts * bulls} intentos): " + ", ".join(f"`{s.model}`" for s in subjects.bulls),
        f"- bear `{subjects.bear.model}`: {bear} (≤ {attempts * bear} intentos)",
        f"- decisor `{subjects.decider.model}`: ≤ {decider} (≤ {attempts * decider} intentos), una "
        "por (activación, bull con alegato válido)",
        f"- total: ≤ {upper} intentos.",
        "",
        f"Precios de Zen al {DEFAULT_PRICING.as_of(Billing.PAYG)} (USD por millón de tokens):",
        "",
        "| modelo | entrada | caché | salida | escritura de caché |",
        "| --- | --- | --- | --- | --- |",
    ]
    models = dict.fromkeys([s.model for s in subjects.pings if s.model])
    for model in models:
        row = DEFAULT_PRICING.price_for(model, Billing.PAYG, utc_now())
        if row is None:
            lines.append(f"| `{model}` | sin precio | | | |")
            continue
        write = "—" if row.cache_write is None else f"{row.cache_write}"
        lines.append(
            f"| `{model}` | {row.input_per_mtok} | {row.cached_per_mtok} | "
            f"{row.output_per_mtok} | {write} |"
        )
    lines += [
        "",
        "Dólares: **no determinado antes de llamar**. El repo no fija `max_tokens` (la salida, "
        "razonamiento incluido, no tiene techo) y los prompts de las mesas dependen de veredictos "
        "que aún no existen; estimar tokens rompe la regla del repo.",
    ]
    if cap_usd is None:
        lines.append("Cota de coste: la corrida real exige `--max-usd` y no arranca sin él.")
    else:
        lines.append(
            f"Cota de coste: **{cap_usd:.2f} USD** (`--max-usd`). La corrida no abre una "
            "invocación nueva cuando lo medido, en el extremo alto, la alcanza. Exceso posible: "
            f"las invocaciones en vuelo (hasta {len(subjects.bulls) + 1} a la vez en las mesas, de "
            f"hasta {attempts} intentos cada una); las llamadas sin tokens no suman al tope."
        )
    return "\n".join([*lines, ""])


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
    parser.add_argument(
        "--desks",
        action="store_true",
        help="sondea las mesas (bull, bear, decisor) sobre una evidencia técnica común",
    )
    parser.add_argument(
        "--technicals-from",
        type=Path,
        default=None,
        help="directorio de un sondeo técnico previo: de él sale quién produce la evidencia",
    )
    parser.add_argument(
        "--max-usd",
        type=float,
        default=None,
        help="tope duro de gasto de la corrida de mesas (extremo alto de lo medido)",
    )
    args = parser.parse_args(argv)
    if args.verdicts < 1:
        parser.error("--verdicts debe ser al menos 1")
    if args.desks and args.technicals_from is None:
        parser.error("--desks necesita --technicals-from")
    if not args.desks and (args.technicals_from is not None or args.max_usd is not None):
        parser.error("--technicals-from y --max-usd son de --desks")
    if args.desks and not args.dry_run and args.max_usd is None:
        parser.error("--desks necesita --max-usd: sin tope no hay cota de coste")
    if args.max_usd is not None and args.max_usd <= 0:
        parser.error("--max-usd debe ser positivo")
    args.raw_argv = tuple(argv if argv is not None else sys.argv[1:])
    return args


async def _main_desks(args: argparse.Namespace, settings: Settings) -> str:
    """El sondeo de mesas: dry-run, o la corrida con su tope."""
    assert settings.openai is not None  # lo garantiza `refuse_unless_zen`
    source = technical_source(args.technicals_from, utc_now())
    subjects = desk_subjects(settings, source)
    if args.dry_run:
        return render_desks_dry_run(subjects, source, args.verdicts, args.max_usd)

    manifest = load_manifest(args.manifest)
    plan = plan_from_manifest(manifest, load_selection_histories(manifest, args.history_dir))
    started = utc_now()
    directory = args.out / started.strftime("%Y%m%dT%H%M%SZ")
    commit, dirty = git_state()
    open_run_directory(
        directory,
        build_meta(
            settings,
            args.manifest,
            args.raw_argv,
            started,
            commit,
            dirty,
            arms=all_desk_arms(subjects),
        ),
    )
    print(f"sondeo de mesas en {directory}", file=sys.stderr)
    findings = await run_desks_probe(
        settings,
        plan,
        directory,
        args.machine,
        build_backends(settings),
        source,
        SpendGuard(args.max_usd),
        utc_now,
        args.verdicts,
    )
    report = render_desks_report(read_run(directory), findings, subjects)
    (directory / "report.md").write_text(report, "utf-8")
    return report


async def _main(args: argparse.Namespace) -> str:
    settings = load_settings(DEFAULT_ENV_FILE)
    refuse_unless_zen(settings)
    assert settings.openai is not None  # lo garantiza `refuse_unless_zen`
    if args.desks:
        return await _main_desks(args, settings)
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
