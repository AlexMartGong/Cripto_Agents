"""Ablación: ¿los seis modelos deciden mejor que uno?

Corre la misma ventana histórica bajo varias formas del pipeline y compara. Todo
lo que cambia entre brazos es la forma del grafo y el reparto rol → modelo; el
histórico, el gate de activación, el gate de riesgo y la construcción de la orden
son idénticos. Si cada brazo tuviera su propio camino hasta el mercado, la tabla
mediría el arnés.

Qué mide cada brazo, y contra qué pregunta:

- `full` es el pipeline que opera. Es la referencia, no la respuesta.
- `no_debate` quita las dos mesas: ¿aporta algo el debate, o el decisor ya tenía
  todo lo que necesitaba en los veredictos técnicos?
- `bull_only` quita la contraparte: ¿aporta la mesa bajista, o solo ruido?
- `solo` es un modelo y una llamada. Es el brazo incómodo: si iguala a `full`, la
  arquitectura de seis modelos es cara y bonita, no buena.
- `local_technicals` y `local_bull` mueven roles concretos al respaldo local: ¿en
  qué roles se puede gastar menos sin perder decisión?

La coincidencia se mide sobre la acción, no sobre la confianza: dos decisores que
compran con 0.7 y 0.6 de confianza tomaron la misma decisión, y tratar esa
diferencia como desacuerdo inventaría señal donde no la hay.
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from pydantic import Field

from crypto_agents.activation import ActivationConfig
from crypto_agents.audit import (
    AuditError,
    PlanKind,
    RunMeta,
    arm_journal_path,
    file_sha256,
    ratio,
    run_chain,
    write_meta,
)
from crypto_agents.cache import JsonFileResponseCache, cache_key
from crypto_agents.consumption import PAGE_LABEL
from crypto_agents.context import AgentContext, utc_now
from crypto_agents.execution import PaperExecutor
from crypto_agents.graph import PipelineVariant, build_graph
from crypto_agents.indicators import DEFAULT_PRESET, IndicatorPreset
from crypto_agents.journal import JsonlJournal
from crypto_agents.llm import build_backends, prompt_digest
from crypto_agents.market import (
    MarketDataError,
    read_ohlcv_csv,
    timeframe_to_timedelta,
    to_dataframe,
)
from crypto_agents.metrics import (
    MIN_PAIRED_N,
    AbortKind,
    AttemptCounts,
    LatencyStats,
    PairedDifference,
    QuotaSplit,
    ReturnStats,
    RiskFlow,
    RunSummary,
    WeightedRate,
    WorstPair,
    attempt_counts,
    live_latency,
    paired_difference,
    quota_by_locality,
    return_stats,
    risk_flow,
    summarise,
    undecided_causes,
    validation_failure,
    wilson_interval,
    worst_pair,
)
from crypto_agents.nodes import (
    BASELINE_NODES,
    JOURNAL_NODE,
    PreparedEvaluation,
    prepare_evaluation,
)
from crypto_agents.outcomes import (
    OutcomeStats,
    ScoredRun,
    Scoring,
    resolved_returns,
    score_outcomes,
    score_run,
)
from crypto_agents.prompts import solo_prompt, technical_prompt
from crypto_agents.quota import QuotaLedger
from crypto_agents.replay import (
    HistoricalMarketClient,
    ReplayCacheMissError,
    ReplaySettings,
    replay_router,
    replay_run_id,
    replay_selection,
)
from crypto_agents.risk import AccountState
from crypto_agents.selection import (
    DEFAULT_SEED,
    PlannedEvaluation,
    SelectionError,
    SelectionManifest,
    load_manifest,
    load_selection_histories,
    verify_histories,
    window_digest,
)
from crypto_agents.selection import (
    HISTORY_DIR as SELECTION_HISTORY_DIR,
)
from crypto_agents.settings import DEFAULT_ENV_FILE, ConfigError, RoleConfig, load_settings
from crypto_agents.state import (
    Action,
    AgentRole,
    Backend,
    Billing,
    DebateBrief,
    Decision,
    Dimension,
    FrozenModel,
    LLMOutput,
    Proposal,
    TechnicalVerdict,
)
from crypto_agents.stops import COMMON_STOP_DESCRIPTION

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime
    from uuid import UUID

    from crypto_agents.cache import ResponseCache
    from crypto_agents.journal import EvaluationRecord, Journal
    from crypto_agents.llm import ChatBackend, ModelRouter
    from crypto_agents.quota import Clock
    from crypto_agents.settings import ModelChoice, Settings

__all__ = [
    "ARMS",
    "EXACT_NODES",
    "AblationArm",
    "ArmResult",
    "DryRunReport",
    "DryRunRow",
    "QuotaLine",
    "ReplayPlan",
    "agreement",
    "agreement_counts",
    "agreement_decided",
    "arm_settings",
    "build_arm_result",
    "decision_actions",
    "default_journal_dir",
    "dry_run",
    "git_state",
    "llm_nodes",
    "main",
    "open_run_directory",
    "plan_from_history",
    "plan_from_manifest",
    "render_dry_run",
    "render_report",
    "run_arm",
    "run_arms",
    "seed_from_previous",
]


class AblationArm(FrozenModel):
    """Una configuración a comparar."""

    name: str = Field(min_length=1)
    variant: PipelineVariant
    question: str = Field(min_length=10)
    """Qué pregunta responde este brazo. Si no responde ninguna, sobra."""

    local_roles: frozenset[AgentRole] = frozenset()
    """Roles que se sirven desde el respaldo local en vez del primario remoto."""


ARMS: tuple[AblationArm, ...] = (
    AblationArm(
        name="full",
        variant=PipelineVariant.FULL,
        question="referencia: tres técnicos, dos mesas y decisor",
    ),
    AblationArm(
        name="no_debate",
        variant=PipelineVariant.NO_DEBATE,
        question="¿aporta algo el debate sobre los veredictos técnicos?",
    ),
    AblationArm(
        name="bull_only",
        variant=PipelineVariant.BULL_ONLY,
        question="¿aporta la contraparte bajista, o basta una mesa?",
    ),
    AblationArm(
        name="solo",
        variant=PipelineVariant.SOLO,
        question="¿un modelo y una llamada igualan a seis modelos y seis llamadas?",
    ),
    AblationArm(
        name="local_technicals",
        variant=PipelineVariant.FULL,
        question="¿se pueden servir las tres lecturas técnicas en local sin perder decisión?",
        local_roles=frozenset({AgentRole.STRUCTURE, AgentRole.MOMENTUM, AgentRole.VOLUME}),
    ),
    AblationArm(
        name="local_bull",
        variant=PipelineVariant.FULL,
        question="¿aguanta una mesa en local contra una remota?",
        local_roles=frozenset({AgentRole.BULL}),
    ),
    AblationArm(
        name="always_buy",
        variant=PipelineVariant.ALWAYS_BUY,
        question="línea base sin modelo: ¿qué da comprar en cada activación del gate?",
    ),
    AblationArm(
        name="always_sell",
        variant=PipelineVariant.ALWAYS_SELL,
        question="línea base sin modelo: ¿qué da vender en cada activación del gate?",
    ),
    AblationArm(
        name="random_uniform",
        variant=PipelineVariant.RANDOM_UNIFORM,
        question="línea base sin modelo: ¿qué da un azar uniforme entre buy, sell y hold?",
    ),
    AblationArm(
        name="rule_trend",
        variant=PipelineVariant.RULE_TREND,
        question="línea base sin modelo: ¿qué da una regla de tendencia fija, pila de EMAs y ADX?",
    ),
)
"""Los seis brazos con modelos y, al final, las cuatro líneas base que no llaman a nadie.

Las líneas base van detrás para que los índices de los seis no cambien. Todas emiten un
`Proposal` por la misma cola que el resto, con el stop común de `stops.py`.
"""

NOTIONAL_ACCOUNT = AccountState(equity=10_000.0, day_start_equity=10_000.0)
"""Cuenta nocional del arnés.

Es una elección del experimento y no del sistema: todos los tamaños del pipeline
son fracciones, así que el nivel absoluto solo afecta a los vetos de drawdown, que
en una ablación sin equity viva nunca se disparan. Se declara para que nadie lea
los retornos como si vinieran de una cuenta real.
"""


class ArmResult(FrozenModel):
    """Lo que produjo un brazo sobre el histórico."""

    arm: AblationArm
    summary: RunSummary
    outcomes: OutcomeStats
    actions: tuple[Action | None, ...]
    """Acción por evaluación, en orden. `None` donde el brazo no llegó a decidir."""

    wall_clock_seconds: float = Field(ge=0.0)
    models: Mapping[AgentRole, str] = Field(default_factory=dict)

    quota: QuotaSplit = QuotaSplit(remote=0.0, local=0.0)
    latency: LatencyStats | None = None
    """Latencia de las llamadas vivas, o `None` si el brazo salió entero de la caché.

    Sustituye a una «latencia media» que era el reloj de pared del brazo dividido
    entre sus intentos vivos: con la caché tibia el denominador se encogía y la
    cifra dejaba de medir a ningún modelo.
    """

    validation: WeightedRate = WeightedRate(numerator=0, denominator=0)
    worst: WorstPair | None = None

    attempts: AttemptCounts = AttemptCounts(
        total=0, cache_hits=0, live=0, valid=0, invalid=0, cached_invalid=0
    )
    """Todas las filas `LLMCall` del brazo, con los aciertos de caché contados aparte."""

    risk: RiskFlow = RiskFlow(actionable=0, holds=0, orders=0, unexecuted=0)
    """Del decisor a la orden: accionables, vetos por regla, órdenes y aprobadas sin orden."""

    undecided: dict[tuple[str, AbortKind], int] = Field(default_factory=dict)
    """Evaluaciones sin decisión por nodo y causa, las del gate cerrado incluidas."""

    returns: ReturnStats | None = None
    """Retorno de las órdenes resueltas con su error estándar. `None` sin ninguna."""

    scored: dict[Scoring, ScoredRun] = Field(default_factory=dict)
    """La corrida puntuada de las tres formas: stop propio, stop común y cierre del horizonte."""


def decision_actions(records: Sequence[EvaluationRecord]) -> tuple[Action | None, ...]:
    """Acción de cada evaluación, en orden, con `None` donde no hubo decisión."""
    return tuple(
        record.proposed.action if record.proposed is not None else None for record in records
    )


def agreement(left: Sequence[Action | None], right: Sequence[Action | None]) -> float | None:
    """Fracción de evaluaciones en que dos brazos tomaron la misma acción.

    Se comparan por posición, que es la misma vela en ambos brazos porque los dos
    recorren el mismo histórico. Cuenta también los acuerdos en «no decidí»: que
    dos configuraciones se callen a la vez es una coincidencia real.
    """
    if len(left) != len(right) or not left:
        return None
    return sum(1 for a, b in zip(left, right, strict=True) if a == b) / len(left)


def agreement_counts(
    left: Sequence[Action | None], right: Sequence[Action | None]
) -> WeightedRate | None:
    """El recuento detrás de `agreement`: coincidencias sobre todas las evaluaciones.

    Es el mismo cociente con sus dos números, para que la tabla no publique una
    fracción sin denominador. Cuenta también el «no decidí» compartido, igual que
    `agreement`, y `tests/test_ablation.py` ata las dos.
    """
    if len(left) != len(right) or not left:
        return None
    return WeightedRate(
        numerator=sum(1 for a, b in zip(left, right, strict=True) if a == b),
        denominator=len(left),
    )


def agreement_decided(
    left: Sequence[Action | None], right: Sequence[Action | None]
) -> WeightedRate | None:
    """Coincidencia solo donde los dos brazos decidieron, con cuántas evaluaciones son.

    Contestar «qué hacen los decisores cuando ambos deciden» exige dejar fuera lo
    demás: dos brazos que se callan a la vez coinciden en `agreement` y no dicen nada
    sobre su criterio. El denominador puede ser cero, y entonces no hay tasa.
    """
    if len(left) != len(right):
        return None
    pairs = [(a, b) for a, b in zip(left, right, strict=True) if a is not None and b is not None]
    return WeightedRate(
        numerator=sum(1 for a, b in pairs if a == b),
        denominator=len(pairs),
    )


def arm_settings(settings: Settings, arm: AblationArm) -> Settings:
    """Configuración del brazo: los roles marcados se sirven desde su respaldo local.

    Se intercambia el primario por el respaldo en vez de tocar el router: así el
    brazo local recorre exactamente el mismo código que el remoto, y la diferencia
    medida es el modelo y no el camino.
    """
    if not arm.local_roles:
        return settings

    roles = dict(settings.roles)
    for role in sorted(arm.local_roles):
        config = roles[role]
        if config.fallback is None:
            raise ConfigError(
                f"el brazo {arm.name!r} pide {role.value} en local, "
                f"pero ese rol no declara respaldo"
            )
        roles[role] = RoleConfig(primary=config.fallback)
    return settings.model_copy(update={"roles": roles})


class ReplayPlan(NamedTuple):
    """Qué evaluaciones corre un brazo, sobre qué series y con qué ventana.

    Existe para que la corrida y el conteo previo recorran lo mismo por el mismo
    camino, vengan las evaluaciones de un tramo contiguo de un símbolo o de un
    manifiesto que salta entre siete. Con un `if` en cada uno, el presupuesto y la
    tabla podrían acabar midiendo recorridos distintos.

    `NamedTuple` y no `FrozenModel` porque `histories` son decenas de miles de
    filas: validarlas en cada construcción es tiempo pagado por nada, y ya vienen
    de `read_ohlcv_csv`, que sí valida.
    """

    entries: tuple[PlannedEvaluation, ...]
    histories: Mapping[str, Sequence[Sequence[float]]]
    timeframe: str
    candle_limit: int
    seed: int = DEFAULT_SEED
    """Semilla del plan: la del manifiesto, o la de la selección por defecto sin él.

    Solo la lee `random_uniform`, que la combina con el `run_id` de cada evaluación.
    """


def plan_from_history(
    rows: Sequence[Sequence[float]], config: ReplaySettings, candle_limit: int | None = None
) -> ReplayPlan:
    """Plan contiguo: cada vela cerrada del histórico a partir del warm-up.

    Es el recorrido de siempre, escrito como plan. Las entradas llevan su digest
    igual que las de un manifiesto, así que el modo por defecto obtiene gratis la
    misma comprobación de que el histórico no cambió bajo los pies.
    """
    limit = candle_limit if candle_limit is not None else config.candle_limit
    frame = to_dataframe(rows)
    entries: list[PlannedEvaluation] = []
    for cursor in range(config.warmup_bars - 1, len(frame) - 1):
        if config.max_evaluations is not None and len(entries) >= config.max_evaluations:
            break
        entries.append(
            PlannedEvaluation(
                symbol=config.symbol,
                timeframe=config.timeframe,
                index=cursor,
                at=frame.index[cursor].to_pydatetime(),
                candles_digest=window_digest(rows, cursor, limit),
            )
        )
    return ReplayPlan(tuple(entries), {config.symbol: rows}, config.timeframe, limit)


def plan_from_manifest(
    manifest: SelectionManifest, histories: Mapping[str, Sequence[Sequence[float]]]
) -> ReplayPlan:
    """Plan de un manifiesto ya verificado contra sus históricos."""
    verify_histories(manifest, histories)
    return ReplayPlan(
        manifest.entries,
        dict(histories),
        manifest.timeframe,
        manifest.candle_limit,
        manifest.seed,
    )


async def run_arm(
    arm: AblationArm,
    plan: ReplayPlan,
    settings: Settings,
    account: AccountState,
    cache: ResponseCache,
    clock: Clock,
    ledger: QuotaLedger,
    journal: Journal,
    fill_with: Mapping[Backend, ChatBackend] | None = None,
    horizon: int = 6,
    preset: IndicatorPreset = DEFAULT_PRESET,
    activation: ActivationConfig | None = None,
) -> ArmResult:
    """Corre un brazo completo sobre el plan y lo consolida.

    El preset es un parámetro y no una constante porque decide cuántas velas de
    warm-up se comen antes de la primera evaluación: con el de producción son 400,
    y sobre un histórico de 500 quedan menos de cien evaluaciones. Todos los brazos
    reciben el mismo, que es lo que los hace comparables.

    El contador de cuota **entra por parámetro y no se construye aquí**. Uno por
    brazo son seis contadores creyéndose cada uno dentro del presupuesto mientras
    el proveedor ve la suma: el mismo defecto que el runner evitaba entre símbolos
    con un contador único. Que sea obligatorio y sin valor por defecto es lo que
    impide que vuelva a colarse uno privado.

    **El journal también, y por lo mismo.** Tenía valor por defecto —uno en
    memoria— y el comando nunca pasaba otro: la primera corrida completa dejó una
    tabla agregada y ningún registro detrás, así que no se pudo preguntar por qué
    31 evaluaciones no decidieron. Sin valor por defecto, dónde queda escrito cada
    brazo lo decide quien llama, y olvidarlo es un error de tipos.
    """
    gate = activation if activation is not None else ActivationConfig(preset=preset)
    tuned = arm_settings(settings, arm)
    router = replay_router(tuned, ledger, cache, clock, fill_with)
    graph = build_graph(arm.variant)

    def build_context(
        symbol: str,
        run_id: UUID,
        router_: ModelRouter,
        market: HistoricalMarketClient,
        moment: datetime,
    ) -> AgentContext:
        return AgentContext(
            settings=tuned,
            router=router_,
            market=market,
            account=account,
            run_id=run_id,
            symbol=symbol,
            timeframe=plan.timeframe,
            executor=PaperExecutor(),
            journal=journal,
            candle_limit=plan.candle_limit,
            preset=preset,
            activation=gate,
            clock=lambda: moment,
            now=moment,
            seed=plan.seed,
        )

    started = time.perf_counter()
    records = await replay_selection(
        plan.entries, plan.histories, plan.candle_limit, router, build_context, graph
    )
    elapsed = time.perf_counter() - started

    return build_arm_result(
        arm,
        records,
        plan.histories,
        wall_clock_seconds=elapsed,
        horizon=horizon,
        models={role: tuned.role_config(role).primary for role in AgentRole},
    )


def build_arm_result(
    arm: AblationArm,
    records: Sequence[EvaluationRecord],
    histories: Mapping[str, Sequence[Sequence[float]]],
    wall_clock_seconds: float,
    horizon: int = 6,
    models: Mapping[AgentRole, ModelChoice] | None = None,
) -> ArmResult:
    """Consolida una corrida de un brazo."""
    return ArmResult(
        arm=arm,
        summary=summarise(records),
        outcomes=score_outcomes(records, histories, horizon),
        actions=decision_actions(records),
        wall_clock_seconds=wall_clock_seconds,
        models={role: choice.model for role, choice in (models or {}).items()},
        quota=quota_by_locality(records),
        latency=live_latency(records),
        validation=validation_failure(records),
        worst=worst_pair(records),
        attempts=attempt_counts(records),
        risk=risk_flow(records),
        undecided=undecided_causes(records),
        returns=return_stats(resolved_returns(records, histories, horizon)),
        scored={scoring: score_run(records, histories, horizon, scoring) for scoring in Scoring},
    )


# ─────────────────────────────────── Directorio de la corrida ─────────────────────────────────────
# Lo que queda en disco cuando el proceso termina, o cuando no termina. Un journal
# por brazo y un `meta.json` escrito antes de la primera llamada: la corrida que más
# hace falta poder leer después es la que se interrumpió.

RUNS_DIR = Path("var/ablation")
"""Raíz de los directorios de corrida. Bajo `var/`, que no se versiona."""


def default_journal_dir(started: datetime) -> Path:
    """Directorio de una corrida que empieza en ese instante: `var/ablation/<inicio UTC>/`.

    Uno nuevo por invocación, también al reanudar. El `run_id` de un replay es un
    UUID5 determinista, así que añadir una segunda pasada al mismo archivo dejaría
    dos líneas con el mismo id y ninguna forma de saber cuál es de cuál.
    """
    return RUNS_DIR / started.strftime("%Y%m%dT%H%M%SZ")


def git_state(cwd: Path | None = None) -> tuple[str | None, bool | None]:
    """Commit actual y si el árbol tiene cambios sin comprometer.

    `(None, None)` si git no responde: fuera de un repositorio, o sin git
    instalado. No se inventa un valor —la auditoría lo imprime como no
    determinado— porque un commit equivocado es peor que ninguno.

    El segundo valor importa tanto como el primero: con cambios sin comprometer,
    el commit nombra un código que no es el que corrió.
    """

    def ask(*arguments: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", *arguments],
                cwd=cwd,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    commit = ask("rev-parse", "HEAD")
    if not commit:
        return None, None
    status = ask("status", "--porcelain")
    return commit, (None if status is None else bool(status))


def open_run_directory(directory: Path, meta: RunMeta) -> Path:
    """Crea el directorio de la corrida y escribe sus metadatos. No reutiliza uno.

    Un directorio que ya tiene una corrida se rechaza: escribir encima mezclaría
    dos pasadas en los mismos archivos y los digests que la auditoría cita dejarían
    de identificar ninguna de las dos.
    """
    if directory.is_dir() and any(directory.iterdir()):
        raise ConfigError(
            f"{directory} ya contiene algo: cada corrida escribe en un directorio propio"
        )
    write_meta(directory, meta)
    return directory


def seed_from_previous(ledger: QuotaLedger, previous: Path, plan_sha256: str) -> int:
    """Siembra el contador con lo que gastó la corrida que se reanuda. Devuelve cuántas contó.

    Recorre la cadena entera y no solo el directorio indicado: dos interrupciones
    en menos de una ventana dejan gasto vigente en ambos, y `seed()` ya descarta lo
    que quedó fuera.

    Reanudar es continuar la misma comparación, así que el plan tiene que ser el
    mismo. Con otro manifiesto las evaluaciones son otras y la corrida no reanuda
    nada: se rechaza en vez de sembrar un presupuesto que no le corresponde.
    """
    try:
        chain = run_chain(previous)
    except AuditError as error:
        raise ConfigError(f"no se puede reanudar desde {previous}: {error}") from error

    for run in chain:
        if run.meta.plan_sha256 != plan_sha256:
            raise ConfigError(
                f"{run.path} corrió otro plan (sha-256 {run.meta.plan_sha256[:12]}…, "
                f"ahora {plan_sha256[:12]}…): eso no es una reanudación"
            )
    return ledger.seed(
        call for run in chain for arm in run.arms for record in arm.records for call in record.calls
    )


async def run_arms(
    arms: Sequence[AblationArm],
    plan: ReplayPlan,
    settings: Settings,
    cache: ResponseCache,
    clock: Clock,
    ledger: QuotaLedger,
    directory: Path,
    fill_with: Mapping[Backend, ChatBackend] | None = None,
    horizon: int = 6,
    preset: IndicatorPreset = DEFAULT_PRESET,
) -> list[ArmResult]:
    """Corre los brazos en orden, cada uno contra su journal en disco.

    Un archivo por brazo dentro de `directory`. La línea se escribe al terminar
    cada evaluación, no al terminar el brazo: lo que una interrupción deja atrás es
    todo lo evaluado hasta entonces, abortos incluidos.
    """
    results: list[ArmResult] = []
    for arm in arms:
        print(f"· {arm.name}: {arm.question}", file=sys.stderr)
        results.append(
            await run_arm(
                arm,
                plan,
                settings,
                NOTIONAL_ACCOUNT,
                cache,
                clock,
                ledger,
                JsonlJournal(arm_journal_path(directory, arm.name)),
                fill_with,
                horizon,
                preset,
            )
        )
    return results


# ──────────────────────────────────── Conteo previo (dry-run) ─────────────────────────────────────
# Cuánto costaría la tabla antes de emitir una sola llamada. A ~90 s por evaluación
# y seis brazos, enterarse a mitad de la corrida de que un rol no cabe en su
# ventana cuesta las horas ya gastadas; enterarse antes cuesta un minuto de CPU.


DETERMINISTIC_NODES = frozenset(
    {"__start__", "__end__", "prepare", "consolidate", "risk", "execute", JOURNAL_NODE}
    | set(BASELINE_NODES)
)
"""Nodos que no llaman a ningún modelo. Lo que sobre de aquí es coste.

Los decisores de las líneas base están aquí: es lo que hace que el conteo previo les
encuentre cero llamadas en vez de fallar por «nodo sin rol declarado».
"""

_LLM_NODES: Mapping[str, tuple[AgentRole, type[LLMOutput]]] = {
    "structure": (AgentRole.STRUCTURE, TechnicalVerdict),
    "momentum": (AgentRole.MOMENTUM, TechnicalVerdict),
    "volume": (AgentRole.VOLUME, TechnicalVerdict),
    "bull": (AgentRole.BULL, DebateBrief),
    "bear": (AgentRole.BEAR, DebateBrief),
    "decide": (AgentRole.DECIDER, Decision),
    "decide_single_desk": (AgentRole.DECIDER, Decision),
    "decide_without_debate": (AgentRole.DECIDER, Proposal),
    "decide_solo": (AgentRole.DECIDER, Proposal),
}
"""Qué rol y qué esquema gasta cada nodo que llama a un modelo.

El esquema hace falta porque entra en la clave de caché: el mismo decisor sobre el
mismo prompt no comparte entrada entre `Decision` y `Proposal`.
"""

EXACT_NODES = frozenset({"structure", "momentum", "volume", "decide_solo"})
"""Nodos cuyo prompt no depende de la salida de ningún modelo.

Son los que reciben indicadores directamente, así que su prompt —y por tanto su
digest y su clave de caché— se puede calcular sin llamar a nadie. Todo lo que
consume veredictos o alegatos solo admite cota superior: cuántas veces *podría*
correr, no con qué texto.
"""


def llm_nodes(variant: PipelineVariant) -> tuple[str, ...]:
    """Nodos de esa variante que gastan cuota, leídos del grafo ya compilado.

    Se leen del grafo y no de una tabla escrita a mano para que añadir un nodo con
    modelo a una variante no pueda pasar inadvertido al conteo: si no está
    declarado en `_LLM_NODES`, esto falla en vez de subestimar la factura.
    """
    names = set(build_graph(variant).get_graph().nodes) - DETERMINISTIC_NODES
    unknown = sorted(names - set(_LLM_NODES))
    if unknown:
        raise ConfigError(f"nodos sin rol declarado para el conteo: {', '.join(unknown)}")
    return tuple(sorted(names))


class DryRunRow(FrozenModel):
    """Lo que un nodo de un brazo costaría, antes de emitir una llamada."""

    arm: str = Field(min_length=1)
    node: str = Field(min_length=1)
    role: AgentRole
    backend: Backend
    model: str = Field(min_length=1)

    calls: int = Field(ge=0)
    """Intentos que el grafo hará si ninguna evaluación aborta antes de llegar."""

    exact: bool
    """Si `calls` es el número real o una cota superior."""

    cached: int | None = None
    """Intentos que se resolverían desde la caché. `None` cuando no se puede saber."""

    to_pay: int | None = None
    """Llamadas distintas que llegarían a un proveedor. `None` cuando no se puede saber."""


class QuotaLine(FrozenModel):
    """Gasto agregado de un par (rol, modelo) sobre todos los brazos.

    Es la cifra que decide si la ablación puede lanzarse, y no una curiosidad:
    los brazos comparten un solo `QuotaLedger`, así que este agregado es
    exactamente lo que ese contador verá y lo que el proveedor cobrará. Un desglose
    por brazo diría seis veces «cabe» sobre un presupuesto que solo existe una vez.
    """

    role: AgentRole
    model: str = Field(min_length=1)
    calls: int = Field(ge=0)
    quota: float = Field(ge=0.0)
    per_window: int = Field(gt=0)

    page_estimate: int | None = Field(default=None, gt=0)
    """Peticiones por 5 h que la página de la suscripción publica para ese modelo.

    `None` si no la publica, o si la corrida no se paga con la suscripción. Es una estimación
    de la página y no una medida: mientras no haya tokens medidos es lo único con lo que
    expresar el consumo del pool antes de gastarlo.
    """

    @property
    def fits(self) -> bool:
        """Si la ablación entera cabe en una ventana de ese par."""
        return self.quota <= self.per_window

    @property
    def window_fraction(self) -> float | None:
        """Qué parte de una ventana de 5 h del pool son esas llamadas, según la página.

        Una llamada vale `1 / estimado` de la ventana porque el estimado son las peticiones
        que caben si ese modelo fuera el único que se usara.
        """
        return None if self.page_estimate is None else self.calls / self.page_estimate


class DryRunReport(FrozenModel):
    """Conteo completo: qué se va a gastar y qué ya está pagado."""

    evaluations: int = Field(ge=0)
    activations: int = Field(ge=0)
    prepare_failures: int = Field(ge=0)
    """Velas donde la capa determinista falló, así que el gate nunca llegó a opinar."""

    billing: Billing = Billing.GO
    """Forma de pago de la corrida: decide si hay pool con el que expresar el consumo."""

    rows: tuple[DryRunRow, ...] = ()
    quota: tuple[QuotaLine, ...] = ()
    prompts: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    """Digest de cada prompt que se puede escribir sin llamar a nadie, por nodo.

    Es lo que el conteo ya calculó para preguntarle a la caché. Se publica porque
    es la única forma de comprobar que dos conteos sobre el mismo plan van a pedir
    exactamente lo mismo: si un digest cambia, la caché deja de servir y la
    corrida vuelve a pagar sin que la tabla de presupuesto lo dijera.
    """

    @property
    def calls(self) -> int:
        """Intentos totales, sumando brazos."""
        return sum(row.calls for row in self.rows)

    @property
    def known_to_pay(self) -> int:
        """Llamadas que se sabe con certeza que llegarán a un proveedor."""
        return sum(row.to_pay or 0 for row in self.rows)


async def dry_run(
    arms: Sequence[AblationArm],
    plan: ReplayPlan,
    settings: Settings,
    cache: ResponseCache,
    clock: Clock = utc_now,
    preset: IndicatorPreset = DEFAULT_PRESET,
    activation: ActivationConfig | None = None,
) -> DryRunReport:
    """Cuenta llamadas por rol y por brazo sin emitir ninguna.

    Recorre el mismo plan que la corrida y ejecuta solo la capa determinista, que
    es la misma función que corre el nodo `prepare`. De ahí salen dos cosas: en
    cuántas velas abre el gate —lo que multiplica todo lo demás— y el prompt exacto
    de los nodos que leen indicadores, cuyo digest permite preguntarle a la caché
    si esa llamada ya está pagada.

    El router que se le da al contexto es el del replay sin `fill_with`, es decir
    `CacheOnlyBackend` en todas las ranuras: aunque alguien añada mañana una llamada
    aquí dentro, no hay proveedor al que pueda llegar.

    Los brazos se procesan en orden y lo que uno dejaría en caché cuenta como
    acierto para los siguientes, que es como se va a comportar la corrida real: los
    seis comparten una sola caché y la clave lleva el modelo, no el brazo.
    """
    gate = activation if activation is not None else ActivationConfig(preset=preset)
    ledger = QuotaLedger(settings.quota_window, clock)
    router = replay_router(settings, ledger, cache, clock)

    digests: dict[str, list[str]] = {node: [] for node in EXACT_NODES}
    evaluations = 0
    activations = 0
    failures = 0

    for entry in plan.entries:
        evaluations += 1
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
            activation=gate,
            clock=lambda moment=moment: moment,  # type: ignore[misc]
            now=moment,
        )
        try:
            prepared = await prepare_evaluation(context)
        except Exception:  # la misma vela abortaría en `prepare` durante la corrida
            failures += 1
            continue

        if not prepared.activation.should_run:
            continue
        activations += 1
        for node, prompt in _exact_prompts(prepared).items():
            digests[node].append(prompt_digest(prompt))

    return _build_dry_run(arms, settings, cache, digests, evaluations, activations, failures)


def _exact_prompts(prepared: PreparedEvaluation) -> dict[str, str]:
    """Prompts que se pueden escribir sin haber llamado a ningún modelo."""
    triggers = prepared.activation.triggers
    prompts = {
        dimension.value: technical_prompt(
            dimension, prepared.snapshot, prepared.indicators, triggers
        )
        for dimension in Dimension
    }
    prompts["decide_solo"] = solo_prompt(prepared.snapshot, prepared.indicators, triggers)
    return prompts


def _build_dry_run(
    arms: Sequence[AblationArm],
    settings: Settings,
    cache: ResponseCache,
    digests: Mapping[str, Sequence[str]],
    evaluations: int,
    activations: int,
    failures: int,
) -> DryRunReport:
    """Convierte los digests contados en filas por brazo y en el agregado de cuota."""
    filled: set[str] = set()
    rows: list[DryRunRow] = []
    spend: dict[tuple[AgentRole, str], list[float]] = {}

    for arm in arms:
        tuned = arm_settings(settings, arm)
        for node in llm_nodes(arm.variant):
            role, schema = _LLM_NODES[node]
            choice = tuned.role_config(role).primary
            cached: int | None = None
            to_pay: int | None = None

            if node in EXACT_NODES:
                keys = [
                    cache_key(
                        choice.backend, choice.model, digest, schema, choice.structured_output
                    )
                    for digest in digests[node]
                ]
                known = {key for key in keys if key in filled or cache.get(key) is not None}
                cached = sum(1 for key in keys if key in known)
                to_pay = len(set(keys) - known)
                filled.update(keys)

            rows.append(
                DryRunRow(
                    arm=arm.name,
                    node=node,
                    role=role,
                    backend=choice.backend,
                    model=choice.model,
                    calls=activations,
                    exact=node in EXACT_NODES,
                    cached=cached,
                    to_pay=to_pay,
                )
            )
            spend.setdefault((role, choice.model), []).append(activations * choice.quota_weight)

    quota = tuple(
        QuotaLine(
            role=role,
            model=model,
            calls=len(weights) * activations,
            quota=sum(weights),
            per_window=_per_window(settings, role, model),
            page_estimate=(
                settings.pricing.page_estimates.get(model)
                if settings.billing is Billing.GO
                else None
            ),
        )
        for (role, model), weights in sorted(spend.items(), key=lambda item: item[0][0].value)
    )
    return DryRunReport(
        evaluations=evaluations,
        activations=activations,
        prepare_failures=failures,
        billing=settings.billing,
        rows=tuple(rows),
        quota=quota,
        prompts={node: tuple(items) for node, items in sorted(digests.items())},
    )


def _per_window(settings: Settings, role: AgentRole, model: str) -> int:
    """Presupuesto declarado del par. El brazo local usa el respaldo, no el primario."""
    for choice in settings.role_choices(role):
        if choice.model == model:
            return choice.quota_per_window
    raise ConfigError(f"{role.value} no declara el modelo {model}")


def render_dry_run(report: DryRunReport) -> str:
    """Tabla del conteo previo, en Markdown.

    Separa explícitamente lo exacto de la cota superior. Presentar las dos como un
    número solo invitaría a leer la suma como la factura, y la factura de los nodos
    que dependen de un veredicto no se puede conocer sin producir ese veredicto.
    """
    lines = [
        f"Evaluaciones: {report.evaluations} · gate abierto: {report.activations} "
        f"· preparación fallida: {report.prepare_failures}",
        "",
        "| brazo | nodo | rol | modelo | backend | llamadas | en caché | a pagar |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report.rows:
        calls = f"{row.calls}" if row.exact else f"≤ {row.calls}"
        lines.append(
            f"| `{row.arm}` | {row.node} | {row.role.value} | `{row.model}` | "
            f"{row.backend.value} | {calls} | {_count(row.cached)} | "
            f"{_count(row.to_pay)} |"
        )

    lines.extend(
        [
            "",
            "Cuota agregada sobre los brazos pedidos. Es la única que existe: los brazos",
            "comparten un `QuotaLedger`, así que esto es lo que ese contador verá.",
            "",
            "| rol | modelo | llamadas | cuota | por ventana | cabe |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    lines.extend(
        f"| {line.role.value} | `{line.model}` | ≤ {line.calls} | {line.quota:.1f} | "
        f"{line.per_window} | {'sí' if line.fits else '**NO**'} |"
        for line in report.quota
    )
    lines.extend(_pool_estimate(report))
    return "\n".join(lines) + "\n"


def _pool_estimate(report: DryRunReport) -> list[str]:
    """Cuánto del pool de la suscripción gastaría la corrida, según lo que dice la página.

    Es lo único que se puede decir antes de medir tokens, y se rotula como lo que es. Cada par
    vale `llamadas / estimado de la página` de una ventana de 5 h; se suman porque el pool es
    uno y lo comparten todos los modelos. Con un par sin estimado el total no se determina: un
    total que omite un modelo en silencio es una cifra menor de lo que costará.
    """
    lines = ["", f"Consumo del pool — {PAGE_LABEL}.", ""]
    if report.billing is not Billing.GO:
        lines += [
            "Con pago por uso no hay pool (sin pool). Los dólares dependen de los tokens de "
            "cada llamada y todavía no hay tokens medidos: no determinado.",
        ]
        return lines
    lines += [
        "Cada llamada vale 1 / (peticiones por 5 h que estima la página para ese modelo) de una",
        "ventana de 5 h. Las fracciones se suman: el pool es uno y lo comparten todos los modelos.",
        "",
        "| rol | modelo | llamadas | estimado por 5 h | fracción de una ventana de 5 h |",
        "| --- | --- | --- | --- | --- |",
    ]
    known = 0.0
    missing = 0
    for line in report.quota:
        fraction = line.window_fraction
        if fraction is None:
            missing += 1
            lines.append(
                f"| {line.role.value} | `{line.model}` | ≤ {line.calls} | — | sin estimado |"
            )
            continue
        known += fraction
        lines.append(
            f"| {line.role.value} | `{line.model}` | ≤ {line.calls} | {line.page_estimate} | "
            f"≤ {fraction:.2%} |"
        )
    total = (
        f"≤ {known:.2%}"
        if missing == 0
        else f"no determinado: {missing} par(es) sin estimado de la página"
    )
    lines.append(f"| **total** | | | | {total} |")
    return lines


def _count(value: int | None) -> str:
    """Un conteo, o un guion cuando el prompt no se puede calcular por adelantado."""
    return "—" if value is None else str(value)


# ────────────────────────────────────────────── Reporte ───────────────────────────────────────────


def _num(value: float | None, digits: int = 2) -> str:
    """Número, o un guion si no hay dato."""
    return "—" if value is None else f"{value:.{digits}f}"


def _no_data(reason: str) -> str:
    """Celda de lo que no se pudo calcular, con su causa. Igual que en la auditoría."""
    return f"no determinado: {reason}"


def _fraction(numerator: int, denominator: int, empty: str) -> str:
    """`k/n (p%)`, o `no determinado` con su causa si no hay con qué dividir."""
    if denominator == 0:
        return _no_data(empty)
    return ratio(numerator, denominator)


def _agreement_cell(rate: WeightedRate | None) -> str:
    """Coincidencia con su recuento. `None` es que los dos brazos no recorrieron lo mismo."""
    if rate is None:
        return _no_data("brazos de distinta longitud o sin evaluaciones")
    return _fraction(rate.numerator, rate.denominator, "ninguna evaluación con ambos decidiendo")


def _wins_cell(outcomes: OutcomeStats) -> str:
    """Aciertos sobre resueltas, con el intervalo de Wilson al 95%."""
    interval = wilson_interval(outcomes.wins, outcomes.resolved)
    if interval is None:
        return _no_data("sin órdenes resueltas")
    return (
        f"{ratio(outcomes.wins, outcomes.resolved)} IC95 [{interval.low:.0%}, {interval.high:.0%}]"
    )


def _return_cell(stats: ReturnStats | None) -> str:
    """Retorno medio con decimales, su error estándar y la muestra que lo sostiene.

    Dos decimales sobre el porcentaje: un retorno de 0.4% redondeado a entero sale
    «0%», y nadie sabría si la media es positiva.
    """
    if stats is None:
        return _no_data("sin órdenes resueltas")
    error = (
        "no determinado (EE: una sola orden)"
        if stats.stderr is None
        else f"{stats.stderr:.2%} (EE)"
    )
    return f"{stats.mean:+.2%} ± {error}, n={stats.n}"


_SCORING_LABELS = {
    Scoring.OWN_STOP: "stop propio",
    Scoring.COMMON_STOP: f"stop común ({COMMON_STOP_DESCRIPTION})",
    Scoring.HORIZON_CLOSE: "cierre del horizonte",
}
"""Cómo se nombra cada puntuación en el reporte. El stop común cita su múltiplo desde `stops`."""


def _paired_cell(
    run: ScoredRun | None, reference: ScoredRun | None, reference_name: str, is_reference: bool
) -> str:
    """Diferencia media por evaluación contra el brazo de referencia, con su IC95% y n."""
    if is_reference:
        return "— (referencia)"
    if reference is None:
        return _no_data(f"no hay brazo `{reference_name}`")
    if run is None:
        return _no_data("sin puntuación")
    if len(run.per_evaluation) != len(reference.per_evaluation):
        return _no_data("brazos de distinta longitud")
    difference = paired_difference(run.per_evaluation, reference.per_evaluation)
    if difference is None:
        return _no_data(f"n={len(run.per_evaluation)} < {MIN_PAIRED_N}")
    return _paired_text(difference)


def _paired_text(difference: PairedDifference) -> str:
    """`Δ [IC95%], n`, con signo y dos decimales de porcentaje."""
    return (
        f"{difference.mean:+.2%} [{difference.low:+.2%}, {difference.high:+.2%}], n={difference.n}"
    )


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """Tabla Markdown: cabecera, separador y filas."""
    return [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def render_report(
    results: Sequence[ArmResult], reference: str = "full", paired: str = "solo"
) -> str:
    """Tablas comparativas en Markdown, una por pregunta.

    Deliberadamente sin veredicto automático: qué significa que `solo` iguale a
    `full` es una conclusión que hay que escribir a mano, y una plantilla que la
    generase sola invitaría a no leerla.

    Las columnas de coste son las de la auditoría, calculadas por las mismas
    funciones: cuota remota y local por separado, latencia solo de llamadas vivas,
    y el fallo de validación como inválidas sobre respondidas con el peor par en
    su propia columna. La tabla anterior publicaba ese máximo como si fuera la
    tasa del brazo, y un reloj de pared dividido entre intentos como si fuera una
    latencia.

    Ninguna fracción sale sin sus dos números: cada etapa del embudo va sobre la
    anterior, los aciertos y el retorno llevan su incertidumbre y su n, y lo que no
    se puede calcular lo dice en vez de dar cero.

    La tabla «Retorno según el stop» puntúa cada brazo de tres formas —con el stop que
    declaró, con el común y al cierre del horizonte— para separar la dirección del
    nivel de invalidación, y empareja cada brazo contra `paired` por evaluación. Son
    unas 27 diferencias al 95% sin corrección por comparaciones múltiples: alguna
    excluirá el cero por azar, y el reporte lo dice.
    """
    if not results:
        return "# Ablación\n\nSin corridas.\n"

    baseline = next((item for item in results if item.arm.name == reference), results[0])
    name = baseline.arm.name
    paired_result = next((item for item in results if item.arm.name == paired), None)

    calls: list[list[str]] = []
    funnel: list[list[str]] = []
    lost: list[list[str]] = []
    scored: list[list[str]] = []
    stops: list[list[str]] = []
    for result in results:
        arm = f"`{result.arm.name}`"
        summary, attempts, risk, outcomes = (
            result.summary,
            result.attempts,
            result.risk,
            result.outcomes,
        )

        no_live = _no_data("sin llamadas vivas")
        latency = result.latency
        worst = (
            "—"
            if result.worst is None
            else f"{result.worst.role.value}/{result.worst.backend.value} "
            + ratio(result.worst.stats.invalid, result.worst.stats.answered)
        )
        calls.append(
            [
                arm,
                str(attempts.total),
                _fraction(attempts.cache_hits, attempts.total, "sin llamadas"),
                _fraction(attempts.live, attempts.total, "sin llamadas"),
                _num(result.quota.remote, 1),
                _num(result.quota.local, 1),
                no_live if latency is None else f"{latency.n}, {latency.median_ms:.0f} ms",
                no_live if latency is None else f"{latency.mean_ms:.0f} ms",
                no_live if latency is None else f"{latency.p95_ms:.0f} ms",
                ratio(result.validation.numerator, result.validation.denominator),
                worst,
            ]
        )

        mix = ", ".join(
            f"{action.value} {count}" for action, count in sorted(summary.actions.items())
        )
        vetoes = ", ".join(f"{rule} {count}" for rule, count in risk.vetoes.items())
        funnel.append(
            [
                arm,
                str(summary.evaluations),
                _fraction(summary.decided, summary.evaluations, "sin evaluaciones"),
                mix or "—",
                _fraction(risk.actionable, summary.decided, "sin decisiones"),
                _fraction(sum(risk.vetoes.values()), risk.actionable, "sin accionables"),
                vetoes or "—",
                _fraction(risk.orders, risk.actionable, "sin accionables"),
                _fraction(risk.unexecuted, risk.actionable, "sin accionables"),
                _agreement_cell(agreement_counts(baseline.actions, result.actions)),
                _agreement_cell(agreement_decided(baseline.actions, result.actions)),
            ]
        )

        if not result.undecided:
            lost.append([arm, "—", "—", _fraction(0, summary.evaluations, "sin evaluaciones")])
        lost.extend(
            [arm, node, kind.value, _fraction(count, summary.evaluations, "sin evaluaciones")]
            for (node, kind), count in result.undecided.items()
        )

        for scoring in Scoring:
            run = result.scored.get(scoring)
            reference_run = None if paired_result is None else paired_result.scored.get(scoring)
            if run is None:
                stops.append([arm, _SCORING_LABELS[scoring], *[_no_data("sin puntuación")] * 4])
                continue
            position_cell = _fraction(run.resolved, run.positions, "sin posiciones")
            if run.unscorable:
                position_cell += f"; {run.unscorable} sin indicadores"
            stops.append(
                [
                    arm,
                    _SCORING_LABELS[scoring],
                    position_cell,
                    _return_cell(return_stats(run.per_position)),
                    _return_cell(return_stats(run.per_evaluation)),
                    _paired_cell(run, reference_run, paired, result.arm.name == paired),
                ]
            )

        scored.append(
            [
                arm,
                _fraction(outcomes.resolved, outcomes.orders, "sin órdenes"),
                _fraction(outcomes.invalidated, outcomes.resolved, "sin órdenes resueltas"),
                _wins_cell(outcomes),
                _return_cell(result.returns),
            ]
        )

    lines = [
        "### Llamadas y coste",
        "",
        *_table(
            [
                "brazo",
                "llamadas",
                "aciertos de caché",
                "vivas",
                "cuota remota",
                "cuota local",
                "latencia viva (n, mediana)",
                "latencia media",
                "latencia p95",
                "fallo validación (inválidas/respondidas)",
                "peor par",
            ],
            calls,
        ),
        "",
        "### Del decisor a la orden",
        "",
        *_table(
            [
                "brazo",
                "evals",
                "decididas",
                "acciones",
                "accionables",
                "vetadas",
                "vetos por regla",
                "órdenes",
                "aprobadas sin orden",
                f"coincidencia con `{name}` (todas las evals)",
                f"coincidencia con `{name}` (ambos decidieron)",
            ],
            funnel,
        ),
        "",
        "### Evaluaciones sin decisión",
        "",
        *_table(["brazo", "nodo", "causa", "evaluaciones"], lost),
        "",
        "### Resultado de las órdenes",
        "",
        *_table(
            [
                "brazo",
                "órdenes resueltas",
                "invalidadas",
                "aciertos (IC95% Wilson)",
                "retorno medio ± EE (n)",
            ],
            scored,
        ),
        "",
        "### Retorno según el stop",
        "",
        "Retorno bruto por unidad nocional, sin comisiones ni tamaño. Por posición promedia las "
        "resueltas; por evaluación, todas, con 0 donde no hay posición. «Stop propio» puntúa "
        "órdenes; «stop común» y «cierre del horizonte» puntúan propuestas accionables. La "
        "diferencia es contra el brazo de referencia, evaluación por evaluación, con un "
        "intervalo normal al 95% desde n = 30. Hay unas 27 diferencias sin corrección por "
        "comparaciones múltiples: alguna excluirá el cero por azar.",
        "",
        *_table(
            [
                "brazo",
                "puntuación",
                "posiciones resueltas",
                "retorno por posición (media ± EE, n)",
                "retorno por evaluación (media ± EE, n)",
                f"Δ por evaluación vs `{paired}` (IC95%)",
            ],
            stops,
        ),
        "",
        "### Qué pregunta responde cada brazo",
        "",
        *_table(
            ["brazo", "qué pregunta responde"],
            [[f"`{r.arm.name}`", r.arm.question] for r in results],
        ),
    ]
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────── Comando ──────────────────────────────────────────────


DEFAULT_HISTORY = Path("tests/data/btcusdt_4h.csv")
DEFAULT_REPORT = Path("docs/ablation.md")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Argumentos del comando."""
    parser = argparse.ArgumentParser(
        prog="python -m crypto_agents.ablation",
        description="Corre la misma ventana histórica bajo varias formas del pipeline.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "selección versionada de activaciones repartidas por símbolo y por tramo; "
            "sin esto, se recorre --history de principio a fin"
        ),
    )
    parser.add_argument("--history-dir", type=Path, default=SELECTION_HISTORY_DIR)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--warmup", type=int, default=DEFAULT_PRESET.min_bars)
    parser.add_argument("--evaluations", type=int, default=25)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--cache", type=Path, default=Path("var/ablation-cache"))
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--arms",
        default=",".join(arm.name for arm in ARMS),
        help="brazos a correr, separados por comas",
    )
    parser.add_argument(
        "--fill",
        action="store_true",
        help="permite llamar a los proveedores para llenar la caché; sin esto, solo caché",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="cuenta llamadas por rol y por brazo sin emitir ninguna, y termina",
    )
    parser.add_argument(
        "--journal-dir",
        type=Path,
        default=None,
        help=(
            "directorio de la corrida: un journal por brazo y meta.json; "
            f"por defecto {RUNS_DIR}/<inicio UTC>/"
        ),
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help=(
            "directorio de la corrida interrumpida: siembra el contador de cuota con lo "
            "que ya gastó; la reanudación escribe en un directorio nuevo"
        ),
    )
    args = parser.parse_args(argv)
    args.raw_argv = tuple(argv) if argv is not None else tuple(sys.argv[1:])
    return args


def _plan_source(args: argparse.Namespace) -> tuple[PlanKind, Path]:
    """El archivo del que sale el plan: el manifiesto, o el histórico si no lo hay."""
    if args.manifest is None:
        return PlanKind.HISTORY, args.history
    return PlanKind.MANIFEST, args.manifest


def _plan_of(args: argparse.Namespace) -> ReplayPlan:
    """El plan pedido: el manifiesto si lo hay, y si no el histórico contiguo.

    Con manifiesto, `plan_from_manifest` verifica antes de devolver nada: un
    histórico revisado por el exchange se detecta aquí y no en la tercera hora de
    corrida, cuando la tabla ya no compararía con la anterior.
    """
    if args.manifest is None:
        return plan_from_history(
            read_ohlcv_csv(args.history),
            ReplaySettings(
                symbol=args.symbol,
                timeframe=args.timeframe,
                warmup_bars=args.warmup,
                max_evaluations=args.evaluations,
            ),
        )
    manifest = load_manifest(args.manifest)
    return plan_from_manifest(manifest, load_selection_histories(manifest, args.history_dir))


async def _run(args: argparse.Namespace) -> str:
    """Corre los brazos pedidos y devuelve el reporte."""
    settings = load_settings(DEFAULT_ENV_FILE)
    plan = _plan_of(args)
    cache = JsonFileResponseCache(args.cache)
    clock = utc_now

    wanted = [name.strip() for name in args.arms.split(",") if name.strip()]
    by_name = {arm.name: arm for arm in ARMS}
    unknown = [name for name in wanted if name not in by_name]
    if unknown:
        raise ConfigError(f"brazos desconocidos: {', '.join(unknown)}")

    if args.dry_run:
        # `build_backends` no se llama aquí: el conteo no puede tener a mano un
        # proveedor al que llamar, ni siquiera sin usarlo.
        report = await dry_run([by_name[name] for name in wanted], plan, settings, cache, clock)
        return render_dry_run(report)

    started = clock()
    kind, source = _plan_source(args)
    plan_sha256 = file_sha256(source)
    directory = args.journal_dir if args.journal_dir is not None else default_journal_dir(started)

    fill_with = build_backends(settings) if args.fill else None
    # Un solo contador para los seis brazos: el proveedor ve la suma, no seis
    # presupuestos independientes. El brazo que agote la ventana aborta y lo
    # registra, en vez de creerse dentro.
    ledger = QuotaLedger(settings.quota_window, clock)
    if args.resume_from is not None:
        # El contador es de este proceso y la ventana del proveedor no: sin
        # sembrar, la reanudación se cree con el presupuesto entero.
        seeded = seed_from_previous(ledger, args.resume_from, plan_sha256)
        print(
            f"reanudación: {seeded} llamadas remotas de {args.resume_from} siguen en la ventana",
            file=sys.stderr,
        )

    commit, dirty = git_state()
    open_run_directory(
        directory,
        RunMeta(
            plan_kind=kind,
            plan_path=str(source),
            plan_sha256=plan_sha256,
            argv=args.raw_argv,
            fill=args.fill,
            arms=tuple(wanted),
            started_at=started,
            git_commit=commit,
            git_dirty=dirty,
            resumed_from=None if args.resume_from is None else str(args.resume_from.resolve()),
            billing=settings.billing,
        ),
    )
    print(f"journals en {directory}", file=sys.stderr)

    results = await run_arms(
        [by_name[name] for name in wanted],
        plan,
        settings,
        cache,
        clock,
        ledger,
        directory,
        fill_with,
        args.horizon,
    )
    return render_report(results)


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada: `python -m crypto_agents.ablation`."""
    args = _parse_args(argv)
    try:
        report = asyncio.run(_run(args))
    except (ConfigError, MarketDataError, ReplayCacheMissError, SelectionError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(report)
    if args.dry_run:
        # El conteo no pisa `docs/ablation.md`: no es la tabla, es su presupuesto.
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"tabla escrita en {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
