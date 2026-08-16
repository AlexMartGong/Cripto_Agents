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
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import Field

from crypto_agents.activation import ActivationConfig
from crypto_agents.cache import JsonFileResponseCache, cache_key
from crypto_agents.context import AgentContext, utc_now
from crypto_agents.execution import PaperExecutor
from crypto_agents.graph import PipelineVariant, build_graph
from crypto_agents.indicators import DEFAULT_PRESET, IndicatorPreset
from crypto_agents.journal import InMemoryJournal
from crypto_agents.llm import build_backends, prompt_digest
from crypto_agents.market import (
    MarketDataError,
    read_ohlcv_csv,
    timeframe_to_timedelta,
    to_dataframe,
)
from crypto_agents.metrics import RunSummary, summarise
from crypto_agents.nodes import JOURNAL_NODE, PreparedEvaluation, prepare_evaluation
from crypto_agents.outcomes import OutcomeStats, score_outcomes
from crypto_agents.prompts import solo_prompt, technical_prompt
from crypto_agents.quota import QuotaLedger
from crypto_agents.replay import (
    HistoricalMarketClient,
    ReplayCacheMissError,
    ReplaySettings,
    replay,
    replay_router,
    replay_run_id,
)
from crypto_agents.risk import AccountState
from crypto_agents.settings import DEFAULT_ENV_FILE, ConfigError, RoleConfig, load_settings
from crypto_agents.state import (
    Action,
    AgentRole,
    Backend,
    DebateBrief,
    Decision,
    Dimension,
    FrozenModel,
    LLMOutput,
    Proposal,
    TechnicalVerdict,
)

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
    "agreement",
    "arm_settings",
    "build_arm_result",
    "decision_actions",
    "dry_run",
    "llm_nodes",
    "main",
    "render_dry_run",
    "render_report",
    "run_arm",
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
)
"""Los seis brazos, en el orden en que se leen en el reporte."""

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

    @property
    def mean_latency_ms(self) -> float | None:
        """Latencia media por llamada que llegó a un proveedor."""
        stats = self.summary.calls
        attempts = sum(item.attempts for item in stats.values())
        if attempts == 0:
            return None
        return self.wall_clock_seconds * 1000.0 / attempts


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


async def run_arm(
    arm: AblationArm,
    rows: Sequence[Sequence[float]],
    settings: Settings,
    config: ReplaySettings,
    account: AccountState,
    cache: ResponseCache,
    clock: Clock,
    ledger: QuotaLedger,
    fill_with: Mapping[Backend, ChatBackend] | None = None,
    horizon: int = 6,
    preset: IndicatorPreset = DEFAULT_PRESET,
    activation: ActivationConfig | None = None,
    journal: Journal | None = None,
) -> ArmResult:
    """Corre un brazo completo sobre el histórico y lo consolida.

    El preset es un parámetro y no una constante porque decide cuántas velas de
    warm-up se comen antes de la primera evaluación: con el de producción son 400,
    y sobre un histórico de 500 quedan menos de cien evaluaciones. Todos los brazos
    reciben el mismo, que es lo que los hace comparables.

    El contador de cuota **entra por parámetro y no se construye aquí**. Uno por
    brazo son seis contadores creyéndose cada uno dentro del presupuesto mientras
    el proveedor ve la suma: el mismo defecto que el runner evitaba entre símbolos
    con un contador único. Que sea obligatorio y sin valor por defecto es lo que
    impide que vuelva a colarse uno privado.
    """
    gate = activation if activation is not None else ActivationConfig(preset=preset)
    tuned = arm_settings(settings, arm)
    router = replay_router(tuned, ledger, cache, clock, fill_with)
    graph = build_graph(arm.variant)
    destination = journal if journal is not None else InMemoryJournal()

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
            timeframe=config.timeframe,
            executor=PaperExecutor(),
            journal=destination,
            candle_limit=config.candle_limit,
            preset=preset,
            activation=gate,
            clock=lambda: moment,
            now=moment,
        )

    started = time.perf_counter()
    records = await replay(rows, config, router, build_context, graph)
    elapsed = time.perf_counter() - started

    return build_arm_result(
        arm,
        records,
        rows,
        wall_clock_seconds=elapsed,
        horizon=horizon,
        models={role: tuned.role_config(role).primary for role in AgentRole},
    )


def build_arm_result(
    arm: AblationArm,
    records: Sequence[EvaluationRecord],
    rows: Sequence[Sequence[float]],
    wall_clock_seconds: float,
    horizon: int = 6,
    models: Mapping[AgentRole, ModelChoice] | None = None,
) -> ArmResult:
    """Consolida una corrida de un brazo."""
    return ArmResult(
        arm=arm,
        summary=summarise(records),
        outcomes=score_outcomes(records, rows, horizon),
        actions=decision_actions(records),
        wall_clock_seconds=wall_clock_seconds,
        models={role: choice.model for role, choice in (models or {}).items()},
    )


# ──────────────────────────────────── Conteo previo (dry-run) ─────────────────────────────────────
# Cuánto costaría la tabla antes de emitir una sola llamada. A ~90 s por evaluación
# y seis brazos, enterarse a mitad de la corrida de que un rol no cabe en su
# ventana cuesta las horas ya gastadas; enterarse antes cuesta un minuto de CPU.


DETERMINISTIC_NODES = frozenset(
    {"__start__", "__end__", "prepare", "consolidate", "risk", "execute", JOURNAL_NODE}
)
"""Nodos que no llaman a ningún modelo. Lo que sobre de aquí es coste."""

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

    @property
    def fits(self) -> bool:
        """Si la ablación entera cabe en una ventana de ese par."""
        return self.quota <= self.per_window


class DryRunReport(FrozenModel):
    """Conteo completo: qué se va a gastar y qué ya está pagado."""

    evaluations: int = Field(ge=0)
    activations: int = Field(ge=0)
    prepare_failures: int = Field(ge=0)
    """Velas donde la capa determinista falló, así que el gate nunca llegó a opinar."""

    rows: tuple[DryRunRow, ...] = ()
    quota: tuple[QuotaLine, ...] = ()

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
    rows: Sequence[Sequence[float]],
    settings: Settings,
    config: ReplaySettings,
    cache: ResponseCache,
    clock: Clock = utc_now,
    preset: IndicatorPreset = DEFAULT_PRESET,
    activation: ActivationConfig | None = None,
) -> DryRunReport:
    """Cuenta llamadas por rol y por brazo sin emitir ninguna.

    Recorre el histórico igual que `replay()` y ejecuta solo la capa determinista,
    que es la misma función que corre el nodo `prepare`. De ahí salen dos cosas: en
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

    frame = to_dataframe(rows)
    period = timeframe_to_timedelta(config.timeframe)
    digests: dict[str, list[str]] = {node: [] for node in EXACT_NODES}
    evaluations = 0
    activations = 0
    failures = 0

    for cursor in range(config.warmup_bars - 1, len(frame) - 1):
        if config.max_evaluations is not None and evaluations >= config.max_evaluations:
            break
        evaluations += 1

        moment = frame.index[cursor].to_pydatetime() + period
        context = AgentContext(
            settings=settings,
            router=router,
            market=HistoricalMarketClient(rows, cursor, config.candle_limit),
            account=NOTIONAL_ACCOUNT,
            run_id=replay_run_id(config.symbol, config.timeframe, moment),
            symbol=config.symbol,
            timeframe=config.timeframe,
            candle_limit=config.candle_limit,
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
                    cache_key(choice.model, digest, schema, choice.structured_output)
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
        )
        for (role, model), weights in sorted(spend.items(), key=lambda item: item[0][0].value)
    )
    return DryRunReport(
        evaluations=evaluations,
        activations=activations,
        prepare_failures=failures,
        rows=tuple(rows),
        quota=quota,
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
    return "\n".join(lines) + "\n"


def _count(value: int | None) -> str:
    """Un conteo, o un guion cuando el prompt no se puede calcular por adelantado."""
    return "—" if value is None else str(value)


# ────────────────────────────────────────────── Reporte ───────────────────────────────────────────


def _pct(value: float | None) -> str:
    """Porcentaje, o un guion si no hay dato."""
    return "—" if value is None else f"{value:.0%}"


def _num(value: float | None, digits: int = 2) -> str:
    """Número, o un guion si no hay dato."""
    return "—" if value is None else f"{value:.{digits}f}"


def render_report(results: Sequence[ArmResult], reference: str = "full") -> str:
    """Tabla comparativa en Markdown.

    Deliberadamente sin veredicto automático: qué significa que `solo` iguale a
    `full` es una conclusión que hay que escribir a mano, y una plantilla que la
    generase sola invitaría a no leerla.
    """
    if not results:
        return "# Ablación\n\nSin corridas.\n"

    baseline = next((item for item in results if item.arm.name == reference), results[0])
    lines = [
        "| brazo | evals | decididas | acciones | órdenes | coincidencia con "
        f"`{baseline.arm.name}` | cuota | latencia media | fallo validación |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        summary = result.summary
        mix = ", ".join(
            f"{action.value} {count}" for action, count in sorted(summary.actions.items())
        )
        failures = [stats.failure_rate for stats in summary.calls.values() if stats.answered > 0]
        lines.append(
            f"| `{result.arm.name}` | {summary.evaluations} | {summary.decided} | "
            f"{mix or '—'} | {summary.traded} | "
            f"{_pct(agreement(baseline.actions, result.actions))} | "
            f"{_num(summary.quota_used, 1)} | "
            f"{_num(result.mean_latency_ms, 0)} ms | "
            f"{_pct(max(failures) if failures else None)} |"
        )

    lines.append("")
    lines.append("| brazo | órdenes resueltas | invalidadas | aciertos | retorno medio |")
    lines.append("| --- | --- | --- | --- | --- |")
    for result in results:
        outcomes = result.outcomes
        lines.append(
            f"| `{result.arm.name}` | {outcomes.resolved} | {outcomes.invalidated} | "
            f"{_pct(outcomes.win_rate)} | {_pct(outcomes.mean_return)} |"
        )

    lines.append("")
    lines.append("| brazo | qué pregunta responde |")
    lines.append("| --- | --- |")
    lines.extend(f"| `{result.arm.name}` | {result.arm.question} |" for result in results)
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
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> str:
    """Corre los brazos pedidos y devuelve el reporte."""
    settings = load_settings(DEFAULT_ENV_FILE)
    rows = read_ohlcv_csv(args.history)
    config = ReplaySettings(
        symbol=args.symbol,
        timeframe=args.timeframe,
        warmup_bars=args.warmup,
        max_evaluations=args.evaluations,
    )
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
        report = await dry_run(
            [by_name[name] for name in wanted], rows, settings, config, cache, clock
        )
        return render_dry_run(report)

    fill_with = build_backends(settings) if args.fill else None
    # Un solo contador para los seis brazos: el proveedor ve la suma, no seis
    # presupuestos independientes. El brazo que agote la ventana aborta y lo
    # registra, en vez de creerse dentro.
    ledger = QuotaLedger(settings.quota_window, clock)

    results: list[ArmResult] = []
    for name in wanted:
        arm = by_name[name]
        print(f"· {arm.name}: {arm.question}", file=sys.stderr)
        results.append(
            await run_arm(
                arm,
                rows,
                settings,
                config,
                NOTIONAL_ACCOUNT,
                cache,
                clock,
                ledger,
                fill_with,
                args.horizon,
            )
        )
    return render_report(results)


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada: `python -m crypto_agents.ablation`."""
    args = _parse_args(argv)
    try:
        report = asyncio.run(_run(args))
    except (ConfigError, MarketDataError, ReplayCacheMissError) as error:
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
