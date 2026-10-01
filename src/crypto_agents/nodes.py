"""Nodos del grafo.

Un nodo determinista al principio y uno por agente después. Los nodos LLM no
tocan ccxt ni pandas; el nodo determinista no llama a ningún modelo.

Todos los nodos devuelven diccionarios parciales. Los campos con varios
escritores (`verdicts`, `briefs`, `calls`, `errors`) llevan reducer en
`TradingState`, así que acumulan en vez de pisarse.

Un fallo de un agente técnico aborta la evaluación en lugar de continuar con
evidencia parcial: decidir con dos tercios de la evidencia es exactamente el
fallo silencioso que el reducer existe para evitar.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, NamedTuple

# LangGraph resuelve las anotaciones de cada nodo con `get_type_hints`, así que
# estos tres tipos deben existir en runtime aunque solo aparezcan en firmas.
from langgraph.runtime import Runtime  # noqa: TC002

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from crypto_agents.llm import ContextCheck

from crypto_agents.activation import evaluate_activation
from crypto_agents.baselines import (
    always_buy_proposal,
    always_sell_proposal,
    random_proposal,
    trend_proposal,
)
from crypto_agents.context import AgentContext  # noqa: TC001
from crypto_agents.execution import build_order
from crypto_agents.indicators import enrich, to_indicator_set
from crypto_agents.journal import build_record
from crypto_agents.llm import ModelInvocationError
from crypto_agents.market import build_snapshot, load_candles
from crypto_agents.prompts import (
    debate_prompt,
    decision_prompt,
    no_debate_prompt,
    solo_prompt,
    technical_prompt,
)
from crypto_agents.quota import QuotaExhaustedError
from crypto_agents.risk import apply_risk
from crypto_agents.state import (
    ActivationCheck,
    AgentRole,
    DebateBrief,
    Decision,
    Dimension,
    IndicatorSet,
    MarketSnapshot,
    NodeError,
    Proposal,
    Side,
    TechnicalEvidence,
    TechnicalVerdict,
    TradingState,
    ungrounded_claim_refs,
    unknown_indicators,
)

__all__ = [
    "BASELINE_NODES",
    "CONTEXT_BYPASSED",
    "DEBATE_NODES",
    "JOURNAL_NODE",
    "TECHNICAL_NODES",
    "PreparedEvaluation",
    "consolidate_evidence",
    "debate_bear",
    "debate_bull",
    "debate_context",
    "decide",
    "decide_always_buy",
    "decide_always_sell",
    "decide_random_uniform",
    "decide_rule_trend",
    "decide_single_desk",
    "decide_solo",
    "decide_without_debate",
    "evidence_router",
    "execute_order",
    "preparation_router",
    "prepare_evaluation",
    "prepare_market_data",
    "record_evaluation",
    "risk_gate",
    "route_after_decision",
    "route_after_evidence",
    "route_after_gate",
    "route_after_preparation",
    "technical_context",
]

TECHNICAL_NODES = ("structure", "momentum", "volume")
DEBATE_NODES = ("bull", "bear")
JOURNAL_NODE = "journal"

_ROLE_BY_DIMENSION = {
    Dimension.STRUCTURE: AgentRole.STRUCTURE,
    Dimension.MOMENTUM: AgentRole.MOMENTUM,
    Dimension.VOLUME: AgentRole.VOLUME,
}
_ROLE_BY_SIDE = {Side.BULL: AgentRole.BULL, Side.BEAR: AgentRole.BEAR}


CONTEXT_BYPASSED = "validación de contexto saltada"
"""Prefijo del error de un nodo que recibió del router una salida que debía rechazar.

No debería ocurrir: el nodo le pasa su validación al router, que no devuelve nada
que no la cumpla. Si ocurre es un bug —alguien quitó la validación de una llamada,
o el router dejó de aplicarla— y se registra como `NodeError` en vez de dejar
pasar la salida: la comprobación que antes era la única defensa se queda como
aserción.
"""


def technical_context(
    dimension: Dimension, indicators: IndicatorSet
) -> ContextCheck[TechnicalVerdict]:
    """Lo que un veredicto técnico tiene que cumplir además de su esquema.

    Que responda por la dimensión que se le pidió y que solo cite indicadores que
    existen. Lo segundo se comprobaba en `consolidate_evidence`, sobre los tres
    veredictos ya aceptados: una cita inventada abortaba la evaluación entera sin
    darle al modelo la ocasión de corregirla, con su llamada marcada como válida.
    """

    def check(verdict: TechnicalVerdict) -> list[str]:
        problems: list[str] = []
        if verdict.dimension is not dimension:
            problems.append(
                f"dimension: respondiste como {verdict.dimension.value} "
                f"y se te pidió {dimension.value}"
            )
        unknown = unknown_indicators(verdict, indicators)
        if unknown:
            problems.append(f"cites: indicadores que no existen -> {', '.join(unknown)}")
        return problems

    return check


def debate_context(side: Side, verdicts: Sequence[TechnicalVerdict]) -> ContextCheck[DebateBrief]:
    """Lo que un alegato tiene que cumplir además de su esquema.

    Que hable por la mesa que se le pidió y que cada afirmación cite ids que algún
    veredicto emitió. Es la regla 3 llevada hasta donde puede hacerse cumplir: el
    esquema exige que haya al menos una cita, y esto, que la cita exista.
    """

    def check(brief: DebateBrief) -> list[str]:
        problems: list[str] = []
        if brief.side is not side:
            problems.append(f"side: respondiste como {brief.side.value} y se te pidió {side.value}")
        invented = ungrounded_claim_refs(brief, verdicts)
        if invented:
            problems.append(
                f"grounded_in: ids que ningún veredicto emitió -> {', '.join(invented)}"
            )
        return problems

    return check


def _error(node: str, message: str, runtime: Runtime[AgentContext]) -> dict[str, object]:
    """Registro de fallo atribuido a un nodo, sellado con el reloj inyectado."""
    return {"errors": [NodeError(node=node, message=message, at=runtime.context.clock())]}


def _model_failure(
    node: str, error: ModelInvocationError | QuotaExhaustedError, runtime: Runtime[AgentContext]
) -> dict[str, object]:
    """Fallo de modelo: el mensaje y los intentos que ya se pagaron.

    Los intentos se escriben en el estado aunque la evaluación aborte. El
    contador de cuota ya los tiene, pero vive en memoria: sin copiarlos aquí, el
    journal de una evaluación que se cayó por el proveedor sale con `calls`
    vacío, y en el archivo queda indistinguible de una que no llamó a nadie.

    `QuotaExhaustedError` no trae ninguno, y es correcto: se lanza antes de tocar
    el backend, así que no se gastó nada que registrar.
    """
    calls = list(error.calls) if isinstance(error, ModelInvocationError) else []
    return {"calls": calls, **_error(node, str(error), runtime)}


# ────────────────────────────────────────── Capa determinista ─────────────────────────────────────


class PreparedEvaluation(NamedTuple):
    """Lo determinista de una evaluación: el momento, sus indicadores y el gate."""

    snapshot: MarketSnapshot
    indicators: IndicatorSet
    activation: ActivationCheck


async def prepare_evaluation(context: AgentContext) -> PreparedEvaluation:
    """Velas, indicadores y gate de activación. Ni un modelo de por medio.

    Está separada del nodo para que se pueda ejecutar sin grafo: el conteo previo
    de la ablación necesita exactamente esto —cuántas evaluaciones abren el gate y
    con qué prompts— y una copia de estas cinco líneas allí mediría el arnés en
    lugar del sistema. El prompt técnico es función de lo que se devuelve aquí, así
    que su digest se puede calcular antes de gastar nada.
    """
    candles = await load_candles(
        context.market, context.symbol, context.timeframe, context.candle_limit, context.now
    )
    enriched = enrich(candles, context.preset)
    return PreparedEvaluation(
        snapshot=build_snapshot(
            candles,
            context.run_id,
            context.settings.exchange.exchange_id,
            context.symbol,
            context.timeframe,
        ),
        indicators=to_indicator_set(enriched, context.preset),
        activation=evaluate_activation(enriched, context.activation),
    )


async def prepare_market_data(
    state: TradingState, runtime: Runtime[AgentContext]
) -> dict[str, object]:
    """Descarga velas, calcula indicadores y evalúa el gate de activación.

    Todo lo determinista ocurre aquí, en un solo nodo, para que el DataFrame nunca
    entre en el estado: se serializaría en cada checkpoint.
    """
    try:
        prepared = await prepare_evaluation(runtime.context)
    except Exception as error:  # cualquier fallo aquí aborta la corrida antes de gastar cuota
        return _error("prepare_market_data", f"{type(error).__name__}: {error}", runtime)

    return {
        "snapshot": prepared.snapshot,
        "indicators": prepared.indicators,
        "activation": prepared.activation,
    }


def route_after_gate(state: TradingState) -> list[str]:
    """Abre el abanico técnico solo si el gate disparó.

    Cortar no significa saltarse el registro: una evaluación que decidió no
    operar es exactamente la que uno querrá auditar después.
    """
    if state.activation is None or not state.activation.should_run:
        return [JOURNAL_NODE]
    return list(TECHNICAL_NODES)


# ─────────────────────────────────────────── Agentes técnicos ─────────────────────────────────────


async def _run_technical(
    dimension: Dimension, state: TradingState, runtime: Runtime[AgentContext]
) -> dict[str, object]:
    """Un agente técnico: interpreta indicadores ya calculados."""
    node = dimension.value
    if state.snapshot is None or state.indicators is None or state.activation is None:
        return _error(node, "falta la preparación determinista", runtime)

    prompt = technical_prompt(
        dimension, state.snapshot, state.indicators, state.activation.triggers
    )
    check = technical_context(dimension, state.indicators)
    try:
        verdict, calls = await runtime.context.router.invoke(
            _ROLE_BY_DIMENSION[dimension], prompt, TechnicalVerdict, check
        )
    except (ModelInvocationError, QuotaExhaustedError) as error:
        return _model_failure(node, error, runtime)

    problems = check(verdict)
    if problems:
        return {
            "calls": calls,
            **_error(node, f"{CONTEXT_BYPASSED}: {'; '.join(problems)}", runtime),
        }
    return {"verdicts": [verdict], "calls": calls}


async def technical_structure(
    state: TradingState, runtime: Runtime[AgentContext]
) -> dict[str, object]:
    """Agente de estructura."""
    return await _run_technical(Dimension.STRUCTURE, state, runtime)


async def technical_momentum(
    state: TradingState, runtime: Runtime[AgentContext]
) -> dict[str, object]:
    """Agente de momentum."""
    return await _run_technical(Dimension.MOMENTUM, state, runtime)


async def technical_volume(
    state: TradingState, runtime: Runtime[AgentContext]
) -> dict[str, object]:
    """Agente de volumen."""
    return await _run_technical(Dimension.VOLUME, state, runtime)


# ───────────────────────────────────────── Evidencia común ────────────────────────────────────────


def consolidate_evidence(state: TradingState, runtime: Runtime[AgentContext]) -> dict[str, object]:
    """Consolida los tres veredictos en el objeto que reciben ambas mesas.

    Exige los tres. Las citas de indicadores inexistentes ya las rechazó el router
    en cada agente técnico, con reintento; aquí queda la misma comprobación como
    aserción, porque este es el último punto antes de que la evidencia llegue a
    las mesas y una cita inventada la vuelve inauditable.
    """
    node = "consolidate_evidence"
    if state.snapshot is None or state.indicators is None:
        return _error(node, "falta la preparación determinista", runtime)

    missing = sorted(set(Dimension) - {verdict.dimension for verdict in state.verdicts})
    if missing:
        return _error(
            node,
            f"faltan veredictos técnicos: {', '.join(item.value for item in missing)}",
            runtime,
        )

    hallucinated = {
        verdict.dimension.value: names
        for verdict in state.verdicts
        if (names := unknown_indicators(verdict, state.indicators))
    }
    if hallucinated:
        detail = "; ".join(
            f"{dimension}: {', '.join(names)}" for dimension, names in sorted(hallucinated.items())
        )
        return _error(
            node, f"{CONTEXT_BYPASSED}: indicadores citados que no existen -> {detail}", runtime
        )

    evidence = TechnicalEvidence(
        snapshot=state.snapshot,
        indicators=state.indicators,
        verdicts=tuple(sorted(state.verdicts, key=lambda item: item.dimension.value)),
    )
    return {"evidence": evidence}


def evidence_router(targets: Sequence[str]) -> Callable[[TradingState], list[str]]:
    """Router hacia lo que venga después de la evidencia en esta variante.

    Los destinos se declaran porque no todas las variantes tienen dos mesas: un
    router fijo mandaría a `bear` en una forma del grafo donde ese nodo no existe.
    """

    def route(state: TradingState) -> list[str]:
        if state.evidence is None:
            return [JOURNAL_NODE]
        return list(targets)

    return route


route_after_evidence = evidence_router(DEBATE_NODES)
"""Router del pipeline completo: abre las dos mesas si la evidencia se consolidó."""


# ──────────────────────────────────────────── Mesas ───────────────────────────────────────────────


async def _run_debate(
    side: Side, state: TradingState, runtime: Runtime[AgentContext]
) -> dict[str, object]:
    """Una mesa argumenta sobre la evidencia común."""
    node = side.value
    if state.evidence is None:
        return _error(node, "no hay evidencia consolidada", runtime)

    evidence = state.evidence
    prompt = debate_prompt(side, evidence.snapshot, evidence.indicators, evidence.verdicts)
    check = debate_context(side, evidence.verdicts)
    try:
        brief, calls = await runtime.context.router.invoke(
            _ROLE_BY_SIDE[side], prompt, DebateBrief, check
        )
    except (ModelInvocationError, QuotaExhaustedError) as error:
        return _model_failure(node, error, runtime)

    problems = check(brief)
    if problems:
        return {
            "calls": calls,
            **_error(node, f"{CONTEXT_BYPASSED}: {'; '.join(problems)}", runtime),
        }
    return {"briefs": [brief], "calls": calls}


async def debate_bull(state: TradingState, runtime: Runtime[AgentContext]) -> dict[str, object]:
    """Mesa alcista."""
    return await _run_debate(Side.BULL, state, runtime)


async def debate_bear(state: TradingState, runtime: Runtime[AgentContext]) -> dict[str, object]:
    """Mesa bajista."""
    return await _run_debate(Side.BEAR, state, runtime)


# ─────────────────────────────────────────── Decisor ──────────────────────────────────────────────


async def decide(state: TradingState, runtime: Runtime[AgentContext]) -> dict[str, object]:
    """Emite la decisión sobre la evidencia y los dos alegatos.

    Exige ambas mesas: decidir con un solo alegato es leer propaganda sin
    contraparte, que es justo lo que el debate existe para evitar.
    """
    node = "decide"
    if state.evidence is None:
        return _error(node, "no hay evidencia consolidada", runtime)

    bull, bear = state.brief(Side.BULL), state.brief(Side.BEAR)
    missing = [
        side.value for side, brief in ((Side.BULL, bull), (Side.BEAR, bear)) if brief is None
    ]
    if missing:
        return _error(node, f"faltan alegatos: {', '.join(missing)}", runtime)

    evidence = state.evidence
    prompt = decision_prompt(evidence.snapshot, evidence.indicators, evidence.verdicts, bull, bear)
    try:
        decision, calls = await runtime.context.router.invoke(AgentRole.DECIDER, prompt, Decision)
    except (ModelInvocationError, QuotaExhaustedError) as error:
        return _model_failure(node, error, runtime)

    return {"decision": decision, "calls": calls}


# ────────────────────────────────── Decisores sin debate (ablación) ───────────────────────────────
# Emiten `Proposal` en vez de `Decision`: sin mesas no hay lado que descartar, y
# `Decision` exige nombrarlo. Escriben en `state.proposal`, nunca en
# `state.decision`, para que el journal distinga «no hubo mesas» de «el decisor no
# dijo a quién descartaba».


async def decide_single_desk(
    state: TradingState, runtime: Runtime[AgentContext]
) -> dict[str, object]:
    """Decide con una sola mesa. Variante de ablación.

    Nodo aparte y no un `decide` más permisivo: en el pipeline que opera, exigir
    los dos alegatos es una garantía, y ablandarla para que quepa un experimento la
    destruiría en producción también. Sigue emitiendo `Decision`, porque con una
    mesa sí hay un lado que descartar.
    """
    node = "decide_single_desk"
    if state.evidence is None:
        return _error(node, "no hay evidencia consolidada", runtime)

    bull = state.brief(Side.BULL)
    if bull is None:
        return _error(node, "faltan alegatos: bull", runtime)

    evidence = state.evidence
    prompt = decision_prompt(evidence.snapshot, evidence.indicators, evidence.verdicts, bull, None)
    try:
        decision, calls = await runtime.context.router.invoke(AgentRole.DECIDER, prompt, Decision)
    except (ModelInvocationError, QuotaExhaustedError) as error:
        return _model_failure(node, error, runtime)

    return {"decision": decision, "calls": calls}


async def decide_without_debate(
    state: TradingState, runtime: Runtime[AgentContext]
) -> dict[str, object]:
    """Decide sobre la evidencia técnica, sin alegatos. Variante de ablación."""
    node = "decide_without_debate"
    if state.evidence is None:
        return _error(node, "no hay evidencia consolidada", runtime)

    evidence = state.evidence
    prompt = no_debate_prompt(evidence.snapshot, evidence.indicators, evidence.verdicts)
    try:
        proposal, calls = await runtime.context.router.invoke(AgentRole.DECIDER, prompt, Proposal)
    except (ModelInvocationError, QuotaExhaustedError) as error:
        return _model_failure(node, error, runtime)

    return {"proposal": proposal, "calls": calls}


async def decide_solo(state: TradingState, runtime: Runtime[AgentContext]) -> dict[str, object]:
    """Un modelo, una llamada: de indicadores a decisión. El baseline de la ablación."""
    node = "decide_solo"
    if state.snapshot is None or state.indicators is None or state.activation is None:
        return _error(node, "falta la preparación determinista", runtime)

    prompt = solo_prompt(state.snapshot, state.indicators, state.activation.triggers)
    try:
        proposal, calls = await runtime.context.router.invoke(AgentRole.DECIDER, prompt, Proposal)
    except (ModelInvocationError, QuotaExhaustedError) as error:
        return _model_failure(node, error, runtime)

    return {"proposal": proposal, "calls": calls}


def preparation_router(target: str) -> Callable[[TradingState], list[str]]:
    """Router del gate al único nodo que decide, en las variantes sin técnicos ni mesas.

    El destino se declara porque cada una tiene el suyo: `decide_solo` llama a un
    modelo y las líneas base no. Con el gate cerrado todas van al journal.
    """

    def route(state: TradingState) -> list[str]:
        if state.activation is None or not state.activation.should_run:
            return [JOURNAL_NODE]
        return [target]

    return route


route_after_preparation = preparation_router("decide_solo")
"""Variante generalista: del gate de activación directo al único modelo."""


# ──────────────────────────────────────────── Líneas base ─────────────────────────────────────────
# Cuatro decisores que no llaman a nadie. Escriben `proposal`, como `decide_solo`, y de ahí en
# adelante el camino es el de siempre: riesgo, ejecución, journal. Ninguno toca el router, que
# ni siquiera necesitan: `baselines.py` no puede importarlo.

BASELINE_NODES = (
    "decide_always_buy",
    "decide_always_sell",
    "decide_random_uniform",
    "decide_rule_trend",
)
"""Nodos que deciden sin modelo. La ablación los cuenta como coste cero."""


def _decide_baseline(
    node: str,
    state: TradingState,
    runtime: Runtime[AgentContext],
    build: Callable[[MarketSnapshot, IndicatorSet], Proposal],
) -> dict[str, object]:
    """Construye la propuesta de una línea base con la preparación determinista."""
    if state.snapshot is None or state.indicators is None:
        return _error(node, "falta la preparación determinista", runtime)
    try:
        proposal = build(state.snapshot, state.indicators)
    except (KeyError, ValueError) as error:
        return _error(node, f"{type(error).__name__}: {error}", runtime)
    return {"proposal": proposal}


def decide_always_buy(state: TradingState, runtime: Runtime[AgentContext]) -> dict[str, object]:
    """Compra en cada activación. Línea base de la ablación."""
    return _decide_baseline("decide_always_buy", state, runtime, always_buy_proposal)


def decide_always_sell(state: TradingState, runtime: Runtime[AgentContext]) -> dict[str, object]:
    """Vende en cada activación. Línea base de la ablación."""
    return _decide_baseline("decide_always_sell", state, runtime, always_sell_proposal)


def decide_random_uniform(state: TradingState, runtime: Runtime[AgentContext]) -> dict[str, object]:
    """Buy, sell o hold al azar, con la semilla del plan y el `run_id` de la evaluación."""
    context = runtime.context
    build = partial(random_proposal, seed=context.seed, run_id=context.run_id)
    return _decide_baseline("decide_random_uniform", state, runtime, build)


def decide_rule_trend(state: TradingState, runtime: Runtime[AgentContext]) -> dict[str, object]:
    """La regla de tendencia con el preset de la evaluación."""
    build = partial(trend_proposal, preset=runtime.context.preset)
    return _decide_baseline("decide_rule_trend", state, runtime, build)


# ─────────────────────────────────────── Gate de riesgo y salida ──────────────────────────────────


def risk_gate(state: TradingState, runtime: Runtime[AgentContext]) -> dict[str, object]:
    """Aplica los límites deterministas. Es código, no un modelo.

    Recibe lo que se haya decidido pero no le concede autoridad sobre el tamaño:
    lo que sale es un `RiskVerdict`, y de ahí se lee todo lo que llega al mercado.
    Lee `state.proposed`, así que su código es idéntico en todas las variantes.

    El kill switch se consulta aquí, en cada evaluación, y no en el runner: este es
    el único punto por el que pasan todos los caminos a una orden. El I/O de
    comprobarlo queda en el nodo; `apply_risk` sigue leyendo un booleano y siendo
    una función pura.

    El cierre que recibe el gate es `snapshot.close`, el mismo que `build_order`
    escribe como `reference_price`: el lado del stop se juzga contra el precio con
    el que se construiría la orden, no contra otro.
    """
    if state.proposed is None:
        return _error("risk_gate", "no hay decisión que evaluar", runtime)
    if state.snapshot is None:
        return _error("risk_gate", "falta la preparación determinista", runtime)

    context = runtime.context
    limits = context.settings.risk
    if context.kill_switch.engaged():
        limits = limits.model_copy(update={"kill_switch": True})

    verdict = apply_risk(
        state.proposed, context.account, limits, context.clock(), state.snapshot.close
    )
    return {"risk": verdict}


def route_after_decision(state: TradingState) -> list[str]:
    """Solo se pasa al gate de riesgo si hubo decisión, la tomara quien la tomara."""
    if state.proposed is None:
        return [JOURNAL_NODE]
    return ["risk"]


async def execute_order(state: TradingState, runtime: Runtime[AgentContext]) -> dict[str, object]:
    """Envía la orden si el gate la aprobó.

    El tamaño viene de `state.risk`, nunca de `state.decision`.
    """
    proposed = state.proposed
    if proposed is None or state.risk is None or state.snapshot is None:
        return {}

    order = build_order(
        proposed, state.risk, state.snapshot, runtime.context.settings.execution.mode
    )
    if order is None:
        return {}

    try:
        await runtime.context.executor.submit(order)
    except Exception as error:
        return {
            "errors": [
                NodeError(
                    node="execute_order",
                    message=f"{type(error).__name__}: {error}",
                    at=runtime.context.clock(),
                )
            ]
        }
    return {"order": order}


def record_evaluation(state: TradingState, runtime: Runtime[AgentContext]) -> dict[str, object]:
    """Escribe el registro estructurado de la evaluación, haya operado o no."""
    context = runtime.context
    record = build_record(
        state,
        run_id=context.run_id,
        symbol=context.symbol,
        timeframe=context.timeframe,
        at=context.clock(),
        order=state.order,
    )
    context.journal.write(record)
    return {}
