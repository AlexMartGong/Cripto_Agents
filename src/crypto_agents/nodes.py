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

# LangGraph resuelve las anotaciones de cada nodo con `get_type_hints`, así que
# estos tres tipos deben existir en runtime aunque solo aparezcan en firmas.
from langgraph.runtime import Runtime  # noqa: TC002

from crypto_agents.activation import evaluate_activation
from crypto_agents.context import AgentContext  # noqa: TC001
from crypto_agents.execution import build_order
from crypto_agents.indicators import enrich, to_indicator_set
from crypto_agents.journal import build_record
from crypto_agents.llm import InvalidModelOutputError
from crypto_agents.market import build_snapshot, load_candles
from crypto_agents.prompts import debate_prompt, decision_prompt, technical_prompt
from crypto_agents.quota import QuotaExhaustedError
from crypto_agents.risk import apply_risk
from crypto_agents.state import (
    AgentRole,
    DebateBrief,
    Decision,
    Dimension,
    NodeError,
    Side,
    TechnicalEvidence,
    TechnicalVerdict,
    TradingState,
    ungrounded_claim_refs,
    unknown_indicators,
)

__all__ = [
    "DEBATE_NODES",
    "JOURNAL_NODE",
    "TECHNICAL_NODES",
    "consolidate_evidence",
    "debate_bear",
    "debate_bull",
    "decide",
    "execute_order",
    "prepare_market_data",
    "record_evaluation",
    "risk_gate",
    "route_after_decision",
    "route_after_evidence",
    "route_after_gate",
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


def _error(node: str, message: str, runtime: Runtime[AgentContext]) -> dict[str, object]:
    """Registro de fallo atribuido a un nodo, sellado con el reloj inyectado."""
    return {"errors": [NodeError(node=node, message=message, at=runtime.context.clock())]}


# ────────────────────────────────────────── Capa determinista ─────────────────────────────────────


async def prepare_market_data(
    state: TradingState, runtime: Runtime[AgentContext]
) -> dict[str, object]:
    """Descarga velas, calcula indicadores y evalúa el gate de activación.

    Todo lo determinista ocurre aquí, en un solo nodo, para que el DataFrame nunca
    entre en el estado: se serializaría en cada checkpoint.
    """
    context = runtime.context
    try:
        candles = await load_candles(
            context.market, context.symbol, context.timeframe, context.candle_limit, context.now
        )
        enriched = enrich(candles, context.preset)
        indicators = to_indicator_set(enriched, context.preset)
        activation = evaluate_activation(enriched, context.activation)
    except Exception as error:  # cualquier fallo aquí aborta la corrida antes de gastar cuota
        return _error("prepare_market_data", f"{type(error).__name__}: {error}", runtime)

    snapshot = build_snapshot(
        candles,
        context.run_id,
        context.settings.exchange.exchange_id,
        context.symbol,
        context.timeframe,
    )
    return {"snapshot": snapshot, "indicators": indicators, "activation": activation}


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
    try:
        verdict, calls = await runtime.context.router.invoke(
            _ROLE_BY_DIMENSION[dimension], prompt, TechnicalVerdict
        )
    except (InvalidModelOutputError, QuotaExhaustedError) as error:
        return _error(node, str(error), runtime)

    if verdict.dimension is not dimension:
        return {
            "calls": calls,
            **_error(node, f"el agente respondió como {verdict.dimension.value}", runtime),
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

    Exige los tres. Y rechaza indicadores citados que no existen: una cita
    inventada convierte el veredicto en algo que no se puede auditar, y dejarlo
    pasar haría que `unknown_indicators()` fuera decorativo.
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
        return _error(node, f"indicadores citados que no existen -> {detail}", runtime)

    evidence = TechnicalEvidence(
        snapshot=state.snapshot,
        indicators=state.indicators,
        verdicts=tuple(sorted(state.verdicts, key=lambda item: item.dimension.value)),
    )
    return {"evidence": evidence}


def route_after_evidence(state: TradingState) -> list[str]:
    """Abre las dos mesas solo si la evidencia se consolidó."""
    if state.evidence is None:
        return [JOURNAL_NODE]
    return list(DEBATE_NODES)


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
    try:
        brief, calls = await runtime.context.router.invoke(_ROLE_BY_SIDE[side], prompt, DebateBrief)
    except (InvalidModelOutputError, QuotaExhaustedError) as error:
        return _error(node, str(error), runtime)

    invented = ungrounded_claim_refs(brief, evidence.verdicts)
    if invented:
        return {
            "calls": calls,
            **_error(node, f"cita ids inexistentes: {', '.join(invented)}", runtime),
        }
    if brief.side is not side:
        return {
            "calls": calls,
            **_error(node, f"la mesa respondió como {brief.side.value}", runtime),
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
    except (InvalidModelOutputError, QuotaExhaustedError) as error:
        return _error(node, str(error), runtime)

    return {"decision": decision, "calls": calls}


# ─────────────────────────────────────── Gate de riesgo y salida ──────────────────────────────────


def risk_gate(state: TradingState, runtime: Runtime[AgentContext]) -> dict[str, object]:
    """Aplica los límites deterministas. Es código, no un modelo.

    Recibe la `Decision` pero no le concede autoridad sobre el tamaño: lo que
    sale es un `RiskVerdict`, y de ahí se lee todo lo que llega al mercado.
    """
    if state.decision is None:
        return _error("risk_gate", "no hay decisión que evaluar", runtime)

    context = runtime.context
    verdict = apply_risk(state.decision, context.account, context.settings.risk, context.clock())
    return {"risk": verdict}


def route_after_decision(state: TradingState) -> list[str]:
    """Solo se pasa al gate de riesgo si hubo decisión."""
    if state.decision is None:
        return [JOURNAL_NODE]
    return ["risk"]


async def execute_order(state: TradingState, runtime: Runtime[AgentContext]) -> dict[str, object]:
    """Envía la orden si el gate la aprobó.

    El tamaño viene de `state.risk`, nunca de `state.decision`.
    """
    if state.decision is None or state.risk is None or state.snapshot is None:
        return {}

    order = build_order(
        state.decision, state.risk, state.snapshot, runtime.context.settings.execution.mode
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
