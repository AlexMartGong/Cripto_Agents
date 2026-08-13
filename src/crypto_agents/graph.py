"""Construcción del grafo y contexto de ejecución.

El contexto lleva el cableado —configuración, router, cliente de mercado, reloj—
y el objetivo de la evaluación. Nada de eso entra en `TradingState`: el estado se
serializa en cada checkpoint y un cliente de exchange no es serializable.

El checkpointing está activo desde el principio. Sin él, una evaluación
interrumpida a mitad del abanico técnico pierde los veredictos ya pagados y hay
que volver a gastarlos.

Forma del grafo:

    START -> prepare -> (sin triggers) ------------------------------> journal
                     -> structure ┐
                        momentum  ├-> consolidate -> (sin evidencia) -> journal
                        volume    ┘                -> bull ┐
                                                     bear  ┴-> decide -> risk
                                                              -> execute -> journal -> END

Todas las salidas pasan por el journal, incluidas las abortadas: una evaluación
que no operó es justo la que se querrá auditar después.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from crypto_agents.context import AgentContext
from crypto_agents.nodes import (
    DEBATE_NODES,
    JOURNAL_NODE,
    TECHNICAL_NODES,
    consolidate_evidence,
    debate_bear,
    debate_bull,
    decide,
    execute_order,
    prepare_market_data,
    record_evaluation,
    risk_gate,
    route_after_decision,
    route_after_evidence,
    route_after_gate,
    technical_momentum,
    technical_structure,
    technical_volume,
)
from crypto_agents.state import TradingState

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph

__all__ = ["AgentContext", "build_graph"]


def build_graph(
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> CompiledStateGraph[TradingState, AgentContext, TradingState, TradingState]:
    """Compila el grafo. Sin checkpointer explícito usa uno en memoria."""
    builder = StateGraph(TradingState, context_schema=AgentContext)

    builder.add_node("prepare", prepare_market_data)
    builder.add_node("structure", technical_structure)
    builder.add_node("momentum", technical_momentum)
    builder.add_node("volume", technical_volume)
    builder.add_node("consolidate", consolidate_evidence)
    builder.add_node("bull", debate_bull)
    builder.add_node("bear", debate_bear)
    builder.add_node("decide", decide)
    builder.add_node("risk", risk_gate)
    builder.add_node("execute", execute_order)
    builder.add_node(JOURNAL_NODE, record_evaluation)

    builder.add_edge(START, "prepare")
    builder.add_conditional_edges("prepare", route_after_gate, [*TECHNICAL_NODES, JOURNAL_NODE])
    for name in TECHNICAL_NODES:
        builder.add_edge(name, "consolidate")
    builder.add_conditional_edges(
        "consolidate", route_after_evidence, [*DEBATE_NODES, JOURNAL_NODE]
    )
    for name in DEBATE_NODES:
        builder.add_edge(name, "decide")
    builder.add_conditional_edges("decide", route_after_decision, ["risk", JOURNAL_NODE])
    builder.add_edge("risk", "execute")
    builder.add_edge("execute", JOURNAL_NODE)
    builder.add_edge(JOURNAL_NODE, END)

    return builder.compile(checkpointer=checkpointer or InMemorySaver())
