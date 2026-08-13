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

from enum import StrEnum
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
    decide_single_desk,
    decide_solo,
    decide_without_debate,
    evidence_router,
    execute_order,
    prepare_market_data,
    record_evaluation,
    risk_gate,
    route_after_decision,
    route_after_gate,
    route_after_preparation,
    technical_momentum,
    technical_structure,
    technical_volume,
)
from crypto_agents.state import TradingState

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph

__all__ = ["AgentContext", "PipelineVariant", "build_graph"]


class PipelineVariant(StrEnum):
    """Formas del pipeline. Todas menos `FULL` existen solo para la ablación.

    Ninguna variante toca la cola determinista: decisor → riesgo → ejecución →
    journal es idéntica en las cuatro. Es lo que hace que la comparación mida los
    modelos y no el arnés, y lo que mantiene en pie la regla de que un LLM nunca
    es el último paso antes de una orden.
    """

    FULL = "full"
    """Tres técnicos, dos mesas, decisor. El pipeline que opera."""

    NO_DEBATE = "no_debate"
    """Tres técnicos y decisor. Mide qué aporta el debate."""

    BULL_ONLY = "bull_only"
    """Tres técnicos y una sola mesa. Mide qué aporta la contraparte."""

    SOLO = "solo"
    """Un modelo, una llamada: indicadores a decisión. Mide qué aporta todo lo demás."""


def build_graph(
    variant: PipelineVariant = PipelineVariant.FULL,
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> CompiledStateGraph[TradingState, AgentContext, TradingState, TradingState]:
    """Compila el grafo de una variante. Sin checkpointer explícito usa uno en memoria."""
    builder = StateGraph(TradingState, context_schema=AgentContext)

    builder.add_node("prepare", prepare_market_data)
    builder.add_node("risk", risk_gate)
    builder.add_node("execute", execute_order)
    builder.add_node(JOURNAL_NODE, record_evaluation)
    builder.add_edge(START, "prepare")

    if variant is PipelineVariant.SOLO:
        builder.add_node("decide_solo", decide_solo)
        builder.add_conditional_edges(
            "prepare", route_after_preparation, ["decide_solo", JOURNAL_NODE]
        )
        builder.add_conditional_edges("decide_solo", route_after_decision, ["risk", JOURNAL_NODE])
    else:
        _add_technical_fan_out(builder)
        desks = _decision_stage(builder, variant)
        builder.add_conditional_edges("consolidate", evidence_router(desks), [*desks, JOURNAL_NODE])

    builder.add_edge("risk", "execute")
    builder.add_edge("execute", JOURNAL_NODE)
    builder.add_edge(JOURNAL_NODE, END)

    return builder.compile(checkpointer=checkpointer or InMemorySaver())


def _add_technical_fan_out(builder: StateGraph[TradingState, AgentContext, TradingState]) -> None:
    """Los tres técnicos en paralelo y su consolidación. Común a todo menos a `SOLO`."""
    builder.add_node("structure", technical_structure)
    builder.add_node("momentum", technical_momentum)
    builder.add_node("volume", technical_volume)
    builder.add_node("consolidate", consolidate_evidence)

    builder.add_conditional_edges("prepare", route_after_gate, [*TECHNICAL_NODES, JOURNAL_NODE])
    for name in TECHNICAL_NODES:
        builder.add_edge(name, "consolidate")


def _decision_stage(
    builder: StateGraph[TradingState, AgentContext, TradingState], variant: PipelineVariant
) -> tuple[str, ...]:
    """Añade las mesas y el decisor de la variante. Devuelve los nodos que abre `consolidate`."""
    if variant is PipelineVariant.NO_DEBATE:
        builder.add_node("decide_without_debate", decide_without_debate)
        builder.add_conditional_edges(
            "decide_without_debate", route_after_decision, ["risk", JOURNAL_NODE]
        )
        return ("decide_without_debate",)

    builder.add_node("bull", debate_bull)
    desks: tuple[str, ...]
    if variant is PipelineVariant.FULL:
        builder.add_node("bear", debate_bear)
        desks, decider = DEBATE_NODES, decide
    else:
        desks, decider = ("bull",), decide_single_desk

    builder.add_node("decide", decider)
    for name in desks:
        builder.add_edge(name, "decide")
    builder.add_conditional_edges("decide", route_after_decision, ["risk", JOURNAL_NODE])
    return desks
