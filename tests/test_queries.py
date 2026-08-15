"""Pruebas de las consultas sobre el journal."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from crypto_agents.journal import EvaluationRecord
from crypto_agents.queries import (
    COMPLETED,
    abort_cause,
    by_abort_cause,
    filter_records,
    with_action,
    with_backend,
    with_symbol,
)
from crypto_agents.state import (
    Action,
    AgentRole,
    Backend,
    Decision,
    LLMCall,
    NodeError,
    Proposal,
    Side,
    StructuredOutputMode,
)

NOW = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)


def call(backend: Backend = Backend.OLLAMA) -> LLMCall:
    """Llamada atendida por ese proveedor."""
    return LLMCall(
        role=AgentRole.STRUCTURE,
        backend=backend,
        model="modelo",
        structured_output=StructuredOutputMode.JSON_SCHEMA,
        quota_weight=1.0,
        prompt_digest="f" * 64,
        valid=True,
        latency_ms=1.0,
        at=NOW,
    )


def buy() -> Decision:
    """Decisión de compra completa."""
    return Decision(
        action=Action.BUY,
        confidence=0.8,
        size_fraction=0.2,
        invalidation_price=99.0,
        rationale="La estructura acompana y el impulso todavia no esta agotado.",
        dismissed_side=Side.BEAR,
        dismissal_reason="Su nivel de referencia ya se perdio en la vela anterior.",
    )


def record(
    symbol: str = "BTC/USDT",
    decision: Decision | None = None,
    proposal: Proposal | None = None,
    errors: tuple[NodeError, ...] = (),
    calls: tuple[LLMCall, ...] = (),
) -> EvaluationRecord:
    """Registro mínimo con lo que la consulta mira."""
    return EvaluationRecord(
        run_id=uuid4(),
        at=NOW,
        symbol=symbol,
        timeframe="4h",
        decision=decision,
        proposal=proposal,
        errors=errors,
        calls=calls,
    )


def test_a_completed_evaluation_has_no_abort_cause() -> None:
    """Llegar al final es un estado, no la ausencia de uno."""
    assert abort_cause(record()) == COMPLETED


def test_the_abort_cause_is_the_first_node_that_failed() -> None:
    """Los nodos posteriores fallan por consecuencia; culpar al último sería culpar al síntoma."""
    aborted = record(
        errors=(
            NodeError(node="structure", message="cuota agotada", at=NOW),
            NodeError(node="consolidate_evidence", message="faltan veredictos", at=NOW),
        )
    )
    assert abort_cause(aborted) == "structure"


def test_records_group_by_where_they_died() -> None:
    """La mitad de las preguntas a un sistema de trading son sobre lo que no hizo."""
    records = [
        record(),
        record(errors=(NodeError(node="structure", message="x", at=NOW),)),
        record(errors=(NodeError(node="structure", message="y", at=NOW),)),
    ]
    grouped = by_abort_cause(records)

    assert len(grouped[COMPLETED]) == 1
    assert len(grouped["structure"]) == 2


def test_filtering_by_symbol() -> None:
    """Un mercado a la vez."""
    records = [record("BTC/USDT"), record("ETH/USDT"), record("BTC/USDT")]
    assert len(with_symbol(records, "BTC/USDT")) == 2


def test_filtering_by_action_covers_variants_without_desks() -> None:
    """Una decisión sin mesas cuenta igual: la acción es la acción."""
    records = [
        record(decision=buy()),
        record(
            proposal=Proposal(
                action=Action.BUY,
                confidence=0.6,
                size_fraction=0.1,
                invalidation_price=98.0,
                rationale="Sin mesas, pero la lectura tecnica es suficientemente clara.",
            )
        ),
        record(),
    ]
    assert len(with_action(records, Action.BUY)) == 2


def test_filtering_by_backend_separates_local_from_remote() -> None:
    """Es lo que distingue una decisión de un modelo grande de una de un respaldo pequeño."""
    records = [
        record(calls=(call(Backend.OPENAI),)),
        record(calls=(call(Backend.OLLAMA),)),
        record(calls=(call(Backend.OPENAI), call(Backend.OLLAMA))),
    ]
    assert len(with_backend(records, Backend.OPENAI)) == 2
    assert len(with_backend(records, Backend.OLLAMA)) == 2


def test_filters_compose() -> None:
    """Los cuatro filtros se encadenan sobre el mismo conjunto."""
    records = [
        record("BTC/USDT", decision=buy(), calls=(call(Backend.OPENAI),)),
        record("ETH/USDT", decision=buy(), calls=(call(Backend.OPENAI),)),
        record("BTC/USDT", calls=(call(Backend.OLLAMA),)),
    ]
    selected = filter_records(
        records, symbol="BTC/USDT", action=Action.BUY, backend=Backend.OPENAI, cause=COMPLETED
    )
    assert len(selected) == 1


def test_no_filters_returns_everything() -> None:
    """Sin criterios no se filtra nada, en vez de devolver vacío."""
    records = [record(), record()]
    assert len(filter_records(records)) == 2
