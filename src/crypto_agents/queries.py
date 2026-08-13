"""Consultas sobre el journal.

La mitad de las preguntas que se le hacen a un sistema de trading son sobre lo que
*no* hizo, así que la causa de aborto es un filtro de primera clase y no una
búsqueda de texto sobre los mensajes de error.

Todo son funciones sobre secuencias de registros: quien las llama decide de dónde
salen —un `JsonlJournal` releído, un `InMemoryJournal` en caliente— y el filtrado
no depende del almacén.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from crypto_agents.journal import EvaluationRecord
    from crypto_agents.state import Action, Backend

__all__ = [
    "abort_cause",
    "by_abort_cause",
    "filter_records",
    "with_action",
    "with_backend",
    "with_symbol",
]

COMPLETED = "completed"
"""Causa de las evaluaciones que terminaron sin fallo de nodo."""


def abort_cause(record: EvaluationRecord) -> str:
    """Nodo que abortó la evaluación, o `completed` si llegó al final.

    Se devuelve el primero: los nodos posteriores fallan por consecuencia del
    primero, y agrupar por el último culparía al síntoma en vez de a la causa.
    """
    if not record.errors:
        return COMPLETED
    return record.errors[0].node


def with_symbol(records: Iterable[EvaluationRecord], symbol: str) -> list[EvaluationRecord]:
    """Evaluaciones de un mercado."""
    return [record for record in records if record.symbol == symbol]


def with_action(records: Iterable[EvaluationRecord], action: Action) -> list[EvaluationRecord]:
    """Evaluaciones cuya decisión fue esa acción, con mesas o sin ellas."""
    return [
        record
        for record in records
        if record.proposed is not None and record.proposed.action is action
    ]


def with_backend(records: Iterable[EvaluationRecord], backend: Backend) -> list[EvaluationRecord]:
    """Evaluaciones en las que ese proveedor atendió al menos una llamada.

    Es lo que permite separar decisiones producidas por un modelo remoto grande de
    las que salieron de un respaldo local, que de otro modo se leen igual.
    """
    return [record for record in records if any(call.backend is backend for call in record.calls)]


def by_abort_cause(records: Iterable[EvaluationRecord]) -> dict[str, list[EvaluationRecord]]:
    """Agrupa por dónde murió cada evaluación."""
    grouped: dict[str, list[EvaluationRecord]] = {}
    for record in records:
        grouped.setdefault(abort_cause(record), []).append(record)
    return dict(sorted(grouped.items()))


def filter_records(
    records: Sequence[EvaluationRecord],
    symbol: str | None = None,
    action: Action | None = None,
    backend: Backend | None = None,
    cause: str | None = None,
) -> list[EvaluationRecord]:
    """Aplica los filtros pedidos, en cadena. Sin ninguno, devuelve todo."""
    result = list(records)
    if symbol is not None:
        result = with_symbol(result, symbol)
    if action is not None:
        result = with_action(result, action)
    if backend is not None:
        result = with_backend(result, backend)
    if cause is not None:
        result = [record for record in result if abort_cause(record) == cause]
    return result
