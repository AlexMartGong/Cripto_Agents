"""Registro estructurado de cada evaluación.

Una línea por evaluación, incluidas las que murieron en el gate de activación o
en un fallo de agente: si una evaluación que no operó no deja rastro, el log no
sirve para auditar por qué no se operó, que es la mitad de las preguntas que uno
le hace a un sistema de trading.

El registro es un modelo Pydantic, así que se serializa y se vuelve a leer con
el mismo esquema que produjo la evaluación.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Protocol
from uuid import UUID

from pydantic import AwareDatetime, Field, ValidationError

from crypto_agents.state import (
    ActivationCheck,
    DebateBrief,
    Decision,
    FrozenModel,
    LLMCall,
    MarketSnapshot,
    NodeError,
    OrderIntent,
    Proposal,
    RiskVerdict,
    TechnicalEvidence,
)

if TYPE_CHECKING:
    from datetime import datetime

    from crypto_agents.state import TradingState

__all__ = [
    "EvaluationRecord",
    "InMemoryJournal",
    "Journal",
    "JournalError",
    "JsonlJournal",
    "build_record",
]


class JournalError(ValueError):
    """Una línea del journal no se pudo leer como evaluación.

    Nombra archivo y línea. Quien relee un journal lo hace para decidir algo con
    él —cuánta cuota queda, qué pasó en una corrida—, y una línea saltada en
    silencio es un recuento corto que nadie sabe que está corto.
    """

    def __init__(self, path: Path, line: int, cause: Exception) -> None:
        self.path = path
        self.line = line
        super().__init__(f"journal {path} ilegible en la línea {line}: {cause}")


class EvaluationRecord(FrozenModel):
    """Todo lo que ocurrió en una evaluación, en un solo objeto."""

    run_id: UUID
    at: AwareDatetime
    symbol: str = Field(min_length=1)
    timeframe: str = Field(min_length=1)

    snapshot: MarketSnapshot | None = None
    activation: ActivationCheck | None = None
    evidence: TechnicalEvidence | None = None
    briefs: tuple[DebateBrief, ...] = ()
    decision: Decision | None = None
    proposal: Proposal | None = None
    """Decisión de una variante sin mesas. Ver `TradingState.proposal`."""

    risk: RiskVerdict | None = None
    order: OrderIntent | None = None

    calls: tuple[LLMCall, ...] = ()
    errors: tuple[NodeError, ...] = ()

    @property
    def quota_used(self) -> float:
        """Cuota consumida por la evaluación. Los aciertos de caché no cuentan."""
        return sum(call.quota_weight for call in self.calls if not call.cache_hit)

    @property
    def proposed(self) -> Proposal | None:
        """Lo que se decidió, con mesas o sin ellas."""
        if self.decision is not None:
            return self.decision
        return self.proposal

    @property
    def traded(self) -> bool:
        """Si la evaluación acabó produciendo una orden."""
        return self.order is not None


class Journal(Protocol):
    """Destino de los registros de evaluación."""

    def write(self, record: EvaluationRecord) -> None:
        """Guarda un registro."""
        ...


class InMemoryJournal:
    """Journal por proceso, para pruebas y para inspección en caliente."""

    def __init__(self) -> None:
        self.records: list[EvaluationRecord] = []

    def write(self, record: EvaluationRecord) -> None:
        """Añade el registro a la lista."""
        self.records.append(record)

    def __len__(self) -> int:
        """Evaluaciones registradas."""
        return len(self.records)


class JsonlJournal:
    """Una línea JSON por evaluación: apto para grep y para reproceso."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        """Archivo de destino."""
        return self._path

    def write(self, record: EvaluationRecord) -> None:
        """Añade una línea al archivo."""
        line = record.model_dump_json()
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def read_all(self) -> list[EvaluationRecord]:
        """Relee el archivo. Valida cada línea contra el esquema que la produjo.

        Una línea que no valida lanza `JournalError` con su número en vez de
        saltarse: el resto del archivo no sirve para contar si falta un trozo.
        """
        if not self._path.is_file():
            return []
        records: list[EvaluationRecord] = []
        lines = self._path.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                records.append(EvaluationRecord.model_validate(json.loads(line)))
            except (json.JSONDecodeError, ValidationError) as error:
                raise JournalError(self._path, number, error) from error
        return records


def build_record(
    state: TradingState,
    run_id: UUID,
    symbol: str,
    timeframe: str,
    at: datetime,
    order: OrderIntent | None = None,
) -> EvaluationRecord:
    """Consolida el estado final en un registro."""
    return EvaluationRecord(
        run_id=run_id,
        at=at,
        symbol=symbol,
        timeframe=timeframe,
        snapshot=state.snapshot,
        activation=state.activation,
        evidence=state.evidence,
        briefs=tuple(state.briefs),
        decision=state.decision,
        proposal=state.proposal,
        risk=state.risk,
        order=order,
        calls=tuple(state.calls),
        errors=tuple(state.errors),
    )
