"""Qué pasó después de cada orden.

Camina las velas posteriores a la orden y decide si la tesis se rompió antes de
llegar al horizonte. Usa el `invalidation_price` que declaró el propio decisor:
puntuar con un stop inventado por el arnés mediría el stop del arnés, no el
criterio del sistema.

Trabaja sobre filas crudas de OHLCV, no sobre un DataFrame, y por dos razones. La
primera es que no hace falta: es aritmética de índices. La segunda es la regla de
capas del paquete —pandas solo vive en `market`, `indicators` y `activation`— que
un import aquí rompería sin ganar nada.

La invalidación se comprueba contra el rango de la vela (`low`/`high`), no contra
su cierre: un stop se toca intradía, y puntuar solo por cierres daría al sistema
un crédito que el mercado no le habría dado.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import Field

from crypto_agents.state import Action, FrozenModel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from crypto_agents.journal import EvaluationRecord

__all__ = [
    "Outcome",
    "OutcomeStats",
    "TradeOutcome",
    "score_outcomes",
    "score_record",
]

_TIMESTAMP, _OPEN, _HIGH, _LOW, _CLOSE = 0, 1, 2, 3, 4


class Outcome(StrEnum):
    """Cómo terminó una orden."""

    INVALIDATED = "invalidated"
    """El precio tocó el nivel donde el decisor dijo que su tesis se rompía."""

    HELD = "held"
    """Llegó al horizonte sin romperse. El retorno es el que dé el cierre."""

    UNRESOLVED = "unresolved"
    """El histórico se acabó antes del horizonte. No cuenta para nada."""


class TradeOutcome(FrozenModel):
    """Resultado de una orden concreta."""

    at: int
    """Índice de la vela en la que se emitió la orden."""

    side: Action
    outcome: Outcome
    entry_price: float = Field(gt=0.0)
    exit_price: float = Field(gt=0.0)
    bars_held: int = Field(ge=0)

    @property
    def gross_return(self) -> float:
        """Retorno de la posición, con signo según el lado. Sin comisiones ni slippage."""
        change = (self.exit_price - self.entry_price) / self.entry_price
        return change if self.side is Action.BUY else -change


class OutcomeStats(FrozenModel):
    """Resumen de las órdenes de una corrida.

    Con pocas órdenes estos números tienen un error de muestreo enorme, así que
    `resolved` se publica al lado: una tasa de acierto sobre tres operaciones no
    es una tasa de acierto.
    """

    orders: int = Field(ge=0)
    resolved: int = Field(ge=0)
    invalidated: int = Field(ge=0)
    wins: int = Field(ge=0)
    total_return: float
    unresolved: int = Field(ge=0)

    @property
    def win_rate(self) -> float | None:
        """Fracción de órdenes resueltas con retorno positivo. `None` sin resueltas."""
        if self.resolved == 0:
            return None
        return self.wins / self.resolved

    @property
    def mean_return(self) -> float | None:
        """Retorno medio por orden resuelta. `None` sin resueltas."""
        if self.resolved == 0:
            return None
        return self.total_return / self.resolved


def _index_of(rows: Sequence[Sequence[float]], timestamp_ms: float) -> int | None:
    """Posición de la vela con ese timestamp, o `None` si no está en el histórico."""
    for index, row in enumerate(rows):
        if row[_TIMESTAMP] == timestamp_ms:
            return index
    return None


def score_record(
    record: EvaluationRecord, rows: Sequence[Sequence[float]], horizon: int
) -> TradeOutcome | None:
    """Puntúa la orden de un registro, o `None` si no operó o no se puede situar.

    La entrada es el cierre de la vela evaluada, que es el `reference_price` con el
    que se construyó la orden: suponer la apertura siguiente sería más realista pero
    dejaría de comparar contra el precio que el sistema dijo estar mirando.
    """
    order = record.order
    if order is None or record.snapshot is None:
        return None

    start = _index_of(rows, record.snapshot.timestamp.timestamp() * 1000.0)
    if start is None:
        return None

    entry = order.reference_price
    invalidation = order.invalidation_price
    long = order.side is Action.BUY

    last = min(start + horizon, len(rows) - 1)
    for offset in range(start + 1, last + 1):
        low, high = rows[offset][_LOW], rows[offset][_HIGH]
        touched = low <= invalidation if long else high >= invalidation
        if touched:
            return TradeOutcome(
                at=start,
                side=order.side,
                outcome=Outcome.INVALIDATED,
                entry_price=entry,
                exit_price=invalidation,
                bars_held=offset - start,
            )

    if last <= start or last - start < horizon:
        return TradeOutcome(
            at=start,
            side=order.side,
            outcome=Outcome.UNRESOLVED,
            entry_price=entry,
            exit_price=entry,
            bars_held=max(0, last - start),
        )

    return TradeOutcome(
        at=start,
        side=order.side,
        outcome=Outcome.HELD,
        entry_price=entry,
        exit_price=rows[last][_CLOSE],
        bars_held=last - start,
    )


def score_outcomes(
    records: Sequence[EvaluationRecord], rows: Sequence[Sequence[float]], horizon: int = 6
) -> OutcomeStats:
    """Puntúa todas las órdenes de una corrida contra el histórico que las produjo.

    El horizonte por defecto son seis velas: un día entero en 4h. Es una elección
    del arnés y no del sistema, así que se declara en vez de esconderse.
    """
    scored = [
        outcome
        for record in records
        if (outcome := score_record(record, rows, horizon)) is not None
    ]
    resolved = [item for item in scored if item.outcome is not Outcome.UNRESOLVED]
    return OutcomeStats(
        orders=len(scored),
        resolved=len(resolved),
        invalidated=sum(1 for item in resolved if item.outcome is Outcome.INVALIDATED),
        wins=sum(1 for item in resolved if item.gross_return > 0.0),
        total_return=sum(item.gross_return for item in resolved),
        unresolved=len(scored) - len(resolved),
    )
