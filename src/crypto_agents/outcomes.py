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

Hay tres formas de puntuar una posición, porque el resultado de un brazo mezcla dos
cosas: si acertó la dirección y dónde puso el stop. `Scoring.OWN_STOP` es la de
siempre, con el stop que declaró el decisor, y no se ha tocado. `COMMON_STOP` usa el
mismo stop para todos los brazos y `HORIZON_CLOSE` no usa ninguno: la salida es el
cierre del horizonte. Las dos últimas puntúan la *propuesta*, no la orden: un stop
declarado en el lado equivocado vetaba la orden y borraba la dirección de la medida.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import Field

from crypto_agents.state import Action, FrozenModel, stop_on_wrong_side
from crypto_agents.stops import atr_of, common_stop

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from crypto_agents.journal import EvaluationRecord

__all__ = [
    "Outcome",
    "OutcomeError",
    "OutcomeStats",
    "ScoredRun",
    "Scoring",
    "TradeOutcome",
    "UnscorableError",
    "resolved_returns",
    "score_outcomes",
    "score_position",
    "score_record",
    "score_run",
]

_TIMESTAMP, _OPEN, _HIGH, _LOW, _CLOSE = 0, 1, 2, 3, 4


class OutcomeError(ValueError):
    """Una orden que no se puede puntuar sin inventar el resultado.

    Hoy solo hay un caso: el stop en el lado equivocado. Un `buy` invalidado por
    encima de su entrada «sale» en la primera vela que toque ese precio, con
    ganancia, y la tabla lo contaría como acierto. El gate de riesgo veta esas
    propuestas y `OrderIntent` se niega a construirlas, así que llegar aquí con
    una es un bug del camino a la orden: se dice, no se puntúa.
    """


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

    Lanza `OutcomeError` si la orden tiene el stop del lado equivocado.
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
    if stop_on_wrong_side(order.side, invalidation, entry):
        raise OutcomeError(
            f"orden {record.run_id} con la invalidación en el lado equivocado: "
            f"{order.side.value} con entrada {entry} e invalidación {invalidation}"
        )

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
    records: Sequence[EvaluationRecord],
    histories: Mapping[str, Sequence[Sequence[float]]],
    horizon: int = 6,
) -> OutcomeStats:
    """Puntúa todas las órdenes de una corrida contra el histórico que las produjo.

    Las series entran por símbolo porque una corrida ya no recorre uno solo: la
    selección de la ablación salta entre siete. Con una única serie, la orden de
    ETH se puntuaría contra las velas de BTC y el resultado sería ruido con
    aspecto de medida.

    El horizonte por defecto son seis velas: un día entero en 4h. Es una elección
    del arnés y no del sistema, así que se declara en vez de esconderse.
    """
    scored = [
        outcome
        for record in records
        if (rows := histories.get(record.symbol)) is not None
        and (outcome := score_record(record, rows, horizon)) is not None
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


def resolved_returns(
    records: Sequence[EvaluationRecord],
    histories: Mapping[str, Sequence[Sequence[float]]],
    horizon: int = 6,
) -> tuple[float, ...]:
    """Retorno de cada orden resuelta, una a una, en el orden de los registros.

    `OutcomeStats` guarda el total y no los retornos, así que de ahí no sale una
    dispersión. Esta función repite el filtro de `score_outcomes` —puntúa con el
    mismo `score_record`, descarta lo que no se puede situar y lo no resuelto— en vez
    de reescribirla, y `tests/test_outcomes.py` ata las dos: mismo recuento, mismo
    total, mismos aciertos.
    """
    return tuple(
        outcome.gross_return
        for record in records
        if (rows := histories.get(record.symbol)) is not None
        and (outcome := score_record(record, rows, horizon)) is not None
        and outcome.outcome is not Outcome.UNRESOLVED
    )


# ─────────────────────────────────── Puntuación común ─────────────────────────────────────────────
# Lo que separa dirección de stop. Todo lo de arriba queda como estaba: `score_record`,
# `score_outcomes` y `resolved_returns` no se tocan, y `OWN_STOP` delega en el primero.


class UnscorableError(OutcomeError):
    """El registro no lleva lo que esta puntuación necesita.

    No es un bug del camino a la orden como `OutcomeError`: un journal escrito antes
    de que el registro guardara los indicadores no puede reconstruir el stop común, y
    eso se cuenta aparte en vez de tumbar la corrida o inventar el ATR.
    """


class Scoring(StrEnum):
    """Cómo se sale de una posición."""

    OWN_STOP = "own_stop"
    """Con el stop que declaró el decisor, sobre órdenes. La puntuación de siempre."""

    COMMON_STOP = "common_stop"
    """Con el stop común de `stops.py`, sobre propuestas accionables."""

    HORIZON_CLOSE = "horizon_close"
    """Sin stop: el cierre de la vela `i + horizonte`, sobre propuestas accionables."""


def _walk(
    side: Action,
    entry: float,
    stop: float | None,
    start: int,
    rows: Sequence[Sequence[float]],
    horizon: int,
) -> TradeOutcome:
    """Recorre las velas posteriores a `start`: sale por el stop, si hay, o al cierre del horizonte.

    Es el recorrido de `score_record` con el stop como parámetro. Lo ata a él
    `tests/test_outcomes.py`: con el stop declarado como stop, las dos dan lo mismo.
    El stop se comprueba contra el rango de la vela, no contra su cierre.
    """
    long = side is Action.BUY
    last = min(start + horizon, len(rows) - 1)
    if stop is not None:
        for offset in range(start + 1, last + 1):
            low, high = rows[offset][_LOW], rows[offset][_HIGH]
            touched = low <= stop if long else high >= stop
            if touched:
                return TradeOutcome(
                    at=start,
                    side=side,
                    outcome=Outcome.INVALIDATED,
                    entry_price=entry,
                    exit_price=stop,
                    bars_held=offset - start,
                )

    if last <= start or last - start < horizon:
        return TradeOutcome(
            at=start,
            side=side,
            outcome=Outcome.UNRESOLVED,
            entry_price=entry,
            exit_price=entry,
            bars_held=max(0, last - start),
        )
    return TradeOutcome(
        at=start,
        side=side,
        outcome=Outcome.HELD,
        entry_price=entry,
        exit_price=rows[last][_CLOSE],
        bars_held=last - start,
    )


def score_position(
    record: EvaluationRecord,
    rows: Sequence[Sequence[float]],
    horizon: int,
    scoring: Scoring,
) -> TradeOutcome | None:
    """Puntúa la posición de un registro, o `None` si no hay ninguna que puntuar.

    `OWN_STOP` es `score_record`: solo hay posición si hubo orden. Las otras dos
    puntúan la propuesta —acción buy o sell, entrada en `snapshot.close`— hubiera o no
    orden, y no leen el stop que declaró el decisor: ni la orden ni la propuesta lo
    cambian. El riesgo tampoco las filtra, que en la ablación solo vetaba stops del
    lado equivocado.

    Lanza `UnscorableError` si `COMMON_STOP` no puede reconstruir el stop.
    """
    if scoring is Scoring.OWN_STOP:
        return score_record(record, rows, horizon)

    proposed = record.proposed
    if proposed is None or proposed.action is Action.HOLD or record.snapshot is None:
        return None
    start = _index_of(rows, record.snapshot.timestamp.timestamp() * 1000.0)
    if start is None:
        return None

    side, entry = proposed.action, record.snapshot.close
    stop: float | None = None
    if scoring is Scoring.COMMON_STOP:
        if record.indicators is None:
            raise UnscorableError(
                f"el registro {record.run_id} no lleva indicadores: "
                "no se puede reconstruir el stop común"
            )
        try:
            stop = common_stop(side, entry, atr_of(record.indicators))
        except ValueError as error:
            raise UnscorableError(f"registro {record.run_id}: {error}") from error
    return _walk(side, entry, stop, start, rows, horizon)


class ScoredRun(FrozenModel):
    """Una corrida puntuada de una forma, con el retorno por posición y por evaluación."""

    scoring: Scoring

    positions: int = Field(ge=0)
    """Posiciones que se pudieron situar en el histórico, resueltas o no."""

    resolved: int = Field(ge=0)
    unscorable: int = Field(ge=0)
    """Registros a los que esta puntuación no pudo llegar, por faltarles los indicadores."""

    per_position: tuple[float, ...]
    """Retorno de cada posición resuelta, en el orden de los registros."""

    per_evaluation: tuple[float, ...]
    """Un valor por registro, en su orden: el retorno de su posición, o 0.0 si no hay.

    Lo que permite emparejar dos brazos por evaluación: mismo índice, misma vela.
    Una posición sin resolver y un registro sin posición valen lo mismo, cero.
    """


def score_run(
    records: Sequence[EvaluationRecord],
    histories: Mapping[str, Sequence[Sequence[float]]],
    horizon: int,
    scoring: Scoring,
) -> ScoredRun:
    """Puntúa una corrida con una de las tres formas.

    Un registro sin los indicadores que pide `COMMON_STOP` se cuenta como
    `unscorable` y la corrida sigue: leer un directorio viejo no debe tumbar la tabla.
    Cualquier otro `OutcomeError` —un stop propio del lado equivocado— sigue siendo
    un bug y sube.
    """
    per_position: list[float] = []
    per_evaluation: list[float] = []
    positions = unscorable = 0
    for record in records:
        rows = histories.get(record.symbol)
        outcome: TradeOutcome | None = None
        if rows is not None:
            try:
                outcome = score_position(record, rows, horizon, scoring)
            except UnscorableError:
                unscorable += 1
        if outcome is None:
            per_evaluation.append(0.0)
            continue
        positions += 1
        if outcome.outcome is Outcome.UNRESOLVED:
            per_evaluation.append(0.0)
            continue
        per_position.append(outcome.gross_return)
        per_evaluation.append(outcome.gross_return)

    return ScoredRun(
        scoring=scoring,
        positions=positions,
        resolved=len(per_position),
        unscorable=unscorable,
        per_position=tuple(per_position),
        per_evaluation=tuple(per_evaluation),
    )
