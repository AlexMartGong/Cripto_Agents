"""Agregaciones sobre las llamadas registradas.

Lee `LLMCall`, no llama a nadie. La pregunta que responde este módulo es si un
respaldo local está ahorrando algo: un modelo pequeño que necesita tres intentos
por veredicto consume tres veces la latencia y produce la misma línea de journal
que uno que acierta a la primera.

La tasa sola miente cuando el denominador es pequeño —un fallo de un intento es
100%—, así que la tasa se publica junto al recuento que la produce.
"""

from __future__ import annotations

import math
import statistics
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import Field

from crypto_agents.queries import abort_cause
from crypto_agents.quota import LOCAL_BACKENDS
from crypto_agents.state import (
    Action,
    AgentRole,
    Backend,
    FailureKind,
    FrozenModel,
    Side,
    stop_on_wrong_side,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from crypto_agents.journal import EvaluationRecord
    from crypto_agents.state import LLMCall

__all__ = [
    "NO_ERROR",
    "AbortKind",
    "AttemptCounts",
    "BackendStats",
    "Interval",
    "InvalidationStats",
    "LatencyStats",
    "QuotaSplit",
    "ResumeDelta",
    "ReturnStats",
    "RiskFlow",
    "RunSummary",
    "WeightedRate",
    "WorstPair",
    "attempt_counts",
    "backend_stats",
    "conviction_cross",
    "dismissal_cross",
    "failure_counts",
    "invalidation_stats",
    "live_latency",
    "nearest_rank",
    "quota_by_locality",
    "resume_delta",
    "return_stats",
    "risk_flow",
    "summarise",
    "undecided_causes",
    "validation_failure",
    "validation_failure_rate",
    "wilson_interval",
    "worst_pair",
]


class BackendStats(FrozenModel):
    """Intentos de un par (rol, backend), separados por dónde se rompieron.

    Transporte y validación se cuentan aparte porque responden a preguntas
    distintas: uno mide si el proveedor está sirviendo ese id, el otro si el
    modelo sabe producir el esquema. Sumarlos en una sola columna es lo que hace
    ilegible una tabla de comparación entre modelos — un id mal escrito y un
    modelo que alucina campos salen con la misma cifra.
    """

    attempts: int = Field(ge=0)
    """Intentos que llegaron a un proveedor. Los aciertos de caché no cuentan."""

    invalid: int = Field(ge=0)
    """Produjeron contenido y no pasó la validación, de esquema o de contexto."""

    context: int = Field(default=0, ge=0)
    """De los inválidos, los que pasaron el esquema y contradecían lo que tenían delante."""

    transport: int = Field(default=0, ge=0)
    """No llegaron a producir contenido: red, autenticación, 4xx, 5xx."""

    timeout: int = Field(default=0, ge=0)
    """No contestaron dentro del plazo. Tampoco produjeron contenido."""

    @property
    def answered(self) -> int:
        """Intentos que sí devolvieron algo que validar."""
        return self.attempts - self.transport - self.timeout

    @property
    def failure_rate(self) -> float:
        """Fracción de lo respondido que no pasó la validación. Sin respuestas, cero.

        El denominador excluye transporte y timeout: a un intento que nunca llegó al
        modelo no se le puede reprochar no haber producido el esquema.
        """
        if self.answered == 0:
            return 0.0
        return self.invalid / self.answered


def backend_stats(calls: Iterable[LLMCall]) -> dict[tuple[AgentRole, Backend], BackendStats]:
    """Intentos, fallos de transporte y fallos de validación por rol y backend.

    Los aciertos de caché quedan fuera: no llegaron a ningún proveedor, así que no
    dicen nada sobre si el modelo sabe producir el esquema.
    """
    attempts: dict[tuple[AgentRole, Backend], int] = {}
    invalid: dict[tuple[AgentRole, Backend], int] = {}
    transport: dict[tuple[AgentRole, Backend], int] = {}
    timeout: dict[tuple[AgentRole, Backend], int] = {}
    context: dict[tuple[AgentRole, Backend], int] = {}
    for call in calls:
        if call.cache_hit:
            continue
        key = (call.role, call.backend)
        attempts[key] = attempts.get(key, 0) + 1
        if call.failure_kind is None:
            continue
        counter = {
            FailureKind.TRANSPORT: transport,
            FailureKind.TIMEOUT: timeout,
            FailureKind.SCHEMA: invalid,
            FailureKind.CONTEXT: invalid,
        }[call.failure_kind]
        counter[key] = counter.get(key, 0) + 1
        if call.failure_kind is FailureKind.CONTEXT:
            context[key] = context.get(key, 0) + 1
    return {
        key: BackendStats(
            attempts=total,
            invalid=invalid.get(key, 0),
            context=context.get(key, 0),
            transport=transport.get(key, 0),
            timeout=timeout.get(key, 0),
        )
        for key, total in sorted(attempts.items())
    }


def validation_failure_rate(calls: Iterable[LLMCall]) -> dict[tuple[AgentRole, Backend], float]:
    """Tasa de fallo de validación por rol y backend."""
    return {key: stats.failure_rate for key, stats in backend_stats(calls).items()}


class RunSummary(FrozenModel):
    """Qué ocurrió en una corrida completa, contado por etapas.

    El embudo importa más que los totales: una corrida que activó el gate 200
    veces y llegó al decisor 12 no tiene el mismo problema que una que lo activó 12
    y decidió las 12.
    """

    evaluations: int = Field(ge=0)
    activated: int = Field(ge=0)
    """Evaluaciones en las que el gate encontró al menos un disparador."""

    consolidated: int = Field(ge=0)
    """Llegaron a tener evidencia técnica completa: los tres veredictos."""

    decided: int = Field(ge=0)
    """El decisor respondió, con mesas o sin ellas.

    Cuenta sobre `EvaluationRecord.proposed`, no sobre `decision`: las variantes
    sin debate registran su salida en `proposal`, y mirar solo `decision` haría que
    la ablación informara de cero decisiones justo en los brazos que existen para
    compararse con el completo.
    """

    traded: int = Field(ge=0)
    actions: dict[Action, int] = Field(default_factory=dict)
    vetoes: dict[str, int] = Field(default_factory=dict)
    """Vetos del gate de riesgo agrupados por `veto_rule`, no por su frase legible."""

    applied_limits: dict[str, int] = Field(default_factory=dict)
    """Recortes que sí dejaron pasar la orden, más pequeña."""

    quota_by_role: dict[AgentRole, float] = Field(default_factory=dict)
    quota_by_backend: dict[Backend, float] = Field(default_factory=dict)
    calls: dict[tuple[AgentRole, Backend], BackendStats] = Field(default_factory=dict)

    @property
    def quota_used(self) -> float:
        """Cuota total de la corrida. Los aciertos de caché no cuentan."""
        return sum(self.quota_by_role.values())


def summarise(records: Sequence[EvaluationRecord]) -> RunSummary:
    """Consolida una corrida de evaluaciones en un solo objeto contable."""
    actions: dict[Action, int] = {}
    vetoes: dict[str, int] = {}
    limits: dict[str, int] = {}
    by_role: dict[AgentRole, float] = {}
    by_backend: dict[Backend, float] = {}

    for record in records:
        proposed = record.proposed
        if proposed is not None:
            actions[proposed.action] = actions.get(proposed.action, 0) + 1
        if record.risk is not None:
            if record.risk.veto_rule is not None:
                vetoes[record.risk.veto_rule] = vetoes.get(record.risk.veto_rule, 0) + 1
            for label in record.risk.applied_limits:
                limits[label] = limits.get(label, 0) + 1
        for call in record.calls:
            if call.cache_hit:
                continue
            by_role[call.role] = by_role.get(call.role, 0.0) + call.quota_weight
            by_backend[call.backend] = by_backend.get(call.backend, 0.0) + call.quota_weight

    return RunSummary(
        evaluations=len(records),
        activated=sum(
            1
            for record in records
            if record.activation is not None and record.activation.should_run
        ),
        consolidated=sum(1 for record in records if record.evidence is not None),
        decided=sum(1 for record in records if record.proposed is not None),
        traded=sum(1 for record in records if record.traded),
        actions=dict(sorted(actions.items())),
        vetoes=dict(sorted(vetoes.items())),
        applied_limits=dict(sorted(limits.items())),
        quota_by_role=dict(sorted(by_role.items())),
        quota_by_backend=dict(sorted(by_backend.items())),
        calls=backend_stats(call for record in records for call in record.calls),
    )


# ────────────────────────────────── Auditoría de una corrida ──────────────────────────────────────
# Lo que la primera tabla de la ablación no pudo contestar. Todo son funciones puras
# sobre `EvaluationRecord`: la misma cifra sale de un journal en caliente y del
# archivo releído tres días después, que es cuando alguien la pide.
#
# Tres reglas comunes. Una tasa nunca sale sin su denominador. Lo que no se puede
# calcular es `None`, no cero: un cero es una medida. Y nada de esto importa el
# router, así que auditar una corrida no puede costar una llamada.


def _calls(records: Iterable[EvaluationRecord]) -> list[LLMCall]:
    """Todos los intentos de la corrida, en el orden del journal."""
    return [call for record in records for call in record.calls]


def nearest_rank(values: Sequence[float], percent: int) -> float:
    """Percentil por rango más cercano: el valor en la posición `ceil(percent * n / 100)`.

    Sin interpolar, para que la cifra sea siempre un valor que ocurrió y se pueda
    comprobar a mano contra la lista ordenada. El rango se calcula con enteros:
    `7 / 100 * 100` es 7.000000000000001 en coma flotante y su techo sería 8.
    """
    if not values:
        raise ValueError("no hay percentil de una lista vacía")
    if not 0 < percent <= 100:
        raise ValueError("el percentil va de 1 a 100")
    ordered = sorted(values)
    rank = max(1, -(-percent * len(ordered) // 100))
    return ordered[rank - 1]


# ── 1. Intentos ───────────────────────────────────────────────────────────────


class AttemptCounts(FrozenModel):
    """Filas `LLMCall` de una corrida, contadas por cada eje."""

    total: int = Field(ge=0)
    """Todas las filas, aciertos de caché incluidos."""

    by_role: dict[AgentRole, int] = Field(default_factory=dict)
    by_backend: dict[Backend, int] = Field(default_factory=dict)
    cache_hits: int = Field(ge=0)
    live: int = Field(ge=0)
    """Llegaron a un proveedor. `total - cache_hits`."""

    valid: int = Field(ge=0)
    invalid: int = Field(ge=0)
    cached_invalid: int = Field(ge=0)
    """Intentos inválidos reproducidos desde la caché.

    La caché guarda cada intento, también el que no validó, y al releerlo da el
    mismo error. En una pasada que paga todo esto es cero; en una reanudación o un
    replay es el número de intentos fallidos que no hubo que volver a pagar. Nunca
    es una respuesta servida: un intento inválido sigue siendo inválido.
    """


def attempt_counts(records: Iterable[EvaluationRecord]) -> AttemptCounts:
    """Intentos totales, por rol, por backend, por acierto de caché y por validez."""
    calls = _calls(records)
    by_role: dict[AgentRole, int] = {}
    by_backend: dict[Backend, int] = {}
    for call in calls:
        by_role[call.role] = by_role.get(call.role, 0) + 1
        by_backend[call.backend] = by_backend.get(call.backend, 0) + 1
    hits = sum(1 for call in calls if call.cache_hit)
    valid = sum(1 for call in calls if call.valid)
    return AttemptCounts(
        total=len(calls),
        by_role=by_role,
        by_backend=by_backend,
        cache_hits=hits,
        live=len(calls) - hits,
        valid=valid,
        invalid=len(calls) - valid,
        cached_invalid=sum(1 for call in calls if call.cache_hit and not call.valid),
    )


def failure_counts(records: Iterable[EvaluationRecord]) -> dict[FailureKind, int]:
    """Intentos fallidos por tipo. Los cuatro tipos aparecen siempre, aunque sea con cero.

    `schema` y `context` miden al modelo; `timeout` y `transport`, al proveedor.
    Sumados en una cifra no dicen de quién es el problema.
    """
    counts = dict.fromkeys(FailureKind, 0)
    for call in _calls(records):
        if call.failure_kind is not None:
            counts[call.failure_kind] += 1
    return counts


class WeightedRate(FrozenModel):
    """Una tasa con los dos números que la producen."""

    numerator: int = Field(ge=0)
    denominator: int = Field(ge=0)

    @property
    def rate(self) -> float | None:
        """La fracción, o `None` sin denominador: 0 de 0 no es 0%."""
        if self.denominator == 0:
            return None
        return self.numerator / self.denominator


def validation_failure(records: Iterable[EvaluationRecord]) -> WeightedRate:
    """Fallo de validación de la corrida: inválidas sobre respondidas, sumadas.

    Es la media de los pares (rol, backend) ponderada por sus intentos, que es lo
    mismo que dividir los totales. La media simple de las tasas deja que un par con
    dos intentos pese igual que uno con doscientos, y el máximo —lo que publicaba
    la primera tabla— es la tasa de un par presentada como la del brazo.

    El transporte queda fuera del denominador, igual que en `BackendStats`.
    """
    stats = backend_stats(_calls(records)).values()
    return WeightedRate(
        numerator=sum(item.invalid for item in stats),
        denominator=sum(item.answered for item in stats),
    )


class WorstPair(FrozenModel):
    """El par (rol, backend) con peor tasa de validación, con sus recuentos."""

    role: AgentRole
    backend: Backend
    stats: BackendStats


def worst_pair(records: Iterable[EvaluationRecord]) -> WorstPair | None:
    """El peor par, nombrado como lo que es.

    Solo compiten los pares que respondieron algo. A igual tasa gana el de más
    respuestas, que es el que la sostiene con más evidencia.
    """
    answered = [
        (key, stats) for key, stats in backend_stats(_calls(records)).items() if stats.answered > 0
    ]
    if not answered:
        return None
    (role, backend), stats = max(
        answered, key=lambda item: (item[1].failure_rate, item[1].answered)
    )
    return WorstPair(role=role, backend=backend, stats=stats)


# ── 2. Latencia ───────────────────────────────────────────────────────────────


class LatencyStats(FrozenModel):
    """Latencia de las llamadas que llegaron a un proveedor."""

    n: int = Field(gt=0)
    mean_ms: float = Field(ge=0.0)
    median_ms: float = Field(ge=0.0)
    p95_ms: float = Field(ge=0.0)


def live_latency(records: Iterable[EvaluationRecord]) -> LatencyStats | None:
    """Latencia sobre llamadas vivas. `None` si la corrida salió entera de la caché.

    Un acierto de caché registra 0 ms: con ellos dentro, la media mide cuánta
    caché había y no lo que tarda el modelo. Entran también los rechazos de
    transporte, que son llamadas vivas con su tiempo hasta el rechazo.
    """
    latencies = [call.latency_ms for call in _calls(records) if not call.cache_hit]
    if not latencies:
        return None
    return LatencyStats(
        n=len(latencies),
        mean_ms=statistics.fmean(latencies),
        median_ms=statistics.median(latencies),
        p95_ms=nearest_rank(latencies, 95),
    )


# ── 3. Cuota ──────────────────────────────────────────────────────────────────


class QuotaSplit(FrozenModel):
    """Cuota consumida, separada por quién la cuenta."""

    remote: float = Field(ge=0.0)
    """La que resta de la ventana de un proveedor."""

    local: float = Field(ge=0.0)
    """La servida en esta máquina. No le cuesta ventana a nadie."""


def quota_by_locality(records: Iterable[EvaluationRecord]) -> QuotaSplit:
    """Cuota remota y local. Los aciertos de caché no consumen."""
    remote = 0.0
    local = 0.0
    for call in _calls(records):
        if call.cache_hit:
            continue
        if call.backend in LOCAL_BACKENDS:
            local += call.quota_weight
        else:
            remote += call.quota_weight
    return QuotaSplit(remote=remote, local=local)


# ── 4. Evaluaciones sin decisión ──────────────────────────────────────────────


class AbortKind(StrEnum):
    """Por qué una evaluación terminó sin decisión, en categorías cerradas.

    `NodeError` solo trae nodo y mensaje, y el mensaje lleva ids y errores de
    validación dentro: agrupar por él haría de cada aborto una categoría propia.
    """

    VALIDATION = "validation"
    """El modelo agotó los intentos sin una salida válida, por esquema o por contexto.

    Cuál de los dos lo dice `failure_kind` en cada intento, no esta categoría.
    """

    TRANSPORT = "transport"
    """El proveedor no llegó a producir contenido."""

    TIMEOUT = "timeout"
    """El proveedor no contestó dentro del plazo."""

    QUOTA = "quota"
    """Ningún modelo del rol cabía en la ventana."""

    CONTEXT_BYPASSED = "context_bypassed"
    """Un nodo recibió del router una salida que su validación de contexto rechaza.

    Es un bug, no un fallo del modelo: esas salidas las rechaza el router, con
    reintento, y se cuentan como intentos de tipo `context`. Cualquier recuento
    aquí dice que una llamada perdió su validación.
    """

    MISSING_UPSTREAM = "missing_upstream"
    """El nodo no tenía con qué trabajar. Como primera causa, es un hueco del grafo."""

    GATE_CLOSED = "gate_closed"
    """No es un aborto: el gate de activación no disparó."""

    OTHER = "other"
    """Sin clasificar. Cualquier recuento aquí pide leer los mensajes."""


NO_ERROR = "sin error"
"""Nodo de las evaluaciones sin decisión que no registraron ningún fallo."""

_ABORT_MARKERS: tuple[tuple[str, AbortKind], ...] = (
    ("cuota agotada", AbortKind.QUOTA),
    ("sin salida válida", AbortKind.VALIDATION),
    ("falló antes de producir", AbortKind.TRANSPORT),
    ("validación de contexto saltada", AbortKind.CONTEXT_BYPASSED),
    ("faltan veredictos técnicos", AbortKind.MISSING_UPSTREAM),
    ("faltan alegatos", AbortKind.MISSING_UPSTREAM),
    ("no hay evidencia consolidada", AbortKind.MISSING_UPSTREAM),
    ("falta la preparación determinista", AbortKind.MISSING_UPSTREAM),
    ("no hay decisión que evaluar", AbortKind.MISSING_UPSTREAM),
)
"""Fragmento del mensaje que identifica cada causa.

Se lee el texto porque el contrato no ofrece otra cosa: `NodeError` no lleva un
tipo de fallo. Los fragmentos son los que escriben `QuotaExhaustedError`, las dos
excepciones del router y los nodos; `tests/test_metrics.py` construye cada uno con
el código real, así que cambiar una redacción rompe una prueba en vez de mandar la
causa a `other` en silencio.
"""


def _abort_kind(record: EvaluationRecord) -> AbortKind:
    """Clase del primer fallo de una evaluación sin decisión."""
    if not record.errors:
        gate = record.activation
        if gate is not None and not gate.should_run:
            return AbortKind.GATE_CLOSED
        return AbortKind.OTHER
    message = record.errors[0].message
    kind = next((kind for marker, kind in _ABORT_MARKERS if marker in message), AbortKind.OTHER)
    if kind is AbortKind.TRANSPORT:
        # El mensaje del router es el mismo para un rechazo y para un plazo
        # vencido; lo que los distingue es el tipo del intento que falló.
        failed = {call.failure_kind for call in record.calls}
        if FailureKind.TIMEOUT in failed and FailureKind.TRANSPORT not in failed:
            return AbortKind.TIMEOUT
    return kind


def undecided_causes(records: Iterable[EvaluationRecord]) -> dict[tuple[str, AbortKind], int]:
    """Evaluaciones sin decisión, agrupadas por el nodo que falló y por qué.

    Manda el primer fallo, igual que en `queries.abort_cause`: los nodos de
    después fallan por consecuencia, y agruparlos culparía al síntoma.
    """
    grouped: dict[tuple[str, AbortKind], int] = {}
    for record in records:
        if record.proposed is not None:
            continue
        node = abort_cause(record) if record.errors else NO_ERROR
        key = (node, _abort_kind(record))
        grouped[key] = grouped.get(key, 0) + 1
    return dict(sorted(grouped.items()))


# ── 5. Del decisor a la orden ─────────────────────────────────────────────────


class RiskFlow(FrozenModel):
    """Qué pasó con cada propuesta entre el decisor y el mercado."""

    actionable: int = Field(ge=0)
    """Propuestas `buy` o `sell`."""

    holds: int = Field(ge=0)
    orders: int = Field(ge=0)
    vetoes: dict[str, int] = Field(default_factory=dict)
    """Vetos del gate de riesgo por `veto_rule`."""

    unexecuted: int = Field(ge=0)
    """Accionables aprobadas por el gate que no acabaron en orden."""


def risk_flow(records: Iterable[EvaluationRecord]) -> RiskFlow:
    """Propuestas accionables, órdenes, vetos por regla y órdenes que no salieron.

    `actionable == orders` solo dice algo junto a las otras dos columnas: sin
    vetos y sin fallos del ejecutor es una identidad, no una coincidencia.
    """
    actionable = holds = orders = unexecuted = 0
    vetoes: dict[str, int] = {}
    for record in records:
        proposed = record.proposed
        if proposed is None:
            continue
        if proposed.action is Action.HOLD:
            holds += 1
            continue
        actionable += 1
        if record.traded:
            orders += 1
        elif record.risk is not None and record.risk.veto_rule is not None:
            vetoes[record.risk.veto_rule] = vetoes.get(record.risk.veto_rule, 0) + 1
        else:
            unexecuted += 1
    return RiskFlow(
        actionable=actionable,
        holds=holds,
        orders=orders,
        vetoes=dict(sorted(vetoes.items())),
        unexecuted=unexecuted,
    )


# ── 6. Invalidación ───────────────────────────────────────────────────────────


class InvalidationStats(FrozenModel):
    """Dónde pusieron los decisores el precio de invalidación respecto al cierre."""

    n: int = Field(ge=0)
    """Propuestas accionables con cierre conocido."""

    wrong_side: int = Field(ge=0)
    """Invalidación en el lado que no invalida: `>=` cierre en un `buy`, `<=` en un `sell`.

    La igualdad cuenta: un stop en el propio cierre no deja recorrido. Es la
    misma definición que usa el veto `invalid_stop_side`, así que este recuento
    sobre las propuestas coincide con los vetos de esa regla.
    """

    p10: float | None = None
    median: float | None = None
    p90: float | None = None
    """`|invalidación - cierre| / cierre`. `None` sin propuestas accionables."""


def invalidation_stats(records: Iterable[EvaluationRecord]) -> InvalidationStats:
    """Lado y distancia de la invalidación de cada propuesta accionable."""
    wrong = 0
    distances: list[float] = []
    for record in records:
        proposed = record.proposed
        if proposed is None or proposed.action is Action.HOLD or record.snapshot is None:
            continue
        invalidation = proposed.invalidation_price
        if invalidation is None:
            continue
        close = record.snapshot.close
        wrong += stop_on_wrong_side(proposed.action, invalidation, close)
        distances.append(abs(invalidation - close) / close)

    if not distances:
        return InvalidationStats(n=0, wrong_side=0)
    return InvalidationStats(
        n=len(distances),
        wrong_side=wrong,
        p10=nearest_rank(distances, 10),
        median=statistics.median(distances),
        p90=nearest_rank(distances, 90),
    )


# ── 7. Decisión contra mesas ──────────────────────────────────────────────────


def conviction_cross(records: Iterable[EvaluationRecord]) -> dict[tuple[Action, int], int]:
    """Acción por signo de `bull.conviction - bear.conviction`: -1, 0 o 1.

    Solo evaluaciones con `Decision` y las dos mesas: con una sola no hay
    diferencia que tomar. Conteos, sin leer nada en ellos.
    """
    grouped: dict[tuple[Action, int], int] = {}
    for record in records:
        if record.decision is None:
            continue
        by_side = {brief.side: brief.conviction for brief in record.briefs}
        if Side.BULL not in by_side or Side.BEAR not in by_side:
            continue
        gap = by_side[Side.BULL] - by_side[Side.BEAR]
        key = (record.decision.action, (gap > 0) - (gap < 0))
        grouped[key] = grouped.get(key, 0) + 1
    return dict(sorted(grouped.items()))


def dismissal_cross(
    records: Iterable[EvaluationRecord],
) -> dict[tuple[Action, Side | None], int]:
    """Acción por mesa descartada. `None` es un `hold` que no descartó a ninguna."""
    grouped: dict[tuple[Action, Side | None], int] = {}
    for record in records:
        if record.decision is None:
            continue
        key = (record.decision.action, record.decision.dismissed_side)
        grouped[key] = grouped.get(key, 0) + 1
    return dict(sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1] or "")))


# ── 8. Incertidumbre ──────────────────────────────────────────────────────────
# Con 20 o 140 órdenes una tasa y una media son estimaciones, no medidas. Estas dos
# funciones ponen al lado de cada una lo que se puede decir de su error: sin ellas la
# tabla de la ablación presentaría 55% y 52% como si fueran dos resultados distintos.


class Interval(FrozenModel):
    """Un intervalo cerrado dentro de [0, 1]."""

    low: float = Field(ge=0.0, le=1.0)
    high: float = Field(ge=0.0, le=1.0)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> Interval | None:
    """Intervalo de Wilson para una proporción; `z = 1.96` es el 95%. `None` sin observaciones.

    Wilson y no el normal: con 0 de 10 o 10 de 10 —y con n pequeña, que es lo que hay
    aquí— el intervalo normal se sale de [0, 1] o colapsa a un punto, mientras que
    el de Wilson queda dentro sin recortar a ciegas y es más ancho exactamente donde
    hay menos datos. Sin corrección de continuidad: es la forma estándar y la que
    cualquiera reproduce a mano.

    Más aciertos que observaciones es un bug de quien cuenta y se dice, no se acota.
    """
    if not 0 <= successes <= n:
        raise ValueError(f"aciertos fuera de rango: {successes} de {n}")
    if n == 0:
        return None
    proportion = successes / n
    scale = 1.0 + z * z / n
    centre = (proportion + z * z / (2 * n)) / scale
    half = z * math.sqrt(proportion * (1.0 - proportion) / n + z * z / (4 * n * n)) / scale
    return Interval(low=max(0.0, centre - half), high=min(1.0, centre + half))


class ReturnStats(FrozenModel):
    """Retorno medio de las órdenes resueltas, con su error estándar y el tamaño de la muestra."""

    n: int = Field(gt=0)
    mean: float
    stderr: float | None = None
    """Error estándar de la media. `None` con una sola orden: sin dispersión que estimar."""


def return_stats(returns: Sequence[float]) -> ReturnStats | None:
    """Media y error estándar de la media. `None` sin retornos.

    La desviación es la muestral (entre n - 1): cuatro órdenes no conocen la varianza
    de la que salen, y dividir entre n presumiría más precisión de la que hay. Con
    una orden el error es `None` y no cero: cero diría que la media es exacta.
    """
    if not returns:
        return None
    n = len(returns)
    stderr = None if n == 1 else statistics.stdev(returns) / math.sqrt(n)
    return ReturnStats(n=n, mean=statistics.fmean(returns), stderr=stderr)


# ── Reanudación ───────────────────────────────────────────────────────────────


class ResumeDelta(FrozenModel):
    """Lo que una reanudación cambió respecto a la pasada anterior."""

    undecided_before: int = Field(ge=0)
    """Evaluaciones de la pasada anterior que terminaron sin decisión."""

    rescued: int = Field(ge=0)
    """De esas, las que esta pasada sí decidió."""


def resume_delta(
    previous: Iterable[EvaluationRecord], current: Iterable[EvaluationRecord]
) -> ResumeDelta:
    """Cuántas evaluaciones que antes no decidieron deciden ahora.

    Un intento inválido no se cachea, así que al reanudar se vuelve a llamar en
    vivo justo a las evaluaciones que abortaron: tienen una segunda tirada que las
    de una pasada única no tuvieron. Se emparejan por `run_id`, que en un replay es
    un UUID5 de símbolo, timeframe e instante.
    """
    undecided = {record.run_id for record in previous if record.proposed is None}
    decided_now = {record.run_id for record in current if record.proposed is not None}
    return ResumeDelta(undecided_before=len(undecided), rescued=len(undecided & decided_now))
