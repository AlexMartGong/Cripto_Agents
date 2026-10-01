"""Agregaciones sobre las llamadas registradas.

Lee `LLMCall`, no llama a nadie. La pregunta que responde este módulo es si un
respaldo local está ahorrando algo: un modelo pequeño que necesita tres intentos
por veredicto consume tres veces la latencia y produce la misma línea de journal
que uno que acierta a la primera.

La tasa sola miente cuando el denominador es pequeño —un fallo de un intento es
100%—, así que la tasa se publica junto al recuento que la produce.
"""

from __future__ import annotations

import statistics
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import Field

from crypto_agents.queries import abort_cause
from crypto_agents.quota import LOCAL_BACKENDS
from crypto_agents.state import Action, AgentRole, Backend, FailureKind, FrozenModel, Side

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from crypto_agents.journal import EvaluationRecord
    from crypto_agents.state import LLMCall

__all__ = [
    "NO_ERROR",
    "AbortKind",
    "AttemptCounts",
    "BackendStats",
    "InvalidationStats",
    "LatencyStats",
    "QuotaSplit",
    "ResumeDelta",
    "RiskFlow",
    "RunSummary",
    "WeightedRate",
    "WorstPair",
    "attempt_counts",
    "backend_stats",
    "conviction_cross",
    "dismissal_cross",
    "invalidation_stats",
    "live_latency",
    "nearest_rank",
    "quota_by_locality",
    "resume_delta",
    "risk_flow",
    "summarise",
    "undecided_causes",
    "validation_failure",
    "validation_failure_rate",
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
    """Produjeron contenido y no pasó el esquema."""

    transport: int = Field(default=0, ge=0)
    """No llegaron a producir contenido: red, autenticación, 4xx, 5xx."""

    @property
    def answered(self) -> int:
        """Intentos que sí devolvieron algo que validar."""
        return self.attempts - self.transport

    @property
    def failure_rate(self) -> float:
        """Fracción de lo respondido que no pasó la validación. Sin respuestas, cero.

        El denominador excluye el transporte: a un intento que nunca llegó al
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
    for call in calls:
        if call.cache_hit:
            continue
        key = (call.role, call.backend)
        attempts[key] = attempts.get(key, 0) + 1
        if call.failure is None:
            continue
        counter = transport if call.failure.kind is FailureKind.TRANSPORT else invalid
        counter[key] = counter.get(key, 0) + 1
    return {
        key: BackendStats(
            attempts=total, invalid=invalid.get(key, 0), transport=transport.get(key, 0)
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
    """Aciertos de caché marcados inválidos.

    El router revalida cada entrada al leerla y descarta la que no pasa, así que
    aquí debería haber cero. Se cuenta aparte porque cualquier otro valor dice que
    la caché sirvió algo que no cumple el esquema.
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
    """El modelo agotó los intentos sin una salida que pase el esquema."""

    TRANSPORT = "transport"
    """El proveedor no llegó a producir contenido."""

    QUOTA = "quota"
    """Ningún modelo del rol cabía en la ventana."""

    UNGROUNDED = "ungrounded"
    """Una mesa citó ids de observación que ningún veredicto emitió."""

    WRONG_SIDE = "wrong_side"
    """Una mesa respondió como la contraria."""

    WRONG_DIMENSION = "wrong_dimension"
    """Un agente técnico respondió como otra dimensión."""

    UNKNOWN_INDICATOR = "unknown_indicator"
    """Un veredicto citó un indicador que no existe."""

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
    ("cita ids inexistentes", AbortKind.UNGROUNDED),
    ("la mesa respondió como", AbortKind.WRONG_SIDE),
    ("el agente respondió como", AbortKind.WRONG_DIMENSION),
    ("indicadores citados que no existen", AbortKind.UNKNOWN_INDICATOR),
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
    return next((kind for marker, kind in _ABORT_MARKERS if marker in message), AbortKind.OTHER)


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

    La igualdad cuenta: un stop en el propio cierre no deja recorrido.
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
        if proposed.action is Action.BUY:
            wrong += invalidation >= close
        else:
            wrong += invalidation <= close
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
