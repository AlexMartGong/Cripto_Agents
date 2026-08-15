"""Agregaciones sobre las llamadas registradas.

Lee `LLMCall`, no llama a nadie. La pregunta que responde este módulo es si un
respaldo local está ahorrando algo: un modelo pequeño que necesita tres intentos
por veredicto consume tres veces la latencia y produce la misma línea de journal
que uno que acierta a la primera.

La tasa sola miente cuando el denominador es pequeño —un fallo de un intento es
100%—, así que la tasa se publica junto al recuento que la produce.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from crypto_agents.state import Action, AgentRole, Backend, FailureKind, FrozenModel

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from crypto_agents.journal import EvaluationRecord
    from crypto_agents.state import LLMCall

__all__ = [
    "BackendStats",
    "RunSummary",
    "backend_stats",
    "summarise",
    "validation_failure_rate",
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
