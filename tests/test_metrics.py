"""Pruebas de las agregaciones sobre llamadas registradas."""

from __future__ import annotations

from datetime import UTC, datetime

from crypto_agents.metrics import backend_stats, validation_failure_rate
from crypto_agents.state import AgentRole, Backend, CallFailure, FailureKind, LLMCall

NOW = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)
DIGEST = "d" * 64


def call(
    role: AgentRole = AgentRole.STRUCTURE,
    backend: Backend = Backend.OLLAMA,
    valid: bool = True,
    cache_hit: bool = False,
    kind: FailureKind = FailureKind.VALIDATION,
) -> LLMCall:
    """Intento registrado. Un intento inválido lleva causa: el contrato la exige."""
    return LLMCall(
        role=role,
        backend=backend,
        model="modelo",
        quota_weight=1.0,
        prompt_digest=DIGEST,
        cache_hit=cache_hit,
        valid=valid,
        failure=None if valid else CallFailure(kind=kind, message="fallo de prueba"),
        latency_ms=10.0,
        at=NOW,
    )


def test_failure_rate_separates_backends_of_the_same_role() -> None:
    """La comparación que importa: el mismo rol, atendido por dos modelos distintos.

    Es la pregunta que decide si un respaldo local ahorra algo. Sin separar por
    backend, un local que falla la mitad de las veces queda diluido en la media
    del remoto que casi nunca falla.
    """
    calls = [
        call(backend=Backend.OPENAI),
        call(backend=Backend.OPENAI),
        call(backend=Backend.OLLAMA, valid=False),
        call(backend=Backend.OLLAMA),
    ]
    rates = validation_failure_rate(calls)
    assert rates[(AgentRole.STRUCTURE, Backend.OPENAI)] == 0.0
    assert rates[(AgentRole.STRUCTURE, Backend.OLLAMA)] == 0.5


def test_cache_hits_are_excluded() -> None:
    """Un acierto de caché no llegó a ningún proveedor: no dice nada del modelo."""
    stats = backend_stats([call(cache_hit=True), call(cache_hit=True), call(valid=False)])
    assert stats[(AgentRole.STRUCTURE, Backend.OLLAMA)].attempts == 1
    assert stats[(AgentRole.STRUCTURE, Backend.OLLAMA)].invalid == 1


def test_stats_publish_the_denominator() -> None:
    """Una tasa sin recuento miente: un fallo de un intento también es el 100%."""
    stats = backend_stats([call(valid=False)])[(AgentRole.STRUCTURE, Backend.OLLAMA)]
    assert stats.failure_rate == 1.0
    assert stats.attempts == 1


def test_roles_are_counted_separately() -> None:
    """Dos roles sobre el mismo backend son dos preguntas distintas."""
    stats = backend_stats([call(role=AgentRole.BULL, valid=False), call(role=AgentRole.BEAR)])
    assert stats[(AgentRole.BULL, Backend.OLLAMA)].invalid == 1
    assert stats[(AgentRole.BEAR, Backend.OLLAMA)].invalid == 0


def test_transport_failures_are_counted_apart_from_validation() -> None:
    """Un 400 no dice nada sobre si el modelo sabe producir el esquema.

    Es la confusión que contaminó la primera tabla de comparación: un id que el
    gateway no sirve y un modelo que alucina campos salían con la misma cifra, y
    la lectura obvia —«este modelo no sabe seguir el esquema»— era falsa para
    todos menos uno.
    """
    calls = [
        call(valid=False, kind=FailureKind.TRANSPORT),
        call(valid=False, kind=FailureKind.TRANSPORT),
        call(valid=False, kind=FailureKind.VALIDATION),
        call(),
    ]
    stats = backend_stats(calls)[(AgentRole.STRUCTURE, Backend.OLLAMA)]

    assert stats.attempts == 4
    assert stats.transport == 2
    assert stats.invalid == 1
    assert stats.answered == 2
    assert stats.failure_rate == 0.5


def test_a_backend_that_never_answers_has_no_validation_rate() -> None:
    """Sin una sola respuesta, la tasa de validación no tiene denominador.

    Publicar 100% ahí mandaría a arreglar el esquema cuando lo que hay que
    arreglar es el id o la credencial.
    """
    stats = backend_stats([call(valid=False, kind=FailureKind.TRANSPORT) for _ in range(5)])
    item = stats[(AgentRole.STRUCTURE, Backend.OLLAMA)]

    assert item.attempts == 5
    assert item.answered == 0
    assert item.failure_rate == 0.0


def test_no_calls_reports_nothing() -> None:
    """Sin llamadas no hay tasa que inventar."""
    assert backend_stats([]) == {}
    assert validation_failure_rate([]) == {}
