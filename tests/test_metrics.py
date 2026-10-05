"""Pruebas de las agregaciones sobre llamadas registradas."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from crypto_agents import metrics as metrics_module
from crypto_agents.journal import EvaluationRecord
from crypto_agents.llm import InvalidModelOutputError, ModelCallError
from crypto_agents.metrics import (
    NO_ERROR,
    AbortKind,
    attempt_counts,
    backend_stats,
    conviction_cross,
    dismissal_cross,
    failure_counts,
    invalidation_stats,
    live_latency,
    nearest_rank,
    paired_difference,
    provider_rejections,
    quota_by_locality,
    resume_delta,
    return_stats,
    risk_flow,
    undecided_causes,
    validation_failure,
    validation_failure_rate,
    wilson_interval,
    worst_pair,
)
from crypto_agents.quota import QuotaExhaustedError
from crypto_agents.state import (
    Action,
    ActivationCheck,
    AgentRole,
    Backend,
    Claim,
    DebateBrief,
    Decision,
    Dimension,
    ExecutionMode,
    FailureKind,
    LLMCall,
    MarketSnapshot,
    NodeError,
    OrderIntent,
    Proposal,
    RiskVerdict,
    Side,
    Strength,
    StructuredOutputMode,
)
from tests.conftest import (
    SCARCE,
    FakeLLM,
    brief_payload,
    insufficient_funds_error,
    verdict_payload,
)
from tests.test_replay import Harness, run, synthetic_rows

NOW = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)
DIGEST = "d" * 64


def call(
    role: AgentRole = AgentRole.STRUCTURE,
    backend: Backend = Backend.OLLAMA,
    valid: bool = True,
    cache_hit: bool = False,
    kind: FailureKind = FailureKind.SCHEMA,
) -> LLMCall:
    """Intento registrado. Un intento inválido lleva causa: el contrato la exige."""
    return LLMCall(
        role=role,
        backend=backend,
        model="modelo",
        structured_output=StructuredOutputMode.JSON_SCHEMA,
        quota_weight=1.0,
        prompt_digest=DIGEST,
        cache_hit=cache_hit,
        valid=valid,
        failure_kind=None if valid else kind,
        failure_message=None if valid else "fallo de prueba",
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
        call(valid=False, kind=FailureKind.SCHEMA),
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


# ─────────────────────────────── Auditoría de una corrida, punto por punto ────────────────────────
# Cada valor esperado está calculado a mano sobre registros escritos aquí mismo. Con
# agentes no deterministas, una métrica que solo se ha visto correr sobre una corrida
# real no se ha probado: no hay con qué comparar la cifra que imprime.


def timed(
    latency_ms: float,
    role: AgentRole = AgentRole.STRUCTURE,
    backend: Backend = Backend.OPENAI,
    cache_hit: bool = False,
    weight: float = 1.0,
    failure: FailureKind | None = None,
) -> LLMCall:
    """Intento con latencia, peso y causa de fallo elegidos por la prueba."""
    return LLMCall(
        role=role,
        backend=backend,
        model="modelo",
        structured_output=StructuredOutputMode.JSON_SCHEMA,
        quota_weight=weight,
        prompt_digest=DIGEST,
        cache_hit=cache_hit,
        valid=failure is None,
        failure_kind=failure,
        failure_message=None if failure is None else "fallo de prueba",
        latency_ms=latency_ms,
        at=NOW,
    )


def snapshot(close: float = 100.0) -> MarketSnapshot:
    """Momento evaluado con el cierre que la prueba necesita."""
    return MarketSnapshot(
        run_id=uuid4(),
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="4h",
        timestamp=NOW,
        close=close,
        candles_digest="a" * 64,
        candles_count=500,
    )


def proposal(action: Action, invalidation: float | None = None) -> Proposal:
    """Decisión sin mesas."""
    return Proposal(
        action=action,
        confidence=0.6,
        size_fraction=0.0 if action is Action.HOLD else 0.2,
        invalidation_price=invalidation,
        rationale="La lectura tecnica acompana la direccion propuesta.",
    )


def decision(action: Action, dismissed: Side | None = None) -> Decision:
    """Decisión con mesas."""
    return Decision(
        action=action,
        confidence=0.6,
        size_fraction=0.0 if action is Action.HOLD else 0.2,
        invalidation_price=None if action is Action.HOLD else 95.0,
        rationale="La lectura tecnica acompana la direccion propuesta.",
        dismissed_side=dismissed,
        dismissal_reason=None if dismissed is None else "Su nivel de referencia ya se perdio.",
    )


def brief(side: Side, conviction: float) -> DebateBrief:
    """Alegato con la convicción que la prueba necesita."""
    claim = Claim(
        text="La media rapida actua como soporte dinamico en cada retroceso.",
        grounded_in=["structure-1"],
        strength=Strength.MODERATE,
    )
    return DebateBrief(
        side=side,
        thesis="La estructura sigue intacta mientras el soporte aguante el retroceso.",
        claims=[claim, claim],
        conviction=conviction,
        strongest_counterargument="Un cierre bajo el soporte invalida toda la lectura.",
    )


def record(
    calls: tuple[LLMCall, ...] = (),
    errors: tuple[NodeError, ...] = (),
    run_id: UUID | None = None,
    **fields: object,
) -> EvaluationRecord:
    """Registro de evaluación con solo lo que la prueba declara."""
    return EvaluationRecord.model_validate(
        {
            "run_id": run_id if run_id is not None else uuid4(),
            "at": NOW,
            "symbol": "BTC/USDT",
            "timeframe": "4h",
            "calls": calls,
            "errors": errors,
            **fields,
        }
    )


def failed(node: str, message: str) -> tuple[NodeError, ...]:
    """Un único fallo de nodo."""
    return (NodeError(node=node, message=message, at=NOW),)


OPEN_GATE = ActivationCheck(should_run=True, triggers=["range_breakout"], reason="disparó")
CLOSED_GATE = ActivationCheck(should_run=False, reason="ninguna regla de activación disparó")


# ── 1. Intentos ──────────────────────────────────────────────────────────────


def test_attempts_are_counted_along_every_axis() -> None:
    """Cinco filas: 4 por rol/backend/caché/validez y un intento inválido releído de la caché.

    El último es un intento fallido reproducido sin pagarlo otra vez. Se cuenta
    aparte porque no es ni una llamada viva ni una respuesta servida.
    """
    stale = LLMCall(
        role=AgentRole.BULL,
        backend=Backend.OLLAMA,
        model="modelo",
        structured_output=StructuredOutputMode.JSON_SCHEMA,
        quota_weight=1.0,
        prompt_digest=DIGEST,
        cache_hit=True,
        valid=False,
        failure_kind=FailureKind.SCHEMA,
        failure_message="entrada vieja",
        latency_ms=0.0,
        at=NOW,
    )
    records = [
        record(
            calls=(
                timed(10.0),
                timed(10.0, failure=FailureKind.SCHEMA),
                timed(0.0, role=AgentRole.BULL, backend=Backend.OLLAMA, cache_hit=True),
            )
        ),
        record(calls=(timed(10.0, role=AgentRole.DECIDER, failure=FailureKind.TRANSPORT), stale)),
    ]

    counts = attempt_counts(records)

    assert counts.total == 5
    assert counts.by_role == {AgentRole.STRUCTURE: 2, AgentRole.BULL: 2, AgentRole.DECIDER: 1}
    assert counts.by_backend == {Backend.OPENAI: 3, Backend.OLLAMA: 2}
    assert counts.cache_hits == 2
    assert counts.live == 3
    assert counts.valid == 2
    assert counts.invalid == 3
    assert counts.cached_invalid == 1


def unbalanced_calls() -> tuple[LLMCall, ...]:
    """Un par grande que acierta y uno pequeño que falla la mitad.

    structure/openai: 8 válidas y 2 rechazos de transporte. bull/ollama: 1 válida y
    1 inválida. Respondidas: 8 + 2 = 10. Inválidas: 1.
    """
    return (
        *(timed(10.0) for _ in range(8)),
        *(timed(10.0, failure=FailureKind.TRANSPORT) for _ in range(2)),
        timed(10.0, role=AgentRole.BULL, backend=Backend.OLLAMA),
        timed(10.0, role=AgentRole.BULL, backend=Backend.OLLAMA, failure=FailureKind.SCHEMA),
    )


def test_the_failure_rate_is_weighted_by_attempts_and_shows_its_denominator() -> None:
    """1 inválida de 10 respondidas es 10%, no la media de 0% y 50%.

    La media simple de los pares daría 25% y el máximo 50%: tres cifras para la
    misma corrida. La que se publica como tasa del brazo es la ponderada, con el
    denominador al lado, y el transporte fuera de él.
    """
    rate = validation_failure([record(calls=unbalanced_calls())])

    assert rate.numerator == 1
    assert rate.denominator == 10
    assert rate.rate == pytest.approx(0.10)


def test_the_worst_pair_is_reported_under_its_own_name() -> None:
    """El 50% de la tabla anterior existe, pero es de un par y trae sus dos intentos."""
    worst = worst_pair([record(calls=unbalanced_calls())])

    assert worst is not None
    assert (worst.role, worst.backend) == (AgentRole.BULL, Backend.OLLAMA)
    assert worst.stats.failure_rate == 0.5
    assert worst.stats.answered == 2


def test_without_answers_there_is_no_rate_and_no_worst_pair() -> None:
    """Solo rechazos de transporte: nada que dividir."""
    records = [record(calls=(timed(10.0, failure=FailureKind.TRANSPORT),))]

    assert validation_failure(records).rate is None
    assert worst_pair(records) is None


def test_failed_attempts_are_counted_by_kind() -> None:
    """Dos de esquema, una de contexto, una de timeout y ninguna de transporte.

    Los cuatro tipos aparecen siempre: un cero es un dato, y una clave ausente
    obligaría a quien lea la tabla a adivinar si es cero o si no se midió.
    """
    calls = (
        timed(10.0),
        timed(10.0, failure=FailureKind.SCHEMA),
        timed(10.0, failure=FailureKind.SCHEMA),
        timed(10.0, failure=FailureKind.CONTEXT),
        timed(10.0, failure=FailureKind.TIMEOUT),
    )
    assert failure_counts([record(calls=calls)]) == {
        FailureKind.SCHEMA: 2,
        FailureKind.CONTEXT: 1,
        FailureKind.TIMEOUT: 1,
        FailureKind.TRANSPORT: 0,
    }


def test_context_failures_count_against_the_model_and_timeouts_do_not() -> None:
    """Respondidas: 4 de 5. Inválidas: 3, una de ellas de contexto. El timeout queda fuera.

    Un fallo de contexto es contenido que el modelo produjo mal, igual que uno de
    esquema. Un timeout no produjo nada: entra en el denominador tanto como un 503.
    """
    calls = (
        timed(10.0),
        timed(10.0, failure=FailureKind.SCHEMA),
        timed(10.0, failure=FailureKind.SCHEMA),
        timed(10.0, failure=FailureKind.CONTEXT),
        timed(10.0, failure=FailureKind.TIMEOUT),
    )
    stats = backend_stats(calls)[(AgentRole.STRUCTURE, Backend.OPENAI)]

    assert (stats.attempts, stats.answered) == (5, 4)
    assert (stats.invalid, stats.context) == (3, 1)
    assert (stats.timeout, stats.transport) == (1, 0)
    assert stats.failure_rate == 0.75


# ── 2. Latencia ──────────────────────────────────────────────────────────────


def test_latency_is_measured_over_live_calls_only() -> None:
    """100, 200, 300, 400 y 1000 ms vivas, más un acierto de caché a 0 ms.

    n = 5, media = 2000 / 5 = 400, mediana = 300, p95 = rango ceil(0.95 * 5) = 5,
    es decir 1000. Con el acierto dentro la media bajaría a 333: es la columna que
    una caché tibia hacía incomparable entre brazos.
    """
    calls = (
        timed(400.0),
        timed(100.0),
        timed(0.0, cache_hit=True),
        timed(1000.0, failure=FailureKind.TRANSPORT),
        timed(300.0),
        timed(200.0),
    )
    stats = live_latency([record(calls=calls)])

    assert stats is not None
    assert stats.n == 5
    assert stats.mean_ms == pytest.approx(400.0)
    assert stats.median_ms == pytest.approx(300.0)
    assert stats.p95_ms == pytest.approx(1000.0)


def test_a_run_served_entirely_from_cache_has_no_latency() -> None:
    """Sin llamadas vivas no hay latencia que publicar: ni cero, ni la del reloj."""
    assert live_latency([record(calls=(timed(0.0, cache_hit=True),))]) is None


def test_nearest_rank_uses_integer_arithmetic() -> None:
    """Sobre 1..30: p10 es el 3.º valor, p90 el 27.º y p95 el 29.º (techo de 28.5).

    Y sobre 1..100, el percentil 7 es el 7.º. `7 / 100 * 100` vale
    7.000000000000001 en coma flotante y su techo es 8: el rango se calcula con
    enteros para que el percentil sea el que sale a mano con cualquier tamaño.
    """
    values = [float(value) for value in range(30, 0, -1)]

    assert nearest_rank(values, 10) == 3.0
    assert nearest_rank(values, 90) == 27.0
    assert nearest_rank(values, 95) == 29.0
    assert nearest_rank([7.0], 10) == 7.0
    assert nearest_rank([float(value) for value in range(1, 101)], 7) == 7.0


# ── 3. Cuota ─────────────────────────────────────────────────────────────────


def test_quota_is_split_between_remote_and_local() -> None:
    """Dos remotas de peso 2 y tres locales de peso 1: 4.0 remota, 3.0 local.

    El acierto de caché remoto no suma. Sumadas en una columna, un brazo local y
    uno remoto no se pueden comparar: solo una de las dos mitades cuesta ventana.
    """
    calls = (
        timed(10.0, weight=2.0),
        timed(10.0, weight=2.0, failure=FailureKind.SCHEMA),
        timed(0.0, weight=2.0, cache_hit=True),
        *(timed(10.0, backend=Backend.OLLAMA) for _ in range(3)),
    )
    split = quota_by_locality([record(calls=calls)])

    assert split.remote == 4.0
    assert split.local == 3.0


# ── 4. Evaluaciones sin decisión ─────────────────────────────────────────────


def test_undecided_evaluations_are_grouped_by_node_and_kind_of_failure() -> None:
    """Seis sin decisión y una decidida: la decidida no aparece.

    Los mensajes se construyen con las excepciones reales del router y del
    contador: si su redacción cambia, esta prueba deja de clasificar y falla, en
    vez de que la causa acabe en «other» sin que nadie lo vea.
    """
    invalid = str(InvalidModelOutputError(AgentRole.DECIDER, 2, "action: Field required"))
    rejected = str(ModelCallError(AgentRole.BEAR, SCARCE, RuntimeError("503 Upstream")))
    exhausted = str(QuotaExhaustedError(AgentRole.DECIDER, ("glm-5.2",)))
    records = [
        record(activation=OPEN_GATE, errors=failed("decide", invalid)),
        record(activation=OPEN_GATE, errors=failed("decide", invalid)),
        record(activation=OPEN_GATE, errors=failed("bear", rejected)),
        record(activation=OPEN_GATE, errors=failed("decide", exhausted)),
        record(activation=CLOSED_GATE),
        record(activation=OPEN_GATE, errors=failed("execute_order", "algo que nadie previó")),
        record(activation=OPEN_GATE, decision=decision(Action.HOLD)),
    ]

    assert undecided_causes(records) == {
        ("bear", AbortKind.TRANSPORT): 1,
        ("decide", AbortKind.QUOTA): 1,
        ("decide", AbortKind.VALIDATION): 2,
        ("execute_order", AbortKind.OTHER): 1,
        (NO_ERROR, AbortKind.GATE_CLOSED): 1,
    }


def test_the_first_error_is_the_cause_and_the_rest_are_consequences() -> None:
    """La mesa que citó un id inventado es la causa; el decisor sin alegato, el síntoma."""
    errors = (
        NodeError(node="bull", message=str(invalid_output(AgentRole.BULL)), at=NOW),
        NodeError(node="decide", message="faltan alegatos: bull", at=NOW),
    )
    assert undecided_causes([record(activation=OPEN_GATE, errors=errors)]) == {
        ("bull", AbortKind.VALIDATION): 1
    }


def invalid_output(role: AgentRole) -> InvalidModelOutputError:
    """El error real con que el router dice que un rol agotó sus intentos."""
    return InvalidModelOutputError(role, 2, "grounded_in: ids que ningún veredicto emitió")


def test_a_timeout_is_not_reported_as_a_transport_rejection() -> None:
    """El mensaje del router es el mismo; lo que los separa es el tipo del intento.

    Un rechazo dice que el proveedor no sirve ese id; un plazo vencido, que tardó
    más de lo concedido. Se arreglan en sitios distintos, y en una sola categoría
    la tabla de abortos manda a mirar el que no es.
    """
    message = str(ModelCallError(AgentRole.BEAR, SCARCE, TimeoutError("120 s")))
    timed_out = record(
        activation=OPEN_GATE,
        errors=failed("bear", message),
        calls=(timed(120_000.0, role=AgentRole.BEAR, failure=FailureKind.TIMEOUT),),
    )
    rejected = record(
        activation=OPEN_GATE,
        errors=failed("bear", message),
        calls=(timed(200.0, role=AgentRole.BEAR, failure=FailureKind.TRANSPORT),),
    )

    assert undecided_causes([timed_out, rejected]) == {
        ("bear", AbortKind.TIMEOUT): 1,
        ("bear", AbortKind.TRANSPORT): 1,
    }


def test_a_node_assertion_is_reported_as_a_bug_not_as_a_model_failure() -> None:
    """Si un nodo recibe lo que el router debía rechazar, la causa tiene nombre propio."""
    errors = failed("bull", "validación de contexto saltada: grounded_in: ids que ningún...")
    assert undecided_causes([record(activation=OPEN_GATE, errors=errors)]) == {
        ("bull", AbortKind.CONTEXT_BYPASSED): 1
    }


def test_an_open_gate_without_error_or_decision_is_not_called_a_closed_gate() -> None:
    """Un hueco sin explicación se nombra como tal, no se esconde en otra categoría."""
    assert undecided_causes([record(activation=OPEN_GATE)]) == {(NO_ERROR, AbortKind.OTHER): 1}


def funds_message() -> str:
    """El mensaje del nodo ante un 402: la excepción real del SDK envuelta por el router."""
    return str(ModelCallError(AgentRole.BEAR, SCARCE, insufficient_funds_error()))


def assert_a_402_is_insufficient_funds() -> None:
    rejected = record(
        activation=OPEN_GATE,
        errors=failed("bear", funds_message()),
        calls=(timed(400.0, role=AgentRole.BEAR, failure=FailureKind.TRANSPORT),),
    )
    assert undecided_causes([rejected]) == {("bear", AbortKind.INSUFFICIENT_FUNDS): 1}


def test_the_real_402_message_carries_what_the_classifier_reads() -> None:
    """Si el SDK o el router reescriben el mensaje, esto falla antes de que el 402 vaya a `other`.

    El texto es el que el SDK escribe, no uno a mano.
    """
    message = funds_message()
    assert "Error code: 402" in message
    assert "Insufficient account funds" in message


def test_a_402_is_insufficient_funds_and_not_a_generic_transport_rejection() -> None:
    assert_a_402_is_insufficient_funds()


def test_a_503_is_still_a_transport_rejection() -> None:
    message = str(ModelCallError(AgentRole.BEAR, SCARCE, RuntimeError("503 Upstream")))
    assert undecided_causes([record(activation=OPEN_GATE, errors=failed("bear", message))]) == {
        ("bear", AbortKind.TRANSPORT): 1
    }


def rejected(model: str, message: str, failure: FailureKind = FailureKind.TRANSPORT) -> LLMCall:
    return timed(100.0, failure=failure).model_copy(
        update={"model": model, "failure_message": message}
    )


def test_a_410_and_a_503_are_two_rows_each_with_its_code_and_body() -> None:
    gone = "APIStatusError: Error code: 410 - {'error': {'message': 'Endpoint is unavailable.'}}"
    down = "APIStatusError: Error code: 503 - {'error': {'message': 'Service Unavailable'}}"
    calls = [rejected("m", gone), rejected("m", down), rejected("m", gone), timed(10.0)]
    found = provider_rejections([record(calls=tuple(calls))])

    assert found == {
        ("m", AgentRole.STRUCTURE, "410", "{'error': {'message': 'Endpoint is unavailable.'}}"): 2,
        ("m", AgentRole.STRUCTURE, "503", "{'error': {'message': 'Service Unavailable'}}"): 1,
    }


def test_a_timeout_and_a_message_without_a_code_are_listed_under_their_own_names() -> None:
    calls = [
        rejected("m", "ReadTimeout: 120 s", FailureKind.TIMEOUT),
        rejected("m", "ConnectError: dns"),
    ]
    codes = {code for (_, _, code, _) in provider_rejections([record(calls=tuple(calls))])}
    assert codes == {"timeout", "sin código"}


def test_content_failures_are_not_provider_rejections() -> None:
    calls = [timed(10.0, failure=FailureKind.SCHEMA), timed(10.0, failure=FailureKind.CONTEXT)]
    assert provider_rejections([record(calls=tuple(calls))]) == {}


def test_the_402_attempt_stays_a_transport_failure_kind() -> None:
    """`FailureKind` es cerrado: el saldo es una categoría del aborto, no un tipo de intento."""
    assert {kind.value for kind in FailureKind} == {"schema", "context", "timeout", "transport"}


def test_mutation_a_402_classified_as_generic_transport_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert_a_402_is_insufficient_funds()  # control: el código real pasa
    without_funds = tuple(
        marker
        for marker in metrics_module._ABORT_MARKERS
        if marker[1] is not AbortKind.INSUFFICIENT_FUNDS
    )
    monkeypatch.setattr(metrics_module, "_ABORT_MARKERS", without_funds)
    with pytest.raises(AssertionError):
        assert_a_402_is_insufficient_funds()


class FundsLLM(FakeLLM):
    """Backend cuya cuenta se quedó sin saldo: el 402 real, antes de producir contenido."""

    async def complete(self, choice: object, prompt: str, schema: type) -> str:
        if self._target(prompt, schema) == "bear":
            raise insufficient_funds_error()
        return await super().complete(choice, prompt, schema)  # type: ignore[arg-type]


class RejectingLLM(FakeLLM):
    """Backend cuyo proveedor rechaza a la mesa bajista antes de producir contenido."""

    async def complete(self, choice: object, prompt: str, schema: type) -> str:
        """Falla como fallaría un 503 del gateway."""
        if self._target(prompt, schema) == "bear":
            raise RuntimeError("503 Upstream request failed")
        return await super().complete(choice, prompt, schema)  # type: ignore[arg-type]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("backend", "expected", "kind"),
    [
        (FakeLLM({"decider": "{}"}), ("decide", AbortKind.VALIDATION), FailureKind.SCHEMA),
        (
            FakeLLM({"bull": brief_payload(Side.BULL, grounded_in="structure-9")}),
            ("bull", AbortKind.VALIDATION),
            FailureKind.CONTEXT,
        ),
        (
            FakeLLM({"bull": brief_payload(Side.BEAR)}),
            ("bull", AbortKind.VALIDATION),
            FailureKind.CONTEXT,
        ),
        (
            FakeLLM({"structure": verdict_payload(Dimension.MOMENTUM)}),
            ("structure", AbortKind.VALIDATION),
            FailureKind.CONTEXT,
        ),
        (
            FakeLLM({"structure": verdict_payload(Dimension.STRUCTURE, cites="RSI_999")}),
            ("structure", AbortKind.VALIDATION),
            FailureKind.CONTEXT,
        ),
        (RejectingLLM(), ("bear", AbortKind.TRANSPORT), FailureKind.TRANSPORT),
        (FundsLLM(), ("bear", AbortKind.INSUFFICIENT_FUNDS), FailureKind.TRANSPORT),
    ],
    ids=[
        "schema",
        "ungrounded",
        "wrong_side",
        "wrong_dimension",
        "unknown_indicator",
        "503",
        "402",
    ],
)
async def test_every_abort_the_real_graph_produces_is_classified(
    backend: FakeLLM, expected: tuple[str, AbortKind], kind: FailureKind
) -> None:
    """Los mensajes que escriben el router y los nodos, leídos del grafo real.

    Es lo que ata la clasificación al código que la produce: ninguna de estas
    causas puede acabar en «other», que es donde iría a parar en silencio el día
    que alguien reescriba un mensaje.

    Los cuatro casos de contexto abortan como `validation` en el nodo que llamó —el
    modelo agotó sus intentos— y lo que dice que fue contexto y no esquema es el
    tipo de cada intento. Ninguno llega ya a una aserción de nodo.
    """
    harness = Harness()
    harness.backend = backend
    records = await run(harness, synthetic_rows(), fill=True, evaluations=12)

    causes = undecided_causes(records)

    assert causes.get(expected, 0) > 0, causes
    assert set(causes) <= {expected, (NO_ERROR, AbortKind.GATE_CLOSED)}, causes
    assert sum(causes.values()) == 12, "con este backend ninguna evaluación debería decidir"
    counts = failure_counts(records)
    assert counts[kind] > 0
    assert sum(counts.values()) == counts[kind], counts


# ── 5. Del decisor a la orden ────────────────────────────────────────────────


def test_the_risk_flow_separates_vetoes_from_orders_that_never_left() -> None:
    """Tres accionables: una orden, un veto y una aprobada que el ejecutor no envió.

    Más un hold y una sin decisión, que no son accionables. `órdenes == accionables`
    solo se puede afirmar si las otras dos columnas son cero, así que van al lado.
    """
    order = OrderIntent(
        symbol="BTC/USDT",
        side=Action.BUY,
        size_fraction=0.1,
        reference_price=100.0,
        invalidation_price=95.0,
        mode=ExecutionMode.PAPER,
    )
    approved = RiskVerdict(approved=True, final_size_fraction=0.1)
    vetoed = RiskVerdict(
        approved=False, final_size_fraction=0.0, veto_rule="cooldown", veto_reason="quedan 30 min"
    )
    records = [
        record(proposal=proposal(Action.BUY, 95.0), risk=approved, order=order),
        record(proposal=proposal(Action.SELL, 105.0), risk=vetoed),
        record(
            proposal=proposal(Action.BUY, 95.0),
            risk=approved,
            errors=failed("execute_order", "RuntimeError: exchange caído"),
        ),
        record(
            proposal=proposal(Action.HOLD), risk=RiskVerdict(approved=True, final_size_fraction=0.0)
        ),
        record(activation=CLOSED_GATE),
    ]

    flow = risk_flow(records)

    assert flow.actionable == 3
    assert flow.holds == 1
    assert flow.orders == 1
    assert flow.vetoes == {"cooldown": 1}
    assert flow.unexecuted == 1


# ── 6. Invalidación ──────────────────────────────────────────────────────────


def test_the_stop_side_and_its_distance_are_measured_against_the_close() -> None:
    """Cierre 100. buy@95, buy@101, sell@110, sell@98, sell@97, buy@100 y un hold.

    Lado equivocado: buy@101 (stop por encima en un largo), sell@98 y sell@97 (por
    debajo en un corto) y buy@100 (sin recorrido): 4 de 6. Los dos lados no pesan
    igual a propósito: con uno equivocado por lado, invertir la regla del `sell`
    daría el mismo total.

    Distancias ordenadas: 0, 0.01, 0.02, 0.03, 0.05, 0.10. p10 = rango ceil(0.6) =
    1.º = 0; mediana = (0.02 + 0.03) / 2 = 0.025; p90 = rango ceil(5.4) = 6.º = 0.10.
    """
    records = [
        record(snapshot=snapshot(), proposal=proposal(action, invalidation))
        for action, invalidation in (
            (Action.BUY, 95.0),
            (Action.BUY, 101.0),
            (Action.SELL, 110.0),
            (Action.SELL, 98.0),
            (Action.SELL, 97.0),
            (Action.BUY, 100.0),
        )
    ]
    records.append(record(snapshot=snapshot(), proposal=proposal(Action.HOLD)))

    stats = invalidation_stats(records)

    assert stats.n == 6
    assert stats.wrong_side == 4
    assert stats.p10 == pytest.approx(0.0)
    assert stats.median == pytest.approx(0.025)
    assert stats.p90 == pytest.approx(0.10)


def test_without_actionable_proposals_there_is_no_distance_to_report() -> None:
    """Solo holds: n = 0 y los percentiles vacíos, no ceros."""
    stats = invalidation_stats([record(snapshot=snapshot(), proposal=proposal(Action.HOLD))])

    assert stats.n == 0
    assert stats.wrong_side == 0
    assert stats.median is None


# ── 7. Decisión contra mesas ─────────────────────────────────────────────────


def debated() -> list[EvaluationRecord]:
    """Cuatro decisiones con las dos mesas, una con una sola y una sin mesas."""
    return [
        record(
            decision=decision(Action.BUY, Side.BEAR),
            briefs=(brief(Side.BULL, 0.8), brief(Side.BEAR, 0.4)),
        ),
        record(
            decision=decision(Action.BUY, Side.BULL),
            briefs=(brief(Side.BEAR, 0.6), brief(Side.BULL, 0.3)),
        ),
        record(
            decision=decision(Action.SELL, Side.BULL),
            briefs=(brief(Side.BULL, 0.5), brief(Side.BEAR, 0.5)),
        ),
        record(
            decision=decision(Action.HOLD), briefs=(brief(Side.BULL, 0.7), brief(Side.BEAR, 0.2))
        ),
        record(decision=decision(Action.BUY, Side.BEAR), briefs=(brief(Side.BULL, 0.9),)),
        record(proposal=proposal(Action.BUY, 95.0)),
    ]


def test_actions_are_crossed_with_the_sign_of_the_conviction_gap() -> None:
    """Signo de bull - bear: +1, -1, 0 y +1. La de una sola mesa no tiene diferencia."""
    assert conviction_cross(debated()) == {
        (Action.BUY, -1): 1,
        (Action.BUY, 1): 1,
        (Action.HOLD, 1): 1,
        (Action.SELL, 0): 1,
    }


def test_actions_are_crossed_with_the_dismissed_desk() -> None:
    """Toda `Decision` cuenta, también la de una mesa; la `Proposal` no trae descarte."""
    assert dismissal_cross(debated()) == {
        (Action.BUY, Side.BEAR): 2,
        (Action.BUY, Side.BULL): 1,
        (Action.HOLD, None): 1,
        (Action.SELL, Side.BULL): 1,
    }


# ── Reanudación ──────────────────────────────────────────────────────────────


def test_a_resumed_run_reports_how_many_evaluations_got_a_second_draw() -> None:
    """Dos sin decidir antes; una de ellas decide ahora: 1 rescatada de 2.

    Un intento inválido no se cachea, así que la reanudación vuelve a llamar en
    vivo justo a las evaluaciones que abortaron. Las «decididas» de una corrida
    reanudada incluyen esa segunda tirada, y sin esta cifra no se distinguen de las
    de una pasada única.
    """
    first, second, third, fourth = uuid4(), uuid4(), uuid4(), uuid4()
    previous = [
        record(run_id=first, activation=OPEN_GATE),
        record(run_id=second, activation=OPEN_GATE),
        record(run_id=third, proposal=proposal(Action.HOLD)),
    ]
    current = [
        record(run_id=first, proposal=proposal(Action.HOLD)),
        record(run_id=second, activation=OPEN_GATE),
        record(run_id=third, proposal=proposal(Action.HOLD)),
        record(run_id=fourth, proposal=proposal(Action.HOLD)),
    ]

    delta = resume_delta(previous, current)

    assert delta.undecided_before == 2
    assert delta.rescued == 1


# ── Incertidumbre ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("successes", "n", "low", "high"),
    [
        (8, 10, 0.4902, 0.9433),
        (0, 10, 0.0, 0.2775),
        (10, 10, 0.7225, 1.0),
        (50, 100, 0.4038, 0.5962),
    ],
    ids=["8 de 10", "0 de 10", "10 de 10", "50 de 100"],
)
def test_wilson_interval_matches_the_published_values(
    successes: int, n: int, low: float, high: float
) -> None:
    """Valores calculados aparte con la fórmula de Wilson al 95%, no con el código bajo prueba.

    Los dos extremos importan: con 0 de 10 el intervalo normal daría un límite
    inferior negativo y con 10 de 10 un superior de 1.0 exacto —ninguno es un
    intervalo—, y Wilson los mantiene dentro de [0, 1] sin recortar a ciegas.
    """
    interval = wilson_interval(successes, n)

    assert interval is not None
    assert interval.low == pytest.approx(low, abs=1e-4)
    assert interval.high == pytest.approx(high, abs=1e-4)


def test_wilson_interval_is_wider_with_fewer_observations() -> None:
    """Misma tasa, 50%: con 10 observaciones el intervalo es más ancho que con 100."""
    small = wilson_interval(5, 10)
    large = wilson_interval(50, 100)

    assert small is not None
    assert large is not None
    assert small.high - small.low > large.high - large.low


def test_wilson_interval_without_observations_is_undefined() -> None:
    """0 de 0 no tiene intervalo: devolver [0, 1] sería una cifra donde no hay medida."""
    assert wilson_interval(0, 0) is None


@pytest.mark.parametrize(("successes", "n"), [(-1, 10), (11, 10)])
def test_wilson_interval_refuses_impossible_counts(successes: int, n: int) -> None:
    """Más aciertos que observaciones es un bug de quien cuenta, no un dato."""
    with pytest.raises(ValueError, match="aciertos"):
        wilson_interval(successes, n)


def test_return_stats_use_the_sample_standard_deviation() -> None:
    """Retornos 2%, -1%, 3% y 0%: media 1%, desviación muestral 1.826%, error 0.913%.

    Con la desviación poblacional (dividir entre n) el error saldría 0.79%: una
    muestra de cuatro órdenes no conoce la varianza de la que viene, y el estimador
    que ignora eso presume más precisión de la que hay.
    """
    stats = return_stats([0.02, -0.01, 0.03, 0.0])

    assert stats is not None
    assert stats.n == 4
    assert stats.mean == pytest.approx(0.01)
    assert stats.stderr == pytest.approx(0.009128709, rel=1e-6)


def test_return_stats_of_one_order_has_a_mean_and_no_error() -> None:
    """Con una orden hay media y no hay dispersión: el error es `None`, no cero."""
    stats = return_stats([0.02])

    assert stats is not None
    assert stats.n == 1
    assert stats.mean == pytest.approx(0.02)
    assert stats.stderr is None


def test_return_stats_without_orders_is_undefined() -> None:
    """Sin órdenes resueltas no hay retorno medio: cero sería una afirmación."""
    assert return_stats([]) is None


# ── Diferencia pareada ───────────────────────────────────────────────────────


def test_the_paired_difference_has_a_mean_a_standard_error_and_an_interval() -> None:
    """Treinta evaluaciones: 15 con +1% y 15 con +3% contra un brazo que no ganó nada.

    Las diferencias valen 0.01 y 0.03: media 0.02, desviación muestral 0.010171, error
    estándar 0.0018570 y un intervalo al 95% de 0.01636 a 0.02364 (±1.96 errores).
    """
    arm = [0.01] * 15 + [0.03] * 15

    result = paired_difference(arm, [0.0] * 30)

    assert result is not None
    assert result.n == 30
    assert result.mean == pytest.approx(0.02)
    assert result.stderr == pytest.approx(0.001857, rel=1e-3)
    assert result.low == pytest.approx(0.01636, abs=1e-5)
    assert result.high == pytest.approx(0.02364, abs=1e-5)


def test_the_difference_is_paired_not_a_comparison_of_two_independent_series() -> None:
    """Dos series muy dispersas que difieren siempre en 0.001: la diferencia no tiene error.

    Tratarlas como independientes sumaría la dispersión de cada una y daría un
    intervalo enorme para una diferencia que es constante: justo el ruido que el
    emparejamiento por evaluación existe para quitar.
    """
    reference = [(-1) ** index * 0.05 for index in range(40)]
    arm = [value + 0.001 for value in reference]

    result = paired_difference(arm, reference)

    assert result is not None
    assert result.mean == pytest.approx(0.001)
    assert result.stderr == pytest.approx(0.0, abs=1e-12)
    assert result.high - result.low == pytest.approx(0.0, abs=1e-12)


def test_the_difference_of_an_arm_with_itself_is_exactly_zero() -> None:
    """Contra sí mismo no hay diferencia ni error: el brazo de referencia no se compara."""
    series = [0.01 * (index % 5) for index in range(30)]

    result = paired_difference(series, series)

    assert result is not None
    assert (result.mean, result.stderr, result.low, result.high) == (0.0, 0.0, 0.0, 0.0)


def test_fewer_than_thirty_evaluations_have_no_interval() -> None:
    """Con 29 el 1.96 es una aproximación demasiado optimista: se dice que no se calcula."""
    assert paired_difference([0.01] * 29, [0.0] * 29) is None
    assert paired_difference([0.01] * 30, [0.0] * 30) is not None


def test_series_of_different_length_are_not_paired() -> None:
    """Pareado es posición contra posición: sin la misma longitud no hay pareja."""
    assert paired_difference([0.01] * 40, [0.0] * 35) is None
