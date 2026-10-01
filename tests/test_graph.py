"""Pruebas del grafo de extremo a extremo con un cliente LLM falso.

El falso devuelve payloads fijos y se limita a leer del prompt qué dimensión o
qué mesa se le está pidiendo, igual que haría un modelo. Ninguna prueba toca la
red ni un proveedor real.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from crypto_agents.activation import ActivationConfig
from crypto_agents.cache import InMemoryResponseCache
from crypto_agents.execution import PaperExecutor
from crypto_agents.graph import AgentContext, build_graph
from crypto_agents.journal import InMemoryJournal
from crypto_agents.llm import ContextCheck, ModelRouter
from crypto_agents.nodes import CONTEXT_BYPASSED
from crypto_agents.prompts import format_indicators, format_verdicts
from crypto_agents.quota import QuotaLedger
from crypto_agents.risk import AccountState, RiskLimits
from crypto_agents.settings import Backend, ModelChoice, Settings, load_settings
from crypto_agents.state import (
    Action,
    AgentRole,
    Dimension,
    ExecutionMode,
    FailureKind,
    LLMCall,
    LLMOutput,
    Side,
    TradingState,
)
from tests.conftest import (
    BAR_COUNT,
    HEALTHY,
    PRESET,
    START,
    STEP,
    FakeLLM,
    FakeMarketClient,
    brief_payload,
    drifting_closes,
    flat_closes,
    raw_ohlcv,
    role_map,
    verdict_payload,
)

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig


def make_settings(risk: RiskLimits | None = None) -> Settings:
    """Configuración válida con ambas mesas en familias distintas."""
    return load_settings(
        roles=role_map(),
        ollama={"host": "http://localhost:11434"},
        risk=risk or RiskLimits(),
    )


class BrokenLLM(FakeLLM):
    """Falla el transporte para un agente concreto y responde normal al resto.

    Reproduce lo que hace un gateway que no sirve uno de los seis ids: los demás
    modelos contestan y ese revienta antes de producir contenido.
    """

    def __init__(self, target: str, error: Exception) -> None:
        super().__init__()
        self.broken = target
        self.error = error

    async def complete(self, choice: ModelChoice, prompt: str, schema: type) -> str:
        """Lanza si el prompt es del agente roto; si no, delega en el falso normal."""
        if self._target(prompt, schema) == self.broken:
            raise self.error
        return await super().complete(choice, prompt, schema)


def make_context(
    *,
    closes: list[float] | None = None,
    overrides: dict[str, str] | None = None,
    cache: InMemoryResponseCache | None = None,
    risk: RiskLimits | None = None,
    account: AccountState = HEALTHY,
    backend: FakeLLM | None = None,
) -> tuple[AgentContext, FakeLLM]:
    """Contexto completo con mercado, modelos, cuenta y ejecutor falsos."""
    series = closes if closes is not None else [*flat_closes(PRESET.min_bars), 110.0]
    settings = make_settings(risk)
    clock = lambda: START  # noqa: E731  # reloj fijo: hace determinista el `at` de cada LLMCall
    backend = backend if backend is not None else FakeLLM(overrides)
    ledger = QuotaLedger(settings.quota_window, clock)
    router = ModelRouter(settings, ledger, {Backend.OLLAMA: backend}, clock, cache)

    context = AgentContext(
        settings=settings,
        router=router,
        market=FakeMarketClient(raw_ohlcv(series)),
        account=account,
        run_id=uuid4(),
        symbol="BTC/USDT",
        timeframe="1h",
        executor=PaperExecutor(),
        journal=InMemoryJournal(),
        candle_limit=500,
        preset=PRESET,
        activation=ActivationConfig(preset=PRESET),
        clock=clock,
        now=START + (len(series) + 1) * STEP,
    )
    return context, backend


def thread_config(thread: str) -> RunnableConfig:
    """Configuración de hilo para el checkpointer."""
    return {"configurable": {"thread_id": thread}}


async def run(context: AgentContext, thread: str = "t1") -> TradingState:
    """Ejecuta el grafo sobre un hilo de checkpoint y devuelve el estado final.

    El grafo es asíncrono de punta a punta: el cliente de mercado usa
    `ccxt.async_support` y el router es async, así que se invoca con `ainvoke`.
    La salida cruda solo trae los canales que algún nodo escribió, así que se
    revalida contra `TradingState` para que los ausentes tomen su valor por
    defecto en vez de faltar.
    """
    graph = build_graph()
    raw = await graph.ainvoke(TradingState(), context=context, config=thread_config(thread))
    return TradingState.model_validate(raw)


# ─────────────────────────────────────── Extremo a extremo ────────────────────────────────────────


@pytest.mark.asyncio
async def test_graph_runs_end_to_end() -> None:
    """Del mercado a la decisión sin tocar red ni proveedor real."""
    context, _ = make_context()
    result = await run(context)

    assert result.errors == []
    assert result.snapshot is not None
    assert result.evidence is not None
    assert result.decision is not None
    assert result.decision.action is Action.BUY


@pytest.mark.asyncio
async def test_three_verdicts_survive_the_parallel_fan_out() -> None:
    """El reducer conserva los tres veredictos técnicos.

    Sin él se decidiría con un tercio de la evidencia; este es el test que lo
    demuestra sobre el grafo real, no sobre nodos de juguete.
    """
    context, _ = make_context()
    result = await run(context)

    verdicts = result.verdicts
    assert len(verdicts) == 3
    assert {item.dimension for item in verdicts} == set(Dimension)


@pytest.mark.asyncio
async def test_both_desks_produce_a_brief() -> None:
    """Las dos mesas escriben y el reducer conserva ambos alegatos."""
    context, _ = make_context()
    result = await run(context)

    briefs = result.briefs
    assert {item.side for item in briefs} == {Side.BULL, Side.BEAR}


@pytest.mark.asyncio
async def test_every_model_call_is_recorded() -> None:
    """Tres técnicos, dos mesas y el decisor: seis llamadas con rastro."""
    context, _ = make_context()
    result = await run(context)

    calls = result.calls
    assert len(calls) == 6
    assert sum(call.quota_weight for call in calls if not call.cache_hit) == 6.0


@pytest.mark.asyncio
async def test_both_desks_see_identical_evidence() -> None:
    """Las mesas argumentan sobre la misma evidencia; solo cambia el lado pedido."""
    context, backend = make_context()
    result = await run(context)

    evidence = result.evidence
    assert evidence is not None
    rendered = format_verdicts(evidence.verdicts)
    indicators = format_indicators(evidence.indicators)

    prompts = dict(backend.prompts)
    bull, bear = prompts["bull"], prompts["bear"]
    assert rendered in bull
    assert rendered in bear
    assert indicators in bull
    assert indicators in bear
    assert 'literalmente `"bull"`' in bull
    assert 'literalmente `"bear"`' in bear


# ────────────────────────────────────────── El gate corta ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_closed_gate_ends_before_spending_any_quota() -> None:
    """Sin cambio material el grafo termina sin llamar a ningún modelo.

    La serie lleva deriva mínima en vez de ser plana: una serie perfectamente
    plana no tiene movimiento direccional, el ADX sale NaN y el fallo ocurriría
    en el cálculo de indicadores en vez de en el gate, que es lo que se prueba.
    """
    context, backend = make_context(closes=drifting_closes(BAR_COUNT))
    result = await run(context)

    activation = result.activation
    assert activation is not None
    assert activation.should_run is False
    assert result.decision is None
    assert result.calls == []
    assert backend.prompts == []


@pytest.mark.asyncio
async def test_market_failure_stops_before_the_agents() -> None:
    """Un fallo de datos se registra y no gasta cuota."""
    context, backend = make_context(closes=flat_closes(10))
    result = await run(context)

    errors = result.errors
    assert errors[0].node == "prepare_market_data"
    assert backend.prompts == []


# ──────────────────────────────────────── Fallos de agente ────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_failing_technical_agent_aborts_the_evaluation() -> None:
    """Con dos tercios de la evidencia no se decide: se aborta."""
    context, _ = make_context(overrides={"momentum": "{}"})
    result = await run(context)

    assert result.decision is None
    assert result.evidence is None
    assert result.briefs == []
    nodes = {error.node for error in result.errors}
    assert "momentum" in nodes
    assert "consolidate_evidence" in nodes


class FirstTryWrongLLM(FakeLLM):
    """Responde mal la primera vez a un agente concreto, y bien a partir de ahí.

    Es lo que hace un modelo de verdad cuando se le devuelve el error: la
    corrección solo se puede comprobar si el segundo intento es distinto del
    primero.
    """

    def __init__(self, target: str, wrong: str) -> None:
        super().__init__()
        self.target = target
        self.wrong = wrong
        self.served = False

    async def complete(self, choice: ModelChoice, prompt: str, schema: type) -> str:
        """El payload equivocado una sola vez; después, el normal."""
        if not self.served and self._target(prompt, schema) == self.target:
            self.served = True
            self.prompts.append((self.target, prompt))
            return self.wrong
        return await super().complete(choice, prompt, schema)


class TrustingRouter(ModelRouter):
    """Router que ignora la validación de contexto. Solo existe para probar la aserción."""

    async def invoke[T: LLMOutput](
        self,
        role: AgentRole,
        prompt: str,
        schema: type[T],
        check: ContextCheck[T] | None = None,
    ) -> tuple[T, list[LLMCall]]:
        """Llama sin la validación que el nodo le pasó."""
        del check
        return await super().invoke(role, prompt, schema)


def context_failures(result: TradingState, role: AgentRole) -> list[LLMCall]:
    """Intentos de ese rol rechazados por contexto."""
    return [
        call
        for call in result.calls
        if call.role is role and call.failure_kind is FailureKind.CONTEXT
    ]


CONTEXT_CASES = [
    ("bull", AgentRole.BULL, brief_payload(Side.BULL, grounded_in="volume-9"), "volume-9"),
    ("bull", AgentRole.BULL, brief_payload(Side.BEAR), "respondiste como bear"),
    (
        "volume",
        AgentRole.VOLUME,
        verdict_payload(Dimension.VOLUME, cites="ICHIMOKU_9"),
        "ICHIMOKU_9",
    ),
    (
        "volume",
        AgentRole.VOLUME,
        verdict_payload(Dimension.MOMENTUM),
        "respondiste como momentum",
    ),
]
CONTEXT_IDS = ["ids inventados", "mesa contraria", "indicador inexistente", "otra dimension"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("node", "role", "wrong", "fragment"), CONTEXT_CASES, ids=CONTEXT_IDS)
async def test_a_context_failure_is_recorded_and_aborts_when_it_persists(
    node: str, role: AgentRole, wrong: str, fragment: str
) -> None:
    """El modelo insiste en una salida que contradice su contexto: dos intentos, los dos inválidos.

    Antes la llamada quedaba como válida y el fallo era un `NodeError` suelto. Ahora
    cada intento dice por qué se rechazó, se cobró, y el nodo aborta con el error
    del router: la salida nunca llega a ser un veredicto ni un alegato.
    """
    context, _ = make_context(overrides={node: wrong})
    result = await run(context)

    rejected = context_failures(result, role)
    assert len(rejected) == 2, [call.failure_kind for call in result.calls]
    assert all(not call.valid and fragment in (call.failure_message or "") for call in rejected)
    assert result.decision is None
    error = next(error for error in result.errors if error.node == node)
    assert "sin salida válida" in error.message
    assert CONTEXT_BYPASSED not in error.message
    assert all(brief.side is not Side.BULL for brief in result.briefs) or node != "bull"


@pytest.mark.asyncio
@pytest.mark.parametrize(("node", "role", "wrong", "fragment"), CONTEXT_CASES, ids=CONTEXT_IDS)
async def test_a_context_failure_is_corrected_by_the_retry(
    node: str, role: AgentRole, wrong: str, fragment: str
) -> None:
    """Con el error delante, el segundo intento corrige y la evaluación llega a decidir.

    Es lo que la comprobación en el nodo no permitía: un id inventado en la mesa
    alcista tiraba la evaluación entera sin darle al modelo la ocasión de citar uno
    que existiera.
    """
    backend = FirstTryWrongLLM(node, wrong)
    context, _ = make_context(backend=backend)
    result = await run(context)

    assert len(context_failures(result, role)) == 1
    assert result.decision is not None
    assert not result.errors
    retry = [prompt for target, prompt in backend.prompts if target == node][1]
    assert fragment in retry


@pytest.mark.asyncio
@pytest.mark.parametrize(("node", "role", "wrong", "fragment"), CONTEXT_CASES, ids=CONTEXT_IDS)
async def test_a_node_still_refuses_what_the_router_should_have_rejected(
    node: str, role: AgentRole, wrong: str, fragment: str
) -> None:
    """La comprobación del nodo se queda como aserción: si se dispara, es un bug y se registra.

    Con un router que no aplica la validación, la salida inválida llega al nodo
    marcada como válida. No entra en el estado: el nodo la rechaza con un error
    que dice qué pasó, en vez de dejarla seguir hasta el decisor.
    """
    context, _ = make_context(overrides={node: wrong})
    router = TrustingRouter(
        context.settings,
        QuotaLedger(context.settings.quota_window, context.clock),
        {Backend.OLLAMA: FakeLLM({node: wrong})},
        context.clock,
    )
    result = await run(replace(context, router=router))

    assert context_failures(result, role) == []
    assert result.decision is None
    error = next(error for error in result.errors if error.node == node)
    assert error.message.startswith(CONTEXT_BYPASSED)
    assert fragment in error.message


# ──────────────────────────────────── Caché y checkpointing ───────────────────────────────────────


@pytest.mark.asyncio
async def test_repeating_an_evaluation_hits_the_cache() -> None:
    """El mismo input no gasta cuota dos veces."""
    cache = InMemoryResponseCache()
    first_context, first_backend = make_context(cache=cache)
    await run(first_context, thread="a")

    second_context, second_backend = make_context(cache=cache)
    result = await run(second_context, thread="b")

    assert len(first_backend.prompts) == 6
    assert second_backend.prompts == []
    calls = result.calls
    assert all(call.cache_hit for call in calls)
    assert sum(call.quota_weight for call in calls if not call.cache_hit) == 0.0


@pytest.mark.asyncio
async def test_checkpointer_keeps_the_final_state() -> None:
    """Checkpointing activo desde el inicio: el estado se recupera por hilo."""
    context, _ = make_context()
    graph = build_graph()
    config = thread_config("persistente")
    await graph.ainvoke(TradingState(), context=context, config=config)

    snapshot = graph.get_state(config)
    assert len(snapshot.values["verdicts"]) == 3
    assert snapshot.values["decision"] is not None


@pytest.mark.asyncio
async def test_run_id_travels_into_the_snapshot() -> None:
    """La identidad de la corrida llega al estado desde el contexto."""
    context, _ = make_context()
    result = await run(context)

    assert result.snapshot is not None
    assert result.snapshot.run_id == context.run_id
    assert result.snapshot.candles_count == BAR_COUNT
    assert (
        result.snapshot.timestamp == datetime(2026, 8, 1, 0, 0, tzinfo=UTC) + (BAR_COUNT - 1) * STEP
    )


# ─────────────────────────────────────── Riesgo y salida ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_graph_reaches_an_order_in_paper_mode() -> None:
    """Del mercado a la orden, sin tocar el mercado real."""
    context, _ = make_context()
    result = await run(context)

    assert result.risk is not None
    assert result.risk.approved is True
    assert result.order is not None
    assert result.order.mode is ExecutionMode.PAPER
    executor = context.executor
    assert isinstance(executor, PaperExecutor)
    assert executor.submitted == [result.order]


@pytest.mark.asyncio
async def test_the_graph_never_sends_the_size_the_model_asked_for() -> None:
    """El decisor pide 0.25; el límite por operación es 0.10 y eso es lo que sale."""
    context, _ = make_context(risk=RiskLimits(max_position_fraction=0.10))
    result = await run(context)

    assert result.decision is not None
    assert result.decision.size_fraction == 0.25
    assert result.order is not None
    assert result.order.size_fraction == 0.10
    assert result.risk is not None
    assert "max_position_fraction" in result.risk.applied_limits


@pytest.mark.asyncio
async def test_kill_switch_stops_the_order_but_not_the_evaluation() -> None:
    """Con el interruptor puesto se analiza igual, pero no sale nada al mercado."""
    context, _ = make_context(risk=RiskLimits(kill_switch=True))
    result = await run(context)

    assert result.decision is not None
    assert result.risk is not None
    assert result.risk.approved is False
    assert result.order is None
    executor = context.executor
    assert isinstance(executor, PaperExecutor)
    assert executor.submitted == []


@pytest.mark.asyncio
async def test_cooldown_blocks_the_order() -> None:
    """Una pérdida reciente veta la operación aunque el debate sea concluyente."""
    account = AccountState(
        equity=10_000.0, day_start_equity=10_000.0, last_loss_at=START - timedelta(minutes=10)
    )
    context, _ = make_context(account=account)
    result = await run(context)

    assert result.risk is not None
    assert "cooldown" in (result.risk.veto_reason or "")
    assert result.order is None


@pytest.mark.asyncio
async def test_drawdown_blocks_the_order() -> None:
    """Perdido el drawdown del día no se opera."""
    account = AccountState(equity=9_000.0, day_start_equity=10_000.0)
    context, _ = make_context(account=account)
    result = await run(context)

    assert result.risk is not None
    assert "drawdown" in (result.risk.veto_reason or "")
    assert result.order is None


# ────────────────────────────────────────── Journal ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_every_evaluation_is_journaled() -> None:
    """Una evaluación completa deja un registro con todo lo que ocurrió."""
    context, _ = make_context()
    result = await run(context)

    journal = context.journal
    assert isinstance(journal, InMemoryJournal)
    assert len(journal) == 1
    record = journal.records[0]
    assert record.run_id == context.run_id
    assert record.decision == result.decision
    assert record.risk == result.risk
    assert record.order == result.order
    assert record.quota_used == 6.0
    assert record.traded is True


@pytest.mark.asyncio
async def test_a_closed_gate_is_journaled_too() -> None:
    """La evaluación que no operó es justo la que se querrá auditar."""
    context, _ = make_context(closes=drifting_closes(BAR_COUNT))
    await run(context)

    journal = context.journal
    assert isinstance(journal, InMemoryJournal)
    assert len(journal) == 1
    record = journal.records[0]
    assert record.traded is False
    assert record.activation is not None
    assert record.activation.should_run is False
    assert record.quota_used == 0.0


@pytest.mark.asyncio
async def test_an_aborted_evaluation_is_journaled_with_its_errors() -> None:
    """Un fallo de agente deja registro del error, no un hueco."""
    context, _ = make_context(overrides={"momentum": "{}"})
    await run(context)

    journal = context.journal
    assert isinstance(journal, InMemoryJournal)
    record = journal.records[0]
    assert record.decision is None
    assert record.traded is False
    assert {error.node for error in record.errors} >= {"momentum", "consolidate_evidence"}


@pytest.mark.asyncio
async def test_a_transport_failure_is_journaled_with_snapshot_and_activation() -> None:
    """Un fallo del proveedor a mitad del abanico técnico no puede vaciar el registro.

    Es la corrida que no dejó rastro: la excepción del adaptador subía por el
    grafo entero y solo quedaba la línea que escribe el runner, firmada por él y
    sin la vela, sin el gate y sin las llamadas que ya se habían pagado. Con eso
    no hay forma de saber si falló el mercado, el gate o el proveedor.
    """
    backend = BrokenLLM("momentum", RuntimeError("400 model not found"))
    context, _ = make_context(backend=backend)

    await run(context)

    journal = context.journal
    assert isinstance(journal, InMemoryJournal)
    record = journal.records[0]
    assert record.snapshot is not None
    assert record.activation is not None and record.activation.should_run is True
    assert record.decision is None and record.traded is False

    failure = next(error for error in record.errors if error.node == "momentum")
    assert "400 model not found" in failure.message


@pytest.mark.asyncio
async def test_a_transport_failure_journals_the_call_it_paid_for() -> None:
    """La fila del intento fallido tiene que llegar al archivo, no solo al contador.

    El `QuotaLedger` la anota, pero vive en memoria y muere con el proceso. En el
    journal es donde alguien va a buscar por qué no se operó tres días después.
    """
    backend = BrokenLLM("momentum", RuntimeError("401 unauthorized"))
    context, _ = make_context(backend=backend)

    await run(context)

    journal = context.journal
    assert isinstance(journal, InMemoryJournal)
    record = journal.records[0]

    failed = [call for call in record.calls if not call.valid]
    assert len(failed) == 1
    assert failed[0].role is AgentRole.MOMENTUM
    assert failed[0].failure_kind is FailureKind.TRANSPORT
    assert "401 unauthorized" in (failed[0].failure_message or "")


@pytest.mark.asyncio
async def test_a_vetoed_evaluation_journals_the_reason() -> None:
    """El veto queda registrado con su causa."""
    context, _ = make_context(risk=RiskLimits(kill_switch=True))
    await run(context)

    journal = context.journal
    assert isinstance(journal, InMemoryJournal)
    record = journal.records[0]
    assert record.risk is not None
    assert record.risk.veto_reason == "kill switch activo"
    assert record.traded is False
