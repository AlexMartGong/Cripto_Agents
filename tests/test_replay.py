"""Pruebas del arnés de replay.

Ninguna prueba toca la red. El histórico real está versionado en `tests/data/` y
los modelos son falsos; lo único que se ejerce de verdad es el grafo entero,
vela a vela.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from crypto_agents.activation import ActivationConfig
from crypto_agents.cache import InMemoryResponseCache
from crypto_agents.execution import PaperExecutor
from crypto_agents.graph import AgentContext, build_graph
from crypto_agents.journal import EvaluationRecord, InMemoryJournal
from crypto_agents.market import candles_digest, to_dataframe
from crypto_agents.metrics import summarise
from crypto_agents.quota import QuotaLedger
from crypto_agents.replay import (
    HistoricalMarketClient,
    ReplayCacheMissError,
    ReplaySettings,
    replay,
    replay_router,
    replay_run_id,
    run_digest,
)
from crypto_agents.settings import Backend, ModelChoice, RoleConfig, Settings, load_settings
from crypto_agents.state import Action, AgentRole, Decision, LLMCall, StructuredOutputMode
from tests.conftest import (
    HEALTHY,
    PRESET,
    REAL_HISTORY_DIGEST,
    FakeLLM,
    raw_ohlcv,
    real_candles,
    real_rows,
)

if TYPE_CHECKING:
    from datetime import datetime
    from uuid import UUID

    from crypto_agents.llm import ModelRouter

TIMEFRAME = "1h"


def synthetic_rows(count: int = 260) -> list[list[float]]:
    """Serie determinista con oscilación y deriva.

    Construida para que el gate dispare a menudo —hace falta que las evaluaciones
    lleguen al decisor para que haya decisiones que comparar— y sin aleatoriedad,
    para que el histórico sea el mismo en cada corrida.
    """
    closes = [100.0 + 8.0 * math.sin(index / 7.0) + 0.05 * index for index in range(count)]
    volumes = [1000.0 + 200.0 * math.sin(index / 5.0) for index in range(count)]
    return raw_ohlcv(closes, volumes=volumes)


def make_settings() -> Settings:
    """Seis roles locales en familias distintas."""
    roles = {
        role: RoleConfig(
            primary=ModelChoice(
                backend=Backend.OLLAMA,
                model=f"modelo-{role.value}",
                family=f"fam-{role.value}",
                structured_output=StructuredOutputMode.JSON_SCHEMA,
                quota_per_window=100_000,
            )
        )
        for role in AgentRole
    }
    return load_settings(roles=roles, ollama={"host": "http://localhost:11434"})


class Harness:
    """Cableado de un replay: caché compartida, router único, grafo real."""

    def __init__(self, cache: InMemoryResponseCache | None = None) -> None:
        self.settings = make_settings()
        self.cache = cache if cache is not None else InMemoryResponseCache()
        self.backend = FakeLLM()
        self.journal = InMemoryJournal()
        self.graph = build_graph()
        self.clock_moment: datetime | None = None

    def router(self, *, fill: bool) -> ModelRouter:
        """Router del replay. Con `fill=False` no puede llamar a ningún proveedor."""
        ledger = QuotaLedger(self.settings, lambda: self.clock_moment or _EPOCH)
        return replay_router(
            self.settings,
            ledger,
            self.cache,
            lambda: self.clock_moment or _EPOCH,
            fill_with={Backend.OLLAMA: self.backend} if fill else None,
        )

    def build_context(
        self,
        symbol: str,
        run_id: UUID,
        router: ModelRouter,
        market: HistoricalMarketClient,
        moment: datetime,
    ) -> AgentContext:
        """Contexto de una vela. El reloj es el instante evaluado, no el real."""
        self.clock_moment = moment
        return AgentContext(
            settings=self.settings,
            router=router,
            market=market,
            account=HEALTHY,
            run_id=run_id,
            symbol=symbol,
            timeframe=TIMEFRAME,
            executor=PaperExecutor(),
            journal=self.journal,
            preset=PRESET,
            activation=ActivationConfig(preset=PRESET),
            clock=lambda: moment,
            now=moment,
        )


_EPOCH = to_dataframe(synthetic_rows(3)).index[0].to_pydatetime()


async def run(
    harness: Harness, rows: list[list[float]], *, fill: bool, evaluations: int = 40
) -> list[EvaluationRecord]:
    """Una corrida completa sobre el histórico dado."""
    config = ReplaySettings(
        symbol="BTC/USDT",
        timeframe=TIMEFRAME,
        warmup_bars=PRESET.min_bars,
        max_evaluations=evaluations,
    )
    return await replay(
        rows, config, harness.router(fill=fill), harness.build_context, harness.graph
    )


def _record_with_calls() -> EvaluationRecord:
    """Registro mínimo con decisión y varias llamadas de roles distintos."""
    moment = real_candles().index[100].to_pydatetime()
    calls = tuple(
        LLMCall(
            role=role,
            backend=Backend.OLLAMA,
            model=f"modelo-{role.value}",
            structured_output=StructuredOutputMode.JSON_SCHEMA,
            quota_weight=1.0,
            prompt_digest=f"{index}" * 64,
            valid=True,
            latency_ms=0.0,
            at=moment,
        )
        for index, role in enumerate([AgentRole.STRUCTURE, AgentRole.MOMENTUM, AgentRole.VOLUME])
    )
    return EvaluationRecord(
        run_id=replay_run_id("BTC/USDT", "4h", moment),
        at=moment,
        symbol="BTC/USDT",
        timeframe="4h",
        decision=Decision(
            action=Action.HOLD,
            confidence=0.5,
            size_fraction=0.0,
            rationale="Ninguna mesa aporta un argumento que mueva la balanza.",
        ),
        calls=calls,
    )


# ──────────────────────────────────────── El histórico ────────────────────────────────────────────


def test_the_committed_history_still_hashes_to_its_recorded_digest() -> None:
    """Si alguien edita el CSV, dos backtests dejan de ser comparables en silencio."""
    assert candles_digest(real_candles()) == REAL_HISTORY_DIGEST


def test_the_history_is_normalised_and_ordered() -> None:
    """500 velas de 4h, cronológicas y sin huecos: `to_dataframe` lo impone."""
    frame = real_candles()
    assert len(frame) == 500
    assert frame.index.is_monotonic_increasing


# ─────────────────────────────────── No mirar hacia adelante ──────────────────────────────────────


@pytest.mark.asyncio
async def test_the_window_never_reaches_past_the_forming_candle() -> None:
    """La ventana termina en la vela evaluada más la que está en formación.

    Entregar la vela en formación es deliberado: es lo que hace el exchange, y
    esconderla en el arnés dejaría `drop_forming_candle` sin ejercitar durante todo
    un replay, que es justo donde un look-ahead saldría más caro.
    """
    rows = real_rows()
    client = HistoricalMarketClient(rows, cursor=100, limit=500)
    window = await client.fetch_ohlcv("BTC/USDT", "4h", 500)

    assert len(window) == 102  # 0..100 evaluadas, 101 en formación
    assert window[-1][0] == rows[101][0]


@pytest.mark.asyncio
async def test_the_window_respects_the_candle_limit() -> None:
    """Un histórico largo no manda mil velas al cálculo de indicadores."""
    client = HistoricalMarketClient(real_rows(), cursor=300, limit=120)
    window = await client.fetch_ohlcv("BTC/USDT", "4h", 120)
    assert len(window) == 120


# ─────────────────────────────────────── Determinismo ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_two_warm_runs_produce_identical_decisions() -> None:
    """El criterio de la fase: mismo histórico y misma caché, mismas decisiones."""
    rows = synthetic_rows()
    cache = InMemoryResponseCache()

    await run(Harness(cache), rows, fill=True)  # llena la caché
    first = await run(Harness(cache), rows, fill=False)
    second = await run(Harness(cache), rows, fill=False)

    assert [record.decision for record in first] == [record.decision for record in second]
    assert any(record.decision is not None for record in first), "sin decisiones no se prueba nada"


@pytest.mark.asyncio
async def test_two_warm_runs_produce_the_same_run_digest() -> None:
    """No solo las decisiones: el registro entero, canonicalizado, coincide.

    `run_digest` ordena las llamadas antes de hashear porque el abanico paralelo
    las escribe en orden de resolución; sin eso, dos corridas idénticas diferirían
    por algo que no es una diferencia de decisión.
    """
    rows = synthetic_rows()
    cache = InMemoryResponseCache()

    await run(Harness(cache), rows, fill=True)
    first = await run(Harness(cache), rows, fill=False)
    second = await run(Harness(cache), rows, fill=False)

    assert run_digest(first) == run_digest(second)


@pytest.mark.asyncio
async def test_a_cold_run_is_not_byte_identical_to_a_warm_one() -> None:
    """La latencia es la diferencia, y por eso el determinismo se exige con caché caliente.

    Una llamada real tarda lo que tarda; un acierto de caché fija `latency_ms` en
    cero. Documentar esto aquí evita que alguien persiga un falso no-determinismo.
    """
    rows = synthetic_rows()
    cache = InMemoryResponseCache()

    cold = await run(Harness(cache), rows, fill=True)
    warm = await run(Harness(cache), rows, fill=False)

    assert [record.decision for record in cold] == [record.decision for record in warm]
    assert {call.latency_ms for record in warm for call in record.calls} == {0.0}
    assert run_digest(cold) != run_digest(warm)


def test_the_digest_ignores_the_order_in_which_calls_landed() -> None:
    """Reordenar las llamadas de un registro no cambia su huella.

    Es lo que hace comparable una corrida caliente con una que no lo estaba: con
    la caché fría los tres técnicos tardan cosas distintas y terminan en cualquiera
    de los seis órdenes posibles, sin que ninguna decisión haya cambiado.
    """
    record = _record_with_calls()
    shuffled = record.model_copy(update={"calls": tuple(reversed(record.calls))})

    assert [call.role for call in shuffled.calls] != [call.role for call in record.calls]
    assert run_digest([record]) == run_digest([shuffled])


def test_the_digest_notices_a_different_decision() -> None:
    """Canonicalizar no puede volverse ciego a lo que sí importa."""
    record = _record_with_calls()
    assert record.decision is not None
    other = record.model_copy(
        update={"decision": record.decision.model_copy(update={"confidence": 0.11})}
    )
    assert run_digest([record]) != run_digest([other])


def test_the_digest_notices_a_missing_evaluation() -> None:
    """Una corrida más corta no puede tener la misma huella que una completa."""
    record = _record_with_calls()
    assert run_digest([record, record]) != run_digest([record])
    assert run_digest([]) != run_digest([record])


def test_run_ids_are_derived_from_the_evaluated_instant() -> None:
    """Un `uuid4()` haría diferir cada registro sin que cambiara ninguna decisión."""
    moment = real_candles().index[100].to_pydatetime()
    assert replay_run_id("BTC/USDT", "4h", moment) == replay_run_id("BTC/USDT", "4h", moment)
    assert replay_run_id("BTC/USDT", "4h", moment) != replay_run_id("ETH/USDT", "4h", moment)
    assert replay_run_id("BTC/USDT", "4h", moment) != uuid4()


# ──────────────────────────────────── Caché fría y gasto ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_replay_refuses_to_call_a_provider_by_default() -> None:
    """Un replay de 500 velas son 3000 llamadas: el accidente tiene que doler."""
    with pytest.raises(ReplayCacheMissError) as excinfo:
        await run(Harness(), synthetic_rows(), fill=False)

    assert "falta en caché" in str(excinfo.value)
    assert excinfo.value.schema == "TechnicalVerdict"


@pytest.mark.asyncio
async def test_the_fill_mode_populates_the_cache() -> None:
    """La primera pasada sobre un histórico nuevo es la que paga."""
    cache = InMemoryResponseCache()
    harness = Harness(cache)
    await run(harness, synthetic_rows(), fill=True)

    assert len(cache) > 0
    assert harness.backend.prompts, "el modo de relleno sí llama al backend"


@pytest.mark.asyncio
async def test_a_warm_replay_spends_no_quota() -> None:
    """Con la caché caliente el presupuesto no se toca: son todos aciertos."""
    cache = InMemoryResponseCache()
    await run(Harness(cache), synthetic_rows(), fill=True)
    harness = Harness(cache)
    records = await run(harness, synthetic_rows(), fill=False)

    assert harness.backend.prompts == []
    assert summarise(records).quota_used == 0.0


# ───────────────────────────────────────── Métricas ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_summary_describes_the_funnel() -> None:
    """Cada etapa cuenta por separado: activadas, consolidadas, decididas, operadas."""
    cache = InMemoryResponseCache()
    records = await run(Harness(cache), synthetic_rows(), fill=True)
    summary = summarise(records)

    assert summary.evaluations == len(records)
    assert summary.activated <= summary.evaluations
    assert summary.consolidated <= summary.activated
    assert summary.decided <= summary.consolidated
    assert summary.traded <= summary.decided
    assert sum(summary.actions.values()) == summary.decided


@pytest.mark.asyncio
async def test_the_summary_splits_quota_by_role_and_backend() -> None:
    """Las dos vistas del mismo gasto: quién lo pidió y quién lo atendió."""
    records = await run(Harness(), synthetic_rows(), fill=True)
    summary = summarise(records)

    assert set(summary.quota_by_backend) == {Backend.OLLAMA}
    assert sum(summary.quota_by_role.values()) == sum(summary.quota_by_backend.values())
    assert summary.quota_by_role[AgentRole.DECIDER] == summary.decided


@pytest.mark.asyncio
async def test_vetoes_are_grouped_by_a_stable_rule_name() -> None:
    """Con el kill switch puesto, toda decisión accionable acaba en el mismo grupo."""
    harness = Harness()
    harness.settings = harness.settings.model_copy(
        update={"risk": harness.settings.risk.model_copy(update={"kill_switch": True})}
    )
    records = await run(harness, synthetic_rows(), fill=True)
    summary = summarise(records)

    actionable = sum(
        count for action, count in summary.actions.items() if action is not Action.HOLD
    )
    assert summary.vetoes.get("kill_switch", 0) == actionable
    assert summary.traded == 0


# ────────────────────────────────── Sobre el histórico real ───────────────────────────────────────


@pytest.mark.asyncio
async def test_the_harness_runs_over_the_real_history() -> None:
    """Prueba de humo del arnés completo contra datos de mercado de verdad."""
    records = await run(Harness(), real_rows(), fill=True, evaluations=25)
    summary = summarise(records)

    assert summary.evaluations == 25
    assert summary.activated > 0, "el gate no disparó ni una vez en 25 velas reales"
    assert all(record.snapshot is not None for record in records)
    assert all(not record.errors for record in records)
