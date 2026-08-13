"""Pruebas del bucle de operación.

El reloj y la espera son falsos: el runner se alinea a cierres de vela de horas,
así que con `asyncio.sleep` real ninguna de estas pruebas terminaría. El falso
adelanta el reloj en vez de esperar, que es lo que hace que un ciclo de 4h corra
en microsegundos.

Ninguna prueba toca la red ni un proveedor: el grafo es el de verdad, y lo que se
sustituye por debajo son el mercado y el backend de modelos.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from crypto_agents.activation import ActivationConfig
from crypto_agents.execution import PaperExecutor
from crypto_agents.graph import AgentContext, build_graph
from crypto_agents.journal import InMemoryJournal
from crypto_agents.llm import ModelRouter
from crypto_agents.quota import QuotaLedger
from crypto_agents.runner import RUNNER_NODE, Runner, RunnerSettings
from crypto_agents.settings import Backend, ModelChoice, RoleConfig, Settings, load_settings
from crypto_agents.state import AgentRole
from tests.conftest import (
    HEALTHY,
    PRESET,
    START,
    STEP,
    FakeLLM,
    FakeMarketClient,
    flat_closes,
    raw_ohlcv,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from uuid import UUID

    from crypto_agents.journal import EvaluationRecord

SERIES = [*flat_closes(PRESET.min_bars), 110.0]
"""Serie que enciende el gate de activación: plana y un salto al final."""

MARKET_NOW = START + (len(SERIES) + 1) * STEP


class FakeClock:
    """Reloj controlado por la prueba."""

    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        """Instante actual."""
        return self.now

    def advance(self, delta: timedelta) -> None:
        """Mueve el reloj hacia adelante."""
        self.now += delta


class FakeSleeper:
    """Espera que adelanta el reloj en vez de bloquear.

    Sin esto, `_sleep_until` no avanzaría nunca: el reloj es falso y el sleep real
    no lo mueve. El gancho `on_sleep` permite a una prueba pedir el apagado en
    mitad de la espera.
    """

    def __init__(self, clock: FakeClock, on_sleep: Callable[[int], None] | None = None) -> None:
        self.clock = clock
        self.slept: list[float] = []
        self.on_sleep = on_sleep

    async def __call__(self, seconds: float) -> None:
        """Anota, adelanta el reloj y devuelve el control."""
        self.slept.append(seconds)
        self.clock.advance(timedelta(seconds=seconds))
        if self.on_sleep is not None:
            self.on_sleep(len(self.slept))


def local(model: str, family: str, quota: int = 1000) -> ModelChoice:
    """Modelo servido por el backend local."""
    return ModelChoice(backend=Backend.OLLAMA, model=model, family=family, quota_per_window=quota)


def remote(model: str, family: str, quota: int) -> ModelChoice:
    """Modelo servido por el backend remoto."""
    return ModelChoice(backend=Backend.OPENAI, model=model, family=family, quota_per_window=quota)


def make_settings(
    structure: RoleConfig | None = None, decider: RoleConfig | None = None
) -> Settings:
    """Configuración generosa salvo en los roles que la prueba quiera estrechar."""
    roles = {
        AgentRole.STRUCTURE: structure or RoleConfig(primary=local("local-a", "fam-a")),
        AgentRole.MOMENTUM: RoleConfig(primary=local("local-b", "fam-b")),
        AgentRole.VOLUME: RoleConfig(primary=local("local-c", "fam-c")),
        AgentRole.BULL: RoleConfig(primary=local("local-bull", "fam-bull")),
        AgentRole.BEAR: RoleConfig(primary=local("local-bear", "fam-bear")),
        AgentRole.DECIDER: decider or RoleConfig(primary=local("local-dec", "fam-dec")),
    }
    return load_settings(
        roles=roles,
        openai={"api_key": "sk-test"},
        ollama={"host": "http://localhost:11434"},
        _env_file=None,
    )


class Harness:
    """Runner cableado sobre un grafo real, con mercado y modelos falsos.

    El `ModelRouter` —y por tanto el `QuotaLedger`— se construye una sola vez y lo
    comparten todos los símbolos: es exactamente la invariante que las pruebas de
    abajo comprueban.
    """

    def __init__(
        self,
        symbols: tuple[str, ...] = ("BTC/USDT", "ETH/USDT"),
        timeframe: str = "1h",
        settings: Settings | None = None,
        max_concurrent: int = 1,
        settle_seconds: float = 0.0,
        start: datetime = datetime(2026, 8, 13, 11, 30, tzinfo=UTC),
        on_context: Callable[[str], None] | None = None,
    ) -> None:
        self.settings = settings or make_settings()
        self.clock = FakeClock(start)
        self.backend = FakeLLM()
        self.ledger = QuotaLedger(self.settings, self.clock)
        self.router = ModelRouter(
            self.settings,
            self.ledger,
            {Backend.OLLAMA: self.backend, Backend.OPENAI: self.backend},
            self.clock,
        )
        self.journal = InMemoryJournal()
        self.executor = PaperExecutor()
        self.contexts: list[str] = []
        self.on_context = on_context

        self.config = RunnerSettings(
            symbols=symbols,
            timeframe=timeframe,
            max_concurrent=max_concurrent,
            settle_seconds=settle_seconds,
        )
        self.sleeper = FakeSleeper(self.clock)
        self.runner = Runner(
            config=self.config,
            router=self.router,
            build_context=self.build_context,
            graph=build_graph(),
            journal=self.journal,
            clock=self.clock,
            sleep=self.sleeper,
        )

    def build_context(self, symbol: str, run_id: UUID, router: ModelRouter) -> AgentContext:
        """Fábrica de contexto: recibe el router compartido, no construye uno propio."""
        self.contexts.append(symbol)
        if self.on_context is not None:
            self.on_context(symbol)
        return AgentContext(
            settings=self.settings,
            router=router,
            market=FakeMarketClient(raw_ohlcv(SERIES)),
            account=HEALTHY,
            run_id=run_id,
            symbol=symbol,
            timeframe=self.config.timeframe,
            executor=self.executor,
            journal=self.journal,
            preset=PRESET,
            activation=ActivationConfig(preset=PRESET),
            clock=self.clock,
            now=MARKET_NOW,
        )

    @property
    def records(self) -> list[EvaluationRecord]:
        """Lo registrado hasta ahora."""
        return self.journal.records


def runner_errors(records: list[EvaluationRecord]) -> list[EvaluationRecord]:
    """Registros firmados por el runner, no por un nodo del grafo."""
    return [
        record for record in records if any(error.node == RUNNER_NODE for error in record.errors)
    ]


# ───────────────────────────────────────── Calendario ─────────────────────────────────────────────


@pytest.mark.parametrize(
    ("timeframe", "now", "expected"),
    [
        ("1h", datetime(2026, 8, 13, 11, 30, tzinfo=UTC), datetime(2026, 8, 13, 12, 0, tzinfo=UTC)),
        ("4h", datetime(2026, 8, 13, 11, 30, tzinfo=UTC), datetime(2026, 8, 13, 12, 0, tzinfo=UTC)),
        ("4h", datetime(2026, 8, 13, 12, 1, tzinfo=UTC), datetime(2026, 8, 13, 16, 0, tzinfo=UTC)),
        ("1d", datetime(2026, 8, 13, 12, 0, tzinfo=UTC), datetime(2026, 8, 14, 0, 0, tzinfo=UTC)),
    ],
)
def test_next_close_aligns_to_absolute_candle_boundaries(
    timeframe: str, now: datetime, expected: datetime
) -> None:
    """El calendario se alinea a la vela, no al momento de arrancar el proceso.

    Dos runners lanzados con minutos de diferencia deben evaluar las mismas velas;
    si el calendario contara desde el arranque, cada proceso miraría un instante
    distinto y dos corridas dejarían de ser comparables.
    """
    harness = Harness(timeframe=timeframe, start=now)
    assert harness.runner.next_close(now) == expected


def test_settle_margin_delays_the_evaluation_past_the_close() -> None:
    """El exchange no publica la vela cerrada en el instante exacto del cierre."""
    now = datetime(2026, 8, 13, 11, 30, tzinfo=UTC)
    harness = Harness(timeframe="1h", start=now, settle_seconds=30.0)
    assert harness.runner.next_close(now) == datetime(2026, 8, 13, 12, 0, 30, tzinfo=UTC)


def test_starting_mid_period_waits_instead_of_evaluating_a_stale_candle() -> None:
    """Arrancar a mitad de vela espera al próximo cierre."""
    now = datetime(2026, 8, 13, 11, 59, tzinfo=UTC)
    harness = Harness(timeframe="1h", start=now)
    assert harness.runner.next_close(now) > now


# ──────────────────────────────────── Contador compartido ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_quota_ledger_is_shared_across_symbols() -> None:
    """Un contador por símbolo multiplicaría el presupuesto real por el número de símbolos.

    Con N símbolos y un contador cada uno, cada contador se creería dentro de
    límite mientras el consumo agregado es N veces el declarado. Aquí las dos
    evaluaciones suman en el mismo sitio.
    """
    harness = Harness(symbols=("BTC/USDT", "ETH/USDT"))
    await harness.runner.run_cycle()

    assert harness.ledger.used(AgentRole.DECIDER, "local-dec") == 2.0
    assert harness.ledger.used(AgentRole.STRUCTURE, "local-a") == 2.0


@pytest.mark.asyncio
async def test_what_the_first_symbol_spends_degrades_the_second() -> None:
    """La prueba de que el contador es uno: el gasto de un símbolo afecta al siguiente.

    El rol de estructura tiene sitio para una sola llamada remota. El primer
    símbolo la consume y el segundo sale ya por el respaldo local. Con contadores
    separados, ambos habrían salido por el modelo remoto.
    """
    settings = make_settings(
        structure=RoleConfig(
            primary=remote("remote-a", "fam-a", quota=1),
            fallback=local("local-a-alt", "fam-a-alt"),
        )
    )
    harness = Harness(symbols=("BTC/USDT", "ETH/USDT"), settings=settings)
    await harness.runner.run_cycle()

    used = [
        (call.model, call.backend)
        for record in harness.records
        for call in record.calls
        if call.role is AgentRole.STRUCTURE
    ]
    assert used == [("remote-a", Backend.OPENAI), ("local-a-alt", Backend.OLLAMA)]


# ─────────────────────────────── Degradación y abort del decisor ──────────────────────────────────


@pytest.mark.asyncio
async def test_exhausting_the_decider_aborts_and_records_instead_of_degrading() -> None:
    """Sin presupuesto de decisor la evaluación muere con causa, no decide peor.

    Es el único rol sin respaldo: degradarlo cambiaría quién toma la decisión
    final sin que eso apareciera en ninguna parte antes de que la orden ya
    estuviera puesta. El segundo símbolo llega hasta el debate, se queda sin
    decisor y termina en el journal sin decisión y sin orden.
    """
    settings = make_settings(decider=RoleConfig(primary=local("local-dec", "fam-dec", quota=1)))
    harness = Harness(symbols=("BTC/USDT", "ETH/USDT"), settings=settings)
    await harness.runner.run_cycle()

    first, second = harness.records
    assert first.decision is not None
    assert second.decision is None
    assert second.order is None
    assert second.evidence is not None, "el abort ocurre en el decisor, no antes"
    assert [error.node for error in second.errors] == ["decide"]
    assert "cuota agotada" in second.errors[0].message


# ──────────────────────────────────────── Ciclo completo ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_every_symbol_leaves_a_record() -> None:
    """Un ciclo deja una línea por símbolo, haya operado o no."""
    harness = Harness(symbols=("BTC/USDT", "ETH/USDT", "SOL/USDT"))
    await harness.runner.run_cycle()

    assert [record.symbol for record in harness.records] == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    assert all(record.traded for record in harness.records)


@pytest.mark.asyncio
async def test_a_broken_symbol_does_not_take_down_the_cycle() -> None:
    """Un fallo antes de que el grafo arranque deja línea y no arrastra a los demás.

    Sin registro, el hueco sería indistinguible de un símbolo que nadie pidió
    evaluar.
    """

    def explode(symbol: str) -> None:
        if symbol == "ETH/USDT":
            raise RuntimeError("no hay mercado para este símbolo")

    harness = Harness(symbols=("BTC/USDT", "ETH/USDT"), on_context=explode)
    await harness.runner.run_cycle()

    assert len(harness.records) == 2
    failed = runner_errors(harness.records)
    assert [record.symbol for record in failed] == ["ETH/USDT"]
    assert "RuntimeError" in failed[0].errors[0].message


@pytest.mark.asyncio
async def test_symbols_run_one_at_a_time_by_default() -> None:
    """`max_concurrent=1`: Ollama serializa por GPU, solaparse solo haría cola."""
    live: list[str] = []
    peak = 0

    def enter(symbol: str) -> None:
        nonlocal peak
        live.append(symbol)
        peak = max(peak, len(live))
        live.clear()

    harness = Harness(symbols=("BTC/USDT", "ETH/USDT", "SOL/USDT"), on_context=enter)
    await harness.runner.run_cycle()

    assert peak == 1


# ───────────────────────────────────────── Apagado limpio ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_stop_before_the_first_cycle_evaluates_nothing() -> None:
    """Pedir el apagado antes de empezar no arranca ninguna evaluación."""
    harness = Harness()
    harness.runner.stop()
    await harness.runner.run_forever()

    assert harness.records == []


@pytest.mark.asyncio
async def test_stop_during_the_wait_exits_without_evaluating() -> None:
    """El apagado se atiende durante la espera, sin abrir un ciclo nuevo."""
    harness = Harness(start=datetime(2026, 8, 13, 11, 30, tzinfo=UTC))
    harness.sleeper.on_sleep = lambda count: harness.runner.stop() if count >= 2 else None
    await harness.runner.run_forever()

    assert harness.records == []
    assert harness.sleeper.slept, "debió esperar antes de salir"


@pytest.mark.asyncio
async def test_a_cycle_in_flight_finishes_and_is_recorded() -> None:
    """`stop()` a mitad de ciclo no deja trabajo a medias sin registrar."""
    harness = Harness(symbols=("BTC/USDT", "ETH/USDT"))
    runner = harness.runner
    harness.on_context = lambda symbol: runner.stop() if symbol == "BTC/USDT" else None

    await harness.runner.run_forever()

    assert [record.symbol for record in harness.records] == ["BTC/USDT", "ETH/USDT"]


# ──────────────────────────────────────── Retraso y saltos ────────────────────────────────────────


@pytest.mark.asyncio
async def test_an_overrun_skips_missed_closes_instead_of_piling_up() -> None:
    """Un ciclo que se pasa de largo salta los cierres perdidos y los registra.

    Encadenar ciclos atrasados haría que el sistema operase sobre velas viejas
    creyendo que va al día.
    """
    cycles = 0
    harness = Harness(symbols=("BTC/USDT",), timeframe="1h")

    def slow(symbol: str) -> None:
        nonlocal cycles
        cycles += 1
        harness.clock.advance(timedelta(hours=2))  # el ciclo dura más que el periodo
        if cycles >= 2:
            harness.runner.stop()

    harness.on_context = slow
    await harness.runner.run_forever()

    skipped = runner_errors(harness.records)
    assert len(skipped) == 1
    assert "ciclo omitido" in skipped[0].errors[0].message
    assert "1 cierre de 1h" in skipped[0].errors[0].message


@pytest.mark.asyncio
async def test_a_skipped_cycle_is_recorded_once_per_symbol() -> None:
    """La pregunta posterior es cuántas evaluaciones se perdió un mercado concreto."""
    cycles = 0
    harness = Harness(symbols=("BTC/USDT", "ETH/USDT"), timeframe="1h")

    def slow(symbol: str) -> None:
        nonlocal cycles
        cycles += 1
        harness.clock.advance(timedelta(hours=2))
        if cycles >= 4:  # dos símbolos por ciclo
            harness.runner.stop()

    harness.on_context = slow
    await harness.runner.run_forever()

    skipped = runner_errors(harness.records)
    assert sorted(record.symbol for record in skipped) == ["BTC/USDT", "ETH/USDT"]
    assert len({record.run_id for record in skipped}) == 1, "un solo ciclo omitido"


# ──────────────────────────────────────── Configuración ───────────────────────────────────────────


def test_duplicate_symbols_are_rejected() -> None:
    """Un símbolo repetido gasta cuota dos veces por el mismo mercado."""
    with pytest.raises(ValueError, match="repetidos: BTC/USDT"):
        RunnerSettings(symbols=("BTC/USDT", "ETH/USDT", "BTC/USDT"))


def test_an_invalid_timeframe_fails_at_configuration_time() -> None:
    """Un timeframe inválido debe fallar al configurar, no en el primer ciclo."""
    with pytest.raises(ValueError, match="timeframe inválido"):
        RunnerSettings(symbols=("BTC/USDT",), timeframe="4x")


def test_symbols_cannot_be_empty() -> None:
    """Un runner sin símbolos no tiene nada que evaluar."""
    with pytest.raises(ValueError):
        RunnerSettings(symbols=())


def test_the_context_factory_receives_the_shared_router() -> None:
    """La firma es lo que impide un contador por símbolo, no la disciplina."""
    harness = Harness()
    context = harness.build_context("BTC/USDT", uuid4(), harness.router)
    assert context.router is harness.router
