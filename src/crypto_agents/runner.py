"""Bucle de operación: un ciclo por cierre de vela, varios símbolos por ciclo.

Tres decisiones que definen el módulo:

- **El contador de cuota es uno solo para todos los símbolos.** No por disciplina
  sino por firma: el `Runner` recibe el `ModelRouter` —que contiene el ledger— y
  se lo pasa a la fábrica de contexto en cada símbolo. La fábrica no tiene de
  dónde sacar uno propio. Con un contador por símbolo, el consumo real sería N
  veces el presupuesto declarado y cada contador creería estar dentro de límite.
- **Retrasarse no acumula deuda.** Si un ciclo termina después del siguiente
  cierre, los cierres que ya pasaron se registran como omitidos y el calendario
  salta al próximo. Encadenar ciclos atrasados hace que el sistema opere sobre
  velas viejas creyendo que va al día.
- **Nada termina sin registrarse.** `stop()` deja terminar el ciclo en curso, y
  cada evaluación pasa por el nodo journal antes de que el proceso salga. Un
  fallo antes de que el grafo arranque también deja línea: si no, el hueco en el
  journal es indistinguible de un símbolo que nadie pidió evaluar.

El reloj y la espera se inyectan, igual que en `QuotaLedger`: un runner alineado
a velas de 4h con `asyncio.sleep` real solo se podría probar esperando 4 horas.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from math import ceil
from typing import TYPE_CHECKING, Self
from uuid import UUID, uuid4

from pydantic import Field, model_validator

from crypto_agents.journal import EvaluationRecord
from crypto_agents.market import MarketDataError, timeframe_to_timedelta
from crypto_agents.state import FrozenModel, NodeError, TradingState

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langgraph.graph.state import CompiledStateGraph

    from crypto_agents.context import AgentContext
    from crypto_agents.journal import Journal
    from crypto_agents.llm import ModelRouter
    from crypto_agents.quota import Clock

__all__ = ["RUNNER_NODE", "ContextFactory", "Runner", "RunnerSettings", "Sleeper"]

RUNNER_NODE = "runner"
"""Nombre con el que el runner firma sus `NodeError`.

No es un nodo del grafo. Comparte el campo porque un ciclo omitido y un fallo de
un agente se buscan en el mismo sitio cuando faltan decisiones.
"""

type ContextFactory = Callable[[str, UUID, ModelRouter], AgentContext]
"""Construye el contexto de un símbolo. Recibe el router: el ledger es compartido."""

type Sleeper = Callable[[float], Awaitable[None]]
"""Espera inyectable, en segundos. En producción, `asyncio.sleep`."""


class RunnerSettings(FrozenModel):
    """Calendario y concurrencia del bucle."""

    symbols: tuple[str, ...] = Field(min_length=1)
    timeframe: str = Field(default="4h", min_length=2)

    max_concurrent: int = Field(default=1, ge=1)
    """Símbolos evaluados a la vez.

    Uno por defecto. Ollama serializa por GPU: con cualquier rol degradado al
    respaldo local, dos símbolos en paralelo no se solapan, solo hacen cola, y la
    cola compite además con el abanico de los tres agentes técnicos.
    """

    settle_seconds: float = Field(default=5.0, ge=0.0)
    """Margen tras el cierre antes de pedir velas.

    El exchange no publica la vela cerrada en el mismo instante del cierre. Sin
    margen, `drop_forming_candle` descarta la que acaba de cerrar y la evaluación
    corre sobre la anterior.
    """

    poll_seconds: float = Field(default=1.0, gt=0.0)
    """Granularidad de la espera; acota lo que tarda `stop()` en notarse."""

    @model_validator(mode="after")
    def _timeframe_is_parseable(self) -> Self:
        """Un timeframe inválido debe fallar al configurar, no en el primer ciclo.

        Se traduce a `ValueError` para que Pydantic lo recoja como el resto de los
        validadores: `MarketDataError` es un `RuntimeError` y saldría crudo,
        saltándose el mensaje que nombra el campo.
        """
        try:
            timeframe_to_timedelta(self.timeframe)
        except MarketDataError as error:
            raise ValueError(str(error)) from error
        return self

    @model_validator(mode="after")
    def _symbols_are_unique(self) -> Self:
        """Un símbolo repetido gasta cuota dos veces por el mismo mercado."""
        duplicates = sorted({item for item in self.symbols if self.symbols.count(item) > 1})
        if duplicates:
            raise ValueError(f"símbolos repetidos: {', '.join(duplicates)}")
        return self


class Runner:
    """Evalúa una lista de símbolos en cada cierre de vela."""

    def __init__(
        self,
        config: RunnerSettings,
        router: ModelRouter,
        build_context: ContextFactory,
        graph: CompiledStateGraph[TradingState, AgentContext, TradingState, TradingState],
        journal: Journal,
        clock: Clock,
        sleep: Sleeper | None = None,
    ) -> None:
        self._config = config
        self._router = router
        self._build_context = build_context
        self._graph = graph
        self._journal = journal
        self._clock = clock
        self._sleep: Sleeper = sleep if sleep is not None else asyncio.sleep
        self._period = timeframe_to_timedelta(config.timeframe)
        self._stopping = asyncio.Event()
        self._last_target: datetime | None = None

    # ── Control ───────────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Pide el apagado. El ciclo en curso termina y se registra."""
        self._stopping.set()

    @property
    def stopping(self) -> bool:
        """Si ya se pidió el apagado."""
        return self._stopping.is_set()

    # ── Calendario ────────────────────────────────────────────────────────────

    def next_close(self, now: datetime) -> datetime:
        """Primer instante de evaluación en o después de `now`.

        Se alinea al cierre de vela en tiempo absoluto —no al arranque del
        proceso— para que dos runners lanzados con minutos de diferencia evalúen
        las mismas velas. Arrancar a mitad de periodo espera al próximo cierre en
        vez de evaluar de inmediato la vela vieja que ya está cerrada.
        """
        seconds = self._period.total_seconds()
        boundary = ceil(now.timestamp() / seconds) * seconds
        settle = timedelta(seconds=self._config.settle_seconds)
        return datetime.fromtimestamp(boundary, UTC) + settle

    def _next_target(self) -> datetime:
        """Siguiente instante de evaluación, saltando los cierres ya perdidos."""
        now = self._clock()
        if self._last_target is None:
            target = self.next_close(now)
        else:
            target = self._last_target + self._period
            skipped = 0
            while target < now:
                target += self._period
                skipped += 1
            if skipped:
                self._record_skipped(skipped, now)
        self._last_target = target
        return target

    async def _sleep_until(self, target: datetime) -> None:
        """Espera hasta `target`, en tramos, para poder atender `stop()`."""
        while not self._stopping.is_set():
            remaining = (target - self._clock()).total_seconds()
            if remaining <= 0.0:
                return
            await self._sleep(min(remaining, self._config.poll_seconds))

    # ── Ejecución ─────────────────────────────────────────────────────────────

    async def run_forever(self) -> None:
        """Bucle principal. Sale tras `stop()`, con el ciclo en curso terminado."""
        while not self._stopping.is_set():
            target = self._next_target()
            await self._sleep_until(target)
            if self._stopping.is_set():
                return
            await self.run_cycle()

    async def run_cycle(self) -> None:
        """Evalúa todos los símbolos una vez, con la concurrencia acotada.

        No se corta a mitad si llega `stop()`: lo que ya empezó termina y pasa por
        el journal. Lo observable de un ciclo es lo que quedó registrado, no lo
        que devuelve esta función.
        """
        semaphore = asyncio.Semaphore(self._config.max_concurrent)

        async def guarded(symbol: str) -> None:
            async with semaphore:
                await self._evaluate(symbol)

        await asyncio.gather(*(guarded(symbol) for symbol in self._config.symbols))

    async def _evaluate(self, symbol: str) -> None:
        """Una evaluación. El grafo ya registra por su cuenta; aquí solo el fallo previo."""
        run_id = uuid4()
        try:
            context = self._build_context(symbol, run_id, self._router)
            await self._graph.ainvoke(
                TradingState(),
                context=context,
                config={"configurable": {"thread_id": f"{symbol}:{run_id}"}},
            )
        except Exception as error:  # un símbolo roto no puede llevarse el ciclo entero
            self._write_error(run_id, symbol, f"{type(error).__name__}: {error}")

    # ── Registro ──────────────────────────────────────────────────────────────

    def _record_skipped(self, count: int, at: datetime) -> None:
        """Deja constancia de los cierres perdidos, uno por símbolo.

        Comparten `run_id` para que se lean como un solo ciclo omitido, y van por
        símbolo porque la pregunta que se hace después es cuántas evaluaciones se
        perdió un mercado concreto.
        """
        run_id = uuid4()
        periods = "cierre" if count == 1 else "cierres"
        message = (
            f"ciclo omitido: {count} {periods} de {self._config.timeframe} pasaron "
            "mientras el anterior seguía corriendo"
        )
        for symbol in self._config.symbols:
            self._write_error(run_id, symbol, message, at=at)

    def _write_error(
        self, run_id: UUID, symbol: str, message: str, at: datetime | None = None
    ) -> None:
        """Registro sin evaluación: solo el fallo, atribuido al runner."""
        moment = at if at is not None else self._clock()
        self._journal.write(
            EvaluationRecord(
                run_id=run_id,
                at=moment,
                symbol=symbol,
                timeframe=self._config.timeframe,
                errors=(NodeError(node=RUNNER_NODE, message=message, at=moment),),
            )
        )
