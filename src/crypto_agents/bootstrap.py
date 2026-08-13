"""Cableado único del sistema.

Es el sitio donde `Settings` se convierte en objetos vivos: backends, router,
caché, journal, ejecutor, interruptor de parada. Ningún otro módulo construye
estas piezas, así que hay un solo lugar donde mirar para saber qué habla con qué.

El ejecutor real solo se construye si se cumplen sus tres condiciones. No se
relajan aquí ni en ningún otro sitio: `CcxtExecutor` se niega a existir si falta
alguna, y este módulo se limita a preguntar antes para poder decirlo con un
mensaje entendible en vez de una excepción a medio arranque.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING
from uuid import uuid4

from crypto_agents.cache import JsonFileResponseCache
from crypto_agents.context import AgentContext, utc_now
from crypto_agents.execution import CcxtExecutor, ExecutionMode, PaperExecutor
from crypto_agents.journal import JsonlJournal
from crypto_agents.llm import ModelRouter, build_backends
from crypto_agents.market import CcxtMarketClient
from crypto_agents.quota import QuotaLedger
from crypto_agents.risk import AccountState, AnyKillSwitch, FileKillSwitch, StaticKillSwitch
from crypto_agents.settings import ConfigError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from uuid import UUID

    from crypto_agents.execution import Executor
    from crypto_agents.journal import Journal
    from crypto_agents.quota import Clock
    from crypto_agents.risk import KillSwitch
    from crypto_agents.settings import Settings

__all__ = [
    "build_executor",
    "build_kill_switch",
    "build_router",
    "evaluation_context",
    "open_journal",
]

NOTIONAL_ACCOUNT = AccountState(equity=10_000.0, day_start_equity=10_000.0)
"""Cuenta por defecto en modo papel. Todos los tamaños son fracciones, así que el
nivel absoluto solo afecta al veto de drawdown, que sin equity viva no dispara."""


def build_kill_switch(settings: Settings) -> KillSwitch:
    """Interruptor efectivo: el de configuración o el archivo centinela.

    Basta uno de los dos. El de configuración detiene de forma permanente hasta
    que se cambie el entorno; el archivo se pone y se quita sin reiniciar nada.
    """
    return AnyKillSwitch(
        StaticKillSwitch(settings.risk.kill_switch),
        FileKillSwitch(settings.operations.kill_switch_file),
    )


def build_router(settings: Settings, clock: Clock = utc_now) -> ModelRouter:
    """Router con la caché en disco declarada en la configuración."""
    ledger = QuotaLedger(settings, clock)
    cache = JsonFileResponseCache(settings.operations.cache_dir)
    return ModelRouter(settings, ledger, build_backends(settings), clock, cache)


def open_journal(settings: Settings) -> Journal:
    """Journal en disco. Una línea por evaluación, apta para grep y para reproceso."""
    return JsonlJournal(settings.operations.journal_path)


def build_executor(settings: Settings) -> Executor:
    """Ejecutor de papel salvo que las tres condiciones de real estén cumplidas.

    Las condiciones no se comprueban aquí: se le pide a `CcxtExecutor` que se
    construya, y es él quien se niega. Duplicar la comprobación crearía dos sitios
    donde relajarla.
    """
    if settings.execution.mode is not ExecutionMode.LIVE:
        return PaperExecutor()
    return CcxtExecutor(settings.execution, settings.exchange)


@asynccontextmanager
async def evaluation_context(
    settings: Settings,
    symbol: str,
    timeframe: str,
    router: ModelRouter | None = None,
    journal: Journal | None = None,
    run_id: UUID | None = None,
    clock: Clock = utc_now,
) -> AsyncIterator[AgentContext]:
    """Contexto completo de una evaluación, con el cliente de exchange cerrado al salir.

    Sin el cierre, cada evaluación deja una sesión HTTP abierta en el event loop y
    un runner de días acaba quedándose sin descriptores.
    """
    if settings.account is None:
        account = NOTIONAL_ACCOUNT
        if settings.execution.mode is ExecutionMode.LIVE:
            raise ConfigError(
                "modo live sin cuenta declarada: falta CA_ACCOUNT__EQUITY "
                "y CA_ACCOUNT__DAY_START_EQUITY"
            )
    else:
        account = settings.account

    market = CcxtMarketClient(settings.exchange)
    try:
        yield AgentContext(
            settings=settings,
            router=router if router is not None else build_router(settings, clock),
            market=market,
            account=account,
            run_id=run_id if run_id is not None else uuid4(),
            symbol=symbol,
            timeframe=timeframe,
            executor=build_executor(settings),
            journal=journal if journal is not None else open_journal(settings),
            kill_switch=build_kill_switch(settings),
            clock=clock,
        )
    finally:
        await market.close()
