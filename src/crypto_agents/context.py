"""Contexto de ejecución de una evaluación.

Vive en su propio módulo y no en `graph.py` porque los nodos lo necesitan en
runtime —LangGraph resuelve las anotaciones de cada nodo con `get_type_hints`—
y `graph.py` importa los nodos: tenerlo allí sería un import circular.

Lleva el cableado y el objetivo de la corrida. Nada de esto entra en
`TradingState`: el estado se serializa en cada checkpoint y un cliente de
exchange no es serializable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from crypto_agents.activation import DEFAULT_CONFIG, ActivationConfig
from crypto_agents.execution import PaperExecutor
from crypto_agents.indicators import DEFAULT_PRESET, IndicatorPreset
from crypto_agents.journal import InMemoryJournal
from crypto_agents.risk import StaticKillSwitch

if TYPE_CHECKING:
    from uuid import UUID

    from crypto_agents.execution import Executor
    from crypto_agents.journal import Journal
    from crypto_agents.llm import ModelRouter
    from crypto_agents.market import MarketClient
    from crypto_agents.quota import Clock
    from crypto_agents.risk import AccountState, KillSwitch
    from crypto_agents.settings import Settings

__all__ = ["AgentContext", "utc_now"]


def utc_now() -> datetime:
    """Reloj por defecto. Se sustituye en pruebas."""
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class AgentContext:
    """Cableado y objetivo de una evaluación. Inmutable durante la corrida."""

    settings: Settings
    router: ModelRouter
    market: MarketClient
    account: AccountState
    run_id: UUID
    symbol: str
    timeframe: str
    executor: Executor = field(default_factory=PaperExecutor)
    journal: Journal = field(default_factory=InMemoryJournal)
    candle_limit: int = 500
    preset: IndicatorPreset = field(default=DEFAULT_PRESET)
    activation: ActivationConfig = field(default=DEFAULT_CONFIG)
    kill_switch: KillSwitch = field(default_factory=StaticKillSwitch)
    """Parada consultable en cada evaluación.

    Va en el contexto y no en `RiskLimits` porque `Settings` está congelado: un
    interruptor que solo se pudiera leer al arrancar no serviría para detener un
    proceso que ya está corriendo.
    """

    clock: Clock = field(default=utc_now)
    now: datetime | None = None
    """Instante que decide qué vela sigue en formación. `None` usa el reloj real."""

    seed: int = 0
    """Semilla del plan. Solo la lee `random_uniform`, que la combina con el `run_id`."""
