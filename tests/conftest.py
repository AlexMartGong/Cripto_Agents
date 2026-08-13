"""Fábrica de velas sintéticas compartida.

Las velas se construyen a partir de una serie de cierres explícita, no de datos
aleatorios: una regla del gate debe dispararse por una forma concreta del precio
que la prueba pueda escribir a mano y leer después.

Ninguna prueba toca la red.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pandas as pd

from crypto_agents.market import OHLCV_COLUMNS
from crypto_agents.settings import Backend, ModelChoice, RoleConfig
from crypto_agents.state import AgentRole

if TYPE_CHECKING:
    from collections.abc import Sequence

START = datetime(2026, 8, 1, 0, 0, tzinfo=UTC)
STEP = timedelta(hours=1)

CHEAP = ModelChoice(backend=Backend.OLLAMA, model="qwen3:8b", family="qwen", quota_per_window=63000)
SCARCE = ModelChoice(
    backend=Backend.OPENAI,
    model="gpt-x",
    family="gpt",
    quota_weight=2.0,
    quota_per_window=4,
)


def role_map(
    primary: ModelChoice = CHEAP, fallback: ModelChoice | None = None
) -> dict[AgentRole, RoleConfig]:
    """Mapa completo de roles con las dos mesas en familias distintas.

    El validador de `Settings` rechaza que bull y bear compartan familia, así que
    a la mesa bajista se le da una copia con familia propia. El resto del
    comportamiento (modelo, peso, cuota) queda idéntico para no alterar lo que
    las pruebas de cuota miden.
    """
    roles = {role: RoleConfig(primary=primary, fallback=fallback) for role in AgentRole}
    bear_primary = primary.model_copy(update={"family": f"{primary.family}-alt"})
    bear_fallback = (
        None
        if fallback is None
        else fallback.model_copy(update={"family": f"{fallback.family}-alt"})
    )
    roles[AgentRole.BEAR] = RoleConfig(primary=bear_primary, fallback=bear_fallback)
    return roles


def raw_ohlcv(
    closes: Sequence[float],
    start: datetime = START,
    step: timedelta = STEP,
    spread: float = 0.5,
    volume: float = 1000.0,
    volumes: Sequence[float] | None = None,
) -> list[list[float]]:
    """Filas crudas como las devuelve ccxt: `[ts_ms, open, high, low, close, volume]`.

    La apertura de cada vela es el cierre de la anterior, y el rango se abre
    simétricamente `spread` a cada lado: velas coherentes sin ruido que estorbe.
    """
    rows: list[list[float]] = []
    for index, close in enumerate(closes):
        opening = closes[index - 1] if index else close
        timestamp = start + index * step
        rows.append(
            [
                float(timestamp.timestamp() * 1000),
                float(opening),
                float(max(opening, close) + spread),
                float(min(opening, close) - spread),
                float(close),
                float(volumes[index]) if volumes is not None else float(volume),
            ]
        )
    return rows


def candles(
    closes: Sequence[float],
    start: datetime = START,
    step: timedelta = STEP,
    spread: float = 0.5,
    volume: float = 1000.0,
    volumes: Sequence[float] | None = None,
) -> pd.DataFrame:
    """DataFrame ya normalizado, saltándose la ida y vuelta por `to_dataframe`."""
    rows = raw_ohlcv(closes, start, step, spread, volume, volumes)
    frame = pd.DataFrame(rows, columns=["timestamp", *OHLCV_COLUMNS])
    frame.index = pd.to_datetime(frame.pop("timestamp"), unit="ms", utc=True)
    frame.index.name = "timestamp"
    return frame.astype("float64")


def flat_closes(count: int, level: float = 100.0) -> list[float]:
    """Serie plana: base sobre la que ninguna regla debe disparar."""
    return [level] * count


def drifting_closes(count: int, level: float = 100.0, drift: float = 0.01) -> list[float]:
    """Deriva mínima y constante: evita divisiones por cero en ATR sin crear señales."""
    return [level + drift * index for index in range(count)]


class FakeMarketClient:
    """Origen de velas en memoria. Registra qué se le pidió."""

    def __init__(self, rows: list[list[float]]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, str, int]] = []

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> list[list[float]]:
        """Devuelve las filas preparadas sin tocar la red."""
        self.calls.append((symbol, timeframe, limit))
        return self.rows
