"""Fábricas compartidas por las pruebas: velas, roles y agentes falsos.

Las velas se construyen a partir de una serie de cierres explícita, no de datos
aleatorios: una regla del gate debe dispararse por una forma concreta del precio
que la prueba pueda escribir a mano y leer después.

Ninguna prueba toca la red.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from crypto_agents.indicators import IndicatorPreset
from crypto_agents.market import OHLCV_COLUMNS, to_dataframe
from crypto_agents.replay import read_ohlcv_csv
from crypto_agents.risk import AccountState
from crypto_agents.settings import Backend, ModelChoice, RoleConfig
from crypto_agents.state import (
    Action,
    AgentRole,
    Bias,
    Claim,
    DebateBrief,
    Decision,
    Dimension,
    Observation,
    Proposal,
    Side,
    Strength,
    TechnicalVerdict,
)

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

    El decisor nunca lleva respaldo: `Settings` lo rechaza, porque degradarlo
    cambiaría quién toma la decisión final sin que quede dicho en ninguna parte.
    """
    roles = {role: RoleConfig(primary=primary, fallback=fallback) for role in AgentRole}
    bear_primary = primary.model_copy(update={"family": f"{primary.family}-alt"})
    bear_fallback = (
        None
        if fallback is None
        else fallback.model_copy(update={"family": f"{fallback.family}-alt"})
    )
    roles[AgentRole.BEAR] = RoleConfig(primary=bear_primary, fallback=bear_fallback)
    roles[AgentRole.DECIDER] = RoleConfig(primary=primary)
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


# ─────────────────────────────────────── Histórico real ──────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"
REAL_HISTORY = DATA_DIR / "btcusdt_4h.csv"
REAL_HISTORY_DIGEST = "3460640ac8fb6ee9ad903c27d5c534e17b9954707c9d4a528bc7537e5142fae2"
"""Digest esperado del histórico. Ver `tests/data/README.md` para su procedencia."""


def real_rows() -> list[list[float]]:
    """Histórico fijo del repositorio, en formato crudo de ccxt.

    Ninguna prueba descarga nada: el CSV está versionado justo para que el replay
    sea reproducible entre máquinas. Se lee con el mismo lector que usa el comando
    de ablación, para que test y comando no puedan divergir en el parseo.
    """
    return read_ohlcv_csv(REAL_HISTORY)


def real_candles() -> pd.DataFrame:
    """El mismo histórico ya normalizado."""
    return to_dataframe(real_rows())


class FakeMarketClient:
    """Origen de velas en memoria. Registra qué se le pidió."""

    def __init__(self, rows: list[list[float]]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, str, int]] = []

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> list[list[float]]:
        """Devuelve las filas preparadas sin tocar la red."""
        self.calls.append((symbol, timeframe, limit))
        return self.rows


# ─────────────────────────────────────── Agentes falsos ───────────────────────────────────────────
# Un preset corto para que las pruebas no necesiten cientos de velas de warm-up.

PRESET = IndicatorPreset(
    rsi=5, ema_fast=3, ema_slow=8, ema_trend=21, atr=5, adx=5, bbands=5, volume_ma=5
)
CITED = "EMA_3"
BAR_COUNT = PRESET.min_bars + 1
HEALTHY = AccountState(equity=10_000.0, day_start_equity=10_000.0)


def verdict_payload(dimension: Dimension, cites: str = CITED) -> str:
    """Veredicto válido para la dimensión pedida."""
    return TechnicalVerdict(
        dimension=dimension,
        bias=Bias.BULLISH,
        confidence=0.7,
        observations=[
            Observation(
                id=f"{dimension.value}-1",
                text="El precio sostiene el soporte previo y marca un maximo superior.",
                cites=[cites],
                supports=Bias.BULLISH,
            )
        ],
        invalidation="Pierde el soporte de 63000.",
    ).model_dump_json()


def brief_payload(side: Side, grounded_in: str = "structure-1") -> str:
    """Alegato válido para la mesa pedida."""
    return DebateBrief(
        side=side,
        thesis="La estructura sigue intacta mientras el soporte aguante el retroceso.",
        claims=[
            Claim(
                text="La media rapida actua como soporte dinamico en cada retroceso.",
                grounded_in=[grounded_in],
                strength=Strength.MODERATE,
            ),
            Claim(
                text="El impulso acompana sin llegar a sobrecompra extrema todavia.",
                grounded_in=[grounded_in],
                strength=Strength.WEAK,
            ),
        ],
        conviction=0.6,
        strongest_counterargument="Un cierre bajo el soporte invalida toda la lectura.",
    ).model_dump_json()


def proposal_payload() -> str:
    """Decisión sin mesas: el esquema de las variantes de ablación.

    `Decision` no vale aquí: `Proposal` cierra los extras, así que los campos de
    descarte harían fallar la validación en vez de ignorarse.
    """
    return Proposal(
        action=Action.BUY,
        confidence=0.7,
        size_fraction=0.25,
        invalidation_price=99.0,
        rationale="Estructura, impulso y volumen coinciden en direccion alcista.",
    ).model_dump_json()


def decision_payload() -> str:
    """Decisión accionable completa."""
    return Decision(
        action=Action.BUY,
        confidence=0.7,
        size_fraction=0.25,
        invalidation_price=99.0,
        rationale="Estructura, impulso y volumen coinciden en direccion alcista.",
        dismissed_side=Side.BEAR,
        dismissal_reason="Su contraargumento depende de un nivel que ya se perdio.",
    ).model_dump_json()


class FakeLLM:
    """Backend falso: lee del prompt qué se le pide y devuelve un payload fijo."""

    def __init__(self, overrides: dict[str, str] | None = None) -> None:
        self.overrides = overrides or {}
        self.prompts: list[tuple[str, str]] = []

    async def complete(self, choice: ModelChoice, prompt: str, schema: type) -> str:
        """Devuelve el payload que corresponde al agente que hizo la pregunta."""
        target = self._target(prompt, schema)
        self.prompts.append((target, prompt))
        if target in self.overrides:
            return self.overrides[target]
        if schema is TechnicalVerdict:
            return verdict_payload(Dimension(target))
        if schema is DebateBrief:
            return brief_payload(Side(target))
        if schema is Proposal:
            return proposal_payload()
        return decision_payload()

    @staticmethod
    def _target(prompt: str, schema: type) -> str:
        """Identifica al solicitante por lo que el prompt le exige responder."""
        if schema is TechnicalVerdict:
            return next(
                item.value for item in Dimension if f'literalmente `"{item.value}"`' in prompt
            )
        if schema is DebateBrief:
            return next(item.value for item in Side if f'literalmente `"{item.value}"`' in prompt)
        return "decider"
