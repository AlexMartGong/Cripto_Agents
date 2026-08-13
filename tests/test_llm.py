"""Pruebas del router de modelos.

Verifican lo que hace intercambiable el backend: el router resuelve el modelo por
presupuesto, delega en el adaptador y deja rastro de cada llamada. Ningún nodo
necesita saber qué proveedor hay detrás.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from crypto_agents.llm import ModelRouter, build_backends, prompt_digest
from crypto_agents.quota import QuotaExhaustedError, QuotaLedger
from crypto_agents.settings import Backend, ModelChoice, RoleConfig, Settings, load_settings
from crypto_agents.state import AgentRole, Bias, Dimension, Observation, TechnicalVerdict

if TYPE_CHECKING:
    from crypto_agents.state import LLMOutput

START = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)

CHEAP = ModelChoice(backend=Backend.OLLAMA, model="qwen3:8b", quota_per_window=63000)
SCARCE = ModelChoice(backend=Backend.OPENAI, model="gpt-x", quota_weight=2.0, quota_per_window=2)


class FakeClock:
    """Reloj controlado por la prueba."""

    def __init__(self, start: datetime = START) -> None:
        self.now = start

    def __call__(self) -> datetime:
        """Instante actual según la prueba."""
        return self.now

    def advance(self, delta: timedelta) -> None:
        """Mueve el reloj hacia adelante."""
        self.now += delta


class RecordingBackend:
    """Adaptador falso que devuelve un veredicto fijo y anota qué se le pidió."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.seen: list[tuple[str, str]] = []

    async def structured[T: LLMOutput](
        self, choice: ModelChoice, prompt: str, schema: type[T]
    ) -> T:
        """Registra la petición y devuelve una instancia válida del esquema."""
        self.seen.append((choice.model, prompt))
        verdict = TechnicalVerdict(
            dimension=Dimension.STRUCTURE,
            bias=Bias.BULLISH,
            confidence=0.6,
            observations=[
                Observation(
                    id="structure-1",
                    text="El precio sostiene el soporte previo y marca un maximo superior.",
                    cites=["EMA_50"],
                    supports=Bias.BULLISH,
                )
            ],
            invalidation="Pierde el soporte de 63000.",
        )
        return verdict  # type: ignore[return-value]


def make_settings(primary: ModelChoice, fallback: ModelChoice | None = None) -> Settings:
    """Configuración donde todos los roles comparten el mismo par primario/respaldo."""
    return load_settings(
        roles={role: RoleConfig(primary=primary, fallback=fallback) for role in AgentRole},
        openai={"api_key": "sk-test"},
        ollama={"host": "http://localhost:11434"},
        _env_file=None,
    )


def make_router(
    settings: Settings, clock: FakeClock
) -> tuple[ModelRouter, QuotaLedger, dict[Backend, RecordingBackend]]:
    """Router con adaptadores falsos para ambos backends."""
    backends = {
        Backend.OPENAI: RecordingBackend("openai"),
        Backend.OLLAMA: RecordingBackend("ollama"),
    }
    ledger = QuotaLedger(settings, clock)
    return ModelRouter(settings, ledger, backends, clock), ledger, backends


@pytest.mark.asyncio
async def test_invoke_returns_output_and_records_the_call() -> None:
    """Toda llamada deja un `LLMCall`: no hay forma de gastar cuota sin rastro."""
    clock = FakeClock()
    router, ledger, _ = make_router(make_settings(SCARCE), clock)

    verdict, call = await router.invoke(AgentRole.STRUCTURE, "analiza", TechnicalVerdict)

    assert verdict.dimension is Dimension.STRUCTURE
    assert call.role is AgentRole.STRUCTURE
    assert call.model == "gpt-x"
    assert call.quota_weight == 2.0
    assert call.prompt_digest == prompt_digest("analiza")
    assert call.at == START
    assert ledger.used(AgentRole.STRUCTURE, "gpt-x") == 2.0


@pytest.mark.asyncio
async def test_router_dispatches_to_the_backend_of_the_resolved_model() -> None:
    """El rol no elige proveedor: lo elige el modelo que resolvió el presupuesto."""
    clock = FakeClock()
    router, _, backends = make_router(make_settings(CHEAP), clock)

    await router.invoke(AgentRole.BULL, "argumenta", TechnicalVerdict)

    assert backends[Backend.OLLAMA].seen == [("qwen3:8b", "argumenta")]
    assert backends[Backend.OPENAI].seen == []


@pytest.mark.asyncio
async def test_router_degrades_to_fallback_backend_when_quota_runs_out() -> None:
    """Agotado el primario, la siguiente llamada sale por el proveedor del respaldo."""
    clock = FakeClock()
    router, _, backends = make_router(make_settings(SCARCE, CHEAP), clock)

    await router.invoke(AgentRole.DECIDER, "primera", TechnicalVerdict)
    await router.invoke(AgentRole.DECIDER, "segunda", TechnicalVerdict)

    assert [model for model, _ in backends[Backend.OPENAI].seen] == ["gpt-x"]
    assert [model for model, _ in backends[Backend.OLLAMA].seen] == ["qwen3:8b"]


@pytest.mark.asyncio
async def test_router_raises_before_spending_when_quota_is_exhausted() -> None:
    """`QuotaExhaustedError` se lanza antes de tocar el backend: no se gasta nada."""
    clock = FakeClock()
    router, _, backends = make_router(make_settings(SCARCE), clock)

    await router.invoke(AgentRole.DECIDER, "primera", TechnicalVerdict)
    with pytest.raises(QuotaExhaustedError):
        await router.invoke(AgentRole.DECIDER, "segunda", TechnicalVerdict)

    assert len(backends[Backend.OPENAI].seen) == 1


@pytest.mark.asyncio
async def test_router_reports_a_missing_backend_by_name() -> None:
    """Un backend declarado pero no construido debe fallar señalando el rol."""
    clock = FakeClock()
    settings = make_settings(CHEAP)
    ledger = QuotaLedger(settings, clock)
    router = ModelRouter(settings, ledger, {}, clock)

    with pytest.raises(LookupError, match="ollama"):
        await router.invoke(AgentRole.VOLUME, "analiza", TechnicalVerdict)


def test_build_backends_only_creates_what_is_configured() -> None:
    """Sin credenciales de OpenAI no se instancia su cliente."""
    settings = load_settings(
        roles={role: RoleConfig(primary=CHEAP) for role in AgentRole},
        ollama={"host": "http://localhost:11434"},
        _env_file=None,
    )
    assert set(build_backends(settings)) == {Backend.OLLAMA}


def test_prompt_digest_is_stable_and_hex() -> None:
    """El digest identifica el prompt para replay y detección de cache hits."""
    digest = prompt_digest("analiza")
    assert digest == prompt_digest("analiza")
    assert len(digest) == 64
    assert digest != prompt_digest("analiza ")
