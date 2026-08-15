"""Pruebas del contador de cuota.

El reloj se inyecta siempre. Ninguna prueba depende de `datetime.now()`: si el
tiempo real entrara aquí, la expiración de ventana solo se podría probar
esperando cinco horas.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from crypto_agents.quota import QuotaExhaustedError, QuotaLedger
from crypto_agents.settings import Backend, ModelChoice, Settings, load_settings
from crypto_agents.state import AgentRole, LLMCall
from tests.conftest import CHEAP, SCARCE, role_map

DIGEST = "c" * 64
START = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)


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


def make_settings(primary: ModelChoice, fallback: ModelChoice | None = None) -> Settings:
    """Configuración donde todos los roles comparten el mismo par primario/respaldo."""
    return load_settings(
        roles=role_map(primary, fallback),
        openai={"api_key": "sk-test"},
        ollama={"host": "http://localhost:11434"},
    )


def make_call(
    choice: ModelChoice,
    at: datetime,
    role: AgentRole = AgentRole.DECIDER,
    cache_hit: bool = False,
) -> LLMCall:
    """Registro de una llamada al modelo indicado."""
    return LLMCall(
        role=role,
        backend=choice.backend,
        model=choice.model,
        quota_weight=choice.quota_weight,
        prompt_digest=DIGEST,
        cache_hit=cache_hit,
        valid=True,
        latency_ms=100.0,
        at=at,
    )


# ────────────────────────────────────────────── Conteo ────────────────────────────────────────────


def test_empty_ledger_reports_full_budget() -> None:
    """Sin llamadas, la ventana está entera."""
    clock = FakeClock()
    ledger = QuotaLedger(make_settings(SCARCE), clock)
    assert ledger.used(AgentRole.DECIDER, SCARCE.model) == 0.0
    assert ledger.remaining(AgentRole.DECIDER, SCARCE) == 4.0


def test_record_accumulates_declared_weight() -> None:
    """Un modelo de doble consumo gasta 2.0 por llamada, no 1.0."""
    clock = FakeClock()
    ledger = QuotaLedger(make_settings(SCARCE), clock)
    ledger.record(make_call(SCARCE, clock.now))
    assert ledger.used(AgentRole.DECIDER, SCARCE.model) == 2.0


def test_cache_hits_do_not_consume_quota() -> None:
    """Un cache hit no llegó al proveedor."""
    clock = FakeClock()
    ledger = QuotaLedger(make_settings(SCARCE), clock)
    ledger.record(make_call(SCARCE, clock.now, cache_hit=True))
    assert ledger.used(AgentRole.DECIDER, SCARCE.model) == 0.0


def test_consumption_is_tracked_per_role() -> None:
    """La cuota se declara por rol: lo que gasta el decisor no descuenta a la mesa alcista."""
    clock = FakeClock()
    ledger = QuotaLedger(make_settings(SCARCE), clock)
    ledger.record(make_call(SCARCE, clock.now, role=AgentRole.DECIDER))
    assert ledger.used(AgentRole.DECIDER, SCARCE.model) == 2.0
    assert ledger.used(AgentRole.BULL, SCARCE.model) == 0.0


# ────────────────────────────────────────── Ventana deslizante ────────────────────────────────────


def test_calls_expire_when_the_window_slides_past_them() -> None:
    """Pasada la ventana de 5 h, el consumo antiguo deja de contar."""
    clock = FakeClock()
    ledger = QuotaLedger(make_settings(SCARCE), clock)
    ledger.record(make_call(SCARCE, clock.now))
    assert ledger.used(AgentRole.DECIDER, SCARCE.model) == 2.0

    clock.advance(timedelta(hours=5, seconds=1))
    assert ledger.used(AgentRole.DECIDER, SCARCE.model) == 0.0


def test_calls_inside_the_window_still_count() -> None:
    """Justo dentro de la ventana el consumo sigue vigente."""
    clock = FakeClock()
    ledger = QuotaLedger(make_settings(SCARCE), clock)
    ledger.record(make_call(SCARCE, clock.now))

    clock.advance(timedelta(hours=4, minutes=59))
    assert ledger.used(AgentRole.DECIDER, SCARCE.model) == 2.0


def test_window_slides_instead_of_resetting() -> None:
    """Solo expira lo viejo: una llamada reciente sobrevive a la purga."""
    clock = FakeClock()
    ledger = QuotaLedger(make_settings(SCARCE), clock)
    ledger.record(make_call(SCARCE, clock.now))
    clock.advance(timedelta(hours=3))
    ledger.record(make_call(SCARCE, clock.now))

    clock.advance(timedelta(hours=2, minutes=1))
    assert ledger.used(AgentRole.DECIDER, SCARCE.model) == 2.0


def test_extend_rehydrates_from_unordered_calls() -> None:
    """El replay recibe llamadas paralelas fuera de orden y debe contarlas igual."""
    clock = FakeClock()
    ledger = QuotaLedger(make_settings(SCARCE), clock)
    ledger.extend(
        [
            make_call(SCARCE, START + timedelta(minutes=30)),
            make_call(SCARCE, START),
            make_call(SCARCE, START + timedelta(minutes=10)),
        ]
    )
    clock.advance(timedelta(hours=1))
    assert ledger.used(AgentRole.DECIDER, SCARCE.model) == 6.0


# ─────────────────────────────────────── Resolución y degradación ─────────────────────────────────


def test_resolve_prefers_the_primary_model() -> None:
    """Con presupuesto disponible se usa el primario."""
    ledger = QuotaLedger(make_settings(SCARCE, CHEAP), FakeClock())
    assert ledger.resolve(AgentRole.DECIDER) == SCARCE


def test_resolve_degrades_to_fallback_when_primary_is_full() -> None:
    """Sin presupuesto en el primario, la llamada baja al respaldo en vez de fallar.

    Se prueba sobre un rol técnico y no sobre el decisor: el decisor no admite
    respaldo, así que degradarlo no es un caso que pueda existir.
    """
    clock = FakeClock()
    ledger = QuotaLedger(make_settings(SCARCE, CHEAP), clock)
    ledger.record(make_call(SCARCE, clock.now, role=AgentRole.STRUCTURE))
    ledger.record(make_call(SCARCE, clock.now, role=AgentRole.STRUCTURE))

    assert ledger.used(AgentRole.STRUCTURE, SCARCE.model) == 4.0
    assert ledger.resolve(AgentRole.STRUCTURE) == CHEAP


def test_partial_room_is_not_enough_for_a_double_weight_call() -> None:
    """Queda 1.0 libre y la llamada pesa 2.0: no cabe entera, degrada."""
    clock = FakeClock()
    settings = load_settings(
        roles=role_map(
            ModelChoice(
                backend=Backend.OPENAI,
                model="gpt-x",
                family="gpt",
                quota_weight=2.0,
                quota_per_window=3,
            ),
            CHEAP,
        ),
        openai={"api_key": "sk-test"},
        ollama={"host": "http://localhost:11434"},
    )
    ledger = QuotaLedger(settings, clock)
    primary = settings.role_config(AgentRole.STRUCTURE).primary
    ledger.record(make_call(primary, clock.now, role=AgentRole.STRUCTURE))

    assert ledger.resolve(AgentRole.STRUCTURE) == CHEAP


def test_resolve_raises_when_neither_model_fits() -> None:
    """Degradación de un solo salto: agotados los dos, la llamada falla sin gastar."""
    clock = FakeClock()
    ledger = QuotaLedger(make_settings(SCARCE, SCARCE), clock)
    ledger.record(make_call(SCARCE, clock.now, role=AgentRole.STRUCTURE))
    ledger.record(make_call(SCARCE, clock.now, role=AgentRole.STRUCTURE))

    with pytest.raises(QuotaExhaustedError) as excinfo:
        ledger.resolve(AgentRole.STRUCTURE)
    assert excinfo.value.role is AgentRole.STRUCTURE
    assert excinfo.value.tried == ("gpt-x", "gpt-x")


def test_resolve_raises_when_exhausted_without_fallback() -> None:
    """Sin respaldo declarado no hay a dónde degradar."""
    clock = FakeClock()
    ledger = QuotaLedger(make_settings(SCARCE), clock)
    ledger.record(make_call(SCARCE, clock.now))
    ledger.record(make_call(SCARCE, clock.now))

    with pytest.raises(QuotaExhaustedError, match="decider"):
        ledger.resolve(AgentRole.DECIDER)


def test_budget_recovers_when_the_window_expires() -> None:
    """Agotada la cuota, el primario vuelve a estar disponible al salir la ventana."""
    clock = FakeClock()
    ledger = QuotaLedger(make_settings(SCARCE, CHEAP), clock)
    ledger.record(make_call(SCARCE, clock.now, role=AgentRole.STRUCTURE))
    ledger.record(make_call(SCARCE, clock.now, role=AgentRole.STRUCTURE))
    assert ledger.resolve(AgentRole.STRUCTURE) == CHEAP

    clock.advance(timedelta(hours=5, seconds=1))
    assert ledger.resolve(AgentRole.STRUCTURE) == SCARCE
