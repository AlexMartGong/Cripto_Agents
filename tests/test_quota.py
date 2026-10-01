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
from crypto_agents.state import AgentRole, LLMCall, StructuredOutputMode
from tests.conftest import CHEAP, SCARCE, role_map

DIGEST = "c" * 64
START = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)
WINDOW = timedelta(hours=5)
"""La ventana declarada por defecto en `Settings`, aquí explícita.

El contador ya no recibe la configuración, así que estas pruebas separan dos
cosas que antes viajaban juntas: cuánto dura la ventana, que es suya, y qué
modelos puede usar un rol, que es de quien llama.
"""


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
        structured_output=choice.structured_output,
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
    ledger = QuotaLedger(WINDOW, clock)
    assert ledger.used(AgentRole.DECIDER, SCARCE.model) == 0.0
    assert ledger.remaining(AgentRole.DECIDER, SCARCE) == 4.0


def test_record_accumulates_declared_weight() -> None:
    """Un modelo de doble consumo gasta 2.0 por llamada, no 1.0."""
    clock = FakeClock()
    ledger = QuotaLedger(WINDOW, clock)
    ledger.record(make_call(SCARCE, clock.now))
    assert ledger.used(AgentRole.DECIDER, SCARCE.model) == 2.0


def test_cache_hits_do_not_consume_quota() -> None:
    """Un cache hit no llegó al proveedor."""
    clock = FakeClock()
    ledger = QuotaLedger(WINDOW, clock)
    ledger.record(make_call(SCARCE, clock.now, cache_hit=True))
    assert ledger.used(AgentRole.DECIDER, SCARCE.model) == 0.0


def test_consumption_is_tracked_per_role() -> None:
    """La cuota se declara por rol: lo que gasta el decisor no descuenta a la mesa alcista."""
    clock = FakeClock()
    ledger = QuotaLedger(WINDOW, clock)
    ledger.record(make_call(SCARCE, clock.now, role=AgentRole.DECIDER))
    assert ledger.used(AgentRole.DECIDER, SCARCE.model) == 2.0
    assert ledger.used(AgentRole.BULL, SCARCE.model) == 0.0


# ────────────────────────────────────────── Ventana deslizante ────────────────────────────────────


def test_calls_expire_when_the_window_slides_past_them() -> None:
    """Pasada la ventana de 5 h, el consumo antiguo deja de contar."""
    clock = FakeClock()
    ledger = QuotaLedger(WINDOW, clock)
    ledger.record(make_call(SCARCE, clock.now))
    assert ledger.used(AgentRole.DECIDER, SCARCE.model) == 2.0

    clock.advance(timedelta(hours=5, seconds=1))
    assert ledger.used(AgentRole.DECIDER, SCARCE.model) == 0.0


def test_calls_inside_the_window_still_count() -> None:
    """Justo dentro de la ventana el consumo sigue vigente."""
    clock = FakeClock()
    ledger = QuotaLedger(WINDOW, clock)
    ledger.record(make_call(SCARCE, clock.now))

    clock.advance(timedelta(hours=4, minutes=59))
    assert ledger.used(AgentRole.DECIDER, SCARCE.model) == 2.0


def test_window_slides_instead_of_resetting() -> None:
    """Solo expira lo viejo: una llamada reciente sobrevive a la purga."""
    clock = FakeClock()
    ledger = QuotaLedger(WINDOW, clock)
    ledger.record(make_call(SCARCE, clock.now))
    clock.advance(timedelta(hours=3))
    ledger.record(make_call(SCARCE, clock.now))

    clock.advance(timedelta(hours=2, minutes=1))
    assert ledger.used(AgentRole.DECIDER, SCARCE.model) == 2.0


def test_extend_rehydrates_from_unordered_calls() -> None:
    """El replay recibe llamadas paralelas fuera de orden y debe contarlas igual."""
    clock = FakeClock()
    ledger = QuotaLedger(WINDOW, clock)
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
    settings = make_settings(SCARCE, CHEAP)
    ledger = QuotaLedger(WINDOW, FakeClock())
    assert ledger.resolve(AgentRole.DECIDER, settings.role_choices(AgentRole.DECIDER)) == SCARCE


def test_resolve_degrades_to_fallback_when_primary_is_full() -> None:
    """Sin presupuesto en el primario, la llamada baja al respaldo en vez de fallar.

    Se prueba sobre un rol técnico y no sobre el decisor: el decisor no admite
    respaldo, así que degradarlo no es un caso que pueda existir.
    """
    clock = FakeClock()
    settings = make_settings(SCARCE, CHEAP)
    ledger = QuotaLedger(WINDOW, clock)
    ledger.record(make_call(SCARCE, clock.now, role=AgentRole.STRUCTURE))
    ledger.record(make_call(SCARCE, clock.now, role=AgentRole.STRUCTURE))

    assert ledger.used(AgentRole.STRUCTURE, SCARCE.model) == 4.0
    choices = settings.role_choices(AgentRole.STRUCTURE)
    assert ledger.resolve(AgentRole.STRUCTURE, choices) == CHEAP


def test_the_same_ledger_serves_two_configurations_of_the_same_role() -> None:
    """Un contador, dos repartos: lo gastado por uno se lo encuentra el otro.

    Es la propiedad que hace posible un solo presupuesto para los seis brazos de
    la ablación, donde cada brazo declara modelos distintos para el mismo rol. Con
    la configuración dentro del contador haría falta uno por brazo, y seis
    contadores creyéndose dentro del presupuesto es el defecto que esto cierra.
    """
    clock = FakeClock()
    remote = make_settings(SCARCE, CHEAP)
    local_only = make_settings(CHEAP)
    ledger = QuotaLedger(WINDOW, clock)

    ledger.record(make_call(CHEAP, clock.now, role=AgentRole.STRUCTURE))
    assert ledger.resolve(AgentRole.STRUCTURE, remote.role_choices(AgentRole.STRUCTURE)) == SCARCE
    assert ledger.used(AgentRole.STRUCTURE, CHEAP.model) == 1.0

    ledger.resolve(AgentRole.STRUCTURE, local_only.role_choices(AgentRole.STRUCTURE))
    assert ledger.used(AgentRole.STRUCTURE, CHEAP.model) == 1.0, "resolver no gasta"


def test_partial_room_is_not_enough_for_a_double_weight_call() -> None:
    """Queda 1.0 libre y la llamada pesa 2.0: no cabe entera, degrada."""
    clock = FakeClock()
    settings = load_settings(
        roles=role_map(
            ModelChoice(
                backend=Backend.OPENAI,
                model="gpt-x",
                family="gpt",
                structured_output=StructuredOutputMode.JSON_SCHEMA,
                quota_weight=2.0,
                quota_per_window=3,
            ),
            CHEAP,
        ),
        openai={"api_key": "sk-test"},
        ollama={"host": "http://localhost:11434"},
    )
    ledger = QuotaLedger(WINDOW, clock)
    primary = settings.role_config(AgentRole.STRUCTURE).primary
    ledger.record(make_call(primary, clock.now, role=AgentRole.STRUCTURE))

    choices = settings.role_choices(AgentRole.STRUCTURE)
    assert ledger.resolve(AgentRole.STRUCTURE, choices) == CHEAP


def test_resolve_raises_when_neither_model_fits() -> None:
    """Degradación de un solo salto: agotados los dos, la llamada falla sin gastar."""
    clock = FakeClock()
    settings = make_settings(SCARCE, SCARCE)
    ledger = QuotaLedger(WINDOW, clock)
    ledger.record(make_call(SCARCE, clock.now, role=AgentRole.STRUCTURE))
    ledger.record(make_call(SCARCE, clock.now, role=AgentRole.STRUCTURE))

    with pytest.raises(QuotaExhaustedError) as excinfo:
        ledger.resolve(AgentRole.STRUCTURE, settings.role_choices(AgentRole.STRUCTURE))
    assert excinfo.value.role is AgentRole.STRUCTURE
    assert excinfo.value.tried == ("gpt-x", "gpt-x")


def test_resolve_raises_when_exhausted_without_fallback() -> None:
    """Sin respaldo declarado no hay a dónde degradar."""
    clock = FakeClock()
    settings = make_settings(SCARCE)
    ledger = QuotaLedger(WINDOW, clock)
    ledger.record(make_call(SCARCE, clock.now))
    ledger.record(make_call(SCARCE, clock.now))

    with pytest.raises(QuotaExhaustedError, match="decider"):
        ledger.resolve(AgentRole.DECIDER, settings.role_choices(AgentRole.DECIDER))


def test_budget_recovers_when_the_window_expires() -> None:
    """Agotada la cuota, el primario vuelve a estar disponible al salir la ventana."""
    clock = FakeClock()
    settings = make_settings(SCARCE, CHEAP)
    choices = settings.role_choices(AgentRole.STRUCTURE)
    ledger = QuotaLedger(WINDOW, clock)
    ledger.record(make_call(SCARCE, clock.now, role=AgentRole.STRUCTURE))
    ledger.record(make_call(SCARCE, clock.now, role=AgentRole.STRUCTURE))
    assert ledger.resolve(AgentRole.STRUCTURE, choices) == CHEAP

    clock.advance(timedelta(hours=5, seconds=1))
    assert ledger.resolve(AgentRole.STRUCTURE, choices) == SCARCE


# ───────────────────────────────────────── Siembra al arrancar ────────────────────────────────────
# El contador vive en memoria y el proveedor no: tras una interrupción, uno limpio
# cree tener la ventana entera mientras el gateway sigue contando lo de antes.


def test_seed_counts_only_what_is_still_inside_the_window() -> None:
    """N llamadas dentro de la ventana y M fuera: cuenta N.

    Lo que ya expiró no le resta nada al presupuesto de ahora, y contarlo dejaría
    al sistema sin cuota por un gasto que el proveedor ya olvidó.
    """
    clock = FakeClock()
    ledger = QuotaLedger(WINDOW, clock)
    inside = [make_call(SCARCE, START - timedelta(hours=hours)) for hours in (1, 2, 4)]
    outside = [make_call(SCARCE, START - timedelta(hours=hours)) for hours in (5, 6)]

    counted = ledger.seed([*outside, *inside])

    assert counted == 3
    assert ledger.used(AgentRole.DECIDER, SCARCE.model) == 6.0


def test_seed_ignores_calls_dated_after_now() -> None:
    """Una llamada «del futuro» no pudo consumir la ventana que termina ahora."""
    ledger = QuotaLedger(WINDOW, FakeClock())
    assert ledger.seed([make_call(SCARCE, START + timedelta(minutes=1))]) == 0


def test_seed_ignores_the_local_backend() -> None:
    """Solo se siembra lo que un proveedor remoto sigue recordando.

    El servidor local no lleva cuenta entre procesos: su límite es una declaración
    nuestra, no una ventana que alguien más esté midiendo.
    """
    ledger = QuotaLedger(WINDOW, FakeClock())
    recent = START - timedelta(minutes=10)

    assert ledger.seed([make_call(CHEAP, recent), make_call(SCARCE, recent)]) == 1
    assert ledger.used(AgentRole.DECIDER, CHEAP.model) == 0.0
    assert ledger.used(AgentRole.DECIDER, SCARCE.model) == 2.0


def test_seed_ignores_cache_hits() -> None:
    """Un acierto de caché no llegó al proveedor ni la primera vez."""
    ledger = QuotaLedger(WINDOW, FakeClock())
    assert ledger.seed([make_call(SCARCE, START - timedelta(minutes=10), cache_hit=True)]) == 0


def test_seeded_calls_expire_like_recorded_ones() -> None:
    """Lo sembrado entra en la cola en orden, así que la purga lo alcanza.

    `_purge` solo mira la cabeza. Sembrar desordenado, o detrás de una llamada más
    reciente, dejaría una entrada vieja atascada contando para siempre.
    """
    clock = FakeClock()
    ledger = QuotaLedger(WINDOW, clock)
    ledger.record(make_call(SCARCE, START))
    ledger.seed(
        [
            make_call(SCARCE, START - timedelta(hours=1)),
            make_call(SCARCE, START - timedelta(hours=4)),
        ]
    )
    assert ledger.used(AgentRole.DECIDER, SCARCE.model) == 6.0

    clock.advance(timedelta(hours=2))
    assert ledger.used(AgentRole.DECIDER, SCARCE.model) == 4.0
    clock.advance(timedelta(hours=2, minutes=30))
    assert ledger.used(AgentRole.DECIDER, SCARCE.model) == 2.0


def test_a_seeded_ledger_degrades_where_a_clean_one_would_not() -> None:
    """La consecuencia que importa: al reanudar, el primario ya no cabe."""
    settings = make_settings(SCARCE, CHEAP)
    choices = settings.role_choices(AgentRole.STRUCTURE)
    spent = [
        make_call(SCARCE, START - timedelta(minutes=minutes), role=AgentRole.STRUCTURE)
        for minutes in (5, 10)
    ]

    clean = QuotaLedger(WINDOW, FakeClock())
    seeded = QuotaLedger(WINDOW, FakeClock())
    seeded.seed(spent)

    assert clean.resolve(AgentRole.STRUCTURE, choices) == SCARCE
    assert seeded.resolve(AgentRole.STRUCTURE, choices) == CHEAP
