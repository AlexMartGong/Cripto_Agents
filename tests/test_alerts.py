"""Pruebas de las alertas.

Los registros se construyen a mano: para comprobar un umbral hay que poder poner
el valor justo por encima y justo por debajo.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from crypto_agents.alerts import (
    AlertKind,
    AlertThresholds,
    evaluate_alerts,
    quota_alerts,
    repeated_veto_alerts,
    skipped_cycle_alerts,
    validation_alerts,
)
from crypto_agents.journal import EvaluationRecord
from crypto_agents.runner import RUNNER_NODE
from crypto_agents.settings import ModelChoice, RoleConfig, Settings, load_settings
from crypto_agents.state import AgentRole, Backend, LLMCall, NodeError, RiskVerdict
from tests.conftest import role_map

NOW = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)
DIGEST = "e" * 64


def settings_with(quota: int = 100) -> Settings:
    """Configuración donde cada rol tiene la cuota indicada."""
    choice = ModelChoice(
        backend=Backend.OLLAMA, model="modelo", family="fam", quota_per_window=quota
    )
    roles = role_map(primary=choice)
    roles[AgentRole.BEAR] = RoleConfig(
        primary=choice.model_copy(update={"family": "fam-alt", "model": "modelo-bear"})
    )
    return load_settings(roles=roles, ollama={"host": "http://localhost:11434"})


def call(
    role: AgentRole = AgentRole.STRUCTURE,
    model: str = "modelo",
    at: datetime = NOW,
    valid: bool = True,
    cache_hit: bool = False,
) -> LLMCall:
    """Llamada registrada."""
    return LLMCall(
        role=role,
        backend=Backend.OLLAMA,
        model=model,
        quota_weight=1.0,
        prompt_digest=DIGEST,
        cache_hit=cache_hit,
        valid=valid,
        latency_ms=10.0,
        at=at,
    )


def record(
    calls: list[LLMCall] | None = None,
    veto_rule: str | None = None,
    errors: tuple[NodeError, ...] = (),
    symbol: str = "BTC/USDT",
) -> EvaluationRecord:
    """Registro con solo lo que la alerta mira."""
    risk = (
        None
        if veto_rule is None
        else RiskVerdict(
            approved=False,
            final_size_fraction=0.0,
            veto_rule=veto_rule,
            veto_reason=f"veto de {veto_rule}",
        )
    )
    return EvaluationRecord(
        run_id=uuid4(),
        at=NOW,
        symbol=symbol,
        timeframe="4h",
        risk=risk,
        calls=tuple(calls or []),
        errors=errors,
    )


# ─────────────────────────────────────────── Cuota ────────────────────────────────────────────────


def test_quota_alerts_when_the_window_is_nearly_spent() -> None:
    """Avisa antes de agotarse, no después: después ya no hay nada que hacer."""
    spent = [record([call() for _ in range(85)])]
    alerts = quota_alerts(spent, settings_with(quota=100), NOW, AlertThresholds())

    low = [item for item in alerts if item.subject.startswith("structure/")]
    assert low, "no avisó del rol que gastó el 85% de su ventana"
    assert low[0].kind is AlertKind.QUOTA_LOW
    assert "15%" in low[0].detail


def test_quota_alerts_stay_quiet_with_room_left() -> None:
    """Con presupuesto de sobra no hay nada que decir."""
    alerts = quota_alerts([record([call()])], settings_with(quota=100), NOW, AlertThresholds())
    assert [item for item in alerts if item.subject.startswith("structure/")] == []


def test_calls_outside_the_window_do_not_count() -> None:
    """La ventana es deslizante: lo de hace seis horas ya se recuperó."""
    old = NOW - timedelta(hours=6)
    spent = [record([call(at=old) for _ in range(95)])]
    alerts = quota_alerts(spent, settings_with(quota=100), NOW, AlertThresholds())
    assert [item for item in alerts if item.subject.startswith("structure/")] == []


def test_cache_hits_do_not_consume_the_window() -> None:
    """Un acierto de caché no llegó al proveedor, así que no gastó cuota."""
    spent = [record([call(cache_hit=True) for _ in range(95)])]
    alerts = quota_alerts(spent, settings_with(quota=100), NOW, AlertThresholds())
    assert [item for item in alerts if item.subject.startswith("structure/")] == []


# ──────────────────────────────────────── Veto repetido ───────────────────────────────────────────


def test_a_rule_that_vetoes_repeatedly_alerts() -> None:
    """Tres vetos de la misma regla en la ventana: algo pasa con la cuenta, no con el mercado."""
    records = [record(veto_rule="daily_drawdown") for _ in range(3)]
    alerts = repeated_veto_alerts(records, AlertThresholds())

    assert len(alerts) == 1
    assert alerts[0].subject == "daily_drawdown"
    assert alerts[0].kind is AlertKind.REPEATED_VETO


def test_a_single_veto_is_not_a_pattern() -> None:
    """Un veto suelto es el gate haciendo su trabajo."""
    assert repeated_veto_alerts([record(veto_rule="cooldown")], AlertThresholds()) == []


def test_only_the_recent_window_counts() -> None:
    """Vetos viejos empujados fuera de la ventana no cuentan."""
    records = [
        *[record(veto_rule="cooldown") for _ in range(3)],
        *[record() for _ in range(10)],
    ]
    assert repeated_veto_alerts(records, AlertThresholds(veto_window=10)) == []


def test_the_kill_switch_does_not_raise_a_repeated_veto_alert() -> None:
    """Si lo pusiste tú, repetirse es su trabajo, no una anomalía."""
    records = [record(veto_rule="kill_switch") for _ in range(8)]
    assert repeated_veto_alerts(records, AlertThresholds()) == []


# ────────────────────────────────────── Fallos de validación ──────────────────────────────────────


def test_a_backend_that_keeps_failing_validation_alerts() -> None:
    """Un modelo que reintenta la mitad de las veces no está ahorrando nada."""
    calls = [call(valid=False) for _ in range(4)] + [call() for _ in range(4)]
    alerts = validation_alerts([record(calls)], AlertThresholds())

    assert len(alerts) == 1
    assert alerts[0].kind is AlertKind.VALIDATION_FAILURES
    assert alerts[0].value == 0.5


def test_a_rate_over_too_few_attempts_is_not_reported() -> None:
    """Un fallo sobre un intento es el 100% y no significa nada."""
    assert validation_alerts([record([call(valid=False)])], AlertThresholds()) == []


# ──────────────────────────────────────── Ciclos saltados ─────────────────────────────────────────


def test_skipped_cycles_alert_per_symbol() -> None:
    """La pregunta es qué mercado se quedó sin evaluar."""
    skipped = record(
        errors=(NodeError(node=RUNNER_NODE, message="ciclo omitido: 2 cierres", at=NOW),),
        symbol="ETH/USDT",
    )
    alerts = skipped_cycle_alerts([skipped], AlertThresholds())

    assert len(alerts) == 1
    assert alerts[0].subject == "ETH/USDT"
    assert alerts[0].kind is AlertKind.SKIPPED_CYCLES


def test_a_node_failure_is_not_a_skipped_cycle() -> None:
    """Un agente que falló no es lo mismo que un ciclo que nunca corrió."""
    failed = record(errors=(NodeError(node="structure", message="cuota agotada", at=NOW),))
    assert skipped_cycle_alerts([failed], AlertThresholds()) == []


# ──────────────────────────────────────────── Conjunto ────────────────────────────────────────────


def test_a_healthy_run_raises_nothing() -> None:
    """Sin alertas es el estado normal; si siempre hubiera, nadie las leería."""
    healthy = [record([call()]) for _ in range(5)]
    assert evaluate_alerts(healthy, settings_with(quota=1000), NOW) == []


def test_every_alert_carries_the_number_that_triggered_it() -> None:
    """Un aviso sin el dato obliga a repetir la consulta a mano."""
    records = [record([call(valid=False) for _ in range(6)], veto_rule="cooldown")]
    for alert in evaluate_alerts(records, settings_with(quota=10), NOW):
        assert alert.detail
        assert alert.threshold >= 0.0
