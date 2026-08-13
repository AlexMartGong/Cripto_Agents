"""Pruebas del gate de riesgo.

La parte importante no son los casos concretos sino la propiedad del final: para
cualquier `Decision` y cualquier estado de cuenta que hypothesis sepa construir,
la orden resultante respeta todos los límites o no existe.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from hypothesis import given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st
from pydantic import ValidationError

from crypto_agents.execution import build_order
from crypto_agents.risk import (
    AccountState,
    RiskLimits,
    apply_risk,
    cap_position_size,
    cap_total_exposure,
)
from crypto_agents.state import (
    Action,
    Decision,
    ExecutionMode,
    MarketSnapshot,
    RiskVerdict,
    Side,
)

NOW = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)
LIMITS = RiskLimits(
    max_position_fraction=0.10,
    max_total_exposure_fraction=0.30,
    max_daily_drawdown_fraction=0.05,
    cooldown_after_loss_minutes=240,
)
HEALTHY = AccountState(equity=10_000.0, day_start_equity=10_000.0)


def buy(size: float = 0.05, action: Action = Action.BUY) -> Decision:
    """Decisión accionable con el tamaño pedido."""
    return Decision(
        action=action,
        confidence=0.8,
        size_fraction=size,
        invalidation_price=99.0,
        rationale="Estructura, impulso y volumen coinciden en direccion alcista.",
        dismissed_side=Side.BEAR,
        dismissal_reason="Su contraargumento depende de un nivel que ya se perdio.",
    )


def snapshot() -> MarketSnapshot:
    """Snapshot mínimo para construir órdenes."""
    from uuid import uuid4

    return MarketSnapshot(
        run_id=uuid4(),
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1h",
        timestamp=NOW,
        close=100.0,
        candles_digest="a" * 64,
        candles_count=300,
    )


# ───────────────────────────────────────────── Límites ────────────────────────────────────────────


def test_incoherent_limits_are_rejected() -> None:
    """Un tamaño por operación mayor que la exposición total no tiene sentido."""
    with pytest.raises(ValidationError, match="max_position_fraction"):
        RiskLimits(max_position_fraction=0.5, max_total_exposure_fraction=0.3)


def test_hold_passes_without_generating_an_order() -> None:
    """Un `hold` no es un veto: pasa con tamaño cero."""
    verdict = apply_risk(buy(action=Action.HOLD, size=0.0), HEALTHY, LIMITS, NOW)
    assert verdict.approved is True
    assert verdict.final_size_fraction == 0.0
    assert (
        build_order(buy(action=Action.HOLD, size=0.0), verdict, snapshot(), ExecutionMode.PAPER)
        is None
    )


# ────────────────────────────────────────────── Vetos ─────────────────────────────────────────────


def test_kill_switch_vetoes_everything() -> None:
    """Con el interruptor puesto no sale nada, pase lo que pase."""
    limits = LIMITS.model_copy(update={"kill_switch": True})
    verdict = apply_risk(buy(), HEALTHY, limits, NOW)
    assert verdict.approved is False
    assert verdict.veto_reason == "kill switch activo"
    assert verdict.final_size_fraction == 0.0


def test_kill_switch_wins_over_a_perfect_setup() -> None:
    """El orden importa: el interruptor corta antes de mirar nada más."""
    limits = LIMITS.model_copy(update={"kill_switch": True})
    account = AccountState(equity=20_000.0, day_start_equity=10_000.0)
    assert apply_risk(buy(0.01), account, limits, NOW).approved is False


def test_daily_drawdown_vetoes() -> None:
    """Perdido el drawdown del día, se deja de operar."""
    account = AccountState(equity=9_400.0, day_start_equity=10_000.0)
    verdict = apply_risk(buy(), account, LIMITS, NOW)
    assert verdict.approved is False
    assert "drawdown diario" in (verdict.veto_reason or "")


def test_drawdown_just_below_the_limit_passes() -> None:
    """El límite es un umbral, no una zona difusa."""
    account = AccountState(equity=9_501.0, day_start_equity=10_000.0)
    assert apply_risk(buy(), account, LIMITS, NOW).approved is True


def test_drawdown_exactly_at_the_limit_vetoes() -> None:
    """Alcanzar el límite ya cuenta como alcanzarlo."""
    account = AccountState(equity=9_500.0, day_start_equity=10_000.0)
    assert apply_risk(buy(), account, LIMITS, NOW).approved is False


def test_cooldown_vetoes_right_after_a_loss() -> None:
    """Operar en caliente es cómo se encadenan las pérdidas."""
    account = AccountState(
        equity=10_000.0, day_start_equity=10_000.0, last_loss_at=NOW - timedelta(minutes=30)
    )
    verdict = apply_risk(buy(), account, LIMITS, NOW)
    assert verdict.approved is False
    assert "cooldown" in (verdict.veto_reason or "")


def test_cooldown_expires() -> None:
    """Pasado el cooldown se vuelve a operar."""
    account = AccountState(
        equity=10_000.0, day_start_equity=10_000.0, last_loss_at=NOW - timedelta(minutes=241)
    )
    assert apply_risk(buy(), account, LIMITS, NOW).approved is True


def test_cooldown_can_be_disabled() -> None:
    """Con cooldown de cero minutos la regla no aplica."""
    limits = LIMITS.model_copy(update={"cooldown_after_loss_minutes": 0})
    account = AccountState(
        equity=10_000.0, day_start_equity=10_000.0, last_loss_at=NOW - timedelta(seconds=1)
    )
    assert apply_risk(buy(), account, limits, NOW).approved is True


# ───────────────────────────────────────────── Recortes ───────────────────────────────────────────


def test_position_size_is_capped() -> None:
    """Un modelo puede pedir todo el capital; sale recortado al máximo por operación."""
    verdict = apply_risk(buy(1.0), HEALTHY, LIMITS, NOW)
    assert verdict.approved is True
    assert verdict.final_size_fraction == 0.10
    assert "max_position_fraction" in verdict.applied_limits


def test_size_below_the_cap_passes_untouched() -> None:
    """Un tamaño razonable no se toca y no registra recorte."""
    verdict = apply_risk(buy(0.03), HEALTHY, LIMITS, NOW)
    assert verdict.final_size_fraction == 0.03
    assert verdict.applied_limits == ()


def test_total_exposure_caps_further() -> None:
    """Con exposición abierta el hueco restante manda."""
    account = AccountState(equity=10_000.0, day_start_equity=10_000.0, open_exposure_fraction=0.27)
    verdict = apply_risk(buy(1.0), account, LIMITS, NOW)
    assert verdict.final_size_fraction == pytest.approx(0.03)
    assert verdict.applied_limits == ("max_position_fraction", "max_total_exposure_fraction")


def test_no_headroom_becomes_a_veto() -> None:
    """Un recorte hasta cero no es una orden diminuta: es un veto."""
    account = AccountState(equity=10_000.0, day_start_equity=10_000.0, open_exposure_fraction=0.30)
    verdict = apply_risk(buy(), account, LIMITS, NOW)
    assert verdict.approved is False
    assert verdict.veto_reason == "no queda hueco de exposición"
    assert verdict.final_size_fraction == 0.0


def test_cap_helpers_report_which_limit_bit() -> None:
    """Cada recorte dice qué límite lo produjo, para que el registro sea auditable."""
    assert cap_position_size(0.5, LIMITS) == (0.10, "max_position_fraction")
    assert cap_position_size(0.05, LIMITS) == (0.05, None)
    assert cap_total_exposure(0.5, HEALTHY, LIMITS) == (0.30, "max_total_exposure_fraction")


# ───────────────────────── La propiedad: ninguna Decision viola un límite ─────────────────────────


@st.composite
def decisions(draw: st.DrawFn) -> Decision:
    """Cualquier `Decision` que el contrato acepte, incluidas las extremas.

    La estrategia respeta los validadores de fase 1 (una acción distinta de
    `hold` exige invalidación, tamaño positivo y mesa descartada). Generar
    objetos inválidos probaría el contrato, no el gate de riesgo.
    """
    action = draw(st.sampled_from(Action))
    confidence = draw(st.floats(0.0, 1.0))
    rationale = "Un racional cualquiera con longitud suficiente para validar."
    if action is Action.HOLD:
        return Decision(
            action=action,
            confidence=confidence,
            size_fraction=draw(st.floats(0.0, 1.0)),
            rationale=rationale,
        )
    return Decision(
        action=action,
        confidence=confidence,
        size_fraction=draw(st.floats(1e-6, 1.0)),
        invalidation_price=draw(st.floats(0.01, 1e9)),
        rationale=rationale,
        dismissed_side=draw(st.sampled_from(Side)),
        dismissal_reason="Una razon cualquiera de descarte.",
    )


accounts = st.builds(
    AccountState,
    equity=st.floats(1.0, 1e9),
    day_start_equity=st.floats(1.0, 1e9),
    open_exposure_fraction=st.floats(0.0, 1.0),
    last_loss_at=st.none()
    | st.datetimes(min_value=datetime(2026, 8, 1), max_value=NOW.replace(tzinfo=None)).map(
        lambda value: value.replace(tzinfo=UTC)
    ),
)
"""Cuentas posibles. La última pérdida nunca es posterior a `NOW`.

El límite superior era el 2026-08-20, siete días por delante del instante
evaluado. Una pérdida registrada en el futuro no es un estado en el que la cuenta
pueda estar, y generaba un contraejemplo espurio: con `cooldown_after_loss_minutes`
en cero la regla está desactivada, así que la orden pasa y el tiempo transcurrido
sale negativo sin que nada esté mal.
"""

limits = st.builds(
    RiskLimits,
    max_position_fraction=st.floats(0.001, 0.5),
    max_total_exposure_fraction=st.floats(0.5, 1.0),
    max_daily_drawdown_fraction=st.floats(0.001, 1.0),
    cooldown_after_loss_minutes=st.integers(0, 1440),
    kill_switch=st.booleans(),
)


@given(decision=decisions(), account=accounts, risk_limits=limits)
@hypothesis_settings(max_examples=500)
def test_no_decision_can_produce_an_order_that_breaks_a_limit(
    decision: Decision, account: AccountState, risk_limits: RiskLimits
) -> None:
    """La propiedad central de la fase: da igual lo que diga el modelo.

    Se construye la orden por el único camino que existe —`apply_risk` y luego
    `build_order`— y se comprueban todos los límites sobre el resultado.
    """
    verdict = apply_risk(decision, account, risk_limits, NOW)
    order = build_order(decision, verdict, snapshot(), ExecutionMode.PAPER)

    if order is None:
        return

    assert risk_limits.kill_switch is False
    assert account.daily_drawdown_fraction < risk_limits.max_daily_drawdown_fraction
    assert order.size_fraction <= risk_limits.max_position_fraction
    assert (
        account.open_exposure_fraction + order.size_fraction
        <= risk_limits.max_total_exposure_fraction + 1e-9
    )
    assert order.size_fraction <= decision.size_fraction
    assert order.size_fraction == verdict.final_size_fraction
    assert order.side is not Action.HOLD

    if account.last_loss_at is not None:
        elapsed = (NOW - account.last_loss_at).total_seconds() / 60.0
        assert elapsed >= risk_limits.cooldown_after_loss_minutes


@given(decision=decisions(), account=accounts, risk_limits=limits)
@hypothesis_settings(max_examples=300)
def test_a_veto_never_carries_size(
    decision: Decision, account: AccountState, risk_limits: RiskLimits
) -> None:
    """Un veto con tamaño no es un veto; el contrato ya lo prohíbe y aquí se confirma."""
    verdict = apply_risk(decision, account, risk_limits, NOW)
    if not verdict.approved:
        assert verdict.final_size_fraction == 0.0
        assert verdict.veto_reason
        assert build_order(decision, verdict, snapshot(), ExecutionMode.PAPER) is None


@given(size=st.floats(1e-6, 1.0))
def test_order_size_never_comes_from_the_decision(size: float) -> None:
    """El tamaño de la orden sale del veredicto, no de lo que pidió el modelo."""
    verdict = RiskVerdict(approved=True, final_size_fraction=0.01)
    order = build_order(buy(size), verdict, snapshot(), ExecutionMode.PAPER)
    assert order is not None
    assert order.size_fraction == 0.01


def test_the_property_actually_builds_orders() -> None:
    """Guarda contra una propiedad vacía.

    Si un cambio hiciera que `build_order` devolviese siempre `None`, el test de
    la propiedad seguiría pasando sin comprobar nada. Este fija que el espacio
    explorado contiene tanto órdenes reales como vetos.
    """
    built = 0
    vetoed = 0
    for index in range(200):
        size = (index % 100) / 100.0 + 0.005
        exposure = (index % 7) / 10.0
        account = AccountState(
            equity=10_000.0 - (index % 10) * 40.0,
            day_start_equity=10_000.0,
            open_exposure_fraction=exposure,
        )
        verdict = apply_risk(buy(size), account, LIMITS, NOW)
        if build_order(buy(size), verdict, snapshot(), ExecutionMode.PAPER) is not None:
            built += 1
        if not verdict.approved:
            vetoed += 1

    assert built > 0
    assert vetoed > 0
