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
    INVALID_STOP_SIDE,
    VETOES,
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
    OrderIntent,
    RiskVerdict,
    Side,
    stop_on_wrong_side,
)

NOW = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)
CLOSE = 100.0
"""Cierre de la vela evaluada en los casos concretos. La invalidación de `buy()` es 99."""
LIMITS = RiskLimits(
    max_position_fraction=0.10,
    max_total_exposure_fraction=0.30,
    max_daily_drawdown_fraction=0.05,
    cooldown_after_loss_minutes=240,
)
HEALTHY = AccountState(equity=10_000.0, day_start_equity=10_000.0)


def buy(
    size: float = 0.05, action: Action = Action.BUY, invalidation: float | None = 99.0
) -> Decision:
    """Decisión accionable con el tamaño pedido."""
    return Decision(
        action=action,
        confidence=0.8,
        size_fraction=size,
        invalidation_price=None if action is Action.HOLD else invalidation,
        rationale="Estructura, impulso y volumen coinciden en direccion alcista.",
        dismissed_side=Side.BEAR,
        dismissal_reason="Su contraargumento depende de un nivel que ya se perdio.",
    )


def snapshot(close: float = CLOSE) -> MarketSnapshot:
    """Snapshot mínimo para construir órdenes."""
    from uuid import uuid4

    return MarketSnapshot(
        run_id=uuid4(),
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1h",
        timestamp=NOW,
        close=close,
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
    verdict = apply_risk(buy(action=Action.HOLD, size=0.0), HEALTHY, LIMITS, NOW, CLOSE)
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
    verdict = apply_risk(buy(), HEALTHY, limits, NOW, CLOSE)
    assert verdict.approved is False
    assert verdict.veto_reason == "kill switch activo"
    assert verdict.final_size_fraction == 0.0


def test_kill_switch_wins_over_a_perfect_setup() -> None:
    """El interruptor corta aunque la cuenta vaya ganando y el tamaño sea mínimo."""
    limits = LIMITS.model_copy(update={"kill_switch": True})
    account = AccountState(equity=20_000.0, day_start_equity=10_000.0)
    assert apply_risk(buy(0.01), account, limits, NOW, CLOSE).approved is False


def test_daily_drawdown_vetoes() -> None:
    """Perdido el drawdown del día, se deja de operar."""
    account = AccountState(equity=9_400.0, day_start_equity=10_000.0)
    verdict = apply_risk(buy(), account, LIMITS, NOW, CLOSE)
    assert verdict.approved is False
    assert "drawdown diario" in (verdict.veto_reason or "")


def test_drawdown_just_below_the_limit_passes() -> None:
    """El límite es un umbral, no una zona difusa."""
    account = AccountState(equity=9_501.0, day_start_equity=10_000.0)
    assert apply_risk(buy(), account, LIMITS, NOW, CLOSE).approved is True


def test_drawdown_exactly_at_the_limit_vetoes() -> None:
    """Alcanzar el límite ya cuenta como alcanzarlo."""
    account = AccountState(equity=9_500.0, day_start_equity=10_000.0)
    assert apply_risk(buy(), account, LIMITS, NOW, CLOSE).approved is False


def test_cooldown_vetoes_right_after_a_loss() -> None:
    """Operar en caliente es cómo se encadenan las pérdidas."""
    account = AccountState(
        equity=10_000.0, day_start_equity=10_000.0, last_loss_at=NOW - timedelta(minutes=30)
    )
    verdict = apply_risk(buy(), account, LIMITS, NOW, CLOSE)
    assert verdict.approved is False
    assert "cooldown" in (verdict.veto_reason or "")


def test_cooldown_expires() -> None:
    """Pasado el cooldown se vuelve a operar."""
    account = AccountState(
        equity=10_000.0, day_start_equity=10_000.0, last_loss_at=NOW - timedelta(minutes=241)
    )
    assert apply_risk(buy(), account, LIMITS, NOW, CLOSE).approved is True


def test_cooldown_can_be_disabled() -> None:
    """Con cooldown de cero minutos la regla no aplica."""
    limits = LIMITS.model_copy(update={"cooldown_after_loss_minutes": 0})
    account = AccountState(
        equity=10_000.0, day_start_equity=10_000.0, last_loss_at=NOW - timedelta(seconds=1)
    )
    assert apply_risk(buy(), account, limits, NOW, CLOSE).approved is True


# ───────────────────────────────────── El lado del stop ───────────────────────────────────────────
# Un `buy` cuya invalidación está por encima del cierre no tiene stop: tiene un
# precio que ya se cumplió. La primera ablación emitió órdenes así y `outcomes`
# las puntuó, porque nada entre el decisor y el mercado miraba de qué lado quedaba.


def test_a_buy_with_its_stop_above_the_close_is_vetoed() -> None:
    """Invalidación 101 con cierre 100: en un largo, eso no es una invalidación."""
    verdict = apply_risk(buy(invalidation=101.0), HEALTHY, LIMITS, NOW, CLOSE)

    assert verdict.approved is False
    assert verdict.veto_rule == INVALID_STOP_SIDE == "invalid_stop_side"
    assert verdict.final_size_fraction == 0.0
    assert "101" in (verdict.veto_reason or "")
    assert "100" in (verdict.veto_reason or "")


def test_a_sell_with_its_stop_below_the_close_is_vetoed() -> None:
    """El caso simétrico: en un corto la tesis se rompe hacia arriba."""
    verdict = apply_risk(buy(action=Action.SELL, invalidation=99.0), HEALTHY, LIMITS, NOW, CLOSE)
    assert verdict.veto_rule == INVALID_STOP_SIDE


def test_a_sell_with_its_stop_above_the_close_passes() -> None:
    """El lado correcto de un corto no se veta."""
    verdict = apply_risk(buy(action=Action.SELL, invalidation=101.0), HEALTHY, LIMITS, NOW, CLOSE)
    assert verdict.approved is True


@pytest.mark.parametrize("action", [Action.BUY, Action.SELL])
def test_a_stop_at_the_close_is_on_the_wrong_side(action: Action) -> None:
    """Sin recorrido no hay stop: la igualdad se veta en los dos sentidos."""
    verdict = apply_risk(buy(action=action, invalidation=CLOSE), HEALTHY, LIMITS, NOW, CLOSE)
    assert verdict.veto_rule == INVALID_STOP_SIDE


def test_the_stop_side_is_checked_before_every_other_veto() -> None:
    """Con el interruptor puesto y la cuenta en drawdown, la regla registrada es la del stop.

    Las otras tres hablan del estado del sistema; esta, de la propuesta. Evaluarla
    primero deja en el journal que el decisor propuso algo incoherente aunque en
    ese momento no se fuera a operar de todos modos.
    """
    limits = LIMITS.model_copy(update={"kill_switch": True})
    account = AccountState(equity=9_000.0, day_start_equity=10_000.0)

    verdict = apply_risk(buy(invalidation=101.0), account, limits, NOW, CLOSE)

    assert verdict.veto_rule == INVALID_STOP_SIDE
    assert [rule for rule, _ in VETOES] == [
        "invalid_stop_side",
        "kill_switch",
        "daily_drawdown",
        "cooldown",
    ]


def test_a_hold_has_no_stop_to_check() -> None:
    """`hold` pasa con tamaño cero sea cual sea el cierre."""
    assert apply_risk(buy(action=Action.HOLD, size=0.0), HEALTHY, LIMITS, NOW, CLOSE).approved


# ───────────────────────────────────────────── Recortes ───────────────────────────────────────────


def test_position_size_is_capped() -> None:
    """Un modelo puede pedir todo el capital; sale recortado al máximo por operación."""
    verdict = apply_risk(buy(1.0), HEALTHY, LIMITS, NOW, CLOSE)
    assert verdict.approved is True
    assert verdict.final_size_fraction == 0.10
    assert "max_position_fraction" in verdict.applied_limits


def test_size_below_the_cap_passes_untouched() -> None:
    """Un tamaño razonable no se toca y no registra recorte."""
    verdict = apply_risk(buy(0.03), HEALTHY, LIMITS, NOW, CLOSE)
    assert verdict.final_size_fraction == 0.03
    assert verdict.applied_limits == ()


def test_total_exposure_caps_further() -> None:
    """Con exposición abierta el hueco restante manda."""
    account = AccountState(equity=10_000.0, day_start_equity=10_000.0, open_exposure_fraction=0.27)
    verdict = apply_risk(buy(1.0), account, LIMITS, NOW, CLOSE)
    assert verdict.final_size_fraction == pytest.approx(0.03)
    assert verdict.applied_limits == ("max_position_fraction", "max_total_exposure_fraction")


def test_no_headroom_becomes_a_veto() -> None:
    """Un recorte hasta cero no es una orden diminuta: es un veto."""
    account = AccountState(equity=10_000.0, day_start_equity=10_000.0, open_exposure_fraction=0.30)
    verdict = apply_risk(buy(), account, LIMITS, NOW, CLOSE)
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
            invalidation_price=None,
            rationale=rationale,
            dismissed_side=None,
            dismissal_reason=None,
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


prices = st.floats(0.01, 1e9)
"""Cierres posibles, en el mismo rango que las invalidaciones que genera `decisions`."""


@given(decision=decisions(), account=accounts, risk_limits=limits, close=prices)
@hypothesis_settings(max_examples=500)
def test_no_decision_can_produce_an_order_that_breaks_a_limit(
    decision: Decision, account: AccountState, risk_limits: RiskLimits, close: float
) -> None:
    """La propiedad central de la fase: da igual lo que diga el modelo.

    Se construye la orden por el único camino que existe —`apply_risk` y luego
    `build_order`— y se comprueban todos los límites sobre el resultado.
    """
    verdict = apply_risk(decision, account, risk_limits, NOW, close)
    order = build_order(decision, verdict, snapshot(close), ExecutionMode.PAPER)

    if order is None:
        return

    if order.side is Action.BUY:
        assert order.invalidation_price < order.reference_price
    else:
        assert order.invalidation_price > order.reference_price
    assert order.reference_price == close
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


@given(decision=decisions(), account=accounts, risk_limits=limits, close=prices)
@hypothesis_settings(max_examples=300)
def test_a_veto_never_carries_size(
    decision: Decision, account: AccountState, risk_limits: RiskLimits, close: float
) -> None:
    """Un veto con tamaño no es un veto; el contrato ya lo prohíbe y aquí se confirma."""
    verdict = apply_risk(decision, account, risk_limits, NOW, close)
    if not verdict.approved:
        assert verdict.final_size_fraction == 0.0
        assert verdict.veto_reason
        assert build_order(decision, verdict, snapshot(close), ExecutionMode.PAPER) is None


@given(decision=decisions(), account=accounts, risk_limits=limits, close=prices)
@hypothesis_settings(max_examples=500)
def test_a_stop_on_the_wrong_side_is_always_vetoed_under_its_own_rule(
    decision: Decision, account: AccountState, risk_limits: RiskLimits, close: float
) -> None:
    """Para cualquier precio y cualquier lado: stop equivocado ⇔ veto `invalid_stop_side`.

    En los dos sentidos. Que todo stop equivocado se vete es la garantía; que solo
    ellos lleven esa regla es lo que hace que el recuento por `veto_rule` cuente lo
    que dice contar, pase lo que pase con el interruptor, el drawdown o el cooldown.
    """
    verdict = apply_risk(decision, account, risk_limits, NOW, close)
    wrong = decision.action is not Action.HOLD and (
        decision.invalidation_price is not None
        and (
            decision.invalidation_price >= close
            if decision.action is Action.BUY
            else decision.invalidation_price <= close
        )
    )

    assert (verdict.veto_rule == INVALID_STOP_SIDE) is wrong


@given(side=st.sampled_from([Action.BUY, Action.SELL]), invalidation=prices, reference=prices)
@hypothesis_settings(max_examples=500)
def test_an_order_with_its_stop_on_the_wrong_side_cannot_be_built(
    side: Action, invalidation: float, reference: float
) -> None:
    """El respaldo del veto: `OrderIntent` se construye si y solo si el lado es correcto.

    El veto es la vía normal. Esto es lo que queda si un camino futuro llega a
    `build_order` sin pasar por el gate: la orden no existe, en vez de existir mal.
    """
    wrong = invalidation >= reference if side is Action.BUY else invalidation <= reference
    assert stop_on_wrong_side(side, invalidation, reference) is wrong

    def build() -> OrderIntent:
        return OrderIntent(
            symbol="BTC/USDT",
            side=side,
            size_fraction=0.05,
            reference_price=reference,
            invalidation_price=invalidation,
            mode=ExecutionMode.PAPER,
        )

    if wrong:
        with pytest.raises(ValidationError, match="lado equivocado"):
            build()
    else:
        assert build().side is side


def test_the_stop_property_sees_both_sides() -> None:
    """Guarda contra una propiedad vacía: la rejilla contiene vetos por stop y órdenes."""
    vetoed = built = 0
    for step in range(1, 40):
        for action in (Action.BUY, Action.SELL):
            decision = buy(action=action, invalidation=80.0 + step)
            verdict = apply_risk(decision, HEALTHY, LIMITS, NOW, CLOSE)
            vetoed += verdict.veto_rule == INVALID_STOP_SIDE
            built += build_order(decision, verdict, snapshot(), ExecutionMode.PAPER) is not None

    assert vetoed == 40, "19 sell por debajo, 1 sell en el cierre, 19 buy por encima, 1 buy en él"
    assert built == 38


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
        verdict = apply_risk(buy(size), account, LIMITS, NOW, CLOSE)
        if build_order(buy(size), verdict, snapshot(), ExecutionMode.PAPER) is not None:
            built += 1
        if not verdict.approved:
            vetoed += 1

    assert built > 0
    assert vetoed > 0
