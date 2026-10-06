"""Pruebas de la salida: construcción de órdenes y modo de ejecución."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import SecretStr, ValidationError

from crypto_agents.execution import (
    CcxtExecutor,
    ExecutionSettings,
    LiveExecutionRefusedError,
    PaperExecutor,
    build_order,
)
from crypto_agents.settings import ExchangeSettings
from crypto_agents.state import (
    Action,
    Decision,
    ExecutionMode,
    MarketSnapshot,
    OrderIntent,
    RiskVerdict,
    Side,
)

NOW = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)
SNAPSHOT = MarketSnapshot(
    run_id=uuid4(),
    exchange="binance",
    symbol="BTC/USDT",
    timeframe="1h",
    timestamp=NOW,
    close=100.0,
    candles_digest="a" * 64,
    candles_count=300,
)
APPROVED = RiskVerdict(approved=True, final_size_fraction=0.05)


def decision(action: Action = Action.BUY, invalidation: float | None = 99.0) -> Decision:
    """Decisión accionable, o con la invalidación que se le pida."""
    if action is Action.HOLD:
        return Decision(
            action=action,
            confidence=0.5,
            size_fraction=0.0,
            invalidation_price=None,
            rationale="La evidencia no es concluyente en ninguna direccion.",
            dismissed_side=None,
            dismissal_reason=None,
        )
    return Decision(
        action=action,
        confidence=0.8,
        size_fraction=0.5,
        invalidation_price=invalidation,
        rationale="Estructura, impulso y volumen coinciden en direccion alcista.",
        dismissed_side=Side.BEAR,
        dismissal_reason="Su contraargumento depende de un nivel que ya se perdio.",
    )


# ─────────────────────────────────────── Construcción de orden ────────────────────────────────────


def test_order_takes_its_size_from_the_verdict() -> None:
    """La decisión pedía 0.5; sale 0.05 porque es lo que aprobó el riesgo."""
    order = build_order(decision(), APPROVED, SNAPSHOT, ExecutionMode.PAPER)
    assert order is not None
    assert order.size_fraction == 0.05
    assert order.side is Action.BUY
    assert order.symbol == "BTC/USDT"
    assert order.reference_price == 100.0
    assert order.mode is ExecutionMode.PAPER


def test_a_vetoed_verdict_produces_no_order() -> None:
    """Sin aprobación no hay orden que enviar."""
    vetoed = RiskVerdict(
        approved=False,
        final_size_fraction=0.0,
        veto_rule="kill_switch",
        veto_reason="kill switch activo",
    )
    assert build_order(decision(), vetoed, SNAPSHOT, ExecutionMode.PAPER) is None


def test_zero_size_produces_no_order() -> None:
    """Aprobado con tamaño cero tampoco genera orden."""
    verdict = RiskVerdict(approved=True, final_size_fraction=0.0)
    assert build_order(decision(), verdict, SNAPSHOT, ExecutionMode.PAPER) is None


def test_hold_produces_no_order() -> None:
    """`hold` no tiene lado de mercado."""
    assert build_order(decision(Action.HOLD), APPROVED, SNAPSHOT, ExecutionMode.PAPER) is None


def test_an_order_cannot_have_a_hold_side() -> None:
    """El propio modelo lo prohíbe, no solo la función que lo construye."""
    with pytest.raises(ValidationError, match="lado 'hold'"):
        OrderIntent(
            symbol="BTC/USDT",
            side=Action.HOLD,
            size_fraction=0.05,
            reference_price=100.0,
            invalidation_price=99.0,
            mode=ExecutionMode.PAPER,
        )


def test_sell_orders_are_built_too() -> None:
    """La dirección sale de la decisión; solo el tamaño viene del riesgo."""
    order = build_order(decision(Action.SELL, 101.0), APPROVED, SNAPSHOT, ExecutionMode.PAPER)
    assert order is not None
    assert order.side is Action.SELL


def test_an_approved_verdict_cannot_carry_a_wrong_sided_stop_into_an_order() -> None:
    """El respaldo del veto: aunque el veredicto diga aprobado, la orden no se construye.

    Aquí el veredicto se fabrica a mano, saltándose `apply_risk`, que es justo el
    camino que no debería existir. Un `sell` con la invalidación por debajo del
    cierre no llega a ser una orden: falla al construirse en vez de salir al
    mercado con un stop que ya estaba cruzado.
    """
    with pytest.raises(ValidationError, match="lado equivocado"):
        build_order(decision(Action.SELL, 99.0), APPROVED, SNAPSHOT, ExecutionMode.PAPER)


# ────────────────────────────────────────── Paper y live ──────────────────────────────────────────


def test_paper_is_the_default_mode() -> None:
    """Sin configurar nada, no se manda nada al mercado."""
    assert ExecutionSettings().mode is ExecutionMode.PAPER


@pytest.mark.asyncio
async def test_paper_executor_records_without_touching_the_market() -> None:
    """El ejecutor de papel anota y devuelve recibo."""
    executor = PaperExecutor()
    order = build_order(decision(), APPROVED, SNAPSHOT, ExecutionMode.PAPER)
    assert order is not None

    receipt = await executor.submit(order)

    assert receipt.accepted is True
    assert receipt.reference == "paper-1"
    assert executor.submitted == [order]


@pytest.mark.parametrize(
    ("execution", "exchange", "expected"),
    [
        (
            ExecutionSettings(),
            ExchangeSettings(api_key=SecretStr("k"), api_secret=SecretStr("s"), sandbox=False),
            "MODE debe ser 'live'",
        ),
        (
            ExecutionSettings(mode=ExecutionMode.LIVE),
            ExchangeSettings(sandbox=False),
            "faltan credenciales",
        ),
        (
            ExecutionSettings(mode=ExecutionMode.LIVE),
            ExchangeSettings(api_key=SecretStr("k"), api_secret=SecretStr("s")),
            "SANDBOX debe ser false",
        ),
    ],
)
def test_live_execution_needs_all_three_conditions(
    execution: ExecutionSettings, exchange: ExchangeSettings, expected: str
) -> None:
    """Cada condición por separado basta para impedir la ejecución real."""
    with pytest.raises(LiveExecutionRefusedError, match=expected):
        CcxtExecutor(execution, exchange)


def test_live_executor_builds_only_with_everything_in_place() -> None:
    """Las tres condiciones juntas son las que habilitan el ejecutor real."""
    executor = CcxtExecutor(
        ExecutionSettings(mode=ExecutionMode.LIVE),
        ExchangeSettings(api_key=SecretStr("k"), api_secret=SecretStr("s"), sandbox=False),
    )
    assert isinstance(executor, CcxtExecutor)


def test_default_settings_refuse_live_execution() -> None:
    """El valor por defecto de todo el sistema no puede ejecutar en real."""
    with pytest.raises(LiveExecutionRefusedError):
        CcxtExecutor(ExecutionSettings(), ExchangeSettings())
