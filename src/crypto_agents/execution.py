"""Salida del sistema: intención de orden y ejecución.

La orden se construye a partir del `RiskVerdict`, no de la `Decision`. La
decisión aporta dirección y precio de invalidación; el tamaño sale del veredicto
de riesgo. No hay ninguna función en este módulo que acepte un tamaño propuesto
por un modelo, así que no existe camino por el que un modelo convincente mande
una orden mayor de la permitida.

Paper trading por defecto. La ejecución real exige tres condiciones
independientes, verificadas al arrancar: modo `live`, credenciales de exchange
presentes y sandbox desactivado. Ninguna sola basta.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from crypto_agents.state import Action, ExecutionMode, FrozenModel, OrderIntent, OrderReceipt

if TYPE_CHECKING:
    from crypto_agents.settings import ExchangeSettings
    from crypto_agents.state import MarketSnapshot, Proposal, RiskVerdict

__all__ = [
    "CcxtExecutor",
    "ExecutionMode",
    "ExecutionSettings",
    "Executor",
    "LiveExecutionRefusedError",
    "OrderIntent",
    "OrderReceipt",
    "PaperExecutor",
    "build_order",
]


class ExecutionSettings(FrozenModel):
    """Configuración de salida. El valor por defecto no manda nada al mercado."""

    mode: ExecutionMode = ExecutionMode.PAPER


class LiveExecutionRefusedError(RuntimeError):
    """Se pidió ejecución real sin cumplir las tres condiciones."""


class Executor(Protocol):
    """Destino de una orden ya aprobada."""

    async def submit(self, order: OrderIntent) -> OrderReceipt:
        """Envía la orden y devuelve el recibo."""
        ...


def build_order(
    decision: Proposal,
    verdict: RiskVerdict,
    snapshot: MarketSnapshot,
    mode: ExecutionMode,
) -> OrderIntent | None:
    """Traduce una decisión aprobada en una orden, o `None` si no hay nada que enviar.

    El tamaño se lee de `verdict.final_size_fraction`. `decision.size_fraction`
    no se consulta: es lo que hace estructuralmente imposible que un modelo
    convincente supere un límite.
    """
    if not verdict.approved or verdict.final_size_fraction <= 0.0:
        return None
    if decision.action is Action.HOLD:
        return None
    if decision.invalidation_price is None:
        return None

    return OrderIntent(
        symbol=snapshot.symbol,
        side=decision.action,
        size_fraction=verdict.final_size_fraction,
        reference_price=snapshot.close,
        invalidation_price=decision.invalidation_price,
        mode=mode,
    )


class PaperExecutor:
    """Ejecutor por defecto: anota la orden y no toca el mercado."""

    def __init__(self) -> None:
        self.submitted: list[OrderIntent] = []

    async def submit(self, order: OrderIntent) -> OrderReceipt:
        """Registra la orden en memoria."""
        self.submitted.append(order)
        return OrderReceipt(order=order, accepted=True, reference=f"paper-{len(self.submitted)}")


class CcxtExecutor:
    """Ejecutor real. Se niega a construirse si no se cumplen las tres condiciones."""

    def __init__(self, execution: ExecutionSettings, exchange: ExchangeSettings) -> None:
        missing = _live_requirements_missing(execution, exchange)
        if missing:
            raise LiveExecutionRefusedError(f"ejecución real no habilitada: {'; '.join(missing)}")
        self._exchange_settings = exchange

    async def submit(self, order: OrderIntent) -> OrderReceipt:
        """Envía la orden al exchange."""
        raise NotImplementedError("el envío real de órdenes se implementa al conectar una cuenta")


def _live_requirements_missing(
    execution: ExecutionSettings, exchange: ExchangeSettings
) -> list[str]:
    """Condiciones que faltan para poder ejecutar en real."""
    missing: list[str] = []
    if execution.mode is not ExecutionMode.LIVE:
        missing.append("CA_EXECUTION__MODE debe ser 'live'")
    if exchange.api_key is None or exchange.api_secret is None:
        missing.append("faltan credenciales de exchange")
    if exchange.sandbox:
        missing.append("CA_EXCHANGE__SANDBOX debe ser false")
    return missing
