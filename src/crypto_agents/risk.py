"""Gate de riesgo: determinista, sin modelos, sin excepciones.

Es el único punto donde se decide cuánto se arriesga. Un modelo puede pedir lo
que quiera; lo que sale de aquí es un `RiskVerdict`, y el tamaño de cualquier
orden se lee de ese veredicto, nunca de la `Decision`.

Las reglas se aplican en orden: primero los vetos, que apagan la operación
entera, y después los recortes, que la dejan pasar más pequeña. Un recorte que
llega a cero se convierte en veto: una orden de tamaño cero no es una orden.

El `now` se inyecta, igual que el reloj del contador de cuota: un cooldown que
dependiera del reloj real solo se podría probar esperando.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from pydantic import AwareDatetime, Field, PositiveFloat, model_validator

from crypto_agents.state import Action, FrozenModel, RiskVerdict

if TYPE_CHECKING:
    from datetime import datetime

    from crypto_agents.state import Decision

__all__ = [
    "AccountState",
    "RiskLimits",
    "apply_risk",
    "cap_position_size",
    "cap_total_exposure",
    "veto_cooldown",
    "veto_daily_drawdown",
    "veto_kill_switch",
]


class RiskLimits(FrozenModel):
    """Límites duros del sistema. Ningún nodo puede modificarlos en caliente."""

    max_position_fraction: float = Field(default=0.10, gt=0.0, le=1.0)
    max_total_exposure_fraction: float = Field(default=0.30, gt=0.0, le=1.0)
    max_daily_drawdown_fraction: float = Field(default=0.05, gt=0.0, le=1.0)
    cooldown_after_loss_minutes: int = Field(default=240, ge=0)
    kill_switch: bool = False
    """Interruptor manual. Con esto activo no sale nada, pase lo que pase."""

    @model_validator(mode="after")
    def _position_fits_inside_total_exposure(self) -> Self:
        """Un tamaño por operación mayor que la exposición total es un límite incoherente."""
        if self.max_position_fraction > self.max_total_exposure_fraction:
            raise ValueError("max_position_fraction no puede superar a max_total_exposure_fraction")
        return self


class AccountState(FrozenModel):
    """Fotografía de la cuenta en el momento de evaluar.

    Se inyecta en el contexto, como el reloj. De dónde salgan estos números es
    responsabilidad de quien construye el contexto: el gate se mantiene puro.
    """

    equity: PositiveFloat
    day_start_equity: PositiveFloat
    open_exposure_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    last_loss_at: AwareDatetime | None = None

    @property
    def daily_drawdown_fraction(self) -> float:
        """Caída desde el inicio del día, en fracción. Cero o negativa si va en ganancia."""
        return max(0.0, (self.day_start_equity - self.equity) / self.day_start_equity)


# ────────────────────────────────────────────── Vetos ─────────────────────────────────────────────


def veto_kill_switch(
    decision: Decision, account: AccountState, limits: RiskLimits, now: datetime
) -> str | None:
    """Interruptor manual: corta antes que cualquier otra consideración."""
    del decision, account, now
    if limits.kill_switch:
        return "kill switch activo"
    return None


def veto_daily_drawdown(
    decision: Decision, account: AccountState, limits: RiskLimits, now: datetime
) -> str | None:
    """Perdido el drawdown del día, se deja de operar hasta el día siguiente."""
    del decision, now
    drawdown = account.daily_drawdown_fraction
    if drawdown >= limits.max_daily_drawdown_fraction:
        return (
            f"drawdown diario {drawdown:.2%} alcanza el límite "
            f"{limits.max_daily_drawdown_fraction:.2%}"
        )
    return None


def veto_cooldown(
    decision: Decision, account: AccountState, limits: RiskLimits, now: datetime
) -> str | None:
    """Tras una pérdida se espera. Operar en caliente es cómo se encadenan las pérdidas."""
    del decision
    if account.last_loss_at is None or limits.cooldown_after_loss_minutes == 0:
        return None
    elapsed_minutes = (now - account.last_loss_at).total_seconds() / 60.0
    if elapsed_minutes < limits.cooldown_after_loss_minutes:
        remaining = limits.cooldown_after_loss_minutes - elapsed_minutes
        return f"cooldown tras pérdida: quedan {remaining:.0f} minutos"
    return None


VETOES = (veto_kill_switch, veto_daily_drawdown, veto_cooldown)


# ───────────────────────────────────────────── Recortes ───────────────────────────────────────────


def cap_position_size(size: float, limits: RiskLimits) -> tuple[float, str | None]:
    """Recorta al tamaño máximo por operación."""
    if size > limits.max_position_fraction:
        return limits.max_position_fraction, "max_position_fraction"
    return size, None


def cap_total_exposure(
    size: float, account: AccountState, limits: RiskLimits
) -> tuple[float, str | None]:
    """Recorta a lo que quede libre de exposición total."""
    headroom = max(0.0, limits.max_total_exposure_fraction - account.open_exposure_fraction)
    if size > headroom:
        return headroom, "max_total_exposure_fraction"
    return size, None


# ────────────────────────────────────────── Gate completo ─────────────────────────────────────────


def apply_risk(
    decision: Decision, account: AccountState, limits: RiskLimits, now: datetime
) -> RiskVerdict:
    """Veredicto de riesgo para una decisión.

    Un `hold` pasa con tamaño cero: no es un veto, simplemente no genera orden.
    """
    if decision.action is Action.HOLD:
        return RiskVerdict(approved=True, final_size_fraction=0.0)

    for veto in VETOES:
        reason = veto(decision, account, limits, now)
        if reason is not None:
            return RiskVerdict(approved=False, final_size_fraction=0.0, veto_reason=reason)

    size = decision.size_fraction
    applied: list[str] = []

    size, label = cap_position_size(size, limits)
    if label is not None:
        applied.append(label)

    size, label = cap_total_exposure(size, account, limits)
    if label is not None:
        applied.append(label)

    if size <= 0.0:
        return RiskVerdict(
            approved=False,
            final_size_fraction=0.0,
            applied_limits=tuple(applied),
            veto_reason="no queda hueco de exposición",
        )
    return RiskVerdict(approved=True, final_size_fraction=size, applied_limits=tuple(applied))
