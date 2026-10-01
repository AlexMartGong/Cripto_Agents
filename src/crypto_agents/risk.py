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

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Self

from pydantic import AwareDatetime, Field, PositiveFloat, model_validator

from crypto_agents.state import Action, FrozenModel, RiskVerdict, stop_on_wrong_side

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

    from crypto_agents.state import Proposal

type Veto = Callable[["Proposal", "AccountState", "RiskLimits", "datetime", float], str | None]
"""Regla de veto: mira decisión, cuenta, límites, instante y precio de referencia.

Devuelve la causa o nada. Todas reciben lo mismo aunque cada una lea una parte:
con firmas distintas, `VETOES` dejaría de ser la lista completa de reglas en su
orden y alguna tendría que comprobarse fuera del bucle, donde nadie la buscaría.
"""

__all__ = [
    "INVALID_STOP_SIDE",
    "NO_EXPOSURE_HEADROOM",
    "VETOES",
    "AccountState",
    "AnyKillSwitch",
    "FileKillSwitch",
    "KillSwitch",
    "RiskLimits",
    "StaticKillSwitch",
    "Veto",
    "apply_risk",
    "cap_position_size",
    "cap_total_exposure",
    "veto_cooldown",
    "veto_daily_drawdown",
    "veto_invalid_stop_side",
    "veto_kill_switch",
]


class KillSwitch(Protocol):
    """Interruptor de parada consultable desde fuera del proceso.

    Vive en el gate de riesgo y no en el runner a propósito. Si lo mirase el
    runner, una evaluación ya en vuelo colocaría su orden igual, y cualquier otra
    entrada al pipeline —un replay con ejecución, una corrida a mano— lo saltaría
    entero. En el gate, todos los caminos a una orden pasan por él.
    """

    def engaged(self) -> bool:
        """Si la operación está detenida ahora mismo."""
        ...


class StaticKillSwitch:
    """Interruptor fijo, el de la configuración. No cambia durante la corrida."""

    def __init__(self, engaged: bool = False) -> None:
        self._engaged = engaged

    def engaged(self) -> bool:
        """Valor declarado al construir."""
        return self._engaged


class FileKillSwitch:
    """Archivo centinela: si existe, no sale nada al mercado.

    Se consulta en cada evaluación, sin cachear. Un valor cacheado en un runner de
    velas de 4h no se enteraría del archivo hasta el siguiente arranque, que es
    exactamente cuando el interruptor no sirve de nada.

    Un error al leerlo cuenta como activado. Un kill switch que falla abierto no es
    un kill switch: ante un permiso roto o un disco lleno, la respuesta correcta es
    dejar de operar y que alguien mire.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        """Archivo vigilado."""
        return self._path

    def engaged(self) -> bool:
        """Si el centinela existe, o si no se pudo comprobar."""
        try:
            return self._path.exists()
        except OSError:
            return True


class AnyKillSwitch:
    """Combina varios interruptores: basta uno activo.

    Es lo que permite que convivan la parada por configuración y la de archivo sin
    que ninguna de las dos tenga prioridad sobre la otra.
    """

    def __init__(self, *switches: KillSwitch) -> None:
        self._switches = switches

    def engaged(self) -> bool:
        """Si alguno de los interruptores está activo."""
        return any(switch.engaged() for switch in self._switches)


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


INVALID_STOP_SIDE = "invalid_stop_side"
"""Regla del stop en el lado equivocado. Nombre estable: se agrupa por él."""


def veto_invalid_stop_side(
    decision: Proposal,
    account: AccountState,
    limits: RiskLimits,
    now: datetime,
    reference_price: float,
) -> str | None:
    """Un stop que no invalida nada no es un stop.

    Un `buy` con la invalidación por encima del cierre —o un `sell` con ella por
    debajo— declara rota la tesis en un precio que el mercado ya cruzó. La primera
    ablación emitió órdenes así sin que nada las detuviera, y la puntuación de
    resultados las contó: tocar el «stop» en la vela siguiente era salir con
    ganancia.

    Va la primera a propósito, antes incluso que el interruptor. Las demás reglas
    hablan del estado del sistema; esta, de la propuesta. Evaluada primero, una
    propuesta incoherente queda registrada como tal aunque en ese momento no se
    fuera a operar por otra razón, y el recuento por `veto_rule` cuenta todas.
    """
    del account, limits, now
    invalidation = decision.invalidation_price
    if invalidation is None:
        return None
    if stop_on_wrong_side(decision.action, invalidation, reference_price):
        return (
            f"invalidación {invalidation} en el lado equivocado de un "
            f"{decision.action.value} con cierre {reference_price}"
        )
    return None


def veto_kill_switch(
    decision: Proposal,
    account: AccountState,
    limits: RiskLimits,
    now: datetime,
    reference_price: float,
) -> str | None:
    """Interruptor manual: corta sea cual sea el estado de la cuenta."""
    del decision, account, now, reference_price
    if limits.kill_switch:
        return "kill switch activo"
    return None


def veto_daily_drawdown(
    decision: Proposal,
    account: AccountState,
    limits: RiskLimits,
    now: datetime,
    reference_price: float,
) -> str | None:
    """Perdido el drawdown del día, se deja de operar hasta el día siguiente."""
    del decision, now, reference_price
    drawdown = account.daily_drawdown_fraction
    if drawdown >= limits.max_daily_drawdown_fraction:
        return (
            f"drawdown diario {drawdown:.2%} alcanza el límite "
            f"{limits.max_daily_drawdown_fraction:.2%}"
        )
    return None


def veto_cooldown(
    decision: Proposal,
    account: AccountState,
    limits: RiskLimits,
    now: datetime,
    reference_price: float,
) -> str | None:
    """Tras una pérdida se espera. Operar en caliente es cómo se encadenan las pérdidas."""
    del decision, reference_price
    if account.last_loss_at is None or limits.cooldown_after_loss_minutes == 0:
        return None
    elapsed_minutes = (now - account.last_loss_at).total_seconds() / 60.0
    if elapsed_minutes < limits.cooldown_after_loss_minutes:
        remaining = limits.cooldown_after_loss_minutes - elapsed_minutes
        return f"cooldown tras pérdida: quedan {remaining:.0f} minutos"
    return None


VETOES: tuple[tuple[str, Veto], ...] = (
    (INVALID_STOP_SIDE, veto_invalid_stop_side),
    ("kill_switch", veto_kill_switch),
    ("daily_drawdown", veto_daily_drawdown),
    ("cooldown", veto_cooldown),
)
"""Reglas de veto con su nombre estable, en orden de aplicación.

El nombre se declara en vez de sacarse de `__name__`: así renombrar la función no
cambia en silencio la clave con la que se agrupan los vetos en las métricas de una
corrida, que es lo que haría incomparables dos backtests.
"""

NO_EXPOSURE_HEADROOM = "no_exposure_headroom"
"""Recorte que llegó a cero. No es un veto declarado, pero acaba en lo mismo."""


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
    decision: Proposal,
    account: AccountState,
    limits: RiskLimits,
    now: datetime,
    reference_price: float,
) -> RiskVerdict:
    """Veredicto de riesgo para una decisión.

    Un `hold` pasa con tamaño cero: no es un veto, simplemente no genera orden.

    `reference_price` es el cierre de la vela evaluada, el mismo precio que
    `build_order` pondrá en la orden. No tiene valor por defecto: con uno, un
    llamador que lo olvidara dejaría de comprobar el lado del stop sin que nada
    fallara, que es exactamente como estaba antes de existir la regla.
    """
    if decision.action is Action.HOLD:
        return RiskVerdict(approved=True, final_size_fraction=0.0)

    for rule, veto in VETOES:
        reason = veto(decision, account, limits, now, reference_price)
        if reason is not None:
            return RiskVerdict(
                approved=False, final_size_fraction=0.0, veto_rule=rule, veto_reason=reason
            )

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
            veto_rule=NO_EXPOSURE_HEADROOM,
            veto_reason="no queda hueco de exposición",
        )
    return RiskVerdict(approved=True, final_size_fraction=size, applied_limits=tuple(applied))
