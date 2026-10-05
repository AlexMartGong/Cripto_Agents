"""Alertas sobre lo que ya quedó registrado.

Funciones puras sobre registros del journal. No consultan el `QuotaLedger` vivo
aunque hablen de cuota: `LLMCall` lleva su marca de tiempo y su peso, así que el
consumo de la ventana se reconstruye del journal, y eso hace que las mismas
alertas se calculen igual en caliente que sobre un archivo de ayer. Una alerta que
solo existiera en memoria no se podría auditar después, que es cuando se pregunta
por qué nadie avisó.

Cada alerta dice qué la disparó y con qué números, no solo que algo va mal: un
aviso sin el dato que lo provocó obliga a repetir la consulta a mano.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import Field

from crypto_agents.metrics import AbortKind, backend_stats, undecided_causes
from crypto_agents.runner import RUNNER_NODE
from crypto_agents.settings import QUOTA_NOT_APPLICABLE
from crypto_agents.state import AgentRole, Billing, FrozenModel

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from crypto_agents.journal import EvaluationRecord
    from crypto_agents.settings import Settings

__all__ = [
    "Alert",
    "AlertKind",
    "AlertThresholds",
    "evaluate_alerts",
    "funds_alerts",
    "quota_alerts",
    "quota_status",
    "repeated_veto_alerts",
    "skipped_cycle_alerts",
    "validation_alerts",
]

KILL_SWITCH_RULE = "kill_switch"
"""Veto excluido de la alerta de repetición: si lo pusiste tú, repetirse es su trabajo."""


class AlertKind(StrEnum):
    """Qué clase de problema se detectó."""

    QUOTA_LOW = "quota_low"
    INSUFFICIENT_FUNDS = "insufficient_funds"
    REPEATED_VETO = "repeated_veto"
    VALIDATION_FAILURES = "validation_failures"
    SKIPPED_CYCLES = "skipped_cycles"


class Alert(FrozenModel):
    """Un aviso, con el dato que lo provocó."""

    kind: AlertKind
    subject: str = Field(min_length=1)
    """A quién se refiere: un rol, una regla de veto, un símbolo."""

    detail: str = Field(min_length=1)
    value: float
    threshold: float


class AlertThresholds(FrozenModel):
    """Umbrales de aviso. Se declaran para poder discutirlos."""

    quota_remaining_fraction: float = Field(default=0.2, gt=0.0, le=1.0)
    """Avisa cuando a un rol le queda menos de esta fracción de su ventana."""

    veto_window: int = Field(default=10, ge=1)
    veto_repeats: int = Field(default=3, ge=1)
    """`veto_repeats` vetos de la misma regla dentro de las últimas `veto_window`."""

    validation_failure_rate: float = Field(default=0.2, ge=0.0, le=1.0)
    min_attempts: int = Field(default=5, ge=1)
    """Intentos mínimos antes de opinar sobre una tasa: 1 de 1 no es el 100%."""

    skipped_cycles: int = Field(default=1, ge=1)

    insufficient_funds: int = Field(default=1, ge=1)
    """Evaluaciones perdidas por saldo insuficiente a partir de las cuales se avisa: una basta."""


def quota_alerts(
    records: Sequence[EvaluationRecord],
    settings: Settings,
    now: datetime,
    thresholds: AlertThresholds,
) -> list[Alert]:
    """Roles a los que les queda poca ventana.

    Reconstruye el consumo desde las llamadas registradas dentro de la ventana que
    termina en `now`, igual que haría el contador vivo.

    Con pago por uso no hay ventana que agotar: Zen no publica límite y la cuota declarada es un
    centinela (`ZEN_UNPUBLISHED_QUOTA`), así que no se calcula ningún porcentaje contra ella.
    `quota_status` dice por qué esta alerta calla.
    """
    if settings.billing is Billing.PAYG:
        return []
    cutoff = now - settings.quota_window
    used: dict[tuple[AgentRole, str], float] = {}
    for record in records:
        for call in record.calls:
            if call.cache_hit or call.at <= cutoff:
                continue
            key = (call.role, call.model)
            used[key] = used.get(key, 0.0) + call.quota_weight

    alerts: list[Alert] = []
    for role in AgentRole:
        for choice in settings.role_choices(role):
            spent = used.get((role, choice.model), 0.0)
            remaining = (choice.quota_per_window - spent) / choice.quota_per_window
            if remaining < thresholds.quota_remaining_fraction:
                alerts.append(
                    Alert(
                        kind=AlertKind.QUOTA_LOW,
                        subject=f"{role.value}/{choice.model}",
                        detail=(
                            f"queda {remaining:.0%} de la ventana "
                            f"({spent:.0f} de {choice.quota_per_window} consumidos)"
                        ),
                        value=remaining,
                        threshold=thresholds.quota_remaining_fraction,
                    )
                )
    return alerts


def quota_status(settings: Settings) -> str | None:
    """La línea que sustituye a la alerta de cuota cuando no se puede medir, o `None`."""
    if settings.billing is Billing.PAYG:
        return f"cuota: {QUOTA_NOT_APPLICABLE}"
    return None


def repeated_veto_alerts(
    records: Sequence[EvaluationRecord], thresholds: AlertThresholds
) -> list[Alert]:
    """Una regla de riesgo que veta una y otra vez dentro de la ventana reciente.

    El kill switch queda fuera: si está puesto, vetará todas las evaluaciones por
    diseño, y avisarlo sería repetir lo que ya se sabe.
    """
    window = records[-thresholds.veto_window :]
    counts: dict[str, int] = {}
    for record in window:
        rule = record.risk.veto_rule if record.risk is not None else None
        if rule is None or rule == KILL_SWITCH_RULE:
            continue
        counts[rule] = counts.get(rule, 0) + 1

    return [
        Alert(
            kind=AlertKind.REPEATED_VETO,
            subject=rule,
            detail=f"{count} vetos en las últimas {len(window)} evaluaciones",
            value=float(count),
            threshold=float(thresholds.veto_repeats),
        )
        for rule, count in sorted(counts.items())
        if count >= thresholds.veto_repeats
    ]


def validation_alerts(
    records: Sequence[EvaluationRecord], thresholds: AlertThresholds
) -> list[Alert]:
    """Pares (rol, backend) que no consiguen producir el esquema.

    Se exige un mínimo de intentos antes de opinar: un fallo sobre un intento es el
    100% y no significa nada.

    Cuenta sobre lo respondido, no sobre lo intentado. Un proveedor que rechaza
    todas las peticiones dispararía esta alerta al 100% sin que ningún modelo
    haya dicho una palabra, y mandaría a cambiar el esquema cuando lo que hay que
    cambiar es el id o la credencial.
    """
    stats = backend_stats(call for record in records for call in record.calls)
    return [
        Alert(
            kind=AlertKind.VALIDATION_FAILURES,
            subject=f"{role.value}/{backend.value}",
            detail=f"{item.invalid} de {item.answered} respuestas no validaron",
            value=item.failure_rate,
            threshold=thresholds.validation_failure_rate,
        )
        for (role, backend), item in stats.items()
        if item.answered >= thresholds.min_attempts
        and item.failure_rate > thresholds.validation_failure_rate
    ]


def funds_alerts(records: Sequence[EvaluationRecord], thresholds: AlertThresholds) -> list[Alert]:
    """Evaluaciones que el proveedor rechazó con `402 Insufficient account funds`, por nodo.

    Con pago por uso el saldo es el único tope, y el rechazo llega como un `transport` más: sin
    esta alerta la evaluación se pierde sin ruido y la corrida sigue gastando intentos contra un
    crédito agotado. No depende del tipo de facturación: un 402 es un 402.
    """
    lost: dict[str, int] = {}
    for (node, kind), count in undecided_causes(records).items():
        if kind is AbortKind.INSUFFICIENT_FUNDS:
            lost[node] = lost.get(node, 0) + count
    return [
        Alert(
            kind=AlertKind.INSUFFICIENT_FUNDS,
            subject=node,
            detail=f"{count} evaluación(es) perdidas por saldo insuficiente (402)",
            value=float(count),
            threshold=float(thresholds.insufficient_funds),
        )
        for node, count in sorted(lost.items())
        if count >= thresholds.insufficient_funds
    ]


def skipped_cycle_alerts(
    records: Sequence[EvaluationRecord], thresholds: AlertThresholds
) -> list[Alert]:
    """Cierres de vela que el runner se saltó por ir con retraso."""
    skipped: dict[str, int] = {}
    for record in records:
        if any(error.node == RUNNER_NODE for error in record.errors):
            skipped[record.symbol] = skipped.get(record.symbol, 0) + 1

    return [
        Alert(
            kind=AlertKind.SKIPPED_CYCLES,
            subject=symbol,
            detail=f"{count} ciclo(s) omitidos o fallidos",
            value=float(count),
            threshold=float(thresholds.skipped_cycles),
        )
        for symbol, count in sorted(skipped.items())
        if count >= thresholds.skipped_cycles
    ]


def evaluate_alerts(
    records: Sequence[EvaluationRecord],
    settings: Settings,
    now: datetime,
    thresholds: AlertThresholds | None = None,
) -> list[Alert]:
    """Todas las alertas sobre un conjunto de registros."""
    limits = thresholds if thresholds is not None else AlertThresholds()
    return [
        *quota_alerts(records, settings, now, limits),
        *repeated_veto_alerts(records, limits),
        *validation_alerts(records, limits),
        *funds_alerts(records, limits),
        *skipped_cycle_alerts(records, limits),
    ]
