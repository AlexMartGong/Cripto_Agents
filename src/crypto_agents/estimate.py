"""Estimación en dólares de la etapa 1 de la ablación con pago por uso. Es una estimación.

    python -m crypto_agents.estimate <directorio del sondeo>

Parte de dos cosas que ya existen y no inventa una tercera:

- **Cuántas llamadas**: el `--dry-run` de la ablación sobre `data/ablation_selection.json`, con la
  caché vacía (sin aciertos), por brazo y por nodo. Es el conteo que decide si la corrida cabe.
- **Cuánto cuesta una**: la media del coste *medido* de las filas `LLMCall` que dejó el sondeo
  (`zen_probe`), por (rol, modelo). Los tokens no se estiman en ningún punto: una llamada sin
  tokens no tiene coste y no entra en la media, y la cifra lo dice.

Reglas, cada una con su prueba:

- **Las llamadas locales y las de las líneas base cuestan 0, por regla.** No hacen petición.
- **Solo el decisor reintenta**, y se presupuesta con `DECIDER_ATTEMPTS`. Se dan las dos cifras: un
  intento por llamada en todo, y el decisor a 1.2.
- **structure y volume dan un rango**, del candidato más barato al más caro de los que
  respondieron (≥ 1 veredicto válido). La elección es de quien lee la tabla, no de este módulo.
- **El decisor se mide con el prompt de `full`** (evidencia y las dos mesas). `solo`, `no_debate` y
  `bull_only` le pasan prompts más cortos, así que su coste aquí es una cota superior.
- **La recarga es una sola** y se lee «4.4% + 0.30 USD» como `crédito x 1.044 + 0.30`
  (`PriceTable.topup_charge`). Es una lectura de la página, no un recibo.
- **Una llamada puede costar un intervalo, no una cifra.** Con `cached_tokens` en `null` va de
  «todo el prompt cacheado» a «todo nuevo»; con escritura de caché (`qwen3.8-max`), de no cobrarla
  a cobrar la entrada nueva a ese precio (`consumption.call_cost_range_usd`). Ningún token se
  estima.
- **`--desks <directorio>`** suma el sondeo de mesas y da una línea por candidato a `bull`, cada una
  con su decisor *condicionado* a ese bull (el prompt del decisor lleva el alegato).
- **`--balance X`** es el saldo que se lee en la consola, sin red. Pasa si `X >= coste x
  LAUNCH_MARGIN`, con el coste del decisor a `DECIDER_ATTEMPTS` y juzgado en el extremo alto; un
  coste no determinado no pasa.

Cada cifra cita el directorio y el sha-256 del archivo de donde sale. `tests/data/usage` entra como
contraste (12 `Ping` con ~1.4 k tokens de relleno, no el prompt real).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from pydantic import Field

from crypto_agents.ablation import (
    ARMS,
    DECIDER_ATTEMPTS,
    LAUNCH_MARGIN,
    DryRunReport,
    DryRunRow,
    dry_run,
    plan_from_manifest,
)
from crypto_agents.audit import AuditError, RunDirectory, file_sha256, read_run
from crypto_agents.cache import InMemoryResponseCache
from crypto_agents.consumption import call_cost_usd, consume
from crypto_agents.context import utc_now
from crypto_agents.selection import (
    HISTORY_DIR,
    SelectionError,
    load_manifest,
    load_selection_histories,
)
from crypto_agents.settings import (
    DEFAULT_ENV_FILE,
    DEFAULT_PRICING,
    ConfigError,
    load_settings,
)
from crypto_agents.state import (
    AgentRole,
    Backend,
    Billing,
    FrozenModel,
    LLMCall,
    StructuredOutputMode,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

__all__ = [
    "USAGE_FIXTURE",
    "BalanceCheck",
    "CallCost",
    "Estimate",
    "RoleLine",
    "ScenarioLine",
    "balance_check",
    "decider_given_bull",
    "estimate",
    "main",
    "measured_costs",
    "render_estimate",
    "render_scenarios",
    "scenarios",
]

LABEL = "ESTIMACIÓN, no medida"

DEFAULT_MANIFEST = Path("data/ablation_selection.json")
USAGE_FIXTURE = Path("tests/data/usage/gateway_usage.json")
USAGE_MEASURED_AT = datetime(2026, 10, 3, 4, 55, tzinfo=UTC)
"""Cuándo se hicieron las 12 llamadas del fixture (su README: 2026-10-03, hacia las 04:55Z)."""

RANGED_ROLES = (AgentRole.STRUCTURE, AgentRole.VOLUME)
"""Los roles cuyo modelo aún no se ha elegido: se dan como rango entre candidatos."""

_ROLE_NAMES = {role.value for role in AgentRole}


class CallCost(FrozenModel):
    """Lo que cuesta de media un intento a un (rol, modelo), de las filas que lo respaldan."""

    role: AgentRole
    model: str
    attempts: int
    """Intentos vivos y respondidos que entran en la media, medidos o no."""

    measured: int
    unmeasured: int
    mean_usd: float | None
    """Coste medio de un intento medido, o `None` si ninguno lo está."""

    valid_verdicts: int
    """Registros del brazo sin error: veredictos que el modelo sí produjo."""

    source: str
    """De dónde salen las filas: el brazo del sondeo, o el fixture de `Ping`."""

    mean_low_usd: float | None = None
    mean_high_usd: float | None = None
    """Intervalo por intento sobre las llamadas que lo tienen: las medidas y las de `cached_tokens`
    en `null`. Coincide con `mean_usd` salvo por escritura de caché o por ese `null`. `None` en las
    construidas a mano sin intervalo: entonces vale `mean_usd`."""

    @property
    def bounds(self) -> tuple[float, float] | None:
        """El intervalo por intento, o `None` si ninguna llamada tiene coste."""
        if self.mean_low_usd is not None and self.mean_high_usd is not None:
            return self.mean_low_usd, self.mean_high_usd
        return None if self.mean_usd is None else (self.mean_usd, self.mean_usd)


def _answered(call: LLMCall) -> bool:
    return not call.cache_hit and call.backend is not Backend.OLLAMA


def measured_costs(run: RunDirectory) -> dict[tuple[AgentRole, str], CallCost]:
    """Coste medio por intento de cada (rol, modelo) que el sondeo midió con su prompt real.

    Los brazos `@ping` y `@header-*` quedan fuera: miden un `Ping` de sesenta caracteres, no un
    prompt que vaya a correr. Entran los intentos inválidos —también se facturaron—.
    """
    grouped: dict[tuple[AgentRole, str], list[LLMCall]] = {}
    valid: dict[tuple[AgentRole, str], int] = {}
    sources: dict[tuple[AgentRole, str], str] = {}
    for arm in run.arms:
        if arm.arm.rpartition("@")[2] not in _ROLE_NAMES:
            continue
        for record in arm.records:
            for call in record.calls:
                key = (call.role, call.model)
                if _answered(call):
                    grouped.setdefault(key, []).append(call)
                    sources[key] = f"{run.path.name}/{arm.path.name}"
            if record.calls and not record.errors:
                key = (record.calls[0].role, record.calls[0].model)
                valid[key] = valid.get(key, 0) + 1
    return {
        key: _call_cost(key[0], key[1], calls, valid.get(key, 0), sources[key])
        for key, calls in grouped.items()
    }


def _call_cost(
    role: AgentRole, model: str, calls: Sequence[LLMCall], valid: int, source: str
) -> CallCost:
    """El coste por intento de un grupo de llamadas: la media medida y el intervalo."""
    total = consume(calls, DEFAULT_PRICING, Billing.PAYG)
    ranged = total.measured + total.cached_unreported
    return CallCost(
        role=role,
        model=model,
        attempts=len(calls),
        measured=total.measured,
        unmeasured=total.unmeasured + total.no_response,
        mean_usd=total.cost_usd / total.measured if total.measured else None,
        valid_verdicts=valid,
        source=source,
        mean_low_usd=(total.cost_usd + total.unreported_floor_usd) / ranged if ranged else None,
        mean_high_usd=(total.cost_upper_usd + total.unreported_ceiling_usd) / ranged
        if ranged
        else None,
    )


def decider_given_bull(run: RunDirectory) -> dict[str, CallCost]:
    """Coste por intento del decisor condicionado a cada bull (brazos `<modelo>@decider+<bull>`).

    El prompt del decisor lleva el alegato del bull, así que lo que cuesta depende de qué bull lo
    escribió. Un brazo por bull, medido con el mismo bear de cada activación.
    """
    result: dict[str, CallCost] = {}
    for arm in run.arms:
        label = arm.arm.rpartition("@")[2]
        role, plus, bull = label.partition("+")
        if not plus or role != AgentRole.DECIDER.value:
            continue
        calls = [c for record in arm.records for c in record.calls if _answered(c)]
        if not calls:
            continue
        valid = sum(1 for record in arm.records if record.calls and not record.errors)
        result[bull] = _call_cost(
            AgentRole.DECIDER, calls[0].model, calls, valid, f"{run.path.name}/{arm.path.name}"
        )
    return result


def usage_fixture_costs(path: Path, roles: Mapping[str, AgentRole]) -> dict[str, CallCost]:
    """Coste medio de un `Ping` por modelo, del `usage` crudo que devolvió la pasarela.

    Es un contraste y un último recurso: el prompt era un esquema `Ping` más ~1.4 k tokens de
    relleno, no un prompt de la corrida. Se convierte cada fila en un `LLMCall` para que cueste lo
    mismo que cuesta cualquier otra, con la misma tabla de precios.
    """
    rows = json.loads(path.read_text(encoding="utf-8"))
    calls: dict[str, list[LLMCall]] = {}
    for row in rows:
        usage = row["token_usage"]
        details = usage.get("prompt_tokens_details") or {}
        model = row["model"]
        calls.setdefault(model, []).append(
            LLMCall(
                role=roles.get(model, AgentRole.STRUCTURE),
                backend=Backend.OPENAI,
                model=model,
                structured_output=StructuredOutputMode(row["mode"]),
                quota_weight=1.0,
                prompt_digest="0" * 64,
                valid=True,
                latency_ms=0.0,
                at=USAGE_MEASURED_AT,
                prompt_tokens=usage["prompt_tokens"],
                cached_tokens=details.get("cached_tokens"),
                completion_tokens=usage["completion_tokens"],
            )
        )
    result: dict[str, CallCost] = {}
    for model, items in calls.items():
        priced = [
            cost
            for call in items
            if (cost := call_cost_usd(call, DEFAULT_PRICING, Billing.PAYG)) is not None
        ]
        result[model] = CallCost(
            role=items[0].role,
            model=model,
            attempts=len(items),
            measured=len(priced),
            unmeasured=len(items) - len(priced),
            mean_usd=sum(priced) / len(priced) if priced else None,
            valid_verdicts=0,
            source=f"{path.name} (Ping, no el prompt real)",
        )
    return result


# ───────────────────────────────────────── Coste por rol y brazo ──────────────────────────────────


@dataclass(frozen=True)
class Span:
    """Un coste que puede ser un rango. `low == high` cuando el modelo está fijado.

    `None` es «no determinado», y contagia: un total al que le falta un sumando no es la suma de
    los demás, es una cifra menor de lo que costará.
    """

    low: float | None
    high: float | None

    def plus(self, other: Span) -> Span:
        if self.low is None or self.high is None or other.low is None or other.high is None:
            return Span(None, None)
        return Span(self.low + other.low, self.high + other.high)

    def scaled(self, factor: float) -> Span:
        return Span(
            None if self.low is None else self.low * factor,
            None if self.high is None else self.high * factor,
        )


NOTHING = Span(0.0, 0.0)


class RoleLine(FrozenModel):
    """Un rol sobre todos los brazos: cuántas llamadas pagará y cuánto cuesta cada una."""

    role: AgentRole
    calls: int
    """Llamadas que llegarán a un proveedor remoto, sin los aciertos de caché."""

    model: str | None
    """El modelo fijado, o `None` en los roles que van como rango."""

    per_call_low: float | None
    per_call_high: float | None
    cost_low: float | None
    cost_high: float | None
    source: str


class Estimate(FrozenModel):
    """La estimación completa, con sus entradas y las suposiciones que la sostienen."""

    activations: int
    roles: tuple[RoleLine, ...]
    arms: dict[str, tuple[float | None, float | None]]
    """Coste por brazo, con un intento por llamada."""

    arms_with_retries: dict[str, tuple[float | None, float | None]]
    """Lo mismo con el decisor a `DECIDER_ATTEMPTS`."""

    total: tuple[float | None, float | None]
    total_with_retries: tuple[float | None, float | None]
    candidates: tuple[CallCost, ...]
    inputs: tuple[tuple[str, str], ...]
    contrast: tuple[CallCost, ...] = ()


def _per_call(
    role: AgentRole,
    costs: Mapping[tuple[AgentRole, str], CallCost],
    fixed: Mapping[AgentRole, str],
) -> tuple[float | None, float | None, str | None, str]:
    """Coste por llamada de un rol: (bajo, alto, modelo fijado, procedencia)."""
    if role in RANGED_ROLES:
        answered = [
            b
            for (r, _), c in costs.items()
            if r is role and c.valid_verdicts > 0 and (b := c.bounds) is not None and b[1] > 0
        ]
        if not answered:
            return None, None, None, "ningún candidato respondió"
        low = min(b[0] for b in answered)
        high = max(b[1] for b in answered)
        return low, high, None, "sondeo: del más barato al más caro que respondió"
    model = fixed.get(role)
    found = costs.get((role, model)) if model is not None else None
    bounds = None if found is None else found.bounds
    if found is None or bounds is None:
        return None, None, model, "el sondeo no midió este rol"
    return bounds[0], bounds[1], model, found.source


def estimate(
    report: DryRunReport,
    costs: Mapping[tuple[AgentRole, str], CallCost],
    fixed: Mapping[AgentRole, str],
    inputs: Sequence[tuple[str, str]] = (),
    contrast: Sequence[CallCost] = (),
    arms: Sequence[str] = (),
) -> Estimate:
    """Las llamadas del conteo previo por el coste medido de cada una. Ningún token se estima.

    `report` viene de `dry_run` con la caché vacía. Una fila local (Ollama) cuesta 0 por regla, y
    las líneas base no tienen filas: no llaman a nadie.
    """
    per_role: dict[AgentRole, tuple[float | None, float | None, str | None, str]] = {
        role: _per_call(role, costs, fixed) for role in AgentRole
    }
    paid: dict[AgentRole, int] = {}
    # Todos los brazos pedidos aparecen, también los que no llaman a nadie: una línea base con
    # cero filas es un brazo que cuesta 0, no un brazo ausente de la tabla.
    per_arm: dict[str, Span] = {name: NOTHING for name in arms}
    per_arm_retry: dict[str, Span] = {name: NOTHING for name in arms}
    for row in report.rows:
        per_arm.setdefault(row.arm, NOTHING)
        per_arm_retry.setdefault(row.arm, NOTHING)
        if row.backend is Backend.OLLAMA:
            continue
        calls = _paid_calls(row)
        paid[row.role] = paid.get(row.role, 0) + calls
        low, high, _, _ = per_role[row.role]
        span = Span(None if low is None else low * calls, None if high is None else high * calls)
        per_arm[row.arm] = per_arm[row.arm].plus(span)
        factor = DECIDER_ATTEMPTS if row.role is AgentRole.DECIDER else 1.0
        per_arm_retry[row.arm] = per_arm_retry[row.arm].plus(span.scaled(factor))

    lines: list[RoleLine] = []
    for role in AgentRole:
        low, high, model, source = per_role[role]
        calls = paid.get(role, 0)
        lines.append(
            RoleLine(
                role=role,
                calls=calls,
                model=model,
                per_call_low=low,
                per_call_high=high,
                cost_low=None if low is None else low * calls,
                cost_high=None if high is None else high * calls,
                source=source,
            )
        )
    total = sum_spans(per_arm.values())
    total_retry = sum_spans(per_arm_retry.values())
    answered = tuple(
        sorted(
            (c for c in costs.values() if c.role in RANGED_ROLES and c.valid_verdicts > 0),
            key=lambda c: (c.role.value, c.mean_usd if c.mean_usd is not None else float("inf")),
        )
    )
    return Estimate(
        activations=report.activations,
        roles=tuple(lines),
        arms={name: (s.low, s.high) for name, s in per_arm.items()},
        arms_with_retries={name: (s.low, s.high) for name, s in per_arm_retry.items()},
        total=(total.low, total.high),
        total_with_retries=(total_retry.low, total_retry.high),
        candidates=answered,
        inputs=tuple(inputs),
        contrast=tuple(contrast),
    )


def _paid_calls(row: DryRunRow) -> int:
    """Llamadas del nodo al proveedor: las exactas a pagar, o la cota si sigue a un veredicto."""
    return (row.to_pay or 0) if row.exact else row.calls


def sum_spans(spans: Iterable[Span]) -> Span:
    """Suma de rangos; con un término sin determinar, el total tampoco lo está."""
    total = NOTHING
    for span in spans:
        total = total.plus(span)
    return total


# ────────────────────────────────────── Saldo frente a estimación ─────────────────────────────────


Pair = tuple[float | None, float | None]


class BalanceCheck(FrozenModel):
    """Si el saldo que se lee en la consola cubre el coste con el margen de lanzamiento."""

    balance: float
    cost: tuple[float | None, float | None]
    """Coste con el decisor a `DECIDER_ATTEMPTS`, de más barato a más caro."""

    required: tuple[float | None, float | None]
    """`cost x LAUNCH_MARGIN`. Vacío si el coste no está determinado."""

    passes: bool
    reason: str


def balance_check(cost: Pair, balance: float) -> BalanceCheck:
    """`PASA` si `balance >= cost x LAUNCH_MARGIN` en el extremo **alto**; si no, `NO PASA`.

    El extremo alto porque el intervalo es entre candidatos (o entre cobrar y no cobrar la
    escritura de caché) y el saldo tiene que alcanzar para el que se elija. Un coste no
    determinado no pasa: un saldo no cubre lo que no se sabe cuánto cuesta.
    """
    low, high = cost
    if low is None or high is None:
        return BalanceCheck(
            balance=balance,
            cost=cost,
            required=(None, None),
            passes=False,
            reason="coste no determinado",
        )
    required = (low * LAUNCH_MARGIN, high * LAUNCH_MARGIN)
    passes = balance >= required[1]
    comparison = "cubre" if passes else "no cubre"
    return BalanceCheck(
        balance=balance,
        cost=cost,
        required=required,
        passes=passes,
        reason=f"el saldo {comparison} el extremo alto del requerido",
    )


def _verdict(check: BalanceCheck) -> str:
    return "PASA" if check.passes else f"NO PASA ({check.reason})"


# ───────────────────────────────── Una línea por candidato a bull ─────────────────────────────────


DESK_ROLES = (AgentRole.BULL, AgentRole.BEAR, AgentRole.DECIDER)
"""Los roles que dependen del sondeo de mesas; los técnicos vienen del sondeo de T."""


class ScenarioLine(FrozenModel):
    """La etapa 1 con un bull concreto, y con el decisor medido sobre el alegato de ese bull."""

    bull: str
    valid_briefs: int
    note: str | None = None
    """Por qué no hay coste, si no lo hay: el candidato no produjo ningún alegato válido."""

    per_role: dict[AgentRole, tuple[float | None, float | None]] = Field(default_factory=dict)
    """Coste total de cada rol sobre todos los brazos; el decisor ya a `DECIDER_ATTEMPTS`."""

    total: tuple[float | None, float | None] = (None, None)
    topup: tuple[float | None, float | None] = (None, None)
    balance: BalanceCheck | None = None


def scenarios(
    report: DryRunReport,
    technical: Mapping[tuple[AgentRole, str], CallCost],
    desks: Mapping[tuple[AgentRole, str], CallCost],
    given: Mapping[str, CallCost],
    fixed: Mapping[AgentRole, str],
    bulls: Sequence[str],
    arms: Sequence[str] = (),
    balance: float | None = None,
) -> tuple[ScenarioLine, ...]:
    """Una estimación por candidato a bull, cada una con su decisor condicionado.

    `technical` viene del sondeo técnico (structure, volume, momentum); `desks` del de las mesas
    (bull, bear); `given` es el decisor medido sobre el alegato de cada bull. Un bull sin alegato
    válido no tiene coste propio ni decisor que medir: su línea lo dice y no se rellena con el de
    otro. El resto de roles usa **exactamente** las mismas cifras en todas las líneas, así que dos
    líneas solo difieren en lo que depende del bull.
    """
    lines: list[ScenarioLine] = []
    decider_model = fixed[AgentRole.DECIDER]
    shared = {key: c for key, c in technical.items() if key[0] not in DESK_ROLES}
    shared.update({key: c for key, c in desks.items() if key[0] is AgentRole.BEAR})
    for bull in bulls:
        bull_cost = desks.get((AgentRole.BULL, bull))
        if bull_cost is None or bull_cost.valid_verdicts == 0:
            lines.append(
                ScenarioLine(
                    bull=bull,
                    valid_briefs=0,
                    note="ningún alegato válido: sin coste de bull ni de decisor condicionado",
                )
            )
            continue
        costs = {**shared, (AgentRole.BULL, bull): bull_cost}
        if bull in given:
            costs[(AgentRole.DECIDER, decider_model)] = given[bull]
        result = estimate(report, costs, {**fixed, AgentRole.BULL: bull}, (), (), arms)
        per_role: dict[AgentRole, tuple[float | None, float | None]] = {
            line.role: (line.cost_low, line.cost_high) for line in result.roles
        }
        low, high = per_role[AgentRole.DECIDER]
        per_role[AgentRole.DECIDER] = (
            None if low is None else low * DECIDER_ATTEMPTS,
            None if high is None else high * DECIDER_ATTEMPTS,
        )
        total = result.total_with_retries
        topup = (
            None if total[0] is None else DEFAULT_PRICING.topup_charge(total[0]),
            None if total[1] is None else DEFAULT_PRICING.topup_charge(total[1]),
        )
        lines.append(
            ScenarioLine(
                bull=bull,
                valid_briefs=bull_cost.valid_verdicts,
                note=None
                if bull in given
                else "sin decisor medido sobre este bull: el total no está determinado",
                per_role=per_role,
                total=total,
                topup=topup,
                balance=None if balance is None else balance_check(total, balance),
            )
        )
    return tuple(lines)


# ───────────────────────────────────────────── Informe ────────────────────────────────────────────


def _usd(value: float | None) -> str:
    return "no determinado" if value is None else f"{value:.2f}"


def _span(low: float | None, high: float | None) -> str:
    if low is None or high is None:
        return "no determinado"
    if abs(high - low) < 0.005:
        return f"{low:.2f}"
    return f"{low:.2f} a {high:.2f}"


def _base_sections(result: Estimate) -> list[str]:
    """Por rol, por brazo y la recarga única: la estimación de un solo escenario."""
    lines = [
        "## Por rol (sumando brazos)",
        "",
        "| rol | modelo | llamadas a pagar | USD por llamada | USD total | de dónde sale |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for line in result.roles:
        model = f"`{line.model}`" if line.model else "rango de candidatos"
        lines.append(
            f"| {line.role.value} | {model} | {line.calls} | "
            f"{_span_small(line.per_call_low, line.per_call_high)} | "
            f"{_span(line.cost_low, line.cost_high)} | {line.source} |"
        )
    lines += [
        "",
        "## Por brazo",
        "",
        f"| brazo | USD, un intento por llamada | USD, decisor a {DECIDER_ATTEMPTS:g} intentos |",
        "| --- | --- | --- |",
    ]
    for name, (low, high) in result.arms.items():
        retry_low, retry_high = result.arms_with_retries[name]
        lines.append(f"| `{name}` | {_span(low, high)} | {_span(retry_low, retry_high)} |")
    lines += [
        f"| **total etapa 1** | {_span(*result.total)} | {_span(*result.total_with_retries)} |",
        "",
        "## Recarga única",
        "",
        "Se lee «4.4% + 0.30 USD por transacción» como `crédito x 1.044 + 0.30`. Es una lectura de "
        "la página (2026-10-04), no un recibo: la primera recarga real dice si es esa.",
        "",
        "| escenario | crédito necesario | cargo en tarjeta |",
        "| --- | --- | --- |",
    ]
    for label, (low, high) in (
        ("un intento por llamada", result.total),
        (f"decisor a {DECIDER_ATTEMPTS:g}", result.total_with_retries),
    ):
        for end, value in (("más barato", low), ("más caro", high)):
            charge = None if value is None else DEFAULT_PRICING.topup_charge(value)
            lines.append(f"| {label}, {end} | {_usd(value)} | {_usd(charge)} |")
    lines.append("")
    return lines


def render_balance(check: BalanceCheck) -> list[str]:
    """El saldo declarado frente al coste con el decisor a `DECIDER_ATTEMPTS`, con el veredicto."""
    return [
        "## Saldo frente a estimación",
        "",
        f"- saldo declarado (el de la consola; no se consultó la red): {check.balance:.2f} USD",
        f"- coste con el decisor a {DECIDER_ATTEMPTS:g} intentos: {_span(*check.cost)} USD",
        f"- saldo requerido = coste x {LAUNCH_MARGIN:g}: {_span(*check.required)} USD",
        f"- **{_verdict(check)}** — se juzga en el extremo alto del requerido",
        "",
    ]


_SCENARIO_ROLES = (
    ("structure + volume", (AgentRole.STRUCTURE, AgentRole.VOLUME)),
    ("momentum", (AgentRole.MOMENTUM,)),
    ("bull", (AgentRole.BULL,)),
    ("bear", (AgentRole.BEAR,)),
    (f"decisor x{DECIDER_ATTEMPTS:g}", (AgentRole.DECIDER,)),
)


def render_scenarios(lines: Sequence[ScenarioLine]) -> list[str]:
    """Una línea por candidato a bull. El mismo `bear`, técnicos y reintentos en todas."""
    with_balance = any(item.balance is not None for item in lines)
    header = ["bull", "alegatos válidos", *(label for label, _ in _SCENARIO_ROLES)]
    header += ["total (USD)", "recarga: cargo en tarjeta (USD)"]
    if with_balance:
        header += ["saldo requerido (USD)", "veredicto"]
    out = [
        "## Etapa 1 por candidato a bull",
        "",
        "Cada línea cambia solo lo que depende del bull: su propio coste y el del decisor medido "
        f"sobre **su** alegato (a {DECIDER_ATTEMPTS:g} intentos). Structure, volume, momentum y "
        "bear son las mismas cifras en todas.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + " --- |" * len(header),
    ]
    for item in lines:
        if item.note is not None and not item.per_role:
            out.append(f"| `{item.bull}` | {item.valid_briefs} | {item.note} |")
            continue
        cells = [f"`{item.bull}`", str(item.valid_briefs)]
        for _, roles in _SCENARIO_ROLES:
            parts = [item.per_role.get(role, (None, None)) for role in roles]
            cells.append(_span(*sum_pairs(parts)))
        cells += [_span(*item.total), _span(*item.topup)]
        if with_balance and item.balance is not None:
            cells += [_span(*item.balance.required), _verdict(item.balance)]
        out.append("| " + " | ".join(cells) + " |")
    notes = [f"- `{item.bull}`: {item.note}" for item in lines if item.note and item.per_role]
    return [*out, *(["", *notes] if notes else []), ""]


def sum_pairs(pairs: Iterable[Pair]) -> Pair:
    """Suma de pares (bajo, alto); con un extremo sin determinar, el resultado tampoco lo está."""
    total = NOTHING
    for low, high in pairs:
        total = total.plus(Span(low, high))
    return total.low, total.high


def render_estimate(
    result: Estimate,
    check: BalanceCheck | None = None,
    scenario_lines: Sequence[ScenarioLine] | None = None,
) -> str:
    """Las tablas, rotuladas como estimación, con el cargo de la recarga única.

    Con `scenario_lines` se imprime una línea por candidato a bull en lugar de los tres cuadros de
    un solo escenario: con los roles de las mesas sin fijar, esos cuadros dirían «no determinado».
    """
    lines = [
        f"# Etapa 1 en USD — {LABEL}",
        "",
        f"{result.activations} activaciones del manifiesto, caché vacía, un `--dry-run` por brazo. "
        "Coste por llamada = media del coste **medido** de las filas del sondeo; ningún token se "
        "estima. Las llamadas locales y las de las líneas base valen 0 por regla.",
        "",
    ]
    if scenario_lines is None:
        lines += _base_sections(result)
    else:
        lines += render_scenarios(scenario_lines)
    if check is not None:
        lines += render_balance(check)
    lines += ["## Candidatos a structure y volume (solo los que respondieron)", ""]
    lines += [
        "| rol | modelo | veredictos válidos | intentos | medidos | sin medir | USD por intento |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in result.candidates:
        bounds = c.bounds
        low, high = (None, None) if bounds is None else bounds
        lines.append(
            f"| {c.role.value} | `{c.model}` | {c.valid_verdicts} | {c.attempts} | {c.measured} | "
            f"{c.unmeasured} | {_span_small(low, high)} |"
        )
    if result.contrast:
        lines += [
            "",
            "## Contraste: los 12 `Ping` de `tests/data/usage` (no es el prompt de la corrida)",
            "",
            "| modelo | llamadas | USD por `Ping` |",
            "| --- | --- | --- |",
        ]
        lines += [
            f"| `{c.model}` | {c.attempts} | {_span_small(c.mean_usd, c.mean_usd)} |"
            for c in result.contrast
        ]
    lines += [
        "",
        "## Suposiciones",
        "",
        "- Caché vacía: ningún acierto. Una corrida que reanude paga menos.",
        "- El decisor se mide con el prompt de `full`; `solo`, `no_debate` y `bull_only` lo pagan "
        "más corto, así que su coste aquí es una cota superior.",
        "- Un coste por llamada que es un intervalo viene de dos cosas: `cached_tokens` en `null` "
        "(de «todo el prompt cacheado» a «todo nuevo») o la escritura de caché de `qwen3.8-max` "
        "(sin cobrarla, o con la entrada nueva a su precio). Es una cota, no una estimación; una "
        "llamada sin tokens no entra en la media y la hace una cota inferior (columna «sin "
        "medir»).",
        "- El saldo requerido multiplica el coste por `LAUNCH_MARGIN`, una convención fijada antes "
        "de ver ninguna estimación.",
        "",
        "## Entradas",
        "",
    ]
    lines += [f"- `{name}` sha-256 `{digest}`" for name, digest in result.inputs]
    lines.append("")
    return "\n".join(lines)


def _span_small(low: float | None, high: float | None) -> str:
    """Un coste por llamada, que son milésimas de dólar: dos decimales lo harían cero."""
    if low is None or high is None:
        return "no determinado"
    if low == high:
        return f"{low:.5f}"
    return f"{low:.5f} a {high:.5f}"


# ───────────────────────────────────────────── Comando ────────────────────────────────────────────


class Built(NamedTuple):
    """Lo que el comando arma antes de imprimir."""

    result: Estimate
    check: BalanceCheck | None
    lines: tuple[ScenarioLine, ...] | None

    @property
    def launchable(self) -> bool | None:
        """Si algo pasa con el saldo dado: la estimación, o alguna línea. `None` sin saldo."""
        if self.lines is not None:
            checks = [item.balance for item in self.lines if item.balance is not None]
            return any(c.passes for c in checks) if checks else None
        return None if self.check is None else self.check.passes


def _sources(run: RunDirectory) -> list[tuple[str, str]]:
    return [(f"{run.path.name}/{arm.path.name}", arm.sha256) for arm in run.arms if arm.sha256]


async def _build(args: argparse.Namespace) -> Built:
    settings = load_settings(args.env_file)
    run = read_run(args.probe)
    manifest = load_manifest(args.manifest)
    plan = plan_from_manifest(manifest, load_selection_histories(manifest, args.history_dir))
    report = await dry_run(ARMS, plan, settings, InMemoryResponseCache(), utc_now)

    costs = measured_costs(run)
    fixed = {
        role: settings.role_config(role).primary.model
        for role in AgentRole
        if role not in RANGED_ROLES
    }
    roles = {settings.role_config(role).primary.model: role for role in AgentRole}
    contrast = tuple(usage_fixture_costs(args.usage, roles).values())

    inputs = _sources(run)
    arm_names = [arm.name for arm in ARMS]
    lines: tuple[ScenarioLine, ...] | None = None
    if args.desks is not None:
        desks = read_run(args.desks)
        inputs += _sources(desks)
        bulls = [
            arm.arm.rpartition("@")[0]
            for arm in desks.arms
            if arm.arm.rpartition("@")[2] == AgentRole.BULL.value
        ]
        lines = scenarios(
            report,
            costs,
            measured_costs(desks),
            decider_given_bull(desks),
            fixed,
            bulls,
            arm_names,
            args.balance,
        )
    inputs += [
        (str(args.manifest), file_sha256(args.manifest)),
        (str(args.usage), file_sha256(args.usage)),
    ]
    result = estimate(report, costs, fixed, inputs, contrast, arm_names)
    check = None
    if args.balance is not None and lines is None:
        check = balance_check(result.total_with_retries, args.balance)
    return Built(result, check, lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada: `python -m crypto_agents.estimate <directorio del sondeo>`.

    Sale con 0, salvo con `--balance`: ahí 1 si el saldo no alcanza para nada (la estimación, o
    ninguno de los candidatos a bull), de modo que se pueda encadenar antes de lanzar. 1 es también
    el código de un error de lectura; el motivo va a `stderr` y el veredicto a `stdout`.
    """
    parser = argparse.ArgumentParser(
        prog="python -m crypto_agents.estimate",
        description="Estimación en USD de la etapa 1. No llama a ningún modelo.",
    )
    parser.add_argument("probe", type=Path, help="directorio de `zen_probe` (veredictos técnicos)")
    parser.add_argument(
        "--desks",
        type=Path,
        default=None,
        help="directorio de `zen_probe --desks`: una línea por candidato a bull",
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=None,
        help="saldo en USD que se lee en la consola de Zen (no se consulta por red)",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--history-dir", type=Path, default=HISTORY_DIR)
    parser.add_argument("--usage", type=Path, default=USAGE_FIXTURE)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    args = parser.parse_args(argv)
    if args.balance is not None and args.balance <= 0:
        parser.error("--balance debe ser positivo")
    try:
        built = asyncio.run(_build(args))
    except (AuditError, ConfigError, SelectionError, OSError, KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(render_estimate(built.result, built.check, built.lines))
    return 1 if built.launchable is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
