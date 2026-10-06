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
- **Los intentos por llamada se leen del sondeo, no se suponen.** Cada rol se presupuesta con los
  intentos vivos por invocación que midió el directorio del que sale su coste (22 intentos para 12
  alegatos son 1.83). `DECIDER_ATTEMPTS` y el 1.0 de los demás quedan como respaldo de un rol sin
  filas, y la tabla lo rotula. Se dan las dos cifras: un intento por llamada en todo, y con los
  intentos medidos. Hasta el bloque T5 eran constantes, y el sondeo de mesas las desmintió: el
  decisor tomó 1.42 a 1.58 intentos frente a 1.2, y el bear 1.83 frente a 1.0.
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
- **`--source ROL=DIRECTORIO`** (repetible) dice de qué sondeo sale un rol. Sin él, los técnicos
  salen del directorio posicional y las mesas de `--desks`. Es explícito a propósito: una
  precedencia entre directorios mezclaría mediciones sin decirlo.
- **`--balance X`** es el saldo que se lee en la consola, sin red. Pasa si `X >= coste x
  LAUNCH_MARGIN`, con el coste a los intentos medidos y juzgado en el extremo alto; un coste no
  determinado no pasa.
- **El desglose por brazo y rol suma el total, no lo sustituye.** Cada celda lleva sus llamadas,
  sus intentos medidos, su rango en USD y el archivo del que sale; la suma de las celdas de un
  brazo es el coste de ese brazo, y el veredicto se sigue juzgando sobre el total.
- **Los candidatos a `bull` son los de `candidates.BULL_CANDIDATES`**, no los brazos que haya en
  el directorio de `--desks`: un candidato ya descartado no tiene línea por haber sido medido una
  vez, y uno de la lista que el directorio no midió sale diciendo que no tiene alegatos.

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
from crypto_agents.candidates import BULL_CANDIDATES
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
    "SHORT_DECIDER_ARMS",
    "USAGE_FIXTURE",
    "ArmRoleLine",
    "BalanceCheck",
    "CallCost",
    "Estimate",
    "RoleLine",
    "ScenarioLine",
    "attempt_rate",
    "balance_check",
    "decider_given_bull",
    "estimate",
    "main",
    "measured_costs",
    "render_breakdown",
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

    invocations: int = 0
    """Registros del brazo con al menos un intento: las veces que se le pidió un veredicto.

    `0` es «no se sabe»: el fixture de `Ping` y las construidas a mano no son invocaciones."""

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

    @property
    def attempts_per_verdict(self) -> float | None:
        """Intentos vivos por invocación pedida, o `None` si no se sabe cuántas se pidieron.

        El denominador son las invocaciones y no los veredictos válidos: lo que el conteo previo
        cuenta son llamadas que se van a hacer, validen o no. Entran los intentos que el proveedor
        no contestó, que también son intentos.
        """
        return self.attempts / self.invocations if self.invocations else None


def _answered(call: LLMCall) -> bool:
    return not call.cache_hit and call.backend is not Backend.OLLAMA


def measured_costs(run: RunDirectory) -> dict[tuple[AgentRole, str], CallCost]:
    """Coste medio por intento de cada (rol, modelo) que el sondeo midió con su prompt real.

    Los brazos `@ping` y `@header-*` quedan fuera: miden un `Ping` de sesenta caracteres, no un
    prompt que vaya a correr. Entran los intentos inválidos —también se facturaron—.
    """
    grouped: dict[tuple[AgentRole, str], list[LLMCall]] = {}
    valid: dict[tuple[AgentRole, str], int] = {}
    asked: dict[tuple[AgentRole, str], int] = {}
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
            if any(_answered(call) for call in record.calls):
                key = (record.calls[0].role, record.calls[0].model)
                asked[key] = asked.get(key, 0) + 1
            if record.calls and not record.errors:
                key = (record.calls[0].role, record.calls[0].model)
                valid[key] = valid.get(key, 0) + 1
    return {
        key: _call_cost(key[0], key[1], calls, valid.get(key, 0), sources[key], asked.get(key, 0))
        for key, calls in grouped.items()
    }


def _call_cost(
    role: AgentRole,
    model: str,
    calls: Sequence[LLMCall],
    valid: int,
    source: str,
    invocations: int = 0,
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
        invocations=invocations,
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
        asked = sum(1 for record in arm.records if any(_answered(c) for c in record.calls))
        result[bull] = _call_cost(
            AgentRole.DECIDER,
            calls[0].model,
            calls,
            valid,
            f"{run.path.name}/{arm.path.name}",
            asked,
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

    attempts_low: float = 1.0
    attempts_high: float = 1.0
    """Intentos por llamada con que se presupuesta el rol; un rango entre candidatos si los hay."""

    attempts_source: str = ""
    """De qué filas sale ese factor, o que es la constante de respaldo."""

    retried_low: float | None = None
    retried_high: float | None = None
    """El coste total del rol con esos intentos. `cost_*` es a un intento por llamada."""


SHORT_DECIDER_ARMS = frozenset({"solo", "no_debate", "bull_only"})
"""Los brazos cuyo decisor lee un prompt más corto que el de `full`, que es el que se midió."""

UPPER_BOUND_NOTE = "cota superior: es el coste del decisor con el prompt de `full`"
LOCAL_NOTE = "local: 0 por regla"
NO_CALLS_NOTE = "no llama a ningún modelo: 0 por regla"
CACHED_NOTE = "0 a pagar: lo resuelve la caché con lo que ya pagó otro brazo"


class ArmRoleLine(FrozenModel):
    """Lo que un rol cuesta dentro de un brazo: sus llamadas, sus intentos y de dónde sale.

    Es una celda del total, no otra estimación: la suma de las líneas de un brazo es el coste de
    ese brazo con los intentos medidos, y la de todas, el total sobre el que se juzga el saldo.
    """

    arm: str
    node: str | None = None
    role: AgentRole | None = None
    """`None` en un brazo que no llama a ningún modelo: una línea base."""

    model: str | None = None
    """El modelo fijado, o `None` si el rol va como rango de candidatos."""

    calls: int = 0
    """Llamadas que llegan a un proveedor remoto. Un nodo local o resuelto por la caché, 0."""

    exact: bool = True
    """Si `calls` es el número real o la cota de lo que sigue a un veredicto."""

    attempts_low: float | None = None
    attempts_high: float | None = None
    """Intentos por llamada con que se presupuesta. `None` donde no se paga nada."""

    usd_low: float | None = 0.0
    usd_high: float | None = 0.0
    """El coste con esos intentos. `None` es «no determinado», nunca cero."""

    source: str = ""
    attempts_source: str = ""
    note: str | None = None


class Estimate(FrozenModel):
    """La estimación completa, con sus entradas y las suposiciones que la sostienen."""

    activations: int
    roles: tuple[RoleLine, ...]
    arms: dict[str, tuple[float | None, float | None]]
    """Coste por brazo, con un intento por llamada."""

    arms_with_retries: dict[str, tuple[float | None, float | None]]
    """Lo mismo con los intentos por llamada medidos de cada rol."""

    total: tuple[float | None, float | None]
    total_with_retries: tuple[float | None, float | None]
    candidates: tuple[CallCost, ...]
    inputs: tuple[tuple[str, str], ...]
    contrast: tuple[CallCost, ...] = ()
    breakdown: tuple[ArmRoleLine, ...] = ()
    """Por brazo y rol, con los intentos medidos: las celdas que suman `arms_with_retries`."""


class AttemptRate(NamedTuple):
    """Los intentos por llamada con que se presupuesta un rol, y de dónde salen."""

    value: float
    source: str


FALLBACK_NOTE = "sin datos: constante de respaldo"


def attempt_rate(role: AgentRole, cost: CallCost | None) -> AttemptRate:
    """Los intentos por veredicto que midió el sondeo, o la constante si no midió ninguno.

    La constante es la de antes —`DECIDER_ATTEMPTS` para el decisor, 1.0 para los demás— y solo
    entra cuando no hay invocaciones que contar. Va rotulada: una cifra que sale de una convención
    no puede leerse como una que salió de un journal.
    """
    measured = None if cost is None else cost.attempts_per_verdict
    if cost is None or measured is None:
        fallback = DECIDER_ATTEMPTS if role is AgentRole.DECIDER else 1.0
        return AttemptRate(fallback, f"{FALLBACK_NOTE} ({fallback:g})")
    return AttemptRate(
        measured, f"{cost.attempts} intentos / {cost.invocations} invocaciones · `{cost.source}`"
    )


class PerCall(NamedTuple):
    """Lo que cuesta una llamada de un rol: por intento, y a los intentos que se le presupuestan."""

    low: float | None
    high: float | None
    """USD por intento."""

    retried_low: float | None
    retried_high: float | None
    """USD por llamada, a los intentos medidos."""

    model: str | None
    source: str
    attempts: tuple[float, float]
    attempts_source: str


def _per_call(
    role: AgentRole,
    costs: Mapping[tuple[AgentRole, str], CallCost],
    fixed: Mapping[AgentRole, str],
) -> PerCall:
    """Coste por llamada de un rol, con su modelo fijado (o el rango) y su procedencia.

    En un rol dado como rango, los intentos son de cada candidato: se multiplica el coste de cada
    uno por los suyos y después se toma el más barato y el más caro. Aplicar un factor común al
    rango cruzaría los intentos de un modelo con el precio de otro.
    """
    if role in RANGED_ROLES:
        answered = [
            (c, b, attempt_rate(role, c))
            for (r, _), c in costs.items()
            if r is role and c.valid_verdicts > 0 and (b := c.bounds) is not None and b[1] > 0
        ]
        if not answered:
            rate = attempt_rate(role, None)
            return PerCall(
                None,
                None,
                None,
                None,
                None,
                "ningún candidato respondió",
                (rate.value, rate.value),
                rate.source,
            )
        rates = [rate.value for _, _, rate in answered]
        cheapest = min(answered, key=lambda item: item[1][0] * item[2].value)[0]
        dearest = max(answered, key=lambda item: item[1][1] * item[2].value)[0]
        return PerCall(
            min(b[0] for _, b, _ in answered),
            max(b[1] for _, b, _ in answered),
            min(b[0] * rate.value for _, b, rate in answered),
            max(b[1] * rate.value for _, b, rate in answered),
            None,
            "sondeo: del más barato al más caro que respondió, "
            f"`{cheapest.model}` (`{cheapest.source}`) a `{dearest.model}` (`{dearest.source}`)",
            (min(rates), max(rates)),
            "por candidato: "
            + "; ".join(f"`{c.model}` {rate.value:.2f} ({rate.source})" for c, _, rate in answered),
        )
    model = fixed.get(role)
    found = costs.get((role, model)) if model is not None else None
    bounds = None if found is None else found.bounds
    rate = attempt_rate(role, found)
    attempts = (rate.value, rate.value)
    if found is None or bounds is None:
        return PerCall(
            None, None, None, None, model, "el sondeo no midió este rol", attempts, rate.source
        )
    return PerCall(
        bounds[0],
        bounds[1],
        bounds[0] * rate.value,
        bounds[1] * rate.value,
        model,
        found.source,
        attempts,
        rate.source,
    )


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
    per_role = {role: _per_call(role, costs, fixed) for role in AgentRole}
    paid: dict[AgentRole, int] = {}
    # Todos los brazos pedidos aparecen, también los que no llaman a nadie: una línea base con
    # cero filas es un brazo que cuesta 0, no un brazo ausente de la tabla.
    per_arm: dict[str, Span] = {name: NOTHING for name in arms}
    per_arm_retry: dict[str, Span] = {name: NOTHING for name in arms}
    cells: dict[str, list[ArmRoleLine]] = {name: [] for name in arms}
    for row in report.rows:
        per_arm.setdefault(row.arm, NOTHING)
        per_arm_retry.setdefault(row.arm, NOTHING)
        if row.backend is Backend.OLLAMA:
            cells.setdefault(row.arm, []).append(
                ArmRoleLine(
                    arm=row.arm, node=row.node, role=row.role, model=row.model, source=LOCAL_NOTE
                )
            )
            continue
        calls = _paid_calls(row)
        paid[row.role] = paid.get(row.role, 0) + calls
        each = per_role[row.role]
        retried = Span(each.retried_low, each.retried_high).scaled(calls)
        per_arm[row.arm] = per_arm[row.arm].plus(Span(each.low, each.high).scaled(calls))
        per_arm_retry[row.arm] = per_arm_retry[row.arm].plus(retried)
        cells.setdefault(row.arm, []).append(
            ArmRoleLine(
                arm=row.arm,
                node=row.node,
                role=row.role,
                model=each.model,
                calls=calls,
                exact=row.exact,
                attempts_low=each.attempts[0],
                attempts_high=each.attempts[1],
                usd_low=retried.low,
                usd_high=retried.high,
                source=each.source,
                attempts_source=each.attempts_source,
                note=_cell_note(row, calls),
            )
        )
    breakdown = tuple(
        line
        for name, lines_of_arm in cells.items()
        for line in (lines_of_arm or [ArmRoleLine(arm=name, source=NO_CALLS_NOTE)])
    )

    lines: list[RoleLine] = []
    for role in AgentRole:
        each = per_role[role]
        calls = paid.get(role, 0)
        one, retried = (
            Span(each.low, each.high).scaled(calls),
            Span(each.retried_low, each.retried_high).scaled(calls),
        )
        lines.append(
            RoleLine(
                role=role,
                calls=calls,
                model=each.model,
                per_call_low=each.low,
                per_call_high=each.high,
                cost_low=one.low,
                cost_high=one.high,
                source=each.source,
                attempts_low=each.attempts[0],
                attempts_high=each.attempts[1],
                attempts_source=each.attempts_source,
                retried_low=retried.low,
                retried_high=retried.high,
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
        breakdown=breakdown,
    )


def _cell_note(row: DryRunRow, calls: int) -> str | None:
    """Lo que hay que saber de una celda además de su cifra."""
    if calls == 0:
        return CACHED_NOTE
    if row.role is AgentRole.DECIDER and row.arm in SHORT_DECIDER_ARMS:
        return UPPER_BOUND_NOTE
    return None


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
    """Coste con los intentos medidos de cada rol, de más barato a más caro."""

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
    """Coste total de cada rol sobre todos los brazos, ya con sus intentos medidos."""

    roles: tuple[RoleLine, ...] = ()
    """Las líneas por rol de este escenario: de ahí salen los intentos y su procedencia."""

    total: tuple[float | None, float | None] = (None, None)
    topup: tuple[float | None, float | None] = (None, None)
    balance: BalanceCheck | None = None
    breakdown: tuple[ArmRoleLine, ...] = ()
    """Las celdas por brazo y rol de este escenario. Suman `total`."""


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
            line.role: (line.retried_low, line.retried_high) for line in result.roles
        }
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
                roles=result.roles,
                total=total,
                topup=topup,
                balance=None if balance is None else balance_check(total, balance),
                breakdown=result.breakdown,
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
        "| brazo | USD, un intento por llamada | USD, con los intentos medidos |",
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
        ("con los intentos medidos", result.total_with_retries),
    ):
        for end, value in (("más barato", low), ("más caro", high)):
            charge = None if value is None else DEFAULT_PRICING.topup_charge(value)
            lines.append(f"| {label}, {end} | {_usd(value)} | {_usd(charge)} |")
    lines.append("")
    return lines


def render_balance(check: BalanceCheck) -> list[str]:
    """El saldo declarado frente al coste con los intentos medidos, con el veredicto."""
    return [
        "## Saldo frente a estimación",
        "",
        f"- saldo declarado (el de la consola; no se consultó la red): {check.balance:.2f} USD",
        f"- coste con los intentos medidos: {_span(*check.cost)} USD",
        f"- saldo requerido = coste x {LAUNCH_MARGIN:g}: {_span(*check.required)} USD",
        f"- **{_verdict(check)}** — se juzga en el extremo alto del requerido",
        "",
    ]


_SCENARIO_ROLES = (
    ("structure + volume", (AgentRole.STRUCTURE, AgentRole.VOLUME)),
    ("momentum", (AgentRole.MOMENTUM,)),
    ("bull", (AgentRole.BULL,)),
    ("bear", (AgentRole.BEAR,)),
    ("decisor", (AgentRole.DECIDER,)),
)


def _rate(low: float, high: float) -> str:
    return f"{low:.2f}" if abs(high - low) < 0.005 else f"{low:.2f} a {high:.2f}"


def render_attempts(rows: Sequence[tuple[str, RoleLine]]) -> list[str]:
    """Los intentos por veredicto de cada rol, con su n y el directorio del que salen.

    Es la tabla que permite discutir el factor: cuántos intentos, sobre cuántas invocaciones y en
    qué archivo. Un rol sin filas lleva la constante de respaldo y lo dice.
    """
    lines = [
        "## Intentos por veredicto",
        "",
        "Intentos vivos por invocación pedida, leídos del sondeo del que sale el coste de cada rol "
        "(los que el proveedor no contestó también cuentan). Multiplican el coste por llamada.",
        "",
        "| escenario | rol | modelo | intentos por veredicto | de dónde sale |",
        "| --- | --- | --- | --- | --- |",
    ]
    for label, line in rows:
        model = f"`{line.model}`" if line.model else "rango de candidatos"
        lines.append(
            f"| {label} | {line.role.value} | {model} | "
            f"{_rate(line.attempts_low, line.attempts_high)} | {line.attempts_source} |"
        )
    return [*lines, ""]


def attempt_rows(
    result: Estimate, scenario_lines: Sequence[ScenarioLine] | None
) -> list[tuple[str, RoleLine]]:
    """Qué filas lleva la tabla de intentos: las del escenario único, o las de cada bull.

    Con una línea por bull, los roles que no dependen de él salen una vez y `bull` y `decider`
    una vez por candidato, que es donde cambian.
    """
    costed = [item for item in (scenario_lines or ()) if item.roles]
    if not costed:
        return [("único", line) for line in result.roles]
    varying = (AgentRole.BULL, AgentRole.DECIDER)
    rows = [("todos", line) for line in costed[0].roles if line.role not in varying]
    for item in costed:
        rows += [(f"bull `{item.bull}`", line) for line in item.roles if line.role in varying]
    return rows


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
        "sobre **su** alegato. Cada rol va a sus intentos medidos (tabla de abajo). Structure, "
        "volume, momentum y bear son las mismas cifras en todas.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + " --- |" * len(header),
    ]
    for item in lines:
        if item.note is not None and not item.per_role:
            filler = ["—"] * (len(header) - 3)
            out.append(
                "| "
                + " | ".join([f"`{item.bull}`", str(item.valid_briefs), item.note, *filler])
                + " |"
            )
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


def render_breakdown(lines: Sequence[ArmRoleLine], title: str) -> list[str]:
    """Por brazo y por rol: llamadas, intentos medidos, USD y de dónde sale cada cifra.

    Cada brazo cierra con su subtotal y la tabla con el total, que es el mismo sobre el que se
    juzga el saldo. Una celda sin determinar deja sin determinar el subtotal y el total: no se
    suma lo que hay como si lo que falta costara cero.
    """
    out = [
        f"## {title}",
        "",
        "Coste con los intentos medidos de cada rol. «≤» es una cota: el nodo sigue a un veredicto "
        "y no se sabe cuántas evaluaciones llegan. El sha-256 de cada archivo está en «Entradas».",
        "",
        "| brazo | rol | modelo | llamadas a pagar | intentos por llamada | USD | "
        "de dónde sale el coste | de dónde salen los intentos |",
        "|" + " --- |" * 8,
    ]
    total = NOTHING
    by_arm: dict[str, list[ArmRoleLine]] = {}
    for line in lines:
        by_arm.setdefault(line.arm, []).append(line)
    for arm, group in by_arm.items():
        subtotal = NOTHING
        for line in group:
            subtotal = subtotal.plus(Span(line.usd_low, line.usd_high))
            if line.role is None:
                out.append(f"| `{arm}` | — | — | 0 | — | {_span(0.0, 0.0)} | {line.source} | — |")
                continue
            model = f"`{line.model}`" if line.model else "rango de candidatos"
            calls = str(line.calls) if line.exact else f"≤ {line.calls}"
            attempts = (
                "—"
                if line.attempts_low is None or line.attempts_high is None
                else _rate(line.attempts_low, line.attempts_high)
            )
            source = line.source if line.note is None else f"{line.source} · {line.note}"
            out.append(
                f"| `{arm}` | {line.role.value} | {model} | {calls} | {attempts} | "
                f"{_span(line.usd_low, line.usd_high)} | {source} | {line.attempts_source or '—'} |"
            )
        out.append(
            f"| `{arm}` | **total del brazo** | | | | {_span(subtotal.low, subtotal.high)} | | |"
        )
        total = total.plus(subtotal)
    out.append(f"| **total etapa 1** | | | | | {_span(total.low, total.high)} | | |")
    return [*out, ""]


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
    if scenario_lines is None:
        lines += render_breakdown(result.breakdown, "Desglose por brazo y rol")
    else:
        for item in scenario_lines:
            if item.breakdown:
                lines += render_breakdown(
                    item.breakdown, f"Desglose por brazo y rol — bull `{item.bull}`"
                )
    lines += render_attempts(attempt_rows(result, scenario_lines))
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


def replace_roles(
    costs: Mapping[tuple[AgentRole, str], CallCost],
    overrides: Sequence[tuple[AgentRole, RunDirectory]],
) -> dict[tuple[AgentRole, str], CallCost]:
    """`costs` con cada rol de `overrides` sustituido por lo que midió su directorio.

    Se sustituye el rol entero, no se mezcla: las filas de ese rol que traía `costs` salen todas
    y entran las del otro sondeo. Un rol que el otro directorio no midió queda sin coste, que es
    «no determinado» y no un cero.
    """
    result = dict(costs)
    for role, run in overrides:
        result = {key: cost for key, cost in result.items() if key[0] is not role}
        result.update({key: c for key, c in measured_costs(run).items() if key[0] is role})
    return result


def _source(value: str) -> tuple[AgentRole, Path]:
    """`ROL=DIRECTORIO`, para `--source`."""
    name, separator, directory = value.partition("=")
    if not separator or not directory:
        raise argparse.ArgumentTypeError(f"se esperaba ROL=DIRECTORIO, no {value!r}")
    try:
        return AgentRole(name), Path(directory)
    except ValueError:
        roles = ", ".join(role.value for role in AgentRole)
        raise argparse.ArgumentTypeError(f"rol desconocido {name!r}: uno de {roles}") from None


async def _build(args: argparse.Namespace) -> Built:
    settings = load_settings(args.env_file)
    run = read_run(args.probe)
    manifest = load_manifest(args.manifest)
    plan = plan_from_manifest(manifest, load_selection_histories(manifest, args.history_dir))
    report = await dry_run(ARMS, plan, settings, InMemoryResponseCache(), utc_now)

    overrides = [(role, read_run(directory)) for role, directory in args.source]
    costs = replace_roles(measured_costs(run), overrides)
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
        bulls = [candidate.model for candidate in BULL_CANDIDATES]
        given = decider_given_bull(desks)
        for role, other in overrides:
            if role is AgentRole.DECIDER:
                given = decider_given_bull(other)
        lines = scenarios(
            report,
            costs,
            replace_roles(measured_costs(desks), overrides),
            given,
            fixed,
            bulls,
            arm_names,
            args.balance,
        )
    for _, other in overrides:
        inputs += [item for item in _sources(other) if item not in inputs]
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
        "--source",
        type=_source,
        action="append",
        default=[],
        metavar="ROL=DIRECTORIO",
        help="de qué sondeo salen el coste y los intentos de ese rol (repetible)",
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
