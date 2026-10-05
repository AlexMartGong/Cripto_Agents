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
from typing import TYPE_CHECKING

from crypto_agents.ablation import (
    ARMS,
    DECIDER_ATTEMPTS,
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
    "CallCost",
    "Estimate",
    "RoleLine",
    "estimate",
    "main",
    "measured_costs",
    "render_estimate",
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
    result: dict[tuple[AgentRole, str], CallCost] = {}
    for key, calls in grouped.items():
        total = consume(calls, DEFAULT_PRICING, Billing.PAYG)
        result[key] = CallCost(
            role=key[0],
            model=key[1],
            attempts=len(calls),
            measured=total.measured,
            unmeasured=total.unmeasured + total.no_response,
            mean_usd=total.cost_usd / total.measured if total.measured else None,
            valid_verdicts=valid.get(key, 0),
            source=sources[key],
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
            c for (r, _), c in costs.items() if r is role and c.valid_verdicts > 0 and c.mean_usd
        ]
        if not answered:
            return None, None, None, "ningún candidato respondió"
        prices = [c.mean_usd for c in answered if c.mean_usd is not None]
        return min(prices), max(prices), None, "sondeo: del más barato al más caro que respondió"
    model = fixed.get(role)
    found = costs.get((role, model)) if model is not None else None
    if found is None or found.mean_usd is None:
        return None, None, model, "el sondeo no midió este rol"
    return found.mean_usd, found.mean_usd, model, found.source


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


# ───────────────────────────────────────────── Informe ────────────────────────────────────────────


def _usd(value: float | None) -> str:
    return "no determinado" if value is None else f"{value:.2f}"


def _span(low: float | None, high: float | None) -> str:
    if low is None or high is None:
        return "no determinado"
    if abs(high - low) < 0.005:
        return f"{low:.2f}"
    return f"{low:.2f} a {high:.2f}"


def render_estimate(result: Estimate) -> str:
    """Las tablas, rotuladas como estimación, con el cargo de la recarga única."""
    lines = [
        f"# Etapa 1 en USD — {LABEL}",
        "",
        f"{result.activations} activaciones del manifiesto, caché vacía, un `--dry-run` por brazo. "
        "Coste por llamada = media del coste **medido** de las filas del sondeo; ningún token se "
        "estima. Las llamadas locales y las de las líneas base valen 0 por regla.",
        "",
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
    lines += ["", "## Candidatos a structure y volume (solo los que respondieron)", ""]
    lines += [
        "| rol | modelo | veredictos válidos | intentos | medidos | sin medir | USD por intento |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in result.candidates:
        lines.append(
            f"| {c.role.value} | `{c.model}` | {c.valid_verdicts} | {c.attempts} | {c.measured} | "
            f"{c.unmeasured} | {_span_small(c.mean_usd, c.mean_usd)} |"
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
        "- Una llamada con `cached_tokens` en `null` no tiene coste exacto y no entra en la media: "
        "con ellas la media es una cota inferior (columna «sin medir»).",
        "- `qwen3.8-max` tiene precio de escritura de caché que `PriceRow` no modela.",
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


async def _build(args: argparse.Namespace) -> Estimate:
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

    inputs = [(f"{run.path.name}/{arm.path.name}", arm.sha256) for arm in run.arms if arm.sha256]
    inputs += [
        (str(args.manifest), file_sha256(args.manifest)),
        (str(args.usage), file_sha256(args.usage)),
    ]
    return estimate(report, costs, fixed, inputs, contrast, [arm.name for arm in ARMS])


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada: `python -m crypto_agents.estimate <directorio del sondeo>`."""
    parser = argparse.ArgumentParser(
        prog="python -m crypto_agents.estimate",
        description="Estimación en USD de la etapa 1. No llama a ningún modelo.",
    )
    parser.add_argument("probe", type=Path, help="directorio de `zen_probe`")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--history-dir", type=Path, default=HISTORY_DIR)
    parser.add_argument("--usage", type=Path, default=USAGE_FIXTURE)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    args = parser.parse_args(argv)
    try:
        print(render_estimate(asyncio.run(_build(args))))
    except (AuditError, ConfigError, SelectionError, OSError, KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
