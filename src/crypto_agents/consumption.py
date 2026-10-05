"""Cuánto cuesta cada llamada y cuánto de la suscripción gasta una corrida. Solo mide.

OpenCode Go limita por dólares: 5 h = 20% del límite mensual, la semana 50%, el mes 100%, y cada
modelo consume el pool con un peso fijado por su «límite mensual». `QuotaLedger` cuenta peticiones
por rol; esto cuenta dólares y fracciones de pool a partir de los tokens que el proveedor declaró.
No cambia el ledger, ni los reintentos, ni el reparto rol → modelo.

Cuatro decisiones que lo mantienen honesto:

- **Una llamada sin tokens no cuesta cero.** Su coste es `None`, y el informe la cuenta aparte
  en vez de sumarla: la cifra de un brazo con llamadas sin medir es una cota inferior y se marca
  `≥`. Nada aquí estima tokens, ni de un prompt ni de una media.
- **`null` no es cero.** DeepSeek V4 Flash devuelve `cached_tokens: null` en frío. El coste exacto
  exige saber cuántos fueron, así que esa llamada queda sin medir; lo único que se puede decir es
  una cota superior —cobrar todo como entrada nueva—, que se rotula como tal.
- **Gratis es gratis por regla.** Un acierto de caché y una llamada local valen 0, lleven los
  tokens que lleven: no hicieron petición, o corrieron en la GPU de la casa.
- **Un fallo sin respuesta no es una medición ausente.** Un 4xx no se cobra y un timeout no se
  sabe; van en su propia columna y no inflan «sin medir».

Los precios y el pico viven en `settings.py` (`PriceTable`), no aquí. Este módulo no importa el
router, el grafo ni la caché: leer un directorio de corrida no puede costar una llamada.

    python -m crypto_agents.consumption <directorio>
    python -m crypto_agents.consumption --quotas
"""

from __future__ import annotations

import argparse
import sys
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from crypto_agents.audit import AuditError, run_chain
from crypto_agents.settings import (
    DEFAULT_ENV_FILE,
    DEFAULT_PRICING,
    ConfigError,
    PriceTable,
    load_settings,
)
from crypto_agents.state import AgentRole, Backend, Billing, FailureKind, FrozenModel

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from crypto_agents.settings import PriceRow, Settings
    from crypto_agents.state import LLMCall

__all__ = [
    "Consumption",
    "CostGap",
    "QuotaCheck",
    "by_role",
    "call_cost_usd",
    "consume",
    "cost_bound_usd",
    "cost_gap",
    "is_free",
    "main",
    "pool_fraction",
    "quota_checks",
    "render_consumption",
    "render_quota_checks",
]

TOKENS_PER_MTOK = 1_000_000
"""Los precios son por millón de tokens."""

PAGE_LABEL = "estimación de la página, no medida"


class CostGap(StrEnum):
    """Por qué una llamada no tiene coste exacto. Cada una se cuenta aparte."""

    NO_RESPONSE = "sin respuesta"
    """El proveedor no devolvió contenido (transporte o timeout): no hay tokens que esperar."""

    NO_TOKENS = "sin tokens"
    """Respondió y no declaró contadores de entrada o de salida."""

    CACHED_UNREPORTED = "cached no informado"
    """Declaró entrada y salida pero dejó `cached_tokens` en `null`."""

    INCONSISTENT = "incoherentes"
    """Declaró más tokens cacheados que de entrada, cuando la entrada los incluye."""

    NO_PRICE = "sin precio"
    """La tabla no tiene tarifa de ese modelo con esa forma de pago."""


def is_free(call: LLMCall) -> bool:
    """Un acierto de caché o una llamada local: no cuestan, lleven los tokens que lleven."""
    return call.cache_hit or call.backend is Backend.OLLAMA


def _row(call: LLMCall, prices: PriceTable, billing: Billing) -> PriceRow | None:
    return prices.price_for(call.model, billing, call.at)


def cost_gap(call: LLMCall, prices: PriceTable, billing: Billing) -> CostGap | None:
    """Por qué `call_cost_usd` es `None`, o `None` si la llamada tiene coste (cero incluido)."""
    if is_free(call):
        return None
    if call.failure_kind in (FailureKind.TRANSPORT, FailureKind.TIMEOUT):
        return CostGap.NO_RESPONSE
    if call.prompt_tokens is None or call.completion_tokens is None:
        return CostGap.NO_TOKENS
    if call.cached_tokens is None:
        return CostGap.CACHED_UNREPORTED
    if call.cached_tokens > call.prompt_tokens:
        return CostGap.INCONSISTENT
    if _row(call, prices, billing) is None:
        return CostGap.NO_PRICE
    return None


def _usd(row: PriceRow, fresh: int, cached: int, completion: int) -> float:
    return (
        fresh * row.input_per_mtok + cached * row.cached_per_mtok + completion * row.output_per_mtok
    ) / TOKENS_PER_MTOK


def call_cost_usd(call: LLMCall, prices: PriceTable, billing: Billing) -> float | None:
    """Coste en USD de la llamada, o `None` si faltan tokens o el precio.

    La entrada nueva —`prompt_tokens` menos los cacheados— se cobra a la tarifa de entrada,
    los cacheados a la de lectura de caché, y la salida a la suya. `prompt_tokens` incluye los
    cacheados: así lo midió la pasarela, y cobrarlos otra vez como entrada daría de más.
    Un acierto de caché y una llamada local valen cero.
    """
    if is_free(call):
        return 0.0
    if cost_gap(call, prices, billing) is not None:
        return None
    row = _row(call, prices, billing)
    assert row is not None  # lo garantiza `cost_gap`
    assert call.prompt_tokens is not None
    assert call.cached_tokens is not None
    assert call.completion_tokens is not None
    return _usd(
        row,
        call.prompt_tokens - call.cached_tokens,
        call.cached_tokens,
        call.completion_tokens,
    )


def cost_bound_usd(call: LLMCall, prices: PriceTable, billing: Billing) -> float | None:
    """Lo máximo que pudo costar la llamada, o `None` si ni siquiera eso se puede decir.

    Con coste exacto es ese coste. Con `cached_tokens` en `null` es la cota que sale de
    cobrar toda la entrada como nueva: los cacheados cuestan menos, así que no se puede
    superar. Es una cota y no una estimación; con tokens ausentes no hay de dónde sacarla.
    """
    exact = call_cost_usd(call, prices, billing)
    if exact is not None:
        return exact
    if cost_gap(call, prices, billing) is not CostGap.CACHED_UNREPORTED:
        return None
    row = _row(call, prices, billing)
    if row is None:
        return None
    assert call.prompt_tokens is not None
    assert call.completion_tokens is not None
    return _usd(row, call.prompt_tokens, 0, call.completion_tokens)


def pool_fraction(call: LLMCall, prices: PriceTable, billing: Billing) -> float | None:
    """Fracción del pool mensual que gastó la llamada. `None` con pago por uso: no hay pool.

    Es el coste entre el límite mensual del modelo. Con `go` las llamadas gratis valen
    `0.0`; las que no tienen coste, `None`.
    """
    if billing is not Billing.GO:
        return None
    cost = call_cost_usd(call, prices, billing)
    if cost is None:
        return None
    if is_free(call):
        return 0.0
    row = _row(call, prices, billing)
    assert row is not None
    assert row.monthly_limit_usd is not None  # lo exige `PriceRow` para `go`
    return cost / row.monthly_limit_usd


# ──────────────────────────────────────── Agregación ──────────────────────────────────────────────


class Consumption(FrozenModel):
    """El consumo de un conjunto de llamadas, con cuántas hay detrás de cada cifra."""

    calls: int
    measured: int
    """Llamadas vivas remotas con coste exacto."""

    free: int
    """Aciertos de caché y llamadas locales: cero por regla."""

    no_response: int
    no_tokens: int
    cached_unreported: int
    inconsistent: int
    no_price: int
    cost_usd: float
    """Suma de las medidas. Con llamadas sin medir es una cota inferior."""

    pool: float | None
    """Fracción del pool mensual de las medidas. `None` con pago por uso."""

    unreported_ceiling_usd: float
    """Cota superior de lo que costaron las llamadas con `cached_tokens` en `null`."""

    @property
    def unmeasured(self) -> int:
        """Llamadas que respondieron y no tienen coste exacto."""
        return self.no_tokens + self.cached_unreported + self.inconsistent + self.no_price

    @property
    def ceiling_usd(self) -> float | None:
        """Cota superior del total, solo si lo único que falta es la caché.

        Con tokens ausentes, un precio ausente o cifras incoherentes no hay cota que dar.
        """
        if self.no_tokens or self.inconsistent or self.no_price:
            return None
        return self.cost_usd + self.unreported_ceiling_usd

    def window_share(self, prices: PriceTable) -> float | None:
        """Qué parte de una ventana de 5 h equivale ese consumo del pool."""
        return None if self.pool is None else self.pool / prices.five_hour_share


def consume(calls: Iterable[LLMCall], prices: PriceTable, billing: Billing) -> Consumption:
    """Clasifica cada llamada en exactamente una clase y suma lo medido."""
    counts = dict.fromkeys(CostGap, 0)
    total = measured = free = 0
    cost = 0.0
    pool = 0.0
    ceiling = 0.0
    for call in calls:
        total += 1
        if is_free(call):
            free += 1
            continue
        gap = cost_gap(call, prices, billing)
        if gap is None:
            measured += 1
            exact = call_cost_usd(call, prices, billing)
            assert exact is not None
            cost += exact
            fraction = pool_fraction(call, prices, billing)
            pool += fraction or 0.0
            continue
        counts[gap] += 1
        if gap is CostGap.CACHED_UNREPORTED:
            ceiling += cost_bound_usd(call, prices, billing) or 0.0
    return Consumption(
        calls=total,
        measured=measured,
        free=free,
        no_response=counts[CostGap.NO_RESPONSE],
        no_tokens=counts[CostGap.NO_TOKENS],
        cached_unreported=counts[CostGap.CACHED_UNREPORTED],
        inconsistent=counts[CostGap.INCONSISTENT],
        no_price=counts[CostGap.NO_PRICE],
        cost_usd=cost,
        pool=pool if billing is Billing.GO else None,
        unreported_ceiling_usd=ceiling,
    )


def by_role(arms: Mapping[str, Sequence[LLMCall]]) -> dict[AgentRole, list[LLMCall]]:
    """Las llamadas de todos los brazos, agrupadas por rol y en el orden de `AgentRole`."""
    grouped: dict[AgentRole, list[LLMCall]] = {}
    for call in (call for calls in arms.values() for call in calls):
        grouped.setdefault(call.role, []).append(call)
    return {role: grouped[role] for role in AgentRole if role in grouped}


# ───────────────────────────────────────── Informe ────────────────────────────────────────────────

_HEADER = (
    "llamadas",
    "medidas",
    "sin medir",
    "sin respuesta",
    "gratis",
    "coste USD",
    "% pool mensual",
    "% ventana 5 h",
    "cota superior USD",
)


def _bounded(text: str, lower_bound: bool) -> str:
    """Una cifra, con `≥` delante si faltan llamadas por sumar."""
    return f"≥ {text}" if lower_bound else text


def _row_cells(c: Consumption, prices: PriceTable) -> list[str]:
    lower = c.unmeasured > 0
    window = c.window_share(prices)
    if c.pool is None or window is None:
        pool_cell = window_cell = "sin pool (payg)"
    else:
        pool_cell = _bounded(f"{c.pool * 100:.4f}%", lower)
        window_cell = _bounded(f"{window * 100:.3f}%", lower)
    if not lower:
        ceiling = "—"
    elif c.ceiling_usd is None:
        ceiling = "no determinado: hay llamadas sin tokens"
    else:
        ceiling = f"{c.ceiling_usd:.4f}"
    return [
        str(c.calls),
        str(c.measured),
        str(c.unmeasured),
        str(c.no_response),
        str(c.free),
        _bounded(f"{c.cost_usd:.4f}", lower),
        pool_cell,
        window_cell,
        ceiling,
    ]


def _table(label: str, rows: Mapping[str, Consumption], prices: PriceTable) -> list[str]:
    lines = [f"| {label} | " + " | ".join(_HEADER) + " |", "| --- |" + " --- |" * len(_HEADER)]
    lines.extend(
        f"| {name} | " + " | ".join(_row_cells(c, prices)) + " |" for name, c in rows.items()
    )
    lines.append("")
    return lines


def render_consumption(
    arms: Mapping[str, Sequence[LLMCall]],
    prices: PriceTable,
    billing: Billing,
    source: str,
    inputs: Sequence[tuple[str, str]] = (),
) -> str:
    """Consumo por brazo y por rol, con las llamadas medidas y las que no, al lado.

    `inputs` son los archivos de donde salieron las llamadas y el sha-256 de cada uno: una
    cifra sin el digest de su origen no se puede volver a comprobar.
    """
    everything = [call for calls in arms.values() for call in calls]
    total = consume(everything, prices, billing)
    pool = (
        f"ventana de 5 h = {prices.five_hour_share:.0%} del pool mensual"
        if billing is Billing.GO
        else "pago por uso: no hay pool, el consumo se expresa en dólares"
    )
    lines = [
        f"# Consumo de la corrida `{source}`",
        "",
        f"facturación: {billing.value} · precios de la página al {prices.as_of(billing)} · {pool}",
        "",
        "- **medidas**: llamadas vivas remotas con sus tres contadores y precio; coste exacto.",
        "- **sin medir**: respondieron y falta algo para calcularlo (ver causas abajo).",
        "- **sin respuesta**: transporte o timeout; no hay tokens que esperar.",
        "- **gratis**: acierto de caché o llamada local, cero por regla.",
        "- Con `sin medir` > 0 la cifra es una cota inferior (`≥`). Nada aquí estima tokens.",
        "",
    ]
    if inputs:
        lines += ["## Entradas", ""]
        lines += [f"- `{name}` sha-256 `{digest}`" for name, digest in inputs]
        lines.append("")

    per_arm = {name: consume(calls, prices, billing) for name, calls in arms.items()}
    lines += ["## Por brazo", ""]
    lines += _table("brazo", per_arm, prices)
    per_role = {
        role.value: consume(calls, prices, billing) for role, calls in by_role(arms).items()
    }
    lines += ["## Por rol (sumando brazos)", ""]
    lines += _table("rol", per_role, prices)
    lines += [
        "Sin medir, por causa: "
        f"sin tokens {total.no_tokens} · cached no informado {total.cached_unreported} · "
        f"incoherentes {total.inconsistent} · sin precio {total.no_price}",
        "",
    ]
    return "\n".join(lines)


# ──────────────────────────────── Cuota declarada frente a la página ──────────────────────────────


class QuotaCheck(FrozenModel):
    """La cuota declarada de un rol contra lo que la página estima para su modelo."""

    role: AgentRole
    model: str
    weight: float
    configured: int
    """`quota_per_window` tal como está en la configuración."""

    page: int | None
    """Peticiones por 5 h que la página publica para ese modelo, o `None` si no las publica."""

    billing: Billing = Billing.GO
    """Con qué forma de pago se leyó la cuota: con `payg` no hay estimado que comparar."""

    @property
    def effective(self) -> float:
        """Llamadas que caben de verdad: el ledger cobra `quota_weight` por llamada."""
        return self.configured / self.weight

    @property
    def matches(self) -> bool | None:
        """Si coincide con la página, o `None` si no hay estimado con qué comparar."""
        return None if self.page is None else self.configured == self.page


def quota_checks(settings: Settings) -> tuple[QuotaCheck, ...]:
    """Una fila por rol, sobre su modelo primario. No corrige nada: solo compara.

    Un modelo que la tabla no conoce —un respaldo local, por ejemplo— no tiene estimado de
    la página: se informa como no comparable en vez de omitirse.

    Los estimados son de la página de Go. Con pago por uso no se comparan: Zen no publica límite
    de peticiones, y contrastar una cifra de Zen con un estimado de Go inventaría discrepancias.
    """
    checks: list[QuotaCheck] = []
    for role in AgentRole:
        if role not in settings.roles:
            continue
        primary = settings.role_config(role).primary
        page = (
            settings.pricing.page_estimates.get(primary.model)
            if settings.billing is Billing.GO
            else None
        )
        checks.append(
            QuotaCheck(
                role=role,
                model=primary.model,
                weight=primary.quota_weight,
                configured=primary.quota_per_window,
                page=page,
                billing=settings.billing,
            )
        )
    return tuple(checks)


def render_quota_checks(checks: Sequence[QuotaCheck], prices: PriceTable) -> str:
    """La tabla de comparación y la lista de discrepancias."""
    billing = checks[0].billing if checks else Billing.GO
    lines = [
        f"# Cuota declarada frente a la página (por 5 h) — {PAGE_LABEL}",
        "",
        f"precios y estimados de la página al {prices.as_of(billing)}. No se corrige la "
        "configuración: solo se listan las diferencias.",
        "",
    ]
    if billing is Billing.PAYG:
        lines += [
            "Pago por uso: Zen no publica límite de peticiones, así que no hay estimado con el "
            "que comparar y la cuota declarada es un centinela (`ZEN_UNPUBLISHED_QUOTA`), "
            "no una medición.",
            "",
        ]
    lines += [
        "| rol | modelo | declarada | efectiva (declarada / peso) | página | estado |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    disagreements: list[str] = []
    for check in checks:
        if check.page is None:
            state = "no publicado por Zen" if check.billing is Billing.PAYG else "sin estimado"
        elif check.matches:
            state = "coincide"
        else:
            state = f"DISCREPA x{check.configured / check.page:.2f}"
            disagreements.append(
                f"- DISCREPA {check.role.value} `{check.model}`: declarada {check.configured} "
                f"(efectiva {check.effective:.0f} con peso {check.weight:g}) frente a "
                f"{check.page} de la página"
            )
        page = "—" if check.page is None else str(check.page)
        lines.append(
            f"| {check.role.value} | `{check.model}` | {check.configured} | "
            f"{check.effective:.0f} | {page} | {state} |"
        )
    lines += ["", f"Discrepancias: {len(disagreements)}", *disagreements, ""]
    return "\n".join(lines)


# ──────────────────────────────────────────── Comando ─────────────────────────────────────────────


def _resolve_billing(recorded: Billing | None, flag: Billing | None) -> Billing:
    """La forma de pago de la corrida: la que dejó escrita, o la que se diga si no la dejó.

    Suponer `go` sobre una corrida que no lo registró sería inventar una cifra de cuota. Y si
    la corrida dice una cosa y la bandera otra, no se elige en silencio.
    """
    if recorded is None and flag is None:
        raise AuditError(
            "meta.json no registra la facturación de esta corrida: pásala con --billing go|payg"
        )
    if recorded is not None and flag is not None and recorded is not flag:
        raise AuditError(f"meta.json dice {recorded.value} y --billing {flag.value} lo contradice")
    resolved = recorded if recorded is not None else flag
    assert resolved is not None
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada: `python -m crypto_agents.consumption <directorio>` o `--quotas`."""
    parser = argparse.ArgumentParser(
        prog="python -m crypto_agents.consumption",
        description="Consumo del pool de una corrida, o cuota declarada frente a la página.",
    )
    parser.add_argument("directory", nargs="?", type=Path, help="directorio de la corrida")
    parser.add_argument("--billing", type=Billing, choices=list(Billing), default=None)
    parser.add_argument(
        "--quotas",
        action="store_true",
        help="compara `quota_per_window` con el estimado de la página; lee la configuración",
    )
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    args = parser.parse_args(argv)

    if args.quotas:
        try:
            settings = load_settings(args.env_file)
        except ConfigError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        print(render_quota_checks(quota_checks(settings), settings.pricing))
        return 0

    if args.directory is None:
        parser.error("indica el directorio de una corrida, o usa --quotas")

    try:
        chain = run_chain(args.directory)
        billing = _resolve_billing(chain[0].meta.billing, args.billing)
    except AuditError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    arms: dict[str, list[LLMCall]] = {}
    inputs: list[tuple[str, str]] = []
    for run in chain:
        for arm in run.arms:
            calls = arms.setdefault(arm.arm, [])
            calls.extend(call for record in arm.records for call in record.calls)
            if arm.sha256 is not None:
                inputs.append((f"{run.path.name}/{arm.path.name}", arm.sha256))
    print(render_consumption(arms, DEFAULT_PRICING, billing, str(args.directory), inputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
