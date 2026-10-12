"""Recálculo de dispersión de la enmienda 3: la desviación medida de la diferencia pareada.

La enmienda 3 publicó una tabla de potencia con una correlación barrida entre 0 y 0.5 y dejó
dicho que era una cota: con el stop común, dos brazos que proponen la misma acción en una
activación rinden exactamente lo mismo en ella, así que la dispersión de la diferencia pareada
depende de dónde discrepan y no de un rho supuesto. Prometió recalcularla tras la etapa 1 con lo
medido, para decidir el n de cualquier etapa posterior. Este módulo lee el directorio de una
corrida y dice, par a par, cuánto vale esa desviación (`sigma_D`), qué n pide, si ese n cabe en
el pool de activaciones y qué costaría. No llama a nadie.

    python -m crypto_agents.paired_dispersion <directorio> --delta 0.002

Cinco decisiones que lo mantienen honesto:

- **No resta, no puntúa y no reparte por su cuenta.** Los dos vectores salen de
  `criteria.paired_returns`, la resta de `metrics.paired_differences`, el acuerdo de
  `metrics.agreement_counts` y los tramos del pool de `selection.stratify`. El error estándar
  contra el que se comprueba `sigma_D/√n` es el que calcula `criteria.evaluate_criteria`: si los
  dos no coinciden a la precisión impresa, la fila lo dice y el comando sale con 1.
- **Ninguna media.** Ningún modelo de este módulo tiene campo de media y el reporte no imprime
  ninguna: el n depende de `sigma_D` y de δ, no de la diferencia observada. Calcular la potencia
  para esa diferencia sería potencia post hoc, que no dice nada.
- **Mide sobre la corrida, y lo dice.** `dispersion.py` midió sobre el pool para no mirar el
  resultado antes de fijar δ. Aquí δ ya está fijado y lo que se dimensiona es otra etapa; aun
  así `sigma_D` es una muestra de 140, y por eso va con su rango bootstrap.
- **El rango es un rango.** Percentiles 2.5 y 97.5 de `sigma_D` al remuestrear evaluaciones
  dentro de cada símbolo, con B y semilla declarados. No se rotula como intervalo de confianza.
- **Da n, coste y viabilidad; no decide.** Qué etapa correr no le toca a este reporte.

Este módulo no importa el router, el grafo, los nodos, el replay, la ablación, el contador de
cuota ni la caché. Un test de arquitectura lo fija.
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from pydantic import AwareDatetime, Field

from crypto_agents.activation_sweep import activated_bars, history_path
from crypto_agents.audit import (
    META_FILE,
    AuditError,
    PlanKind,
    RunKind,
    file_sha256,
    load_plan,
    resolve_horizon,
    run_chain,
)
from crypto_agents.consumption import Consumption, consume, format_cost
from crypto_agents.criteria import (
    BASELINE_ARMS,
    PERCENT_DIGITS,
    REFERENCE,
    CriteriaError,
    check_validity,
    evaluate_criteria,
    final_records,
    paired_returns,
    render_invalid,
)
from crypto_agents.dispersion import DEFAULT_POWER, DEFAULT_SAMPLE_SIZES, Z_INTERVAL, effect_for
from crypto_agents.metrics import (
    agreement_counts,
    decision_actions,
    nearest_rank,
    paired_differences,
)
from crypto_agents.selection import (
    HISTORY_DIR,
    SelectionError,
    confirm_activation,
    load_manifest,
    stratify,
)
from crypto_agents.settings import DEFAULT_PRICING
from crypto_agents.state import Billing, FrozenModel

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from crypto_agents.audit import RunDirectory
    from crypto_agents.criteria import PairedReturns, ValidityReport
    from crypto_agents.journal import EvaluationRecord
    from crypto_agents.selection import SelectionManifest
    from crypto_agents.settings import PriceTable
    from crypto_agents.state import LLMCall

__all__ = [
    "BOOTSTRAP_PERMILLE",
    "DEFAULT_RESAMPLES",
    "DEFAULT_SEED",
    "MODEL_ARMS",
    "PAIRS",
    "ArmCost",
    "Dispersion",
    "InvalidRunError",
    "MissingPair",
    "PairDispersion",
    "Pool",
    "PoolCell",
    "Recalculation",
    "RecalculationError",
    "SampleSize",
    "SigmaRange",
    "Sources",
    "StageRun",
    "WallClock",
    "arm_costs",
    "bootstrap_sigma",
    "build_report",
    "commit_label",
    "dispersion_of",
    "fit",
    "main",
    "n_for_half_width",
    "n_for_power",
    "paid_arms",
    "pair_dispersion",
    "pool_cells",
    "read_stage",
    "render_report",
    "round_up_to",
    "sigma_of",
    "stage_hours",
    "stage_usd",
    "wall_clock",
]

MODEL_ARMS = (REFERENCE, "solo")
"""Los dos brazos con modelo de la etapa 1. Son los que una etapa posterior volvería a pagar."""

PAIRS: tuple[tuple[str, str], ...] = (
    (REFERENCE, "solo"),
    *((arm, baseline) for arm in MODEL_ARMS for baseline in BASELINE_ARMS),
)
"""`full` contra `solo`, que es lo que la enmienda 3 promete, y cada uno contra cada línea base,
que es lo que la etapa 2 del criterio 8 compara."""

DEFAULT_RESAMPLES = 10_000
"""B del bootstrap. Con 10 000 los percentiles 2.5 y 97.5 son los rangos 250 y 9 750, enteros."""

DEFAULT_SEED = 20261011
"""Semilla del bootstrap: la fecha del bloque. Declarada, impresa en la cabecera y la misma para
todos los pares, cada uno con su propio generador derivado de ella."""

BOOTSTRAP_PERMILLE = (25, 975)
"""Percentiles 2.5 y 97.5, en tantos por mil para que el rango se calcule con enteros."""

_SIGMA_LABELS = ("extremo bajo", "puntual", "extremo alto")


class RecalculationError(RuntimeError):
    """La corrida no admite el recálculo: no es de un manifiesto, o una celda no cuadra."""


# ─────────────────────────────────────────── Modelos ──────────────────────────────────────────────


class SigmaRange(FrozenModel):
    """Entre qué valores cae `sigma_D` al remuestrear las evaluaciones. Un rango, no un IC."""

    low: float = Field(ge=0.0)
    high: float = Field(ge=0.0)


class SampleSize(FrozenModel):
    """El n que pide una desviación para un objetivo, tal cual y redondeado."""

    sigma: float = Field(ge=0.0)
    exact: float = Field(ge=0.0)
    """Sin redondear: lo que da la fórmula."""

    rounded: int = Field(ge=0)
    """Hacia arriba, al múltiplo de las celdas del manifiesto (símbolos por tramos)."""


class Dispersion(FrozenModel):
    """`sigma_D` de un vector de diferencias y los n que pide.

    **No tiene campo de media, a propósito**, igual que `dispersion.ReturnDispersion`: lo que no
    está en el modelo no se imprime por descuido, y ningún n de aquí depende de ella.
    """

    n: int = Field(ge=2)
    sigma: float = Field(ge=0.0)
    """Desviación estándar muestral (n - 1) de las diferencias por evaluación."""

    bootstrap: SigmaRange
    half_width_n: tuple[SampleSize, SampleSize, SampleSize]
    """n con el que el semiancho del IC95% queda en δ o menos: con el extremo bajo del rango, con
    `sigma` y con el extremo alto."""

    power_n: tuple[SampleSize, SampleSize, SampleSize]
    """n con el que una diferencia real de δ deja el IC entero sobre 0 con la potencia dada."""

    @property
    def stderr(self) -> float:
        """`sigma/√n`: lo que `criteria` imprime como EE, si el vector es el mismo."""
        return self.sigma / math.sqrt(self.n)


class PairDispersion(FrozenModel):
    """Un brazo contra otro: cuánto coinciden y cuánto se dispersa su diferencia."""

    arm: str
    reference: str
    agreed: int = Field(ge=0)
    """Evaluaciones con la misma acción: buy, sell, hold o «no decidió» en los dos."""

    nonzero: int = Field(ge=0)
    """Evaluaciones cuya diferencia no es exactamente 0."""

    nonzero_agreed: int = Field(ge=0)
    """De esas, las que tienen la misma acción en los dos brazos. Con el stop común es 0: otro
    número desmiente la premisa de la enmienda 3 y el comando sale con 1."""

    dispersion: Dispersion
    criteria_stderr: float = Field(ge=0.0)
    """El EE que `criteria.evaluate_criteria` calcula para este par."""

    @property
    def n(self) -> int:
        """Evaluaciones emparejadas."""
        return self.dispersion.n

    @property
    def disagreed(self) -> int:
        """Evaluaciones en que los dos brazos no toman la misma acción."""
        return self.n - self.agreed

    @property
    def reproduces(self) -> bool:
        """Si `sigma_D/√n` es el EE de `criteria` a la precisión con que este se imprime."""
        return _pct(self.dispersion.stderr) == _pct(self.criteria_stderr)


class MissingPair(FrozenModel):
    """Un par que no se pudo calcular, con el porqué. No es un par con dispersión cero."""

    arm: str
    reference: str
    why: str


class PoolCell(FrozenModel):
    """Una celda (símbolo, tramo) del pool de activaciones."""

    symbol: str
    stratum: int = Field(ge=0)
    candidates: int = Field(ge=0)
    """Activaciones del barrido con el horizonte por delante: lo que `stratify` reparte."""

    confirmed: int = Field(ge=0)
    """De esas, las que el gate abre también con la ventana del replay. Son las que una selección
    puede entregar."""

    used: int = Field(ge=0)
    """Las que ya están en el manifiesto de la corrida."""

    @property
    def fresh(self) -> int:
        """Confirmadas que la corrida no usó."""
        return self.confirmed - self.used


class Pool(FrozenModel):
    """El pool de activaciones del histórico del manifiesto, celda a celda."""

    activations: int = Field(ge=0)
    """Activaciones del barrido, tengan o no horizonte por delante."""

    cells: tuple[PoolCell, ...] = Field(min_length=1)

    def smallest(self, fresh: bool) -> PoolCell:
        """La celda con menos sitio: contando las de la corrida, o solo las que no usó."""
        return min(self.cells, key=lambda cell: cell.fresh if fresh else cell.confirmed)

    def capacity(self, fresh: bool) -> int:
        """Activaciones que admite cada celda sin dejar ninguna corta."""
        cell = self.smallest(fresh)
        return cell.fresh if fresh else cell.confirmed


class ArmCost(FrozenModel):
    """Lo que un brazo costó en la corrida, según `consumption`."""

    arm: str
    evaluations: int = Field(ge=0)
    consumption: Consumption
    upper_usd: float | None = Field(default=None, ge=0.0)
    """Cota superior de lo medido, o `None` si hay llamadas sin tokens o sin precio: ahí
    `consumption` no da cota y un coste por evaluación sería inventado."""

    @property
    def per_evaluation(self) -> float | None:
        """La cota superior entre las evaluaciones del brazo. Un brazo sin llamadas vale 0."""
        if self.upper_usd is None or self.evaluations == 0:
            return None
        return self.upper_usd / self.evaluations


class WallClock(FrozenModel):
    """Pared de un brazo, entre su primera y su última `LLMCall.at` viva."""

    arm: str
    evaluations: int = Field(ge=0)
    live_calls: int = Field(ge=0)
    first: AwareDatetime | None = None
    last: AwareDatetime | None = None

    @property
    def seconds(self) -> float | None:
        """Segundos entre las dos marcas, o `None` con menos de dos llamadas vivas."""
        if self.first is None or self.last is None or self.live_calls < 2:
            return None
        return (self.last - self.first).total_seconds()

    @property
    def per_evaluation(self) -> float | None:
        """Segundos de pared por evaluación del brazo."""
        if self.seconds is None or self.evaluations == 0:
            return None
        return self.seconds / self.evaluations


class Recalculation(FrozenModel):
    """Todo lo que el reporte sabe. Ninguna media entre estos campos."""

    delta: float = Field(gt=0.0)
    power: float = Field(gt=0.0, lt=1.0)
    horizon: int = Field(gt=0)
    symbols: int = Field(gt=0)
    strata: int = Field(gt=0)
    stage_n: int = Field(gt=0)
    """Evaluaciones del manifiesto de la corrida: lo que una etapa posterior puede reutilizar."""

    resamples: int = Field(gt=0)
    seed: int
    pairs: tuple[PairDispersion, ...]
    missing: tuple[MissingPair, ...] = ()
    pool: Pool
    costs: tuple[ArmCost, ...] = ()
    cost_note: str | None = None
    """Por qué no hay coste, cuando no lo hay."""

    prices_as_of: date | None = None
    billing: Billing | None = None
    clocks: tuple[WallClock, ...] = ()
    clock_note: str | None = None
    """Por qué no hay pared, cuando no la hay."""

    @property
    def unit(self) -> int:
        """Celdas del manifiesto: a su múltiplo se redondea todo n."""
        return self.symbols * self.strata

    @property
    def consistent(self) -> bool:
        """Si cada par reproduce el EE de `criteria` y ninguna diferencia desmiente la premisa.

        Un par que no se pudo calcular —un brazo que la corrida no trae— no cuenta en contra:
        tiene su fila con el porqué. Sin ningún par no hay nada que dar por bueno.
        """
        return bool(self.pairs) and all(
            pair.reproduces and pair.nonzero_agreed == 0 for pair in self.pairs
        )


class Sources(NamedTuple):
    """De dónde sale cada cifra: lo que la cabecera cita."""

    directory: str
    command: str
    commit: str
    plan: str
    journals: tuple[tuple[str, str], ...]
    """Nombre y sha-256 de cada journal."""

    histories: tuple[tuple[str, str, str], ...]
    """Símbolo, ruta y sha-256 de cada histórico."""


# ─────────────────────────────────────────── Cálculo ──────────────────────────────────────────────


def sigma_of(differences: Sequence[float]) -> float:
    """Desviación estándar muestral (entre n - 1) de las diferencias por evaluación.

    Muestral porque es la de `metrics.paired_difference`: el error estándar que `criteria`
    publica es esta cifra entre √n, y dividir entre n daría otro.
    """
    if len(differences) < 2:
        raise ValueError(f"hacen falta al menos dos diferencias: {len(differences)}")
    return statistics.stdev(differences)


def _require(sigma: float, delta: float) -> None:
    if sigma < 0.0 or not math.isfinite(sigma):
        raise ValueError(f"la desviación tiene que ser finita y no negativa: {sigma}")
    if delta <= 0.0 or not math.isfinite(delta):
        raise ValueError(f"delta tiene que ser positivo: {delta}")


def n_for_half_width(sigma: float, delta: float) -> float:
    """n desde el que el semiancho del IC95% queda en δ o menos: `(1.96·sigma/δ)²`.

    Es el n del criterio 7 de la enmienda 2 con la desviación medida en lugar de la cota. Con ese
    n «no concluyente» deja de estar al alcance: el intervalo, si incluye 0, ya es un empate.
    """
    _require(sigma, delta)
    return (Z_INTERVAL * sigma / delta) ** 2


def n_for_power(sigma: float, delta: float, power: float = DEFAULT_POWER) -> float:
    """n con el que una diferencia real de δ deja el IC95% entero sobre 0 con esa potencia.

    `((1.96 + z_power)·sigma/δ)²`, con `z_power` el cuantil normal de la potencia, el mismo que
    usa `dispersion.effect_for`. Es la pregunta de diseño —qué n hace detectable un efecto de δ—
    y no la potencia para la diferencia observada.
    """
    _require(sigma, delta)
    if not 0.0 < power < 1.0:
        raise ValueError(f"la potencia está en (0, 1): {power}")
    return ((Z_INTERVAL + statistics.NormalDist().inv_cdf(power)) * sigma / delta) ** 2


def round_up_to(n: float, unit: int) -> int:
    """`n` redondeado hacia arriba al múltiplo de `unit`. Un múltiplo exacto se queda como está."""
    if unit <= 0:
        raise ValueError(f"el múltiplo tiene que ser positivo: {unit}")
    if n < 0.0 or not math.isfinite(n):
        raise ValueError(f"n tiene que ser finito y no negativo: {n}")
    return math.ceil(n / unit) * unit


def bootstrap_sigma(
    differences: Sequence[float],
    symbols: Sequence[str],
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    label: str = "",
) -> SigmaRange:
    """Rango de `sigma_D` al remuestrear las evaluaciones, percentiles 2.5 y 97.5.

    Se remuestrean pares —la evaluación entera, que aquí es su diferencia—, con reemplazo y
    **dentro de cada símbolo**: el manifiesto fija cuántas evaluaciones aporta cada uno, y una
    etapa posterior también, así que cada réplica conserva ese reparto. Un remuestreo simple lo
    dejaría variar y metería en el rango una dispersión que el diseño no tiene.

    `label` separa los generadores: cada par remuestrea con el suyo, derivado de la semilla, y
    añadir o quitar un par no mueve el rango de los demás. La semilla es una cadena, que
    `random.Random` siembra de forma estable entre ejecuciones.
    """
    if len(differences) != len(symbols):
        raise ValueError(f"{len(differences)} diferencias y {len(symbols)} símbolos: no cuadran")
    if resamples <= 0:
        raise ValueError(f"hace falta al menos un remuestreo: {resamples}")
    groups: dict[str, list[float]] = {}
    for value, symbol in zip(differences, symbols, strict=True):
        groups.setdefault(symbol, []).append(value)
    ordered = [groups[symbol] for symbol in sorted(groups)]

    rng = random.Random(f"{seed}|{label}")
    sigmas: list[float] = []
    for _ in range(resamples):
        sample: list[float] = []
        for group in ordered:
            sample.extend(rng.choices(group, k=len(group)))
        sigmas.append(sigma_of(sample))
    low, high = (nearest_rank(sigmas, permille, per=1000) for permille in BOOTSTRAP_PERMILLE)
    return SigmaRange(low=low, high=high)


def _sizes(
    n_for: Callable[[float], float], sigmas: tuple[float, float, float], unit: int
) -> tuple[SampleSize, SampleSize, SampleSize]:
    low, point, high = (
        SampleSize(sigma=sigma, exact=n_for(sigma), rounded=round_up_to(n_for(sigma), unit))
        for sigma in sigmas
    )
    return low, point, high


def dispersion_of(
    differences: Sequence[float],
    symbols: Sequence[str],
    delta: float,
    unit: int,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    label: str = "",
    power: float = DEFAULT_POWER,
) -> Dispersion:
    """`sigma_D` de un vector de diferencias, su rango bootstrap y los n que piden los tres.

    Pura sobre el vector: sumar una constante a todas las diferencias no cambia nada de lo que
    devuelve, porque nada de aquí lee la media.
    """
    sigma = sigma_of(differences)
    spread = bootstrap_sigma(differences, symbols, resamples, seed, label)
    sigmas = (spread.low, sigma, spread.high)
    return Dispersion(
        n=len(differences),
        sigma=sigma,
        bootstrap=spread,
        half_width_n=_sizes(lambda value: n_for_half_width(value, delta), sigmas, unit),
        power_n=_sizes(lambda value: n_for_power(value, delta, power), sigmas, unit),
    )


def pair_dispersion(
    arm: str,
    reference: str,
    pair: PairedReturns,
    criteria_stderr: float,
    delta: float,
    unit: int,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    power: float = DEFAULT_POWER,
) -> PairDispersion:
    """El acuerdo de acción y la dispersión de un par ya alineado por `criteria`."""
    differences = paired_differences(pair.arm_returns, pair.reference_returns)
    ours = decision_actions(pair.arm_records)
    theirs = decision_actions(pair.reference_records)
    agreed = agreement_counts(ours, theirs)
    assert agreed is not None  # los dos vectores vienen alineados y no vacíos
    same = [a == b for a, b in zip(ours, theirs, strict=True)]
    return PairDispersion(
        arm=arm,
        reference=reference,
        agreed=agreed.numerator,
        nonzero=sum(1 for value in differences if value != 0.0),
        nonzero_agreed=sum(
            1 for value, equal in zip(differences, same, strict=True) if equal and value != 0.0
        ),
        dispersion=dispersion_of(
            differences,
            [record.symbol for record in pair.arm_records],
            delta,
            unit,
            resamples,
            seed,
            f"{arm}|{reference}",
            power,
        ),
        criteria_stderr=criteria_stderr,
    )


def pool_cells(
    manifest: SelectionManifest,
    histories: Mapping[str, Sequence[Sequence[float]]],
    progress: bool = False,
) -> Pool:
    """El pool de activaciones de cada serie del manifiesto, repartido en sus celdas.

    Las celdas son las de `selection.stratify`, la misma función que reparte al seleccionar, y
    cada candidata se confirma con `selection.confirm_activation`: el barrido calcula los
    indicadores sobre la serie entera y la evaluación sobre la ventana del replay, así que una
    candidata puede no abrir el gate allí y entonces ninguna selección la entrega.

    Una entrada del manifiesto que no esté en la celda que su `stratum` dice es un error: el
    histórico o el reparto ya no son los de la selección, y contar con ellos mediría otro pool.
    """
    used: dict[tuple[str, int], set[int]] = {}
    for entry in manifest.entries:
        used.setdefault((entry.symbol, entry.stratum), set()).add(entry.index)

    cells: list[PoolCell] = []
    activations = 0
    for ref in manifest.series:
        rows = histories[ref.symbol]
        found = activated_bars(rows, ref.symbol, manifest.timeframe)
        activations += len(found)
        buckets = stratify(rows, found, manifest.strata, manifest.horizon)
        for stratum, bucket in enumerate(buckets):
            taken = used.get((ref.symbol, stratum), set())
            stray = sorted(taken - {bar.index for bar in bucket})
            if stray:
                raise RecalculationError(
                    f"{ref.symbol} tramo {stratum}: las velas {stray} del manifiesto no están en "
                    "esa celda del pool; el histórico o el reparto no son los de la selección"
                )
            cells.append(
                PoolCell(
                    symbol=ref.symbol,
                    stratum=stratum,
                    candidates=len(bucket),
                    confirmed=sum(
                        1
                        for bar in bucket
                        if confirm_activation(rows, bar.index, manifest.candle_limit) is not None
                    ),
                    used=len(taken),
                )
            )
        if progress:
            print(f"{ref.symbol:10} {len(found):>5} activaciones", file=sys.stderr)
    return Pool(activations=activations, cells=tuple(cells))


def arm_costs(
    calls: Mapping[str, Sequence[LLMCall]],
    evaluations: Mapping[str, int],
    prices: PriceTable,
    billing: Billing,
) -> tuple[ArmCost, ...]:
    """Lo que cada brazo costó, medido por `consumption.consume`, con sus evaluaciones al lado.

    La cifra que se escala es la cota superior (`ceiling_usd`) y no el coste medido: donde el
    proveedor no informó los tokens cacheados el medido es una cota inferior, y presupuestar
    otra etapa con una cota inferior es quedarse corto a sabiendas.
    """
    costs: list[ArmCost] = []
    for arm, count in evaluations.items():
        spent = consume(calls.get(arm, ()), prices, billing)
        costs.append(
            ArmCost(arm=arm, evaluations=count, consumption=spent, upper_usd=spent.ceiling_usd)
        )
    return tuple(costs)


def wall_clock(arm: str, records: Sequence[EvaluationRecord]) -> WallClock:
    """Primera y última `LLMCall.at` viva de un brazo.

    Solo las vivas: un acierto de caché lleva en `at` el instante evaluado y no el de pared.
    `at` es el final de cada llamada, así que a la pared le falta la latencia de la primera.
    """
    marks = [call.at for record in records for call in record.calls if not call.cache_hit]
    return WallClock(
        arm=arm,
        evaluations=len(records),
        live_calls=len(marks),
        first=min(marks) if marks else None,
        last=max(marks) if marks else None,
    )


def stage_usd(costs: Sequence[ArmCost], arms: Sequence[str], n: int) -> float | None:
    """`n` por el coste por evaluación de esos brazos, o `None` si alguno no lo tiene.

    Un brazo que no llama a ningún modelo aporta 0: lo dice `consumption`, no una regla de aquí.
    """
    by_arm = {cost.arm: cost for cost in costs}
    spent = 0.0
    for arm in arms:
        cost = by_arm.get(arm)
        if cost is None or cost.per_evaluation is None:
            return None
        spent += cost.per_evaluation * n
    return spent


def stage_hours(clocks: Sequence[WallClock], arms: Sequence[str], n: int) -> float | None:
    """Horas de pared de `n` evaluaciones de esos brazos, uno detrás de otro.

    Un brazo sin llamadas vivas no tiene pared que medir con `LLMCall.at` y aquí cuenta 0; uno
    con una sola llamada deja la cuenta sin determinar.
    """
    by_arm = {clock.arm: clock for clock in clocks}
    seconds = 0.0
    for arm in arms:
        clock = by_arm.get(arm)
        if clock is None:
            return None
        if clock.live_calls == 0:
            continue
        if clock.per_evaluation is None:
            return None
        seconds += clock.per_evaluation * n
    return seconds / 3600.0


def build_report(
    arms: Mapping[str, Sequence[EvaluationRecord]],
    histories: Mapping[str, Sequence[Sequence[float]]],
    horizon: int,
    delta: float,
    manifest: SelectionManifest,
    pool: Pool,
    calls: Mapping[str, Sequence[LLMCall]],
    prices: PriceTable,
    billing: Billing | None,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    power: float = DEFAULT_POWER,
    resumed: bool = False,
) -> Recalculation:
    """El recálculo de una corrida ya leída. Pura sobre sus argumentos.

    `calls` son las llamadas de cada brazo sobre toda la cadena de reanudación —lo que se pagó—
    y `arms` los registros del último eslabón, que es lo que `criteria` juzga. El pool llega
    calculado: es lo único lento, y no depende de la corrida sino del manifiesto.
    """
    unit = len(manifest.series) * manifest.strata
    criteria = evaluate_criteria(arms, histories, horizon, delta)
    published = {
        (item.arm, item.reference): item
        for item in (
            *criteria.criterion_1,
            *(row for table in criteria.criterion_3 for row in table.comparisons),
        )
    }

    pairs: list[PairDispersion] = []
    missing: list[MissingPair] = []
    for arm, reference in PAIRS:
        aligned = paired_returns(arm, reference, arms, histories, horizon)
        comparison = published.get((arm, reference))
        if isinstance(aligned, str):
            missing.append(MissingPair(arm=arm, reference=reference, why=aligned))
        elif comparison is None or comparison.diff is None:
            why = "criteria no lo compara" if comparison is None else str(comparison.why)
            missing.append(MissingPair(arm=arm, reference=reference, why=why))
        else:
            pairs.append(
                pair_dispersion(
                    arm,
                    reference,
                    aligned,
                    comparison.diff.stderr,
                    delta,
                    unit,
                    resamples,
                    seed,
                    power,
                )
            )

    evaluations = {name: len(records) for name, records in arms.items()}
    return Recalculation(
        delta=delta,
        power=power,
        horizon=horizon,
        symbols=len(manifest.series),
        strata=manifest.strata,
        stage_n=len(manifest.entries),
        resamples=resamples,
        seed=seed,
        pairs=tuple(pairs),
        missing=tuple(missing),
        pool=pool,
        costs=() if billing is None else arm_costs(calls, evaluations, prices, billing),
        cost_note=(
            f"{META_FILE} no registra la facturación de la corrida" if billing is None else None
        ),
        prices_as_of=None if billing is None else prices.as_of(billing),
        billing=billing,
        clocks=()
        if resumed
        else tuple(wall_clock(name, records) for name, records in arms.items()),
        clock_note=(
            "corrida reanudada: sus pasadas no son una sola pared que medir" if resumed else None
        ),
    )


# ─────────────────────────────────────────── Reporte ──────────────────────────────────────────────


def _pct(value: float, digits: int = PERCENT_DIGITS) -> str:
    """Porcentaje con los decimales de `criteria`, que es con lo que se compara."""
    return f"{value:.{digits}%}"


def _usd(value: float | None) -> str:
    return "no determinado" if value is None else f"{value:.2f}"


def _hours(value: float | None) -> str:
    return "no determinado" if value is None else f"{value:.1f}"


def fit(per_cell: int, capacity: int) -> str:
    """Si caben `per_cell` activaciones en la celda más pequeña, con los dos números delante."""
    if per_cell <= capacity:
        return f"cabe ({per_cell} ≤ {capacity})"
    return f"no cabe ({per_cell} > {capacity})"


def paid_arms(arm: str, reference: str) -> tuple[str, ...]:
    """Los brazos del par que una etapa posterior tendría que pagar: los que llevan modelo."""
    return tuple(name for name in (arm, reference) if name in MODEL_ARMS)


def _reproduce(command: str, also: str = "") -> list[str]:
    return ["", f"Reproduce: `{command}`{also}", ""]


def _dispersion_table(report: Recalculation) -> list[str]:
    lines = [
        "| brazo | referencia | n | acuerdo de acción | discrepan | diferencias ≠ 0 "
        "| ≠ 0 con la misma acción | sigma_D | rango bootstrap de sigma_D | sigma_D/√n "
        "| EE de criteria | reproduce |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for pair in report.pairs:
        found = pair.dispersion
        lines.append(
            f"| {pair.arm} | {pair.reference} | {pair.n} | {pair.agreed}/{pair.n} "
            f"({pair.agreed / pair.n:.0%}) | {pair.disagreed} | {pair.nonzero} "
            f"| {pair.nonzero_agreed} | {_pct(found.sigma)} "
            f"| [{_pct(found.bootstrap.low)}, {_pct(found.bootstrap.high)}] "
            f"| {_pct(found.stderr)} | {_pct(pair.criteria_stderr)} "
            f"| {'sí' if pair.reproduces else '**NO**'} |"
        )
    lines += [
        f"| {item.arm} | {item.reference} | — | — | — | — | — | — | — | — | — "
        f"| no determinado: {item.why} |"
        for item in report.missing
    ]
    return lines


def _size_table(report: Recalculation) -> list[str]:
    unit = report.unit
    lines = [
        "| brazo | referencia | sigma_D | valor | n para semiancho ≤ δ "
        f"| al múltiplo de {unit} | n para potencia {report.power:.0%} | al múltiplo de {unit} |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for pair in report.pairs:
        found = pair.dispersion
        for label, tie, power in zip(_SIGMA_LABELS, found.half_width_n, found.power_n, strict=True):
            lines.append(
                f"| {pair.arm} | {pair.reference} | {label} | {_pct(tie.sigma)} "
                f"| {tie.exact:.1f} | {tie.rounded} | {power.exact:.1f} | {power.rounded} |"
            )
    return lines


def _effect_table(report: Recalculation, sample_sizes: Sequence[int]) -> list[str]:
    lines = [
        "| brazo | referencia | n | sigma_dif | semiancho IC95% | detectable |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for pair in report.pairs:
        for n in sample_sizes:
            half_width, detectable = effect_for(pair.dispersion.sigma, n, report.power)
            lines.append(
                f"| {pair.arm} | {pair.reference} | {n} | {pair.dispersion.sigma:.2%} "
                f"| {half_width:.2%} | {detectable:.2%} |"
            )
    return lines


def _stage_table(report: Recalculation) -> list[str]:
    reusing = report.pool.capacity(fresh=False)
    fresh = report.pool.capacity(fresh=True)
    lines = [
        "| brazo | referencia | objetivo | sigma_D | n | por celda "
        f"| reutilizando las {report.stage_n} | sin ellas | n adicional | USD del n "
        "| USD del adicional | horas de pared del n |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for pair in report.pairs:
        found = pair.dispersion
        paid = paid_arms(pair.arm, pair.reference)
        targets = (
            ("semiancho ≤ δ", found.half_width_n),
            (f"potencia {report.power:.0%}", found.power_n),
        )
        for target, sizes in targets:
            for label, size in zip(_SIGMA_LABELS, sizes, strict=True):
                per_cell = size.rounded // report.unit
                extra = max(0, size.rounded - report.stage_n)
                lines.append(
                    f"| {pair.arm} | {pair.reference} | {target} | {label} | {size.rounded} "
                    f"| {per_cell} | {fit(per_cell, reusing)} | {fit(per_cell, fresh)} "
                    f"| {extra} | {_usd(stage_usd(report.costs, paid, size.rounded))} "
                    f"| {_usd(stage_usd(report.costs, paid, extra))} "
                    f"| {_hours(stage_hours(report.clocks, paid, size.rounded))} |"
                )
    return lines


def _pool_table(report: Recalculation) -> list[str]:
    lines = [
        "| símbolo | tramo | candidatas | confirmadas | en la corrida | libres |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines += [
        f"| {cell.symbol} | {cell.stratum} | {cell.candidates} | {cell.confirmed} "
        f"| {cell.used} | {cell.fresh} |"
        for cell in report.pool.cells
    ]
    return lines


def _cost_table(report: Recalculation) -> list[str]:
    if report.cost_note is not None:
        return [f"No determinado: {report.cost_note}."]
    lines = [
        "| brazo | evaluaciones | llamadas | sin medir | coste medido USD | cota superior USD "
        "| USD por evaluación |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cost in report.costs:
        spent = cost.consumption
        upper = "no determinado" if cost.upper_usd is None else f"{cost.upper_usd:.4f}"
        each = "no determinado" if cost.per_evaluation is None else f"{cost.per_evaluation:.5f}"
        lines.append(
            f"| {cost.arm} | {cost.evaluations} | {spent.calls} | {spent.unmeasured} "
            f"| {format_cost(spent)} | {upper} | {each} |"
        )
    return lines


def _clock_table(report: Recalculation) -> list[str]:
    if report.clock_note is not None:
        return [f"No determinado: {report.clock_note}."]
    lines = [
        "| brazo | evaluaciones | llamadas vivas | primera `LLMCall.at` | última `LLMCall.at` "
        "| pared (h) | s por evaluación |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for clock in report.clocks:
        if clock.seconds is None or clock.per_evaluation is None:
            why = "sin llamadas" if clock.live_calls == 0 else "una sola llamada"
            lines.append(
                f"| {clock.arm} | {clock.evaluations} | {clock.live_calls} | — | — "
                f"| no determinado: {why} | no determinado: {why} |"
            )
            continue
        assert clock.first is not None
        assert clock.last is not None
        lines.append(
            f"| {clock.arm} | {clock.evaluations} | {clock.live_calls} "
            f"| {clock.first.isoformat(timespec='seconds')} "
            f"| {clock.last.isoformat(timespec='seconds')} "
            f"| {clock.seconds / 3600.0:.2f} | {clock.per_evaluation:.1f} |"
        )
    return lines


def render_report(
    report: Recalculation,
    sources: Sources,
    sample_sizes: Sequence[int] = DEFAULT_SAMPLE_SIZES,
) -> str:
    """El recálculo en Markdown. Sin medias, sin veredicto y sin decir qué etapa correr."""
    command = sources.command
    unit = report.unit
    smallest = report.pool.smallest(fresh=False)
    smallest_fresh = report.pool.smallest(fresh=True)
    candidates = sum(cell.candidates for cell in report.pool.cells)
    confirmed = sum(cell.confirmed for cell in report.pool.cells)
    criteria_command = f"python -m crypto_agents.criteria {sources.directory} --delta"
    criteria_command += f" {report.delta:g}"

    lines = [
        f"# Recálculo de dispersión (enmienda 3) de la corrida `{sources.directory}`",
        "",
        f"Reproduce: `{command}`",
        "",
        f"- commit de la corrida: {sources.commit}",
        f"- plan: {sources.plan}",
        *(f"- `{name}` sha-256 `{digest}`" for name, digest in sources.journals),
        *(
            f"- histórico `{symbol}`: `{path}`, sha-256 `{digest}`"
            for symbol, path, digest in sources.histories
        ),
        f"- δ = {report.delta:g} ({report.delta:.3%} de retorno por evaluación) · horizonte "
        f"{report.horizon} velas · puntuación: stop común, retorno bruto · diferencias pareadas "
        "por `run_id`, las mismas que resta `criteria`",
        f"- todo n se redondea hacia arriba al múltiplo de {unit}: {report.symbols} símbolos por "
        f"{report.strata} tramos del manifiesto, que tiene {report.stage_n} evaluaciones",
        f"- bootstrap: {report.resamples} remuestreos de las evaluaciones, con reemplazo y dentro "
        f"de cada símbolo, semilla {report.seed}, percentiles 2.5 y 97.5 por rango más cercano",
        "",
        "## 1. Acuerdo de acción y dispersión de la diferencia pareada",
        "",
        "`sigma_D` es la desviación estándar muestral (n - 1) de la diferencia `brazo - "
        "referencia` por evaluación. El acuerdo cuenta como categorías buy, sell, hold y «no "
        "decidió». Con el stop común dos brazos con la misma acción rinden lo mismo, así que una "
        "diferencia solo puede ser distinta de 0 donde discrepan: la columna «≠ 0 con la misma "
        "acción» tiene que ser 0. `sigma_D/√n` tiene que ser el EE que imprime `criteria`.",
        "",
        "La enmienda 3 promete este recálculo para `full` frente a `solo`. Las filas contra las "
        "líneas base se añaden porque son las comparaciones del criterio 3, que es lo que la "
        "etapa 2 del criterio 8 mide.",
        "",
        *_dispersion_table(report),
        *_reproduce(command, f" (los EE: `{criteria_command}`)"),
        "## 2. n que pide cada par",
        "",
        "- **n para semiancho ≤ δ**: `(1.96·sigma_D/δ)²`. Desde ese n el intervalo, si incluye 0, "
        "es un «empate» y no un «no concluyente».",
        f"- **n para potencia {report.power:.0%}**: `((1.96 + z)·sigma_D/δ)²`, con z el cuantil "
        f"normal de {report.power:.0%}. Es el n con el que una diferencia real de δ daría "
        f"«justifica» con probabilidad {report.power:.0%}. No es la potencia para la diferencia "
        "observada, que no se calcula.",
        "- Cada n va con `sigma_D` puntual y con los dos extremos de su rango bootstrap.",
        "",
        *_size_table(report),
        *_reproduce(command),
        f"## 3. Semiancho y diferencia detectable con n = {', '.join(map(str, sample_sizes))}",
        "",
        "La tabla de `python -m crypto_agents.dispersion` que la enmienda 3 copió, con la "
        "`sigma_D` medida de cada par en lugar de una cota y un rho barrido: semiancho "
        "`1.96·sigma_dif/√n`, detectable `(1.96 + z)·sigma_dif/√n`.",
        "",
        *_effect_table(report, sample_sizes),
        *_reproduce(command),
        "## 4. Viabilidad y coste de cada n redondeado",
        "",
        f"- **por celda**: n entre las {unit} celdas (símbolo, tramo).",
        f"- **reutilizando las {report.stage_n}**: contra la celda más pequeña del pool, "
        f"{smallest.symbol} tramo {smallest.stratum}, con {smallest.confirmed} activaciones "
        "confirmadas.",
        f"- **sin ellas**: contra la celda con menos activaciones que la corrida no usó, "
        f"{smallest_fresh.symbol} tramo {smallest_fresh.stratum}, con {smallest_fresh.fresh}.",
        f"- **n adicional**: n menos las {report.stage_n} ya corridas.",
        "- **USD**: n por el coste por evaluación de los brazos con modelo del par (sección 6); "
        "las líneas base no llaman a ningún modelo. **Horas**: n por la pared por evaluación de "
        "esos brazos, uno detrás de otro (sección 7).",
        "",
        *_stage_table(report),
        *_reproduce(command),
        "## 5. Pool de activaciones por celda",
        "",
        f"{report.pool.activations} activaciones del barrido sobre {report.symbols} series; "
        f"{candidates} con el horizonte por delante, repartidas en tramos por "
        f"`selection.stratify`; {confirmed} que el gate abre también con la ventana del replay "
        "(`selection.confirm_activation`), que son las que una selección puede entregar. «En la "
        "corrida» son las entradas del manifiesto.",
        "",
        *_pool_table(report),
        *_reproduce(command),
        "## 6. Coste por evaluación medido en la corrida",
        "",
        "De `consumption.consume`. La cota superior cobra la entrada nueva a `cache_write` donde "
        "el modelo lo tiene y todo el prompt como nuevo donde el proveedor no informó "
        "`cached_tokens`. "
        + (
            f"Facturación {report.billing.value}, precios de la página al {report.prices_as_of}."
            if report.billing is not None
            else ""
        ),
        "",
        *_cost_table(report),
        *_reproduce(
            command, f" (el consumo: `python -m crypto_agents.consumption {sources.directory}`)"
        ),
        "## 7. Pared observada en la corrida",
        "",
        "Entre la primera y la última `LLMCall.at` viva de cada brazo, dividida entre sus "
        "evaluaciones. `at` es el final de cada llamada: falta la latencia de la primera. "
        "Observada en esta corrida y en la máquina que la corrió.",
        "",
        *_clock_table(report),
        *_reproduce(command),
        "## Lo que estas cifras suponen",
        "",
        f"- `sigma_D` sale de las mismas {report.stage_n} evaluaciones cuyo veredicto ya se "
        "leyó. Es una muestra: su rango dice cuánto se mueve, y el n se mueve con su cuadrado.",
        "- El rango es el de los percentiles 2.5 y 97.5 de `sigma_D` al remuestrear, no un "
        "intervalo de confianza al 95%. Con muchas diferencias exactamente 0 —las evaluaciones "
        "en que los dos brazos coinciden— y colas largas, ese rango se queda corto.",
        "- El error estándar `sigma_D/√n` supone evaluaciones independientes. Horizontes que se "
        "solapan y símbolos que se mueven juntos reducen el n efectivo: los n de arriba son "
        "optimistas en eso.",
        "- El coste y la pared son los de esta corrida por evaluación, escalados en línea recta "
        "con n: una estimación, no una cota de otra etapa.",
        "- «Cabe» exige que ninguna celda quede corta. Una selección con una celda corta se "
        "rellena desde otros tramos del mismo símbolo y deja de estar repartida en el tiempo.",
        "- Ninguna cifra de este reporte es una media de retorno.",
        "",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────── Comando ──────────────────────────────────────────────


def commit_label(commit: str | None, dirty: bool | None) -> str:
    if commit is None:
        return "no determinado: git no respondió al arrancar"
    state = {True: "con cambios sin comprometer", False: "árbol limpio", None: "estado desconocido"}
    return f"`{commit}` ({state[dirty]})"


class InvalidRunError(RecalculationError):
    """El criterio 6 invalida la corrida: no se interpreta, tampoco para dimensionar otra."""

    def __init__(self, validity: ValidityReport) -> None:
        super().__init__("la corrida es inválida según el criterio 6")
        self.validity = validity


class StageRun(NamedTuple):
    """Una corrida válida de un manifiesto, leída con lo que hace falta para dimensionar otra."""

    chain: list[RunDirectory]
    """La corrida y las que reanuda, de la más reciente a la más antigua."""

    arms: dict[str, tuple[EvaluationRecord, ...]]
    """Los registros que `criteria` juzga: los del último eslabón."""

    histories: dict[str, list[list[float]]]
    manifest: SelectionManifest
    plan_path: Path
    horizon: int
    calls: dict[str, list[LLMCall]]
    """Las llamadas de cada brazo sobre toda la cadena: lo que se pagó."""

    history_files: tuple[tuple[str, str, str], ...]
    """Símbolo, ruta y sha-256 de cada histórico."""


def read_stage(
    directory: Path, plan: Path | None, history_dir: Path, horizon: int | None
) -> StageRun:
    """Lee una corrida y comprueba que se puede dimensionar otra etapa con ella.

    Tres negativas, antes de leer una sola vela: un sondeo no es una corrida, sin `arm_roles` no
    se puede comprobar el criterio 6, y sin manifiesto no hay tramos ni celdas. Una corrida que
    el criterio 6 invalida levanta `InvalidRunError` con el motivo.

    Es lo que hacía `main` antes de calcular nada, sacado fuera para que quien mide otra cosa
    sobre la misma corrida —el efecto de diseño— entre por la misma puerta y no por una copia de
    estas negativas.
    """
    chain = run_chain(directory)
    meta = chain[0].meta
    probes = [item.path for item in chain if item.meta.kind is RunKind.PROBE]
    if probes:
        raise RecalculationError(
            f"{probes[0]} es un sondeo (kind=probe), no una corrida de la ablación"
        )
    if meta.arm_roles is None:
        raise RecalculationError(
            f"{META_FILE} no registra arm_roles: sin el backend de cada brazo no se puede "
            "comprobar que la corrida es válida"
        )
    if meta.plan_kind is not PlanKind.MANIFEST:
        raise RecalculationError(
            "el plan de la corrida no es un manifiesto: los tramos, las celdas y el múltiplo "
            "de n salen de él"
        )
    arms = final_records(chain)
    validity = check_validity(
        arms,
        {
            name: {role: entry.backend for role, entry in roles.items()}
            for name, roles in meta.arm_roles.items()
        },
    )
    if not validity.valid:
        raise InvalidRunError(validity)

    records = [record for entries in arms.values() for record in entries]
    histories, _ = load_plan(meta, plan, history_dir, records)
    plan_path = plan if plan is not None else Path(meta.plan_path)
    manifest = load_manifest(plan_path)
    calls: dict[str, list[LLMCall]] = {}
    for link in chain:
        for arm in link.arms:
            calls.setdefault(arm.arm, []).extend(
                call for record in arm.records for call in record.calls
            )
    return StageRun(
        chain=chain,
        arms=arms,
        histories=histories,
        manifest=manifest,
        plan_path=plan_path,
        horizon=resolve_horizon(meta.horizon, horizon),
        calls=calls,
        history_files=tuple(
            (
                ref.symbol,
                str(history_path(ref.symbol, ref.timeframe, history_dir)),
                file_sha256(history_path(ref.symbol, ref.timeframe, history_dir)),
            )
            for ref in manifest.series
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada: `python -m crypto_agents.paired_dispersion <directorio> --delta X`.

    Sale con 0 si el recálculo reproduce los EE de `criteria`, con 1 si alguno no se reproduce o
    la corrida es inválida, y con 2 si el directorio no se pudo leer.
    """
    parser = argparse.ArgumentParser(
        prog="python -m crypto_agents.paired_dispersion",
        description="Dispersión medida de la diferencia pareada de una corrida, y el n que pide.",
    )
    parser.add_argument("directory", type=Path, help="directorio de la corrida, con meta.json")
    parser.add_argument(
        "--delta",
        type=float,
        required=True,
        help="el δ de los criterios, en fracción de retorno por evaluación (0.002 = 0.2%%)",
    )
    parser.add_argument("--plan", type=Path, default=None, help="manifiesto del plan")
    parser.add_argument("--history-dir", type=Path, default=HISTORY_DIR)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    if args.delta <= 0:
        parser.error("--delta debe ser positivo")
    if args.resamples <= 0:
        parser.error("--resamples debe ser positivo")

    try:
        stage = read_stage(args.directory, args.plan, args.history_dir, args.horizon)
        run = stage.chain[0]
        meta = run.meta
        pool = pool_cells(stage.manifest, stage.histories, progress=True)
        report = build_report(
            stage.arms,
            stage.histories,
            stage.horizon,
            args.delta,
            stage.manifest,
            pool,
            stage.calls,
            DEFAULT_PRICING,
            meta.billing,
            args.resamples,
            args.seed,
            resumed=len(stage.chain) > 1,
        )
    except InvalidRunError as error:
        print(render_invalid(error.validity, str(args.directory)))
        return 1
    except (AuditError, SelectionError, CriteriaError, RecalculationError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    flags = [f"--delta {args.delta:g}"]
    if args.plan is not None:
        flags.append(f"--plan {args.plan}")
    if args.history_dir != HISTORY_DIR:
        flags.append(f"--history-dir {args.history_dir}")
    if args.horizon is not None:
        flags.append(f"--horizon {args.horizon}")
    if args.resamples != DEFAULT_RESAMPLES:
        flags.append(f"--resamples {args.resamples}")
    if args.seed != DEFAULT_SEED:
        flags.append(f"--seed {args.seed}")
    sources = Sources(
        directory=str(args.directory),
        command=f"python -m crypto_agents.paired_dispersion {args.directory} {' '.join(flags)}",
        commit=commit_label(meta.git_commit, meta.git_dirty),
        plan=f"`{stage.plan_path}` ({meta.plan_kind.value}), sha-256 `{meta.plan_sha256}`",
        journals=tuple((arm.path.name, arm.sha256) for arm in run.arms if arm.sha256 is not None),
        histories=stage.history_files,
    )
    print(render_report(report, sources))
    return 0 if report.consistent else 1


if __name__ == "__main__":
    raise SystemExit(main())
