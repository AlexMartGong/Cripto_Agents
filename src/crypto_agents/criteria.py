"""Evaluador mecánico de criterios de la ablación, y las guardas que invalidan una corrida.

Los criterios de la enmienda se evalúan por código y no a mano, y una corrida que incumple las
guardas se declara inválida por sí sola, sin que nadie tenga que mirar y decidir. Lee el
directorio de una corrida —un journal por brazo y su `meta.json`— y no llama a nadie.

    python -m crypto_agents.criteria <directorio>

**El intervalo es el de la enmienda 4.** `media ± 1.96 · EE agrupado`, con el error estándar
agrupado CR1 de `metrics.clustered_error` y un clúster por bloque UTC de `CLUSTER_DAYS` días del
instante evaluado, todos los símbolos juntos. El ingenuo (`stdev/√n`) supone evaluaciones
independientes, y el efecto de diseño midió que no lo son: se imprime al lado, con G y deff, y
no entra en ningún veredicto. Con menos de `MIN_CLUSTERS` clústeres no hay intervalo.

**δ es una constante.** `DELTA` es el δ de la enmienda 2, 0.20 % por evaluación, y el comando no
lo pide: `--delta` solo acepta ese valor —`--delta 0.2` es δ = 20 % y con él todo semiancho
«empata»— y `--exploratory-delta` deja probar otro, con cada veredicto marcado «δ no registrado».
La marca sale del valor, no del flag.

**Criterio 1.** `full` contra `solo`, `no_debate` y `bull_only`: diferencia pareada del retorno
por evaluación con el stop común, con su intervalo al 95%. El veredicto depende de `delta`, el
efecto por debajo del cual dos brazos se consideran iguales:

- el intervalo queda por encima de 0: *justifica*;
- queda por debajo de 0: *peor*;
- incluye 0 y su semiancho no supera `delta`: *empate*;
- incluye 0 y su semiancho **supera** `delta`: *no concluyente*. El intervalo incluye 0, pero es
  tan ancho que ni siquiera permite decir que los dos brazos son iguales.

**Criterio 3.** Cada brazo con modelo contra *cada una* de las cuatro líneas base, por separado. No
se calcula un «mejor de las líneas base»: «supera a k de 4» es una conjunción de cuatro
comparaciones, no un máximo.

**Criterio 6, la corrida es válida.** Cinco condiciones, y basta que falle una:

1. el decisor perdió evaluaciones por cuota. El único rastro es el `NodeError` con «cuota agotada»
   —no hay fila de `LLMCall` porque no se gastó nada—, que `metrics.undecided_causes` clasifica
   como `AbortKind.QUOTA` y de cuyo nodo se lee que es el decisor;
2. un acierto de caché con un backend que no es el que el brazo declara. Se compara el backend de
   cada `LLMCall` con el de `meta.json`, que lo registra por brazo y por rol;
3. un veto que no sea `invalid_stop_side`;
4. una evaluación perdida por saldo insuficiente (`AbortKind.INSUFFICIENT_FUNDS`, el `402` de Zen)
   en cualquier nodo: con pago por uso el saldo es el único tope, y una corrida a la que se le
   acabó no mide los modelos sino el crédito. Es la cuarta, añadida con el pago por uso; la
   enmienda 2 la lista desde el bloque T4, antes de la segunda corrida;
5. una evaluación perdida por un fallo del proveedor —un rechazo antes de producir contenido
   (`AbortKind.TRANSPORT`) o un plazo vencido (`AbortKind.TIMEOUT`)— en cualquier nodo y en
   cualquier brazo. `resolve()` no degrada por transporte, así que una ruta caída en un nodo
   técnico deja sin evidencia a todo brazo que lo use, y lo que esa corrida compara son los
   brazos que el proveedor dejó en pie. Añadida en el bloque T5, antes de la segunda corrida,
   tras 15 rechazos 404 seguidos de `deepseek-v4-flash`. Manda el primer fallo del registro
   (`metrics.undecided_causes`): una evaluación que ya había agotado sus intentos por validación
   en un nodo y además sufrió un rechazo en otro se cuenta por la validación, que es del modelo;
6. una evaluación cortada por el tope de gasto en USD (`AbortKind.SPEND_CAP`, el `--max-usd` de
   la corrida) en cualquier nodo y en cualquier brazo. Con pago por uso la cuota declarada ya no
   frena nada y el tope es el freno que se declara antes de llegar al saldo: lo que corta no se
   midió. Añadida en el bloque T7 y escrita en la enmienda 3, antes de la etapa 1.

En una corrida reanudada las seis se miran sobre el último eslabón de la cadena
(`final_records`): la reanudación recorre el plan entero, de modo que una evaluación perdida en la
primera pasada y decidida en la segunda no invalida, y una que sigue sin decidir al final sí. El
pico de consumo es lo único que suma la cadena entera, porque la ventana del proveedor es una.

Con la corrida inválida solo se imprime el motivo y el pico de consumo, ningún veredicto, y el
código de salida es 1. Si el directorio no se puede evaluar —no es una corrida, el plan no es el
suyo— el código es 2: no es lo mismo una corrida mala que una que no se pudo leer.

Este módulo no importa el router, el grafo, el replay ni la ablación, y no puntúa de otra forma
que `outcomes.score_run`, que no toca.
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import Field

from crypto_agents.audit import (
    META_FILE,
    AuditError,
    RunDirectory,
    RunKind,
    chain_calls,
    load_plan,
    resolve_horizon,
    run_chain,
)
from crypto_agents.funding import FUNDING_DIR, FundingError, funding_digests, load_funding
from crypto_agents.journal import EvaluationRecord
from crypto_agents.metrics import (
    MIN_CLUSTERS,
    MIN_PAIRED_N,
    NET_LABEL,
    AbortKind,
    ClusteredPair,
    NetPairs,
    PairedDifference,
    clustered_paired_difference,
    clusters_of,
    paired_net_difference,
    undecided_causes,
)
from crypto_agents.outcomes import NetInputs, Scoring, score_run
from crypto_agents.quota import LOCAL_BACKENDS
from crypto_agents.risk import INVALID_STOP_SIDE
from crypto_agents.selection import HISTORY_DIR, SelectionError
from crypto_agents.settings import DEFAULT_COSTS, QUOTA_NOT_APPLICABLE
from crypto_agents.state import AgentRole, Billing, FrozenModel

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Mapping, Sequence
    from uuid import UUID

    from crypto_agents.state import Backend, LLMCall

__all__ = [
    "AMENDMENT",
    "BASELINE_ARMS",
    "CLUSTER_DAYS",
    "DECIDER_NODES",
    "DELTA",
    "PEAK_WARNING_SHARE",
    "ArmBaselines",
    "ArmValidity",
    "Cell",
    "Comparison",
    "CriteriaError",
    "CriteriaReport",
    "Descriptive",
    "PairedReturns",
    "PeakWindow",
    "ValidityReport",
    "Verdict",
    "add_delta_arguments",
    "check_validity",
    "classify",
    "compare_returns",
    "delta_flag",
    "evaluate_criteria",
    "final_records",
    "main",
    "paired_returns",
    "peak_window_usage",
    "render_criteria",
    "render_invalid",
    "render_peaks",
    "render_validity",
    "resolve_delta",
    "unregistered_delta",
]

REFERENCE = "full"
CRITERION_1_REFERENCES = ("solo", "no_debate", "bull_only")
"""Contra qué se compara `full` en el criterio 1."""

BASELINE_ARMS = ("always_buy", "always_sell", "random_uniform", "rule_trend")
"""Las cuatro líneas base del criterio 3. Cada una se compara por separado."""

DECIDER_NODES = frozenset({"decide", "decide_single_desk", "decide_without_debate", "decide_solo"})

PROVIDER_FAILURES = (AbortKind.TRANSPORT, AbortKind.TIMEOUT)
"""Las dos formas en que el proveedor, y no el modelo, se lleva una evaluación.

El 402 no está aquí aunque su `failure_kind` sea `transport`: `undecided_causes` lo clasifica
aparte (`INSUFFICIENT_FUNDS`) y tiene su propia condición.
"""
"""Nodos de la ablación que gastan el rol decisor. Una prueba los ata a `_LLM_NODES`."""

PEAK_WARNING_SHARE = 0.80
"""Fracción de `quota_per_window` a partir de la cual el pico de consumo avisa."""

SCORING = Scoring.COMMON_STOP
"""Con qué se puntúa: el stop común, el mismo para todos los brazos y las líneas base."""

PERCENT_DIGITS = 3

AMENDMENT = 4
"""La enmienda de `docs/ablation.md` que fija el intervalo de los veredictos."""

CLUSTER_DAYS = 7
"""L de la enmienda 4: un clúster por bloque UTC de 7 días del instante evaluado, todos los
símbolos juntos. Es el L que dio la regla conservadora del efecto de diseño en los siete n de su
rejilla. Sin parámetro ni flag: otro L es otra enmienda, no otro argumento."""

DELTA = 0.002
"""δ de la enmienda 2: 0.20 % de retorno por evaluación, como fracción.

Una constante y no un argumento porque se fijó antes de los datos y no hay otro valor que
pasar: `--delta 0.20` es δ = 20 %, y con él los nueve veredictos de la etapa 1 imprimían
«empate» sin que nada lo impidiera. `tests/test_criteria.py` fija el valor.
"""


class CriteriaError(RuntimeError):
    """La corrida no se puede evaluar: faltan datos, no es la que dice ser, o δ no vale."""


# ───────────────────────────────────── Veredicto y comparación ────────────────────────────────────


class Verdict(StrEnum):
    """Qué se puede decir de una diferencia pareada, dado `delta`."""

    ABOVE = "above"
    """El intervalo al 95% queda entero por encima de 0."""

    BELOW = "below"
    """El intervalo al 95% queda entero por debajo de 0."""

    TIE = "tie"
    """Incluye 0 y es lo bastante estrecho para decir que los brazos son iguales."""

    INCONCLUSIVE = "inconclusive"
    """Incluye 0, pero su semiancho supera `delta`: los datos no permiten decir ni empate."""

    UNDETERMINED = "undetermined"
    """No se pudo calcular el intervalo."""


def _require_delta(delta: float) -> None:
    if delta <= 0:
        raise CriteriaError(f"delta debe ser positivo, recibido {delta}")


def add_delta_arguments(parser: argparse.ArgumentParser) -> None:
    """Los dos argumentos de δ, los mismos en cada comando que lo lee.

    `--delta` se conserva para que los comandos ya escritos —el de la enmienda 3, las líneas
    «Reproduce» de los reportes— se sigan ejecutando tal cual, pero ya no elige nada: solo
    admite el valor registrado. Probar otro es `--exploratory-delta`, que no se puede
    confundir con él.
    """
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--delta",
        type=float,
        default=None,
        help=f"el δ registrado, en fracción de retorno por evaluación. Solo admite {DELTA:g} "
        f"({DELTA:.3%}), que es también el valor sin el argumento",
    )
    group.add_argument(
        "--exploratory-delta",
        type=float,
        default=None,
        help="otro δ, en fracción de retorno por evaluación, para explorar: el reporte lo marca "
        "como «δ no registrado»",
    )


def resolve_delta(parser: argparse.ArgumentParser, args: argparse.Namespace) -> float:
    """El δ con el que corre el comando, o un error de uso si `--delta` no es el registrado.

    El error dice cuánto es el valor en porcentaje: quien escribe `--delta 0.2` quería decir
    0.2 %, y lo que hay que enseñarle es que escribió 20 %.
    """
    explored: float | None = args.exploratory_delta
    if explored is not None:
        if explored <= 0:
            parser.error("--exploratory-delta debe ser positivo")
        return explored
    given: float | None = args.delta
    if given is not None and given != DELTA:
        parser.error(
            f"--delta {given:g} es δ = {given:.3%}; el δ registrado es {DELTA:.3%} "
            f"(--delta {DELTA:g}). Para explorar otro valor: --exploratory-delta {given:g}, "
            "que marca el reporte como «δ no registrado»"
        )
    return DELTA


def delta_flag(delta: float) -> str:
    """El argumento que reproduce un reporte hecho con ese δ.

    El registrado se escribe `--delta`, como en los comandos ya publicados; cualquier otro
    solo entra por `--exploratory-delta`, así que la línea «Reproduce» de un reporte
    exploratorio no se puede pegar creyendo que evalúa el criterio.
    """
    return f"--delta {delta:g}" if delta == DELTA else f"--exploratory-delta {delta:g}"


def unregistered_delta(delta: float) -> str | None:
    """La línea que marca un reporte hecho con otro δ, o `None` si es el registrado.

    Se decide por el valor y no por el argumento que lo trajo: quien llame a las funciones
    desde código con otro δ recibe la misma marca que quien lo pidió por la línea de comandos.
    """
    if delta == DELTA:
        return None
    return (
        f"**δ no registrado**: {delta:.3%}. El δ de la enmienda 2 es {DELTA:.3%}; lo que sigue "
        "es exploratorio y no evalúa ningún criterio."
    )


def half_width(diff: PairedDifference) -> float:
    """Semiancho del intervalo: `1.96 * error estándar`, el agrupado desde la enmienda 4."""
    return (diff.high - diff.low) / 2.0


def classify(diff: PairedDifference | None, delta: float) -> Verdict:
    """El veredicto de una diferencia pareada.

    Con el intervalo fuera de 0, `delta` no interviene: la diferencia es distinta de cero y
    el signo manda. Solo cuando el intervalo incluye 0 hace falta saber si es estrecho o ancho,
    y eso es lo que `delta` decide. La comparación es estricta: un semiancho igual a `delta`
    todavía permite hablar de empate.
    """
    _require_delta(delta)
    if diff is None:
        return Verdict.UNDETERMINED
    if diff.low > 0:
        return Verdict.ABOVE
    if diff.high < 0:
        return Verdict.BELOW
    if half_width(diff) > delta:
        return Verdict.INCONCLUSIVE
    return Verdict.TIE


class Comparison(FrozenModel):
    """Un brazo contra otro sobre las mismas evaluaciones, con todo lo que hay detrás."""

    arm: str
    reference: str
    n: int = Field(ge=0)
    """Evaluaciones emparejadas."""

    unpaired: int = Field(default=0, ge=0)
    """Evaluaciones que solo tiene uno de los dos brazos y por eso no entran."""

    pair: ClusteredPair | None = None
    """La diferencia con sus dos errores estándar y el intervalo del agrupado. `None` si los
    brazos no se pudieron emparejar o hay menos de `MIN_PAIRED_N` parejas."""

    verdict: Verdict
    why: str | None = None
    """Por qué no se pudo determinar, cuando `verdict` es `UNDETERMINED`."""

    net: NetPairs | None = None
    """La misma diferencia sobre el retorno neto. Descriptiva: no entra en `verdict`, y es `None`
    si no se pidió el neto o los brazos no se pudieron alinear."""

    @property
    def diff(self) -> PairedDifference | None:
        """El intervalo del que sale el veredicto: el del error estándar agrupado."""
        return None if self.pair is None else self.pair.interval

    @property
    def naive(self) -> PairedDifference | None:
        """La diferencia con el error ingenuo, `stdev/√n`. Se imprime; no juzga."""
        return None if self.pair is None else self.pair.naive

    @property
    def half_width(self) -> float | None:
        """Semiancho del intervalo, o `None` sin él."""
        return None if self.diff is None else half_width(self.diff)


def compare_returns(
    arm: str,
    reference: str,
    arm_returns: Sequence[float],
    reference_returns: Sequence[float],
    clusters: Sequence[Hashable],
    delta: float,
    unpaired: int = 0,
) -> Comparison:
    """Compara dos vectores de retorno por evaluación, ya emparejados por posición.

    `clusters[i]` es el clúster de la evaluación `i` y no tiene valor por defecto: el veredicto
    sale del intervalo del error agrupado, y un llamador que lo olvidara volvería al ingenuo
    sin que nada fallara. Con pocos clústeres —o uno solo— quedan las cifras y no hay
    veredicto.
    """
    _require_delta(delta)
    n = len(arm_returns)
    pair = clustered_paired_difference(arm_returns, reference_returns, clusters)
    why: str | None = None
    if pair is None:
        why = (
            "retornos de distinta longitud"
            if len(arm_returns) != len(reference_returns)
            else f"menos de {MIN_PAIRED_N} evaluaciones emparejadas ({n})"
        )
    elif pair.interval is None:
        why = pair.why
    return Comparison(
        arm=arm,
        reference=reference,
        n=n,
        unpaired=unpaired,
        pair=pair,
        verdict=classify(None if pair is None else pair.interval, delta),
        why=why,
    )


class ArmBaselines(FrozenModel):
    """Un brazo contra cada una de las cuatro líneas base."""

    arm: str
    comparisons: tuple[Comparison, ...]

    @property
    def beaten(self) -> int:
        """En cuántas de las comparaciones el intervalo queda por encima de 0."""
        return sum(1 for c in self.comparisons if c.verdict is Verdict.ABOVE)

    @property
    def beats_all(self) -> bool:
        """Si supera a todas: la conjunción de las comparaciones, no un máximo."""
        return self.beaten == len(self.comparisons)


class Cell(FrozenModel):
    """n y media de una diferencia en un tramo. Sin intervalo: n < 30 no lo permite."""

    label: str
    n: int = Field(ge=0)
    mean: float


class Descriptive(FrozenModel):
    """Diferencia pareada de `full` con otro brazo por símbolo y por tramo. Sin veredicto."""

    arm: str
    reference: str
    by_symbol: tuple[Cell, ...] = ()
    by_stratum: tuple[Cell, ...] | None = None
    """`None` sin manifiesto: sin él no se sabe a qué tramo pertenece cada evaluación."""

    why: str | None = None


class CriteriaReport(FrozenModel):
    """Los criterios evaluados sobre una corrida."""

    delta: float
    horizon: int
    criterion_1: tuple[Comparison, ...]
    criterion_3: tuple[ArmBaselines, ...]
    descriptive: tuple[Descriptive, ...]

    net_note: str | None = None
    """Por qué no hay columnas netas, o `None` si las hay. Los veredictos no dependen de esto."""


# ───────────────────────────────────────── Evaluación ─────────────────────────────────────────────


class PairedReturns(FrozenModel):
    """Los dos brazos sobre las evaluaciones que comparten, en el mismo orden.

    Es lo que los criterios restan: `arm_returns[i]` y `reference_returns[i]` son la misma
    evaluación, y `arm_records[i]` y `reference_records[i]` lo que cada brazo dejó de ella.
    """

    arm_records: tuple[EvaluationRecord, ...]
    reference_records: tuple[EvaluationRecord, ...]
    arm_returns: tuple[float, ...]
    reference_returns: tuple[float, ...]
    unpaired: int
    arm_net: tuple[float | None, ...] | None = None
    reference_net: tuple[float | None, ...] | None = None


def _by_run_id(arm: str, records: Iterable[EvaluationRecord]) -> dict[UUID, EvaluationRecord]:
    indexed: dict[UUID, EvaluationRecord] = {}
    for record in records:
        if record.run_id in indexed:
            raise CriteriaError(f"el brazo {arm} tiene la evaluación {record.run_id} repetida")
        indexed[record.run_id] = record
    return indexed


def _align(
    arm: str,
    reference: str,
    index: Mapping[str, Mapping[UUID, EvaluationRecord]],
    histories: Mapping[str, Sequence[Sequence[float]]],
    horizon: int,
    net: NetInputs | None = None,
) -> PairedReturns | str:
    """Empareja por `run_id` —no por posición— o devuelve por qué no se puede.

    Con `net` puntúa además el neto de cada brazo. Los vectores brutos salen del mismo
    `score_run` y no dependen de él: el neto se calcula al lado, no en lugar de.
    """
    ours, theirs = index.get(arm, {}), index.get(reference, {})
    for name, side in ((arm, ours), (reference, theirs)):
        if not side:
            return f"el brazo {name} no tiene evaluaciones"
    common = sorted(set(ours) & set(theirs), key=str)
    if not common:
        return "los dos brazos no comparten ninguna evaluación"
    ours_records = [ours[run_id] for run_id in common]
    theirs_records = [theirs[run_id] for run_id in common]
    scored_ours = score_run(ours_records, histories, horizon, SCORING, net)
    scored_theirs = score_run(theirs_records, histories, horizon, SCORING, net)
    bare = scored_ours.unscorable + scored_theirs.unscorable
    if bare:
        return (
            f"{bare} registro(s) sin indicadores: no se puede reconstruir el stop común "
            "y un cero silencioso sesgaría la diferencia"
        )
    return PairedReturns(
        arm_records=tuple(ours_records),
        reference_records=tuple(theirs_records),
        arm_returns=scored_ours.per_evaluation,
        reference_returns=scored_theirs.per_evaluation,
        unpaired=len(set(ours) | set(theirs)) - len(common),
        arm_net=scored_ours.per_evaluation_net,
        reference_net=scored_theirs.per_evaluation_net,
    )


def paired_returns(
    arm: str,
    reference: str,
    arms: Mapping[str, Sequence[EvaluationRecord]],
    histories: Mapping[str, Sequence[Sequence[float]]],
    horizon: int,
) -> PairedReturns | str:
    """Los dos vectores de retorno bruto que los criterios restan, o por qué no los hay.

    La puerta para quien necesita el vector de diferencias y no solo su media y su error
    —el recálculo de dispersión—: empareja y puntúa por `_align`, lo mismo que
    `evaluate_criteria`, así que no hay una segunda forma de alinear dos brazos que pueda
    divergir de la que da los veredictos. Sin neto: los criterios son del bruto.
    """
    index = {name: _by_run_id(name, arms[name]) for name in (arm, reference) if name in arms}
    return _align(arm, reference, index, histories, horizon)


def _compare(
    arm: str,
    reference: str,
    index: Mapping[str, Mapping[UUID, EvaluationRecord]],
    histories: Mapping[str, Sequence[Sequence[float]]],
    horizon: int,
    delta: float,
    net: NetInputs | None = None,
) -> Comparison:
    aligned = _align(arm, reference, index, histories, horizon, net)
    if isinstance(aligned, str):
        return Comparison(
            arm=arm, reference=reference, n=0, verdict=Verdict.UNDETERMINED, why=aligned
        )
    clusters = clusters_of(aligned.arm_records, CLUSTER_DAYS)
    comparison = compare_returns(
        arm,
        reference,
        aligned.arm_returns,
        aligned.reference_returns,
        clusters,
        delta,
        aligned.unpaired,
    )
    if aligned.arm_net is None or aligned.reference_net is None:
        return comparison
    return comparison.model_copy(
        update={"net": paired_net_difference(aligned.arm_net, aligned.reference_net, clusters)}
    )


def _cells(groups: Mapping[str, list[float]]) -> tuple[Cell, ...]:
    return tuple(
        Cell(label=label, n=len(values), mean=sum(values) / len(values))
        for label, values in sorted(groups.items())
    )


def _descriptive(
    reference: str,
    index: Mapping[str, Mapping[UUID, EvaluationRecord]],
    histories: Mapping[str, Sequence[Sequence[float]]],
    horizon: int,
    strata: Mapping[tuple[str, datetime], int] | None,
) -> Descriptive:
    aligned = _align(REFERENCE, reference, index, histories, horizon)
    if isinstance(aligned, str):
        return Descriptive(arm=REFERENCE, reference=reference, why=aligned)
    by_symbol: dict[str, list[float]] = {}
    by_stratum: dict[str, list[float]] = {}
    for record, ours, theirs in zip(
        aligned.arm_records, aligned.arm_returns, aligned.reference_returns, strict=True
    ):
        by_symbol.setdefault(record.symbol, []).append(ours - theirs)
        if strata is not None:
            label = (
                str(strata[(record.symbol, record.at)])
                if (record.symbol, record.at) in strata
                else "sin tramo"
            )
            by_stratum.setdefault(label, []).append(ours - theirs)
    return Descriptive(
        arm=REFERENCE,
        reference=reference,
        by_symbol=_cells(by_symbol),
        by_stratum=None if strata is None else _cells(by_stratum),
    )


def evaluate_criteria(
    arms: Mapping[str, Sequence[EvaluationRecord]],
    histories: Mapping[str, Sequence[Sequence[float]]],
    horizon: int,
    delta: float,
    strata: Mapping[tuple[str, datetime], int] | None = None,
    net: NetInputs | None = None,
    net_note: str | None = None,
) -> CriteriaReport:
    """Evalúa los criterios 1 y 3 sobre los registros de cada brazo.

    Pura sobre sus argumentos: los registros dicen qué decidió cada brazo y las velas
    —que son una entrada, no algo que esta función vaya a buscar— dicen cuánto rindió. Se
    puntúa con `outcomes.score_run` y el stop común, el mismo para los brazos con modelo y
    para las líneas base, que declaran exactamente ese stop.

    El intervalo de cada diferencia es el del error agrupado de la enmienda 4: el clúster de
    una evaluación es el bloque UTC de `CLUSTER_DAYS` días de su `at`, y de ahí sale el
    veredicto. No hay argumento para otro L ni para volver al error ingenuo.

    `net` añade a cada comparación la diferencia pareada del retorno neto, descriptiva. No
    entra en ningún veredicto: con o sin él, `criterion_1` y `criterion_3` son los mismos
    salvo por ese campo, y `net_note` dice por qué faltan si no se pasó.
    """
    _require_delta(delta)
    missing = sorted({r.symbol for records in arms.values() for r in records} - set(histories))
    if missing:
        raise CriteriaError(f"faltan históricos para: {', '.join(missing)}")
    index = {name: _by_run_id(name, records) for name, records in arms.items()}

    criterion_1 = tuple(
        _compare(REFERENCE, other, index, histories, horizon, delta, net)
        for other in CRITERION_1_REFERENCES
    )
    criterion_3 = tuple(
        ArmBaselines(
            arm=name,
            comparisons=tuple(
                _compare(name, baseline, index, histories, horizon, delta, net)
                for baseline in BASELINE_ARMS
            ),
        )
        for name in arms
        if name not in BASELINE_ARMS
    )
    descriptive = tuple(
        _descriptive(other, index, histories, horizon, strata) for other in CRITERION_1_REFERENCES
    )
    return CriteriaReport(
        delta=delta,
        horizon=horizon,
        criterion_1=criterion_1,
        criterion_3=criterion_3,
        descriptive=descriptive,
        net_note=None if net is not None else (net_note or "no se pasó el funding"),
    )


# ─────────────────────────────────── Criterio 6: validez de la corrida ────────────────────────────


class ArmValidity(FrozenModel):
    """Lo que las guardas encontraron en un brazo. Todo en cero es un brazo limpio."""

    arm: str
    decider_quota_lost: int = Field(ge=0)
    """Evaluaciones que el decisor perdió por cuota. Invalida."""

    other_quota_lost: dict[str, int] = Field(default_factory=dict)
    """Evaluaciones perdidas por cuota en otro nodo, por nodo. Avisa."""

    funds_lost: int = Field(default=0, ge=0)
    """Evaluaciones perdidas por saldo insuficiente (402), en cualquier nodo. Invalida."""

    provider_lost: dict[str, int] = Field(default_factory=dict)
    """Evaluaciones perdidas por un rechazo o un plazo vencido, por `nodo/tipo`. Invalida."""

    spend_cap_lost: int = Field(default=0, ge=0)
    """Evaluaciones que el tope de gasto (`--max-usd`) cortó, en cualquier nodo. Invalida."""

    cache_backend_mismatch: int = Field(ge=0)
    """Aciertos de caché con un backend distinto al del brazo. Invalida."""

    live_backend_mismatch: int = Field(ge=0)
    """Llamadas vivas con un backend distinto al del brazo, o sea degradaciones. Avisa."""

    foreign_vetoes: dict[str, int] = Field(default_factory=dict)
    """Vetos distintos de `invalid_stop_side`, por regla. Invalida."""


class ValidityReport(FrozenModel):
    """El criterio 6 sobre una corrida."""

    arms: tuple[ArmValidity, ...]

    @staticmethod
    def _per_arm(pairs: Iterable[tuple[str, int]]) -> str:
        return ", ".join(f"{arm} {count}" for arm, count in pairs if count)

    @property
    def reasons(self) -> list[str]:
        """Una razón por condición que falla, con el desglose por brazo."""
        reasons: list[str] = []
        quota = [(a.arm, a.decider_quota_lost) for a in self.arms]
        if total := sum(count for _, count in quota):
            reasons.append(
                f"el decisor perdió {total} evaluación(es) por cuota ({self._per_arm(quota)})"
            )
        funds = [(a.arm, a.funds_lost) for a in self.arms]
        if total := sum(count for _, count in funds):
            reasons.append(
                f"{total} evaluación(es) perdidas por saldo insuficiente, 402 «Insufficient "
                f"account funds» ({self._per_arm(funds)})"
            )
        provider: dict[str, list[str]] = {}
        for entry in self.arms:
            for where, count in sorted(entry.provider_lost.items()):
                provider.setdefault(where, []).append(f"{entry.arm} {count}")
        if provider:
            total = sum(sum(entry.provider_lost.values()) for entry in self.arms)
            detail = "; ".join(f"{where}: {', '.join(arms)}" for where, arms in provider.items())
            reasons.append(
                f"{total} evaluación(es) perdidas por un fallo del proveedor, rechazo o plazo "
                f"vencido ({detail})"
            )
        capped = [(a.arm, a.spend_cap_lost) for a in self.arms]
        if total := sum(count for _, count in capped):
            reasons.append(
                f"{total} evaluación(es) cortadas por el tope de gasto, `--max-usd` "
                f"({self._per_arm(capped)})"
            )
        cache = [(a.arm, a.cache_backend_mismatch) for a in self.arms]
        if total := sum(count for _, count in cache):
            reasons.append(
                f"{total} acierto(s) de caché con un backend distinto al del brazo "
                f"({self._per_arm(cache)})"
            )
        vetoes: dict[str, list[str]] = {}
        for entry in self.arms:
            for rule, count in sorted(entry.foreign_vetoes.items()):
                vetoes.setdefault(rule, []).append(f"{entry.arm} {count}")
        if vetoes:
            detail = "; ".join(f"{rule}: {', '.join(where)}" for rule, where in vetoes.items())
            reasons.append(f"vetos distintos de {INVALID_STOP_SIDE}: {detail}")
        return reasons

    @property
    def warnings(self) -> list[str]:
        """Lo que se avisa sin invalidar: cuota perdida fuera del decisor y degradaciones."""
        warnings: list[str] = []
        lost: dict[str, list[str]] = {}
        for entry in self.arms:
            for node, count in sorted(entry.other_quota_lost.items()):
                lost.setdefault(node, []).append(f"{entry.arm} {count}")
        if lost:
            detail = "; ".join(f"{node}: {', '.join(where)}" for node, where in lost.items())
            warnings.append(f"evaluaciones perdidas por cuota fuera del decisor ({detail})")
        live = [(a.arm, a.live_backend_mismatch) for a in self.arms]
        if sum(count for _, count in live):
            warnings.append(
                "llamadas vivas con un backend distinto al del brazo, es decir degradadas por "
                f"cuota ({self._per_arm(live)})"
            )
        return warnings

    @property
    def valid(self) -> bool:
        """Si la corrida cumple el criterio 6."""
        return not self.reasons


def check_validity(
    arms: Mapping[str, Sequence[EvaluationRecord]],
    expected: Mapping[str, Mapping[AgentRole, Backend]],
) -> ValidityReport:
    """Las seis condiciones del criterio 6 sobre los registros de cada brazo.

    `expected` es el backend que `meta.json` declara para cada rol de cada brazo. Un brazo que
    el meta no describe no se puede comprobar, y eso es un error y no un «sin problemas».

    La sexta es el tope de gasto (enmienda 3): una evaluación que `--max-usd` no dejó terminar
    es una evaluación que el brazo no tuvo ocasión de decidir, igual que la que se llevó un 402.
    Como el saldo, el tope es de la corrida y no de un nodo: cuenta en cualquiera.
    """
    results: list[ArmValidity] = []
    for name, records in arms.items():
        if name not in expected:
            raise CriteriaError(f"meta.json no describe los roles del brazo {name}")
        declared = expected[name]
        decider_lost = funds_lost = cap_lost = 0
        other_lost: dict[str, int] = {}
        provider_lost: dict[str, int] = {}
        cache_mismatch = live_mismatch = 0
        vetoes: dict[str, int] = {}
        for record in records:
            for (node, kind), count in undecided_causes([record]).items():
                if kind is AbortKind.INSUFFICIENT_FUNDS:
                    funds_lost += count
                    continue
                if kind in PROVIDER_FAILURES:
                    where = f"{node}/{kind.value}"
                    provider_lost[where] = provider_lost.get(where, 0) + count
                    continue
                if kind is AbortKind.SPEND_CAP:
                    cap_lost += count
                    continue
                if kind is not AbortKind.QUOTA:
                    continue
                if node in DECIDER_NODES:
                    decider_lost += count
                else:
                    other_lost[node] = other_lost.get(node, 0) + count
            for call in record.calls:
                if call.backend is declared.get(call.role):
                    continue
                if call.cache_hit:
                    cache_mismatch += 1
                else:
                    live_mismatch += 1
            verdict = record.risk
            if verdict is not None and verdict.veto_rule not in (None, INVALID_STOP_SIDE):
                vetoes[verdict.veto_rule] = vetoes.get(verdict.veto_rule, 0) + 1
        results.append(
            ArmValidity(
                arm=name,
                decider_quota_lost=decider_lost,
                other_quota_lost=dict(sorted(other_lost.items())),
                funds_lost=funds_lost,
                provider_lost=dict(sorted(provider_lost.items())),
                spend_cap_lost=cap_lost,
                cache_backend_mismatch=cache_mismatch,
                live_backend_mismatch=live_mismatch,
                foreign_vetoes=dict(sorted(vetoes.items())),
            )
        )
    return ValidityReport(arms=tuple(results))


def final_records(chain: Sequence[RunDirectory]) -> dict[str, tuple[EvaluationRecord, ...]]:
    """Lo que se juzga de una corrida reanudada: los registros de su último eslabón, por brazo.

    `chain` viene de `run_chain`, de la pasada más reciente a la más antigua. Una reanudación
    vuelve a recorrer el plan entero —lo que ya tenía contenido sale de la caché, lo que murió en un
    rechazo del proveedor o en una ventana agotada se llama de nuevo—, así que su directorio es el
    estado final de cada evaluación. Lo que una pasada anterior perdió y esta decidió ya no está
    perdido; lo que esta tampoco decidió, sí. Sumar los eslabones contaría como perdida una
    evaluación que tiene su decisión dos líneas más abajo.

    Límite: se confía en que el último eslabón esté completo. Una reanudación interrumpida, o
    lanzada con menos `--arms`, no trae lo que no volvió a ejecutar, y eso aquí no se ve.
    """
    return {arm.arm: arm.records for arm in chain[0].arms}


# ───────────────────────────────────── Pico de consumo de la ventana ──────────────────────────────


class PeakWindow(FrozenModel):
    """El máximo consumo remoto de un par (rol, modelo) en cualquier ventana deslizante."""

    role: AgentRole
    model: str
    calls: int = Field(ge=0)
    """El mayor número de llamadas vivas remotas dentro de una ventana."""

    quota: float = Field(ge=0.0)
    """La mayor cuota consumida —la suma de `quota_weight`— dentro de una ventana."""

    end: datetime
    """Instante en que termina la ventana donde la cuota alcanzó su máximo."""


def peak_window_usage(
    calls: Iterable[LLMCall], role: AgentRole, window: timedelta
) -> tuple[PeakWindow, ...]:
    """Máximo de llamadas vivas remotas de un rol en cualquier ventana deslizante.

    Un par por modelo remoto del rol: la cuota se lleva por (rol, modelo). Cuenta lo que el
    proveedor vio —ni aciertos de caché ni llamadas locales— y usa la ventana del
    contador: una llamada cuenta si su `at` está en `(t - window, t]`, de modo que una hecha
    exactamente `window` antes ya no cuenta. Sirve el orden en que lleguen.
    """
    by_model: dict[str, list[LLMCall]] = {}
    for call in calls:
        if call.role is role and not call.cache_hit and call.backend not in LOCAL_BACKENDS:
            by_model.setdefault(call.model, []).append(call)

    peaks: list[PeakWindow] = []
    for model, group in sorted(by_model.items()):
        ordered = sorted(group, key=lambda call: call.at)
        live: deque[LLMCall] = deque()
        quota = 0.0
        best_calls, best_quota, best_end = 0, 0.0, ordered[0].at
        for call in ordered:
            live.append(call)
            quota += call.quota_weight
            while live[0].at <= call.at - window:
                quota -= live.popleft().quota_weight
            best_calls = max(best_calls, len(live))
            if quota > best_quota:
                best_quota, best_end = quota, call.at
        peaks.append(
            PeakWindow(
                role=role, model=model, calls=best_calls, quota=max(best_quota, 0.0), end=best_end
            )
        )
    return tuple(peaks)


# ───────────────────────────────────────── Informe ────────────────────────────────────────────────

_VERDICT_LABEL = {
    Verdict.ABOVE: "justifica",
    Verdict.BELOW: "peor",
    Verdict.TIE: "empate",
    Verdict.INCONCLUSIVE: "no concluyente",
}
_BASELINE_LABEL = {**_VERDICT_LABEL, Verdict.ABOVE: "supera"}

_COMPARISON_HEADER = (
    "| brazo | referencia | n | diferencia media | EE ingenuo | EE agrupado | G | deff | IC 95% "
    f"| semiancho | veredicto | {NET_LABEL}: n | {NET_LABEL}: diferencia media [IC 95%] |",
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
)

_UNREGISTERED = " (δ no registrado)"


def _pct(value: float) -> str:
    return f"{value:+.{PERCENT_DIGITS}%}"


def _net_cells(c: Comparison, note: str | None) -> str:
    """Las dos columnas netas: cuántas parejas entraron y la diferencia, o por qué no hay."""
    if c.net is None:
        why = _NET_UNDETERMINED.format(note or "los brazos no se pudieron alinear")
        return f"{why} | {why}"
    dropped = f" (no determinado: {c.net.dropped})" if c.net.dropped else ""
    if c.net.diff is None:
        reason = (
            f"menos de {MIN_PAIRED_N} parejas con el funding completo"
            if c.net.pair is None
            else c.net.pair.why
        )
        return f"{c.net.n}{dropped} | no determinado: {reason}"
    diff = c.net.diff
    return f"{c.net.n}{dropped} | {_pct(diff.mean)} [{_pct(diff.low)}, {_pct(diff.high)}]"


_NET_UNDETERMINED = "no determinado: {}"


def _error(value: float | None) -> str:
    return "—" if value is None else f"{value:.{PERCENT_DIGITS}%}"


def _deff(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _comparison_row(
    c: Comparison,
    labels: Mapping[Verdict, str],
    note: str | None = None,
    registered: bool = True,
) -> str:
    """Una fila: la diferencia, sus dos errores, G, deff y el intervalo del agrupado.

    Sin intervalo —pocos clústeres— se imprimen las cifras que hay y el motivo en lugar del
    veredicto. `registered` es si δ es el de la enmienda: si no, el veredicto va marcado.
    """
    net = _net_cells(c, note)
    if c.pair is None:
        return (
            f"| {c.arm} | {c.reference} | {c.n} | — | — | — | — | — | — | — "
            f"| no determinado: {c.why} | {net} |"
        )
    error = c.pair.error
    figures = (
        f"| {c.arm} | {c.reference} | {c.n} | {_pct(c.pair.naive.mean)} | {_error(error.naive)} "
        f"| {_error(error.clustered)} | {error.clusters} | {_deff(error.deff)} "
    )
    if c.diff is None or c.half_width is None:
        return f"{figures}| — | — | no determinado: {c.why} | {net} |"
    pairing = f" (sin pareja: {c.unpaired})" if c.unpaired else ""
    mark = "" if registered else _UNREGISTERED
    return (
        f"{figures}| [{_pct(c.diff.low)}, {_pct(c.diff.high)}] "
        f"| {c.half_width:.{PERCENT_DIGITS}%} | {labels[c.verdict]}{mark}{pairing} | {net} |"
    )


def _cell_rows(label: str, reference: str, cells: Sequence[Cell]) -> list[str]:
    return [
        f"| {reference} | {label} {cell.label} | {cell.n} | {_pct(cell.mean)} |" for cell in cells
    ]


def render_criteria(report: CriteriaReport, source: str, preamble: str = "") -> str:
    """Los criterios 1 y 3 y la diferencia descriptiva, cada cifra con su n y su intervalo.

    `preamble` va entre el encabezado y el criterio 1: lo que el comando imprime antes de los
    veredictos —de dónde salen las cifras, la validez, el pico de consumo—.
    """
    unregistered = unregistered_delta(report.delta)
    registered = unregistered is None
    lines = [
        f"# Criterios de la corrida `{source}`",
        "",
        f"δ = {report.delta:.{PERCENT_DIGITS}%} de retorno por evaluación (fracción "
        f"{report.delta:g}) · horizonte {report.horizon} velas · puntuación: stop común · "
        "diferencias pareadas por `run_id`",
        "",
        *([] if unregistered is None else [unregistered, ""]),
        f"Enmienda {AMENDMENT}: el IC al 95% es `media ± 1.96 · EE agrupado`, con el error "
        f"estándar agrupado CR1 y un clúster por bloque UTC de {CLUSTER_DAYS} días del instante "
        "evaluado, todos los símbolos juntos. El EE ingenuo (`stdev/√n`) supone evaluaciones "
        "independientes: se imprime al lado y no entra en ningún veredicto. G son los "
        "clústeres y deff, `(EE agrupado / EE ingenuo)²`; con menos de "
        f"{MIN_CLUSTERS} clústeres no hay intervalo.",
        "",
        *([preamble, ""] if preamble else []),
        "## Criterio 1: ¿`full` justifica su coste?",
        "",
        "Justifica solo si el IC queda por encima de 0. Un empate es derrota para `full`; "
        "«no concluyente» es que el IC incluye 0 con un semiancho mayor que δ.",
        "",
        *_COMPARISON_HEADER,
    ]
    lines += [
        _comparison_row(c, _VERDICT_LABEL, report.net_note, registered) for c in report.criterion_1
    ]

    lines += [
        "",
        "## Criterio 3: ¿tiene señal el brazo?",
        "",
        "Cada brazo contra cada una de las cuatro líneas base, por separado: no hay un «mejor "
        "de las líneas base».",
        "",
        *_COMPARISON_HEADER,
    ]
    for table in report.criterion_3:
        lines += [
            _comparison_row(c, _BASELINE_LABEL, report.net_note, registered)
            for c in table.comparisons
        ]
    lines.append("")
    for table in report.criterion_3:
        total = len(table.comparisons)
        lines.append(f"- `{table.arm}`: supera a {table.beaten} de {total} líneas base.")

    lines += [
        "",
        "## Diferencia pareada descriptiva",
        "",
        "`full` menos cada referencia, por símbolo y por tramo del manifiesto: n y media, sin "
        "veredicto ni intervalo (n < 30 por celda).",
        "",
    ]
    for item in report.descriptive:
        if item.why is not None:
            lines += [f"`full` contra `{item.reference}`: no determinado: {item.why}", ""]
            continue
        lines += ["| referencia | grupo | n | diferencia media |", "| --- | --- | --- | --- |"]
        lines += _cell_rows("símbolo", item.reference, item.by_symbol)
        if item.by_stratum is None:
            lines.append(f"| {item.reference} | tramo | — | no determinado: sin manifiesto |")
        else:
            lines += _cell_rows("tramo", item.reference, item.by_stratum)
        lines.append("")
    return "\n".join(lines) + "\n"


def render_validity(result: ValidityReport) -> str:
    """El criterio 6 por brazo, con las seis condiciones y los avisos, pasen o no."""
    lines = [
        "## Criterio 6: la corrida es válida",
        "",
        "| brazo | decisor perdido por cuota | saldo insuficiente (402) | fallo del proveedor "
        "| tope de gasto | caché de otro backend | vetos ajenos | avisos |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in result.arms:
        vetoes = ", ".join(f"{rule} {n}" for rule, n in entry.foreign_vetoes.items()) or "0"
        provider = ", ".join(f"{where} {n}" for where, n in entry.provider_lost.items()) or "0"
        notes = []
        if entry.other_quota_lost:
            notes.append("cuota fuera del decisor " + str(sum(entry.other_quota_lost.values())))
        if entry.live_backend_mismatch:
            notes.append(f"degradadas {entry.live_backend_mismatch}")
        lines.append(
            f"| {entry.arm} | {entry.decider_quota_lost} | {entry.funds_lost} | {provider} | "
            f"{entry.spend_cap_lost} | {entry.cache_backend_mismatch} | {vetoes} | "
            f"{', '.join(notes) or '—'} |"
        )
    lines.append("")
    lines += [f"AVISO: {warning}" for warning in result.warnings]
    if result.warnings:
        lines.append("")
    return "\n".join(lines)


def render_invalid(result: ValidityReport, source: str) -> str:
    """Una corrida inválida: el motivo primero y ningún veredicto."""
    reasons = "; ".join(result.reasons)
    return "\n".join(
        [
            f"CORRIDA INVÁLIDA: {reasons}",
            "",
            f"No se emiten veredictos de la corrida `{source}`.",
            "",
            render_validity(result),
        ]
    )


def render_peaks(
    calls: Iterable[LLMCall],
    window: timedelta,
    limits: Mapping[tuple[AgentRole, str], int],
    billing: Billing | None = None,
) -> str:
    """El pico de consumo remoto de cada rol en una ventana deslizante. Se imprime siempre.

    Con pago por uso la columna del uso de la cuota dice `no aplica (payg)` y no hay aviso: Zen no
    publica límite y lo declarado es un centinela, así que un porcentaje contra él no mediría
    nada. Los picos de llamadas y de cuota son medidas y se quedan. `None` —una corrida que no
    registró su facturación— es lo de siempre: se compara con lo declarado.
    """
    hours = window.total_seconds() / 3600
    collected = list(calls)
    lines = [
        f"## Ventana de {hours:g} h: pico de consumo remoto",
        "",
        "Máximo, en cualquier ventana deslizante, de llamadas vivas remotas —sin aciertos de "
        "caché ni llamadas locales— y de cuota (suma de `quota_weight`), sobre todas las "
        "pasadas de la corrida.",
        "",
        "| rol | modelo | pico de llamadas | pico de cuota | uso de la cuota por ventana "
        "| termina |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    warnings: list[str] = []
    found = False
    for role in AgentRole:
        for peak in peak_window_usage(collected, role, window):
            found = True
            limit = limits.get((role, peak.model))
            if billing is Billing.PAYG:
                usage = QUOTA_NOT_APPLICABLE
            elif limit is None:
                usage = "no determinado: meta.json no registra la cuota de ese modelo"
            else:
                share = peak.quota / limit
                usage = f"{peak.quota:g} de {limit} ({share:.0%})"
                if share > PEAK_WARNING_SHARE:
                    warnings.append(
                        f"AVISO: {role.value} `{peak.model}` llegó a {peak.quota:g} de {limit} "
                        f"({share:.0%}) de su cuota por ventana, más del {PEAK_WARNING_SHARE:.0%}"
                    )
            lines.append(
                f"| {role.value} | `{peak.model}` | {peak.calls} | {peak.quota:g} | {usage} | "
                f"{peak.end.isoformat()} |"
            )
    if not found:
        lines.append("| (ninguna llamada viva remota) | | | | | |")
    lines.append("")
    lines += warnings
    if warnings:
        lines.append("")
    return "\n".join(lines)


# ────────────────────────────────────────── Comando ───────────────────────────────────────────────


def _header(run: RunDirectory, extra: Sequence[str] = ()) -> list[str]:
    meta = run.meta
    roles = meta.arm_roles or {}
    temperatures = {
        (arm, role.value): entry.temperature
        for arm, by_role in roles.items()
        for role, entry in by_role.items()
    }
    odd = sorted(f"{arm}/{role} {value:g}" for (arm, role), value in temperatures.items() if value)
    temperature = (
        f"AVISO: temperatura distinta de 0 en {', '.join(odd)}"
        if odd
        else f"temperatura efectiva 0 en los {len(temperatures)} pares (brazo, rol)"
    )
    switch = {True: "ACTIVO", False: "no activo", None: "no registrado"}[meta.kill_switch]
    lines = [
        f"- plan ({meta.plan_kind.value}): `{meta.plan_path}`, sha-256 `{meta.plan_sha256}`",
        f"- {temperature}",
        f"- kill switch de la configuración: {switch}",
    ]
    lines += [
        f"- `{arm.path.name}` sha-256 `{arm.sha256}`" for arm in run.arms if arm.sha256 is not None
    ]
    return [*lines, *extra, ""]


def _net_inputs(
    records: Sequence[EvaluationRecord], funding_dir: Path
) -> tuple[NetInputs | None, str | None, list[str]]:
    """El funding y los costes de las columnas netas, con lo que hay que citar de ellos.

    Sin las series no se cae: las columnas netas dicen por qué no están y los criterios, que son
    del bruto, se evalúan igual. Devuelve las entradas, el motivo si faltan y las líneas de la
    cabecera —costes y sha-256 de cada archivo de funding— para que la cifra cite su origen.
    """
    costs = DEFAULT_COSTS
    symbols = sorted({record.symbol for record in records})
    lines = [
        f"- {NET_LABEL}: comisión taker {costs.taker_fee:.4%} por lado (confirmada, "
        f"{costs.as_of.isoformat()}), deslizamiento {costs.slippage:.4%} por lado (supuesto sin "
        f"medir); ida y vuelta {costs.round_trip:.4%}; funding de `{funding_dir}`",
    ]
    try:
        series = load_funding(symbols, funding_dir)
        digests = funding_digests(symbols, funding_dir)
    except FundingError as error:
        return None, str(error), [*lines, f"- funding: no determinado: {error}"]
    lines += [f"- funding `{symbol}` sha-256 `{digest}`" for symbol, digest in digests.items()]
    return NetInputs(funding=series, costs=costs), None, lines


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada: `python -m crypto_agents.criteria <directorio>`.

    Sale con 0 si la corrida es válida y se evaluó, con 1 si es inválida y con 2 si no se pudo
    evaluar.
    """
    parser = argparse.ArgumentParser(
        prog="python -m crypto_agents.criteria",
        description="Evalúa los criterios de la ablación sobre el directorio de una corrida.",
    )
    parser.add_argument("directory", type=Path, help="directorio de la corrida, con meta.json")
    add_delta_arguments(parser)
    parser.add_argument("--plan", type=Path, default=None, help="manifiesto o histórico del plan")
    parser.add_argument("--history-dir", type=Path, default=HISTORY_DIR)
    parser.add_argument(
        "--funding-dir",
        type=Path,
        default=FUNDING_DIR,
        help="series de funding para las columnas netas (descriptivas); sin ellas dicen "
        "«no determinado» y los veredictos no cambian",
    )
    parser.add_argument("--horizon", type=int, default=None)
    args = parser.parse_args(argv)
    delta = resolve_delta(parser, args)

    try:
        chain = run_chain(args.directory)
        run = chain[0]
        meta = run.meta
        probes = [item.path for item in chain if item.meta.kind is RunKind.PROBE]
        if probes:
            raise CriteriaError(
                f"{probes[0]} es un sondeo (kind=probe), no una corrida de la ablación: sus brazos "
                "son un modelo cada uno y no hay `full` ni líneas base con que compararlos. "
                "Léelo con `audit` o `consumption`"
            )
        if meta.arm_roles is None:
            raise CriteriaError(
                f"{META_FILE} no registra arm_roles: sin el backend de cada brazo no se puede "
                "comprobar el criterio 6"
            )
        arms = final_records(chain)
        validity = check_validity(
            arms,
            {
                name: {r: m.backend for r, m in roles.items()}
                for name, roles in meta.arm_roles.items()
            },
        )
        window = meta.quota_window
        if window is None:
            raise CriteriaError(f"{META_FILE} no registra quota_window")
        limits = {
            (role, entry.model): entry.quota_per_window
            for roles in meta.arm_roles.values()
            for role, entry in roles.items()
        }
        peaks = render_peaks(chain_calls(args.directory), window, limits, meta.billing)

        if not validity.valid:
            print(render_invalid(validity, str(args.directory)))
            print(peaks)
            return 1

        records = [record for entries in arms.values() for record in entries]
        histories, strata = load_plan(meta, args.plan, args.history_dir, records)
        horizon = resolve_horizon(meta.horizon, args.horizon)
        net, net_note, net_lines = _net_inputs(records, args.funding_dir)
        report = evaluate_criteria(arms, histories, horizon, delta, strata, net, net_note)
    except (AuditError, SelectionError, CriteriaError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    preamble = "\n".join([*_header(run, net_lines), render_validity(validity), peaks])
    print(render_criteria(report, str(args.directory), preamble))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
