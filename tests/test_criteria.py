"""Pruebas del evaluador de criterios y de las guardas de validez.

Todo parte de registros sintéticos con respuesta conocida: cada evaluación tiene su retorno
escrito de antemano, de modo que la diferencia pareada, su intervalo y el veredicto se
pueden calcular a mano y no leer de la función que se prueba.

Las mutaciones que fijan el diseño están escritas como pruebas, no como comentarios:

- ignorar δ, y el veredicto «no concluyente» desaparece;
- contar el borde de la ventana de 5 h con el signo contrario;
- dejar pasar un veto que no sea `invalid_stop_side`.

Las de la enmienda 4, cada una reescribiendo el código de producción con el error dentro:

- sacar el veredicto del error ingenuo y no del agrupado;
- quitar la corrección `G/(G - 1)` del error agrupado;
- dar veredicto con menos de 50 clústeres;
- aceptar un `--delta` que no es el registrado, o no marcar un δ exploratorio;
- unir los tramos del manifiesto por la apertura de la vela y no por su cierre.
"""

from __future__ import annotations

import asyncio
import functools
import json
import math
import statistics
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, NamedTuple
from uuid import NAMESPACE_URL, uuid5

import pytest

from crypto_agents import audit as audit_module
from crypto_agents import criteria as criteria_module
from crypto_agents import metrics as metrics_module
from crypto_agents.ablation import ARMS, plan_from_manifest, run_arm
from crypto_agents.audit import (
    PlanKind,
    RoleMeta,
    RunKind,
    RunMeta,
    arm_journal_path,
    file_sha256,
    read_meta,
    write_meta,
)
from crypto_agents.audit import main as audit_main
from crypto_agents.cache import InMemoryResponseCache
from crypto_agents.consumption import main as consumption_main
from crypto_agents.criteria import (
    AMENDMENT,
    BASELINE_ARMS,
    CLUSTER_DAYS,
    DECIDER_NODES,
    PEAK_WARNING_SHARE,
    Comparison,
    CriteriaError,
    CriteriaReport,
    PairedReturns,
    PeakWindow,
    Verdict,
    check_validity,
    classify,
    compare_returns,
    evaluate_criteria,
    final_records,
    main,
    paired_returns,
    peak_window_usage,
    render_criteria,
    render_invalid,
    render_peaks,
)
from crypto_agents.journal import EvaluationRecord, InMemoryJournal, JsonlJournal
from crypto_agents.llm import ModelCallError, SpendCapReachedError
from crypto_agents.market import write_ohlcv_csv
from crypto_agents.metrics import MIN_CLUSTERS, AbortKind, PairedDifference, clusters_of
from crypto_agents.quota import QuotaExhaustedError, QuotaLedger
from crypto_agents.risk import INVALID_STOP_SIDE, NO_EXPOSURE_HEADROOM
from crypto_agents.selection import (
    DEFAULT_MANIFEST,
    HISTORY_DIR,
    PlannedEvaluation,
    SelectionManifest,
    SeriesRef,
    load_manifest,
    load_selection_histories,
    window_digest,
    write_manifest,
)
from crypto_agents.state import (
    Action,
    AgentRole,
    Backend,
    Billing,
    FailureKind,
    IndicatorSet,
    LLMCall,
    MarketSnapshot,
    NodeError,
    Proposal,
    RiskVerdict,
    StructuredOutputMode,
)
from tests.conftest import (
    HEALTHY,
    SCARCE,
    insufficient_funds_error,
    no_route_error,
    timeout_error,
)
from tests.test_ablation import ablation_settings, fixed_clock
from tests.test_metrics import record as bare_record
from tests.test_net_outcomes import mutated
from tests.test_selection import ROOT

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

START = datetime(2026, 8, 1, tzinfo=UTC)
HORIZON = 3
SYMBOLS = ("S0", "S1")
PERIOD = timedelta(hours=1)
STEP = CLUSTER_DAYS * 24
"""Velas de 1 h que hay en un bloque de 7 días."""

EVEN, ODD = 0.012, 0.008
"""Retorno bruto de una compra: 1.2% en las evaluaciones pares y 0.8% en las impares."""

DELTA = 0.002


# ───────────────────────────────── Escenario sintético con respuesta conocida ─────────────────────


def evaluation_return(j: int) -> float:
    return EVEN if j % 2 == 0 else ODD


def symbol_of(j: int) -> str:
    return SYMBOLS[j % 2]


class Slot(NamedTuple):
    """Una entrada de un manifiesto sintético: qué vela de qué símbolo y de qué tramo."""

    symbol: str
    candle: int
    stratum: int


def record_at(symbol: str, index: int, action: Action | None) -> EvaluationRecord:
    """Una evaluación de `symbol` en la vela `index`, con `at` en su cierre, como el replay.

    La apertura es `snapshot.timestamp` y lo que el manifiesto guarda en su entrada; el instante
    evaluado, una vela después. Con los dos iguales —como estaba este escenario— la tabla por
    tramo no veía que `load_plan` indexaba por la apertura.
    """
    opened = START + timedelta(hours=index)
    run_id = uuid5(NAMESPACE_URL, f"{symbol}|{index}")
    proposal = None
    if action is not None:
        stop = {Action.BUY: 96.0, Action.SELL: 104.0, Action.HOLD: None}[action]
        proposal = Proposal(
            action=action,
            confidence=0.5,
            size_fraction=0.0 if action is Action.HOLD else 0.1,
            invalidation_price=stop,
            rationale="Posición sintética para comprobar el evaluador.",
        )
    return EvaluationRecord(
        run_id=run_id,
        at=opened + PERIOD,
        symbol=symbol,
        timeframe="1h",
        snapshot=MarketSnapshot(
            run_id=run_id,
            exchange="binance",
            symbol=symbol,
            timeframe="1h",
            timestamp=opened,
            close=100.0,
            candles_digest="a" * 64,
            candles_count=100,
        ),
        indicators=IndicatorSet(values={"ATRr_14": 2.0}),
        proposal=proposal,
    )


class Scenario:
    """Dos símbolos, velas de 1 h y un retorno conocido por evaluación, repartidas como se diga.

    Lo único que distingue a un escenario de otro es en qué vela cae cada evaluación y a qué
    tramo pertenece. Hay dos:

    - `SPACED`, el de este archivo: 60 evaluaciones, cada una sola en su bloque de 7 días. Con
      una evaluación por clúster el error agrupado es exactamente el ingenuo, así que las cifras
      calculadas a mano con `statistics.stdev` son las del intervalo que juzga. Son 60 porque
      con menos de `MIN_CLUSTERS` clústeres no hay intervalo. Lo que cambia cuando un bloque
      lleva varias lo prueba el escenario agrupado, más abajo.
    - `PACKED`, el que había antes de la enmienda 4: 40 evaluaciones a cuatro velas una de otra,
      todas en dos bloques. Lo conservan las pruebas del recálculo de dispersión, cuyas cifras a
      mano —celdas del pool, acuerdos, n— están escritas sobre él y no leen ningún veredicto.
    """

    def __init__(
        self,
        per_symbol: int,
        index: Callable[[int], int],
        stratum: Callable[[int], int],
        candles: int,
    ) -> None:
        self.n = len(SYMBOLS) * per_symbol
        self._index = index
        self._stratum = stratum
        self.histories = {symbol: self._history(symbol, candles) for symbol in SYMBOLS}

    def candle_index(self, j: int) -> int:
        """La vela de la evaluación `j`. Su salida, tres velas después, no pisa otra entrada."""
        return self._index(j)

    def stratum_of(self, j: int) -> int:
        return self._stratum(j)

    def closes_at(self, j: int) -> datetime:
        """El instante evaluado: el cierre de la vela `j`, que es lo que el replay deja en `at`."""
        return START + timedelta(hours=self.candle_index(j)) + PERIOD

    def _history(self, symbol: str, candles: int) -> list[list[float]]:
        """Velas de 1 h de un símbolo. Cada evaluación entra a 100 y sale a `100 * (1 + r)`.

        Los rangos quedan dentro de [97, 103]: el stop común (`entry -/+ 2 * 2.0`) no se toca
        nunca, así que toda posición sale al cierre del horizonte y su retorno es `+/- r`.
        Solo las evaluaciones de este símbolo mueven su cierre; las del otro lo dejan en 100.
        """
        exits = {
            self.candle_index(j) + HORIZON: 100.0 * (1.0 + evaluation_return(j))
            for j in range(self.n)
            if symbol_of(j) == symbol
        }
        return [
            [
                START.timestamp() * 1000 + index * 3_600_000,
                100.0,
                101.0,
                99.0,
                exits.get(index, 100.0),
                1.0,
            ]
            for index in range(candles)
        ]

    def make_record(self, j: int, action: Action | None) -> EvaluationRecord:
        """La evaluación `j` de un brazo que decide `action`; `None` es una sin decisión."""
        return record_at(symbol_of(j), self.candle_index(j), action)

    def arm(self, rule: str) -> list[EvaluationRecord]:
        """Un brazo sintético: `buy`, `sell`, `hold`, `even` (compra pares) u `odd` (impares)."""
        n = self.n
        actions: dict[str, list[Action]] = {
            "buy": [Action.BUY] * n,
            "sell": [Action.SELL] * n,
            "hold": [Action.HOLD] * n,
            "even": [Action.BUY if j % 2 == 0 else Action.HOLD for j in range(n)],
            "odd": [Action.BUY if j % 2 == 1 else Action.HOLD for j in range(n)],
        }
        return [self.make_record(j, action) for j, action in enumerate(actions[rule])]

    def strata(self) -> dict[tuple[str, datetime], int]:
        """Los tramos como los entrega `load_plan`: por símbolo e instante evaluado."""
        return {(symbol_of(j), self.closes_at(j)): self.stratum_of(j) for j in range(self.n)}

    def slots(self) -> list[Slot]:
        return [Slot(symbol_of(j), self.candle_index(j), self.stratum_of(j)) for j in range(self.n)]

    def write_plan(self, tmp_path: Path) -> Path:
        return write_plan(tmp_path, self.histories, self.slots())

    def write_run(
        self,
        tmp_path: Path,
        arms: dict[str, list[EvaluationRecord]],
        *,
        billing: Billing | None = None,
        kind: RunKind = RunKind.ABLATION,
    ) -> Path:
        return write_run(tmp_path, arms, plan=self.write_plan(tmp_path), billing=billing, kind=kind)


PER_SYMBOL = 30
SPACED = Scenario(
    per_symbol=PER_SYMBOL,
    index=lambda j: STEP * j + 4 * ((j // 2) % 2),
    stratum=lambda j: j // (2 * PER_SYMBOL // 5),
    candles=STEP * (2 * PER_SYMBOL - 1) + 4 + HORIZON + 2,
)
"""Una evaluación cada `STEP` velas. Las de `k = j // 2` impar van 4 velas más tarde: no cambia
su bloque, y hace que la mitad de las ventanas `(entrada, salida]` contenga una hora múltiplo de
8, que es donde `test_net_reports.py` pone las liquidaciones del funding."""

PACKED = Scenario(
    per_symbol=20,
    index=lambda j: 4 * (j // 2),
    stratum=lambda j: (j // 2) // 4,
    candles=4 * 20 + 4,
)
"""Una evaluación de cada símbolo cada 4 velas: la salida (a las 3) no pisa la entrada siguiente."""

N = SPACED.n
HISTORIES = SPACED.histories
candle_index = SPACED.candle_index
stratum_of = SPACED.stratum_of
make_record = SPACED.make_record
arm = SPACED.arm
strata = SPACED.strata
base_slots = SPACED.slots


def expected(diffs: Sequence[float]) -> tuple[float, float, float, float]:
    """Media, error estándar, y extremos del intervalo normal, con `statistics`."""
    mean = statistics.fmean(diffs)
    stderr = statistics.stdev(diffs) / len(diffs) ** 0.5
    return mean, stderr, mean - 1.96 * stderr, mean + 1.96 * stderr


# ──────────────────────────────────────── Veredicto y δ ───────────────────────────────────────────


def spread(half: float) -> PairedDifference:
    """Una diferencia con ese semiancho, centrada donde la prueba diga."""
    return PairedDifference(n=140, mean=0.0, stderr=half / 1.96, low=-half, high=half)


def test_an_interval_that_excludes_zero_on_the_positive_side_justifies() -> None:
    diff = PairedDifference(n=140, mean=0.01, stderr=0.002, low=0.00608, high=0.01392)
    assert classify(diff, DELTA) is Verdict.ABOVE


def test_an_interval_that_excludes_zero_on_the_negative_side_is_worse() -> None:
    diff = PairedDifference(n=140, mean=-0.01, stderr=0.002, low=-0.01392, high=-0.00608)
    assert classify(diff, DELTA) is Verdict.BELOW


def test_a_narrow_interval_around_zero_is_a_tie() -> None:
    assert classify(spread(0.0019), DELTA) is Verdict.TIE


def test_a_wide_interval_around_zero_is_inconclusive_even_though_it_includes_zero() -> None:
    """Semiancho 0.003 > δ 0.002: el IC incluye 0, pero ni siquiera permite hablar de empate."""
    assert classify(spread(0.003), DELTA) is Verdict.INCONCLUSIVE


def test_ignoring_delta_would_call_a_wide_interval_a_tie() -> None:
    """La mutación del criterio: sin δ, todo IC que incluye 0 es «empate».

    Con δ grande el mismo intervalo sí es un empate: lo único que cambia entre los dos
    veredictos es δ, así que ignorarlo hace fallar una de estas dos líneas.
    """
    wide = spread(0.003)
    assert classify(wide, 0.002) is Verdict.INCONCLUSIVE
    assert classify(wide, 0.0035) is Verdict.TIE


def test_the_semiwidth_is_compared_with_delta_strictly() -> None:
    assert classify(spread(DELTA), DELTA) is Verdict.TIE
    assert classify(spread(DELTA + 1e-9), DELTA) is Verdict.INCONCLUSIVE


def test_an_interval_touching_zero_is_not_above() -> None:
    touching = PairedDifference(n=140, mean=0.003, stderr=0.003 / 1.96, low=0.0, high=0.006)
    assert classify(touching, 0.01) is Verdict.TIE


def test_without_a_difference_the_verdict_is_undetermined() -> None:
    assert classify(None, DELTA) is Verdict.UNDETERMINED


def test_delta_must_be_positive() -> None:
    with pytest.raises(CriteriaError, match="delta"):
        classify(spread(0.001), 0.0)
    with pytest.raises(CriteriaError, match="delta"):
        classify(spread(0.001), -0.1)


def test_the_comparison_of_two_vectors_carries_the_numbers_behind_the_verdict() -> None:
    """Retornos ±0.01 contra cero, cada uno en su clúster: media 0, semiancho 0.002552 a mano.

    `1.96 · 0.01 · √(60/59) / √60`, con n = 60. Una por clúster, el agrupado es el ingenuo.
    """
    arm_returns = [0.01 if j % 2 == 0 else -0.01 for j in range(60)]
    alone = list(range(60))
    comparison = compare_returns("a", "b", arm_returns, [0.0] * 60, alone, 0.0025)

    assert comparison.n == 60
    assert comparison.diff is not None
    mean, stderr, low, high = expected(arm_returns)
    assert comparison.diff.mean == pytest.approx(mean)
    assert comparison.diff.stderr == pytest.approx(stderr)
    assert (comparison.diff.low, comparison.diff.high) == pytest.approx((low, high))
    assert comparison.half_width == pytest.approx(0.0025517, abs=1e-6)
    assert comparison.verdict is Verdict.INCONCLUSIVE
    tie = compare_returns("a", "b", arm_returns, [0.0] * 60, alone, 0.0026)
    assert tie.verdict is Verdict.TIE


def test_fewer_than_thirty_pairs_have_no_interval_and_say_why() -> None:
    comparison = compare_returns("a", "b", [0.01] * 29, [0.0] * 29, list(range(29)), DELTA)
    assert comparison.verdict is Verdict.UNDETERMINED
    assert comparison.diff is None
    assert comparison.why is not None
    assert "30" in comparison.why


# ───────────────────────────────────── Criterios 1 y 3 ────────────────────────────────────────────


def criteria_run() -> dict[str, list[EvaluationRecord]]:
    return {
        "full": arm("buy"),
        "solo": arm("odd"),
        "no_debate": arm("buy"),
        "bull_only": arm("sell"),
        "always_buy": arm("even"),
        "always_sell": arm("sell"),
        "random_uniform": arm("buy"),
        "rule_trend": arm("buy"),
    }


def report() -> CriteriaReport:
    return evaluate_criteria(criteria_run(), HISTORIES, HORIZON, DELTA, strata())


def test_criterion_one_compares_full_with_the_three_other_shapes() -> None:
    comparisons = {c.reference: c for c in report().criterion_1}

    assert set(comparisons) == {"solo", "no_debate", "bull_only"}
    assert all(c.arm == "full" for c in comparisons.values())


def test_criterion_one_numbers_match_the_hand_computed_differences() -> None:
    """`full` compra todo y `solo` las impares: la diferencia es `r` en las pares y 0 en las otras.

    Pares: 0.012, impares: 0 → media 0.006. Contra `bull_only`, que vende todo, es
    `2r`: 0.024 y 0.016.
    """
    comparisons = {c.reference: c for c in report().criterion_1}

    vs_solo = comparisons["solo"]
    mean, stderr, low, high = expected([EVEN if j % 2 == 0 else 0.0 for j in range(N)])
    assert vs_solo.n == N
    assert vs_solo.diff is not None
    assert (vs_solo.diff.mean, vs_solo.diff.stderr) == pytest.approx((mean, stderr))
    assert (vs_solo.diff.low, vs_solo.diff.high) == pytest.approx((low, high))
    assert vs_solo.verdict is Verdict.ABOVE

    vs_sell = comparisons["bull_only"]
    assert vs_sell.diff is not None
    assert vs_sell.diff.mean == pytest.approx(
        statistics.fmean(2 * evaluation_return(j) for j in range(N))
    )
    assert vs_sell.verdict is Verdict.ABOVE


def test_an_identical_arm_is_a_tie_and_a_tie_is_not_a_justification() -> None:
    """`no_debate` decide como `full`: diferencia cero, intervalo de ancho cero, empate."""
    comparison = next(c for c in report().criterion_1 if c.reference == "no_debate")
    assert comparison.diff is not None
    assert (comparison.diff.mean, comparison.diff.stderr) == (0.0, 0.0)
    assert comparison.verdict is Verdict.TIE


def test_criterion_three_compares_every_arm_with_each_of_the_four_baselines() -> None:
    tables = {t.arm: t for t in report().criterion_3}

    assert set(tables) == {"full", "solo", "no_debate", "bull_only"}
    for table in tables.values():
        assert [c.reference for c in table.comparisons] == list(BASELINE_ARMS)


def test_the_baselines_are_not_compared_with_themselves() -> None:
    assert {t.arm for t in report().criterion_3}.isdisjoint(BASELINE_ARMS)


def test_criterion_three_verdicts_follow_the_known_differences() -> None:
    """`full` compra todo.

    - contra `always_buy` (solo pares): gana en las impares, 0.008 → media 0.004, supera;
    - contra `always_sell`: `2r`, supera;
    - contra `random_uniform` y `rule_trend`, que deciden igual que `full`: empate.
    """
    full = next(t for t in report().criterion_3 if t.arm == "full")
    verdicts = {c.reference: c.verdict for c in full.comparisons}

    assert verdicts == {
        "always_buy": Verdict.ABOVE,
        "always_sell": Verdict.ABOVE,
        "random_uniform": Verdict.TIE,
        "rule_trend": Verdict.TIE,
    }
    assert full.beaten == 2
    assert not full.beats_all


def test_an_arm_that_does_worse_than_a_baseline_is_marked_worse() -> None:
    """`solo` compra las impares: a los baselines que compran todo les deja las pares."""
    solo = next(t for t in report().criterion_3 if t.arm == "solo")
    verdicts = {c.reference: c.verdict for c in solo.comparisons}

    # contra `always_buy` (pares): -0.002 de media, con semiancho 0.0031 > δ: ni empate ni derrota
    assert verdicts["always_buy"] is Verdict.INCONCLUSIVE
    assert verdicts["always_sell"] is Verdict.ABOVE
    assert verdicts["random_uniform"] is Verdict.BELOW
    assert verdicts["rule_trend"] is Verdict.BELOW
    assert solo.beaten == 1


def test_beating_all_four_baselines_is_a_conjunction_and_not_a_best_of() -> None:
    """Cada comparación es contra cada una: no hay «mejor de las líneas base» que calcular."""
    run = criteria_run()
    run["always_buy"] = arm("hold")
    run["random_uniform"] = arm("hold")
    run["rule_trend"] = arm("sell")
    table = next(
        t
        for t in evaluate_criteria(run, HISTORIES, HORIZON, DELTA, strata()).criterion_3
        if t.arm == "full"
    )

    assert table.beaten == 4
    assert table.beats_all


def test_the_aligned_pairs_are_the_evaluations_both_arms_have() -> None:
    """Se empareja por `run_id`, no por posición: a `solo` le falta una evaluación."""
    run = criteria_run()
    run["solo"] = [r for j, r in enumerate(run["solo"]) if j != 7]
    comparison = next(
        c
        for c in evaluate_criteria(run, HISTORIES, HORIZON, DELTA, strata()).criterion_1
        if c.reference == "solo"
    )

    assert comparison.n == N - 1
    assert comparison.unpaired == 1


def test_the_paired_returns_are_the_vectors_the_criteria_subtract() -> None:
    """La puerta pública entrega lo que `evaluate_criteria` resta: los dos vectores por `run_id`,
    con lo que cada brazo dejó de cada evaluación. Barajar un journal no cambia nada, y la media
    y el error de la diferencia son los de la comparación publicada."""
    run = criteria_run()
    shuffled = {**run, "solo": list(reversed(run["solo"]))}

    pair = paired_returns("full", "solo", shuffled, HISTORIES, HORIZON)

    assert isinstance(pair, PairedReturns)
    assert pair == paired_returns("full", "solo", run, HISTORIES, HORIZON)
    assert [r.run_id for r in pair.arm_records] == [r.run_id for r in pair.reference_records]
    assert {r.run_id for r in pair.arm_records} == {r.run_id for r in run["full"]}
    by_id = {r.run_id: j for j, r in enumerate(run["full"])}
    for record, ours, theirs in zip(
        pair.arm_records, pair.arm_returns, pair.reference_returns, strict=True
    ):
        j = by_id[record.run_id]
        assert ours == pytest.approx(evaluation_return(j))
        assert theirs == pytest.approx(evaluation_return(j) if j % 2 == 1 else 0.0)

    published = next(
        c
        for c in evaluate_criteria(run, HISTORIES, HORIZON, DELTA).criterion_1
        if c.reference == "solo"
    )
    assert published.diff is not None
    assert published.naive is not None
    differences = metrics_module.paired_differences(pair.arm_returns, pair.reference_returns)
    assert published.diff.mean == statistics.fmean(differences)
    assert published.naive.stderr == statistics.stdev(differences) / len(differences) ** 0.5
    assert published.diff.stderr == pytest.approx(published.naive.stderr)


def test_the_paired_returns_say_why_when_an_arm_is_missing() -> None:
    run = {"full": arm("buy")}

    assert paired_returns("full", "solo", run, HISTORIES, HORIZON) == (
        "el brazo solo no tiene evaluaciones"
    )
    assert paired_returns("full", "solo", {**run, "solo": []}, HISTORIES, HORIZON) == (
        "el brazo solo no tiene evaluaciones"
    )


def test_arms_that_are_not_in_the_run_are_undetermined_and_not_zero() -> None:
    run = {"full": arm("buy"), "always_buy": arm("buy")}
    result = evaluate_criteria(run, HISTORIES, HORIZON, DELTA, strata())

    assert {c.verdict for c in result.criterion_1} == {Verdict.UNDETERMINED}
    full = next(t for t in result.criterion_3 if t.arm == "full")
    missing = [c for c in full.comparisons if c.reference != "always_buy"]
    assert {c.verdict for c in missing} == {Verdict.UNDETERMINED}
    assert all("no tiene evaluaciones" in (c.why or "") for c in missing)


def test_records_without_indicators_make_the_comparison_undetermined() -> None:
    """Sin ATR no hay stop común: un cero silencioso sesgaría la diferencia."""
    run = criteria_run()
    run["solo"] = [r.model_copy(update={"indicators": None}) for r in run["solo"]]
    comparison = next(
        c
        for c in evaluate_criteria(run, HISTORIES, HORIZON, DELTA, strata()).criterion_1
        if c.reference == "solo"
    )

    assert comparison.verdict is Verdict.UNDETERMINED
    assert "indicadores" in (comparison.why or "")


def test_a_duplicated_evaluation_in_a_journal_is_an_error() -> None:
    run = criteria_run()
    run["solo"] = [*run["solo"], run["solo"][0]]
    with pytest.raises(CriteriaError, match="repetida"):
        evaluate_criteria(run, HISTORIES, HORIZON, DELTA, strata())


# ──────────────────────────────── Diferencia pareada descriptiva ──────────────────────────────────


def test_the_descriptive_breakdown_has_counts_and_means_and_no_verdict() -> None:
    """`full` contra `solo`: la diferencia solo existe en las impares, que son del símbolo S1."""
    descriptive = {d.reference: d for d in report().descriptive}["solo"]

    by_symbol = {cell.label: cell for cell in descriptive.by_symbol}
    assert set(by_symbol) == set(SYMBOLS)
    assert by_symbol["S0"].n == PER_SYMBOL
    assert by_symbol["S0"].mean == pytest.approx(EVEN)
    assert by_symbol["S1"].n == PER_SYMBOL
    assert by_symbol["S1"].mean == pytest.approx(0.0)
    assert not hasattr(by_symbol["S0"], "verdict")


def test_the_descriptive_breakdown_by_span_uses_the_manifest_strata() -> None:
    descriptive = {d.reference: d for d in report().descriptive}["solo"]

    assert descriptive.by_stratum is not None
    cells = {cell.label: cell for cell in descriptive.by_stratum}
    assert set(cells) == {str(i) for i in range(5)}
    assert {cell.n for cell in cells.values()} == {N // 5}
    assert {round(cell.mean, 9) for cell in cells.values()} == {round(EVEN / 2, 9)}


def test_without_a_manifest_the_spans_are_not_determined() -> None:
    result = evaluate_criteria(criteria_run(), HISTORIES, HORIZON, DELTA, None)
    assert all(d.by_stratum is None for d in result.descriptive)


# ───────────────────────────────── Criterio 6: la corrida es válida ───────────────────────────────


def live_call(
    role: AgentRole = AgentRole.STRUCTURE,
    backend: Backend = Backend.OPENAI,
    cache_hit: bool = False,
    at: datetime = START,
    weight: float = 1.0,
    model: str = "modelo",
) -> LLMCall:
    return LLMCall(
        role=role,
        backend=backend,
        model=model,
        structured_output=StructuredOutputMode.JSON_SCHEMA,
        quota_weight=weight,
        prompt_digest="a" * 64,
        cache_hit=cache_hit,
        valid=True,
        latency_ms=0.0 if cache_hit else 10.0,
        at=at,
    )


EXPECTED = {
    "full": dict.fromkeys(AgentRole, Backend.OPENAI),
    "local_technicals": {
        **dict.fromkeys(AgentRole, Backend.OPENAI),
        AgentRole.STRUCTURE: Backend.OLLAMA,
        AgentRole.MOMENTUM: Backend.OLLAMA,
        AgentRole.VOLUME: Backend.OLLAMA,
    },
}


def quota_lost(node: str = "decide") -> EvaluationRecord:
    message = f"{node}: {QuotaExhaustedError(AgentRole.DECIDER, ('glm-5.2',))}"
    return bare_record(errors=(NodeError(node=node, message=message, at=START),))


def funds_lost(node: str = "bull") -> EvaluationRecord:
    """Una evaluación que el proveedor rechazó con el 402 real de Zen, en el nodo indicado."""
    message = f"{node}: {ModelCallError(AgentRole.BULL, SCARCE, insufficient_funds_error())}"
    return bare_record(errors=(NodeError(node=node, message=message, at=START),))


PROVIDER_ROLE = {
    "structure": AgentRole.STRUCTURE,
    "momentum": AgentRole.MOMENTUM,
    "volume": AgentRole.VOLUME,
    "bull": AgentRole.BULL,
    "bear": AgentRole.BEAR,
    "decide": AgentRole.DECIDER,
}


def provider_lost(node: str, kind: FailureKind) -> EvaluationRecord:
    """Una evaluación que el proveedor se llevó en ese nodo: un 404 real de Zen, o un plazo vencido.

    El registro lleva el intento fallido, como lo deja `_model_failure`: el mensaje del router es
    el mismo para un rechazo y para un plazo, y lo que los distingue es el tipo de la fila.
    """
    role = PROVIDER_ROLE[node]
    cause = no_route_error() if kind is FailureKind.TRANSPORT else timeout_error()
    failed = live_call(role).model_copy(
        update={"valid": False, "failure_kind": kind, "failure_message": str(cause)}
    )
    message = f"{node}: {ModelCallError(role, SCARCE, cause, (failed,))}"
    return bare_record(calls=(failed,), errors=(NodeError(node=node, message=message, at=START),))


def cap_lost(node: str = "decide") -> EvaluationRecord:
    """Una evaluación que el tope de gasto no dejó abrir en ese nodo: la excepción real del router.

    Sin llamadas: el tope se mira antes del primer intento vivo, así que no se gastó nada.
    """
    message = f"{node}: {SpendCapReachedError(PROVIDER_ROLE[node], 5.0, 5.0123)}"
    return bare_record(errors=(NodeError(node=node, message=message, at=START),))


def test_a_clean_run_has_nothing_to_report() -> None:
    result = check_validity({"full": [bare_record()]}, EXPECTED)
    assert result.valid
    assert result.reasons == []


def test_a_decider_evaluation_lost_to_quota_invalidates_the_run() -> None:
    result = check_validity({"full": [bare_record(), quota_lost("decide")]}, EXPECTED)

    assert not result.valid
    assert len(result.reasons) == 1
    assert "cuota" in result.reasons[0]
    assert "full 1" in result.reasons[0]
    assert result.arms[0].decider_quota_lost == 1


@pytest.mark.parametrize("node", ["structure", "bull", "bear", "decide"])
def test_an_evaluation_lost_to_insufficient_funds_invalidates_the_run_at_any_node(
    node: str,
) -> None:
    """El saldo es de la cuenta: se acaba para todos los roles a la vez, no solo para el decisor."""
    result = check_validity({"full": [bare_record(), funds_lost(node)]}, EXPECTED)

    assert not result.valid
    assert result.arms[0].funds_lost == 1
    assert result.arms[0].decider_quota_lost == 0
    assert len(result.reasons) == 1
    assert "saldo insuficiente" in result.reasons[0]
    assert "full 1" in result.reasons[0]


@pytest.mark.parametrize("node", sorted(PROVIDER_ROLE))
@pytest.mark.parametrize("kind", [FailureKind.TRANSPORT, FailureKind.TIMEOUT])
def test_an_evaluation_lost_to_the_provider_invalidates_the_run_at_any_node(
    node: str, kind: FailureKind
) -> None:
    """La quinta condición: un rechazo o un plazo vencido, también en un nodo técnico.

    `resolve()` no degrada por transporte: con la ruta de `momentum` caída, la evaluación aborta
    en `consolidate_evidence` y lo que la corrida compara son los brazos que no la usaban.
    """
    result = check_validity({"full": [bare_record(), provider_lost(node, kind)]}, EXPECTED)

    assert not result.valid
    assert result.arms[0].provider_lost == {f"{node}/{kind.value}": 1}
    assert (result.arms[0].funds_lost, result.arms[0].decider_quota_lost) == (0, 0)
    assert len(result.reasons) == 1
    assert "fallo del proveedor" in result.reasons[0]
    assert f"{node}/{kind.value}: full 1" in result.reasons[0]


@pytest.mark.parametrize("node", sorted(PROVIDER_ROLE))
def test_an_evaluation_cut_by_the_spend_cap_invalidates_the_run_at_any_node(node: str) -> None:
    """La sexta condición (enmienda 3): el tope es de la corrida, no de un nodo ni de un brazo.

    Lo que `--max-usd` no dejó terminar es una evaluación que el brazo no tuvo ocasión de decidir,
    igual que la que se llevó un 402; puntuarla 0 sería medir el tope y no los modelos.
    """
    result = check_validity({"full": [bare_record(), cap_lost(node)]}, EXPECTED)

    assert not result.valid
    assert result.arms[0].spend_cap_lost == 1
    assert result.arms[0].funds_lost == 0
    assert result.arms[0].provider_lost == {}
    assert result.arms[0].decider_quota_lost == 0
    assert len(result.reasons) == 1
    assert "tope de gasto" in result.reasons[0]
    assert "full 1" in result.reasons[0]


def test_the_cap_column_is_in_the_validity_table() -> None:
    text = render_invalid(check_validity({"full": [cap_lost("bull")]}, EXPECTED), "var/ablation/x")

    assert text.splitlines()[0].startswith("CORRIDA INVÁLIDA: ")
    assert "| tope de gasto |" in text
    assert "| full | 0 | 0 | 0 | 1 | 0 | 0 | — |" in text
    assert not any(word in text for word in VERDICT_WORDS)


def test_a_402_is_still_counted_as_funds_and_not_twice() -> None:
    """Su `failure_kind` es `transport`, pero es la cuarta condición y no la quinta."""
    result = check_validity({"full": [funds_lost("momentum")]}, EXPECTED)

    assert result.arms[0].funds_lost == 1
    assert result.arms[0].provider_lost == {}
    assert len(result.reasons) == 1


def test_a_validation_loss_is_the_models_and_does_not_invalidate() -> None:
    """Lo que la enmienda deja fuera a propósito: agotar los intentos es parte de lo que se mide."""
    exhausted = bare_record(
        errors=(
            NodeError(
                node="momentum",
                message="momentum: 2 intento(s) sin salida válida; último error: x",
                at=START,
            ),
        )
    )
    result = check_validity({"full": [exhausted]}, EXPECTED)

    assert result.valid
    assert result.arms[0].provider_lost == {}


def test_the_provider_column_is_in_the_validity_table() -> None:
    lost = provider_lost("momentum", FailureKind.TRANSPORT)
    text = render_invalid(check_validity({"full": [lost]}, EXPECTED), "var/ablation/x")

    assert text.splitlines()[0].startswith("CORRIDA INVÁLIDA: ")
    assert "| fallo del proveedor |" in text
    assert "| full | 0 | 0 | momentum/transport 1 | 0 | 0 | 0 | — |" in text
    assert not any(word in text for word in VERDICT_WORDS)


def test_the_funds_column_is_in_the_validity_table() -> None:
    text = render_invalid(check_validity({"full": [funds_lost()]}, EXPECTED), "var/ablation/x")

    assert text.splitlines()[0].startswith("CORRIDA INVÁLIDA: ")
    assert "saldo insuficiente (402)" in text
    assert not any(word in text for word in VERDICT_WORDS)


def assert_a_402_is_reported_as_funds() -> None:
    result = check_validity({"full": [funds_lost()]}, EXPECTED)
    assert not result.valid
    assert result.arms[0].funds_lost == 1, "el 402 se cuenta como saldo"
    assert "saldo insuficiente" in result.reasons[0]


def test_mutation_a_402_read_as_generic_transport_loses_its_name_and_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sin la categoría, el 402 sería un rechazo cualquiera: inválida igual, pero por otra razón.

    Hasta el bloque T5 esa mutación dejaba la corrida *válida*, porque un rechazo de transporte
    no invalidaba. Con la quinta condición la corrida sigue saliendo inválida, y lo que la
    categoría protege es el nombre: «se acabó el saldo» no se arregla reanudando, «el proveedor
    rechazó» puede que sí, y quien lee el motivo decide con él qué hacer.
    """
    assert_a_402_is_reported_as_funds()  # control: el código real lo nombra
    without_funds = tuple(
        marker
        for marker in metrics_module._ABORT_MARKERS
        if marker[1] is not AbortKind.INSUFFICIENT_FUNDS
    )
    monkeypatch.setattr(metrics_module, "_ABORT_MARKERS", without_funds)
    with pytest.raises(AssertionError, match="el 402 se cuenta como saldo"):
        assert_a_402_is_reported_as_funds()
    mislabelled = check_validity({"full": [funds_lost()]}, EXPECTED)
    assert not mislabelled.valid
    assert "fallo del proveedor" in mislabelled.reasons[0]


@pytest.mark.parametrize("node", sorted(DECIDER_NODES))
def test_every_decider_node_counts(node: str) -> None:
    assert not check_validity({"full": [quota_lost(node)]}, EXPECTED).valid


def test_the_decider_nodes_are_exactly_the_ones_the_ablation_counts_as_decider() -> None:
    """`criteria.py` no importa la ablación; esta prueba es lo que impide que diverjan."""
    from crypto_agents.ablation import _LLM_NODES

    assert {n for n, (role, _) in _LLM_NODES.items() if role is AgentRole.DECIDER} == DECIDER_NODES


def test_a_quota_loss_in_another_role_warns_without_invalidating() -> None:
    """`bear` no tiene respaldo y puede perder la evaluación por cuota: se avisa, no se invalida."""
    result = check_validity({"full": [quota_lost("bear")]}, EXPECTED)

    assert result.valid
    assert result.arms[0].other_quota_lost == {"bear": 1}
    assert "bear" in "\n".join(result.warnings)


def test_an_abort_for_another_reason_is_not_a_quota_loss() -> None:
    other = bare_record(
        errors=(
            NodeError(
                node="decide",
                message="decider: 2 intento(s) sin salida válida; último error: x",
                at=START,
            ),
        )
    )
    assert check_validity({"full": [other]}, EXPECTED).valid


def test_a_cache_hit_from_another_backend_invalidates_the_run() -> None:
    """Un acierto de caché local en un brazo que dice llamar al remoto: no es el brazo que dice."""
    hit = live_call(AgentRole.STRUCTURE, Backend.OLLAMA, cache_hit=True)
    result = check_validity({"full": [bare_record(calls=(hit,))]}, EXPECTED)

    assert not result.valid
    assert result.arms[0].cache_backend_mismatch == 1
    assert "caché" in result.reasons[0]


def test_the_expected_backend_is_the_one_of_that_arm_and_not_the_base_one() -> None:
    """`local_technicals` pide los técnicos en local: un acierto local ahí es lo correcto."""
    hit = live_call(AgentRole.STRUCTURE, Backend.OLLAMA, cache_hit=True)
    assert check_validity({"local_technicals": [bare_record(calls=(hit,))]}, EXPECTED).valid

    remote = live_call(AgentRole.STRUCTURE, Backend.OPENAI, cache_hit=True)
    assert not check_validity({"local_technicals": [bare_record(calls=(remote,))]}, EXPECTED).valid


def test_a_live_call_degraded_to_another_backend_warns_without_invalidating() -> None:
    """Degradar por cuota es del diseño; solo un acierto de caché ajeno contamina el brazo."""
    degraded = live_call(AgentRole.STRUCTURE, Backend.OLLAMA)
    result = check_validity({"full": [bare_record(calls=(degraded,))]}, EXPECTED)

    assert result.valid
    assert result.arms[0].live_backend_mismatch == 1
    assert result.warnings


def test_matching_backends_are_clean_whether_cached_or_live() -> None:
    calls = (
        live_call(AgentRole.STRUCTURE, Backend.OPENAI, cache_hit=True),
        live_call(AgentRole.DECIDER, Backend.OPENAI),
    )
    assert check_validity({"full": [bare_record(calls=calls)]}, EXPECTED).valid


def vetoed(rule: str) -> EvaluationRecord:
    return bare_record(
        risk=RiskVerdict(
            approved=False, final_size_fraction=0.0, veto_rule=rule, veto_reason="prueba"
        )
    )


def test_the_only_veto_the_run_may_have_is_the_invalid_stop_side() -> None:
    assert check_validity({"full": [vetoed(INVALID_STOP_SIDE)]}, EXPECTED).valid


@pytest.mark.parametrize(
    "rule", ["kill_switch", "daily_drawdown", "cooldown", NO_EXPOSURE_HEADROOM]
)
def test_any_other_veto_invalidates_the_run(rule: str) -> None:
    """La mutación: dejar pasar un veto que no sea `invalid_stop_side` haría fallar esto."""
    result = check_validity({"full": [vetoed(rule)]}, EXPECTED)

    assert not result.valid
    assert result.arms[0].foreign_vetoes == {rule: 1}
    assert rule in result.reasons[0]


def test_every_failing_condition_is_named_and_none_hides_another() -> None:
    hit = live_call(AgentRole.STRUCTURE, Backend.OLLAMA, cache_hit=True)
    result = check_validity(
        {"full": [quota_lost(), vetoed("kill_switch"), bare_record(calls=(hit,))]}, EXPECTED
    )

    assert len(result.reasons) == 3
    assert [r for r in result.reasons if "cuota" in r]
    assert [r for r in result.reasons if "caché" in r]
    assert [r for r in result.reasons if "veto" in r]


def test_an_arm_the_meta_does_not_describe_cannot_be_checked() -> None:
    with pytest.raises(CriteriaError, match=r"meta\.json"):
        check_validity({"desconocido": [bare_record()]}, EXPECTED)


# ───────────────────────────────── Pico de consumo en la ventana de 5 h ───────────────────────────

WINDOW = timedelta(hours=5)


def at(hours: float) -> datetime:
    return START + timedelta(hours=hours)


def peak(calls: Sequence[LLMCall], role: AgentRole = AgentRole.STRUCTURE) -> PeakWindow:
    (found,) = peak_window_usage(calls, role, WINDOW)
    return found


def test_the_peak_is_the_most_calls_in_any_sliding_window() -> None:
    """Llamadas en 0, 1, 4.99 y 5 h: la ventana que termina en 5 h deja fuera la de 0 h."""
    calls = [live_call(at=at(h)) for h in (0, 1, 4.99, 5)]
    found = peak(calls)

    assert found.calls == 3
    assert found.quota == 3.0


def test_the_window_edge_is_open_on_the_old_side_like_the_ledger() -> None:
    """Una llamada exactamente 5 h antes ya no cuenta, igual que `QuotaLedger._purge`.

    La mutación —contarla— daría 2 y no 1.
    """
    found = peak([live_call(at=at(0)), live_call(at=at(5))])
    assert found.calls == 1


def test_the_peak_adds_quota_weights_and_not_just_calls() -> None:
    found = peak([live_call(at=at(h), weight=2.0) for h in (0, 1, 2)])
    assert found.calls == 3
    assert found.quota == 6.0


def test_cache_hits_and_local_calls_do_not_count() -> None:
    calls = [
        live_call(at=at(0)),
        live_call(at=at(0.5), cache_hit=True),
        live_call(at=at(1), backend=Backend.OLLAMA),
    ]
    assert peak(calls).calls == 1


def test_each_model_of_a_role_has_its_own_peak() -> None:
    calls = [live_call(at=at(h), model="a") for h in (0, 1)] + [live_call(at=at(2), model="b")]
    found = {p.model: p for p in peak_window_usage(calls, AgentRole.STRUCTURE, WINDOW)}

    assert (found["a"].calls, found["b"].calls) == (2, 1)


def test_a_role_with_no_remote_calls_has_no_peak() -> None:
    assert peak_window_usage([live_call(role=AgentRole.BULL)], AgentRole.STRUCTURE, WINDOW) == ()


def test_the_peak_does_not_depend_on_the_order_the_calls_arrive_in() -> None:
    calls = [live_call(at=at(h)) for h in (3, 0, 2, 1, 4)]
    assert peak(calls).calls == 5


def test_the_peak_says_where_it_happened() -> None:
    found = peak([live_call(at=at(h)) for h in (0, 10, 11, 12)])
    assert found.calls == 3
    assert found.end == at(12)


def test_the_warning_threshold_is_eighty_percent_of_the_quota() -> None:
    assert PEAK_WARNING_SHARE == 0.80
    calls = [live_call(at=at(i / 100), model="m") for i in range(81)]
    limits = {(AgentRole.STRUCTURE, "m"): 100}

    warned = render_peaks(calls, WINDOW, limits)
    assert "AVISO" in warned
    quiet = render_peaks(calls[:80], WINDOW, limits)
    assert "AVISO" not in quiet
    assert "80 de 100" in quiet


def test_the_peak_is_printed_even_when_the_quota_is_not_known() -> None:
    text = render_peaks([live_call(at=at(0))], WINDOW, {})
    assert "no determinado" in text
    assert "AVISO" not in text


def assert_payg_peaks_do_not_measure(render: Callable[..., Any]) -> None:
    """81 de 100 es un AVISO con Go; con `payg` el 100 es un centinela y no se compara."""
    calls = [live_call(at=at(i / 100), model="m") for i in range(81)]
    limits = {(AgentRole.STRUCTURE, "m"): 100}
    text = render(calls, WINDOW, limits, Billing.PAYG)

    assert "no aplica (payg)" in text
    assert "AVISO" not in text
    assert "%" not in text
    assert "| 81 | 81 |" in text  # el pico es una medida y se queda


def test_under_payg_the_peak_is_measured_but_not_compared_with_the_sentinel() -> None:
    assert_payg_peaks_do_not_measure(render_peaks)


def test_a_run_that_did_not_record_its_billing_is_compared_as_before() -> None:
    calls = [live_call(at=at(i / 100), model="m") for i in range(81)]
    limits = {(AgentRole.STRUCTURE, "m"): 100}
    for billing in (None, Billing.GO):
        text = render_peaks(calls, WINDOW, limits, billing)
        assert "AVISO" in text
        assert "81 de 100" in text


def test_mutation_computing_the_peak_share_under_payg_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mutant = mutated(render_peaks, "if billing is Billing.PAYG:", "if False:", criteria_module)
    monkeypatch.setattr(criteria_module, "render_peaks", mutant)
    with pytest.raises(AssertionError):
        assert_payg_peaks_do_not_measure(criteria_module.render_peaks)


# ─────────────────────────────────────────── Informe ──────────────────────────────────────────────

VERDICT_WORDS = ("justifica", "supera", "empate", "no concluyente", "peor")


def test_the_report_shows_every_comparison_with_its_numbers() -> None:
    text = render_criteria(report(), "var/ablation/x")

    assert "## Criterio 1" in text
    assert "## Criterio 3" in text
    assert "δ = 0.200% de retorno por evaluación" in text
    for reference in ("solo", "no_debate", "bull_only"):
        assert any(line.startswith(f"| full | {reference} ") for line in text.splitlines())
    assert text.count("always_buy") >= 4
    full_vs_solo = next(line for line in text.splitlines() if line.startswith("| full | solo "))
    assert "justifica" in full_vs_solo
    assert f"| {N} |" in full_vs_solo


def test_the_report_says_how_many_baselines_each_arm_beats() -> None:
    text = render_criteria(report(), "d")
    assert "supera a 2 de 4" in text
    assert "supera a 1 de 4" in text


def test_the_descriptive_section_says_it_has_no_verdict() -> None:
    text = render_criteria(report(), "d")
    section = text.split("## Diferencia pareada descriptiva")[1]
    assert "sin veredicto" in section
    assert not any(word in section.split("\n\n", 1)[1] for word in ("justifica", "no concluyente"))


def test_an_invalid_run_prints_the_reason_first_and_no_verdict() -> None:
    result = check_validity({"full": [quota_lost()]}, EXPECTED)
    text = render_invalid(result, "var/ablation/x")

    assert text.splitlines()[0].startswith("CORRIDA INVÁLIDA: ")
    assert "cuota" in text.splitlines()[0]
    assert "## Criterio 1" not in text
    assert "## Criterio 3" not in text
    assert not any(word in text for word in VERDICT_WORDS)


# ──────────────────────────────────── El comando, de un directorio a otro ─────────────────────────


def write_plan(
    tmp_path: Path,
    histories: dict[str, list[list[float]]] | None = None,
    slots: Sequence[Slot] | None = None,
) -> Path:
    """Manifiesto, históricos y su digest, exactamente como los lee la ablación.

    Cada entrada lleva en `at` la **apertura** de su vela, como las de un manifiesto real; el
    instante evaluado, que es lo que llevan los registros, es una vela después.
    """
    histories = HISTORIES if histories is None else histories
    slots = base_slots() if slots is None else slots
    history_dir = tmp_path / "history"
    for symbol, rows in histories.items():
        write_ohlcv_csv(history_dir / f"{symbol.lower()}_1h.csv", rows)
    entries = tuple(
        PlannedEvaluation(
            symbol=slot.symbol,
            timeframe="1h",
            index=slot.candle,
            at=START + timedelta(hours=slot.candle),
            candles_digest=window_digest(histories[slot.symbol], slot.candle, 5),
            stratum=slot.stratum,
        )
        for slot in slots
    )
    series = tuple(
        SeriesRef(
            symbol=symbol,
            timeframe="1h",
            bars=len(rows),
            first=START,
            last=START + timedelta(hours=len(rows) - 1),
            digest="b" * 64,
            activations=sum(1 for slot in slots if slot.symbol == symbol),
        )
        for symbol, rows in histories.items()
    )
    manifest = SelectionManifest(
        created_at=START,
        seed=1,
        target=len(slots),
        strata=5,
        timeframe="1h",
        warmup_bars=1,
        candle_limit=5,
        horizon=HORIZON,
        series=series,
        entries=entries,
    )
    path = tmp_path / "manifest.json"
    write_manifest(path, manifest)
    return path


def role_meta(backend: Backend = Backend.OPENAI, quota: int = 1000) -> dict[AgentRole, RoleMeta]:
    return {
        role: RoleMeta(
            model=f"modelo-{role.value}",
            backend=backend,
            temperature=0.0,
            quota_per_window=quota,
            quota_weight=1.0,
        )
        for role in AgentRole
    }


def write_run(
    tmp_path: Path,
    arms: dict[str, list[EvaluationRecord]],
    *,
    plan: Path | None = None,
    arm_roles: bool = True,
    horizon: int | None = HORIZON,
    billing: Billing | None = None,
    kind: RunKind = RunKind.ABLATION,
) -> Path:
    plan = plan if plan is not None else write_plan(tmp_path)
    directory = tmp_path / "corrida"
    meta = RunMeta(
        kind=kind,
        plan_kind=PlanKind.MANIFEST,
        plan_path=str(plan),
        plan_sha256=file_sha256(plan),
        argv=("--manifest", str(plan), "--fill"),
        fill=True,
        arms=tuple(arms),
        started_at=START,
        arm_roles={name: role_meta() for name in arms} if arm_roles else None,
        kill_switch=False,
        quota_window=WINDOW,
        horizon=horizon,
        billing=billing,
    )
    write_meta(directory, meta)
    for name, records in arms.items():
        journal = JsonlJournal(arm_journal_path(directory, name))
        for item in records:
            journal.write(item)
    return directory


def run_command(directory: Path, tmp_path: Path, *extra: str) -> int:
    return main(
        [str(directory), "--delta", str(DELTA), "--history-dir", str(tmp_path / "history"), *extra]
    )


def test_a_valid_run_prints_the_verdicts_and_exits_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = write_run(tmp_path, criteria_run())

    assert run_command(directory, tmp_path) == 0

    out = capsys.readouterr().out
    assert "CORRIDA INVÁLIDA" not in out
    assert "## Criterio 1" in out
    assert "## Criterio 3" in out
    assert "## Ventana de 5 h" in out
    full_vs_solo = next(line for line in out.splitlines() if line.startswith("| full | solo "))
    assert "justifica" in full_vs_solo


def test_the_command_prints_not_applicable_in_the_peak_table_of_a_payg_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arms = criteria_run()
    arms["full"] = [
        arms["full"][0].model_copy(
            update={"calls": (live_call(AgentRole.DECIDER, at=at(0), model="modelo-decider"),)}
        ),
        *arms["full"][1:],
    ]
    directory = write_run(tmp_path, arms, billing=Billing.PAYG)

    assert run_command(directory, tmp_path) == 0

    out = capsys.readouterr().out
    decider = next(line for line in out.splitlines() if line.startswith("| decider"))
    assert "no aplica (payg)" in decider
    assert "%" not in decider


def test_a_synthetic_quota_failure_exits_one_and_prints_no_verdict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """El criterio de aceptación: una cuota perdida invalida la corrida por sí sola."""
    arms = criteria_run()
    arms["full"] = [quota_lost("decide"), *arms["full"][1:]]
    directory = write_run(tmp_path, arms)

    assert run_command(directory, tmp_path) == 1

    out = capsys.readouterr().out
    assert out.splitlines()[0].startswith("CORRIDA INVÁLIDA: ")
    assert "## Criterio 1" not in out
    assert not any(word in out for word in VERDICT_WORDS)
    assert "## Ventana de 5 h" in out  # el pico se imprime siempre


def test_a_run_that_ran_out_of_funds_exits_one_and_prints_no_verdict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arms = criteria_run()
    arms["full"] = [funds_lost("bull"), *arms["full"][1:]]
    directory = write_run(tmp_path, arms, billing=Billing.PAYG)

    assert run_command(directory, tmp_path) == 1

    out = capsys.readouterr().out
    assert out.splitlines()[0].startswith("CORRIDA INVÁLIDA: ")
    assert "saldo insuficiente" in out.splitlines()[0]
    assert "## Criterio 1" not in out
    assert not any(word in out for word in VERDICT_WORDS)


def assert_a_run_cut_by_the_cap_is_invalid(tmp_path: Path) -> None:
    arms = criteria_run()
    arms["full"] = [cap_lost("decide"), *arms["full"][1:]]
    directory = write_run(tmp_path, arms, billing=Billing.PAYG)
    assert run_command(directory, tmp_path) == 1, "una evaluación cortada por el tope invalida"


def test_a_run_cut_by_the_spend_cap_exits_one_and_prints_no_verdict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert_a_run_cut_by_the_cap_is_invalid(tmp_path)

    out = capsys.readouterr().out
    assert out.splitlines()[0].startswith("CORRIDA INVÁLIDA: ")
    assert "tope de gasto" in out.splitlines()[0]
    assert "## Criterio 1" not in out
    assert not any(word in out for word in VERDICT_WORDS)


def test_mutation_a_validity_check_that_ignores_the_cap_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sin la sexta condición, la evaluación cortada puntúa 0 y la corrida sale válida."""
    assert_a_run_cut_by_the_cap_is_invalid(tmp_path / "control")
    mutant = mutated(
        check_validity, "if kind is AbortKind.SPEND_CAP:", "if False:", criteria_module
    )
    monkeypatch.setattr(criteria_module, "check_validity", mutant)
    with pytest.raises(AssertionError, match="cortada por el tope invalida"):
        assert_a_run_cut_by_the_cap_is_invalid(tmp_path / "mutante")


def test_a_meta_written_before_the_kind_field_loads_as_an_ablation(tmp_path: Path) -> None:
    directory = write_run(tmp_path, criteria_run())
    path = directory / "meta.json"
    stored = json.loads(path.read_text("utf-8"))
    assert stored.pop("kind") == "ablation"
    path.write_text(json.dumps(stored), encoding="utf-8")

    assert read_meta(directory).kind is RunKind.ABLATION


def test_criteria_refuses_a_probe_directory_and_says_why(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = write_run(tmp_path, criteria_run(), billing=Billing.PAYG, kind=RunKind.PROBE)

    assert run_command(directory, tmp_path) == 2

    captured = capsys.readouterr()
    assert "sondeo" in captured.err
    assert "kind=probe" in captured.err
    assert "## Criterio" not in captured.out
    assert not any(word in captured.out for word in VERDICT_WORDS)


def test_criteria_refuses_a_run_that_resumes_a_probe(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """El rechazo recorre la cadena: reanudar un sondeo no lo convierte en una corrida."""
    probe_dir = write_run(tmp_path / "sondeo", criteria_run(), kind=RunKind.PROBE)
    directory = write_run(
        tmp_path / "corrida", criteria_run(), plan=tmp_path / "sondeo" / "manifest.json"
    )
    meta = json.loads((directory / "meta.json").read_text("utf-8"))
    meta["resumed_from"] = str(probe_dir)
    (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    assert run_command(directory, tmp_path / "sondeo") == 2
    assert "sondeo" in capsys.readouterr().err


def test_audit_and_consumption_read_a_probe_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = write_run(tmp_path, criteria_run(), billing=Billing.PAYG, kind=RunKind.PROBE)

    assert audit_main([str(directory), "--history-dir", str(tmp_path / "history")]) == 0
    assert "tipo: probe" in capsys.readouterr().out
    assert consumption_main([str(directory)]) == 0
    assert "Consumo de la corrida" in capsys.readouterr().out


def test_a_foreign_veto_exits_one(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    arms = criteria_run()
    arms["solo"] = [
        arms["solo"][0].model_copy(update={"risk": vetoed("kill_switch").risk}),
        *arms["solo"][1:],
    ]
    directory = write_run(tmp_path, arms)

    assert run_command(directory, tmp_path) == 1
    assert "kill_switch" in capsys.readouterr().out.splitlines()[0]


def test_a_run_that_did_not_record_its_horizon_needs_it_on_the_command_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Puntuar con otro horizonte que el de la corrida cambiaría el retorno sin avisar."""
    directory = write_run(tmp_path, criteria_run(), horizon=None)

    assert run_command(directory, tmp_path) == 2
    assert "--horizon" in capsys.readouterr().err

    assert run_command(directory, tmp_path, "--horizon", str(HORIZON)) == 0


def test_a_horizon_flag_that_contradicts_the_run_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = write_run(tmp_path, criteria_run())

    assert run_command(directory, tmp_path, "--horizon", "5") == 2
    assert "contradice" in capsys.readouterr().err


def test_a_plan_that_is_not_the_one_the_run_used_is_an_error_and_not_a_verdict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = write_run(tmp_path, criteria_run())
    manifest = json.loads((tmp_path / "manifest.json").read_text("utf-8"))
    manifest["seed"] = 999
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert run_command(directory, tmp_path) == 2
    captured = capsys.readouterr()
    assert "sha-256" in captured.err
    assert "## Criterio" not in captured.out


def test_a_run_whose_meta_does_not_describe_its_roles_cannot_be_evaluated(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = write_run(tmp_path, criteria_run(), arm_roles=False)

    assert run_command(directory, tmp_path) == 2
    assert "arm_roles" in capsys.readouterr().err


def test_a_directory_that_is_not_a_run_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(tmp_path), "--delta", "0.002"]) == 2
    assert "meta.json" in capsys.readouterr().err


def test_the_peak_comes_from_every_pass_of_a_resumed_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """El ledger es uno: lo gastado en la pasada anterior también cuenta contra la ventana."""
    first_arms = {
        "full": [
            make_record(0, Action.BUY).model_copy(
                update={"calls": (live_call(AgentRole.DECIDER, at=at(0)),)}
            )
        ]
    }
    first = write_run(tmp_path / "uno", first_arms, plan=write_plan(tmp_path))
    second_arms = criteria_run()
    second_arms["full"] = [
        arm("buy")[0].model_copy(update={"calls": (live_call(AgentRole.DECIDER, at=at(1)),)}),
        *arm("buy")[1:],
    ]
    directory = write_run(tmp_path / "dos", second_arms, plan=tmp_path / "manifest.json")
    meta = json.loads((directory / "meta.json").read_text("utf-8"))
    meta["resumed_from"] = str(first)
    (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    assert run_command(directory, tmp_path) == 0

    out = capsys.readouterr().out
    decider = next(line for line in out.splitlines() if line.startswith("| decider"))
    assert "| 2 |" in decider


LOSSES: dict[str, Callable[[], EvaluationRecord]] = {
    "funds": lambda: funds_lost("bull"),
    "decider_quota": lambda: quota_lost("decide"),
    "momentum_404": lambda: provider_lost("momentum", FailureKind.TRANSPORT),
    "decide_timeout": lambda: provider_lost("decide", FailureKind.TIMEOUT),
    "spend_cap": lambda: cap_lost("decide"),
}
"""Las pérdidas que invalidan y que una reanudación rescata: ninguna deja entrada en caché.

La tercera y la cuarta son la quinta condición (bloque T5): el 404 de `deepseek-v4-flash` en un
nodo técnico y un plazo vencido en el decisor. La última es la sexta (bloque T7): el tope de
gasto, que una reanudación con más `--max-usd` rescata.
"""


def lost_evaluation(j: int, kind: str) -> EvaluationRecord:
    """La evaluación `j` del escenario perdida de esa forma: su mismo `run_id`, sin decisión."""
    lost = LOSSES[kind]()
    return make_record(j, None).model_copy(update={"errors": lost.errors, "calls": lost.calls})


def resumed_chain(
    tmp_path: Path, first_full: list[EvaluationRecord], last_full: list[EvaluationRecord]
) -> tuple[Path, Path]:
    """Dos pasadas del mismo plan encadenadas por `resumed_from`: la primera y la que la reanuda."""
    first_arms = criteria_run()
    first_arms["full"] = first_full
    first = write_run(tmp_path / "uno", first_arms, plan=write_plan(tmp_path))
    last_arms = criteria_run()
    last_arms["full"] = last_full
    last = write_run(tmp_path / "dos", last_arms, plan=tmp_path / "manifest.json")
    meta = json.loads((last / "meta.json").read_text("utf-8"))
    meta["resumed_from"] = str(first)
    (last / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return first, last


def assert_a_rescued_loss_leaves_the_chain_valid(tmp_path: Path, kind: str) -> None:
    """Perdida en la primera pasada, decidida en la segunda: la cadena termina sin pérdidas."""
    full = arm("buy")
    first, last = resumed_chain(tmp_path, [lost_evaluation(0, kind), *full[1:]], full)
    assert run_command(first, tmp_path) == 1, "la primera pasada, sola, sí es la que perdió"
    assert run_command(last, tmp_path) == 0, "lo que la reanudación decidió ya no está perdido"


@pytest.mark.parametrize("kind", sorted(LOSSES))
def test_a_loss_the_resume_rescued_leaves_the_run_valid(
    kind: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert_a_rescued_loss_leaves_the_chain_valid(tmp_path, kind)

    out = capsys.readouterr().out.split("# Criterios")[-1]
    assert "CORRIDA INVÁLIDA" not in out
    assert "## Criterio 1" in out


@pytest.mark.parametrize("kind", sorted(LOSSES))
def test_a_loss_still_there_at_the_end_of_the_chain_invalidates_the_run(
    kind: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Reanudar no absuelve: lo que la última pasada tampoco decidió sigue perdido."""
    still = [lost_evaluation(0, kind), *arm("buy")[1:]]
    _, last = resumed_chain(tmp_path, still, still)

    assert run_command(last, tmp_path) == 1

    out = capsys.readouterr().out
    assert out.splitlines()[0].startswith("CORRIDA INVÁLIDA: ")
    assert "## Criterio 1" not in out
    assert not any(word in out for word in VERDICT_WORDS)


@pytest.mark.parametrize("kind", sorted(LOSSES))
def test_mutation_counting_the_loss_of_the_first_link_is_caught(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sumar los eslabones cuenta como perdida una evaluación que la reanudación ya decidió."""
    mutant = mutated(
        final_records,
        "arm.arm: arm.records for",
        "arm.arm: tuple(r for run in chain for a in run.arms if a.arm == arm.arm "
        "for r in a.records) for",
        criteria_module,
    )
    monkeypatch.setattr(criteria_module, "final_records", mutant)
    with pytest.raises(AssertionError, match="ya no está perdido"):
        assert_a_rescued_loss_leaves_the_chain_valid(tmp_path, kind)


def assert_a_momentum_404_at_the_end_of_the_chain_invalidates(tmp_path: Path) -> None:
    still = [lost_evaluation(0, "momentum_404"), *arm("buy")[1:]]
    _, last = resumed_chain(tmp_path, still, still)
    assert run_command(last, tmp_path) == 1, "un 404 en un nodo técnico invalida la corrida"


def test_mutation_a_condition_that_ignores_the_technical_nodes_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mirando solo al decisor, la corrida que perdió su evidencia por un 404 saldría válida."""
    assert_a_momentum_404_at_the_end_of_the_chain_invalidates(tmp_path / "control")
    mutant = mutated(
        check_validity,
        "if kind in PROVIDER_FAILURES:",
        "if kind in PROVIDER_FAILURES and node in DECIDER_NODES:",
        criteria_module,
    )
    monkeypatch.setattr(criteria_module, "check_validity", mutant)
    with pytest.raises(AssertionError, match="un 404 en un nodo técnico invalida"):
        assert_a_momentum_404_at_the_end_of_the_chain_invalidates(tmp_path / "mutante")


def test_the_mutant_still_sees_a_timeout_in_the_decider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lo que la mutación rompe son los nodos técnicos: el decisor sigue invalidando con ella."""
    mutant = mutated(
        check_validity,
        "if kind in PROVIDER_FAILURES:",
        "if kind in PROVIDER_FAILURES and node in DECIDER_NODES:",
        criteria_module,
    )
    monkeypatch.setattr(criteria_module, "check_validity", mutant)
    still = [lost_evaluation(0, "decide_timeout"), *arm("buy")[1:]]
    _, last = resumed_chain(tmp_path, still, still)

    assert run_command(last, tmp_path) == 1


def test_no_key_and_no_url_reaches_the_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = write_run(tmp_path, criteria_run())
    run_command(directory, tmp_path)
    out = capsys.readouterr().out
    assert "http" not in out
    assert "sk-" not in out


# ─────────────────────────── Enmienda 4: el intervalo del error agrupado ──────────────────────────


def test_the_cluster_length_the_amendment_and_the_registered_delta_are_frozen() -> None:
    """Fijadas por la enmienda 4. Cambiar una es otra enmienda, no otro commit."""
    assert CLUSTER_DAYS == 7
    assert AMENDMENT == 4
    assert MIN_CLUSTERS == 50
    assert criteria_module.DELTA == 0.002
    assert criteria_module.DELTA == DELTA


def test_the_header_names_the_amendment_the_clustering_and_delta_as_a_percentage() -> None:
    text = render_criteria(report(), "d")

    assert "Enmienda 4" in text
    assert "`media ± 1.96 · EE usado`" in text
    assert "`EE usado = max(EE ingenuo, EE agrupado)`" in text
    assert "agrupar solo puede ensanchar el intervalo, nunca estrecharlo" in text
    assert "error estándar CR1" in text
    assert "un clúster por bloque UTC de 7 días del instante evaluado" in text
    assert "δ = 0.200% de retorno por evaluación (fracción 0.002)" in text
    assert "| EE ingenuo | EE agrupado | EE usado | G | deff | IC 95% |" in text


def test_every_evaluation_of_the_base_scenario_is_alone_in_its_block() -> None:
    assert len(set(clusters_of(arm("buy"), CLUSTER_DAYS))) == N


def assert_one_per_cluster_is_the_naive_interval() -> None:
    """Con G = n el agrupado es el ingenuo, y los dos son `pstdev/√n · √(G/(G - 1))`."""
    comparison = next(c for c in report().criterion_1 if c.reference == "solo")
    differences = [EVEN if j % 2 == 0 else 0.0 for j in range(N)]
    uncorrected = statistics.pstdev(differences) / math.sqrt(N)

    assert comparison.pair is not None
    assert comparison.diff is not None
    assert comparison.naive is not None
    error = comparison.pair.error
    assert (error.n, error.clusters) == (N, N)
    assert comparison.diff.stderr == pytest.approx(uncorrected * math.sqrt(N / (N - 1)))
    assert comparison.diff.stderr == pytest.approx(comparison.naive.stderr)
    assert (comparison.diff.low, comparison.diff.high) == pytest.approx(
        (comparison.naive.low, comparison.naive.high)
    )
    assert error.deff == pytest.approx(1.0)


def test_with_one_evaluation_per_cluster_the_interval_is_the_naive_one() -> None:
    assert_one_per_cluster_is_the_naive_interval()


def test_mutation_a_clustered_error_without_the_cluster_count_correction_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mutant = mutated(
        metrics_module.clustered_errors, "groups / (groups - 1) * ", "", metrics_module
    )
    monkeypatch.setattr(metrics_module, "clustered_errors", mutant)
    with pytest.raises(AssertionError):
        assert_one_per_cluster_is_the_naive_interval()


BLOCKS = 60
PER_BLOCK = 4
GAP = 8
"""Velas entre dos evaluaciones del mismo bloque: la salida de una, a las 3, no pisa la otra."""

SWING = 0.01
DRIFT = 0.0019
CLUSTERED_SYMBOL = "C0"


def block_return(block: int) -> float:
    """Lo que rinde comprar en cualquier evaluación del bloque: `DRIFT ± SWING`, alternando."""
    return DRIFT + (SWING if block % 2 == 0 else -SWING)


def clustered_slots() -> list[Slot]:
    return [
        Slot(CLUSTERED_SYMBOL, STEP * block + GAP * k, block * 5 // BLOCKS)
        for block in range(BLOCKS)
        for k in range(PER_BLOCK)
    ]


def clustered_history() -> list[list[float]]:
    exits = {
        slot.candle + HORIZON: 100.0 * (1.0 + block_return(slot.candle // STEP))
        for slot in clustered_slots()
    }
    return [
        [
            START.timestamp() * 1000 + index * 3_600_000,
            100.0,
            101.0,
            99.0,
            exits.get(index, 100.0),
            1.0,
        ]
        for index in range(clustered_slots()[-1].candle + HORIZON + 2)
    ]


CLUSTERED_HISTORIES = {CLUSTERED_SYMBOL: clustered_history()}


def clustered_run() -> dict[str, list[EvaluationRecord]]:
    """`full` compra las 240 y `solo` no entra en ninguna: la diferencia es la del bloque."""
    slots = clustered_slots()
    return {
        "full": [record_at(slot.symbol, slot.candle, Action.BUY) for slot in slots],
        "solo": [record_at(slot.symbol, slot.candle, Action.HOLD) for slot in slots],
    }


def assert_the_verdict_comes_from_the_clustered_interval() -> None:
    """240 evaluaciones en 60 bloques de 4, iguales dentro de cada bloque.

    A mano, con a = 0.01 y media 0.0019. El ingenuo es `a·√(240/239)/√240` = 0.000647: la media
    queda a 2.94 errores de 0 y el intervalo lo excluye, «justifica». El agrupado es `a/√59` =
    0.001302 —las sumas por bloque son ±4a, así que `EE² = 60/59 · 60·16a²/240²`—: la media
    queda a 1.46, el intervalo incluye 0 y su semiancho, 0.00255, supera δ. «No concluyente».
    deff = 239/59.
    """
    result = evaluate_criteria(clustered_run(), CLUSTERED_HISTORIES, HORIZON, DELTA)
    comparison = next(c for c in result.criterion_1 if c.reference == "solo")
    n = BLOCKS * PER_BLOCK

    assert comparison.pair is not None
    assert comparison.naive is not None
    assert comparison.diff is not None
    error = comparison.pair.error
    assert (error.n, error.clusters) == (n, BLOCKS)
    assert comparison.naive.mean == pytest.approx(DRIFT)
    assert comparison.naive.stderr == pytest.approx(SWING * math.sqrt(n / (n - 1)) / math.sqrt(n))
    assert comparison.naive.low > 0.0, "el intervalo ingenuo excluye el 0"
    assert comparison.diff.stderr == pytest.approx(SWING / math.sqrt(BLOCKS - 1))
    assert error.clustered is not None
    assert error.clustered > error.naive, "agrupar aquí ensancha"
    assert comparison.used == error.clustered, "el error usado es el agrupado"
    assert error.deff == pytest.approx((n - 1) / (BLOCKS - 1))
    assert comparison.diff.low < 0.0 < comparison.diff.high, "el agrupado lo incluye"
    assert comparison.half_width == pytest.approx(1.96 * SWING / math.sqrt(BLOCKS - 1))
    assert comparison.verdict is Verdict.INCONCLUSIVE


def test_where_the_naive_interval_excludes_zero_the_verdict_is_the_clustered_one() -> None:
    assert_the_verdict_comes_from_the_clustered_interval()


def test_mutation_a_verdict_taken_from_the_naive_error_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mutant = mutated(
        metrics_module.clustered_paired_difference,
        "stderr = max(error.naive, error.clustered)",
        "stderr = error.naive",
        metrics_module,
    )
    monkeypatch.setattr(criteria_module, "clustered_paired_difference", mutant)
    with pytest.raises(AssertionError):
        assert_the_verdict_comes_from_the_clustered_interval()


def test_the_command_prints_the_three_errors_and_judges_with_the_clustered_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """La misma fila, de un directorio a la salida: las cifras de arriba a tres decimales."""
    plan = write_plan(tmp_path, CLUSTERED_HISTORIES, clustered_slots())
    directory = write_run(tmp_path, clustered_run(), plan=plan)

    assert run_command(directory, tmp_path) == 0

    out = capsys.readouterr().out
    row = next(line for line in out.splitlines() if line.startswith("| full | solo "))
    cells = [cell.strip() for cell in row.strip("|").split("|")]
    assert cells[:12] == [
        "full",
        "solo",
        "240",
        "+0.190%",
        "0.065%",
        "0.130%",
        "0.130%",
        "60",
        "4.051",
        "[-0.065%, +0.445%]",
        "0.255%",
        "no concluyente",
    ]


# ─────────────────────── Enmienda 4: el suelo en el error estándar ingenuo ────────────────────────

PAIRED_BLOCKS = 60
SWAY = 0.008
"""Lo que separa a las dos evaluaciones de un bloque: una rinde `+SWAY` y la otra `-SWAY`."""

TILT = 0.001
"""Lo que mueve al bloque entero, alternando de signo de un bloque al siguiente."""

LIFT = 0.001
"""La media: lo que rinde comprar, quitadas las dos oscilaciones."""

COMPENSATED_SYMBOL = "K0"


def compensated_return(block: int, k: int) -> float:
    """`LIFT ± TILT ± SWAY`: el bloque decide el signo de `TILT`, y la posición, el de `SWAY`."""
    return LIFT + (TILT if block % 2 == 0 else -TILT) + (SWAY if k == 0 else -SWAY)


def compensated_slots() -> list[Slot]:
    return [
        Slot(COMPENSATED_SYMBOL, STEP * block + GAP * k, block * 5 // PAIRED_BLOCKS)
        for block in range(PAIRED_BLOCKS)
        for k in range(2)
    ]


def compensated_history() -> list[list[float]]:
    exits = {
        slot.candle + HORIZON: 100.0
        * (1.0 + compensated_return(slot.candle // STEP, (slot.candle % STEP) // GAP))
        for slot in compensated_slots()
    }
    return [
        [
            START.timestamp() * 1000 + index * 3_600_000,
            100.0,
            101.0,
            99.0,
            exits.get(index, 100.0),
            1.0,
        ]
        for index in range(compensated_slots()[-1].candle + HORIZON + 2)
    ]


COMPENSATED_HISTORIES = {COMPENSATED_SYMBOL: compensated_history()}


def compensated_run() -> dict[str, list[EvaluationRecord]]:
    """`full` compra las 120 y `solo` no entra en ninguna: la diferencia es la de la evaluación."""
    slots = compensated_slots()
    return {
        "full": [record_at(slot.symbol, slot.candle, Action.BUY) for slot in slots],
        "solo": [record_at(slot.symbol, slot.candle, Action.HOLD) for slot in slots],
    }


def assert_the_naive_error_is_the_floor() -> None:
    """120 evaluaciones en 60 bloques de 2 que se compensan dentro de cada bloque.

    A mano, con S = 0.008, T = 0.001 y media 0.001. Las dos de un bloque valen `media ± T + S` y
    `media ± T - S`: su suma centrada es `±2T`, sin rastro de S. El agrupado es `T/√59` =
    0.000130 —`EE² = 60/59 · 60·4T²/120²`—, y con él la media queda a 7.68 errores de 0: el
    intervalo lo excluye, «justifica». El ingenuo sí ve S: `Σ(x - x̄)² = 120·(T² + S²)`, así que
    es `√((T² + S²)/119)` = 0.000739, 5.7 veces mayor; deff = 119·T²/(59·(T² + S²)) = 0.031.

    El suelo manda: el error usado es el ingenuo, el semiancho es 1.96 · 0.000739 = 0.00145, el
    intervalo incluye 0 y no supera δ. «Empate», que para `full` es derrota. Que 120
    evaluaciones se compensen dentro de sus bloques no las hace decir más que si fueran
    independientes.
    """
    result = evaluate_criteria(compensated_run(), COMPENSATED_HISTORIES, HORIZON, DELTA)
    comparison = next(c for c in result.criterion_1 if c.reference == "solo")
    n = PAIRED_BLOCKS * 2
    naive = math.sqrt((TILT**2 + SWAY**2) / (n - 1))
    clustered = TILT / math.sqrt(PAIRED_BLOCKS - 1)

    assert comparison.pair is not None
    assert comparison.naive is not None
    assert comparison.diff is not None
    error = comparison.pair.error
    assert (error.n, error.clusters) == (n, PAIRED_BLOCKS)
    assert comparison.naive.mean == pytest.approx(LIFT)
    assert error.naive == pytest.approx(naive)
    assert error.clustered is not None
    assert error.clustered == pytest.approx(clustered)
    assert error.clustered < error.naive, "agrupar aquí estrecharía"
    assert error.deff == pytest.approx(
        (n - 1) * TILT**2 / ((PAIRED_BLOCKS - 1) * (TILT**2 + SWAY**2))
    )
    assert LIFT - 1.96 * clustered > 0.0, "sin suelo, el intervalo excluiría el 0"
    assert comparison.used == error.naive, "el error usado es el ingenuo"
    assert comparison.diff.stderr == pytest.approx(naive)
    assert (comparison.diff.low, comparison.diff.high) == pytest.approx(
        (comparison.naive.low, comparison.naive.high)
    )
    assert comparison.diff.low < 0.0 < comparison.diff.high
    assert comparison.half_width == pytest.approx(1.96 * naive)
    assert comparison.verdict is Verdict.TIE


def test_where_the_clustered_error_is_smaller_the_naive_one_decides() -> None:
    assert_the_naive_error_is_the_floor()


def test_mutation_an_interval_without_the_floor_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sin el `max`, el intervalo es el del agrupado: excluye 0 y `full` «justifica»."""
    mutant = mutated(
        metrics_module.clustered_paired_difference,
        "stderr = max(error.naive, error.clustered)",
        "stderr = error.clustered",
        metrics_module,
    )
    monkeypatch.setattr(criteria_module, "clustered_paired_difference", mutant)
    with pytest.raises(AssertionError):
        assert_the_naive_error_is_the_floor()

    result = evaluate_criteria(compensated_run(), COMPENSATED_HISTORIES, HORIZON, DELTA)
    comparison = next(c for c in result.criterion_1 if c.reference == "solo")
    assert comparison.verdict is Verdict.ABOVE, "lo que el suelo impide"


def test_the_command_prints_the_three_errors_and_judges_with_the_naive_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """La misma fila, de un directorio a la salida: el usado es el ingenuo y el deff baja de 1."""
    plan = write_plan(tmp_path, COMPENSATED_HISTORIES, compensated_slots())
    directory = write_run(tmp_path, compensated_run(), plan=plan)

    assert run_command(directory, tmp_path) == 0

    out = capsys.readouterr().out
    row = next(line for line in out.splitlines() if line.startswith("| full | solo "))
    cells = [cell.strip() for cell in row.strip("|").split("|")]
    assert cells[:12] == [
        "full",
        "solo",
        "120",
        "+0.100%",
        "0.074%",
        "0.013%",
        "0.074%",
        "60",
        "0.031",
        "[-0.045%, +0.245%]",
        "0.145%",
        "empate",
    ]


FEW = MIN_CLUSTERS - 10


def assert_too_few_clusters_have_figures_and_no_verdict() -> None:
    """40 evaluaciones, cada una en su bloque: hay error estándar que imprimir y no intervalo."""
    run = {name: records[:FEW] for name, records in criteria_run().items()}
    result = evaluate_criteria(run, HISTORIES, HORIZON, DELTA)
    comparison = next(c for c in result.criterion_1 if c.reference == "solo")

    assert comparison.verdict is Verdict.UNDETERMINED
    assert comparison.diff is None
    assert comparison.naive is not None
    assert comparison.pair is not None
    assert comparison.pair.error.clusters == FEW
    assert comparison.why == f"G = {FEW} clústeres, menos de {MIN_CLUSTERS}"

    row = next(
        line
        for line in render_criteria(result, "d").splitlines()
        if line.startswith("| full | solo ")
    )
    cells = [cell.strip() for cell in row.strip("|").split("|")]
    assert cells[2] == str(FEW)
    assert cells[4].endswith("%"), "el EE ingenuo se imprime"
    assert cells[5].endswith("%"), "el EE agrupado se imprime"
    assert cells[6] == "—", "sin intervalo no hay error usado"
    assert cells[7] == str(FEW)
    assert cells[9:12] == ["—", "—", f"no determinado: G = {FEW} clústeres, menos de 50"]
    assert not any(word in row for word in VERDICT_WORDS)


def test_fewer_than_fifty_clusters_print_their_figures_and_give_no_verdict() -> None:
    assert_too_few_clusters_have_figures_and_no_verdict()


def test_mutation_a_verdict_given_with_too_few_clusters_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mutant = mutated(
        metrics_module.clustered_paired_difference,
        "if not error.eligible:",
        "if False:",
        metrics_module,
    )
    monkeypatch.setattr(criteria_module, "clustered_paired_difference", mutant)
    with pytest.raises(AssertionError):
        assert_too_few_clusters_have_figures_and_no_verdict()


def test_exactly_fifty_clusters_are_enough() -> None:
    run = {name: records[:MIN_CLUSTERS] for name, records in criteria_run().items()}
    comparison = next(
        c
        for c in evaluate_criteria(run, HISTORIES, HORIZON, DELTA).criterion_1
        if c.reference == "solo"
    )
    assert comparison.diff is not None
    assert comparison.verdict is Verdict.ABOVE


def test_a_single_cluster_has_no_clustered_error_and_no_verdict() -> None:
    arm_returns = [0.01 if j % 2 == 0 else -0.01 for j in range(60)]
    comparison = compare_returns("a", "b", arm_returns, [0.0] * 60, [0] * 60, DELTA)

    assert comparison.verdict is Verdict.UNDETERMINED
    assert comparison.pair is not None
    assert comparison.pair.error.clustered is None
    assert comparison.why is not None
    assert "un solo clúster" in comparison.why


def test_the_clusters_have_to_line_up_with_the_evaluations() -> None:
    with pytest.raises(ValueError, match="no cuadran"):
        compare_returns("a", "b", [0.01] * 60, [0.0] * 60, list(range(59)), DELTA)


# ──────────────────────────────────── δ: una constante con guarda ─────────────────────────────────


def history_flag(tmp_path: Path) -> list[str]:
    return ["--history-dir", str(tmp_path / "history")]


def test_without_the_argument_the_command_uses_the_registered_delta(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--delta 0.002` sigue funcionando —es el comando que la enmienda 3 dejó escrito— y da lo
    mismo que no pasar nada."""
    directory = write_run(tmp_path, criteria_run())

    assert main([str(directory), *history_flag(tmp_path)]) == 0
    bare = capsys.readouterr().out
    assert main([str(directory), "--delta", "0.002", *history_flag(tmp_path)]) == 0

    assert capsys.readouterr().out == bare
    assert "δ = 0.200% de retorno por evaluación (fracción 0.002)" in bare
    assert "δ no registrado" not in bare


def assert_a_delta_that_is_not_the_registered_one_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--delta 0.2` es δ = 20%: con él los nueve veredictos de la etapa 1 imprimían «empate»."""
    directory = write_run(tmp_path, criteria_run())
    refused = False
    try:
        main([str(directory), "--delta", "0.2", *history_flag(tmp_path)])
    except SystemExit as stop:
        refused = stop.code == 2
    captured = capsys.readouterr()

    assert refused, "un δ que no es el registrado tiene que ser un error de uso"
    assert captured.out == ""
    assert "--delta 0.2 es δ = 20.000%" in captured.err
    assert "el δ registrado es 0.200%" in captured.err
    assert "--exploratory-delta 0.2" in captured.err


def test_a_delta_that_is_not_the_registered_one_is_refused_naming_its_percentage(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert_a_delta_that_is_not_the_registered_one_is_refused(tmp_path, capsys)


def test_mutation_a_command_that_accepts_any_delta_is_caught(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    mutant = mutated(
        criteria_module.resolve_delta,
        "given is not None and given != DELTA",
        "False",
        criteria_module,
    )
    monkeypatch.setattr(criteria_module, "resolve_delta", mutant)
    with pytest.raises(AssertionError):
        assert_a_delta_that_is_not_the_registered_one_is_refused(tmp_path, capsys)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--delta", "0"],
        ["--delta", "-0.002"],
        ["--delta", "0.02"],
        ["--exploratory-delta", "0"],
        ["--delta", "0.002", "--exploratory-delta", "0.01"],
    ],
)
def test_the_two_delta_arguments_do_not_mix_and_have_to_make_sense(
    arguments: list[str], tmp_path: Path
) -> None:
    directory = write_run(tmp_path, criteria_run())
    with pytest.raises(SystemExit) as raised:
        main([str(directory), *arguments, *history_flag(tmp_path)])
    assert raised.value.code == 2


def verdict_rows(text: str) -> list[str]:
    """Las filas de las tablas de comparación que llevan un veredicto."""
    return [
        line
        for line in text.splitlines()
        if line.startswith("| ") and any(f"| {word}" in line for word in VERDICT_WORDS)
    ]


def assert_an_exploratory_delta_marks_every_verdict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = write_run(tmp_path, criteria_run())

    code = main([str(directory), "--exploratory-delta", "0.2", *history_flag(tmp_path)])
    out = capsys.readouterr().out

    assert code == 0
    assert "δ = 20.000% de retorno por evaluación (fracción 0.2)" in out
    assert "**δ no registrado**: 20.000%. El δ de la enmienda 2 es 0.200%" in out
    rows = verdict_rows(out)
    assert len(rows) == 3 + 4 * len(BASELINE_ARMS), "criterio 1 y los cuatro brazos del 3"
    assert all("(δ no registrado)" in row for row in rows)


def test_an_exploratory_delta_marks_the_header_and_every_verdict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert_an_exploratory_delta_marks_every_verdict(tmp_path, capsys)


def test_mutation_an_exploratory_delta_that_is_not_marked_is_caught(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    mutant = mutated(
        criteria_module.unregistered_delta, "if delta == DELTA:", "if True:", criteria_module
    )
    monkeypatch.setattr(criteria_module, "unregistered_delta", mutant)
    with pytest.raises(AssertionError):
        assert_an_exploratory_delta_marks_every_verdict(tmp_path, capsys)


def test_the_mark_comes_from_the_value_and_not_from_the_argument() -> None:
    """Quien llame a `evaluate_criteria` desde código con otro δ recibe la misma marca."""
    explored = render_criteria(evaluate_criteria(criteria_run(), HISTORIES, HORIZON, 0.2), "d")
    registered = render_criteria(report(), "d")

    assert all("(δ no registrado)" in row for row in verdict_rows(explored))
    assert "δ no registrado" not in registered
    assert len(verdict_rows(explored)) == len(verdict_rows(registered))


def test_the_flag_that_reproduces_a_report_says_whether_its_delta_is_the_registered_one() -> None:
    assert criteria_module.delta_flag(0.002) == "--delta 0.002"
    assert criteria_module.delta_flag(0.2) == "--exploratory-delta 0.2"


# ──────────────────────────────── Tramos: la clave de unión con el manifiesto ─────────────────────


def test_the_command_prints_the_five_spans_of_the_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """De un directorio a la salida, pasando por `load_plan`: el manifiesto guarda aperturas y
    los registros llevan cierres. Era lo que el escenario de antes no distinguía."""
    directory = write_run(tmp_path, criteria_run())

    assert run_command(directory, tmp_path) == 0

    out = capsys.readouterr().out
    spans = [line for line in out.splitlines() if line.startswith("| solo | tramo ")]
    assert [line.split("|")[2:4] for line in spans] == [
        [f" tramo {index} ", f" {N // 5} "] for index in range(5)
    ]
    assert "sin tramo" not in out


@functools.cache
def stage_one_replayed() -> dict[str, tuple[EvaluationRecord, ...]]:
    """Dos líneas base sobre el manifiesto versionado, por el replay de verdad.

    Solo caché y sin `fill_with`: una llamada a un modelo levantaría `ReplayCacheMissError`.
    Hacen de `full` y de `solo` porque la tabla descriptiva compara esos nombres; lo que la
    prueba mira es en qué tramo cae cada evaluación, y eso lo decide el `at` que escribe el
    replay, no quién decidió.
    """
    manifest = load_manifest(ROOT / DEFAULT_MANIFEST)
    plan = plan_from_manifest(manifest, load_selection_histories(manifest, ROOT / HISTORY_DIR))
    settings = ablation_settings()

    async def replayed(name: str) -> tuple[EvaluationRecord, ...]:
        journal = InMemoryJournal()
        await run_arm(
            next(item for item in ARMS if item.name == name),
            plan,
            settings,
            HEALTHY,
            InMemoryResponseCache(),
            fixed_clock,
            QuotaLedger(settings.quota_window, fixed_clock),
            journal,
            guard=None,
            horizon=manifest.horizon,
        )
        return tuple(journal.records)

    async def both() -> dict[str, tuple[EvaluationRecord, ...]]:
        return {"full": await replayed("always_buy"), "solo": await replayed("always_sell")}

    return asyncio.run(both())


def stage_one_meta() -> RunMeta:
    plan = ROOT / DEFAULT_MANIFEST
    return RunMeta(
        plan_kind=PlanKind.MANIFEST,
        plan_path=str(plan),
        plan_sha256=file_sha256(plan),
        argv=("--manifest", str(plan)),
        fill=False,
        arms=("full", "solo"),
        started_at=START,
    )


def assert_stage_one_has_five_spans_of_twenty_eight(load: Callable[..., Any]) -> None:
    """El manifiesto de la etapa 1 reparte 140 activaciones en 5 tramos de 28."""
    arms = stage_one_replayed()
    records = [record for entries in arms.values() for record in entries]
    histories, strata = load(stage_one_meta(), None, ROOT / HISTORY_DIR, records)
    horizon = load_manifest(ROOT / DEFAULT_MANIFEST).horizon

    result = evaluate_criteria(arms, histories, horizon, DELTA, strata)
    descriptive = next(item for item in result.descriptive if item.reference == "solo")

    assert descriptive.by_stratum is not None
    assert {cell.label: cell.n for cell in descriptive.by_stratum} == {
        str(index): 28 for index in range(5)
    }


def test_the_spans_of_stage_one_are_five_cells_of_twenty_eight() -> None:
    assert len(stage_one_replayed()["full"]) == 140
    assert_stage_one_has_five_spans_of_twenty_eight(audit_module.load_plan)


def test_mutation_spans_joined_by_the_opening_of_the_candle_are_caught() -> None:
    """La clave de antes: 139 evaluaciones «sin tramo» y una con el tramo de la vela siguiente."""
    mutant = mutated(
        audit_module.load_plan,
        "entry.at + timeframe_to_timedelta(entry.timeframe)",
        "entry.at",
        audit_module,
    )
    with pytest.raises(AssertionError):
        assert_stage_one_has_five_spans_of_twenty_eight(mutant)


# ───────────────────────────── Regresión: la etapa 1 bajo la enmienda 4 ───────────────────────────

STAGE_ONE_FIXTURE = ROOT / "tests" / "data" / "stage1_paired_returns.jsonl"
STAGE_ONE_SHA256 = "b521801e0545d3965d3f96c0fba32debd92b30988af6d31e3cfa1d034e55841f"

STAGE_ONE_PUBLISHED = {
    ("full", "solo"): ("+0.017%", "0.192%", "0.954"),
    ("full", "always_buy"): ("-0.432%", "0.418%", "0.565"),
    ("full", "always_sell"): ("+0.301%", "0.361%", "0.888"),
    ("full", "random_uniform"): ("-0.044%", "0.396%", "0.793"),
    ("full", "rule_trend"): ("-0.029%", "0.223%", "0.928"),
    ("solo", "always_buy"): ("-0.449%", "0.345%", "0.649"),
    ("solo", "always_sell"): ("+0.284%", "0.375%", "0.877"),
    ("solo", "random_uniform"): ("-0.061%", "0.364%", "0.939"),
    ("solo", "rule_trend"): ("-0.046%", "0.186%", "1.076"),
}
"""Por par, lo que `docs/ablation.md` ya tiene publicado de `var/ablation/20261009T045033Z`: la
diferencia media y el EE ingenuo de «Etapa 1 — resultado», y el deff con L = 7 de «6.
Comprobación sobre la corrida» del efecto de diseño. Atan el fixture a esas tablas."""

STAGE_ONE_CLUSTERS = 71


class Scored(NamedTuple):
    """Una evaluación de la etapa 1: dónde, cuándo y cuánto rindió cada brazo."""

    symbol: str
    at: datetime
    returns: dict[str, float]


def stage_one_scored() -> list[Scored]:
    return [
        Scored(line["symbol"], datetime.fromisoformat(line["at"]), line["returns"])
        for line in map(json.loads, STAGE_ONE_FIXTURE.read_text("utf-8").splitlines())
    ]


def stage_one_comparisons() -> dict[tuple[str, str], Comparison]:
    scored = stage_one_scored()
    clusters = clusters_of(scored, CLUSTER_DAYS)
    return {
        (name, reference): compare_returns(
            name,
            reference,
            [item.returns[name] for item in scored],
            [item.returns[reference] for item in scored],
            clusters,
            DELTA,
        )
        for name, reference in STAGE_ONE_PUBLISHED
    }


def assert_stage_one_is_still_inconclusive() -> None:
    comparisons = stage_one_comparisons()

    assert len(comparisons) == 9
    for pair, (mean, naive, deff) in STAGE_ONE_PUBLISHED.items():
        comparison = comparisons[pair]
        assert comparison.pair is not None, pair
        error = comparison.pair.error
        assert (comparison.n, error.clusters) == (140, STAGE_ONE_CLUSTERS), pair
        assert f"{comparison.pair.naive.mean:+.3%}" == mean, pair
        assert f"{error.naive:.3%}" == naive, pair
        assert error.deff is not None
        assert f"{error.deff:.3f}" == deff, pair
        assert comparison.verdict is Verdict.INCONCLUSIVE, pair


def test_the_stage_one_fixture_is_the_one_whose_provenance_is_recorded() -> None:
    assert file_sha256(STAGE_ONE_FIXTURE) == STAGE_ONE_SHA256
    assert len(stage_one_scored()) == 140


STAGE_ONE_WIDENED = ("solo", "rule_trend")
"""El único par de la etapa 1 donde el error agrupado supera al ingenuo y es el usado."""

STAGE_ONE_WIDENED_ERROR = "0.193%"


def assert_stage_one_never_judges_below_the_naive_error() -> None:
    """El error usado no baja del ingenuo en ninguno de los nueve pares.

    En ocho el agrupado sale menor que el ingenuo y el usado es el ingenuo: sin suelo, esos
    ocho intervalos se estrechaban. En `solo` contra `rule_trend` el agrupado es mayor, y es el
    usado: 0.193% donde el ingenuo era 0.186%.
    """
    comparisons = stage_one_comparisons()

    assert len(comparisons) == 9
    for pair, comparison in comparisons.items():
        assert comparison.pair is not None, pair
        error = comparison.pair.error
        assert error.clustered is not None, pair
        assert comparison.used is not None, pair
        assert comparison.used >= error.naive, pair
        if pair == STAGE_ONE_WIDENED:
            assert error.clustered > error.naive, pair
            assert comparison.used == error.clustered, pair
            assert f"{comparison.used:.3%}" == STAGE_ONE_WIDENED_ERROR
        else:
            assert error.clustered < error.naive, pair
            assert comparison.used == error.naive, pair
            assert comparison.used != error.clustered, pair


def test_the_nine_verdicts_of_stage_one_do_not_change_with_the_interval_of_the_amendment() -> None:
    """Los nueve pares siguen «no concluyente», con el deff que el efecto de diseño ya publicó."""
    assert_stage_one_is_still_inconclusive()


def test_stage_one_uses_the_naive_error_in_eight_pairs_and_the_clustered_in_one() -> None:
    assert_stage_one_never_judges_below_the_naive_error()


def test_mutation_without_the_floor_stage_one_is_judged_below_the_naive_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sin el `max` los ocho intervalos se estrechan. Los veredictos, con este n, no cambian."""
    mutant = mutated(
        metrics_module.clustered_paired_difference,
        "stderr = max(error.naive, error.clustered)",
        "stderr = error.clustered",
        metrics_module,
    )
    monkeypatch.setattr(criteria_module, "clustered_paired_difference", mutant)
    with pytest.raises(AssertionError):
        assert_stage_one_never_judges_below_the_naive_error()
    assert_stage_one_is_still_inconclusive()


def test_mutation_a_looser_threshold_changes_the_verdicts_of_stage_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Con el umbral diez veces más ancho los nueve «empatan», como con `--delta 0.02`."""
    mutant = mutated(
        criteria_module.classify,
        "half_width(diff) > delta",
        "half_width(diff) > delta * 10",
        criteria_module,
    )
    monkeypatch.setattr(criteria_module, "classify", mutant)
    with pytest.raises(AssertionError):
        assert_stage_one_is_still_inconclusive()
