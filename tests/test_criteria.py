"""Pruebas del evaluador de criterios y de las guardas de validez.

Todo parte de registros sintéticos con respuesta conocida: cada evaluación tiene su retorno
escrito de antemano, de modo que la diferencia pareada, su intervalo y el veredicto se
pueden calcular a mano y no leer de la función que se prueba.

Las mutaciones que fijan el diseño están escritas como pruebas, no como comentarios:

- ignorar δ, y el veredicto «no concluyente» desaparece;
- contar el borde de la ventana de 5 h con el signo contrario;
- dejar pasar un veto que no sea `invalid_stop_side`.
"""

from __future__ import annotations

import json
import statistics
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import NAMESPACE_URL, uuid5

import pytest

from crypto_agents import criteria as criteria_module
from crypto_agents import metrics as metrics_module
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
from crypto_agents.consumption import main as consumption_main
from crypto_agents.criteria import (
    BASELINE_ARMS,
    DECIDER_NODES,
    PEAK_WARNING_SHARE,
    CriteriaError,
    CriteriaReport,
    PeakWindow,
    Verdict,
    check_validity,
    classify,
    compare_returns,
    evaluate_criteria,
    final_records,
    main,
    peak_window_usage,
    render_criteria,
    render_invalid,
    render_peaks,
)
from crypto_agents.journal import EvaluationRecord, JsonlJournal
from crypto_agents.llm import ModelCallError, SpendCapReachedError
from crypto_agents.market import write_ohlcv_csv
from crypto_agents.metrics import AbortKind, PairedDifference
from crypto_agents.quota import QuotaExhaustedError
from crypto_agents.risk import INVALID_STOP_SIDE, NO_EXPOSURE_HEADROOM
from crypto_agents.selection import (
    PlannedEvaluation,
    SelectionManifest,
    SeriesRef,
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
from tests.conftest import SCARCE, insufficient_funds_error, no_route_error, timeout_error
from tests.test_metrics import record as bare_record
from tests.test_net_outcomes import mutated

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

START = datetime(2026, 8, 1, tzinfo=UTC)
HORIZON = 3
SYMBOLS = ("S0", "S1")
PER_SYMBOL = 20
N = len(SYMBOLS) * PER_SYMBOL
EVEN, ODD = 0.012, 0.008
"""Retorno bruto de una compra: 1.2% en las evaluaciones pares y 0.8% en las impares."""

DELTA = 0.002


# ───────────────────────────────── Escenario sintético con respuesta conocida ─────────────────────


def evaluation_return(j: int) -> float:
    return EVEN if j % 2 == 0 else ODD


def symbol_of(j: int) -> str:
    return SYMBOLS[j % 2]


def candle_index(j: int) -> int:
    """Una evaluación cada 4 velas: la salida (a las 3) no pisa la entrada de la siguiente."""
    return 4 * (j // 2)


def stratum_of(j: int) -> int:
    return (j // 2) // 4


def history_for(symbol: str) -> list[list[float]]:
    """Velas de 1 h de un símbolo. Cada evaluación entra a 100 y sale a `100 * (1 + r)`.

    Los rangos quedan dentro de [97, 103]: el stop común (`entry -/+ 2 * 2.0`) no se toca
    nunca, así que toda posición sale al cierre del horizonte y su retorno es `+/- r`.
    Solo las evaluaciones de este símbolo mueven su cierre; las del otro lo dejan en 100.
    """
    exits = {
        candle_index(j) + HORIZON: 100.0 * (1.0 + evaluation_return(j))
        for j in range(N)
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
        for index in range(4 * PER_SYMBOL + 4)
    ]


HISTORIES = {symbol: history_for(symbol) for symbol in SYMBOLS}


def run_id_of(j: int) -> object:
    return uuid5(NAMESPACE_URL, f"{symbol_of(j)}|{candle_index(j)}")


def make_record(j: int, action: Action | None) -> EvaluationRecord:
    """La evaluación `j` de un brazo que decide `action`; `None` es una evaluación sin decisión."""
    moment = START + timedelta(hours=candle_index(j))
    symbol = symbol_of(j)
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
        run_id=run_id_of(j),  # type: ignore[arg-type]
        at=moment,
        symbol=symbol,
        timeframe="1h",
        snapshot=MarketSnapshot(
            run_id=run_id_of(j),  # type: ignore[arg-type]
            exchange="binance",
            symbol=symbol,
            timeframe="1h",
            timestamp=moment,
            close=100.0,
            candles_digest="a" * 64,
            candles_count=100,
        ),
        indicators=IndicatorSet(values={"ATRr_14": 2.0}),
        proposal=proposal,
    )


def arm(rule: str) -> list[EvaluationRecord]:
    """Un brazo sintético. `buy`, `sell`, `hold`, `even` (compra pares) o `odd` (compra impares)."""
    actions: dict[str, list[Action]] = {
        "buy": [Action.BUY] * N,
        "sell": [Action.SELL] * N,
        "hold": [Action.HOLD] * N,
        "even": [Action.BUY if j % 2 == 0 else Action.HOLD for j in range(N)],
        "odd": [Action.BUY if j % 2 == 1 else Action.HOLD for j in range(N)],
    }
    return [make_record(j, action) for j, action in enumerate(actions[rule])]


def strata() -> dict[tuple[str, datetime], int]:
    return {
        (symbol_of(j), START + timedelta(hours=candle_index(j))): stratum_of(j) for j in range(N)
    }


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
    """Retornos ±0.01 contra cero: media 0, semiancho 0.003139 a mano (n = 40)."""
    arm_returns = [0.01 if j % 2 == 0 else -0.01 for j in range(40)]
    comparison = compare_returns("a", "b", arm_returns, [0.0] * 40, 0.003)

    assert comparison.n == 40
    assert comparison.diff is not None
    mean, stderr, low, high = expected(arm_returns)
    assert comparison.diff.mean == pytest.approx(mean)
    assert comparison.diff.stderr == pytest.approx(stderr)
    assert (comparison.diff.low, comparison.diff.high) == pytest.approx((low, high))
    assert comparison.half_width == pytest.approx(0.0031386, abs=1e-6)
    assert comparison.verdict is Verdict.INCONCLUSIVE
    assert compare_returns("a", "b", arm_returns, [0.0] * 40, 0.0032).verdict is Verdict.TIE


def test_fewer_than_thirty_pairs_have_no_interval_and_say_why() -> None:
    comparison = compare_returns("a", "b", [0.01] * 29, [0.0] * 29, DELTA)
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
    assert {cell.n for cell in cells.values()} == {8}
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
    assert f"δ = {DELTA}" in text
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


def write_plan(tmp_path: Path) -> Path:
    """Manifiesto, históricos y su digest, exactamente como los lee la ablación."""
    history_dir = tmp_path / "history"
    for symbol, rows in HISTORIES.items():
        write_ohlcv_csv(history_dir / f"{symbol.lower()}_1h.csv", rows)
    entries = tuple(
        PlannedEvaluation(
            symbol=symbol_of(j),
            timeframe="1h",
            index=candle_index(j),
            at=START + timedelta(hours=candle_index(j)),
            candles_digest=window_digest(HISTORIES[symbol_of(j)], candle_index(j), 5),
            stratum=stratum_of(j),
        )
        for j in range(N)
    )
    series = tuple(
        SeriesRef(
            symbol=symbol,
            timeframe="1h",
            bars=len(rows),
            first=START,
            last=START + timedelta(hours=len(rows) - 1),
            digest="b" * 64,
            activations=PER_SYMBOL,
        )
        for symbol, rows in HISTORIES.items()
    )
    manifest = SelectionManifest(
        created_at=START,
        seed=1,
        target=N,
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


def test_the_delta_is_mandatory(tmp_path: Path) -> None:
    directory = write_run(tmp_path, criteria_run())
    with pytest.raises(SystemExit) as raised:
        main([str(directory)])
    assert raised.value.code == 2


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
