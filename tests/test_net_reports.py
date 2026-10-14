"""El retorno neto en `criteria` y en `audit`: columnas aparte, descriptivas, sin tocar el bruto.

Se monta sobre el escenario sintético de `test_criteria.py`: dos símbolos, velas de 1 h, 60
evaluaciones con retorno bruto conocido (1.2% en las pares, 0.8% en las impares), cada una sola
en su bloque de 7 días. Encima se le pone funding de 8 h por símbolo —0.0001 el de `S0`, 0.0004
el de `S1`— y costes de 0.003 la ida y vuelta, y todo lo neto se calcula a mano.

La evaluación `j` es la vela `168j`, o la `168j + 4` si `k = j // 2` es impar. Entra al cierre de
esa vela y sale al cierre de la tercera siguiente: la ventana es `(168j + 1, 168j + 4]` con `k`
par y `(168j + 5, 168j + 8]` con `k` impar. Solo la segunda contiene una hora múltiplo de 8, que
es donde hay liquidación. Por eso paga funding la mitad de las evaluaciones de cada símbolo.

Dos cosas que estas pruebas fijan, porque son el contrato del bloque:

- el neto es **descriptivo**: los veredictos, el bruto y el resto de la tabla son idénticos con
  o sin costes, con costes absurdos y con el funding ausente;
- lo que no se puede determinar se dice, y se cuenta: nunca se rellena con cero.
"""

from __future__ import annotations

import hashlib
import statistics
from typing import TYPE_CHECKING

import pytest

from crypto_agents import audit
from crypto_agents import metrics as metrics_module
from crypto_agents.audit import net_scores, net_section, read_run, render_audit
from crypto_agents.criteria import evaluate_criteria, render_criteria
from crypto_agents.funding import FundingSeries
from crypto_agents.metrics import (
    MIN_PAIRED_N,
    NET_LABEL,
    clustered_paired_difference,
    net_stats,
    paired_difference,
    paired_net_difference,
)
from crypto_agents.outcomes import NetInputs, Scoring
from crypto_agents.perp_probe import FundingRow, series_csv
from crypto_agents.settings import DEFAULT_COSTS, CostModel
from tests.test_criteria import (
    DELTA,
    EVEN,
    HISTORIES,
    HORIZON,
    ODD,
    START,
    N,
    criteria_run,
    expected,
    run_command,
    strata,
    symbol_of,
    write_run,
)
from tests.test_net_outcomes import mutated

if TYPE_CHECKING:
    from pathlib import Path

HOUR_MS = 3_600_000
GROSS_CELLS = 12
"""Celdas del bruto en una fila de comparación; las dos del neto van detrás."""
ORIGIN_MS = int(START.timestamp() * 1000)
COSTS = CostModel(taker_fee=0.001, slippage=0.0005)
RATE = {"S0": 0.0001, "S1": 0.0004}
ROUND_TRIP = 0.003


LAST_HOUR = (max(len(rows) for rows in HISTORIES.values()) // 8 + 2) * 8


def settlements(symbol: str, *, first_hour: int = -8, last_hour: int = LAST_HOUR) -> FundingSeries:
    """Liquidaciones cada 8 h con la tasa del símbolo, cubriendo todo el histórico sintético."""
    return FundingSeries.from_rows(
        [
            FundingRow(ORIGIN_MS + hour * HOUR_MS, RATE[symbol])
            for hour in range(first_hour, last_hour + 1, 8)
        ]
    )


FUNDING = {symbol: settlements(symbol) for symbol in ("S0", "S1")}
INPUTS = NetInputs(funding=FUNDING, costs=COSTS)


def pays(j: int) -> float:
    """La tasa que paga un largo en la evaluación `j`: una liquidación si `k` es impar, si no 0."""
    return RATE[symbol_of(j)] if (j // 2) % 2 == 1 else 0.0


def gross(j: int) -> float:
    return EVEN if j % 2 == 0 else ODD


def net_buy(j: int) -> float:
    return gross(j) - ROUND_TRIP - pays(j)


def net_sell(j: int) -> float:
    return -gross(j) - ROUND_TRIP + pays(j)


# ───────────────────────────────── Ayudantes de metrics ───────────────────────────────────────


def test_net_stats_trims_the_gross_to_the_same_evaluations() -> None:
    """Sin recortar, la diferencia bruto-neto mediría también el cambio de muestra."""
    stats = net_stats([0.02, 0.04, 0.10], [0.01, None, 0.03])
    assert stats.n == 2
    assert stats.undetermined == 1
    assert stats.net is not None
    assert stats.gross is not None
    assert stats.net.mean == pytest.approx(0.02)
    assert stats.gross.mean == pytest.approx(0.06), "0.10 y 0.02, no la media de las tres"


def test_net_stats_with_nothing_determined_has_no_figure_not_zero() -> None:
    stats = net_stats([0.02, 0.04], [None, None])
    assert (stats.n, stats.undetermined) == (0, 2)
    assert stats.net is None
    assert stats.gross is None


def test_net_stats_refuses_vectors_that_do_not_line_up() -> None:
    with pytest.raises(ValueError, match="distinta longitud"):
        net_stats([0.1], [0.1, 0.2])


PAIRS = 60
ALONE = list(range(PAIRS))
"""Cada pareja en su clúster: así el error agrupado es el ingenuo, y las cifras de
`paired_difference` son las del intervalo. Son 60 porque con menos de 50 clústeres no lo hay."""


def test_the_paired_net_difference_drops_the_pairs_it_cannot_make_and_counts_them() -> None:
    arm: list[float | None] = [0.01 * (i % 3) for i in range(PAIRS)]
    reference: list[float | None] = [0.005 * (i % 4) for i in range(PAIRS)]
    arm[3], reference[10], arm[20], reference[20] = None, None, None, None
    pairs = paired_net_difference(arm, reference, ALONE)

    assert pairs.dropped == 3, "la 20 tiene los dos en None y cuenta una vez"
    assert pairs.n == PAIRS - 3
    kept = [i for i in range(PAIRS) if i not in (3, 10, 20)]
    expected_diff = paired_difference(
        [arm[i] for i in kept],  # type: ignore[misc]
        [reference[i] for i in kept],  # type: ignore[misc]
    )
    assert expected_diff is not None
    assert pairs.pair is not None
    assert pairs.pair.naive == expected_diff
    assert pairs.pair.error.clusters == PAIRS - 3
    assert pairs.diff is not None
    assert (pairs.diff.n, pairs.diff.mean) == (expected_diff.n, expected_diff.mean)
    assert (pairs.diff.stderr, pairs.diff.low, pairs.diff.high) == pytest.approx(
        (expected_diff.stderr, expected_diff.low, expected_diff.high)
    )


def test_a_missing_net_is_not_filled_with_zero_when_pairing() -> None:
    """Rellenar con 0 daría otra media: la mutación que esta prueba cierra."""
    arm: list[float | None] = [0.02] * PAIRS
    reference: list[float | None] = [0.01] * (PAIRS - 1) + [None]
    pairs = paired_net_difference(arm, reference, ALONE)
    assert pairs.diff is not None
    assert pairs.diff.mean == pytest.approx(0.01)
    zero_filled = paired_difference([0.02] * PAIRS, [0.01] * (PAIRS - 1) + [0.0])
    assert zero_filled is not None
    assert zero_filled.mean != pytest.approx(pairs.diff.mean)


def test_fewer_than_thirty_pairs_have_no_interval() -> None:
    pairs = paired_net_difference([0.01] * 29, [0.0] * 29, list(range(29)))
    assert (pairs.n, pairs.dropped) == (29, 0)
    assert pairs.diff is None
    assert pairs.pair is None
    assert MIN_PAIRED_N == 30


def test_vectors_of_different_length_do_not_pair() -> None:
    pairs = paired_net_difference([0.01] * 40, [0.0] * 39, list(range(40)))
    assert (pairs.n, pairs.dropped, pairs.diff) == (0, 0, None)


def test_the_clusters_of_the_net_have_to_line_up_with_the_pairs() -> None:
    with pytest.raises(ValueError, match="no cuadran"):
        paired_net_difference([0.01] * 40, [0.0] * 40, list(range(39)))


def test_too_few_clusters_leave_the_net_without_an_interval_and_say_why() -> None:
    pairs = paired_net_difference([0.01, 0.02] * 20, [0.0] * 40, list(range(40)))
    assert pairs.diff is None
    assert pairs.pair is not None
    assert pairs.pair.why == "G = 40 clústeres, menos de 50"


NET_BLOCKS = 60
NET_CLUSTERS = [block for block in range(NET_BLOCKS) for _ in range(2)]
"""Dos parejas por bloque, con la misma diferencia dentro de cada uno."""


def assert_the_net_interval_is_clustered_with_the_blocks_of_the_pairs_it_keeps() -> None:
    """Se descartan una pareja del bloque 0 y las dos del bloque 5: cada una se lleva su clúster.

    Lo que queda —117 parejas en 59 bloques, el 0 con una sola— tiene que salir como si se
    hubiera pasado así desde el principio: el neto se agrupa con los bloques de las parejas
    que conserva, no con los primeros 117 de la lista.
    """
    arm: list[float | None] = [
        0.002 + (0.01 if block % 2 == 0 else -0.01) + 0.001 * (block % 5) for block in NET_CLUSTERS
    ]
    reference: list[float | None] = [0.0] * len(arm)
    for position in (0, 10, 11):
        reference[position] = None
    kept = [position for position in range(len(arm)) if position not in (0, 10, 11)]

    pairs = paired_net_difference(arm, reference, NET_CLUSTERS)
    expected_pair = clustered_paired_difference(
        [arm[position] for position in kept],  # type: ignore[misc]
        [0.0] * len(kept),
        [NET_CLUSTERS[position] for position in kept],
    )

    assert (pairs.n, pairs.dropped) == (117, 3)
    assert expected_pair is not None
    assert expected_pair.interval is not None
    assert expected_pair.error.clusters == NET_BLOCKS - 1
    assert expected_pair.error.clustered is not None
    assert expected_pair.error.clustered > expected_pair.error.naive, "agrupar aquí ensancha"
    assert pairs.pair == expected_pair
    assert pairs.diff == expected_pair.interval


def test_the_net_interval_is_the_clustered_one_over_the_pairs_it_keeps() -> None:
    assert_the_net_interval_is_clustered_with_the_blocks_of_the_pairs_it_keeps()


def assert_the_net_interval_has_the_floor_of_the_gross_one() -> None:
    """Dos parejas por bloque que se compensan: el agrupado sale menor y manda el ingenuo.

    Es la regla del bruto, no otra: con el agrupado a secas, la columna neta de una fila
    llevaría un intervalo más estrecho que el bruto que decide su veredicto.
    """
    arm: list[float | None] = [
        0.002 + 0.001 * (1 if block % 2 == 0 else -1) + (0.01 if position % 2 == 0 else -0.01)
        for position, block in enumerate(NET_CLUSTERS)
    ]
    reference: list[float | None] = [0.0] * len(arm)

    pairs = paired_net_difference(arm, reference, NET_CLUSTERS)

    assert pairs.pair is not None
    assert pairs.diff is not None
    error = pairs.pair.error
    assert error.clustered is not None
    assert error.clustered < error.naive, "agrupar aquí estrecharía"
    assert pairs.diff.stderr == error.naive
    assert pairs.diff == pairs.pair.naive


def test_the_net_interval_never_goes_below_the_naive_error() -> None:
    assert_the_net_interval_has_the_floor_of_the_gross_one()


def test_mutation_a_net_interval_without_the_floor_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mutant = mutated(
        metrics_module.clustered_paired_difference,
        "stderr = max(error.naive, error.clustered)",
        "stderr = error.clustered",
        metrics_module,
    )
    monkeypatch.setattr(metrics_module, "clustered_paired_difference", mutant)
    with pytest.raises(AssertionError):
        assert_the_net_interval_has_the_floor_of_the_gross_one()


def test_mutation_net_clusters_that_do_not_follow_the_dropped_pairs_are_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mutant = mutated(
        metrics_module.paired_net_difference,
        "[c for _, _, c in both]",
        "list(clusters)[: len(both)]",
        metrics_module,
    )
    monkeypatch.setitem(globals(), "paired_net_difference", mutant)
    with pytest.raises(AssertionError):
        assert_the_net_interval_is_clustered_with_the_blocks_of_the_pairs_it_keeps()


# ───────────────────────────────── Criterios: columnas netas ──────────────────────────────────


def test_the_net_difference_of_full_against_solo_matches_the_hand_computed_one() -> None:
    """`full` compra todo y `solo` las impares: la diferencia es el neto de `full` en las pares."""
    comparison = {
        c.reference: c
        for c in evaluate_criteria(
            criteria_run(), HISTORIES, HORIZON, DELTA, strata(), INPUTS
        ).criterion_1
    }["solo"]

    assert comparison.net is not None
    diffs = [net_buy(j) if j % 2 == 0 else 0.0 for j in range(N)]
    mean, stderr, low, high = expected(diffs)
    assert (comparison.net.n, comparison.net.dropped) == (N, 0)
    assert comparison.net.diff is not None
    assert comparison.net.diff.mean == pytest.approx(mean, abs=1e-12)
    assert comparison.net.diff.stderr == pytest.approx(stderr, abs=1e-12)
    assert (comparison.net.diff.low, comparison.net.diff.high) == pytest.approx((low, high))


def test_against_an_arm_that_sells_the_costs_cancel_and_the_funding_doubles() -> None:
    """`2r - 2·funding`: las dos patas pagan lo mismo de costes, y el funding se cobra y se paga."""
    comparison = {
        c.reference: c
        for c in evaluate_criteria(
            criteria_run(), HISTORIES, HORIZON, DELTA, strata(), INPUTS
        ).criterion_1
    }["bull_only"]

    assert comparison.net is not None
    assert comparison.net.diff is not None
    diffs = [net_buy(j) - net_sell(j) for j in range(N)]
    assert diffs == pytest.approx([2 * gross(j) - 2 * pays(j) for j in range(N)])
    assert comparison.net.diff.mean == pytest.approx(statistics.fmean(diffs), abs=1e-12)


def test_the_baselines_get_their_net_columns_too() -> None:
    report = evaluate_criteria(criteria_run(), HISTORIES, HORIZON, DELTA, strata(), INPUTS)
    comparisons = [c for table in report.criterion_3 for c in table.comparisons]
    assert comparisons
    assert all(c.net is not None and c.net.n + c.net.dropped == N for c in comparisons)


def without_net(comparisons: tuple[object, ...]) -> list[object]:
    return [c.model_copy(update={"net": None}) for c in comparisons]  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "inputs",
    [
        NetInputs(funding=FUNDING, costs=CostModel(taker_fee=0.0, slippage=0.0)),
        INPUTS,
        NetInputs(funding=FUNDING, costs=CostModel(taker_fee=0.4, slippage=0.5)),
        NetInputs(funding={}, costs=COSTS),
        NetInputs(funding={"S0": settlements("S0", last_hour=30)}, costs=COSTS),
    ],
    ids=["sin-costes", "costes", "costes-absurdos", "sin-funding", "funding-corto"],
)
def test_the_criteria_do_not_change_with_or_without_costs(inputs: NetInputs) -> None:
    """Los criterios siguen sobre el bruto: ni los costes ni el funding mueven un veredicto."""
    base = evaluate_criteria(criteria_run(), HISTORIES, HORIZON, DELTA, strata())
    other = evaluate_criteria(criteria_run(), HISTORIES, HORIZON, DELTA, strata(), inputs)

    assert without_net(other.criterion_1) == without_net(base.criterion_1)
    for table_a, table_b in zip(other.criterion_3, base.criterion_3, strict=True):
        assert table_a.arm == table_b.arm
        assert without_net(table_a.comparisons) == without_net(table_b.comparisons)
        assert table_a.beaten == table_b.beaten
    assert other.descriptive == base.descriptive
    assert (other.delta, other.horizon) == (base.delta, base.horizon)
    assert [c.verdict for c in other.criterion_1] == [c.verdict for c in base.criterion_1]


def test_without_funding_the_net_is_undetermined_and_the_rest_is_the_same() -> None:
    report = evaluate_criteria(
        criteria_run(), HISTORIES, HORIZON, DELTA, strata(), None, "no hay carpeta"
    )
    assert report.net_note == "no hay carpeta"
    assert all(c.net is None for c in report.criterion_1)
    text = render_criteria(report, "x")
    assert "no determinado: no hay carpeta" in text


def test_a_symbol_without_funding_drops_its_evaluations_and_says_how_many() -> None:
    """`S1` sin serie: sus 30 evaluaciones salen del neto. Quedan 30 parejas en 30 bloques, y
    con menos de 50 clústeres el neto tampoco tiene intervalo: lo dice, con su motivo."""
    partial = NetInputs(funding={"S0": FUNDING["S0"]}, costs=COSTS)
    report = evaluate_criteria(criteria_run(), HISTORIES, HORIZON, DELTA, strata(), partial)
    comparison = {c.reference: c for c in report.criterion_1}["solo"]

    assert comparison.net is not None
    assert (comparison.net.n, comparison.net.dropped) == (N // 2, N // 2)
    assert comparison.net.diff is None
    text = render_criteria(report, "x")
    row = next(line for line in text.splitlines() if line.startswith("| full | solo "))
    assert "30 (no determinado: 30)" in row
    assert "no determinado: G = 30 clústeres, menos de 50" in row
    assert "justifica" in row, "el veredicto del bruto no depende de cuántas parejas netas hay"


def test_a_funding_series_that_stops_early_leaves_too_few_pairs_and_says_so() -> None:
    """Una serie de `S0` que acaba a las 30 h solo cubre su primera evaluación: una pareja."""
    short = NetInputs(funding={"S0": settlements("S0", last_hour=30)}, costs=COSTS)
    report = evaluate_criteria(criteria_run(), HISTORIES, HORIZON, DELTA, strata(), short)
    comparison = {c.reference: c for c in report.criterion_1}["solo"]

    assert comparison.net is not None
    assert (comparison.net.n, comparison.net.dropped) == (1, N - 1)
    assert comparison.net.pair is None
    row = next(
        line
        for line in render_criteria(report, "x").splitlines()
        if line.startswith("| full | solo ")
    )
    assert f"1 (no determinado: {N - 1})" in row
    assert f"menos de {MIN_PAIRED_N} parejas" in row


def test_the_rendered_tables_carry_the_label_in_two_separate_columns() -> None:
    text = render_criteria(
        evaluate_criteria(criteria_run(), HISTORIES, HORIZON, DELTA, strata(), INPUTS), "x"
    )
    header = next(line for line in text.splitlines() if line.startswith("| brazo | referencia"))
    cells = [cell.strip() for cell in header.strip("|").split("|")]
    assert cells[:GROSS_CELLS] == [
        "brazo",
        "referencia",
        "n",
        "diferencia media",
        "EE ingenuo",
        "EE agrupado",
        "EE usado",
        "G",
        "deff",
        "IC 95%",
        "semiancho",
        "veredicto",
    ], "las doce columnas del bruto, con los tres errores de la enmienda 4"
    assert cells[GROSS_CELLS:] == [
        f"{NET_LABEL}: n",
        f"{NET_LABEL}: diferencia media [IC 95%]",
    ]
    assert text.count(NET_LABEL) >= 4, "criterio 1 y criterio 3, dos columnas cada uno"


def comparison_rows(text: str) -> list[list[str]]:
    """Las filas de las dos tablas de comparación, con todas sus celdas."""
    rows = []
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if (
            line.startswith("| ")
            and len(cells) == GROSS_CELLS + 2
            and cells[0] != "brazo"
            and "---" not in line
        ):
            rows.append(cells)
    return rows


def test_the_gross_cells_of_every_row_are_identical_with_and_without_costs() -> None:
    """Las doce primeras celdas de cada fila, que son las del bruto, no se mueven."""
    with_net = render_criteria(
        evaluate_criteria(criteria_run(), HISTORIES, HORIZON, DELTA, strata(), INPUTS), "x"
    )
    without = render_criteria(
        evaluate_criteria(criteria_run(), HISTORIES, HORIZON, DELTA, strata()), "x"
    )
    rows_with, rows_without = comparison_rows(with_net), comparison_rows(without)

    assert len(rows_with) == len(rows_without) >= 7
    assert [row[:GROSS_CELLS] for row in rows_with] == [row[:GROSS_CELLS] for row in rows_without]
    assert [row[GROSS_CELLS:] for row in rows_with] != [
        row[GROSS_CELLS:] for row in rows_without
    ], "lo único que cambia son las dos columnas netas"


# ───────────────────────────────────────── El comando ─────────────────────────────────────────


def write_funding(directory: Path) -> Path:
    directory.mkdir()
    for symbol, series in FUNDING.items():
        rows = [
            FundingRow(timestamp_ms=time, rate=rate)
            for time, rate in zip(series.times, series.rates, strict=True)
        ]
        (directory / f"{symbol.lower()}.csv").write_bytes(series_csv(rows))
    return directory


def test_the_command_prints_the_net_columns_the_costs_and_the_digest_of_each_series(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = write_run(tmp_path, criteria_run())
    funding_dir = write_funding(tmp_path / "funding")

    assert run_command(directory, tmp_path, "--funding-dir", str(funding_dir)) == 0
    out = capsys.readouterr().out

    assert NET_LABEL in out
    assert f"comisión taker {DEFAULT_COSTS.taker_fee:.4%}" in out
    assert "supuesto sin medir" in out
    for symbol in ("S0", "S1"):
        digest = hashlib.sha256((funding_dir / f"{symbol.lower()}.csv").read_bytes()).hexdigest()
        assert f"funding `{symbol}` sha-256 `{digest}`" in out, "cada serie cita su archivo"
    vs_solo = next(line for line in out.splitlines() if line.startswith("| full | solo "))
    assert "justifica" in vs_solo
    assert "no determinado" not in vs_solo


def test_the_command_without_funding_still_evaluates_the_criteria(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = write_run(tmp_path, criteria_run())
    assert run_command(directory, tmp_path, "--funding-dir", str(tmp_path / "nada")) == 0
    out = capsys.readouterr().out
    vs_solo = next(line for line in out.splitlines() if line.startswith("| full | solo "))
    assert "justifica" in vs_solo, "el veredicto es del bruto y no necesita el funding"
    assert "no determinado" in vs_solo
    assert "funding: no determinado" in out


def test_the_verdicts_printed_by_the_command_do_not_depend_on_the_funding(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = write_run(tmp_path, criteria_run())
    funding_dir = write_funding(tmp_path / "funding")

    run_command(directory, tmp_path, "--funding-dir", str(funding_dir))
    with_funding = capsys.readouterr().out
    run_command(directory, tmp_path, "--funding-dir", str(tmp_path / "nada"))
    without = capsys.readouterr().out

    def verdicts(text: str) -> list[list[str]]:
        return [row[:8] for row in comparison_rows(text)]

    assert verdicts(with_funding) == verdicts(without)
    assert len(verdicts(with_funding)) >= 7


# ──────────────────────────────────────── Auditoría ───────────────────────────────────────────


def audit_args(tmp_path: Path, funding_dir: Path) -> tuple[Path, Path]:
    return tmp_path / "history", funding_dir


def test_the_audit_publishes_the_net_return_per_evaluation_apart_and_labelled(
    tmp_path: Path,
) -> None:
    run = read_run(write_run(tmp_path, criteria_run()))
    history_dir, funding_dir = audit_args(tmp_path, write_funding(tmp_path / "funding"))
    section = net_section(run, None, history_dir, funding_dir, None)

    assert section.why is None
    assert section.horizon == HORIZON
    text = render_audit(run, None, section)
    assert f"## 8. Retorno por evaluación, {NET_LABEL}" in text
    assert "no los toca" in text, "dice que los criterios siguen sobre el bruto"
    assert "supuesto sin medir" in text


def test_the_net_return_of_an_arm_matches_the_hand_computed_one(tmp_path: Path) -> None:
    """`full` compra todo: neto medio = bruto medio - costes - funding medio pagado."""
    run = read_run(write_run(tmp_path, criteria_run()))
    scores = net_scores(run, HISTORIES, HORIZON, INPUTS)

    full = scores["full"][Scoring.COMMON_STOP]
    assert full.per_evaluation_net is not None
    assert full.per_evaluation_net == pytest.approx([net_buy(j) for j in range(N)], abs=1e-12)
    stats = net_stats(full.per_evaluation, full.per_evaluation_net)
    assert stats.net is not None
    payers = N // 4
    assert payers == 15, "de las 30 de cada símbolo paga la mitad: las de `k` impar"
    assert stats.net.mean == pytest.approx(
        0.010 - ROUND_TRIP - (payers * 0.0001 + payers * 0.0004) / N
    )
    assert stats.gross is not None
    assert stats.gross.mean == pytest.approx(0.010), "el bruto es el de siempre"

    sells = scores["bull_only"][Scoring.COMMON_STOP]
    assert sells.per_evaluation_net == pytest.approx([net_sell(j) for j in range(N)], abs=1e-12)


def test_an_arm_that_never_trades_pays_nothing(tmp_path: Path) -> None:
    """Un hold vale 0 en el bruto y en el neto: no hay posición a la que cobrar nada."""
    arms = criteria_run()
    arms["solo"] = [r.model_copy(update={"proposal": None}) for r in arms["solo"]]
    run = read_run(write_run(tmp_path, arms))
    solo = net_scores(run, HISTORIES, HORIZON, INPUTS)["solo"][Scoring.COMMON_STOP]
    assert solo.per_evaluation_net == (0.0,) * N


def test_the_audit_scores_the_three_scorings(tmp_path: Path) -> None:
    run = read_run(write_run(tmp_path, criteria_run()))
    scores = net_scores(run, HISTORIES, HORIZON, INPUTS)
    assert set(scores["full"]) == set(Scoring)
    text = render_audit(
        run, None, net_section(run, None, tmp_path / "history", write_funding(tmp_path / "f"), None)
    )
    for label in ("stop declarado", "stop común", "cierre del horizonte"):
        assert label in text


def test_without_the_candles_or_funding_the_audit_still_reports_and_says_why(
    tmp_path: Path,
) -> None:
    run = read_run(write_run(tmp_path, criteria_run()))
    missing = net_section(run, None, tmp_path / "history", tmp_path / "nada", None)
    assert missing.scores is None
    assert missing.why is not None

    text = render_audit(run, None, missing)
    assert "no determinado" in text.split("## 8.")[1]
    assert text.split("## 8.")[0].rstrip("\n") == render_audit(run).rstrip("\n"), (
        "todo lo anterior a la sección neta es el informe de siempre"
    )


def test_a_plan_that_is_not_the_runs_leaves_the_net_undetermined_not_wrong(
    tmp_path: Path,
) -> None:
    run = read_run(write_run(tmp_path, criteria_run()))
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    section = net_section(
        run, None, tmp_path / "history", write_funding(tmp_path / "funding"), None
    )
    assert section.scores is None
    assert section.why is not None
    assert "no es el de la corrida" in section.why


def test_the_report_without_a_net_section_is_unchanged(tmp_path: Path) -> None:
    run = read_run(write_run(tmp_path, criteria_run()))
    assert "## 8." not in render_audit(run)


def test_the_audit_command_prints_the_net_section(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = write_run(tmp_path, criteria_run())
    funding_dir = write_funding(tmp_path / "funding")
    code = audit.main(
        [
            str(directory),
            "--history-dir",
            str(tmp_path / "history"),
            "--funding-dir",
            str(funding_dir),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert NET_LABEL in out
    assert "funding `S0`: sha-256" in out
    assert "funding `S1`: sha-256" in out


def test_the_audit_command_does_not_fail_when_the_net_cannot_be_computed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = write_run(tmp_path, criteria_run())
    code = audit.main(
        [
            str(directory),
            "--history-dir",
            str(tmp_path / "history"),
            "--funding-dir",
            str(tmp_path / "nada"),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "## 1." in out
    assert "no determinado" in out.split("## 8.")[1]
