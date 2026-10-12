"""Pruebas del recálculo de dispersión de la enmienda 3.

Tres cosas tienen que ser ciertas para que la cifra sirva, y cada una tiene su mutación:

- **La aritmética.** `sigma_D` y los dos n sobre un vector hecho a mano, con los valores
  calculados a mano. Cambiar el z, usar el error estándar en lugar de la desviación, olvidar el
  cuadrado o dividir entre n en lugar de n - 1 tiene que romper la comprobación.
- **El vector.** Las diferencias son las que resta `criteria`, no una copia: un espía en la resta
  de `metrics` ve pasar los mismos vectores por los dos caminos.
- **Lo que no se dice.** Ninguna media, ninguna recomendación, ningún archivo escrito.

El escenario sintético es el de `tests/test_criteria.py`: cada evaluación tiene su retorno
escrito de antemano.
"""

from __future__ import annotations

import inspect
import statistics
from datetime import timedelta
from typing import TYPE_CHECKING

import pytest

from crypto_agents import metrics as metrics_module
from crypto_agents import outcomes as outcomes_module
from crypto_agents import paired_dispersion as module
from crypto_agents.activation_sweep import ActivatedBar
from crypto_agents.audit import RunKind, file_sha256
from crypto_agents.consumption import consume
from crypto_agents.criteria import PairedReturns, evaluate_criteria, paired_returns
from crypto_agents.dispersion import Z_INTERVAL
from crypto_agents.outcomes import score_position
from crypto_agents.paired_dispersion import (
    PAIRS,
    ArmCost,
    PairDispersion,
    Pool,
    PoolCell,
    Recalculation,
    RecalculationError,
    Sources,
    WallClock,
    arm_costs,
    bootstrap_sigma,
    build_report,
    main,
    n_for_half_width,
    n_for_power,
    pair_dispersion,
    pool_cells,
    render_report,
    round_up_to,
    sigma_of,
    stage_hours,
    stage_usd,
    wall_clock,
)
from crypto_agents.selection import load_manifest
from crypto_agents.settings import DEFAULT_PRICING
from crypto_agents.state import Action, AgentRole, Billing, FrozenModel
from tests.test_criteria import (
    DELTA,
    HISTORIES,
    HORIZON,
    START,
    N,
    arm,
    live_call,
    make_record,
    quota_lost,
    write_plan,
    write_run,
)
from tests.test_net_outcomes import mutated

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from crypto_agents.journal import EvaluationRecord
    from crypto_agents.state import LLMCall

RESAMPLES = 200
"""Remuestreos de las pruebas: los suficientes para tener rango, sin pagar los 10 000."""

HAND = [0.02, -0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, -0.02]
"""Diez diferencias de media 0: Σd² = 4·0.0004 = 0.0016, s² = 0.0016/9, s = 0.04/3."""

HAND_SYMBOLS = ["A"] * 5 + ["B"] * 5
HAND_SIGMA = 0.04 / 3
HAND_HALF_WIDTH_N = 170.7378
"""`(1.96·s/δ)²` con δ = 0.002: `1.96·20/3 = 13.0667`, al cuadrado."""

HAND_POWER_N = 348.848
"""`((1.96 + 0.8416)·s/δ)²`: `2.8016·20/3 = 18.6775`, al cuadrado."""

UNIT = 35


# ─────────────────────────────── sigma_D y los n, calculados a mano ───────────────────────────────


def assert_hand_computed_figures() -> None:
    """La comprobación que cada mutación tiene que romper. Llama por el módulo, no por el nombre
    importado, para que vea lo que la mutación haya dejado puesto."""
    found = module.dispersion_of(HAND, HAND_SYMBOLS, DELTA, UNIT, RESAMPLES, seed=1)

    assert found.n == 10
    assert found.sigma == pytest.approx(HAND_SIGMA, rel=1e-12)
    assert found.stderr == pytest.approx(HAND_SIGMA / 10**0.5, rel=1e-12)
    low, point, high = found.half_width_n
    assert point.sigma == found.sigma
    assert point.exact == pytest.approx(HAND_HALF_WIDTH_N, abs=1e-4)
    assert point.rounded == 175
    assert found.power_n[1].exact == pytest.approx(HAND_POWER_N, abs=1e-3)
    assert found.power_n[1].rounded == 350
    assert (low.sigma, high.sigma) == (found.bootstrap.low, found.bootstrap.high)
    assert low.exact <= point.exact <= high.exact


def test_the_hand_computed_figures_hold_for_the_real_code() -> None:
    """El control de las mutaciones: sin error puesto, la comprobación pasa."""
    assert_hand_computed_figures()


@pytest.mark.parametrize("function", [n_for_half_width, n_for_power])
def test_mutation_a_ninety_percent_z_is_caught(
    function: Callable[..., float], monkeypatch: pytest.MonkeyPatch
) -> None:
    """z = 1.645 es el de un intervalo al 90%: el n sale un 30% más corto y nadie lo vería."""
    mutant = mutated(function, "Z_INTERVAL", "1.645", module)
    monkeypatch.setattr(module, function.__name__, mutant)
    with pytest.raises(AssertionError):
        assert_hand_computed_figures()


def test_mutation_the_standard_error_in_place_of_the_deviation_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """El EE ya está dividido entre √n: usarlo como desviación divide el n entre las 140."""
    mutant = mutated(
        module.dispersion_of,
        "sigma = sigma_of(differences)",
        "sigma = sigma_of(differences) / math.sqrt(len(differences))",
        module,
    )
    monkeypatch.setattr(module, "dispersion_of", mutant)
    with pytest.raises(AssertionError):
        assert_hand_computed_figures()


@pytest.mark.parametrize("function", [n_for_half_width, n_for_power])
def test_mutation_forgetting_the_square_is_caught(
    function: Callable[..., float], monkeypatch: pytest.MonkeyPatch
) -> None:
    mutant = mutated(function, " ** 2", "", module)
    monkeypatch.setattr(module, function.__name__, mutant)
    with pytest.raises(AssertionError):
        assert_hand_computed_figures()


def test_mutation_dividing_by_n_instead_of_n_minus_one_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """La poblacional da 0.04/√10 y un n de 153.7 donde la muestral da 170.7."""
    mutant = mutated(sigma_of, "statistics.stdev(", "statistics.pstdev(", module)
    monkeypatch.setattr(module, "sigma_of", mutant)
    with pytest.raises(AssertionError):
        assert_hand_computed_figures()


def assert_a_constant_changes_no_n() -> None:
    base = module.dispersion_of(HAND, HAND_SYMBOLS, DELTA, UNIT, RESAMPLES, seed=1)
    shifted = module.dispersion_of(
        [value + 0.05 for value in HAND], HAND_SYMBOLS, DELTA, UNIT, RESAMPLES, seed=1
    )

    assert shifted.sigma == pytest.approx(base.sigma, rel=1e-9)
    assert shifted.bootstrap.low == pytest.approx(base.bootstrap.low, rel=1e-9)
    assert shifted.bootstrap.high == pytest.approx(base.bootstrap.high, rel=1e-9)
    for moved, still in zip(
        (*shifted.half_width_n, *shifted.power_n), (*base.half_width_n, *base.power_n), strict=True
    ):
        assert moved.exact == pytest.approx(still.exact, rel=1e-9)
        assert moved.rounded == still.rounded


def test_adding_a_constant_to_every_difference_changes_no_n() -> None:
    """El n depende de `sigma_D` y de δ, no de la media observada: sumar un 5% a todas las
    diferencias no mueve ninguno, ni el puntual ni los de los extremos del rango."""
    assert_a_constant_changes_no_n()


def test_mutation_a_deviation_around_zero_would_follow_the_mean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """La mutación: la raíz del cuadrado medio, que no resta la media. Sobre el vector a mano,
    que tiene media 0, da lo mismo que la desviación; desplazado, ya no."""
    mutant = mutated(
        sigma_of,
        "statistics.stdev(differences)",
        "math.sqrt(sum(value * value for value in differences) / (len(differences) - 1))",
        module,
    )
    monkeypatch.setattr(module, "sigma_of", mutant)
    with pytest.raises(AssertionError):
        assert_a_constant_changes_no_n()


@pytest.mark.parametrize(
    ("arguments", "function"),
    [
        ({"sigma": -0.01, "delta": DELTA}, n_for_half_width),
        ({"sigma": float("nan"), "delta": DELTA}, n_for_half_width),
        ({"sigma": 0.01, "delta": 0.0}, n_for_half_width),
        ({"sigma": 0.01, "delta": DELTA, "power": 1.0}, n_for_power),
        ({"sigma": 0.01, "delta": -DELTA}, n_for_power),
    ],
)
def test_arguments_outside_their_domain_are_refused(
    arguments: dict[str, float], function: Callable[..., float]
) -> None:
    with pytest.raises(ValueError):
        function(**arguments)


def test_one_difference_has_no_deviation() -> None:
    with pytest.raises(ValueError, match="al menos dos"):
        sigma_of([0.01])


def test_the_power_n_is_the_one_the_dispersion_module_calls_detectable() -> None:
    """Con el n de la potencia, lo que `dispersion.effect_for` llama detectable vale δ: las dos
    cuentas son la misma leída al revés, y con el mismo z."""
    sigma = 0.0227
    n = n_for_power(sigma, DELTA)
    exact = (Z_INTERVAL + statistics.NormalDist().inv_cdf(0.8)) * sigma / n**0.5

    assert exact == pytest.approx(DELTA, rel=1e-12)
    half_width = Z_INTERVAL * sigma / n_for_half_width(sigma, DELTA) ** 0.5
    assert half_width == pytest.approx(DELTA, rel=1e-12)


# ──────────────────────────────────── El redondeo al múltiplo ─────────────────────────────────────


def assert_rounding_goes_up_and_keeps_an_exact_multiple() -> None:
    assert module.round_up_to(525.0, 35) == 525
    assert module.round_up_to(525.0001, 35) == 560
    assert module.round_up_to(496.4, 35) == 525
    assert module.round_up_to(1.0, 35) == 35
    assert module.round_up_to(0.0, 35) == 0


def test_rounding_goes_up_to_the_multiple_and_keeps_an_exact_one() -> None:
    assert_rounding_goes_up_and_keeps_an_exact_multiple()


@pytest.mark.parametrize(
    "replacement", ["math.floor(n / unit) * unit", "(math.floor(n / unit) + 1) * unit"]
)
def test_mutation_rounding_down_or_always_one_more_is_caught(
    replacement: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hacia abajo pierde el 525.0001; «siempre uno más» convierte el 525 exacto en 560."""
    mutant = mutated(round_up_to, "math.ceil(n / unit) * unit", replacement, module)
    monkeypatch.setattr(module, "round_up_to", mutant)
    with pytest.raises(AssertionError):
        assert_rounding_goes_up_and_keeps_an_exact_multiple()


@pytest.mark.parametrize(("n", "unit"), [(10.0, 0), (-1.0, 35), (float("inf"), 35)])
def test_rounding_refuses_what_it_cannot_round(n: float, unit: int) -> None:
    with pytest.raises(ValueError):
        round_up_to(n, unit)


# ────────────────────────────────────────── Bootstrap ─────────────────────────────────────────────

SPREAD = [0.03, -0.01, 0.0, 0.02, -0.04, 0.0, 0.01, 0.0, -0.02, 0.05, 0.0, -0.03]
SPREAD_SYMBOLS = ["A", "B", "C"] * 4


def assert_the_same_seed_gives_the_same_range() -> None:
    for seed in range(7, 12):
        first = module.bootstrap_sigma(SPREAD, SPREAD_SYMBOLS, RESAMPLES, seed, "full|solo")
        second = module.bootstrap_sigma(SPREAD, SPREAD_SYMBOLS, RESAMPLES, seed, "full|solo")
        assert first == second
        assert first.low < sigma_of(SPREAD) < first.high


def assert_another_seed_gives_another_range() -> None:
    first = module.bootstrap_sigma(SPREAD, SPREAD_SYMBOLS, RESAMPLES, seed=7, label="full|solo")
    other = module.bootstrap_sigma(SPREAD, SPREAD_SYMBOLS, RESAMPLES, seed=8, label="full|solo")

    assert (first.low, first.high) != (other.low, other.high)


def test_the_same_seed_gives_the_same_range() -> None:
    assert_the_same_seed_gives_the_same_range()


def test_another_seed_gives_another_range() -> None:
    assert_another_seed_gives_another_range()


@pytest.mark.parametrize(
    ("generator", "check"),
    [
        ("random.Random()", assert_the_same_seed_gives_the_same_range),
        ('random.Random("fija")', assert_another_seed_gives_another_range),
    ],
)
def test_mutation_a_generator_that_ignores_the_declared_seed_is_caught(
    generator: str, check: Callable[[], None], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sin sembrar, dos ejecuciones no dan el mismo rango y la cifra no se puede reproducir; con
    una semilla escrita dentro, la declarada en la cabecera sería decoración."""
    mutant = mutated(bootstrap_sigma, 'random.Random(f"{seed}|{label}")', generator, module)
    monkeypatch.setattr(module, "bootstrap_sigma", mutant)
    with pytest.raises(AssertionError):
        check()


def test_each_pair_resamples_with_its_own_generator() -> None:
    """Con la misma semilla, otro par remuestrea distinto: añadir un par no mueve a los demás,
    y dos pares no comparten los mismos índices por haber salido del mismo generador."""
    first = bootstrap_sigma(SPREAD, SPREAD_SYMBOLS, RESAMPLES, seed=7, label="full|solo")
    other = bootstrap_sigma(SPREAD, SPREAD_SYMBOLS, RESAMPLES, seed=7, label="solo|rule_trend")

    assert (first.low, first.high) != (other.low, other.high)


CONSTANT = [0.0] * 6 + [0.02] * 6
CONSTANT_SYMBOLS = ["A"] * 6 + ["B"] * 6


def assert_resampling_stays_inside_each_symbol() -> None:
    found = module.bootstrap_sigma(CONSTANT, CONSTANT_SYMBOLS, RESAMPLES, seed=7)
    exact = sigma_of(CONSTANT)

    assert found.low == pytest.approx(exact, rel=1e-12)
    assert found.high == pytest.approx(exact, rel=1e-12)


def test_the_resampling_keeps_how_many_evaluations_each_symbol_has() -> None:
    """Dos símbolos de valor constante: remuestreando dentro de cada uno, toda réplica es la
    muestra misma y el rango no tiene ancho."""
    assert_resampling_stays_inside_each_symbol()


def test_mutation_a_simple_resampling_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """La mutación: un solo grupo con todas las evaluaciones. El reparto por símbolo ya varía de
    réplica en réplica y el rango se abre."""
    mutant = mutated(
        bootstrap_sigma,
        "ordered = [groups[symbol] for symbol in sorted(groups)]",
        "ordered = [list(differences)]",
        module,
    )
    monkeypatch.setattr(module, "bootstrap_sigma", mutant)
    with pytest.raises(AssertionError):
        assert_resampling_stays_inside_each_symbol()


def test_the_range_is_the_nearest_rank_of_the_resampled_deviations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Con las desviaciones remuestreadas sustituidas por 1..1000, los percentiles 2.5 y 97.5
    son los valores de rango 25 y 975: valores que ocurrieron, sin interpolar."""
    counter = iter(range(1, 1001))
    monkeypatch.setattr(module, "sigma_of", lambda _sample: float(next(counter)))

    found = module.bootstrap_sigma(SPREAD, SPREAD_SYMBOLS, 1000, seed=7)

    assert (found.low, found.high) == (25.0, 975.0)


def test_differences_and_symbols_must_line_up() -> None:
    with pytest.raises(ValueError, match="no cuadran"):
        bootstrap_sigma(SPREAD, SPREAD_SYMBOLS[:-1], RESAMPLES)
    with pytest.raises(ValueError, match="remuestreo"):
        bootstrap_sigma(SPREAD, SPREAD_SYMBOLS, 0)


# ──────────────────────────────── El vector es el que resta criteria ──────────────────────────────


def stage_run() -> dict[str, list[EvaluationRecord]]:
    """Seis brazos como los de la etapa 1, con decisiones escritas a mano.

    `full` compra todo salvo la evaluación 5, que no decide. `solo` compra las impares y hace
    hold en las pares. Las líneas base: comprar todo, vender todo, comprar las pares, hold.
    """
    full = arm("buy")
    full[5] = make_record(5, None)
    return {
        "full": full,
        "solo": arm("odd"),
        "always_buy": arm("buy"),
        "always_sell": arm("sell"),
        "random_uniform": arm("even"),
        "rule_trend": arm("hold"),
    }


def aligned(left: str, right: str) -> PairedReturns:
    found = paired_returns(left, right, stage_run(), HISTORIES, HORIZON)
    assert isinstance(found, PairedReturns)
    return found


def spy_on_the_subtraction(
    monkeypatch: pytest.MonkeyPatch, target: object
) -> list[tuple[tuple[float, ...], tuple[float, ...]]]:
    """Sustituye `paired_differences` en `target` por la real con un testigo delante."""
    seen: list[tuple[tuple[float, ...], tuple[float, ...]]] = []
    real = metrics_module.paired_differences

    def spy(arm_returns: Sequence[float], reference_returns: Sequence[float]) -> list[float]:
        seen.append((tuple(arm_returns), tuple(reference_returns)))
        return real(arm_returns, reference_returns)

    monkeypatch.setattr(target, "paired_differences", spy)
    return seen


def assert_the_module_subtracts_what_criteria_subtracts(monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as patch:
        by_criteria = spy_on_the_subtraction(patch, metrics_module)
        evaluate_criteria(stage_run(), HISTORIES, HORIZON, DELTA)
    with monkeypatch.context() as patch:
        by_module = spy_on_the_subtraction(patch, module)
        for left, right in PAIRS:
            module.pair_dispersion(
                left, right, aligned(left, right), 0.0, DELTA, UNIT, RESAMPLES, seed=1
            )

    assert len(by_module) == len(PAIRS)
    assert set(by_module) <= set(by_criteria)


def test_the_differences_are_the_ones_criteria_subtracts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Un testigo en la resta de `metrics` ve lo que `evaluate_criteria` resta; otro en la del
    módulo, lo que resta el recálculo. Los nueve pares de aquí están entre los de allí, vector
    por vector y en el mismo orden de brazos."""
    assert vars(module)["paired_differences"] is metrics_module.paired_differences
    assert_the_module_subtracts_what_criteria_subtracts(monkeypatch)


def test_mutation_subtracting_the_other_way_round_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """La mutación que la comprobación del EE no ve: `referencia - brazo` tiene la misma
    desviación. El testigo sí la ve, porque los vectores llegan cambiados de sitio."""
    mutant = mutated(
        pair_dispersion,
        "paired_differences(pair.arm_returns, pair.reference_returns)",
        "paired_differences(pair.reference_returns, pair.arm_returns)",
        module,
    )
    monkeypatch.setattr(module, "pair_dispersion", mutant)
    with pytest.raises(AssertionError):
        assert_the_module_subtracts_what_criteria_subtracts(monkeypatch)


def synthetic_pool() -> Pool:
    """Dos símbolos por cinco tramos, cuatro usadas por celda y una celda algo más corta."""
    return Pool(
        activations=64,
        cells=tuple(
            PoolCell(
                symbol=symbol,
                stratum=stratum,
                candidates=7,
                confirmed=5 if (symbol, stratum) == ("S1", 3) else 6,
                used=4,
            )
            for symbol in ("S0", "S1")
            for stratum in range(5)
        ),
    )


def decider_call(minutes: float, cached: int | None = 100) -> LLMCall:
    """Una llamada viva del decisor con sus tokens, `minutes` después del inicio."""
    return live_call(
        role=AgentRole.DECIDER, model="glm-5.2", at=START + timedelta(minutes=minutes)
    ).model_copy(update={"prompt_tokens": 1000, "cached_tokens": cached, "completion_tokens": 500})


def with_calls(
    records: Sequence[EvaluationRecord], step: float, cached: int | None = 100
) -> list[EvaluationRecord]:
    """Los mismos registros, cada uno con una llamada del decisor cada `step` minutos."""
    return [
        record.model_copy(update={"calls": (decider_call(index * step, cached),)})
        for index, record in enumerate(records)
    ]


def paid_run() -> dict[str, list[EvaluationRecord]]:
    """`stage_run` con las llamadas de los dos brazos con modelo: `full` sin `cached_tokens`."""
    arms = stage_run()
    arms["full"] = with_calls(arms["full"], step=2.0, cached=None)
    arms["solo"] = with_calls(arms["solo"], step=1.0)
    return arms


def report_of(
    arms: dict[str, list[EvaluationRecord]] | None = None,
    tmp_path: Path | None = None,
    billing: Billing | None = Billing.PAYG,
    resumed: bool = False,
) -> Recalculation:
    arms = paid_run() if arms is None else arms
    assert tmp_path is not None
    manifest = load_manifest(write_plan(tmp_path))
    calls = {
        name: [call for record in records for call in record.calls]
        for name, records in arms.items()
    }
    return build_report(
        arms,
        HISTORIES,
        HORIZON,
        DELTA,
        manifest,
        synthetic_pool(),
        calls,
        DEFAULT_PRICING,
        billing,
        RESAMPLES,
        seed=1,
        resumed=resumed,
    )


def test_every_pair_reproduces_the_standard_error_criteria_computes(tmp_path: Path) -> None:
    """`sigma_D/√n` es, número por número, el EE de `evaluate_criteria` para los nueve pares."""
    report = report_of(tmp_path=tmp_path)
    criteria = evaluate_criteria(paid_run(), HISTORIES, HORIZON, DELTA)
    published = {
        (item.arm, item.reference): item.diff
        for item in (
            *criteria.criterion_1,
            *(row for table in criteria.criterion_3 for row in table.comparisons),
        )
    }

    assert [(pair.arm, pair.reference) for pair in report.pairs] == list(PAIRS)
    assert report.missing == ()
    for pair in report.pairs:
        diff = published[(pair.arm, pair.reference)]
        assert diff is not None
        assert pair.n == N
        assert pair.dispersion.stderr == diff.stderr
        assert pair.criteria_stderr == diff.stderr
        assert pair.reproduces
    assert report.consistent


def test_a_deviation_that_is_not_the_one_of_criteria_is_marked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Con la desviación poblacional `sigma_D/√n` ya no es el EE de `criteria`: la fila lo dice
    y el reporte deja de darse por bueno."""
    monkeypatch.setattr(
        module, "sigma_of", mutated(sigma_of, "statistics.stdev(", "statistics.pstdev(", module)
    )

    report = report_of(tmp_path=tmp_path)

    assert not report.pairs[0].reproduces
    assert not report.consistent
    assert "**NO**" in render_report(report, SOURCES)


# ─────────────────────────── La misma acción aporta exactamente cero ──────────────────────────────


@pytest.mark.parametrize("action", [Action.BUY, Action.SELL, Action.HOLD, None])
def test_two_arms_with_the_same_action_contribute_exactly_zero(action: Action | None) -> None:
    """Con el stop común la misma acción en la misma evaluación es el mismo retorno: la
    diferencia es 0.0 exacto, no un número pequeño. Vale para «no decidió» también."""
    records = [make_record(j, action) for j in range(N)]
    pair = paired_returns("uno", "otro", {"uno": records, "otro": records}, HISTORIES, HORIZON)

    assert isinstance(pair, PairedReturns)
    differences = metrics_module.paired_differences(pair.arm_returns, pair.reference_returns)
    assert differences == [0.0] * N
    found = pair_dispersion("uno", "otro", pair, 0.0, DELTA, UNIT, RESAMPLES)
    assert (found.agreed, found.nonzero, found.nonzero_agreed) == (N, 0, 0)
    assert found.dispersion.sigma == 0.0
    assert found.dispersion.half_width_n[1].rounded == 0


def assert_the_declared_stop_does_not_reach_the_difference() -> None:
    for action, tight in ((Action.BUY, 99.5), (Action.SELL, 100.5)):
        ours = [make_record(j, action) for j in range(N)]
        theirs = []
        for record in ours:
            assert record.proposal is not None
            proposal = record.proposal.model_copy(update={"invalidation_price": tight})
            theirs.append(record.model_copy(update={"proposal": proposal}))

        pair = paired_returns("uno", "otro", {"uno": ours, "otro": theirs}, HISTORIES, HORIZON)

        assert isinstance(pair, PairedReturns)
        assert all(value != 0.0 for value in pair.arm_returns)
        assert metrics_module.paired_differences(pair.arm_returns, pair.reference_returns) == (
            [0.0] * N
        )


def test_the_same_action_with_another_declared_stop_still_contributes_zero() -> None:
    """Es el caso real: `full` y `solo` compran la misma vela con stops distintos. Uno declara
    el suyo a medio punto de la entrada, donde la primera vela lo tocaría; con el stop común ese
    stop no se lee, y la diferencia sigue siendo 0.0 exacto."""
    assert_the_declared_stop_does_not_reach_the_difference()


def test_mutation_scoring_with_the_declared_stop_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """La mutación: el «stop común» lee el stop de cada decisor. Dos brazos con la misma acción
    dejan de rendir lo mismo y la premisa de la enmienda 3 cae."""
    mutant = mutated(
        score_position,
        "stop = common_stop(side, entry, atr_of(record.indicators))",
        "stop = proposed.invalidation_price",
        outcomes_module,
    )
    monkeypatch.setattr(outcomes_module, "score_position", mutant)
    with pytest.raises(AssertionError):
        assert_the_declared_stop_does_not_reach_the_difference()


def test_a_difference_is_only_nonzero_where_the_arms_disagree() -> None:
    """`full` compra salvo en la 5, que no decide; `solo` compra las impares. Coinciden en las
    19 impares restantes y discrepan en 21: las 20 pares, que rinden, y la 5, donde `solo` gana
    lo que `full` no."""
    found = pair_dispersion("full", "solo", aligned("full", "solo"), 0.0, DELTA, UNIT, RESAMPLES)

    assert (found.agreed, found.disagreed) == (19, 21)
    assert found.nonzero == 21
    assert found.nonzero_agreed == 0


def test_a_hold_against_no_decision_disagrees_and_still_adds_zero() -> None:
    """«No decidió» y hold son categorías distintas del acuerdo, y los dos puntúan 0: hay
    discrepancias que no aportan dispersión."""
    undecided = [make_record(j, None) for j in range(N)]
    pair = paired_returns("a", "b", {"a": undecided, "b": arm("hold")}, HISTORIES, HORIZON)

    assert isinstance(pair, PairedReturns)
    found = pair_dispersion("a", "b", pair, 0.0, DELTA, UNIT, RESAMPLES)
    assert (found.agreed, found.disagreed, found.nonzero) == (0, N, 0)


def test_the_agreement_is_the_count_the_ablation_table_prints() -> None:
    """Es `metrics.agreement_counts`, la función que rinde la columna de coincidencia de la
    ablación, y no una cuenta propia."""
    pair = aligned("full", "solo")
    expected = metrics_module.agreement_counts(
        metrics_module.decision_actions(pair.arm_records),
        metrics_module.decision_actions(pair.reference_records),
    )

    assert expected is not None
    found = pair_dispersion("full", "solo", pair, 0.0, DELTA, UNIT, RESAMPLES)
    assert (found.agreed, found.n) == (expected.numerator, expected.denominator)


def test_a_nonzero_difference_under_the_same_action_is_counted_and_fails_the_report(
    tmp_path: Path,
) -> None:
    """Lo que la enmienda 3 da por hecho, comprobado y no supuesto: si dos brazos con la misma
    acción rindieran distinto, la fila lo cuenta y el reporte no se da por bueno."""
    same = aligned("full", "always_buy")
    broken = same.model_copy(
        update={"reference_returns": tuple(value + 0.001 for value in same.reference_returns)}
    )

    found = pair_dispersion("full", "always_buy", broken, 0.0, DELTA, UNIT, RESAMPLES)

    assert found.nonzero_agreed == found.agreed == N - 1
    report = report_of(tmp_path=tmp_path)
    tampered = report.model_copy(update={"pairs": (found, *report.pairs[1:])})
    assert not tampered.consistent


def test_a_pair_the_run_does_not_have_is_missing_and_not_zero(tmp_path: Path) -> None:
    """Sin `full` en la corrida —la etapa 2 del criterio 8 es `solo` contra las líneas base—
    sus cinco pares quedan con el porqué, los cuatro de `solo` se calculan, y lo calculado se
    da por bueno."""
    arms = paid_run()
    del arms["full"]

    report = report_of(arms, tmp_path)

    assert [(pair.arm, pair.reference) for pair in report.pairs] == [
        pair for pair in PAIRS if pair[0] == "solo"
    ]
    assert [(item.arm, item.reference) for item in report.missing] == [
        pair for pair in PAIRS if pair[0] == "full"
    ]
    assert all("full" in item.why for item in report.missing)
    assert report.consistent
    assert "no determinado: el brazo full no tiene evaluaciones" in render_report(report, SOURCES)


# ───────────────────────────────────────── Coste y pared ──────────────────────────────────────────


def assert_the_cost_scaled_is_the_upper_bound() -> None:
    calls = {"full": [decider_call(index, cached=None) for index in range(4)]}
    (cost,) = module.arm_costs(calls, {"full": 4}, DEFAULT_PRICING, Billing.PAYG)
    spent = consume(calls["full"], DEFAULT_PRICING, Billing.PAYG)

    assert spent.ceiling_usd is not None
    assert spent.cost_usd == 0.0
    assert spent.ceiling_usd > 0.0
    assert cost.upper_usd == spent.ceiling_usd
    assert cost.per_evaluation == pytest.approx(spent.ceiling_usd / 4)


def test_the_cost_per_evaluation_is_the_upper_bound_consumption_gives() -> None:
    """Con `cached_tokens` sin informar el coste medido es 0 y una cota inferior; lo que se
    escala a otra etapa es la cota superior de `consumption`."""
    assert_the_cost_scaled_is_the_upper_bound()


def test_mutation_scaling_the_measured_cost_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    mutant = mutated(arm_costs, "upper_usd=spent.ceiling_usd", "upper_usd=spent.cost_usd", module)
    monkeypatch.setattr(module, "arm_costs", mutant)
    with pytest.raises(AssertionError):
        assert_the_cost_scaled_is_the_upper_bound()


def test_a_call_without_tokens_leaves_the_cost_undetermined() -> None:
    """Sin tokens `consumption` no da cota, y un coste por evaluación sería inventado."""
    calls = {"full": [live_call(model="glm-5.2")]}

    (cost,) = arm_costs(calls, {"full": 1}, DEFAULT_PRICING, Billing.PAYG)

    assert cost.upper_usd is None
    assert cost.per_evaluation is None
    assert stage_usd((cost,), ("full",), 35) is None


def test_the_cost_of_a_stage_adds_the_arms_that_are_paid() -> None:
    """Un brazo sin llamadas vale 0 porque lo dice `consumption`; uno que no está, no se sabe."""
    costs = arm_costs(
        {"full": [decider_call(0), decider_call(1)], "solo": [decider_call(2)]},
        {"full": 2, "solo": 1, "always_buy": 3},
        DEFAULT_PRICING,
        Billing.PAYG,
    )
    full, solo, baseline = costs

    assert full.per_evaluation is not None
    assert solo.per_evaluation is not None
    assert full.per_evaluation == pytest.approx(solo.per_evaluation)
    assert baseline.per_evaluation == 0.0
    assert stage_usd(costs, ("full", "solo"), 70) == pytest.approx(140 * full.per_evaluation)
    assert stage_usd(costs, ("always_buy",), 70) == 0.0
    assert stage_usd(costs, ("full", "no_debate"), 70) is None


def clocked(*calls: LLMCall) -> list[EvaluationRecord]:
    return [make_record(0, Action.BUY).model_copy(update={"calls": calls})]


def assert_the_wall_clock_is_between_the_first_and_the_last_live_call() -> None:
    hit = live_call(cache_hit=True, at=START - timedelta(days=400))
    records = [
        *clocked(decider_call(30.0), hit),
        *clocked(decider_call(0.0)),
        *clocked(decider_call(10.0)),
    ]

    found = module.wall_clock("full", records)

    assert (found.first, found.last) == (START, START + timedelta(minutes=30))
    assert (found.evaluations, found.live_calls) == (3, 3)
    assert found.seconds == 1800.0
    assert found.per_evaluation == 600.0


def test_the_wall_clock_is_between_the_first_and_the_last_live_call() -> None:
    """Mínimo y máximo de `LLMCall.at`, lleguen en el orden que lleguen, sin los aciertos de
    caché: en un replay su `at` es el instante evaluado y no el de pared."""
    assert_the_wall_clock_is_between_the_first_and_the_last_live_call()


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (" if not call.cache_hit", ""),
        ("first=min(marks) if marks else None", "first=marks[0] if marks else None"),
        ("last=max(marks) if marks else None", "last=marks[-1] if marks else None"),
    ],
)
def test_mutation_a_wall_clock_read_another_way_is_caught(
    old: str, new: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "wall_clock", mutated(wall_clock, old, new, module))
    with pytest.raises(AssertionError):
        assert_the_wall_clock_is_between_the_first_and_the_last_live_call()


def test_an_arm_without_two_live_calls_has_no_wall_clock() -> None:
    silent = wall_clock("always_buy", [make_record(0, Action.BUY)])
    single = wall_clock("solo", clocked(decider_call(0.0)))

    assert (silent.live_calls, silent.seconds, silent.per_evaluation) == (0, None, None)
    assert (single.live_calls, single.seconds, single.per_evaluation) == (1, None, None)


def test_the_hours_of_a_stage_add_the_arms_one_after_another() -> None:
    """600 s y 60 s por evaluación: 36 evaluaciones de los dos son 6.6 h. Un brazo sin llamadas
    no tiene pared que medir y cuenta 0; uno con una sola deja la cuenta sin determinar."""
    full = WallClock(
        arm="full",
        evaluations=3,
        live_calls=3,
        first=START,
        last=START + timedelta(minutes=30),
    )
    solo = WallClock(
        arm="solo", evaluations=10, live_calls=10, first=START, last=START + timedelta(minutes=10)
    )
    silent = WallClock(arm="always_buy", evaluations=10, live_calls=0)
    single = WallClock(arm="bull_only", evaluations=1, live_calls=1, first=START, last=START)
    clocks = (full, solo, silent, single)

    assert stage_hours(clocks, ("full", "solo"), 36) == pytest.approx(6.6)
    assert stage_hours(clocks, ("solo", "always_buy"), 36) == pytest.approx(0.6)
    assert stage_hours(clocks, ("bull_only",), 36) is None
    assert stage_hours(clocks, ("no_debate",), 36) is None


# ─────────────────────────────────────────── El pool ──────────────────────────────────────────────

MANIFEST_INDICES = tuple(range(0, 80, 4))
"""Las velas del manifiesto sintético: 4 por tramo en cada símbolo."""

POOL_INDICES = {
    "S0": (*MANIFEST_INDICES, 1, 2, 17, 33, 49, 65),
    "S1": (*MANIFEST_INDICES, 1),
}
"""El barrido sintético: lo del manifiesto y algunas activaciones más. Con la primera en 0 y la
última en 76, el tramo de la vela `i` es `i·5 // 77`: 1 y 2 caen en el 0, 17 en el 1, 33 en el 2,
49 en el 3 y 65 en el 4."""

UNCONFIRMED = {("S0", 2)}
"""Activaciones del barrido que la ventana del replay no confirma."""


def fake_pool(monkeypatch: pytest.MonkeyPatch, indices: dict[str, tuple[int, ...]]) -> None:
    """Sustituye el barrido y la confirmación por los escritos a mano: el gate tiene sus pruebas."""

    def bars(_rows: object, symbol: str, timeframe: str) -> tuple[ActivatedBar, ...]:
        return tuple(
            ActivatedBar(
                symbol=symbol,
                timeframe=timeframe,
                index=index,
                at=START + timedelta(hours=index),
                triggers=("range_breakout",),
            )
            for index in sorted(indices[symbol])
        )

    def confirm(rows: list[list[float]], index: int, _limit: int) -> tuple[str, ...] | None:
        symbol = next(name for name, series in HISTORIES.items() if series == rows)
        return None if (symbol, index) in UNCONFIRMED else ("range_breakout",)

    monkeypatch.setattr(module, "activated_bars", bars)
    monkeypatch.setattr(module, "confirm_activation", confirm)


def assert_the_pool_is_counted_cell_by_cell(tmp_path: Path) -> None:
    manifest = load_manifest(write_plan(tmp_path))

    pool = module.pool_cells(manifest, HISTORIES)

    assert pool.activations == 26 + 21
    cells = {(cell.symbol, cell.stratum): cell for cell in pool.cells}
    assert len(cells) == 10
    first = cells[("S0", 0)]
    assert (first.candidates, first.confirmed, first.used, first.fresh) == (6, 5, 4, 1)
    assert [cells[("S0", stratum)].candidates for stratum in range(5)] == [6, 5, 5, 5, 5]
    assert [cells[("S1", stratum)].candidates for stratum in range(5)] == [5, 4, 4, 4, 4]
    assert {cell.used for cell in pool.cells} == {4}
    assert (pool.capacity(fresh=False), pool.capacity(fresh=True)) == (4, 0)
    assert pool.smallest(fresh=False).symbol == "S1"


def test_the_pool_is_counted_cell_by_cell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Candidatas las que `stratify` reparte, confirmadas las que el gate abre con la ventana del
    replay, usadas las del manifiesto. La celda más pequeña decide cuánto cabe."""
    fake_pool(monkeypatch, POOL_INDICES)
    assert_the_pool_is_counted_cell_by_cell(tmp_path)


def test_mutation_counting_every_candidate_as_confirmed_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Una candidata que la ventana del replay no confirma no se puede seleccionar: contarla
    diría que cabe un n que `select_activations` no entrega."""
    fake_pool(monkeypatch, POOL_INDICES)
    mutant = mutated(
        pool_cells,
        "if confirm_activation(rows, bar.index, manifest.candle_limit) is not None",
        "if True",
        module,
    )
    monkeypatch.setattr(module, "pool_cells", mutant)
    with pytest.raises(AssertionError):
        assert_the_pool_is_counted_cell_by_cell(tmp_path)


def test_a_manifest_entry_outside_its_cell_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Si el barrido ya no encuentra una vela del manifiesto en la celda que su tramo dice, el
    pool que se contaría no es el de la selección."""
    fake_pool(monkeypatch, {**POOL_INDICES, "S1": (*MANIFEST_INDICES[1:], 1)})
    manifest = load_manifest(write_plan(tmp_path))

    with pytest.raises(RecalculationError, match="S1 tramo 0"):
        pool_cells(manifest, HISTORIES)


def test_whether_an_n_fits_shows_both_numbers() -> None:
    assert module.fit(15, 103) == "cabe (15 ≤ 103)"
    assert module.fit(103, 103) == "cabe (103 ≤ 103)"
    assert module.fit(104, 103) == "no cabe (104 > 103)"


def test_only_the_model_arms_of_a_pair_are_paid() -> None:
    assert module.paid_arms("full", "solo") == ("full", "solo")
    assert module.paid_arms("full", "always_buy") == ("full",)
    assert module.paid_arms("solo", "rule_trend") == ("solo",)


# ─────────────────────────────────────────── El reporte ───────────────────────────────────────────

SOURCES = Sources(
    directory="var/ablation/corrida",
    command="python -m crypto_agents.paired_dispersion var/ablation/corrida --delta 0.002",
    commit="`abc1234` (árbol limpio)",
    plan="`manifest.json` (manifest), sha-256 `" + "c" * 64 + "`",
    journals=(("full.jsonl", "d" * 64), ("solo.jsonl", "e" * 64)),
    histories=(("S0", "history/s0_1h.csv", "f" * 64),),
)


def test_no_model_in_the_module_has_a_field_for_a_mean() -> None:
    """Lo que no está en el modelo no se puede imprimir por descuido."""
    models = [
        item
        for _, item in inspect.getmembers(module, inspect.isclass)
        if issubclass(item, FrozenModel) and item.__module__ == module.__name__
    ]
    assert PairDispersion in models
    assert ArmCost in models

    banned = ("mean", "avg", "average", "expected", "total")
    offenders = [
        f"{model.__name__}.{name}"
        for model in models
        for name in model.model_fields
        if any(word in name for word in banned)
    ]
    assert offenders == []


def test_the_report_prints_no_mean_difference(tmp_path: Path) -> None:
    """`full` menos `solo` tiene una media de 0.580% en el escenario —veinte pares a 1.2% y la
    evaluación 5 a -0.8%, entre 40—: no aparece. La desviación de ese mismo vector sí."""
    pair = aligned("full", "solo")
    differences = metrics_module.paired_differences(pair.arm_returns, pair.reference_returns)
    mean = statistics.fmean(differences)

    text = render_report(report_of(tmp_path=tmp_path), SOURCES)

    assert f"{mean:.3%}" == "0.580%"
    assert "0.580%" not in text
    assert f"{statistics.stdev(differences):.3%}" in text
    assert "media de retorno" in text


def test_the_report_recommends_nothing(tmp_path: Path) -> None:
    """Da n, coste y viabilidad. Qué etapa correr no le toca."""
    text = render_report(report_of(tmp_path=tmp_path), SOURCES).lower()

    for word in ("conviene", "recom", "debería", "aconsej", "sugier", "vale la pena", "hay que"):
        assert word not in text
    for verdict in ("justifica su coste", "derrota", "es mejor", "es peor"):
        assert verdict not in text


def test_the_report_cites_every_digest(tmp_path: Path) -> None:
    text = render_report(report_of(tmp_path=tmp_path), SOURCES)

    for digest in ("c" * 64, "d" * 64, "e" * 64, "f" * 64):
        assert f"`{digest}`" in text
    assert "`full.jsonl`" in text
    assert "`history/s0_1h.csv`" in text
    assert "`abc1234`" in text


def test_every_table_says_which_command_reproduces_it(tmp_path: Path) -> None:
    text = render_report(report_of(tmp_path=tmp_path), SOURCES)

    sections = text.split("\n## ")[1:]
    numbered = [section for section in sections if section[0].isdigit()]
    assert len(numbered) == 7
    for section in numbered:
        assert "| --- |" in section
        assert f"Reproduce: `{SOURCES.command}`" in section
    assert "python -m crypto_agents.criteria var/ablation/corrida --delta 0.002" in text
    assert "python -m crypto_agents.consumption var/ablation/corrida" in text


def test_the_report_states_delta_both_ways_and_how_it_resampled(tmp_path: Path) -> None:
    """δ como fracción y como porcentaje, para que un `--delta 0.20` se vea en la cabecera."""
    text = render_report(report_of(tmp_path=tmp_path), SOURCES)

    assert "δ = 0.002 (0.200% de retorno por evaluación)" in text
    assert f"{RESAMPLES} remuestreos" in text
    assert "dentro de cada símbolo, semilla 1," in text
    assert "al múltiplo de 10: 2 símbolos por 5 tramos" in text
    assert "retorno bruto" in text


def test_the_report_has_a_row_for_every_pair_size_and_target(tmp_path: Path) -> None:
    """Nueve pares: 9 filas de dispersión, 27 de n (tres valores de sigma), 27 de semiancho
    (140, 280 y 420) y 54 de viabilidad (dos objetivos por tres valores)."""
    text = render_report(report_of(tmp_path=tmp_path), SOURCES)
    sections = {section.split("\n", 1)[0][:2]: section for section in text.split("\n## ")[1:]}

    def rows(key: str) -> list[str]:
        return [line for line in sections[key].splitlines() if line.startswith("| full | solo ")]

    assert len(rows("1.")) == 1
    assert len(rows("2.")) == 3
    assert [line.split("|")[3].strip() for line in rows("3.")] == ["140", "280", "420"]
    assert len(rows("4.")) == 6
    assert sections["4."].count("| semiancho ≤ δ |") == 27
    assert sections["4."].count("| potencia 80% |") == 27


def stage_row(text: str, target: str) -> list[str]:
    row = next(
        line
        for line in text.splitlines()
        if line.startswith(f"| full | solo | {target} | puntual |")
    )
    return [cell.strip() for cell in row.split("|")[1:-1]]


def test_the_stage_rows_carry_the_room_the_extra_n_and_the_cost(tmp_path: Path) -> None:
    """Las dos filas puntuales de `full` contra `solo`, contra sus propias piezas: n entre las
    celdas, las dos capacidades del pool (5 con lo ya corrido, 1 sin ello), n menos lo ya
    corrido, y coste y pared de los dos brazos con modelo.

    El semiancho pide 39.4, que redondea a los 40 que ya hay: caben reutilizándolos y no hay
    nada que añadir. La potencia pide 80.4, que redondea a 90: 9 por celda, que no caben.
    """
    report = report_of(tmp_path=tmp_path)
    found = report.pairs[0].dispersion
    text = render_report(report, SOURCES)

    tie = stage_row(text, "semiancho ≤ δ")
    assert found.half_width_n[1].rounded == N == 40
    assert tie[4:9] == ["40", "4", "cabe (4 ≤ 5)", "no cabe (4 > 1)", "0"]
    assert tie[10] == "0.00"

    power = stage_row(text, "potencia 80%")
    assert found.power_n[1].rounded == 90
    assert power[4:9] == ["90", "9", "no cabe (9 > 5)", "no cabe (9 > 1)", "50"]
    usd = stage_usd(report.costs, ("full", "solo"), 90)
    extra = stage_usd(report.costs, ("full", "solo"), 50)
    hours = stage_hours(report.clocks, ("full", "solo"), 90)
    assert usd is not None
    assert extra is not None
    assert hours is not None
    assert usd > extra > 0.0
    assert power[9:] == [f"{usd:.2f}", f"{extra:.2f}", f"{hours:.1f}"]


def test_the_cost_section_dates_its_prices_and_labels_the_estimate(tmp_path: Path) -> None:
    text = render_report(report_of(tmp_path=tmp_path), SOURCES)

    assert f"precios de la página al {DEFAULT_PRICING.as_of(Billing.PAYG)}" in text
    assert "una estimación, no una cota de otra etapa" in text
    assert "Observada en esta corrida" in text


def test_a_run_without_its_billing_has_no_cost_and_says_why(tmp_path: Path) -> None:
    report = report_of(tmp_path=tmp_path, billing=None)

    assert report.costs == ()
    text = render_report(report, SOURCES)
    assert "No determinado: meta.json no registra la facturación de la corrida." in text
    assert "| no determinado | no determinado |" in text


def test_a_resumed_run_has_no_single_wall_clock(tmp_path: Path) -> None:
    report = report_of(tmp_path=tmp_path, resumed=True)

    assert report.clocks == ()
    assert "No determinado: corrida reanudada" in render_report(report, SOURCES)


# ─────────────────────────────────── El comando, de un directorio ─────────────────────────────────


def snapshot(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def run_command(directory: Path, tmp_path: Path, *extra: str) -> int:
    return main(
        [
            str(directory),
            "--delta",
            str(DELTA),
            "--history-dir",
            str(tmp_path / "history"),
            "--resamples",
            str(RESAMPLES),
            *extra,
        ]
    )


def test_the_command_reads_a_run_cites_its_files_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_pool(monkeypatch, POOL_INDICES)
    directory = write_run(tmp_path, paid_run(), billing=Billing.PAYG)
    monkeypatch.chdir(tmp_path)
    before = snapshot(tmp_path)

    assert run_command(directory, tmp_path) == 0

    assert snapshot(tmp_path) == before
    out = capsys.readouterr().out
    for name in ("full.jsonl", "solo.jsonl", "always_buy.jsonl", "manifest.json"):
        path = directory / name if name.endswith(".jsonl") else tmp_path / name
        assert f"sha-256 `{file_sha256(path)}`" in out
    for symbol in ("s0", "s1"):
        assert f"sha-256 `{file_sha256(tmp_path / 'history' / f'{symbol}_1h.csv')}`" in out
    assert f"{directory} --delta 0.002 --history-dir {tmp_path / 'history'} --resamples 200" in out
    assert "| full | solo | 40 | 19/40 (48%) | 21 | 21 | 0 |" in out
    assert "S1 tramo 1, con 4 activaciones confirmadas" in out
    assert "**NO**" not in out


def test_the_command_is_reproducible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_pool(monkeypatch, POOL_INDICES)
    directory = write_run(tmp_path, paid_run(), billing=Billing.PAYG)

    assert run_command(directory, tmp_path) == 0
    first = capsys.readouterr().out
    assert run_command(directory, tmp_path) == 0
    assert capsys.readouterr().out == first
    assert run_command(directory, tmp_path, "--seed", "5") == 0
    assert capsys.readouterr().out != first


def test_a_standard_error_that_does_not_reproduce_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_pool(monkeypatch, POOL_INDICES)
    monkeypatch.setattr(
        module, "sigma_of", mutated(sigma_of, "statistics.stdev(", "statistics.pstdev(", module)
    )
    directory = write_run(tmp_path, paid_run(), billing=Billing.PAYG)

    assert run_command(directory, tmp_path) == 1
    assert "**NO**" in capsys.readouterr().out


def test_an_invalid_run_exits_one_and_prints_no_figure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Una corrida que el criterio 6 invalida no se interpreta, tampoco para dimensionar otra."""
    fake_pool(monkeypatch, POOL_INDICES)
    arms = paid_run()
    arms["full"] = [*arms["full"], quota_lost()]
    directory = write_run(tmp_path, arms, billing=Billing.PAYG)

    assert run_command(directory, tmp_path) == 1

    out = capsys.readouterr().out
    assert out.startswith("CORRIDA INVÁLIDA")
    assert "sigma_D" not in out


def test_a_probe_and_a_missing_directory_exit_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_pool(monkeypatch, POOL_INDICES)
    probe = write_run(tmp_path, paid_run(), billing=Billing.PAYG, kind=RunKind.PROBE)

    assert run_command(probe, tmp_path) == 2
    assert "es un sondeo" in capsys.readouterr().err
    assert run_command(tmp_path / "no-existe", tmp_path) == 2
    assert "no es una corrida" in capsys.readouterr().err


def test_delta_and_resamples_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main([str(tmp_path), "--delta", "0"])
    with pytest.raises(SystemExit):
        main([str(tmp_path), "--delta", "0.002", "--resamples", "0"])
