"""La estimación de la etapa 1 en dólares: cuentas a mano, y las reglas que la mantienen honesta.

Las cifras de `estimate()` salen de multiplicar un conteo (el `--dry-run`) por un coste medido por
llamada. Aquí se arma el conteo y los costes a mano para que cada total se pueda comprobar con una
calculadora, y aparte se comprueba que el coste medido sale de las filas de un sondeo de verdad.
"""

from __future__ import annotations
import __future__ as future_flags

import argparse
import inspect
import textwrap
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from crypto_agents import estimate as estimate_module
from crypto_agents import zen_probe as zen_probe_module
from crypto_agents.ablation import (
    ARMS,
    DECIDER_ATTEMPTS,
    LAUNCH_MARGIN,
    DryRunReport,
    DryRunRow,
)
from crypto_agents.audit import file_sha256, read_run
from crypto_agents.candidates import BULL_CANDIDATES, Candidate
from crypto_agents.consumption import consume
from crypto_agents.estimate import (
    CACHED_NOTE,
    FALLBACK_NOTE,
    FULL_PROMPT_NOTE,
    LOCAL_NOTE,
    NO_CALLS_NOTE,
    SHORT_DECIDER_ARMS,
    USAGE_FIXTURE,
    ArmRoleLine,
    AttemptRate,
    CallCost,
    Estimate,
    ScenarioLine,
    arm_cost,
    attempt_rate,
    balance_check,
    estimate,
    main,
    measured_costs,
    render_breakdown,
    render_estimate,
    replace_roles,
    scenarios,
    usage_fixture_costs,
)
from crypto_agents.estimate import _source as source_option
from crypto_agents.settings import DEFAULT_PRICING
from crypto_agents.state import AgentRole, Backend, Billing
from tests.conftest import DATA_DIR
from tests.test_zen_payg import payg_settings
from tests.test_zen_probe import (  # noqa: F401
    COUNT,
    ZenFake,
    desks,
    manifest_on_disk,
    pro_settings,
    probe,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from crypto_agents.state import LLMCall

N = 10
NOW = datetime(2026, 10, 4, tzinfo=UTC)


def cost(
    role: AgentRole,
    model: str,
    mean: float | None,
    valid: int = N,
    attempts: int = N,
    invocations: int = 0,
) -> CallCost:
    """Un coste por intento. Sin `invocations` no hay intentos medidos: vale la constante."""
    return CallCost(
        role=role,
        model=model,
        attempts=attempts,
        measured=attempts if mean is not None else 0,
        unmeasured=0 if mean is not None else attempts,
        mean_usd=mean,
        valid_verdicts=valid,
        source="sondeo/prueba.jsonl",
        invocations=invocations,
    )


def row(
    arm: str, node: str, role: AgentRole, *, exact: bool, backend: Backend = Backend.OPENAI
) -> DryRunRow:
    return DryRunRow(
        arm=arm,
        node=node,
        role=role,
        backend=backend,
        model="m",
        calls=N,
        exact=exact,
        cached=0 if exact else None,
        to_pay=N if exact else None,
    )


def report() -> DryRunReport:
    """`full` con los seis roles remotos, `local_technicals` con los técnicos en local, `solo`
    con su decisor y una línea base sin ninguna fila."""
    local = Backend.OLLAMA
    return DryRunReport(
        evaluations=N,
        activations=N,
        prepare_failures=0,
        billing=Billing.PAYG,
        rows=(
            row("full", "structure", AgentRole.STRUCTURE, exact=True),
            row("full", "momentum", AgentRole.MOMENTUM, exact=True),
            row("full", "volume", AgentRole.VOLUME, exact=True),
            row("full", "bull", AgentRole.BULL, exact=False),
            row("full", "bear", AgentRole.BEAR, exact=False),
            row("full", "decide", AgentRole.DECIDER, exact=False),
            row("local_technicals", "structure", AgentRole.STRUCTURE, exact=True, backend=local),
            row("local_technicals", "momentum", AgentRole.MOMENTUM, exact=True, backend=local),
            row("local_technicals", "volume", AgentRole.VOLUME, exact=True, backend=local),
            row("local_technicals", "bull", AgentRole.BULL, exact=False),
            row("local_technicals", "bear", AgentRole.BEAR, exact=False),
            row("local_technicals", "decide", AgentRole.DECIDER, exact=False),
            row("solo", "decide_solo", AgentRole.DECIDER, exact=True),
        ),
    )


FIXED = {
    AgentRole.MOMENTUM: "mom",
    AgentRole.BULL: "bull",
    AgentRole.BEAR: "bear",
    AgentRole.DECIDER: "dec",
}


def costs() -> dict[tuple[AgentRole, str], CallCost]:
    return {
        (AgentRole.MOMENTUM, "mom"): cost(AgentRole.MOMENTUM, "mom", 0.001),
        (AgentRole.BULL, "bull"): cost(AgentRole.BULL, "bull", 0.002),
        (AgentRole.BEAR, "bear"): cost(AgentRole.BEAR, "bear", 0.003),
        (AgentRole.DECIDER, "dec"): cost(AgentRole.DECIDER, "dec", 0.01),
        (AgentRole.STRUCTURE, "A"): cost(AgentRole.STRUCTURE, "A", 0.004),
        (AgentRole.STRUCTURE, "B"): cost(AgentRole.STRUCTURE, "B", 0.008),
        # El más barato de todos, pero no produjo ni un veredicto válido: no respondió.
        (AgentRole.STRUCTURE, "C"): cost(AgentRole.STRUCTURE, "C", 0.0001, valid=0),
        (AgentRole.VOLUME, "A"): cost(AgentRole.VOLUME, "A", 0.005),
        (AgentRole.VOLUME, "B"): cost(AgentRole.VOLUME, "B", 0.001),
    }


# ──────────────────────────────────────────── Cuentas a mano ──────────────────────────────────────


def test_the_totals_are_the_hand_computed_ones() -> None:
    """full: structure 10 x (.004-.008), momentum .01, volume 10 x (.001-.005), bull .02, bear .03,
    decisor .10. Suma .21 a .29; local_technicals: solo mesas y decisor, .15; solo: .10.
    """
    result = estimate(report(), costs(), FIXED)

    assert result.arms["full"] == pytest.approx((0.21, 0.29))
    assert result.arms["local_technicals"] == pytest.approx((0.15, 0.15))
    assert result.arms["solo"] == pytest.approx((0.10, 0.10))
    assert result.total == pytest.approx((0.46, 0.54))


def test_without_measured_invocations_the_constants_are_the_fallback_and_it_is_said() -> None:
    """Sin invocaciones que contar: el decisor a 1.2 y los demás a 1.0, rotulado como respaldo.

    full .10 -> .12, local_technicals .10 -> .12, solo .10 -> .12.
    """
    result = estimate(report(), costs(), FIXED)

    assert DECIDER_ATTEMPTS == 1.2
    assert result.arms_with_retries["full"] == pytest.approx((0.23, 0.31))
    assert result.arms_with_retries["local_technicals"] == pytest.approx((0.17, 0.17))
    assert result.arms_with_retries["solo"] == pytest.approx((0.12, 0.12))
    assert result.total_with_retries == pytest.approx((0.52, 0.60))
    by_role = {line.role: line for line in result.roles}
    assert by_role[AgentRole.DECIDER].attempts_low == pytest.approx(1.2)
    assert by_role[AgentRole.BEAR].attempts_low == pytest.approx(1.0)
    assert all(FALLBACK_NOTE in line.attempts_source for line in result.roles)


# ─────────────────────────────────── Intentos medidos, no supuestos ───────────────────────────────


def measured() -> dict[tuple[AgentRole, str], CallCost]:
    """Los costes de siempre, con intentos medidos en el bear y en el decisor.

    bear: 22 intentos para 12 alegatos (lo que midió `20261005T233053Z`); decisor: 19 para 12.
    """
    found = costs()
    found[(AgentRole.BEAR, "bear")] = cost(
        AgentRole.BEAR, "bear", 0.003, attempts=22, invocations=12
    )
    found[(AgentRole.DECIDER, "dec")] = cost(
        AgentRole.DECIDER, "dec", 0.01, attempts=19, invocations=12
    )
    return found


def assert_the_measured_attempts_price_the_roles() -> None:
    """A mano: bear 20 llamadas x .003 x 22/12 = .11; decisor 30 x .01 x 19/12 = .475.

    El resto a un intento: structure .04 a .08, momentum .01, volume .01 a .05, bull .04.
    Total .685 a .765. Con las constantes viejas (bear 1.0, decisor 1.2) sería .52 a .60.
    """
    result = estimate(report(), measured(), FIXED)
    by_role = {line.role: line for line in result.roles}

    assert by_role[AgentRole.BEAR].attempts_low == pytest.approx(22 / 12)
    bear = by_role[AgentRole.BEAR]
    assert (bear.retried_low, bear.retried_high) == pytest.approx((0.11, 0.11))
    assert by_role[AgentRole.DECIDER].retried_low == pytest.approx(0.475)
    assert result.total_with_retries == pytest.approx((0.685, 0.765))
    assert result.total == pytest.approx((0.46, 0.54)), "a un intento por llamada no cambia"


def test_each_role_is_priced_with_the_attempts_its_probe_measured() -> None:
    assert_the_measured_attempts_price_the_roles()


def test_mutation_the_old_constants_instead_of_the_measured_attempts_are_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Volver a 1.2 y 1.0 deja el total en .52 a .60: lo que T4 midió que era falso."""

    def constants(role: AgentRole, _cost: CallCost | None) -> AttemptRate:
        return AttemptRate(DECIDER_ATTEMPTS if role is AgentRole.DECIDER else 1.0, "constante")

    monkeypatch.setattr(estimate_module, "attempt_rate", constants)
    with pytest.raises(AssertionError):
        assert_the_measured_attempts_price_the_roles()


def test_the_attempts_count_the_ones_the_provider_never_answered() -> None:
    """Un plazo vencido es un intento: 17 para 12 son 1.42 aunque cuatro no trajeran tokens."""
    timed_out = CallCost(
        role=AgentRole.DECIDER,
        model="dec",
        attempts=17,
        measured=13,
        unmeasured=4,
        mean_usd=0.01,
        valid_verdicts=7,
        source="sondeo/decider.jsonl",
        invocations=12,
    )
    rate = attempt_rate(AgentRole.DECIDER, timed_out)

    assert rate.value == pytest.approx(17 / 12)
    assert rate.source == "17 intentos / 12 invocaciones · `sondeo/decider.jsonl`"


def test_a_ranged_role_applies_to_each_candidate_its_own_attempts() -> None:
    """A: .004 por intento a 2.0 intentos = .008; B: .008 a 1.0 = .008. El rango es .008 a .008.

    Un factor común sobre el rango (.004 a .008) x 2.0 daría .008 a .016: los intentos de A con
    el precio de B.
    """
    found = costs()
    found[(AgentRole.STRUCTURE, "A")] = cost(
        AgentRole.STRUCTURE, "A", 0.004, attempts=24, invocations=12
    )
    found[(AgentRole.STRUCTURE, "B")] = cost(
        AgentRole.STRUCTURE, "B", 0.008, attempts=12, invocations=12
    )
    structure = next(
        line for line in estimate(report(), found, FIXED).roles if line.role is AgentRole.STRUCTURE
    )

    assert (structure.attempts_low, structure.attempts_high) == pytest.approx((1.0, 2.0))
    assert (structure.retried_low, structure.retried_high) == pytest.approx((0.08, 0.08))
    assert "`A` 2.00" in structure.attempts_source
    assert "`B` 1.00" in structure.attempts_source


def test_the_report_shows_the_attempts_their_n_and_where_they_come_from() -> None:
    text = render_estimate(estimate(report(), measured(), FIXED))
    table = text.split("## Intentos por veredicto")[1].split("## ")[0]

    bear = (
        "| único | bear | `bear` | 1.83 | 22 intentos / 12 invocaciones · `sondeo/prueba.jsonl` |"
    )
    assert bear in table
    assert "| único | decider | `dec` | 1.58 | 19 intentos / 12 invocaciones" in table
    assert f"| único | bull | `bull` | 1.00 | {FALLBACK_NOTE} (1) |" in table


def test_local_calls_and_baselines_cost_nothing_by_rule() -> None:
    """local_technicals paga mesas y decisor (.15): sus tres técnicos locales no suman nada.

    Una línea base no tiene filas en el conteo, y aun así sale en la tabla, con cero.
    """
    arms = ("full", "local_technicals", "solo", "always_buy")
    result = estimate(report(), costs(), FIXED, arms=arms)
    structure = next(line for line in result.roles if line.role is AgentRole.STRUCTURE)

    assert result.arms["local_technicals"] == pytest.approx((0.15, 0.15))
    assert structure.calls == N, (
        "solo las 10 de full: las 10 locales de local_technicals no cuentan"
    )
    assert result.arms["always_buy"] == (0.0, 0.0)
    assert result.arms_with_retries["always_buy"] == (0.0, 0.0)
    assert estimate(report(), costs(), FIXED).total == result.total


def test_the_range_is_between_the_cheapest_and_the_dearest_that_answered() -> None:
    """C sale más barato que A y B, pero no respondió: no entra en el rango."""
    result = estimate(report(), costs(), FIXED)
    structure = next(line for line in result.roles if line.role is AgentRole.STRUCTURE)
    volume = next(line for line in result.roles if line.role is AgentRole.VOLUME)

    assert (structure.per_call_low, structure.per_call_high) == pytest.approx((0.004, 0.008))
    assert (volume.per_call_low, volume.per_call_high) == pytest.approx((0.001, 0.005))
    assert structure.model is None, "no se elige modelo: se da el rango"
    assert "C" not in {c.model for c in result.candidates}


def test_the_decider_line_counts_every_paid_call_across_the_arms() -> None:
    result = estimate(report(), costs(), FIXED)
    decider = next(line for line in result.roles if line.role is AgentRole.DECIDER)

    assert decider.calls == 30  # full, local_technicals y solo: 10 cada uno
    assert decider.cost_low == pytest.approx(0.30)
    assert decider.model == "dec"


def test_a_role_nobody_measured_makes_the_total_undetermined_not_smaller() -> None:
    """Un total que se salta un rol en silencio es una cifra menor de lo que costará."""
    partial = costs()
    del partial[(AgentRole.BEAR, "bear")]

    result = estimate(report(), partial, FIXED)

    assert result.total == (None, None)
    assert result.arms["solo"] == pytest.approx((0.10, 0.10)), "el brazo sin ese rol sí se calcula"
    text = render_estimate(result)
    assert "no determinado" in text
    assert "el sondeo no midió este rol" in text


def test_with_no_candidate_that_answered_the_ranged_roles_are_undetermined() -> None:
    nobody = {key: value for key, value in costs().items() if key[1] not in {"A", "B", "C"}}
    result = estimate(report(), nobody, FIXED)
    structure = next(line for line in result.roles if line.role is AgentRole.STRUCTURE)

    assert structure.cost_low is None
    assert "ningún candidato respondió" in structure.source
    assert result.total == (None, None)


# ───────────────────────────────────────────── La recarga ─────────────────────────────────────────


def test_the_single_topup_is_the_credit_times_the_rate_plus_the_fixed_fee() -> None:
    text = render_estimate(estimate(report(), costs(), FIXED))

    # .46 x 1.044 + .30 = .78024 ; .60 x 1.044 + .30 = .9264
    assert "| un intento por llamada, más barato | 0.46 | 0.78 |" in text
    assert "| con los intentos medidos, más caro | 0.60 | 0.93 |" in text
    assert "ESTIMACIÓN" in text
    assert "no un recibo" in text


def test_the_report_cites_the_input_files_with_their_digests() -> None:
    inputs = [("sondeo/a.jsonl", "a" * 64), ("data/ablation_selection.json", "b" * 64)]
    text = render_estimate(estimate(report(), costs(), FIXED, inputs))

    for name, digest in inputs:
        assert f"`{name}` sha-256 `{digest}`" in text


# ──────────────────────────────────────── Saldo frente a estimación ───────────────────────────────


def test_the_launch_margin_is_one_and_a_half() -> None:
    assert LAUNCH_MARGIN == 1.5


def test_the_balance_must_cover_the_dearest_end_times_the_margin() -> None:
    """Coste de 10 a 12 con el decisor a 1.2: requerido 15 a 18. Con 18.00 pasa; con 17.99, no."""
    passes = balance_check((10.0, 12.0), 18.0)
    assert passes.required == (15.0, 18.0)
    assert passes.passes

    fails = balance_check((10.0, 12.0), 17.99)
    assert fails.required == (15.0, 18.0)
    assert not fails.passes


def test_a_balance_between_the_two_ends_does_not_pass() -> None:
    """16 cubre el extremo bajo (15) y no el alto (18): se juzga en el alto."""
    assert not balance_check((10.0, 12.0), 16.0).passes


def test_an_undetermined_cost_never_passes() -> None:
    for cost_pair in ((None, None), (10.0, None), (None, 12.0)):
        check = balance_check(cost_pair, 1_000_000.0)
        assert not check.passes
        assert check.required == (None, None)
        assert "no determinado" in check.reason


def test_the_balance_section_prints_the_cost_the_requirement_and_the_verdict() -> None:
    result = estimate(report(), costs(), FIXED)  # con el decisor a 1.2: 0.52 a 0.60
    low, high = result.total_with_retries
    text = render_estimate(result, balance_check((low, high), 0.90))

    assert "## Saldo frente a estimación" in text
    assert "no se consultó la red" in text
    assert "0.52 a 0.60 USD" in text
    assert "saldo requerido = coste x 1.5: 0.78 a 0.90 USD" in text
    assert "**PASA**" in text
    assert "**NO PASA" in render_estimate(result, balance_check((low, high), 0.89))


# ───────────────────────────────────── Una línea por candidato a bull ─────────────────────────────

DESKS = {
    (AgentRole.BULL, "b1"): cost(AgentRole.BULL, "b1", 0.002),
    (AgentRole.BULL, "b2"): cost(AgentRole.BULL, "b2", 0.006),
    (AgentRole.BULL, "b3"): cost(AgentRole.BULL, "b3", 0.0001, valid=0),
    (AgentRole.BEAR, "bear"): cost(AgentRole.BEAR, "bear", 0.003),
}
GIVEN = {
    "b1": cost(AgentRole.DECIDER, "dec", 0.010),
    "b2": cost(AgentRole.DECIDER, "dec", 0.020),
}


def lines(balance: float | None = None) -> tuple[ScenarioLine, ...]:
    technical = {k: v for k, v in costs().items() if k[0] not in (AgentRole.BULL, AgentRole.BEAR)}
    return scenarios(
        report(), technical, DESKS, GIVEN, FIXED, ["b1", "b2", "b3", "b4"], balance=balance
    )


def test_each_bull_has_its_own_totals_computed_by_hand() -> None:
    """Llamadas: structure 10, momentum 10, volume 10, bull 20, bear 20, decisor 30.

    b1: bull 20 x .002 = .04, decisor 30 x .010 x 1.2 = .36 -> total .52 a .60 (como el base).
    b2: bull 20 x .006 = .12, decisor 30 x .020 x 1.2 = .72 -> total .96 a 1.04.
    """
    one, two, *_ = lines()

    assert one.per_role[AgentRole.BULL] == pytest.approx((0.04, 0.04))
    assert one.per_role[AgentRole.DECIDER] == pytest.approx((0.36, 0.36))
    assert one.total == pytest.approx((0.52, 0.60))
    assert two.per_role[AgentRole.BULL] == pytest.approx((0.12, 0.12))
    assert two.per_role[AgentRole.DECIDER] == pytest.approx((0.72, 0.72))
    assert two.total == pytest.approx((0.96, 1.04))
    assert one.topup == pytest.approx((0.52 * 1.044 + 0.30, 0.60 * 1.044 + 0.30))


def test_only_what_depends_on_the_bull_changes_between_lines() -> None:
    one, two, *_ = lines()
    for role in (
        AgentRole.STRUCTURE,
        AgentRole.MOMENTUM,
        AgentRole.VOLUME,
        AgentRole.BEAR,
    ):
        assert one.per_role[role] == two.per_role[role], role


def test_a_bull_with_no_valid_brief_has_a_line_saying_so_and_no_borrowed_cost() -> None:
    _, _, nothing, absent = lines()

    for item in (nothing, absent):
        assert item.valid_briefs == 0
        assert item.total == (None, None)
        assert item.per_role == {}
        assert item.note is not None
        assert "ningún alegato válido" in item.note


def test_a_bull_whose_decider_was_not_measured_is_undetermined_and_not_cheaper() -> None:
    technical = {k: v for k, v in costs().items() if k[0] not in (AgentRole.BULL, AgentRole.BEAR)}
    (item,) = scenarios(report(), technical, DESKS, {}, FIXED, ["b1"])

    assert item.total == (None, None)
    assert item.note is not None
    assert "sin decisor medido" in item.note


def test_the_balance_is_judged_for_each_bull() -> None:
    """Con 1.00: b1 requiere .78 a .90 y pasa; b2 requiere 1.44 a 1.56 y no."""
    one, two, nothing, _ = lines(balance=1.0)

    assert one.balance is not None
    assert one.balance.passes
    assert two.balance is None or not two.balance.passes
    assert nothing.balance is None


def test_the_scenario_table_has_one_line_per_candidate_with_the_verdict() -> None:
    result = estimate(report(), costs(), FIXED)
    text = render_estimate(result, None, lines(balance=1.0))

    table = text.split("## Etapa 1 por candidato a bull")[1].split("## ")[0]
    rows = [line for line in table.splitlines() if line.startswith("| `b")]
    assert [row.split("|")[1].strip() for row in rows] == ["`b1`", "`b2`", "`b3`", "`b4`"]
    assert "PASA" in rows[0]
    assert "NO PASA" in rows[1]
    assert "ningún alegato válido" in rows[2]
    header = next(line for line in table.splitlines() if line.startswith("| bull "))
    assert {row.count("|") for row in rows} == {header.count("|")}, (
        "todas las filas, del mismo ancho"
    )
    assert "Por rol (sumando brazos)" not in text, "con escenarios no hay cuadro de un solo bull"


# ──────────────────────────────────── El coste medido sale de las filas ───────────────────────────


def test_the_measured_cost_per_call_comes_from_the_rows_of_the_probe(tmp_path: Path) -> None:
    """1000 de entrada y 200 de salida a 0.30 / 1.20 por millón: 0.00054 por intento."""
    directory, _, _ = probe(tmp_path)
    found = measured_costs(read_run(directory))

    pro = found[(AgentRole.STRUCTURE, "deepseek-v4.1-flash")]
    assert pro.mean_usd == pytest.approx((1000 * 0.30 + 200 * 1.20) / 1_000_000)
    assert pro.attempts == COUNT
    assert pro.valid_verdicts == COUNT
    assert pro.measured == COUNT
    assert pro.source.endswith("deepseek-v4.1-flash@structure.jsonl")


def test_the_mode_pings_do_not_pollute_the_per_call_cost(tmp_path: Path) -> None:
    """El Ping dura sesenta caracteres: contarlo con el decisor bajaría su media."""
    directory, _, _ = probe(tmp_path)
    found = measured_costs(read_run(directory))

    assert found[(AgentRole.DECIDER, "glm-5.2")].attempts == COUNT  # los 3 del encadenado
    assert found[(AgentRole.BULL, "kimi-k2.6")].attempts == COUNT


def test_a_retried_verdict_costs_both_attempts(tmp_path: Path) -> None:
    """El intento inválido también se facturó: entra en la media y en los intentos."""
    directory, _, _ = probe(tmp_path, ZenFake(flaky={"deepseek-v4-pro"}))
    found = measured_costs(read_run(directory))

    flaky = found[(AgentRole.STRUCTURE, "deepseek-v4-pro")]
    assert flaky.attempts == 2 * COUNT
    assert flaky.valid_verdicts == COUNT
    assert flaky.invocations == COUNT
    assert flaky.attempts_per_verdict == pytest.approx(2.0)


def test_the_ping_fixture_is_priced_with_the_same_table_and_keeps_null_apart() -> None:
    """deepseek-v4-flash: la segunda llamada trae 1536 cacheados; la primera, `null`.

    Con `null` no hay coste exacto, así que la media es la de la segunda:
    (71 x 0.14 + 1536 x 0.028 + 177 x 0.28) / 1e6.
    """
    roles = {"deepseek-v4-flash": AgentRole.MOMENTUM}
    found = usage_fixture_costs(USAGE_FIXTURE, roles)

    flash = found["deepseek-v4-flash"]
    assert (flash.measured, flash.unmeasured) == (1, 1)
    assert flash.mean_usd == pytest.approx((71 * 0.14 + 1536 * 0.028 + 177 * 0.28) / 1_000_000)
    assert "Ping, no el prompt real" in flash.source
    assert found["mimo-v2.5"].mean_usd is None, "mimo-v2.5 no existe en payg: sin precio, sin coste"
    assert DEFAULT_PRICING.price_for("mimo-v2.5", Billing.PAYG, NOW) is None


# ───────────────────────────────────────────── El comando ─────────────────────────────────────────


def test_the_command_runs_over_the_real_manifest_without_calling_anyone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    directory, _, _ = probe(tmp_path)
    monkeypatch.setattr(estimate_module, "load_settings", lambda _path: payg_settings())

    code = main([str(directory)])

    out = capsys.readouterr().out
    assert code == 0
    assert "ESTIMACIÓN" in out
    assert "140 activaciones" in out
    assert "total etapa 1" in out
    assert "con los intentos medidos" in out
    assert "no determinado" not in out.split("## Candidatos")[0], "el sondeo midió todos los roles"
    attempts = out.split("## Intentos por veredicto")[1].split("## ")[0]
    assert f"3 intentos / 3 invocaciones · `{directory.name}/glm-5.2@decider.jsonl`" in attempts


def test_the_command_gives_one_line_per_bull_with_its_own_conditioned_decider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    previous = probe(tmp_path)[0]
    mesas, _, _, _ = desks(tmp_path, technical=previous, settings=pro_settings())
    monkeypatch.setattr(estimate_module, "load_settings", lambda _path: pro_settings())
    argv = [str(previous), "--desks", str(mesas), "--balance", "1000"]

    code = main([*argv, "--source", f"momentum={mesas}"])

    out = capsys.readouterr().out
    assert code == 0
    table = out.split("## Etapa 1 por candidato a bull")[1].split("## ")[0]
    rows = [line for line in table.splitlines() if line.startswith("| `")]
    assert [row.split("|")[1].strip() for row in rows] == ["`kimi-k3`", "`qwen3.8-max`"]
    assert all("PASA" in row and "NO PASA" not in row for row in rows)
    assert "no determinado" not in table, "las mesas midieron todos los roles"
    assert "Saldo frente a estimación" not in out, "con escenarios el veredicto va en cada línea"
    assert f"`{mesas.name}/minimax-m3@bear.jsonl` sha-256" in out
    assert f"`{mesas.name}/glm-5.2@decider+kimi-k3.jsonl` sha-256" in out
    attempts = out.split("## Intentos por veredicto")[1].split("## ")[0]
    assert f"| todos | momentum | `deepseek-v4-pro` | 1.00 | {COUNT} intentos / {COUNT}" in attempts
    assert f"`{mesas.name}/deepseek-v4-pro@momentum.jsonl`" in attempts
    for bull in ("kimi-k3", "qwen3.8-max"):
        assert f"| bull `{bull}` | decider | `glm-5.2` | 1.00 | {COUNT} intentos" in attempts
        assert f"`{mesas.name}/glm-5.2@decider+{bull}.jsonl`" in attempts

    # Sin decir de dónde sale momentum, el sondeo técnico no lo midió con ese modelo: no hay total.
    assert main(argv) == 1
    silent = capsys.readouterr().out
    assert "no determinado" in silent.split("## Etapa 1 por candidato a bull")[1].split("## ")[0]


def test_a_source_replaces_the_whole_role_and_never_mixes_two_probes(tmp_path: Path) -> None:
    """El decisor de otro sondeo sustituye al del primero: sus filas, sus intentos, su archivo."""
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "manifiesto.json").write_text("{}", encoding="utf-8")
    first = probe(tmp_path / "a")[0]
    flaky = probe(tmp_path / "b", ZenFake(flaky={"glm-5.2"}))[0]
    base = measured_costs(read_run(first))

    replaced = replace_roles(base, [(AgentRole.DECIDER, read_run(flaky))])

    decider = replaced[(AgentRole.DECIDER, "glm-5.2")]
    assert decider.attempts == 2 * COUNT
    assert decider.attempts_per_verdict == pytest.approx(2.0)
    assert decider.source.startswith(f"{flaky.name}/")
    untouched = {key: value for key, value in replaced.items() if key[0] is not AgentRole.DECIDER}
    assert untouched == {k: v for k, v in base.items() if k[0] is not AgentRole.DECIDER}


def test_a_source_that_is_not_a_role_and_a_directory_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for bad in ("momentum", "nadie=x", "momentum="):
        with pytest.raises(SystemExit) as caught:
            main([str(tmp_path), "--source", bad])
        assert caught.value.code == 2
    assert "ROL=DIRECTORIO" in capsys.readouterr().err


def test_the_command_exits_one_when_the_balance_covers_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    previous = probe(tmp_path)[0]
    mesas, _, _, _ = desks(tmp_path, technical=previous, settings=pro_settings())
    monkeypatch.setattr(estimate_module, "load_settings", lambda _path: pro_settings())
    argv = [str(previous), "--desks", str(mesas), "--source", f"momentum={mesas}"]

    assert main([*argv, "--balance", "0.01"]) == 1
    assert "NO PASA" in capsys.readouterr().out


def test_a_single_probe_with_a_balance_prints_the_check_and_the_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    directory, _, _ = probe(tmp_path)
    monkeypatch.setattr(estimate_module, "load_settings", lambda _path: payg_settings())

    assert main([str(directory), "--balance", "1000"]) == 0
    assert "**PASA**" in capsys.readouterr().out
    assert main([str(directory), "--balance", "0.01"]) == 1
    assert "**NO PASA" in capsys.readouterr().out


def test_a_non_positive_balance_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as caught:
        main([str(tmp_path), "--balance", "0"])
    assert caught.value.code == 2


# ──────────────────────────────────── Desglose por brazo y rol ────────────────────────────────────

BREAKDOWN_ARMS = ("full", "local_technicals", "solo", "always_buy")


def breakdown() -> tuple[ArmRoleLine, ...]:
    return estimate(report(), measured(), FIXED, arms=BREAKDOWN_ARMS).breakdown


def cell(arm: str, role: AgentRole) -> ArmRoleLine:
    (found,) = (line for line in breakdown() if line.arm == arm and line.role is role)
    return found


def test_each_cell_is_the_hand_computed_one() -> None:
    """Las mismas cuentas de arriba, celda a celda.

    full: bear 10 x .003 x 22/12 = .055; decisor 10 x .01 x 19/12 = .15833; bull 10 x .002 = .02;
    momentum .01; structure 10 x (.004 a .008); volume 10 x (.001 a .005). solo: decisor .15833.
    """
    bear = cell("full", AgentRole.BEAR)
    assert (bear.calls, bear.exact) == (N, False)
    assert (bear.attempts_low, bear.attempts_high) == pytest.approx((22 / 12, 22 / 12))
    assert (bear.usd_low, bear.usd_high) == pytest.approx((0.055, 0.055))
    assert bear.model == "bear"

    decider = cell("full", AgentRole.DECIDER)
    assert (decider.usd_low, decider.usd_high) == pytest.approx((0.15833333, 0.15833333))
    assert cell("solo", AgentRole.DECIDER).usd_low == pytest.approx(0.15833333)

    structure = cell("full", AgentRole.STRUCTURE)
    assert (structure.calls, structure.exact, structure.model) == (N, True, None)
    assert (structure.usd_low, structure.usd_high) == pytest.approx((0.04, 0.08))
    assert cell("full", AgentRole.BULL).usd_low == pytest.approx(0.02)
    assert cell("full", AgentRole.MOMENTUM).usd_low == pytest.approx(0.01)


def test_the_cells_of_an_arm_add_up_to_that_arm_and_all_of_them_to_the_total() -> None:
    """El desglose no es otra estimación: es el total, repartido."""
    result = estimate(report(), measured(), FIXED, arms=BREAKDOWN_ARMS)

    for arm, expected in result.arms_with_retries.items():
        own = [line for line in result.breakdown if line.arm == arm]
        assert own, f"{arm} no tiene ninguna celda"
        assert sum(line.usd_low or 0.0 for line in own) == pytest.approx(expected[0])
        assert sum(line.usd_high or 0.0 for line in own) == pytest.approx(expected[1])
    assert sum(line.usd_low or 0.0 for line in result.breakdown) == pytest.approx(
        result.total_with_retries[0]
    )
    assert sum(line.usd_high or 0.0 for line in result.breakdown) == pytest.approx(
        result.total_with_retries[1]
    )


def test_every_cell_says_where_its_cost_and_its_attempts_come_from() -> None:
    bear = cell("full", AgentRole.BEAR)
    assert bear.source == "sondeo/prueba.jsonl"
    assert bear.attempts_source == "22 intentos / 12 invocaciones · `sondeo/prueba.jsonl`"

    ranged = cell("full", AgentRole.STRUCTURE)
    assert "`A` (`sondeo/prueba.jsonl`) a `B` (`sondeo/prueba.jsonl`)" in ranged.source

    fallback = cell("full", AgentRole.BULL)
    assert FALLBACK_NOTE in fallback.attempts_source


def test_a_local_node_and_a_baseline_are_cells_that_cost_nothing_by_rule() -> None:
    """Están en la tabla: un brazo ausente y un brazo que cuesta 0 no son lo mismo."""
    local = cell("local_technicals", AgentRole.STRUCTURE)
    assert (local.calls, local.usd_low, local.usd_high) == (0, 0.0, 0.0)
    assert local.source == LOCAL_NOTE
    assert local.model == "m"

    (baseline,) = (line for line in breakdown() if line.arm == "always_buy")
    assert baseline.role is None
    assert (baseline.usd_low, baseline.usd_high) == (0.0, 0.0)
    assert baseline.source == NO_CALLS_NOTE


def test_a_node_the_cache_resolves_is_a_cell_with_nothing_to_pay() -> None:
    """`no_debate` reutiliza los técnicos de `full`: la celda existe, con 0 llamadas a pagar."""
    cached = DryRunRow(
        arm="no_debate",
        node="structure",
        role=AgentRole.STRUCTURE,
        backend=Backend.OPENAI,
        model="m",
        calls=N,
        exact=True,
        cached=N,
        to_pay=0,
    )
    counted = report().model_copy(update={"rows": (*report().rows, cached)})

    (line,) = (
        item for item in estimate(counted, measured(), FIXED).breakdown if item.arm == "no_debate"
    )

    assert (line.calls, line.usd_low, line.usd_high) == (0, 0.0, 0.0)
    assert line.note == CACHED_NOTE


def test_a_decider_priced_with_the_prompt_of_full_is_an_estimate_and_never_a_bound() -> None:
    """El decisor se midió con el prompt de `full`; en `solo` esa cifra es una estimación.

    Se rotulaba «cota superior», y no lo es: el coste del decisor es sobre todo salida, y un
    prompt más corto no acota cuánto escribe el modelo. Lo que acota es medirlo.
    """
    assert cell("solo", AgentRole.DECIDER).note == FULL_PROMPT_NOTE
    assert cell("full", AgentRole.DECIDER).note is None
    assert cell("local_technicals", AgentRole.DECIDER).note is None
    assert {arm.name for arm in ARMS} >= SHORT_DECIDER_ARMS
    assert "estimación con el prompt de `full`" in FULL_PROMPT_NOTE
    assert "cota" not in FULL_PROMPT_NOTE

    text = render_estimate(estimate(report(), measured(), FIXED, arms=BREAKDOWN_ARMS))
    assert "cota superior" not in text
    assert FULL_PROMPT_NOTE in text


def test_an_undetermined_cell_makes_its_arm_and_the_total_undetermined() -> None:
    """Sin coste de bull, `full` no tiene subtotal: lo que falta no se suma como si fuera 0."""
    without_bull = {key: value for key, value in measured().items() if key[0] is not AgentRole.BULL}
    lines = estimate(report(), without_bull, FIXED, arms=BREAKDOWN_ARMS).breakdown

    text = "\n".join(render_breakdown(lines, "Desglose"))

    assert "| `full` | **total del brazo** | | | | | no determinado | | |" in text
    assert "| `solo` | **total del brazo** | | | | | 0.16 | | |" in text
    assert "| **total etapa 1** | | | | | | no determinado | | |" in text


def test_the_breakdown_table_has_a_row_per_cell_a_subtotal_per_arm_and_the_total() -> None:
    """Las filas se comprueban donde la cifra no cae en una frontera de redondeo."""
    result = estimate(report(), measured(), FIXED, arms=BREAKDOWN_ARMS)

    text = "\n".join(render_breakdown(result.breakdown, "Desglose por brazo y rol"))

    decider = "| `dec` | ≤ 10 | 1.58 | 0.01000 | 0.16 | sondeo/prueba.jsonl |"
    assert f"| `full` | decider {decider}" in text
    assert "| `full` | bear | `bear` | ≤ 10 | 1.83 | 0.00300 |" in text
    ranged = "| rango de candidatos | 10 | 1.00 | 0.00400 a 0.00800 | 0.04 a 0.08 |"
    assert f"| `full` | structure {ranged}" in text
    solo = f"| `dec` | 10 | 1.58 | 0.01000 | 0.16 | sondeo/prueba.jsonl · {FULL_PROMPT_NOTE}"
    assert f"| `solo` | decider {solo}" in text
    assert f"| `local_technicals` | structure | `m` | 0 | — | — | 0.00 | {LOCAL_NOTE} | — |" in text
    assert f"| `always_buy` | — | — | 0 | — | — | 0.00 | {NO_CALLS_NOTE} | — |" in text
    assert "| `full` | **total del brazo** | | | | | 0.29 a 0.37 | | |" in text
    assert "| `solo` | **total del brazo** | | | | | 0.16 | | |" in text
    assert text.rstrip().splitlines()[-1].startswith("| **total etapa 1** | | | | | | 0.6")
    rows = [line for line in text.splitlines() if line.startswith("|")]
    assert {row.count("|") for row in rows} == {10}, "todas las filas, del mismo ancho"


def test_the_breakdown_changes_neither_the_totals_nor_the_verdict() -> None:
    """El veredicto x1.5 se juzga sobre el total de siempre; el desglose solo lo reparte."""
    for item in lines(balance=1.0):
        if not item.breakdown:
            assert item.total == (None, None)
            continue
        assert sum(cell.usd_low or 0.0 for cell in item.breakdown) == pytest.approx(item.total[0])
        assert sum(cell.usd_high or 0.0 for cell in item.breakdown) == pytest.approx(item.total[1])
    one, two, *_ = lines(balance=1.0)
    assert one.total == pytest.approx((0.52, 0.60))
    assert one.balance is not None
    assert one.balance.required == pytest.approx((0.52 * LAUNCH_MARGIN, 0.60 * LAUNCH_MARGIN))
    assert (one.balance.passes, two.balance is not None and two.balance.passes) == (True, False)


def test_the_report_carries_one_breakdown_per_bull_that_has_a_cost() -> None:
    result = estimate(report(), costs(), FIXED)

    text = render_estimate(result, None, lines())

    assert "## Desglose por brazo y rol — bull `b1`" in text
    assert "## Desglose por brazo y rol — bull `b2`" in text
    assert "bull `b3`" not in text.split("## Desglose", 1)[1].split("## Intentos")[0]
    assert "## Desglose por brazo y rol" in render_estimate(result)


def test_the_bull_lines_come_from_the_candidate_list_and_not_from_the_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """El directorio de T4 midió un tercer bull que ya no es candidato: no tiene línea.

    Antes la lista salía de los brazos `<modelo>@bull` que hubiera en `--desks`, y
    `deepseek-v4-pro` —retirado como candidato en T5— seguía saliendo con «no determinado».
    """
    previous = probe(tmp_path)[0]
    retired = Candidate("deepseek-v4-pro", "deepseek")
    monkeypatch.setattr(zen_probe_module, "BULL_CANDIDATES", (*BULL_CANDIDATES, retired))
    mesas, _, _, _ = desks(tmp_path, technical=previous, settings=pro_settings())
    assert "deepseek-v4-pro@bull" in {arm.arm for arm in read_run(mesas).arms}
    monkeypatch.setattr(estimate_module, "load_settings", lambda _path: pro_settings())

    main([str(previous), "--desks", str(mesas), "--source", f"momentum={mesas}"])

    out = capsys.readouterr().out
    table = out.split("## Etapa 1 por candidato a bull")[1].split("## ")[0]
    rows = [line.split("|")[1].strip() for line in table.splitlines() if line.startswith("| `")]
    assert rows == [f"`{candidate.model}`" for candidate in BULL_CANDIDATES]
    assert "bull `deepseek-v4-pro`" not in out


def test_a_candidate_the_directory_did_not_measure_has_a_line_saying_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    previous = probe(tmp_path)[0]
    mesas, _, _, _ = desks(tmp_path, technical=previous, settings=pro_settings())
    monkeypatch.setattr(estimate_module, "load_settings", lambda _path: pro_settings())
    added = Candidate("un-bull-nuevo", "otra")
    monkeypatch.setattr(estimate_module, "BULL_CANDIDATES", (*BULL_CANDIDATES, added))

    main([str(previous), "--desks", str(mesas), "--source", f"momentum={mesas}"])

    table = capsys.readouterr().out.split("## Etapa 1 por candidato a bull")[1].split("## ")[0]
    (row,) = (line for line in table.splitlines() if line.startswith("| `un-bull-nuevo`"))
    assert "ningún alegato válido" in row


# ─────────────────────────────── Fuente por brazo para el decisor ─────────────────────────────────

DECIDERS = DATA_DIR / "zen_probe" / "20261006T235701Z"
"""Los journals de `var/zen-probe/20261006T235701Z`, byte a byte. Ver el README de datos."""

DECIDER_FILES = {
    "glm-5.2@solo.jsonl": "349f08fbf806d31757c3313c9ffb49f06ebfe539dbfe3e8ef8f335e3ff9f1064",
    "glm-5.2@no_debate.jsonl": "aa2f11f13bbe60139a0d25282d877bb49d718cfda2e16b8f71d8d1a640ee230a",
}
GLM = "glm-5.2"
GLM_FIXED = {**FIXED, AgentRole.DECIDER: GLM}


def glm_costs() -> dict[tuple[AgentRole, str], CallCost]:
    """Los costes de `measured()`, con el decisor de `full` bajo el id real del modelo."""
    found = {key: value for key, value in measured().items() if key[0] is not AgentRole.DECIDER}
    found[(AgentRole.DECIDER, GLM)] = cost(
        AgentRole.DECIDER, GLM, 0.01, attempts=19, invocations=12
    )
    return found


def measured_rows(arm: str) -> list[LLMCall]:
    """Las filas `LLMCall` que el sondeo dejó en el brazo `glm-5.2@<arm>`."""
    (found,) = (item for item in read_run(DECIDERS).arms if item.arm == f"{GLM}@{arm}")
    return [call for record in found.records for call in record.calls]


def own_sources(*arms: str) -> dict[tuple[str, AgentRole], CallCost]:
    run = read_run(DECIDERS)
    return {
        (arm, AgentRole.DECIDER): arm_cost(run, AgentRole.DECIDER, GLM, arm)
        for arm in arms or ("solo", "no_debate")
    }


def with_no_debate() -> DryRunReport:
    """El conteo de siempre más `no_debate`: técnicos resueltos por la caché, decisor de pago."""
    extra = (
        DryRunRow(
            arm="no_debate",
            node="structure",
            role=AgentRole.STRUCTURE,
            backend=Backend.OPENAI,
            model="m",
            calls=N,
            exact=True,
            cached=N,
            to_pay=0,
        ),
        row("no_debate", "decide_without_debate", AgentRole.DECIDER, exact=False),
    )
    return report().model_copy(update={"rows": (*report().rows, *extra)})


def test_the_decider_fixture_is_the_journal_of_the_measured_run() -> None:
    """Se fija el digest para que la cifra de abajo salga de las filas que se midieron."""
    for name, digest in DECIDER_FILES.items():
        assert file_sha256(DECIDERS / name) == digest, name


def assert_the_solo_decider_is_priced_from_its_own_rows(
    run: Callable[..., Estimate] = estimate,
) -> None:
    """0.01456 USD por llamada: lo que costaron de media las 12 propuestas de `solo`.

    La cifra no está escrita: se saca de las 12 filas `LLMCall` del journal con la tabla de
    precios, y la celda del decisor de `solo` tiene que dar esa misma. Los 12 intentos son 12
    invocaciones, así que va a 1.00 intentos por llamada y no a los 19/12 del decisor de `full`.
    """
    rows = measured_rows("solo")
    spent = consume(rows, DEFAULT_PRICING, Billing.PAYG)
    assert (len(rows), spent.measured, spent.unmeasured) == (12, 12, 0)
    per_call = spent.cost_usd / len(rows)
    assert per_call == pytest.approx(0.01456, abs=5e-6)

    result = run(
        with_no_debate(), glm_costs(), GLM_FIXED, arms=BREAKDOWN_ARMS, by_arm=own_sources()
    )
    by_cell = {(line.arm, line.role): line for line in result.breakdown}

    solo = by_cell[("solo", AgentRole.DECIDER)]
    assert (solo.per_call_low, solo.per_call_high) == pytest.approx((per_call, per_call))
    assert (solo.attempts_low, solo.attempts_high) == (1.0, 1.0)
    assert (solo.usd_low, solo.usd_high) == pytest.approx((N * per_call, N * per_call))
    assert solo.source == "20261006T235701Z/glm-5.2@solo.jsonl"
    assert solo.attempts_source == (
        "12 intentos / 12 invocaciones · `20261006T235701Z/glm-5.2@solo.jsonl`"
    )
    assert solo.note is None, "con fuente propia no es una estimación con el prompt de otro brazo"
    assert result.arms_with_retries["solo"] == pytest.approx((N * per_call, N * per_call))

    # El decisor de `full` no se toca: sigue siendo el medido con su prompt.
    full = by_cell[("full", AgentRole.DECIDER)]
    assert full.usd_low == pytest.approx(N * 0.01 * 19 / 12)
    assert full.source == "sondeo/prueba.jsonl"


def test_with_a_source_per_arm_the_solo_decider_costs_what_its_own_rows_measured() -> None:
    assert_the_solo_decider_is_priced_from_its_own_rows()


def rewritten(function: Callable[..., Any], old: str, new: str) -> Callable[..., Any]:
    """La función con un fragmento cambiado, compilada en el espacio de `estimate.py`.

    Con las anotaciones diferidas, como el módulo: `Mapping` y `Sequence` solo existen allí bajo
    `TYPE_CHECKING`, y evaluarlas al definir la función fallaría por algo que no es la mutación.
    """
    source = textwrap.dedent(inspect.getsource(function))
    assert source.count(old) == 1, f"el fragmento {old!r} ya no está una sola vez en el código"
    namespace: dict[str, Any] = dict(vars(estimate_module))
    flags = future_flags.annotations.compiler_flag
    exec(compile(source.replace(old, new), "<mutante>", "exec", flags=flags), namespace)
    return namespace[function.__name__]  # type: ignore[no-any-return]


def test_mutation_ignoring_the_source_per_arm_is_caught() -> None:
    """Si la fuente por brazo no se mira, `solo` vuelve a costar lo que el decisor de `full`."""
    ignoring = rewritten(estimate, "own = sources.get((row.arm, row.role))", "own = None")

    with pytest.raises(AssertionError):
        assert_the_solo_decider_is_priced_from_its_own_rows(ignoring)


def test_without_a_source_per_arm_the_estimate_is_the_one_it_was() -> None:
    """Sin fuentes por brazo, o con una para un brazo que el conteo no tiene, nada cambia."""
    before = estimate(report(), measured(), FIXED, arms=BREAKDOWN_ARMS)
    empty = estimate(report(), measured(), FIXED, arms=BREAKDOWN_ARMS, by_arm={})
    elsewhere = estimate(
        report(), glm_costs(), GLM_FIXED, arms=BREAKDOWN_ARMS, by_arm=own_sources("no_debate")
    )
    plain = estimate(report(), glm_costs(), GLM_FIXED, arms=BREAKDOWN_ARMS)

    assert empty == before
    assert elsewhere == plain, "`report()` no tiene el brazo `no_debate`"
    # Las cifras de hoy, las de las cuentas a mano de arriba.
    assert before.total_with_retries == pytest.approx((0.685, 0.765))
    assert before.arms_with_retries["solo"] == pytest.approx((0.15833333, 0.15833333))


def test_the_breakdown_still_adds_up_with_sources_per_arm() -> None:
    """Las celdas siguen sumando el brazo y el total, y el rol suma sus celdas de cada brazo."""
    result = estimate(
        with_no_debate(), glm_costs(), GLM_FIXED, arms=BREAKDOWN_ARMS, by_arm=own_sources()
    )

    for arm, expected in result.arms_with_retries.items():
        own = [line for line in result.breakdown if line.arm == arm]
        assert sum(line.usd_low or 0.0 for line in own) == pytest.approx(expected[0]), arm
        assert sum(line.usd_high or 0.0 for line in own) == pytest.approx(expected[1]), arm
    assert sum(line.usd_low or 0.0 for line in result.breakdown) == pytest.approx(
        result.total_with_retries[0]
    )
    decider = next(line for line in result.roles if line.role is AgentRole.DECIDER)
    cells = [line for line in result.breakdown if line.role is AgentRole.DECIDER]
    assert {line.arm for line in cells} == {"full", "local_technicals", "solo", "no_debate"}
    assert decider.retried_low == pytest.approx(sum(line.usd_low or 0.0 for line in cells))
    assert decider.calls == 4 * N


def test_each_arm_with_its_own_source_is_cheaper_or_dearer_by_what_its_rows_say() -> None:
    """`no_debate` también: 12 filas, su propio coste, sus propios intentos."""
    rows = measured_rows("no_debate")
    per_call = consume(rows, DEFAULT_PRICING, Billing.PAYG).cost_usd / len(rows)
    assert per_call == pytest.approx(0.02014, abs=5e-6)

    result = estimate(
        with_no_debate(), glm_costs(), GLM_FIXED, arms=BREAKDOWN_ARMS, by_arm=own_sources()
    )

    (line,) = (
        item
        for item in result.breakdown
        if item.arm == "no_debate" and item.role is AgentRole.DECIDER
    )
    assert (line.per_call_low, line.usd_low) == pytest.approx((per_call, N * per_call))
    assert line.source == "20261006T235701Z/glm-5.2@no_debate.jsonl"


def test_the_scenarios_carry_the_source_per_arm_and_judge_the_total_that_results() -> None:
    """Por bull cambia lo de siempre; `solo` cuesta lo mismo en todas las líneas, lo suyo."""
    technical = {k: v for k, v in costs().items() if k[0] not in (AgentRole.BULL, AgentRole.BEAR)}
    arguments = (report(), technical, DESKS, GIVEN, GLM_FIXED, ["b1", "b2"])
    plain = scenarios(*arguments, balance=1.0)
    own = scenarios(*arguments, balance=1.0, by_arm=own_sources("solo"))
    per_call = consume(measured_rows("solo"), DEFAULT_PRICING, Billing.PAYG).cost_usd / 12

    for before, after, given in zip(plain, own, (0.010, 0.020), strict=True):
        saved = N * (given * DECIDER_ATTEMPTS - per_call)
        was_low, was_high = before.total
        low, high = after.total
        assert was_low is not None
        assert was_high is not None
        assert low is not None
        assert high is not None
        assert (low, high) == pytest.approx((was_low - saved, was_high - saved))
        assert sum(c.usd_high or 0.0 for c in after.breakdown) == pytest.approx(high)
        assert after.balance is not None
        assert after.balance.required == pytest.approx((low * LAUNCH_MARGIN, high * LAUNCH_MARGIN))
        solo = next(c for c in after.breakdown if c.arm == "solo" and c.role is AgentRole.DECIDER)
        assert solo.usd_low == pytest.approx(N * per_call)


def test_an_arm_the_directory_did_not_measure_is_refused_naming_it() -> None:
    with pytest.raises(ValueError, match=r"glm-5\.2@bull_only"):
        arm_cost(read_run(DECIDERS), AgentRole.DECIDER, GLM, "bull_only")


def test_the_source_option_reads_a_role_and_optionally_an_arm() -> None:
    assert source_option("bear=un/dir") == (AgentRole.BEAR, None, Path("un/dir"))
    assert source_option("decider@solo=un/dir") == (AgentRole.DECIDER, "solo", Path("un/dir"))


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("decider@solo", "ROL=DIRECTORIO"),
        ("nadie=dir", "rol desconocido"),
        ("bear@solo=dir", "solo el decisor"),
        ("decider@inventado=dir", "brazo desconocido"),
        ("decider@=dir", "brazo desconocido"),
    ],
)
def test_a_source_that_does_not_make_sense_is_refused(value: str, message: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match=message):
        source_option(value)


def test_the_command_prices_solo_and_no_debate_from_their_own_arms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Las dos fuentes por brazo, por el comando, sobre el conteo real de 140 activaciones."""
    previous = probe(tmp_path)[0]
    mesas, _, _, _ = desks(tmp_path, technical=previous, settings=pro_settings())
    monkeypatch.setattr(estimate_module, "load_settings", lambda _path: pro_settings())
    argv = [str(previous), "--desks", str(mesas), "--source", f"momentum={mesas}"]
    per_call = consume(measured_rows("solo"), DEFAULT_PRICING, Billing.PAYG).cost_usd / 12

    assert main(argv) == 0
    plain = capsys.readouterr().out
    own = [
        *argv,
        "--source",
        f"decider@solo={DECIDERS}",
        "--source",
        f"decider@no_debate={DECIDERS}",
    ]
    assert main(own) == 0
    out = capsys.readouterr().out

    source = "20261006T235701Z/glm-5.2@solo.jsonl"
    cost_cells = f"{per_call:.5f} | {140 * per_call:.2f}"
    assert f"| `solo` | decider | `glm-5.2` | 140 | 1.00 | {cost_cells} | {source} |" in out
    assert "| `no_debate` | decider | `glm-5.2` | ≤ 140 | 1.00 | 0.02014 |" in out
    assert "| `bull_only` | decider | `glm-5.2` | ≤ 140 | 1.00 |" in out
    bull_only = next(
        line for line in out.splitlines() if line.startswith("| `bull_only` | decider")
    )
    assert FULL_PROMPT_NOTE in bull_only
    assert FULL_PROMPT_NOTE not in next(
        line for line in out.splitlines() if line.startswith("| `solo` | decider")
    )
    assert FULL_PROMPT_NOTE in next(
        line for line in plain.splitlines() if line.startswith("| `solo` | decider")
    )
    for name, digest in DECIDER_FILES.items():
        assert f"`20261006T235701Z/{name}` sha-256 `{digest}`" in out
    attempts = out.split("## Intentos por veredicto")[1].split("## ")[0]
    assert (
        f"| brazo `solo` | decider | `glm-5.2` | 1.00 | 12 intentos / 12 invocaciones · `{source}`"
        in attempts
    )
    assert "cota superior" not in out


def test_the_command_refuses_a_source_per_arm_the_directory_does_not_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    previous = probe(tmp_path)[0]
    monkeypatch.setattr(estimate_module, "load_settings", lambda _path: payg_settings())

    code = main([str(previous), "--source", f"decider@solo={previous}"])

    assert code == 1
    assert "glm-5.2@solo" in capsys.readouterr().err
