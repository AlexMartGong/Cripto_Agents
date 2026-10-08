"""Pruebas del consumo del pool: lo que cuesta cada llamada y cuánto de la suscripción gasta.

Cada cifra de aquí está calculada a mano con los precios de la página, no leída de la función
que se prueba. Las dos mutaciones que fijan el diseño están escritas como pruebas:

- cobrar los cacheados como entrada, y
- ignorar el día de la semana al decidir el pico de DeepSeek.
"""

from __future__ import annotations

import os
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

import pytest

from crypto_agents import consumption as consumption_module
from crypto_agents.audit import PlanKind, RunMeta, arm_journal_path, write_meta
from crypto_agents.consumption import (
    WRITE_NOTE,
    CostGap,
    call_cost_range_usd,
    call_cost_usd,
    consume,
    cost_bound_usd,
    cost_gap,
    format_cost,
    is_free,
    main,
    pool_fraction,
    quota_checks,
    render_consumption,
    render_quota_checks,
)
from crypto_agents.journal import JsonlJournal
from crypto_agents.settings import DEFAULT_PRICING, load_settings
from crypto_agents.state import (
    AgentRole,
    Backend,
    Billing,
    FailureKind,
    LLMCall,
    StructuredOutputMode,
)
from tests.conftest import role_map
from tests.test_metrics import record
from tests.test_net_outcomes import mutated
from tests.test_settings import TEMPLATE

if TYPE_CHECKING:
    from pathlib import Path

    from crypto_agents.journal import EvaluationRecord
    from crypto_agents.settings import Settings

GO, PAYG = Billing.GO, Billing.PAYG
PRICES = DEFAULT_PRICING
DIGEST = "a" * 64

TUESDAY_PEAK = datetime(2026, 10, 6, 2, 0, tzinfo=UTC)
SATURDAY_OFF = datetime(2026, 10, 3, 2, 0, tzinfo=UTC)


def call(
    model: str = "glm-5.2",
    prompt: int | None = 10_000,
    cached: int | None = 4_000,
    completion: int | None = 2_000,
    *,
    at: datetime = TUESDAY_PEAK,
    backend: Backend = Backend.OPENAI,
    cache_hit: bool = False,
    failure: FailureKind | None = None,
    role: AgentRole = AgentRole.DECIDER,
) -> LLMCall:
    """Una llamada con los contadores que la prueba dicte."""
    return LLMCall(
        role=role,
        backend=backend,
        model=model,
        structured_output=StructuredOutputMode.JSON_SCHEMA,
        quota_weight=1.0,
        prompt_digest=DIGEST,
        cache_hit=cache_hit,
        valid=failure is None,
        failure_kind=failure,
        failure_message=None if failure is None else "fallo de prueba",
        latency_ms=0.0 if cache_hit else 100.0,
        at=at,
        prompt_tokens=prompt,
        cached_tokens=cached,
        completion_tokens=completion,
    )


# ───────────────────────────────── Coste y fracción del pool, a mano ──────────────────────────────


def test_a_call_costs_the_uncached_input_the_cached_input_and_the_output_each_at_its_rate() -> None:
    """GLM-5.2: (6 000 * 1.40 + 4 000 * 0.26 + 2 000 * 4.40) / 1e6 = 18 240 / 1e6."""
    cost = call_cost_usd(call(), PRICES, GO)
    assert cost == pytest.approx(0.01824)


def test_the_pool_fraction_is_the_cost_over_the_monthly_limit_of_the_model() -> None:
    """0.01824 / 60 = 0.000304 del pool mensual."""
    assert pool_fraction(call(), PRICES, GO) == pytest.approx(0.000304)


def test_charging_the_cached_tokens_as_input_would_give_another_figure() -> None:
    """La mutación del criterio: cobrar los cacheados como entrada.

    Con 10 000 de entrada a 1.40 y 2 000 de salida a 4.40 saldría 0.0228 y no 0.01824.
    """
    wrong = (10_000 * 1.40 + 2_000 * 4.40) / 1_000_000
    assert wrong == pytest.approx(0.0228)
    assert call_cost_usd(call(), PRICES, GO) != pytest.approx(wrong)


def test_a_fully_cached_prompt_costs_only_the_cached_rate() -> None:
    """Todo el prompt cacheado: (0 * 1.40 + 1 751 * 0.26 + 37 * 4.40) / 1e6."""
    cost = call_cost_usd(call(prompt=1_751, cached=1_751, completion=37), PRICES, GO)
    assert cost == pytest.approx((1_751 * 0.26 + 37 * 4.40) / 1_000_000)


def test_pay_as_you_go_has_a_cost_in_dollars_and_no_pool() -> None:
    """Kimi K2.6: (4 000 * 0.95 + 1 000 * 0.16 + 500 * 4.00) / 1e6 = 0.00596."""
    kimi = call("kimi-k2.6", prompt=5_000, cached=1_000, completion=500)
    assert call_cost_usd(kimi, PRICES, PAYG) == pytest.approx(0.00596)
    assert pool_fraction(kimi, PRICES, PAYG) is None


def test_pay_as_you_go_deepseek_costs_the_same_in_the_peak_and_outside_it() -> None:
    """0.14 / 0.028 / 0.28 sin distinguir pico: (2 000*0.14 + 1 000*0.028 + 400*0.28) / 1e6."""
    peak = call_cost_usd(call("deepseek-v4-flash", 3_000, 1_000, 400), PRICES, PAYG)
    off = call_cost_usd(call("deepseek-v4-flash", 3_000, 1_000, 400, at=SATURDAY_OFF), PRICES, PAYG)
    assert peak == off
    assert peak == pytest.approx((2_000 * 0.14 + 1_000 * 0.028 + 400 * 0.28) / 1_000_000)


# ───────────────────────────── Escritura de caché: un intervalo, a mano ───────────────────────────
#
# qwen3.8-max (Zen, 2026-10-04): 2.00 entrada, 0.25 caché, 6.00 salida y 2.50 escribir en la caché.
# Con 10 000 de prompt, 2 000 leídos de la caché y 1 000 de salida, quedan 8 000 de entrada nueva:
#   bajo = (8 000 * 2.00 + 2 000 * 0.25 + 1 000 * 6.00) / 1e6 = 22 500 / 1e6 = 0.0225
#   alto = (8 000 * 2.50 + 2 000 * 0.25 + 1 000 * 6.00) / 1e6 = 26 500 / 1e6 = 0.0265


def qwen_call() -> LLMCall:
    return call("qwen3.8-max", prompt=10_000, cached=2_000, completion=1_000)


def assert_the_qwen_interval_by_hand() -> None:
    low, high = call_cost_range_usd(qwen_call(), PRICES, PAYG) or (0.0, 0.0)
    assert low == pytest.approx(0.0225)
    assert high == pytest.approx(0.0265)
    total = consume([qwen_call()], PRICES, PAYG)
    assert total.cost_usd == pytest.approx(0.0225)
    assert total.cost_upper_usd == pytest.approx(0.0265)
    assert total.write_priced == 1


def test_a_model_that_charges_to_write_the_cache_has_a_cost_interval() -> None:
    assert_the_qwen_interval_by_hand()


def test_the_low_end_is_the_cost_the_rest_of_the_code_already_gave() -> None:
    assert call_cost_usd(qwen_call(), PRICES, PAYG) == pytest.approx(0.0225)


def test_a_model_without_cache_write_has_an_exact_cost_and_no_marker() -> None:
    kimi = call("kimi-k3", prompt=10_000, cached=2_000, completion=1_000)
    # (8 000 * 3.00 + 2 000 * 0.30 + 1 000 * 15.00) / 1e6 = 39 600 / 1e6
    low, high = call_cost_range_usd(kimi, PRICES, PAYG) or (0.0, 0.0)
    assert low == high == pytest.approx(0.0396)
    total = consume([kimi], PRICES, PAYG)
    assert total.write_priced == 0
    assert format_cost(total) == "0.0396"


def test_the_interval_is_printed_with_its_marker_and_its_footnote() -> None:
    text = format_cost(consume([qwen_call()], PRICES, PAYG))
    assert text == "0.0225 a 0.0265†"
    report = render_consumption({"qwen3.8-max@bull": [qwen_call()]}, PRICES, PAYG, "x")
    assert "0.0225 a 0.0265†" in report
    assert WRITE_NOTE in report


def test_a_report_without_cache_write_rows_has_no_footnote() -> None:
    report = render_consumption({"a": [call("kimi-k3")]}, PRICES, PAYG, "x")
    assert "†" not in report


def test_a_fully_cached_prompt_leaves_nothing_to_charge_as_write() -> None:
    allcached = call("qwen3.8-max", prompt=2_000, cached=2_000, completion=1_000)
    low, high = call_cost_range_usd(allcached, PRICES, PAYG) or (0.0, 1.0)
    assert low == high == pytest.approx((2_000 * 0.25 + 1_000 * 6.00) / 1_000_000)
    assert consume([allcached], PRICES, PAYG).write_priced == 0


def test_a_free_call_has_a_zero_interval_whatever_it_carries() -> None:
    assert call_cost_range_usd(call("qwen3.8-max", cache_hit=True), PRICES, PAYG) == (0.0, 0.0)


def test_a_call_without_tokens_has_no_interval() -> None:
    assert call_cost_range_usd(call("qwen3.8-max", prompt=None), PRICES, PAYG) is None


def test_cached_null_is_an_interval_from_all_cached_to_all_new() -> None:
    """deepseek-v4-flash (payg 0.14 / 0.028 / 0.28), 3 000 de prompt, `cached` null, 400 de salida.

    bajo = (3 000 * 0.028 + 400 * 0.28) / 1e6 = 196 / 1e6 = 0.000196
    alto = (3 000 * 0.14 + 400 * 0.28) / 1e6 = 532 / 1e6 = 0.000532
    """
    unreported = call("deepseek-v4-flash", prompt=3_000, cached=None, completion=400)
    low, high = call_cost_range_usd(unreported, PRICES, PAYG) or (0.0, 0.0)
    assert low == pytest.approx(0.000196)
    assert high == pytest.approx(0.000532)
    total = consume([unreported, unreported], PRICES, PAYG)
    assert total.measured == 0
    assert total.range_usd is not None
    assert total.range_usd[0] == pytest.approx(0.000392)
    assert total.range_usd[1] == pytest.approx(0.001064)
    assert cost_bound_usd(unreported, PRICES, PAYG) == pytest.approx(0.000532)


def test_the_total_has_no_interval_when_a_call_has_no_tokens() -> None:
    total = consume([call("kimi-k3"), call("kimi-k3", prompt=None)], PRICES, PAYG)
    assert total.range_usd is None


def test_mutation_ignoring_the_cache_write_price_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    assert_the_qwen_interval_by_hand()  # control: el código real pasa
    mutant = mutated(
        consumption_module._upper_rate,
        "return row.cache_write if row.cache_write is not None else row.input_per_mtok",
        "return row.input_per_mtok",
        consumption_module,
    )
    monkeypatch.setattr(consumption_module, "_upper_rate", mutant)
    with pytest.raises(AssertionError):
        assert_the_qwen_interval_by_hand()


# ───────────────────────────────────── El pico de DeepSeek ────────────────────────────────────────


def test_deepseek_charges_the_peak_tariff_on_a_tuesday_at_2_utc() -> None:
    """0.30 / 0.006 / 1.20: (2 000 * 0.30 + 1 000 * 0.006 + 400 * 1.20) / 1e6 = 0.001086."""
    peak = call("deepseek-v4-flash", 3_000, 1_000, 400, at=TUESDAY_PEAK)
    assert call_cost_usd(peak, PRICES, GO) == pytest.approx(0.001086)
    assert pool_fraction(peak, PRICES, GO) == pytest.approx(0.001086 / 30)


def test_deepseek_charges_the_off_peak_tariff_on_a_saturday_at_2_utc() -> None:
    """0.15 / 0.003 / 0.60: (2 000 * 0.15 + 1 000 * 0.003 + 400 * 0.60) / 1e6 = 0.000543."""
    off = call("deepseek-v4-flash", 3_000, 1_000, 400, at=SATURDAY_OFF)
    assert call_cost_usd(off, PRICES, GO) == pytest.approx(0.000543)
    assert pool_fraction(off, PRICES, GO) == pytest.approx(0.000543 / 30)


def test_the_same_hour_costs_double_on_a_weekday_and_the_weekend_does_not() -> None:
    """Las mismas 02:00 UTC valen distinto: lo único que cambia es el día de la semana.

    Con la mutación —ignorar el día— el sábado se cobraría como pico y las dos cifras
    serían iguales; este test no pasaría.
    """
    peak = call_cost_usd(call("deepseek-v4-flash", at=TUESDAY_PEAK), PRICES, GO)
    off = call_cost_usd(call("deepseek-v4-flash", at=SATURDAY_OFF), PRICES, GO)
    assert peak is not None
    assert off is not None
    assert peak == pytest.approx(2 * off)


def test_the_peak_is_read_from_the_call_moment_in_utc() -> None:
    """02:00 UTC del martes es lunes por la noche en México: sigue siendo pico."""
    from zoneinfo import ZoneInfo

    local = TUESDAY_PEAK.astimezone(ZoneInfo("America/Mexico_City"))
    in_mexico = call("deepseek-v4-flash", at=local)
    assert call_cost_usd(in_mexico, PRICES, GO) == call_cost_usd(
        call("deepseek-v4-flash", at=TUESDAY_PEAK), PRICES, GO
    )


# ──────────────────────────────── Gratis, ausente y no calculable ─────────────────────────────────


def test_a_cache_hit_costs_nothing_and_uses_none_of_the_pool() -> None:
    hit = call(prompt=None, cached=None, completion=None, cache_hit=True)
    assert is_free(hit)
    assert call_cost_usd(hit, PRICES, GO) == 0.0
    assert pool_fraction(hit, PRICES, GO) == 0.0


def test_a_local_call_costs_nothing_whatever_its_counters_say() -> None:
    """Ollama cuenta tokens, y su cuenta no es un coste: corre en la GPU de la casa."""
    counted = call("qwen3:8b", backend=Backend.OLLAMA, prompt=412, cached=None, completion=96)
    silent = call("qwen3:8b", backend=Backend.OLLAMA, prompt=None, cached=None, completion=None)
    for local in (counted, silent):
        assert is_free(local)
        assert call_cost_usd(local, PRICES, GO) == 0.0
        assert pool_fraction(local, PRICES, GO) == 0.0


def test_free_calls_have_no_pool_under_pay_as_you_go_either() -> None:
    hit = call(cache_hit=True, prompt=None, cached=None, completion=None)
    assert call_cost_usd(hit, PRICES, PAYG) == 0.0
    assert pool_fraction(hit, PRICES, PAYG) is None


@pytest.mark.parametrize(
    "missing",
    [
        {"prompt": None},
        {"completion": None},
        {"prompt": None, "cached": None, "completion": None},
    ],
)
def test_a_remote_call_without_tokens_has_no_cost_and_it_is_not_zero(
    missing: dict[str, None],
) -> None:
    unknown = call(**missing)  # type: ignore[arg-type]
    assert call_cost_usd(unknown, PRICES, GO) is None
    assert pool_fraction(unknown, PRICES, GO) is None
    assert cost_gap(unknown, PRICES, GO) is CostGap.NO_TOKENS


def test_a_cached_counter_the_provider_left_null_gives_no_exact_cost() -> None:
    """DeepSeek informa `null` en frío: puede ser cero o no, y no se supone."""
    cold = call("deepseek-v4-flash", prompt=1_607, cached=None, completion=173, at=SATURDAY_OFF)
    assert call_cost_usd(cold, PRICES, GO) is None
    assert cost_gap(cold, PRICES, GO) is CostGap.CACHED_UNREPORTED


def test_the_ceiling_of_an_unreported_cache_charges_everything_as_new_input() -> None:
    """Cotas, no estimación: los cacheados cuestan menos, así que esto no se puede superar.

    1 607 * 0.15 + 173 * 0.60 = 344.85 → 0.00034485 USD.
    """
    cold = call("deepseek-v4-flash", prompt=1_607, cached=None, completion=173, at=SATURDAY_OFF)
    assert cost_bound_usd(cold, PRICES, GO) == pytest.approx(0.00034485)


def test_the_ceiling_is_the_exact_cost_when_there_is_one_and_none_when_tokens_are_missing() -> None:
    assert cost_bound_usd(call(), PRICES, GO) == pytest.approx(0.01824)
    assert cost_bound_usd(call(prompt=None), PRICES, GO) is None


def test_more_cached_than_prompt_is_incoherent_and_gives_no_cost() -> None:
    """`prompt_tokens` incluye los cacheados: no pueden ser más que el total."""
    broken = call(prompt=100, cached=500, completion=10)
    assert call_cost_usd(broken, PRICES, GO) is None
    assert cost_gap(broken, PRICES, GO) is CostGap.INCONSISTENT


def test_a_model_without_a_price_in_that_billing_has_no_cost() -> None:
    """MiMo-V2.5 no existe con pago por uso."""
    mimo = call("mimo-v2.5", prompt=1_602, cached=0, completion=88)
    assert call_cost_usd(mimo, PRICES, GO) is not None
    assert call_cost_usd(mimo, PRICES, PAYG) is None
    assert cost_gap(mimo, PRICES, PAYG) is CostGap.NO_PRICE


def test_a_provider_that_never_answered_has_no_cost_and_is_not_counted_as_missing_tokens() -> None:
    for kind in (FailureKind.TRANSPORT, FailureKind.TIMEOUT):
        dead = call(prompt=None, cached=None, completion=None, failure=kind)
        assert call_cost_usd(dead, PRICES, GO) is None
        assert cost_gap(dead, PRICES, GO) is CostGap.NO_RESPONSE


def test_an_invalid_attempt_was_billed_all_the_same() -> None:
    """Un intento que no pasó la validación ya se pagó: cuenta como cualquier otro."""
    bad = call(failure=FailureKind.SCHEMA)
    assert call_cost_usd(bad, PRICES, GO) == pytest.approx(0.01824)


# ──────────────────────────────────────── Agregación ──────────────────────────────────────────────


def mixed_calls() -> list[LLMCall]:
    """Ocho llamadas, una de cada clase de cosa que puede pasar."""
    return [
        call(),  # medida: 0.01824
        call("kimi-k2.6", 5_000, 1_000, 500),  # medida: 0.00596 en payg; en go 0.00596 / 60
        call(prompt=None, cached=None, completion=None),  # sin tokens
        call("deepseek-v4-flash", 1_607, None, 173, at=SATURDAY_OFF),  # cached no informado
        call(prompt=100, cached=500, completion=10),  # incoherente
        call(cache_hit=True, prompt=None, cached=None, completion=None),  # gratis
        call("qwen3:8b", backend=Backend.OLLAMA, prompt=7, cached=None, completion=3),  # gratis
        call(
            prompt=None, cached=None, completion=None, failure=FailureKind.TIMEOUT
        ),  # sin respuesta
    ]


def test_every_call_lands_in_exactly_one_bucket() -> None:
    total = consume(mixed_calls(), PRICES, GO)
    assert total.calls == 8
    assert (total.measured, total.free, total.no_response) == (2, 2, 1)
    assert (total.no_tokens, total.cached_unreported, total.inconsistent, total.no_price) == (
        1,
        1,
        1,
        0,
    )
    assert total.unmeasured == 3
    assert total.measured + total.free + total.no_response + total.unmeasured == total.calls


def test_the_cost_adds_only_what_was_measured() -> None:
    total = consume(mixed_calls(), PRICES, GO)
    assert total.cost_usd == pytest.approx(0.01824 + 0.00596)
    assert total.pool == pytest.approx(0.01824 / 60 + 0.00596 / 60)


def test_pay_as_you_go_aggregates_dollars_and_no_pool() -> None:
    total = consume(mixed_calls(), PRICES, PAYG)
    assert total.cost_usd == pytest.approx(0.01824 + 0.00596)
    assert total.pool is None


def test_the_five_hour_share_is_the_pool_over_a_fifth() -> None:
    total = consume([call()], PRICES, GO)
    assert total.window_share(PRICES) == pytest.approx(0.000304 / 0.20)
    assert consume([call()], PRICES, PAYG).window_share(PRICES) is None


def test_the_ceiling_exists_only_when_every_gap_has_one() -> None:
    """Hay cota solo si lo que falta es la caché; con tokens ausentes no hay de dónde sacarla."""
    only_cache = consume(
        [call(), call("deepseek-v4-flash", 1_607, None, 173, at=SATURDAY_OFF)], PRICES, GO
    )
    assert only_cache.ceiling_usd == pytest.approx(0.01824 + 0.00034485)
    assert consume(mixed_calls(), PRICES, GO).ceiling_usd is None
    assert consume([call()], PRICES, GO).ceiling_usd == pytest.approx(0.01824)


def test_an_empty_run_has_nothing_measured() -> None:
    total = consume([], PRICES, GO)
    assert (total.calls, total.measured, total.cost_usd) == (0, 0, 0.0)


# ──────────────────────────────────────── Informe ─────────────────────────────────────────────────


def arms() -> dict[str, list[LLMCall]]:
    return {
        "full": [
            call(role=AgentRole.DECIDER),
            call("deepseek-v4-flash", 1_607, None, 173, at=SATURDAY_OFF, role=AgentRole.MOMENTUM),
        ],
        "solo": [call(role=AgentRole.DECIDER, prompt=None, cached=None, completion=None)],
    }


def test_the_report_puts_the_counts_next_to_every_figure() -> None:
    text = render_consumption(arms(), PRICES, GO, "var/ablation/x")
    row = next(line for line in text.splitlines() if line.startswith("| full"))
    cells = [cell.strip() for cell in row.strip("|").split("|")]
    assert cells[0] == "full"
    assert cells[1:5] == ["2", "1", "1", "0"]  # llamadas, medidas, sin medir, sin respuesta


def test_a_figure_with_missing_calls_is_marked_as_a_lower_bound() -> None:
    text = render_consumption(arms(), PRICES, GO, "var/ablation/x")
    solo = next(line for line in text.splitlines() if line.startswith("| solo"))
    assert "≥" in solo
    exact = render_consumption({"full": [call()]}, PRICES, GO, "var/ablation/x")
    assert "≥" not in next(line for line in exact.splitlines() if line.startswith("| full"))


def test_a_run_with_no_measured_call_says_the_figure_is_not_determined() -> None:
    text = render_consumption(
        {"solo": [call(prompt=None, cached=None, completion=None)]}, PRICES, GO, "d"
    )
    solo = next(line for line in text.splitlines() if line.startswith("| solo"))
    assert "≥ 0.0000" in solo


def test_the_ceiling_column_says_why_when_it_cannot_be_computed() -> None:
    text = render_consumption(arms(), PRICES, GO, "d")
    solo = next(line for line in text.splitlines() if line.startswith("| solo"))
    assert "no determinado" in solo


def test_the_report_has_a_table_by_arm_and_another_by_role() -> None:
    text = render_consumption(arms(), PRICES, GO, "d")
    assert "## Por brazo" in text
    assert "## Por rol" in text
    decider = next(line for line in text.splitlines() if line.startswith("| decider"))
    momentum = next(line for line in text.splitlines() if line.startswith("| momentum"))
    assert decider.split("|")[2].strip() == "2"  # el decisor suma los dos brazos
    assert momentum.split("|")[2].strip() == "1"


def test_the_report_states_the_billing_the_prices_date_and_the_window_share() -> None:
    text = render_consumption(arms(), PRICES, GO, "var/ablation/x")
    assert "facturación: go" in text
    assert str(date(2026, 10, 2)) in text
    assert "20%" in text
    assert "var/ablation/x" in text


def test_under_pay_as_you_go_the_pool_columns_say_there_is_none() -> None:
    text = render_consumption(arms(), PRICES, PAYG, "d")
    assert "facturación: payg" in text
    full = next(line for line in text.splitlines() if line.startswith("| full"))
    assert "sin pool" in full


def test_the_report_names_the_causes_behind_the_unmeasured_calls() -> None:
    text = render_consumption({"a": mixed_calls()}, PRICES, GO, "d")
    assert "sin tokens 1" in text
    assert "cached no informado 1" in text
    assert "incoherentes 1" in text
    assert "sin precio 0" in text


def test_the_report_carries_no_key_and_no_gateway_url() -> None:
    text = render_consumption(arms(), PRICES, GO, "d")
    assert "http" not in text
    assert "sk-" not in text


# ───────────────────────────────── Leer el directorio de una corrida ──────────────────────────────


STARTED = datetime(2026, 10, 1, 17, 0, tzinfo=UTC)


def meta(billing: Billing | None, resumed_from: Path | None = None) -> RunMeta:
    return RunMeta(
        plan_kind=PlanKind.MANIFEST,
        plan_path="data/ablation_selection.json",
        plan_sha256="b" * 64,
        argv=("--fill",),
        fill=True,
        arms=("full", "solo"),
        started_at=STARTED,
        billing=billing,
        resumed_from=None if resumed_from is None else str(resumed_from),
    )


def write_arm(directory: Path, arm: str, calls: list[LLMCall]) -> None:
    journal = JsonlJournal(arm_journal_path(directory, arm))
    records: list[EvaluationRecord] = [record(calls=tuple(calls))]
    for item in records:
        journal.write(item)


def test_the_command_reads_the_arms_of_a_run_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_meta(tmp_path, meta(GO))
    write_arm(tmp_path, "full", [call(), call(prompt=None, cached=None, completion=None)])
    write_arm(tmp_path, "solo", [call(cache_hit=True, prompt=None, cached=None, completion=None)])

    assert main([str(tmp_path)]) == 0

    out = capsys.readouterr().out
    assert "facturación: go" in out
    full = next(line for line in out.splitlines() if line.startswith("| full"))
    assert [cell.strip() for cell in full.strip("|").split("|")][1:5] == ["2", "1", "1", "0"]


def test_the_command_sums_the_passes_of_a_resumed_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Lo gastado en la primera pasada también cuenta: reanudar no devuelve el presupuesto."""
    first, second = tmp_path / "a", tmp_path / "b"
    write_meta(first, meta(GO))
    write_arm(first, "full", [call()])
    write_meta(second, meta(GO, resumed_from=first))
    write_arm(second, "full", [call(), call()])

    assert main([str(second)]) == 0

    full = next(line for line in capsys.readouterr().out.splitlines() if line.startswith("| full"))
    assert [cell.strip() for cell in full.strip("|").split("|")][1:3] == ["3", "3"]


def test_an_old_run_without_billing_needs_it_on_the_command_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Suponer `go` sería inventar una cifra de cuota sobre una corrida que no lo dijo."""
    write_meta(tmp_path, meta(None))
    write_arm(tmp_path, "full", [call()])

    assert main([str(tmp_path)]) == 1
    assert "--billing" in capsys.readouterr().err

    assert main([str(tmp_path), "--billing", "go"]) == 0
    assert "facturación: go" in capsys.readouterr().out


def test_a_billing_flag_that_contradicts_the_run_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_meta(tmp_path, meta(GO))
    write_arm(tmp_path, "full", [call()])

    assert main([str(tmp_path), "--billing", "payg"]) == 1
    assert "contradice" in capsys.readouterr().err


def test_a_directory_that_is_not_a_run_is_an_error_and_not_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(tmp_path)]) == 1
    assert "meta.json" in capsys.readouterr().err


# ──────────────────────── Cuota declarada frente al estimado de la página ─────────────────────────


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Aísla del entorno real del desarrollador, como `tests/test_settings.py`."""
    for key in list(os.environ):
        if key.startswith("CA_"):
            monkeypatch.delenv(key, raising=False)


def settings_from_template() -> Settings:
    """Las seis cuotas del `.env.example` que se reparte."""
    return load_settings(TEMPLATE)


def test_the_template_quotas_that_disagree_with_the_page_are_listed() -> None:
    """Hoy: ninguna discrepa, y momentum y bull no se pueden comparar.

    Hasta el bloque T5 momentum era `deepseek-v4-flash` y discrepaba (63 300 frente a 13 000).
    Ahora es `deepseek-v4-pro`, que la página de Go no estima: su cuota es una declaración y la
    fila lo dice, en vez de contar como acuerdo o como discrepancia. Con bull pasó lo mismo en el
    bloque T7: era `kimi-k2.6` y discrepaba (4 300 frente a 1 150); es `qwen3.8-max`, medido solo
    en Zen, y su cuota es el centinela.
    """
    settings = settings_from_template()
    checks = quota_checks(settings)
    disagreeing = {c.role: (c.configured, c.page) for c in checks if c.matches is False}
    assert disagreeing == {}
    agreeing = {c.role for c in checks if c.matches is True}
    assert agreeing == {
        AgentRole.STRUCTURE,
        AgentRole.VOLUME,
        AgentRole.BEAR,
        AgentRole.DECIDER,
    }
    not_comparable = {c.role: (c.model, c.configured, c.page) for c in checks if c.matches is None}
    assert not_comparable == {
        AgentRole.MOMENTUM: ("deepseek-v4-pro", 100_000, None),
        AgentRole.BULL: ("qwen3.8-max", 100_000, None),
    }
    assert "sin estimado" in render_quota_checks(checks, PRICES)


def test_the_quota_report_shows_the_effective_figure_and_never_corrects_anything() -> None:
    """Con un peso distinto de 1 la cifra que cuenta es la efectiva; se informa y no se corrige.

    La plantilla ya no lleva ningún rol de doble uso, así que el caso se arma con lo que momentum
    declaraba hasta el bloque T5: `deepseek-v4-flash`, 63 300 a peso 2.0.
    """
    template = settings_from_template()
    declared = template.role_config(AgentRole.MOMENTUM)
    flash = declared.primary.model_copy(
        update={"model": "deepseek-v4-flash", "quota_weight": 2.0, "quota_per_window": 63_300}
    )
    double_use = declared.model_copy(update={"primary": flash})
    settings = template.model_copy(
        update={"roles": {**template.roles, AgentRole.MOMENTUM: double_use}}
    )
    checks = quota_checks(settings)
    momentum = next(c for c in checks if c.role is AgentRole.MOMENTUM)
    assert momentum.effective == pytest.approx(31_650)  # 63 300 / peso 2.0
    text = render_quota_checks(checks, PRICES)
    assert "estimación de la página, no medida" in text
    assert "DISCREPA" in text
    assert "momentum" in text
    assert "63300" in text.replace(" ", "")


def test_a_role_on_a_model_the_page_does_not_estimate_is_reported_as_not_comparable() -> None:
    settings = load_settings(
        roles=role_map(),
        openai={"api_key": "sk-test"},
        ollama={"host": "http://localhost:11434"},
    )
    checks = quota_checks(settings)
    assert all(c.matches is None and c.page is None for c in checks)
    assert "sin estimado" in render_quota_checks(checks, PRICES)


def test_with_pay_as_you_go_the_quota_is_not_compared_with_the_go_page() -> None:
    """Zen no publica límite: contrastar su cuota con los estimados de Go inventaría discrepancias.

    La plantilla trae las cifras de Go; con `payg` esas mismas cifras ya no discrepan de nada.
    """
    paying = settings_from_template().model_copy(update={"billing": PAYG})
    checks = quota_checks(paying)

    assert all(check.page is None and check.matches is None for check in checks)
    text = render_quota_checks(checks, DEFAULT_PRICING)
    assert "no publicado por Zen" in text
    assert "DISCREPA" not in text
    assert "Discrepancias: 0" in text
    assert f"al {DEFAULT_PRICING.as_of(PAYG)}" in text


def test_the_consumption_report_dates_the_prices_of_the_billing_it_used() -> None:
    go = render_consumption({}, DEFAULT_PRICING, GO, "x")
    payg = render_consumption({}, DEFAULT_PRICING, PAYG, "x")

    assert f"precios de la página al {DEFAULT_PRICING.as_of(GO)}" in go
    assert f"precios de la página al {DEFAULT_PRICING.as_of(PAYG)}" in payg
    assert DEFAULT_PRICING.as_of(GO) != DEFAULT_PRICING.as_of(PAYG)
