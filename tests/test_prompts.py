"""Pruebas de carga y renderizado de prompts."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from crypto_agents.prompts import (
    PROMPT_DIR,
    PromptError,
    debate_prompt,
    decision_prompt,
    format_brief,
    format_indicators,
    format_verdicts,
    render,
    technical_prompt,
)
from crypto_agents.state import (
    Bias,
    Claim,
    DebateBrief,
    Dimension,
    IndicatorSet,
    MarketSnapshot,
    Observation,
    Side,
    Strength,
    TechnicalVerdict,
)

SNAPSHOT = MarketSnapshot(
    run_id=uuid4(),
    exchange="binance",
    symbol="BTC/USDT",
    timeframe="1h",
    timestamp=datetime(2026, 8, 13, 12, 0, tzinfo=UTC),
    close=64250.5,
    candles_digest="a" * 64,
    candles_count=300,
)
INDICATORS = IndicatorSet(values={"RSI_14": 61.2, "EMA_50": 63900.0})


def verdict(dimension: Dimension = Dimension.STRUCTURE) -> TechnicalVerdict:
    """Veredicto mínimo válido."""
    return TechnicalVerdict(
        dimension=dimension,
        bias=Bias.BULLISH,
        confidence=0.7,
        observations=[
            Observation(
                id=f"{dimension.value}-1",
                text="El precio sostiene el soporte previo y marca un maximo superior.",
                cites=["EMA_50"],
                supports=Bias.BULLISH,
            )
        ],
        invalidation="Pierde el soporte de 63000.",
    )


def brief(side: Side = Side.BULL) -> DebateBrief:
    """Alegato mínimo válido."""
    return DebateBrief(
        side=side,
        thesis="La estructura alcista sigue intacta mientras el soporte aguante.",
        claims=[
            Claim(
                text="La media de 50 actua como soporte dinamico en cada retroceso.",
                grounded_in=["structure-1"],
                strength=Strength.MODERATE,
            ),
            Claim(
                text="El impulso acompana sin llegar a sobrecompra extrema todavia.",
                grounded_in=["structure-1"],
                strength=Strength.WEAK,
            ),
        ],
        conviction=0.6,
        strongest_counterargument="Un cierre bajo 63000 invalida toda la lectura alcista.",
    )


# ─────────────────────────────────────────── Carga ────────────────────────────────────────────────


def test_prompts_live_in_files_not_in_code() -> None:
    """Las plantillas se versionan como archivos y se leen en un diff."""
    names = {path.stem for path in PROMPT_DIR.glob("*.md")}
    assert names == {
        "technical",
        "debate",
        "decider",
        "decider_no_debate",  # variante de ablación: técnicos sin mesas
        "decider_solo",  # variante de ablación: un modelo, una llamada
    }


def test_missing_template_is_reported() -> None:
    """Pedir una plantilla que no existe falla nombrándola."""
    with pytest.raises(PromptError, match="no existe la plantilla"):
        render("inexistente")


def test_missing_variable_is_an_error_not_a_hole() -> None:
    """Una variable sin valor debe fallar, no dejar `$variable` dentro del prompt."""
    with pytest.raises(PromptError, match="falta la variable"):
        render("technical", dimension="structure")


# ───────────────────────────────────────── Formateo ───────────────────────────────────────────────


def test_indicators_are_listed_in_stable_order() -> None:
    """El orden es estable: dos renders del mismo estado dan el mismo digest."""
    assert format_indicators(INDICATORS) == format_indicators(INDICATORS)
    assert format_indicators(INDICATORS).index("EMA_50") < format_indicators(INDICATORS).index(
        "RSI_14"
    )


def test_verdicts_expose_observation_ids() -> None:
    """Las mesas solo pueden citar ids que el prompt les mostró."""
    rendered = format_verdicts([verdict(Dimension.STRUCTURE), verdict(Dimension.VOLUME)])
    assert "`structure-1`" in rendered
    assert "`volume-1`" in rendered


def test_missing_brief_is_marked_explicitly() -> None:
    """Un alegato ausente se dice, no se deja en blanco."""
    assert "no produjo alegato" in format_brief(None)


# ──────────────────────────────────── Prompts por agente ──────────────────────────────────────────


def test_technical_prompt_carries_indicators_and_triggers() -> None:
    """El agente técnico recibe indicadores ya calculados, nunca velas."""
    rendered = technical_prompt(Dimension.MOMENTUM, SNAPSHOT, INDICATORS, ["ma_cross_bullish"])
    assert "momentum" in rendered
    assert "RSI_14" in rendered
    assert "ma_cross_bullish" in rendered
    assert "BTC/USDT" in rendered
    assert "$" not in rendered


def test_both_desks_receive_the_same_evidence() -> None:
    """La evidencia es idéntica; solo cambia el lado que se argumenta."""
    verdicts = [verdict(Dimension.STRUCTURE)]
    bull = debate_prompt(Side.BULL, SNAPSHOT, INDICATORS, verdicts)
    bear = debate_prompt(Side.BEAR, SNAPSHOT, INDICATORS, verdicts)

    assert bull.replace('"bull"', "X") == bear.replace('"bear"', "X").replace(
        "**bear**", "**bull**"
    )
    assert format_verdicts(verdicts) in bull
    assert format_verdicts(verdicts) in bear


def test_debate_prompt_demands_the_counterargument() -> None:
    """El contraargumento es la razón de ser del debate y el prompt lo exige."""
    rendered = debate_prompt(Side.BULL, SNAPSHOT, INDICATORS, [verdict()])
    assert "strongest_counterargument" in rendered
    assert "propaganda" in rendered


def test_decision_prompt_includes_both_briefs() -> None:
    """El decisor ve los dos alegatos completos."""
    rendered = decision_prompt(
        SNAPSHOT, INDICATORS, [verdict()], brief(Side.BULL), brief(Side.BEAR)
    )
    assert rendered.count("contraargumento más fuerte") == 2
    assert "$" not in rendered


def test_editing_a_template_changes_the_rendered_prompt() -> None:
    """Un cambio en el `.md` cambia el prompt y por tanto invalida la caché por sí solo."""
    first = technical_prompt(Dimension.STRUCTURE, SNAPSHOT, INDICATORS, ["ma_cross_bullish"])
    second = technical_prompt(Dimension.STRUCTURE, SNAPSHOT, INDICATORS, ["range_breakout_up"])
    assert first != second
