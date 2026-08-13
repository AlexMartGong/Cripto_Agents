"""Pruebas del contrato de datos.

Cada invariante descrita en `state.py` debe tener aquí un caso que la respeta y
otro que la viola. Si un caso inválido pasa, la regla no existe.
"""

from __future__ import annotations

import math
import operator
from datetime import UTC, datetime
from typing import Annotated, get_args, get_origin
from uuid import uuid4

import pytest
from pydantic import ValidationError

from crypto_agents.state import (
    Action,
    ActivationCheck,
    AgentRole,
    Bias,
    Claim,
    DebateBrief,
    Decision,
    Dimension,
    IndicatorSet,
    LLMCall,
    MarketSnapshot,
    Observation,
    RiskVerdict,
    Side,
    Strength,
    TechnicalEvidence,
    TechnicalVerdict,
    TradingState,
    ungrounded_claim_refs,
    unknown_indicators,
)

DIGEST = "a" * 64
OTHER_DIGEST = "b" * 64


# ───────────────────────────────────────────── Fábricas ───────────────────────────────────────────


def make_snapshot() -> MarketSnapshot:
    """Snapshot mínimo válido."""
    return MarketSnapshot(
        run_id=uuid4(),
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1h",
        timestamp=datetime(2026, 8, 13, 12, 0, tzinfo=UTC),
        close=64250.5,
        candles_digest=DIGEST,
        candles_count=300,
    )


def make_indicators() -> IndicatorSet:
    """Indicadores finitos, todos citables."""
    return IndicatorSet(values={"RSI_14": 61.2, "EMA_50": 63900.0, "ATRr_14": 812.4})


def make_observation(
    dimension: Dimension = Dimension.STRUCTURE,
    index: int = 1,
    cites: list[str] | None = None,
    supports: Bias = Bias.BULLISH,
) -> Observation:
    """Observación válida en la dimensión pedida."""
    return Observation(
        id=f"{dimension.value}-{index}",
        text="El precio sostiene el soporte previo y marca un maximo superior confirmado.",
        cites=cites if cites is not None else ["EMA_50"],
        supports=supports,
    )


def make_verdict(
    dimension: Dimension = Dimension.STRUCTURE,
    bias: Bias = Bias.BULLISH,
    confidence: float = 0.7,
    observations: list[Observation] | None = None,
) -> TechnicalVerdict:
    """Veredicto válido en la dimensión pedida."""
    return TechnicalVerdict(
        dimension=dimension,
        bias=bias,
        confidence=confidence,
        observations=observations if observations is not None else [make_observation(dimension)],
        invalidation="Pierde el soporte de 63000.",
    )


def make_brief(side: Side = Side.BULL, grounded_in: list[str] | None = None) -> DebateBrief:
    """Alegato válido con dos claims."""
    refs = grounded_in if grounded_in is not None else ["structure-1"]
    return DebateBrief(
        side=side,
        thesis="La estructura alcista sigue intacta mientras el soporte aguante.",
        claims=[
            Claim(
                text="La media de 50 actua como soporte dinamico en cada retroceso reciente.",
                grounded_in=refs,
                strength=Strength.MODERATE,
            ),
            Claim(
                text="El impulso acompana sin llegar a lecturas de sobrecompra extrema.",
                grounded_in=refs,
                strength=Strength.WEAK,
            ),
        ],
        conviction=0.6,
        strongest_counterargument="Un cierre bajo 63000 invalida toda la lectura alcista.",
    )


def make_call(cache_hit: bool = False, weight: float = 1.0) -> LLMCall:
    """Registro de llamada a modelo."""
    return LLMCall(
        role=AgentRole.STRUCTURE,
        model="gpt-x",
        quota_weight=weight,
        prompt_digest=OTHER_DIGEST,
        cache_hit=cache_hit,
        latency_ms=812.0,
    )


# ────────────────────────────────────── Bases y capa determinista ─────────────────────────────────


def test_llm_output_forbids_extra_fields() -> None:
    """Un campo alucinado debe reventar la validación."""
    with pytest.raises(ValidationError):
        Observation(
            id="structure-1",
            text="El precio sostiene el soporte previo y marca un maximo superior confirmado.",
            cites=["EMA_50"],
            supports=Bias.BULLISH,
            certainty=0.9,  # type: ignore[call-arg]
        )


def test_llm_output_is_frozen() -> None:
    """Ningún nodo posterior puede editar lo que dijo otro agente."""
    observation = make_observation()
    with pytest.raises(ValidationError):
        observation.text = "reescrito"


def test_indicator_set_rejects_nan() -> None:
    """Un NaN significa warm-up insuficiente, no una lectura válida."""
    with pytest.raises(ValidationError, match="no finitos"):
        IndicatorSet(values={"RSI_14": 61.2, "EMA_200": math.nan})


def test_indicator_set_rejects_infinity() -> None:
    """Un infinito tampoco es una lectura válida."""
    with pytest.raises(ValidationError, match="no finitos"):
        IndicatorSet(values={"RSI_14": math.inf})


def test_indicator_set_rejects_empty() -> None:
    """Un conjunto vacío no permite citar nada."""
    with pytest.raises(ValidationError):
        IndicatorSet(values={})


def test_indicator_set_exposes_names() -> None:
    """`names` es el universo citable por los agentes técnicos."""
    assert make_indicators().names == frozenset({"RSI_14", "EMA_50", "ATRr_14"})


def test_activation_check_requires_trigger_when_running() -> None:
    """Activarse sin disparador explícito es gasto injustificado."""
    with pytest.raises(ValidationError, match="trigger"):
        ActivationCheck(should_run=True, triggers=[], reason="algo cambio")


def test_activation_check_allows_skip_without_triggers() -> None:
    """No activarse sin triggers es el caso normal."""
    check = ActivationCheck(should_run=False, triggers=[], reason="sin cambio material")
    assert check.should_run is False


# ─────────────────────────────────────────── Capa técnica ─────────────────────────────────────────


@pytest.mark.parametrize("bad_id", ["obs-1", "structure", "structure-", "sentiment-1", "-1"])
def test_observation_id_pattern_is_enforced(bad_id: str) -> None:
    """Un id sin prefijo de dimensión válido no es rastreable hasta su agente."""
    with pytest.raises(ValidationError):
        Observation(
            id=bad_id,
            text="El precio sostiene el soporte previo y marca un maximo superior confirmado.",
            cites=["EMA_50"],
            supports=Bias.BULLISH,
        )


def test_observation_derives_dimension_from_id() -> None:
    """La dimensión sale del prefijo, no de un campo aparte que pueda contradecirlo."""
    assert make_observation(Dimension.VOLUME).dimension is Dimension.VOLUME


def test_observation_requires_at_least_one_citation() -> None:
    """Una observación sin indicador citado es opinión, no evidencia."""
    with pytest.raises(ValidationError):
        make_observation(cites=[])


def test_verdict_rejects_observations_from_another_dimension() -> None:
    """El agente de momentum no puede haber visto una observación de estructura."""
    with pytest.raises(ValidationError, match="ajenas a la dimensión"):
        make_verdict(
            dimension=Dimension.MOMENTUM,
            observations=[make_observation(Dimension.STRUCTURE)],
        )


def test_verdict_rejects_duplicate_observation_ids() -> None:
    """Un id repetido vuelve ambigua la fundamentación de las mesas."""
    with pytest.raises(ValidationError, match="duplicados"):
        make_verdict(
            observations=[
                make_observation(Dimension.STRUCTURE, 1),
                make_observation(Dimension.STRUCTURE, 1, cites=["RSI_14"]),
            ]
        )


def test_verdict_caps_observations_at_five() -> None:
    """Más de cinco observaciones diluye la evidencia y encarece el debate."""
    with pytest.raises(ValidationError):
        make_verdict(
            observations=[make_observation(Dimension.STRUCTURE, i) for i in range(1, 7)],
        )


def test_evidence_rejects_two_verdicts_of_same_dimension() -> None:
    """Dos veredictos de la misma dimensión indican un fan-in mal armado."""
    with pytest.raises(ValidationError, match="más de un veredicto"):
        TechnicalEvidence(
            snapshot=make_snapshot(),
            indicators=make_indicators(),
            verdicts=(make_verdict(Dimension.STRUCTURE), make_verdict(Dimension.STRUCTURE)),
        )


def test_evidence_exposes_observation_ids() -> None:
    """El universo de ids citables por las mesas."""
    evidence = TechnicalEvidence(
        snapshot=make_snapshot(),
        indicators=make_indicators(),
        verdicts=(make_verdict(Dimension.STRUCTURE), make_verdict(Dimension.MOMENTUM)),
    )
    assert evidence.observation_ids == frozenset({"structure-1", "momentum-1"})


def test_net_bias_is_signed_average_in_range() -> None:
    """Sesgo neto en [-1, 1]: dos alcistas y un bajista dan positivo."""
    evidence = TechnicalEvidence(
        snapshot=make_snapshot(),
        indicators=make_indicators(),
        verdicts=(
            make_verdict(Dimension.STRUCTURE, Bias.BULLISH, 0.9),
            make_verdict(Dimension.MOMENTUM, Bias.BEARISH, 0.3),
            make_verdict(Dimension.VOLUME, Bias.NEUTRAL, 1.0),
        ),
    )
    assert evidence.net_bias == pytest.approx((0.9 - 0.3) / 3)


# ─────────────────────────────────────────── Capa de debate ───────────────────────────────────────


def test_claim_requires_grounding() -> None:
    """Una afirmación sin anclaje no es auditable."""
    with pytest.raises(ValidationError):
        Claim(
            text="El mercado va a subir porque el sentimiento general ha mejorado mucho.",
            grounded_in=[],
            strength=Strength.STRONG,
        )


def test_brief_requires_strongest_counterargument() -> None:
    """Sin contraargumento cada mesa produce propaganda."""
    with pytest.raises(ValidationError):
        DebateBrief(
            side=Side.BULL,
            thesis="La estructura alcista sigue intacta mientras el soporte aguante.",
            claims=make_brief().claims,
            conviction=0.6,
        )  # type: ignore[call-arg]


def test_brief_requires_at_least_two_claims() -> None:
    """Una sola afirmación no es un alegato."""
    with pytest.raises(ValidationError):
        DebateBrief(
            side=Side.BULL,
            thesis="La estructura alcista sigue intacta mientras el soporte aguante.",
            claims=[make_brief().claims[0]],
            conviction=0.6,
            strongest_counterargument="Un cierre bajo 63000 invalida toda la lectura alcista.",
        )


# ──────────────────────────────────────── Decisión y riesgo ───────────────────────────────────────


def test_hold_needs_no_invalidation_price() -> None:
    """Quedarse fuera no requiere nivel de invalidación."""
    decision = Decision(
        action=Action.HOLD,
        confidence=0.5,
        size_fraction=0.0,
        rationale="La evidencia no es concluyente en ninguna direccion.",
    )
    assert decision.invalidation_price is None


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"invalidation_price": None}, "invalidation_price"),
        ({"size_fraction": 0.0}, "size_fraction > 0"),
        ({"dismissed_side": None}, "dismissed_side"),
        ({"dismissal_reason": None}, "dismissal_reason"),
    ],
)
def test_actionable_decision_is_incomplete_without(
    overrides: dict[str, object], expected: str
) -> None:
    """Actuar sin invalidación, tamaño o mesa descartada no es una decisión."""
    payload: dict[str, object] = {
        "action": Action.BUY,
        "confidence": 0.8,
        "size_fraction": 0.25,
        "invalidation_price": 63000.0,
        "rationale": "La estructura y el volumen coinciden en direccion alcista.",
        "dismissed_side": Side.BEAR,
        "dismissal_reason": "Su contraargumento depende de un nivel ya perdido.",
    }
    payload.update(overrides)
    with pytest.raises(ValidationError, match=expected):
        Decision(**payload)  # type: ignore[arg-type]


def test_veto_requires_reason() -> None:
    """Un veto sin causa registrada no es auditable."""
    with pytest.raises(ValidationError, match="veto_reason"):
        RiskVerdict(approved=False, final_size_fraction=0.0, veto_reason=None)


def test_veto_requires_zero_size() -> None:
    """Un veto con tamaño distinto de cero no es un veto."""
    with pytest.raises(ValidationError, match="final_size_fraction == 0"):
        RiskVerdict(approved=False, final_size_fraction=0.1, veto_reason="drawdown maximo")


def test_risk_verdict_is_not_an_llm_output() -> None:
    """El gate de riesgo es código determinista, no un modelo."""
    from crypto_agents.state import LLMOutput

    assert not issubclass(RiskVerdict, LLMOutput)


# ──────────────────────────────────────── Estado y reducers ───────────────────────────────────────


@pytest.mark.parametrize("field", ["verdicts", "briefs", "calls", "errors"])
def test_concurrent_fields_declare_add_reducer(field: str) -> None:
    """Sin reducer, LangGraph pierde escrituras paralelas sin lanzar excepción."""
    annotation = TradingState.model_fields[field].rebuild_annotation()
    assert get_origin(annotation) is Annotated
    assert operator.add in get_args(annotation)


def test_reducer_accumulates_partial_updates() -> None:
    """Dos actualizaciones parciales sobre `verdicts` acumulan en vez de sobrescribir."""
    from langgraph.graph import END, START, StateGraph

    def emit_structure(state: TradingState) -> dict[str, list[TechnicalVerdict]]:
        return {"verdicts": [make_verdict(Dimension.STRUCTURE)]}

    def emit_momentum(state: TradingState) -> dict[str, list[TechnicalVerdict]]:
        return {"verdicts": [make_verdict(Dimension.MOMENTUM)]}

    def emit_volume(state: TradingState) -> dict[str, list[TechnicalVerdict]]:
        return {"verdicts": [make_verdict(Dimension.VOLUME)]}

    builder = StateGraph(TradingState)
    builder.add_node("structure", emit_structure)
    builder.add_node("momentum", emit_momentum)
    builder.add_node("volume", emit_volume)
    for name in ("structure", "momentum", "volume"):
        builder.add_edge(START, name)
        builder.add_edge(name, END)

    result = builder.compile().invoke(TradingState())

    assert {verdict.dimension for verdict in result["verdicts"]} == set(Dimension)
    assert len(result["verdicts"]) == 3


def test_reducer_accumulates_sequential_updates() -> None:
    """El reducer también cubre la cadena secuencial, que es donde la pérdida es silenciosa.

    Sin reducer, el fan-out paralelo revienta con `InvalidUpdateError`, pero tres
    escrituras en cadena se pisan sin error y dejan solo la última.
    """
    from langgraph.graph import END, START, StateGraph

    def emit_structure(state: TradingState) -> dict[str, list[TechnicalVerdict]]:
        return {"verdicts": [make_verdict(Dimension.STRUCTURE)]}

    def emit_momentum(state: TradingState) -> dict[str, list[TechnicalVerdict]]:
        return {"verdicts": [make_verdict(Dimension.MOMENTUM)]}

    def emit_volume(state: TradingState) -> dict[str, list[TechnicalVerdict]]:
        return {"verdicts": [make_verdict(Dimension.VOLUME)]}

    builder = StateGraph(TradingState)
    builder.add_node("structure", emit_structure)
    builder.add_node("momentum", emit_momentum)
    builder.add_node("volume", emit_volume)
    builder.add_edge(START, "structure")
    builder.add_edge("structure", "momentum")
    builder.add_edge("momentum", "volume")
    builder.add_edge("volume", END)

    result = builder.compile().invoke(TradingState())

    assert len(result["verdicts"]) == 3


def test_quota_used_ignores_cache_hits() -> None:
    """Un cache hit no llegó al proveedor, así que no consume presupuesto."""
    state = TradingState(
        calls=[make_call(weight=2.0), make_call(cache_hit=True, weight=2.0), make_call()]
    )
    assert state.quota_used == pytest.approx(3.0)


def test_brief_accessor_by_side() -> None:
    """Accesor por lado; None si esa mesa todavía no escribió."""
    state = TradingState(briefs=[make_brief(Side.BULL)])
    bull = state.brief(Side.BULL)
    assert bull is not None
    assert bull.side is Side.BULL
    assert state.brief(Side.BEAR) is None


# ──────────────────────────────────────── Validación cruzada ──────────────────────────────────────


def test_unknown_indicators_empty_when_all_cited_exist() -> None:
    """Lista vacía significa que el agente no inventó indicadores."""
    verdict = make_verdict(
        observations=[make_observation(Dimension.STRUCTURE, 1, cites=["EMA_50", "ATRr_14"])]
    )
    assert unknown_indicators(verdict, make_indicators()) == []


def test_unknown_indicators_reports_hallucinated_names() -> None:
    """Un indicador citado que no se calculó es una alucinación."""
    verdict = make_verdict(
        observations=[
            make_observation(Dimension.STRUCTURE, 1, cites=["EMA_50", "ICHIMOKU_9"]),
            make_observation(Dimension.STRUCTURE, 2, cites=["SUPERTREND_7"]),
        ]
    )
    assert unknown_indicators(verdict, make_indicators()) == ["ICHIMOKU_9", "SUPERTREND_7"]


def test_ungrounded_claim_refs_empty_when_all_ids_exist() -> None:
    """Lista vacía significa que la mesa citó solo evidencia real."""
    verdicts = [make_verdict(Dimension.STRUCTURE)]
    assert ungrounded_claim_refs(make_brief(grounded_in=["structure-1"]), verdicts) == []


def test_ungrounded_claim_refs_reports_invented_ids() -> None:
    """Un id que no salió de ningún veredicto técnico es una cita inventada."""
    verdicts = [make_verdict(Dimension.STRUCTURE)]
    brief = make_brief(grounded_in=["structure-1", "volume-9", "momentum-3"])
    assert ungrounded_claim_refs(brief, verdicts) == ["momentum-3", "volume-9"]
