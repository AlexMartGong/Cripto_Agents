"""Contrato de datos entre todos los nodos del grafo.

Este módulo fija qué puede decir cada agente. Las invariantes se imponen por
validación, no por convención: si un agente puede romper una regla y el proceso
sigue corriendo, la regla no existe.

Reglas estructurales:

- Todo lo que emite un modelo hereda de `LLMOutput`: `extra="forbid"` hace que un
  campo alucinado reviente la validación, y `frozen=True` impide que un nodo
  posterior edite lo que dijo otro agente.
- `RiskVerdict` es determinista y NO hereda de `LLMOutput`.
- El estado no contiene DataFrames, clientes de exchange ni objetos de LangChain:
  se serializa entero en cada checkpoint.
"""

from __future__ import annotations

import math
import operator
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Self
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, PositiveFloat, model_validator

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "OBSERVATION_ID_PATTERN",
    "Action",
    "ActivationCheck",
    "AgentRole",
    "Bias",
    "Claim",
    "DebateBrief",
    "Decision",
    "Dimension",
    "FrozenModel",
    "IndicatorSet",
    "LLMCall",
    "LLMOutput",
    "MarketSnapshot",
    "NodeError",
    "Observation",
    "RiskVerdict",
    "Side",
    "Strength",
    "TechnicalEvidence",
    "TechnicalVerdict",
    "TradingState",
    "ungrounded_claim_refs",
    "unknown_indicators",
]

_DIGEST_PATTERN = r"^[0-9a-f]{64}$"


# ─────────────────────────────────────────── Vocabulario ──────────────────────────────────────────


class Action(StrEnum):
    """Acción de trading que puede emitir el decisor."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class Bias(StrEnum):
    """Dirección que soporta una pieza de evidencia."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class Dimension(StrEnum):
    """Dimensión técnica analizada por un agente en paralelo."""

    STRUCTURE = "structure"
    MOMENTUM = "momentum"
    VOLUME = "volume"


class Side(StrEnum):
    """Mesa de debate."""

    BULL = "bull"
    BEAR = "bear"


class Strength(StrEnum):
    """Fuerza declarada de una afirmación de debate.

    Categórica y no numérica a propósito: un float invita a comparar 0.71 contra
    0.68 como si la diferencia significara algo.
    """

    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


class AgentRole(StrEnum):
    """Rol que consume cuota de modelo.

    Vive aquí y no en la configuración porque el contrato de datos no debe
    depender de la config; la config importa de este módulo, nunca al revés.
    """

    STRUCTURE = "structure"
    MOMENTUM = "momentum"
    VOLUME = "volume"
    BULL = "bull"
    BEAR = "bear"
    DECIDER = "decider"


_DIMENSION_ALTERNATION = "|".join(dimension.value for dimension in Dimension)
OBSERVATION_ID_PATTERN = rf"^(?:{_DIMENSION_ALTERNATION})-\d+$"
"""Patrón `<dimension>-<n>`: el prefijo hace los ids citables sin colisión entre agentes."""


# ────────────────────────────────────────────── Bases ─────────────────────────────────────────────


class FrozenModel(BaseModel):
    """Base inmutable y cerrada, común a la capa determinista y a la de modelos."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class LLMOutput(FrozenModel):
    """Base de todo lo que produce un modelo.

    Marcador semántico además de configuración: hace explícito qué partes del
    estado provienen de un modelo y por tanto no son de fiar sin validación.
    """


# ───────────────────────────────────────── Capa determinista ──────────────────────────────────────


class MarketSnapshot(FrozenModel):
    """Identidad del momento evaluado.

    El DataFrame de velas no viaja en el estado: solo su digest, que basta para
    detectar que dos ejecuciones vieron datos distintos.
    """

    run_id: UUID
    exchange: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    timeframe: str = Field(min_length=1)
    timestamp: AwareDatetime
    close: PositiveFloat
    candles_digest: str = Field(pattern=_DIGEST_PATTERN)
    candles_count: int = Field(ge=2)


class IndicatorSet(FrozenModel):
    """Valores ya calculados por pandas-ta.

    La clave es el nombre del indicador y el valor está tipado: no es un escape
    tipo `dict[str, Any]`. Campos fijos amarrarían el esquema a un preset y
    romperían al cambiar de timeframe.
    """

    values: dict[str, float] = Field(min_length=1)

    @model_validator(mode="after")
    def _reject_non_finite(self) -> Self:
        """Un NaN o un infinito significa warm-up insuficiente, no una lectura válida."""
        offenders = sorted(name for name, value in self.values.items() if not math.isfinite(value))
        if offenders:
            raise ValueError(
                f"indicadores no finitos (warm-up insuficiente): {', '.join(offenders)}"
            )
        return self

    @property
    def names(self) -> frozenset[str]:
        """Conjunto de indicadores disponibles para citar."""
        return frozenset(self.values)


class ActivationCheck(FrozenModel):
    """Veredicto del gate que decide si vale la pena gastar llamadas a modelos."""

    should_run: bool
    triggers: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def _require_trigger_when_running(self) -> Self:
        """Activarse sin un disparador explícito es gasto injustificado."""
        if self.should_run and not self.triggers:
            raise ValueError("should_run=True exige al menos un trigger explícito")
        return self


# ─────────────────────────────────────────── Capa técnica ─────────────────────────────────────────


class Observation(LLMOutput):
    """Unidad mínima de evidencia técnica.

    El id lleva la dimensión como prefijo, así que una mesa de debate puede
    citarlo y el chequeo de fundamentación puede rastrearlo hasta su origen.
    """

    id: str = Field(pattern=OBSERVATION_ID_PATTERN)
    text: str = Field(min_length=20, max_length=280)
    cites: list[str] = Field(min_length=1)
    supports: Bias

    @property
    def dimension(self) -> Dimension:
        """Dimensión derivada del prefijo del id."""
        return Dimension(self.id.split("-", 1)[0])


class TechnicalVerdict(LLMOutput):
    """Lectura de una dimensión técnica sobre indicadores ya calculados.

    Los agentes interpretan indicadores; nunca los calculan.
    """

    dimension: Dimension
    bias: Bias
    confidence: float = Field(ge=0.0, le=1.0)
    observations: list[Observation] = Field(min_length=1, max_length=5)
    invalidation: str = Field(min_length=10)

    @model_validator(mode="after")
    def _observations_belong_to_dimension(self) -> Self:
        """Un id de otra dimensión sería evidencia que este agente no puede haber visto."""
        foreign = sorted(
            observation.id
            for observation in self.observations
            if observation.dimension is not self.dimension
        )
        if foreign:
            raise ValueError(
                f"observaciones ajenas a la dimensión {self.dimension.value}: {', '.join(foreign)}"
            )
        return self

    @model_validator(mode="after")
    def _observation_ids_unique(self) -> Self:
        """Un id repetido vuelve ambigua la fundamentación de las mesas."""
        seen: set[str] = set()
        duplicates: set[str] = set()
        for observation in self.observations:
            if observation.id in seen:
                duplicates.add(observation.id)
            seen.add(observation.id)
        if duplicates:
            raise ValueError(f"ids de observación duplicados: {', '.join(sorted(duplicates))}")
        return self


class TechnicalEvidence(FrozenModel):
    """Consolidado que reciben ambas mesas, idéntico para las dos."""

    snapshot: MarketSnapshot
    indicators: IndicatorSet
    verdicts: tuple[TechnicalVerdict, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _one_verdict_per_dimension(self) -> Self:
        """Dos veredictos de la misma dimensión indican un fan-in mal armado."""
        dimensions = [verdict.dimension for verdict in self.verdicts]
        if len(set(dimensions)) != len(dimensions):
            raise ValueError("hay más de un veredicto por dimensión")
        return self

    @property
    def observation_ids(self) -> frozenset[str]:
        """Ids que las mesas pueden citar legítimamente."""
        return frozenset(
            observation.id for verdict in self.verdicts for observation in verdict.observations
        )

    @property
    def net_bias(self) -> float:
        """Sesgo neto en [-1, 1]. Solo referencia: ningún nodo decide con esto."""
        total = 0.0
        for verdict in self.verdicts:
            if verdict.bias is Bias.BULLISH:
                total += verdict.confidence
            elif verdict.bias is Bias.BEARISH:
                total -= verdict.confidence
        return total / len(self.verdicts)


# ─────────────────────────────────────────── Capa de debate ───────────────────────────────────────


class Claim(LLMOutput):
    """Afirmación de una mesa, anclada a evidencia técnica existente."""

    text: str = Field(min_length=20, max_length=400)
    grounded_in: list[str] = Field(min_length=1)
    strength: Strength


class DebateBrief(LLMOutput):
    """Alegato de una mesa sobre la evidencia común."""

    side: Side
    thesis: str = Field(min_length=20, max_length=400)
    claims: list[Claim] = Field(min_length=2, max_length=5)
    conviction: float = Field(ge=0.0, le=1.0)
    strongest_counterargument: str = Field(min_length=20)
    """Obligatorio: sin esto cada mesa produce propaganda y el decisor cuenta párrafos."""


# ──────────────────────────────────────── Decisión y riesgo ───────────────────────────────────────


class Decision(LLMOutput):
    """Salida del decisor: qué hacer y con qué tamaño propuesto."""

    action: Action
    confidence: float = Field(ge=0.0, le=1.0)
    size_fraction: float = Field(ge=0.0, le=1.0)
    invalidation_price: PositiveFloat | None = None
    rationale: str = Field(min_length=20)
    dismissed_side: Side | None = None
    dismissal_reason: str | None = None

    @model_validator(mode="after")
    def _actionable_decisions_are_complete(self) -> Self:
        """Actuar sin invalidación, sin tamaño o sin descartar una mesa no es una decisión."""
        if self.action is Action.HOLD:
            return self
        missing: list[str] = []
        if self.invalidation_price is None:
            missing.append("invalidation_price")
        if self.size_fraction <= 0.0:
            missing.append("size_fraction > 0")
        if self.dismissed_side is None:
            missing.append("dismissed_side")
        if not self.dismissal_reason:
            missing.append("dismissal_reason")
        if missing:
            raise ValueError(f"la acción {self.action.value} exige: {', '.join(missing)}")
        return self


class RiskVerdict(FrozenModel):
    """Veredicto del gate de riesgo. Determinista: es código, no un modelo."""

    approved: bool
    final_size_fraction: float = Field(ge=0.0, le=1.0)
    applied_limits: tuple[str, ...] = ()
    veto_reason: str | None = None

    @model_validator(mode="after")
    def _veto_is_recorded_and_flat(self) -> Self:
        """Un veto sin causa no es auditable; un veto con tamaño no es un veto."""
        if self.approved:
            return self
        if not self.veto_reason:
            raise ValueError("un veto exige veto_reason")
        if self.final_size_fraction != 0.0:
            raise ValueError("un veto exige final_size_fraction == 0")
        return self


# ───────────────────────────────────────────── Contabilidad ───────────────────────────────────────


class LLMCall(FrozenModel):
    """Registro de una llamada a modelo. Base del presupuesto y del replay.

    `at` es lo que permite reconstruir una ventana deslizante desde el estado:
    sin marca de tiempo el contador solo existe en memoria y el replay no puede
    recalcular cuánta cuota había consumida en cada punto de la ejecución.
    """

    role: AgentRole
    model: str = Field(min_length=1)
    quota_weight: float = Field(gt=0.0)
    prompt_digest: str = Field(pattern=_DIGEST_PATTERN)
    cache_hit: bool = False
    latency_ms: float = Field(ge=0.0)
    at: AwareDatetime


class NodeError(FrozenModel):
    """Fallo atribuido a un nodo concreto, para replay y diagnóstico."""

    node: str = Field(min_length=1)
    message: str = Field(min_length=1)
    at: AwareDatetime


# ──────────────────────────────────────── Estado del grafo ────────────────────────────────────────


class TradingState(BaseModel):
    """Estado que LangGraph acarrea por el pipeline.

    Los campos escritos por varios nodos en paralelo llevan `operator.add` como
    reducer. Sin él, LangGraph aplica last-write-wins dentro del mismo super-step
    y se pierden dos de los tres veredictos técnicos sin lanzar excepción: la
    ejecución termina normal y decide con un tercio de la evidencia.
    """

    model_config = ConfigDict(extra="forbid")

    snapshot: MarketSnapshot | None = None
    indicators: IndicatorSet | None = None
    activation: ActivationCheck | None = None
    evidence: TechnicalEvidence | None = None
    decision: Decision | None = None
    risk: RiskVerdict | None = None

    verdicts: Annotated[list[TechnicalVerdict], operator.add] = Field(default_factory=list)
    briefs: Annotated[list[DebateBrief], operator.add] = Field(default_factory=list)
    calls: Annotated[list[LLMCall], operator.add] = Field(default_factory=list)
    errors: Annotated[list[NodeError], operator.add] = Field(default_factory=list)

    @property
    def quota_used(self) -> float:
        """Cuota consumida. Los cache hits no cuentan: no llegaron al proveedor."""
        return sum(call.quota_weight for call in self.calls if not call.cache_hit)

    def brief(self, side: Side) -> DebateBrief | None:
        """Alegato de una mesa, o None si esa mesa todavía no escribió."""
        return next((brief for brief in self.briefs if brief.side is side), None)


# ────────────────────────────────────── Validación cruzada ────────────────────────────────────────
# Pydantic no puede verificar esto dentro de un solo modelo: son relaciones entre
# objetos que se construyen en nodos distintos. Lista vacía = el modelo no inventó nada.


def unknown_indicators(verdict: TechnicalVerdict, indicators: IndicatorSet) -> list[str]:
    """Indicadores citados por un veredicto que no existen en el `IndicatorSet`."""
    known = indicators.names
    return sorted(
        {
            name
            for observation in verdict.observations
            for name in observation.cites
            if name not in known
        }
    )


def ungrounded_claim_refs(brief: DebateBrief, verdicts: Sequence[TechnicalVerdict]) -> list[str]:
    """Ids citados por una mesa que no salieron de ningún veredicto técnico."""
    known = {observation.id for verdict in verdicts for observation in verdict.observations}
    return sorted({ref for claim in brief.claims for ref in claim.grounded_in if ref not in known})
