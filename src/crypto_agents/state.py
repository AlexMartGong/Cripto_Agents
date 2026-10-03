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
    "TOKEN_FIELDS",
    "Action",
    "ActivationCheck",
    "AgentRole",
    "Backend",
    "Bias",
    "Billing",
    "Claim",
    "DebateBrief",
    "Decision",
    "Dimension",
    "ExecutionMode",
    "FailureKind",
    "FrozenModel",
    "IndicatorSet",
    "LLMCall",
    "LLMOutput",
    "MarketSnapshot",
    "NodeError",
    "Observation",
    "OrderIntent",
    "OrderReceipt",
    "Proposal",
    "RiskVerdict",
    "Side",
    "Strength",
    "StructuredOutputMode",
    "TechnicalEvidence",
    "TechnicalVerdict",
    "TradingState",
    "stop_on_wrong_side",
    "ungrounded_claim_refs",
    "unknown_indicators",
]

_DIGEST_PATTERN = r"^[0-9a-f]{64}$"

TOKEN_FIELDS = ("prompt_tokens", "cached_tokens", "completion_tokens")
"""Los contadores de uso de un `LLMCall`. Nombrados aquí para que el digest de una corrida
los omita cuando valen `None` sin repetir la lista."""


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


class Backend(StrEnum):
    """Proveedor que atendió una llamada.

    Vive en el contrato y no en la configuración porque `LLMCall` lo registra:
    `state.py` no importa nada del paquete, así que un enum de configuración aquí
    cerraría el ciclo. La configuración lo reexporta para quien lo declare.
    """

    OPENAI = "openai"
    OLLAMA = "ollama"


class Billing(StrEnum):
    """Cómo se paga el proveedor: suscripción con pool de dólares, o por uso.

    Vive en el contrato por lo mismo que `Backend`: la corrida lo deja escrito en su
    `meta.json`, y `audit.py` lo lee sin poder importar la configuración entera.
    """

    GO = "go"
    """Suscripción OpenCode Go: un pool de dólares que cada modelo consume con su propio peso."""

    PAYG = "payg"
    """Pago por uso: cada llamada cuesta sus dólares y no hay pool que agotar."""


class StructuredOutputMode(StrEnum):
    """Cómo se le pide a un modelo que se ajuste a un esquema.

    Era una constante implícita de LangChain —`function_calling`, su valor por
    defecto— y eso la volvía invisible: el mismo esquema contra el mismo modelo
    daba respuesta vacía o respuesta buena según un parámetro que nadie había
    escrito en ninguna parte. Es un dato del modelo, no del framework, porque
    cada uno soporta un subconjunto distinto y no hay forma de saber cuál sin
    preguntárselo.

    Vive en el contrato por lo mismo que `Backend`: `LLMCall` lo registra.
    """

    JSON_SCHEMA = "json_schema"
    """El esquema viaja como `response_format`. La respuesta llega en `content`."""

    JSON_MODE = "json_mode"
    """Solo se exige JSON válido, sin forma. El esquema lo impone la validación."""

    FUNCTION_CALLING = "function_calling"
    """El esquema viaja como herramienta. La respuesta llega en `tool_calls`, no en `content`."""


_DIMENSION_ALTERNATION = "|".join(dimension.value for dimension in Dimension)
OBSERVATION_ID_PATTERN = rf"^({_DIMENSION_ALTERNATION})-[0-9]+$"
"""Patrón `<dimension>-<n>`: el prefijo hace los ids citables sin colisión entre agentes.

Escrito en el subconjunto de regex que Ollama sabe compilar. Este patrón viaja en
el JSON Schema con el que se restringe la generación local, y el compilador de
gramáticas de Ollama 0.18 muere con SIGSEGV —se lleva el servidor entero, no solo
la petición— ante dos construcciones: el grupo no capturante `(?:...)` y la clase
abreviada `\\d`. De ahí el grupo capturante y `[0-9]` explícito. Para la
validación en Python las tres formas son equivalentes, así que evitarlas no
cuesta nada; descubrirlo en producción, sí.
"""


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


class Proposal(LLMOutput):
    """Lo que declara cualquier decisor, haya debatido o no.

    Existe porque la ablación necesita medir ramas sin mesas de debate. `Decision`
    exige nombrar la mesa descartada, y con razón —actuar sin decir qué argumento
    se rechaza es no haber leído la contraparte—, pero en una variante que no tiene
    mesas ese mismo validador haría toda compra invalidable y la rama sería
    inmedible. Se separa en vez de ablandar `Decision`: el contrato del pipeline
    que opera de verdad no se relaja para que quepa un experimento.

    El gate de riesgo y la construcción de la orden leen de aquí, así que todas las
    variantes recorren exactamente el mismo camino hasta el mercado. Si cada rama
    tuviera el suyo, la comparación mediría el arnés en vez de los modelos.
    """

    action: Action
    confidence: float = Field(ge=0.0, le=1.0)
    size_fraction: float = Field(ge=0.0, le=1.0)
    invalidation_price: PositiveFloat | None = None
    rationale: str = Field(min_length=20)

    @model_validator(mode="after")
    def _actionable_proposals_are_complete(self) -> Self:
        """Actuar sin invalidación o sin tamaño no es una decisión."""
        if self.action is Action.HOLD:
            return self
        missing: list[str] = []
        if self.invalidation_price is None:
            missing.append("invalidation_price")
        if self.size_fraction <= 0.0:
            missing.append("size_fraction > 0")
        if missing:
            raise ValueError(f"la acción {self.action.value} exige: {', '.join(missing)}")
        return self


class Decision(Proposal):
    """Salida del decisor del pipeline completo: además, qué mesa descarta."""

    dismissed_side: Side | None = None
    dismissal_reason: str | None = None

    @model_validator(mode="after")
    def _actionable_decisions_name_the_dismissed_desk(self) -> Self:
        """Habiendo mesas, actuar sin descartar una explícitamente no es decidir."""
        if self.action is Action.HOLD:
            return self
        missing: list[str] = []
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
    veto_rule: str | None = None
    """Nombre estable de la regla que vetó, para poder agrupar.

    Va aparte de `veto_reason` porque esa es una frase con porcentajes y minutos
    dentro: agrupar por ella convertiría cada veto de drawdown en una categoría
    propia y la métrica no contaría nada.
    """

    veto_reason: str | None = None

    @model_validator(mode="after")
    def _veto_is_recorded_and_flat(self) -> Self:
        """Un veto sin causa no es auditable; un veto con tamaño no es un veto."""
        if self.approved:
            return self
        if not self.veto_reason:
            raise ValueError("un veto exige veto_reason")
        if not self.veto_rule:
            raise ValueError("un veto exige veto_rule")
        if self.final_size_fraction != 0.0:
            raise ValueError("un veto exige final_size_fraction == 0")
        return self


# ───────────────────────────────────────────── Contabilidad ───────────────────────────────────────


class ExecutionMode(StrEnum):
    """Destino de las órdenes. El valor por defecto no manda nada al mercado."""

    PAPER = "paper"
    LIVE = "live"


def stop_on_wrong_side(action: Action, invalidation_price: float, reference_price: float) -> bool:
    """Si la invalidación queda del lado que no invalida nada.

    Un largo se rompe hacia abajo y un corto hacia arriba: en un `buy` la
    invalidación tiene que estar por debajo del precio de referencia, y en un
    `sell` por encima. La igualdad cuenta como lado equivocado —un stop en el
    propio precio no deja recorrido—, y un `hold` no tiene stop que mirar.

    Es la única definición del paquete. El veto del gate de riesgo, el validador
    de `OrderIntent`, la puntuación de resultados y el recuento de la auditoría
    leen de aquí: con cuatro comparaciones escritas por separado, bastaría que una
    tratase la igualdad distinto para que el gate aprobara lo que la orden rechaza.
    """
    if action is Action.BUY:
        return invalidation_price >= reference_price
    if action is Action.SELL:
        return invalidation_price <= reference_price
    return False


class OrderIntent(FrozenModel):
    """Orden a enviar. Su tamaño ya pasó por el gate de riesgo.

    Vive en el contrato y no en `execution.py` porque viaja en el estado entre el
    nodo que ejecuta y el que registra.
    """

    symbol: str = Field(min_length=1)
    side: Action
    size_fraction: float = Field(gt=0.0, le=1.0)
    reference_price: PositiveFloat
    invalidation_price: PositiveFloat
    mode: ExecutionMode

    @model_validator(mode="after")
    def _side_is_directional(self) -> Self:
        """`hold` no es un lado: no genera orden."""
        if self.side is Action.HOLD:
            raise ValueError("una orden no puede tener lado 'hold'")
        return self

    @model_validator(mode="after")
    def _stop_is_on_the_side_that_invalidates(self) -> Self:
        """Una orden con el stop del lado equivocado no se puede construir.

        El veto `invalid_stop_side` del gate de riesgo es la vía normal y deja
        registro. Esto es el respaldo: si algún camino llegara aquí sin pasar por
        el gate, la orden no existe, en lugar de existir con un stop que el
        mercado ya había cruzado antes de enviarla.
        """
        if stop_on_wrong_side(self.side, self.invalidation_price, self.reference_price):
            raise ValueError(
                f"invalidación en el lado equivocado: un {self.side.value} con referencia "
                f"{self.reference_price} no se invalida en {self.invalidation_price}"
            )
        return self


class OrderReceipt(FrozenModel):
    """Resultado de enviar una orden."""

    order: OrderIntent
    accepted: bool
    reference: str = Field(min_length=1)
    """Identificador del exchange, o una marca sintética en modo papel."""


class FailureKind(StrEnum):
    """En qué punto se rompió un intento. Cerrado: cuatro causas y ninguna más.

    La distinción no es cosmética. Las dos primeras miden al modelo y las dos
    últimas al proveedor, y contarlas juntas es lo que convierte una tabla de
    comparación entre modelos en algo que no se puede leer: un 400 del gateway y
    un JSON que no valida se parecen en el journal y no se parecen en nada más.
    """

    SCHEMA = "schema"
    """Hubo contenido y no pasó el esquema: falta un campo, sobra otro, un tipo no cuadra."""

    CONTEXT = "context"
    """Pasó el esquema y contradice lo que tenía delante.

    Un alegato que cita un id de observación que ningún veredicto emitió, un
    veredicto que cita un indicador que no existe, una mesa que responde como la
    contraria. Ningún JSON Schema lo expresa, porque depende de la evaluación y no
    de la forma de la respuesta. Es tan fallo del modelo como el de esquema, y se
    trata igual: el intento se registra, consume cuota y se reintenta con el error.
    """

    TIMEOUT = "timeout"
    """El proveedor no contestó dentro del plazo declarado.

    Aparte del transporte porque no es la misma noticia: un rechazo dice que el
    proveedor no quiere o no puede servir ese id; un timeout, que tardó más de lo
    que se le concedió, y eso se arregla en otro sitio.
    """

    TRANSPORT = "transport"
    """La petición no llegó a producir contenido: red, autenticación, 4xx, 5xx."""


_LEGACY_FAILURE_KINDS = {"validation": FailureKind.SCHEMA.value}
"""Nombres que escribieron versiones anteriores, y a qué equivalen hoy.

`validation` era el único fallo de contenido que existía como fila de llamada, y
era siempre de esquema: los de contexto se registraban como `NodeError` del nodo,
no como intento. `transport` se queda como está, aunque entonces incluía los
timeouts: ya no hay con qué distinguirlos.
"""


class LLMCall(FrozenModel):
    """Registro de una llamada a modelo. Base del presupuesto y del replay.

    `at` es lo que permite reconstruir una ventana deslizante desde el estado:
    sin marca de tiempo el contador solo existe en memoria y el replay no puede
    recalcular cuánta cuota había consumida en cada punto de la ejecución.
    """

    role: AgentRole
    backend: Backend
    """Quién atendió el intento.

    Sin esta columna el journal mezcla veredictos de un modelo remoto grande con
    los de un respaldo local pequeño y no hay forma de separarlos después: dos
    decisiones de calidades distintas quedan indistinguibles en la misma línea.
    """

    model: str = Field(min_length=1)
    structured_output: StructuredOutputMode
    """Con qué modo se le pidió el esquema.

    Mismo argumento que `backend`: sin esta columna, dos corridas del mismo
    modelo en modos distintos producen líneas idénticas, y la comparación que
    decide cuál usar deja de poder hacerse sobre el journal.
    """

    quota_weight: float = Field(gt=0.0)
    prompt_digest: str = Field(pattern=_DIGEST_PATTERN)
    cache_hit: bool = False
    valid: bool
    """Si la salida del intento pasó la validación: el esquema y, si la hay, la de contexto.

    Un intento inválido ya se pagó en el proveedor, así que se registra igual. Sin
    distinguirlo del bueno, un modelo local que necesita tres intentos por
    veredicto parece tan barato como uno que acierta a la primera.
    """

    failure_kind: FailureKind | None = None
    """Por qué falló el intento. `None` si y solo si `valid`."""

    failure_message: str | None = None
    """Qué dijo el validador o el proveedor. Obligatorio cuando hay `failure_kind`.

    Un intento que falla sin causa registrada obliga a reconstruirla desde los
    logs del proveedor, que es justo lo que no existe cuando alguien pregunta
    tres días después por qué el sistema no operó.

    Son dos campos planos y no un objeto anidado para que el journal se pueda
    filtrar por tipo de fallo sin parsear nada: `grep '"failure_kind":"context"'`.
    El estado a medias que el objeto impedía —tipo sin mensaje, mensaje sin tipo—
    lo impide el validador.
    """

    latency_ms: float = Field(ge=0.0)
    at: AwareDatetime

    prompt_tokens: int | None = Field(default=None, ge=0)
    """Tokens de entrada que contó el proveedor, **incluidos los cacheados**.

    Es la convención de OpenAI y la que midió la pasarela: la segunda de dos llamadas
    idénticas repite este número con una parte en `cached_tokens`. `None` es que el
    proveedor no lo informó, y nunca se rellena con una estimación: un conteo propio
    del prompt sería exactamente el número que este campo existe para no inventar.
    """

    cached_tokens: int | None = Field(default=None, ge=0)
    """De los `prompt_tokens`, los que el proveedor sirvió desde su caché de prefijo.

    `None` no es `0`. DeepSeek V4 Flash devuelve `null` en una llamada en frío y la
    cifra en una en caliente, mientras que los demás modelos devuelven `0`.
    """

    completion_tokens: int | None = Field(default=None, ge=0)
    """Tokens de salida, **razonamiento incluido**.

    Kimi K2.6 gastó 519 para un `ok=true`, 512 de ellos razonando. Se cobran como salida.
    """

    @model_validator(mode="before")
    @classmethod
    def _accept_the_previous_failure_shape(cls, data: object) -> object:
        """Lee los registros escritos antes de que el fallo fuera dos campos planos.

        Entonces era `failure: {kind, message} | null`. Un journal es un archivo
        que se relee meses después: si el contrato de hoy no supiera cargar las
        líneas de ayer, cada cambio de esquema dejaría la historia ilegible, y el
        runner se niega a arrancar con un journal que no puede leer.
        """
        if not isinstance(data, dict) or "failure" not in data:
            return data
        converted = {key: value for key, value in data.items() if key != "failure"}
        legacy = data["failure"]
        if isinstance(legacy, dict):
            kind = legacy.get("kind")
            if isinstance(kind, str):
                kind = _LEGACY_FAILURE_KINDS.get(kind, kind)
            converted.setdefault("failure_kind", kind)
            converted.setdefault("failure_message", legacy.get("message"))
        elif legacy is not None:
            return data
        return converted

    @model_validator(mode="after")
    def _failure_matches_validity(self) -> Self:
        """`valid` si y solo si no hay tipo de fallo; y un fallo siempre dice qué pasó."""
        if self.valid and self.failure_kind is not None:
            raise ValueError("un intento válido no puede llevar causa de fallo")
        if not self.valid and self.failure_kind is None:
            raise ValueError("un intento inválido debe registrar su causa")
        if self.failure_kind is not None and not self.failure_message:
            raise ValueError("un fallo exige failure_message")
        if self.failure_kind is None and self.failure_message is not None:
            raise ValueError("un failure_message sin failure_kind no describe ningún fallo")
        return self


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
    proposal: Proposal | None = None
    """Decisión de una variante sin mesas. Nunca conviven las dos.

    Campo aparte y no una unión: un `Proposal` guardado bajo `decision` se releería
    como una `Decision` a la que simplemente le faltan los campos de descarte, y en
    el journal no habría forma de distinguir «no hubo mesas» de «el decisor no dijo
    a quién descartaba».
    """

    risk: RiskVerdict | None = None
    order: OrderIntent | None = None

    verdicts: Annotated[list[TechnicalVerdict], operator.add] = Field(default_factory=list)
    briefs: Annotated[list[DebateBrief], operator.add] = Field(default_factory=list)
    calls: Annotated[list[LLMCall], operator.add] = Field(default_factory=list)
    errors: Annotated[list[NodeError], operator.add] = Field(default_factory=list)

    @property
    def quota_used(self) -> float:
        """Cuota consumida. Los cache hits no cuentan: no llegaron al proveedor."""
        return sum(call.quota_weight for call in self.calls if not call.cache_hit)

    @property
    def proposed(self) -> Proposal | None:
        """Lo que se decidió, venga de un pipeline con mesas o sin ellas.

        El gate de riesgo lee de aquí, así que su código es el mismo para todas las
        variantes de ablación.
        """
        if self.decision is not None:
            return self.decision
        return self.proposal

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
