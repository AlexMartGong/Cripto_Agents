"""Acceso a modelos: protocolo, adaptadores concretos y router.

Ningún nodo importa un cliente concreto. Los nodos piden `ModelRouter.invoke(role,
prompt, schema)` y el router resuelve qué modelo cabe en presupuesto, consulta la
caché, llama al backend, valida la salida y devuelve el objeto junto al registro
de cada intento. Cambiar de OpenAI a Ollama es cambiar configuración, no código.

El backend devuelve **texto JSON**, no un objeto ya validado. La validación es
responsabilidad del router porque es quien puede reintentar: un esquema como
`TechnicalVerdict` lleva validadores propios (dimensión de las observaciones, ids
únicos) que ningún JSON Schema expresa, así que la salida puede ser
estructuralmente correcta y aun así inválida.

El reintento adjunta el error al prompt. Reenviar el mismo texto daría el mismo
digest, la caché devolvería la misma respuesta inválida y el bucle no avanzaría.

La caché guarda cada intento que produjo contenido, válido o no, cada uno bajo el
digest de su prompt. Un intento inválido leído de la caché se valida otra vez, da
el mismo error y por tanto el mismo prompt de reintento: la conversación entera
se reproduce sin llamar a nadie, que es lo que hace que un replay solo-caché
complete una evaluación que necesitó reintentos.

Todo lo que salga del adaptador se envuelve aquí en `ModelCallError`, con los
intentos ya pagados dentro. El router es el único módulo que habla con un
proveedor, así que también es el único sitio donde se pueden convertir las
excepciones de cada cliente —httpx, openai, ollama— en un tipo que un nodo pueda
capturar sin importar ninguno de los tres.

Cada intento recoge además lo que el proveedor contó: tokens de entrada, cacheados y de salida.
Es una medida, no una estimación: si el proveedor no la devuelve queda `None`, y los adaptadores
leen el `usage` crudo de la respuesta porque el resumen de LangChain convierte un contador
ausente en cero. El router no sabe de precios; eso es de `consumption.py`.

Y quién respondió aguas arriba: la pasarela lo declara en dos cabeceras de la respuesta, y el id
que se pidió no lo dice. Se leen aquí y en ningún otro módulo, de la respuesta de **esa** llamada
—viajan en su mensaje, no en un hook compartido—, y solo esas dos: ninguna otra cabecera sale del
adaptador.

La degradación a un modelo local no vive aquí: es `QuotaLedger.resolve()` quien
elige, y lo hace en cada intento, así que un veredicto puede empezar remoto y
terminar local. Por eso cada `LLMCall` registra su propio `backend` y su propio
`valid`: sin esas dos columnas, un respaldo local que necesita tres intentos por
veredicto queda en el journal indistinguible de un remoto que acierta a la
primera, y la comparación entre ambos deja de ser posible.

Con pago por uso esa degradación no ocurre en un rol remoto: el proveedor no publica
cuota, así que el router no le pregunta al contador (`_choose`) y lo que frena es un
tope en dólares, `SpendGuard`, que se mira antes de abrir cada invocación (`_admit`).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from collections.abc import Mapping
from typing import TYPE_CHECKING, NamedTuple, Protocol

from pydantic import Field, ValidationError

from crypto_agents.cache import CacheEntry, cache_key
from crypto_agents.quota import LOCAL_BACKENDS
from crypto_agents.state import (
    Backend,
    Billing,
    FailureKind,
    FrozenModel,
    LLMCall,
    LLMOutput,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from crypto_agents.cache import ResponseCache
    from crypto_agents.quota import Clock, QuotaLedger
    from crypto_agents.settings import ModelChoice, Settings
    from crypto_agents.spend import SpendGuard
    from crypto_agents.state import AgentRole

__all__ = [
    "SESSION_HEADER",
    "UPSTREAM_ENDPOINT_HEADER",
    "UPSTREAM_MODEL_HEADER",
    "BackendNotCalledError",
    "ChatBackend",
    "Completion",
    "ContextCheck",
    "InvalidModelOutputError",
    "ModelCallError",
    "ModelCatalog",
    "ModelInvocationError",
    "ModelRouter",
    "OllamaBackend",
    "OpenAIBackend",
    "ResidentModel",
    "SpendCapReachedError",
    "TokenUsage",
    "Upstream",
    "as_completion",
    "build_backends",
    "content_from_rejected_parse",
    "json_payload",
    "prompt_digest",
    "raw_text",
    "structured_runnable",
    "upstream_from_headers",
    "upstream_from_message",
    "usage_from_message",
    "usage_from_ollama",
    "usage_from_provider",
    "usage_from_rejected_parse",
]

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
"""Bloque de razonamiento de un modelo que piensa en voz alta antes de responder."""

_CODE_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
"""Valla de código markdown alrededor del JSON."""

_RETRY_TEMPLATE = (
    "\n\n---\n"
    "Tu respuesta anterior no pasó la validación del esquema.\n"
    "Error: {error}\n"
    "Corrige exactamente eso y responde de nuevo, solo con JSON válido."
)


type ContextCheck[T: LLMOutput] = Callable[[T], list[str]]
"""Validación que el esquema no puede expresar. Devuelve los errores, o lista vacía.

Una función pura sobre la salida ya validada contra su esquema, que el nodo
construye con lo que tiene delante: qué ids de observación existen, qué
indicadores hay, qué mesa o qué dimensión se pidió. Es pura a propósito —el
router la ejecuta en cada intento y otra vez al leer la caché, y tiene que decir
lo mismo las dos veces.

Vive en la firma de `invoke()` y no dentro del nodo porque solo el router puede
hacer algo con un fallo: registrarlo como intento, cobrarlo y reintentar con el
error adjunto. Comprobado después, en el nodo, un alegato que cita un id
inventado abortaba la evaluación con un `LLMCall` marcado como válido y la
respuesta ya guardada en caché.
"""


class _Rejected(NamedTuple):
    """Una salida que no pasó, con el tipo de fallo y lo que se le dirá al modelo."""

    kind: FailureKind
    message: str


class _Replayed[T: LLMOutput](NamedTuple):
    """Un intento leído de la caché: cómo se juzga hoy y quién lo había contestado."""

    judged: T | _Rejected
    upstream: Upstream | None


class BackendNotCalledError(RuntimeError):
    """El adaptador se negó a llamar; ningún proveedor vio la petición.

    El router envuelve todo lo que salga de un backend, pero esto no es un fallo
    del proveedor: es el arnés diciendo que pare. `CacheOnlyBackend` la usa para
    que un hueco de caché en un replay siga nombrando modelo, esquema y prompt en
    lugar de disfrazarse de error de transporte, que mandaría a revisar la red
    cuando lo que falta es rellenar la caché. No se registra `LLMCall` porque no
    hubo llamada que registrar.
    """


class ModelInvocationError(RuntimeError):
    """Fallo de invocación que deja atrás intentos ya pagados.

    Los intentos viajan dentro de la excepción porque quien la captura —el nodo—
    es quien escribe en el estado. Sin ellos las filas se quedan en el contador
    de cuota, que vive en memoria y muere con el proceso, y no llegan al journal:
    la regla 4 se cumpliría solo mientras el sistema siguiera corriendo, que es
    exactamente cuando nadie necesita consultarla.
    """

    def __init__(self, message: str, calls: tuple[LLMCall, ...] = ()) -> None:
        self.calls = calls
        super().__init__(message)


class InvalidModelOutputError(ModelInvocationError):
    """El modelo agotó los intentos sin producir una salida que valide."""

    def __init__(
        self, role: AgentRole, attempts: int, last_error: str, calls: tuple[LLMCall, ...] = ()
    ) -> None:
        self.role = role
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"{role.value}: {attempts} intento(s) sin salida válida; último error: {last_error}",
            calls,
        )


class ModelCallError(ModelInvocationError):
    """El proveedor no llegó a devolver contenido: red, autenticación o rechazo.

    Existe para que un fallo de transporte tenga un tipo que un nodo pueda
    capturar. Antes salía crudo del adaptador, se llevaba el grafo por delante y
    la evaluación no dejaba ni la vela que estaba mirando: el journal quedaba con
    una línea firmada por el runner y sin contexto, que es justo lo que hace
    falta para saber si el fallo fue del mercado, del gate o del proveedor.
    """

    def __init__(
        self,
        role: AgentRole,
        choice: ModelChoice,
        cause: Exception,
        calls: tuple[LLMCall, ...] = (),
    ) -> None:
        self.role = role
        self.model = choice.model
        self.backend = choice.backend
        self.cause = cause
        super().__init__(
            f"{role.value}: {choice.model} ({choice.backend.value}) falló antes de producir "
            f"contenido: {type(cause).__name__}: {cause}",
            calls,
        )


class SpendCapReachedError(ModelInvocationError):
    """El tope de gasto no dejó abrir la invocación: ningún proveedor vio esta petición.

    Hereda de `ModelInvocationError` por lo que trae, no por lo que pagó: una invocación puede
    haber reproducido de la caché un intento inválido antes de necesitar la primera llamada viva, y
    esas filas tienen que llegar al journal igual que las de cualquier otro fallo. Lo que nunca
    trae es un intento pagado: el tope se mira antes del primer intento vivo y no después.

    Como `QuotaExhaustedError`, dice que no había con qué; a diferencia de ella, el límite es en
    dólares y lo declaró quien lanzó la corrida, no el proveedor.
    """

    def __init__(
        self,
        role: AgentRole,
        cap_usd: float,
        spent_usd: float,
        calls: tuple[LLMCall, ...] = (),
    ) -> None:
        self.role = role
        self.cap_usd = cap_usd
        self.spent_usd = spent_usd
        super().__init__(
            f"{role.value}: tope de gasto alcanzado, {spent_usd:.4f} USD de {cap_usd:.4f}; "
            "no se abrió la invocación",
            calls,
        )


def prompt_digest(prompt: str) -> str:
    """Huella del prompt para caché y replay."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def json_payload(text: str) -> str:
    """Quita el envoltorio con que un modelo rodea su JSON, si lo hay.

    Dos envoltorios medidos sobre los seis modelos del gateway:

    - **Razonamiento en voz alta.** minimax-m3 antepone un bloque
      `<think>…</think>` de miles de caracteres y luego emite el veredicto. En los
      tres modos. El JSON es correcto; lo que llega al validador son 4215
      caracteres que empiezan por `<`, y pydantic muere en la columna 1.
    - **Valla de código.** mimo-v2.5 en `function_calling` envuelve la respuesta
      en ```` ```json ````. Mismo efecto.

    Es el defecto de los `tool_calls` otra vez: la salida del modelo existe y el
    adaptador no la encuentra. Y como allí, el reintento no ayuda — le adjunta un
    error que no describe nada de lo que el modelo hizo, así que vuelve a razonar
    en voz alta y se paga otra llamada por el mismo resultado.

    Tres cautelas, porque esto toca lo que el validador va a ver:

    1. Si el texto ya es JSON, se devuelve intacto. Un JSON cuyo contenido lleve
       acentos graves no puede acabar destrozado por una expresión regular.
    2. Solo se desenvuelve si el resultado *parsea*. Si no, se devuelve el
       original, para que el error del reintento siga describiendo lo que el
       modelo emitió de verdad.
    3. No se valida contra el esquema aquí. Eso es del router, que es quien
       reintenta; esto solo decide dónde empieza y acaba el JSON.
    """
    stripped = text.strip()
    if not stripped or _parses(stripped):
        return stripped

    without_reasoning = _THINK_BLOCK.sub("", stripped).strip()
    fenced = _CODE_FENCE.search(without_reasoning)
    candidate = fenced.group(1).strip() if fenced else without_reasoning
    return candidate if _parses(candidate) else stripped


def _parses(text: str) -> bool:
    """Si el texto es JSON sintácticamente válido. No dice nada del esquema."""
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


def raw_text(result: object) -> str:
    """Texto tal como lo emitió el modelo, sin validar, salga por donde salga.

    Depende del modo: con `function_calling` —el valor por defecto de
    `with_structured_output`, y el que estuvo operando sin que nadie lo hubiera
    escrito— el mensaje llega con `content` vacío y lo que el modelo dijo está en
    los argumentos de la llamada a herramienta. Leer solo `content` devuelve
    cadena vacía, así que el router reintenta adjuntando un error que enumera
    todos los campos como ausentes en vez del fallo real — y el modelo recibe una
    corrección que no describe nada de lo que hizo. Con `json_schema` y
    `json_mode` la salida sí viene en `content`.

    El orden va de más crudo a más procesado: contenido, argumentos de la
    herramienta, argumentos que ni siquiera eran JSON, y en último lugar el
    objeto ya validado por LangChain, que solo existe cuando no hubo problema.
    """
    if not isinstance(result, dict):
        return ""

    raw = result.get("raw")
    content = getattr(raw, "content", None)
    if isinstance(content, str) and content.strip():
        return json_payload(content)

    for call in getattr(raw, "tool_calls", None) or ():
        arguments = call.get("args") if isinstance(call, dict) else None
        if arguments:
            return json.dumps(arguments, ensure_ascii=False, default=str)

    for call in getattr(raw, "invalid_tool_calls", None) or ():
        arguments = call.get("args") if isinstance(call, dict) else None
        if isinstance(arguments, str) and arguments.strip():
            return json_payload(arguments)

    parsed = result.get("parsed")
    if isinstance(parsed, LLMOutput):
        return parsed.model_dump_json()
    return ""


def _raw_from_validation_error(error: ValidationError) -> str:
    """Rescata del propio error lo que el modelo había emitido, cuando no hay otra copia.

    Pydantic guarda en `input` el valor que rechazó. Se prefiere el error más
    externo —el de `loc` más corto— porque su `input` es lo más parecido a la
    respuesta entera.

    **Solo lo es cuando el fallo está en la raíz.** Con el fallo en un campo
    anidado, el `input` del error más externo es ese campo o el objeto que lo
    contiene: un fragmento. El router lo validaba otra vez, fallaba por otra cosa
    —`Invalid JSON`, o campos de la raíz «ausentes»— y el reintento le describía
    al modelo un error que no había cometido. Por eso esto es ya el último recurso:
    la respuesta entera está en el cuerpo de la respuesta HTTP
    (`content_from_rejected_parse`), y aquí solo se llega si ese cuerpo no se puede
    leer.
    """
    for detail in sorted(error.errors(), key=lambda item: len(item["loc"])):
        candidate = detail.get("input")
        if isinstance(candidate, str) and candidate.strip():
            return json_payload(candidate)
        if isinstance(candidate, dict) and candidate:
            return json.dumps(candidate, ensure_ascii=False, default=str)
    return ""


def structured_runnable(client: object, schema: type[LLMOutput], choice: ModelChoice) -> object:
    """Enlaza el esquema al cliente con el modo declarado para ese modelo.

    Es una función y no una línea dentro de `complete()` para que el modo se
    pueda comprobar sin red: qué método pide el adaptador es justo el dato que
    antes no estaba escrito en ninguna parte.
    """
    bind = client.with_structured_output  # type: ignore[attr-defined]
    return bind(schema, method=choice.structured_output.value, include_raw=True)


class TokenUsage(FrozenModel):
    """Lo que un proveedor contó de una llamada. `None` es que no lo informó, nunca `0`."""

    prompt_tokens: int | None = Field(default=None, ge=0)
    """Entrada total, cacheados incluidos."""

    cached_tokens: int | None = Field(default=None, ge=0)
    """Parte de la entrada servida desde la caché de prefijo del proveedor."""

    completion_tokens: int | None = Field(default=None, ge=0)
    """Salida, razonamiento incluido."""


UPSTREAM_MODEL_HEADER = "x-opencode-upstream-model-id"
"""Cabecera de respuesta con el modelo que la pasarela usó aguas arriba para contestar."""

UPSTREAM_ENDPOINT_HEADER = "x-opencode-endpoint-id"
"""Cabecera de respuesta con el proveedor o la ruta que la pasarela usó."""


class Upstream(FrozenModel):
    """Quién contestó según la pasarela. Dos nombres copiados de la respuesta, sin interpretar."""

    model: str | None = Field(default=None, min_length=1)
    endpoint: str | None = Field(default=None, min_length=1)


class Completion(FrozenModel):
    """Texto JSON sin validar, el uso que el proveedor declaró y quién dijo la pasarela que fue.

    De las cabeceras de la respuesta salen dos nombres y nada más: el diccionario entero no cruza
    del adaptador al router, así que no hay campo por el que una cookie o una credencial puedan
    llegar a un `LLMCall`.
    """

    text: str
    usage: TokenUsage | None = None
    upstream: Upstream | None = None


def as_completion(result: str | Completion) -> Completion:
    """Un backend que solo sabe devolver texto devuelve una `Completion` sin uso.

    Es lo que permite que `ChatBackend.complete()` crezca sin tocar a quien solo
    tiene texto que dar —los falsos de las pruebas, el backend de solo-caché—: no
    declaran uso porque no lo conocen, y eso es `None`, no una estimación.
    """
    return result if isinstance(result, Completion) else Completion(text=result)


def _count(value: object) -> int | None:
    """Un contador del proveedor: un entero natural, o `None` si es otra cosa."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def usage_from_provider(usage: object) -> TokenUsage | None:
    """El `usage` crudo de una respuesta OpenAI-compatible, tal como llegó.

    `prompt_tokens_details.cached_tokens` puede venir como `0`, como una cifra o como
    `null` —DeepSeek V4 Flash lo devuelve `null` en frío—, y los tres se conservan:
    `null` es «no informado» y no se convierte en cero. Si no hay ningún contador, no
    hay uso que devolver.
    """
    if not isinstance(usage, Mapping):
        return None
    details = usage.get("prompt_tokens_details")
    result = TokenUsage(
        prompt_tokens=_count(usage.get("prompt_tokens")),
        cached_tokens=_count(details.get("cached_tokens"))
        if isinstance(details, Mapping)
        else None,
        completion_tokens=_count(usage.get("completion_tokens")),
    )
    return None if result == TokenUsage() else result


def usage_from_message(result: object) -> TokenUsage | None:
    """El uso de un mensaje de LangChain, leído de `response_metadata["token_usage"]`.

    No de `usage_metadata`: LangChain lo construye con `prompt_tokens or 0`, así que
    un proveedor que no informó nada produciría ceros que parecen una medición.
    `response_metadata["token_usage"]` es el `usage` del proveedor sin tocar.
    """
    metadata = getattr(result, "response_metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    return usage_from_provider(metadata.get("token_usage"))


def _header(headers: Mapping[str, object], name: str) -> str | None:
    """El valor de una cabecera, o `None` si falta, está vacío o no es texto."""
    value = headers.get(name)
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def upstream_from_headers(headers: object) -> Upstream | None:
    """Las dos cabeceras de upstream de una respuesta, o `None` si no trae ninguna.

    Recibe todas las cabeceras y devuelve dos nombres: es el único punto por el que pasan, y lo
    demás —cookies, identificadores de petición, lo que el proveedor añada mañana— se queda aquí.
    Los nombres se comparan sin distinguir mayúsculas, como manda HTTP. Que falten no es un error:
    un proveedor que no las manda deja el intento sin upstream, igual que sin contadores.
    """
    if not isinstance(headers, Mapping):
        return None
    lowered = {key.lower(): value for key, value in headers.items() if isinstance(key, str)}
    result = Upstream(
        model=_header(lowered, UPSTREAM_MODEL_HEADER),
        endpoint=_header(lowered, UPSTREAM_ENDPOINT_HEADER),
    )
    return None if result == Upstream() else result


def upstream_from_message(result: object) -> Upstream | None:
    """El upstream de un mensaje de LangChain, leído de `response_metadata["headers"]`.

    Con `include_response_headers=True`, `ChatOpenAI` deja ahí las cabeceras de la respuesta que
    produjo ese mensaje. Es lo que hace la lectura correlacionada: cada llamada trae las suyas, sin
    estado compartido entre llamadas concurrentes.
    """
    metadata = getattr(result, "response_metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    return upstream_from_headers(metadata.get("headers"))


def upstream_from_rejected_parse(error: BaseException) -> Upstream | None:
    """El upstream de una respuesta que el SDK no pudo convertir en el esquema.

    Con `json_schema` el SDK de OpenAI valida el contenido contra la clase dentro de `parse()`. Si
    no pasa, lanza y el mensaje no llega a existir; LangChain cuelga la respuesta HTTP de la propia
    excepción antes de relanzarla. Es la respuesta de esa llamada y de ninguna otra, así que el
    intento inválido —justo el que se quiere poder atribuir— sigue diciendo quién lo contestó.
    """
    return upstream_from_headers(getattr(getattr(error, "response", None), "headers", None))


def _rejected_body(error: BaseException) -> Mapping[str, object] | None:
    """El cuerpo de la respuesta que LangChain cuelga de la excepción, o `None` si no se lee.

    Es la respuesta de esa llamada y de ninguna otra, ya leída entera por el SDK antes de intentar
    convertirla. Que no esté, o que no sea JSON, no es un error: el intento se queda sin lo que
    de ahí se iba a sacar, igual que una respuesta sin contadores.
    """
    reader = getattr(getattr(error, "response", None), "json", None)
    if not callable(reader):
        return None
    try:
        body = reader()
    except (ValueError, RuntimeError):
        return None
    return body if isinstance(body, Mapping) else None


def usage_from_rejected_parse(error: BaseException) -> TokenUsage | None:
    """El `usage` de una respuesta que el SDK no pudo convertir en el esquema.

    Con `json_schema` el SDK valida dentro de `parse()` y lanza antes de que exista un mensaje, así
    que `usage_from_message` no tiene de dónde leer. El intento se facturó igual: los contadores
    están en el cuerpo de esa misma respuesta, y sin ellos un intento inválido —o uno válido cuyo
    contenido llegó envuelto y se rescató después— quedaba como «sin medir» por culpa del
    adaptador y no del proveedor.
    """
    body = _rejected_body(error)
    return None if body is None else usage_from_provider(body.get("usage"))


def content_from_rejected_parse(error: BaseException) -> str | None:
    """El contenido entero de una respuesta que el SDK no pudo convertir en el esquema.

    Es lo que el modelo contestó, tal cual, leído del cuerpo de su propia respuesta: el mismo
    sitio del que sale el `usage`. Con él, lo que el router valida —y por tanto el error que el
    reintento adjunta y lo que queda en la caché— es la respuesta y no un trozo de ella.

    `None` si el cuerpo no se lee o no trae contenido de texto: quien llama cae entonces a lo que
    Pydantic guardó en el error, que es peor y es lo que había.
    """
    body = _rejected_body(error)
    choices = None if body is None else body.get("choices")
    first = choices[0] if isinstance(choices, list) and choices else None
    message = first.get("message") if isinstance(first, Mapping) else None
    content = message.get("content") if isinstance(message, Mapping) else None
    if not isinstance(content, str) or not content.strip():
        return None
    return json_payload(content)


def usage_from_ollama(response: object) -> TokenUsage | None:
    """Los contadores de Ollama: tokens del prompt evaluado y de la respuesta.

    Ollama no informa una caché de prefijo en el sentido de la pasarela, así que
    `cached_tokens` queda `None`. `prompt_eval_count` falta cuando reutilizó el prefijo
    de la llamada anterior; sigue siendo `None` y no cero.
    """
    result = TokenUsage(
        prompt_tokens=_count(getattr(response, "prompt_eval_count", None)),
        completion_tokens=_count(getattr(response, "eval_count", None)),
    )
    return None if result == TokenUsage() else result


class ChatBackend(Protocol):
    """Contrato mínimo de un proveedor: prompt más esquema, texto JSON de vuelta.

    Devuelve el texto solo, o una `Completion` si además sabe el uso que declaró el
    proveedor. Los adaptadores reales devuelven siempre `Completion`.
    """

    async def complete(
        self, choice: ModelChoice, prompt: str, schema: type[LLMOutput]
    ) -> str | Completion:
        """Invoca el modelo y devuelve su salida como texto JSON sin validar."""
        ...


class ModelCatalog(Protocol):
    """Qué modelos sirve un proveedor. Sondas de arranque, no de inferencia.

    Vive aquí y no en `doctor.py` porque este módulo es el único al que la regla 4
    le permite hablar con un proveedor. Listar un catálogo no gasta cuota ni emite
    `LLMCall`, pero abrir esa puerta en otro módulo sí dejaría sitio para meter
    mañana una llamada de inferencia sin registrar.
    """

    async def available_models(self) -> frozenset[str]:
        """Identificadores que el proveedor declara servir."""
        ...


class ResidentModel(FrozenModel):
    """Un modelo cargado ahora mismo en el servidor local.

    `size_vram` contra `size` es la única lectura honesta de si cabe entero en la
    GPU: es lo que `ollama ps` resume como `PROCESSOR`, y cualquier reparto con CPU
    multiplica la latencia sin que la llamada falle.
    """

    name: str = Field(min_length=1)
    size: int = Field(ge=0)
    size_vram: int = Field(ge=0)
    context_length: int | None = None

    @property
    def fully_on_gpu(self) -> bool:
        """Sin una sola capa en CPU."""
        return self.size > 0 and self.size_vram == self.size

    @property
    def gpu_fraction(self) -> float:
        """Fracción de los pesos residente en GPU."""
        return self.size_vram / self.size if self.size > 0 else 0.0


SESSION_HEADER = "x-opencode-session"
"""Cabecera que OpenCode Go exige para enrutar; sin ella responde 400."""


class OpenAIBackend:
    """Adaptador para cualquier endpoint compatible con OpenAI.

    El timeout y el número de reintentos del cliente se pasan siempre, nunca se
    heredan. Los dos valores por defecto del SDK son inaceptables aquí por
    razones distintas:

    - `timeout=None` se resuelve en 600 s de lectura, diez minutos que un rol
      colgado retiene mientras el resto de la evaluación espera.
    - `max_retries=2` reintenta por dentro. El proveedor ve tres peticiones y el
      router escribe un `LLMCall`, que es la regla 4 rota sin que nadie pueda
      verlo: la única evidencia de las otras dos está en la factura. El reintento
      es del router, que lo hace adjuntando el error y lo registra fila a fila.

    Cada petición lleva además `x-opencode-session`. OpenCode Go empezó a
    rechazar con `400 MissingSessionID` cualquier llamada sin ella, antes de
    generar contenido: los seis roles caían como fallo de transporte. El gateway
    pide un id estable por conversación; aquí es uno por backend, es decir por
    proceso, porque llevarlo por evaluación cambiaría el contrato de
    `ChatBackend.complete()`. La caché no lo ve: su clave es modelo, prompt y
    esquema, así que un replay no depende de él.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        timeout_seconds: float = 120.0,
        session_id: str | None = None,
    ) -> None:
        if session_id is not None and not session_id.strip():
            raise ValueError(f"{SESSION_HEADER} no puede ir vacío: el gateway lo rechaza con 400")
        self._api_key = api_key
        self._base_url = base_url
        self._timeout_seconds = timeout_seconds
        self._session_id = session_id or f"crypto-agents-{uuid.uuid4().hex[:8]}"

    @property
    def session_id(self) -> str:
        """Identificador que viaja en `x-opencode-session` en cada petición."""
        return self._session_id

    def _headers(self) -> dict[str, str]:
        """Cabeceras comunes al cliente de chat y a la sonda del catálogo."""
        return {SESSION_HEADER: self._session_id}

    def _http_client(self) -> object | None:
        """El cliente HTTP del chat, o `None` para que el SDK construya el suyo.

        En producción es siempre `None`. Existe para que una prueba pueda poner un transporte
        simulado debajo del `ChatOpenAI` y del SDK reales, y comprobar lo que se lee de una
        respuesta sin sustituir nada de lo que la lee.
        """
        return None

    async def complete(
        self, choice: ModelChoice, prompt: str, schema: type[LLMOutput]
    ) -> Completion:
        """Pide salida estructurada y devuelve el texto crudo del modelo y su uso.

        Se usa `include_raw=True` para quedarse con lo que el modelo emitió antes
        de que LangChain lo valide: la validación la hace el router, que es quien
        sabe reintentar.

        Una `ValidationError` que se escape de LangChain se captura aquí y no se
        deja subir: este método debe devolver texto, y una excepción de validación
        cruzando la frontera del adaptador se lleva por delante el reintento que
        el router tiene documentado. El intento se facturó igual y el mensaje no
        sobrevive a la excepción, pero la respuesta HTTP sí: va colgada de ella, y
        de su cuerpo se leen el contenido entero y los contadores. El contenido
        entero y no lo que Pydantic guardó en el error, que con el fallo en un
        campo anidado es un fragmento: validarlo otra vez daba un error que el
        modelo no había cometido, y ese era el que recibía en el reintento.

        `include_response_headers=True` hace que las cabeceras de la respuesta viajen en el
        mensaje que esa respuesta produjo, y de ahí se leen las dos de upstream. No hay hook
        sobre el cliente HTTP: uno compartido no sabría a qué llamada pertenece cada respuesta.
        En la ruta de la `ValidationError` no hay mensaje, pero la respuesta va colgada de la
        excepción, y de ella se leen las dos cabeceras y el `usage`.
        """
        from langchain_openai import ChatOpenAI

        client = ChatOpenAI(
            model=choice.model,
            temperature=choice.temperature,
            api_key=self._api_key,  # type: ignore[arg-type]
            base_url=self._base_url,
            timeout=self._timeout_seconds,
            max_retries=0,
            default_headers=self._headers(),
            include_response_headers=True,
            http_async_client=self._http_client(),
        )
        structured = structured_runnable(client, schema, choice)
        try:
            result = await structured.ainvoke(prompt)  # type: ignore[attr-defined]
        except ValidationError as error:
            whole = content_from_rejected_parse(error)
            return Completion(
                text=whole if whole is not None else _raw_from_validation_error(error),
                usage=usage_from_rejected_parse(error),
                upstream=upstream_from_rejected_parse(error),
            )
        raw = result.get("raw") if isinstance(result, dict) else None
        return Completion(
            text=raw_text(result),
            usage=usage_from_message(raw),
            upstream=upstream_from_message(raw),
        )

    async def available_models(self) -> frozenset[str]:
        """Catálogo del gateway.

        Se pide con el cliente de OpenAI y no con una petición HTTP a mano para que
        la sonda hable exactamente la misma pila que las llamadas que valida: el
        armado de `base_url` y la cabecera de autorización son justo donde una
        configuración equivocada se manifiesta.
        """
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self._timeout_seconds,
            max_retries=0,
            default_headers=self._headers(),
        )
        try:
            page = await client.models.list()
            return frozenset(model.id for model in page.data)
        finally:
            await client.close()


class OllamaBackend:
    """Adaptador para un servidor Ollama local.

    El cliente se construye una vez y se reutiliza: uno por llamada abriría una
    sesión HTTP nueva en cada intento.

    El timeout se pasa explícito porque el cliente de Ollama no trae ninguno —su
    httpx sale con `Timeout(None)`— y un servidor local que se cuelga no devuelve
    ni un error: deja la corrida esperando para siempre, sin una línea que mirar.
    """

    def __init__(
        self,
        host: str,
        keep_alive: str = "30m",
        num_ctx: int = 4096,
        timeout_seconds: float = 300.0,
    ) -> None:
        from ollama import AsyncClient

        self._client = AsyncClient(host=host, timeout=timeout_seconds)
        self._keep_alive = keep_alive
        self._num_ctx = num_ctx
        self._timeout_seconds = timeout_seconds

    async def complete(
        self, choice: ModelChoice, prompt: str, schema: type[LLMOutput]
    ) -> Completion:
        """Ollama restringe la generación al JSON Schema, pero no aplica nuestros validadores.

        No lee `choice.structured_output`: pasar el esquema en `format` *es*
        `json_schema`, y un validador de `ModelChoice` impide declarar otra cosa
        sobre este backend. Ollama no expone herramientas, así que la rama de
        `function_calling` no existiría aquí.

        `keep_alive` evita que los pesos se descarguen entre ciclos del runner y
        `num_ctx` acota la KV cache, que es lo que decide si el modelo cabe entero
        en la GPU o Ollama empieza a descargar capas a CPU.
        """
        response = await self._client.chat(
            model=choice.model,
            messages=[{"role": "user", "content": prompt}],
            format=schema.model_json_schema(),
            options={"temperature": choice.temperature, "num_ctx": self._num_ctx},
            keep_alive=self._keep_alive,
        )
        return Completion(
            text=json_payload(response.message.content or ""), usage=usage_from_ollama(response)
        )

    async def available_models(self) -> frozenset[str]:
        """Tags descargados en el servidor."""
        listing = await self._client.list()
        return frozenset(model.model for model in listing.models if model.model)

    async def resident(self) -> tuple[ResidentModel, ...]:
        """Lo que hay cargado ahora mismo, con su reparto GPU/CPU.

        Es `ollama ps` por HTTP. Solo dice la verdad justo después de una llamada:
        pasado `keep_alive` el servidor descarga los pesos y la lista queda vacía.
        """
        running = await self._client.ps()
        return tuple(
            ResidentModel(
                name=model.model or "",
                size=int(model.size or 0),
                size_vram=int(model.size_vram or 0),
                context_length=model.context_length,
            )
            for model in running.models
            if model.model
        )


def build_backends(settings: Settings) -> dict[Backend, ChatBackend]:
    """Construye solo los backends que la configuración declara."""
    backends: dict[Backend, ChatBackend] = {}
    if settings.openai is not None:
        backends[Backend.OPENAI] = OpenAIBackend(
            api_key=settings.openai.api_key.get_secret_value(),
            base_url=settings.openai.base_url,
            timeout_seconds=settings.openai.timeout_seconds,
        )
    if settings.ollama is not None:
        backends[Backend.OLLAMA] = OllamaBackend(
            host=settings.ollama.host,
            keep_alive=settings.ollama.keep_alive,
            num_ctx=settings.ollama.num_ctx,
            timeout_seconds=settings.ollama.timeout_seconds,
        )
    return backends


class ModelRouter:
    """Único punto por el que pasa toda llamada a modelo.

    Concentrar aquí resolución de modelo, caché, validación y registro es lo que
    hace medible el presupuesto: no hay forma de gastar cuota sin dejar rastro.
    """

    def __init__(
        self,
        settings: Settings,
        ledger: QuotaLedger,
        backends: Mapping[Backend, ChatBackend],
        clock: Clock,
        cache: ResponseCache | None = None,
        max_attempts: int = 2,
        guard: SpendGuard | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts debe ser al menos 1")
        self._settings = settings
        self._ledger = ledger
        self._backends = dict(backends)
        self._clock = clock
        self._cache = cache
        self._max_attempts = max_attempts
        self._guard = guard

    async def invoke[T: LLMOutput](
        self,
        role: AgentRole,
        prompt: str,
        schema: type[T],
        check: ContextCheck[T] | None = None,
    ) -> tuple[T, list[LLMCall]]:
        """Resuelve, consulta caché, llama y valida. Devuelve la salida y todos los intentos.

        Cada intento fallido ya consumió una llamada en el proveedor, así que se
        registra igual: un reintento cuesta cuota.

        `check` es la validación de contexto del nodo. Un fallo suyo es un intento
        inválido igual que uno de esquema —se registra, consume cuota y se
        reintenta con el error adjunto—, y los dos comparten el mismo número de
        intentos: fallar primero el esquema y después el contexto los agota.
        """
        calls: list[LLMCall] = []
        current = prompt
        last_error = ""
        admitted = False

        for _attempt in range(self._max_attempts):
            digest = prompt_digest(current)

            choices = self._settings.role_choices(role)
            choice = choices[0]
            replayed = self._replay_attempt(choice, digest, schema, check)
            if replayed is None:
                # Sin entrada del primario hay que llamar, y a quién lo decide el
                # presupuesto. Solo si el contador degradó de verdad en este
                # intento se mira la entrada del respaldo: es el modelo que iba a
                # responder de todos modos.
                choice = self._choose(role, choices)
                if choice != choices[0]:
                    replayed = self._replay_attempt(choice, digest, schema, check)

            cache_hit = replayed is not None
            elapsed_ms = 0.0
            usage: TokenUsage | None = None
            judged: T | _Rejected
            upstream: Upstream | None
            if replayed is not None:
                # Un acierto no hizo petición: el upstream es el de la respuesta guardada.
                judged, upstream = replayed
            else:
                if not admitted and choice.backend not in LOCAL_BACKENDS:
                    # El tope se mira aquí y no antes: un acierto de caché no gasta, así
                    # que no pregunta, y una llamada local tampoco. Y una sola vez por
                    # invocación: el reintento de una que ya se abrió no se niega, o un
                    # intento pagado se quedaría sin su corrección.
                    self._admit(role, calls)
                    admitted = True
                backend = self._backends.get(choice.backend)
                if backend is None:
                    raise LookupError(
                        f"backend {choice.backend.value} no configurado para {role.value}"
                    )

                started = time.perf_counter()
                try:
                    completion = as_completion(await backend.complete(choice, current, schema))
                except BackendNotCalledError:
                    raise  # no llegó a salir: no hay intento que registrar ni que envolver
                except Exception as error:
                    # La petición salió: el proveedor la vio y puede haberla
                    # cobrado. Registrarla es lo que impide que la regla 4 se
                    # incumpla justo cuando el proveedor se porta mal, que es
                    # cuando el registro importa. No se reintenta: la plantilla de
                    # reintento corrige un error de validación, y aquí no hay
                    # salida que corregir. Tampoco se guarda en caché: no hay texto
                    # que guardar, y un 503 cacheado sería un 503 para siempre.
                    calls.append(
                        self._record(
                            role,
                            choice,
                            digest,
                            cache_hit=False,
                            latency_ms=(time.perf_counter() - started) * 1000.0,
                            failure=_Rejected(
                                _provider_failure_kind(error), f"{type(error).__name__}: {error}"
                            ),
                        )
                    )
                    raise ModelCallError(role, choice, error, tuple(calls)) from error
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                usage = completion.usage
                upstream = completion.upstream
                judged = _validate(completion.text, schema, check)
                self._store(role, choice, digest, schema, completion.text, judged, upstream)

            if isinstance(judged, _Rejected):
                last_error = judged.message
                calls.append(
                    self._record(
                        role,
                        choice,
                        digest,
                        cache_hit=cache_hit,
                        latency_ms=elapsed_ms,
                        failure=judged,
                        usage=usage,
                        upstream=upstream,
                    )
                )
                current = prompt + _RETRY_TEMPLATE.format(error=last_error)
                continue

            calls.append(
                self._record(
                    role,
                    choice,
                    digest,
                    cache_hit=cache_hit,
                    latency_ms=elapsed_ms,
                    usage=usage,
                    upstream=upstream,
                )
            )
            return judged, calls

        raise InvalidModelOutputError(role, self._max_attempts, last_error, tuple(calls))

    def _choose(self, role: AgentRole, choices: Sequence[ModelChoice]) -> ModelChoice:
        """A quién se llama ahora: el primario, salvo que su cuota lo mande al respaldo.

        Con pago por uso la cuota declarada no frena un rol remoto. OpenCode Zen no publica
        límite de peticiones, así que una cifra en `quota_per_window` —la de Go que se quedó en
        `.env`, o cualquier otra— es un límite que el proveedor no impone: aplicarla degradaría el
        rol al modelo local, o abortaría al decisor, por nada. Lo que frena ahí es el tope en
        dólares (`_admit`). El contador sigue anotando cada llamada; solo deja de decidir.

        La regla vive aquí y no en `QuotaLedger` porque el router es el único que tiene a la vez
        la configuración y el contador, y el contador no debe tenerla. Un primario local —los
        brazos de la ablación que fuerzan un rol al respaldo— sigue por el contador, y con la
        suscripción nada cambia.

        Consecuencia: con pago por uso el respaldo local de un rol remoto no se activa nunca por
        cuota. Tampoco se activaba por un rechazo del proveedor.
        """
        primary = choices[0]
        if self._settings.billing is Billing.PAYG and primary.backend not in LOCAL_BACKENDS:
            return primary
        return self._ledger.resolve(role, choices)

    def _admit(self, role: AgentRole, calls: Sequence[LLMCall]) -> None:
        """Deja abrir una invocación de pago, o lanza si el tope de gasto ya se alcanzó.

        Solo pregunta quien va a llamar a un proveedor remoto: una llamada local es gratis por
        regla y no hay nada que topar. `calls` son los intentos que esta invocación ya reprodujo
        de la caché: viajan en la excepción para que lleguen al journal.
        """
        if self._guard is None or self._guard.allow(role.value):
            return
        raise SpendCapReachedError(role, self._guard.cap_usd, self._guard.spent_usd, tuple(calls))

    def _replay_attempt[T: LLMOutput](
        self, choice: ModelChoice, digest: str, schema: type[T], check: ContextCheck[T] | None
    ) -> _Replayed[T] | None:
        """Lo que ese modelo ya respondió a ese prompt, juzgado de nuevo. O `None` si no hay.

        Una clave, un modelo. Antes se recorrían todos los candidatos del rol y se
        devolvía el primero que hubiera: con la entrada del primario ausente y la
        del respaldo presente, el rol recibía como acierto la respuesta de otro
        modelo. En la ablación eso es un brazo remoto sirviendo lo que un 8B local
        le contestó a otro brazo, registrado además con el backend del respaldo y
        sin gastar cuota —nada en la tabla lo delataba.

        Quién decide qué modelo se mira es `invoke()`: primero el primario, sin
        consultar el presupuesto porque un acierto no gasta —`resolve()` lanza
        cuando no queda cuota, y un replay sobre una caché caliente fallaría sin ir
        a llamar a nadie—, y el respaldo solo cuando el contador lo resolvió.

        El texto guardado se valida otra vez, esquema y contexto. Tres resultados:

        - **Pasa**: es la respuesta.
        - **No pasa, y se guardó como inválido**: es un intento reproducido. Se
          devuelve el rechazo recalculado, que da el mismo error, el mismo prompt
          de reintento y el mismo digest que la primera vez.
        - **No pasa, y se guardó como válido**: el esquema o el contexto cambiaron
          desde entonces. Es una entrada vieja: se descarta y cuenta como hueco.

        Igual que una que no se puede leer o que no es de esta pregunta.
        """
        if self._cache is None:
            return None
        key = cache_key(choice.backend, choice.model, digest, schema, choice.structured_output)
        stored = self._cache.get(key)
        if stored is None:
            return None
        try:
            entry = CacheEntry.model_validate_json(stored)
            if not entry.answers(
                choice.backend, choice.model, digest, schema, choice.structured_output
            ):
                raise ValueError("la entrada no corresponde a esta clave")
        except ValueError:
            self._cache.discard(key)
            return None
        judged = _validate(entry.raw, schema, check)
        if isinstance(judged, _Rejected) and entry.valid:
            self._cache.discard(key)
            return None
        answered_by = Upstream(model=entry.upstream_model, endpoint=entry.upstream_endpoint)
        return _Replayed(judged, None if answered_by == Upstream() else answered_by)

    def _store[T: LLMOutput](
        self,
        role: AgentRole,
        choice: ModelChoice,
        digest: str,
        schema: type[T],
        payload: str,
        judged: T | _Rejected,
        upstream: Upstream | None = None,
    ) -> None:
        """Guarda el intento, válido o no, bajo el digest de su propio prompt.

        La clave es la de siempre —modelo, prompt, esquema, modo—, así que una
        respuesta válida a la primera queda exactamente donde quedaba. Lo nuevo es
        que el intento rechazado también queda, bajo el digest del prompt que lo
        produjo, y el reintento bajo el suyo, que incluye el error.
        """
        if self._cache is None:
            return
        rejected = judged if isinstance(judged, _Rejected) else None
        entry = CacheEntry(
            backend=choice.backend,
            model=choice.model,
            role=role,
            prompt_digest=digest,
            schema_name=schema.__name__,
            structured_output=choice.structured_output,
            raw=payload,
            valid=rejected is None,
            failure_kind=None if rejected is None else rejected.kind,
            failure_message=None if rejected is None else rejected.message,
            upstream_model=None if upstream is None else upstream.model,
            upstream_endpoint=None if upstream is None else upstream.endpoint,
        )
        self._cache.set(entry.key, entry.model_dump_json(indent=2))

    def _record(
        self,
        role: AgentRole,
        choice: ModelChoice,
        digest: str,
        cache_hit: bool,
        latency_ms: float,
        failure: _Rejected | None = None,
        usage: TokenUsage | None = None,
        upstream: Upstream | None = None,
    ) -> LLMCall:
        """Anota el intento. Los aciertos de caché no consumen presupuesto.

        `valid` se deriva de `failure` en vez de pasarse aparte: son la misma
        afirmación, y dos parámetros permitirían registrar un intento fallido sin
        decir por qué.

        `usage` es lo que el proveedor contó, y solo existe en una llamada viva: un
        acierto de caché no hizo petición, y un fallo del proveedor no devolvió contenido.

        `upstream` es quién contestó según la pasarela. En una llamada viva viene de su
        respuesta; en un acierto de caché, de la entrada guardada.
        """
        call = LLMCall(
            role=role,
            backend=choice.backend,
            model=choice.model,
            structured_output=choice.structured_output,
            quota_weight=choice.quota_weight,
            prompt_digest=digest,
            cache_hit=cache_hit,
            valid=failure is None,
            failure_kind=None if failure is None else failure.kind,
            failure_message=None if failure is None else failure.message,
            latency_ms=latency_ms,
            at=self._clock(),
            prompt_tokens=None if usage is None else usage.prompt_tokens,
            cached_tokens=None if usage is None else usage.cached_tokens,
            completion_tokens=None if usage is None else usage.completion_tokens,
            upstream_model=None if upstream is None else upstream.model,
            upstream_endpoint=None if upstream is None else upstream.endpoint,
        )
        self._ledger.record(call)
        if self._guard is not None and not cache_hit:
            self._guard.add((call,))
        return call


def _validate[T: LLMOutput](
    payload: str, schema: type[T], check: ContextCheck[T] | None
) -> T | _Rejected:
    """Las dos validaciones de una salida: primero su forma, después su contexto.

    En ese orden porque la segunda necesita un objeto: no se puede preguntar qué
    ids cita un alegato que ni siquiera es un alegato. Es una función y no dos
    bloques en `invoke()` para que el intento vivo y la entrada de caché pasen
    exactamente por lo mismo.
    """
    try:
        output = schema.model_validate_json(payload)
    except ValidationError as error:
        return _Rejected(FailureKind.SCHEMA, _summarise(error))
    problems = check(output) if check is not None else []
    if problems:
        return _Rejected(FailureKind.CONTEXT, "; ".join(problems))
    return output


def _provider_failure_kind(error: BaseException) -> FailureKind:
    """Timeout o transporte, mirando la excepción y todo lo que la causó.

    Se recorre la cadena porque los clientes envuelven: el timeout de httpx llega
    dentro de un error de conexión del SDK, y juzgando solo la excepción de fuera
    todo sería transporte. Los tipos de cada proveedor se nombran aquí porque este
    es el único módulo que puede importarlos.
    """
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (TimeoutError, *_provider_timeouts())):
            return FailureKind.TIMEOUT
        current = current.__cause__ or current.__context__
    return FailureKind.TRANSPORT


def _provider_timeouts() -> tuple[type[BaseException], ...]:
    """Excepciones de timeout de los clientes instalados."""
    import httpx
    import openai

    return (httpx.TimeoutException, openai.APITimeoutError)


def _summarise(error: ValidationError) -> str:
    """Resumen corto del fallo, apto para meter en el prompt de reintento."""
    parts = [
        f"{'.'.join(str(item) for item in detail['loc'])}: {detail['msg']}"
        for detail in error.errors()[:5]
    ]
    return "; ".join(parts) or str(error)
