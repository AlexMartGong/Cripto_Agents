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

Todo lo que salga del adaptador se envuelve aquí en `ModelCallError`, con los
intentos ya pagados dentro. El router es el único módulo que habla con un
proveedor, así que también es el único sitio donde se pueden convertir las
excepciones de cada cliente —httpx, openai, ollama— en un tipo que un nodo pueda
capturar sin importar ninguno de los tres.

La degradación a un modelo local no vive aquí: es `QuotaLedger.resolve()` quien
elige, y lo hace en cada intento, así que un veredicto puede empezar remoto y
terminar local. Por eso cada `LLMCall` registra su propio `backend` y su propio
`valid`: sin esas dos columnas, un respaldo local que necesita tres intentos por
veredicto queda en el journal indistinguible de un remoto que acierta a la
primera, y la comparación entre ambos deja de ser posible.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import TYPE_CHECKING, Protocol

from pydantic import Field, ValidationError

from crypto_agents.cache import cache_key
from crypto_agents.state import (
    Backend,
    CallFailure,
    FailureKind,
    FrozenModel,
    LLMCall,
    LLMOutput,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from crypto_agents.cache import ResponseCache
    from crypto_agents.quota import Clock, QuotaLedger
    from crypto_agents.settings import ModelChoice, Settings
    from crypto_agents.state import AgentRole

__all__ = [
    "BackendNotCalledError",
    "ChatBackend",
    "InvalidModelOutputError",
    "ModelCallError",
    "ModelCatalog",
    "ModelInvocationError",
    "ModelRouter",
    "OllamaBackend",
    "OpenAIBackend",
    "ResidentModel",
    "build_backends",
    "json_payload",
    "prompt_digest",
    "raw_text",
    "structured_runnable",
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
    """Rescata del propio error lo que el modelo había emitido.

    Pydantic guarda en `input` el valor que rechazó, y cuando la validación
    ocurre dentro del adaptador esa es la única copia que queda del texto crudo.
    Se prefiere el error más externo —el de `loc` más corto— porque su `input` es
    la respuesta entera y no el campo suelto que la rompió.
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


class ChatBackend(Protocol):
    """Contrato mínimo de un proveedor: prompt más esquema, texto JSON de vuelta."""

    async def complete(self, choice: ModelChoice, prompt: str, schema: type[LLMOutput]) -> str:
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
    """

    def __init__(
        self, api_key: str, base_url: str | None = None, timeout_seconds: float = 120.0
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._timeout_seconds = timeout_seconds

    async def complete(self, choice: ModelChoice, prompt: str, schema: type[LLMOutput]) -> str:
        """Pide salida estructurada y devuelve el texto crudo del modelo.

        Se usa `include_raw=True` para quedarse con lo que el modelo emitió antes
        de que LangChain lo valide: la validación la hace el router, que es quien
        sabe reintentar.

        Una `ValidationError` que se escape de LangChain se captura aquí y no se
        deja subir: este método debe devolver texto, y una excepción de validación
        cruzando la frontera del adaptador se lleva por delante el reintento que
        el router tiene documentado. Lo que el modelo dijo se recupera del propio
        error, porque en ese punto ya no queda en ningún otro sitio.
        """
        from langchain_openai import ChatOpenAI

        client = ChatOpenAI(
            model=choice.model,
            temperature=choice.temperature,
            api_key=self._api_key,  # type: ignore[arg-type]
            base_url=self._base_url,
            timeout=self._timeout_seconds,
            max_retries=0,
        )
        structured = structured_runnable(client, schema, choice)
        try:
            result = await structured.ainvoke(prompt)  # type: ignore[attr-defined]
        except ValidationError as error:
            return _raw_from_validation_error(error)
        return raw_text(result)

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

    async def complete(self, choice: ModelChoice, prompt: str, schema: type[LLMOutput]) -> str:
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
        return json_payload(response.message.content or "")

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
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts debe ser al menos 1")
        self._settings = settings
        self._ledger = ledger
        self._backends = dict(backends)
        self._clock = clock
        self._cache = cache
        self._max_attempts = max_attempts

    async def invoke[T: LLMOutput](
        self, role: AgentRole, prompt: str, schema: type[T]
    ) -> tuple[T, list[LLMCall]]:
        """Resuelve, consulta caché, llama y valida. Devuelve la salida y todos los intentos.

        Cada intento fallido ya consumió una llamada en el proveedor, así que se
        registra igual: un reintento cuesta cuota.
        """
        calls: list[LLMCall] = []
        current = prompt
        last_error = ""

        for _attempt in range(self._max_attempts):
            digest = prompt_digest(current)

            hit = self._read_any_cache(role, digest, schema)
            if hit is not None:
                cached, cached_choice = hit
                calls.append(
                    self._record(role, cached_choice, digest, cache_hit=True, latency_ms=0.0)
                )
                return cached, calls

            choice = self._ledger.resolve(role)
            key = cache_key(choice.model, digest, schema, choice.structured_output)

            backend = self._backends.get(choice.backend)
            if backend is None:
                raise LookupError(
                    f"backend {choice.backend.value} no configurado para {role.value}"
                )

            started = time.perf_counter()
            try:
                payload = await backend.complete(choice, current, schema)
            except BackendNotCalledError:
                raise  # no llegó a salir: no hay intento que registrar ni que envolver
            except Exception as error:
                # La petición salió: el proveedor la vio y puede haberla cobrado.
                # Registrarla es lo que impide que la regla 4 se incumpla justo
                # cuando el proveedor se porta mal, que es cuando el registro
                # importa. No se reintenta: la plantilla de reintento corrige un
                # error de validación, y aquí no hay salida que corregir.
                calls.append(
                    self._record(
                        role,
                        choice,
                        digest,
                        cache_hit=False,
                        latency_ms=(time.perf_counter() - started) * 1000.0,
                        failure=CallFailure(
                            kind=FailureKind.TRANSPORT,
                            message=f"{type(error).__name__}: {error}",
                        ),
                    )
                )
                raise ModelCallError(role, choice, error, tuple(calls)) from error
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            try:
                output = schema.model_validate_json(payload)
            except ValidationError as error:
                last_error = _summarise(error)
                calls.append(
                    self._record(
                        role,
                        choice,
                        digest,
                        cache_hit=False,
                        latency_ms=elapsed_ms,
                        failure=CallFailure(kind=FailureKind.VALIDATION, message=last_error),
                    )
                )
                current = prompt + _RETRY_TEMPLATE.format(error=last_error)
                continue

            calls.append(self._record(role, choice, digest, cache_hit=False, latency_ms=elapsed_ms))
            if self._cache is not None:
                self._cache.set(key, payload)
            return output, calls

        raise InvalidModelOutputError(role, self._max_attempts, last_error, tuple(calls))

    def _read_any_cache[T: LLMOutput](
        self, role: AgentRole, digest: str, schema: type[T]
    ) -> tuple[T, ModelChoice] | None:
        """Busca en la caché por todos los modelos que ese rol puede usar.

        Se mira antes de resolver la cuota, y no después, porque `resolve()` lanza
        cuando no queda presupuesto: consultando en el otro orden, un replay sobre
        una caché caliente fallaría por cuota agotada aunque no fuera a llamar a
        ningún proveedor. La clave de caché incluye el modelo, así que hay que
        probar candidato por candidato; se devuelve el primero, que es el primario.
        """
        for choice in self._settings.role_choices(role):
            key = cache_key(choice.model, digest, schema, choice.structured_output)
            cached = self._read_cache(key, schema)
            if cached is not None:
                return cached, choice
        return None

    def _read_cache[T: LLMOutput](self, key: str, schema: type[T]) -> T | None:
        """Lee y revalida. Una entrada que ya no valida se descarta, no se usa."""
        if self._cache is None:
            return None
        payload = self._cache.get(key)
        if payload is None:
            return None
        try:
            return schema.model_validate_json(payload)
        except ValidationError:
            self._cache.discard(key)
            return None

    def _record(
        self,
        role: AgentRole,
        choice: ModelChoice,
        digest: str,
        cache_hit: bool,
        latency_ms: float,
        failure: CallFailure | None = None,
    ) -> LLMCall:
        """Anota el intento. Los aciertos de caché no consumen presupuesto.

        `valid` se deriva de `failure` en vez de pasarse aparte: son la misma
        afirmación, y dos parámetros permitirían registrar un intento fallido sin
        decir por qué.
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
            failure=failure,
            latency_ms=latency_ms,
            at=self._clock(),
        )
        self._ledger.record(call)
        return call


def _summarise(error: ValidationError) -> str:
    """Resumen corto del fallo, apto para meter en el prompt de reintento."""
    parts = [
        f"{'.'.join(str(item) for item in detail['loc'])}: {detail['msg']}"
        for detail in error.errors()[:5]
    ]
    return "; ".join(parts) or str(error)
