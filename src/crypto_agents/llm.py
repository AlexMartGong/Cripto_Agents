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

La degradación a un modelo local no vive aquí: es `QuotaLedger.resolve()` quien
elige, y lo hace en cada intento, así que un veredicto puede empezar remoto y
terminar local. Por eso cada `LLMCall` registra su propio `backend` y su propio
`valid`: sin esas dos columnas, un respaldo local que necesita tres intentos por
veredicto queda en el journal indistinguible de un remoto que acierta a la
primera, y la comparación entre ambos deja de ser posible.
"""

from __future__ import annotations

import hashlib
import time
from typing import TYPE_CHECKING, Protocol

from pydantic import ValidationError

from crypto_agents.cache import cache_key
from crypto_agents.state import Backend, LLMCall, LLMOutput

if TYPE_CHECKING:
    from collections.abc import Mapping

    from crypto_agents.cache import ResponseCache
    from crypto_agents.quota import Clock, QuotaLedger
    from crypto_agents.settings import ModelChoice, Settings
    from crypto_agents.state import AgentRole

__all__ = [
    "ChatBackend",
    "InvalidModelOutputError",
    "ModelRouter",
    "OllamaBackend",
    "OpenAIBackend",
    "build_backends",
    "prompt_digest",
]

_RETRY_TEMPLATE = (
    "\n\n---\n"
    "Tu respuesta anterior no pasó la validación del esquema.\n"
    "Error: {error}\n"
    "Corrige exactamente eso y responde de nuevo, solo con JSON válido."
)


class InvalidModelOutputError(RuntimeError):
    """El modelo agotó los intentos sin producir una salida que valide."""

    def __init__(self, role: AgentRole, attempts: int, last_error: str) -> None:
        self.role = role
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"{role.value}: {attempts} intento(s) sin salida válida; último error: {last_error}"
        )


def prompt_digest(prompt: str) -> str:
    """Huella del prompt para caché y replay."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


class ChatBackend(Protocol):
    """Contrato mínimo de un proveedor: prompt más esquema, texto JSON de vuelta."""

    async def complete(self, choice: ModelChoice, prompt: str, schema: type[LLMOutput]) -> str:
        """Invoca el modelo y devuelve su salida como texto JSON sin validar."""
        ...


class OpenAIBackend:
    """Adaptador para cualquier endpoint compatible con OpenAI."""

    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        self._api_key = api_key
        self._base_url = base_url

    async def complete(self, choice: ModelChoice, prompt: str, schema: type[LLMOutput]) -> str:
        """Pide salida estructurada y devuelve el texto crudo del modelo.

        Se usa `include_raw=True` para quedarse con lo que el modelo emitió antes
        de que LangChain lo valide: la validación la hace el router, que es quien
        sabe reintentar.
        """
        from langchain_openai import ChatOpenAI

        client = ChatOpenAI(
            model=choice.model,
            temperature=choice.temperature,
            api_key=self._api_key,  # type: ignore[arg-type]
            base_url=self._base_url,
        )
        result = await client.with_structured_output(schema, include_raw=True).ainvoke(prompt)
        raw = result["raw"] if isinstance(result, dict) else None
        content = getattr(raw, "content", None)
        if isinstance(content, str) and content.strip():
            return content
        parsed = result["parsed"] if isinstance(result, dict) else None
        if isinstance(parsed, LLMOutput):
            return parsed.model_dump_json()
        return ""  # texto vacío: falla la validación y el router reintenta


class OllamaBackend:
    """Adaptador para un servidor Ollama local.

    El cliente se construye una vez y se reutiliza: uno por llamada abriría una
    sesión HTTP nueva en cada intento.
    """

    def __init__(self, host: str, keep_alive: str = "30m", num_ctx: int = 4096) -> None:
        from ollama import AsyncClient

        self._client = AsyncClient(host=host)
        self._keep_alive = keep_alive
        self._num_ctx = num_ctx

    async def complete(self, choice: ModelChoice, prompt: str, schema: type[LLMOutput]) -> str:
        """Ollama restringe la generación al JSON Schema, pero no aplica nuestros validadores.

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
        return response.message.content or ""


def build_backends(settings: Settings) -> dict[Backend, ChatBackend]:
    """Construye solo los backends que la configuración declara."""
    backends: dict[Backend, ChatBackend] = {}
    if settings.openai is not None:
        backends[Backend.OPENAI] = OpenAIBackend(
            api_key=settings.openai.api_key.get_secret_value(),
            base_url=settings.openai.base_url,
        )
    if settings.ollama is not None:
        backends[Backend.OLLAMA] = OllamaBackend(
            host=settings.ollama.host,
            keep_alive=settings.ollama.keep_alive,
            num_ctx=settings.ollama.num_ctx,
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
                    self._record(
                        role, cached_choice, digest, cache_hit=True, valid=True, latency_ms=0.0
                    )
                )
                return cached, calls

            choice = self._ledger.resolve(role)
            key = cache_key(choice.model, digest, schema)

            backend = self._backends.get(choice.backend)
            if backend is None:
                raise LookupError(
                    f"backend {choice.backend.value} no configurado para {role.value}"
                )

            started = time.perf_counter()
            payload = await backend.complete(choice, current, schema)
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            try:
                output = schema.model_validate_json(payload)
            except ValidationError as error:
                calls.append(
                    self._record(
                        role, choice, digest, cache_hit=False, valid=False, latency_ms=elapsed_ms
                    )
                )
                last_error = _summarise(error)
                current = prompt + _RETRY_TEMPLATE.format(error=last_error)
                continue

            calls.append(
                self._record(
                    role, choice, digest, cache_hit=False, valid=True, latency_ms=elapsed_ms
                )
            )
            if self._cache is not None:
                self._cache.set(key, payload)
            return output, calls

        raise InvalidModelOutputError(role, self._max_attempts, last_error)

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
            cached = self._read_cache(cache_key(choice.model, digest, schema), schema)
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
        valid: bool,
        latency_ms: float,
    ) -> LLMCall:
        """Anota el intento. Los aciertos de caché no consumen presupuesto."""
        call = LLMCall(
            role=role,
            backend=choice.backend,
            model=choice.model,
            quota_weight=choice.quota_weight,
            prompt_digest=digest,
            cache_hit=cache_hit,
            valid=valid,
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
