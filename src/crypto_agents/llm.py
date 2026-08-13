"""Acceso a modelos: protocolo, adaptadores concretos y router.

Ningún nodo importa un cliente concreto. Los nodos piden `ModelRouter.invoke(role,
prompt, schema)` y el router resuelve qué modelo cabe en presupuesto, llama al
backend que corresponda y devuelve la salida validada junto al registro de la
llamada. Cambiar de OpenAI a Ollama es cambiar configuración, no código de nodo.
"""

from __future__ import annotations

import hashlib
import time
from typing import TYPE_CHECKING, Protocol

from crypto_agents.settings import Backend
from crypto_agents.state import LLMCall, LLMOutput

if TYPE_CHECKING:
    from collections.abc import Mapping

    from crypto_agents.quota import Clock, QuotaLedger
    from crypto_agents.settings import ModelChoice, Settings
    from crypto_agents.state import AgentRole

__all__ = [
    "ChatBackend",
    "ModelRouter",
    "OllamaBackend",
    "OpenAIBackend",
    "build_backends",
    "prompt_digest",
]


def prompt_digest(prompt: str) -> str:
    """Huella del prompt para replay y detección de cache hits."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


class ChatBackend(Protocol):
    """Contrato mínimo de un proveedor: prompt más esquema, salida validada."""

    async def structured[T: LLMOutput](
        self, choice: ModelChoice, prompt: str, schema: type[T]
    ) -> T:
        """Invoca el modelo y devuelve una instancia de `schema`."""
        ...


class OpenAIBackend:
    """Adaptador para cualquier endpoint compatible con OpenAI."""

    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        self._api_key = api_key
        self._base_url = base_url

    async def structured[T: LLMOutput](
        self, choice: ModelChoice, prompt: str, schema: type[T]
    ) -> T:
        """Salida estructurada vía `with_structured_output`, que impone el esquema."""
        from langchain_openai import ChatOpenAI

        client = ChatOpenAI(
            model=choice.model,
            temperature=choice.temperature,
            api_key=self._api_key,  # type: ignore[arg-type]
            base_url=self._base_url,
        )
        return await client.with_structured_output(schema).ainvoke(prompt)  # type: ignore[return-value]


class OllamaBackend:
    """Adaptador para un servidor Ollama local."""

    def __init__(self, host: str) -> None:
        self._host = host

    async def structured[T: LLMOutput](
        self, choice: ModelChoice, prompt: str, schema: type[T]
    ) -> T:
        """Ollama valida contra el JSON Schema del modelo Pydantic."""
        from ollama import AsyncClient

        response = await AsyncClient(host=self._host).chat(
            model=choice.model,
            messages=[{"role": "user", "content": prompt}],
            format=schema.model_json_schema(),
            options={"temperature": choice.temperature},
        )
        return schema.model_validate_json(response.message.content or "")


def build_backends(settings: Settings) -> dict[Backend, ChatBackend]:
    """Construye solo los backends que la configuración declara."""
    backends: dict[Backend, ChatBackend] = {}
    if settings.openai is not None:
        backends[Backend.OPENAI] = OpenAIBackend(
            api_key=settings.openai.api_key.get_secret_value(),
            base_url=settings.openai.base_url,
        )
    if settings.ollama is not None:
        backends[Backend.OLLAMA] = OllamaBackend(host=settings.ollama.host)
    return backends


class ModelRouter:
    """Único punto por el que pasa toda llamada a modelo.

    Concentrar aquí la resolución de modelo y el registro de la llamada es lo que
    hace medible el presupuesto: no hay forma de gastar cuota sin dejar rastro.
    """

    def __init__(
        self,
        settings: Settings,
        ledger: QuotaLedger,
        backends: Mapping[Backend, ChatBackend],
        clock: Clock,
    ) -> None:
        self._settings = settings
        self._ledger = ledger
        self._backends = dict(backends)
        self._clock = clock

    async def invoke[T: LLMOutput](
        self, role: AgentRole, prompt: str, schema: type[T]
    ) -> tuple[T, LLMCall]:
        """Resuelve modelo, llama y registra. Propaga `QuotaExhaustedError` sin gastar nada."""
        choice = self._ledger.resolve(role)
        backend = self._backends.get(choice.backend)
        if backend is None:
            raise LookupError(f"backend {choice.backend.value} no configurado para {role.value}")

        started = time.perf_counter()
        output = await backend.structured(choice, prompt, schema)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        call = LLMCall(
            role=role,
            model=choice.model,
            quota_weight=choice.quota_weight,
            prompt_digest=prompt_digest(prompt),
            cache_hit=False,
            latency_ms=elapsed_ms,
            at=self._clock(),
        )
        self._ledger.record(call)
        return output, call
