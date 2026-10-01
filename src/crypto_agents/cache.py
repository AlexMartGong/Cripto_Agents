"""Caché de respuestas de modelo, indexada por digest de prompt.

Dos propósitos: no gastar cuota dos veces por la misma pregunta, y hacer el
replay determinista. Se guarda el JSON crudo, no el objeto ya validado, para que
al leerlo se vuelva a validar: si el esquema cambió, la entrada guardada falla la
validación y se trata como un fallo de caché en vez de colarse desactualizada.

La clave incluye el backend, el modelo, el modo de salida estructurada y el
nombre del esquema además del digest del prompt. El mismo texto contra otro
modelo es otra respuesta, el mismo nombre de modelo en otro proveedor también, un
esquema distinto vuelve inservible lo guardado, y el modo decide por dónde llega
la salida —`content` o `tool_calls`—, así que dos modos sobre el mismo prompt no
comparten entrada aunque compartan modelo.

Lo que se guarda bajo la clave es un `CacheEntry`: el texto del modelo dentro de
un sobre que dice quién lo respondió y a qué. La clave es un sha-256, irreversible;
sin el sobre, un directorio de caché son miles de respuestas que no se pueden
atribuir a ningún rol ni a ningún modelo.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pydantic import Field

from crypto_agents.state import AgentRole, Backend, FrozenModel, StructuredOutputMode

if TYPE_CHECKING:
    from crypto_agents.state import LLMOutput

__all__ = [
    "CacheEntry",
    "InMemoryResponseCache",
    "JsonFileResponseCache",
    "ResponseCache",
    "cache_key",
]


def cache_key(
    backend: Backend,
    model: str,
    prompt_digest: str,
    schema: type[LLMOutput] | str,
    mode: StructuredOutputMode,
) -> str:
    """Clave estable para una respuesta: backend, modelo, prompt, esquema y modo.

    El esquema entra por su nombre; se acepta la clase o el nombre ya extraído,
    que es lo que una entrada guardada tiene a mano.
    """
    name = schema if isinstance(schema, str) else schema.__name__
    material = "\n".join((backend.value, model, prompt_digest, name, mode.value))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class CacheEntry(FrozenModel):
    """Una respuesta guardada, con lo necesario para saber de quién es.

    Los cinco primeros campos después de `role` son los de la clave, en claro. El
    rol no entra en la clave: es quién pagó la llamada, y si otro rol hiciera
    exactamente la misma pregunta al mismo modelo la respuesta sería la misma.
    """

    backend: Backend
    model: str = Field(min_length=1)
    role: AgentRole
    """Rol que hizo la llamada que llenó esta entrada."""

    prompt_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_name: str = Field(min_length=1)
    structured_output: StructuredOutputMode
    raw: str
    """Lo que devolvió el modelo, como texto. Se revalida contra el esquema al leer."""

    @property
    def key(self) -> str:
        """La clave bajo la que esta entrada debe estar guardada."""
        return cache_key(
            self.backend, self.model, self.prompt_digest, self.schema_name, self.structured_output
        )

    def answers(
        self,
        backend: Backend,
        model: str,
        prompt_digest: str,
        schema: type[LLMOutput],
        mode: StructuredOutputMode,
    ) -> bool:
        """Si esta entrada es la respuesta a esa pregunta, hecha a ese modelo.

        La clave ya debería garantizarlo, pero es un hash y los archivos se
        copian y se renombran: lo que decide si una entrada vale es lo que dice
        dentro, no dónde está.
        """
        return self.key == cache_key(backend, model, prompt_digest, schema, mode)


class ResponseCache(Protocol):
    """Almacén de payloads JSON indexado por clave."""

    def get(self, key: str) -> str | None:
        """Payload guardado, o `None` si no hay nada."""
        ...

    def set(self, key: str, payload: str) -> None:
        """Guarda el payload bajo esa clave."""
        ...

    def discard(self, key: str) -> None:
        """Elimina una entrada que ya no valida contra su esquema."""
        ...


class InMemoryResponseCache:
    """Caché por proceso. Muere con él: el replay entre corridas no la aprovecha."""

    def __init__(self) -> None:
        self._entries: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        """Payload guardado, o `None`."""
        return self._entries.get(key)

    def set(self, key: str, payload: str) -> None:
        """Guarda el payload."""
        self._entries[key] = payload

    def discard(self, key: str) -> None:
        """Olvida la entrada."""
        self._entries.pop(key, None)

    def __len__(self) -> int:
        """Entradas vivas, útil para comprobar aciertos en pruebas."""
        return len(self._entries)


class JsonFileResponseCache:
    """Caché en disco, un archivo por clave.

    Sobrevive al proceso, así que una reejecución sobre las mismas velas no
    vuelve a gastar cuota.
    """

    def __init__(self, directory: Path | str) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self._directory / f"{key}.json"

    def get(self, key: str) -> str | None:
        """Payload guardado, o `None` si el archivo no existe o está corrupto."""
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            payload = path.read_text(encoding="utf-8")
            json.loads(payload)
        except (OSError, json.JSONDecodeError):
            return None
        return payload

    def set(self, key: str, payload: str) -> None:
        """Escribe el payload. La escritura es atómica dentro del directorio."""
        temporary = self._path(key).with_suffix(".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(self._path(key))

    def discard(self, key: str) -> None:
        """Borra la entrada si existe."""
        self._path(key).unlink(missing_ok=True)
