"""Caché de respuestas de modelo, indexada por digest de prompt.

Dos propósitos: no gastar cuota dos veces por la misma pregunta, y hacer el
replay determinista. Se guarda el JSON crudo, no el objeto ya validado, para que
al leerlo se vuelva a validar: si el esquema cambió, la entrada guardada falla la
validación y se trata como un fallo de caché en vez de colarse desactualizada.

La clave incluye el modelo y el nombre del esquema además del digest del prompt.
El mismo texto contra otro modelo es otra respuesta, y un esquema distinto vuelve
inservible lo guardado.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from crypto_agents.state import LLMOutput

__all__ = [
    "InMemoryResponseCache",
    "JsonFileResponseCache",
    "ResponseCache",
    "cache_key",
]


def cache_key(model: str, prompt_digest: str, schema: type[LLMOutput]) -> str:
    """Clave estable para una respuesta: modelo, prompt y esquema."""
    material = "\n".join((model, prompt_digest, schema.__name__))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


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
