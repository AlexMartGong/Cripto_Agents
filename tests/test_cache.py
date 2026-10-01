"""Pruebas de la caché de respuestas."""

from __future__ import annotations

from typing import TYPE_CHECKING

from crypto_agents.cache import (
    CacheEntry,
    InMemoryResponseCache,
    JsonFileResponseCache,
    cache_key,
)
from crypto_agents.state import (
    AgentRole,
    Backend,
    DebateBrief,
    StructuredOutputMode,
    TechnicalVerdict,
)

if TYPE_CHECKING:
    from pathlib import Path

DIGEST = "a" * 64
OTHER = "b" * 64
SCHEMA_MODE = StructuredOutputMode.JSON_SCHEMA


def test_key_depends_on_the_model() -> None:
    """La misma pregunta a otro modelo es otra respuesta."""
    assert cache_key(Backend.OPENAI, "gpt-x", DIGEST, TechnicalVerdict, SCHEMA_MODE) != cache_key(
        Backend.OPENAI, "qwen3:8b", DIGEST, TechnicalVerdict, SCHEMA_MODE
    )


def test_key_depends_on_the_backend() -> None:
    """El mismo nombre de modelo servido por otro proveedor es otra respuesta.

    Un tag local y un id del gateway pueden coincidir en el nombre y no en los
    pesos, la cuantización ni el modo en que se les pide el esquema.
    """
    assert cache_key(
        Backend.OPENAI, "qwen3:8b", DIGEST, TechnicalVerdict, SCHEMA_MODE
    ) != cache_key(Backend.OLLAMA, "qwen3:8b", DIGEST, TechnicalVerdict, SCHEMA_MODE)


def entry(**changes: object) -> CacheEntry:
    """Entrada de caché con lo que la prueba cambie."""
    fields: dict[str, object] = {
        "backend": Backend.OPENAI,
        "model": "gpt-x",
        "role": AgentRole.STRUCTURE,
        "prompt_digest": DIGEST,
        "schema_name": "TechnicalVerdict",
        "structured_output": SCHEMA_MODE,
        "raw": '{"a": 1}',
    }
    return CacheEntry.model_validate(fields | changes)


def test_an_entry_answers_the_question_it_was_stored_for() -> None:
    """La entrada sabe para qué pregunta es, y solo dice que sí a esa."""
    stored = entry()

    assert stored.answers(Backend.OPENAI, "gpt-x", DIGEST, TechnicalVerdict, SCHEMA_MODE)
    assert stored.key == cache_key(Backend.OPENAI, "gpt-x", DIGEST, TechnicalVerdict, SCHEMA_MODE)
    assert not stored.answers(Backend.OLLAMA, "gpt-x", DIGEST, TechnicalVerdict, SCHEMA_MODE)
    assert not stored.answers(Backend.OPENAI, "otro", DIGEST, TechnicalVerdict, SCHEMA_MODE)
    assert not stored.answers(Backend.OPENAI, "gpt-x", OTHER, TechnicalVerdict, SCHEMA_MODE)
    assert not stored.answers(Backend.OPENAI, "gpt-x", DIGEST, DebateBrief, SCHEMA_MODE)
    assert not stored.answers(
        Backend.OPENAI, "gpt-x", DIGEST, TechnicalVerdict, StructuredOutputMode.JSON_MODE
    )


def test_key_depends_on_the_prompt() -> None:
    """Otro prompt, otra entrada."""
    assert cache_key(Backend.OPENAI, "gpt-x", DIGEST, TechnicalVerdict, SCHEMA_MODE) != cache_key(
        Backend.OPENAI, "gpt-x", OTHER, TechnicalVerdict, SCHEMA_MODE
    )


def test_key_depends_on_the_schema() -> None:
    """Cambiar el esquema vuelve inservible lo guardado."""
    assert cache_key(Backend.OPENAI, "gpt-x", DIGEST, TechnicalVerdict, SCHEMA_MODE) != cache_key(
        Backend.OPENAI, "gpt-x", DIGEST, DebateBrief, SCHEMA_MODE
    )


def test_key_depends_on_the_mode() -> None:
    """El modo decide por dónde llega la salida, así que no es la misma respuesta.

    Con `function_calling` viene en los argumentos de la herramienta y con
    `json_schema` en el contenido. Compartir entrada haría que cambiar el modo
    devolviera lo que produjo el anterior, que es justo lo que se estaba
    intentando dejar de leer.
    """
    assert cache_key(Backend.OPENAI, "gpt-x", DIGEST, TechnicalVerdict, SCHEMA_MODE) != cache_key(
        Backend.OPENAI, "gpt-x", DIGEST, TechnicalVerdict, StructuredOutputMode.FUNCTION_CALLING
    )


def test_key_is_stable() -> None:
    """Dos ejecuciones sobre la misma entrada producen la misma clave."""
    assert cache_key(Backend.OPENAI, "gpt-x", DIGEST, TechnicalVerdict, SCHEMA_MODE) == cache_key(
        Backend.OPENAI, "gpt-x", DIGEST, TechnicalVerdict, SCHEMA_MODE
    )


def test_memory_cache_round_trip() -> None:
    """Guardar y recuperar."""
    cache = InMemoryResponseCache()
    assert cache.get("k") is None
    cache.set("k", '{"a": 1}')
    assert cache.get("k") == '{"a": 1}'
    cache.discard("k")
    assert cache.get("k") is None


def test_file_cache_survives_a_new_instance(tmp_path: Path) -> None:
    """Persistir es el punto: una reejecución no vuelve a gastar cuota."""
    JsonFileResponseCache(tmp_path).set("k", '{"a": 1}')
    assert JsonFileResponseCache(tmp_path).get("k") == '{"a": 1}'


def test_file_cache_ignores_corrupt_entries(tmp_path: Path) -> None:
    """Un archivo truncado se trata como fallo de caché, no como respuesta."""
    cache = JsonFileResponseCache(tmp_path)
    (tmp_path / "k.json").write_text("{roto", encoding="utf-8")
    assert cache.get("k") is None


def test_file_cache_discard_is_idempotent(tmp_path: Path) -> None:
    """Borrar algo que no existe no debe fallar."""
    JsonFileResponseCache(tmp_path).discard("inexistente")


def test_file_cache_creates_its_directory(tmp_path: Path) -> None:
    """El directorio se crea solo la primera vez."""
    target = tmp_path / "anidado" / "cache"
    JsonFileResponseCache(target).set("k", "{}")
    assert (target / "k.json").is_file()
