"""Pruebas de la caché de respuestas."""

from __future__ import annotations

from typing import TYPE_CHECKING

from crypto_agents.cache import InMemoryResponseCache, JsonFileResponseCache, cache_key
from crypto_agents.state import DebateBrief, TechnicalVerdict

if TYPE_CHECKING:
    from pathlib import Path

DIGEST = "a" * 64
OTHER = "b" * 64


def test_key_depends_on_the_model() -> None:
    """La misma pregunta a otro modelo es otra respuesta."""
    assert cache_key("gpt-x", DIGEST, TechnicalVerdict) != cache_key(
        "qwen3:8b", DIGEST, TechnicalVerdict
    )


def test_key_depends_on_the_prompt() -> None:
    """Otro prompt, otra entrada."""
    assert cache_key("gpt-x", DIGEST, TechnicalVerdict) != cache_key(
        "gpt-x", OTHER, TechnicalVerdict
    )


def test_key_depends_on_the_schema() -> None:
    """Cambiar el esquema vuelve inservible lo guardado."""
    assert cache_key("gpt-x", DIGEST, TechnicalVerdict) != cache_key("gpt-x", DIGEST, DebateBrief)


def test_key_is_stable() -> None:
    """Dos ejecuciones sobre la misma entrada producen la misma clave."""
    assert cache_key("gpt-x", DIGEST, TechnicalVerdict) == cache_key(
        "gpt-x", DIGEST, TechnicalVerdict
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
