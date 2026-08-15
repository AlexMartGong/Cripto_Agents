"""Pruebas de configuración.

El criterio central: si falta una variable requerida, el arranque falla con un
mensaje que la nombra. Un `KeyError` a mitad de una evaluación no cuenta.
"""

from __future__ import annotations

import json
import os
from datetime import timedelta
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from crypto_agents.settings import (
    DEFAULT_ENV_FILE,
    Backend,
    ConfigError,
    ExchangeSettings,
    ModelChoice,
    RoleConfig,
    load_settings,
)
from crypto_agents.state import AgentRole, StructuredOutputMode
from tests.conftest import CHEAP, role_map

if TYPE_CHECKING:
    from pathlib import Path

EXPENSIVE = ModelChoice(
    backend=Backend.OPENAI,
    model="gpt-x",
    family="gpt",
    structured_output=StructuredOutputMode.JSON_SCHEMA,
    quota_weight=2.0,
    quota_per_window=120,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Aísla del entorno real del desarrollador; el `.env` se desactiva por kwarg."""
    for key in list(os.environ):
        if key.startswith("CA_"):
            monkeypatch.delenv(key, raising=False)


def base_kwargs(**overrides: object) -> dict[str, object]:
    """Configuración mínima válida; usa Ollama para no exigir clave de OpenAI."""
    kwargs: dict[str, object] = {
        "roles": role_map(),
        "ollama": {"host": "http://localhost:11434"},
    }
    kwargs.update(overrides)
    return kwargs


# ─────────────────────────────────────────── Casos válidos ────────────────────────────────────────


def test_minimal_configuration_loads() -> None:
    """Ventana por defecto de 5 horas y todos los roles mapeados."""
    settings = load_settings(**base_kwargs())
    assert settings.quota_window == timedelta(hours=5)
    assert set(settings.roles) == set(AgentRole)


def test_choices_includes_primaries_and_fallbacks() -> None:
    """El router necesita ver todos los modelos declarados, no solo los primarios.

    Uno por rol más un respaldo para cada rol salvo el decisor, que no admite.
    """
    settings = load_settings(**base_kwargs(roles=role_map(primary=CHEAP, fallback=CHEAP)))
    assert len(settings.choices()) == 2 * len(AgentRole) - 1


def test_settings_are_frozen() -> None:
    """La configuración no cambia a mitad de una ejecución."""
    settings = load_settings(**base_kwargs())
    with pytest.raises(ValidationError):
        settings.quota_window = timedelta(hours=1)


def test_exchange_without_credentials_is_valid() -> None:
    """Leer OHLCV no necesita claves."""
    assert load_settings(**base_kwargs()).exchange == ExchangeSettings()


def test_roles_load_from_nested_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """El reparto rol → modelo es un dato del entorno, no una constante del código."""
    for role in AgentRole:
        prefix = f"CA_ROLES__{role.value.upper()}__PRIMARY__"
        monkeypatch.setenv(f"{prefix}BACKEND", "ollama")
        monkeypatch.setenv(f"{prefix}MODEL", "qwen3:8b")
        monkeypatch.setenv(f"{prefix}FAMILY", "qwen" if role is not AgentRole.BEAR else "llama")
        monkeypatch.setenv(f"{prefix}STRUCTURED_OUTPUT", "json_schema")
        monkeypatch.setenv(f"{prefix}QUOTA_PER_WINDOW", "63000")
    monkeypatch.setenv("CA_OLLAMA__HOST", "http://localhost:11434")

    settings = load_settings()
    assert settings.role_config(AgentRole.DECIDER).primary.model == "qwen3:8b"


def test_quota_weight_is_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Los modelos de doble consumo se declaran como dato, con peso 2.0."""
    for role in AgentRole:
        prefix = f"CA_ROLES__{role.value.upper()}__PRIMARY__"
        monkeypatch.setenv(f"{prefix}BACKEND", "openai")
        monkeypatch.setenv(f"{prefix}MODEL", "gpt-x")
        monkeypatch.setenv(f"{prefix}FAMILY", "gpt" if role is not AgentRole.BEAR else "claude")
        monkeypatch.setenv(f"{prefix}STRUCTURED_OUTPUT", "function_calling")
        monkeypatch.setenv(f"{prefix}QUOTA_WEIGHT", "2.0")
        monkeypatch.setenv(f"{prefix}QUOTA_PER_WINDOW", "120")
    monkeypatch.setenv("CA_OPENAI__API_KEY", "sk-test")

    settings = load_settings()
    assert settings.role_config(AgentRole.BULL).primary.quota_weight == 2.0


# ─────────────────────────────────────────── Archivo .env ─────────────────────────────────────────


def _write_env_file(directory: Path) -> Path:
    """Un `.env` mínimo y válido, con los seis roles en local."""
    roles = {
        role.value: {
            "primary": {
                "backend": "ollama",
                "model": "qwen3:8b",
                "family": "qwen" if role is not AgentRole.BEAR else "llama",
                "structured_output": "json_schema",
                "quota_per_window": 63000,
            }
        }
        for role in AgentRole
    }
    path = directory / ".env"
    path.write_text(
        f"CA_ROLES={json.dumps(roles)}\nCA_OLLAMA__HOST=http://localhost:11434\n",
        encoding="utf-8",
    )
    return path


def test_a_env_file_in_the_working_directory_is_not_read_on_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Estar parado junto a un `.env` no es pedirlo.

    Cuando `Settings` lo declaraba por defecto, cualquier llamada a
    `load_settings()` leía el archivo del desarrollador: la suite entera pasaba a
    depender de en qué máquina corría, y en la que opera —la única que tiene
    `.env`— dos pruebas de la CLI fallaban, una de ellas colgándose hasta el
    siguiente cierre de vela.
    """
    _write_env_file(tmp_path)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ConfigError, match=r"falta CA_ROLES"):
        load_settings()


def test_the_env_file_loads_when_it_is_asked_for_by_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Y con la ruta explícita sí se lee, que es como lo piden los dos comandos."""
    path = _write_env_file(tmp_path)
    monkeypatch.chdir(tmp_path)

    settings = load_settings(path)
    assert settings.role_config(AgentRole.DECIDER).primary.model == "qwen3:8b"
    assert load_settings(DEFAULT_ENV_FILE).roles == settings.roles


# ────────────────────────────────────────── Fallos de arranque ────────────────────────────────────


def test_missing_roles_names_the_variable() -> None:
    """Sin mapa de roles, el mensaje nombra la variable que falta."""
    with pytest.raises(ConfigError, match=r"falta CA_ROLES"):
        load_settings(ollama={"host": "http://x"})


def test_incomplete_role_map_names_the_missing_roles() -> None:
    """Un rol sin modelo revienta al arrancar, no a mitad de una evaluación."""
    partial = role_map()
    del partial[AgentRole.DECIDER]
    del partial[AgentRole.BULL]
    with pytest.raises(ConfigError, match="bull, decider"):
        load_settings(**base_kwargs(roles=partial))


def test_openai_backend_without_credentials_fails_at_startup() -> None:
    """Usar un backend sin credenciales debe fallar antes de la primera llamada."""
    with pytest.raises(ConfigError, match=r"CA_OPENAI__API_KEY"):
        load_settings(roles=role_map(primary=EXPENSIVE))


def test_fallback_backend_also_requires_credentials() -> None:
    """El respaldo se usa de verdad: sus credenciales cuentan igual que las del primario."""
    with pytest.raises(ConfigError, match=r"CA_OPENAI__API_KEY"):
        load_settings(**base_kwargs(roles=role_map(primary=CHEAP, fallback=EXPENSIVE)))


def test_ollama_only_accepts_the_mode_it_actually_implements() -> None:
    """Declarar `function_calling` sobre el respaldo local sería una configuración que miente.

    `OllamaBackend` restringe la generación pasando el esquema en `format`, que es
    `json_schema`, y no expone herramientas. Aceptar otra cosa y luego ignorarla
    dejaría al journal registrando un modo que nunca se pidió.
    """
    with pytest.raises(ValidationError, match="ollama solo admite"):
        ModelChoice(
            backend=Backend.OLLAMA,
            model="qwen3:8b",
            family="qwen",
            structured_output=StructuredOutputMode.FUNCTION_CALLING,
            quota_per_window=100,
        )


def test_a_model_without_a_declared_mode_does_not_load() -> None:
    """El modo no tiene default: un valor por omisión sería la constante implícita de vuelta.

    Y el fallo nombra la variable, como el resto de la configuración que falta.
    """
    roles = {
        role.value: {
            "primary": {
                "backend": "ollama",
                "model": "qwen3:8b",
                "family": "qwen" if role is not AgentRole.BEAR else "llama",
                "quota_per_window": 100,
            }
        }
        for role in AgentRole
    }
    with pytest.raises(ConfigError, match="STRUCTURED_OUTPUT"):
        load_settings(roles=roles, ollama={"host": "http://localhost:11434"})


def test_the_decider_cannot_declare_a_fallback() -> None:
    """Degradar al decisor cambiaría quién decide sin que quede dicho en ninguna parte.

    Los demás roles sí degradan: una lectura técnica más pobre sigue siendo una
    lectura y el journal registra con qué backend se produjo. La decisión final no
    admite ese trato, así que la configuración que lo intente no llega a cargar.
    """
    roles = role_map()
    roles[AgentRole.DECIDER] = RoleConfig(primary=CHEAP, fallback=CHEAP)
    with pytest.raises(ConfigError, match="decider no admite fallback"):
        load_settings(**base_kwargs(roles=roles))


def test_debate_desks_sharing_a_family_are_rejected() -> None:
    """Con la misma familia en ambas mesas, sus errores están correlacionados.

    Dos mesas sobre el mismo modelo pasan por alto lo mismo: el decisor recibiría
    dos versiones del mismo sesgo creyendo que son puntos de vista independientes.
    """
    roles = role_map()
    roles[AgentRole.BEAR] = RoleConfig(primary=CHEAP)
    with pytest.raises(ConfigError, match="comparten familia: qwen"):
        load_settings(**base_kwargs(roles=roles))


def test_debate_desks_sharing_a_family_through_a_fallback_are_rejected() -> None:
    """El respaldo cuenta: la correlación aparece igual cuando una mesa degrada."""
    roles = role_map()
    roles[AgentRole.BULL] = RoleConfig(
        primary=CHEAP.model_copy(update={"family": "gpt"}), fallback=CHEAP
    )
    roles[AgentRole.BEAR] = RoleConfig(
        primary=CHEAP.model_copy(update={"family": "claude"}), fallback=CHEAP
    )
    with pytest.raises(ConfigError, match="comparten familia: qwen"):
        load_settings(**base_kwargs(roles=roles))


def test_debate_desks_in_different_families_are_accepted() -> None:
    """Familias distintas en ambas mesas es la configuración válida."""
    settings = load_settings(**base_kwargs())
    bull = {choice.family for choice in settings.role_choices(AgentRole.BULL)}
    bear = {choice.family for choice in settings.role_choices(AgentRole.BEAR)}
    assert not bull & bear


def test_half_exchange_credentials_are_rejected() -> None:
    """Media credencial es configuración a medio hacer, no modo de solo lectura."""
    with pytest.raises(ConfigError, match="juntas o ninguna"):
        load_settings(**base_kwargs(exchange={"api_key": "solo-la-clave"}))


def test_unknown_field_is_rejected() -> None:
    """`extra="forbid"`: una variable mal escrita no se ignora en silencio."""
    with pytest.raises(ConfigError):
        load_settings(**base_kwargs(qouta_window=timedelta(hours=1)))


def test_zero_quota_weight_is_rejected() -> None:
    """Un peso de cero haría que un modelo pareciera gratis."""
    raw = {
        role.value: {
            "primary": {
                "backend": "ollama",
                "model": "qwen3:8b",
                "family": "qwen" if role is not AgentRole.BEAR else "llama",
                "quota_weight": 0.0,
                "quota_per_window": 10,
            }
        }
        for role in AgentRole
    }
    with pytest.raises(ConfigError, match="QUOTA_WEIGHT"):
        load_settings(**base_kwargs(roles=raw))
