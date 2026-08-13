"""Pruebas de configuración.

El criterio central: si falta una variable requerida, el arranque falla con un
mensaje que la nombra. Un `KeyError` a mitad de una evaluación no cuenta.
"""

from __future__ import annotations

import os
from datetime import timedelta

import pytest
from pydantic import ValidationError

from crypto_agents.settings import (
    Backend,
    ConfigError,
    ExchangeSettings,
    ModelChoice,
    RoleConfig,
    load_settings,
)
from crypto_agents.state import AgentRole

CHEAP = ModelChoice(backend=Backend.OLLAMA, model="qwen3:8b", quota_per_window=63000)
EXPENSIVE = ModelChoice(
    backend=Backend.OPENAI, model="gpt-x", quota_weight=2.0, quota_per_window=120
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Aísla del entorno real del desarrollador; el `.env` se desactiva por kwarg."""
    for key in list(os.environ):
        if key.startswith("CA_"):
            monkeypatch.delenv(key, raising=False)


def all_roles(
    primary: ModelChoice = CHEAP, fallback: ModelChoice | None = None
) -> dict[AgentRole, RoleConfig]:
    """Mapa completo de roles, que es lo que exige el validador."""
    return {role: RoleConfig(primary=primary, fallback=fallback) for role in AgentRole}


def base_kwargs(**overrides: object) -> dict[str, object]:
    """Configuración mínima válida; usa Ollama para no exigir clave de OpenAI."""
    kwargs: dict[str, object] = {
        "roles": all_roles(),
        "ollama": {"host": "http://localhost:11434"},
        "_env_file": None,
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
    """El router necesita ver todos los modelos declarados, no solo los primarios."""
    settings = load_settings(**base_kwargs(roles=all_roles(primary=CHEAP, fallback=CHEAP)))
    assert len(settings.choices()) == 2 * len(AgentRole)


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
        monkeypatch.setenv(f"{prefix}QUOTA_PER_WINDOW", "63000")
    monkeypatch.setenv("CA_OLLAMA__HOST", "http://localhost:11434")

    settings = load_settings(_env_file=None)
    assert settings.role_config(AgentRole.DECIDER).primary.model == "qwen3:8b"


def test_quota_weight_is_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Los modelos de doble consumo se declaran como dato, con peso 2.0."""
    for role in AgentRole:
        prefix = f"CA_ROLES__{role.value.upper()}__PRIMARY__"
        monkeypatch.setenv(f"{prefix}BACKEND", "openai")
        monkeypatch.setenv(f"{prefix}MODEL", "gpt-x")
        monkeypatch.setenv(f"{prefix}QUOTA_WEIGHT", "2.0")
        monkeypatch.setenv(f"{prefix}QUOTA_PER_WINDOW", "120")
    monkeypatch.setenv("CA_OPENAI__API_KEY", "sk-test")

    settings = load_settings(_env_file=None)
    assert settings.role_config(AgentRole.BULL).primary.quota_weight == 2.0


# ────────────────────────────────────────── Fallos de arranque ────────────────────────────────────


def test_missing_roles_names_the_variable() -> None:
    """Sin mapa de roles, el mensaje nombra la variable que falta."""
    with pytest.raises(ConfigError, match=r"falta CA_ROLES"):
        load_settings(ollama={"host": "http://x"}, _env_file=None)


def test_incomplete_role_map_names_the_missing_roles() -> None:
    """Un rol sin modelo revienta al arrancar, no a mitad de una evaluación."""
    partial = all_roles()
    del partial[AgentRole.DECIDER]
    del partial[AgentRole.BULL]
    with pytest.raises(ConfigError, match="bull, decider"):
        load_settings(**base_kwargs(roles=partial))


def test_openai_backend_without_credentials_fails_at_startup() -> None:
    """Usar un backend sin credenciales debe fallar antes de la primera llamada."""
    with pytest.raises(ConfigError, match=r"CA_OPENAI__API_KEY"):
        load_settings(roles=all_roles(primary=EXPENSIVE), _env_file=None)


def test_fallback_backend_also_requires_credentials() -> None:
    """El respaldo se usa de verdad: sus credenciales cuentan igual que las del primario."""
    with pytest.raises(ConfigError, match=r"CA_OPENAI__API_KEY"):
        load_settings(**base_kwargs(roles=all_roles(primary=CHEAP, fallback=EXPENSIVE)))


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
                "quota_weight": 0.0,
                "quota_per_window": 10,
            }
        }
        for role in AgentRole
    }
    with pytest.raises(ConfigError, match="QUOTA_WEIGHT"):
        load_settings(**base_kwargs(roles=raw))
