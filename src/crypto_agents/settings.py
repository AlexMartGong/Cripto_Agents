"""Configuración por entorno: el reparto rol → modelo es un dato, no una constante.

Ningún módulo lee `os.environ`. Todo entra por `Settings`, y `load_settings()`
convierte un fallo de validación en un mensaje que nombra las variables que
faltan en vez de volcar el error crudo de Pydantic.

La cuota se declara por rol, no por modelo: cada rol trae su propia ventana. Si
dos roles apuntan al mismo modelo, cada uno lleva su cuenta por separado y el
presupuesto agregado queda sobreestimado; es una decisión explícita del diseño.
"""

from __future__ import annotations

from datetime import timedelta
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from crypto_agents.state import AgentRole

__all__ = [
    "ENV_PREFIX",
    "Backend",
    "ConfigError",
    "ExchangeSettings",
    "ModelChoice",
    "OllamaSettings",
    "OpenAISettings",
    "RoleConfig",
    "Settings",
    "load_settings",
]

ENV_PREFIX = "CA_"
_NESTED_DELIMITER = "__"


class ConfigError(RuntimeError):
    """Arranque abortado por configuración incompleta o incoherente."""


class Backend(StrEnum):
    """Proveedor de modelos. Los nodos nunca lo consultan: solo el router."""

    OPENAI = "openai"
    OLLAMA = "ollama"


class ModelChoice(BaseModel):
    """Un modelo concreto con su presupuesto de cuota.

    `quota_weight` es lo que consume una llamada: 2.0 en los modelos que el
    proveedor cobra como doble uso.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: Backend
    model: str = Field(min_length=1)
    quota_weight: float = Field(default=1.0, gt=0.0)
    quota_per_window: int = Field(gt=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class RoleConfig(BaseModel):
    """Modelo primario de un rol y su respaldo.

    El respaldo es de un solo salto: si tampoco cabe en presupuesto, la llamada
    falla en vez de encadenar degradaciones hasta un modelo inservible.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    primary: ModelChoice
    fallback: ModelChoice | None = None


class ExchangeSettings(BaseModel):
    """Acceso al exchange. Las claves son opcionales: leer OHLCV no las necesita."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    exchange_id: str = Field(default="binance", min_length=1)
    api_key: SecretStr | None = None
    api_secret: SecretStr | None = None
    sandbox: bool = True

    @model_validator(mode="after")
    def _keys_come_in_pairs(self) -> Self:
        """Media credencial es una configuración a medio hacer, no un modo de solo lectura."""
        if (self.api_key is None) != (self.api_secret is None):
            raise ValueError("api_key y api_secret deben declararse juntas o ninguna")
        return self


class OpenAISettings(BaseModel):
    """Credenciales de un backend compatible con OpenAI."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    api_key: SecretStr
    base_url: str | None = None


class OllamaSettings(BaseModel):
    """Ubicación del servidor Ollama local."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    host: str = Field(default="http://localhost:11434", min_length=1)


class Settings(BaseSettings):
    """Configuración completa del sistema.

    El mapa `roles` acepta tanto JSON en una variable (`CA_ROLES`) como claves
    anidadas (`CA_ROLES__STRUCTURE__PRIMARY__MODEL`).
    """

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_nested_delimiter=_NESTED_DELIMITER,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
        frozen=True,
    )

    exchange: ExchangeSettings = ExchangeSettings()
    openai: OpenAISettings | None = None
    ollama: OllamaSettings | None = None
    roles: dict[AgentRole, RoleConfig]
    quota_window: timedelta = timedelta(hours=5)

    @model_validator(mode="after")
    def _every_role_is_mapped(self) -> Self:
        """Un rol sin modelo revienta a mitad de una evaluación, no al arrancar."""
        missing = sorted(role.value for role in AgentRole if role not in self.roles)
        if missing:
            raise ValueError(f"roles sin modelo asignado: {', '.join(missing)}")
        return self

    @model_validator(mode="after")
    def _backends_in_use_have_credentials(self) -> Self:
        """Faltar credenciales del backend que sí usas debe fallar en el arranque."""
        required = {choice.backend for choice in self.choices()}
        missing: list[str] = []
        if Backend.OPENAI in required and self.openai is None:
            missing.append(f"{ENV_PREFIX}OPENAI{_NESTED_DELIMITER}API_KEY")
        if Backend.OLLAMA in required and self.ollama is None:
            missing.append(f"{ENV_PREFIX}OLLAMA{_NESTED_DELIMITER}HOST")
        if missing:
            raise ValueError(f"backends en uso sin credenciales: {', '.join(missing)}")
        return self

    def choices(self) -> tuple[ModelChoice, ...]:
        """Todos los modelos declarados, primarios y de respaldo."""
        result: list[ModelChoice] = []
        for config in self.roles.values():
            result.append(config.primary)
            if config.fallback is not None:
                result.append(config.fallback)
        return tuple(result)

    def role_config(self, role: AgentRole) -> RoleConfig:
        """Configuración de un rol. El validador garantiza que existe."""
        return self.roles[role]


def _as_env_var(location: tuple[int | str, ...]) -> str:
    """Traduce una ruta de error de Pydantic al nombre de variable de entorno."""
    parts = [str(part).upper() for part in location if not isinstance(part, int)]
    return ENV_PREFIX + _NESTED_DELIMITER.join(parts)


def _format_validation_error(error: ValidationError) -> str:
    """Mensaje accionable: qué variable falta y por qué, una por línea."""
    lines: list[str] = []
    for detail in error.errors():
        location = detail["loc"]
        message = detail["msg"]
        if detail["type"] == "missing":
            lines.append(f"  falta {_as_env_var(location)}")
        elif location:
            lines.append(f"  {_as_env_var(location)}: {message}")
        else:
            lines.append(f"  {message}")
    return "configuración inválida:\n" + "\n".join(lines)


def load_settings(**overrides: object) -> Settings:
    """Carga la configuración o aborta con un mensaje que nombra lo que falta."""
    try:
        return Settings(**overrides)  # type: ignore[arg-type]
    except ValidationError as error:
        raise ConfigError(_format_validation_error(error)) from error
