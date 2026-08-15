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
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from crypto_agents.execution import ExecutionSettings
from crypto_agents.risk import AccountState, RiskLimits
from crypto_agents.runner import RunnerSettings
from crypto_agents.state import AgentRole, Backend

__all__ = [
    "DEFAULT_ENV_FILE",
    "ENV_PREFIX",
    "Backend",
    "ConfigError",
    "ExchangeSettings",
    "ExecutionSettings",
    "ModelChoice",
    "OllamaSettings",
    "OpenAISettings",
    "OperationsSettings",
    "RoleConfig",
    "RunnerSettings",
    "Settings",
    "load_settings",
]

ENV_PREFIX = "CA_"
_NESTED_DELIMITER = "__"

DEFAULT_ENV_FILE = Path(".env")
"""Archivo que leen los puntos de entrada, relativo al directorio de trabajo.

Está aquí y no en cada comando para que exista un único sitio donde ver qué
archivo acaba en la configuración de una corrida real.
"""


class ConfigError(RuntimeError):
    """Arranque abortado por configuración incompleta o incoherente."""


class ModelChoice(BaseModel):
    """Un modelo concreto con su presupuesto de cuota.

    `quota_weight` es lo que consume una llamada: 2.0 en los modelos que el
    proveedor cobra como doble uso.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: Backend
    model: str = Field(min_length=1)
    family: str = Field(min_length=1)
    """Familia del modelo, declarada a mano.

    Se declara en vez de deducirse del nombre: una heurística por prefijo dejaría
    pasar un modelo con nombre inesperado, que es justo el fallo que la regla de
    diversidad entre mesas intenta evitar.
    """

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
    """Servidor Ollama local y los dos parámetros que deciden si cabe en la GPU."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    host: str = Field(default="http://localhost:11434", min_length=1)
    keep_alive: str = Field(default="30m", min_length=1)
    """Cuánto mantiene Ollama los pesos cargados tras la última llamada.

    Con el valor por defecto de Ollama (5 min) un runner de velas de 4h recarga
    el modelo entero en cada ciclo, y esa recarga domina el tiempo total de la
    etapa técnica. Se declara aquí y no como variable de entorno del servidor
    para que quede en la misma configuración que el resto.
    """

    num_ctx: int = Field(default=4096, ge=512)
    """Ventana de contexto del modelo local.

    En 8 GB de VRAM el contexto es lo primero que se come el margen: la KV cache
    crece con `num_ctx` y con el número de peticiones concurrentes. Un valor alto
    obliga a Ollama a descargar capas a CPU y la latencia se multiplica.
    """


class OperationsSettings(BaseModel):
    """Dónde vive el estado operativo: journal, caché y centinela de parada."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    journal_path: Path = Field(default=Path("var/journal.jsonl"))
    cache_dir: Path = Field(default=Path("var/cache"))
    kill_switch_file: Path = Field(default=Path("var/STOP"))
    """Centinela de parada. Su sola existencia detiene cualquier orden.

    Un archivo y no una señal: una señal solo alcanza al proceso que la recibe y
    se pierde al reiniciar, mientras que el archivo sobrevive, lo pone cualquiera
    con acceso al disco y expresa «sigue parado».
    """


class Settings(BaseSettings):
    """Configuración completa del sistema.

    El mapa `roles` acepta tanto JSON en una variable (`CA_ROLES`) como claves
    anidadas (`CA_ROLES__STRUCTURE__PRIMARY__MODEL`).

    No se declara `env_file` aquí a propósito: con un archivo por defecto, la
    configuración pasa a depender del directorio desde el que se arranque, y quien
    carga `Settings` no puede saber si acabó leyendo disco o no. El archivo se pide
    por ruta explícita en `load_settings()`, y solo lo piden los dos puntos de
    entrada.
    """

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_nested_delimiter=_NESTED_DELIMITER,
        env_file_encoding="utf-8",
        extra="forbid",
        frozen=True,
    )

    exchange: ExchangeSettings = ExchangeSettings()
    openai: OpenAISettings | None = None
    ollama: OllamaSettings | None = None
    roles: dict[AgentRole, RoleConfig]
    quota_window: timedelta = timedelta(hours=5)
    risk: RiskLimits = RiskLimits()
    execution: ExecutionSettings = ExecutionSettings()
    runner: RunnerSettings | None = None
    """Calendario del bucle. Sin esto, `crypto-agents run` aborta nombrándolo."""

    account: AccountState | None = None
    """Fotografía de la cuenta. Opcional porque consultar el journal no la necesita."""

    operations: OperationsSettings = OperationsSettings()

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

    @model_validator(mode="after")
    def _the_decider_never_degrades(self) -> Self:
        """El decisor no admite respaldo. Es preferible no decidir a decidir peor.

        Todos los demás roles degradan a un modelo local cuando se les acaba el
        presupuesto: una lectura técnica más pobre sigue siendo una lectura, y el
        journal registra con qué backend se produjo. El decisor no, porque
        degradarlo cambia quién toma la decisión final sin que eso aparezca en
        ninguna parte antes de que la orden ya esté puesta. Sin respaldo, la cuota
        agotada aborta la evaluación y la aborta con causa.
        """
        config = self.roles.get(AgentRole.DECIDER)
        if config is not None and config.fallback is not None:
            raise ValueError(
                "el rol decider no admite fallback: agotada su cuota la evaluación se aborta"
            )
        return self

    @model_validator(mode="after")
    def _debate_desks_use_different_families(self) -> Self:
        """Las dos mesas no pueden compartir familia de modelo.

        Con el mismo modelo y distinto system prompt, los errores de las dos
        mesas están correlacionados: ambas pasan por alto lo mismo y el debate
        deja de aportar información. El decisor recibiría dos versiones del mismo
        sesgo creyendo que son puntos de vista independientes.
        """
        bull = {choice.family for choice in self.role_choices(AgentRole.BULL)}
        bear = {choice.family for choice in self.role_choices(AgentRole.BEAR)}
        shared = sorted(bull & bear)
        if shared:
            raise ValueError(f"las mesas bull y bear comparten familia: {', '.join(shared)}")
        return self

    def role_choices(self, role: AgentRole) -> tuple[ModelChoice, ...]:
        """Modelos que ese rol puede llegar a usar: primario y respaldo."""
        config = self.roles.get(role)
        if config is None:
            return ()
        if config.fallback is None:
            return (config.primary,)
        return (config.primary, config.fallback)

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


def load_settings(env_file: Path | str | None = None, /, **overrides: object) -> Settings:
    """Carga la configuración o aborta con un mensaje que nombra lo que falta.

    Sin `env_file` no se lee ningún archivo: solo el entorno y lo que se pase por
    argumento. Leer `.env` por defecto ataba la configuración al directorio de
    trabajo, y con ello la suite entera —que llama aquí decenas de veces— pasaba a
    depender de que la máquina *no* tuviera un `.env` al lado. Verde en la máquina
    de desarrollo y colgada en la que opera es la peor forma de estar verde.
    """
    try:
        return Settings(_env_file=env_file, **overrides)  # type: ignore[arg-type, call-arg]
    except ValidationError as error:
        raise ConfigError(_format_validation_error(error)) from error
