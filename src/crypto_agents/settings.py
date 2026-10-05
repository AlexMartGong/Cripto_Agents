"""Configuración por entorno: el reparto rol → modelo es un dato, no una constante.

Ningún módulo lee `os.environ`. Todo entra por `Settings`, y `load_settings()`
convierte un fallo de validación en un mensaje que nombra las variables que
faltan en vez de volcar el error crudo de Pydantic.

La cuota se declara por rol, no por modelo: cada rol trae su propia ventana. Si
dos roles apuntan al mismo modelo, cada uno lleva su cuenta por separado y el
presupuesto agregado queda sobreestimado; es una decisión explícita del diseño.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from crypto_agents.execution import ExecutionSettings
from crypto_agents.risk import AccountState, RiskLimits
from crypto_agents.runner import RunnerSettings
from crypto_agents.state import AgentRole, Backend, Billing, StructuredOutputMode

__all__ = [
    "DEFAULT_COSTS",
    "DEFAULT_ENV_FILE",
    "DEFAULT_PRICING",
    "ENV_PREFIX",
    "ZEN_UNPUBLISHED_QUOTA",
    "Backend",
    "Billing",
    "ConfigError",
    "CostModel",
    "ExchangeSettings",
    "ExecutionSettings",
    "ModelChoice",
    "OllamaSettings",
    "OpenAISettings",
    "OperationsSettings",
    "PriceRow",
    "PriceTable",
    "RoleConfig",
    "RunnerSettings",
    "Settings",
    "StructuredOutputMode",
    "load_settings",
    "public_url",
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


ZEN_UNPUBLISHED_QUOTA = 100_000
"""`quota_per_window` de un rol remoto con pago por uso: un centinela, no una medición.

OpenCode Zen no publica límite de peticiones (comprobado en su página el 2026-10-04): lo único
que corta el gasto es el saldo y el límite mensual que se fije en el workspace, y los dos son
dólares, no peticiones. Una cifra de Go declarada ahí haría que el contador degradara un rol
remoto al local, o abortara el decisor, por un límite que el proveedor no impone. Este valor tiene
el mismo estatus que el 10 000 de los respaldos locales —«no hay cuota que modelar»— y queda dos
órdenes de magnitud por encima de la cota del decisor sobre el manifiesto (840 llamadas, 1 008
con los reintentos medidos). `tests/test_ablation.py` lo ata a esa cota.
"""


def public_url(url: str | None) -> str | None:
    """De una URL, solo lo que se puede publicar: esquema, host[:puerto] y ruta.

    Sin usuario ni contraseña, sin query y sin fragmento: una `base_url` puede llevar credenciales
    en cualquiera de los tres sitios, y lo que sale en `meta.json` o por pantalla no puede.
    El puerto se conserva porque otro puerto es otro endpoint. Los mensajes de error no citan la
    URL: es justo lo que esta función existe para no repetir.
    """
    if url is None:
        return None
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as error:
        raise ConfigError(f"base_url ilegible ({type(error).__name__})") from error
    host = parts.hostname
    if not parts.scheme or host is None:
        raise ConfigError("base_url sin esquema o sin host")
    if ":" in host:  # IPv6: urlsplit le quita los corchetes
        host = f"[{host}]"
    authority = host if port is None else f"{host}:{port}"
    return f"{parts.scheme}://{authority}{parts.path}"


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

    structured_output: StructuredOutputMode
    """Cómo pedirle a este modelo que se ajuste al esquema.

    Sin valor por defecto a propósito. Un default reinstalaría la constante
    implícita que este campo existe para quitar —solo que mudada de LangChain a
    aquí— y volvería a haber seis modelos operando con un modo que nadie eligió.
    Se declara por modelo y `crypto-agents doctor` lo verifica contra el
    proveedor antes de que se pague una evaluación.
    """

    quota_weight: float = Field(default=1.0, gt=0.0)
    quota_per_window: int = Field(gt=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    @model_validator(mode="after")
    def _mode_is_supported_by_the_backend(self) -> Self:
        """Ollama solo sabe un modo, y decir lo contrario no lo cambia.

        `OllamaBackend` restringe la generación con `format=<esquema>`, que es
        `json_schema`. No expone herramientas, así que `function_calling` no
        existe ahí; y `json_mode` sería superficie sin probar para nada que el
        respaldo local necesite. Aceptar la declaración y luego ignorarla dejaría
        una configuración que miente sobre lo que hace el sistema.
        """
        if self.backend is Backend.OLLAMA and self.structured_output is not (
            StructuredOutputMode.JSON_SCHEMA
        ):
            raise ValueError(
                f"backend ollama solo admite structured_output=json_schema, "
                f"declarado {self.structured_output.value}"
            )
        return self


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
    """Credenciales de un backend compatible con OpenAI y su corte de paciencia."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    api_key: SecretStr
    base_url: str | None = None

    timeout_seconds: float = Field(default=120.0, gt=0.0)
    """Cuánto se espera a una respuesta antes de darla por perdida.

    Se declara porque heredarlo es peor que no tenerlo: sin este valor el SDK de
    OpenAI aplica `Timeout(connect=5, read=600, write=600, pool=600)`, y un rol
    colgado retiene diez minutos por intento. El peor rol medido contra el
    gateway es `bull` con 60 s, así que 120 s deja el doble de margen sobre lo
    que hoy sí completa y sigue cortando un cuelgue dentro de la misma vela.
    """


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

    timeout_seconds: float = Field(default=300.0, gt=0.0)
    """Corte de paciencia con el servidor local.

    El cliente de Ollama no trae ninguno: su httpx sale con `Timeout(None)`, así
    que un servidor colgado cuelga la corrida entera sin producir una sola línea.
    El valor es holgado a propósito, porque aquí la espera legítima no es la
    latencia de un modelo sino la cola: Ollama serializa en la GPU, así que los
    tres técnicos del abanico se ejecutan uno detrás de otro.
    """


# ───────────────────────────────────────── Precios ───────────────────────────────────────────────


class PriceRow(BaseModel):
    """Lo que cuesta un modelo bajo una forma de pago, en USD por millón de tokens.

    `peak` distingue las dos tarifas de un modelo que cobra distinto según la hora
    (`True` y `False`); `None` es que no distingue. `monthly_limit_usd` es el límite
    con el que ese modelo consume el pool de la suscripción: a 60 un dólar gasta 1/60
    del pool, a 30 gasta el doble. Solo existe en `go`: con pago por uso no hay pool.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(min_length=1)
    """Identificador de la pasarela, el mismo que registra `LLMCall.model`."""

    billing: Billing
    peak: bool | None = None
    input_per_mtok: float = Field(ge=0.0)
    """Entrada que **no** vino de la caché."""

    cached_per_mtok: float = Field(ge=0.0)
    """Entrada servida desde la caché de prefijo."""

    output_per_mtok: float = Field(ge=0.0)
    """Salida, razonamiento incluido."""

    monthly_limit_usd: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def _the_pool_exists_only_in_the_subscription(self) -> Self:
        if self.billing is Billing.GO and self.monthly_limit_usd is None:
            raise ValueError(f"{self.model}: una tarifa go necesita su límite mensual")
        if self.billing is Billing.PAYG and self.monthly_limit_usd is not None:
            raise ValueError(f"{self.model}: el pago por uso no tiene pool ni límite mensual")
        return self


class PriceTable(BaseModel):
    """Precios de la página de OpenCode Go y lo que hace falta para leerlos.

    Vive en la configuración y no en un módulo: son datos con fecha, y cambiar uno tiene
    que ser un diff de aquí, no una constante enterrada en el código que calcula.
    `prices_as_of` es cuándo se copiaron: una tabla sin fecha no dice si todavía vale.

    `page_estimates` son las peticiones por ventana de 5 h que la página publica por
    modelo. Son una estimación de la página y no una medida: sirven para contrastar lo
    declarado en `quota_per_window`, no para sustituirlo.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    rows: tuple[PriceRow, ...]
    prices_as_of: date
    page_estimates: dict[str, int]
    five_hour_share: float = Field(default=0.20, gt=0.0, le=1.0)
    """Fracción del pool mensual que cabe en una ventana de 5 h."""

    payg_prices_as_of: date | None = None
    """Cuándo se copiaron las filas de pago por uso, si es otro día que `prices_as_of`.

    Las filas `go` salen de la página de OpenCode Go y las `payg` de la de Zen: dos páginas, dos
    fechas. `None` es que la tabla no distingue (las construidas a mano en las pruebas).
    """

    topup_fee_rate: float = Field(default=0.044, ge=0.0, lt=1.0)
    topup_fee_usd: float = Field(default=0.30, ge=0.0)
    """Comisión de recarga de Zen: «4.4% + $0.30 por transacción», pasada a coste (2026-10-04)."""

    peak_hours_utc: tuple[tuple[int, int], ...] = ((1, 4), (6, 10))
    """Franjas `[inicio, fin)` en horas UTC enteras. El fin es exclusivo."""

    peak_weekdays: tuple[int, ...] = (0, 1, 2, 3, 4)
    """Días en que rige el pico, con lunes = 0. Fines de semana no hay pico."""

    @model_validator(mode="after")
    def _the_table_is_coherent(self) -> Self:
        for start, end in self.peak_hours_utc:
            if not 0 <= start < end <= 24:
                raise ValueError(f"franja de horas de pico inválida: {start}-{end}")
        seen: dict[tuple[str, Billing], list[bool | None]] = {}
        for entry in self.rows:
            seen.setdefault((entry.model, entry.billing), []).append(entry.peak)
        for (model, billing), peaks in seen.items():
            if len(peaks) != len(set(peaks)):
                raise ValueError(f"{model} ({billing.value}): tarifa repetida")
            if None in peaks and len(peaks) > 1:
                raise ValueError(f"{model} ({billing.value}): mezcla tarifa de pico y plana")
            if None not in peaks and set(peaks) != {True, False}:
                raise ValueError(f"{model} ({billing.value}): falta la otra tarifa de pico")
        return self

    def as_of(self, billing: Billing) -> date:
        """La fecha de las filas de esa forma de pago."""
        if billing is Billing.PAYG and self.payg_prices_as_of is not None:
            return self.payg_prices_as_of
        return self.prices_as_of

    def topup_charge(self, credit_usd: float) -> float:
        """Lo que cuesta en tarjeta una recarga única que deja `credit_usd` de saldo.

        Es una lectura de «4.4% + $0.30 por transacción» (la comisión se suma al crédito, sobre el
        crédito), no un recibo: la primera recarga real dice si la página quiere decir esto.
        """
        return credit_usd * (1.0 + self.topup_fee_rate) + self.topup_fee_usd

    def is_peak(self, at: datetime) -> bool:
        """Si `at` cae en el pico, juzgado siempre en UTC.

        Es de la hora de la llamada, así que solo vale para llamadas vivas: un acierto
        de caché en un replay lleva el instante evaluado, no el de la petición.
        """
        moment = at.astimezone(UTC)
        if moment.weekday() not in self.peak_weekdays:
            return False
        return any(start <= moment.hour < end for start, end in self.peak_hours_utc)

    def price_for(self, model: str, billing: Billing, at: datetime) -> PriceRow | None:
        """La tarifa de ese modelo a esa hora, o `None` si la tabla no la tiene."""
        candidates = [r for r in self.rows if r.model == model and r.billing is billing]
        if not candidates:
            return None
        flat = next((r for r in candidates if r.peak is None), None)
        if flat is not None:
            return flat
        wanted = self.is_peak(at)
        return next(r for r in candidates if r.peak is wanted)


def _go(
    model: str, entry: float, cached: float, out: float, limit: float, peak: bool | None = None
) -> PriceRow:
    return PriceRow(
        model=model,
        billing=Billing.GO,
        peak=peak,
        input_per_mtok=entry,
        cached_per_mtok=cached,
        output_per_mtok=out,
        monthly_limit_usd=limit,
    )


def _payg(model: str, entry: float, cached: float, out: float) -> PriceRow:
    return PriceRow(
        model=model,
        billing=Billing.PAYG,
        input_per_mtok=entry,
        cached_per_mtok=cached,
        output_per_mtok=out,
    )


DEFAULT_PRICING = PriceTable(
    prices_as_of=date(2026, 10, 2),
    rows=(
        _go("mimo-v2.5", 0.14, 0.0028, 0.28, 60.0),
        _go("hy3", 0.14, 0.035, 0.58, 60.0),
        _go("kimi-k2.6", 0.95, 0.16, 4.00, 60.0),
        _go("minimax-m3", 0.30, 0.06, 1.20, 60.0),
        _go("glm-5.2", 1.40, 0.26, 4.40, 60.0),
        _go("deepseek-v4-flash", 0.15, 0.003, 0.60, 30.0, peak=False),
        _go("deepseek-v4-flash", 0.30, 0.006, 1.20, 30.0, peak=True),
        _payg("kimi-k2.6", 0.95, 0.16, 4.00),
        _payg("minimax-m3", 0.30, 0.06, 1.20),
        _payg("glm-5.2", 1.40, 0.26, 4.40),
        _payg("deepseek-v4-flash", 0.14, 0.028, 0.28),
        _payg("deepseek-v4.1-flash", 0.30, 0.006, 1.20),
        _payg("deepseek-v4-pro", 1.74, 0.145, 3.48),
        _payg("glm-5.3-flash", 0.15, 0.03, 0.50),
        _payg("minimax-m2.7", 0.30, 0.06, 1.20),
        _payg("kimi-k2.7-code", 0.95, 0.19, 4.00),
        _payg("qwen3.8-max", 2.00, 0.25, 6.00),
    ),
    payg_prices_as_of=date(2026, 10, 4),
    page_estimates={
        "mimo-v2.5": 30_100,
        "deepseek-v4-flash": 13_000,
        "hy3": 4_300,
        "kimi-k2.6": 1_150,
        "minimax-m3": 3_200,
        "glm-5.2": 880,
    },
)
"""Precios de la página de OpenCode Go (2026-10-02) y de la de OpenCode Zen (2026-10-04).

Con la suscripción, MiMo-V2.5 y Hy3 tienen precio y los pago-por-uso de Kimi K2.6, MiniMax M3 y
GLM-5.2 son iguales; DeepSeek V4 Flash tiene dos tarifas según el pico. En pago por uso, MiMo-V2.5
y Hy3 no existen. Las cuatro filas `payg` que ya estaban se contrastaron con la página de Zen el
2026-10-04 y no cambiaron; las seis nuevas son los candidatos a structure y volume, todas servidas
por `/chat/completions`.

`qwen3.8-max` tiene además un precio de escritura de caché (2.50 USD/Mtok) que `PriceRow` no
modela: el `usage` del proveedor informa lecturas, no escrituras. Si Zen cobra la escritura, el
coste medido de ese modelo es una cota inferior, y los informes lo dicen.
"""


# ─────────────────────────────────────── Costes de operar ─────────────────────────────────────────


class CostModel(BaseModel):
    """Lo que cuesta cruzar el mercado en perpetuos USDT de Binance, por lado y como fracción.

    Vive aquí, con fecha, por la misma razón que los precios: son datos que caducan, y cambiar
    uno tiene que ser un diff de la configuración y no una constante enterrada en el código que
    puntúa. Ningún otro módulo lleva un número de coste.

    Solo alimenta el retorno **neto**, que es descriptivo: los criterios de la enmienda 2 se
    siguen evaluando sobre el retorno bruto y este modelo no los toca.

    `taker_fee` está confirmado en la cuenta (VIP 0, sin descuento por BNB) a `as_of`.
    `slippage` es un **supuesto sin medir**, pendiente de confirmar con la cuenta: nadie ha
    comparado todavía el precio de referencia con el de ejecución. Una orden de mercado paga
    el taker y la salida, que es otra orden de mercado, lo paga otra vez.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    taker_fee: float = Field(
        default=0.0005,
        ge=0.0,
        lt=1.0,
        description="Comisión taker por lado, fracción del nominal. "
        "Confirmada en la cuenta: VIP 0, sin descuento BNB.",
    )

    slippage: float = Field(
        default=0.0002,
        ge=0.0,
        lt=1.0,
        description="Deslizamiento por lado, fracción del nominal. "
        "SUPUESTO sin medir, pendiente de confirmar con la cuenta.",
    )

    as_of: date = date(2026, 10, 3)
    """Cuándo se confirmó la comisión. El deslizamiento no está confirmado a ninguna fecha."""

    @property
    def round_trip(self) -> float:
        """Coste de entrar y salir: dos lados, cada uno con comisión y deslizamiento."""
        return 2.0 * (self.taker_fee + self.slippage)


DEFAULT_COSTS = CostModel()
"""Los costes por omisión: comisión taker confirmada el 2026-10-03, deslizamiento supuesto."""


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
    billing: Billing = Billing.GO
    """Cómo se paga el proveedor. Solo mide: no cambia la cuota ni los reintentos.

    Con `go` el consumo se expresa como fracción del pool de la suscripción; con `payg`,
    como dólares. La ablación lo deja escrito en el `meta.json` de la corrida.
    """

    pricing: PriceTable = DEFAULT_PRICING
    costs: CostModel = DEFAULT_COSTS
    """Comisión y deslizamiento de los perpetuos. Solo puntúan el retorno neto, descriptivo."""

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
