"""Comprobaciones de arranque: ¿puede este sistema operar, antes de gastar nada?

Cinco preguntas, cada una respondida contra el sistema real y no contra la
configuración que lo describe:

- los seis identificadores remotos existen en el catálogo del gateway;
- cada modelo responde de verdad en el modo de salida estructurada declarado;
- el servidor local responde y tiene descargado el tag del respaldo;
- ese modelo cabe entero en la GPU con el `num_ctx` configurado;
- el exchange contesta, respeta el modo sandbox y acepta las credenciales.

Las dos primeras son las que justifican el comando, y por la misma razón. Un id
equivocado no falla al cargar la configuración: falla en la primera llamada, a
mitad de una evaluación que ya pagó las anteriores. Un modo equivocado tampoco —
el modelo contesta, pero por un canal que el adaptador no está leyendo, y lo que
se ve es una salida vacía atribuida al modelo. Las dos son configuración que solo
el proveedor puede confirmar.

Ninguna sonda deja escapar una excepción: cada una la convierte en un `FAIL` con
el mensaje dentro, porque una traza de ccxt o de httpx no le dice a nadie qué
variable de entorno tiene que tocar.

Sobre las llamadas de prueba, dos regímenes distintos y a propósito:

- **La local no pasa por `ModelRouter`** y por tanto no emite `LLMCall`. La regla
  4 existe para que ninguna llamada *de una evaluación* quede sin registrar; esta
  no pertenece a ninguna, es local y gratuita, y su registro es la línea que el
  comando imprime.
- **Las remotas sí van por el router.** Cuestan cuota de verdad, así que el
  argumento anterior no las cubre: cada intento emite su `LLMCall` y el contador
  lo descuenta. `crypto-agents doctor` deja de ser un comando gratis — gasta seis
  llamadas por invocación, más dos por cada rol que falle *por contenido*
  mientras se busca el modo que sí funciona. Un rechazo de transporte no paga esa
  búsqueda: si el proveedor no generó nada, el modo declarado no quedó desmentido.

Los seis sondeos de modo corren a la vez; el resto de comprobaciones, en serie.
La diferencia está en qué comparten: seis peticiones al mismo gateway no
comparten nada, mientras que la sonda de VRAM carga un modelo en la GPU y su
latencia es justo el número que interesa.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

from pydantic import Field, ValidationError

from crypto_agents.execution import ExecutionMode
from crypto_agents.indicators import DEFAULT_PRESET
from crypto_agents.llm import (
    Completion,
    ModelCallError,
    ModelInvocationError,
    ModelRouter,
    OllamaBackend,
    OpenAIBackend,
    as_completion,
    build_backends,
)
from crypto_agents.market import CcxtMarketClient, CcxtTradingClient, MarketDataError
from crypto_agents.quota import QuotaExhaustedError, QuotaLedger
from crypto_agents.settings import ENV_PREFIX, RoleConfig
from crypto_agents.state import AgentRole, Backend, FrozenModel, LLMOutput, StructuredOutputMode

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from crypto_agents.llm import ChatBackend, ResidentModel
    from crypto_agents.quota import Clock
    from crypto_agents.settings import ModelChoice, Settings
    from crypto_agents.state import LLMCall

_NESTED = "__"
"""Separador de claves anidadas, el mismo que usa `Settings`."""

__all__ = [
    "CheckResult",
    "CheckStatus",
    "CredentialProbe",
    "LocalProbe",
    "MarketProbe",
    "check_exchange",
    "check_gateway",
    "check_local_vram",
    "check_modes",
    "check_ollama",
    "probe_router",
    "probe_settings",
    "render",
    "run_checks",
]

_PROBE_PROMPT = 'Responde solo con este JSON, sin añadir nada: {"ok": true}'

_PROBE_CANDLES = 500
"""Velas que se piden al sondear, las mismas que `AgentContext.candle_limit`.

Pedir dos, como antes, comprobaba que el exchange contesta y nada más. La
pregunta que importa es si esta fuente entrega histórico suficiente para el
preset: testnet contesta perfectamente y devuelve 58 velas de 4h.
"""
_GIB = 1024**3

_MODE_ORDER = (
    StructuredOutputMode.JSON_SCHEMA,
    StructuredOutputMode.FUNCTION_CALLING,
    StructuredOutputMode.JSON_MODE,
)
"""Orden en que se buscan alternativas cuando el modo declarado no funciona.

De más restrictivo a menos: `json_schema` obliga a la forma, `function_calling`
solo a que exista la herramienta, `json_mode` únicamente a que sea JSON. Si dos
funcionan conviene quedarse con el primero, porque el que menos exige es el que
más trabajo deja al validador —y a los reintentos que ese validador provoca.
"""


class CheckStatus(StrEnum):
    """Resultado de una comprobación."""

    OK = "ok"
    FAIL = "fail"


class CheckResult(FrozenModel):
    """Lo que dio una comprobación, con su detalle en texto.

    El detalle no es decorativo: es lo único que separa «algo falla» de «falta
    `CA_OLLAMA__NUM_CTX` más bajo», y es lo que se lee en un terminal a las tres
    de la mañana.
    """

    name: str = Field(min_length=1)
    status: CheckStatus
    detail: str = Field(min_length=1)

    @property
    def ok(self) -> bool:
        """Si la comprobación pasó."""
        return self.status is CheckStatus.OK


class Ping(LLMOutput):
    """Esquema mínimo para la llamada de prueba local.

    Se usa un esquema de verdad y no texto libre porque lo que se comprueba no es
    que el modelo hable, sino que obedece una gramática JSON: es como lo llamará
    el router en cada evaluación.
    """

    ok: bool


class LocalProbe(Protocol):
    """Lo que el chequeo de VRAM necesita del backend local."""

    async def complete(
        self, choice: ModelChoice, prompt: str, schema: type[LLMOutput]
    ) -> str | Completion:
        """Una llamada real, con esquema."""
        ...

    async def resident(self) -> tuple[ResidentModel, ...]:
        """Lo que hay cargado ahora mismo."""
        ...


class Catalog(Protocol):
    """Lo que los chequeos de catálogo necesitan de un proveedor."""

    async def available_models(self) -> frozenset[str]:
        """Identificadores que el proveedor declara servir."""
        ...


class MarketProbe(Protocol):
    """Lo que el chequeo necesita del origen de datos. No incluye credenciales.

    Que este protocolo no declare `verify_credentials` es la mitad del punto: el
    cliente de lectura no tiene claves que comprobar, y no puede tenerlas.
    """

    @property
    def api_base_url(self) -> str:
        """URL efectiva. En el cliente de lectura, siempre producción."""
        ...

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> list[list[float]]:
        """Velas crudas."""
        ...


class CredentialProbe(Protocol):
    """Lo que el chequeo necesita del cliente autenticado. No sabe leer velas."""

    @property
    def api_base_url(self) -> str:
        """URL efectiva tras aplicar el modo sandbox."""
        ...

    async def verify_credentials(self) -> None:
        """Llamada privada de lectura."""
        ...


def _declared(settings: Settings, backend: Backend) -> dict[AgentRole, list[ModelChoice]]:
    """Roles que usan un backend, con las elecciones concretas que lo usan."""
    declared: dict[AgentRole, list[ModelChoice]] = {}
    for role in settings.roles:
        choices = [choice for choice in settings.role_choices(role) if choice.backend is backend]
        if choices:
            declared[role] = choices
    return declared


# ────────────────────────────────────────── Comprobaciones ────────────────────────────────────────


async def check_gateway(settings: Settings, catalog: Catalog | None = None) -> CheckResult:
    """Los ids remotos existen en el catálogo del gateway."""
    name = "gateway"
    wanted = _declared(settings, Backend.OPENAI)
    if not wanted:
        return CheckResult(name=name, status=CheckStatus.OK, detail="sin roles remotos declarados")
    if settings.openai is None:
        return CheckResult(
            name=name,
            status=CheckStatus.FAIL,
            detail="hay roles con backend openai y falta CA_OPENAI__API_KEY",
        )

    if catalog is None:
        backend = build_backends(settings).get(Backend.OPENAI)
        if not isinstance(backend, OpenAIBackend):
            return CheckResult(
                name=name, status=CheckStatus.FAIL, detail="el backend openai no expone catálogo"
            )
        catalog = backend

    try:
        available = await catalog.available_models()
    except Exception as error:  # httpx, openai y el gateway tienen jerarquías propias
        return CheckResult(
            name=name,
            status=CheckStatus.FAIL,
            detail=f"catálogo inaccesible: {type(error).__name__}: {error}",
        )

    missing = sorted(
        f"{role.value} → {choice.model}"
        for role, choices in wanted.items()
        for choice in choices
        if choice.model not in available
    )
    total = sum(len(choices) for choices in wanted.values())
    if missing:
        return CheckResult(
            name=name,
            status=CheckStatus.FAIL,
            detail=f"ids que el gateway no sirve: {', '.join(missing)}",
        )
    return CheckResult(
        name=name,
        status=CheckStatus.OK,
        detail=f"{total}/{total} ids presentes en {settings.openai.base_url or 'api.openai.com'}",
    )


def probe_settings(
    settings: Settings, overrides: Mapping[AgentRole, ModelChoice] | None = None
) -> Settings:
    """Copia donde cada rol es exactamente un modelo y ninguno tiene respaldo.

    El respaldo se quita porque `QuotaLedger.resolve()` degradaría al modelo local
    en cuanto el remoto no cupiera, y entonces el sondeo diría que el modo del
    remoto funciona cuando quien contestó fue otro modelo entero.
    """
    primaries = {role: settings.role_config(role).primary for role in settings.roles}
    if overrides:
        primaries.update(overrides)
    return settings.model_copy(
        update={"roles": {role: RoleConfig(primary=choice) for role, choice in primaries.items()}}
    )


def probe_router(
    settings: Settings,
    ledger: QuotaLedger,
    backends: Mapping[Backend, ChatBackend],
    clock: Clock,
) -> ModelRouter:
    """Router para sondear, no para operar: sin caché y de un solo intento.

    Sin caché porque una entrada guardada haría que la segunda ejecución de
    `doctor` no comprobara nada y respondiera que sí a un proveedor apagado. De un
    solo intento porque un reintento mide si el modelo se corrige, que es otra
    pregunta: aquí solo interesa si el modo declarado produce salida a la primera.
    """
    return ModelRouter(settings, ledger, backends, clock, cache=None, max_attempts=1)


class _ProbeOutcome(FrozenModel):
    """Qué pasó al sondear un modo concreto.

    `transport` separa «el proveedor rechazó la petición» de «respondió y no
    valida», que es la distinción que la primera tabla de modelos no tenía. Con
    seis sondeos a la vez importa más que antes: un 429 del gateway no puede
    leerse como «este modo no funciona» y mandar a cambiar una configuración
    correcta.
    """

    ok: bool
    transport: bool = False
    latency_ms: float = 0.0


class _RoleReport(FrozenModel):
    """Lo que el sondeo averiguó sobre un rol."""

    role: AgentRole
    model: str = Field(min_length=1)
    declared: StructuredOutputMode
    verified: bool
    transport: bool = False
    working: StructuredOutputMode | None = None
    """Modo alternativo que sí responde, cuando el declarado falló."""

    latency_ms: float = 0.0
    exhausted: str | None = None
    """Mensaje si el sondeo se quedó sin cuota antes de poder concluir."""


async def _answers_in_mode(
    settings: Settings,
    role: AgentRole,
    choice: ModelChoice,
    backends: Mapping[Backend, ChatBackend],
    clock: Clock,
    spent: list[LLMCall],
) -> _ProbeOutcome:
    """Una llamada mínima con esquema. Acumula lo gastado, funcione o no.

    El contador se reconstruye con `extend()` desde lo ya gastado por *este* rol
    en vez de crearse limpio: cada modo probado necesita su propia copia de la
    configuración, y sin rehidratar, tres sondeos del mismo modelo creerían cada
    uno ser el primero. Entre roles no se comparte nada, y es correcto: la cuota
    se lleva por par (rol, modelo), así que un rol no consume la del otro.
    """
    probe = probe_settings(settings, {role: choice})
    ledger = QuotaLedger(probe.quota_window, clock)
    ledger.extend(spent)
    router = probe_router(probe, ledger, backends, clock)
    try:
        _, calls = await router.invoke(role, _PROBE_PROMPT, Ping)
    except QuotaExhaustedError:
        raise  # quedarse sin presupuesto no es que el modo falle: es no haber preguntado
    except ModelInvocationError as error:
        spent.extend(error.calls)
        return _ProbeOutcome(
            ok=False,
            transport=isinstance(error, ModelCallError),
            latency_ms=_total_latency(error.calls),
        )
    except Exception:  # backend ausente, o lo que traiga el cliente
        return _ProbeOutcome(ok=False, transport=True)
    spent.extend(calls)
    return _ProbeOutcome(ok=True, latency_ms=_total_latency(calls))


def _total_latency(calls: Sequence[LLMCall]) -> float:
    """Lo que tardó el sondeo, sumando sus intentos."""
    return sum(call.latency_ms for call in calls)


async def _first_working_mode(
    settings: Settings,
    role: AgentRole,
    choice: ModelChoice,
    backends: Mapping[Backend, ChatBackend],
    clock: Clock,
    spent: list[LLMCall],
) -> StructuredOutputMode | None:
    """Qué modo sí funciona con ese modelo, si alguno.

    Solo se ejecuta cuando el declarado ya falló: convierte un «no funciona» en
    la línea de configuración que hay que escribir, que es la diferencia entre un
    diagnóstico y una tarea. Los modos de un mismo rol se prueban en serie porque
    comparten el par (rol, modelo) del que se descuenta la cuota.
    """
    for mode in _MODE_ORDER:
        if mode is choice.structured_output:
            continue
        candidate = choice.model_copy(update={"structured_output": mode})
        if (await _answers_in_mode(settings, role, candidate, backends, clock, spent)).ok:
            return mode
    return None


async def _probe_role(
    settings: Settings,
    role: AgentRole,
    choice: ModelChoice,
    backends: Mapping[Backend, ChatBackend],
    clock: Clock,
) -> _RoleReport:
    """Sondea un rol de principio a fin. Es la unidad que corre en paralelo.

    Cada rol lleva su propia lista de gasto, así que no hay estado compartido
    entre las tareas concurrentes — que es lo que hace seguro lanzarlas juntas.
    """
    spent: list[LLMCall] = []
    base = _RoleReport(
        role=role, model=choice.model, declared=choice.structured_output, verified=False
    )
    try:
        outcome = await _answers_in_mode(settings, role, choice, backends, clock, spent)
        if outcome.ok:
            return base.model_copy(update={"verified": True, "latency_ms": outcome.latency_ms})
        if outcome.transport:
            # El proveedor rechazó la petición antes de generar nada, así que el
            # modo declarado no ha quedado desmentido. Buscar alternativa serían
            # dos rechazos más y dos llamadas tiradas.
            return base.model_copy(update={"transport": True, "latency_ms": outcome.latency_ms})
        working = await _first_working_mode(settings, role, choice, backends, clock, spent)
    except QuotaExhaustedError as error:
        # Sin presupuesto no se puede afirmar que el modo falle: no se llegó a
        # preguntar. Decir «ningún modo funciona» mandaría a cambiar una
        # configuración que puede estar bien.
        return base.model_copy(update={"exhausted": str(error)})
    return base.model_copy(
        update={
            "working": working,
            "transport": outcome.transport,
            "latency_ms": outcome.latency_ms,
        }
    )


async def check_modes(
    settings: Settings,
    backends: Mapping[Backend, ChatBackend] | None = None,
    clock: Clock | None = None,
) -> CheckResult:
    """Cada modelo remoto responde de verdad en el modo declarado.

    Es la comprobación que la configuración no puede dar: el modo no se rechaza
    al cargar ni al llamar. El modelo contesta, pero por un canal que el
    adaptador no lee, y el síntoma es una salida vacía atribuida al modelo —a
    mitad de una evaluación que ya pagó las llamadas anteriores.

    **Los seis roles se sondean a la vez.** En serie cuesta la suma de las seis
    latencias y en paralelo la del más lento: medido en el escritorio contra el
    gateway, 12.9 s de sondeos se convierten en 3.3 s. Es seguro porque son seis
    peticiones HTTP independientes contra el mismo gateway, sin GPU de por medio
    y sin contador compartido: la cuota se lleva por par (rol, modelo) y cada
    tarea tiene el suyo. Es lo contrario de `run_checks`, que sí va en serie
    porque la sonda de VRAM carga un modelo en la GPU y su latencia es el dato.
    """
    name = "modes"
    remote = _declared(settings, Backend.OPENAI)
    if not remote:
        return CheckResult(
            name=name, status=CheckStatus.OK, detail="sin modelos remotos que sondear"
        )
    if settings.openai is None:
        return CheckResult(
            name=name,
            status=CheckStatus.FAIL,
            detail="hay roles con backend openai y falta CA_OPENAI__API_KEY",
        )

    if backends is None:
        backends = build_backends(settings)
    at: Clock = clock if clock is not None else (lambda: datetime.now(UTC))

    reports = await asyncio.gather(
        *(
            _probe_role(settings, role, choice, backends, at)
            for role in sorted(remote, key=lambda item: item.value)
            for choice in remote[role]
        )
    )

    exhausted = [report for report in reports if report.exhausted is not None]
    if exhausted:
        return CheckResult(
            name=name,
            status=CheckStatus.FAIL,
            detail=f"sin cuota para sondear {exhausted[0].role.value}: {exhausted[0].exhausted}",
        )

    verified = [
        f"{report.role.value} {report.declared.value} ({report.latency_ms / 1000:.1f} s)"
        for report in reports
        if report.verified
    ]
    failures = [_mode_failure(report) for report in reports if not report.verified]

    if failures:
        return CheckResult(name=name, status=CheckStatus.FAIL, detail="; ".join(failures))
    return CheckResult(name=name, status=CheckStatus.OK, detail=", ".join(verified))


def _mode_failure(report: _RoleReport) -> str:
    """El fallo de un rol, con la variable que hay que cambiar si hay arreglo.

    Un rechazo de transporte se nombra como tal. Decir «este modo no da salida
    válida» cuando el proveedor no llegó a generar nada manda a cambiar una
    configuración que puede estar correcta — y con seis sondeos simultáneos, un
    límite de tasa del gateway es exactamente esa clase de falso negativo.
    """
    head = f"{report.role.value} → {report.model}: {report.declared.value}"
    if report.transport:
        return f"{head} rechazado por el proveedor antes de producir contenido"
    head = f"{head} no dio salida válida"
    if report.working is None:
        return f"{head} y ningún otro modo tampoco"
    variable = (
        f"{ENV_PREFIX}ROLES{_NESTED}{report.role.value.upper()}"
        f"{_NESTED}PRIMARY{_NESTED}STRUCTURED_OUTPUT"
    )
    return f"{head}; {report.working.value} sí → {variable}={report.working.value}"


async def check_ollama(settings: Settings, catalog: Catalog | None = None) -> CheckResult:
    """El servidor local responde y tiene descargados los tags de los respaldos."""
    name = "ollama"
    wanted = _declared(settings, Backend.OLLAMA)
    if not wanted:
        return CheckResult(name=name, status=CheckStatus.OK, detail="sin respaldos locales")
    if settings.ollama is None:
        return CheckResult(
            name=name,
            status=CheckStatus.FAIL,
            detail="hay roles con backend ollama y falta CA_OLLAMA__HOST",
        )

    if catalog is None:
        backend = build_backends(settings).get(Backend.OLLAMA)
        if not isinstance(backend, OllamaBackend):
            return CheckResult(
                name=name, status=CheckStatus.FAIL, detail="el backend ollama no expone catálogo"
            )
        catalog = backend

    try:
        available = await catalog.available_models()
    except Exception as error:  # el servidor puede no estar levantado
        return CheckResult(
            name=name,
            status=CheckStatus.FAIL,
            detail=f"{settings.ollama.host} no responde: {type(error).__name__}: {error}",
        )

    tags = sorted({choice.model for choices in wanted.values() for choice in choices})
    missing = [tag for tag in tags if tag not in available]
    if missing:
        pulls = "; ".join(f"ollama pull {tag}" for tag in missing)
        return CheckResult(
            name=name,
            status=CheckStatus.FAIL,
            detail=f"tags sin descargar en {settings.ollama.host}: {', '.join(missing)} ({pulls})",
        )
    descargado = "descargado" if len(tags) == 1 else "descargados"
    return CheckResult(
        name=name,
        status=CheckStatus.OK,
        detail=f"{', '.join(tags)} {descargado} en {settings.ollama.host}",
    )


async def check_local_vram(settings: Settings, probe: LocalProbe | None = None) -> CheckResult:
    """Una llamada real al respaldo local y después el reparto GPU/CPU.

    El orden importa: `ps` solo informa de lo que está cargado, así que sin la
    llamada previa la lista viene vacía y la comprobación no comprueba nada.
    """
    name = "vram"
    wanted = _declared(settings, Backend.OLLAMA)
    if not wanted or settings.ollama is None:
        return CheckResult(name=name, status=CheckStatus.OK, detail="sin modelo local que medir")

    choice = next(choice for choices in wanted.values() for choice in choices)
    if probe is None:
        backend = build_backends(settings).get(Backend.OLLAMA)
        if not isinstance(backend, OllamaBackend):
            return CheckResult(
                name=name, status=CheckStatus.FAIL, detail="el backend ollama no admite sondeo"
            )
        probe = backend

    started = time.perf_counter()
    try:
        raw = as_completion(await probe.complete(choice, _PROBE_PROMPT, Ping)).text
        Ping.model_validate_json(raw)
    except ValidationError as error:
        return CheckResult(
            name=name,
            status=CheckStatus.FAIL,
            detail=f"{choice.model} no respetó el esquema JSON: {error.error_count()} error(es)",
        )
    except Exception as error:
        return CheckResult(
            name=name,
            status=CheckStatus.FAIL,
            detail=f"{choice.model} no respondió: {type(error).__name__}: {error}",
        )
    elapsed = time.perf_counter() - started

    try:
        running = await probe.resident()
    except Exception as error:
        return CheckResult(
            name=name,
            status=CheckStatus.FAIL,
            detail=f"no se pudo leer el estado del servidor: {type(error).__name__}: {error}",
        )

    loaded = next((model for model in running if model.name == choice.model), None)
    if loaded is None:
        return CheckResult(
            name=name,
            status=CheckStatus.FAIL,
            detail=f"{choice.model} respondió pero no aparece cargado; sin reparto que comprobar",
        )

    size = loaded.size / _GIB
    if not loaded.fully_on_gpu:
        return CheckResult(
            name=name,
            status=CheckStatus.FAIL,
            detail=(
                f"{choice.model} {size:.1f} GiB, solo {loaded.gpu_fraction:.0%} en GPU con "
                f"num_ctx={settings.ollama.num_ctx}: baja CA_OLLAMA__NUM_CTX o usa un modelo "
                f"más pequeño antes de medir latencias"
            ),
        )
    return CheckResult(
        name=name,
        status=CheckStatus.OK,
        detail=(
            f"{choice.model} {size:.1f} GiB, 100% GPU con "
            f"num_ctx={settings.ollama.num_ctx} ({elapsed:.1f} s)"
        ),
    )


async def check_exchange(
    settings: Settings,
    market: MarketProbe | None = None,
    credentials: CredentialProbe | None = None,
    production_url: str | None = None,
) -> CheckResult:
    """La fuente de datos entrega histórico suficiente y las claves de órdenes valen.

    Dos clientes porque son dos preguntas. La primera se le hace al origen de
    velas, que va a producción y sin credenciales; la segunda al cliente
    autenticado, que es el que respeta `sandbox`. Sondear las dos con un solo
    cliente es lo que hacía que las claves de testnet rompieran una lectura
    pública y que testnet acabara siendo la fuente de datos.
    """
    name = "exchange"
    exchange = settings.exchange
    owned_market: CcxtMarketClient | None = None
    owned_credentials: CcxtTradingClient | None = None

    if market is None:
        try:
            owned_market = CcxtMarketClient(exchange.exchange_id)
        except MarketDataError as error:
            return CheckResult(name=name, status=CheckStatus.FAIL, detail=str(error))
        market = owned_market

    try:
        symbol, timeframe = _probe_market(settings)
        try:
            candles = await market.fetch_ohlcv(symbol, timeframe, _PROBE_CANDLES)
        except Exception as error:
            return CheckResult(
                name=name,
                status=CheckStatus.FAIL,
                detail=f"{symbol} no responde en {market.api_base_url}: "
                f"{type(error).__name__}: {error}",
            )

        required = DEFAULT_PRESET.min_bars
        if len(candles) < required:
            return CheckResult(
                name=name,
                status=CheckStatus.FAIL,
                detail=(
                    f"{symbol} {timeframe} devolvió {len(candles)} velas en "
                    f"{market.api_base_url} y el preset exige {required}: con esta fuente "
                    f"ninguna evaluación puede completarse"
                ),
            )

        has_keys = exchange.api_key is not None and exchange.api_secret is not None
        if not has_keys:
            if settings.execution.mode is ExecutionMode.LIVE:
                return CheckResult(
                    name=name,
                    status=CheckStatus.FAIL,
                    detail="modo live sin CA_EXCHANGE__API_KEY ni CA_EXCHANGE__API_SECRET",
                )
            return CheckResult(
                name=name,
                status=CheckStatus.OK,
                detail=(
                    f"{exchange.exchange_id} lee {len(candles)} velas de {symbol} en "
                    f"{market.api_base_url}; sin credenciales de órdenes declaradas"
                ),
            )

        if credentials is None:
            try:
                owned_credentials = CcxtTradingClient(exchange)
            except MarketDataError as error:
                return CheckResult(name=name, status=CheckStatus.FAIL, detail=str(error))
            credentials = owned_credentials

        if (
            exchange.sandbox
            and production_url is not None
            and credentials.api_base_url == production_url
        ):
            return CheckResult(
                name=name,
                status=CheckStatus.FAIL,
                detail=(
                    f"CA_EXCHANGE__SANDBOX=true pero las órdenes de {exchange.exchange_id} "
                    f"siguen apuntando a {production_url}: ccxt no cambió de URL para este "
                    f"exchange"
                ),
            )

        try:
            await credentials.verify_credentials()
        except MarketDataError as error:
            return CheckResult(
                name=name,
                status=CheckStatus.FAIL,
                detail=f"credenciales rechazadas por {credentials.api_base_url}: {error}",
            )
        return CheckResult(
            name=name,
            status=CheckStatus.OK,
            detail=(
                f"{exchange.exchange_id} lee {len(candles)} velas de {symbol} en "
                f"{market.api_base_url}; órdenes contra {credentials.api_base_url} "
                f"(sandbox={str(exchange.sandbox).lower()}) con credenciales válidas"
            ),
        )
    finally:
        if owned_market is not None:
            await owned_market.close()
        if owned_credentials is not None:
            await owned_credentials.close()


def _probe_market(settings: Settings) -> tuple[str, str]:
    """Símbolo y timeframe con los que sondear el exchange."""
    if settings.runner is not None and settings.runner.symbols:
        return settings.runner.symbols[0], settings.runner.timeframe
    return "BTC/USDT", "4h"


# ─────────────────────────────────────────── Orquestación ─────────────────────────────────────────


async def run_checks(settings: Settings) -> tuple[CheckResult, ...]:
    """Las cinco comprobaciones, en orden de coste creciente.

    Secuenciales a propósito: la de VRAM carga un modelo en la GPU y la del
    exchange abre una sesión HTTP; lanzarlas a la vez mezclaría sus latencias y la
    de la llamada local es justo el número que interesa.

    `modes` va justo después de `gateway` porque depende de él: sondear el modo de
    un id que el gateway no sirve gasta seis llamadas para redescubrir lo que la
    comprobación anterior ya dijo.
    """
    production_url: str | None = None
    if settings.exchange.sandbox:
        # Sobre el cliente autenticado: es el único al que `sandbox` le aplica ya.
        live = CcxtTradingClient(settings.exchange.model_copy(update={"sandbox": False}))
        try:
            production_url = live.api_base_url
        finally:
            await live.close()

    gateway = await check_gateway(settings)
    modes = (
        await check_modes(settings)
        if gateway.ok
        else CheckResult(
            name="modes",
            status=CheckStatus.FAIL,
            detail="no sondeado: el catálogo del gateway no cuadra",
        )
    )
    return (
        gateway,
        modes,
        await check_ollama(settings),
        await check_local_vram(settings),
        await check_exchange(settings, production_url=production_url),
    )


def render(results: Sequence[CheckResult]) -> str:
    """Una línea por comprobación, con su resultado y su detalle."""
    width = max((len(result.name) for result in results), default=0)
    return "\n".join(
        f"{result.name:<{width}}  {'OK' if result.ok else 'FALLA':<5}  {result.detail}"
        for result in results
    )
