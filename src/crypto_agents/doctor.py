"""Comprobaciones de arranque: ¿puede este sistema operar, antes de gastar nada?

Cuatro preguntas, cada una respondida contra el sistema real y no contra la
configuración que lo describe:

- los seis identificadores remotos existen en el catálogo del gateway;
- el servidor local responde y tiene descargado el tag del respaldo;
- ese modelo cabe entero en la GPU con el `num_ctx` configurado;
- el exchange contesta, respeta el modo sandbox y acepta las credenciales.

La primera es la que justifica el comando. Un id equivocado no falla al cargar la
configuración: falla en la primera llamada, a mitad de una evaluación que ya pagó
las anteriores. Comprobarlo cuesta una petición al catálogo.

Ninguna sonda deja escapar una excepción: cada una la convierte en un `FAIL` con
el mensaje dentro, porque una traza de ccxt o de httpx no le dice a nadie qué
variable de entorno tiene que tocar.

Sobre la llamada de prueba local: no pasa por `ModelRouter` y por tanto no emite
`LLMCall`. La regla 4 existe para que ninguna llamada *de una evaluación* quede
sin registrar; esta no pertenece a ninguna, es local y gratuita, y su registro es
la línea que el comando imprime. Si alguna vez hace falta sondear un modelo
remoto de esta forma, deja de ser cierto y hay que enrutarla.
"""

from __future__ import annotations

import time
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

from pydantic import Field, ValidationError

from crypto_agents.execution import ExecutionMode
from crypto_agents.llm import OllamaBackend, OpenAIBackend, build_backends
from crypto_agents.market import CcxtMarketClient, MarketDataError
from crypto_agents.state import Backend, FrozenModel, LLMOutput

if TYPE_CHECKING:
    from collections.abc import Sequence

    from crypto_agents.llm import ResidentModel
    from crypto_agents.settings import ModelChoice, Settings
    from crypto_agents.state import AgentRole

__all__ = [
    "CheckResult",
    "CheckStatus",
    "ExchangeProbe",
    "LocalProbe",
    "check_exchange",
    "check_gateway",
    "check_local_vram",
    "check_ollama",
    "render",
    "run_checks",
]

_PROBE_PROMPT = 'Responde solo con este JSON, sin añadir nada: {"ok": true}'
_GIB = 1024**3


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

    async def complete(self, choice: ModelChoice, prompt: str, schema: type[LLMOutput]) -> str:
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


class ExchangeProbe(Protocol):
    """Lo que el chequeo de exchange necesita del cliente de mercado."""

    @property
    def api_base_url(self) -> str:
        """URL efectiva tras aplicar el modo sandbox."""
        ...

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> list[list[float]]:
        """Velas crudas."""
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
        raw = await probe.complete(choice, _PROBE_PROMPT, Ping)
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
    client: ExchangeProbe | None = None,
    production_url: str | None = None,
) -> CheckResult:
    """El exchange contesta, respeta el sandbox y acepta las credenciales."""
    name = "exchange"
    exchange = settings.exchange
    owned: CcxtMarketClient | None = None
    if client is None:
        try:
            owned = CcxtMarketClient(exchange)
        except MarketDataError as error:
            return CheckResult(name=name, status=CheckStatus.FAIL, detail=str(error))
        client = owned

    try:
        if (
            exchange.sandbox
            and production_url is not None
            and client.api_base_url == production_url
        ):
            return CheckResult(
                name=name,
                status=CheckStatus.FAIL,
                detail=(
                    f"CA_EXCHANGE__SANDBOX=true pero {exchange.exchange_id} sigue apuntando a "
                    f"{production_url}: ccxt no cambió de URL para este exchange"
                ),
            )

        symbol, timeframe = _probe_market(settings)
        try:
            candles = await client.fetch_ohlcv(symbol, timeframe, 2)
        except Exception as error:
            return CheckResult(
                name=name,
                status=CheckStatus.FAIL,
                detail=f"{symbol} no responde en {client.api_base_url}: "
                f"{type(error).__name__}: {error}",
            )
        if not candles:
            return CheckResult(
                name=name,
                status=CheckStatus.FAIL,
                detail=f"{symbol} devolvió cero velas en {client.api_base_url}",
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
                    f"{exchange.exchange_id} en {client.api_base_url}, {symbol} responde; "
                    f"sin credenciales declaradas (leer velas no las necesita)"
                ),
            )

        try:
            await client.verify_credentials()
        except MarketDataError as error:
            return CheckResult(
                name=name,
                status=CheckStatus.FAIL,
                detail=f"credenciales rechazadas por {client.api_base_url}: {error}",
            )
        return CheckResult(
            name=name,
            status=CheckStatus.OK,
            detail=(
                f"{exchange.exchange_id} en {client.api_base_url} "
                f"(sandbox={str(exchange.sandbox).lower()}), {symbol} responde, "
                f"credenciales válidas"
            ),
        )
    finally:
        if owned is not None:
            await owned.close()


def _probe_market(settings: Settings) -> tuple[str, str]:
    """Símbolo y timeframe con los que sondear el exchange."""
    if settings.runner is not None and settings.runner.symbols:
        return settings.runner.symbols[0], settings.runner.timeframe
    return "BTC/USDT", "4h"


# ─────────────────────────────────────────── Orquestación ─────────────────────────────────────────


async def run_checks(settings: Settings) -> tuple[CheckResult, ...]:
    """Las cuatro comprobaciones, en orden de coste creciente.

    Secuenciales a propósito: la de VRAM carga un modelo en la GPU y la del
    exchange abre una sesión HTTP; lanzarlas a la vez mezclaría sus latencias y la
    de la llamada local es justo el número que interesa.
    """
    production_url: str | None = None
    if settings.exchange.sandbox:
        live = CcxtMarketClient(settings.exchange.model_copy(update={"sandbox": False}))
        try:
            production_url = live.api_base_url
        finally:
            await live.close()

    return (
        await check_gateway(settings),
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
