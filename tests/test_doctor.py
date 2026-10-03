"""Pruebas de las comprobaciones de arranque.

Todas las sondas son falsas: `doctor` existe para hablar con el exterior, así que
lo único que se puede probar en frío es que traduce cada respuesta —incluida una
excepción— a un resultado con el detalle que hace falta para arreglarlo.

El criterio que ordena el archivo: un id inexistente sale por la puerta de `FAIL`
nombrando rol e id, nunca por la de la traza.
"""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from crypto_agents.doctor import (
    CheckStatus,
    check_exchange,
    check_gateway,
    check_local_vram,
    check_modes,
    check_ollama,
    render,
)
from crypto_agents.llm import Completion, ResidentModel
from crypto_agents.settings import RoleConfig, Settings, load_settings
from crypto_agents.state import AgentRole, Backend, StructuredOutputMode
from tests.conftest import CHEAP, SCARCE, raw_ohlcv, role_map

if TYPE_CHECKING:
    from crypto_agents.settings import ModelChoice
    from crypto_agents.state import LLMOutput

GIB = 1024**3
NOW = datetime(2026, 8, 15, 12, 0, tzinfo=UTC)


def remote_settings(**overrides: object) -> Settings:
    """Los seis roles en el gateway, que es donde vive la comprobación cara."""
    kwargs: dict[str, object] = {
        "roles": role_map(primary=SCARCE),
        "openai": {"api_key": "sk-test", "base_url": "https://gateway.example/v1"},
        "exchange": {"exchange_id": "binance", "sandbox": True},
    }
    kwargs.update(overrides)
    return load_settings(**kwargs)


def local_settings(**overrides: object) -> Settings:
    """Los seis roles en local, para las comprobaciones de Ollama y VRAM."""
    kwargs: dict[str, object] = {
        "roles": role_map(primary=CHEAP),
        "ollama": {"host": "http://localhost:11434", "num_ctx": 4096},
    }
    kwargs.update(overrides)
    return load_settings(**kwargs)


class FakeCatalog:
    """Catálogo que devuelve lo que se le diga, o revienta."""

    def __init__(self, models: set[str] | None = None, error: Exception | None = None) -> None:
        self._models = frozenset(models or set())
        self._error = error

    async def available_models(self) -> frozenset[str]:
        if self._error is not None:
            raise self._error
        return self._models


class FakeLocal:
    """Backend local falso: una respuesta y un estado de residencia."""

    def __init__(
        self,
        response: str = '{"ok": true}',
        resident: tuple[ResidentModel, ...] = (),
        error: Exception | None = None,
    ) -> None:
        self._response = response
        self._resident = resident
        self._error = error
        self.prompts: list[str] = []

    async def complete(self, choice: ModelChoice, prompt: str, schema: type[LLMOutput]) -> str:
        del choice, schema
        if self._error is not None:
            raise self._error
        self.prompts.append(prompt)
        return self._response

    async def resident(self) -> tuple[ResidentModel, ...]:
        return self._resident


ENOUGH = 500
"""Velas que devuelve la fuente sana: por encima de los 400 que exige el preset."""


class FakeMarket:
    """Origen de velas falso. No sabe nada de credenciales, igual que el real."""

    def __init__(
        self,
        url: str = "https://api.example/api",
        candles: list[list[float]] | None = None,
        fetch_error: Exception | None = None,
    ) -> None:
        self._url = url
        self._candles = candles if candles is not None else raw_ohlcv([100.0] * ENOUGH)
        self._fetch_error = fetch_error
        self.asked: list[tuple[str, str, int]] = []

    @property
    def api_base_url(self) -> str:
        return self._url

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> list[list[float]]:
        self.asked.append((symbol, timeframe, limit))
        if self._fetch_error is not None:
            raise self._fetch_error
        return self._candles[:limit]


class FakeCredentials:
    """Cliente autenticado falso. No sabe leer velas, igual que el real."""

    def __init__(
        self,
        url: str = "https://testnet.example/api",
        credentials_error: Exception | None = None,
    ) -> None:
        self._url = url
        self._credentials_error = credentials_error
        self.verified = False

    @property
    def api_base_url(self) -> str:
        return self._url

    async def verify_credentials(self) -> None:
        self.verified = True
        if self._credentials_error is not None:
            raise self._credentials_error


def loaded(model: str, gpu_fraction: float = 1.0, size: int = 6 * GIB) -> ResidentModel:
    """Un modelo residente con el reparto GPU que se quiera."""
    return ResidentModel(
        name=model, size=size, size_vram=int(size * gpu_fraction), context_length=4096
    )


# ─────────────────────────────────────────────  gateway  ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_every_declared_id_present_passes() -> None:
    """Con los seis ids en el catálogo, la comprobación pasa y dice cuántos."""
    settings = remote_settings()
    catalog = FakeCatalog({SCARCE.model})

    result = await check_gateway(settings, catalog)

    assert result.status is CheckStatus.OK
    assert "6/6" in result.detail


@pytest.mark.asyncio
async def test_an_unknown_id_names_the_role_and_the_id() -> None:
    """El fallo que justifica el comando: un id que el gateway no sirve.

    Sin esto no se entera nadie hasta la primera llamada, a mitad de una
    evaluación que ya pagó las anteriores.
    """
    roles = role_map(primary=SCARCE)
    roles[AgentRole.STRUCTURE] = roles[AgentRole.STRUCTURE].model_copy(
        update={"primary": SCARCE.model_copy(update={"model": "modelo-que-no-existe"})}
    )
    settings = remote_settings(roles=roles)

    result = await check_gateway(settings, FakeCatalog({SCARCE.model}))

    assert result.status is CheckStatus.FAIL
    assert "modelo-que-no-existe" in result.detail
    assert "structure" in result.detail


@pytest.mark.asyncio
async def test_an_unreachable_catalog_is_a_failure_not_a_traceback() -> None:
    """Una excepción del cliente se traduce a detalle legible."""
    settings = remote_settings()

    result = await check_gateway(settings, FakeCatalog(error=ConnectionError("sin ruta al host")))

    assert result.status is CheckStatus.FAIL
    assert "ConnectionError" in result.detail
    assert "sin ruta al host" in result.detail


@pytest.mark.asyncio
async def test_without_remote_roles_the_gateway_check_is_vacuous() -> None:
    """Todo en local: no hay nada que comprobar contra un gateway."""
    result = await check_gateway(local_settings(), FakeCatalog())

    assert result.status is CheckStatus.OK


# ───────────────────────────────────────────────  modos  ──────────────────────────────────────────
# El modo de salida estructurada no se puede validar leyendo la configuración: el
# modelo contesta igual y lo que cambia es por dónde. Sondearlo cuesta llamadas y
# es lo que evita descubrirlo a mitad de una evaluación ya pagada.


ROOMY = SCARCE.model_copy(update={"quota_per_window": 100, "quota_weight": 1.0})
"""Como `SCARCE` pero con presupuesto de sobra.

Sondear tres modos son tres llamadas contra el mismo par (rol, modelo), y con la
cuota justa de `SCARCE` la búsqueda de alternativa se queda sin presupuesto antes
de terminar. Eso es un caso real y tiene su propia prueba; aquí estorba.
"""


class ModeAwareBackend:
    """Solo responde en los modos que se le declaren; en el resto, vacío.

    Refleja lo que hace un proveedor de verdad: `function_calling` sobre un modelo
    sin herramientas no da error, da una respuesta que el adaptador no puede leer.
    """

    def __init__(self, *supported: StructuredOutputMode) -> None:
        self.supported = frozenset(supported)
        self.seen: list[tuple[str, StructuredOutputMode]] = []

    async def complete(self, choice: ModelChoice, prompt: str, schema: type[LLMOutput]) -> str:
        """Devuelve el ping si el modo está soportado, y si no cadena vacía."""
        self.seen.append((choice.model, choice.structured_output))
        if choice.structured_output in self.supported:
            return '{"ok": true}'
        return ""


@pytest.mark.asyncio
async def test_every_role_reports_the_mode_that_was_verified() -> None:
    """El criterio de aceptación: qué modo quedó comprobado, rol por rol."""
    backend = ModeAwareBackend(StructuredOutputMode.JSON_SCHEMA)

    result = await check_modes(remote_settings(), {Backend.OPENAI: backend}, lambda: NOW)

    assert result.status is CheckStatus.OK
    for role in AgentRole:
        assert f"{role.value} json_schema" in result.detail
    assert len(backend.seen) == len(AgentRole)


class ConcurrencyTrackingBackend:
    """Anota cuántos sondeos hay en vuelo a la vez."""

    def __init__(self) -> None:
        self.inflight = 0
        self.peak = 0

    async def complete(self, choice: ModelChoice, prompt: str, schema: type[LLMOutput]) -> str:
        """Cede el control una vez, para que las tareas puedan solaparse de verdad."""
        del choice, prompt, schema
        self.inflight += 1
        self.peak = max(self.peak, self.inflight)
        await asyncio.sleep(0)
        self.inflight -= 1
        return '{"ok": true}'


class AlwaysRefusingBackend:
    """El proveedor rechaza la petición antes de generar nada. Un 429, un 401."""

    async def complete(self, choice: ModelChoice, prompt: str, schema: type[LLMOutput]) -> str:
        """Lanza como lanza un cliente HTTP."""
        del choice, prompt, schema
        raise RuntimeError("429 rate limit exceeded")


@pytest.mark.asyncio
async def test_the_six_roles_are_probed_at_the_same_time() -> None:
    """En serie el comando cuesta la suma de las seis latencias; en paralelo, la peor.

    Con los modelos configurados eso son 118 s contra 56, dominados por uno solo.
    Es seguro porque son seis peticiones independientes y cada tarea lleva su
    propio contador de cuota: no hay estado compartido que proteger.
    """
    backend = ConcurrencyTrackingBackend()

    result = await check_modes(remote_settings(), {Backend.OPENAI: backend}, lambda: NOW)

    assert result.status is CheckStatus.OK
    assert backend.peak == len(AgentRole)


@pytest.mark.asyncio
async def test_a_provider_rejection_is_not_reported_as_a_broken_mode() -> None:
    """Un rechazo de transporte no dice nada sobre el modo declarado.

    Con seis sondeos simultáneos, un límite de tasa del gateway es justo esta
    clase de falso negativo: mandaría a cambiar `STRUCTURED_OUTPUT` cuando lo que
    hay que hacer es esperar.
    """
    settings = remote_settings(roles=role_map(primary=ROOMY))

    result = await check_modes(settings, {Backend.OPENAI: AlwaysRefusingBackend()}, lambda: NOW)

    assert result.status is CheckStatus.FAIL
    assert "rechazado por el proveedor" in result.detail
    assert "no dio salida válida" not in result.detail


@pytest.mark.asyncio
async def test_a_provider_rejection_does_not_pay_for_a_mode_search() -> None:
    """Si el proveedor no llegó a generar nada, el modo declarado no quedó desmentido.

    Buscar alternativa serían dos rechazos más y dos llamadas tiradas por rol —
    doce en total justo cuando el gateway está diciendo que pares.
    """

    class CountingRefusal(AlwaysRefusingBackend):
        def __init__(self) -> None:
            self.attempts = 0

        async def complete(self, choice: ModelChoice, prompt: str, schema: type[LLMOutput]) -> str:
            self.attempts += 1
            return await super().complete(choice, prompt, schema)

    backend = CountingRefusal()
    settings = remote_settings(roles=role_map(primary=ROOMY))

    await check_modes(settings, {Backend.OPENAI: backend}, lambda: NOW)

    assert backend.attempts == len(AgentRole)


@pytest.mark.asyncio
async def test_the_verified_line_carries_the_latency() -> None:
    """Cuánto tardó cada rol es la mitad del valor de sondearlos: dice qué cuesta operar."""
    backend = ModeAwareBackend(StructuredOutputMode.JSON_SCHEMA)

    result = await check_modes(remote_settings(), {Backend.OPENAI: backend}, lambda: NOW)

    assert result.status is CheckStatus.OK
    assert re.search(r"decider json_schema \(\d+\.\d s\)", result.detail)


@pytest.mark.asyncio
async def test_a_declared_mode_that_does_not_work_names_the_role_and_exits_one() -> None:
    """Un modo equivocado tiene que salir por la misma puerta que un id equivocado."""
    roles = role_map(primary=ROOMY)
    roles[AgentRole.DECIDER] = RoleConfig(
        primary=ROOMY.model_copy(
            update={"structured_output": StructuredOutputMode.FUNCTION_CALLING}
        )
    )
    settings = remote_settings(roles=roles)
    backend = ModeAwareBackend(StructuredOutputMode.JSON_SCHEMA)

    result = await check_modes(settings, {Backend.OPENAI: backend}, lambda: NOW)

    assert result.status is CheckStatus.FAIL
    assert "decider" in result.detail
    assert "function_calling no dio salida válida" in result.detail


@pytest.mark.asyncio
async def test_the_failure_names_the_mode_that_does_work() -> None:
    """Un diagnóstico que no dice qué escribir deja el trabajo a medias.

    Es el caso real del reporte: el modelo estaba bien y el modo no, así que el
    detalle tiene que traer la variable ya redactada.
    """
    roles = role_map(primary=ROOMY)
    roles[AgentRole.DECIDER] = RoleConfig(
        primary=ROOMY.model_copy(update={"structured_output": StructuredOutputMode.JSON_MODE})
    )
    settings = remote_settings(roles=roles)
    backend = ModeAwareBackend(StructuredOutputMode.FUNCTION_CALLING)

    result = await check_modes(settings, {Backend.OPENAI: backend}, lambda: NOW)

    assert result.status is CheckStatus.FAIL
    assert "function_calling sí" in result.detail
    assert "CA_ROLES__DECIDER__PRIMARY__STRUCTURED_OUTPUT=function_calling" in result.detail


@pytest.mark.asyncio
async def test_a_model_that_answers_in_no_mode_says_so() -> None:
    """Sin ningún modo que funcione no hay variable que sugerir, y hay que decirlo."""
    backend = ModeAwareBackend()
    settings = remote_settings(roles=role_map(primary=ROOMY))

    result = await check_modes(settings, {Backend.OPENAI: backend}, lambda: NOW)

    assert result.status is CheckStatus.FAIL
    assert "ningún otro modo tampoco" in result.detail


@pytest.mark.asyncio
async def test_the_probe_does_not_fall_back_to_the_local_model() -> None:
    """Sondear el modo de un remoto y que conteste el respaldo local no comprueba nada.

    `resolve()` degrada en cuanto el primario no cabe en presupuesto, así que el
    sondeo corre sobre una copia sin respaldos: lo que se mide es el modelo
    declarado o nada.
    """
    exhausted = SCARCE.model_copy(update={"quota_per_window": 1, "quota_weight": 2.0})
    roles = role_map(primary=exhausted, fallback=CHEAP)
    roles[AgentRole.DECIDER] = RoleConfig(primary=exhausted)
    settings = remote_settings(roles=roles, ollama={"host": "http://localhost:11434"})
    remote = ModeAwareBackend(StructuredOutputMode.JSON_SCHEMA)
    local = ModeAwareBackend(StructuredOutputMode.JSON_SCHEMA)

    await check_modes(settings, {Backend.OPENAI: remote, Backend.OLLAMA: local}, lambda: NOW)

    assert local.seen == []


@pytest.mark.asyncio
async def test_the_probe_does_not_answer_from_cache() -> None:
    """Con caché, la segunda ejecución de `doctor` diría que sí a un proveedor apagado."""
    backend = ModeAwareBackend(StructuredOutputMode.JSON_SCHEMA)
    settings = remote_settings()

    await check_modes(settings, {Backend.OPENAI: backend}, lambda: NOW)
    await check_modes(settings, {Backend.OPENAI: backend}, lambda: NOW)

    assert len(backend.seen) == 2 * len(AgentRole)


class RetryOnlyBackend:
    """Solo acierta cuando el prompt ya trae adjunto el error del intento anterior."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    async def complete(self, choice: ModelChoice, prompt: str, schema: type[LLMOutput]) -> str:
        """Vacío a la primera; el ping solo si el router reintentó."""
        del choice, schema
        self.seen.append(prompt)
        return '{"ok": true}' if "no pasó la validación" in prompt else ""


@pytest.mark.asyncio
async def test_the_probe_does_not_retry() -> None:
    """Un modo que solo funciona al reintentar no es un modo que funcione.

    El reintento mide si el modelo se corrige cuando se le enseña el error, que
    es otra pregunta y además cuesta el doble. Aquí solo interesa si el modo
    declarado produce salida a la primera, que es como lo va a usar cada
    evaluación.
    """
    settings = remote_settings(roles=role_map(primary=ROOMY))

    result = await check_modes(settings, {Backend.OPENAI: RetryOnlyBackend()}, lambda: NOW)

    assert result.status is CheckStatus.FAIL
    assert "no dio salida válida" in result.detail


@pytest.mark.asyncio
async def test_running_out_of_budget_is_not_reported_as_a_broken_mode() -> None:
    """Sin cuota no se llegó a preguntar, y decir «no funciona» mandaría a cambiar
    una configuración que puede estar perfectamente bien.

    Con la cuota de `SCARCE` caben dos sondeos por rol: el declarado y uno más.
    El tercero ya no, así que la búsqueda se corta a media pregunta.
    """
    backend = ModeAwareBackend()

    result = await check_modes(remote_settings(), {Backend.OPENAI: backend}, lambda: NOW)

    assert result.status is CheckStatus.FAIL
    assert "sin cuota para sondear" in result.detail


@pytest.mark.asyncio
async def test_probing_a_local_only_configuration_costs_nothing() -> None:
    """Sin modelos remotos no hay nada que sondear ni cuota que gastar."""
    result = await check_modes(local_settings(), {}, lambda: NOW)

    assert result.status is CheckStatus.OK
    assert "sin modelos remotos" in result.detail


# ──────────────────────────────────────────────  ollama  ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_downloaded_tag_passes() -> None:
    """El tag configurado está entre los descargados."""
    result = await check_ollama(local_settings(), FakeCatalog({CHEAP.model}))

    assert result.status is CheckStatus.OK
    assert CHEAP.model in result.detail


@pytest.mark.asyncio
async def test_a_missing_tag_names_it_and_how_to_get_it() -> None:
    """El detalle trae el comando que lo arregla, no solo el diagnóstico."""
    result = await check_ollama(local_settings(), FakeCatalog({"otro:tag"}))

    assert result.status is CheckStatus.FAIL
    assert CHEAP.model in result.detail
    assert f"ollama pull {CHEAP.model}" in result.detail


@pytest.mark.asyncio
async def test_a_dead_server_names_the_host() -> None:
    """Si el servidor no responde, el detalle dice a qué host se intentó llegar."""
    result = await check_ollama(local_settings(), FakeCatalog(error=OSError("connection refused")))

    assert result.status is CheckStatus.FAIL
    assert "http://localhost:11434" in result.detail


# ───────────────────────────────────────────────  vram  ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_real_local_adapter_returns_a_completion_and_the_check_reads_its_text() -> None:
    """`OllamaBackend.complete()` devuelve `Completion` desde que registra tokens.

    La sonda de VRAM validaba lo que recibía como JSON: con un `Completion` en vez de
    texto, el chequeo habría fallado contra el backend real y pasado contra todos los falsos.
    """

    class CompletionLocal:
        def __init__(self) -> None:
            self.inner = FakeLocal(resident=(loaded(CHEAP.model),))

        async def complete(
            self, choice: ModelChoice, prompt: str, schema: type[LLMOutput]
        ) -> Completion:
            return Completion(text=await self.inner.complete(choice, prompt, schema))

        async def resident(self) -> tuple[ResidentModel, ...]:
            return await self.inner.resident()

    probe = CompletionLocal()

    result = await check_local_vram(local_settings(), probe)

    assert result.status is CheckStatus.OK


@pytest.mark.asyncio
async def test_a_model_entirely_on_the_gpu_passes() -> None:
    """100% GPU y una llamada que respeta el esquema."""
    probe = FakeLocal(resident=(loaded(CHEAP.model),))

    result = await check_local_vram(local_settings(), probe)

    assert result.status is CheckStatus.OK
    assert "100% GPU" in result.detail
    assert probe.prompts, "la comprobación tiene que llamar antes de leer `ps`"


@pytest.mark.asyncio
async def test_layers_on_the_cpu_name_the_variable_to_lower() -> None:
    """El reparto con CPU es un fallo, y el detalle dice qué bajar.

    Es la comprobación que evita medir latencias sobre un modelo que no cabe:
    la llamada funciona igual, solo que varias veces más lenta.
    """
    probe = FakeLocal(resident=(loaded(CHEAP.model, gpu_fraction=0.72),))

    result = await check_local_vram(local_settings(), probe)

    assert result.status is CheckStatus.FAIL
    assert "72% en GPU" in result.detail
    assert "CA_OLLAMA__NUM_CTX" in result.detail


@pytest.mark.asyncio
async def test_output_that_breaks_the_schema_is_a_failure() -> None:
    """Lo que se comprueba es la gramática JSON, no que el modelo hable."""
    probe = FakeLocal(response='{"ok": "sí"}', resident=(loaded(CHEAP.model),))

    result = await check_local_vram(local_settings(), probe)

    assert result.status is CheckStatus.FAIL
    assert "esquema" in result.detail


@pytest.mark.asyncio
async def test_a_model_that_answers_but_is_not_resident_is_a_failure() -> None:
    """Sin entrada en `ps` no hay reparto que comprobar, y callarlo sería peor."""
    probe = FakeLocal(resident=())

    result = await check_local_vram(local_settings(), probe)

    assert result.status is CheckStatus.FAIL
    assert "no aparece cargado" in result.detail


@pytest.mark.asyncio
async def test_a_local_model_that_does_not_answer_is_a_failure() -> None:
    """La excepción del cliente se convierte en detalle, no en traza."""
    probe = FakeLocal(error=TimeoutError("agotado"))

    result = await check_local_vram(local_settings(), probe)

    assert result.status is CheckStatus.FAIL
    assert "TimeoutError" in result.detail


# ─────────────────────────────────────────────  exchange  ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_happy_path_names_both_destinations() -> None:
    """Camino feliz: velas de producción, órdenes en testnet, credenciales aceptadas."""
    settings = remote_settings(
        exchange={
            "exchange_id": "binance",
            "sandbox": True,
            "api_key": "clave",
            "api_secret": "secreto",
        }
    )
    market, credentials = FakeMarket(), FakeCredentials()

    result = await check_exchange(
        settings, market, credentials, production_url="https://api.example/api"
    )

    assert result.status is CheckStatus.OK
    assert credentials.verified
    assert "https://api.example/api" in result.detail
    assert "https://testnet.example/api" in result.detail


@pytest.mark.asyncio
async def test_the_probe_asks_for_a_full_window_of_candles() -> None:
    """Pedir dos velas comprueba que el exchange contesta y nada más.

    La pregunta que importa es si esta fuente entrega histórico suficiente, y eso
    solo se ve pidiendo lo que pide una evaluación.
    """
    market = FakeMarket()

    await check_exchange(remote_settings(), market)

    assert market.asked == [("BTC/USDT", "4h", 500)]


@pytest.mark.asyncio
async def test_a_source_with_too_little_history_is_a_failure() -> None:
    """El fallo que mató la primera corrida, ahora en el arranque y no a mitad.

    Testnet contesta perfectamente y devuelve 58 velas de 4h contra un preset que
    exige 400: ninguna evaluación puede completarse con esa fuente, y hasta ahora
    nada lo decía antes de intentarlo.
    """
    market = FakeMarket(candles=raw_ohlcv([100.0] * 58))

    result = await check_exchange(remote_settings(), market)

    assert result.status is CheckStatus.FAIL
    assert "58 velas" in result.detail
    assert "400" in result.detail


@pytest.mark.asyncio
async def test_a_sandbox_that_did_not_change_the_url_is_a_failure() -> None:
    """`sandbox=true` es una intención; la URL de las órdenes es la consecuencia.

    Se comprueba sobre el cliente autenticado, que es al único al que esa variable
    le aplica desde que la lectura va siempre a producción.
    """
    settings = remote_settings(
        exchange={
            "exchange_id": "binance",
            "sandbox": True,
            "api_key": "clave",
            "api_secret": "secreto",
        }
    )
    production = "https://api.example/api"

    result = await check_exchange(
        settings, FakeMarket(), FakeCredentials(url=production), production_url=production
    )

    assert result.status is CheckStatus.FAIL
    assert "SANDBOX" in result.detail


@pytest.mark.asyncio
async def test_rejected_credentials_name_the_url_that_rejected_them() -> None:
    """Unas claves mal copiadas no dan señal leyendo velas: hace falta la privada."""
    from crypto_agents.market import MarketDataError

    settings = remote_settings(
        exchange={
            "exchange_id": "binance",
            "sandbox": True,
            "api_key": "clave",
            "api_secret": "mala",
        }
    )
    credentials = FakeCredentials(
        credentials_error=MarketDataError("AuthenticationError: firma inválida")
    )

    result = await check_exchange(
        settings, FakeMarket(), credentials, production_url="https://api.example/api"
    )

    assert result.status is CheckStatus.FAIL
    assert "firma inválida" in result.detail


@pytest.mark.asyncio
async def test_an_empty_candle_response_is_a_failure() -> None:
    """Un exchange que contesta con cero velas no sirve para evaluar nada."""
    result = await check_exchange(remote_settings(), FakeMarket(candles=[]))

    assert result.status is CheckStatus.FAIL
    assert "0 velas" in result.detail


@pytest.mark.asyncio
async def test_a_source_that_does_not_answer_is_a_failure_not_a_traceback() -> None:
    """La excepción del cliente se convierte en detalle con la URL dentro."""
    market = FakeMarket(fetch_error=ConnectionError("sin ruta al host"))

    result = await check_exchange(remote_settings(), market)

    assert result.status is CheckStatus.FAIL
    assert "ConnectionError" in result.detail
    assert "https://api.example/api" in result.detail


@pytest.mark.asyncio
async def test_live_without_credentials_is_a_failure() -> None:
    """En papel se puede vivir sin claves; en live es la condición de arranque."""
    settings = remote_settings(execution={"mode": "live"})

    result = await check_exchange(settings, FakeMarket())

    assert result.status is CheckStatus.FAIL
    assert "CA_EXCHANGE__API_KEY" in result.detail


@pytest.mark.asyncio
async def test_paper_without_credentials_passes() -> None:
    """Leer velas no necesita claves —ni puede usarlas— y el papel no manda órdenes."""
    result = await check_exchange(remote_settings(), FakeMarket())

    assert result.status is CheckStatus.OK
    assert "sin credenciales de órdenes" in result.detail


# ──────────────────────────────────────────────  salida  ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_report_names_every_check_with_its_result() -> None:
    """Una línea por comprobación: nombre, resultado y detalle."""
    settings = local_settings()
    results = (
        await check_gateway(settings, FakeCatalog()),
        await check_ollama(settings, FakeCatalog({CHEAP.model})),
        await check_local_vram(settings, FakeLocal(resident=(loaded(CHEAP.model),))),
    )

    text = render(results)

    assert len(text.splitlines()) == 3
    for name in ("gateway", "ollama", "vram"):
        assert name in text
