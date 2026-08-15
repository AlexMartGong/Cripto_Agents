"""Pruebas de las comprobaciones de arranque.

Todas las sondas son falsas: `doctor` existe para hablar con el exterior, así que
lo único que se puede probar en frío es que traduce cada respuesta —incluida una
excepción— a un resultado con el detalle que hace falta para arreglarlo.

El criterio que ordena el archivo: un id inexistente sale por la puerta de `FAIL`
nombrando rol e id, nunca por la de la traza.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from crypto_agents.doctor import (
    CheckStatus,
    check_exchange,
    check_gateway,
    check_local_vram,
    check_ollama,
    render,
)
from crypto_agents.llm import ResidentModel
from crypto_agents.settings import Settings, load_settings
from crypto_agents.state import AgentRole
from tests.conftest import CHEAP, SCARCE, raw_ohlcv, role_map

if TYPE_CHECKING:
    from crypto_agents.settings import ModelChoice
    from crypto_agents.state import LLMOutput

GIB = 1024**3


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


class FakeExchange:
    """Cliente de mercado falso, con URL y credenciales controladas."""

    def __init__(
        self,
        url: str = "https://testnet.example/api",
        candles: list[list[float]] | None = None,
        credentials_error: Exception | None = None,
        fetch_error: Exception | None = None,
    ) -> None:
        self._url = url
        self._candles = candles if candles is not None else raw_ohlcv([100.0, 101.0])
        self._credentials_error = credentials_error
        self._fetch_error = fetch_error
        self.verified = False

    @property
    def api_base_url(self) -> str:
        return self._url

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> list[list[float]]:
        del symbol, timeframe
        if self._fetch_error is not None:
            raise self._fetch_error
        return self._candles[:limit]

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
async def test_a_sandbox_that_answers_with_valid_credentials_passes() -> None:
    """Camino feliz: URL de testnet, velas y credenciales aceptadas."""
    settings = remote_settings(
        exchange={
            "exchange_id": "binance",
            "sandbox": True,
            "api_key": "clave",
            "api_secret": "secreto",
        }
    )
    client = FakeExchange()

    result = await check_exchange(settings, client, production_url="https://api.example/api")

    assert result.status is CheckStatus.OK
    assert client.verified


@pytest.mark.asyncio
async def test_a_sandbox_that_did_not_change_the_url_is_a_failure() -> None:
    """`sandbox=true` es una intención; la URL es la consecuencia.

    Si ccxt no cambia de destino para este exchange, se estaría operando contra
    producción creyendo lo contrario.
    """
    settings = remote_settings(exchange={"exchange_id": "binance", "sandbox": True})
    production = "https://api.example/api"

    result = await check_exchange(settings, FakeExchange(url=production), production_url=production)

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
    client = FakeExchange(credentials_error=MarketDataError("AuthenticationError: firma inválida"))

    result = await check_exchange(settings, client, production_url="https://api.example/api")

    assert result.status is CheckStatus.FAIL
    assert "firma inválida" in result.detail


@pytest.mark.asyncio
async def test_an_empty_candle_response_is_a_failure() -> None:
    """Un exchange que contesta con cero velas no sirve para evaluar nada."""
    settings = remote_settings()

    result = await check_exchange(settings, FakeExchange(candles=[]))

    assert result.status is CheckStatus.FAIL
    assert "cero velas" in result.detail


@pytest.mark.asyncio
async def test_live_without_credentials_is_a_failure() -> None:
    """En papel se puede vivir sin claves; en live es la condición de arranque."""
    settings = remote_settings(execution={"mode": "live"})

    result = await check_exchange(settings, FakeExchange())

    assert result.status is CheckStatus.FAIL
    assert "CA_EXCHANGE__API_KEY" in result.detail


@pytest.mark.asyncio
async def test_paper_without_credentials_passes() -> None:
    """Leer velas no necesita claves, y el modo papel no manda órdenes."""
    result = await check_exchange(remote_settings(), FakeExchange())

    assert result.status is CheckStatus.OK
    assert "sin credenciales" in result.detail


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
