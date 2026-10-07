"""Quién respondió de verdad: el upstream que la pasarela declara en cada respuesta.

`LLMCall.model` es el id que se pidió. La pasarela lo enruta a un proveedor y a un modelo aguas
arriba que pueden cambiar sin que el id cambie: el 2026-10-06 `glm-5.2` lo servía `fireworks` con
`accounts/fireworks/models/glm-5p3` (`var/zen-probe/20261006T061405Z`). Sin esa columna dos brazos
de la ablación pueden estar comparando modelos distintos sin que nada lo registre.

Lo que se comprueba aquí pasa por el `ChatOpenAI` real y el SDK real de OpenAI: lo único simulado
es el transporte. La prueba que importa es la de concurrencia —cada intento lleva las cabeceras de
**su** respuesta—, y lleva su mutación: un adaptador que tomara las de la última respuesta vista,
que es lo que hace un hook global de httpx, la hace fallar.
"""

from __future__ import annotations
import __future__ as future_flags

import asyncio
import hashlib
import inspect
import json
import textwrap
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import httpx
import pytest

import crypto_agents.llm as llm_module
from crypto_agents.cache import (
    CacheEntry,
    InMemoryResponseCache,
    JsonFileResponseCache,
    cache_key,
)
from crypto_agents.journal import EvaluationRecord, JsonlJournal
from crypto_agents.llm import (
    UPSTREAM_ENDPOINT_HEADER,
    UPSTREAM_MODEL_HEADER,
    Completion,
    InvalidModelOutputError,
    ModelRouter,
    OpenAIBackend,
    Upstream,
    prompt_digest,
    upstream_from_headers,
    upstream_from_message,
)
from crypto_agents.quota import QuotaLedger
from crypto_agents.replay import RUN_DIGEST_VERSION, replay_run_id, run_digest
from crypto_agents.settings import Backend, ModelChoice, RoleConfig, Settings, load_settings
from crypto_agents.state import (
    UPSTREAM_FIELDS,
    AgentRole,
    Dimension,
    FailureKind,
    LLMCall,
    StructuredOutputMode,
    TechnicalVerdict,
)
from tests.conftest import DATA_DIR, role_map, verdict_payload

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Mapping
    from pathlib import Path

    from crypto_agents.cache import ResponseCache

    Handler = (
        Callable[[httpx.Request], httpx.Response]
        | Callable[[httpx.Request], Coroutine[None, None, httpx.Response]]
    )

START = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)
MODEL_A, MODEL_B = "modelo-a", "modelo-b"
ROLE_OF = {MODEL_A: AgentRole.STRUCTURE, MODEL_B: AgentRole.MOMENTUM}
DIMENSION_OF = {MODEL_A: Dimension.STRUCTURE, MODEL_B: Dimension.MOMENTUM}

SECRETS = {
    "set-cookie": "sesion=galleta-que-no-debe-salir",
    "authorization": "Bearer credencial-que-no-debe-salir",
    "x-request-id": "peticion-que-no-debe-salir",
}
"""Cabeceras que una respuesta real trae y que no son de nadie más: ninguna llega al journal."""


def upstream_of(model: str) -> Upstream:
    """Lo que la pasarela simulada declara para ese id: distinto por modelo, a propósito."""
    return Upstream(model=f"proveedor/{model}-0813", endpoint=f"ruta-{model}")


def headers_of(model: str) -> dict[str, str]:
    declared = upstream_of(model)
    assert declared.model is not None
    assert declared.endpoint is not None
    return {
        UPSTREAM_MODEL_HEADER: declared.model,
        UPSTREAM_ENDPOINT_HEADER: declared.endpoint,
        **SECRETS,
    }


def body_of(
    model: str, text: str, mode: StructuredOutputMode, tool: str = TechnicalVerdict.__name__
) -> dict[str, object]:
    """La respuesta del proveedor con ese texto, por el canal que corresponde al modo."""
    message: dict[str, object]
    if mode is StructuredOutputMode.FUNCTION_CALLING:
        call = {
            "id": "c",
            "type": "function",
            "function": {"name": tool, "arguments": text},
        }
        message = {"role": "assistant", "content": None, "tool_calls": [call]}
        finish = "tool_calls"
    else:
        message = {"role": "assistant", "content": text}
        finish = "stop"
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "finish_reason": finish, "message": message}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


class MockedBackend(OpenAIBackend):
    """El adaptador de producción con el transporte simulado.

    Sobrescribe `_http_client()`, la misma costura que `_headers()`: todo lo demás —el
    `ChatOpenAI`, el SDK de OpenAI, la lectura de cabeceras— es el código que corre de verdad.
    """

    def __init__(self, handler: Handler) -> None:
        super().__init__(
            api_key="clave-de-prueba", base_url="https://zen.invalid/v1", timeout_seconds=5.0
        )
        self._mock = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    def _http_client(self) -> object | None:
        return self._mock


class Clock:
    def __init__(self) -> None:
        self.now = START

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now


def settings_for(mode: StructuredOutputMode) -> Settings:
    """Dos roles remotos con ids distintos, los dos en el modo pedido."""

    def remote(model: str, family: str) -> ModelChoice:
        return ModelChoice(
            backend=Backend.OPENAI,
            model=model,
            family=family,
            structured_output=mode,
            quota_per_window=1000,
        )

    roles = role_map()
    roles[AgentRole.STRUCTURE] = RoleConfig(primary=remote(MODEL_A, "familia-a"))
    roles[AgentRole.MOMENTUM] = RoleConfig(primary=remote(MODEL_B, "familia-b"))
    return load_settings(
        roles=roles, openai={"api_key": "sk-test"}, ollama={"host": "http://localhost:11434"}
    )


def router_with(
    backend: object,
    mode: StructuredOutputMode = StructuredOutputMode.JSON_SCHEMA,
    cache: ResponseCache | None = None,
    max_attempts: int = 2,
) -> ModelRouter:
    settings = settings_for(mode)
    clock = Clock()
    return ModelRouter(
        settings,
        QuotaLedger(settings.quota_window, clock),
        {Backend.OPENAI: backend},  # type: ignore[dict-item]
        clock,
        cache,
        max_attempts,
    )


def answering(
    mode: StructuredOutputMode, headers: Callable[[str], Mapping[str, str]] = headers_of
) -> Callable[[httpx.Request], httpx.Response]:
    """Un transporte que contesta un veredicto válido con las cabeceras de ese modelo."""

    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        return httpx.Response(
            200,
            headers=dict(headers(model)),
            json=body_of(model, verdict_payload(DIMENSION_OF[model]), mode),
        )

    return handler


def carried(call: LLMCall) -> Upstream:
    return Upstream(model=call.upstream_model, endpoint=call.upstream_endpoint)


# ───────────────────────────────────── Leer las dos cabeceras ─────────────────────────────────────


def test_only_the_two_headers_are_read() -> None:
    found = upstream_from_headers({**headers_of(MODEL_A), "x-otra": "valor"})

    assert found == upstream_of(MODEL_A)
    assert set(Upstream.model_fields) == {"model", "endpoint"}


def test_the_header_names_are_matched_without_case() -> None:
    """httpx las entrega en minúsculas; un proxy intermedio puede no hacerlo."""
    found = upstream_from_headers(
        {UPSTREAM_MODEL_HEADER.upper(): "m", UPSTREAM_ENDPOINT_HEADER.title(): "e"}
    )

    assert found == Upstream(model="m", endpoint="e")


def test_one_header_alone_is_kept() -> None:
    assert upstream_from_headers({UPSTREAM_MODEL_HEADER: "m"}) == Upstream(model="m")
    assert upstream_from_headers({UPSTREAM_ENDPOINT_HEADER: "e"}) == Upstream(endpoint="e")


@pytest.mark.parametrize(
    "headers",
    [
        None,
        "texto",
        7,
        {},
        {"x-request-id": "r"},
        {UPSTREAM_MODEL_HEADER: "", UPSTREAM_ENDPOINT_HEADER: "   "},
        {UPSTREAM_MODEL_HEADER: 7},
    ],
)
def test_no_header_is_none_and_not_an_error(headers: object) -> None:
    """Sin cabecera no hay upstream que anotar. Una cadena vacía no es un nombre."""
    assert upstream_from_headers(headers) is None


@pytest.mark.parametrize("message", [None, "texto", 7, {}])
def test_a_result_that_is_not_a_message_has_no_upstream(message: object) -> None:
    assert upstream_from_message(message) is None


def test_the_header_value_is_kept_as_it_came() -> None:
    """`glm-5.2` contestó con `glm-5p3` en la cabecera: se copia, no se interpreta."""
    found = upstream_from_headers(
        {
            UPSTREAM_MODEL_HEADER: "accounts/fireworks/models/glm-5p3",
            UPSTREAM_ENDPOINT_HEADER: "fireworks",
        }
    )

    assert found == Upstream(model="accounts/fireworks/models/glm-5p3", endpoint="fireworks")


# ─────────────────────────────── Del transporte al `LLMCall` ──────────────────────────────────────


@pytest.mark.parametrize("mode", list(StructuredOutputMode))
@pytest.mark.asyncio
async def test_the_call_carries_the_headers_of_its_response(mode: StructuredOutputMode) -> None:
    """En los tres modos: la cabecera viaja en el mensaje, no en el canal del contenido."""
    router = router_with(MockedBackend(answering(mode)), mode)

    verdict, calls = await router.invoke(AgentRole.STRUCTURE, "analiza", TechnicalVerdict)

    (call,) = calls
    assert verdict.dimension is Dimension.STRUCTURE
    assert call.valid
    assert call.model == MODEL_A
    assert carried(call) == upstream_of(MODEL_A)
    assert (call.prompt_tokens, call.completion_tokens) == (11, 7)


@pytest.mark.asyncio
async def test_a_response_without_the_headers_is_a_valid_attempt_with_none() -> None:
    mode = StructuredOutputMode.JSON_SCHEMA
    router = router_with(MockedBackend(answering(mode, lambda _model: SECRETS)), mode)

    _, (call,) = await router.invoke(AgentRole.STRUCTURE, "analiza", TechnicalVerdict)

    assert call.valid
    assert carried(call) == Upstream()


@pytest.mark.asyncio
async def test_an_answer_the_sdk_rejects_still_carries_its_upstream() -> None:
    """Bajo `json_schema` el SDK valida dentro de `parse()` y el mensaje no sobrevive.

    Es la ruta en la que el adaptador ya perdía el `usage`. La respuesta sigue colgada de la
    excepción —es la de esa llamada—, así que el intento inválido dice quién lo contestó, que es
    justo el intento que se quiere poder atribuir.
    """
    mode = StructuredOutputMode.JSON_SCHEMA

    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        return httpx.Response(
            200, headers=headers_of(model), json=body_of(model, '{"dimension": "structure"}', mode)
        )

    router = router_with(MockedBackend(handler), mode, max_attempts=1)

    with pytest.raises(InvalidModelOutputError) as failure:
        await router.invoke(AgentRole.STRUCTURE, "analiza", TechnicalVerdict)

    (call,) = failure.value.calls
    assert call.failure_kind is FailureKind.SCHEMA
    assert carried(call) == upstream_of(MODEL_A)
    assert call.prompt_tokens is None  # el uso sigue perdiéndose en esta ruta: no cambió


@pytest.mark.asyncio
async def test_a_provider_failure_has_no_upstream() -> None:
    """Sin contenido no hay `Completion`: el intento queda sin cabecera, como queda sin tokens."""

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(404, headers=headers_of(MODEL_A), json={"error": {"message": "no"}})

    router = router_with(MockedBackend(handler))

    with pytest.raises(llm_module.ModelCallError) as failure:
        await router.invoke(AgentRole.STRUCTURE, "analiza", TechnicalVerdict)

    (call,) = failure.value.calls
    assert call.failure_kind is FailureKind.TRANSPORT
    assert carried(call) == Upstream()


@pytest.mark.asyncio
async def test_a_backend_that_only_returns_text_has_no_upstream() -> None:
    """Los falsos, el de solo-caché y Ollama: no hay pasarela que declare nada."""

    class TextOnly:
        async def complete(self, choice: ModelChoice, prompt: str, schema: type) -> str:
            del choice, prompt, schema
            return verdict_payload(Dimension.STRUCTURE)

    _, (call,) = await router_with(TextOnly()).invoke(
        AgentRole.STRUCTURE, "analiza", TechnicalVerdict
    )

    assert call.valid
    assert carried(call) == Upstream()


# ─────────────────────────────── Cada intento, las suyas ──────────────────────────────────────────


class Released:
    """Un transporte que retiene las dos respuestas hasta que las dos peticiones han llegado.

    Así las dos respuestas se entregan seguidas, antes de que ninguna de las dos llamadas termine
    de procesar la suya. Con una sola respuesta en vuelo, «la última vista» y «la mía» coinciden
    siempre y la prueba no distinguiría un adaptador correcto de uno que las confunde.
    """

    def __init__(self, mode: StructuredOutputMode) -> None:
        self._mode = mode
        self._both = asyncio.Event()
        self.arrived: list[str] = []
        self.last_seen: Upstream | None = None

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        self.arrived.append(model)
        if len(self.arrived) == len(ROLE_OF):
            self._both.set()
        await self._both.wait()
        self.last_seen = upstream_of(model)
        return httpx.Response(
            200,
            headers=headers_of(model),
            json=body_of(model, verdict_payload(DIMENSION_OF[model]), self._mode),
        )


async def assert_each_call_carries_its_own_headers(
    build: Callable[[Released], OpenAIBackend],
) -> None:
    """Dos llamadas a la vez, a dos modelos con cabeceras distintas: cada fila lleva las suyas."""
    mode = StructuredOutputMode.JSON_SCHEMA
    transport = Released(mode)
    router = router_with(build(transport), mode)

    (_, first), (_, second) = await asyncio.gather(
        router.invoke(AgentRole.STRUCTURE, "analiza", TechnicalVerdict),
        router.invoke(AgentRole.MOMENTUM, "analiza", TechnicalVerdict),
    )

    assert sorted(transport.arrived) == [MODEL_A, MODEL_B]
    for call in (*first, *second):
        assert carried(call) == upstream_of(call.model), (
            f"{call.model} lleva las cabeceras de otra respuesta: {carried(call)}"
        )


@pytest.mark.asyncio
async def test_two_concurrent_calls_keep_their_own_headers() -> None:
    await assert_each_call_carries_its_own_headers(MockedBackend)


def mutated_method(
    function: Callable[..., Any], old: str, new: str, extra: Mapping[str, object]
) -> Callable[..., Any]:
    """El método con un fragmento cambiado, compilado en el espacio de `llm.py` más `extra`.

    Se compila con las anotaciones diferidas, como el módulo: `ModelChoice` solo existe allí bajo
    `TYPE_CHECKING`, y evaluarla al definir la función fallaría por algo que no es la mutación.
    """
    source = textwrap.dedent(inspect.getsource(function))
    assert source.count(old) == 1, f"el fragmento {old!r} ya no está una sola vez en el código"
    namespace: dict[str, Any] = {**vars(llm_module), **extra}
    code = compile(
        source.replace(old, new),
        "<mutante>",
        "exec",
        flags=future_flags.annotations.compiler_flag,
    )
    exec(code, namespace)
    return namespace[function.__name__]  # type: ignore[no-any-return]


@pytest.mark.asyncio
async def test_taking_the_headers_of_the_last_response_seen_fails_the_check() -> None:
    """La mutación: un adaptador que lee «la última respuesta» en vez de la de su llamada.

    Es lo que hace un hook de httpx sin correlación, que fue como se miraron las cabeceras a mano
    en los bloques T4 y T5. Con una llamada a la vez no se nota; con dos, una fila se queda con
    las cabeceras de la otra.
    """

    def build(transport: Released) -> OpenAIBackend:
        class LastSeen(MockedBackend):
            complete = mutated_method(
                OpenAIBackend.complete,
                "upstream=upstream_from_message(raw)",
                "upstream=TRANSPORT.last_seen",
                {"TRANSPORT": transport},
            )

        return LastSeen(transport)

    with pytest.raises(AssertionError, match="lleva las cabeceras de otra respuesta"):
        await assert_each_call_carries_its_own_headers(build)


# ───────────────────────────────── Nada más llega al journal ──────────────────────────────────────


@pytest.mark.asyncio
async def test_no_other_header_reaches_the_journal(tmp_path: Path) -> None:
    """La respuesta trae `set-cookie`, `authorization` y `x-request-id`: ninguna se escribe."""
    mode = StructuredOutputMode.FUNCTION_CALLING
    router = router_with(MockedBackend(answering(mode)), mode)
    _, calls = await router.invoke(AgentRole.STRUCTURE, "analiza", TechnicalVerdict)

    path = tmp_path / "journal.jsonl"
    JsonlJournal(path).write(
        EvaluationRecord(
            run_id=replay_run_id("BTC/USDT", "4h", START),
            at=START,
            symbol="BTC/USDT",
            timeframe="4h",
            calls=tuple(calls),
        )
    )
    written = path.read_text(encoding="utf-8").lower()

    assert f'"upstream_model":"{upstream_of(MODEL_A).model}"' in path.read_text(encoding="utf-8")
    for name, value in SECRETS.items():
        assert name not in written
        assert value.lower() not in written
    assert set(LLMCall.model_fields) >= set(UPSTREAM_FIELDS)
    assert not {field for field in LLMCall.model_fields if "header" in field}


def test_the_completion_carries_an_upstream_and_no_headers() -> None:
    """Lo que cruza del adaptador al router son dos nombres, no el diccionario de cabeceras."""
    assert set(Completion.model_fields) == {"text", "usage", "upstream"}


# ─────────────────────────────────────────── Caché ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_cache_hit_reports_the_upstream_of_the_stored_entry() -> None:
    """Un acierto no hizo petición: lo que dice es quién contestó lo que está guardado."""
    mode = StructuredOutputMode.JSON_SCHEMA
    cache = InMemoryResponseCache()
    router = router_with(MockedBackend(answering(mode)), mode, cache)

    _, (live,) = await router.invoke(AgentRole.STRUCTURE, "analiza", TechnicalVerdict)
    _, (hit,) = await router.invoke(AgentRole.STRUCTURE, "analiza", TechnicalVerdict)

    assert (live.cache_hit, hit.cache_hit) == (False, True)
    assert carried(live) == carried(hit) == upstream_of(MODEL_A)
    (stored,) = (CacheEntry.model_validate_json(raw) for raw in cache.snapshot().values())
    assert (stored.upstream_model, stored.upstream_endpoint) == (
        upstream_of(MODEL_A).model,
        upstream_of(MODEL_A).endpoint,
    )


@pytest.mark.asyncio
async def test_an_entry_written_before_the_field_gives_none() -> None:
    """Las entradas viejas no llevan el campo: cargan, y el acierto no inventa un upstream."""
    mode = StructuredOutputMode.JSON_SCHEMA
    digest = prompt_digest("analiza")
    old = {
        "backend": "openai",
        "model": MODEL_A,
        "role": "structure",
        "prompt_digest": digest,
        "schema_name": TechnicalVerdict.__name__,
        "structured_output": mode.value,
        "raw": verdict_payload(Dimension.STRUCTURE),
        "valid": True,
        "failure_kind": None,
        "failure_message": None,
    }
    cache = InMemoryResponseCache()
    cache.set(cache_key(Backend.OPENAI, MODEL_A, digest, TechnicalVerdict, mode), json.dumps(old))

    def unreachable(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"un acierto de caché no llama: {request.url}")

    router = router_with(MockedBackend(unreachable), mode, cache)
    _, (hit,) = await router.invoke(AgentRole.STRUCTURE, "analiza", TechnicalVerdict)

    assert hit.cache_hit
    assert carried(hit) == Upstream()


def assert_no_secret_in(text: str, where: str) -> None:
    """Ni el nombre ni el valor de ninguna de las cabeceras que no son las dos de upstream."""
    lowered = text.lower()
    for name, value in SECRETS.items():
        assert name not in lowered, f"{where} lleva la cabecera {name}"
        assert value.lower() not in lowered, f"{where} lleva el valor de {name}"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", list(StructuredOutputMode))
async def test_no_other_header_reaches_the_cache_on_disk(
    mode: StructuredOutputMode, tmp_path: Path
) -> None:
    """La entrada guardada lleva dos nombres de upstream; nada más de la respuesta HTTP.

    Se mira el archivo que queda en disco, no el objeto: es lo que alguien copia, sube o adjunta.
    """
    cache = JsonFileResponseCache(tmp_path)
    router = router_with(MockedBackend(answering(mode)), mode, cache)

    await router.invoke(AgentRole.STRUCTURE, "analiza", TechnicalVerdict)

    (stored,) = sorted(tmp_path.glob("*.json"))
    written = stored.read_text(encoding="utf-8")
    assert_no_secret_in(written, stored.name)
    entry = CacheEntry.model_validate_json(written)
    assert (entry.upstream_model, entry.upstream_endpoint) == (
        upstream_of(MODEL_A).model,
        upstream_of(MODEL_A).endpoint,
    )
    assert not {field for field in CacheEntry.model_fields if "header" in field}
    assert set(json.loads(written)) == set(CacheEntry.model_fields)


@pytest.mark.asyncio
async def test_no_other_header_reaches_the_cache_when_the_sdk_rejects_the_answer(
    tmp_path: Path,
) -> None:
    """La ruta en que las cabeceras se leen de la excepción: también ahí salen solo dos nombres."""
    mode = StructuredOutputMode.JSON_SCHEMA

    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        return httpx.Response(
            200, headers=headers_of(model), json=body_of(model, '{"dimension": "structure"}', mode)
        )

    router = router_with(MockedBackend(handler), mode, JsonFileResponseCache(tmp_path), 1)
    with pytest.raises(InvalidModelOutputError):
        await router.invoke(AgentRole.STRUCTURE, "analiza", TechnicalVerdict)

    (stored,) = sorted(tmp_path.glob("*.json"))
    written = stored.read_text(encoding="utf-8")
    assert_no_secret_in(written, stored.name)
    entry = CacheEntry.model_validate_json(written)
    assert not entry.valid
    assert entry.upstream_model == upstream_of(MODEL_A).model


def test_the_upstream_is_not_part_of_the_cache_key() -> None:
    """La misma pregunta al mismo id es la misma entrada, la sirva quien la sirva."""
    base = {
        "backend": Backend.OPENAI,
        "model": MODEL_A,
        "role": AgentRole.STRUCTURE,
        "prompt_digest": "0" * 64,
        "schema_name": "TechnicalVerdict",
        "structured_output": StructuredOutputMode.JSON_SCHEMA,
        "raw": "{}",
    }

    assert CacheEntry(**base).key == CacheEntry(**base, upstream_model="otro").key  # type: ignore[arg-type]


# ─────────────────────────────── Lo ya escrito sigue cargando ─────────────────────────────────────

PRE_T6 = DATA_DIR / "llmcall_pre_t6.jsonl"
PRE_T6_SHA256 = "a56531255afe9169dec72a29df7a8a4447ad08cea70582f1cd6bfd6108cb98e6"
PRE_T6_RUN_DIGEST = "e71e27a74d16f85201e03f5caa08aecb7c9d7032b14e4a80ce11c8b7a7cc6910"
"""Un registro con cinco `LLMCall` que serializó el código de `60b09bc`. Ver el README de datos."""


@pytest.mark.parametrize("name", ["llmcall_pre_t6.jsonl", "journal_pre_t5.jsonl"])
def test_journal_lines_written_before_the_block_still_load(name: str) -> None:
    records = JsonlJournal(DATA_DIR / name).read_all()

    assert records
    for record in records:
        for call in record.calls:
            assert carried(call) == Upstream()


def test_the_digest_of_a_run_without_headers_did_not_move() -> None:
    """Por eso `RUN_DIGEST_VERSION` sigue en `replay-v3`: sin cabecera, el hash es el de antes."""
    assert hashlib.sha256(PRE_T6.read_bytes()).hexdigest() == PRE_T6_SHA256
    (record,) = JsonlJournal(PRE_T6).read_all()

    assert len(record.calls) == 5
    assert run_digest([record]) == PRE_T6_RUN_DIGEST
    assert RUN_DIGEST_VERSION == b"replay-v3"


def test_a_run_with_headers_hashes_them() -> None:
    """Como los tokens y la latencia: lo que se midió entra en la huella."""
    (record,) = JsonlJournal(PRE_T6).read_all()
    first, *rest = record.calls

    def with_upstream(**fields: str) -> str:
        changed = first.model_copy(update=fields)
        return run_digest([record.model_copy(update={"calls": (changed, *rest)})])

    one = with_upstream(upstream_model="accounts/fireworks/models/glm-5p3")
    other = with_upstream(upstream_model="z-ai/glm-5.2")

    assert len({PRE_T6_RUN_DIGEST, one, other}) == 3
    assert with_upstream(upstream_endpoint="fireworks") != PRE_T6_RUN_DIGEST
