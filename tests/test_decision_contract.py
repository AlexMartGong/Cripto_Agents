"""El contrato del decisor: el esquema anuncia lo que el validador exige.

`dismissed_side`, `dismissal_reason` e `invalidation_price` son obligatorios para `buy` y `sell`,
y eso lo dice un validador. Mientras llevaron `default=None`, Pydantic los dejaba fuera de
`required` y, con `function_calling`, ese esquema es la definición de la herramienta que ve el
modelo: el contrato decía «opcional» y el validador rechazaba la respuesta que se lo creía. En el
sondeo de mesas del 2026-10-05 (`var/zen-probe/20261005T233053Z`) los 29 fallos de esquema del
decisor fueron ese: `la acción buy|sell exige: dismissed_side`.

Lo que se comprueba aquí es lo que sale por el cable, no solo `model_json_schema()`: entre el
modelo de Python y el proveedor está la conversión de LangChain, y una prueba que se quede en el
primero no dice qué lee el modelo.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import pytest
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

import crypto_agents.state as state_module
from crypto_agents.journal import JsonlJournal
from crypto_agents.llm import structured_runnable
from crypto_agents.state import Action, Decision, Proposal, Side, StructuredOutputMode
from tests.conftest import DATA_DIR, SCARCE

if TYPE_CHECKING:
    from collections.abc import Iterable

    from crypto_agents.state import LLMOutput

DECISION_FIELDS = ("invalidation_price", "dismissed_side", "dismissal_reason")
PROPOSAL_FIELDS = ("invalidation_price",)

RATIONALE = "Estructura, impulso y volumen coinciden en direccion alcista."
REASON = "Su contraargumento depende de un nivel que ya se perdio."


def sell(**overrides: object) -> dict[str, object]:
    """Lo que un modelo contestaría para vender, como el diccionario crudo que llega."""
    payload: dict[str, object] = {
        "action": "sell",
        "confidence": 0.7,
        "size_fraction": 0.25,
        "invalidation_price": 101.0,
        "rationale": RATIONALE,
        "dismissed_side": "bull",
        "dismissal_reason": REASON,
    }
    payload.update(overrides)
    return payload


def without(payload: dict[str, object], *keys: str) -> dict[str, object]:
    """El mismo diccionario sin esas claves: una respuesta que las omite, no que las anula."""
    return {key: value for key, value in payload.items() if key not in keys}


# ─────────────────────────────────── Lo que dice el esquema ───────────────────────────────────────


def test_the_schema_lists_the_conditional_fields_as_required() -> None:
    assert set(DECISION_FIELDS) <= set(Decision.model_json_schema()["required"])
    assert set(PROPOSAL_FIELDS) <= set(Proposal.model_json_schema()["required"])


@pytest.mark.parametrize("field", DECISION_FIELDS)
def test_the_required_fields_stay_nullable_and_carry_no_default(field: str) -> None:
    """Requerido y anulable: el campo existe siempre y en `hold` vale `null`.

    Un `default` en el esquema es lo que lo sacaba de `required`. Que `null` siga entre las
    alternativas es lo que hace que la regla siga siendo del validador y no del esquema.
    """
    declared = Decision.model_json_schema()["properties"][field]

    assert "default" not in declared
    assert {"type": "null"} in declared["anyOf"]


# ─────────────────────────────── Lo que sale por el cable ─────────────────────────────────────────


def _completion(name: str) -> dict[str, object]:
    """Una respuesta mínima del proveedor con una llamada a la herramienta pedida."""
    call = {"id": "c", "type": "function", "function": {"name": name, "arguments": "{}"}}
    message = {"role": "assistant", "content": None, "tool_calls": [call]}
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "glm-5.2",
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": message}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


async def sent_tool(schema: type[LLMOutput]) -> dict[str, Any]:
    """La herramienta que recibe el proveedor al pedir `schema` con `function_calling`.

    El cliente es el `ChatOpenAI` real y el esquema se enlaza con `structured_runnable`, la misma
    función que usa `OpenAIBackend.complete()`. Lo único simulado es el transporte, que guarda el
    cuerpo de la petición: es el JSON que habría viajado.
    """
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_completion(schema.__name__))

    client = ChatOpenAI(
        model="glm-5.2",
        api_key="clave-de-prueba",  # type: ignore[arg-type]
        base_url="https://zen.invalid/v1",
        timeout=5.0,
        max_retries=0,
        http_async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    choice = SCARCE.model_copy(update={"structured_output": StructuredOutputMode.FUNCTION_CALLING})
    await structured_runnable(client, schema, choice).ainvoke("x")  # type: ignore[attr-defined]

    (body,) = bodies
    (tool,) = body["tools"]
    assert body["tool_choice"] == {"type": "function", "function": {"name": schema.__name__}}
    function: dict[str, Any] = tool["function"]
    return function


async def assert_the_sent_tool_requires(schema: type[LLMOutput], fields: Iterable[str]) -> None:
    """La comprobación que la mutación tiene que romper."""
    required = (await sent_tool(schema))["parameters"]["required"]
    missing = sorted(set(fields) - set(required))
    assert not missing, f"la herramienta de {schema.__name__} no exige {', '.join(missing)}"


@pytest.mark.asyncio
async def test_the_tool_that_is_sent_requires_the_conditional_fields() -> None:
    await assert_the_sent_tool_requires(Decision, DECISION_FIELDS)
    await assert_the_sent_tool_requires(Proposal, PROPOSAL_FIELDS)


@pytest.mark.asyncio
async def test_the_tool_parameters_are_one_object_at_the_top_level() -> None:
    """`function_calling` exige un objeto: ni unión por `action` ni esquema reescrito a mano."""
    parameters = (await sent_tool(Decision))["parameters"]

    assert parameters["type"] == "object"
    assert set(parameters["properties"]) == set(Decision.model_fields)
    assert set(parameters["required"]) == set(Decision.model_fields)


def mutated_decision(old: str, new: str) -> type[LLMOutput]:
    """`Decision` con un fragmento de su código cambiado, compilada en el espacio de `state`."""
    source = textwrap.dedent(inspect.getsource(Decision))
    assert source.count(old) == 1, f"el fragmento {old!r} ya no está una sola vez en el código"
    namespace: dict[str, Any] = dict(vars(state_module))
    exec(compile(source.replace(old, new), "<mutante>", "exec"), namespace)
    mutant: type[LLMOutput] = namespace["Decision"]
    return mutant


@pytest.mark.asyncio
async def test_mutation_a_default_takes_the_field_out_of_the_tool_and_is_caught() -> None:
    """Devolverle `= None` a `dismissed_side` es volver al contrato que decía «opcional»."""
    mutant = mutated_decision(
        "dismissed_side: Side | None\n", "dismissed_side: Side | None = None\n"
    )

    assert "dismissed_side" not in mutant.model_json_schema()["required"]
    with pytest.raises(AssertionError):
        await assert_the_sent_tool_requires(mutant, DECISION_FIELDS)


# ───────────────────────────── La regla sigue siendo del validador ────────────────────────────────


def test_an_omitted_field_and_an_explicit_null_fail_with_different_errors() -> None:
    """Lo que el contrato viejo no dejaba distinguir: omitir la clave o escribir `null`.

    Omitida, la respuesta ni siquiera tiene la forma pedida (`Field required`). Con `null`
    explícito la forma es correcta y quien la rechaza es el validador, con el mismo mensaje de
    siempre: es la regla, que no cambió.
    """
    with pytest.raises(ValidationError) as omitted:
        Decision.model_validate(without(sell(), "dismissed_side"))
    assert [(e["loc"], e["type"]) for e in omitted.value.errors()] == [
        (("dismissed_side",), "missing")
    ]

    with pytest.raises(ValidationError) as null:
        Decision.model_validate(sell(dismissed_side=None))
    assert [e["msg"] for e in null.value.errors()] == [
        "Value error, la acción sell exige: dismissed_side"
    ]


def test_a_hold_must_now_say_null_instead_of_saying_nothing() -> None:
    """El coste del cambio, escrito: un `hold` sin las claves ya no valida.

    Antes validaba. Es lo que significa que el campo exista siempre, y es también por dónde una
    entrada de caché escrita con el contrato viejo puede dejar de servir.
    """
    hold = without(sell(action="hold", size_fraction=0.0), *DECISION_FIELDS)

    with pytest.raises(ValidationError) as error:
        Decision.model_validate(hold)
    assert sorted(e["loc"][0] for e in error.value.errors()) == sorted(DECISION_FIELDS)

    explicit = Decision.model_validate({**hold, **dict.fromkeys(DECISION_FIELDS)})
    assert explicit.action is Action.HOLD
    assert explicit.dismissed_side is None


def test_a_proposal_without_desks_follows_the_same_rule() -> None:
    hold = {"action": "hold", "confidence": 0.4, "size_fraction": 0.0, "rationale": RATIONALE}

    with pytest.raises(ValidationError) as error:
        Proposal.model_validate(hold)
    assert [(e["loc"], e["type"]) for e in error.value.errors()] == [
        (("invalidation_price",), "missing")
    ]
    assert Proposal.model_validate({**hold, "invalidation_price": None}).action is Action.HOLD


# ─────────────────────────────── Lo ya escrito sigue cargando ─────────────────────────────────────

PRE_T5 = DATA_DIR / "journal_pre_t5.jsonl"
PRE_T5_SHA256 = "30ddc376c57f488d3abc8ee29c03c3d5fbba84d92a39a452269ea0967802aac6"
"""Tres líneas que serializó el código de `6e736aa`, antes del cambio. Ver el README de datos."""


def test_journal_lines_written_before_the_change_still_load() -> None:
    """`model_dump_json` ya escribía los tres campos con `null`, así que las claves están.

    Una línea de cada tipo: `Decision` con `hold`, `Decision` con `buy` y `Proposal` con `hold`.
    Se fija el digest del archivo para que nadie las «arregle» a mano y deje de probar esto.
    """
    assert hashlib.sha256(PRE_T5.read_bytes()).hexdigest() == PRE_T5_SHA256

    hold, buy, proposal = JsonlJournal(Path(PRE_T5)).read_all()

    assert hold.decision is not None
    assert (hold.decision.action, hold.decision.dismissed_side) == (Action.HOLD, None)
    assert buy.decision is not None
    assert (buy.decision.action, buy.decision.dismissed_side) == (Action.BUY, Side.BEAR)
    assert buy.decision.invalidation_price == pytest.approx(99.0)
    assert proposal.decision is None
    assert proposal.proposal is not None
    assert (proposal.proposal.action, proposal.proposal.invalidation_price) == (Action.HOLD, None)


def test_a_record_serialises_to_the_same_bytes_as_before() -> None:
    """Por eso `RUN_DIGEST_VERSION` no se mueve: la forma serializada no cambió."""
    lines = PRE_T5.read_text(encoding="utf-8").splitlines()

    assert [record.model_dump_json() for record in JsonlJournal(Path(PRE_T5)).read_all()] == lines
