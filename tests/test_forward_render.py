"""El prompt de `solo` fijado por su salida, y la petición que lo lleva al modelo.

`tests/data/solo_prompt_digests.jsonl` guarda, para las dos primeras entradas de cada símbolo del
manifiesto de la etapa 1, el `prompt_digest` que registró el journal. Aquí esos 14 prompts se
reconstruyen desde `data/history/` y el manifiesto, sin modelos, y tienen que dar el mismo digest.
Cuatro cosas, cada una con su mutación:

- **Lo que cambia el prompt rompe la comprobación**: una cadena de la plantilla, el formato de
  `format_indicators`, el de `format_triggers` y el del cierre en `solo_prompt`.
- **Lo que no lo cambia, no**: el código de las tres funciones reescrito solo en espaciado pasa,
  y el sha-256 de su código fuente, que es lo que se fijaba antes, no.
- **Ni la preparación ni el renderizado usan el router**: el contexto lleva uno que levanta ante
  cualquier atributo, y una `prepare_evaluation` que lee la configuración a través de él falla.
- **La petición**: la envoltura de la del decisor registrado es la registrada, no depende del
  prompt, su único mensaje es el prompt, y un mensaje de sistema, otro modo u otra temperatura la
  cambian. Capturarla no abre ninguna conexión.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import shutil
import socket
import textwrap
from typing import TYPE_CHECKING, Any, NoReturn, cast
from uuid import UUID, uuid4

import pytest

import crypto_agents.forward_render as module
import crypto_agents.nodes as nodes_module
import crypto_agents.prompts as prompts_module
from crypto_agents.ablation import plan_from_manifest
from crypto_agents.context import utc_now
from crypto_agents.forward_render import (
    CAPTURE_BASE_URL,
    SOLO_REQUEST_ENVELOPE_SHA256,
    captured_request,
    check_envelope,
    check_run,
    compare,
    envelope_digest,
    journal_digests,
    main,
    request_envelope,
    sample_positions,
    solo_digests,
)
from crypto_agents.journal import EvaluationRecord
from crypto_agents.llm import OpenAIBackend, prompt_digest
from crypto_agents.market import timeframe_to_timedelta
from crypto_agents.replay import replay_run_id
from crypto_agents.selection import load_manifest, load_selection_histories
from crypto_agents.settings import ModelChoice, RoleConfig, load_settings
from crypto_agents.state import (
    AgentRole,
    Backend,
    FailureKind,
    LLMCall,
    StructuredOutputMode,
)
from tests.conftest import DATA_DIR
from tests.test_net_outcomes import mutated
from tests.test_replay import make_settings

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path
    from types import ModuleType

    from crypto_agents.ablation import ReplayPlan
    from crypto_agents.llm import ModelRouter
    from crypto_agents.settings import Settings

FIXTURE = DATA_DIR / "solo_prompt_digests.jsonl"
FIXTURE_SHA256 = "8d1063441c43d49969eba2911eb9ffa8593040eefaff112f464bf6ee5d67644b"
MANIFEST = DATA_DIR.parent.parent / "data" / "ablation_selection.json"

REGISTERED_DECIDER = ModelChoice(
    backend=Backend.OPENAI,
    model="glm-5.2",
    family="zhipu",
    structured_output=StructuredOutputMode.FUNCTION_CALLING,
    temperature=0.0,
    quota_per_window=100_000,
)
"""El decisor de `solo` en la etapa 1, con los valores del apartado 3 de `docs/forward.md`."""


def fixture_rows() -> list[dict[str, Any]]:
    return [json.loads(line) for line in FIXTURE.read_text("utf-8").splitlines()]


def fixture_digests() -> dict[UUID, str]:
    return {UUID(row["run_id"]): row["prompt_digest"] for row in fixture_rows()}


@pytest.fixture(scope="module")
def plan() -> ReplayPlan:
    """El plan de la etapa 1 reducido a las 14 entradas del fixture."""
    manifest = load_manifest(MANIFEST)
    whole = plan_from_manifest(manifest, load_selection_histories(manifest))
    positions = [row["position"] for row in fixture_rows()]
    return whole._replace(entries=tuple(whole.entries[position] for position in positions))


@pytest.fixture
def fresh_templates() -> Iterator[None]:
    """`load_template` guarda en caché lo leído: sin vaciarla, cambiar `PROMPT_DIR` no se ve."""
    prompts_module.load_template.cache_clear()
    yield
    prompts_module.load_template.cache_clear()


class RefusingRouter:
    """Un router que no deja usar nada: pedirle cualquier atributo levanta."""

    def __getattr__(self, name: str) -> NoReturn:
        raise AssertionError(f"se usó el router: {name}")


def refusing_router() -> ModelRouter:
    return cast("ModelRouter", RefusingRouter())


async def assert_the_fourteen_render_as_journaled(plan: ReplayPlan) -> None:
    """La comprobación que cada mutación tiene que romper."""
    rendered = await solo_digests(plan, make_settings(), refusing_router())
    check = compare(fixture_digests(), rendered)

    assert check.matching == len(fixture_rows()) == 14
    assert check.passed, f"distintas: {check.mismatched}, faltan: {check.missing}"


# ─────────────────────────────────────────── El fixture ──────────────────────────────────────────


def test_the_fixture_is_the_one_derived() -> None:
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256


def test_the_fixture_is_the_first_two_entries_of_each_symbol() -> None:
    """Las posiciones no se eligen: son las dos primeras de cada símbolo, en el orden del
    manifiesto, y cada línea es la entrada que dice ser."""
    manifest = load_manifest(MANIFEST)
    rows = fixture_rows()

    positions = sample_positions(manifest.entries, 2)

    assert positions == (0, 1, 20, 21, 40, 41, 60, 61, 80, 81, 100, 101, 120, 121)
    assert tuple(row["position"] for row in rows) == positions
    for row in rows:
        entry = manifest.entries[row["position"]]
        at = entry.at + timeframe_to_timedelta(entry.timeframe)
        assert row["symbol"] == entry.symbol
        assert row["at"] == at.isoformat().replace("+00:00", "Z")
        assert UUID(row["run_id"]) == replay_run_id(entry.symbol, entry.timeframe, at)


# ─────────────────────────────── El prompt, reconstruido sin modelos ──────────────────────────────


@pytest.mark.asyncio
async def test_the_fourteen_prompts_render_as_the_journal_recorded(plan: ReplayPlan) -> None:
    """El control de las mutaciones: sin nada cambiado, los 14 coinciden."""
    await assert_the_fourteen_render_as_journaled(plan)


def test_the_refusing_router_refuses_any_method() -> None:
    """El control del router de la prueba: si no levantara, la prueba de abajo no vería nada."""
    router = refusing_router()
    for name in ("invoke", "settings", "ledger"):
        with pytest.raises(AssertionError, match=name):
            getattr(router, name)


@pytest.mark.asyncio
async def test_mutation_preparing_through_the_router_is_caught(
    plan: ReplayPlan, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`prepare_evaluation` leyendo el exchange a través del router en lugar de los ajustes."""
    mutant = mutated(
        nodes_module.prepare_evaluation,
        "context.settings.exchange.exchange_id",
        "context.router.settings.exchange.exchange_id",
        nodes_module,
    )
    monkeypatch.setattr(module, "prepare_evaluation", mutant)

    with pytest.raises(AssertionError, match="se usó el router"):
        await assert_the_fourteen_render_as_journaled(plan)


@pytest.mark.asyncio
@pytest.mark.usefixtures("fresh_templates")
async def test_mutation_a_changed_template_is_caught(
    plan: ReplayPlan, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Una copia de `prompts/` sin tocar pasa; con una frase de `decider_solo.md` cambiada, no."""
    for template in prompts_module.PROMPT_DIR.glob("*.md"):
        shutil.copy(template, tmp_path / template.name)
    monkeypatch.setattr(prompts_module, "PROMPT_DIR", tmp_path)

    await assert_the_fourteen_render_as_journaled(plan)

    solo = tmp_path / "decider_solo.md"
    text = solo.read_text("utf-8")
    old = "Precio de cierre de la última vela cerrada: $close"
    assert text.count(old) == 1
    solo.write_text(text.replace(old, "Último cierre: $close"), "utf-8")
    prompts_module.load_template.cache_clear()

    with pytest.raises(AssertionError):
        await assert_the_fourteen_render_as_journaled(plan)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("owner", "name", "old", "new"),
    [
        (prompts_module, "format_indicators", ":.4f", ":.3f"),
        (prompts_module, "format_triggers", 'f"- `{trigger}`"', 'f"- {trigger}"'),
        (module, "solo_prompt", 'close=f"{snapshot.close:.8g}"', 'close=f"{snapshot.close:.8f}"'),
    ],
    ids=["indicadores con 3 decimales", "disparadores sin comillas", "cierre en coma fija"],
)
async def test_mutation_a_changed_format_string_is_caught(
    plan: ReplayPlan,
    monkeypatch: pytest.MonkeyPatch,
    owner: ModuleType,
    name: str,
    old: str,
    new: str,
) -> None:
    """Cada función se cambia donde la busca quien la llama: `solo_prompt` busca las dos de
    formato en `prompts`, y `forward_render` busca `solo_prompt` en su propio espacio."""
    function = getattr(prompts_module, name)
    monkeypatch.setattr(owner, name, mutated(function, old, new, prompts_module))

    with pytest.raises(AssertionError):
        await assert_the_fourteen_render_as_journaled(plan)


def respaced(function: Callable[..., Any], namespace_of: ModuleType) -> Callable[..., Any]:
    """La función con su código reescrito solo en espaciado: sangría doble, dos espacios al final
    de cada línea y una línea en blanco entre cada dos. Dentro de las cadenas de una línea no
    cambia nada; en los docstrings, sí, y no importa."""
    source = textwrap.dedent(inspect.getsource(function))

    def respace(line: str) -> str:
        body = line.lstrip(" ")
        return " " * (2 * (len(line) - len(body))) + body + "  " if body else line

    rewritten = "\n\n".join(respace(line) for line in source.splitlines()) + "\n"
    assert "".join(rewritten.split()) == "".join(source.split()), "solo cambia el espaciado"
    assert hashlib.sha256(rewritten.encode()).digest() != hashlib.sha256(source.encode()).digest()
    namespace: dict[str, Any] = dict(vars(namespace_of))
    exec(compile(rewritten, "<respaciado>", "exec"), namespace)
    return cast("Callable[..., Any]", namespace[function.__name__])


@pytest.mark.asyncio
async def test_a_whitespace_only_rewrite_of_the_rendering_code_still_passes(
    plan: ReplayPlan, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lo que el sha-256 del código fuente no aguantaba. Primero las dos de formato, para que la
    copia de `solo_prompt` las vea ya reescritas."""
    for name in ("format_indicators", "format_triggers"):
        function = getattr(prompts_module, name)
        monkeypatch.setattr(prompts_module, name, respaced(function, prompts_module))
    monkeypatch.setattr(module, "solo_prompt", respaced(prompts_module.solo_prompt, prompts_module))

    await assert_the_fourteen_render_as_journaled(plan)


# ───────────────────────────────── El journal y la comparación ───────────────────────────────────


def call(role: AgentRole, digest: str, valid: bool = True) -> LLMCall:
    return LLMCall(
        role=role,
        backend=Backend.OPENAI,
        model="glm-5.2",
        structured_output=StructuredOutputMode.FUNCTION_CALLING,
        quota_weight=1.0,
        prompt_digest=digest,
        valid=valid,
        failure_kind=None if valid else FailureKind.SCHEMA,
        failure_message=None if valid else "action: Field required",
        latency_ms=1.0,
        at=utc_now(),
    )


def record(*calls: LLMCall, run_id: UUID | None = None) -> EvaluationRecord:
    return EvaluationRecord(
        run_id=run_id or uuid4(),
        at=utc_now(),
        symbol="ADA/USDT",
        timeframe="4h",
        calls=calls,
    )


def test_the_journal_digest_is_the_first_attempt_of_the_decider() -> None:
    """El reintento lleva la plantilla de reintento y otro digest; una evaluación sin decisor no
    entra."""
    retried = record(
        call(AgentRole.DECIDER, "a" * 64, valid=False), call(AgentRole.DECIDER, "b" * 64)
    )
    other = record(call(AgentRole.STRUCTURE, "c" * 64), call(AgentRole.DECIDER, "d" * 64))
    gate_closed = record()

    digests = journal_digests([retried, other, gate_closed])

    assert digests == {retried.run_id: "a" * 64, other.run_id: "d" * 64}


def test_the_comparison_names_what_differs_what_is_missing_and_what_is_extra() -> None:
    same, differs, missing, extra = (uuid4() for _ in range(4))

    check = compare({same: "a", differs: "b", missing: "c"}, {same: "a", differs: "x", extra: "d"})

    assert (check.expected, check.rendered, check.matching) == (3, 3, 1)
    assert (check.mismatched, check.missing, check.unexpected) == ((differs,), (missing,), (extra,))
    assert not check.passed
    assert compare({same: "a"}, {same: "a"}).passed
    assert not compare({}, {}).passed, "nada que comparar no es una comprobación que pasa"


# ───────────────────────────────────────────── La petición ───────────────────────────────────────


@pytest.mark.asyncio
async def test_the_envelope_of_the_registered_decider_is_the_registered_one() -> None:
    """Un mensaje del usuario, la herramienta `Proposal` y estos parámetros, nada más."""
    body = await captured_request(REGISTERED_DECIDER, "un prompt")

    assert sorted(body) == [
        "messages",
        "model",
        "parallel_tool_calls",
        "stream",
        "temperature",
        "tool_choice",
        "tools",
    ]
    assert body["messages"] == [{"role": "user", "content": "un prompt"}]
    assert (body["model"], body["temperature"], body["stream"]) == ("glm-5.2", 0.0, False)
    assert body["parallel_tool_calls"] is False
    assert body["tool_choice"] == {"type": "function", "function": {"name": "Proposal"}}
    assert envelope_digest(request_envelope(body)) == SOLO_REQUEST_ENVELOPE_SHA256


@pytest.mark.asyncio
async def test_the_envelope_does_not_depend_on_the_prompt_and_the_message_is_the_prompt() -> None:
    """El prompt lo cubre su digest; la envoltura, todo lo demás. Que el único mensaje sea el
    prompt es lo que ata la petición al digest del journal."""
    first = await check_envelope(REGISTERED_DECIDER, "uno", prompt_digest("uno"))
    second = await check_envelope(REGISTERED_DECIDER, "otro, más largo", prompt_digest("otro"))

    assert first.digest == second.digest == SOLO_REQUEST_ENVELOPE_SHA256
    assert first.passed
    assert first.roles == ("user",)
    assert first.tools == ("Proposal",)
    assert not second.message_matches, "el digest que se manda no es el que el journal dice"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        {"temperature": 0.2},
        {"model": "glm-5.3"},
        {"structured_output": StructuredOutputMode.JSON_SCHEMA},
    ],
    ids=["otra temperatura", "otro modelo", "otro modo"],
)
async def test_another_parameter_changes_the_envelope(change: dict[str, object]) -> None:
    choice = REGISTERED_DECIDER.model_copy(update=change)

    check = await check_envelope(choice, "uno", prompt_digest("uno"))

    assert not check.envelope_matches


@pytest.mark.asyncio
async def test_mutation_a_system_message_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """Un mensaje de sistema delante del prompt: la envoltura lo ve aunque su contenido se vacíe,
    y el único mensaje deja de ser el prompt."""
    import crypto_agents.llm as llm_module

    mutant = mutated(
        OpenAIBackend.complete,
        "result = await structured.ainvoke(prompt)",
        'result = await structured.ainvoke([("system", "x"), ("user", prompt)])',
        llm_module,
    )
    monkeypatch.setattr(OpenAIBackend, "complete", mutant)

    check = await check_envelope(REGISTERED_DECIDER, "uno", prompt_digest("uno"))

    assert check.roles == ("system", "user")
    assert not check.envelope_matches
    assert not check.message_matches


@pytest.mark.asyncio
async def test_capturing_the_request_opens_no_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    """El transporte es simulado y la URL no resuelve; aquí, además, no hay red que abrir."""

    def refuse(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("se intentó abrir una conexión")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)

    check = await check_envelope(REGISTERED_DECIDER, "uno", prompt_digest("uno"))

    assert check.passed
    assert CAPTURE_BASE_URL.split("/")[2].endswith(".invalid")


# ───────────────────────────────────────── La comprobación entera ─────────────────────────────────


def settings_with(decider: ModelChoice) -> Settings:
    roles = dict(make_settings().roles)
    roles[AgentRole.DECIDER] = RoleConfig(primary=decider)
    return load_settings(
        roles=roles, openai={"api_key": "sk-test"}, ollama={"host": "http://localhost:11434"}
    )


def journaled(digests: dict[UUID, str]) -> list[EvaluationRecord]:
    return [
        record(call(AgentRole.DECIDER, digest), run_id=run_id) for run_id, digest in digests.items()
    ]


@pytest.mark.asyncio
async def test_the_whole_check_passes_over_the_fourteen(plan: ReplayPlan) -> None:
    check, envelope = await check_run(
        plan, settings_with(REGISTERED_DECIDER), journaled(fixture_digests())
    )

    assert check.passed
    assert check.matching == 14
    assert envelope.passed


@pytest.mark.asyncio
async def test_the_whole_check_fails_on_a_digest_or_a_parameter(plan: ReplayPlan) -> None:
    digests = fixture_digests()
    changed = next(iter(digests))
    digests[changed] = "0" * 64

    check, envelope = await check_run(plan, settings_with(REGISTERED_DECIDER), journaled(digests))
    assert check.mismatched == (changed,)
    assert not check.passed

    warmer = REGISTERED_DECIDER.model_copy(update={"temperature": 0.2})
    check, envelope = await check_run(plan, settings_with(warmer), journaled(fixture_digests()))
    assert check.passed
    assert not envelope.passed


def test_a_directory_that_cannot_be_read_exits_with_two(tmp_path: Path) -> None:
    assert main([str(tmp_path / "no-existe")]) == 2
