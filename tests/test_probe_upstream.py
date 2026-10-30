"""El sondeo de diagnóstico del upstream: `scripts/probe_upstream.py`.

Nada aquí sale de la máquina: las peticiones van a un `httpx.MockTransport` puesto debajo del
adaptador de producción por la misma costura que usa la sonda, con una clave literal. Lo que se
fija es lo que la sonda promete: que la variante a es la petición del estudio, que b y c cambian
solo lo que dicen cambiar, que se guarda todo menos `set-cookie` y nada de la petición que lleve
credenciales, que son siete peticiones y ni una más, y que se niega antes de llamar.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import httpx
import pytest
import scripts.probe_upstream as module
from pydantic import SecretStr
from scripts.probe_upstream import (
    CAP_USD,
    CONTROL_MODEL,
    FIXED_USER_AGENT,
    PLAN,
    STABLE_SESSION,
    Prepared,
    ProbeRefusedError,
    Variant,
    describe,
    kept_headers,
    prepare,
    probe,
    run,
    tokens_per_second,
)

from crypto_agents.forward import ForwardPaths
from crypto_agents.forward_data import STAGE1_DIR
from crypto_agents.forward_record import REGISTERED_PATH, SOLO_REQUEST_ENVELOPE_SHA256
from crypto_agents.forward_render import captured_request
from crypto_agents.llm import SESSION_HEADER, prompt_digest
from crypto_agents.selection import HISTORY_DIR
from crypto_agents.settings import Billing, Settings
from crypto_agents.state import AgentRole, LLMCall
from tests.test_forward import ROOT, study_settings
from tests.test_net_outcomes import mutated

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

SCRIPT = ROOT / "scripts" / "probe_upstream.py"
KEY = "clave-literal-de-prueba"
STARTED = datetime(2026, 10, 30, 6, 0, tzinfo=UTC)
FIRST_RUN_ID = "bef5177f-eb34-55cd-b0a6-1261366bd515"
FIRST_DIGEST = "f1ec963b791e5a33ef99e4dabe8c03e719400eca1d71db668c4fa3cd4401fcea"
VALID = (
    '{"rationale":"una razón de más de veinte caracteres","action":"hold",'
    '"confidence":0.5,"size_fraction":0,"invalidation_price":null}'
)
INVALID = '{"rationale":"corta","action":"hold","confidence":0.5,"size_fraction":0}'


def paths() -> ForwardPaths:
    return ForwardPaths(stage1=ROOT / STAGE1_DIR, published_history=ROOT / HISTORY_DIR)


def settings(billing: Billing = Billing.PAYG) -> Settings:
    base = study_settings()
    assert base.openai is not None
    openai = base.openai.model_copy(update={"api_key": SecretStr(KEY)})
    return base.model_copy(update={"openai": openai, "billing": billing})


@pytest.fixture(scope="module")
def prepared() -> Prepared:
    """Las comprobaciones previas sobre la etapa 1 de verdad, una vez para todo el módulo."""
    return asyncio.run(prepare(settings(), paths()))


class Gateway:
    """El transporte simulado: guarda cada petición y contesta con la herramienta pedida."""

    def __init__(
        self,
        arguments: str = VALID,
        headers: Sequence[tuple[str, str]] = (),
        usage: dict[str, int] | None = None,
    ) -> None:
        self.arguments = arguments
        self.headers = list(headers)
        self.usage = usage or {"prompt_tokens": 1149, "completion_tokens": 2537, "total_tokens": 0}
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = json.loads(request.content)
        name = body["tool_choice"]["function"]["name"]
        call = {
            "id": "t",
            "type": "function",
            "function": {"name": name, "arguments": self.arguments},
        }
        message = {"role": "assistant", "content": None, "tool_calls": [call]}
        return httpx.Response(
            200,
            headers=self.headers,
            json={
                "id": "a1b2c3",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
                "usage": self.usage,
            },
        )

    @property
    def bodies(self) -> list[dict[str, object]]:
        return [json.loads(request.content) for request in self.requests]

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)


def probed(prompt: str, gateway: Gateway) -> module.ProbeRun:
    return asyncio.run(probe(settings(), prompt, transport=gateway.transport()))


def at(variant: Variant) -> list[int]:
    return [index for index, planned in enumerate(PLAN) if planned is variant]


# ─────────────────────────────────────────── El prompt ───────────────────────────────────────────


def test_the_prompt_is_the_first_of_stage1_and_gives_its_journal_digest(
    prepared: Prepared,
) -> None:
    first = json.loads((ROOT / STAGE1_DIR / "solo.jsonl").read_text("utf-8").splitlines()[0])
    assert prepared.reference.run_id == first["run_id"] == FIRST_RUN_ID
    assert prompt_digest(prepared.prompt) == prepared.reference.prompt_digest == FIRST_DIGEST
    assert prepared.reference.upstream_model == "accounts/fireworks/models/glm-5p3"
    assert prepared.envelope_sha256 == REGISTERED_PATH.envelope_sha256


# ────────────────────────────────────────── La petición ──────────────────────────────────────────


def test_variant_a_is_the_request_of_the_study(prepared: Prepared) -> None:
    gateway = Gateway()
    done = probed(prepared.prompt, gateway)

    choice = settings().role_choices(AgentRole.DECIDER)[0]
    production = asyncio.run(captured_request(choice, prepared.prompt))
    for index in at(Variant.STUDY):
        assert gateway.bodies[index] == production
        call = done.calls[index]
        assert call.request_envelope_sha256 == SOLO_REQUEST_ENVELOPE_SHA256
        assert call.prompt_matches is True
        assert call.request_user_agent != FIXED_USER_AGENT
    sessions = {gateway.requests[index].headers[SESSION_HEADER] for index in at(Variant.STUDY)}
    assert len(sessions) == 1, "una sesión por backend, como el estudio"
    assert sessions != {STABLE_SESSION}


def test_variant_b_changes_only_the_user_agent_and_the_session(prepared: Prepared) -> None:
    first, second = Gateway(), Gateway()
    probed(prepared.prompt, first)
    probed(prepared.prompt, second)

    (study, *_), (headers, *_) = at(Variant.STUDY), at(Variant.HEADERS)
    assert first.bodies[headers] == first.bodies[study]
    plain = dict(first.requests[study].headers)
    changed = dict(first.requests[headers].headers)
    differ = {
        name for name in plain.keys() | changed.keys() if plain.get(name) != changed.get(name)
    }
    assert differ == {"user-agent", SESSION_HEADER}
    for gateway in (first, second):
        for index in at(Variant.HEADERS):
            assert gateway.requests[index].headers["user-agent"] == FIXED_USER_AGENT
            assert gateway.requests[index].headers[SESSION_HEADER] == STABLE_SESSION


def test_variant_c_changes_only_the_model(prepared: Prepared) -> None:
    gateway = Gateway()
    done = probed(prepared.prompt, gateway)

    (study, *_), (control,) = at(Variant.STUDY), at(Variant.CONTROL)
    assert gateway.bodies[control] == {**gateway.bodies[study], "model": CONTROL_MODEL}
    assert done.calls[control].model == CONTROL_MODEL
    assert done.calls[control].request_envelope_sha256 != SOLO_REQUEST_ENVELOPE_SHA256


# ────────────────────────────────────── Lo que se guarda ─────────────────────────────────────────

RESPONSE_HEADERS = (
    ("x-opencode-upstream-model-id", "accounts/fireworks/models/glm-5p3"),
    ("set-cookie", "uno=1"),
    ("x-opencode-endpoint-id", "fireworks"),
    ("via", "a"),
    ("via", "b"),
    ("Set-Cookie", "dos=2"),
)


def assert_every_header_but_set_cookie_is_kept(prompt: str) -> None:
    done = probed(prompt, Gateway(headers=RESPONSE_HEADERS))
    for call in done.calls:
        names = [name for name, _ in call.headers]
        assert "set-cookie" not in names
        kept = [(name, value) for name, value in call.headers if name in ("via",) or "x-" in name]
        assert kept == [
            ("x-opencode-upstream-model-id", "accounts/fireworks/models/glm-5p3"),
            ("x-opencode-endpoint-id", "fireworks"),
            ("via", "a"),
            ("via", "b"),
        ]
        assert call.upstream_model == "accounts/fireworks/models/glm-5p3"
        assert call.upstream_endpoint == "fireworks"


def test_every_response_header_but_set_cookie_is_kept(prepared: Prepared) -> None:
    assert_every_header_but_set_cookie_is_kept(prepared.prompt)


def test_mutation_keeping_set_cookie_is_caught(
    prepared: Prepared, monkeypatch: pytest.MonkeyPatch
) -> None:
    mutant = mutated(kept_headers, " if name.lower() != EXCLUDED_HEADER", "", module)
    monkeypatch.setattr(module, "kept_headers", mutant)
    with pytest.raises(AssertionError):
        assert_every_header_but_set_cookie_is_kept(prepared.prompt)


SHUFFLED = (
    '{"size_fraction":0,"invalidation_price":null,"action":"hold",'
    '"rationale":"una razón de más de veinte caracteres","confidence":0.5}'
)


def assert_the_keys_keep_their_order(prompt: str) -> None:
    done = probed(prompt, Gateway(arguments=SHUFFLED))
    for call in done.calls:
        assert call.argument_keys == (
            "size_fraction",
            "invalidation_price",
            "action",
            "rationale",
            "confidence",
        )
        assert call.body_keys == ("id", "object", "created", "model", "choices", "usage")
        assert call.body_id == "a1b2c3"
        assert call.body_model == call.model
        assert call.system_fingerprint is None
        assert call.valid is True


def test_the_argument_keys_keep_the_order_they_arrived_in(prepared: Prepared) -> None:
    assert_the_keys_keep_their_order(prepared.prompt)


def test_mutation_sorting_the_argument_keys_is_caught(
    prepared: Prepared, monkeypatch: pytest.MonkeyPatch
) -> None:
    mutant = mutated(
        module._keys,
        "return tuple(str(key) for key, _ in pairs)",
        "return tuple(sorted(str(key) for key, _ in pairs))",
        module,
    )
    monkeypatch.setattr(module, "_keys", mutant)
    with pytest.raises(AssertionError):
        assert_the_keys_keep_their_order(prepared.prompt)


def test_tokens_per_second_is_output_tokens_over_wall_seconds() -> None:
    assert tokens_per_second(2537, 23578.88851200005) == pytest.approx(107.596, abs=1e-3)
    assert tokens_per_second(2328, 41449.76930300254) == pytest.approx(56.165, abs=1e-3)
    assert tokens_per_second(None, 1000.0) is None
    assert tokens_per_second(10, None) is None
    assert tokens_per_second(10, 0.0) is None


def test_a_call_the_cap_refused_keeps_its_place_without_a_response() -> None:
    call = describe(4, Variant.HEADERS, "glm-5.2", "p", None, None, "negada por el tope: x")
    assert call.status is None
    assert call.at is None
    assert call.headers == ()
    assert call.failure == "negada por el tope: x"


def test_nothing_of_the_request_that_carries_credentials_is_written(
    prepared: Prepared, tmp_path: Path
) -> None:
    gateway = Gateway(headers=RESPONSE_HEADERS)
    code = asyncio.run(
        run(
            settings(),
            paths(),
            tmp_path,
            False,
            clock=lambda: STARTED,
            transport=gateway.transport(),
            echo=lambda line: None,
        )
    )

    assert code == 0
    assert all("authorization" in request.headers for request in gateway.requests)
    (directory,) = tmp_path.iterdir()
    assert directory.name == "20261030T060000Z-upstream"
    assert sorted(path.name for path in directory.iterdir()) == [
        "calls.jsonl",
        "meta.json",
        "report.md",
        "responses.jsonl",
    ]
    for path in directory.iterdir():
        text = path.read_text("utf-8").lower()
        assert KEY not in text, path.name
        assert "authorization" not in text, path.name
        assert "bearer" not in text, path.name
        assert "uno=1" not in text and "dos=2" not in text, path.name


def test_the_directory_holds_what_the_report_promises(prepared: Prepared, tmp_path: Path) -> None:
    asyncio.run(
        run(
            settings(),
            paths(),
            tmp_path,
            False,
            clock=lambda: STARTED,
            transport=Gateway(headers=RESPONSE_HEADERS).transport(),
            echo=lambda line: None,
        )
    )
    (directory,) = tmp_path.iterdir()
    meta = module.ProbeMeta.model_validate_json((directory / "meta.json").read_text("utf-8"))
    assert meta.script_sha256 == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    assert meta.reference.run_id == FIRST_RUN_ID
    assert meta.envelope_sha256 == SOLO_REQUEST_ENVELOPE_SHA256
    assert meta.max_attempts == 1
    assert meta.cap_usd == CAP_USD
    assert meta.refused_calls == 0
    rows = (directory / "calls.jsonl").read_text("utf-8").splitlines()
    assert len(rows) == len(PLAN)
    assert {LLMCall.model_validate_json(row).cache_hit for row in rows} == {False}
    lines = (directory / "responses.jsonl").read_text("utf-8").splitlines()
    calls = [module.ProbeCall.model_validate_json(line) for line in lines]
    assert [call.variant for call in calls] == list(PLAN)
    report = (directory / "report.md").read_text("utf-8")
    assert report.count("### Llamada ") == len(PLAN)
    for field in ("cuerpo: `id`", "argumentos: orden de las claves", "tok/s", "hora UTC"):
        assert report.count(field) >= len(PLAN), field
    assert "cabecera `x-opencode-endpoint-id`" in report


# ─────────────────────────────────────── Siete y ni una más ──────────────────────────────────────


def assert_seven_requests(prompt: str) -> None:
    gateway = Gateway(arguments=INVALID)
    try:
        done = probed(prompt, gateway)
    except ProbeRefusedError:
        done = None
    assert len(gateway.requests) == len(PLAN) == 7
    assert done is not None
    assert len(done.rows) == 7
    assert not any(row.valid for row in done.rows)


def test_invalid_answers_make_seven_requests_and_no_retry(prepared: Prepared) -> None:
    assert_seven_requests(prepared.prompt)


def test_mutation_a_second_attempt_is_caught(
    prepared: Prepared, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "MAX_ATTEMPTS", 2)
    with pytest.raises(AssertionError):
        assert_seven_requests(prepared.prompt)


def test_the_cap_stops_opening_calls_once_reached(
    prepared: Prepared, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "CAP_USD", 1e-9)
    gateway = Gateway()
    code = asyncio.run(
        run(
            settings(),
            paths(),
            tmp_path,
            False,
            clock=lambda: STARTED,
            transport=gateway.transport(),
            echo=lambda line: None,
        )
    )
    assert code == 1
    assert len(gateway.requests) == 1
    (directory,) = tmp_path.iterdir()
    lines = (directory / "responses.jsonl").read_text("utf-8").splitlines()
    calls = [module.ProbeCall.model_validate_json(line) for line in lines]
    assert [call.status for call in calls] == [200, None, None, None, None, None, None]
    assert all("negada por el tope" in (call.failure or "") for call in calls[1:])


def test_no_cache_and_no_study_data() -> None:
    """El router de la sonda no tiene caché, y el script no nombra la del estudio ni sus datos."""
    source = SCRIPT.read_text("utf-8")
    for named in ("var/forward/cache", "var/ablation-cache", "data/forward", "ResponseCache"):
        assert named not in source, named
    routers = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ModelRouter"
    ]
    assert len(routers) == 1
    (cache,) = [keyword for keyword in routers[0].keywords if keyword.arg == "cache"]
    assert isinstance(cache.value, ast.Constant) and cache.value.value is None


# ───────────────────────────────────── Se niega antes de llamar ──────────────────────────────────


def refused(tmp_path: Path, chosen: Settings, gateway: Gateway, lines: list[str]) -> int:
    return asyncio.run(
        run(
            chosen,
            paths(),
            tmp_path,
            False,
            clock=lambda: STARTED,
            transport=gateway.transport(),
            echo=lines.append,
        )
    )


def assert_refused_before_calling(
    tmp_path: Path, chosen: Settings, expected: str, before: Callable[[], object] = lambda: None
) -> None:
    before()
    gateway, lines = Gateway(), list[str]()
    entries = sorted(tmp_path.iterdir())
    assert refused(tmp_path, chosen, gateway, lines) == 1
    assert gateway.requests == []
    assert sorted(tmp_path.iterdir()) == entries
    assert lines and lines[0].startswith("se niega antes de llamar"), lines
    assert expected in lines[0], lines


def test_it_refuses_a_prompt_that_is_not_the_journal_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        module, "journal_digests", lambda records: {record.run_id: "0" * 64 for record in records}
    )
    assert_refused_before_calling(tmp_path, settings(), "no da el prompt_digest del journal")


def test_it_refuses_an_envelope_that_is_not_the_registered_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = REGISTERED_PATH.model_copy(update={"envelope_sha256": "0" * 64})
    monkeypatch.setattr(module, "REGISTERED_PATH", other)
    assert_refused_before_calling(tmp_path, settings(), "envelope_sha256")


def test_it_refuses_the_subscription(tmp_path: Path) -> None:
    assert_refused_before_calling(tmp_path, settings(Billing.GO), "pago por uso")


def test_it_refuses_a_directory_that_exists(tmp_path: Path) -> None:
    def existing() -> None:
        (tmp_path / "20261030T060000Z-upstream").mkdir()

    assert_refused_before_calling(tmp_path, settings(), "ya existe", existing)


def test_the_dry_run_checks_and_calls_nobody(tmp_path: Path) -> None:
    gateway, lines = Gateway(), list[str]()
    code = asyncio.run(
        run(
            settings(),
            paths(),
            tmp_path,
            True,
            clock=lambda: STARTED,
            transport=gateway.transport(),
            echo=lines.append,
        )
    )
    assert code == 0
    assert gateway.requests == []
    assert list(tmp_path.iterdir()) == []
    assert lines[0].startswith("comprobaciones previas: sí")
    assert "7 c `minimax-m3`" in lines[1]
