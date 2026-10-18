"""El prompt de `solo` y la petición que lo lleva, fijados por lo que producen y no por su código.

`docs/forward.md` congela lo que `solo` envía al modelo. Fijar el sha-256 del código fuente de las
funciones que renderizan el prompt era frágil por los dos lados: un cambio de espaciado lo rompía
sin cambiar un byte del prompt, y no decía nada de lo que hay entre el prompt y el proveedor. Aquí
se fija la salida:

    python -m crypto_agents.forward_render <directorio de la etapa 1>

- **El prompt.** Cada evaluación de `solo` se reconstruye desde las velas y el manifiesto, sin
  modelos, por el mismo camino que la corrida: `nodes.prepare_evaluation` y
  `prompts.solo_prompt`. Su `llm.prompt_digest` se compara con el que registró el journal en el
  primer intento del decisor. `prompt_digest` es el sha-256 de los bytes UTF-8 del texto que el
  router entrega al backend en ese intento, y nada más: ni la herramienta, ni los parámetros, ni el
  modelo.
- **La petición.** Lo que el digest no cubre se fija aparte. La petición se captura saliendo del
  adaptador de producción (`llm.OpenAIBackend.complete`, con el `ChatOpenAI` y el SDK reales)
  hacia un transporte simulado, y de su cuerpo JSON se vacía el contenido de cada mensaje: eso es
  la envoltura —mensajes y sus papeles, herramienta, `tool_choice`, modelo, temperatura y el resto
  de parámetros—, cuyo sha-256 queda registrado en `SOLO_REQUEST_ENVELOPE_SHA256`. Se comprueba
  además que el único mensaje lleva el prompt cuyo digest registró el journal.

Tres decisiones que lo mantienen sin gasto:

- **El contexto lleva el router del replay sin `fill_with`**: todas las ranuras son
  `CacheOnlyBackend` y la caché está vacía. Ni `prepare_evaluation` ni el renderizado lo usan, y
  un test lo comprueba con un router que levanta ante cualquier atributo.
- **La captura no sale de la máquina.** Usa la costura `_http_client` del adaptador, la misma que
  usan las pruebas, con un `httpx.MockTransport`; la clave es un literal y la URL termina en
  `.invalid`, que no resuelve nunca (RFC 6761). Las llamadas que sí van al proveedor siguen sin
  tocar esa costura.
- **No lee la clave.** Los ajustes de `.env` dan el id del exchange y el modelo del decisor; del
  bloque `openai` no se lee nada.

El contexto de cada entrada lo arma `ablation.entry_context`, la misma función que usan el conteo
previo de la ablación y el sondeo: antes eran tres copias, y esta comprobación fue la regresión de
su unificación.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

import httpx
from pydantic import Field

from crypto_agents import llm, prompts
from crypto_agents.ablation import entry_context, plan_from_manifest
from crypto_agents.audit import AuditError, file_sha256
from crypto_agents.cache import InMemoryResponseCache
from crypto_agents.context import utc_now
from crypto_agents.criteria import CriteriaError, render_invalid
from crypto_agents.llm import OpenAIBackend, prompt_digest
from crypto_agents.nodes import prepare_evaluation
from crypto_agents.paired_dispersion import (
    InvalidRunError,
    RecalculationError,
    commit_label,
    read_stage,
)
from crypto_agents.prompts import solo_prompt
from crypto_agents.quota import QuotaLedger
from crypto_agents.replay import replay_router
from crypto_agents.selection import HISTORY_DIR, SelectionError
from crypto_agents.settings import DEFAULT_ENV_FILE, ConfigError, load_settings
from crypto_agents.state import AgentRole, FrozenModel, Proposal

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from crypto_agents.ablation import ReplayPlan
    from crypto_agents.journal import EvaluationRecord
    from crypto_agents.llm import ModelRouter
    from crypto_agents.quota import Clock
    from crypto_agents.selection import PlannedEvaluation
    from crypto_agents.settings import ModelChoice, Settings
    from crypto_agents.state import LLMOutput

__all__ = [
    "CAPTURE_BASE_URL",
    "PROPOSAL_SCHEMA_SHA256",
    "REGISTERED_PATH",
    "RETRY_TEMPLATE_SHA256",
    "SOLO_REQUEST_ENVELOPE_SHA256",
    "SOLO_TEMPLATE",
    "SOLO_TEMPLATE_SHA256",
    "TARGET_ARM",
    "EnvelopeCheck",
    "ForwardRenderError",
    "MeasuredPath",
    "RenderCheck",
    "cache_only_router",
    "captured_request",
    "check_envelope",
    "check_run",
    "compare",
    "envelope_digest",
    "journal_digests",
    "main",
    "measured_path",
    "render_report",
    "request_envelope",
    "retry_template_digest",
    "sample_positions",
    "schema_digest",
    "solo_digests",
    "solo_prompts",
    "template_digest",
]

TARGET_ARM = "solo"

SOLO_REQUEST_ENVELOPE_SHA256 = "f0727ca0e3c51a242b7351e215a0c6596d1f9e7b777f2efda9d402f7ba3a6928"
"""sha-256 registrado de la envoltura de la petición del decisor de `solo` (`envelope_digest`).

Es la de la configuración de la etapa 1 —`glm-5.2`, `function_calling`, temperatura 0.0— con el
esquema `Proposal`. Lo fija `tests/test_forward_render.py` y lo cita `docs/forward.md`, apartado 3.
"""

SOLO_TEMPLATE = "decider_solo.md"
"""La plantilla del prompt de `solo`, dentro de `prompts.PROMPT_DIR`."""

SOLO_TEMPLATE_SHA256 = "fe622bc357f853db3e5b61bc06ff6312f47d1dfff4cc2eff06db15d20822f2de"
"""sha-256 registrado de los bytes de la plantilla (`docs/forward.md`, apartado 3, «El prompt»)."""

PROPOSAL_SCHEMA_SHA256 = "4970a9ec09cfc93e89e3058561325e55368f5b87f3f154d5f4e686a1f21e5bb7"
"""sha-256 registrado del esquema de `Proposal` en JSON canónico (apartado 3, «El decisor»)."""

RETRY_TEMPLATE_SHA256 = "588bf07e71d7a92a4d098a9659a3a9c4e05dc3347743ccc1f077e634a8e39890"
"""sha-256 registrado del texto de `llm._RETRY_TEMPLATE` en UTF-8 (apartado 3, «El decisor»)."""

CAPTURE_BASE_URL = "https://captura.invalid/v1"
"""Adonde iría la petición capturada si faltara el transporte simulado: un dominio que no
resuelve."""

_ENVELOPE_PROMPT = "envoltura"
"""El prompt con que se captura la envoltura de `measured_path`. Da igual cuál: la envoltura vacía
el contenido de cada mensaje, y un test fija que no depende del prompt."""

_CAPTURE_KEY = "sin-clave"
_CAPTURE_TIMEOUT_SECONDS = 5.0
_COMMAND = "python -m crypto_agents.forward_render"


class ForwardRenderError(RuntimeError):
    """La corrida no permite la comprobación, o la petición capturada no tiene la forma esperada."""


# ─────────────────────────────────────────── El prompt ───────────────────────────────────────────


def cache_only_router(settings: Settings, clock: Clock) -> ModelRouter:
    """El router del replay sin `fill_with`: `CacheOnlyBackend` en cada ranura, caché vacía."""
    return replay_router(
        settings, QuotaLedger(settings.quota_window, clock), InMemoryResponseCache(), clock
    )


async def solo_prompts(
    plan: ReplayPlan, settings: Settings, router: ModelRouter
) -> dict[UUID, str]:
    """El prompt de `solo` de cada entrada del plan cuyo gate abre, por `run_id`.

    Una entrada con el gate cerrado no llega a `decide_solo` en la corrida, y aquí tampoco tiene
    prompt. Una que no prepara es un plan roto y la excepción sube.
    """
    prompts: dict[UUID, str] = {}
    for entry in plan.entries:
        context = entry_context(plan, entry, settings, router)
        prepared = await prepare_evaluation(context)
        if not prepared.activation.should_run:
            continue
        prompts[context.run_id] = solo_prompt(
            prepared.snapshot, prepared.indicators, prepared.activation.triggers
        )
    return prompts


async def solo_digests(
    plan: ReplayPlan, settings: Settings, router: ModelRouter
) -> dict[UUID, str]:
    """`llm.prompt_digest` de cada prompt de `solo_prompts`, la función con que registra el
    router."""
    prompts = await solo_prompts(plan, settings, router)
    return {run_id: prompt_digest(prompt) for run_id, prompt in prompts.items()}


def journal_digests(records: Iterable[EvaluationRecord]) -> dict[UUID, str]:
    """El `prompt_digest` del primer intento del decisor en cada registro que lo llamó.

    El primero porque es el prompt renderizado: el de un reintento lleva además la plantilla de
    reintento con el error, y su digest es otro.
    """
    digests: dict[UUID, str] = {}
    for record in records:
        decider = [call for call in record.calls if call.role is AgentRole.DECIDER]
        if decider:
            digests[record.run_id] = decider[0].prompt_digest
    return digests


def sample_positions(entries: Sequence[PlannedEvaluation], per_symbol: int) -> tuple[int, ...]:
    """Las posiciones del manifiesto de las `per_symbol` primeras entradas de cada símbolo, en
    orden. Con 2, son las del fixture de `tests/data/solo_prompt_digests.jsonl`: no se eligen."""
    seen: dict[str, int] = {}
    positions: list[int] = []
    for position, entry in enumerate(entries):
        if seen.get(entry.symbol, 0) < per_symbol:
            seen[entry.symbol] = seen.get(entry.symbol, 0) + 1
            positions.append(position)
    return tuple(positions)


class RenderCheck(FrozenModel):
    """Los digests reconstruidos contra los del journal."""

    expected: int = Field(ge=0)
    """Registros del journal con una llamada del decisor."""

    rendered: int = Field(ge=0)
    matching: int = Field(ge=0)
    mismatched: tuple[UUID, ...] = ()
    """En los dos, con digest distinto."""

    missing: tuple[UUID, ...] = ()
    """En el journal y no reconstruidos."""

    unexpected: tuple[UUID, ...] = ()
    """Reconstruidos y no en el journal."""

    @property
    def passed(self) -> bool:
        return (
            self.expected > 0
            and self.matching == self.expected
            and not self.mismatched
            and not self.missing
            and not self.unexpected
        )


def compare(expected: Mapping[UUID, str], rendered: Mapping[UUID, str]) -> RenderCheck:
    """Coinciden, no coinciden, faltan o sobran, por `run_id`."""
    shared = expected.keys() & rendered.keys()
    return RenderCheck(
        expected=len(expected),
        rendered=len(rendered),
        matching=sum(1 for run_id in shared if expected[run_id] == rendered[run_id]),
        mismatched=tuple(sorted(r for r in shared if expected[r] != rendered[r])),
        missing=tuple(sorted(expected.keys() - rendered.keys())),
        unexpected=tuple(sorted(rendered.keys() - expected.keys())),
    )


# ────────────────────────────────────────── La petición ──────────────────────────────────────────


class _CapturingBackend(OpenAIBackend):
    """El adaptador de producción con un transporte que guarda la petición y contesta él mismo."""

    def __init__(self) -> None:
        super().__init__(
            api_key=_CAPTURE_KEY,
            base_url=CAPTURE_BASE_URL,
            timeout_seconds=_CAPTURE_TIMEOUT_SECONDS,
        )
        self.bodies: list[dict[str, object]] = []
        self.client = httpx.AsyncClient(transport=httpx.MockTransport(self._answer))

    def _http_client(self) -> object | None:
        return self.client

    def _answer(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if not isinstance(body, dict):
            raise ForwardRenderError("el cuerpo de la petición no es un objeto JSON")
        self.bodies.append(body)
        return httpx.Response(200, json=_reply(body))


def _tool_name(body: Mapping[str, object]) -> str | None:
    choice = body.get("tool_choice")
    function = choice.get("function") if isinstance(choice, dict) else None
    name = function.get("name") if isinstance(function, dict) else None
    return name if isinstance(name, str) else None


def _reply(body: Mapping[str, object]) -> dict[str, object]:
    """Una respuesta con la forma que el modo espera. Lo que diga da igual: la petición ya se
    guardó."""
    name = _tool_name(body)
    message: dict[str, object]
    if name is None:
        message = {"role": "assistant", "content": "{}"}
    else:
        call = {"id": "captura", "type": "function", "function": {"name": name, "arguments": "{}"}}
        message = {"role": "assistant", "content": None, "tool_calls": [call]}
    return {
        "id": "captura",
        "object": "chat.completion",
        "created": 0,
        "model": body.get("model"),
        "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
    }


async def captured_request(
    choice: ModelChoice, prompt: str, schema: type[LLMOutput] = Proposal
) -> dict[str, object]:
    """El cuerpo JSON que `OpenAIBackend.complete` envía con ese modelo, ese prompt y ese
    esquema."""
    backend = _CapturingBackend()
    try:
        await backend.complete(choice, prompt, schema)
    finally:
        await backend.client.aclose()
    if len(backend.bodies) != 1:
        raise ForwardRenderError(f"se esperaba una petición y salieron {len(backend.bodies)}")
    return backend.bodies[0]


def request_envelope(body: Mapping[str, object]) -> dict[str, object]:
    """El cuerpo con el contenido de cada mensaje vaciado: todo lo que `prompt_digest` no cubre."""
    messages = body.get("messages")
    if not isinstance(messages, list) or not all(isinstance(item, dict) for item in messages):
        raise ForwardRenderError("la petición no lleva una lista de mensajes")
    return {**body, "messages": [{**message, "content": ""} for message in messages]}


def envelope_digest(envelope: Mapping[str, object]) -> str:
    """sha-256 del JSON canónico: claves ordenadas, sin espacios, UTF-8."""
    canonical = json.dumps(envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _names(items: object, *path: str) -> tuple[str, ...]:
    """El valor en `path` de cada objeto de una lista JSON, para el reporte."""
    if not isinstance(items, list):
        return ()
    names: list[str] = []
    for item in items:
        value: object = item
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        names.append(str(value))
    return tuple(names)


def _user_content(body: Mapping[str, object]) -> str | None:
    """El contenido del único mensaje, si la petición lleva exactamente uno y es del usuario."""
    messages = body.get("messages")
    if not isinstance(messages, list) or len(messages) != 1:
        return None
    (message,) = messages
    if not isinstance(message, dict) or message.get("role") != "user":
        return None
    content = message.get("content")
    return content if isinstance(content, str) else None


class EnvelopeCheck(FrozenModel):
    """La petición del decisor de `solo` para un prompt: su forma, su envoltura y su mensaje."""

    model: str
    temperature: float
    mode: str
    keys: tuple[str, ...]
    roles: tuple[str, ...]
    tools: tuple[str, ...]
    digest: str
    registered: str
    sent_digest: str | None
    """`prompt_digest` del único mensaje del usuario; `None` si la petición no lleva solo ese."""

    expected_digest: str
    """El que registró el journal para ese prompt."""

    @property
    def envelope_matches(self) -> bool:
        return self.digest == self.registered

    @property
    def message_matches(self) -> bool:
        return self.sent_digest == self.expected_digest

    @property
    def passed(self) -> bool:
        return self.envelope_matches and self.message_matches


async def check_envelope(choice: ModelChoice, prompt: str, expected_digest: str) -> EnvelopeCheck:
    """Captura la petición de ese prompt y la compara con lo registrado."""
    body = await captured_request(choice, prompt)
    content = _user_content(body)
    return EnvelopeCheck(
        model=choice.model,
        temperature=choice.temperature,
        mode=choice.structured_output.value,
        keys=tuple(sorted(body)),
        roles=_names(body.get("messages"), "role"),
        tools=_names(body.get("tools"), "function", "name"),
        digest=envelope_digest(request_envelope(body)),
        registered=SOLO_REQUEST_ENVELOPE_SHA256,
        sent_digest=None if content is None else prompt_digest(content),
        expected_digest=expected_digest,
    )


# ──────────────────────────────────────── El camino medido ───────────────────────────────────────


class MeasuredPath(FrozenModel):
    """Los cuatro sha-256 que fijan lo que viaja al modelo en la llamada del decisor de `solo`."""

    template_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    retry_template_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    envelope_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def differences(self, registered: MeasuredPath) -> tuple[str, ...]:
        """Los campos que no son los registrados, por su nombre."""
        return tuple(
            name
            for name in type(self).model_fields
            if getattr(self, name) != getattr(registered, name)
        )


REGISTERED_PATH = MeasuredPath(
    template_sha256=SOLO_TEMPLATE_SHA256,
    schema_sha256=PROPOSAL_SCHEMA_SHA256,
    retry_template_sha256=RETRY_TEMPLATE_SHA256,
    envelope_sha256=SOLO_REQUEST_ENVELOPE_SHA256,
)
"""El camino medido tal como lo registra `docs/forward.md`, apartado 3."""


def template_digest() -> str:
    """sha-256 de los bytes de la plantilla de `solo`, leída de `prompts.PROMPT_DIR` al llamar."""
    return hashlib.sha256((prompts.PROMPT_DIR / SOLO_TEMPLATE).read_bytes()).hexdigest()


def schema_digest(schema: type[LLMOutput] = Proposal) -> str:
    """sha-256 del esquema en JSON canónico: claves ordenadas, sin espacios, UTF-8."""
    canonical = json.dumps(
        schema.model_json_schema(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def retry_template_digest() -> str:
    """sha-256 del texto de la plantilla de reintento del router, leído al llamar."""
    return hashlib.sha256(llm._RETRY_TEMPLATE.encode("utf-8")).hexdigest()


async def measured_path(choice: ModelChoice) -> MeasuredPath:
    """Los cuatro sha-256 de hoy. La envoltura, con el decisor `choice`.

    Nada de esto llama a un proveedor: la envoltura sale del adaptador de producción hacia el
    transporte simulado de `captured_request`.
    """
    body = await captured_request(choice, _ENVELOPE_PROMPT)
    return MeasuredPath(
        template_sha256=template_digest(),
        schema_sha256=schema_digest(),
        retry_template_sha256=retry_template_digest(),
        envelope_sha256=envelope_digest(request_envelope(body)),
    )


# ─────────────────────────────────────────── El reporte ──────────────────────────────────────────


def _yes(value: bool) -> str:
    return "sí" if value else "**no**"


def render_report(
    check: RenderCheck,
    envelope: EnvelopeCheck,
    where: Mapping[UUID, tuple[str, str]],
    header: Sequence[str],
) -> str:
    """Markdown en español. `where` da símbolo e instante de cada `run_id`, para nombrar las que
    fallan; `header` son las líneas de procedencia."""
    lines = ["# El prompt de `solo` y su petición", "", *header, ""]
    lines += [
        "## 1. El prompt, reconstruido sin modelos",
        "",
        "Cada evaluación de `solo` se reconstruye desde las velas y el manifiesto con "
        "`nodes.prepare_evaluation` y `prompts.solo_prompt`, y su `llm.prompt_digest` —el sha-256 "
        "de los bytes UTF-8 del texto que el router entrega al backend— se compara con el del "
        "primer intento del decisor en el journal.",
        "",
        "| en el journal | reconstruidas | coinciden | distintas | sin reconstruir | de más |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {check.expected} | {check.rendered} | {check.matching} | {len(check.mismatched)} | "
        f"{len(check.missing)} | {len(check.unexpected)} |",
    ]
    for label, run_ids in (
        ("distinta", check.mismatched),
        ("sin reconstruir", check.missing),
        ("de más", check.unexpected),
    ):
        for run_id in run_ids:
            symbol, at = where.get(run_id, ("?", "?"))
            lines.append(f"- {label}: `{run_id}`, {symbol} {at}")
    lines += [
        "",
        "## 2. La petición que lo lleva al modelo",
        "",
        "La petición sale del adaptador de producción hacia un transporte simulado; nada sale "
        "de la máquina. La envoltura es su cuerpo JSON con el contenido de cada mensaje vaciado, "
        "y su sha-256 el del JSON canónico (claves ordenadas, sin espacios, UTF-8).",
        "",
        f"- decisor: `{envelope.model}`, `{envelope.mode}`, temperatura {envelope.temperature}",
        f"- claves del cuerpo: {', '.join(f'`{key}`' for key in envelope.keys)}",
        f"- mensajes: {', '.join(envelope.roles) or 'ninguno'}",
        f"- herramientas: {', '.join(f'`{tool}`' for tool in envelope.tools) or 'ninguna'}",
        f"- el único mensaje es el prompt cuyo digest registró el journal: "
        f"{_yes(envelope.message_matches)}",
        f"- sha-256 de la envoltura: `{envelope.digest}`",
        f"- registrado: `{envelope.registered}`",
        f"- coincide: {_yes(envelope.envelope_matches)}",
        "",
        f"Resultado: {'todo coincide' if check.passed and envelope.passed else '**no coincide**'}.",
    ]
    return "\n".join(lines)


async def check_run(
    plan: ReplayPlan, settings: Settings, records: Sequence[EvaluationRecord]
) -> tuple[RenderCheck, EnvelopeCheck]:
    """Las dos comprobaciones sobre un plan y los registros de `solo` que lo corrieron.

    El prompt, con el router del replay sin `fill_with`. La petición, con el decisor de `settings`
    y el prompt de la primera evaluación reconstruida que está en el journal.
    """
    expected = journal_digests(records)
    prompts = await solo_prompts(plan, settings, cache_only_router(settings, utc_now))
    rendered = {run_id: prompt_digest(prompt) for run_id, prompt in prompts.items()}
    first = next((run_id for run_id in prompts if run_id in expected), None)
    if first is None:
        raise ForwardRenderError("ninguna evaluación reconstruida está en el journal")
    choice = settings.role_choices(AgentRole.DECIDER)[0]
    envelope = await check_envelope(choice, prompts[first], expected[first])
    return compare(expected, rendered), envelope


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada: `python -m crypto_agents.forward_render <directorio>`.

    Sale con 0 si los prompts y la petición coinciden con lo registrado, con 1 si algo no coincide
    o la corrida es inválida, y con 2 si el directorio o la configuración no se pudieron leer.
    """
    parser = argparse.ArgumentParser(
        prog=_COMMAND,
        description="El prompt de solo reconstruido sin modelos, contra el journal de la corrida.",
    )
    parser.add_argument("directory", type=Path, help="directorio de la corrida, con meta.json")
    parser.add_argument("--plan", type=Path, default=None, help="manifiesto del plan")
    parser.add_argument("--history-dir", type=Path, default=HISTORY_DIR)
    args = parser.parse_args(argv)

    try:
        stage = read_stage(args.directory, args.plan, args.history_dir, None)
        records = stage.arms.get(TARGET_ARM)
        if records is None:
            raise ForwardRenderError(f"la corrida no tiene el brazo {TARGET_ARM}")
        plan = plan_from_manifest(stage.manifest, stage.histories)
        settings = load_settings(DEFAULT_ENV_FILE)
        check, envelope = asyncio.run(check_run(plan, settings, records))
    except InvalidRunError as error:
        print(render_invalid(error.validity, str(args.directory)))
        return 1
    except (
        AuditError,
        SelectionError,
        CriteriaError,
        RecalculationError,
        ConfigError,
        ForwardRenderError,
        OSError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    run = stage.chain[0]
    journal = run.path / f"{TARGET_ARM}.jsonl"
    flags = [
        *([f"--plan {args.plan}"] if args.plan is not None else []),
        *([f"--history-dir {args.history_dir}"] if args.history_dir != HISTORY_DIR else []),
    ]
    header = [
        f"Reproduce: `{' '.join([_COMMAND, str(args.directory), *flags])}`",
        "",
        f"- commit de la corrida: {commit_label(run.meta.git_commit, run.meta.git_dirty)}",
        f"- plan: `{stage.plan_path}`, sha-256 `{run.meta.plan_sha256}`",
        f"- `{journal.name}` sha-256 `{file_sha256(journal)}`",
        f"- decisor leído de `{DEFAULT_ENV_FILE}`; de su bloque `openai` no se lee nada",
    ]
    where = {
        record.run_id: (record.symbol, record.at.isoformat().replace("+00:00", "Z"))
        for record in records
    }
    print(render_report(check, envelope, where, header))
    return 0 if check.passed and envelope.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
