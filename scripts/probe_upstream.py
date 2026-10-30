"""Sondeo de diagnóstico: quién contesta hoy a la petición del decisor de `solo`, y si lo dice.

    uv run python scripts/probe_upstream.py --env-file <.env> [--out data/probes] [--dry-run]

Las 42 llamadas vivas del decisor en el lote del bloque 1 llegaron sin
`x-opencode-upstream-model-id` ni `x-opencode-endpoint-id`, y más rápidas que en la etapa 1. Este
script repite la petición del estudio, fuera del estudio, y guarda de cada respuesta todo lo que
puede decir quién la sirvió. Siete llamadas, una detrás de otra:

- a) tres a `glm-5.2`, exactamente como las manda el lote: el mismo adaptador, el mismo modo
  `function_calling`, el mismo esquema `Proposal` y la misma envoltura, con una sesión aleatoria
  por proceso, como `OpenAIBackend`;
- b) tres iguales, más un User-Agent fijo y un `x-opencode-session` estable;
- c) una tal cual a `minimax-m3`, que en los sondeos del bloque T mandó las dos cabeceras por el
  mismo endpoint, `fireworks`, por el que salía `glm-5.2`.

El prompt es el de la primera evaluación de `data/stage1/solo.jsonl`, renderizado con las funciones
del lote, para poder compararlo con su llamada de la etapa 1.

Antes de la primera llamada se niega, y sale con 1 sin haber hecho ninguna, si no es Zen con pago
por uso, si `data/stage1/` no es la publicada, si el decisor no es el registrado, si el camino
medido no es el registrado, si el prompt no da el `prompt_digest` del journal o si el directorio de
salida ya existe.

Cada llamada pasa por `ModelRouter` (regla 4), sin caché, con un intento —una respuesta inválida
no se reintenta, así que son siete peticiones y ni una más— y con un tope de 0.20 USD que no abre
una llamada cuando lo gastado lo alcanza. Las cabeceras se leen con un hook sobre el cliente HTTP
del SDK que pasa por la costura `_http_client` del adaptador, sin tocar `llm.py`. Las llamadas son
secuenciales, así que cada respuesta es de la llamada en curso.

Escribe `data/probes/<inicio UTC>-upstream/`: `meta.json`, `calls.jsonl` (las filas `LLMCall`),
`responses.jsonl` (una línea por llamada) y `report.md`. De cada respuesta guarda todas las
cabeceras menos `set-cookie`; del cuerpo, `model`, `id`, `system_fingerprint` y el orden de sus
claves; de los argumentos, el orden de sus claves; tokens de salida, latencia, tok/s y la hora UTC.
De la petición solo guarda el sha-256 de su envoltura, el User-Agent y la sesión: nada que lleve
credenciales. El `.env` lo lee `load_settings` y no se imprime nada de él.

Sale con 0 si las siete llamadas se hicieron, con 1 si se negó antes de llamar o el tope dejó
alguna sin hacer, y con 2 si no pudo leer la configuración o los argumentos.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import socket
import sys
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from openai import DefaultAsyncHttpxClient
from pydantic import AwareDatetime

from crypto_agents.ablation import git_state, plan_from_manifest
from crypto_agents.audit import AuditError, read_meta
from crypto_agents.context import utc_now
from crypto_agents.criteria import CriteriaError
from crypto_agents.forward import ForwardPaths, decider_check, read_decider
from crypto_agents.forward_data import ForwardDataError, verify_stage1
from crypto_agents.forward_record import (
    REGISTERED_MAX_ATTEMPTS,
    REGISTERED_PATH,
    TARGET_ARM,
    ForwardError,
    registered_decider,
)
from crypto_agents.forward_render import (
    ForwardRenderError,
    cache_only_router,
    envelope_digest,
    journal_digests,
    measured_path,
    request_envelope,
    solo_prompts,
)
from crypto_agents.llm import (
    SESSION_HEADER,
    ModelInvocationError,
    ModelRouter,
    OpenAIBackend,
    SpendCapReachedError,
    prompt_digest,
)
from crypto_agents.paired_dispersion import RecalculationError, read_stage
from crypto_agents.quota import QuotaLedger
from crypto_agents.selection import SelectionError
from crypto_agents.settings import (
    Backend,
    ConfigError,
    RoleConfig,
    Settings,
    load_settings,
    public_url,
)
from crypto_agents.spend import SpendGuard
from crypto_agents.state import AgentRole, FrozenModel, LLMCall, Proposal
from crypto_agents.zen_probe import refuse_unless_zen

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import httpx

    from crypto_agents.quota import Clock

CAP_USD = 0.20
"""Tope de las siete llamadas, en el extremo alto de lo que el proveedor declara."""

MAX_ATTEMPTS = 1
"""Un intento por llamada: una respuesta inválida no se reintenta. La petición es la misma que
con los dos intentos del estudio; lo que cambia es que no hay una octava."""

CONTROL_MODEL = "minimax-m3"
CONTROL_FAMILY = "minimax"
"""El modelo de control de la variante c. En el bloque T5 su `Ping` y en el T6 sus 12 llamadas
trajeron las dos cabeceras: `accounts/anomalyinc/routers/zen-minimax-m3` por `fireworks`, el mismo
endpoint por el que salía `glm-5.2`."""

FIXED_USER_AGENT = "crypto-agents-sonda/1"
STABLE_SESSION = "crypto-agents-sonda-upstream"
"""Lo que añade la variante b: un User-Agent fijo y una sesión que no cambia entre procesos."""

EXCLUDED_HEADER = "set-cookie"
"""La única cabecera de respuesta que no se guarda."""

PROBES_DIR = Path("data/probes")
SUFFIX = "-upstream"


class Variant(StrEnum):
    """Las tres formas de la petición."""

    STUDY = "a"
    HEADERS = "b"
    CONTROL = "c"


PLAN: tuple[Variant, ...] = (
    Variant.STUDY,
    Variant.STUDY,
    Variant.STUDY,
    Variant.HEADERS,
    Variant.HEADERS,
    Variant.HEADERS,
    Variant.CONTROL,
)
"""Las siete llamadas, en el orden en que se hacen."""


class ProbeRefusedError(RuntimeError):
    """Una comprobación previa falló: no se hace ninguna llamada."""


# ─────────────────────────────────────── El adaptador ───────────────────────────────────────────


class Exchange(NamedTuple):
    """Una respuesta HTTP y lo que se guarda de la petición que la produjo."""

    status: int
    headers: tuple[tuple[str, str], ...]
    body: bytes
    request_body: bytes
    user_agent: str | None
    session: str | None


def kept_headers(headers: httpx.Headers) -> tuple[tuple[str, str], ...]:
    """Las cabeceras de la respuesta, repetidas incluidas y en su orden, menos `set-cookie`."""
    return tuple(
        (name, value) for name, value in headers.multi_items() if name.lower() != EXCLUDED_HEADER
    )


class RecordingBackend(OpenAIBackend):
    """El adaptador de producción con un hook que guarda cada respuesta que recibe.

    `extra_headers` se suma a las cabeceras del adaptador (la variante b). `transport` existe para
    las pruebas: en la sonda es `None` y el cliente es el que el SDK construye por defecto.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str | None,
        timeout_seconds: float,
        session_id: str | None = None,
        extra_headers: Mapping[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(api_key, base_url, timeout_seconds, session_id)
        self._extra = dict(extra_headers or {})
        self.exchanges: list[Exchange] = []
        self.client = DefaultAsyncHttpxClient(
            transport=transport, event_hooks={"response": [self._seen]}
        )

    def _headers(self) -> dict[str, str]:
        return {**super()._headers(), **self._extra}

    def _http_client(self) -> object | None:
        return self.client

    async def _seen(self, response: httpx.Response) -> None:
        await response.aread()
        request = response.request
        self.exchanges.append(
            Exchange(
                status=response.status_code,
                headers=kept_headers(response.headers),
                body=response.content,
                request_body=request.content,
                user_agent=request.headers.get("user-agent"),
                session=request.headers.get(SESSION_HEADER),
            )
        )


# ────────────────────────────────────── Lo que se guarda ─────────────────────────────────────────


class ProbeCall(FrozenModel):
    """Una llamada de la sonda: lo que dijo la respuesta y lo que registró el router."""

    sequence: int
    variant: Variant
    model: str
    """El id pedido."""

    at: AwareDatetime | None
    """`LLMCall.at`: cuándo terminó la llamada, en UTC. `None` si el tope no la dejó abrir."""

    status: int | None
    headers: tuple[tuple[str, str], ...] = ()
    body_keys: tuple[str, ...] | None = None
    body_model: str | None = None
    body_id: str | None = None
    system_fingerprint: str | None = None
    argument_keys: tuple[str, ...] | None = None
    """El orden de las claves de los argumentos de la herramienta, como llegaron."""

    completion_tokens: int | None = None
    latency_ms: float | None = None
    tokens_per_second: float | None = None
    upstream_model: str | None = None
    upstream_endpoint: str | None = None
    """Lo que leyó el router de las dos cabeceras, para cotejarlo con `headers`."""

    valid: bool | None = None
    failure: str | None = None
    request_envelope_sha256: str | None = None
    request_user_agent: str | None = None
    request_session: str | None = None
    prompt_matches: bool | None = None
    """Si el mensaje que salió es el prompt de la etapa 1."""


class Reference(FrozenModel):
    """La llamada de la etapa 1 con la que se compara."""

    run_id: str
    symbol: str
    evaluated_at: AwareDatetime
    prompt_digest: str
    at: AwareDatetime
    latency_ms: float | None
    completion_tokens: int | None
    upstream_model: str | None
    upstream_endpoint: str | None


class ProbeMeta(FrozenModel):
    """Lo que la sonda deja escrito de sí misma."""

    started_at: AwareDatetime
    finished_at: AwareDatetime | None = None
    hostname: str
    git_commit: str | None
    git_dirty: bool | None
    script_sha256: str
    """El sha-256 de este archivo: nombra el código que corrió aunque el árbol esté sucio."""

    base_url: str | None
    reference: Reference
    envelope_sha256: str
    """La envoltura medida de la variante a: es la registrada, o la sonda no habría llamado."""

    decider_model: str
    control_model: str
    max_attempts: int
    cap_usd: float
    spent_usd: float | None = None
    unpriced_calls: int | None = None
    refused_calls: int | None = None
    plan: tuple[Variant, ...]


def tokens_per_second(tokens: int | None, latency_ms: float | None) -> float | None:
    """Tokens de salida por segundo de pared, o `None` sin los dos números."""
    if tokens is None or latency_ms is None or latency_ms <= 0:
        return None
    return tokens / (latency_ms / 1000)


def _keys(text: str) -> tuple[str, ...] | None:
    """Las claves de un objeto JSON en el orden en que llegaron, o `None` si no es un objeto."""
    try:
        pairs = json.loads(text, object_pairs_hook=lambda items: items)
    except json.JSONDecodeError:
        return None
    if not isinstance(pairs, list):
        return None
    return tuple(str(key) for key, _ in pairs)


def _text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _arguments(body: Mapping[str, object]) -> str | None:
    """Los argumentos crudos de la primera herramienta, como texto."""
    choices = body.get("choices")
    first = choices[0] if isinstance(choices, list) and choices else None
    message = first.get("message") if isinstance(first, dict) else None
    calls = message.get("tool_calls") if isinstance(message, dict) else None
    call = calls[0] if isinstance(calls, list) and calls else None
    function = call.get("function") if isinstance(call, dict) else None
    return _text(function.get("arguments")) if isinstance(function, dict) else None


def _user_content(body: Mapping[str, object]) -> str | None:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return None
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            return _text(message.get("content"))
    return None


def describe(
    sequence: int,
    variant: Variant,
    model: str,
    prompt: str,
    call: LLMCall | None,
    exchange: Exchange | None,
    failure: str | None,
) -> ProbeCall:
    """Lo que se guarda de una llamada: la respuesta HTTP y la fila del router."""
    fields: dict[str, object] = {}
    if exchange is not None:
        request = json.loads(exchange.request_body)
        fields |= {
            "status": exchange.status,
            "headers": exchange.headers,
            "request_envelope_sha256": envelope_digest(request_envelope(request)),
            "request_user_agent": exchange.user_agent,
            "request_session": exchange.session,
            "prompt_matches": _user_content(request) == prompt,
        }
        try:
            body = json.loads(exchange.body)
        except json.JSONDecodeError:
            body = None
        if isinstance(body, dict):
            arguments = _arguments(body)
            fields |= {
                "body_keys": _keys(exchange.body.decode("utf-8")),
                "body_model": _text(body.get("model")),
                "body_id": _text(body.get("id")),
                "system_fingerprint": _text(body.get("system_fingerprint")),
                "argument_keys": None if arguments is None else _keys(arguments),
            }
    if call is not None:
        fields |= {
            "at": call.at,
            "completion_tokens": call.completion_tokens,
            "latency_ms": call.latency_ms,
            "tokens_per_second": tokens_per_second(call.completion_tokens, call.latency_ms),
            "upstream_model": call.upstream_model,
            "upstream_endpoint": call.upstream_endpoint,
            "valid": call.valid,
        }
    return ProbeCall.model_validate(
        {
            "sequence": sequence,
            "variant": variant,
            "model": model,
            "at": None,
            "status": None,
            "failure": failure,
            **fields,
        }
    )


# ────────────────────────────────────── Las comprobaciones ───────────────────────────────────────


class Prepared(NamedTuple):
    """Lo que las comprobaciones dejan listo: el prompt, su referencia y la envoltura medida."""

    prompt: str
    reference: Reference
    envelope_sha256: str


_READ_ERRORS = (
    AuditError,
    CriteriaError,
    RecalculationError,
    SelectionError,
    ForwardError,
    ForwardDataError,
    ForwardRenderError,
    OSError,
)


async def prepare(settings: Settings, paths: ForwardPaths) -> Prepared:
    """Las cinco comprobaciones previas. Cualquiera que falle levanta `ProbeRefusedError`."""
    try:
        refuse_unless_zen(settings)
    except ConfigError as error:
        raise ProbeRefusedError(str(error)) from error
    try:
        verify_stage1(paths.stage1)
        registered = registered_decider(read_meta(paths.stage1))
        stage = read_stage(paths.stage1, None, paths.published_history, None)
    except _READ_ERRORS as error:
        raise ProbeRefusedError(f"la etapa 1 no se pudo leer: {error}") from error
    decider = decider_check(
        registered, read_decider(settings, REGISTERED_MAX_ATTEMPTS), settings.billing
    )
    if not decider.passed:
        raise ProbeRefusedError(f"el decisor no es el registrado: {decider.detail}")
    choice = settings.role_choices(AgentRole.DECIDER)[0]
    measured = await measured_path(choice)
    different = measured.differences(REGISTERED_PATH)
    if different:
        raise ProbeRefusedError(f"el camino medido no es el registrado: {', '.join(different)}")

    records = stage.arms.get(TARGET_ARM, ())
    if not records:
        raise ProbeRefusedError(f"la etapa 1 no tiene registros de {TARGET_ARM}")
    first = records[0]
    expected = journal_digests([first]).get(first.run_id)
    if expected is None:
        raise ProbeRefusedError(f"la primera evaluación de {TARGET_ARM} no llamó al decisor")
    plan = plan_from_manifest(stage.manifest, stage.histories)
    prompts = await solo_prompts(plan, settings, cache_only_router(settings, utc_now))
    prompt = prompts.get(first.run_id)
    if prompt is None or prompt_digest(prompt) != expected:
        raise ProbeRefusedError(
            f"el prompt renderizado de {first.run_id} no da el prompt_digest del journal"
        )
    call = next(call for call in first.calls if call.role is AgentRole.DECIDER)
    reference = Reference(
        run_id=str(first.run_id),
        symbol=first.symbol,
        evaluated_at=first.at,
        prompt_digest=expected,
        at=call.at,
        latency_ms=call.latency_ms,
        completion_tokens=call.completion_tokens,
        upstream_model=call.upstream_model,
        upstream_endpoint=call.upstream_endpoint,
    )
    return Prepared(prompt, reference, measured.envelope_sha256)


def output_directory(root: Path, started: datetime) -> Path:
    """`<root>/<inicio UTC>-upstream`: con la hora, para que dos tandas del mismo día no choquen."""
    return root / f"{started.astimezone(UTC):%Y%m%dT%H%M%SZ}{SUFFIX}"


# ─────────────────────────────────────────── La sonda ────────────────────────────────────────────


def _settings_for(settings: Settings, variant: Variant) -> Settings:
    """Los ajustes con que llama cada variante: el decisor registrado, o el de control."""
    if variant is not Variant.CONTROL:
        return settings
    choice = settings.role_choices(AgentRole.DECIDER)[0]
    control = choice.model_copy(update={"model": CONTROL_MODEL, "family": CONTROL_FAMILY})
    roles = {**settings.roles, AgentRole.DECIDER: RoleConfig(primary=control)}
    return settings.model_copy(update={"roles": roles})


def backends_for(
    settings: Settings, transport: httpx.AsyncBaseTransport | None
) -> dict[Variant, RecordingBackend]:
    """Un backend por variante, como el estudio: uno por proceso, con su sesión.

    a y c son el adaptador tal cual, cada uno con su sesión aleatoria; b le añade el User-Agent fijo
    y la sesión estable.
    """
    openai = settings.openai
    if openai is None:
        raise ProbeRefusedError("falta el bloque openai de la configuración")
    key = openai.api_key.get_secret_value()

    def build(
        session_id: str | None = None, extra_headers: Mapping[str, str] | None = None
    ) -> RecordingBackend:
        return RecordingBackend(
            key,
            openai.base_url,
            openai.timeout_seconds,
            session_id=session_id,
            extra_headers=extra_headers,
            transport=transport,
        )

    return {
        Variant.STUDY: build(),
        Variant.HEADERS: build(STABLE_SESSION, {"User-Agent": FIXED_USER_AGENT}),
        Variant.CONTROL: build(),
    }


class ProbeRun(NamedTuple):
    """Lo que la sonda hizo: una línea por llamada, las filas del router y el tope."""

    calls: tuple[ProbeCall, ...]
    rows: tuple[LLMCall, ...]
    guard: SpendGuard


async def probe(
    settings: Settings,
    prompt: str,
    clock: Clock = utc_now,
    transport: httpx.AsyncBaseTransport | None = None,
) -> ProbeRun:
    """Las siete llamadas, en orden, cada una por un router sin caché y con un intento."""
    guard = SpendGuard(CAP_USD)
    ledger = QuotaLedger(settings.quota_window, clock)
    backends = backends_for(settings, transport)
    routers = {
        variant: ModelRouter(
            _settings_for(settings, variant),
            ledger,
            {Backend.OPENAI: backend},
            clock,
            cache=None,
            max_attempts=MAX_ATTEMPTS,
            guard=guard,
        )
        for variant, backend in backends.items()
    }
    calls: list[ProbeCall] = []
    rows: list[LLMCall] = []
    try:
        for sequence, variant in enumerate(PLAN, start=1):
            backend = backends[variant]
            before = len(backend.exchanges)
            failure: str | None = None
            try:
                _, attempts = await routers[variant].invoke(AgentRole.DECIDER, prompt, Proposal)
            except SpendCapReachedError as error:
                attempts, failure = list(error.calls), f"negada por el tope: {error}"
            except ModelInvocationError as error:
                attempts, failure = list(error.calls), str(error)
            seen = backend.exchanges[before:]
            if len(seen) > 1 or len(attempts) > 1:
                raise ProbeRefusedError(
                    f"la llamada {sequence} hizo {len(seen)} peticiones y {len(attempts)} intentos"
                )
            rows += attempts
            model = _settings_for(settings, variant).role_choices(AgentRole.DECIDER)[0].model
            calls.append(
                describe(
                    sequence,
                    variant,
                    model,
                    prompt,
                    attempts[0] if attempts else None,
                    seen[0] if seen else None,
                    failure,
                )
            )
    finally:
        for backend in backends.values():
            await backend.client.aclose()
    return ProbeRun(tuple(calls), tuple(rows), guard)


# ─────────────────────────────────────────── El reporte ──────────────────────────────────────────


def _cell(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.1f}"
    if isinstance(value, tuple):
        return ", ".join(f"`{item}`" for item in value) or "—"
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    return f"`{value}`".replace("|", "\\|")


def _call_table(call: ProbeCall) -> list[str]:
    rows: list[tuple[str, object]] = [
        ("variante", call.variant.value),
        ("modelo pedido", call.model),
        ("hora UTC (fin de la llamada)", call.at),
        ("estado HTTP", call.status),
        ("cuerpo: `model`", call.body_model),
        ("cuerpo: `id`", call.body_id),
        ("cuerpo: `system_fingerprint`", call.system_fingerprint),
        ("cuerpo: orden de las claves", call.body_keys),
        ("argumentos: orden de las claves", call.argument_keys),
        ("completion_tokens", call.completion_tokens),
        ("latency_ms", call.latency_ms),
        ("tok/s", call.tokens_per_second),
        ("router: `upstream_model`", call.upstream_model),
        ("router: `upstream_endpoint`", call.upstream_endpoint),
        ("válida", call.valid),
        ("fallo", call.failure),
        ("petición: sha-256 de la envoltura", call.request_envelope_sha256),
        ("petición: User-Agent", call.request_user_agent),
        ("petición: `x-opencode-session`", call.request_session),
        ("petición: el prompt es el de la etapa 1", call.prompt_matches),
    ]
    rows += [(f"cabecera `{name}`", value) for name, value in call.headers]
    return [
        f"### Llamada {call.sequence} — variante {call.variant.value}, `{call.model}`",
        "",
        "| campo | valor |",
        "| --- | --- |",
        *(f"| {name} | {_cell(value)} |" for name, value in rows),
        "",
    ]


def render_report(meta: ProbeMeta, calls: Sequence[ProbeCall]) -> list[str]:
    """El reporte: la referencia de la etapa 1, un resumen y una tabla por llamada."""
    reference = meta.reference
    speed = tokens_per_second(reference.completion_tokens, reference.latency_ms)
    names = sorted({name for call in calls for name, _ in call.headers if "opencode" in name})
    lines = [
        f"# Sondeo del upstream del decisor — {meta.started_at:%Y-%m-%dT%H:%M:%SZ}",
        "",
        f"- commit: `{meta.git_commit}` (sucio: {meta.git_dirty}); script sha-256 "
        f"`{meta.script_sha256}`; máquina `{meta.hostname}`",
        f"- base_url: `{meta.base_url}`; un intento por llamada, sin caché; tope "
        f"{meta.cap_usd:.2f} USD, gastado {_cell(meta.spent_usd)} USD en el extremo alto "
        f"({meta.unpriced_calls} sin tokens, {meta.refused_calls} negadas)",
        f"- envoltura de la variante a: `{meta.envelope_sha256}` (la registrada)",
        f"- prompt: el de `{reference.run_id}` ({reference.symbol}, evaluado "
        f"{_cell(reference.evaluated_at)}), `prompt_digest` `{reference.prompt_digest}`",
        "",
        "## La llamada de la etapa 1 con el mismo prompt",
        "",
        "| hora UTC | latency_ms | completion_tokens | tok/s | upstream_model "
        "| upstream_endpoint |",
        "| --- | --- | --- | --- | --- | --- |",
        f"| {_cell(reference.at)} | {_cell(reference.latency_ms)} | "
        f"{_cell(reference.completion_tokens)} | {_cell(speed)} | "
        f"{_cell(reference.upstream_model)} | {_cell(reference.upstream_endpoint)} |",
        "",
        "## Resumen",
        "",
        "| # | variante | modelo | hora UTC | estado | tok/s | cabeceras `x-opencode-*` | "
        "`model` del cuerpo | `id` del cuerpo |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for call in calls:
        present = sorted({name for name, _ in call.headers if "opencode" in name})
        lines.append(
            f"| {call.sequence} | {call.variant.value} | `{call.model}` | {_cell(call.at)} | "
            f"{_cell(call.status)} | {_cell(call.tokens_per_second)} | "
            f"{', '.join(f'`{name}`' for name in present) or 'ninguna'} | "
            f"{_cell(call.body_model)} | {_cell(call.body_id)} |"
        )
    lines += [
        "",
        f"Cabeceras `opencode` vistas en alguna llamada: "
        f"{', '.join(f'`{name}`' for name in names) or 'ninguna'}.",
        "",
        "## Una tabla por llamada",
        "",
    ]
    for call in calls:
        lines += _call_table(call)
    return lines


# ─────────────────────────────────────────── El comando ──────────────────────────────────────────


def _write_jsonl(path: Path, items: Sequence[FrozenModel]) -> None:
    path.write_text("".join(item.model_dump_json() + "\n" for item in items), encoding="utf-8")


def write_probe(directory: Path, meta: ProbeMeta, run: ProbeRun) -> None:
    """Los cuatro archivos del directorio de la sonda."""
    (directory / "meta.json").write_text(meta.model_dump_json(indent=2) + "\n", encoding="utf-8")
    _write_jsonl(directory / "calls.jsonl", run.rows)
    _write_jsonl(directory / "responses.jsonl", run.calls)
    (directory / "report.md").write_text(
        "\n".join(render_report(meta, run.calls)) + "\n", encoding="utf-8"
    )


async def run(
    settings: Settings,
    paths: ForwardPaths,
    out: Path,
    dry_run: bool,
    clock: Clock = utc_now,
    transport: httpx.AsyncBaseTransport | None = None,
    echo: Callable[[str], None] = print,
) -> int:
    """Comprueba, llama y escribe. Devuelve el código de salida."""
    started = clock()
    directory = output_directory(out, started)
    try:
        if directory.exists():
            raise ProbeRefusedError(
                f"{directory} ya existe: cada tanda escribe en un directorio propio"
            )
        prepared = await prepare(settings, paths)
    except ProbeRefusedError as refusal:
        echo(f"se niega antes de llamar: {refusal}")
        return 1
    plan = ", ".join(
        f"{sequence} {variant.value} "
        f"`{_settings_for(settings, variant).role_choices(AgentRole.DECIDER)[0].model}`"
        for sequence, variant in enumerate(PLAN, start=1)
    )
    echo(f"comprobaciones previas: sí; prompt de {prepared.reference.run_id}")
    echo(f"llamadas: {plan}")
    if dry_run:
        echo(f"--dry-run: no se llama a nadie; escribiría {directory}")
        return 0

    commit, dirty = git_state()
    meta = ProbeMeta(
        started_at=started,
        hostname=socket.gethostname(),
        git_commit=commit,
        git_dirty=dirty,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        base_url=public_url(None if settings.openai is None else settings.openai.base_url),
        reference=prepared.reference,
        envelope_sha256=prepared.envelope_sha256,
        decider_model=settings.role_choices(AgentRole.DECIDER)[0].model,
        control_model=CONTROL_MODEL,
        max_attempts=MAX_ATTEMPTS,
        cap_usd=CAP_USD,
        plan=PLAN,
    )
    directory.mkdir(parents=True)
    done = await probe(settings, prepared.prompt, clock, transport)
    meta = meta.model_copy(
        update={
            "finished_at": clock(),
            "spent_usd": done.guard.spent_usd,
            "unpriced_calls": done.guard.unpriced,
            "refused_calls": len(done.guard.refused),
        }
    )
    write_probe(directory, meta, done)
    echo(f"escrito {directory}; gastado {done.guard.spent_usd:.4f} USD en el extremo alto")
    made = sum(1 for call in done.calls if call.status is not None)
    return 0 if made == len(PLAN) else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scripts/probe_upstream.py",
        description="Siete llamadas vivas para ver quién sirve al decisor de solo y si lo dice.",
    )
    parser.add_argument("--env-file", type=Path, required=True, help="el .env con la clave")
    parser.add_argument("--out", type=Path, default=PROBES_DIR, help="raíz de los directorios")
    parser.add_argument("--dry-run", action="store_true", help="comprueba y no llama")
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.env_file)
    except ConfigError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return asyncio.run(run(settings, ForwardPaths(), args.out, args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
