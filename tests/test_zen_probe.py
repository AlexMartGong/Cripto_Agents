"""El sondeo de Zen, con backends falsos: ninguna prueba toca la red ni cuesta dinero.

Lo que se comprueba es el contrato del sondeo, no a Zen: que cada llamada deja su fila (regla 4),
que el directorio lo lee `audit` y lo suma `consumption` sin saber que es un sondeo, que un id
ausente del catálogo se sondea igual, que un rechazo no se confunde con un modo que no funciona, y
que se niega a correr contra Go.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from crypto_agents import zen_probe
from crypto_agents.ablation import open_run_directory, plan_from_manifest
from crypto_agents.audit import RunKind, chain_calls, read_run
from crypto_agents.consumption import consume
from crypto_agents.consumption import main as consumption_main
from crypto_agents.llm import Completion, TokenUsage
from crypto_agents.prompts import debate_prompt
from crypto_agents.settings import DEFAULT_PRICING, ConfigError
from crypto_agents.state import (
    AgentRole,
    Backend,
    Billing,
    Dimension,
    FailureKind,
    Side,
    StructuredOutputMode,
)
from crypto_agents.zen_probe import (
    BULL_CANDIDATES,
    CANDIDATES,
    PRESENT,
    VERDICTS,
    ProbeFindings,
    SpendGuard,
    _NoSessionBackend,
    all_arms,
    all_desk_arms,
    arm_name,
    build_meta,
    decider_label,
    desk_subjects,
    main,
    redact,
    refuse_unless_zen,
    render_desks_report,
    render_families,
    render_report,
    run_desks_probe,
    run_probe,
    subjects_of,
    technical_source,
)
from tests.conftest import PRESET, FakeLLM
from tests.test_ablation import two_symbol_manifest
from tests.test_consumption import call as priced_call
from tests.test_zen_payg import payg_settings

if TYPE_CHECKING:
    from pathlib import Path

    from crypto_agents.llm import ChatBackend
    from crypto_agents.settings import ModelChoice, Settings

NOW = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)
COUNT = 3
"""Veredictos por brazo en las pruebas: los 12 reales multiplican el tiempo sin probar nada más."""

USAGE = TokenUsage(prompt_tokens=1000, cached_tokens=0, completion_tokens=200)
OFFLINE_MANIFEST = "data/ablation_selection.json"


class ProviderRejectedError(RuntimeError):
    """Lo que el cliente de un proveedor lanza cuando contesta con un 4xx."""


class ZenFake(FakeLLM):
    """Un Zen de juguete: responde lo que el prompt pide, o se comporta mal donde se le diga.

    Cada petición que le llega se anota en `requests`: es lo que permite comprobar que el sondeo
    no hizo ninguna llamada sin dejar su fila.
    """

    def __init__(
        self,
        refuse: set[str] | None = None,
        garbage: set[tuple[str, StructuredOutputMode]] | None = None,
        flaky: set[str] | None = None,
        rate_limited: set[str] | None = None,
        requires_session: bool = False,
    ) -> None:
        super().__init__()
        self.refuse = refuse or set()
        self.garbage = garbage or set()
        self.flaky = flaky or set()
        self.rate_limited = rate_limited or set()
        self.requires_session = requires_session
        self.requests: list[tuple[str, str]] = []
        self._first_prompts: list[tuple[str, str]] = []

    async def complete(  # type: ignore[override]
        self, choice: ModelChoice, prompt: str, schema: type
    ) -> Completion:
        self.requests.append((choice.model, schema.__name__))
        if self.requires_session:
            raise ProviderRejectedError("Error code: 400 - MissingSessionID")
        if choice.model in self.refuse:
            raise ProviderRejectedError(f"Error code: 404 - model {choice.model} not found")
        if choice.model in self.rate_limited:
            raise ProviderRejectedError("Error code: 429 - rate limit exceeded")
        if (choice.model, choice.structured_output) in self.garbage:
            return Completion(text="esto no es json", usage=USAGE)
        if (
            choice.model in self.flaky
            and schema.__name__ != "Ping"
            and not self._retry(choice, prompt)
        ):
            return Completion(text='{"dimension": "structure"}', usage=USAGE)
        if schema.__name__ == "Ping":
            return Completion(text='{"ok": true}', usage=USAGE)
        text = await super().complete(choice, prompt, schema)
        return Completion(text=text, usage=USAGE)

    def _retry(self, choice: ModelChoice, prompt: str) -> bool:
        """Si este prompt reintenta uno ya visto: el router lo escribe como prefijo + error.

        El primer intento de cada prompt falla; el reintento, que lo lleva de prefijo, valida.
        """
        seen = any(
            model == choice.model and prompt.startswith(old) for model, old in self._first_prompts
        )
        if not seen:
            self._first_prompts.append((choice.model, prompt))
        return seen


class Catalog:
    def __init__(self, listed: set[str]) -> None:
        self._listed = frozenset(listed)

    async def available_models(self) -> frozenset[str]:
        return self._listed


def every_id() -> set[str]:
    return {model for model, _ in PRESENT} | {c.model for c in CANDIDATES}


def probe(
    tmp_path: Path,
    backend: ZenFake | None = None,
    headerless: ZenFake | None = None,
    listed: set[str] | None = None,
    settings: Settings | None = None,
) -> tuple[Path, ProbeFindings, ZenFake]:
    """Corre el sondeo entero sobre un manifiesto sintético y devuelve su directorio."""
    chosen = settings or payg_settings()
    zen = backend or ZenFake()
    manifest, data = two_symbol_manifest()
    plan = plan_from_manifest(manifest, data)
    directory = tmp_path / "sondeo"
    open_run_directory(
        directory,
        build_meta(chosen, tmp_path / "manifiesto.json", ("--machine", "desktop"), NOW, None, None),
    )
    backends: dict[Backend, ChatBackend] = {Backend.OPENAI: zen}
    findings = asyncio.run(
        run_probe(
            chosen,
            plan,
            directory,
            "desktop",
            backends,
            headerless or ZenFake(),
            Catalog(every_id() if listed is None else listed),
            lambda: NOW,
            COUNT,
            PRESET,
        )
    )
    return directory, findings, zen


@pytest.fixture(autouse=True)
def manifest_on_disk(tmp_path: Path) -> None:
    """`build_meta` lleva el sha-256 del manifiesto: tiene que existir en disco."""
    (tmp_path / "manifiesto.json").write_text("{}", encoding="utf-8")


# ───────────────────────────────────────────── Contrato ───────────────────────────────────────────


def test_twelve_verdicts_is_what_the_probe_asks_by_default() -> None:
    assert VERDICTS == 12


def test_the_probe_writes_kind_probe_in_its_meta(tmp_path: Path) -> None:
    directory, _, _ = probe(tmp_path)
    assert read_run(directory).meta.kind is RunKind.PROBE


def test_the_probe_directory_is_a_run_that_the_audit_can_read(tmp_path: Path) -> None:
    directory, _, _ = probe(tmp_path)
    run = read_run(directory)

    assert run.meta.billing is Billing.PAYG
    assert run.meta.base_url == "https://opencode.ai/zen/v1"
    by_arm = {arm.arm: arm for arm in run.arms}
    for candidate in CANDIDATES:
        for dimension in (Dimension.STRUCTURE, Dimension.VOLUME):
            arm = by_arm[arm_name(candidate.model, dimension.value)]
            assert arm.sha256 is not None
            assert len(arm.records) == COUNT
            assert all(not record.errors for record in arm.records)
    momentum = by_arm[arm_name("deepseek-v4-flash", "momentum")]
    assert len(momentum.records) == COUNT


def test_every_request_the_provider_saw_has_its_row_in_the_journal(tmp_path: Path) -> None:
    """Regla 4: ninguna llamada del sondeo existe solo en la factura."""
    directory, _, zen = probe(tmp_path)

    rows = chain_calls(directory)

    # El Ping sin cabecera va a otro objeto (`headerless`) y deja su fila igualmente: de ahí el + 1.
    assert len(rows) == len(zen.requests) + 1
    assert all(not call.cache_hit for call in rows)


def test_the_probe_goes_through_no_cache_and_no_fallback(tmp_path: Path) -> None:
    """Dos veces el mismo prompt son dos peticiones, y todo es el modelo pedido."""
    directory, _, zen = probe(tmp_path)

    rows = chain_calls(directory)
    assert {call.model for call in rows} <= every_id()
    assert {call.backend for call in rows} == {Backend.OPENAI}
    assert {model for model, _ in zen.requests} <= every_id()
    verdicts = [call for call in rows if call.prompt_digest]
    digests = [call.prompt_digest for call in verdicts if call.role is AgentRole.STRUCTURE]
    assert len(digests) > len(set(digests)), "el mismo prompt a varios modelos son varias llamadas"


def test_the_chain_measures_the_two_desks_and_the_decider_with_their_own_prompts(
    tmp_path: Path,
) -> None:
    directory, findings, _ = probe(tmp_path)
    by_arm = {arm.arm: arm for arm in read_run(directory).arms}

    for model, role in (
        ("kimi-k2.6", AgentRole.BULL),
        ("minimax-m3", AgentRole.BEAR),
        ("glm-5.2", AgentRole.DECIDER),
    ):
        arm = by_arm[arm_name(model, role.value)]
        assert len(arm.records) == COUNT, f"{model} no corrió el encadenado entero"
        assert {call.role for record in arm.records for call in record.calls} == {role}
    assert not any("encadenado" in reason for reason in findings.skipped)


# ───────────────────────────────────── Catálogo, modos y rechazos ─────────────────────────────────


def test_an_id_missing_from_the_catalog_is_probed_all_the_same(tmp_path: Path) -> None:
    """La ausencia en `/models` no prueba que el chat no lo sirva: el Ping lo decide."""
    absent = {"glm-5.2", "kimi-k2.6", "qwen3.8-max"}
    directory, findings, _ = probe(tmp_path, listed=every_id() - absent)
    by_model = {item.model: item for item in findings.models}

    for model in absent:
        assert by_model[model].listed is False
        assert by_model[model].mode is not None, f"{model} contestó y se dio por ausente"
    assert by_model["minimax-m3"].listed is True
    assert read_run(directory).arms  # el directorio sigue siendo legible


def test_an_unreadable_catalog_is_reported_and_does_not_stop_the_probe(tmp_path: Path) -> None:
    class Broken:
        async def available_models(self) -> frozenset[str]:
            raise ConnectionError("sin ruta al host")

    settings = payg_settings()
    manifest, data = two_symbol_manifest()
    directory = tmp_path / "sondeo"
    open_run_directory(
        directory, build_meta(settings, tmp_path / "manifiesto.json", (), NOW, None, None)
    )
    findings = asyncio.run(
        run_probe(
            settings,
            plan_from_manifest(manifest, data),
            directory,
            "desktop",
            {Backend.OPENAI: ZenFake()},
            ZenFake(),
            Broken(),
            lambda: NOW,
            COUNT,
            PRESET,
        )
    )

    assert findings.catalog_size is None
    assert findings.catalog_error is not None
    assert "ConnectionError" in findings.catalog_error
    assert all(item.listed is None for item in findings.models)
    assert all(item.mode is not None for item in findings.models)


def test_a_provider_rejection_is_not_reported_as_a_mode_that_does_not_work(
    tmp_path: Path,
) -> None:
    directory, findings, _ = probe(tmp_path, ZenFake(refuse={"minimax-m2.7"}))
    refused = next(item for item in findings.models if item.model == "minimax-m2.7")

    assert refused.mode is None
    assert "rechazado por el proveedor" in refused.note
    assert any("minimax-m2.7" in reason for reason in findings.skipped)
    by_arm = {arm.arm: arm for arm in read_run(directory).arms}
    assert by_arm[arm_name("minimax-m2.7", "structure")].sha256 is None, "no se sondeó"
    assert by_arm[arm_name("deepseek-v4-pro", "structure")].sha256 is not None, "los demás sí"


def test_a_candidate_whose_declared_mode_returns_garbage_gets_the_mode_that_works(
    tmp_path: Path,
) -> None:
    broken = {("qwen3.8-max", StructuredOutputMode.JSON_SCHEMA)}
    directory, findings, _ = probe(tmp_path, ZenFake(garbage=broken))
    qwen = next(item for item in findings.models if item.model == "qwen3.8-max")

    assert qwen.mode is not None
    assert qwen.mode is not StructuredOutputMode.JSON_SCHEMA
    arm = next(a for a in read_run(directory).arms if a.arm == "qwen3.8-max@structure")
    modes = {call.structured_output for record in arm.records for call in record.calls}
    assert modes == {qwen.mode}, "los veredictos se piden en el modo que el Ping confirmó"


def test_the_present_models_start_from_the_mode_the_env_declares(tmp_path: Path) -> None:
    _, findings, _ = probe(tmp_path)
    declared = {s.model: s.choice.structured_output for s in subjects_of(payg_settings())}

    for item in findings.models:
        assert item.declared is declared[item.model]
    assert next(i for i in findings.models if i.model == "glm-5.2").declared is (
        StructuredOutputMode.FUNCTION_CALLING
    )


def test_a_rate_limit_is_recorded_as_transport_not_as_the_models_fault(tmp_path: Path) -> None:
    directory, _, _ = probe(tmp_path, ZenFake(rate_limited={"glm-5.3-flash"}))

    rejected = [
        call
        for call in chain_calls(directory)
        if call.model == "glm-5.3-flash" and call.failure_kind is not None
    ]
    assert rejected
    assert {call.failure_kind for call in rejected} == {FailureKind.TRANSPORT}
    assert all("429" in (call.failure_message or "") for call in rejected)


def test_a_retry_is_counted_per_verdict_and_billed(tmp_path: Path) -> None:
    """El primer intento falla de esquema, el segundo valida: 2 intentos, 1 veredicto."""
    directory, _, _ = probe(tmp_path, ZenFake(flaky={"deepseek-v4-pro"}))
    arm = next(a for a in read_run(directory).arms if a.arm == "deepseek-v4-pro@structure")

    assert len(arm.records) == COUNT
    assert all(len(record.calls) == 2 for record in arm.records)
    assert {call.failure_kind for record in arm.records for call in record.calls} == {
        None,
        FailureKind.SCHEMA,
    }
    assert all(not record.errors for record in arm.records), "el reintento los rescató"


# ───────────────────────────────────────────── Cabecera ───────────────────────────────────────────


def test_the_session_header_is_tested_with_and_without_it(tmp_path: Path) -> None:
    _, findings, _ = probe(tmp_path, headerless=ZenFake(requires_session=True))

    assert findings.header is not None
    assert findings.header.with_header == "responde"
    assert findings.header.without_header.startswith("rechazado")
    assert "MissingSessionID" in findings.header.without_header


def test_the_header_is_tested_even_when_no_model_answers(tmp_path: Path) -> None:
    """Sin fondos o sin acceso, Zen rechaza antes de generar; aun así la cabecera se distingue.

    Go contestaba `400 MissingSessionID` antes de cualquier otra cosa: basta con que el rechazo con
    cabecera y el rechazo sin ella no sean el mismo para saber si falta.
    """
    directory, findings, _ = probe(
        tmp_path, ZenFake(refuse=every_id()), headerless=ZenFake(requires_session=True)
    )

    assert findings.header is not None
    assert findings.header.model == zen_probe.HEADER_MODEL
    assert findings.header.mode_confirmed is False
    assert "404" in findings.header.with_header
    assert "MissingSessionID" in findings.header.without_header
    assert findings.header.with_header != findings.header.without_header
    report = render_report(read_run(directory), findings)
    assert "solo declarado, sin confirmar" in report


def test_a_rejection_note_carries_what_the_provider_said(tmp_path: Path) -> None:
    """«Rechazado» a secas manda a mirar el modelo; el cuerpo del 402 o el 403 manda a la cuenta."""
    _, findings, _ = probe(tmp_path, ZenFake(refuse={"minimax-m2.7"}))
    refused = next(item for item in findings.models if item.model == "minimax-m2.7")

    assert "rechazado por el proveedor antes de producir contenido" in refused.note
    assert "404" in refused.note
    assert "minimax-m2.7 not found" in refused.note


def test_a_gateway_that_does_not_need_the_header_says_so(tmp_path: Path) -> None:
    _, findings, _ = probe(tmp_path)

    assert findings.header is not None
    assert findings.header.without_header == "responde"


def test_the_headerless_backend_really_sends_no_session_header() -> None:
    backend = _NoSessionBackend("clave-de-prueba", "https://opencode.ai/zen/v1")
    assert backend._headers() == {}
    assert backend.session_id  # sigue habiendo id de sesión: lo que falta es mandarlo


# ───────────────────────────────────────────── Informe ────────────────────────────────────────────


def test_the_report_is_read_from_the_directory_and_names_the_machine(tmp_path: Path) -> None:
    directory, findings, _ = probe(tmp_path)
    report = render_report(read_run(directory), findings)

    assert "latencia media (desktop)" in report
    assert "máquina donde se midió la latencia: **desktop**" in report
    assert f"python -m crypto_agents.consumption {directory}" in report
    assert "`deepseek-v4.1-flash@structure`" in report
    assert f"{COUNT}/{COUNT}" in report
    for name in ("schema", "context", "timeout", "transport"):
        assert name in report, "los cuatro failure_kind van siempre, ceros incluidos"


def test_the_cost_of_a_probe_is_measured_from_its_rows_and_added_by_the_consumption_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`glm-5.2` a 1000 de entrada y 200 de salida: (1000 x 1.40 + 200 x 4.40) / 1e6 por llamada."""
    directory, _, _ = probe(tmp_path)
    per_call = (1000 * 1.40 + 200 * 4.40) / 1_000_000
    decider = [call for call in chain_calls(directory) if call.role is AgentRole.DECIDER]
    assert len(decider) == COUNT + 1  # el Ping de modo y los tres del encadenado
    # `glm-5.2` también contesta el Ping de modo
    measured = consume(decider, DEFAULT_PRICING, Billing.PAYG)
    assert measured.measured == len(decider)
    assert measured.cost_usd == pytest.approx(len(decider) * per_call)

    assert consumption_main([str(directory)]) == 0
    out = capsys.readouterr().out
    assert "glm-5.2@decider" in out
    assert "pago por uso" in out


def test_every_arm_the_probe_may_write_is_declared_in_the_meta_before_the_first_call(
    tmp_path: Path,
) -> None:
    directory, _, _ = probe(tmp_path)
    declared = set(json.loads((directory / "meta.json").read_text("utf-8"))["arms"])

    assert set(all_arms(payg_settings())) == declared
    written = {path.stem for path in directory.glob("*.jsonl")}
    assert written <= declared


def test_a_probe_directory_is_never_reused(tmp_path: Path) -> None:
    directory, _, _ = probe(tmp_path)
    with pytest.raises(ConfigError, match="ya contiene algo"):
        open_run_directory(
            directory,
            build_meta(payg_settings(), tmp_path / "manifiesto.json", (), NOW, None, None),
        )


# ───────────────────────────────────────── Se niega a gastar de Go ────────────────────────────────


def test_it_refuses_to_run_against_the_go_gateway() -> None:
    go = payg_settings().model_copy(
        update={
            "openai": payg_settings().openai.model_copy(  # type: ignore[union-attr]
                update={"base_url": "https://opencode.ai/zen/go/v1"}
            )
        }
    )
    with pytest.raises(ConfigError, match="OpenCode Go"):
        refuse_unless_zen(go)


def test_it_refuses_to_run_without_pay_as_you_go() -> None:
    subscription = payg_settings().model_copy(update={"billing": Billing.GO})
    with pytest.raises(ConfigError, match="payg"):
        refuse_unless_zen(subscription)


def test_it_accepts_zen_with_pay_as_you_go() -> None:
    refuse_unless_zen(payg_settings())


def test_the_refusal_never_quotes_a_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    leaky = payg_settings().model_copy(
        update={
            "openai": payg_settings().openai.model_copy(  # type: ignore[union-attr]
                update={"base_url": "https://user-xyz:hunter2@opencode.ai/zen/go/v1?k=tokenvalue"}
            )
        }
    )
    with pytest.raises(ConfigError) as caught:
        refuse_unless_zen(leaky)
    for secret in ("user-xyz", "hunter2", "tokenvalue"):
        assert secret not in str(caught.value)


def test_the_command_refuses_go_before_building_any_backend(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    go = payg_settings().model_copy(update={"billing": Billing.GO})
    monkeypatch.setattr(zen_probe, "load_settings", lambda _path: go)

    def forbidden(_settings: object) -> object:
        raise AssertionError("se construyó un backend antes de negarse")

    monkeypatch.setattr(zen_probe, "build_backends", forbidden)

    assert main(["--machine", "desktop"]) == 1
    assert "payg" in capsys.readouterr().err


def test_the_dry_run_counts_without_building_a_backend(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(zen_probe, "load_settings", lambda _path: payg_settings())

    def forbidden(_settings: object) -> object:
        raise AssertionError("el conteo previo no puede tener un proveedor a mano")

    monkeypatch.setattr(zen_probe, "build_backends", forbidden)

    assert main(["--machine", "desktop", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "conteo previo" in out
    assert (
        "veredictos técnicos: 156" in out
    )  # 6 candidatos x 2 dimensiones x 12, más 12 de momentum
    assert "no determinado" in out, "sin tokens medidos no hay dólares que dar"


def test_the_machine_is_mandatory(capsys: pytest.CaptureFixture[str]) -> None:
    """Una latencia sin máquina es inservible: no hay forma de saber a cuál de las dos describe."""
    with pytest.raises(SystemExit) as caught:
        main(["--dry-run"])
    assert caught.value.code == 2


# ──────────────────────────────────────── Sondeo de mesas ─────────────────────────────────────────


class DeskFake(ZenFake):
    """Un Zen que además anota los prompts, distingue a los bull y falla donde se le diga.

    `errors` hace que un modelo conteste con ese texto a toda petición (un 410, un 503);
    `bad_briefs` hace que conteste basura solo al esquema del alegato.
    """

    def __init__(
        self,
        errors: dict[str, str] | None = None,
        bad_briefs: set[str] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.errors = errors or {}
        self.bad_briefs = bad_briefs or set()
        self.seen: list[tuple[str, str, str]] = []

    async def complete(  # type: ignore[override]
        self, choice: ModelChoice, prompt: str, schema: type
    ) -> Completion:
        self.seen.append((choice.model, schema.__name__, prompt))
        if choice.model in self.errors:
            raise ProviderRejectedError(self.errors[choice.model])
        if schema.__name__ == "DebateBrief" and choice.model in self.bad_briefs:
            return Completion(text="esto no es un alegato", usage=USAGE)
        done = await super().complete(choice, prompt, schema)
        if schema.__name__ == "DebateBrief":
            # Cada bull escribe una tesis distinta: el prompt del decisor lleva el alegato.
            return Completion(
                text=done.text.replace(
                    "La estructura sigue", f"La estructura de {choice.model} sigue"
                ),
                usage=done.usage,
            )
        return done


GONE = (
    "Error code: 410 - {'error': {'type': 'server_error', "
    "'message': 'Upstream request failed: Endpoint is unavailable.'}}"
)
DOWN = "Error code: 503 - {'error': {'type': 'server_error', 'message': 'Service Unavailable'}}"


def desks(
    tmp_path: Path,
    backend: DeskFake | None = None,
    cap: float = 1_000.0,
    technical: Path | None = None,
) -> tuple[Path, ProbeFindings, DeskFake, SpendGuard]:
    """El sondeo técnico de siempre y, encima, el de las mesas con su tope."""
    chosen = payg_settings()
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "manifiesto.json").write_text("{}", encoding="utf-8")
    previous = technical if technical is not None else probe(tmp_path)[0]
    source = technical_source(previous, NOW)
    subjects = desk_subjects(chosen, source)
    manifest, data = two_symbol_manifest()
    plan = plan_from_manifest(manifest, data)
    directory = tmp_path / "mesas"
    open_run_directory(
        directory,
        build_meta(
            chosen,
            tmp_path / "manifiesto.json",
            (),
            NOW,
            None,
            None,
            arms=all_desk_arms(subjects),
        ),
    )
    zen = backend or DeskFake()
    guard = SpendGuard(cap)
    backends: dict[Backend, ChatBackend] = {Backend.OPENAI: zen}
    findings = asyncio.run(
        run_desks_probe(
            chosen, plan, directory, "desktop", backends, source, guard, lambda: NOW, COUNT, PRESET
        )
    )
    return directory, findings, zen, guard


def bull_arms(directory: Path) -> dict[str, list[str]]:
    """Por candidato a bull, el digest del prompt de cada activación, en orden."""
    run = read_run(directory)
    by_arm = {arm.arm: arm for arm in run.arms}
    return {
        c.model: [
            record.calls[0].prompt_digest for record in by_arm[arm_name(c.model, "bull")].records
        ]
        for c in BULL_CANDIDATES
    }


def test_the_technical_source_applies_the_current_rule_to_a_previous_probe(
    tmp_path: Path,
) -> None:
    """Todos con los mismos válidos: gana el más barato por token de salida, `glm-5.3-flash`."""
    source = technical_source(probe(tmp_path)[0], NOW)
    assert source.producers == {
        Dimension.STRUCTURE: "glm-5.3-flash",
        Dimension.VOLUME: "glm-5.3-flash",
    }
    assert source.sources, "cada productor cita los journals de los que se contó"
    assert all(len(digest) == 64 for _, digest in source.sources)


def test_the_technical_source_prefers_the_candidate_with_more_valid_verdicts(
    tmp_path: Path,
) -> None:
    """Sin ningún válido para el más barato, la regla pasa al siguiente en precio."""
    broken = ZenFake(garbage={("glm-5.3-flash", mode) for mode in StructuredOutputMode})
    source = technical_source(probe(tmp_path, broken)[0], NOW)
    assert source.producers[Dimension.STRUCTURE] == "deepseek-v4.1-flash"


def test_every_bull_candidate_gets_the_same_prompt_and_only_the_model_changes(
    tmp_path: Path,
) -> None:
    directory, _, _, _ = desks(tmp_path)
    digests = bull_arms(directory)

    assert all(len(found) == COUNT for found in digests.values())
    first = digests[BULL_CANDIDATES[0].model]
    for model, found in digests.items():
        assert found == first, f"{model} recibió otro prompt que {BULL_CANDIDATES[0].model}"
    assert len(set(first)) == COUNT, "y las activaciones sí se distinguen entre sí"


def test_every_desk_of_an_activation_reads_the_very_same_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[tuple[Side, tuple[object, ...]]] = []
    original = debate_prompt

    def spy(side: Side, snapshot: object, indicators: object, evidence: tuple[object, ...]) -> str:
        captured.append((side, evidence))
        return original(side, snapshot, indicators, evidence)  # type: ignore[arg-type]

    previous = probe(tmp_path)[0]  # su encadenado también arma prompts de mesa: no se espía
    monkeypatch.setattr("crypto_agents.zen_probe.debate_prompt", spy)
    desks(tmp_path, technical=previous)

    sides = [side for side, _ in captured]
    assert sides.count(Side.BULL) == len(BULL_CANDIDATES) * COUNT
    assert sides.count(Side.BEAR) == COUNT
    assert len({id(evidence) for _, evidence in captured}) == COUNT, (
        "una sola evidencia por activación, compartida por las cinco mesas"
    )


def test_the_technicals_run_once_per_activation_and_not_once_per_candidate(
    tmp_path: Path,
) -> None:
    directory, _, zen, _ = desks(tmp_path)
    run = read_run(directory)

    verdict_requests = [model for model, schema, _ in zen.seen if schema == "TechnicalVerdict"]
    assert len(verdict_requests) == len(Dimension) * COUNT
    arms = {arm.arm for arm in run.arms if arm.records}
    assert {"glm-5.3-flash@structure", "glm-5.3-flash@volume", "deepseek-v4-flash@momentum"} <= arms


def test_the_decider_is_measured_once_per_activation_and_per_bull_with_a_valid_brief(
    tmp_path: Path,
) -> None:
    broken = DeskFake(bad_briefs={"qwen3.8-max"})
    directory, _, zen, _ = desks(tmp_path, broken)
    by_arm = {arm.arm: arm for arm in read_run(directory).arms}

    for bull in ("kimi-k2.6", "kimi-k3", "deepseek-v4-pro"):
        arm = by_arm[arm_name("glm-5.2", decider_label(bull))]
        assert len(arm.records) == COUNT, bull
        assert all(not record.errors for record in arm.records)
    assert by_arm[arm_name("glm-5.2", decider_label("qwen3.8-max"))].records == ()
    decider_requests = [m for m, schema, _ in zen.seen if schema == "Decision"]
    assert len(decider_requests) == 3 * COUNT


def test_each_decider_prompt_carries_the_brief_of_its_own_bull(tmp_path: Path) -> None:
    """Condicionado de verdad: el mismo decisor, la misma activación, otro alegato, otro digest."""
    directory, _, _, _ = desks(tmp_path)
    by_arm = {arm.arm: arm for arm in read_run(directory).arms}
    digests = {
        bull.model: [
            r.calls[0].prompt_digest
            for r in by_arm[arm_name("glm-5.2", decider_label(bull.model))].records
        ]
        for bull in BULL_CANDIDATES
    }
    for position in range(COUNT):
        assert len({found[position] for found in digests.values()}) == len(BULL_CANDIDATES)


def test_a_bull_that_cannot_be_served_does_not_take_the_bear_with_it(tmp_path: Path) -> None:
    gone = {c.model: GONE for c in BULL_CANDIDATES}
    directory, findings, _, _ = desks(tmp_path, DeskFake(errors=gone))
    by_arm = {arm.arm: arm for arm in read_run(directory).arms}

    bear = by_arm[arm_name("minimax-m3", "bear")]
    assert len(bear.records) == COUNT
    assert all(not record.errors for record in bear.records)
    assert all(by_arm[arm_name(c.model, "bull")].records == () for c in BULL_CANDIDATES)
    assert any("sin modo que responda" in reason for reason in findings.skipped)


def test_a_410_and_a_503_are_reported_apart_each_with_its_code_and_body(
    tmp_path: Path,
) -> None:
    directory, findings, _, _ = desks(
        tmp_path, DeskFake(errors={"kimi-k2.6": GONE, "kimi-k3": DOWN})
    )
    subjects = desk_subjects(payg_settings(), technical_source(tmp_path / "sondeo", NOW))
    text = render_desks_report(read_run(directory), findings, subjects)

    section = text.split("## Rechazos del proveedor")[1].split("## ")[0]
    rows = [line for line in section.splitlines() if line.startswith("| `")]
    assert len(rows) == 2
    gone = next(row for row in rows if "kimi-k2.6" in row)
    down = next(row for row in rows if "kimi-k3" in row)
    assert "| 410 |" in gone and "Endpoint is unavailable" in gone
    assert "| 503 |" in down and "Service Unavailable" in down


def test_the_report_never_prints_a_credential_that_came_back_in_an_error_body(
    tmp_path: Path,
) -> None:
    leak = (
        "Error code: 401 - {'error': 'bad key', 'detail': 'Authorization: Bearer sk-supersecret123 "
        "api_key=sk-otherkey4567 via https://user:hunter2@host/v1'}"
    )
    directory, findings, _, _ = desks(tmp_path, DeskFake(errors={"qwen3.8-max": leak}))
    subjects = desk_subjects(payg_settings(), technical_source(tmp_path / "sondeo", NOW))
    text = render_desks_report(read_run(directory), findings, subjects)

    for secret in ("supersecret123", "otherkey4567", "hunter2"):
        assert secret not in text
    assert "| 401 |" in text, "el código se queda: es lo que distingue el rechazo"
    assert "[redactado]" in text


def test_redact_removes_bearer_keys_and_url_credentials() -> None:
    cleaned = redact("Bearer abc.def sk-0123456789 https://u:p4ss@h/x api_key: zzz")
    for secret in ("abc.def", "sk-0123456789", "p4ss", "zzz"):
        assert secret not in cleaned


def test_the_report_has_one_row_per_model_and_role_with_the_four_failure_kinds(
    tmp_path: Path,
) -> None:
    directory, findings, _, _ = desks(tmp_path)
    subjects = desk_subjects(payg_settings(), technical_source(tmp_path / "sondeo", NOW))
    text = render_desks_report(read_run(directory), findings, subjects)

    header = next(line for line in text.splitlines() if line.startswith("| brazo | modo"))
    for column in ("schema", "context", "timeout", "transport", "latencia media (desktop)"):
        assert column in header
    for bull in BULL_CANDIDATES:
        assert f"| `{arm_name(bull.model, 'bull')}` |" in text
        assert f"| `{arm_name('glm-5.2', decider_label(bull.model))}` |" in text
    assert "| `minimax-m3@bear` |" in text
    assert "la mesa contestando como la otra" in text


def test_every_dollar_cites_the_rows_and_the_sha256_of_the_journal_it_comes_from(
    tmp_path: Path,
) -> None:
    directory, findings, _, _ = desks(tmp_path)
    run = read_run(directory)
    subjects = desk_subjects(payg_settings(), technical_source(tmp_path / "sondeo", NOW))
    text = render_desks_report(run, findings, subjects)

    sources = text.split("## Fuentes")[1].split("## ")[0]
    for arm in run.arms:
        if arm.sha256 is None:
            continue
        rows = sum(len(record.calls) for record in arm.records)
        assert f"| `{arm.arm}` | `{arm.path.name}` | {rows} | `{arm.sha256}` |" in sources
    assert "python -m crypto_agents.consumption" in text


def test_the_desks_directory_is_a_probe_the_audit_and_the_consumption_read(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory, _, _, _ = desks(tmp_path)

    assert read_run(directory).meta.kind is RunKind.PROBE
    assert set(json.loads((directory / "meta.json").read_text("utf-8"))["arms"]) == set(
        all_desk_arms(desk_subjects(payg_settings(), technical_source(tmp_path / "sondeo", NOW)))
    )
    assert consumption_main([str(directory)]) == 0
    assert "glm-5.2@decider+kimi-k3" in capsys.readouterr().out


def test_every_request_of_the_desks_probe_left_its_row(tmp_path: Path) -> None:
    """Regla 4: lo que el proveedor vio está en el journal, intento por intento."""
    directory, _, zen, _ = desks(tmp_path)
    assert len(chain_calls(directory)) == len(zen.seen)


# ─────────────────────────────────────────── El tope de gasto ─────────────────────────────────────


def test_the_spend_guard_sums_the_dearest_end_of_what_the_provider_declared() -> None:
    """qwen3.8-max con 10 000 de prompt, 2 000 cacheados y 1 000 de salida: de 0.0225 a 0.0265.

    El tope mira el extremo alto: tras una llamada hay 0.0265 y con un tope de 0.03 cabe otra;
    tras dos hay 0.0530 y ya no. Lo que se deja sin hacer queda anotado con su nombre.
    """
    guard = SpendGuard(0.03)
    assert guard.allow("antes de todo")
    guard.add([priced_call("qwen3.8-max", prompt=10_000, cached=2_000, completion=1_000)])
    assert guard.spent_usd == pytest.approx(0.0265)
    assert guard.allow("la segunda")
    guard.add([priced_call("qwen3.8-max", prompt=10_000, cached=2_000, completion=1_000)])
    assert guard.spent_usd == pytest.approx(0.0530)
    assert not guard.allow("la tercera")
    assert guard.refused == ["la tercera"]


def test_a_call_without_tokens_does_not_count_for_the_cap_and_is_said() -> None:
    guard = SpendGuard(1.0)
    guard.add([priced_call("qwen3.8-max", prompt=None)])
    assert guard.spent_usd == 0.0
    assert guard.unpriced == 1


def test_a_cap_that_is_reached_stops_new_invocations_and_says_what_it_left_undone(
    tmp_path: Path,
) -> None:
    free, _, loose, _ = desks(tmp_path / "libre")
    directory, findings, tight, guard = desks(tmp_path / "con_tope", cap=0.0005)

    assert len(tight.seen) < len(loose.seen)
    assert guard.refused, "el tope tenía que impedir al menos una invocación"
    assert findings.refused == tuple(guard.refused)
    assert any("tope" in reason for reason in findings.skipped)
    assert guard.spent_usd >= guard.cap_usd
    assert free != directory


def test_a_non_positive_cap_is_refused() -> None:
    with pytest.raises(ValueError, match="positivo"):
        SpendGuard(0.0)


# ────────────────────────────────────────────── Familias ──────────────────────────────────────────


def test_no_bull_candidate_shares_a_family_with_the_bear_or_the_decider(tmp_path: Path) -> None:
    subjects = desk_subjects(payg_settings(), technical_source(probe(tmp_path)[0], NOW))
    text = "\n".join(render_families(subjects))

    assert "**IGUAL**" not in text
    for family in ("moonshot", "qwen", "deepseek"):
        assert f"| {family} | distinta | distinta |" in text
    assert "comparte familia con `momentum`" in text


def test_a_bull_in_the_bears_family_is_flagged(tmp_path: Path) -> None:
    subjects = desk_subjects(payg_settings(), technical_source(probe(tmp_path)[0], NOW))
    clashing = subjects._replace(bulls=(subjects.bulls[0]._replace(family="minimax"),))

    assert "**IGUAL**" in "\n".join(render_families(clashing))


# ───────────────────────────────────── Comando: --desks, --dry-run, tope ──────────────────────────


def test_the_desks_dry_run_counts_and_declares_the_cap_without_a_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    previous = probe(tmp_path)[0]
    monkeypatch.setattr(zen_probe, "load_settings", lambda _path: payg_settings())

    def forbidden(_settings: object) -> object:
        raise AssertionError("el conteo previo no puede tener un proveedor a mano")

    monkeypatch.setattr(zen_probe, "build_backends", forbidden)

    argv = ["--machine", "desktop", "--desks", "--dry-run", "--technicals-from", str(previous)]
    assert main([*argv, "--max-usd", "2.5"]) == 0
    out = capsys.readouterr().out
    assert "técnicos, una vez por activación: 36" in out  # 3 dimensiones x 12
    assert "bull: 4 candidatos x 12 = 48" in out
    assert "bear `minimax-m3`: 12" in out
    assert "decisor `glm-5.2`: ≤ 48" in out
    assert "2.50 USD" in out
    assert "no determinado antes de llamar" in out
    assert "| `qwen3.8-max` | 2.0 | 0.25 | 6.0 | 2.5 |" in out
    assert "| `kimi-k3` | 3.0 | 0.3 | 15.0 | — |" in out

    assert main(argv) == 0
    assert "no arranca sin él" in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv",
    [
        ["--desks"],
        ["--desks", "--technicals-from", "x"],
        ["--technicals-from", "x"],
        ["--max-usd", "1"],
        ["--desks", "--technicals-from", "x", "--max-usd", "0"],
    ],
    ids=["no_source", "no_cap", "source_without_desks", "cap_without_desks", "zero_cap"],
)
def test_the_desks_flags_refuse_incoherent_combinations(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--machine", "desktop", *argv])
    assert caught.value.code == 2


def test_the_desks_command_refuses_go_before_building_any_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    go = payg_settings().model_copy(update={"billing": Billing.GO})
    monkeypatch.setattr(zen_probe, "load_settings", lambda _path: go)

    def forbidden(_settings: object) -> object:
        raise AssertionError("se construyó un backend antes de negarse")

    monkeypatch.setattr(zen_probe, "build_backends", forbidden)

    code = main(
        ["--machine", "desktop", "--desks", "--technicals-from", str(tmp_path), "--max-usd", "1"]
    )
    assert code == 1
    assert "payg" in capsys.readouterr().err
