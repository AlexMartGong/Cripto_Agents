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
from crypto_agents.audit import chain_calls, read_run
from crypto_agents.consumption import consume
from crypto_agents.consumption import main as consumption_main
from crypto_agents.llm import Completion, TokenUsage
from crypto_agents.settings import DEFAULT_PRICING, ConfigError
from crypto_agents.state import (
    AgentRole,
    Backend,
    Billing,
    Dimension,
    FailureKind,
    StructuredOutputMode,
)
from crypto_agents.zen_probe import (
    CANDIDATES,
    PRESENT,
    VERDICTS,
    ProbeFindings,
    _NoSessionBackend,
    all_arms,
    arm_name,
    build_meta,
    main,
    refuse_unless_zen,
    render_report,
    run_probe,
    subjects_of,
)
from tests.conftest import PRESET, FakeLLM
from tests.test_ablation import two_symbol_manifest
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
