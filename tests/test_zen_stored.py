"""Preguntar otra vez sobre lo que un sondeo de mesas dejó escrito, sin volver a pedir los insumos.

Dos mediciones del bloque T6, las dos sobre el `content.jsonl` de un sondeo ya hecho:

- los decisores sin mesas (`decide_solo`, `decide_without_debate`) con el contrato de T5, que
  solo se había medido con el decisor de `full`;
- la mesa bear con otro modo de salida estructurada que el de `.env`, con el mismo prompt.

Lo que estas pruebas cuidan es que la medición mida lo que dice: que los prompts son los de
producción, que la evidencia es la guardada y no una nueva, y que eso se comprueba por el
`prompt_digest` **antes** de la primera llamada, no después de haber gastado.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

import pytest

from crypto_agents import zen_probe
from crypto_agents.ablation import open_run_directory, plan_from_manifest
from crypto_agents.audit import file_sha256, read_run
from crypto_agents.criteria import main as criteria_main
from crypto_agents.indicators import IndicatorPreset
from crypto_agents.llm import Completion, Upstream, prompt_digest
from crypto_agents.metrics import wilson_interval
from crypto_agents.prompts import debate_prompt, no_debate_prompt, solo_prompt
from crypto_agents.settings import ConfigError
from crypto_agents.state import (
    Action,
    AgentRole,
    Backend,
    FailureKind,
    Proposal,
    Side,
    StructuredOutputMode,
)
from crypto_agents.zen_probe import (
    CONTENT_FILE,
    DECIDER_ARMS,
    SpendGuard,
    activation_run_id,
    arm_name,
    build_meta,
    main,
    prepare_activations,
    read_content,
    render_bear_report,
    render_deciders_report,
    run_bear_probe,
    run_deciders_probe,
    stored_bear_arms,
    stored_decider_arms,
    stored_inputs,
)
from tests.conftest import PRESET, proposal_payload
from tests.test_ablation import two_symbol_manifest
from tests.test_zen_payg import payg_settings
from tests.test_zen_probe import (  # noqa: F401
    COUNT,
    NOW,
    USAGE,
    DeskFake,
    desks,
    manifest_on_disk,
    probe,
    two_momentum_producers,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from crypto_agents.ablation import ReplayPlan
    from crypto_agents.llm import ChatBackend
    from crypto_agents.settings import ModelChoice, Settings
    from crypto_agents.state import LLMCall
    from crypto_agents.zen_probe import ProbeFindings

GLM_UPSTREAM = Upstream(model="accounts/fireworks/models/glm-5p3", endpoint="fireworks")
"""Lo que la pasarela declaró para `glm-5.2` el 2026-10-06 (`var/zen-probe/20261006T061405Z`)."""

OTHER_PRESET = IndicatorPreset(
    rsi=7, ema_fast=3, ema_slow=8, ema_trend=21, atr=5, adx=5, bbands=5, volume_ma=5
)
"""Otros indicadores sobre las mismas velas: otro prompt técnico, y por tanto otro digest."""


def retried(prompt: str) -> bool:
    """Si el router está reintentando: el prompt lleva pegado el error del intento anterior."""
    return "no pasó la validación" in prompt


def without_key(payload: str, key: str) -> str:
    """La misma respuesta sin esa clave: el modelo la omitió."""
    return json.dumps({k: v for k, v in json.loads(payload).items() if k != key})


def with_null(payload: str, key: str) -> str:
    """La misma respuesta con esa clave en `null`: el modelo la escribió, vacía."""
    return json.dumps({**json.loads(payload), key: None})


class StoredFake(DeskFake):
    """Un Zen que contesta mal la primera vez a quien se le diga, y declara su upstream.

    `first` dice qué texto devuelve el primer intento de cada brazo del decisor (`solo` o
    `no_debate`); el reintento valida. `string_grounding` hace que el bear escriba `grounded_in`
    como cadena en el primer intento, que es lo que `minimax-m3` hizo en T4 y T5.
    """

    def __init__(
        self,
        first: dict[str, str] | None = None,
        string_grounding: bool = False,
        upstream: Upstream | None = GLM_UPSTREAM,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.first = first or {}
        self.string_grounding = string_grounding
        self.upstream = upstream

    async def complete(  # type: ignore[override]
        self, choice: ModelChoice, prompt: str, schema: type
    ) -> Completion:
        done = await super().complete(choice, prompt, schema)
        text = done.text
        if schema is Proposal and not retried(prompt):
            # El prompt sin mesas lleva los veredictos, con sus ids; el del generalista, no.
            arm = "no_debate" if "structure-1" in prompt else "solo"
            text = self.first.get(arm, text)
        if schema.__name__ == "DebateBrief" and self.string_grounding and not retried(prompt):
            brief = json.loads(text)
            for claim in brief["claims"]:
                claim["grounded_in"] = claim["grounded_in"][0]
            text = json.dumps(brief)
        return Completion(text=text, usage=done.usage, upstream=self.upstream)


def plan_and_manifest(tmp_path: Path) -> tuple[ReplayPlan, Path]:
    manifest, data = two_symbol_manifest()
    return plan_from_manifest(manifest, data), tmp_path / "manifiesto.json"


def deciders(
    tmp_path: Path,
    origin: Path,
    backend: DeskFake | None = None,
    cap: float = 1_000.0,
    settings: Settings | None = None,
    preset: IndicatorPreset = PRESET,
    announce: Callable[[str], None] | None = None,
) -> tuple[Path, ProbeFindings, DeskFake, SpendGuard]:
    """Los dos decisores sin mesas sobre lo que dejó `origin`."""
    chosen = settings or payg_settings()
    plan, manifest = plan_and_manifest(tmp_path)
    directory = tmp_path / "deciders"
    stored = stored_inputs(origin, manifest)
    open_run_directory(
        directory,
        build_meta(chosen, manifest, (), NOW, None, None, arms=stored_decider_arms(chosen)),
    )
    zen = backend or StoredFake()
    guard = SpendGuard(cap)
    backends: dict[Backend, ChatBackend] = {Backend.OPENAI: zen}
    findings = asyncio.run(
        run_deciders_probe(
            chosen,
            plan,
            directory,
            "desktop",
            backends,
            stored,
            guard,
            lambda: NOW,
            COUNT,
            preset,
            announce=announce,
        )
    )
    return directory, findings, zen, guard


def bear(
    tmp_path: Path,
    origin: Path,
    mode: StructuredOutputMode = StructuredOutputMode.JSON_SCHEMA,
    backend: DeskFake | None = None,
    cap: float = 1_000.0,
    settings: Settings | None = None,
    preset: IndicatorPreset = PRESET,
    announce: Callable[[str], None] | None = None,
) -> tuple[Path, ProbeFindings, DeskFake, SpendGuard]:
    """La mesa bear en `mode` sobre la evidencia que dejó `origin`."""
    chosen = settings or payg_settings()
    plan, manifest = plan_and_manifest(tmp_path)
    directory = tmp_path / "bear"
    stored = stored_inputs(origin, manifest)
    open_run_directory(
        directory,
        build_meta(chosen, manifest, (), NOW, None, None, arms=stored_bear_arms(chosen)),
    )
    zen = backend or StoredFake()
    guard = SpendGuard(cap)
    backends: dict[Backend, ChatBackend] = {Backend.OPENAI: zen}
    findings = asyncio.run(
        run_bear_probe(
            chosen,
            plan,
            directory,
            "desktop",
            backends,
            stored,
            mode,
            guard,
            lambda: NOW,
            COUNT,
            preset,
            announce=announce,
        )
    )
    return directory, findings, zen, guard


def calls_of(directory: Path, arm: str) -> list[list[LLMCall]]:
    """Por invocación de ese brazo, sus intentos."""
    (found,) = (a for a in read_run(directory).arms if a.arm == arm)
    return [list(record.calls) for record in found.records]


def asked(zen: DeskFake, schema: str) -> int:
    return sum(1 for _, name, _ in zen.seen if name == schema)


# ───────────────────────────────────── Lo que se lee del origen ───────────────────────────────────


def test_the_stored_inputs_are_the_content_of_a_desks_probe(tmp_path: Path) -> None:
    origin = desks(tmp_path)[0]

    stored = stored_inputs(origin, tmp_path / "manifiesto.json")

    assert len(stored.content) == COUNT
    assert stored.content_sha256 == file_sha256(origin / CONTENT_FILE)
    assert [item.run_id for item in read_content(origin)] == list(stored.content)


def test_a_directory_without_content_is_refused(tmp_path: Path) -> None:
    """El sondeo técnico no guarda lo que se contestó: no hay evidencia que leer."""
    technical = probe(tmp_path)[0]

    with pytest.raises(ConfigError, match=CONTENT_FILE):
        stored_inputs(technical, tmp_path / "manifiesto.json")


def test_a_source_that_ran_another_plan_is_refused(tmp_path: Path) -> None:
    """Otras activaciones no son las mismas preguntas: se dice antes de preparar nada."""
    origin = desks(tmp_path)[0]
    other = tmp_path / "otro-manifiesto.json"
    other.write_text('{"otro": true}', encoding="utf-8")

    with pytest.raises(ConfigError, match="otro plan"):
        stored_inputs(origin, other)


def test_an_activation_missing_from_the_content_is_refused_without_calling(
    tmp_path: Path,
) -> None:
    origin = desks(tmp_path)[0]
    content = origin / CONTENT_FILE
    content.write_text(
        "\n".join(content.read_text("utf-8").splitlines()[1:]) + "\n", encoding="utf-8"
    )
    zen = StoredFake()

    with pytest.raises(ConfigError, match=f"1 de las {COUNT} activaciones"):
        deciders(tmp_path, origin, zen)

    assert zen.seen == []


# ────────────────────────────────────── Decisores sin mesas ───────────────────────────────────────


def test_the_two_deciders_ask_the_production_prompts_over_the_stored_evidence(
    tmp_path: Path,
) -> None:
    """`solo` no lee evidencia; `no_debate` lee la guardada. Ninguno vuelve a pedir un veredicto."""
    origin = desks(tmp_path)[0]
    stored = {item.run_id: item for item in read_content(origin)}
    plan, _ = plan_and_manifest(tmp_path)
    activations = asyncio.run(
        prepare_activations(plan, payg_settings(), lambda: NOW, COUNT, PRESET)
    )

    directory, _, zen, _ = deciders(tmp_path, origin)

    assert (asked(zen, "TechnicalVerdict"), asked(zen, "DebateBrief")) == (0, 0)
    assert asked(zen, "Proposal") == 2 * COUNT
    assert asked(zen, "Decision") == 0
    by_arm = {arm.arm: {r.run_id: r for r in arm.records} for arm in read_run(directory).arms}
    for activation in activations:
        run_id = activation_run_id(activation.entry)
        where = activation.prepared
        solo = solo_prompt(where.snapshot, where.indicators, where.activation.triggers)
        bare = no_debate_prompt(where.snapshot, where.indicators, stored[run_id].evidence)
        assert by_arm["glm-5.2@solo"][run_id].calls[0].prompt_digest == prompt_digest(solo)
        assert by_arm["glm-5.2@no_debate"][run_id].calls[0].prompt_digest == prompt_digest(bare)


def test_the_arms_are_named_after_the_ablation_arms_and_not_after_a_bull(tmp_path: Path) -> None:
    """`decider+<algo>` lo lee `estimate` como el decisor condicionado a un bull."""
    assert DECIDER_ARMS == ("solo", "no_debate")
    arms = stored_decider_arms(payg_settings())

    assert arms == ("glm-5.2@solo", "glm-5.2@no_debate")
    assert not [arm for arm in arms if "+" in arm]


def test_the_decider_mode_and_role_are_the_ones_the_env_declares(tmp_path: Path) -> None:
    directory, _, _, _ = deciders(tmp_path, desks(tmp_path)[0])

    for arm in stored_decider_arms(payg_settings()):
        for attempts in calls_of(directory, arm):
            for call in attempts:
                assert call.role is AgentRole.DECIDER
                assert call.structured_output is StructuredOutputMode.FUNCTION_CALLING
                assert not call.cache_hit


def test_the_technical_digests_are_checked_and_announced_before_the_first_call(
    tmp_path: Path,
) -> None:
    """12/12 por dimensión (aquí 3/3): la evidencia guardada contesta las preguntas de hoy."""
    origin = desks(tmp_path)[0]
    zen = StoredFake()
    announced: list[tuple[int, str]] = []

    _, findings, _, _ = deciders(
        tmp_path, origin, zen, announce=lambda text: announced.append((len(zen.seen), text))
    )

    by_label = {match.arm: match for match in findings.digest_matches}
    for dimension in ("structure", "momentum", "volume"):
        found = by_label[dimension]
        assert (found.asked, found.paired, found.equal) == (COUNT, COUNT, COUNT), dimension
    assert findings.compared_with == str(origin)
    assert announced, "la tabla se anuncia"
    assert announced[0][0] == 0, "antes de que el proveedor vea nada"
    assert f"| structure | {COUNT} | {COUNT} | {COUNT} |" in announced[0][1]


def test_the_stored_content_is_tied_to_what_the_source_decider_read(tmp_path: Path) -> None:
    """Los digests técnicos prueban la pregunta; este, que el texto guardado es el que se leyó.

    Con la evidencia y los alegatos de `content.jsonl` se reconstruye el prompt del decisor del
    origen, y tiene que dar el digest de su journal.
    """
    _, findings, _, _ = deciders(tmp_path, desks(tmp_path)[0])

    by_label = {match.arm: match for match in findings.digest_matches}
    for bull in ("kimi-k3", "qwen3.8-max"):
        found = by_label[f"decider+{bull}"]
        assert (found.asked, found.equal) == (COUNT, COUNT)


def test_a_source_whose_technical_questions_differ_is_refused_without_calling(
    tmp_path: Path,
) -> None:
    """Con otros indicadores el prompt técnico es otro: la evidencia guardada ya no lo contesta."""
    origin = desks(tmp_path)[0]
    zen = StoredFake()

    with pytest.raises(ConfigError, match=f"structure: 0 de {COUNT}"):
        deciders(tmp_path, origin, zen, preset=OTHER_PRESET)

    assert zen.seen == []
    assert not list((tmp_path / "deciders").glob("*.jsonl"))


def test_the_proposals_are_written_next_to_the_evidence_they_read(tmp_path: Path) -> None:
    origin = desks(tmp_path)[0]

    directory, findings, _, _ = deciders(tmp_path, origin)

    before = {item.run_id: item for item in read_content(origin)}
    written = read_content(directory)
    assert len(written) == COUNT
    for item in written:
        assert item.evidence == before[item.run_id].evidence
        assert set(item.proposals) == set(DECIDER_ARMS)
        assert item.proposals["solo"].action is Action.BUY
        assert not item.bulls
        assert not item.decisions
    assert findings.proposals == {arm: ("buy",) * COUNT for arm in DECIDER_ARMS}
    assert findings.content_from == str(origin)


def test_a_lost_proposal_is_journaled_with_its_attempts_and_not_written_as_content(
    tmp_path: Path,
) -> None:
    """Dos intentos inválidos: la invocación queda en el journal con su error, sin propuesta."""

    class Stubborn(StoredFake):
        async def complete(  # type: ignore[override]
            self, choice: ModelChoice, prompt: str, schema: type
        ) -> Completion:
            done = await super().complete(choice, prompt, schema)
            if schema is Proposal and "structure-1" not in prompt:
                return Completion(text=with_null(proposal_payload(), "invalidation_price"))
            return done

    directory, findings, _, _ = deciders(tmp_path, desks(tmp_path)[0], Stubborn())

    (solo,) = (a for a in read_run(directory).arms if a.arm == "glm-5.2@solo")
    assert [len(record.calls) for record in solo.records] == [2] * COUNT
    assert all(record.errors for record in solo.records)
    assert {error.node for record in solo.records for error in record.errors} == {"decide_solo"}
    assert findings.proposals["solo"] == (None,) * COUNT
    assert findings.proposals["no_debate"] == ("buy",) * COUNT
    assert all("solo" not in item.proposals for item in read_content(directory))


def faulty() -> StoredFake:
    """`solo` omite la clave la primera vez; `no_debate` la escribe en `null` la primera vez."""
    return StoredFake(
        first={
            "solo": without_key(proposal_payload(), "invalidation_price"),
            "no_debate": with_null(proposal_payload(), "invalidation_price"),
        }
    )


def test_the_report_gives_validity_attempts_failures_and_actions_per_arm(tmp_path: Path) -> None:
    """Las cuatro columnas de fallo salen siempre, con sus ceros."""
    directory, findings, _, _ = deciders(tmp_path, desks(tmp_path)[0], faulty())

    report = render_deciders_report(read_run(directory), findings)

    interval = wilson_interval(COUNT, COUNT)
    assert interval is not None
    wilson = f"{interval.low:.0%} a {interval.high:.0%}"
    for arm in ("glm-5.2@solo", "glm-5.2@no_debate"):
        row = (
            f"| `{arm}` | {COUNT}/{COUNT} | {wilson} | {2 * COUNT} | 2.00 | "
            f"{COUNT} | 0 | 0 | 0 | {COUNT} · 0 · 0 |"
        )
        assert row in report, arm


def test_the_report_separates_an_omitted_key_from_a_null_on_an_action(tmp_path: Path) -> None:
    """`Field required` es una clave que no vino; `exige:` es la clave presente con `null`.

    Es la distinción que el contrato de T5 hizo posible y que T4 no podía hacer.
    """
    directory, findings, _, _ = deciders(tmp_path, desks(tmp_path)[0], faulty())

    report = render_deciders_report(read_run(directory), findings)

    faults = report.split("### Fallos de esquema, por lo que el modelo hizo")[1].split("##")[0]
    assert f"| `glm-5.2@solo` | clave omitida | `invalidation_price` | {COUNT} |" in faults
    assert f"| `glm-5.2@no_debate` | null en buy/sell | `invalidation_price` | {COUNT} |" in faults
    assert f"| `glm-5.2@solo` | schema | invalidation_price: Field required | {COUNT} |" in report
    assert "la acción buy exige: invalidation_price" in report


def test_the_report_states_the_gate_on_field_required_with_its_counts(tmp_path: Path) -> None:
    """La puerta del bloque: si una clave del contrato se omite, se dice cuántas veces."""
    directory, findings, _, _ = deciders(tmp_path, desks(tmp_path)[0], faulty())
    clean_dir, clean, _, _ = deciders(tmp_path / "limpio", desks(tmp_path / "limpio")[0])

    seen = render_deciders_report(read_run(directory), findings)
    nothing = render_deciders_report(read_run(clean_dir), clean)

    assert f"`invalidation_price` omitida (`Field required`): **{COUNT}** intento(s)" in seen
    assert "`glm-5.2@solo`" in seen.split("omitida (`Field required`)")[1].split("\n")[0]
    assert "Ningún intento omitió" in nothing
    assert "omitida (`Field required`): **" not in nothing


def test_the_report_lists_the_upstream_of_every_attempt(tmp_path: Path) -> None:
    directory, findings, _, _ = deciders(tmp_path, desks(tmp_path)[0], faulty())

    report = render_deciders_report(read_run(directory), findings)

    table = report.split("## Upstream de cada intento")[1].split("\n## ")[0]
    rows = [line for line in table.splitlines() if line.startswith("| `glm-5.2@")]
    assert len(rows) == 4 * COUNT  # dos brazos, dos intentos por invocación
    assert all("`accounts/fireworks/models/glm-5p3` | `fireworks` |" in row for row in rows)
    assert sum("| 1 | schema |" in row for row in rows) == 2 * COUNT
    assert sum("| 2 | válido |" in row for row in rows) == 2 * COUNT
    assert (
        f"| decider | `glm-5.2` | `accounts/fireworks/models/glm-5p3` | `fireworks` | {4 * COUNT} |"
        in report
    )


def test_an_attempt_without_the_header_says_so_in_the_report(tmp_path: Path) -> None:
    directory, findings, _, _ = deciders(tmp_path, desks(tmp_path)[0], StoredFake(upstream=None))

    report = render_deciders_report(read_run(directory), findings)

    table = report.split("## Upstream de cada intento")[1].split("\n## ")[0]
    assert table.count("| sin cabecera | — |") == 2 * COUNT


def test_the_cap_cuts_the_deciders_and_the_report_says_how_many_were_asked(
    tmp_path: Path,
) -> None:
    """Con el tope alcanzado no se abre otra invocación: queda k/N, no una corrida relanzada."""
    directory, findings, zen, guard = deciders(tmp_path, desks(tmp_path)[0], cap=0.000001)

    assert asked(zen, "Proposal") < 2 * COUNT
    assert guard.refused
    assert findings.refused == tuple(guard.refused)
    report = render_deciders_report(read_run(directory), findings)
    assert f"invocaciones que el tope impidió: {len(guard.refused)}" in report
    for arm in stored_decider_arms(payg_settings()):
        done = len(calls_of(directory, arm)) if (directory / f"{arm}.jsonl").is_file() else 0
        assert (
            f"| `{arm}` | {done}/{COUNT} |" in report.split("### Lo que el tope dejó sin medir")[1]
        )


def test_the_deciders_directory_is_a_probe_that_criteria_refuses(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory, _, _, _ = deciders(tmp_path, desks(tmp_path)[0])

    assert read_run(directory).meta.kind.value == "probe"
    assert criteria_main([str(directory), "--delta", "0.01"]) == 2
    assert "es un sondeo (kind=probe)" in capsys.readouterr().err


# ───────────────────────────────────── El bear en otro modo ───────────────────────────────────────


def test_the_bear_is_asked_the_very_same_prompt_in_the_requested_mode(tmp_path: Path) -> None:
    """El modo cambia; la pregunta, no: mismo `prompt_digest` que el origen, sin pedir evidencia."""
    origin = desks(tmp_path)[0]
    before = {
        r.run_id: r for a in read_run(origin).arms if a.arm == "minimax-m3@bear" for r in a.records
    }
    assert {c.structured_output for r in before.values() for c in r.calls} == {
        StructuredOutputMode.JSON_MODE
    }

    directory, findings, zen, _ = bear(tmp_path, origin)

    assert (asked(zen, "TechnicalVerdict"), asked(zen, "DebateBrief")) == (0, COUNT)
    (arm,) = (a for a in read_run(directory).arms if a.arm == "minimax-m3@bear")
    assert len(arm.records) == COUNT
    for record in arm.records:
        assert record.calls[0].prompt_digest == before[record.run_id].calls[0].prompt_digest
        assert {call.structured_output for call in record.calls} == {
            StructuredOutputMode.JSON_SCHEMA
        }
        assert {call.role for call in record.calls} == {AgentRole.BEAR}
    assert findings.requested_mode is StructuredOutputMode.JSON_SCHEMA
    assert stored_bear_arms(payg_settings()) == ("minimax-m3@bear",)


def test_the_bear_prompt_is_the_production_one_over_the_stored_evidence(tmp_path: Path) -> None:
    origin = desks(tmp_path)[0]
    stored = {item.run_id: item for item in read_content(origin)}
    plan, _ = plan_and_manifest(tmp_path)
    activations = asyncio.run(
        prepare_activations(plan, payg_settings(), lambda: NOW, COUNT, PRESET)
    )

    directory, _, _, _ = bear(tmp_path, origin)

    (arm,) = (a for a in read_run(directory).arms if a.arm == "minimax-m3@bear")
    asked_digests = {record.run_id: record.calls[0].prompt_digest for record in arm.records}
    for activation in activations:
        run_id = activation_run_id(activation.entry)
        where = activation.prepared
        prompt = debate_prompt(Side.BEAR, where.snapshot, where.indicators, stored[run_id].evidence)
        assert asked_digests[run_id] == prompt_digest(prompt)


def test_the_bear_digest_is_checked_and_announced_before_the_first_call(tmp_path: Path) -> None:
    origin = desks(tmp_path)[0]
    zen = StoredFake()
    announced: list[tuple[int, str]] = []

    _, findings, _, _ = bear(
        tmp_path, origin, backend=zen, announce=lambda text: announced.append((len(zen.seen), text))
    )

    (match,) = findings.digest_matches
    assert (match.arm, match.asked, match.paired, match.equal) == ("bear", COUNT, COUNT, COUNT)
    assert announced[0][0] == 0
    assert f"| bear | {COUNT} | {COUNT} | {COUNT} |" in announced[0][1]


def test_a_bear_prompt_that_differs_from_the_source_is_refused_without_calling(
    tmp_path: Path,
) -> None:
    origin = desks(tmp_path)[0]
    zen = StoredFake()

    with pytest.raises(ConfigError, match=f"bear: 0 de {COUNT}"):
        bear(tmp_path, origin, backend=zen, preset=OTHER_PRESET)

    assert zen.seen == []


def test_a_source_measured_with_another_bear_model_is_refused(tmp_path: Path) -> None:
    """Otro modo sobre otro modelo no compara nada: el origen tiene que haber medido a este bear."""
    origin = desks(tmp_path)[0]
    base = payg_settings()
    declared = base.role_config(AgentRole.BEAR)
    other = declared.model_copy(
        update={"primary": declared.primary.model_copy(update={"model": "otro-bear"})}
    )
    settings = base.model_copy(update={"roles": {**base.roles, AgentRole.BEAR: other}})
    zen = StoredFake()

    with pytest.raises(ConfigError, match="otro-bear@bear"):
        bear(tmp_path, origin, backend=zen, settings=settings)

    assert zen.seen == []


def test_the_bear_report_counts_first_attempts_against_the_source(tmp_path: Path) -> None:
    """Primeros intentos válidos k/12, intentos por alegato y lo mismo del origen al lado."""
    origin = desks(tmp_path)[0]

    directory, findings, _, _ = bear(tmp_path, origin, backend=StoredFake(string_grounding=True))

    report = render_bear_report(read_run(directory), findings, read_run(origin))
    table = report.split("## Bear: `json_schema` frente al origen")[1].split("\n## ")[0]
    done = f"{COUNT}/{COUNT}"
    assert f"| origen | json_mode | {done} | {done} | {COUNT} | 1.00 | 0 | 0 | 0 | 0 |" in table
    now = f"| ahora | json_schema | 0/{COUNT} | {done} | {2 * COUNT} | 2.00 | {COUNT} | 0 | 0 | 0 |"
    assert now in table
    assert "| 1.er intento en el origen | 1.er intento ahora |" in report
    assert report.count("| válido | schema |") == COUNT


def test_the_bear_report_groups_the_failure_messages_without_the_claim_index(
    tmp_path: Path,
) -> None:
    """`claims.0.grounded_in` y `claims.1.grounded_in` son el mismo fallo: una fila."""
    origin = desks(tmp_path)[0]

    directory, findings, _, _ = bear(tmp_path, origin, backend=StoredFake(string_grounding=True))

    report = render_bear_report(read_run(directory), findings, read_run(origin))
    faults = report.split("### Fallos de esquema, por lo que el modelo hizo")[1].split("##")[0]
    rows = [line for line in faults.splitlines() if line.startswith("| `minimax-m3@bear`")]
    fault = "claims.N.grounded_in: Input should be a valid array"
    assert rows == [f"| `minimax-m3@bear` | otro | `{fault}` | {COUNT} |"]


def test_the_bear_report_lists_the_upstream_of_every_attempt(tmp_path: Path) -> None:
    upstream = Upstream(model="accounts/anomalyinc/routers/zen-minimax-m3", endpoint="fireworks")
    origin = desks(tmp_path)[0]

    directory, findings, _, _ = bear(
        tmp_path, origin, backend=StoredFake(string_grounding=True, upstream=upstream)
    )

    report = render_bear_report(read_run(directory), findings, read_run(origin))
    table = report.split("## Upstream de cada intento")[1].split("\n## ")[0]
    rows = [line for line in table.splitlines() if line.startswith("| `minimax-m3@bear`")]
    assert len(rows) == 2 * COUNT
    assert all(
        "`accounts/anomalyinc/routers/zen-minimax-m3` | `fireworks` |" in row for row in rows
    )


def test_a_provider_failure_on_the_bear_is_a_row_and_not_a_first_attempt_that_failed_on_content(
    tmp_path: Path,
) -> None:
    origin = desks(tmp_path)[0]
    down = "Error code: 503 - {'error': {'message': 'Service Unavailable'}}"

    directory, findings, _, _ = bear(
        tmp_path, origin, backend=StoredFake(errors={"minimax-m3": down})
    )

    report = render_bear_report(read_run(directory), findings, read_run(origin))
    table = report.split("## Bear: `json_schema` frente al origen")[1].split("\n## ")[0]
    assert (
        f"| ahora | json_schema | 0/{COUNT} | 0/{COUNT} | {COUNT} | 1.00 | 0 | 0 | 0 | {COUNT} |"
        in table
    )
    for attempts in calls_of(directory, "minimax-m3@bear"):
        assert [c.failure_kind for c in attempts] == [FailureKind.TRANSPORT]
    assert "| sin respuesta | — |" in report


# ───────────────────────────────────────────── Comando ────────────────────────────────────────────


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--deciders-from", "x", "--desks", "--technicals-from", "y"], "uno solo"),
        (["--deciders-from", "x", "--bear-from", "y", "--bear-mode", "json_schema"], "uno solo"),
        (["--bear-from", "x"], "--bear-from necesita --bear-mode"),
        (["--bear-mode", "json_schema"], "--bear-mode es de --bear-from"),
        (["--deciders-from", "x"], "necesita --max-usd"),
        (["--bear-from", "x", "--bear-mode", "json_schema"], "necesita --max-usd"),
        (["--deciders-from", "x", "--technicals-from", "y", "--max-usd", "1"], "--technicals-from"),
    ],
)
def test_the_new_options_are_refused_when_they_do_not_make_sense(
    argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--machine", "desktop", *argv])

    assert caught.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    "option", [["--deciders-from"], ["--bear-mode", "json_schema", "--bear-from"]]
)
def test_the_command_refuses_a_source_of_another_plan_before_building_a_backend(
    option: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """El origen sintético corrió otro manifiesto que el real: se dice, y no se construye nada."""
    origin = desks(tmp_path)[0]
    monkeypatch.setattr(zen_probe, "load_settings", lambda _path: payg_settings())

    def forbidden(_settings: object) -> object:
        raise AssertionError("no se construye un proveedor antes de comprobar el origen")

    monkeypatch.setattr(zen_probe, "build_backends", forbidden)

    code = main(["--machine", "desktop", *option, str(origin), "--max-usd", "0.5"])

    assert code == 1
    assert "otro plan" in capsys.readouterr().err


def test_the_arm_name_helper_is_the_one_the_journals_use(tmp_path: Path) -> None:
    directory, _, _, _ = deciders(tmp_path, desks(tmp_path)[0])

    assert {path.stem for path in directory.glob("*.jsonl")} == {
        arm_name("glm-5.2", label) for label in DECIDER_ARMS
    } | {CONTENT_FILE.removesuffix(".jsonl")}
