"""El comando de lote del estudio hacia adelante: fechas, comprobaciones, tope, reanudación.

Lo que `docs/forward.md`, apartado 10, pide de cada lote, con su mutación donde la hay:

- **Las negativas por fecha**, con sus bordes, sin escribir nada y sin llamar a nadie.
- **Una negativa por cada comprobación previa rota**, con cero llamadas a un backend que las cuenta.
  Las comprobaciones caras sobre la etapa 1 y el pool real corren en el ensayo
  (`python -m crypto_agents.forward rehearse`); aquí cada una se prueba con lo que lee.
- **El tope**: proporcional al manifiesto y acumulado entre pasadas.
- **La reanudación**: una pasada cortada termina en la siguiente sin repetir ninguna llamada pagada.
- **El resumen** no tiene ningún campo de retorno y no recibe velas.
- **El upstream**: un cambio deja preparado el asiento del registro, con la primera llamada.
- **El ensayo pequeño**: las 14 entradas del fixture, por la función de corrida del lote y solo con
  caché, dan las decisiones, los stops y los digests de `data/stage1/`.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import math
import shutil
import socket
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

import crypto_agents.forward as module
import crypto_agents.llm as llm_module
import crypto_agents.prompts as prompts_module
from crypto_agents.ablation import build_run_meta, plan_from_manifest
from crypto_agents.audit import META_FILE, PlanKind, read_meta, read_run
from crypto_agents.cache import InMemoryResponseCache
from crypto_agents.design_effect import StageTie
from crypto_agents.forward import (
    FIRST_BLOCK,
    FORWARD_META_FILE,
    MERGED_AT,
    REGISTERED_MAX_ATTEMPTS,
    UPSTREAM_FILE,
    BatchEnvironment,
    BatchMeta,
    BatchOutcome,
    BatchSummary,
    Check,
    CheckOutcome,
    ForwardPaths,
    PreparedBlock,
    block_dates,
    catalog_check,
    compare_arms,
    current_path,
    date_refusal,
    decider_check,
    forward_arms,
    gap_result,
    main,
    manifest_check,
    measured_check,
    pool_check,
    read_decider,
    registered_decider,
    replay_forward_arms,
    router_attempts,
    run_batch,
    run_pass,
    spend_state,
    stage_check,
    summarise,
    ties_check,
)
from crypto_agents.forward_data import (
    STAGE1_DIR,
    CandleRef,
    HistoryExtension,
    StudyPopulation,
    read_histories,
    read_revisions,
    study_population,
)
from crypto_agents.forward_plan import (
    MAX_BLOCKS,
    POOL_COUNTS,
    PublishedTie,
    first_block,
)
from crypto_agents.forward_render import REGISTERED_PATH, MeasuredPath, sample_positions
from crypto_agents.funding import FUNDING_DIR
from crypto_agents.llm import Completion, ModelRouter, TokenUsage, Upstream, prompt_digest
from crypto_agents.quota import QuotaLedger
from crypto_agents.replay import ReplayCacheMissError, replay_router
from crypto_agents.selection import (
    DEFAULT_MANIFEST,
    HISTORY_DIR,
    PlannedEvaluation,
    SelectionManifest,
    load_manifest,
    load_selection_histories,
    write_manifest,
)
from crypto_agents.settings import ModelChoice, RoleConfig, load_settings
from crypto_agents.state import (
    Action,
    AgentRole,
    Backend,
    Billing,
    LLMCall,
    Proposal,
    StructuredOutputMode,
)
from tests.conftest import DATA_DIR, no_route_error
from tests.test_net_outcomes import mutated
from tests.test_zen_payg import payg_settings

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from crypto_agents.journal import EvaluationRecord
    from crypto_agents.settings import Settings
    from crypto_agents.state import LLMOutput

ROOT = DATA_DIR.parent.parent
DOCS = ROOT / "docs" / "forward.md"
READY = datetime(2026, 10, 30, tzinfo=UTC)
"""Desde aquí se corre el lote del bloque 1."""

ZEN = "https://opencode.ai/zen/v1"
SYMBOLS = ("ADA/USDT", "BNB/USDT", "BTC/USDT", "DOGE/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT")
FORMING = datetime(2026, 8, 15, 20, tzinfo=UTC)
FORMING_FLAGS = tuple(CandleRef(symbol, FORMING) for symbol in SYMBOLS)
"""El flag que acepta las siete velas de la fila 4 380, guardadas en formación."""
STAGE1_UPSTREAM = Upstream(model="accounts/fireworks/models/glm-5p3", endpoint="fireworks")
LOCK_NAME = "lote.lock"
"""El cerrojo de `batch`, `baselines` y `auto`, dentro del directorio de cada prueba."""


# ─────────────────────────────────────── Lo registrado ────────────────────────────────────────────


def test_the_first_block_is_the_one_the_registered_merge_gives() -> None:
    assert datetime(2026, 10, 17, 17, 12, 44, tzinfo=UTC) == MERGED_AT
    assert FIRST_BLOCK == first_block(MERGED_AT) == datetime(2026, 10, 22, tzinfo=UTC)
    committer = datetime(2026, 10, 17, 17, 12, 43, tzinfo=UTC)
    assert first_block(committer) == FIRST_BLOCK


def test_the_register_cites_the_merge_its_two_dates_and_the_start() -> None:
    register = DOCS.read_text("utf-8").split("## Registro del estudio", 1)[1]
    for fragment in (
        "ce3afe2f172f152ec91ca87a836e5359bd081837",
        "af85dd4bda46b0ee9ed67aa5c0530382f72a5497",
        "015f546de37024acc3d1631ccc9076e097fdef80",
        "2026-10-17T11:12:43-06:00",
        "2026-10-17T17:12:44Z",
        "[2026-10-22T00:00Z, 2026-10-29T00:00Z)",
        "Los artefactos de cada lote se versionan",
    ):
        assert fragment in register


# ─────────────────────────────────────────── Fechas ───────────────────────────────────────────────


def test_the_dates_of_the_first_block_are_the_registered_ones() -> None:
    dates = block_dates(FIRST_BLOCK)
    assert dates.number == 1
    assert dates.end == datetime(2026, 10, 29, tzinfo=UTC)
    assert dates.ready == READY
    assert dates.deadline == datetime(2026, 11, 12, tzinfo=UTC)
    assert block_dates(FIRST_BLOCK + timedelta(days=7 * 49)).number == 50


MICRO = timedelta(microseconds=1)


@pytest.mark.parametrize(
    ("start", "now", "reason"),
    [
        (datetime(2026, 10, 23, tzinfo=UTC), READY, "no es el inicio de un bloque"),
        (FIRST_BLOCK + timedelta(seconds=1), READY, "no es el inicio de un bloque"),
        (datetime(2026, 10, 15, tzinfo=UTC), READY, "anterior al bloque 1"),
        (FIRST_BLOCK + timedelta(days=7 * MAX_BLOCKS), READY, "acaba en el 52"),
        (FIRST_BLOCK, READY - MICRO, "todavía no"),
        (FIRST_BLOCK, datetime(2026, 11, 12, tzinfo=UTC), "venció"),
        (FIRST_BLOCK, datetime(2026, 12, 1, tzinfo=UTC), "venció"),
    ],
)
def test_each_date_refusal_with_its_edges(start: datetime, now: datetime, reason: str) -> None:
    refusal = date_refusal(start, now)
    assert refusal is not None
    assert reason in refusal


@pytest.mark.parametrize(
    "now", [READY, datetime(2026, 11, 12, tzinfo=UTC) - MICRO, datetime(2026, 11, 5, tzinfo=UTC)]
)
def test_the_batch_runs_from_its_ready_instant_until_just_before_its_deadline(
    now: datetime,
) -> None:
    assert date_refusal(FIRST_BLOCK, now) is None
    last = FIRST_BLOCK + timedelta(days=7 * (MAX_BLOCKS - 1))
    assert date_refusal(last, block_dates(last).ready) is None


def assert_the_edges(refusal: Callable[[datetime, datetime], str | None]) -> None:
    """Los cuatro bordes del lote del bloque 1 y el del bloque 1 mismo."""
    deadline = datetime(2026, 11, 12, tzinfo=UTC)
    assert refusal(FIRST_BLOCK, READY) is None
    assert refusal(FIRST_BLOCK, READY - MICRO) is not None
    assert refusal(FIRST_BLOCK, deadline - MICRO) is None
    assert refusal(FIRST_BLOCK, deadline) is not None
    assert "anterior al bloque 1" in (refusal(FIRST_BLOCK - timedelta(days=7), READY) or "")


def test_the_edges_hold_for_the_real_code() -> None:
    assert_the_edges(date_refusal)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("if now < dates.ready:", "if now <= dates.ready:"),
        ("if now >= dates.deadline:", "if now > dates.deadline:"),
    ],
)
def test_mutation_an_edge_moved_by_one_is_caught(old: str, new: str) -> None:
    mutant = mutated(date_refusal, old, new, module)
    with pytest.raises((AssertionError, ValueError)):
        assert_the_edges(mutant)


def test_mutation_the_first_block_moved_by_one_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """El borde del bloque 1 vive en `_block_refusal`, que comparten el lote y `baselines`."""
    mutant = mutated(
        module._block_refusal, "if start < FIRST_BLOCK:", "if start < FIRST_BLOCK - STEP:", module
    )
    monkeypatch.setattr(module, "_block_refusal", mutant)
    with pytest.raises((AssertionError, ValueError)):
        assert_the_edges(date_refusal)
    with pytest.raises((AssertionError, ValueError)):
        assert_the_baseline_edges(module.baselines_date_refusal)


DEADLINE = datetime(2026, 11, 12, tzinfo=UTC)
"""El plazo del lote del bloque 1: desde aquí se completan aparte sus líneas base."""


def assert_the_baseline_edges(refusal: Callable[[datetime, datetime], str | None]) -> None:
    """Las líneas base del bloque 1 se completan aparte desde el plazo exacto, nunca antes."""
    assert refusal(FIRST_BLOCK, DEADLINE) is None
    assert refusal(FIRST_BLOCK, DEADLINE + timedelta(days=300)) is None
    assert "todavía no" in (refusal(FIRST_BLOCK, DEADLINE - MICRO) or "")
    assert "anterior al bloque 1" in (refusal(FIRST_BLOCK - timedelta(days=7), DEADLINE) or "")
    assert "acaba en el 52" in (
        refusal(FIRST_BLOCK + timedelta(days=7 * MAX_BLOCKS), DEADLINE + timedelta(days=400)) or ""
    )


def test_the_baselines_are_completed_from_the_deadline_on() -> None:
    assert_the_baseline_edges(module.baselines_date_refusal)


def test_mutation_baselines_one_instant_before_the_deadline_is_caught() -> None:
    mutant = mutated(
        module.baselines_date_refusal,
        "if now < dates.deadline:",
        "if now < dates.deadline - timedelta(microseconds=1):",
        module,
    )
    with pytest.raises(AssertionError):
        assert_the_baseline_edges(mutant)


@pytest.mark.asyncio
async def test_baselines_before_the_deadline_touch_nothing(tmp_path: Path) -> None:
    refuse = Refusing()
    env = BatchEnvironment(
        clock=lambda: DEADLINE - MICRO,
        settings=refuse,  # type: ignore[arg-type]
        backends=refuse,  # type: ignore[arg-type]
        catalog=refuse,  # type: ignore[arg-type]
        fetch_candles=refuse,  # type: ignore[arg-type]
        fetch_funding=refuse,  # type: ignore[arg-type]
        paths=paths_in(tmp_path),
    )
    result = await module.run_baselines(FIRST_BLOCK, env, ())

    assert result.code == 2
    assert "todavía no" in result.lines[0]
    assert list(tmp_path.iterdir()) == []


class Refusing:
    """Cualquier cosa que el lote no debería tocar antes de pasar las fechas."""

    def __call__(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("el lote tocó el mundo antes de pasar las fechas")


@pytest.mark.parametrize(
    ("start", "now"),
    [
        (datetime(2026, 10, 15, tzinfo=UTC), READY),
        (FIRST_BLOCK, READY - MICRO),
        (FIRST_BLOCK, datetime(2026, 11, 12, tzinfo=UTC)),
        (datetime(2026, 10, 23, tzinfo=UTC), READY),
    ],
)
@pytest.mark.asyncio
async def test_a_date_refusal_touches_nothing_and_writes_nothing(
    start: datetime, now: datetime, tmp_path: Path
) -> None:
    refuse = Refusing()
    env = BatchEnvironment(
        clock=lambda: now,
        settings=refuse,  # type: ignore[arg-type]
        backends=refuse,  # type: ignore[arg-type]
        catalog=refuse,  # type: ignore[arg-type]
        fetch_candles=refuse,  # type: ignore[arg-type]
        fetch_funding=refuse,  # type: ignore[arg-type]
        paths=paths_in(tmp_path),
    )
    result = await run_batch(start, env, ())

    assert result.code == 2
    assert result.lines[0].startswith("error:")
    assert list(tmp_path.iterdir()) == []


def test_the_command_refuses_a_date_with_code_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env = BatchEnvironment(clock=lambda: READY, paths=paths_in(tmp_path))
    assert main(["batch", "--block", "2026-10-15"], env) == 2
    assert "anterior al bloque 1" in capsys.readouterr().err


def test_the_baselines_command_refuses_before_the_deadline_with_code_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env = BatchEnvironment(clock=lambda: DEADLINE - MICRO, paths=paths_in(tmp_path))
    assert main(["baselines", "--block", "2026-10-22"], env) == 2
    assert "se completan aparte desde que vence su plazo" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


# ──────────────────────────────────────── Las comprobaciones ──────────────────────────────────────


def study_settings(
    *,
    model: str = "glm-5.2",
    mode: StructuredOutputMode = StructuredOutputMode.FUNCTION_CALLING,
    temperature: float = 0.0,
    timeout: float = 240.0,
    base_url: str = ZEN,
) -> Settings:
    """El mapa de pago por uso con el decisor que registró la etapa 1."""
    roles = dict(payg_settings().roles)
    roles[AgentRole.DECIDER] = RoleConfig(
        primary=ModelChoice(
            backend=Backend.OPENAI,
            model=model,
            family="zhipu",
            structured_output=mode,
            temperature=temperature,
            quota_per_window=100_000,
        )
    )
    return load_settings(
        roles=roles,
        openai={"api_key": "no-se-usa", "base_url": base_url, "timeout_seconds": timeout},
        ollama={"host": "http://localhost:11434"},
        billing=Billing.PAYG,
    )


def registered() -> module.DeciderConfig:
    return registered_decider(read_meta(ROOT / STAGE1_DIR))


def test_the_registered_decider_is_the_one_of_the_document() -> None:
    config = registered()
    assert config.model == "glm-5.2"
    assert config.backend is Backend.OPENAI
    assert config.base_url == ZEN
    assert config.structured_output is StructuredOutputMode.FUNCTION_CALLING
    assert config.temperature == 0.0
    assert config.timeout_seconds == 240.0
    assert config.max_attempts == REGISTERED_MAX_ATTEMPTS == 2


def test_the_attempts_are_read_from_the_router_that_will_call() -> None:
    settings = study_settings()
    clock = lambda: READY  # noqa: E731
    ledger = QuotaLedger(settings.quota_window, clock)
    router = replay_router(settings, ledger, InMemoryResponseCache(), clock)
    assert router_attempts(router) == REGISTERED_MAX_ATTEMPTS
    other = ModelRouter(settings, ledger, {}, clock, max_attempts=3)
    assert router_attempts(other) == 3


def test_the_decider_read_as_registered_passes() -> None:
    check = decider_check(registered(), read_decider(study_settings(), 2), Billing.PAYG)
    assert check.passed, check.detail


@pytest.mark.parametrize(
    ("settings", "attempts", "billing", "field"),
    [
        (lambda: study_settings(temperature=0.1), 2, Billing.PAYG, "temperature"),
        (lambda: study_settings(timeout=120.0), 2, Billing.PAYG, "timeout_seconds"),
        (lambda: study_settings(model="glm-5.3"), 2, Billing.PAYG, "model"),
        (
            lambda: study_settings(mode=StructuredOutputMode.JSON_SCHEMA),
            2,
            Billing.PAYG,
            "structured_output",
        ),
        (
            lambda: study_settings(base_url="https://opencode.ai/zen/go/v1"),
            2,
            Billing.PAYG,
            "base_url",
        ),
        (study_settings, 3, Billing.PAYG, "max_attempts"),
        (study_settings, 2, Billing.GO, "billing go"),
    ],
)
def test_a_decider_other_than_the_registered_is_refused(
    settings: Callable[[], Settings], attempts: int, billing: Billing, field: str
) -> None:
    check = decider_check(registered(), read_decider(settings(), attempts), billing)
    assert not check.passed
    assert field in check.detail


def test_the_stage1_check_reads_the_copy_and_names_what_differs(tmp_path: Path) -> None:
    assert stage_check(ROOT / STAGE1_DIR).passed
    missing = stage_check(tmp_path)
    assert not missing.passed
    assert "falta" in missing.detail


@pytest.mark.asyncio
async def test_the_measured_path_of_the_registered_decider_is_the_registered_one() -> None:
    measured = await current_path(study_settings())
    assert measured == REGISTERED_PATH
    assert measured_check(measured).passed


@pytest.mark.asyncio
async def test_mutation_a_changed_retry_template_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_module, "_RETRY_TEMPLATE", llm_module._RETRY_TEMPLATE + " ")
    check = measured_check(await current_path(study_settings()))
    assert not check.passed
    assert "retry_template_sha256" in check.detail


@pytest.mark.asyncio
async def test_mutation_a_changed_template_is_caught(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for template in prompts_module.PROMPT_DIR.glob("*.md"):
        (tmp_path / template.name).write_bytes(template.read_bytes())
    target = tmp_path / "decider_solo.md"
    target.write_bytes(target.read_bytes() + b"\n")
    monkeypatch.setattr(prompts_module, "PROMPT_DIR", tmp_path)
    check = measured_check(await current_path(study_settings()))
    assert not check.passed
    assert "template_sha256" in check.detail


@pytest.mark.asyncio
async def test_mutation_a_changed_schema_or_envelope_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """El esquema de la clase que usa el decisor, `state.Proposal`, con una propiedad de más."""
    original = Proposal.model_json_schema

    def wider(*args: object, **kwargs: object) -> dict[str, object]:
        schema = original(*args, **kwargs)  # type: ignore[arg-type]
        return {**schema, "properties": {**schema["properties"], "extra": {"type": "string"}}}

    monkeypatch.setattr(Proposal, "model_json_schema", wider)
    check = measured_check(await current_path(study_settings()))
    assert not check.passed
    assert "schema_sha256" in check.detail

    monkeypatch.undo()
    hotter = measured_check(await current_path(study_settings(temperature=0.2)))
    assert not hotter.passed
    assert "envelope_sha256" in hotter.detail


def test_a_path_that_could_not_be_measured_does_not_pass() -> None:
    check = measured_check("sin transporte")
    assert not check.passed
    assert "no se pudo" in check.detail


def block_manifest_of(reference: SelectionManifest, **changes: object) -> SelectionManifest:
    return reference.model_copy(update=changes)


@pytest.mark.parametrize(
    ("changes", "name"),
    [
        ({"seed": 1}, "semilla"),
        ({"candle_limit": 400}, "candle_limit"),
        ({"horizon": 5}, "horizonte"),
        ({"warmup_bars": 300}, "warm-up"),
        ({"timeframe": "1h"}, "temporalidad"),
    ],
)
def test_a_manifest_with_another_parameter_is_refused(
    changes: dict[str, object], name: str
) -> None:
    reference = load_manifest(ROOT / DEFAULT_MANIFEST)
    assert manifest_check(reference, ROOT / DEFAULT_MANIFEST).passed
    check = manifest_check(block_manifest_of(reference, **changes), ROOT / DEFAULT_MANIFEST)
    assert not check.passed
    assert name in check.detail


def test_a_manifest_with_another_symbol_is_refused() -> None:
    reference = load_manifest(ROOT / DEFAULT_MANIFEST)
    fewer = block_manifest_of(reference, series=reference.series[1:])
    check = manifest_check(fewer, ROOT / DEFAULT_MANIFEST)
    assert not check.passed
    assert "símbolos" in check.detail


def tie(arm: str, actions: int = 0) -> StageTie:
    return StageTie(arm=arm, evaluations=140, found=140, actions=actions, stops=0, returns=0)


def published(arm: str, returns: int = 0) -> PublishedTie:
    return PublishedTie(arm=arm, published=140, found=140, returns=returns)


BASELINES = ("always_buy", "always_sell", "random_uniform", "rule_trend")


def test_the_ties_pass_at_560_of_560_and_a_single_difference_fails() -> None:
    check = ties_check([tie(arm) for arm in BASELINES], [published(arm) for arm in BASELINES])
    assert check.passed
    assert "560 de 560" in check.detail

    journal = ties_check(
        [tie(arm, actions=arm == "rule_trend") for arm in BASELINES],
        [published(arm) for arm in BASELINES],
    )
    assert not journal.passed
    assert "rule_trend" in journal.detail
    scored = ties_check(
        [tie(arm) for arm in BASELINES],
        [published(arm, returns=arm == "always_buy") for arm in BASELINES],
    )
    assert not scored.passed
    assert "always_buy (publicado)" in scored.detail


def test_the_pool_counts_pass_only_when_they_are_the_registered() -> None:
    assert pool_check(POOL_COUNTS, POOL_COUNTS).passed
    other = POOL_COUNTS.model_copy(update={"unconfirmed": 113})
    check = pool_check(other, POOL_COUNTS)
    assert not check.passed
    assert "4740 / 113 / 4737 / 4626" in check.detail


class FakeCatalog:
    def __init__(self, ids: set[str] | None = None, error: Exception | None = None) -> None:
        self.ids = frozenset(ids or set())
        self.error = error

    async def available_models(self) -> frozenset[str]:
        if self.error is not None:
            raise self.error
        return self.ids


@pytest.mark.asyncio
async def test_the_catalog_tells_a_missing_id_from_a_catalog_that_cannot_be_read() -> None:
    listed = await catalog_check(FakeCatalog({"glm-5.2", "otro"}), "glm-5.2")
    assert listed.passed

    gone = await catalog_check(FakeCatalog({"otro"}), "glm-5.2")
    assert not gone.passed
    assert gone.ends_study

    unreadable = await catalog_check(FakeCatalog(error=ConnectionError("sin red")), "glm-5.2")
    assert not unreadable.passed
    assert not unreadable.ends_study
    assert "inaccesible" in unreadable.detail


# ─────────────────────────────────────────── Un lote ──────────────────────────────────────────────

HOLD = Proposal(
    action=Action.HOLD,
    confidence=0.5,
    size_fraction=0.0,
    invalidation_price=None,
    rationale="Sin ventaja clara: se espera.",
).model_dump_json()


class FakeDecider:
    """El proveedor simulado: cuenta las llamadas y puede fallar en algunas."""

    def __init__(
        self,
        fail_on: Sequence[int] = (),
        completion_tokens: int = 3000,
        upstream: Upstream = STAGE1_UPSTREAM,
    ) -> None:
        self.fail_on = set(fail_on)
        self.completion_tokens = completion_tokens
        self.upstream = upstream
        self.calls = 0

    async def complete(self, choice: object, prompt: str, schema: type[LLMOutput]) -> Completion:
        self.calls += 1
        if self.calls in self.fail_on:
            raise no_route_error()
        return Completion(
            text=HOLD,
            usage=TokenUsage(
                prompt_tokens=1000, cached_tokens=0, completion_tokens=self.completion_tokens
            ),
            upstream=self.upstream,
        )


class TickingClock:
    """Un reloj que avanza un segundo cada vez que se le pregunta: cada pasada tiene su sello."""

    def __init__(self, start: datetime = READY) -> None:
        self.now = start

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now


def paths_in(tmp_path: Path) -> ForwardPaths:
    return ForwardPaths(
        history=tmp_path / "history",
        funding=tmp_path / "funding",
        manifests=tmp_path / "manifests",
        batches=tmp_path / "batches",
        cache=tmp_path / "cache",
        stage1=ROOT / STAGE1_DIR,
        published_history=ROOT / HISTORY_DIR,
        published_funding=ROOT / FUNDING_DIR,
        reference_manifest=ROOT / DEFAULT_MANIFEST,
        revisions=tmp_path / "revisions.json",
        lock=tmp_path / LOCK_NAME,
    )


def environment(
    tmp_path: Path,
    decider: FakeDecider,
    catalog: FakeCatalog | None = None,
    clock: TickingClock | None = None,
) -> BatchEnvironment:
    """El entorno de una pasada. Las pasadas de una misma prueba comparten `clock`: con un reloj
    nuevo cada una tendría el mismo sello y el mismo directorio."""
    return BatchEnvironment(
        clock=clock if clock is not None else TickingClock(),
        settings=study_settings,
        backends=lambda settings: {Backend.OPENAI: decider, Backend.OLLAMA: decider},
        catalog=lambda backends: catalog if catalog is not None else FakeCatalog({"glm-5.2"}),
        fetch_candles=Refusing(),  # type: ignore[arg-type]
        fetch_funding=Refusing(),  # type: ignore[arg-type]
        paths=paths_in(tmp_path),
    )


def prepared(tmp_path: Path, count: int = 3) -> PreparedBlock:
    """Un bloque ya preparado con las `count` primeras entradas del manifiesto de la etapa 1."""
    reference = load_manifest(ROOT / DEFAULT_MANIFEST)
    manifest = reference.model_copy(
        update={"entries": reference.entries[:count], "target": count, "strata": 1}
    )
    path = tmp_path / "manifests" / "2026-10-22.json"
    write_manifest(path, manifest)
    return PreparedBlock(
        block=block_dates(FIRST_BLOCK),
        history=(),
        funding=(),
        histories=load_selection_histories(reference, ROOT / HISTORY_DIR),
        counts=(count,),
        manifest=manifest,
        manifest_file=path,
    )


def passing(failed: Check | None = None) -> Callable[..., object]:
    """Las comprobaciones sin red, todas bien salvo `failed`. Las de verdad sobre la etapa 1
    tardan minutos y corren en el ensayo."""

    async def checks(*args: object) -> tuple[CheckOutcome, ...]:
        return tuple(
            CheckOutcome(check=check, passed=check is not failed, detail="simulada")
            for check in Check
            if check is not Check.CATALOG
        )

    return checks


def passes(tmp_path: Path) -> list[Path]:
    directory = tmp_path / "batches" / "2026-10-22"
    return sorted(directory.iterdir()) if directory.is_dir() else []


def forward_meta(directory: Path) -> BatchMeta:
    return BatchMeta.model_validate_json((directory / FORWARD_META_FILE).read_text("utf-8"))


@pytest.mark.parametrize("failed", [check for check in Check if check is not Check.CATALOG])
@pytest.mark.asyncio
async def test_each_broken_check_refuses_the_batch_without_a_single_call(
    failed: Check, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "offline_checks", passing(failed))
    decider = FakeDecider()

    result = await run_pass(prepared(tmp_path), environment(tmp_path, decider), ())

    assert result.code == 1
    assert decider.calls == 0
    assert f"«{failed.value}»" in result.lines[0]
    (directory,) = passes(tmp_path)
    assert not (directory / META_FILE).exists()
    meta = forward_meta(directory)
    assert meta.outcome is BatchOutcome.REFUSED
    assert [check.check for check in meta.checks if not check.passed] == [failed]
    assert any(str(directory / FORWARD_META_FILE) in line for line in result.lines)


@pytest.mark.asyncio
async def test_an_unreadable_catalog_refuses_and_a_missing_id_ends_the_study(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "offline_checks", passing())
    decider = FakeDecider()

    clock = TickingClock()
    unreadable = environment(
        tmp_path, decider, FakeCatalog(error=ConnectionError("sin red")), clock=clock
    )
    assert (await run_pass(prepared(tmp_path), unreadable, ())).code == 1

    gone = environment(tmp_path, decider, FakeCatalog({"otro"}), clock=clock)
    result = await run_pass(prepared(tmp_path), gone, ())
    assert result.code == 4
    assert result.lines[0].startswith("El estudio termina en el bloque 1")
    assert forward_meta(passes(tmp_path)[-1]).outcome is BatchOutcome.TERMINATED
    assert decider.calls == 0


@pytest.mark.asyncio
async def test_a_complete_batch_records_its_meta_and_prints_only_counters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "offline_checks", passing())
    decider = FakeDecider()

    result = await run_pass(prepared(tmp_path), environment(tmp_path, decider), ("batch",))

    assert result.code == 0, "\n".join(result.lines)
    assert decider.calls == 3
    (directory,) = passes(tmp_path)
    run = read_meta(directory)
    assert run.hostname == socket.gethostname()
    assert run.arms == ("solo", "always_buy", "always_sell", "random_uniform", "rule_trend")
    assert run.spend_cap_usd == pytest.approx(3 * 3 * 0.01509)
    meta = forward_meta(directory)
    assert meta.outcome is BatchOutcome.RAN
    assert meta.hostname == socket.gethostname()
    assert meta.python_version
    assert meta.uv_lock_sha256 is not None
    assert meta.measured_path == REGISTERED_PATH
    assert meta.decider == registered()
    assert meta.evaluations == 3
    assert all(check.passed for check in meta.checks)
    text = "\n".join(result.lines)
    assert "3 decisiones válidas, 0 respuestas inválidas, 0 faltantes" in text
    assert "lote completo: sí" in text
    assert "retorno" not in text.lower()
    assert str(directory / "solo.jsonl") in text
    assert "2026-10-22.json" in text


@pytest.mark.asyncio
async def test_a_batch_cut_halfway_resumes_without_repeating_a_paid_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "offline_checks", passing())
    block = prepared(tmp_path)
    clock = TickingClock()

    first = await run_pass(block, environment(tmp_path, FakeDecider(fail_on=(2,)), clock=clock), ())
    assert first.code == 3
    assert "2 decisiones válidas, 0 respuestas inválidas, 1 faltantes" in "\n".join(first.lines)

    shutil.rmtree(tmp_path / "cache")
    counting = FakeDecider()
    second = await run_pass(block, environment(tmp_path, counting, clock=clock), ())
    assert second.code == 0, "\n".join(second.lines)
    assert counting.calls == 1, "con la caché vacía, solo se llama a la evaluación que faltaba"
    earlier, later = passes(tmp_path)
    assert read_meta(later).resumed_from == str(earlier.resolve())
    assert forward_meta(later).resumed_from == str(earlier)
    arms = {arm.arm: arm for arm in read_run(later).arms}
    assert len(arms["solo"].records) == 1
    assert all(arms[name].records == () for name in BASELINES)
    whole = module.batch_records(tmp_path / "batches" / "2026-10-22")
    assert sorted(len(records) for records in whole.values()) == [3] * 5
    assert all(record.proposed is not None for record in whole["solo"])

    third = await run_pass(block, environment(tmp_path, FakeDecider(), clock=clock), ())
    assert third.code == 0
    assert "ya está completo" in third.lines[0]
    assert len(passes(tmp_path)) == 2


@pytest.mark.asyncio
async def test_the_cap_accumulates_across_passes_and_a_spent_cap_calls_nobody(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cada llamada cuesta 0.44 USD y el tope de 3 evaluaciones es 0.136: la primera pasada hace
    una y el tope corta las otras dos; la segunda ya no llama a nadie."""
    monkeypatch.setattr(module, "offline_checks", passing())
    block = prepared(tmp_path)
    clock = TickingClock()
    expensive = FakeDecider(completion_tokens=100_000)

    first = await run_pass(block, environment(tmp_path, expensive, clock=clock), ())
    assert first.code == 3
    assert expensive.calls == 1
    assert "tope 2" in "\n".join(first.lines)

    again = FakeDecider()
    second = await run_pass(block, environment(tmp_path, again, clock=clock), ())
    assert second.code == 3
    assert again.calls == 0
    meta = forward_meta(passes(tmp_path)[-1])
    assert meta.outcome is BatchOutcome.CAP_EXHAUSTED
    assert meta.spent_before_usd >= meta.spend_cap_usd


def priced_call(completion_tokens: int) -> LLMCall:
    return LLMCall(
        role=AgentRole.DECIDER,
        backend=Backend.OPENAI,
        model="glm-5.2",
        prompt_digest="a" * 64,
        structured_output=StructuredOutputMode.FUNCTION_CALLING,
        quota_weight=1.0,
        latency_ms=1.0,
        valid=True,
        at=READY,
        prompt_tokens=1000,
        cached_tokens=0,
        completion_tokens=completion_tokens,
    )


def assert_the_cap_is_proportional_and_accumulated() -> None:
    reference = load_manifest(ROOT / DEFAULT_MANIFEST)
    fresh = module.spend_state(study_settings(), reference, (), READY)
    assert fresh.cap_usd == pytest.approx(3 * 140 * 0.01509)
    assert fresh.spent_before_usd == 0.0

    previous = [priced_call(3000), priced_call(3000)]
    spent = module.spend_state(study_settings(), reference, previous, READY)
    assert spent.spent_before_usd == pytest.approx(2 * (1000 * 1.40 + 3000 * 4.40) / 1e6)
    assert spent.guard.allow("siguiente")


def test_the_cap_is_proportional_to_the_manifest_and_accumulated() -> None:
    assert_the_cap_is_proportional_and_accumulated()


def test_mutation_a_fixed_cap_per_batch_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "batch_spend_cap", lambda manifest: 2.22)
    with pytest.raises(AssertionError):
        assert_the_cap_is_proportional_and_accumulated()


def test_mutation_a_cap_that_forgets_what_was_spent_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mutant = mutated(spend_state, "guard.add(previous)", "guard.add(())", module)
    monkeypatch.setattr(module, "spend_state", mutant)
    with pytest.raises(AssertionError):
        assert_the_cap_is_proportional_and_accumulated()


@pytest.mark.asyncio
async def test_an_empty_block_records_itself_once_and_calls_nobody(tmp_path: Path) -> None:
    empty = prepared(tmp_path)._replace(counts=(0,), manifest=None, manifest_file=None)
    decider = FakeDecider()

    clock = TickingClock()
    first = await run_pass(empty, environment(tmp_path, decider, clock=clock), ())
    second = await run_pass(empty, environment(tmp_path, decider, clock=clock), ())

    assert first.code == second.code == 0
    assert decider.calls == 0
    (directory,) = passes(tmp_path)
    assert forward_meta(directory).outcome is BatchOutcome.EMPTY
    assert forward_meta(directory).evaluations == 0
    assert "no tiene activaciones" in second.lines[0]


# ──────────────────────────────────────── El resumen ──────────────────────────────────────────────


def test_the_summary_has_no_field_for_a_return_and_receives_no_candles() -> None:
    names = set(BatchSummary.model_fields)
    for word in ("return", "retorno", "pnl", "profit", "gain", "score", "outcome", "result"):
        assert not any(word in name for name in names), word
    parameters = set(inspect.signature(summarise).parameters)
    assert not parameters & {"histories", "rows", "candles", "horizon", "scoring"}


@pytest.mark.asyncio
async def test_a_new_upstream_leaves_the_register_entry_ready_with_its_first_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "offline_checks", passing())
    other = Upstream(model="accounts/together/models/glm-5p2", endpoint="together")
    document = DOCS.read_bytes()

    result = await run_pass(
        prepared(tmp_path), environment(tmp_path, FakeDecider(upstream=other)), ()
    )

    text = "\n".join(result.lines)
    assert "**cambió**" in text
    (directory,) = passes(tmp_path)
    entry = (directory / UPSTREAM_FILE).read_text("utf-8")
    assert "en el bloque 1" in entry
    assert "`accounts/together/models/glm-5p2` por `together`" in entry
    assert "Ya no aparece `accounts/fireworks/models/glm-5p3` por `fireworks`" in entry
    solo = next(arm for arm in read_run(directory).arms if arm.arm == "solo")
    first = min((call.at, record.run_id) for record in solo.records for call in record.calls)
    assert f"`{first[1]}`" in entry
    assert "la etapa 1" in entry
    assert DOCS.read_bytes() == document, "el lote no escribe el registro: lo copia el operador"


@pytest.mark.asyncio
async def test_the_same_upstream_as_stage1_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "offline_checks", passing())
    result = await run_pass(prepared(tmp_path), environment(tmp_path, FakeDecider()), ())
    assert "respecto a la etapa 1" in "\n".join(result.lines)
    assert "sin cambios" in "\n".join(result.lines)
    (directory,) = passes(tmp_path)
    assert not (directory / UPSTREAM_FILE).exists()


# ──────────────────────────────────── El ensayo, en pequeño ───────────────────────────────────────


class JournalBackend:
    """Contesta a cada prompt de `solo` lo que decidió la etapa 1 para ese mismo prompt."""

    def __init__(self, records: Sequence[EvaluationRecord]) -> None:
        self.answers: dict[str, str] = {}
        for record in records:
            decided = record.proposed
            if decided is None or not record.calls:
                continue
            self.answers[record.calls[0].prompt_digest] = decided.model_dump_json()

    async def complete(self, choice: object, prompt: str, schema: type[LLMOutput]) -> str:
        return self.answers[prompt_digest(prompt)]


@pytest.mark.asyncio
async def test_the_fourteen_through_the_batch_code_reproduce_stage1(tmp_path: Path) -> None:
    """La función de corrida del lote sobre las 14 entradas del fixture de `forward_render`.

    La primera pasada llena una caché con lo que la etapa 1 decidió; la segunda corre solo con
    caché, como el ensayo de verdad, y tiene que dar la acción, el stop y los `prompt_digest` de
    `data/stage1/` en las cinco ramas, sin una sola llamada viva.
    """
    reference = load_manifest(ROOT / DEFAULT_MANIFEST)
    whole = plan_from_manifest(reference, load_selection_histories(reference, ROOT / HISTORY_DIR))
    positions = sample_positions(reference.entries, 2)
    plan = whole._replace(entries=tuple(whole.entries[position] for position in positions))
    stage = {arm.arm: arm.records for arm in read_run(ROOT / STAGE1_DIR).arms}
    settings = study_settings()
    cache = InMemoryResponseCache()
    clock = TickingClock()

    filled = tmp_path / "llenado"
    backend = JournalBackend(stage["solo"])
    await replay_forward_arms(
        plan,
        settings,
        cache,
        clock,
        QuotaLedger(settings.quota_window, clock),
        filled,
        None,
        {Backend.OPENAI: backend, Backend.OLLAMA: backend},
    )
    replayed = await replay_forward_arms(
        plan,
        settings,
        cache,
        clock,
        QuotaLedger(settings.quota_window, clock),
        tmp_path / "ensayo",
        None,
        None,
    )
    run_ids = {record.run_id for record in replayed["solo"]}
    theirs = {
        arm: tuple(record for record in records if record.run_id in run_ids)
        for arm, records in stage.items()
    }

    comparisons = compare_arms(replayed, theirs)

    assert [item.arm for item in comparisons] == [arm.name for arm in forward_arms()]
    for item in comparisons:
        assert item.exact, item
        assert item.shared == 14


def test_a_different_decision_or_a_live_call_breaks_the_comparison() -> None:
    stage = {arm.arm: arm.records for arm in read_run(ROOT / STAGE1_DIR).arms}
    same = compare_arms(stage, stage)
    assert all(item.live_calls > 0 for item in same if item.arm == "solo")
    assert not next(item for item in same if item.arm == "solo").exact
    assert all(item.exact for item in same if item.arm != "solo")

    first = stage["rule_trend"][0]
    decided = first.proposed
    assert decided is not None
    flipped = first.model_copy(
        update={"proposal": decided.model_copy(update={"invalidation_price": 1.0})}
    )
    changed = {**stage, "rule_trend": (flipped, *stage["rule_trend"][1:])}
    item = next(item for item in compare_arms(changed, stage) if item.arm == "rule_trend")
    assert item.stops == 139
    assert not item.exact


def test_build_run_meta_records_the_hostname_and_an_old_meta_loads_without_it(
    tmp_path: Path,
) -> None:
    meta = build_run_meta(
        study_settings(),
        PlanKind.MANIFEST,
        ROOT / DEFAULT_MANIFEST,
        "c" * 64,
        (),
        False,
        forward_arms(),
        READY,
    )
    assert meta.hostname == socket.gethostname()
    assert read_meta(ROOT / STAGE1_DIR).hostname is None
    payload = json.loads((ROOT / STAGE1_DIR / META_FILE).read_text("utf-8"))
    assert "hostname" not in payload


def test_the_measured_path_constants_are_the_ones_of_the_document() -> None:
    text = DOCS.read_text("utf-8")
    for value in REGISTERED_PATH.model_dump().values():
        assert f"`{value}`" in text
    assert REGISTERED_PATH.differences(REGISTERED_PATH) == ()
    other = MeasuredPath(**{**REGISTERED_PATH.model_dump(), "schema_sha256": "0" * 64})
    assert other.differences(REGISTERED_PATH) == ("schema_sha256",)


def summary_of(records: Mapping[str, Sequence[EvaluationRecord]]) -> BatchSummary:
    settings = study_settings()
    guard = spend_state(settings, load_manifest(ROOT / DEFAULT_MANIFEST), (), READY).guard
    expected = {arm.name: {role: Backend.OPENAI for role in AgentRole} for arm in forward_arms()}
    return summarise(
        block_dates(FIRST_BLOCK), 140, records, expected, guard, "la etapa 1", (), (140,)
    )


def test_the_summary_of_stage1_counts_its_decisions() -> None:
    stage = {arm.arm: arm.records for arm in read_run(ROOT / STAGE1_DIR).arms if arm.arm != "full"}
    summary = summary_of(stage)
    assert summary.decided == 140
    assert summary.invalid == summary.missing == summary.other_undecided == 0
    assert summary.baselines_decided == dict.fromkeys(BASELINES, 140)
    assert summary.complete
    assert summary.upstream_changed
    assert summary.filled_blocks == 1


# ─────────────────────────────── Mutaciones de la negativa y la caché ─────────────────────────────


async def assert_a_failed_check_calls_nobody(tmp_path: Path) -> None:
    decider = FakeDecider()
    result = await module.run_pass(prepared(tmp_path), environment(tmp_path, decider), ())
    assert result.code == 1
    assert decider.calls == 0


@pytest.mark.asyncio
async def test_mutation_a_batch_that_ignores_a_failed_check_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "offline_checks", passing(Check.RENDER))
    await assert_a_failed_check_calls_nobody(tmp_path / "real")

    mutant = mutated(
        run_pass,
        "failed = [check for check in checks if not check.passed]",
        "failed = [check for check in checks if check.ends_study]",
        module,
    )
    monkeypatch.setattr(module, "run_pass", mutant)
    with pytest.raises(AssertionError):
        await assert_a_failed_check_calls_nobody(tmp_path / "mutante")


async def assert_the_resume_pays_only_what_is_missing(tmp_path: Path) -> None:
    block = prepared(tmp_path)
    clock = TickingClock()
    cut = await module.run_pass(
        block, environment(tmp_path, FakeDecider(fail_on=(2,)), clock=clock), ()
    )
    assert cut.code == 3
    shutil.rmtree(tmp_path / "cache")
    counting = FakeDecider()
    assert (
        await module.run_pass(block, environment(tmp_path, counting, clock=clock), ())
    ).code == 0
    assert counting.calls == 1


@pytest.mark.asyncio
async def test_mutation_a_resume_that_does_not_skip_the_journaled_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sin saltarse lo ya journalizado y con la caché vacía, la reanudación volvería a pagar las
    tres evaluaciones."""
    monkeypatch.setattr(module, "offline_checks", passing())
    await assert_the_resume_pays_only_what_is_missing(tmp_path / "real")

    mutant = mutated(run_pass, "skip=state.done,", "skip=None,", module)
    monkeypatch.setattr(module, "run_pass", mutant)
    with pytest.raises(AssertionError):
        await assert_the_resume_pays_only_what_is_missing(tmp_path / "mutante")


@pytest.mark.asyncio
async def test_a_run_id_complete_in_two_passes_stops_the_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "offline_checks", passing())
    block = prepared(tmp_path)
    assert (await run_pass(block, environment(tmp_path, FakeDecider()), ())).code == 0
    (first,) = passes(tmp_path)
    shutil.copytree(first, first.parent / "29991231T000000Z")

    decider = FakeDecider()
    result = await run_pass(block, environment(tmp_path, decider), ())

    assert result.code == 1
    assert "ya tenía registro completo" in result.lines[0]
    assert decider.calls == 0
    with pytest.raises(module.DuplicateRecordError):
        module.batch_records(tmp_path / "batches" / "2026-10-22")


# ─────────────────────────────────── El lote, de punta a punta ────────────────────────────────────


def quiet_market(symbol: str, until: datetime) -> list[list[float]]:
    """Lo publicado, la fila 4 380 ya cerrada y velas tranquilas con un salto de vez en cuando:
    unas pocas activaciones por semana, no una por vela."""
    published = load_selection_histories(
        load_manifest(ROOT / DEFAULT_MANIFEST), ROOT / HISTORY_DIR
    )[symbol]
    rows = [list(row) for row in published[:-1]]
    forming = list(published[-1])
    rows.append([forming[0], forming[1], forming[2], forming[3], forming[4], forming[5] * 2])
    level = forming[4]
    moment = datetime.fromtimestamp(forming[0] / 1000, UTC) + timedelta(hours=4)
    step = 0
    while moment + timedelta(hours=4) <= until:
        close = level * (1 + 0.002 * math.sin(step / 3))
        spike = step % 37 == 36
        high = close * (1.06 if spike else 1.001)
        rows.append([
            float(int(moment.timestamp() * 1000)), close, high, close * 0.999,
            close * (1.04 if spike else 1.0), 1000.0 * (4 if spike else 1),
        ])  # fmt: skip
        moment += timedelta(hours=4)
        step += 1
    return rows


class QuietExchange:
    def __init__(self) -> None:
        self.asked = 0

    async def __call__(
        self, symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> list[list[float]]:
        self.asked += 1
        start_ms, end_ms = start.timestamp() * 1000, end.timestamp() * 1000
        return [row for row in quiet_market(symbol, end) if start_ms <= row[0] < end_ms]


async def quiet_funding(start: datetime, end: datetime) -> dict[str, bytes]:
    series: dict[str, bytes] = {}
    for symbol in SYMBOLS:
        content = (ROOT / FUNDING_DIR / f"{symbol.replace('/', '').lower()}.csv").read_bytes()
        moment = datetime(2026, 8, 16, 8, tzinfo=UTC)
        while moment <= end:
            content += f"{int(moment.timestamp() * 1000)},{1e-05!r}\n".encode()
            moment += timedelta(hours=8)
        series[symbol] = content
    return series


@pytest.mark.asyncio
async def test_the_whole_batch_from_its_date_to_the_files_to_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fechas, extensión, población real sobre las series extendidas, manifiesto, comprobaciones
    (simuladas), corrida y lista de archivos. El manifiesto es la población del bloque 1."""
    monkeypatch.setattr(module, "offline_checks", passing())
    decider = FakeDecider()
    candles = QuietExchange()
    env = environment(tmp_path, decider)._replace(
        fetch_candles=candles, fetch_funding=quiet_funding
    )

    stopped = await run_batch(FIRST_BLOCK, env, ("batch", "--block", "2026-10-22"))
    assert stopped.code == 5
    result = await run_batch(FIRST_BLOCK, env, ("batch",), FORMING_FLAGS)

    text = "\n".join(result.lines)
    assert candles.asked == 14
    manifest_file = tmp_path / "manifests" / "2026-10-22.json"
    manifest = load_manifest(manifest_file)
    assert manifest.entries, "el mercado tranquilo tendría que dar alguna activación"
    assert result.code == 0, text
    assert decider.calls == len(manifest.entries)
    histories = read_histories(tmp_path / "history")
    population = study_population(histories, [FIRST_BLOCK])
    assert manifest.entries == population.blocks[FIRST_BLOCK]
    for entry in manifest.entries:
        closes = entry.at + timedelta(hours=4)
        assert FIRST_BLOCK <= closes < FIRST_BLOCK + timedelta(days=7)
    revised, directory = passes(tmp_path)
    assert forward_meta(revised).outcome is BatchOutcome.REVISED
    meta = forward_meta(directory)
    assert meta.manifest_path == str(manifest_file)
    assert len(meta.history) == len(meta.funding) == 7
    assert all(item.last == datetime(2026, 10, 29, 20, tzinfo=UTC) for item in meta.history)
    assert all(item.last == datetime(2026, 10, 30, tzinfo=UTC) for item in meta.funding)
    for item in (*meta.history, *meta.funding):
        assert item.path in text
    assert str(manifest_file) in text
    assert f"{len(manifest.entries)} decisiones válidas" in text
    assert "1 no vacíos, 0 vacíos" in text
    for symbol in SYMBOLS:
        published = ROOT / HISTORY_DIR / f"{symbol.replace('/', '').lower()}_4h.csv"
        extended = tmp_path / "history" / published.name
        assert extended.read_bytes().startswith(published.read_bytes())
    accepted = read_revisions(tmp_path / "revisions.json").accepted
    assert {item.symbol for item in accepted} == set(SYMBOLS)
    assert meta.accepted == accepted
    assert str(tmp_path / "revisions.json") in text


@pytest.mark.asyncio
async def test_a_block_after_the_study_closed_is_refused_before_its_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Con 50 bloques no vacíos antes, el bloque 52 ya no es del estudio: código 2, sin manifiesto
    y sin llamar a nadie."""
    start = FIRST_BLOCK + timedelta(days=7 * 51)
    entry = PlannedEvaluation(
        symbol="BTC/USDT", timeframe="4h", index=0, at=FIRST_BLOCK, candles_digest="0" * 64
    )

    async def no_history(*args: object, **kwargs: object) -> HistoryExtension:
        return HistoryExtension(files=(), accepted=())

    async def no_funding(*args: object, **kwargs: object) -> tuple[()]:
        return ()

    def every_block_filled(histories: object, starts: Sequence[datetime]) -> StudyPopulation:
        return StudyPopulation(blocks={moment: (entry,) for moment in starts}, activations={})

    monkeypatch.setattr(module, "extend_history", no_history)
    monkeypatch.setattr(module, "extend_funding", no_funding)
    monkeypatch.setattr(module, "read_histories", lambda directory: {})
    monkeypatch.setattr(module, "study_population", every_block_filled)
    decider = FakeDecider()
    env = environment(tmp_path, decider)._replace(clock=lambda: block_dates(start).ready)

    result = await run_batch(start, env, ())

    assert result.code == 2
    assert "ya cerró en el bloque 50" in result.lines[0]
    assert not (tmp_path / "manifests").exists()
    assert decider.calls == 0


# ───────────────────────────── Velas revisadas y velas que faltan ─────────────────────────────────


@pytest.mark.asyncio
async def test_a_revised_candle_stops_the_batch_with_its_register_entry_and_flags(
    tmp_path: Path,
) -> None:
    """El exchange de hoy devuelve cerrada la fila 4 380 que `data/history/` guardó en formación: el
    lote se detiene con el código 5, sin extender nada, y deja preparados el asiento y el flag."""
    decider = FakeDecider()
    env = environment(tmp_path, decider)._replace(
        fetch_candles=QuietExchange(), fetch_funding=quiet_funding, clock=lambda: READY
    )

    result = await run_batch(FIRST_BLOCK, env, ())

    text = "\n".join(result.lines)
    assert result.code == 5
    assert decider.calls == 0
    assert not (tmp_path / "history").exists()
    (directory,) = passes(tmp_path)
    meta = forward_meta(directory)
    assert meta.outcome is BatchOutcome.REVISED
    assert {item.symbol for item in meta.revised} == set(SYMBOLS)
    for symbol in SYMBOLS:
        assert f"--accept-revised {symbol}@2026-08-15T20:00Z" in text
        assert f"**{symbol}**, vela que abre el 2026-08-15T20:00Z" in text
    assert "guardado: apertura 0.1781, máximo 0.1783, mínimo 0.1772, cierre 0.1778" in text


class Gapped(QuietExchange):
    """El mercado tranquilo sin una vela nueva de SOL/USDT, mientras `present` sea falso."""

    def __init__(self) -> None:
        super().__init__()
        self.present = False

    async def __call__(
        self, symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> list[list[float]]:
        rows = await super().__call__(symbol, timeframe, start, end)
        hole = datetime(2026, 10, 1, tzinfo=UTC).timestamp() * 1000
        return rows if self.present or symbol != "SOL/USDT" else [r for r in rows if r[0] != hole]


async def assert_a_gap_waits_a_day_before_it_ends_the_study(tmp_path: Path) -> None:
    clock = TickingClock()
    env = environment(tmp_path, FakeDecider(), clock=clock)._replace(
        fetch_candles=Gapped(), fetch_funding=quiet_funding
    )

    seen = await run_batch(FIRST_BLOCK, env, (), FORMING_FLAGS)
    assert seen.code == 6, "\n".join(seen.lines)
    assert "SOL/USDT@2026-10-01T00:00Z" in "\n".join(seen.lines)

    clock.now += timedelta(hours=23)
    assert (await run_batch(FIRST_BLOCK, env, (), FORMING_FLAGS)).code == 6

    clock.now += timedelta(hours=1)
    persistent = await run_batch(FIRST_BLOCK, env, (), FORMING_FLAGS)
    assert persistent.code == 7
    assert persistent.lines[0].startswith("El estudio termina en el bloque 1")
    outcomes = [forward_meta(path).outcome for path in passes(tmp_path)]
    assert outcomes == [BatchOutcome.GAP, BatchOutcome.GAP, BatchOutcome.GAP_PERSISTENT]
    assert not (tmp_path / "history").exists()


@pytest.mark.asyncio
async def test_a_missing_candle_waits_a_day_and_then_ends_the_study(tmp_path: Path) -> None:
    await assert_a_gap_waits_a_day_before_it_ends_the_study(tmp_path)


@pytest.mark.asyncio
async def test_mutation_a_gap_declared_persistent_without_waiting_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mutant = mutated(gap_result, ">= GAP_WAIT", ">= timedelta(0)", module)
    monkeypatch.setattr(module, "gap_result", mutant)
    with pytest.raises(AssertionError):
        await assert_a_gap_waits_a_day_before_it_ends_the_study(tmp_path)


# ─────────────────────────────── Las líneas base después del plazo ────────────────────────────────


def passing_baselines(*args: object) -> tuple[CheckOutcome, ...]:
    """Las comprobaciones de las líneas base, todas bien. Las de verdad leen el pool entero."""
    return tuple(
        CheckOutcome(check=check, passed=True, detail="simulada")
        for check in (Check.STAGE1, Check.TIES, Check.POOL, Check.MANIFEST)
    )


def no_model_environment(tmp_path: Path) -> BatchEnvironment:
    """Después del plazo, sin backends ni catálogo: si alguno se toca, la prueba falla."""
    refuse = Refusing()
    return BatchEnvironment(
        clock=TickingClock(DEADLINE),
        settings=study_settings,
        backends=refuse,  # type: ignore[arg-type]
        catalog=refuse,  # type: ignore[arg-type]
        fetch_candles=refuse,  # type: ignore[arg-type]
        fetch_funding=refuse,  # type: ignore[arg-type]
        paths=paths_in(tmp_path),
    )


async def assert_the_baselines_are_completed_calling_nobody(tmp_path: Path) -> None:
    """Un lote que no corrió: las cuatro líneas base quedan con sus tres registros, sin `solo`, sin
    una sola llamada y sin construir backends. La segunda vez no hay nada que completar."""
    block = prepared(tmp_path)
    env = no_model_environment(tmp_path)

    result = await module.baselines_pass(block, env, ("baselines",))

    found = "\n".join(result.lines)
    assert result.code == 0, found
    assert "llamadas a un modelo: 0" in found
    (directory,) = passes(tmp_path)
    assert forward_meta(directory).outcome is BatchOutcome.BASELINES
    assert forward_meta(directory).measured_path is None
    run = read_run(directory)
    assert {arm.arm for arm in run.arms} == set(BASELINES)
    assert not (directory / "solo.jsonl").exists()
    assert all(len(arm.records) == 3 for arm in run.arms)
    assert all(record.calls == () for arm in run.arms for record in arm.records)
    assert all(record.proposed is not None for arm in run.arms for record in arm.records)

    again = await module.baselines_pass(block, env, ("baselines",))
    assert again.code == 0
    assert "ya tienen registro completo" in again.lines[0]
    assert len(passes(tmp_path)) == 1


@pytest.mark.asyncio
async def test_the_baselines_after_the_deadline_are_completed_with_zero_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "baseline_checks", passing_baselines)
    await assert_the_baselines_are_completed_calling_nobody(tmp_path)


@pytest.mark.asyncio
async def test_mutation_a_baselines_run_with_solo_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Con `solo` entre los brazos, el router solo de caché sobre una caché vacía se niega a llamar:
    la pasada no termina como si nada."""
    monkeypatch.setattr(module, "baseline_checks", passing_baselines)
    monkeypatch.setattr(module, "baseline_arms", forward_arms)
    with pytest.raises((AssertionError, ReplayCacheMissError)):
        await assert_the_baselines_are_completed_calling_nobody(tmp_path)


async def assert_a_call_in_a_baseline_is_refused(tmp_path: Path) -> None:
    result = await module.baselines_pass(prepared(tmp_path), no_model_environment(tmp_path), ())
    assert result.code == 1
    assert result.lines[0].startswith("**Una línea base llamó a un modelo**: 3 llamadas")


def calling_baselines(real: Callable[..., object]) -> Callable[..., object]:
    """La función de corrida, con una llamada al decisor colada en cada registro de `rule_trend`."""

    async def run(*args: object, **kwargs: object) -> dict[str, tuple[EvaluationRecord, ...]]:
        found = await real(*args, **kwargs)  # type: ignore[misc]
        call = priced_call(10)
        return {
            arm: tuple(
                record.model_copy(update={"calls": (call,)}) if arm == "rule_trend" else record
                for record in records
            )
            for arm, records in found.items()
        }

    return run


@pytest.mark.asyncio
async def test_a_baseline_that_called_a_model_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "baseline_checks", passing_baselines)
    monkeypatch.setattr(
        module, "replay_forward_arms", calling_baselines(module.replay_forward_arms)
    )
    await assert_a_call_in_a_baseline_is_refused(tmp_path)


@pytest.mark.asyncio
async def test_mutation_baselines_that_do_not_check_their_calls_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "baseline_checks", passing_baselines)
    monkeypatch.setattr(
        module, "replay_forward_arms", calling_baselines(module.replay_forward_arms)
    )
    mutant = mutated(module.baselines_pass, "    if calls:\n", "    if False:\n", module)
    monkeypatch.setattr(module, "baselines_pass", mutant)
    with pytest.raises(AssertionError):
        await assert_a_call_in_a_baseline_is_refused(tmp_path)


@pytest.mark.asyncio
async def test_a_failed_baseline_check_writes_no_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(*args: object) -> tuple[CheckOutcome, ...]:
        return (CheckOutcome(check=Check.TIES, passed=False, detail="simulada"),)

    monkeypatch.setattr(module, "baseline_checks", failing)

    result = await module.baselines_pass(prepared(tmp_path), no_model_environment(tmp_path), ())

    assert result.code == 1
    assert "«atadura»" in result.lines[0]
    (directory,) = passes(tmp_path)
    assert forward_meta(directory).outcome is BatchOutcome.REFUSED
    assert not (directory / META_FILE).exists()


# ─────────────────────────────── Un lote inválido, repetido entero ────────────────────────────────


def as_another_decider(directory: Path) -> None:
    """La pasada corrió con otra configuración del decisor: el lote es inválido (apartado 5)."""
    meta = forward_meta(directory)
    changed = meta.model_copy(
        update={"decider": registered().model_copy(update={"temperature": 0.7})}
    )
    module.write_forward_meta(directory, changed)


async def assert_the_invalid_batch_is_repeated_whole(tmp_path: Path) -> None:
    """La primera pasada corre las tres evaluaciones y queda inválida. `--supersede` la marca
    sustituida y repite las tres: llama otra vez a las tres, aunque su caché tenga las respuestas,
    porque la repetición tiene la suya. El lote queda con los registros de la repetición, sin
    duplicados, y completo."""
    block = prepared(tmp_path)
    clock = TickingClock()
    assert (await run_pass(block, environment(tmp_path, FakeDecider(), clock=clock), ())).code == 0
    (first,) = passes(tmp_path)
    as_another_decider(first)

    again = FakeDecider()
    result = await run_pass(block, environment(tmp_path, again, clock=clock), (), supersede=True)

    found = "\n".join(result.lines)
    assert result.code == 0, found
    assert again.calls == 3
    earlier, later = passes(tmp_path)
    assert earlier == first
    mark = module.SupersededMark.model_validate_json((first / "superseded.json").read_text())
    assert mark.by == later.name
    assert "temperature" in mark.reasons[0]
    assert str(first / "superseded.json") in found
    whole = module.batch_records(tmp_path / "batches" / "2026-10-22")
    assert sorted(len(records) for records in whole.values()) == [3] * 5
    later_ids = {record.run_id for record in read_run(later).arms[0].records}
    assert {record.run_id for record in whole["solo"]} == later_ids
    assert "lote completo: sí" in found


@pytest.mark.asyncio
async def test_an_invalid_batch_is_repeated_whole_without_duplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "offline_checks", passing())
    await assert_the_invalid_batch_is_repeated_whole(tmp_path)


@pytest.mark.asyncio
async def test_mutation_a_batch_that_keeps_its_superseded_passes_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import crypto_agents.forward_record as record_module

    monkeypatch.setattr(module, "offline_checks", passing())
    mutant = mutated(
        record_module.batch_passes,
        "(include_superseded or superseded(path) is None)",
        "True",
        record_module,
    )
    monkeypatch.setattr(record_module, "batch_passes", mutant)
    monkeypatch.setattr(module, "batch_passes", mutant)
    with pytest.raises((AssertionError, module.DuplicateRecordError)):
        await assert_the_invalid_batch_is_repeated_whole(tmp_path)


@pytest.mark.asyncio
async def test_mutation_a_repetition_with_the_shared_cache_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "offline_checks", passing())
    monkeypatch.setattr(module, "pass_cache", lambda root, batch_dir, rerun=None: root)
    with pytest.raises(AssertionError):
        await assert_the_invalid_batch_is_repeated_whole(tmp_path)


class AnyReasons(module.SupersededMark):
    """La marca sin la exigencia de motivos: lo que deja a la negativa del comando sola."""

    reasons: tuple[str, ...] = ()


def without_the_reason_requirement(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutraliza la exigencia de motivos donde se escribe la marca (`forward`) y donde se lee
    (`forward_record.superseded`), para que la prueba de la negativa no dependa de ella. Va antes
    de construir cualquier mutante: `mutated` copia el espacio del módulo al compilar."""
    import crypto_agents.forward_record as record_module

    monkeypatch.setattr(module, "SupersededMark", AnyReasons)
    monkeypatch.setattr(record_module, "SupersededMark", AnyReasons)


async def assert_a_valid_batch_is_not_superseded(tmp_path: Path) -> None:
    """`batch --supersede` sobre un lote válido sale con 1, sin llamar a nadie y sin escribir
    ningún `superseded.json`."""
    block = prepared(tmp_path)
    clock = TickingClock()
    assert (
        await module.run_pass(block, environment(tmp_path, FakeDecider(), clock=clock), ())
    ).code == 0

    again = FakeDecider()
    result = await module.run_pass(
        block, environment(tmp_path, again, clock=clock), (), supersede=True
    )

    assert result.code == 1
    assert "no es inválido: no se sustituye" in result.lines[0]
    assert again.calls == 0
    assert not list((tmp_path / "batches").rglob("superseded.json"))


@pytest.mark.asyncio
async def test_a_valid_batch_is_not_superseded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "offline_checks", passing())
    without_the_reason_requirement(monkeypatch)
    await assert_a_valid_batch_is_not_superseded(tmp_path)


@pytest.mark.asyncio
async def test_mutation_superseding_a_valid_batch_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sin la negativa del comando, y con la marca neutralizada, el lote válido queda sustituido y
    se repite: la prueba de la negativa lo ve."""
    monkeypatch.setattr(module, "offline_checks", passing())
    without_the_reason_requirement(monkeypatch)
    mutant = mutated(
        module.run_pass,
        "if supersede and not summary.invalidating:",
        "if False:",
        module,
    )
    monkeypatch.setattr(module, "run_pass", mutant)
    with pytest.raises(AssertionError):
        await assert_a_valid_batch_is_not_superseded(tmp_path)


def assert_a_mark_needs_a_reason(mark: Callable[..., object]) -> None:
    """Una marca sin motivos no se puede construir: sin lote inválido no hay qué sustituir."""
    try:
        mark(at=READY, by="20261030T000000Z", reasons=())
    except ValidationError:
        return
    raise AssertionError("una marca de sustitución sin motivos se aceptó")


def test_a_superseded_mark_without_reasons_is_rejected() -> None:
    assert_a_mark_needs_a_reason(module.SupersededMark)
    assert module.SupersededMark(at=READY, by="x", reasons=("inválido",)).reasons == ("inválido",)


def test_mutation_a_mark_that_accepts_no_reasons_is_caught() -> None:
    import crypto_agents.forward_record as record_module

    mutant = mutated(
        record_module.SupersededMark,
        "reasons: tuple[str, ...] = Field(min_length=1)",
        "reasons: tuple[str, ...] = ()",
        record_module,
    )
    with pytest.raises(AssertionError):
        assert_a_mark_needs_a_reason(mutant)


@pytest.mark.asyncio
async def test_a_refused_repetition_supersedes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Si una comprobación falla, ninguna pasada queda sustituida: las marcas se escriben justo
    antes de la primera llamada."""
    block = prepared(tmp_path)
    clock = TickingClock()
    monkeypatch.setattr(module, "offline_checks", passing())
    assert (await run_pass(block, environment(tmp_path, FakeDecider(), clock=clock), ())).code == 0
    (first,) = passes(tmp_path)
    as_another_decider(first)
    monkeypatch.setattr(module, "offline_checks", passing(Check.TIES))

    again = FakeDecider()
    result = await run_pass(block, environment(tmp_path, again, clock=clock), (), supersede=True)

    assert result.code == 1
    assert "Ninguna pasada quedó sustituida" in result.lines[0]
    assert again.calls == 0
    assert not (first / "superseded.json").exists()


@pytest.mark.asyncio
async def test_an_invalid_batch_is_told_to_repeat_with_supersede(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "offline_checks", passing())
    block = prepared(tmp_path)
    clock = TickingClock()
    assert (await run_pass(block, environment(tmp_path, FakeDecider(), clock=clock), ())).code == 0
    as_another_decider(passes(tmp_path)[0])

    again = FakeDecider()
    result = await run_pass(block, environment(tmp_path, again, clock=clock), ())

    assert result.code == 3
    assert again.calls == 0
    assert "se repite entero con `--supersede`" in "\n".join(result.lines)


# ───────────────────────────── Lo que se versiona, y el cerrojo de los lotes ──────────────────────


def printed_files(lines: Sequence[str]) -> tuple[str, ...]:
    """La lista que el lote imprime bajo «Archivos que hay que versionar»."""
    text = "\n".join(lines)
    header = "Archivos que hay que versionar (este comando no hace commits):"
    assert text.count(header) == 1, text
    listed = text.split(header, 1)[1].strip().splitlines()
    return tuple(line.removeprefix("- ") for line in listed)


async def assert_the_batch_lists_every_pass(tmp_path: Path) -> None:
    """Una pasada cortada y la que completa el lote: la segunda lista los archivos de las dos, y
    lo que devuelve es lo que imprime. Lo mismo en una negativa y en un bloque vacío."""
    block = prepared(tmp_path)
    clock = TickingClock()
    cut = await run_pass(block, environment(tmp_path, FakeDecider(fail_on=[2]), clock=clock), ())
    assert cut.code == 3
    assert cut.files == printed_files(cut.lines)
    done = await run_pass(block, environment(tmp_path, FakeDecider(), clock=clock), ())
    assert done.code == 0
    assert done.files == printed_files(done.lines)
    first, second = passes(tmp_path)
    for directory in (first, second):
        for path in directory.iterdir():
            assert str(path) in done.files, f"{path} no está en la lista del lote completo"
    assert str(block.manifest_file) in done.files


@pytest.mark.asyncio
async def test_the_batch_lists_every_pass_and_returns_what_it_prints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "offline_checks", passing())
    await assert_the_batch_lists_every_pass(tmp_path)


@pytest.mark.asyncio
async def test_mutation_a_batch_that_lists_only_its_last_pass_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "offline_checks", passing())
    mutant = mutated(
        module.files_to_version, 'directory.parent.rglob("*")', "directory.iterdir()", module
    )
    monkeypatch.setattr(module, "files_to_version", mutant)
    with pytest.raises(AssertionError):
        await assert_the_batch_lists_every_pass(tmp_path)


@pytest.mark.asyncio
async def test_a_refusal_and_an_empty_block_return_what_they_print(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "offline_checks", passing(Check.TIES))
    refused = await run_pass(prepared(tmp_path), environment(tmp_path, FakeDecider()), ())
    assert refused.code == 1
    assert refused.files == printed_files(refused.lines)
    assert any(path.endswith(FORWARD_META_FILE) for path in refused.files)

    empty = prepared(tmp_path / "vacio")._replace(manifest=None, manifest_file=None, counts=(0,))
    result = await run_pass(empty, environment(tmp_path / "vacio", FakeDecider()), ())
    assert result.code == 0
    assert result.files == printed_files(result.lines)
    (recorded,) = result.files
    assert recorded.endswith(FORWARD_META_FILE)


def refusing_environment(tmp_path: Path, now: datetime) -> BatchEnvironment:
    refuse = Refusing()
    return BatchEnvironment(
        clock=lambda: now,
        settings=refuse,  # type: ignore[arg-type]
        backends=refuse,  # type: ignore[arg-type]
        catalog=refuse,  # type: ignore[arg-type]
        fetch_candles=refuse,  # type: ignore[arg-type]
        fetch_funding=refuse,  # type: ignore[arg-type]
        paths=paths_in(tmp_path),
    )


WRITING_COMMANDS = (
    (["batch", "--block", "2026-10-22"], READY),
    (["batch", "--block", "2026-10-22", "--supersede"], READY),
    (["baselines", "--block", "2026-10-22"], DEADLINE),
    (["auto"], DEADLINE),
)
"""Cada comando que escribe lotes, con un reloj en que su fecha se acepta: el cerrojo se toma
después de la negativa por fecha, así que solo una fecha aceptada llega a él."""

WRITING_IDS = ["batch", "batch --supersede", "baselines", "auto"]


def assert_a_held_lock_refuses(
    command: Sequence[str], now: datetime, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Con el cerrojo tomado por otro, el comando sale con 1 sin tocar nada: ni la red, ni el
    entorno, ni un archivo del estudio. Sin cerrojo, `batch` (con o sin `--supersede`) y
    `baselines` irían a la red, y `auto` también, con el bloque 2 pendiente."""
    env = refusing_environment(tmp_path, now)
    with module.batch_lock(env.paths.lock):
        code = main(command, env)
    output = capsys.readouterr()
    assert code == 1
    assert "otro lote está corriendo" in output.out + output.err
    assert [path.name for path in tmp_path.iterdir()] == [LOCK_NAME]


@pytest.mark.parametrize(("command", "now"), WRITING_COMMANDS, ids=WRITING_IDS)
def test_a_held_lock_refuses_every_command_that_writes_batches(
    command: Sequence[str], now: datetime, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert_a_held_lock_refuses(command, now, tmp_path, capsys)


@pytest.mark.parametrize(("command", "now"), WRITING_COMMANDS, ids=WRITING_IDS)
def test_mutation_a_command_without_the_lock_is_caught(
    command: Sequence[str],
    now: datetime,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    held = module.batch_lock
    monkeypatch.setattr(module, "batch_lock", lambda path: contextlib.nullcontext())
    with held(tmp_path / LOCK_NAME), pytest.raises(AssertionError):
        assert_a_held_lock_refuses(command, now, tmp_path, capsys)


DATE_REFUSALS = (
    (["batch", "--block", "2026-10-22"], DEADLINE),
    (["batch", "--block", "2026-10-22", "--supersede"], READY - MICRO),
    (["baselines", "--block", "2026-10-22"], DEADLINE - MICRO),
)
"""Fechas que cada comando niega: en el plazo, antes del cierre más 24 h, antes del plazo."""


def assert_a_date_refusal_takes_no_lock(
    command: Sequence[str], now: datetime, tmp_path: Path
) -> None:
    """Una fecha negada sale con 2 antes de tomar el cerrojo: no escribe nada, ni el cerrojo."""
    env = BatchEnvironment(clock=lambda: now, paths=paths_in(tmp_path))
    assert module.main(command, env) == 2
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(("command", "now"), DATE_REFUSALS, ids=WRITING_IDS[:3])
def test_a_date_refusal_takes_no_lock(
    command: Sequence[str], now: datetime, tmp_path: Path
) -> None:
    assert_a_date_refusal_takes_no_lock(command, now, tmp_path)


@pytest.mark.parametrize(("command", "now"), DATE_REFUSALS, ids=WRITING_IDS[:3])
def test_mutation_the_lock_taken_before_the_date_is_caught(
    command: Sequence[str], now: datetime, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """El mutante toma el cerrojo antes de mirar la fecha, como lo hacía `cc42962`."""
    check = "date_refusal" if command[0] == "batch" else "baselines_date_refusal"
    old = f"refusal = {check}(args.block, environment.clock())"
    new = f"batch_lock(environment.paths.lock).__enter__()\n            {old}"
    monkeypatch.setattr(module, "main", mutated(module.main, old, new, module))
    with pytest.raises(AssertionError):
        assert_a_date_refusal_takes_no_lock(command, now, tmp_path)


async def assert_an_empty_block_is_recorded_once_by_the_baselines(tmp_path: Path) -> None:
    """Las líneas base de un bloque sin activaciones lo registran vacío una sola vez, como el lote:
    así el temporizador lee en el disco que no le falta nada."""
    block = prepared(tmp_path)
    assert block.manifest_file is not None
    block.manifest_file.unlink()
    block = block._replace(manifest=None, manifest_file=None, counts=(0,))
    env = no_model_environment(tmp_path)

    first = await module.baselines_pass(block, env, ("baselines",))

    assert first.code == 0
    assert first.lines[0] == "El bloque 1 no tiene activaciones: no hay líneas base que completar."
    assert len(passes(tmp_path)) == 1, passes(tmp_path)
    directory = passes(tmp_path)[0]
    assert forward_meta(directory).outcome is BatchOutcome.EMPTY
    assert not (directory / META_FILE).exists()
    assert first.files == printed_files(first.lines)
    again = await module.baselines_pass(block, env, ("baselines",))
    assert again.code == 0
    assert passes(tmp_path) == [directory]


@pytest.mark.asyncio
async def test_an_empty_block_is_recorded_once_by_the_baselines(tmp_path: Path) -> None:
    await assert_an_empty_block_is_recorded_once_by_the_baselines(tmp_path)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (
            "write_forward_meta(directory, meta)\n            recorded = directory",
            "recorded = directory",
        ),
        ("recorded = _empty_recorded(batch_dir)", "recorded = None"),
    ],
    ids=["no lo registra", "lo registra cada vez"],
)
@pytest.mark.asyncio
async def test_mutation_an_empty_block_misrecorded_by_the_baselines_is_caught(
    old: str, new: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mutant = mutated(module.baselines_pass, old, new, module)
    monkeypatch.setattr(module, "baselines_pass", mutant)
    with pytest.raises(AssertionError):
        await assert_an_empty_block_is_recorded_once_by_the_baselines(tmp_path)
