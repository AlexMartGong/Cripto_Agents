"""La comprobación semanal del estudio: `python -m crypto_agents.forward_watch` y su workflow.

Lo que pide el bloque, cada parte con su mutación:

- **Un lote que va tarde** falla: pendiente con la ventana abierta hace más de 24 h, con el borde
  exacto, y también el del bloque 1, que corre el operador.
- **Lo que espera al operador** falla aunque esté dentro de las 24 h: retenido, inválido o con
  duplicados.
- **Un bloque vencido** con el lote incompleto y sus líneas base sin completar falla; con ellas
  completas, pasa.
- **Lo demás pasa**: un lote completo o un bloque vacío; el estudio terminado, cerrado o agotado,
  que además se dice. Con el estudio cerrado, un bloque anterior al cierre que falla sigue
  fallando.
- **`--now`** sustituye al reloj; vacío es el reloj; sin zona o con otra zona se rechaza.
- **El resumen** va a la salida y al final del archivo de `--summary`; un estudio ilegible sale con
  2 y lo dice.
- **El workflow**: los sábados a las 15:17 UTC y `workflow_dispatch` con `now`, solo lectura, sin
  secretos, cada acción fijada por su sha, `master`, y la entrada por el entorno.

Los estudios se arman con las funciones del lote de verdad, como en `tests/test_forward_auto.py`.
"""

from __future__ import annotations

import asyncio
import re
import shutil
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest
import yaml  # type: ignore[import-untyped]

import crypto_agents.forward as forward
import crypto_agents.forward_watch as module
from crypto_agents.audit import AuditError
from crypto_agents.forward import (
    block_dates,
    revised_result,
    run_pass,
    study_paths,
)
from crypto_agents.forward_analysis import study_paths as analysis_paths
from crypto_agents.forward_auto import StudyState
from crypto_agents.forward_data import FORWARD_DIR, STAGE1_DIR
from crypto_agents.forward_plan import MAX_BLOCKS
from crypto_agents.forward_record import FIRST_BLOCK, STEP
from crypto_agents.forward_watch import GRACE, Watch, WatchEnvironment, main, parse_now, watch
from tests.test_forward import (
    MICRO,
    ROOT,
    FakeCatalog,
    FakeDecider,
    TickingClock,
    as_another_decider,
    environment,
    no_model_environment,
    passing,
    passing_baselines,
    paths_in,
)
from tests.test_forward_auto import (
    REVISED_BTC,
    Dying,
    Killed,
    empty_block,
    passes_of,
    prepared_for,
    run_block,
    touch_manifests,
)
from tests.test_net_outcomes import mutated

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping
    from pathlib import Path

    from crypto_agents.forward_analysis import StudyPaths

STAGE1 = ROOT / STAGE1_DIR
ONE = block_dates(FIRST_BLOCK)
TWO = FIRST_BLOCK + STEP
WITHIN = ONE.ready + timedelta(hours=1)
"""Una hora después de abrirse el lote del bloque 1: el temporizador está a tiempo."""

EDGE = ONE.ready + GRACE
"""Justo 24 h después: todavía a tiempo. Falla lo que pasa de aquí."""

LATE = EDGE + MICRO
"""Un microsegundo más tarde. El lote del bloque 2 todavía no se abre."""

EXPIRED_AT = ONE.deadline
"""El plazo del bloque 1, vencido justo ahora. El bloque 2 está registrado vacío en esos estudios,
y el lote del 3 todavía no se abre."""


def study(root: Path) -> StudyPaths:
    return study_paths(paths_in(root))


def environment_of(root: Path, now: datetime) -> WatchEnvironment:
    return WatchEnvironment(clock=lambda: now, paths=study(root), stage1=STAGE1)


def no_clock() -> datetime:
    raise AssertionError("se leyó el reloj con --now puesto")


# ─────────────────────────────────────────── Los estudios ─────────────────────────────────────────


async def record_empty(root: Path, start: datetime, clock: TickingClock) -> None:
    """El bloque sin activaciones, registrado como vacío por su lote."""
    env = environment(root, FakeDecider(), clock=clock)
    assert (await run_pass(empty_block(root, start), env, ())).code == 0


async def cut_short(root: Path, clock: TickingClock) -> None:
    """La pasada del bloque 1 que muere a mitad: el lote queda incompleto, sin líneas base."""
    with pytest.raises(Killed):
        await run_block(root, FIRST_BLOCK, Dying(), clock)


async def build_studies(root: Path) -> None:
    clock = TickingClock(ONE.ready)
    complete = await run_block(root / "completo", FIRST_BLOCK, FakeDecider(), clock)
    assert complete.code == 0
    short = await run_block(root / "incompleto", FIRST_BLOCK, FakeDecider(fail_on=[1]), clock)
    assert short.code == 3
    assert (await run_block(root / "inválido", FIRST_BLOCK, FakeDecider(), clock)).code == 0
    as_another_decider(passes_of(root / "inválido", FIRST_BLOCK)[-1])
    assert (await run_block(root / "con duplicados", FIRST_BLOCK, FakeDecider(), clock)).code == 0
    (first,) = passes_of(root / "con duplicados", FIRST_BLOCK)
    shutil.copytree(first, first.parent / "29991231T000000Z")
    env = environment(root / "retenido", FakeDecider(), clock=clock)
    assert revised_result(ONE, env, REVISED_BTC).code == 5
    await record_empty(root / "vacío", FIRST_BLOCK, clock)
    for name in ("cortado", "cortado y completado"):
        await cut_short(root / name, clock)
        await record_empty(root / name, TWO, clock)
    completed = root / "cortado y completado"
    after = no_model_environment(completed)._replace(clock=TickingClock(ONE.deadline))
    done = await forward.baselines_pass(prepared_for(completed, FIRST_BLOCK), after, ())
    assert done.code == 0


@pytest.fixture(scope="module")
def studies(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Un estudio por estado del bloque 1, armados una vez; las pruebas solo los leen."""
    root = tmp_path_factory.mktemp("vigilancia")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(forward, "offline_checks", passing())
        patch.setattr(forward, "baseline_checks", passing_baselines)
        asyncio.run(build_studies(root))
    yield root


VERDICTS: Mapping[tuple[str, datetime], tuple[bool, str]] = {
    ("sin nada", WITHIN): (False, "a tiempo"),
    ("sin nada", EDGE): (False, "a tiempo"),
    ("sin nada", LATE): (True, "tarde"),
    ("incompleto", WITHIN): (False, "a tiempo"),
    ("incompleto", LATE): (True, "tarde"),
    ("completo", LATE): (False, ""),
    ("vacío", LATE): (False, ""),
    ("inválido", WITHIN): (True, "espera"),
    ("con duplicados", WITHIN): (True, "espera"),
    ("retenido", WITHIN): (True, "espera"),
    ("cortado", EXPIRED_AT): (True, "vencido"),
    ("cortado y completado", EXPIRED_AT): (False, ""),
}
"""Cada estudio en un instante: si la comprobación falla, y dónde aparece el bloque 1."""


def where(found: Watch) -> str:
    """En qué lista está el bloque 1, si está en alguna."""
    lists = {
        "tarde": found.late,
        "a tiempo": found.on_time,
        "espera": found.state.blocked,
        "vencido": found.state.expired,
    }
    named = [name for name, readings in lists.items() if FIRST_BLOCK in starts(readings)]
    assert len(named) <= 1, named
    return named[0] if named else ""


def starts(readings: tuple[Any, ...]) -> list[datetime]:
    return [reading.dates.start for reading in readings]


def assert_the_verdicts(root: Path) -> None:
    for (name, now), (fails, place) in VERDICTS.items():
        found = module.watch(study(root / name), STAGE1, now)
        assert (found.failed, where(found)) == (fails, place), (name, now)


def test_each_state_of_block_one_is_judged(studies: Path) -> None:
    assert_the_verdicts(studies)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("now - reading.dates.ready > GRACE", "now - reading.dates.ready >= GRACE"),
        ("study_state(paths, stage1, now, first=FIRST_BLOCK)", "study_state(paths, stage1, now)"),
    ],
    ids=["el borde de las 24 h", "sin el bloque 1"],
)
def test_mutation_another_watch_is_caught(
    old: str, new: str, studies: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "watch", mutated(watch, old, new, module))
    with pytest.raises(AssertionError):
        assert_the_verdicts(studies)


@pytest.mark.parametrize("grace", [timedelta(0), timedelta(hours=23), timedelta(hours=48)])
def test_mutation_another_grace_is_caught(
    grace: timedelta, studies: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert timedelta(hours=24) == GRACE
    monkeypatch.setattr(module, "GRACE", grace)
    with pytest.raises(AssertionError):
        assert_the_verdicts(studies)


def mutate_failed(monkeypatch: pytest.MonkeyPatch, old: str, new: str) -> None:
    """`Watch.failed` con un fragmento cambiado. El código que se recompila lleva `@property`, así
    que el mutante ya es la propiedad."""
    mutant: object = mutated(vars(Watch)["failed"].fget, old, new, module)
    assert isinstance(mutant, property)
    monkeypatch.setattr(Watch, "failed", mutant)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (
            "self.late or self.state.blocked or self.state.expired",
            "self.state.blocked or self.state.expired",
        ),
        (
            "self.late or self.state.blocked or self.state.expired",
            "self.late or self.state.expired",
        ),
        (
            "self.late or self.state.blocked or self.state.expired",
            "self.late or self.state.blocked",
        ),
    ],
    ids=["tarde pasa", "espera pasa", "vencido pasa"],
)
def test_mutation_a_rule_left_out_is_caught(
    old: str, new: str, studies: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mutate_failed(monkeypatch, old, new)
    with pytest.raises(AssertionError):
        assert_the_verdicts(studies)


# ────────────────────────────── Terminado, cerrado y agotado ──────────────────────────────────────


def stub_state(**fields: object) -> Callable[..., StudyState]:
    """`study_state` sustituido por un estado fijo: para lo que no depende de cómo se lee."""
    values: dict[str, object] = {
        "now": WITHIN,
        "pending": (),
        "blocked": (),
        "expired": (),
        "terminated": None,
        "closed_at": None,
        "upcoming": None,
        "non_empty": 0,
        "exhausted": False,
    } | fields

    def state(*args: object, **kwargs: object) -> StudyState:
        return StudyState(**values)  # type: ignore[arg-type]

    return state


def summary_of(found: Watch) -> str:
    return "\n".join(module.render(found))


def assert_an_ended_study_passes_and_says_so(tmp_path: Path) -> None:
    """Un estudio terminado no deja nada que vigilar, aunque el bloque 1 no tenga lote."""
    terminated = tmp_path / "terminado"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(forward, "offline_checks", passing())
        env = environment(terminated, FakeDecider(), FakeCatalog({"otro"}), clock=TickingClock())
        assert asyncio.run(run_pass(prepared_for(terminated, FIRST_BLOCK), env, ())).code == 4
    found = module.watch(study(terminated), STAGE1, LATE)
    assert not found.failed
    assert "- El estudio terminó en el bloque 1: " in summary_of(found)
    assert "**pasa**" in summary_of(found)


def test_an_ended_study_passes_and_says_so(tmp_path: Path) -> None:
    assert_an_ended_study_passes_and_says_so(tmp_path)


def test_mutation_an_ended_study_left_unsaid_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = "    if state.terminated is not None:\n"
    monkeypatch.setattr(module, "render", mutated(module.render, old, "    if False:\n", module))
    with pytest.raises(AssertionError):
        assert_an_ended_study_passes_and_says_so(tmp_path)


def assert_a_closed_or_exhausted_study_passes_and_says_so(patch: pytest.MonkeyPatch) -> None:
    patch.setattr(module, "study_state", stub_state(closed_at=50, non_empty=50))
    closed = module.watch(study(ROOT), STAGE1, WITHIN)
    assert not closed.failed
    assert "- El estudio cerró en el bloque 50: " in summary_of(closed)
    patch.setattr(module, "study_state", stub_state(exhausted=True, non_empty=48))
    exhausted = module.watch(study(ROOT), STAGE1, WITHIN)
    assert not exhausted.failed
    assert f"- El bloque {MAX_BLOCKS} terminó sin 50 bloques no vacíos: " in summary_of(exhausted)


def test_a_closed_or_exhausted_study_passes_and_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    assert_a_closed_or_exhausted_study_passes_and_says_so(monkeypatch)


@pytest.mark.parametrize(
    "old",
    ["    if state.closed_at is not None:\n", "    if state.exhausted:\n"],
    ids=["cerrado", "agotado"],
)
def test_mutation_a_closing_left_unsaid_is_caught(
    old: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "render", mutated(module.render, old, "    if False:\n", module))
    with pytest.raises(AssertionError):
        assert_a_closed_or_exhausted_study_passes_and_says_so(monkeypatch)


def test_mutation_a_closing_that_fails_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    old = "return bool(self.late or self.state.blocked or self.state.expired)"
    new = f"{old.removesuffix(')')} or self.state.closed_at or self.state.exhausted)"
    mutate_failed(monkeypatch, old, new)
    with pytest.raises(AssertionError):
        assert_a_closed_or_exhausted_study_passes_and_says_so(monkeypatch)


def assert_a_block_before_the_closing_still_fails(tmp_path: Path) -> None:
    """Con 50 manifiestos y ningún lote, el estudio cerró en el 50, y los 50 bloques vencieron
    con el lote incompleto y sus líneas base sin completar: falla, y dice que cerró."""
    touch_manifests(tmp_path, 50)
    now = block_dates(FIRST_BLOCK + 49 * STEP).deadline
    found = module.watch(study(tmp_path), STAGE1, now)
    assert found.state.closed_at == 50
    assert [reading.dates.number for reading in found.state.expired] == list(range(1, 51))
    assert found.failed
    assert "- El estudio cerró en el bloque 50: " in summary_of(found)


def test_a_block_before_the_closing_still_fails(tmp_path: Path) -> None:
    assert_a_block_before_the_closing_still_fails(tmp_path)


def test_mutation_a_closed_study_that_always_passes_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = "return bool(self.late or self.state.blocked or self.state.expired)"
    new = f"return self.state.closed_at is None and {old.removeprefix('return ')}"
    mutate_failed(monkeypatch, old, new)
    with pytest.raises(AssertionError):
        assert_a_block_before_the_closing_still_fails(tmp_path)


# ─────────────────────────────────────────── `--now` ──────────────────────────────────────────────


def exit_code(argv: list[str], env: WatchEnvironment) -> int:
    """El código de `main`, también el 2 con que argparse rechaza un argumento."""
    try:
        return module.main(argv, env)
    except SystemExit as error:
        assert isinstance(error.code, int)
        return error.code


def assert_now_replaces_the_clock(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Con `--now`, ese instante y no el reloj; vacío, el reloj. Un instante sin zona o en otra
    zona no es un instante UTC: uso incorrecto, 2."""
    empty = root / "sin nada"
    assert exit_code(["--now", "2026-10-31T00:00:00.000001Z"], environment_of(empty, WITHIN)) == 1
    assert exit_code(["--now", "2026-10-31T00:00:00+00:00"], environment_of(empty, LATE)) == 0
    assert exit_code(["--now", ""], environment_of(empty, LATE)) == 1
    assert exit_code(["--now", "  "], environment_of(empty, WITHIN)) == 0
    assert exit_code([], environment_of(empty, LATE)) == 1
    assert exit_code(["--now", "2026-10-31T00:00:01"], environment_of(empty, WITHIN)) == 2
    assert exit_code(["--now", "2026-10-30T18:00:01-06:00"], environment_of(empty, WITHIN)) == 2
    without_clock = WatchEnvironment(no_clock, study(empty), STAGE1)
    assert exit_code(["--now", "2026-10-31T00:00:01Z"], without_clock) == 1
    capsys.readouterr()


def test_now_replaces_the_clock(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "sin nada").mkdir()
    assert parse_now("2026-11-07T15:17:00Z") == datetime(2026, 11, 7, 15, 17, tzinfo=UTC)
    assert parse_now("") is None
    assert_now_replaces_the_clock(tmp_path, capsys)


@pytest.mark.parametrize(
    ("function", "old", "new"),
    [
        (
            "main",
            "now: datetime = args.now if args.now is not None else environment.clock()",
            "now: datetime = environment.clock()",
        ),
        ("parse_now", "if instant.utcoffset() != timedelta(0):", "if instant.tzinfo is None:"),
        ("parse_now", "if instant.utcoffset() != timedelta(0):", "if False:"),
        ("parse_now", "if not text:\n        return None", "if False:\n        return None"),
    ],
    ids=["el reloj gana", "otra zona vale", "sin zona vale", "vacío no es el reloj"],
)
def test_mutation_another_now_is_caught(
    function: str,
    old: str,
    new: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "sin nada").mkdir()
    monkeypatch.setattr(module, function, mutated(getattr(module, function), old, new, module))
    with pytest.raises(AssertionError):
        assert_now_replaces_the_clock(tmp_path, capsys)


# ───────────────────────────────── El resumen y lo ilegible ───────────────────────────────────────


def assert_the_summary_is_appended(
    studies: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """El resumen del job es el de la salida, añadido al final de lo que ya hubiera."""
    summary = tmp_path / "resumen.md"
    summary.write_text("lo de antes\n", encoding="utf-8")
    capsys.readouterr()
    code = main(["--summary", str(summary)], environment_of(studies / "inválido", WITHIN))
    out = capsys.readouterr().out
    assert code == 1
    assert summary.read_text(encoding="utf-8") == "lo de antes\n" + out
    assert "- Resultado: **falla**: 1 bloque necesita atención." in out
    assert "- bloque 1 (2026-10-22): inválido. Espera al operador; vence el " in out


def test_the_summary_is_appended(
    studies: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert_the_summary_is_appended(studies, tmp_path, capsys)


def test_mutation_a_summary_that_overwrites_is_caught(
    studies: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    old = 'with summary.open("a", encoding="utf-8") as handle:'
    new = 'with summary.open("w", encoding="utf-8") as handle:'
    monkeypatch.setattr(module, "_emit", mutated(module._emit, old, new, module))
    with pytest.raises(AssertionError):
        assert_the_summary_is_appended(studies, tmp_path, capsys)


def assert_an_unreadable_study_fails_with_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Media línea en un journal: el estudio no se puede leer, sale con 2 y lo dice."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(forward, "offline_checks", passing())
        asyncio.run(cut_short(tmp_path, TickingClock(ONE.ready)))
    with (passes_of(tmp_path, FIRST_BLOCK)[-1] / "solo.jsonl").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write('{"run_id": "')
    summary = tmp_path / "resumen.md"
    capsys.readouterr()
    assert main(["--summary", str(summary)], environment_of(tmp_path, WITHIN)) == 2
    assert "- Resultado: **falla**: el estudio no se pudo leer: " in summary.read_text("utf-8")


def test_an_unreadable_study_fails_with_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert_an_unreadable_study_fails_with_two(tmp_path, capsys)


def test_mutation_an_unreadable_study_as_a_traceback_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(module, "READ_ERRORS", (OSError,))
    with pytest.raises(AuditError):
        assert_an_unreadable_study_fails_with_two(tmp_path, capsys)


def test_today_the_study_in_the_repository_passes(capsys: pytest.CaptureFixture[str]) -> None:
    """Sobre `data/forward/` de este checkout, antes de que se abra el lote del bloque 1."""
    paths = analysis_paths(ROOT / FORWARD_DIR)
    now = datetime(2026, 10, 29, 15, 17, tzinfo=UTC)
    assert main([], WatchEnvironment(lambda: now, paths, STAGE1)) == 0
    out = capsys.readouterr().out
    assert "- Resultado: **pasa**" in out
    assert (
        "El siguiente: el lote del bloque 1 (2026-10-22) se abre el 2026-10-30T00:00:00+00:00."
        in out
    )


# ─────────────────────────────────────────── El workflow ──────────────────────────────────────────

WORKFLOW = ROOT / ".github" / "workflows" / "forward-watch.yml"
PINNED = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+@[0-9a-f]{40}")


def steps(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [step for job in data["jobs"].values() for step in job["steps"]]


def assert_the_workflow(text: str) -> None:
    """Los sábados a las 15:17 UTC y a mano con `now`, en `master`, solo lectura, sin secretos,
    cada acción por su sha, y la entrada por el entorno: nunca escrita dentro de la orden."""
    data = yaml.safe_load(text)
    triggers = data[True]  # YAML 1.1 lee `on` como verdadero
    assert triggers["schedule"] == [{"cron": "17 15 * * 6"}]
    inputs = triggers["workflow_dispatch"]["inputs"]
    assert set(inputs) == {"now"}
    assert inputs["now"]["required"] is False
    assert inputs["now"]["type"] == "string"
    assert data["permissions"] == {"contents": "read"}
    assert all("permissions" not in job for job in data["jobs"].values())
    assert "secrets" not in text
    used = [step["uses"] for step in steps(data) if "uses" in step]
    assert used
    assert all(PINNED.fullmatch(uses) for uses in used), used
    (checkout,) = [
        step for step in steps(data) if step.get("uses", "").startswith("actions/checkout@")
    ]
    assert checkout["with"] == {"ref": "master", "persist-credentials": False}
    runs = [step for step in steps(data) if "run" in step]
    assert all("${{" not in step["run"] for step in runs)
    (check,) = [step for step in runs if "python -m crypto_agents.forward_watch" in step["run"]]
    assert '--now "$NOW"' in check["run"]
    assert '--summary "$GITHUB_STEP_SUMMARY"' in check["run"]
    assert check["env"] == {"NOW": "${{ inputs.now }}"}


def test_the_workflow_runs_the_watch_read_only_on_saturdays() -> None:
    assert_the_workflow(WORKFLOW.read_text("utf-8"))


def unpinned(text: str) -> str:
    """La primera acción por su etiqueta en vez de por su sha."""
    return re.sub(r"@[0-9a-f]{40}", "@v5", text, count=1)


@pytest.mark.parametrize(
    "change",
    [
        lambda text: text.replace('cron: "17 15 * * 6"', 'cron: "17 15 * * 0"'),
        lambda text: text.replace('cron: "17 15 * * 6"', 'cron: "17 3 * * 6"'),
        lambda text: text.replace("contents: read", "contents: write"),
        lambda text: text.replace("NOW: ${{ inputs.now }}", "NOW: ${{ secrets.NOW }}"),
        unpinned,
        lambda text: text.replace('--now "$NOW" ', ""),
        lambda text: text.replace('--now "$NOW"', '--now "${{ inputs.now }}"'),
        lambda text: text.replace("ref: master", "ref: ${{ github.ref }}"),
        lambda text: text.replace("persist-credentials: false", "persist-credentials: true"),
        lambda text: text.replace('--summary "$GITHUB_STEP_SUMMARY"', ""),
        lambda text: text.replace("required: false", "required: true"),
    ],
    ids=[
        "domingo",
        "otra hora",
        "escritura",
        "un secreto",
        "una acción por etiqueta",
        "sin --now",
        "la entrada dentro de la orden",
        "otra rama",
        "credenciales en el checkout",
        "sin resumen del job",
        "now obligatorio",
    ],
)
def test_mutation_another_workflow_is_caught(change: Callable[[str], str]) -> None:
    text = WORKFLOW.read_text("utf-8")
    mutant = change(text)
    assert mutant != text
    with pytest.raises(AssertionError):
        assert_the_workflow(mutant)
