"""Los lotes que corre el temporizador: `python -m crypto_agents.forward auto`.

Lo que el bloque de la automatización pide, cada parte con su mutación:

- **Los bloques pendientes**, con sus bordes: el cierre más 24 h y el plazo, el bloque 1 que no es
  del temporizador, los lotes completos y vacíos, el estudio cerrado y el terminado.
- **La tabla de códigos**, caso por caso, y el aviso del plazo.
- **`auto` llama al lote solo con su inicio, su entorno y su argv**: nunca acepta velas ni
  sustituye pasadas. Que nunca llama al análisis lo fija `tests/test_architecture.py`.
- **Idempotencia y `--dry-run`**: sin nada pendiente no hace nada, y el ensayo no corre ni
  escribe.
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

import crypto_agents.forward as forward
import crypto_agents.forward_auto as module
from crypto_agents.audit import META_FILE, read_meta
from crypto_agents.forward import (
    BatchMeta,
    BatchOutcome,
    BatchResult,
    ForwardPaths,
    PreparedBlock,
    base_meta,
    block_dates,
    main,
    registered_decider,
    run_pass,
    study_paths,
    write_forward_meta,
)
from crypto_agents.forward_analysis import pass_metas
from crypto_agents.forward_auto import (
    ACTIONS,
    AUTO_FIRST_BLOCK,
    UPSTREAM_FILE,
    WARN_BEFORE,
    Action,
    AutoReport,
    BlockStatus,
    action_for,
    block_status,
    run_auto,
    study_state,
)
from crypto_agents.forward_data import (
    BATCH_DIR,
    FORWARD_FUNDING_DIR,
    FORWARD_HISTORY_DIR,
    MANIFEST_DIR,
    REVISIONS_FILE,
    STAGE1_DIR,
    manifest_path,
)
from crypto_agents.forward_record import FIRST_BLOCK, STEP
from crypto_agents.forward_render import REGISTERED_PATH
from crypto_agents.selection import (
    DEFAULT_MANIFEST,
    HISTORY_DIR,
    load_manifest,
    load_selection_histories,
    write_manifest,
)
from tests.test_forward import (
    DOCS,
    LOCK_NAME,
    MICRO,
    ROOT,
    FakeDecider,
    TickingClock,
    as_another_decider,
    environment,
    passing,
    paths_in,
    refusing_environment,
)
from tests.test_net_outcomes import mutated

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from crypto_agents.forward_analysis import StudyPaths

STAGE1 = ROOT / STAGE1_DIR
BLOCK2 = AUTO_FIRST_BLOCK
BLOCK3 = BLOCK2 + STEP
BOTH = block_dates(BLOCK3).ready
"""Con este reloj están pendientes los bloques 2 y 3, y a ninguno le quedan 3 días o menos."""

TODAY = datetime(2026, 10, 25, 20, tzinfo=UTC)
"""El día en que se escribió el temporizador: no hay nada pendiente."""


def study(tmp_path: Path) -> StudyPaths:
    return study_paths(paths_in(tmp_path))


def registered() -> forward.DeciderConfig:
    return registered_decider(read_meta(STAGE1))


def status_of(tmp_path: Path, start: datetime) -> BlockStatus:
    paths = study(tmp_path)
    return block_status(paths, start, registered, pass_metas(paths.batches)).status


def prepared_for(tmp_path: Path, start: datetime, count: int = 3) -> PreparedBlock:
    """El bloque que empieza en `start`, con las `count` primeras entradas de la etapa 1."""
    reference = load_manifest(ROOT / DEFAULT_MANIFEST)
    manifest = reference.model_copy(
        update={"entries": reference.entries[:count], "target": count, "strata": 1}
    )
    path = manifest_path(start, tmp_path / "manifests")
    write_manifest(path, manifest)
    return PreparedBlock(
        block=block_dates(start),
        history=(),
        funding=(),
        histories=load_selection_histories(reference, ROOT / HISTORY_DIR),
        counts=(count,),
        manifest=manifest,
        manifest_file=path,
    )


def passes_of(tmp_path: Path, start: datetime) -> list[Path]:
    directory = tmp_path / "batches" / f"{start:%Y-%m-%d}"
    return sorted(directory.iterdir()) if directory.is_dir() else []


async def run_block(
    tmp_path: Path, start: datetime, decider: FakeDecider, clock: TickingClock
) -> BatchResult:
    env = environment(tmp_path, decider, clock=clock)
    return await run_pass(prepared_for(tmp_path, start), env, ())


class Scripted:
    """El lote simulado: devuelve el código que le toca a cada bloque y apunta cuáles corrió."""

    def __init__(self, codes: Mapping[datetime, int]) -> None:
        self.codes = codes
        self.starts: list[datetime] = []

    async def __call__(self, start: datetime) -> BatchResult:
        self.starts.append(start)
        code = self.codes[start]
        return BatchResult(code, (f"El lote salió con {code}.",), ())


class NeverCalled:
    async def __call__(self, start: datetime) -> BatchResult:
        raise AssertionError(f"el temporizador corrió el lote de {start}")


# ─────────────────────────────────────── Los bloques pendientes ───────────────────────────────────


def test_the_timer_starts_at_block_two() -> None:
    assert datetime(2026, 10, 29, tzinfo=UTC) == AUTO_FIRST_BLOCK
    assert block_dates(AUTO_FIRST_BLOCK).number == 2
    assert block_dates(AUTO_FIRST_BLOCK).ready == datetime(2026, 11, 6, tzinfo=UTC)
    assert timedelta(days=3) == WARN_BEFORE


def pending_starts(tmp_path: Path, now: datetime) -> list[datetime]:
    state = module.study_state(study(tmp_path), STAGE1, now)
    return [reading.dates.start for reading in state.pending]


def assert_the_edges(tmp_path: Path) -> None:
    """Desde el cierre más 24 h exacto, y hasta justo antes del plazo. El bloque 1 nunca."""
    two = block_dates(BLOCK2)
    assert pending_starts(tmp_path, two.ready - MICRO) == []
    assert pending_starts(tmp_path, two.ready) == [BLOCK2]
    assert pending_starts(tmp_path, two.deadline - MICRO) == [BLOCK2, BLOCK3]
    assert pending_starts(tmp_path, two.deadline) == [BLOCK3]
    one = block_dates(FIRST_BLOCK)
    assert pending_starts(tmp_path, one.ready) == []
    assert pending_starts(tmp_path, one.deadline - MICRO) == [BLOCK2]


def test_the_edges_of_a_pending_block(tmp_path: Path) -> None:
    assert_the_edges(tmp_path)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("if now < dates.ready:", "if now <= dates.ready:"),
        ("if now >= dates.deadline:", "if now > dates.deadline:"),
        ("if start < AUTO_FIRST_BLOCK:", "if start < FIRST_BLOCK:"),
    ],
    ids=["el cierre más 24 h", "el plazo", "el bloque 1"],
)
def test_mutation_an_edge_moved_is_caught(
    old: str, new: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "study_state", mutated(study_state, old, new, module))
    with pytest.raises(AssertionError):
        assert_the_edges(tmp_path)


def test_before_block_two_opens_it_names_the_next_one(tmp_path: Path) -> None:
    state = study_state(study(tmp_path), STAGE1, TODAY)
    assert state.pending == ()
    assert state.upcoming == block_dates(BLOCK2)


@pytest.mark.asyncio
async def test_each_state_of_a_block_is_read_from_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sin preparar, incompleto tras una pasada cortada, completo tras reanudarla, e inválido. Solo
    los dos primeros se corren."""
    monkeypatch.setattr(forward, "offline_checks", passing())
    clock = TickingClock()
    assert status_of(tmp_path, BLOCK2) is BlockStatus.NOT_PREPARED
    assert pending_starts(tmp_path, BOTH) == [BLOCK2, BLOCK3]

    assert (await run_block(tmp_path, BLOCK2, FakeDecider(fail_on=[1]), clock)).code == 3
    assert status_of(tmp_path, BLOCK2) is BlockStatus.INCOMPLETE
    assert pending_starts(tmp_path, BOTH) == [BLOCK2, BLOCK3]

    assert (await run_block(tmp_path, BLOCK2, FakeDecider(), clock)).code == 0
    assert status_of(tmp_path, BLOCK2) is BlockStatus.COMPLETE
    assert pending_starts(tmp_path, BOTH) == [BLOCK3]

    as_another_decider(passes_of(tmp_path, BLOCK2)[-1])
    assert status_of(tmp_path, BLOCK2) is BlockStatus.INVALID
    state = study_state(study(tmp_path), STAGE1, BOTH)
    assert [reading.dates.start for reading in state.pending] == [BLOCK3]
    assert [reading.dates.start for reading in state.blocked] == [BLOCK2]


@pytest.mark.asyncio
async def test_mutation_a_complete_batch_read_as_pending_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(forward, "offline_checks", passing())
    assert (await run_block(tmp_path, BLOCK2, FakeDecider(), TickingClock())).code == 0
    assert pending_starts(tmp_path, BOTH) == [BLOCK3]
    mutant = mutated(
        block_status, "status = BlockStatus.COMPLETE", "status = BlockStatus.INCOMPLETE", module
    )
    monkeypatch.setattr(module, "block_status", mutant)
    assert pending_starts(tmp_path, BOTH) == [BLOCK2, BLOCK3]


@pytest.mark.asyncio
async def test_an_empty_block_is_not_pending_once_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(forward, "offline_checks", passing())
    empty = prepared_for(tmp_path, BLOCK2)
    manifest_path(BLOCK2, tmp_path / "manifests").unlink()
    empty = empty._replace(manifest=None, manifest_file=None, counts=(0,))
    result = await run_pass(empty, environment(tmp_path, FakeDecider()), ())
    assert result.code == 0
    assert status_of(tmp_path, BLOCK2) is BlockStatus.EMPTY
    assert pending_starts(tmp_path, BOTH) == [BLOCK3]


@pytest.mark.asyncio
async def test_a_duplicated_batch_is_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(forward, "offline_checks", passing())
    assert (await run_block(tmp_path, BLOCK2, FakeDecider(), TickingClock())).code == 0
    (first,) = passes_of(tmp_path, BLOCK2)
    shutil.copytree(first, first.parent / "29991231T000000Z")
    assert status_of(tmp_path, BLOCK2) is BlockStatus.DUPLICATE
    state = study_state(study(tmp_path), STAGE1, BOTH)
    assert [reading.dates.start for reading in state.blocked] == [BLOCK2]


def touch_manifests(tmp_path: Path, blocks: int) -> None:
    """Un manifiesto (vacío de contenido: solo cuenta que exista) por cada bloque del 1 al
    `blocks`. Ninguno queda dentro de su plazo con el reloj de `assert_the_closing`."""
    for position in range(blocks):
        path = manifest_path(FIRST_BLOCK + position * STEP, tmp_path / "manifests")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")


def assert_the_closing(tmp_path: Path) -> None:
    """Con 50 bloques no vacíos, el 51 ya no es del estudio. Con 49, sí."""
    fifty = block_dates(FIRST_BLOCK + 49 * STEP)
    fifty_one = FIRST_BLOCK + 50 * STEP
    now = fifty.deadline
    touch_manifests(tmp_path / "abierto", 49)
    assert pending_starts(tmp_path / "abierto", now) == [fifty_one]
    touch_manifests(tmp_path / "cerrado", 50)
    state = module.study_state(study(tmp_path / "cerrado"), STAGE1, now)
    assert state.pending == ()
    assert state.closed_at == 50


def test_a_block_after_the_study_closed_is_not_pending(tmp_path: Path) -> None:
    assert_the_closing(tmp_path)


def test_mutation_a_state_that_ignores_the_closing_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mutant = mutated(study_state, "if closed is not None:", "if False:", module)
    monkeypatch.setattr(module, "study_state", mutant)
    with pytest.raises(AssertionError):
        assert_the_closing(tmp_path)


def end_the_study(tmp_path: Path, how: str) -> None:
    """Una pasada del bloque 2 que terminó el estudio, como la deja el lote."""
    block = block_dates(BLOCK2)
    meta = base_meta(block, block.ready)
    if how == "catálogo":
        meta = meta.model_copy(update={"outcome": BatchOutcome.TERMINATED})
    elif how == "hueco":
        meta = meta.model_copy(update={"outcome": BatchOutcome.GAP_PERSISTENT})
    else:
        changed = REGISTERED_PATH.model_copy(update={"template_sha256": "0" * 64})
        meta = meta.model_copy(update={"outcome": BatchOutcome.RAN, "measured_path": changed})
    write_forward_meta(tmp_path / "batches" / "2026-10-29" / "20261106T010000Z", meta)


def assert_the_ended_study_runs_nothing(tmp_path: Path, how: str) -> None:
    end_the_study(tmp_path, how)
    state = module.study_state(study(tmp_path), STAGE1, BOTH)
    assert state.pending == ()
    assert state.terminated is not None
    assert "bloque 2" in state.terminated


@pytest.mark.parametrize("how", ["catálogo", "hueco", "camino medido"])
def test_an_ended_study_has_nothing_pending(how: str, tmp_path: Path) -> None:
    assert_the_ended_study_runs_nothing(tmp_path, how)


@pytest.mark.parametrize("how", ["catálogo", "hueco", "camino medido"])
def test_mutation_a_state_that_ignores_the_termination_is_caught(
    how: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mutant = mutated(study_state, "if ended is not None:", "if False:", module)
    monkeypatch.setattr(module, "study_state", mutant)
    with pytest.raises(AssertionError):
        assert_the_ended_study_runs_nothing(tmp_path, how)


# ─────────────────────────────────────── La tabla de códigos ──────────────────────────────────────


EXPECTED = {
    0: (Action.NEXT, True, False),
    3: (Action.RETRY, True, False),
    6: (Action.RETRY, True, False),
    1: (Action.STOP, False, True),
    5: (Action.STOP, False, True),
    4: (Action.ENDS, False, True),
    7: (Action.ENDS, False, True),
    2: (Action.FAILURE, True, True),
    9: (Action.FAILURE, True, True),
}
"""Por código del primer bloque: la acción, si corre el segundo y si hay aviso."""


async def assert_the_table(tmp_path: Path) -> None:
    for code, (action, second, alert) in EXPECTED.items():
        runner = Scripted({BLOCK2: code, BLOCK3: 0})
        report = await module.run_auto(study(tmp_path / str(code)), STAGE1, lambda: BOTH, runner)
        assert report.runs[0].action is action, code
        assert runner.starts == ([BLOCK2, BLOCK3] if second else [BLOCK2]), code
        assert report.needs_operator is alert, (code, report.alerts)
        assert report.retry is (action is Action.RETRY), code


def test_the_table_covers_every_code_of_the_batch() -> None:
    assert set(ACTIONS) == {0, 1, 2, 3, 4, 5, 6, 7}
    assert action_for(42) is Action.FAILURE


@pytest.mark.asyncio
async def test_each_exit_code_has_its_action(tmp_path: Path) -> None:
    await assert_the_table(tmp_path)


@pytest.mark.parametrize(
    ("code", "wrong"),
    [
        (0, Action.RETRY),
        (3, Action.STOP),
        (6, Action.STOP),
        (1, Action.RETRY),
        (5, Action.NEXT),
        (4, Action.STOP),
        (7, Action.RETRY),
        (2, Action.NEXT),
    ],
)
@pytest.mark.asyncio
async def test_mutation_a_code_with_another_action_is_caught(
    code: int, wrong: Action, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "ACTIONS", {**ACTIONS, code: wrong})
    with pytest.raises(AssertionError):
        await assert_the_table(tmp_path)


@pytest.mark.asyncio
async def test_mutation_a_stop_that_does_not_stop_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mutant = mutated(
        run_auto,
        "unfinished.extend(state.pending[position + 1 :])\n            break",
        "unfinished.extend(state.pending[position + 1 :])",
        module,
    )
    monkeypatch.setattr(module, "run_auto", mutant)
    with pytest.raises(AssertionError):
        await assert_the_table(tmp_path)


async def warned(tmp_path: Path, left: timedelta, code: int) -> bool:
    now = block_dates(BLOCK2).deadline - left
    report = await run_auto(
        study(tmp_path), STAGE1, lambda: now, Scripted({BLOCK2: code, BLOCK3: 3})
    )
    return any(alert.startswith("bloque 2 ") and "le quedan" in alert for alert in report.alerts)


async def assert_the_deadline_is_warned(tmp_path: Path) -> None:
    """Con 3 días justos o menos se avisa, salga el lote con lo que salga si no queda completo.
    Con un instante más, no."""
    for code in (3, 6, 1):
        assert await warned(tmp_path / f"{code}-justo", WARN_BEFORE, code)
        assert await warned(tmp_path / f"{code}-menos", WARN_BEFORE - MICRO, code)
        assert not await warned(tmp_path / f"{code}-mas", WARN_BEFORE + MICRO, code)


@pytest.mark.asyncio
async def test_a_pending_block_close_to_its_deadline_is_always_warned(tmp_path: Path) -> None:
    await assert_the_deadline_is_warned(tmp_path)


@pytest.mark.asyncio
async def test_mutation_a_deadline_warning_one_instant_late_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mutant = mutated(module.near_deadline, "<= WARN_BEFORE", "< WARN_BEFORE", module)
    monkeypatch.setattr(module, "near_deadline", mutant)
    with pytest.raises(AssertionError):
        await assert_the_deadline_is_warned(tmp_path)


@pytest.mark.asyncio
async def test_a_blocked_batch_is_not_run_and_is_warned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(forward, "offline_checks", passing())
    assert (await run_block(tmp_path, BLOCK2, FakeDecider(), TickingClock())).code == 0
    as_another_decider(passes_of(tmp_path, BLOCK2)[-1])
    runner = Scripted({BLOCK3: 0})

    report = await run_auto(study(tmp_path), STAGE1, lambda: BOTH, runner)

    assert runner.starts == [BLOCK3]
    (alert,) = report.alerts
    assert alert.startswith("bloque 2 (2026-10-29): el lote es inválido")
    assert "temperature" in alert
    assert "repetirlo entero es del operador" in alert


@pytest.mark.asyncio
async def test_an_incomplete_batch_that_turns_out_invalid_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Un lote que sale con 3 y al releerlo es inválido no se reintenta: se avisa."""
    monkeypatch.setattr(forward, "offline_checks", passing())
    assert (await run_block(tmp_path, BLOCK2, FakeDecider(fail_on=[1]), TickingClock())).code == 3
    (first,) = passes_of(tmp_path, BLOCK2)
    runner = Scripted({BLOCK2: 3, BLOCK3: 0})

    async def invalidating(start: datetime) -> BatchResult:
        as_another_decider(first)
        return await runner(start)

    report = await run_auto(study(tmp_path), STAGE1, lambda: BOTH, invalidating)

    (run,) = report.runs
    assert run.action is Action.STOP
    assert run.status is BlockStatus.INVALID
    assert runner.starts == [BLOCK2]
    assert any("salió con 3 y es inválido" in alert for alert in report.alerts)


@pytest.mark.asyncio
async def test_a_new_upstream_entry_is_warned(tmp_path: Path) -> None:
    """Una pasada nueva que deja `upstream.md`: el asiento va al registro a mano."""
    runner = Scripted({BLOCK2: 0, BLOCK3: 0})

    async def changing(start: datetime) -> BatchResult:
        directory = tmp_path / "batches" / f"{start:%Y-%m-%d}" / "20261113T010000Z"
        directory.mkdir(parents=True)
        (directory / META_FILE).write_text("{}", encoding="utf-8")
        if start == BLOCK2:
            (directory / UPSTREAM_FILE).write_text("### asiento\n", encoding="utf-8")
        return await runner(start)

    report = await run_auto(study(tmp_path), STAGE1, lambda: BOTH, changing)

    (alert,) = report.alerts
    assert alert.startswith("bloque 2 (2026-10-29): cambió el upstream del decisor")
    assert UPSTREAM_FILE in alert
    assert report.runs[0].upstream_entries
    assert not report.runs[1].upstream_entries


@pytest.mark.asyncio
async def test_the_report_carries_the_counters_and_the_files_of_a_complete_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(forward, "offline_checks", passing())
    clock = TickingClock()

    async def real(start: datetime) -> BatchResult:
        return await run_block(tmp_path, start, FakeDecider(), clock)

    report = await run_auto(study(tmp_path), STAGE1, lambda: block_dates(BLOCK2).ready, real)

    (run,) = report.runs
    assert (run.code, run.action, run.status) == (0, Action.NEXT, BlockStatus.COMPLETE)
    assert (run.evaluations, run.decided, run.invalid, run.missing) == (3, 3, 0, 0)
    assert any(path.endswith("solo.jsonl") for path in run.files)
    assert not report.needs_operator
    assert "retorno" not in report.model_dump_json().lower()


# ──────────────────────────────────────────── El comando ──────────────────────────────────────────


class SpyBatch:
    """`run_batch` simulado: apunta con qué se llamó y devuelve un lote completo."""

    def __init__(self, code: int = 0) -> None:
        self.code = code
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    async def __call__(self, *args: object, **kwargs: object) -> BatchResult:
        self.calls.append((args, kwargs))
        return BatchResult(self.code, (f"El lote salió con {self.code}.",), ())


def assert_auto_passes_only_three_arguments(tmp_path: Path, spy: SpyBatch) -> None:
    """`spy` ya está puesto en lugar de `forward.run_batch`."""
    env = refusing_environment(tmp_path, block_dates(BLOCK2).ready)

    assert main(["auto"], env) == 0

    ((args, kwargs),) = spy.calls
    assert args == (BLOCK2, env, ("auto",))
    assert kwargs == {}


def test_auto_runs_the_batch_with_its_start_its_environment_and_its_argv_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = SpyBatch()
    monkeypatch.setattr(forward, "run_batch", spy)
    assert_auto_passes_only_three_arguments(tmp_path, spy)


@pytest.mark.parametrize(
    "new",
    [
        "run_batch(start, env, argv, supersede=True)",
        "run_batch(start, env, argv, (CandleRef('BTC/USDT', start),))",
    ],
    ids=["sustituye", "acepta una vela"],
)
def test_mutation_auto_that_supersedes_or_accepts_is_caught(
    new: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = SpyBatch()
    monkeypatch.setattr(forward, "run_batch", spy)
    mutant = mutated(forward.batch_runner, "run_batch(start, env, argv)", new, forward)
    monkeypatch.setattr(forward, "batch_runner", mutant)
    with pytest.raises(AssertionError):
        assert_auto_passes_only_three_arguments(tmp_path, spy)
    assert spy.calls, "el mutante llegó al lote simulado"


@pytest.mark.parametrize(
    "arguments",
    [
        ["auto", "--supersede"],
        ["auto", "--accept-revised", "BTC/USDT@2026-08-15T20:00Z"],
        ["auto", "--block", "2026-10-29"],
        ["auto", "--dry-run", "--report", "informe.json"],
    ],
)
def test_auto_has_no_flag_for_what_the_operator_decides(
    arguments: Sequence[str], tmp_path: Path
) -> None:
    with pytest.raises(SystemExit) as stopped:
        main(arguments, refusing_environment(tmp_path, TODAY))
    assert stopped.value.code == 2


def snapshot(directory: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def test_two_runs_with_nothing_pending_do_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Ni red, ni lote, ni archivo del estudio: solo el cerrojo, creado la primera vez."""
    spy = SpyBatch()
    monkeypatch.setattr(forward, "run_batch", spy)
    env = refusing_environment(tmp_path, TODAY)

    assert main(["auto"], env) == 0
    first = snapshot(tmp_path)
    assert main(["auto"], env) == 0

    assert snapshot(tmp_path) == first
    assert list(first) == [LOCK_NAME]
    assert spy.calls == []
    assert "no corrió ningún lote" in capsys.readouterr().out


def test_a_run_that_needs_the_operator_exits_one_and_leaves_its_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(forward, "run_batch", SpyBatch(code=5))
    report_path = tmp_path / "informe" / "auto.json"

    code = main(["auto", "--report", str(report_path)], refusing_environment(tmp_path, BOTH))

    assert code == 1
    report = AutoReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    assert [run.code for run in report.runs] == [5]
    assert report.runs[0].action is Action.STOP
    assert report.needs_operator


def test_the_dry_run_says_what_it_would_do_without_running_or_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    spy = SpyBatch()
    monkeypatch.setattr(forward, "run_batch", spy)

    assert main(["auto", "--dry-run"], refusing_environment(tmp_path, BOTH)) == 0

    assert spy.calls == []
    assert list(tmp_path.iterdir()) == [], "ni el cerrojo"
    out = capsys.readouterr().out
    assert "correría el lote del bloque 2 (2026-10-29)" in out
    assert "correría el lote del bloque 3 (2026-11-05)" in out


def test_today_the_dry_run_on_the_study_says_nothing_is_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Sobre el `data/forward/` del repositorio, con el reloj del día en que se escribió."""
    monkeypatch.setattr(forward, "run_batch", SpyBatch())
    paths = ForwardPaths(
        history=ROOT / FORWARD_HISTORY_DIR,
        funding=ROOT / FORWARD_FUNDING_DIR,
        manifests=ROOT / MANIFEST_DIR,
        batches=ROOT / BATCH_DIR,
        cache=tmp_path / "cache",
        stage1=STAGE1,
        revisions=ROOT / REVISIONS_FILE,
        lock=tmp_path / LOCK_NAME,
    )
    env = refusing_environment(tmp_path, TODAY)._replace(paths=paths)

    assert main(["auto", "--dry-run"], env) == 0

    out = capsys.readouterr().out
    assert "no hay ningún lote pendiente" in out
    assert "el lote del bloque 2 (2026-10-29) se abre el 2026-11-06T00:00:00+00:00" in out
    assert list(tmp_path.iterdir()) == []


# ──────────────────────────────────────────── El registro ─────────────────────────────────────────


def test_the_register_says_the_timer_runs_from_block_two_and_what_stays_manual() -> None:
    register = DOCS.read_text("utf-8").split("## Registro del estudio", 1)[1]
    entry = register.split("— Desde el bloque 2 los lotes los lanza el temporizador", 1)[1]
    entry = entry.split("\n### ", 1)[0]
    for fragment in (
        "python -m crypto_agents.forward auto",
        "19:00",
        "el commit que añade este asiento",
        "El lote del bloque 1 lo corre el operador",
        "--accept-revised",
        "--supersede",
        "forward baselines",
        "forward analyze",
        "apartado 3",
    ):
        assert fragment in entry, fragment


def test_a_batch_meta_written_by_the_timer_loads() -> None:
    """El `forward.json` no cambia de forma: el temporizador no añade campos."""
    meta = base_meta(block_dates(BLOCK2), block_dates(BLOCK2).ready)
    assert BatchMeta.model_validate_json(meta.model_dump_json()) == meta
