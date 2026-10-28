"""Los lotes que corre el temporizador: `python -m crypto_agents.forward auto`.

Lo que el bloque de la automatización pide, cada parte con su mutación:

- **Los bloques pendientes**, con sus bordes: el cierre más 24 h y el plazo, el bloque 1 que no es
  del temporizador, el tope de 52 bloques, los lotes completos y vacíos, el estudio cerrado y el
  terminado (con el `forward.json` que escribe el lote).
- **Los bloques que esperan al operador**: inválidos, con duplicados y retenidos por una pasada
  que se negó o se detuvo por una vela revisada; y los vencidos sin sus líneas base.
- **La tabla de códigos**, caso por caso, y el aviso del plazo; lo que pasa al día siguiente.
- **Un lote interrumpido** nunca se lee como completo.
- **`auto` llama al lote solo con su inicio, su entorno y su argv**: nunca acepta velas ni
  sustituye pasadas. Que nunca llama al análisis lo fija `tests/test_architecture.py`.
- **Idempotencia y `--dry-run`**: sin nada pendiente no hace nada, y el ensayo no corre ni
  escribe.
"""

from __future__ import annotations

import asyncio
import shutil
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

import crypto_agents.forward as forward
import crypto_agents.forward_auto as module
from crypto_agents.audit import META_FILE, AuditError, read_meta
from crypto_agents.forward import (
    GAP_WAIT,
    BatchMeta,
    BatchOutcome,
    BatchResult,
    Check,
    ForwardPaths,
    PreparedBlock,
    SupersededMark,
    base_meta,
    block_dates,
    gap_result,
    main,
    registered_decider,
    revised_result,
    run_pass,
    study_paths,
    write_forward_meta,
    write_superseded,
)
from crypto_agents.forward_analysis import pass_metas
from crypto_agents.forward_auto import (
    ACTIONS,
    AUTO_FIRST_BLOCK,
    BLOCKED,
    HELD_OUTCOMES,
    RUNNABLE,
    UPSTREAM_FILE,
    WARN_BEFORE,
    Action,
    AutoReport,
    BlockStatus,
    action_for,
    block_status,
    latest_pass,
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
    MissingCandle,
    MissingCandlesError,
    RevisedCandle,
    RevisedCandlesError,
    manifest_path,
)
from crypto_agents.forward_plan import closing_block
from crypto_agents.forward_record import FIRST_BLOCK, STEP
from crypto_agents.forward_render import REGISTERED_PATH
from crypto_agents.market import MarketDataError
from crypto_agents.selection import (
    DEFAULT_MANIFEST,
    HISTORY_DIR,
    load_manifest,
    load_selection_histories,
    write_manifest,
)
from tests.test_forward import (
    LOCK_NAME,
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
    refusing_environment,
)
from tests.test_net_outcomes import mutated

if TYPE_CHECKING:
    from collections.abc import Collection, Iterator, Mapping, Sequence
    from pathlib import Path

    from crypto_agents.forward_analysis import StudyPaths
    from crypto_agents.llm import Completion
    from crypto_agents.state import LLMOutput

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
    return module.block_status(paths, start, registered, pass_metas(paths.batches)).status


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


def touch_manifests(tmp_path: Path, blocks: int, skip: Collection[int] = ()) -> None:
    """Un manifiesto de una entrada por cada bloque del 1 al `blocks`, salvo los números de
    `skip`. Para el cierre solo cuenta que exista; tiene que ser válido porque los bloques vencidos
    se leen para saber si les faltan las líneas base."""
    reference = load_manifest(ROOT / DEFAULT_MANIFEST)
    manifest = reference.model_copy(
        update={"entries": reference.entries[:1], "target": 1, "strata": 1}
    )
    for position in range(blocks):
        if position + 1 not in skip:
            write_manifest(
                manifest_path(FIRST_BLOCK + position * STEP, tmp_path / "manifests"), manifest
            )


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


MISSING_SOL = MissingCandlesError(
    [MissingCandle(symbol="SOL/USDT", opens=datetime(2026, 11, 4, 20, tzinfo=UTC))]
)
"""Una vela nueva del bloque 2 que la descarga no trae."""


async def catalog_gone(tmp_path: Path, clock: TickingClock) -> BatchResult:
    """La pasada del bloque 2 con `glm-5.2` fuera del catálogo: `run_pass` sale con 4 y escribe
    `TERMINATED`. Pide `forward.offline_checks` ya simulada."""
    env = environment(tmp_path, FakeDecider(), FakeCatalog({"otro"}), clock=clock)
    return await run_pass(prepared_for(tmp_path, BLOCK2), env, ())


def gap_seen(tmp_path: Path, clock: TickingClock) -> BatchResult:
    """La pasada del bloque 2 a la que le falta una vela nueva: `gap_result`, la función con que el
    lote la registra, sale con 6 la primera vez y con 7 cuando ya pasaron 24 h."""
    env = environment(tmp_path, FakeDecider(), clock=clock)
    return gap_result(block_dates(BLOCK2), env, MISSING_SOL)


async def end_the_study(tmp_path: Path, how: str) -> None:
    """Una pasada del bloque 2 que terminó el estudio. El catálogo y el hueco los escribe el lote
    con sus funciones; el camino medido cambiado en una pasada que corrió no lo escribe ninguna
    función del lote (se niega antes de llamar), así que ese `forward.json` se escribe a mano."""
    block = block_dates(BLOCK2)
    clock = TickingClock(block.ready)
    if how == "catálogo":
        assert (await catalog_gone(tmp_path, clock)).code == 4
    elif how == "hueco":
        assert gap_seen(tmp_path, clock).code == 6
        clock.now += GAP_WAIT
        assert gap_seen(tmp_path, clock).code == 7
    else:
        changed = REGISTERED_PATH.model_copy(update={"template_sha256": "0" * 64})
        meta = base_meta(block, block.ready).model_copy(
            update={"outcome": BatchOutcome.RAN, "measured_path": changed}
        )
        write_forward_meta(tmp_path / "batches" / "2026-10-29" / "20261106T010000Z", meta)


async def assert_the_ended_study_runs_nothing(tmp_path: Path, how: str) -> None:
    await end_the_study(tmp_path, how)
    state = module.study_state(study(tmp_path), STAGE1, BOTH)
    assert state.pending == ()
    assert state.terminated is not None
    assert "bloque 2" in state.terminated


@pytest.mark.parametrize("how", ["catálogo", "hueco", "camino medido"])
@pytest.mark.asyncio
async def test_an_ended_study_has_nothing_pending(
    how: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(forward, "offline_checks", passing())
    await assert_the_ended_study_runs_nothing(tmp_path, how)


@pytest.mark.parametrize("how", ["catálogo", "hueco", "camino medido"])
@pytest.mark.asyncio
async def test_mutation_a_state_that_ignores_the_termination_is_caught(
    how: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(forward, "offline_checks", passing())
    mutant = mutated(study_state, "if ended is not None:", "if False:", module)
    monkeypatch.setattr(module, "study_state", mutant)
    with pytest.raises(AssertionError):
        await assert_the_ended_study_runs_nothing(tmp_path, how)


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
async def test_mutation_an_unknown_code_taken_as_complete_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Un código que la tabla no conoce (el 9 de `EXPECTED`) es un fallo, no un lote completo."""
    old = "ACTIONS.get(code, Action.FAILURE)"
    mutant = mutated(action_for, old, "ACTIONS.get(code, Action.NEXT)", module)
    monkeypatch.setattr(module, "action_for", mutant)
    with pytest.raises(AssertionError):
        await assert_the_table(tmp_path)


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


async def assert_an_incomplete_batch_that_turns_out_invalid_stops(tmp_path: Path) -> None:
    """Un lote que sale con 3 y al releerlo es inválido no se reintenta: se avisa."""
    assert (await run_block(tmp_path, BLOCK2, FakeDecider(fail_on=[1]), TickingClock())).code == 3
    (first,) = passes_of(tmp_path, BLOCK2)
    runner = Scripted({BLOCK2: 3, BLOCK3: 0})

    async def invalidating(start: datetime) -> BatchResult:
        as_another_decider(first)
        return await runner(start)

    report = await module.run_auto(study(tmp_path), STAGE1, lambda: BOTH, invalidating)

    assert [run.start for run in report.runs] == [BLOCK2]
    run = report.runs[0]
    assert run.action is Action.STOP
    assert run.status is BlockStatus.INVALID
    assert runner.starts == [BLOCK2]
    assert any("salió con 3 y es inválido" in alert for alert in report.alerts)


@pytest.mark.asyncio
async def test_an_incomplete_batch_that_turns_out_invalid_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(forward, "offline_checks", passing())
    await assert_an_incomplete_batch_that_turns_out_invalid_stops(tmp_path)


@pytest.mark.asyncio
async def test_mutation_an_invalid_batch_retried_as_incomplete_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(forward, "offline_checks", passing())
    old = "if action is Action.RETRY and after.status is BlockStatus.INVALID:"
    monkeypatch.setattr(module, "run_auto", mutated(run_auto, old, "if False:", module))
    with pytest.raises(AssertionError):
        await assert_an_incomplete_batch_that_turns_out_invalid_stops(tmp_path)


async def assert_a_new_upstream_entry_is_warned(tmp_path: Path) -> None:
    """Una pasada nueva que deja `upstream.md`: el asiento va al registro a mano."""
    runner = Scripted({BLOCK2: 0, BLOCK3: 0})

    async def changing(start: datetime) -> BatchResult:
        directory = tmp_path / "batches" / f"{start:%Y-%m-%d}" / "20261113T010000Z"
        directory.mkdir(parents=True)
        (directory / META_FILE).write_text("{}", encoding="utf-8")
        if start == BLOCK2:
            (directory / UPSTREAM_FILE).write_text("### asiento\n", encoding="utf-8")
        return await runner(start)

    report = await module.run_auto(study(tmp_path), STAGE1, lambda: BOTH, changing)

    assert len(report.alerts) == 1, report.alerts
    alert = report.alerts[0]
    assert alert.startswith("bloque 2 (2026-10-29): cambió el upstream del decisor")
    assert UPSTREAM_FILE in alert
    assert report.runs[0].upstream_entries
    assert not report.runs[1].upstream_entries


@pytest.mark.asyncio
async def test_a_new_upstream_entry_is_warned(tmp_path: Path) -> None:
    await assert_a_new_upstream_entry_is_warned(tmp_path)


@pytest.mark.asyncio
async def test_mutation_an_upstream_entry_left_unread_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = "if (path / UPSTREAM_FILE).is_file()"
    monkeypatch.setattr(module, "run_auto", mutated(run_auto, old, "if False", module))
    with pytest.raises(AssertionError):
        await assert_a_new_upstream_entry_is_warned(tmp_path)


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


def test_a_batch_meta_written_by_the_timer_loads() -> None:
    """El `forward.json` no cambia de forma: el temporizador no añade campos."""
    meta = base_meta(block_dates(BLOCK2), block_dates(BLOCK2).ready)
    assert BatchMeta.model_validate_json(meta.model_dump_json()) == meta


# ──────────────────────────── El nombre de `upstream.md`, en los dos módulos ──────────────────────


def assert_the_upstream_names_agree() -> None:
    """`forward_auto` busca el asiento con su propio literal: el que escribe el lote."""
    assert module.UPSTREAM_FILE == forward.UPSTREAM_FILE


def test_the_timer_looks_for_the_upstream_entry_the_batch_writes() -> None:
    assert_the_upstream_names_agree()


@pytest.mark.parametrize("where", [module, forward], ids=["forward_auto", "forward"])
def test_mutation_an_upstream_name_that_drifts_is_caught(
    where: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(where, "UPSTREAM_FILE", "upstream.txt")
    with pytest.raises(AssertionError):
        assert_the_upstream_names_agree()


# ─────────────────── El tope de 52 bloques, el estudio abierto y los bloques no vacíos ────────────


def assert_the_study_ends_at_block_fifty_two(tmp_path: Path) -> None:
    """Sin 50 bloques no vacíos el estudio no cierra, pero el 52 es su último bloque: cuando se abre
    el lote del 53, el 52 sigue pendiente y el 53 no lo está nunca."""
    touch_manifests(tmp_path, 49)
    fifty_two = FIRST_BLOCK + 51 * STEP
    fifty_three = block_dates(FIRST_BLOCK + 52 * STEP)
    assert pending_starts(tmp_path, fifty_three.ready) == [fifty_two]
    assert pending_starts(tmp_path, fifty_three.deadline - MICRO) == []


def test_block_fifty_two_is_the_last_one(tmp_path: Path) -> None:
    assert_the_study_ends_at_block_fifty_two(tmp_path)


def test_mutation_a_fifty_third_block_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = "for position in range(MAX_BLOCKS):"
    mutant = mutated(study_state, old, "for position in range(MAX_BLOCKS + 1):", module)
    monkeypatch.setattr(module, "study_state", mutant)
    with pytest.raises(AssertionError):
        assert_the_study_ends_at_block_fifty_two(tmp_path)


def test_an_open_study_is_read_as_open() -> None:
    """Con menos de 52 bloques y menos de 50 no vacíos, `closing_block` se niega: el estudio sigue
    abierto, y eso se lee como abierto, no como un error."""
    with pytest.raises(ValueError, match="sigue abierto"):
        closing_block([0])


def test_mutation_an_open_study_read_as_an_error_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sin el `except ValueError`, la negativa de `closing_block` sobre un estudio abierto rompe la
    lectura del estado en vez de dejar el bloque pendiente."""
    assert pending_starts(tmp_path, block_dates(BLOCK2).ready) == [BLOCK2]
    old = (
        "try:\n            closed = closing_block(filled)\n"
        "        except ValueError:\n            closed = None"
    )
    mutant = mutated(study_state, old, "closed = closing_block(filled)", module)
    monkeypatch.setattr(module, "study_state", mutant)
    with pytest.raises(ValueError, match="sigue abierto"):
        pending_starts(tmp_path, block_dates(BLOCK2).ready)


def assert_a_block_without_manifest_counts_as_empty(tmp_path: Path) -> None:
    """Para el cierre, un bloque con manifiesto no está vacío y uno sin él sí: con el 50 sin
    manifiesto, el estudio cierra en el 51, no en el 50, y el 50 y el 51 están pendientes."""
    touch_manifests(tmp_path, 51, skip={50})
    fifty, fifty_one = FIRST_BLOCK + 49 * STEP, FIRST_BLOCK + 50 * STEP
    state = module.study_state(study(tmp_path), STAGE1, block_dates(fifty_one).ready)
    assert state.closed_at == 51
    assert [reading.dates.start for reading in state.pending] == [fifty, fifty_one]


def test_a_block_without_manifest_counts_as_empty(tmp_path: Path) -> None:
    assert_a_block_without_manifest_counts_as_empty(tmp_path)


@pytest.mark.parametrize("new", ["filled.append(1)", "filled.append(0)"], ids=["todos", "ninguno"])
def test_mutation_a_non_empty_block_misread_is_caught(
    new: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = "filled.append(1 if manifest_path(start, paths.manifests).is_file() else 0)"
    monkeypatch.setattr(module, "study_state", mutated(study_state, old, new, module))
    with pytest.raises(AssertionError):
        assert_a_block_without_manifest_counts_as_empty(tmp_path)


# ───────────────────────────────── Lo que se corre y lo que espera ────────────────────────────────

STATES_NOW = block_dates(BLOCK2).ready + timedelta(hours=1)
"""Dentro del plazo del bloque 2, con el lote del 3 todavía cerrado."""

REVISED_BTC = RevisedCandlesError(
    [
        RevisedCandle(
            symbol="BTC/USDT",
            opens=datetime(2026, 8, 15, 20, tzinfo=UTC),
            stored=(1.0, 2.0, 0.5, 1.5, 10.0),
            returned=(1.0, 2.0, 0.5, 1.6, 10.0),
        )
    ]
)
"""Una vela guardada que el exchange devuelve distinta y nadie aceptó."""

STATES = {
    "sin preparar": "pendiente",
    "incompleto": "pendiente",
    "vacío el 3": "pendiente",
    "completo": "nada",
    "vacío": "nada",
    "inválido": "espera",
    "con duplicados": "espera",
    "retenido": "espera",
}
"""Cada estado del bloque 2, en su propio estudio, y qué se hace con él."""


def empty_block(tmp_path: Path, start: datetime) -> PreparedBlock:
    """El bloque preparado sin activaciones: sin manifiesto."""
    block = prepared_for(tmp_path, start)
    assert block.manifest_file is not None
    block.manifest_file.unlink()
    return block._replace(manifest=None, manifest_file=None, counts=(0,))


async def refused_pass(tmp_path: Path, clock: TickingClock) -> BatchResult:
    """La pasada del bloque 2 que se niega porque falla una comprobación: `run_pass` deja su
    `forward.json` con `REFUSED`."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(forward, "offline_checks", passing(Check.TIES))
        env = environment(tmp_path, FakeDecider(), clock=clock)
        return await run_pass(prepared_for(tmp_path, BLOCK2), env, ())


async def build_states(root: Path) -> None:
    clock = TickingClock(block_dates(BLOCK2).ready)
    assert (await run_block(root / "incompleto", BLOCK2, FakeDecider(fail_on=[1]), clock)).code == 3
    assert (await run_block(root / "completo", BLOCK2, FakeDecider(), clock)).code == 0
    assert (await run_block(root / "inválido", BLOCK2, FakeDecider(), clock)).code == 0
    as_another_decider(passes_of(root / "inválido", BLOCK2)[-1])
    assert (await run_block(root / "con duplicados", BLOCK2, FakeDecider(), clock)).code == 0
    (first,) = passes_of(root / "con duplicados", BLOCK2)
    shutil.copytree(first, first.parent / "29991231T000000Z")
    for name, start in (("vacío", BLOCK2), ("vacío el 3", BLOCK3)):
        env = environment(root / name, FakeDecider(), clock=clock)
        assert (await run_pass(empty_block(root / name, start), env, ())).code == 0
    env = environment(root / "retenido", FakeDecider(), clock=clock)
    assert revised_result(block_dates(BLOCK2), env, REVISED_BTC).code == 5


@pytest.fixture(scope="module")
def states(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Los estudios de `STATES`, armados una vez con el lote de verdad y con las funciones con que
    el lote registra sus pasadas. Las pruebas solo los leen."""
    root = tmp_path_factory.mktemp("estados")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(forward, "offline_checks", passing())
        asyncio.run(build_states(root))
    yield root


def assert_what_runs_and_what_waits(root: Path) -> None:
    for name, what in STATES.items():
        state = module.study_state(study(root / name), STAGE1, STATES_NOW)
        pending = [reading.dates.start for reading in state.pending]
        blocked = [reading.dates.start for reading in state.blocked]
        assert pending == ([BLOCK2] if what == "pendiente" else []), name
        assert blocked == ([BLOCK2] if what == "espera" else []), name


def test_what_runs_and_what_waits_for_the_operator(states: Path) -> None:
    assert_what_runs_and_what_waits(states)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("RUNNABLE", RUNNABLE - {BlockStatus.NOT_PREPARED}),
        ("RUNNABLE", RUNNABLE - {BlockStatus.INCOMPLETE}),
        ("RUNNABLE", RUNNABLE | {BlockStatus.COMPLETE}),
        ("RUNNABLE", RUNNABLE | {BlockStatus.EMPTY}),
        ("BLOCKED", BLOCKED - {BlockStatus.INVALID}),
        ("BLOCKED", BLOCKED - {BlockStatus.DUPLICATE}),
        ("BLOCKED", BLOCKED - {BlockStatus.HELD}),
    ],
    ids=[
        "sin preparar no se corre",
        "incompleto no se corre",
        "completo se corre",
        "vacío se corre",
        "inválido no espera",
        "con duplicados no espera",
        "retenido no espera",
    ],
)
def test_mutation_a_state_in_the_wrong_set_is_caught(
    name: str, value: frozenset[BlockStatus], states: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, name, value)
    with pytest.raises(AssertionError):
        assert_what_runs_and_what_waits(states)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("elif counts.invalidating:", "elif False:"),
        (
            "return BlockReading(dates, BlockStatus.DUPLICATE, evaluations, None)",
            "return BlockReading(dates, BlockStatus.INCOMPLETE, evaluations, None)",
        ),
        (
            "meta.outcome is BatchOutcome.EMPTY and meta.block.start == start",
            "meta.block.start == start",
        ),
        (
            "meta.outcome is BatchOutcome.EMPTY and meta.block.start == start",
            "meta.outcome is BatchOutcome.EMPTY",
        ),
    ],
    ids=[
        "inválido leído incompleto",
        "duplicados leídos incompleto",
        "cualquier pasada sin manifiesto es vacío",
        "el vacío de otro bloque",
    ],
)
def test_mutation_a_state_misread_is_caught(
    old: str, new: str, states: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "block_status", mutated(block_status, old, new, module))
    with pytest.raises(AssertionError):
        assert_what_runs_and_what_waits(states)


# ─────────────────────────────── Los bloques retenidos (decisión 1) ───────────────────────────────

HELD_NOW = block_dates(BLOCK2).ready + timedelta(days=1)
HELD = {
    "vela revisada": True,
    "negada": True,
    "negada y después corrida": False,
    "corrida y después negada": True,
    "negada y sustituida": False,
}
"""Las pasadas del bloque 2 en cada estudio, en orden, y si lo retienen."""


async def build_held(root: Path) -> None:
    clock = TickingClock(block_dates(BLOCK2).ready)
    env = environment(root / "vela revisada", FakeDecider(), clock=clock)
    assert revised_result(block_dates(BLOCK2), env, REVISED_BTC).code == 5
    assert (await refused_pass(root / "negada", clock)).code == 1
    first = root / "negada y después corrida"
    assert (await refused_pass(first, clock)).code == 1
    assert (await run_block(first, BLOCK2, FakeDecider(fail_on=[1]), clock)).code == 3
    later = root / "corrida y después negada"
    assert (await run_block(later, BLOCK2, FakeDecider(fail_on=[1]), clock)).code == 3
    assert (await refused_pass(later, clock)).code == 1
    marked = root / "negada y sustituida"
    assert (await run_block(marked, BLOCK2, FakeDecider(fail_on=[1]), clock)).code == 3
    assert (await refused_pass(marked, clock)).code == 1
    mark = SupersededMark(at=clock(), by="29991231T000000Z", reasons=("prueba",))
    write_superseded([passes_of(marked, BLOCK2)[-1]], mark)


@pytest.fixture(scope="module")
def held(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    root = tmp_path_factory.mktemp("retenidos")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(forward, "offline_checks", passing())
        asyncio.run(build_held(root))
    yield root


async def assert_a_held_block_waits_for_a_later_pass(root: Path) -> None:
    """Retiene la última pasada no sustituida si se negó o se detuvo por una vela revisada: el
    bloque no se corre y se avisa con su plazo. Una pasada posterior lo libera."""
    block = block_dates(BLOCK2)
    for name, retained in HELD.items():
        runner = Scripted({BLOCK2: 3})
        report = await module.run_auto(study(root / name), STAGE1, lambda: HELD_NOW, runner)
        assert runner.starts == ([] if retained else [BLOCK2]), name
        if not retained:
            continue
        (alert,) = report.alerts
        last = latest_pass(pass_metas(root / name / "batches"), BLOCK2)
        assert last is not None
        assert alert.startswith("bloque 2 (2026-10-29): retenido"), alert
        assert f"{last[0].name}, {last[1].outcome.value}" in alert
        assert f"Vence el {block.deadline.isoformat()}" in alert
        assert "quedan 288.0 h" in alert


@pytest.mark.asyncio
async def test_a_held_block_waits_for_a_later_pass(held: Path) -> None:
    await assert_a_held_block_waits_for_a_later_pass(held)


@pytest.mark.parametrize(
    "outcome", sorted(HELD_OUTCOMES), ids=lambda outcome: f"sin {outcome.value}"
)
@pytest.mark.asyncio
async def test_mutation_an_outcome_that_no_longer_holds_is_caught(
    outcome: BatchOutcome, held: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "HELD_OUTCOMES", HELD_OUTCOMES - {outcome})
    with pytest.raises(AssertionError):
        await assert_a_held_block_waits_for_a_later_pass(held)


@pytest.mark.parametrize(
    ("function", "old", "new"),
    [
        ("latest_pass", "return max(", "return min("),
        (
            "latest_pass",
            "if meta.block.start == start and superseded(path) is None",
            "if meta.block.start == start",
        ),
        ("_blocked_alert", "{dates.deadline.isoformat()}; quedan", "; quedan"),
    ],
    ids=["la primera pasada", "las sustituidas también", "el aviso sin plazo"],
)
@pytest.mark.asyncio
async def test_mutation_a_held_block_misread_is_caught(
    function: str, old: str, new: str, held: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, function, mutated(getattr(module, function), old, new, module))
    with pytest.raises(AssertionError):
        await assert_a_held_block_waits_for_a_later_pass(held)


class Unreachable:
    """El exchange sin red: la descarga se cae antes de que el lote escriba nada."""

    async def __call__(self, *args: object) -> list[list[float]]:
        raise MarketDataError("sin red: el exchange no responde")


async def assert_a_refusal_without_a_pass_is_retried(tmp_path: Path) -> None:
    """Un 1 que no dejó pasada en disco, como un error de red al descargar: ese día detiene todo y
    al día siguiente el bloque se vuelve a correr."""
    day = block_dates(BLOCK2).ready
    env = environment(tmp_path, FakeDecider(), clock=TickingClock(day))
    batch = forward.batch_runner(env._replace(fetch_candles=Unreachable()), ())
    report = await module.run_auto(study(tmp_path), STAGE1, lambda: day, batch)
    assert [(run.code, run.action) for run in report.runs] == [(1, Action.STOP)]
    assert "sin red" in report.runs[0].headline
    assert passes_of(tmp_path, BLOCK2) == []
    again = Scripted({BLOCK2: 0})
    await module.run_auto(study(tmp_path), STAGE1, lambda: day + timedelta(days=1), again)
    assert again.starts == [BLOCK2]


@pytest.mark.asyncio
async def test_a_refusal_without_a_pass_is_retried_the_next_day(tmp_path: Path) -> None:
    await assert_a_refusal_without_a_pass_is_retried(tmp_path)


@pytest.mark.asyncio
async def test_mutation_a_block_held_without_a_pass_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = "    latest = latest_pass(metas, reading.dates.start)\n"
    new = (
        f"{old}    if latest is None:\n"
        "        return reading._replace(status=BlockStatus.HELD, held_by='sin pasada')\n"
    )
    monkeypatch.setattr(module, "_held", mutated(module._held, old, new, module))
    with pytest.raises(AssertionError):
        await assert_a_refusal_without_a_pass_is_retried(tmp_path)


@pytest.mark.parametrize("code", [4, 5, 7, 1])
@pytest.mark.asyncio
async def test_after_a_stop_the_next_day_does_not_run_the_block_again(
    code: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """El lote de verdad del bloque 2 sale con `code`. Con 4 y 7 el estudio termina y nada vuelve
    a correr; con 5 y 1 el bloque queda retenido y se avisa. El 7 llega tras un 6 a las 24 h."""
    monkeypatch.setattr(forward, "offline_checks", passing())
    block = block_dates(BLOCK2)
    clock = TickingClock(block.ready)

    async def batch(start: datetime) -> BatchResult:
        if code == 4:
            return await catalog_gone(tmp_path, clock)
        if code == 5:
            env = environment(tmp_path, FakeDecider(), clock=clock)
            return revised_result(block, env, REVISED_BTC)
        if code == 1:
            return await refused_pass(tmp_path, clock)
        return gap_seen(tmp_path, clock)

    day = block.ready
    report = await module.run_auto(study(tmp_path), STAGE1, lambda: day, batch)
    if code == 7:
        assert [run.code for run in report.runs] == [6]
        day += GAP_WAIT
        clock.now = day
        report = await module.run_auto(study(tmp_path), STAGE1, lambda: day, batch)
    assert [run.code for run in report.runs] == [code]
    assert report.runs[0].action is (Action.ENDS if code in (4, 7) else Action.STOP)

    following = day + timedelta(days=1)
    later = await module.run_auto(study(tmp_path), STAGE1, lambda: following, NeverCalled())
    assert later.runs == ()
    if code in (4, 7):
        assert later.terminated is not None
        assert later.alerts == ()
    else:
        (alert,) = later.alerts
        assert alert.startswith("bloque 2 (2026-10-29): retenido")


# ───────────────────────────── Los bloques vencidos (decisión 2) ──────────────────────────────────

EXPIRED_NOW = block_dates(BLOCK2).deadline
"""El plazo del bloque 2, vencido justo ahora. El bloque 3 está dentro del suyo."""


class Killed(BaseException):
    """El proceso que muere a mitad de una pasada: nada de lo que hay encima lo atrapa."""


class Dying(FakeDecider):
    """El proveedor simulado que mata el proceso en su tercera llamada, con dos registros de
    `solo` ya escritos y ninguno de las líneas base."""

    async def complete(self, choice: object, prompt: str, schema: type[LLMOutput]) -> Completion:
        if self.calls == 2:
            raise Killed
        return await super().complete(choice, prompt, schema)


async def interrupted(tmp_path: Path, clock: TickingClock) -> None:
    with pytest.raises(Killed):
        await run_block(tmp_path, BLOCK2, Dying(), clock)


def after_deadline(tmp_path: Path) -> forward.BatchEnvironment:
    """El entorno de `forward baselines` del bloque 2: sin backends ni catálogo."""
    return no_model_environment(tmp_path)._replace(clock=TickingClock(EXPIRED_NOW))


EXPIRED = {
    "nunca corrió": True,
    "nunca corrió y vacío completado": False,
    "cortado": True,
    "cortado y completado": False,
    "solo incompleto": False,
    "completo": False,
    "con duplicados": True,
}
"""Cada estudio con el bloque 2 vencido, y si se avisa. «solo incompleto» tiene las líneas base
completas en la misma pasada: el lote está incompleto pero no falta nada que completar."""


async def build_expired(root: Path) -> None:
    clock = TickingClock(block_dates(BLOCK2).ready)
    vacant = root / "nunca corrió y vacío completado"
    done = await forward.baselines_pass(empty_block(vacant, BLOCK2), after_deadline(vacant), ())
    assert done.code == 0
    for name in ("cortado", "cortado y completado"):
        await interrupted(root / name, clock)
    completed = root / "cortado y completado"
    block = prepared_for(completed, BLOCK2)
    assert (await forward.baselines_pass(block, after_deadline(completed), ())).code == 0
    short = root / "solo incompleto"
    assert (await run_block(short, BLOCK2, FakeDecider(fail_on=[1]), clock)).code == 3
    assert (await run_block(root / "completo", BLOCK2, FakeDecider(), clock)).code == 0
    assert (await run_block(root / "con duplicados", BLOCK2, FakeDecider(), clock)).code == 0
    (first,) = passes_of(root / "con duplicados", BLOCK2)
    shutil.copytree(first, first.parent / "29991231T000000Z")


@pytest.fixture(scope="module")
def expired(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    root = tmp_path_factory.mktemp("vencidos")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(forward, "offline_checks", passing())
        patch.setattr(forward, "baseline_checks", passing_baselines)
        asyncio.run(build_expired(root))
    yield root


class Completes:
    """El lote simulado que sale con 0 en cualquier bloque, sin escribir nada: los bloques 3 y 4
    están pendientes cuando vence el 2."""

    async def __call__(self, start: datetime) -> BatchResult:
        return BatchResult(0, ("El lote salió con 0.",), ())


async def expired_alerts(tmp_path: Path, now: datetime) -> list[str]:
    report = await module.run_auto(study(tmp_path), STAGE1, lambda: now, Completes())
    return [alert for alert in report.alerts if "venció" in alert]


async def assert_an_expired_block_is_warned_until_its_baselines_are_complete(root: Path) -> None:
    """Un bloque vencido con el lote incompleto y sus líneas base sin completar se avisa cada día;
    con ellas completas, o con el bloque registrado vacío, no. El bloque 1, vencido en todos los
    estudios, no es del temporizador y no se avisa."""
    for name, warned in EXPIRED.items():
        for now in (EXPIRED_NOW, EXPIRED_NOW + timedelta(days=1)):
            alerts = await expired_alerts(root / name, now)
            if not warned:
                assert alerts == [], (name, alerts)
                continue
            assert len(alerts) == 1, (name, alerts)
            assert alerts[0].startswith(
                "bloque 2 (2026-10-29): su plazo venció el 2026-11-19T00:00:00+00:00"
            ), alerts[0]
            assert "líneas base sin completar" in alerts[0]


@pytest.mark.asyncio
async def test_an_expired_block_is_warned_until_its_baselines_are_complete(expired: Path) -> None:
    await assert_an_expired_block_is_warned_until_its_baselines_are_complete(expired)


@pytest.mark.parametrize(
    ("function", "old", "new"),
    [
        (
            "study_state",
            "if reading.status not in DONE and not baseline_arms_complete(paths, start):",
            "if False:",
        ),
        (
            "study_state",
            "if reading.status not in DONE and not baseline_arms_complete(paths, start):",
            "if reading.status not in DONE:",
        ),
        (
            "study_state",
            "if reading.status not in DONE and not baseline_arms_complete(paths, start):",
            "if not baseline_arms_complete(paths, start):",
        ),
        ("study_state", "if start < AUTO_FIRST_BLOCK:", "if start < FIRST_BLOCK:"),
        (
            "baseline_arms_complete",
            "if not manifest.is_file():\n        return False",
            "if not manifest.is_file():\n        return True",
        ),
        (
            "baseline_arms_complete",
            "except DuplicateRecordError:\n        return False",
            "except DuplicateRecordError:\n        return True",
        ),
        (
            "baseline_arms_complete",
            "return all(expected <= state.done.get(arm, set()) for arm in BASELINE_ARMS)",
            "return True",
        ),
    ],
    ids=[
        "no avisa",
        "avisa aunque estén completas",
        "avisa aunque el bloque esté vacío",
        "avisa del bloque 1",
        "sin manifiesto, completas",
        "con duplicados, completas",
        "siempre completas",
    ],
)
@pytest.mark.asyncio
async def test_mutation_an_expired_block_misread_is_caught(
    function: str, old: str, new: str, expired: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, function, mutated(getattr(module, function), old, new, module))
    with pytest.raises(AssertionError):
        await assert_an_expired_block_is_warned_until_its_baselines_are_complete(expired)


async def assert_completing_an_empty_block_silences_it(tmp_path: Path) -> None:
    """Un bloque vacío que nunca corrió se avisa al vencer; `forward baselines` lo registra vacío,
    y el aviso se apaga."""
    assert len(await expired_alerts(tmp_path, EXPIRED_NOW)) == 1
    result = await forward.baselines_pass(
        empty_block(tmp_path, BLOCK2), after_deadline(tmp_path), ()
    )
    assert result.code == 0
    assert await expired_alerts(tmp_path, EXPIRED_NOW + timedelta(days=1)) == []


@pytest.mark.asyncio
async def test_completing_an_empty_block_silences_it(tmp_path: Path) -> None:
    await assert_completing_an_empty_block_silences_it(tmp_path)


@pytest.mark.asyncio
async def test_mutation_baselines_that_do_not_record_the_empty_block_are_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = "write_forward_meta(directory, meta)\n            recorded = directory"
    mutant = mutated(forward.baselines_pass, old, "recorded = directory", forward)
    monkeypatch.setattr(forward, "baselines_pass", mutant)
    with pytest.raises(AssertionError):
        await assert_completing_an_empty_block_silences_it(tmp_path)


# ───────────────────────────── Un lote interrumpido (decisión 10) ─────────────────────────────────


def assert_an_interrupted_batch_is_never_complete(tmp_path: Path) -> None:
    """La pasada muere a mitad con parte de sus registros escritos: el bloque queda incompleto y
    se vuelve a correr. Si además quedó media línea en un journal, el bloque no se puede leer, y
    `auto` lo avisa sin correr nada."""
    asyncio.run(interrupted(tmp_path, TickingClock(block_dates(BLOCK2).ready)))
    assert status_of(tmp_path, BLOCK2) is BlockStatus.INCOMPLETE
    assert pending_starts(tmp_path, BOTH) == [BLOCK2, BLOCK3]

    solo = passes_of(tmp_path, BLOCK2)[-1] / "solo.jsonl"
    with solo.open("a", encoding="utf-8") as handle:
        handle.write('{"run_id": "')
    with pytest.raises(AuditError):
        status_of(tmp_path, BLOCK2)
    report_path = tmp_path / "informe.json"
    env = refusing_environment(tmp_path, BOTH)
    assert main(["auto", "--report", str(report_path)], env) == 1
    report = AutoReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    assert report.runs == ()
    (alert,) = report.alerts
    assert alert.startswith("auto no corrió ningún lote")


def test_an_interrupted_batch_is_never_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(forward, "offline_checks", passing())
    assert_an_interrupted_batch_is_never_complete(tmp_path)


def test_mutation_an_interrupted_batch_read_as_complete_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(forward, "offline_checks", passing())
    mutant = mutated(block_status, "elif counts.complete:", "elif not counts.invalidating:", module)
    monkeypatch.setattr(module, "block_status", mutant)
    with pytest.raises(AssertionError):
        assert_an_interrupted_batch_is_never_complete(tmp_path)


# ─────────────────────── El aviso del plazo, solo para lo que no queda completo ───────────────────


async def assert_a_batch_completed_near_its_deadline_is_not_warned(tmp_path: Path) -> None:
    now = block_dates(BLOCK2).deadline - timedelta(days=2)
    clock = TickingClock(now)

    async def real(start: datetime) -> BatchResult:
        return await run_block(tmp_path, start, FakeDecider(), clock)

    report = await module.run_auto(study(tmp_path), STAGE1, lambda: now, real)
    assert [run.status for run in report.runs] == [BlockStatus.COMPLETE, BlockStatus.COMPLETE]
    assert report.alerts == ()


@pytest.mark.asyncio
async def test_a_batch_completed_near_its_deadline_is_not_warned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(forward, "offline_checks", passing())
    await assert_a_batch_completed_near_its_deadline_is_not_warned(tmp_path)


@pytest.mark.asyncio
async def test_mutation_a_complete_batch_warned_for_its_deadline_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(forward, "offline_checks", passing())
    old = "if after.status not in (BlockStatus.COMPLETE, BlockStatus.EMPTY):"
    monkeypatch.setattr(module, "run_auto", mutated(run_auto, old, "if True:", module))
    with pytest.raises(AssertionError):
        await assert_a_batch_completed_near_its_deadline_is_not_warned(tmp_path)


# ────────────────────────── Lo que dice `--dry-run` de retenidos y vencidos ───────────────────────


def assert_the_dry_run_names_held_and_expired_blocks(held: Path, expired: Path) -> None:
    on_hold = module.render_state(study_state(study(held / "vela revisada"), STAGE1, HELD_NOW))
    (line,) = [line for line in on_hold if "no correría" in line]
    assert line.startswith("- no correría el lote del bloque 2 (2026-10-29): está retenido (")
    assert line.endswith(", vela revisada), y se avisaría al operador.")
    late = module.render_state(study_state(study(expired / "nunca corrió"), STAGE1, EXPIRED_NOW))
    assert (
        "- el lote del bloque 2 (2026-10-29) venció el 2026-11-19T00:00:00+00:00 sin preparar y "
        "con sus líneas base sin completar: se avisaría."
    ) in late


def test_the_dry_run_names_held_and_expired_blocks(held: Path, expired: Path) -> None:
    assert_the_dry_run_names_held_and_expired_blocks(held, expired)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("for reading in state.expired:", "for reading in ():"),
        ('held = f" ({reading.held_by})" if reading.held_by is not None else ""', 'held = ""'),
    ],
    ids=["sin los vencidos", "sin la pasada que retiene"],
)
def test_mutation_a_dry_run_that_hides_them_is_caught(
    old: str, new: str, held: Path, expired: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "render_state", mutated(module.render_state, old, new, module))
    with pytest.raises(AssertionError):
        assert_the_dry_run_names_held_and_expired_blocks(held, expired)
