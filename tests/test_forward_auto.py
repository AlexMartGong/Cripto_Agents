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
  escribe. Con un journal ilegible, el ensayo avisa por el mismo camino que `auto`.
- **El cierre, contado también con el bloque 52**: el mismo bloque y los mismos no vacíos que el
  código congelado, con bloques vacíos registrados y sin registrar; el aviso de cada ejecución
  mientras está cerrado, que no es un fallo (`auto` sale con 0 y no es un aviso para el operador),
  y lo que dice `--dry-run`.
- **El registro de un bloque vacío**: el análisis lo lee igual lo escriba el lote o
  `forward baselines`, también preparado más tarde sobre series alargadas y con una vela aceptada,
  y en la rama terminada cambia exactamente dos líneas.
- **El registro**: los asientos de los PR #26, #27 y #28 y del lote del bloque 1, en orden y
  sin horas de commit ni locales; los del bloque 1, atados a los `forward.json` y al journal
  del lote.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, NamedTuple

import pytest
from pydantic import create_model

import crypto_agents.forward as forward
import crypto_agents.forward_analysis as analysis_module
import crypto_agents.forward_auto as module
import crypto_agents.forward_watch as watch_module
from crypto_agents.activation_sweep import DEFAULT_SYMBOLS, history_path
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
    UpstreamStatus,
    action_for,
    block_status,
    latest_pass,
    render_report,
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
    AcceptedRevision,
    MissingCandle,
    MissingCandlesError,
    RevisedCandle,
    RevisedCandlesError,
    SeriesFile,
    manifest_path,
    read_histories,
)
from crypto_agents.forward_plan import MAX_BLOCKS, closing_block
from crypto_agents.forward_record import FIRST_BLOCK, FORWARD_META_FILE, STEP, batch_directory
from crypto_agents.forward_render import REGISTERED_PATH
from crypto_agents.forward_watch import WatchEnvironment
from crypto_agents.forward_watch import main as watch_main
from crypto_agents.funding import funding_path
from crypto_agents.journal import JsonlJournal
from crypto_agents.market import MarketDataError
from crypto_agents.selection import (
    DEFAULT_MANIFEST,
    HISTORY_DIR,
    load_manifest,
    load_selection_histories,
    write_manifest,
)
from crypto_agents.state import AgentRole, FrozenModel
from tests.test_forward import (
    DOCS,
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
from tests.test_forward_analysis import (
    CHANGED_PATH,
    LATE,
    build_study,
    only_pass,
    population_of,
    rewrite_meta,
    text,
)
from tests.test_forward_analysis import run as analyze_study
from tests.test_net_outcomes import mutated

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
    from pathlib import Path

    from crypto_agents.forward_analysis import StudyPaths
    from crypto_agents.forward_auto import StudyState
    from crypto_agents.forward_record import BlockDates
    from crypto_agents.journal import EvaluationRecord
    from crypto_agents.llm import Completion
    from crypto_agents.state import LLMCall, LLMOutput
    from tests.test_forward_analysis import Study

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
        ("first: datetime = AUTO_FIRST_BLOCK", "first: datetime = FIRST_BLOCK"),
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


async def assert_an_unverifiable_upstream_is_a_normal_notice(tmp_path: Path) -> None:
    """Un lote cuyo decisor contestó sin cabecera, como el anterior: se avisa como normal, en
    `notices`, y no necesita al operador."""
    statuses = {BLOCK2: UpstreamStatus.UNVERIFIABLE, BLOCK3: UpstreamStatus.UNCHANGED}

    async def runner(start: datetime) -> BatchResult:
        return BatchResult(0, ("El lote salió con 0.",), (), statuses[start])

    report = await module.run_auto(study(tmp_path), STAGE1, lambda: BOTH, runner)

    assert report.alerts == ()
    assert not report.needs_operator
    assert len(report.notices) == 1, report.notices
    notice = report.notices[0]
    assert notice.startswith("bloque 2 (2026-10-29): el upstream del decisor no es verificable")
    assert "No hay asiento que preparar" in notice
    lines = render_report(report)
    assert lines[lines.index("Avisos:") + 1] == f"- {notice}"
    assert "Avisos para el operador:" not in lines


@pytest.mark.asyncio
async def test_an_unverifiable_upstream_is_a_normal_notice(tmp_path: Path) -> None:
    await assert_an_unverifiable_upstream_is_a_normal_notice(tmp_path)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("            notices.append(\n", "            alerts.append(\n"),
        ("if result.upstream is UpstreamStatus.UNVERIFIABLE:", "if False:"),
    ],
    ids=["cuenta para el operador", "no se avisa"],
)
@pytest.mark.asyncio
async def test_mutation_an_unverifiable_upstream_told_as_a_failure_or_untold_is_caught(
    old: str, new: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "run_auto", mutated(run_auto, old, new, module))
    with pytest.raises(AssertionError):
        await assert_an_unverifiable_upstream_is_a_normal_notice(tmp_path)


def assert_the_report_reads_in_a_wrapper_older_than_the_notices(tmp_path: Path) -> None:
    """La envoltura lee el informe con el `AutoReport` que cargó antes de su `git pull`. Uno sin
    `notices` (`extra="forbid"`) tiene que leer el informe de un `auto` nuevo sin avisos."""
    fields = {
        name: (field.annotation, field)
        for name, field in AutoReport.model_fields.items()
        if name != "notices"
    }
    older = create_model("OlderReport", __base__=FrozenModel, **fields)  # type: ignore[call-overload]

    quiet = forward.write_auto_report(tmp_path / "quiet.json", AutoReport(at=BOTH))
    assert "notices" not in json.loads(quiet.read_text("utf-8"))
    older.model_validate_json(quiet.read_text("utf-8"))
    assert AutoReport.model_validate_json(quiet.read_text("utf-8")) == AutoReport(at=BOTH)

    told = AutoReport(at=BOTH, notices=("bloque 2: no verificable",))
    loud = forward.write_auto_report(tmp_path / "loud.json", told)
    assert json.loads(loud.read_text("utf-8"))["notices"] == ["bloque 2: no verificable"]
    assert AutoReport.model_validate_json(loud.read_text("utf-8")) == told


def test_the_report_reads_in_a_wrapper_older_than_the_notices(tmp_path: Path) -> None:
    assert_the_report_reads_in_a_wrapper_older_than_the_notices(tmp_path)


def test_mutation_an_empty_notices_key_written_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = 'empty = {"notices"} if not report.notices else None'
    mutant = mutated(forward.write_auto_report, old, "empty = None", forward)
    monkeypatch.setattr(forward, "write_auto_report", mutant)
    with pytest.raises(AssertionError):
        assert_the_report_reads_in_a_wrapper_older_than_the_notices(tmp_path)


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

TIMER_ENTRY = "El temporizador de los lotes entra en master"
WATCH_ENTRY = "La comprobación semanal del estudio entra en master"
REVISED_ENTRY = "Velas revisadas por el exchange, lote del bloque 1"
UPSTREAM_ENTRY = "Bloque 1: el upstream del decisor no se puede verificar"
PROBE_ENTRY = "El sondeo del upstream y «no verificable» entran en master"
NEW_ENTRIES = (TIMER_ENTRY, WATCH_ENTRY, REVISED_ENTRY, UPSTREAM_ENTRY, PROBE_ENTRY)
"""Los asientos de los PR #26, #27 y #28 y del lote del bloque 1, en el orden en que pasaron."""

BLOCK1_BATCH = batch_directory(ROOT / BATCH_DIR, FIRST_BLOCK)
STOPPED_PASS = BLOCK1_BATCH / "20261030T000029Z"
RAN_PASS = BLOCK1_BATCH / "20261030T000327Z"


def register() -> str:
    return DOCS.read_text("utf-8").split("## Registro del estudio", 1)[1]


def register_entry(title: str) -> str:
    """El asiento con ese título, hasta el siguiente, con los espacios normalizados: una cita que
    cae entre dos líneas se lee entera."""
    entry = register().split(f"— {title}\n", 1)[1].split("\n### ", 1)[0]
    return " ".join(entry.split())


def pass_meta(directory: Path) -> BatchMeta:
    return BatchMeta.model_validate_json((directory / FORWARD_META_FILE).read_text("utf-8"))


def to_the_second(instant: datetime) -> str:
    return f"{instant.astimezone(UTC):%Y-%m-%dT%H:%M:%S}Z"


def to_the_minute(instant: datetime) -> str:
    return f"{instant.astimezone(UTC):%Y-%m-%dT%H:%M}Z"


def decider_calls(journal: Path) -> list[tuple[EvaluationRecord, LLMCall]]:
    return [
        (record, call)
        for record in JsonlJournal(journal).read_all()
        for call in record.calls
        if call.role is AgentRole.DECIDER
    ]


def test_the_register_says_the_timer_runs_from_block_two_and_what_stays_manual() -> None:
    entry = register_entry(TIMER_ENTRY)
    for fragment in (
        "PR #26",
        "cd6a6d655bfe4e6799c7c57db9ab71b95b410cf9",
        "ed8407786d29239efc1f7b7b3897f0c325518a69",
        "74a83205eb1daba98dc0dae8f4b641b75d4e981a",
        "2026-10-29T05:43:37Z",
        "pr_merge",
        "/activity",
        "19:00 de Jalisco",
        "Persistent=true",
        "python -m crypto_agents.forward_service",
        "python -m crypto_agents.forward auto",
        "AUTO_FIRST_BLOCK",
        "El lote del bloque 1 lo corre el operador",
        "1, 5, 4 y 7",
        "--accept-revised",
        "--supersede",
        "forward baselines",
        "forward analyze",
        "No cambia el veredicto",
        "git diff --stat ed84077 cd6a6d6",
        "29 módulos congelados",
        "scripts/compare_rehearsals.py",
        "apartado 3",
    ):
        assert fragment in entry, fragment


def test_the_register_says_the_weekly_check_only_watches() -> None:
    entry = register_entry(WATCH_ENTRY)
    for fragment in (
        "PR #27",
        "32850c37b557a13ae13d5892d543cb45b05606a7",
        "cd6a6d655bfe4e6799c7c57db9ab71b95b410cf9",
        "8fd54371cbbec3872a8a49350ce8e90fa1741cd5",
        "2026-10-29T19:37:06Z",
        "pr_merge",
        ".github/workflows/forward-watch.yml",
        "Solo vigila",
        "permissions: contents: read",
        "`first`",
        "AUTO_FIRST_BLOCK",
        "el temporizador sigue empezando en el bloque 2",
        "`FIRST_BLOCK`",
        "comprobación de longitud",
        "git diff --stat ed84077 32850c3",
    ):
        assert fragment in entry, fragment


def test_the_register_copies_block_one_s_revised_candles_from_the_pass_that_stopped() -> None:
    """Las siete velas, con el valor guardado y el devuelto del `forward.json` de la pasada que se
    detuvo; el commit de las dos pasadas y sus horas, leídas de esos mismos archivos."""
    entry = register_entry(REVISED_ENTRY)
    stopped, ran = pass_meta(STOPPED_PASS), pass_meta(RAN_PASS)
    assert stopped.outcome is BatchOutcome.REVISED
    assert {candle.symbol for candle in stopped.revised} == set(DEFAULT_SYMBOLS)
    for candle in stopped.revised:
        assert f"**{candle.symbol}**, vela que abre el {to_the_minute(candle.opens)}" in entry
        assert f"guardado: {forward._values(candle.stored)}" in entry
        assert f"devuelto: {forward._values(candle.returned)}" in entry
    accepted = {(item.symbol, item.opens, item.stored, item.returned) for item in ran.accepted}
    assert accepted == {
        (item.symbol, item.opens, item.stored, item.returned) for item in stopped.revised
    }
    assert stopped.git_commit == ran.git_commit
    for fragment in (
        STOPPED_PASS.name,
        RAN_PASS.name,
        "--accept-revised",
        f"`git_commit` `{stopped.git_commit}`",
        "Este asiento se escribió después de relanzar",
        f"se detuvo a las {to_the_second(stopped.invoked_at)}",
        f"empezó a las {to_the_second(ran.accepted[0].accepted_at)}",
        f"Su `invoked_at`, {to_the_second(ran.invoked_at)}",
    ):
        assert fragment in entry, fragment


def test_the_register_counts_block_one_s_headerless_decider_from_its_journal() -> None:
    """Las 42 llamadas vivas sin cabecera, la primera que lo muestra y la última de la etapa 1 salen
    de los journals; el `git_dirty` de las dos pasadas, de sus `forward.json`."""
    entry = register_entry(UPSTREAM_ENTRY)
    calls = decider_calls(RAN_PASS / "solo.jsonl")
    live = [(record, call) for record, call in calls if not call.cache_hit]
    assert len(live) == len(calls) == 42
    assert all(call.upstream_model is None and call.upstream_endpoint is None for _, call in live)
    record, first = min(live, key=lambda pair: pair[1].at)
    stage1 = [call for _, call in decider_calls(STAGE1 / "solo.jsonl") if not call.cache_hit]
    assert len(stage1) == 140
    served = {(call.upstream_model, call.upstream_endpoint) for call in stage1}
    assert served == {("accounts/fireworks/models/glm-5p3", "fireworks")}
    last = max(call.at for call in stage1)
    stopped, ran = pass_meta(STOPPED_PASS), pass_meta(RAN_PASS)
    assert (stopped.git_dirty, ran.git_dirty) == (False, True)
    assert stopped.git_commit == ran.git_commit
    assert ran.git_commit is not None
    for fragment in (
        f"Las {len(live)} llamadas al decisor de `solo`",
        "x-opencode-upstream-model-id",
        "x-opencode-endpoint-id",
        f"`run_id` `{record.run_id}`",
        record.symbol,
        f"Terminó a las {to_the_second(first.at)}",
        f"etapa 1, a las {to_the_minute(last)}",
        f"bloque 1, a las {to_the_minute(first.at)}",
        f"git diff --stat b9676eb {ran.git_commit[:7]} -- src/crypto_agents/llm.py uv.lock",
        "solo en aciertos de caché",
        "minimax-m3",
        "apunta a la pasarela",
        "53 de 54",
        "140 de 140",
        "de 10:20 a 10:23Z",
        "El estudio continúa",
        "separado por upstream",
        "`git_dirty: true`",
        "revisions.json",
        "registró `git_dirty: false`",
    ):
        assert fragment in entry, fragment


def test_the_register_says_what_the_upstream_probe_merge_brought() -> None:
    entry = register_entry(PROBE_ENTRY)
    for fragment in (
        "PR #28",
        "537609fcc2e25201919ad5945be9a4598a25282a",
        "ffc94b045d1fc1436be892c9e6ef7e545e9d93fa",
        "f10d90fd3f71b7a4f15e04d1c2a25b3225127e9f",
        "2026-10-30T17:44:24Z",
        "pr_merge",
        "scripts/probe_upstream.py",
        "fuera del estudio",
        "«no verificable»",
        "aviso normal",
        "aviso urgente",
        "excluye `data/forward/`",
        "`meta.json` sigue siendo el estado crudo del árbol",
        "29 módulos congelados",
    ):
        assert fragment in entry, fragment


def test_the_register_entries_are_in_chronological_order() -> None:
    """Las fechas de los asientos no bajan, y los cinco nuevos van al final en el orden en que
    pasaron: los merges del #26 y del #27, las dos pasadas del bloque 1 y el merge del #28."""
    headings = re.findall(r"^### (\d{4}-\d{2}-\d{2}) — (.+)$", register(), flags=re.MULTILINE)
    dates = [date for date, _ in headings]
    assert dates == sorted(dates)
    assert tuple(title for _, title in headings[-len(NEW_ENTRIES) :]) == NEW_ENTRIES
    assert dates[-len(NEW_ENTRIES) :] == ["2026-10-29"] * 2 + ["2026-10-30"] * 3


def test_the_new_entries_take_no_committer_date_and_no_local_time() -> None:
    """Las horas de los merges salen de `/activity` y las de las pasadas, de sus `forward.json`:
    ningún asiento nuevo cita la fecha de committer ni una hora con el desfase local."""
    for title in NEW_ENTRIES:
        entry = register_entry(title)
        assert "committer" not in entry.lower(), title
        assert "-06:00" not in entry, title


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
            "if reading.status not in DONE and not baselines_complete(paths, start):",
            "if False:",
        ),
        (
            "study_state",
            "if reading.status not in DONE and not baselines_complete(paths, start):",
            "if reading.status not in DONE:",
        ),
        (
            "study_state",
            "if reading.status not in DONE and not baselines_complete(paths, start):",
            "if not baselines_complete(paths, start):",
        ),
        ("study_state", "first: datetime = AUTO_FIRST_BLOCK", "first: datetime = FIRST_BLOCK"),
        (
            "baselines_complete",
            "if not manifest.is_file():\n        return False",
            "if not manifest.is_file():\n        return True",
        ),
        (
            "baselines_complete",
            "except DuplicateRecordError:\n        return False",
            "except DuplicateRecordError:\n        return True",
        ),
        (
            "baselines_complete",
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


# ──────────────────────── El registro de un bloque vacío, leído por el análisis ───────────────────

ELEVEN = 10
"""La posición (desde 0) del bloque 11, el vacío de los estudios sintéticos del análisis."""

WRITERS = ("lote", "líneas base")
"""Quién registra vacío un bloque: el lote dentro de su plazo, o `forward baselines` al vencer."""

LATER = 2
"""Los bloques que pasan antes de que `forward baselines` registre el bloque vacío en la prueba de
las series: los lotes de esos bloques ya alargaron las series, y el primero aceptó una vela."""

FIRST_ROW = datetime(2024, 8, 16, tzinfo=UTC)
CANDLE = timedelta(hours=4)
SETTLEMENT = timedelta(hours=8)


class Preparation(NamedTuple):
    """Cuándo prepara un escritor el bloque vacío, y lo que su registro copia de la preparación."""

    at: datetime
    history: tuple[SeriesFile, ...] = ()
    funding: tuple[SeriesFile, ...] = ()
    accepted: tuple[AcceptedRevision, ...] = ()


def series_file(symbol: str, path: Path, step: timedelta, last: datetime) -> SeriesFile:
    """Un archivo extendido hasta `last`, desde la primera fila del histórico. El sha-256 es el de
    su ruta y su última fila: otro alargamiento, otro archivo."""
    return SeriesFile(
        symbol=symbol,
        path=path.as_posix(),
        sha256=hashlib.sha256(f"{path}|{last.isoformat()}".encode()).hexdigest(),
        rows=(last - FIRST_ROW) // step + 1,
        first=FIRST_ROW,
        last=last,
    )


def preparation(dates: BlockDates, writer: str, series: bool) -> Preparation:
    """Sin `series`, el registro no copia ninguna serie: el lote prepara el bloque al abrirse y
    `forward baselines` al vencer. Con `series`, el lote lo prepara a su hora, con las series hasta
    la apertura del bloque; `forward baselines`, `LATER` bloques más tarde, sobre las series que
    alargaron los lotes posteriores y con la vela que aceptó el primero de ellos: la última que
    guardó el lote de este bloque, que el exchange devolvió después con otro cierre."""
    if not series:
        return Preparation(dates.ready if writer == "lote" else dates.deadline)
    later = dates if writer == "lote" else block_dates(dates.start + LATER * STEP)
    symbols = sorted(DEFAULT_SYMBOLS)
    history = tuple(
        series_file(s, history_path(s, "4h", FORWARD_HISTORY_DIR), CANDLE, later.ready - CANDLE)
        for s in symbols
    )
    funding = tuple(
        series_file(s, funding_path(s, FORWARD_FUNDING_DIR), SETTLEMENT, later.ready - SETTLEMENT)
        for s in symbols
    )
    if writer == "lote":
        return Preparation(dates.ready, history, funding)
    accepted = AcceptedRevision(
        symbol="BTC/USDT",
        opens=dates.ready - CANDLE,
        stored=(1.0, 2.0, 0.5, 1.5, 10.0),
        returned=(1.0, 2.0, 0.5, 1.6, 10.0),
        accepted_at=block_dates(dates.start + STEP).ready,
        block=dates.start + STEP,
    )
    return Preparation(later.ready, history, funding, (accepted,))


async def register_empty(root: Path, position: int, writer: str, series: bool = False) -> None:
    """El bloque `position` (desde 0) registrado vacío con la función del lote o con la de
    `forward baselines`, cada una con su `preparation`."""
    dates = block_dates(FIRST_BLOCK + position * STEP)
    prepared = preparation(dates, writer, series)
    block = PreparedBlock(
        block=dates,
        history=prepared.history,
        funding=prepared.funding,
        histories={},
        counts=(0,),
        manifest=None,
        manifest_file=None,
        accepted=prepared.accepted,
    )
    if writer == "lote":
        env = environment(root, FakeDecider(), clock=TickingClock(prepared.at))
        result = await run_pass(block, env, ("batch",))
    else:
        env = no_model_environment(root)._replace(clock=TickingClock(prepared.at))
        result = await forward.baselines_pass(block, env, ("baselines",))
    assert result.code == 0, result.lines


BRANCHES = ("veredicto", "abierto", "bloque 52 sin 50", "terminado")
"""Las cuatro ramas del análisis, cada una sobre un estudio sintético con el bloque 11 vacío: cierra
en el 51 con veredicto; 49 bloques, abierto; 52 bloques con 49 no vacíos; y terminado por una
pasada del bloque 21 que llamó con el camino medido cambiado."""


def branch_study(root: Path, monkeypatch: pytest.MonkeyPatch, branch: str) -> Study:
    if branch == "abierto":
        return build_study(root, monkeypatch, blocks=49)
    if branch == "bloque 52 sin 50":
        return build_study(root, monkeypatch, blocks=MAX_BLOCKS, empty=(ELEVEN, 30, 40))
    found = build_study(root, monkeypatch)
    if branch == "terminado":
        rewrite_meta(only_pass(root, 20), measured_path=CHANGED_PATH)
    return found


async def analyses_by_writer(
    root: Path, monkeypatch: pytest.MonkeyPatch, branch: str, series: bool = False
) -> dict[str | None, list[str]]:
    """Lo que imprime `analyze` sobre el mismo estudio sin registro del bloque 11 (`None`) y con el
    registro que deja cada escritor, uno cada vez, preparado como dice `preparation`."""
    found = branch_study(root, monkeypatch, branch)
    lines: dict[str | None, list[str]] = {None: text(analyze_study(found)).splitlines()}
    for writer in WRITERS:
        await register_empty(root, ELEVEN, writer, series)
        try:
            lines[writer] = text(analyze_study(found)).splitlines()
        finally:
            recorded = batch_directory(root / "batches", FIRST_BLOCK + ELEVEN * STEP)
            if recorded.is_dir():
                shutil.rmtree(recorded)
    return lines


async def assert_the_analysis_reads_both_records_alike(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """En las cuatro ramas, `analyze` imprime lo mismo, línea a línea, lo registre el lote o
    `forward baselines`. La fila del bloque no lleva ninguna fecha del registro: su inicio y su
    plazo son los del bloque, así que tampoco esa fecha cambia."""
    for branch in BRANCHES:
        lines = await analyses_by_writer(root / branch, monkeypatch, branch)
        assert lines["lote"] == lines["líneas base"], branch


@pytest.mark.asyncio
async def test_the_analysis_reads_both_records_of_an_empty_block_alike(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await assert_the_analysis_reads_both_records_alike(tmp_path, monkeypatch)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (
            "write_forward_meta(directory, meta)\n            recorded = directory",
            "recorded = directory",
        ),
        (
            "write_forward_meta(directory, meta)\n            recorded = directory",
            "ended = {'outcome': BatchOutcome.TERMINATED}\n"
            "            write_forward_meta(directory, meta.model_copy(update=ended))\n"
            "            recorded = directory",
        ),
        (
            "meta = base_meta(block, now, prepared.history, prepared.funding, prepared.accepted)\n"
            "            write_forward_meta",
            "meta = base_meta(block_dates(block.start + timedelta(days=7)), now)\n"
            "            write_forward_meta",
        ),
    ],
    ids=["no lo registra", "lo registra como fin del estudio", "registra el bloque siguiente"],
)
@pytest.mark.asyncio
async def test_mutation_a_record_the_analysis_reads_otherwise_is_caught(
    old: str, new: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mutant = mutated(forward.baselines_pass, old, new, forward)
    monkeypatch.setattr(forward, "baselines_pass", mutant)
    with pytest.raises(AssertionError):
        await assert_the_analysis_reads_both_records_alike(tmp_path, monkeypatch)


async def recorded_preparations(root: Path) -> dict[str, BatchMeta]:
    """El `forward.json` que deja cada escritor sobre el bloque 11 con las series de
    `preparation`, cada uno en su directorio."""
    found: dict[str, BatchMeta] = {}
    for writer in WRITERS:
        await register_empty(root / writer, ELEVEN, writer, series=True)
        ((_, meta),) = pass_metas(study(root / writer).batches)
        found[writer] = meta
    return found


async def assert_a_record_prepared_later_reads_alike(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`forward baselines` registra vacío el bloque 11 dos bloques tarde, sobre series que los lotes
    posteriores alargaron y con una vela que aceptó uno de ellos; el lote lo registró a su hora.
    Los dos registros copian series, financiación y velas aceptadas distintas, y aun así `analyze`
    imprime lo mismo con uno que con otro en las cuatro ramas: en un bloque vacío no las lee."""
    by_batch, by_baselines = (await recorded_preparations(root / "registros")).values()
    assert by_batch.accepted == ()
    assert [item.symbol for item in by_baselines.accepted] == ["BTC/USDT"]
    for late, early in zip(by_baselines.history, by_batch.history, strict=True):
        assert (late.symbol, late.path) == (early.symbol, early.path)
        assert late.rows == early.rows + LATER * 7 * 6
        assert late.sha256 != early.sha256
    for late, early in zip(by_baselines.funding, by_batch.funding, strict=True):
        assert late.rows == early.rows + LATER * 7 * 3
        assert late.sha256 != early.sha256
    for branch in BRANCHES:
        lines = await analyses_by_writer(root / branch, monkeypatch, branch, series=True)
        assert lines["lote"] == lines["líneas base"], branch


@pytest.mark.asyncio
async def test_a_record_prepared_later_reads_alike(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await assert_a_record_prepared_later_reads_alike(tmp_path, monkeypatch)


SOURCES_END = "    if paths.revisions.is_file():"
"""Donde el análisis acaba de citar sus fuentes, antes de `revisions.json`."""


@pytest.mark.parametrize(
    "cited",
    [
        "f'- `{item.path}` sha-256 `{item.sha256}`' for item in meta.history",
        "f'- `{item.path}` hasta {item.last.isoformat()}' for item in meta.funding",
        "f'- vela aceptada {item.symbol} {item.opens.isoformat()}' for item in meta.accepted",
    ],
    ids=["las series", "la financiación", "las velas aceptadas"],
)
@pytest.mark.asyncio
async def test_mutation_an_analysis_that_reads_the_preparation_is_caught(
    cited: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """El análisis cita, de cada `forward.json`, lo que copió de la preparación. Es una mutación en
    memoria del código congelado: el archivo no cambia."""
    new = (
        "    for _, meta in pass_metas(paths.batches):\n"
        f"        lines.extend({cited})\n"
        f"{SOURCES_END}"
    )
    mutant = mutated(analysis_module._source_lines, SOURCES_END, new, analysis_module)
    monkeypatch.setattr(analysis_module, "_source_lines", mutant)
    with pytest.raises(AssertionError):
        await assert_a_record_prepared_later_reads_alike(tmp_path, monkeypatch)


ELEVEN_ROW = "| 11 | 2026-12-31 | 0 | 0 | — | — | — | 2027-01-21 | — |"
"""La fila del bloque 11 que añade su registro de bloque vacío en la rama terminada."""


async def assert_the_record_changes_two_lines_of_an_ended_study(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """En la rama terminada, el registro del bloque vacío añade su fila a los contadores, entre la
    del 10 y la del 12, y lo cuenta en la línea de resumen. Nada más cambia: ni el veredicto ni su
    causa. Las otras tres ramas no lo leen."""
    lines = await analyses_by_writer(root, monkeypatch, "terminado")
    without = lines[None]
    twelve = next(index for index, line in enumerate(without) if line.startswith("| 12 | "))
    summary = without.index("- bloques: 50, 50 no vacíos y 0 vacíos")
    expected = [
        *without[:twelve],
        ELEVEN_ROW,
        *without[twelve:summary],
        "- bloques: 51, 50 no vacíos y 1 vacíos (los bloques 11)",
        *without[summary + 1 :],
    ]
    assert lines["líneas base"] == expected


@pytest.mark.asyncio
async def test_the_record_changes_two_lines_of_an_ended_study(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await assert_the_record_changes_two_lines_of_an_ended_study(tmp_path, monkeypatch)


@pytest.mark.asyncio
async def test_mutation_an_empty_block_left_unrecorded_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = "write_forward_meta(directory, meta)\n            recorded = directory"
    mutant = mutated(forward.baselines_pass, old, "recorded = directory", forward)
    monkeypatch.setattr(forward, "baselines_pass", mutant)
    with pytest.raises(AssertionError):
        await assert_the_record_changes_two_lines_of_an_ended_study(tmp_path, monkeypatch)


# ──────────────────────── El cierre, contado también con el último bloque ─────────────────────────

CLOSINGS = {
    "en el 50": ((51, 52), 50),
    "en el 51": ((11, 52), 51),
    "en el 52": ((11, 31), 52),
    "tope del 52 sin 50": ((11, 31, 41), None),
}
"""Los bloques vacíos (desde 1) de cada estudio sintético de 52 bloques, y en cuál cierra según el
código congelado (`None`: el bloque 52 terminó sin 50 no vacíos)."""


@pytest.fixture(scope="module")
def closings(tmp_path_factory: pytest.TempPathFactory) -> Iterator[list[tuple[str, Study]]]:
    """Dos estudios por caso de `CLOSINGS`, armados una vez y solo leídos. En el primero el lote
    registra vacío el primer bloque vacío y los demás quedan sin registrar; en el segundo
    `forward baselines` registra los demás y el primero queda sin registrar."""
    root = tmp_path_factory.mktemp("cierres")
    found: list[tuple[str, Study]] = []
    with pytest.MonkeyPatch.context() as patch:
        for name, (empty, _) in CLOSINGS.items():
            for writer, recorded in (("lote", empty[:1]), ("líneas base", empty[1:])):
                where = root / f"{name}, {writer}"
                positions = [number - 1 for number in empty]
                built = build_study(where, patch, blocks=MAX_BLOCKS, empty=positions)
                for number in recorded:
                    asyncio.run(register_empty(where, number - 1, writer))
                found.append((name, built))
    yield found


def frozen_closing(found: Study, monkeypatch: pytest.MonkeyPatch) -> tuple[int | None, int]:
    """El bloque de cierre y los no vacíos según el código congelado: `study_closing` sobre el
    histórico, como lo llama `analyze`, y los bloques que entran en el análisis: hasta el de cierre,
    o los 52 si no cierra."""
    monkeypatch.setattr(analysis_module, "study_population", population_of(found.entries))
    closing = analysis_module.study_closing(read_histories(found.root / "history"))
    last = MAX_BLOCKS if closing.block is None else closing.block
    return closing.block, sum(1 for count in closing.counts[:last] if count)


def assert_the_timer_closes_where_the_analysis_does(
    found: Sequence[tuple[str, Study]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`forward_auto` da el mismo bloque de cierre y el mismo número de no vacíos que el código
    congelado, con bloques vacíos registrados y sin registrar, en el 50, el 51, el 52 y en el tope
    del 52 sin 50 no vacíos."""
    for name, built in found:
        block, non_empty = frozen_closing(built, monkeypatch)
        assert block == CLOSINGS[name][1], name
        state = module.study_state(study(built.root), STAGE1, LATE)
        assert (state.closed_at, state.non_empty) == (block, non_empty), name
        assert state.exhausted is (block is None), name


def assert_the_watch_closes_where_the_timer_does(
    found: Sequence[tuple[str, Study]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Con `first=FIRST_BLOCK`, la llamada de la comprobación semanal, el cierre, los no vacíos y
    `exhausted` son los del código congelado y los de la llamada sin `first`: `first` decide qué
    bloques se leen, no cuáles cuentan para el cierre."""
    for name, built in found:
        block, non_empty = frozen_closing(built, monkeypatch)
        state = module.study_state(study(built.root), STAGE1, LATE)
        watched = module.study_state(study(built.root), STAGE1, LATE, first=FIRST_BLOCK)
        assert (watched.closed_at, watched.non_empty) == (block, non_empty), name
        assert watched.exhausted is (block is None), name
        closing = (state.closed_at, state.non_empty, state.exhausted)
        assert (watched.closed_at, watched.non_empty, watched.exhausted) == closing, name


def test_the_timer_closes_where_the_analysis_does(
    closings: list[tuple[str, Study]], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert_the_timer_closes_where_the_analysis_does(closings, monkeypatch)
    assert_the_watch_closes_where_the_timer_does(closings, monkeypatch)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (
            "    if not settled:\n        return found(None, None)\n"
            "    closed = closing_block(filled)\n"
            "    return found(closed, None, exhausted=closed is None)",
            "    return found(None, None)",
        ),
        ("            sum(filled),", "            len(filled),"),
        (
            "filled.append(1 if manifest_path(start, paths.manifests).is_file() else 0)",
            "filled.append(1)",
        ),
        (
            "filled.append(1 if manifest_path(start, paths.manifests).is_file() else 0)",
            "filled.append(1 if start >= first and manifest_path(start, paths.manifests)"
            ".is_file() else 0)",
        ),
    ],
    ids=[
        "sin contar el último bloque",
        "los bloques por no vacíos",
        "todos no vacíos",
        "first filtra también el cierre",
    ],
)
def test_mutation_a_timer_that_closes_elsewhere_is_caught(
    old: str,
    new: str,
    closings: list[tuple[str, Study]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "study_state", mutated(study_state, old, new, module))
    with pytest.raises(AssertionError):
        assert_the_timer_closes_where_the_analysis_does(closings, monkeypatch)


def test_mutation_block_one_empty_only_for_the_watch_is_caught(
    closings: list[tuple[str, Study]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Con `first=FIRST_BLOCK`, el bloque 1 cuenta como vacío para el cierre. La llamada sin
    `first` no cambia, así que solo la comparación de la variante con `first` lo ve."""
    old = "filled.append(1 if manifest_path(start, paths.manifests).is_file() else 0)"
    new = (
        "filled.append(0 if start == first == FIRST_BLOCK else 1 if manifest_path(start, "
        "paths.manifests).is_file() else 0)"
    )
    monkeypatch.setattr(module, "study_state", mutated(study_state, old, new, module))
    assert_the_timer_closes_where_the_analysis_does(closings, monkeypatch)
    with pytest.raises(AssertionError):
        assert_the_watch_closes_where_the_timer_does(closings, monkeypatch)


LAST = block_dates(FIRST_BLOCK + (MAX_BLOCKS - 1) * STEP)
"""El bloque 52."""


def closing_of(root: Path, now: datetime) -> tuple[int | None, bool, int]:
    state = module.study_state(study(root), STAGE1, now)
    return state.closed_at, state.exhausted, state.non_empty


def assert_the_last_block_counts_once_it_is_known(tmp_path: Path) -> None:
    """Con 49 no vacíos en los bloques 1 a 51, el bloque 52 decide. Sin manifiesto y dentro de su
    plazo, el estudio sigue abierto y el 52 pendiente; con manifiesto, cierra en él; registrado
    vacío, o sin manifiesto con el plazo vencido, termina sin 50 no vacíos."""
    unknown, known = tmp_path / "sin manifiesto", tmp_path / "con manifiesto"
    touch_manifests(unknown, MAX_BLOCKS - 1, skip={11, 31})
    touch_manifests(known, MAX_BLOCKS, skip={11, 31})
    assert closing_of(unknown, LAST.ready) == (None, False, 49)
    state = module.study_state(study(unknown), STAGE1, LAST.ready)
    assert LAST.start in [reading.dates.start for reading in state.pending]
    assert closing_of(unknown, LAST.deadline) == (None, True, 49)
    assert closing_of(known, LAST.ready) == (52, False, 50)
    asyncio.run(register_empty(unknown, MAX_BLOCKS - 1, "lote"))
    assert closing_of(unknown, LAST.ready) == (None, True, 49)


def test_the_last_block_counts_once_it_is_known(tmp_path: Path) -> None:
    assert_the_last_block_counts_once_it_is_known(tmp_path)


@pytest.mark.parametrize(
    "new",
    [
        "settled = True",
        "settled = bool(filled[-1]) or now >= dates.deadline",
        "settled = bool(filled[-1]) or reading.status is BlockStatus.EMPTY",
    ],
    ids=["antes de saberlo", "sin el registro de vacío", "sin el plazo"],
)
def test_mutation_the_last_block_counted_too_soon_or_never_is_caught(
    new: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = (
        "settled = bool(filled[-1]) or reading.status is BlockStatus.EMPTY or now >= dates.deadline"
    )
    monkeypatch.setattr(module, "study_state", mutated(study_state, old, new, module))
    with pytest.raises(AssertionError):
        assert_the_last_block_counts_once_it_is_known(tmp_path)


def dry_run_of(root: Path, now: datetime) -> list[str]:
    return module.render_state(module.study_state(study(root), STAGE1, now))


def assert_the_dry_run_says_how_many_and_where_it_closed(tmp_path: Path) -> None:
    """`auto --dry-run` dice siempre cuántos bloques no vacíos lleva el estudio, también terminado;
    si cerró, en qué bloque; y si el 52 terminó sin 50 no vacíos, eso."""
    touch_manifests(tmp_path / "abierto", 30)
    shown = dry_run_of(tmp_path / "abierto", block_dates(FIRST_BLOCK + 30 * STEP).ready)
    assert "- bloques no vacíos: 30; el estudio cierra con 50." in shown
    assert not [line for line in shown if "cerró" in line or "ya no cierra" in line]

    touch_manifests(tmp_path / "cerrado", 51, skip={11})
    shown = dry_run_of(tmp_path / "cerrado", LAST.ready)
    assert "- bloques no vacíos: 50; el estudio cierra con 50." in shown
    assert "- el estudio cerró en el bloque 51: se avisaría." in shown

    touch_manifests(tmp_path / "sin cerrar", MAX_BLOCKS, skip={11, 31, 41})
    shown = dry_run_of(tmp_path / "sin cerrar", LAST.ready)
    assert "- bloques no vacíos: 49; el estudio cierra con 50." in shown
    assert (
        "- el bloque 52 terminó sin 50 bloques no vacíos: el estudio ya no cierra, y se avisaría."
    ) in shown

    touch_manifests(tmp_path / "terminado", 3)
    asyncio.run(end_the_study(tmp_path / "terminado", "camino medido"))
    shown = dry_run_of(tmp_path / "terminado", BOTH)
    assert "- bloques no vacíos: 3; el estudio cierra con 50." in shown
    assert any(line.startswith("- el estudio terminó en el bloque 2") for line in shown)


def test_the_dry_run_says_how_many_and_where_it_closed(tmp_path: Path) -> None:
    assert_the_dry_run_says_how_many_and_where_it_closed(tmp_path)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (
            '        f"- bloques no vacíos: {state.non_empty}; el estudio cierra con '
            '{TARGET_BLOCKS}.",\n',
            "",
        ),
        ("if state.closed_at is not None:", "if False:"),
        ("if state.exhausted:", "if False:"),
    ],
    ids=["sin los no vacíos", "sin el bloque de cierre", "sin el tope del 52"],
)
def test_mutation_a_dry_run_that_hides_the_closing_is_caught(
    old: str, new: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "render_state", mutated(module.render_state, old, new, module))
    with pytest.raises(AssertionError):
        assert_the_dry_run_says_how_many_and_where_it_closed(tmp_path)


async def closing_notice(root: Path, now: datetime) -> tuple[str | None, list[str]]:
    """El aviso del cierre de `auto`, y sus avisos para el operador que no son de un bloque."""
    report = await module.run_auto(study(root), STAGE1, lambda: now, Completes())
    return report.closing, [alert for alert in report.alerts if not alert.startswith("bloque ")]


CLOSED_ALERT = (
    "el estudio cerró en el bloque 51, con 50 bloques no vacíos: los bloques posteriores no son "
    "del estudio (apartado 2)."
)
SHORT_ALERT = (
    "el bloque 52, el último del estudio, terminó con 49 bloques no vacíos, menos de 50: el "
    "estudio ya no cierra (apartado 2)."
)


async def assert_a_closed_study_is_warned_every_run(tmp_path: Path) -> None:
    """Mientras el estudio está cerrado, cada ejecución lo dice con el bloque de cierre y los no
    vacíos, leídos del disco: dos días seguidos el mismo aviso, y el disco igual. No es un aviso
    para el operador. Abierto, no hay aviso."""
    fifty_one = block_dates(FIRST_BLOCK + 50 * STEP)
    touch_manifests(tmp_path / "abierto", 49)
    assert await closing_notice(tmp_path / "abierto", fifty_one.ready) == (None, [])

    closed = tmp_path / "cerrado"
    touch_manifests(closed, 51, skip={11})
    before = snapshot(closed)
    for day in (0, 1):
        now = LAST.ready + timedelta(days=day)
        assert await closing_notice(closed, now) == (CLOSED_ALERT, []), day
    assert snapshot(closed) == before

    touch_manifests(tmp_path / "sin cerrar", MAX_BLOCKS, skip={11, 31, 41})
    for day in (0, 1):
        now = LAST.ready + timedelta(days=day)
        assert await closing_notice(tmp_path / "sin cerrar", now) == (SHORT_ALERT, []), day


@pytest.mark.asyncio
async def test_a_closed_study_is_warned_every_run(tmp_path: Path) -> None:
    await assert_a_closed_study_is_warned_every_run(tmp_path)


@pytest.mark.parametrize(
    ("function", "old", "new"),
    [
        ("run_auto", "closing=_closing_alert(state),", "closing=None,"),
        ("_closing_alert", "if state.exhausted:", "if False:"),
        ("_closing_alert", "terminó con {state.non_empty}", "terminó con {TARGET_BLOCKS}"),
    ],
    ids=["no avisa", "no avisa del tope", "sin los no vacíos"],
)
@pytest.mark.asyncio
async def test_mutation_a_closed_study_left_unwarned_is_caught(
    function: str, old: str, new: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, function, mutated(getattr(module, function), old, new, module))
    with pytest.raises(AssertionError):
        await assert_a_closed_study_is_warned_every_run(tmp_path)


ENDED = {
    "cerrado": (51, (ELEVEN,), CLOSED_ALERT),
    "tope del 52": (MAX_BLOCKS, (ELEVEN, 30, 40), SHORT_ALERT),
}
"""Dos estudios sintéticos sin nada más que avisar, con sus bloques y sus vacíos (posiciones desde
0): cerrado en el 51, y el 52 terminado con 49 no vacíos. El lote registra vacíos los vacíos, así
que ninguno vence sin sus líneas base. Y el aviso de cada uno."""


@pytest.fixture(scope="module")
def ended(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Los estudios de `ENDED`, armados una vez."""
    root = tmp_path_factory.mktemp("acabados")
    found: dict[str, Path] = {}
    with pytest.MonkeyPatch.context() as patch:
        for name, (blocks, empty, _) in ENDED.items():
            where = root / name
            build_study(where, patch, blocks=blocks, empty=empty)
            for position in empty:
                asyncio.run(register_empty(where, position, "lote"))
            found[name] = where
    return found


def assert_a_closed_study_is_not_a_failure(
    ended: Mapping[str, Path], reports: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Un estudio cerrado, o con el bloque 52 terminado sin cerrar, no es un fallo: cada día `auto`
    sale con 0, imprime el aviso fuera de los avisos para el operador, y el informe que lee la
    envoltura lo lleva en `closing`, con `alerts` vacío."""
    for name, root in ended.items():
        notice = ENDED[name][2]
        for day in (0, 1):
            report = reports / f"{name}, día {day}.json"
            env = refusing_environment(root, LATE + timedelta(days=day))
            assert main(["auto", "--report", str(report)], env) == 0, (name, day)
            printed = capsys.readouterr().out.splitlines()
            assert f"- {notice}" in printed, (name, day)
            assert "Avisos para el operador:" not in printed, (name, day)
            found = AutoReport.model_validate_json(report.read_text(encoding="utf-8"))
            assert (found.closing, found.alerts) == (notice, ()), (name, day)


def test_a_closed_study_is_not_a_failure(
    ended: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert_a_closed_study_is_not_a_failure(ended, tmp_path, capsys)


ALERTS_FROM_BLOCKS = "alerts = [_blocked_alert(reading, now) for reading in state.blocked]"


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (
            ALERTS_FROM_BLOCKS,
            "alerts = [] if state.closed_at is None else [_closing_alert(state)]\n"
            "    alerts += [_blocked_alert(reading, now) for reading in state.blocked]",
        ),
        (
            ALERTS_FROM_BLOCKS,
            "alerts = [_closing_alert(state)] if state.exhausted else []\n"
            "    alerts += [_blocked_alert(reading, now) for reading in state.blocked]",
        ),
    ],
    ids=["el cierre para el operador", "el tope para el operador"],
)
def test_mutation_a_closing_or_a_cap_for_the_operator_is_caught(
    old: str,
    new: str,
    ended: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """El cierre, o el tope, entre los avisos para el operador: `auto` sale con 1."""
    monkeypatch.setattr(forward, "run_auto", mutated(module.run_auto, old, new, module))
    with pytest.raises(AssertionError):
        assert_a_closed_study_is_not_a_failure(ended, tmp_path, capsys)


@pytest.mark.parametrize(
    "new",
    [
        "return 1 if found.needs_operator or found.closed_at is not None else 0",
        "return 1 if found.needs_operator or found.closing is not None and found.closed_at is None "
        "else 0",
    ],
    ids=["el cierre sale con 1", "el tope sale con 1"],
)
def test_mutation_a_closing_or_a_cap_that_exits_one_is_caught(
    new: str,
    ended: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    old = "return 1 if found.needs_operator else 0"
    monkeypatch.setattr(forward, "auto_command", mutated(forward.auto_command, old, new, forward))
    with pytest.raises(AssertionError):
        assert_a_closed_study_is_not_a_failure(ended, tmp_path, capsys)


# ─────────────────── Un error al contar el cierre no se lee como un estudio abierto ───────────────

BEFORE_LAST = LAST.ready - timedelta(days=1)
"""Con el lote del 51 abierto y el del 52 todavía cerrado: el estudio «cerrado» de `ENDED` ya cerró
en el 51."""

LOOP = "for position in range(MAX_BLOCKS):\n"
START = "        start = FIRST_BLOCK + position * STEP\n"
FROM_BLOCK_TWO = "for position in range(1, MAX_BLOCKS):\n"
"""Un error al contar: el recorrido empieza en el bloque 2, el 1 no entra en la lista del cierre y
la lista se queda un bloque corta, con `first` y sin él."""

COUNT_CHECK = (
    "        if len(filled) != position:\n"
    "            raise ForwardError(\n"
    '                f"la lista del cierre mide {len(filled)} antes del bloque {position + 1}, y '
    'debe "\n'
    '                f"medir {position}: un error al contar, no un estudio abierto"\n'
    "            )\n"
)


def miscounting(*, checked: bool) -> Callable[..., StudyState]:
    """`study_state` con el recorrido desde el bloque 2, en memoria; con la comprobación de la
    longitud, o sin ella."""
    if checked:
        return mutated(study_state, LOOP, FROM_BLOCK_TWO, module)
    return mutated(study_state, f"{LOOP}{START}{COUNT_CHECK}", f"{FROM_BLOCK_TWO}{START}", module)


def assert_a_miscount_fails_in_auto(
    root: Path, reports: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`auto` sale con 1 y deja el error en los avisos de su informe, sin aviso de cierre."""
    report = reports / "informe.json"
    assert main(["auto", "--report", str(report)], refusing_environment(root, BEFORE_LAST)) == 1
    capsys.readouterr()
    found = AutoReport.model_validate_json(report.read_text(encoding="utf-8"))
    assert found.closing is None
    assert found.alerts == (
        "auto no corrió ningún lote: la lista del cierre mide 0 antes del bloque 2, y debe medir "
        "1: un error al contar, no un estudio abierto",
    )


def assert_a_miscount_fails_in_the_watch(root: Path, reports: Path) -> None:
    """La comprobación semanal sale con 2 y lo dice en su resumen."""
    summary = reports / "resumen.md"
    env = WatchEnvironment(clock=lambda: BEFORE_LAST, paths=study(root), stage1=STAGE1)
    assert watch_main(["--summary", str(summary)], env) == 2
    assert (
        "- Resultado: **falla**: el estudio no se pudo leer: la lista del cierre mide 0 antes del "
        "bloque 2, y debe medir 1: un error al contar, no un estudio abierto"
    ) in summary.read_text(encoding="utf-8").splitlines()


def test_a_miscount_is_a_failure_not_an_open_study(
    ended: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Con el código como está, la lista del cierre lleva siempre un elemento por bloque anterior.
    Para probar la comprobación, el recorrido empieza en el bloque 2 y el 1 queda fuera de la
    lista: `study_state` no lo lee como un estudio abierto, sino como un error, que `auto` deja en
    su informe y con el que la comprobación semanal falla."""
    root = ended["cerrado"]
    for first in (AUTO_FIRST_BLOCK, FIRST_BLOCK):
        assert module.study_state(study(root), STAGE1, BEFORE_LAST, first=first).closed_at == 51
    mutant = miscounting(checked=True)
    monkeypatch.setattr(module, "study_state", mutant)
    monkeypatch.setattr(watch_module, "study_state", mutant)
    assert_a_miscount_fails_in_auto(root, tmp_path, capsys)
    assert_a_miscount_fails_in_the_watch(root, tmp_path)


def test_mutation_a_miscount_read_as_open_is_caught(
    ended: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Sin la comprobación, el `except ValueError` toma la negativa de `closing_block` ante la lista
    corta por la de un estudio abierto: el estudio, que cerró en el 51, se lee abierto con el 52 por
    venir, y ni `auto` ni la comprobación semanal fallan."""
    root = ended["cerrado"]
    mutant = miscounting(checked=False)
    for first in (AUTO_FIRST_BLOCK, FIRST_BLOCK):
        state = mutant(study(root), STAGE1, BEFORE_LAST, first=first)
        assert (state.closed_at, state.upcoming) == (None, LAST)
    monkeypatch.setattr(module, "study_state", mutant)
    monkeypatch.setattr(watch_module, "study_state", mutant)
    with pytest.raises(AssertionError):
        assert_a_miscount_fails_in_auto(root, tmp_path, capsys)
    with pytest.raises(AssertionError):
        assert_a_miscount_fails_in_the_watch(root, tmp_path)


# ──────────────────────── `--dry-run` con media línea, por el camino de `auto` ────────────────────


def assert_the_dry_run_warns_like_auto(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Con media línea en un journal, el ensayo lee por el mismo camino que `auto`: imprime lo
    mismo, en la salida y en el error, sale con el mismo código y sin traceback."""
    asyncio.run(interrupted(tmp_path, TickingClock(block_dates(BLOCK2).ready)))
    solo = passes_of(tmp_path, BLOCK2)[-1] / "solo.jsonl"
    with solo.open("a", encoding="utf-8") as handle:
        handle.write('{"run_id": "')
    env = refusing_environment(tmp_path, BOTH)
    capsys.readouterr()

    assert main(["auto"], env) == 1
    ran = capsys.readouterr()
    assert main(["auto", "--dry-run"], env) == 1
    dry = capsys.readouterr()

    assert "- auto no corrió ningún lote: " in ran.out
    assert ran.err.startswith("error: ")
    assert (dry.out, dry.err) == (ran.out, ran.err)


def test_the_dry_run_warns_like_auto(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(forward, "offline_checks", passing())
    assert_the_dry_run_warns_like_auto(tmp_path, capsys)


def test_mutation_a_dry_run_outside_the_path_of_auto_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """El ensayo fuera del `try` de `auto`, como estaba: el journal ilegible sale como traceback."""
    monkeypatch.setattr(forward, "offline_checks", passing())
    old = (
        "    try:\n        if dry_run:\n"
        '            print("\\n".join(render_state(study_state(paths, env.paths.stage1, '
        "env.clock()))))\n"
        "            return 0\n"
    )
    new = (
        "    if dry_run:\n"
        '        print("\\n".join(render_state(study_state(paths, env.paths.stage1, '
        "env.clock()))))\n"
        "        return 0\n    try:\n"
    )
    monkeypatch.setattr(forward, "auto_command", mutated(forward.auto_command, old, new, forward))
    with pytest.raises(AuditError):
        assert_the_dry_run_warns_like_auto(tmp_path, capsys)
