"""La envoltura del temporizador: git, el inhibidor y los avisos alrededor de `auto`.

Cada orden pasa por un ejecutor simulado que la apunta, así que las pruebas ven lo que se habría
lanzado. Lo que se fija, cada parte con su mutación:

- se niega con el árbol sucio fuera de `data/forward/`, y con un pull que no puede;
- añade exactamente lo que listó el lote, y nada fuera de `data/forward/`;
- sin lote completo no hay commit, y un commit que falla no deja push;
- dos ejecuciones sin nada pendiente no hacen nada;
- sin informe de `auto`, avisa y no toca git;
- un aviso de escritorio que falla no rompe nada.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import crypto_agents.forward_service as module
from crypto_agents.forward_auto import Action, AutoReport, BlockRun, BlockStatus
from crypto_agents.forward_service import (
    INHIBIT,
    Completed,
    ServiceEnvironment,
    changed_paths,
    check,
    commit_message,
    inside_study,
    main,
    serve,
)
from tests.test_net_outcomes import mutated

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

NOW = datetime(2026, 11, 6, 1, 0, 5, tzinfo=UTC)
BLOCK2 = datetime(2026, 10, 29, tzinfo=UTC)
BLOCK3 = datetime(2026, 11, 5, tzinfo=UTC)
FILES = (
    "data/forward/history/btcusdt_4h.csv",
    "data/forward/funding/btcusdt.csv",
    "data/forward/manifests/2026-10-29.json",
    "data/forward/batches/2026-10-29/20261106T010000Z/forward.json",
    "data/forward/batches/2026-10-29/20261106T010000Z/solo.jsonl",
)
CLEAN = Completed(0, "", "")


def complete(start: datetime = BLOCK2, files: Sequence[str] = FILES) -> BlockRun:
    number = 2 if start == BLOCK2 else 3
    return BlockRun(
        start=start,
        number=number,
        code=0,
        action=Action.NEXT,
        headline="## Lote del bloque",
        files=tuple(files),
        status=BlockStatus.COMPLETE,
        evaluations=47,
        decided=46,
        invalid=1,
        missing=0,
    )


def incomplete(files: Sequence[str] = FILES) -> BlockRun:
    return complete(files=files).model_copy(
        update={"code": 3, "action": Action.RETRY, "status": BlockStatus.INCOMPLETE}
    )


class Shell:
    """El ejecutor simulado. `auto` deja `report` donde le dicen; git responde lo de `answers`."""

    def __init__(
        self,
        report: AutoReport | None,
        status: str = "",
        answers: Mapping[str, Completed] | None = None,
    ) -> None:
        self.report = report
        self.status = status
        self.answers = dict(answers or {})
        self.commands: list[list[str]] = []

    def __call__(self, command: Sequence[str]) -> Completed:
        command = list(command)
        self.commands.append(command)
        if command[: len(INHIBIT)] == list(INHIBIT):
            if "--report" in command and self.report is not None:
                path = Path(command[command.index("--report") + 1])
                path.write_text(self.report.model_dump_json(), encoding="utf-8")
            return Completed(0, "salida de auto\n", "")
        verb = command[1] if command[0] == "git" else command[0]
        if verb in self.answers:
            return self.answers[verb]
        if verb == "symbolic-ref":
            return Completed(0, "master\n", "")
        if verb == "status":
            return Completed(0, self.status, "")
        if verb == "diff":
            return Completed(1, "", "")
        if verb == "rev-parse":
            return Completed(0, "abc1234\n", "")
        return CLEAN

    def git(self, *verbs: str) -> list[list[str]]:
        return [
            command for command in self.commands if command[:1] == ["git"] and command[1] in verbs
        ]

    @property
    def writes(self) -> list[list[str]]:
        return self.git("add", "commit", "push")


class Desktop:
    def __init__(self, works: bool = True) -> None:
        self.works = works
        self.shown: list[tuple[str, str, bool]] = []

    def __call__(self, title: str, body: str, urgent: bool) -> bool:
        self.shown.append((title, body, urgent))
        return self.works


def service(tmp_path: Path, shell: Shell, desktop: Desktop | None = None) -> ServiceEnvironment:
    return ServiceEnvironment(
        run=shell,
        notify=desktop if desktop is not None else Desktop(),
        clock=lambda: NOW,
        python="python",
        log=tmp_path / "auto.log",
        reports=tmp_path / "informes",
        lock=tmp_path / "servicio.lock",
    )


def report_with(*runs: BlockRun, alerts: Sequence[str] = (), retry: bool = False) -> AutoReport:
    return AutoReport(at=NOW, runs=runs, alerts=tuple(alerts), retry=retry)


# ──────────────────────────────────────────── Antes de correr ─────────────────────────────────────


def test_the_porcelain_paths_include_both_sides_of_a_rename() -> None:
    porcelain = " M src/a.py\0?? data/forward/b.json\0R  data/forward/c.csv\0src/d.py\0"
    assert changed_paths(porcelain) == [
        "src/a.py",
        "data/forward/b.json",
        "data/forward/c.csv",
        "src/d.py",
    ]


@pytest.mark.parametrize(
    ("path", "inside"),
    [
        ("data/forward/history/btcusdt_4h.csv", True),
        ("data/forward/batches/2026-10-29/x/forward.json", True),
        ("data/forward", False),
        ("data/forward/../stage1/meta.json", False),
        ("/home/alex/data/forward/x.csv", False),
        ("data/forwardx/y.csv", False),
        ("src/crypto_agents/forward.py", False),
    ],
)
def test_only_paths_under_the_study_count_as_inside(path: str, inside: bool) -> None:
    assert inside_study(path) is inside


def assert_a_dirty_tree_is_refused(tmp_path: Path) -> None:
    shell = Shell(report_with(complete()), status=" M src/crypto_agents/llm.py\0")
    desktop = Desktop()

    assert serve(service(tmp_path, shell, desktop)) == 1

    assert shell.git("pull") == []
    assert not any(command[: len(INHIBIT)] == list(INHIBIT) for command in shell.commands)
    assert shell.writes == []
    ((_, body, urgent),) = desktop.shown
    assert urgent
    assert "src/crypto_agents/llm.py" in body
    assert "src/crypto_agents/llm.py" in (tmp_path / "auto.log").read_text(encoding="utf-8")


def test_a_tree_dirty_outside_the_study_is_refused(tmp_path: Path) -> None:
    assert_a_dirty_tree_is_refused(tmp_path)


def test_mutation_a_preflight_that_ignores_the_tree_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        module, "preflight", mutated(module.preflight, "if outside:", "if False:", module)
    )
    with pytest.raises(AssertionError):
        assert_a_dirty_tree_is_refused(tmp_path)


def test_changes_inside_the_study_do_not_stop_it(tmp_path: Path) -> None:
    status = (
        "?? data/forward/batches/2026-10-29/20261106T010000Z/forward.json\0"
        " M data/forward/history/btcusdt_4h.csv\0"
    )
    shell = Shell(report_with(), status=status)

    serve(service(tmp_path, shell))

    assert shell.git("pull") == [["git", "pull", "--ff-only", "origin", "master"]]
    assert any(command[: len(INHIBIT)] == list(INHIBIT) for command in shell.commands)


def test_another_branch_is_refused(tmp_path: Path) -> None:
    shell = Shell(report_with(complete()), answers={"symbolic-ref": Completed(0, "bloque-x\n", "")})
    assert serve(service(tmp_path, shell)) == 1
    assert shell.git("pull") == []
    assert shell.writes == []


def test_a_pull_that_cannot_fast_forward_is_refused(tmp_path: Path) -> None:
    refused = Completed(128, "", "fatal: Not possible to fast-forward, aborting.\n")
    shell = Shell(report_with(complete()), answers={"pull": refused})
    desktop = Desktop()

    assert serve(service(tmp_path, shell, desktop)) == 1

    assert not any(command[: len(INHIBIT)] == list(INHIBIT) for command in shell.commands)
    assert shell.writes == []
    assert "Not possible to fast-forward" in desktop.shown[0][1]


# ─────────────────────────────────────────────── Durante ──────────────────────────────────────────


def test_auto_runs_under_the_inhibitor_with_a_report_that_is_removed(tmp_path: Path) -> None:
    shell = Shell(report_with())

    assert serve(service(tmp_path, shell)) == 0

    (launched,) = [command for command in shell.commands if command[0] == "systemd-inhibit"]
    assert launched[: len(INHIBIT)] == [
        "systemd-inhibit",
        "--what=sleep:idle",
        "--who=crypto-agents",
        "--why=lote semanal del estudio hacia adelante",
        "--mode=block",
    ]
    assert launched[len(INHIBIT) : len(INHIBIT) + 5] == [
        "python",
        "-m",
        "crypto_agents.forward",
        "auto",
        "--report",
    ]
    assert list((tmp_path / "informes").iterdir()) == []


def test_without_a_report_it_warns_and_does_not_touch_git(tmp_path: Path) -> None:
    """El inhibidor denegado: `systemd-inhibit` sale sin lanzar `auto`, y no hay informe."""
    shell = Shell(None)
    desktop = Desktop()

    assert serve(service(tmp_path, shell, desktop)) == 1

    assert shell.writes == []
    ((_, body, urgent),) = desktop.shown
    assert urgent
    assert "auto no dejó informe" in body


# ─────────────────────────────────────────────── Después ──────────────────────────────────────────


def assert_only_the_listed_files_are_added(tmp_path: Path) -> None:
    shell = Shell(report_with(complete()))
    desktop = Desktop()

    assert serve(service(tmp_path, shell, desktop)) == 0

    assert shell.git("add") == [["git", "add", "--", *FILES]]
    message = "forward: batch of block 2 (2026-10-29), 47 evaluations, exit code 0"
    assert shell.git("commit") == [["git", "commit", "-m", message, "--", *FILES]]
    assert shell.git("push") == [["git", "push", "origin", "master"]]
    ((_, body, urgent),) = desktop.shown
    assert not urgent
    assert "47 evaluaciones, 46 decididas, 1 inválidas, 0 faltantes" in body
    assert "commit abc1234 subido" in body


def test_a_complete_batch_adds_commits_and_pushes_only_what_it_listed(tmp_path: Path) -> None:
    assert_only_the_listed_files_are_added(tmp_path)


@pytest.mark.parametrize(
    "new",
    ['["git", "add", "-A"]', '["git", "add", "--", str(STUDY_DIR)]'],
    ids=["todo el árbol", "todo data/forward"],
)
def test_mutation_adding_more_than_the_list_is_caught(
    new: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mutant = mutated(module.version_batch, '["git", "add", "--", *files]', new, module)
    monkeypatch.setattr(module, "version_batch", mutant)
    with pytest.raises(AssertionError):
        assert_only_the_listed_files_are_added(tmp_path)


def test_a_listed_file_outside_the_study_versions_nothing(tmp_path: Path) -> None:
    shell = Shell(report_with(complete(files=(*FILES, "src/crypto_agents/llm.py"))))
    desktop = Desktop()

    assert serve(service(tmp_path, shell, desktop)) == 1

    assert shell.writes == []
    assert "src/crypto_agents/llm.py" in desktop.shown[0][1]


def assert_no_commit_without_a_complete_batch(tmp_path: Path) -> None:
    """Un lote que quedó para mañana también lista archivos: no se versionan."""
    shell = Shell(report_with(incomplete(), retry=True))
    assert module.serve(service(tmp_path, shell)) == 0
    assert shell.writes == []


def test_no_commit_without_a_complete_batch(tmp_path: Path) -> None:
    assert_no_commit_without_a_complete_batch(tmp_path)


def test_mutation_a_commit_for_any_batch_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mutant = mutated(
        module.serve, "if run.action is Action.NEXT and run.files", "if run.files", module
    )
    monkeypatch.setattr(module, "serve", mutant)
    with pytest.raises(AssertionError):
        assert_no_commit_without_a_complete_batch(tmp_path)


def assert_a_failed_commit_is_not_pushed(tmp_path: Path) -> None:
    failed = Completed(1, "", "error: hook rechazó el commit\n")
    shell = Shell(report_with(complete(), complete(BLOCK3)), answers={"commit": failed})
    desktop = Desktop()

    assert serve(service(tmp_path, shell, desktop)) == 1

    assert shell.git("push") == []
    assert len(shell.git("commit")) == 1, "no sigue con el lote siguiente"
    assert "el commit no pudo" in desktop.shown[0][1]


def test_a_failed_commit_is_not_pushed(tmp_path: Path) -> None:
    assert_a_failed_commit_is_not_pushed(tmp_path)


def test_mutation_a_push_after_a_failed_commit_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mutant = mutated(module.version_batch, "if committed.code != 0:", "if False:", module)
    monkeypatch.setattr(module, "version_batch", mutant)
    with pytest.raises(AssertionError):
        assert_a_failed_commit_is_not_pushed(tmp_path)


def test_a_failed_push_keeps_the_commit_and_warns(tmp_path: Path) -> None:
    rejected = Completed(128, "", "fatal: Could not read from remote repository.\n")
    shell = Shell(report_with(complete(), complete(BLOCK3)), answers={"push": rejected})
    desktop = Desktop()

    assert serve(service(tmp_path, shell, desktop)) == 1

    assert len(shell.git("commit")) == 2
    assert all(urgent for _, _, urgent in desktop.shown)
    assert "commit abc1234 hecho y sin subir" in desktop.shown[0][1]


def test_nothing_new_to_version_makes_no_commit(tmp_path: Path) -> None:
    shell = Shell(report_with(complete()), answers={"diff": CLEAN})
    assert serve(service(tmp_path, shell)) == 0
    assert shell.git("commit") == shell.git("push") == []


def test_the_commit_message_is_fixed() -> None:
    assert commit_message(complete()) == (
        "forward: batch of block 2 (2026-10-29), 47 evaluations, exit code 0"
    )


# ─────────────────────────────────────────────── Avisos ───────────────────────────────────────────


def test_two_runs_with_nothing_pending_do_nothing(tmp_path: Path) -> None:
    shell = Shell(report_with())
    desktop = Desktop()
    env = service(tmp_path, shell, desktop)

    assert serve(env) == 0
    assert serve(env) == 0

    assert shell.writes == []
    assert desktop.shown == []
    assert not (tmp_path / "auto.log").exists()


def test_every_alert_of_auto_reaches_the_operator(tmp_path: Path) -> None:
    alerts = ("bloque 2 (2026-10-29): el lote salió con 5: vela revisada", "bloque 3: plazo")
    shell = Shell(report_with(alerts=alerts))
    desktop = Desktop()

    assert serve(service(tmp_path, shell, desktop)) == 1

    assert [body for _, body, _ in desktop.shown] == list(alerts)
    assert all(urgent for _, _, urgent in desktop.shown)
    log = (tmp_path / "auto.log").read_text(encoding="utf-8").splitlines()
    assert len(log) == 2
    assert log[0].startswith("2026-11-06T01:00:05Z El estudio necesita al operador: bloque 2")


def test_a_desktop_without_a_session_does_not_stop_anything(tmp_path: Path) -> None:
    shell = Shell(report_with(complete()))

    assert serve(service(tmp_path, shell, Desktop(works=False))) == 0

    assert shell.git("push") == [["git", "push", "origin", "master"]]
    (line,) = (tmp_path / "auto.log").read_text(encoding="utf-8").splitlines()
    assert line.endswith("[sin aviso de escritorio]")


def test_files_left_that_no_pending_batch_will_version_are_warned(tmp_path: Path) -> None:
    left = "?? data/forward/batches/2026-10-29/20261107T010000Z/forward.json\0"
    shell = Shell(report_with())
    desktop = Desktop()
    env = service(tmp_path, shell, desktop)

    def answering(command: Sequence[str]) -> Completed:
        if list(command[-2:]) == ["--", "data/forward/"]:
            shell.commands.append(list(command))
            return Completed(0, left, "")
        return shell(command)

    assert serve(env._replace(run=answering)) == 1
    ((_, body, urgent),) = desktop.shown
    assert urgent
    assert "quedan 1 en data/forward/" in body

    desktop.shown.clear()
    shell.report = report_with(retry=True)
    assert serve(env._replace(run=answering)) == 0
    assert desktop.shown == []


# ─────────────────────────────────────────────── --check ──────────────────────────────────────────


def test_the_check_asks_the_credential_and_the_inhibitor_without_pull_or_writes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    shell = Shell(None)
    desktop = Desktop()

    assert check(service(tmp_path, shell, desktop)) == 0

    assert shell.git("pull") == shell.git("add") == shell.git("commit") == []
    assert shell.git("push") == [["git", "push", "--dry-run", "origin", "master"]]
    assert [*INHIBIT, "true"] in shell.commands
    assert ["python", "-m", "crypto_agents.forward", "auto", "--dry-run"] in shell.commands
    assert desktop.shown == []
    assert list(tmp_path.iterdir()) == []
    assert "credencial del push: sí" in capsys.readouterr().out


def test_the_check_fails_when_the_push_would_fail(tmp_path: Path) -> None:
    denied = Completed(128, "", "git@github.com: Permission denied (publickey).\n")
    shell = Shell(None, answers={"push": denied})
    assert main(["--check"], service(tmp_path, shell)) == 1
