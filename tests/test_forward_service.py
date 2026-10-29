"""La envoltura del temporizador: git, el inhibidor y los avisos alrededor de `auto`.

Cada orden pasa por un ejecutor simulado que la apunta, así que las pruebas ven lo que se habría
lanzado; la red y el reloj del sondeo también son simulados. Lo que se fija, cada parte con su
mutación:

- se niega con otra rama, con el árbol sucio fuera de `data/forward/`, sin red y con un pull que
  no puede, sin lanzar `auto`;
- añade y confirma exactamente lo que listó el lote, y nada fuera de `data/forward/`;
- sin lote completo no hay commit, un commit que falla no deja push, y cada git que falla o agota
  su tiempo deja una línea y un aviso;
- dos ejecuciones sin nada pendiente no hacen nada, y con el cerrojo tomado sale con 0 y una línea;
- sin informe de `auto`, avisa y no toca git;
- un aviso de escritorio que falla no rompe nada, y uno que agota su tiempo hace salir con 1;
- el estudio cerrado, o el bloque 52 terminado sin cerrar, se avisa como normal en cada ejecución
  y sale con 0.
"""

from __future__ import annotations

import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import crypto_agents.forward_service as module
from crypto_agents.forward_auto import Action, AutoReport, BlockRun, BlockStatus
from crypto_agents.forward_service import (
    CLOSING_TITLE,
    GIT_TIMEOUT,
    INHIBIT,
    NETWORK_INTERVAL,
    NETWORK_WAIT,
    NOTIFY_TIMEOUT,
    TIMEOUT_CODE,
    Completed,
    ServiceEnvironment,
    Shown,
    changed_paths,
    check,
    commit_message,
    inside_study,
    main,
    remote_address,
    serve,
    service_lock,
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
ORIGIN = "https://github.com/AlexMartGong/Cripto_Agents.git"
"""La url de `origin` en la laptop: la del pull, la que se sondea."""


def timed_out(*command: str) -> Completed:
    """Lo que devuelve `run_command` cuando una orden agota su tiempo."""
    return Completed(TIMEOUT_CODE, "", f"{' '.join(command)} agotó {GIT_TIMEOUT:g} s", True)


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
        if verb == "remote":
            return Completed(0, f"{ORIGIN}\n", "")
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
    def __init__(self, works: bool = True, answer: Shown | None = None) -> None:
        self.answer = answer if answer is not None else (Shown.YES if works else Shown.NO)
        self.shown: list[tuple[str, str, bool]] = []

    def __call__(self, title: str, body: str, urgent: bool) -> Shown:
        self.shown.append((title, body, urgent))
        return self.answer


class Network:
    """La red y el reloj del sondeo, simulados. Responde a partir del intento `up` (contado desde
    1; `None`, nunca). Cada intento que falla consume `slow` segundos de su tiempo máximo, y cada
    espera que no es de cero dura `late` segundos más de lo pedido."""

    def __init__(self, up: int | None = 1, slow: float = 0.0, late: float = 0.0) -> None:
        self.up, self.slow, self.late = up, slow, late
        self.now = 1000.0
        self.attempts: list[tuple[str, int, float, float]] = []
        self.sleeps: list[float] = []

    def connect(self, host: str, port: int, timeout: float) -> bool:
        self.attempts.append((host, port, timeout, self.now - 1000.0))
        if self.up is not None and len(self.attempts) >= self.up:
            return True
        self.now += min(self.slow, timeout)
        return False

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds + (self.late if seconds > 0 else 0.0)

    def monotonic(self) -> float:
        return self.now


def service(
    tmp_path: Path, shell: Shell, desktop: Desktop | None = None, network: Network | None = None
) -> ServiceEnvironment:
    net = network if network is not None else Network()
    return ServiceEnvironment(
        run=shell,
        notify=desktop if desktop is not None else Desktop(),
        clock=lambda: NOW,
        python="python",
        log=tmp_path / "auto.log",
        reports=tmp_path / "informes",
        lock=tmp_path / "servicio.lock",
        connect=net.connect,
        sleep=net.sleep,
        monotonic=net.monotonic,
    )


def launched_auto(shell: Shell) -> bool:
    return any(command[: len(INHIBIT)] == list(INHIBIT) for command in shell.commands)


def log_lines(tmp_path: Path) -> list[str]:
    path = tmp_path / "auto.log"
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


def report_with(*runs: BlockRun, alerts: Sequence[str] = (), retry: bool = False) -> AutoReport:
    return AutoReport(at=NOW, runs=runs, alerts=tuple(alerts), retry=retry)


# ──────────────────────────────────────────── Antes de correr ─────────────────────────────────────


def assert_both_sides_of_a_rename_are_read() -> None:
    porcelain = " M src/a.py\0?? data/forward/b.json\0R  data/forward/c.csv\0src/d.py\0"
    assert module.changed_paths(porcelain) == [
        "src/a.py",
        "data/forward/b.json",
        "data/forward/c.csv",
        "src/d.py",
    ]


def test_the_porcelain_paths_include_both_sides_of_a_rename() -> None:
    assert_both_sides_of_a_rename_are_read()


def test_mutation_a_rename_read_as_one_path_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    old = 'if "R" in status or "C" in status:'
    monkeypatch.setattr(module, "changed_paths", mutated(changed_paths, old, "if False:", module))
    with pytest.raises(AssertionError):
        assert_both_sides_of_a_rename_are_read()


INSIDE = (
    ("data/forward/history/btcusdt_4h.csv", True),
    ("data/forward/batches/2026-10-29/x/forward.json", True),
    ("data/forward", False),
    ("data/forward/../stage1/meta.json", False),
    ("/home/alex/data/forward/x.csv", False),
    ("/data/forward/x.csv", False),
    ("data/forwardx/y.csv", False),
    ("src/crypto_agents/forward.py", False),
)


def assert_only_paths_under_the_study_count_as_inside() -> None:
    for path, inside in INSIDE:
        assert module.inside_study(path) is inside, path


@pytest.mark.parametrize(("path", "inside"), INSIDE)
def test_only_paths_under_the_study_count_as_inside(path: str, inside: bool) -> None:
    assert inside_study(path) is inside


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ('".." not in pure.parts', "True"),
        ("len(pure.parts) > len(STUDY_DIR.parts)", "len(pure.parts) >= len(STUDY_DIR.parts)"),
        ("pure.parts[: len(STUDY_DIR.parts)] == STUDY_DIR.parts", "True"),
    ],
    ids=["con ..", "el directorio mismo", "otro directorio"],
)
def test_mutation_a_path_outside_read_as_inside_is_caught(
    old: str, new: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "inside_study", mutated(inside_study, old, new, module))
    with pytest.raises(AssertionError):
        assert_only_paths_under_the_study_count_as_inside()


def assert_a_dirty_tree_is_refused(tmp_path: Path) -> None:
    shell = Shell(report_with(complete()), status=" M src/crypto_agents/llm.py\0")
    desktop = Desktop()

    assert module.serve(service(tmp_path, shell, desktop)) == 1

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


def assert_a_refusal_runs_nothing(
    tmp_path: Path, answers: Mapping[str, Completed], reason: str, network: Network | None = None
) -> None:
    """Se niega antes de `auto`: sale con 1, sin pull si no llegó a él, sin lanzar `auto` y sin
    escribir nada en git; deja la razón en un aviso urgente y en una línea del log."""
    shell = Shell(report_with(complete()), answers=answers)
    desktop = Desktop()

    assert module.serve(service(tmp_path, shell, desktop, network)) == 1

    assert not launched_auto(shell)
    assert shell.writes == []
    ((title, body, urgent),) = desktop.shown
    assert (title, urgent) == ("El temporizador no corrió", True)
    assert reason in body, body
    (line,) = log_lines(tmp_path)
    assert reason in line


def test_another_branch_is_refused(tmp_path: Path) -> None:
    answers = {"symbolic-ref": Completed(0, "bloque-x\n", "")}
    assert_a_refusal_runs_nothing(tmp_path, answers, "la rama no es master: bloque-x")


def test_mutation_another_branch_let_through_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = "if branch.stdout.strip() != BRANCH:"
    monkeypatch.setattr(module, "preflight", mutated(module.preflight, old, "if False:", module))
    answers = {"symbolic-ref": Completed(0, "bloque-x\n", "")}
    with pytest.raises(AssertionError):
        assert_a_refusal_runs_nothing(tmp_path, answers, "la rama no es master: bloque-x")


REFUSALS = {
    "rama agotada": (
        {"symbolic-ref": timed_out("git", "symbolic-ref")},
        "git symbolic-ref no pudo: git symbolic-ref agotó 120 s",
        "if branch.code != 0:",
    ),
    "status que falla": (
        {"status": timed_out("git", "status")},
        "git status no pudo: git status agotó 120 s",
        "if status.code != 0:",
    ),
    "pull sin fast-forward": (
        {"pull": Completed(128, "", "fatal: Not possible to fast-forward, aborting.\n")},
        "git pull --ff-only no pudo: fatal: Not possible to fast-forward, aborting.",
        "if pull.code != 0:",
    ),
    "pull agotado": (
        {"pull": timed_out("git", "pull")},
        "git pull --ff-only no pudo: git pull agotó 120 s",
        "if pull.code != 0:",
    ),
    "url de origin que falla": (
        {"remote": timed_out("git", "remote")},
        "git remote get-url origin no pudo: git remote agotó 120 s",
        "if offline is not None:",
    ),
}
"""Cada negativa del preflight: qué responde git, qué dice el aviso, y la guarda que la produce."""


@pytest.mark.parametrize("case", list(REFUSALS))
def test_each_preflight_refusal_runs_nothing(case: str, tmp_path: Path) -> None:
    answers, reason, _ = REFUSALS[case]
    assert_a_refusal_runs_nothing(tmp_path, answers, reason)


@pytest.mark.parametrize("case", list(REFUSALS))
def test_mutation_a_preflight_refusal_let_through_is_caught(
    case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answers, reason, guard = REFUSALS[case]
    monkeypatch.setattr(module, "preflight", mutated(module.preflight, guard, "if False:", module))
    with pytest.raises(AssertionError):
        assert_a_refusal_runs_nothing(tmp_path, answers, reason)


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


def assert_without_a_report_it_warns_and_does_not_touch_git(tmp_path: Path) -> None:
    """El inhibidor denegado: `systemd-inhibit` sale sin lanzar `auto`, y no hay informe."""
    shell = Shell(None)
    desktop = Desktop()

    assert module.serve(service(tmp_path, shell, desktop)) == 1

    assert shell.writes == []
    assert len(desktop.shown) == 1, desktop.shown
    ((_, body, urgent),) = desktop.shown
    assert urgent
    assert "auto no dejó informe" in body


def test_without_a_report_it_warns_and_does_not_touch_git(tmp_path: Path) -> None:
    assert_without_a_report_it_warns_and_does_not_touch_git(tmp_path)


def test_mutation_a_missing_report_left_unwarned_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = 'notices.tell("El temporizador falló", report, urgent=True)\n            return 1'
    monkeypatch.setattr(module, "serve", mutated(serve, old, "return 0", module))
    with pytest.raises(AssertionError):
        assert_without_a_report_it_warns_and_does_not_touch_git(tmp_path)


# ─────────────────────────────────────────────── Después ──────────────────────────────────────────


def assert_only_the_listed_files_are_added(tmp_path: Path) -> None:
    shell = Shell(report_with(complete()))
    desktop = Desktop()

    assert module.serve(service(tmp_path, shell, desktop)) == 0

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


@pytest.mark.parametrize(
    "new",
    [
        '["git", "commit", "-m", commit_message(run)]',
        '["git", "commit", "-a", "-m", commit_message(run)]',
    ],
    ids=["sin pathspec", "todo lo modificado"],
)
def test_mutation_a_commit_of_more_than_the_list_is_caught(
    new: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = '["git", "commit", "-m", commit_message(run), "--", *files]'
    monkeypatch.setattr(module, "version_batch", mutated(module.version_batch, old, new, module))
    with pytest.raises(AssertionError):
        assert_only_the_listed_files_are_added(tmp_path)


def assert_a_listed_file_outside_the_study_versions_nothing(tmp_path: Path) -> None:
    shell = Shell(report_with(complete(files=(*FILES, "src/crypto_agents/llm.py"))))
    desktop = Desktop()

    assert module.serve(service(tmp_path, shell, desktop)) == 1

    assert shell.writes == []
    assert "src/crypto_agents/llm.py" in desktop.shown[0][1]


def test_a_listed_file_outside_the_study_versions_nothing(tmp_path: Path) -> None:
    assert_a_listed_file_outside_the_study_versions_nothing(tmp_path)


def test_mutation_a_listed_file_outside_versioned_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mutant = mutated(module.version_batch, "if outside:", "if False:", module)
    monkeypatch.setattr(module, "version_batch", mutant)
    with pytest.raises(AssertionError):
        assert_a_listed_file_outside_the_study_versions_nothing(tmp_path)


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

    assert module.serve(service(tmp_path, shell, desktop)) == 1

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


def test_mutation_a_failed_commit_that_goes_on_to_the_next_batch_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = "and not version_batch(env, run, notices):\n                break"
    new = "and not version_batch(env, run, notices):\n                continue"
    monkeypatch.setattr(module, "serve", mutated(serve, old, new, module))
    with pytest.raises(AssertionError):
        assert_a_failed_commit_is_not_pushed(tmp_path)


REJECTED = Completed(128, "", "fatal: Could not read from remote repository.\n")


def assert_a_failed_push_keeps_the_commit_and_warns(tmp_path: Path, answer: Completed) -> None:
    """El push que falla, rechazado o agotado: los commits quedan en local, sin force ni rebase,
    y cada lote lo avisa con una línea."""
    shell = Shell(report_with(complete(), complete(BLOCK3)), answers={"push": answer})
    desktop = Desktop()

    assert module.serve(service(tmp_path, shell, desktop)) == 1

    assert len(shell.git("commit")) == 2
    assert shell.git("push") == [["git", "push", "origin", "master"]] * 2
    assert shell.git("pull") == [["git", "pull", "--ff-only", "origin", "master"]]
    assert all(urgent for _, _, urgent in desktop.shown)
    assert "commit abc1234 hecho y sin subir" in desktop.shown[0][1]
    assert _last(answer) in desktop.shown[0][1]
    assert sum("hecho y sin subir" in line for line in log_lines(tmp_path)) == 2


def _last(answer: Completed) -> str:
    return answer.stderr.strip().splitlines()[-1]


@pytest.mark.parametrize(
    "answer", [REJECTED, timed_out("git", "push")], ids=["rechazado", "agotado"]
)
def test_a_failed_push_keeps_the_commit_and_warns(tmp_path: Path, answer: Completed) -> None:
    assert_a_failed_push_keeps_the_commit_and_warns(tmp_path, answer)


def test_mutation_a_failed_push_taken_as_pushed_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mutant = mutated(module.version_batch, "if pushed.code != 0:", "if False:", module)
    monkeypatch.setattr(module, "version_batch", mutant)
    with pytest.raises(AssertionError):
        assert_a_failed_push_keeps_the_commit_and_warns(tmp_path, REJECTED)


def assert_nothing_new_to_version_makes_no_commit(tmp_path: Path) -> None:
    shell = Shell(report_with(complete()), answers={"diff": CLEAN})
    assert module.serve(service(tmp_path, shell)) == 0
    assert shell.git("commit") == shell.git("push") == []


def test_nothing_new_to_version_makes_no_commit(tmp_path: Path) -> None:
    assert_nothing_new_to_version_makes_no_commit(tmp_path)


def test_mutation_a_commit_with_nothing_staged_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mutant = mutated(module.version_batch, "if staged.code == 0:", "if False:", module)
    monkeypatch.setattr(module, "version_batch", mutant)
    with pytest.raises(AssertionError):
        assert_nothing_new_to_version_makes_no_commit(tmp_path)


GIT_FAILURES = {
    "add": (
        {"add": timed_out("git", "add")},
        "git add no pudo: git add agotó 120 s",
        ("if added.code != 0:", "if False:"),
        False,
    ),
    "diff": (
        {"diff": timed_out("git", "diff")},
        "git diff no pudo: git diff agotó 120 s",
        ("if staged.code != 1:", "if False:"),
        False,
    ),
    "rev-parse": (
        {"rev-parse": timed_out("git", "rev-parse")},
        "git rev-parse no pudo: git rev-parse agotó 120 s",
        ("if resolved.code != 0:", "if False:"),
        True,
    ),
}
"""Un git de `version_batch` que falla o agota su tiempo: qué responde, qué dice el aviso, la
guarda que lo produce y si el lote llega a confirmarse y subirse igual."""


def assert_a_git_failure_is_told(tmp_path: Path, case: str) -> None:
    answers, reason, _, committed = GIT_FAILURES[case]
    shell = Shell(report_with(complete()), answers=answers)
    desktop = Desktop()

    assert module.serve(service(tmp_path, shell, desktop)) == 1

    assert any(body == reason for _, body, urgent in desktop.shown if urgent), desktop.shown
    assert any(line.endswith(f": {reason}") for line in log_lines(tmp_path))
    assert (shell.git("commit") != []) is committed
    assert (shell.git("push") != []) is committed


@pytest.mark.parametrize("case", list(GIT_FAILURES))
def test_a_git_failure_while_versioning_is_told(case: str, tmp_path: Path) -> None:
    assert_a_git_failure_is_told(tmp_path, case)


@pytest.mark.parametrize("case", list(GIT_FAILURES))
def test_mutation_a_git_failure_left_untold_is_caught(
    case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old, new = GIT_FAILURES[case][2]
    monkeypatch.setattr(module, "version_batch", mutated(module.version_batch, old, new, module))
    with pytest.raises(AssertionError):
        assert_a_git_failure_is_told(tmp_path, case)


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


CLOSINGS = {
    51: (
        "el estudio cerró en el bloque 51, con 50 bloques no vacíos: los bloques posteriores no "
        "son del estudio (apartado 2)."
    ),
    None: (
        "el bloque 52, el último del estudio, terminó con 49 bloques no vacíos, menos de 50: el "
        "estudio ya no cierra (apartado 2)."
    ),
}
"""Los avisos del cierre que deja `auto` en su informe, por su bloque de cierre: cerrado en el 51,
y el 52 terminado sin 50 bloques no vacíos."""


def assert_a_closed_study_is_a_normal_notice(tmp_path: Path) -> None:
    """El estudio cerrado, o el bloque 52 terminado sin cerrar, no es un fallo: cada ejecución lo
    avisa como normal, con su línea en el log, y sale con 0, así que la unidad no queda en
    `failed`."""
    for closed_at, closing in CLOSINGS.items():
        where = tmp_path / str(closed_at)
        desktop = Desktop()
        report = AutoReport(at=NOW, closed_at=closed_at, closing=closing)
        env = service(where, Shell(report), desktop)
        assert module.serve(env) == 0, closing
        assert module.serve(env) == 0, closing
        assert desktop.shown == [(CLOSING_TITLE, closing, False)] * 2
        assert log_lines(where) == [f"2026-11-06T01:00:05Z {CLOSING_TITLE}: {closing}"] * 2


def test_a_closed_study_is_a_normal_notice(tmp_path: Path) -> None:
    assert_a_closed_study_is_a_normal_notice(tmp_path)


CLOSING_NOTICE = "notices.tell(CLOSING_TITLE, report.closing, urgent=False)"
SERVICE_EXIT = "return 1 if notices.operator or report.needs_operator else 0"


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (CLOSING_NOTICE, "notices.tell(CLOSING_TITLE, report.closing, urgent=True)"),
        (
            CLOSING_NOTICE,
            "notices.tell(CLOSING_TITLE, report.closing, urgent=report.closed_at is not None)",
        ),
        (
            CLOSING_NOTICE,
            "notices.tell(CLOSING_TITLE, report.closing, urgent=report.closed_at is None)",
        ),
        (CLOSING_NOTICE, "notices.env.notify(CLOSING_TITLE, report.closing, True)"),
        (
            SERVICE_EXIT,
            "return 1 if notices.operator or report.needs_operator or report.closed_at else 0",
        ),
        (
            SERVICE_EXIT,
            "return 1 if notices.operator or report.needs_operator or report.closing and "
            "report.closed_at is None else 0",
        ),
        ("if report.closing is not None:", "if False:"),
    ],
    ids=[
        "urgente",
        "el cierre urgente",
        "el tope urgente",
        "urgente con salida 0",
        "el cierre sale con 1",
        "el tope sale con 1",
        "sin aviso",
    ],
)
def test_mutation_a_closing_told_as_a_failure_or_untold_is_caught(
    old: str, new: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "serve", mutated(module.serve, old, new, module))
    with pytest.raises(AssertionError):
        assert_a_closed_study_is_a_normal_notice(tmp_path)


def assert_a_desktop_without_a_session_does_not_stop_anything(tmp_path: Path) -> None:
    shell = Shell(report_with(complete()))

    assert module.serve(service(tmp_path, shell, Desktop(works=False))) == 0

    assert shell.git("push") == [["git", "push", "origin", "master"]]
    (line,) = log_lines(tmp_path)
    assert line.endswith("[sin aviso de escritorio]")


def test_a_desktop_without_a_session_does_not_stop_anything(tmp_path: Path) -> None:
    assert_a_desktop_without_a_session_does_not_stop_anything(tmp_path)


def test_mutation_a_failed_desktop_notice_left_unmarked_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mutant = mutated(module.Notices.tell, "if shown is not Shown.YES:", "if False:", module)
    monkeypatch.setattr(module.Notices, "tell", mutant)
    with pytest.raises(AssertionError):
        assert_a_desktop_without_a_session_does_not_stop_anything(tmp_path)


def assert_a_desktop_notice_that_times_out_fails_the_run(tmp_path: Path) -> None:
    """`notify-send` que agota sus 10 s: el lote completo se versiona igual, la línea del log lo
    dice, y la ejecución sale con 1 para que la unidad quede en `failed`."""
    shell = Shell(report_with(complete()))

    code = module.serve(service(tmp_path, shell, Desktop(answer=Shown.TIMED_OUT)))

    assert shell.git("push") == [["git", "push", "origin", "master"]]
    (line,) = log_lines(tmp_path)
    assert line.endswith(f"[notify-send agotó {NOTIFY_TIMEOUT:g} s]")
    assert code == 1


def test_a_desktop_notice_that_times_out_fails_the_run(tmp_path: Path) -> None:
    assert_a_desktop_notice_that_times_out_fails_the_run(tmp_path)


def test_mutation_a_timed_out_notice_that_exits_zero_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = "self.operator = self.operator or urgent or shown is Shown.TIMED_OUT"
    new = "self.operator = self.operator or urgent"
    monkeypatch.setattr(module.Notices, "tell", mutated(module.Notices.tell, old, new, module))
    with pytest.raises(AssertionError):
        assert_a_desktop_notice_that_times_out_fails_the_run(tmp_path)


LEFT = "?? data/forward/batches/2026-10-29/20261107T010000Z/forward.json\0"


def assert_files_left_are_warned(tmp_path: Path) -> None:
    """Sin ningún lote para mañana, lo que queda sin versionar en `data/forward/` se avisa; con
    uno, no, porque lo versionará él. Si `git status` no puede, también se avisa."""
    shell = Shell(report_with())
    desktop = Desktop()
    env = service(tmp_path, shell, desktop)
    answer = Completed(0, LEFT, "")

    def answering(command: Sequence[str]) -> Completed:
        if list(command[-2:]) == ["--", "data/forward/"]:
            shell.commands.append(list(command))
            return answer
        return shell(command)

    assert module.serve(env._replace(run=answering)) == 1
    ((_, body, urgent),) = desktop.shown
    assert urgent
    assert "quedan 1 en data/forward/" in body

    desktop.shown.clear()
    shell.report = report_with(retry=True)
    assert module.serve(env._replace(run=answering)) == 0
    assert desktop.shown == []

    shell.report = report_with()
    answer = timed_out("git", "status")
    assert module.serve(env._replace(run=answering)) == 1
    ((_, body, urgent),) = desktop.shown
    assert urgent
    assert body == "git status no pudo, no se sabe si quedan: git status agotó 120 s"


def test_files_left_that_no_pending_batch_will_version_are_warned(tmp_path: Path) -> None:
    assert_files_left_are_warned(tmp_path)


@pytest.mark.parametrize(
    "old",
    ["if report.retry:", "if left:", "if status.code != 0:"],
    ids=["con lote", "sin aviso", "status"],
)
def test_mutation_files_left_misread_is_caught(
    old: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "_leftovers", mutated(module._leftovers, old, "if False:", module))
    with pytest.raises(AssertionError):
        assert_files_left_are_warned(tmp_path)


def assert_a_busy_wrapper_leaves_one_line(tmp_path: Path) -> None:
    """Otra ejecución tiene el cerrojo de la envoltura: sale con 0, sin git, sin `auto` y sin
    aviso de escritorio, y deja una sola línea en el log."""
    shell = Shell(report_with(complete()))
    desktop = Desktop()
    env = service(tmp_path, shell, desktop)

    with service_lock(env.lock) as held:
        assert held
        assert module.serve(env) == 0

    assert shell.commands == []
    assert desktop.shown == []
    lines = log_lines(tmp_path)
    assert len(lines) == 1, lines
    assert lines[0].startswith("2026-11-06T01:00:05Z El temporizador no corrió: otra ejecución")


def test_a_busy_wrapper_exits_zero_with_one_line(tmp_path: Path) -> None:
    assert_a_busy_wrapper_leaves_one_line(tmp_path)


@pytest.mark.parametrize(
    ("old", "new"),
    [("if not held:", "if False:"), ("notices.log(", "print(")],
    ids=["corre igual", "sin línea"],
)
def test_mutation_a_busy_wrapper_misbehaving_is_caught(
    old: str, new: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "serve", mutated(serve, old, new, module))
    with pytest.raises(AssertionError):
        assert_a_busy_wrapper_leaves_one_line(tmp_path)


# ─────────────────────────────────────────────── La red ───────────────────────────────────────────


ADDRESSES = (
    ("https://github.com/AlexMartGong/Cripto_Agents.git", ("github.com", 443)),
    ("https://github.com:8443/x/y.git", ("github.com", 8443)),
    ("http://example.org/x.git", ("example.org", 80)),
    ("ssh://git@github.com/x/y.git", ("github.com", 22)),
    ("ssh://git@github.com:2222/x/y.git", ("github.com", 2222)),
    ("git@github.com:AlexMartGong/Cripto_Agents.git", ("github.com", 22)),
    ("/home/alex/repos/cripto.git", None),
    ("file:///home/alex/repos/cripto.git", None),
    ("../cripto.git", None),
    ("./ruta:con/dos-puntos.git", None),
    ("https://github.com:sin-puerto/x.git", None),
    ("https:///x.git", None),
    ("@:ruta.git", None),
)


def assert_the_remote_address_is_read() -> None:
    for url, address in ADDRESSES:
        try:
            found = module.remote_address(url)
        except ValueError as error:
            raise AssertionError(f"{url}: {error}") from error
        assert found == address, url


def test_the_remote_address_is_read_from_the_url() -> None:
    assert_the_remote_address_is_read()


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("return parts.hostname, port or default", "return parts.hostname, default"),
        (
            'return (host, DEFAULT_PORTS["ssh"]) if host else None',
            "return (host, 443) if host else None",
        ),
        ('if not separator or "/" in user_host:', "if not separator:"),
        (
            "except ValueError:\n            return None",
            "except ZeroDivisionError:\n            return None",
        ),
        ("if parts.hostname is None or default is None:", "if default is None:"),
        (
            'return (host, DEFAULT_PORTS["ssh"]) if host else None',
            'return (host, DEFAULT_PORTS["ssh"])',
        ),
    ],
    ids=[
        "sin el puerto de la url",
        "ssh corto al 443",
        "una ruta local",
        "un puerto que no es número",
        "sin host",
        "ssh corto sin host",
    ],
)
def test_mutation_a_remote_address_misread_is_caught(
    old: str, new: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "remote_address", mutated(remote_address, old, new, module))
    with pytest.raises(AssertionError):
        assert_the_remote_address_is_read()


def test_mutation_https_on_another_port_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "DEFAULT_PORTS", {**module.DEFAULT_PORTS, "https": 22})
    with pytest.raises(AssertionError):
        assert_the_remote_address_is_read()


def probe(network: Network, tmp_path: Path) -> str | None:
    return module.wait_for_network(service(tmp_path, Shell(None), network=network))


def assert_the_network_is_probed_every_thirty_seconds_for_five_minutes(tmp_path: Path) -> None:
    """Al host y puerto de `origin`. Con red al primer intento no espera; al tercero, espera dos
    veces 30 s. Sin red, intenta en 0, 30, …, 270 s, cada intento con el tiempo que queda como
    máximo, y a los 300 s dice por qué. Con esperas que se alargan, ningún intento pasa de 300 s."""
    first = Network(up=1)
    assert probe(first, tmp_path) is None
    assert first.attempts == [("github.com", 443, NETWORK_INTERVAL, 0.0)]
    assert first.sleeps == []

    third = Network(up=3)
    assert probe(third, tmp_path) is None
    assert [at for *_, at in third.attempts] == [0.0, 30.0, 60.0]
    assert third.sleeps == [30.0, 30.0]

    never = Network(up=None)
    reason = probe(never, tmp_path)
    assert [at for *_, at in never.attempts] == [30.0 * k for k in range(10)]
    assert never.sleeps == [30.0] * 9
    assert reason == (
        "sin red: github.com:443 no respondió en 300 s (10 intentos, uno cada 30 s). No corrió "
        "ningún lote; siguen pendientes"
    )

    slow = Network(up=None, slow=30.0)
    assert probe(slow, tmp_path) is not None
    assert [at for *_, at in slow.attempts] == [30.0 * k for k in range(10)]
    assert all(timeout <= NETWORK_INTERVAL for *_, timeout, _ in slow.attempts)

    late = Network(up=None, late=7.0)
    assert probe(late, tmp_path) is not None
    assert all(at + timeout <= NETWORK_WAIT for *_, timeout, at in late.attempts), late.attempts
    assert late.now - 1000.0 <= NETWORK_WAIT


def test_the_network_is_probed_every_thirty_seconds_for_five_minutes(tmp_path: Path) -> None:
    assert_the_network_is_probed_every_thirty_seconds_for_five_minutes(tmp_path)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("min(NETWORK_INTERVAL, left)", "NETWORK_INTERVAL"),
        ("if following >= deadline:\n            break", "if False:\n            break"),
        (
            "following = start + attempts * NETWORK_INTERVAL",
            "following = env.monotonic() + NETWORK_INTERVAL",
        ),
        ("if env.connect(host, port, min(NETWORK_INTERVAL, left)):", "if False:"),
        ("if url.code != 0:", "if False:"),
    ],
    ids=[
        "un intento pasa de 300 s",
        "una espera de más al final",
        "30 s desde el final del intento",
        "nunca hay red",
        "sin la url de origin",
    ],
)
def test_mutation_a_probe_misbehaving_is_caught(
    old: str, new: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mutant = mutated(module.wait_for_network, old, new, module)
    monkeypatch.setattr(module, "wait_for_network", mutant)
    with pytest.raises(AssertionError):
        assert_the_network_is_probed_every_thirty_seconds_for_five_minutes(tmp_path)
        assert_a_refusal_runs_nothing(
            tmp_path / "url",
            {"remote": timed_out("git", "remote")},
            "git remote get-url origin no pudo: git remote agotó 120 s",
        )


@pytest.mark.parametrize(("name", "value"), [("NETWORK_INTERVAL", 60.0), ("NETWORK_WAIT", 600.0)])
def test_mutation_another_probe_schedule_is_caught(
    name: str, value: float, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, name, value)
    with pytest.raises(AssertionError):
        assert_the_network_is_probed_every_thirty_seconds_for_five_minutes(tmp_path)


NOT_A_NETWORK = {"remote": Completed(0, "/home/alex/repos/cripto.git\n", "")}


def test_an_origin_that_is_not_a_network_url_is_refused(tmp_path: Path) -> None:
    reason = "la url de origin no es de red: /home/alex/repos/cripto.git"
    assert_a_refusal_runs_nothing(tmp_path, NOT_A_NETWORK, reason)


def test_mutation_an_origin_that_is_not_a_network_url_let_through_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = 'return f"la url de origin no es de red: {url.stdout.strip()}"'
    mutant = mutated(module.wait_for_network, old, "return None", module)
    monkeypatch.setattr(module, "wait_for_network", mutant)
    with pytest.raises(AssertionError):
        reason = "la url de origin no es de red: /home/alex/repos/cripto.git"
        assert_a_refusal_runs_nothing(tmp_path, NOT_A_NETWORK, reason)


def assert_a_tcp_connection_is_tried_for_real() -> None:
    """`tcp_connect` contra un puerto local que escucha y contra el mismo puerto ya cerrado: abre
    la conexión, o dice que no sin dejar escapar el error. No sale de la máquina."""
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        assert module.tcp_connect("127.0.0.1", port, 1.0)
    try:
        assert not module.tcp_connect("127.0.0.1", port, 1.0)
    except OSError as error:
        raise AssertionError(f"el error de conexión se escapó: {error}") from error


def test_a_tcp_connection_is_tried_for_real() -> None:
    assert_a_tcp_connection_is_tried_for_real()


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (
            "except OSError:\n        return False",
            "except ZeroDivisionError:\n        return False",
        ),
        ("            return True", "            return False"),
    ],
    ids=["el error se escapa", "nunca conecta"],
)
def test_mutation_a_tcp_connection_misread_is_caught(
    old: str, new: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "tcp_connect", mutated(module.tcp_connect, old, new, module))
    with pytest.raises(AssertionError):
        assert_a_tcp_connection_is_tried_for_real()


def test_without_network_nothing_runs_and_it_is_told(tmp_path: Path) -> None:
    """Sin red: una línea en el log, un aviso urgente, salida 1 (la unidad en `failed`) y ningún
    pull ni `auto`: los bloques siguen pendientes."""
    reason = "sin red: github.com:443 no respondió en 300 s"
    assert_a_refusal_runs_nothing(tmp_path, {}, reason, Network(up=None))


def test_mutation_a_run_without_network_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = "if offline is not None:\n        return offline"
    new = "if False:\n        return offline"
    monkeypatch.setattr(module, "preflight", mutated(module.preflight, old, new, module))
    with pytest.raises(AssertionError):
        assert_a_refusal_runs_nothing(
            tmp_path, {}, "sin red: github.com:443 no respondió en 300 s", Network(up=None)
        )


# ───────────────────────────────────────── Los tiempos máximos ────────────────────────────────────


class Recorder:
    """`subprocess.run` simulado: apunta el tiempo máximo de cada orden, y agota el de las que se
    le dicen."""

    def __init__(self, expire: Sequence[str] = (), code: int = 0) -> None:
        self.expire = set(expire)
        self.code = code
        self.timeouts: dict[str, float | None] = {}

    def __call__(self, command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        timeout = kwargs.get("timeout")
        assert timeout is None or isinstance(timeout, float)
        self.timeouts[command[0]] = timeout
        if command[0] in self.expire and timeout is not None:
            raise subprocess.TimeoutExpired(command, timeout)
        return subprocess.CompletedProcess(command, self.code, "", "")


def assert_each_program_has_its_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """120 s para git, 10 s para `notify-send`, y nada para el lote bajo `systemd-inhibit`. Una
    orden que agota el suyo vuelve con `TIMEOUT_CODE` y «agotó» en el error, sin escaparse."""
    recorder = Recorder(expire=["/usr/bin/git", "notify-send"])
    monkeypatch.setattr(subprocess, "run", recorder)
    commands = (["/usr/bin/git", "pull"], ["notify-send", "título"], [*INHIBIT, "python"])
    try:
        found = [module.run_command(command) for command in commands]
    except subprocess.TimeoutExpired as escaped:
        raise AssertionError(f"el tiempo agotado se escapó: {escaped}") from escaped
    assert recorder.timeouts == {
        "/usr/bin/git": GIT_TIMEOUT,
        "notify-send": NOTIFY_TIMEOUT,
        "systemd-inhibit": None,
    }
    assert found[0] == Completed(TIMEOUT_CODE, "", "/usr/bin/git pull agotó 120 s", True)
    assert found[1] == Completed(TIMEOUT_CODE, "", "notify-send agotó 10 s", True)
    assert found[2] == Completed(0, "", "")
    assert module.notify_send("título", "cuerpo", urgent=True) is Shown.TIMED_OUT


def test_each_program_has_its_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    assert_each_program_has_its_timeout(monkeypatch)


@pytest.mark.parametrize(
    ("program", "value"),
    [("git", None), ("notify-send", None), ("git", 600.0), ("systemd-inhibit", 120.0)],
    ids=["git sin tiempo", "notify-send sin tiempo", "git con 600 s", "el lote con tiempo"],
)
def test_mutation_another_timeout_is_caught(
    program: str, value: float | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    timeouts = {**module.TIMEOUTS, program: value}
    monkeypatch.setattr(module, "TIMEOUTS", {k: v for k, v in timeouts.items() if v is not None})
    with pytest.raises(AssertionError):
        assert_each_program_has_its_timeout(monkeypatch)


def assert_a_desktop_notice_is_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """`notify-send` que sale con 0 es un aviso mostrado; con otro código, uno que no llegó."""
    monkeypatch.setattr(subprocess, "run", Recorder(code=0))
    assert module.notify_send("título", "cuerpo", urgent=False) is Shown.YES
    monkeypatch.setattr(subprocess, "run", Recorder(code=1))
    assert module.notify_send("título", "cuerpo", urgent=False) is Shown.NO


def test_a_desktop_notice_is_read(monkeypatch: pytest.MonkeyPatch) -> None:
    assert_a_desktop_notice_is_read(monkeypatch)


@pytest.mark.parametrize("new", ["return Shown.YES", "return Shown.NO"], ids=["siempre", "nunca"])
def test_mutation_a_desktop_notice_misread_is_caught(
    new: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = "return Shown.YES if done.code == 0 else Shown.NO"
    monkeypatch.setattr(module, "notify_send", mutated(module.notify_send, old, new, module))
    with pytest.raises(AssertionError):
        assert_a_desktop_notice_is_read(monkeypatch)


@pytest.mark.parametrize(
    ("function", "old", "new"),
    [
        ("run_command", "except subprocess.TimeoutExpired:", "except ZeroDivisionError:"),
        ("notify_send", "if done.timed_out:", "if False:"),
    ],
    ids=["el tiempo agotado se escapa", "notify-send agotado como fallo"],
)
def test_mutation_a_timeout_misread_is_caught(
    function: str, old: str, new: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, function, mutated(getattr(module, function), old, new, module))
    with pytest.raises(AssertionError):
        assert_each_program_has_its_timeout(monkeypatch)


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


def assert_the_check_prints_what_timed_out(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """En `--check` un git agotado se imprime en su fila y sale con 1: sin línea ni aviso."""
    answers = {verb: timed_out("git", verb) for verb in ("symbolic-ref", "status", "push")}
    desktop = Desktop()

    assert module.check(service(tmp_path, Shell(None, answers=answers), desktop)) == 1

    out = capsys.readouterr().out
    assert "- rama: **NO** — git symbolic-ref agotó 120 s" in out
    assert "- árbol fuera de data/forward/: **NO** — git status agotó 120 s" in out
    assert "- credencial del push: **NO** — git push agotó 120 s" in out
    assert desktop.shown == []
    assert list(tmp_path.iterdir()) == []


def test_the_check_prints_what_timed_out(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert_the_check_prints_what_timed_out(tmp_path, capsys)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("tree = _last_line(status)", 'tree = "limpio"'),
        (
            "branch.stdout.strip() if branch.code == 0 else _last_line(branch)",
            "branch.stdout.strip()",
        ),
    ],
    ids=["el árbol", "la rama"],
)
def test_mutation_a_check_that_hides_a_timeout_is_caught(
    old: str,
    new: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "check", mutated(check, old, new, module))
    with pytest.raises(AssertionError):
        assert_the_check_prints_what_timed_out(tmp_path, capsys)
