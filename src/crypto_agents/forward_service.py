"""La envoltura del temporizador: git, el inhibidor de suspensión y los avisos alrededor de `auto`.

    python -m crypto_agents.forward_service [--check]

La lanza cada día `deploy/crypto-agents-forward.service` (instalación en `deploy/README.md`). El
orden:

1. **Antes de correr.** La rama tiene que ser `master`, el árbol no puede tener cambios fuera de
   `data/forward/`, y `git pull --ff-only origin master` tiene que poder. Si algo falla, no corre
   nada y avisa.
2. **Durante.** `python -m crypto_agents.forward auto --report <informe>`, bajo
   `systemd-inhibit --what=sleep:idle`, en un proceso nuevo: así corre el código que acaba de
   bajar el pull, no el que cargó esta envoltura. Si el informe no aparece, `auto` no corrió (el
   inhibidor denegado, o una caída), y se avisa.
3. **Después de cada lote completo.**
   - `git add` de exactamente los archivos que el lote listó para versionar, todos bajo
     `data/forward/`;
   - un commit de solo esas rutas, con un mensaje fijo: bloque, evaluaciones y código;
   - `git push origin master`.

   Un commit que falla no deja push. Un push que falla deja el commit en local, y se avisa.
4. **Avisos.** `notify-send` y una línea en `var/forward/auto.log`, para un lote completo (solo
   contadores) y para todo lo que necesita al operador. Sin nada pendiente no escribe ni avisa.

Nunca hace `--force`, nunca añade nada fuera de `data/forward/`, y no decide nada de lo que
`docs/forward.md` reserva al operador: eso vive en `auto`, que tampoco lo decide. Los ejecutores de
órdenes y de avisos se inyectan, así que las pruebas ven cada orden que se habría lanzado.

`--check` es la comprobación de la instalación, para correrla desde systemd. No hace pull ni
escribe nada, y no avisa. Comprueba el árbol, la credencial del push
(`git push --dry-run origin master`), el inhibidor y `auto --dry-run`.

No importa el router: un test de arquitectura lo importa en un proceso nuevo.
"""

from __future__ import annotations

import argparse
import fcntl
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, NamedTuple

from pydantic import ValidationError

from crypto_agents.context import utc_now
from crypto_agents.forward_auto import Action, AutoReport, BlockRun

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from datetime import datetime

__all__ = [
    "BRANCH",
    "INHIBIT",
    "LOG_FILE",
    "STUDY_DIR",
    "Completed",
    "ServiceEnvironment",
    "changed_paths",
    "check",
    "commit_message",
    "inside_study",
    "main",
    "notify_send",
    "preflight",
    "run_auto",
    "run_command",
    "serve",
    "service_lock",
    "version_batch",
]

BRANCH = "master"
STUDY_DIR = PurePosixPath("data/forward")
"""Lo único que la envoltura versiona, y lo único que puede estar cambiado al empezar."""

LOG_FILE = Path("var/forward/auto.log")
REPORT_DIR = Path("var/forward/auto")
"""Donde `auto` deja su informe para esta envoltura. Se borra al leerlo."""

SERVICE_LOCK = Path("var/forward/servicio.lock")

INHIBIT = (
    "systemd-inhibit",
    "--what=sleep:idle",
    "--who=crypto-agents",
    "--why=lote semanal del estudio hacia adelante",
    "--mode=block",
)
"""Que la laptop no se suspenda a mitad del lote. Cerrar la tapa la suspende igual: logind ignora
este inhibidor con la tapa (`LidSwitchIgnoreInhibited=yes`), ver `deploy/README.md`."""


class Completed(NamedTuple):
    """Lo que devolvió una orden: código, salida y error."""

    code: int
    stdout: str
    stderr: str


def run_command(command: Sequence[str]) -> Completed:
    """Lanza una orden sin entrada, para que nada pueda quedarse esperando a alguien delante."""
    try:
        done = subprocess.run(
            list(command), capture_output=True, text=True, stdin=subprocess.DEVNULL, check=False
        )
    except OSError as error:
        return Completed(127, "", f"{command[0]}: {error}")
    return Completed(done.returncode, done.stdout, done.stderr)


def notify_send(title: str, body: str, urgent: bool) -> bool:
    """Un aviso de escritorio. Sin sesión gráfica falla, y la línea del log queda igual."""
    urgency = "critical" if urgent else "normal"
    done = run_command(
        ["notify-send", "--app-name=crypto-agents", f"--urgency={urgency}", title, body]
    )
    return done.code == 0


class ServiceEnvironment(NamedTuple):
    """Lo que la envoltura toca fuera de sí misma. Las pruebas lo sustituyen entero."""

    run: Callable[[Sequence[str]], Completed] = run_command
    notify: Callable[[str, str, bool], bool] = notify_send
    clock: Callable[[], datetime] = utc_now
    python: str = sys.executable
    log: Path = LOG_FILE
    reports: Path = REPORT_DIR
    lock: Path = SERVICE_LOCK


# ─────────────────────────────────────────── Avisos ───────────────────────────────────────────────


class Notices:
    """Una línea en el log y un aviso de escritorio por cada cosa que contar."""

    def __init__(self, env: ServiceEnvironment) -> None:
        self.env = env
        self.operator = False

    def tell(self, title: str, body: str, *, urgent: bool) -> None:
        """Lo cuenta. `urgent` es lo que necesita al operador: el servicio sale con 1."""
        shown = self.env.notify(title, body, urgent)
        line = f"{self.env.clock():%Y-%m-%dT%H:%M:%SZ} {title}: {body}"
        if not shown:
            line += " [sin aviso de escritorio]"
        self.env.log.parent.mkdir(parents=True, exist_ok=True)
        with self.env.log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)
        self.operator = self.operator or urgent


# ───────────────────────────────────────────── Git ────────────────────────────────────────────────


def _last_line(done: Completed) -> str:
    text = (done.stderr or done.stdout).strip()
    return text.splitlines()[-1] if text else f"código {done.code}"


def changed_paths(porcelain: str) -> list[str]:
    """Las rutas de `git status --porcelain=v1 -z`, las dos de un renombre incluidas."""
    tokens = porcelain.split("\0")
    paths: list[str] = []
    position = 0
    while position < len(tokens):
        entry = tokens[position]
        position += 1
        if not entry:
            continue
        status, path = entry[:2], entry[3:]
        paths.append(path)
        if "R" in status or "C" in status:
            paths.append(tokens[position])
            position += 1
    return paths


def inside_study(path: str) -> bool:
    """Si la ruta es relativa, está bajo `data/forward/` y no sale de él."""
    pure = PurePosixPath(path)
    return (
        not pure.is_absolute()
        and ".." not in pure.parts
        and len(pure.parts) > len(STUDY_DIR.parts)
        and pure.parts[: len(STUDY_DIR.parts)] == STUDY_DIR.parts
    )


def _status(env: ServiceEnvironment, *paths: str) -> Completed:
    return env.run(["git", "status", "--porcelain=v1", "-z", "--untracked-files=all", "--", *paths])


def preflight(env: ServiceEnvironment) -> str | None:
    """Por qué no se puede correr, o `None`: rama, árbol fuera de `data/forward/` y pull."""
    branch = env.run(["git", "symbolic-ref", "--short", "HEAD"])
    if branch.code != 0 or branch.stdout.strip() != BRANCH:
        return f"la rama no es {BRANCH}: {branch.stdout.strip() or _last_line(branch)}"
    status = _status(env)
    if status.code != 0:
        return f"git status no pudo: {_last_line(status)}"
    outside = [path for path in changed_paths(status.stdout) if not inside_study(path)]
    if outside:
        shown = ", ".join(outside[:5]) + (" …" if len(outside) > 5 else "")
        return f"hay cambios fuera de {STUDY_DIR}/ ({len(outside)}): {shown}"
    pull = env.run(["git", "pull", "--ff-only", "origin", BRANCH])
    if pull.code != 0:
        return f"git pull --ff-only no pudo: {_last_line(pull)}"
    return None


def commit_message(run: BlockRun) -> str:
    """El mensaje fijo del commit de un lote completo: bloque, evaluaciones y código."""
    return (
        f"forward: batch of block {run.number} ({run.start:%Y-%m-%d}), "
        f"{run.evaluations} evaluations, exit code {run.code}"
    )


def _counters(run: BlockRun) -> str:
    return (
        f"{run.evaluations} evaluaciones, {run.decided or 0} decididas, {run.invalid or 0} "
        f"inválidas, {run.missing or 0} faltantes"
    )


def version_batch(env: ServiceEnvironment, run: BlockRun, notices: Notices) -> bool:
    """Versiona y sube un lote completo. Devuelve si se puede seguir con el siguiente.

    Solo las rutas que listó el lote, y solo si todas están bajo `data/forward/`. Un commit que
    falla no deja push, y corta los lotes que quedan. Un push que falla deja el commit en local:
    el siguiente push lo sube.
    """
    label = f"Lote del bloque {run.number} ({run.start:%Y-%m-%d})"
    outside = [path for path in run.files if not inside_study(path)]
    if outside:
        notices.tell(
            label,
            f"listó archivos fuera de {STUDY_DIR}/ ({', '.join(outside)}): no se versiona nada",
            urgent=True,
        )
        return False
    files = list(run.files)
    added = env.run(["git", "add", "--", *files])
    if added.code != 0:
        notices.tell(label, f"git add no pudo: {_last_line(added)}", urgent=True)
        return False
    staged = env.run(["git", "diff", "--cached", "--quiet", "--", *files])
    if staged.code == 0:
        notices.tell(label, f"completo, {_counters(run)}; nada nuevo que versionar", urgent=False)
        return True
    committed = env.run(["git", "commit", "-m", commit_message(run), "--", *files])
    if committed.code != 0:
        notices.tell(
            label, f"completo, pero el commit no pudo: {_last_line(committed)}", urgent=True
        )
        return False
    head = env.run(["git", "rev-parse", "--short", "HEAD"]).stdout.strip()
    pushed = env.run(["git", "push", "origin", BRANCH])
    if pushed.code != 0:
        notices.tell(
            label,
            f"completo, {_counters(run)}; commit {head} hecho y sin subir: {_last_line(pushed)}",
            urgent=True,
        )
        return True
    notices.tell(label, f"completo, {_counters(run)}; commit {head} subido", urgent=False)
    return True


# ───────────────────────────────────────── La ejecución ───────────────────────────────────────────


def _auto(env: ServiceEnvironment, *arguments: str) -> list[str]:
    return [env.python, "-m", "crypto_agents.forward", "auto", *arguments]


def run_auto(env: ServiceEnvironment) -> AutoReport | str:
    """`auto` bajo el inhibidor, en un proceso nuevo. Su informe, o por qué no lo hay."""
    env.reports.mkdir(parents=True, exist_ok=True)
    report = env.reports / f"{env.clock():%Y%m%dT%H%M%SZ}.json"
    report.unlink(missing_ok=True)
    done = env.run([*INHIBIT, *_auto(env, "--report", str(report))])
    for stream, text in ((sys.stdout, done.stdout), (sys.stderr, done.stderr)):
        if text:
            print(text, end="" if text.endswith("\n") else "\n", file=stream, flush=True)
    try:
        return AutoReport.model_validate_json(report.read_text(encoding="utf-8"))
    except (OSError, ValidationError):
        return f"auto no dejó informe (código {done.code}): {_last_line(done)}"
    finally:
        report.unlink(missing_ok=True)


@contextmanager
def service_lock(path: Path) -> Iterator[bool]:
    """El cerrojo de la envoltura, sin esperar: `False` si otra ejecución lo tiene."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def serve(env: ServiceEnvironment) -> int:
    """Una ejecución del servicio. Sale con 0 si nada necesita al operador y con 1 si algo sí."""
    with service_lock(env.lock) as held:
        if not held:
            print(f"otra ejecución de la envoltura tiene {env.lock}: no se hace nada", flush=True)
            return 0
        notices = Notices(env)
        refusal = preflight(env)
        if refusal is not None:
            notices.tell("El temporizador no corrió", refusal, urgent=True)
            return 1
        report = run_auto(env)
        if isinstance(report, str):
            notices.tell("El temporizador falló", report, urgent=True)
            return 1
        for run in report.runs:
            if run.action is Action.NEXT and run.files and not version_batch(env, run, notices):
                break
        for alert in report.alerts:
            notices.tell("El estudio necesita al operador", alert, urgent=True)
        _leftovers(env, report, notices)
        return 1 if notices.operator or report.needs_operator else 0


def _leftovers(env: ServiceEnvironment, report: AutoReport, notices: Notices) -> None:
    """Avisa de archivos de `data/forward/` sin versionar que ninguna ejecución recogerá sola.

    Los de un lote que quedó para la ejecución siguiente se versionan cuando se complete. Sin
    ninguno así, lo que queda solo lo versiona el operador.
    """
    if report.retry:
        return
    status = _status(env, f"{STUDY_DIR}/")
    left = changed_paths(status.stdout) if status.code == 0 else []
    if left:
        notices.tell(
            "Archivos sin versionar",
            f"quedan {len(left)} en {STUDY_DIR}/ que ningún lote pendiente versionará "
            f"(el primero, {left[0]}): los versiona el operador",
            urgent=True,
        )


def check(env: ServiceEnvironment) -> int:
    """La comprobación de la instalación: sin pull, sin escribir nada y sin avisos.

    Comprueba la rama y el árbol, la credencial del push con `git push --dry-run`, que el
    inhibidor se puede tomar y lo que `auto --dry-run` haría ahora. Sale con 0 si todo está bien.
    """
    branch = env.run(["git", "symbolic-ref", "--short", "HEAD"])
    status = _status(env)
    outside = [path for path in changed_paths(status.stdout) if not inside_study(path)]
    push = env.run(["git", "push", "--dry-run", "origin", BRANCH])
    inhibit = env.run([*INHIBIT, "true"])
    dry = env.run(_auto(env, "--dry-run"))
    rows = [
        ("rama", branch.code == 0 and branch.stdout.strip() == BRANCH, branch.stdout.strip()),
        (
            f"árbol fuera de {STUDY_DIR}/",
            status.code == 0 and not outside,
            "limpio" if status.code == 0 and not outside else f"{len(outside)} cambios",
        ),
        ("credencial del push", push.code == 0, _last_line(push)),
        ("inhibidor de suspensión", inhibit.code == 0, _last_line(inhibit)),
        ("auto --dry-run", dry.code == 0, f"código {dry.code}"),
    ]
    for name, passed, detail in rows:
        print(f"- {name}: {'sí' if passed else '**NO**'} — {detail}")
    if dry.stdout:
        print("", dry.stdout.rstrip(), sep="\n")
    return 0 if all(passed for _, passed, _ in rows) else 1


def main(argv: Sequence[str] | None = None, env: ServiceEnvironment | None = None) -> int:
    """Punto de entrada: `python -m crypto_agents.forward_service [--check]`."""
    parser = argparse.ArgumentParser(
        prog="python -m crypto_agents.forward_service",
        description="La envoltura del temporizador de los lotes del estudio.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="comprueba la instalación sin pull, sin escribir y sin avisar",
    )
    args = parser.parse_args(argv)
    environment = env if env is not None else ServiceEnvironment()
    return check(environment) if args.check else serve(environment)


if __name__ == "__main__":
    sys.exit(main())
