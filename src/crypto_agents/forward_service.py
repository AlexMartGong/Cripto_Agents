"""La envoltura del temporizador: git, el inhibidor de suspensión y los avisos alrededor de `auto`.

    python -m crypto_agents.forward_service [--check]

La lanza cada día `deploy/crypto-agents-forward.service` (instalación en `deploy/README.md`). El
orden:

1. **Antes de correr.** La rama tiene que ser `master` y el árbol no puede tener cambios fuera de
   `data/forward/`. Después espera a la red: una conexión TCP al host y puerto de la url de
   `origin`, la del pull, cada 30 s y durante 300 s como máximo (`wait_for_network`). Por último,
   `git pull --ff-only origin master` tiene que poder. Si algo falla, no corre nada, deja una
   línea en el log y avisa; la unidad termina en `failed` y los bloques siguen pendientes.
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
   Con el cerrojo de la envoltura tomado por otra ejecución, sale con 0 y deja solo una línea.
   El estudio cerrado, o el bloque 52 terminado sin cerrar, no es un fallo: un aviso normal en
   cada ejecución, y la salida con 0, así que la unidad no queda en `failed`.

Cada orden de git tiene 120 s como máximo y `notify-send` 10 s (`TIMEOUTS`); el lote no lleva
tiempo propio, lo acota `TimeoutStartSec` de la unidad. Un tiempo agotado deja una línea en el log
y un aviso. Si el que se agota es `notify-send`, el aviso es la propia línea y la salida con 1.

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
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, NamedTuple
from urllib.parse import urlsplit

from pydantic import ValidationError

from crypto_agents.context import utc_now
from crypto_agents.forward_auto import Action, AutoReport, BlockRun

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence
    from datetime import datetime

__all__ = [
    "BRANCH",
    "CLOSING_TITLE",
    "GIT_TIMEOUT",
    "INHIBIT",
    "LOG_FILE",
    "NETWORK_INTERVAL",
    "NETWORK_WAIT",
    "NOTIFY_TIMEOUT",
    "STUDY_DIR",
    "TIMEOUTS",
    "Completed",
    "ServiceEnvironment",
    "Shown",
    "changed_paths",
    "check",
    "commit_message",
    "inside_study",
    "main",
    "notify_send",
    "preflight",
    "remote_address",
    "run_auto",
    "run_command",
    "serve",
    "service_lock",
    "tcp_connect",
    "version_batch",
    "wait_for_network",
]

BRANCH = "master"
STUDY_DIR = PurePosixPath("data/forward")
"""Lo único que la envoltura versiona, y lo único que puede estar cambiado al empezar."""

LOG_FILE = Path("var/forward/auto.log")
REPORT_DIR = Path("var/forward/auto")
"""Donde `auto` deja su informe para esta envoltura. Se borra al leerlo."""

SERVICE_LOCK = Path("var/forward/servicio.lock")

CLOSING_TITLE = "Cierre del estudio"
"""El título del aviso del estudio cerrado, o del bloque 52 terminado sin cerrar: normal, no
urgente."""

INHIBIT = (
    "systemd-inhibit",
    "--what=sleep:idle",
    "--who=crypto-agents",
    "--why=lote semanal del estudio hacia adelante",
    "--mode=block",
)
"""Que la laptop no se suspenda a mitad del lote. Cerrar la tapa la suspende igual: logind ignora
este inhibidor con la tapa (`LidSwitchIgnoreInhibited=yes`), ver `deploy/README.md`."""

GIT_TIMEOUT = 120.0
NOTIFY_TIMEOUT = 10.0
TIMEOUTS: Mapping[str, float] = {"git": GIT_TIMEOUT, "notify-send": NOTIFY_TIMEOUT}
"""Segundos como máximo por programa. Lo que no está aquí, como el lote bajo `systemd-inhibit`, no
lleva tiempo propio: lo acota `TimeoutStartSec` de la unidad."""

TIMEOUT_CODE = 124
"""El código de una orden cortada por su tiempo, el mismo que usa `timeout(1)`."""

NETWORK_INTERVAL = 30.0
NETWORK_WAIT = 300.0
"""El sondeo de la red: un intento de conexión cada 30 s, durante 300 s como máximo."""

DEFAULT_PORTS: Mapping[str, int] = {"https": 443, "http": 80, "ssh": 22, "git": 9418}


class Completed(NamedTuple):
    """Lo que devolvió una orden: código, salida, error y si se cortó por su tiempo."""

    code: int
    stdout: str
    stderr: str
    timed_out: bool = False


def _named(command: Sequence[str]) -> str:
    """La orden como se nombra en un aviso: `git pull`, `notify-send`."""
    return " ".join(command[:2]) if PurePosixPath(command[0]).name == "git" else command[0]


def run_command(command: Sequence[str]) -> Completed:
    """Lanza una orden sin entrada, para que nada pueda quedarse esperando a alguien delante.

    Con el tiempo de `TIMEOUTS` para su programa: agotado, la orden se mata y devuelve
    `TIMEOUT_CODE` con «agotó N s» en el error, que es lo que llega a la línea y al aviso.
    """
    timeout = TIMEOUTS.get(PurePosixPath(command[0]).name)
    try:
        done = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return Completed(TIMEOUT_CODE, "", f"{_named(command)} agotó {timeout:g} s", True)
    except OSError as error:
        return Completed(127, "", f"{command[0]}: {error}")
    return Completed(done.returncode, done.stdout, done.stderr)


class Shown(StrEnum):
    """Qué pasó con un aviso de escritorio."""

    YES = "mostrado"
    NO = "sin aviso de escritorio"
    """`notify-send` falló, como sin sesión gráfica: queda la línea del log."""

    TIMED_OUT = f"notify-send agotó {NOTIFY_TIMEOUT:g} s"
    """`notify-send` no respondió a tiempo: queda la línea, y la ejecución sale con 1."""


def notify_send(title: str, body: str, urgent: bool) -> Shown:
    """Un aviso de escritorio. Sin sesión gráfica falla, y la línea del log queda igual."""
    urgency = "critical" if urgent else "normal"
    done = run_command(
        ["notify-send", "--app-name=crypto-agents", f"--urgency={urgency}", title, body]
    )
    if done.timed_out:
        return Shown.TIMED_OUT
    return Shown.YES if done.code == 0 else Shown.NO


def tcp_connect(host: str, port: int, timeout: float) -> bool:
    """Si se abre una conexión TCP a `host:port` antes de `timeout` segundos. No envía nada."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class ServiceEnvironment(NamedTuple):
    """Lo que la envoltura toca fuera de sí misma. Las pruebas lo sustituyen entero."""

    run: Callable[[Sequence[str]], Completed] = run_command
    notify: Callable[[str, str, bool], Shown] = notify_send
    clock: Callable[[], datetime] = utc_now
    python: str = sys.executable
    log: Path = LOG_FILE
    reports: Path = REPORT_DIR
    lock: Path = SERVICE_LOCK
    connect: Callable[[str, int, float], bool] = tcp_connect
    sleep: Callable[[float], None] = time.sleep
    monotonic: Callable[[], float] = time.monotonic


# ─────────────────────────────────────────── Avisos ───────────────────────────────────────────────


class Notices:
    """Una línea en el log y un aviso de escritorio por cada cosa que contar."""

    def __init__(self, env: ServiceEnvironment) -> None:
        self.env = env
        self.operator = False

    def tell(self, title: str, body: str, *, urgent: bool) -> None:
        """Lo cuenta. `urgent` es lo que necesita al operador: el servicio sale con 1. Un
        `notify-send` que agota su tiempo también: entonces el aviso es la línea."""
        shown = self.env.notify(title, body, urgent)
        line = f"{self.env.clock():%Y-%m-%dT%H:%M:%SZ} {title}: {body}"
        if shown is not Shown.YES:
            line += f" [{shown.value}]"
        self._write(line)
        self.operator = self.operator or urgent or shown is Shown.TIMED_OUT

    def log(self, title: str, body: str) -> None:
        """Solo la línea del log, sin aviso de escritorio y sin cambiar la salida."""
        self._write(f"{self.env.clock():%Y-%m-%dT%H:%M:%SZ} {title}: {body}")

    def _write(self, line: str) -> None:
        self.env.log.parent.mkdir(parents=True, exist_ok=True)
        with self.env.log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)


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


def remote_address(url: str) -> tuple[str, int] | None:
    """El host y el puerto TCP de una url de git, o `None` si no es de red.

    `https://` (443), `http://` (80), `ssh://` (22) y `git://` (9418), con el puerto de la url si
    lo trae; y la forma corta `usuario@host:ruta` de SSH (22).
    """
    if "://" in url:
        parts = urlsplit(url)
        default = DEFAULT_PORTS.get(parts.scheme)
        try:
            port = parts.port
        except ValueError:
            return None
        if parts.hostname is None or default is None:
            return None
        return parts.hostname, port or default
    user_host, separator, _ = url.partition(":")
    if not separator or "/" in user_host:
        return None
    host = user_host.rpartition("@")[2]
    return (host, DEFAULT_PORTS["ssh"]) if host else None


def wait_for_network(env: ServiceEnvironment) -> str | None:
    """Espera a la red, o dice por qué no la hubo.

    Una conexión TCP al host y puerto de la url de `origin`, la que usa el pull, cada
    `NETWORK_INTERVAL` segundos. Cada intento tiene como máximo lo que queda de `NETWORK_WAIT`, así
    que el sondeo entero no pasa de ahí. Al volver de una suspensión el temporizador dispara antes
    de que haya red: sin esto, la descarga fallida se leería como una negativa del lote.
    """
    url = env.run(["git", "remote", "get-url", "origin"])
    if url.code != 0:
        return f"git remote get-url origin no pudo: {_last_line(url)}"
    address = remote_address(url.stdout.strip())
    if address is None:
        return f"la url de origin no es de red: {url.stdout.strip()}"
    host, port = address
    start = env.monotonic()
    deadline = start + NETWORK_WAIT
    attempts = 0
    while (left := deadline - env.monotonic()) > 0:
        attempts += 1
        if env.connect(host, port, min(NETWORK_INTERVAL, left)):
            return None
        following = start + attempts * NETWORK_INTERVAL
        if following >= deadline:
            break
        env.sleep(max(0.0, following - env.monotonic()))
    return (
        f"sin red: {host}:{port} no respondió en {NETWORK_WAIT:g} s ({attempts} intentos, uno "
        f"cada {NETWORK_INTERVAL:g} s). No corrió ningún lote; siguen pendientes"
    )


def preflight(env: ServiceEnvironment) -> str | None:
    """Por qué no se puede correr, o `None`: rama, árbol fuera de `data/forward/`, red y pull."""
    branch = env.run(["git", "symbolic-ref", "--short", "HEAD"])
    if branch.code != 0:
        return f"git symbolic-ref no pudo: {_last_line(branch)}"
    if branch.stdout.strip() != BRANCH:
        return f"la rama no es {BRANCH}: {branch.stdout.strip()}"
    status = _status(env)
    if status.code != 0:
        return f"git status no pudo: {_last_line(status)}"
    outside = [path for path in changed_paths(status.stdout) if not inside_study(path)]
    if outside:
        shown = ", ".join(outside[:5]) + (" …" if len(outside) > 5 else "")
        return f"hay cambios fuera de {STUDY_DIR}/ ({len(outside)}): {shown}"
    offline = wait_for_network(env)
    if offline is not None:
        return offline
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
    if staged.code != 1:
        notices.tell(label, f"git diff no pudo: {_last_line(staged)}", urgent=True)
        return False
    committed = env.run(["git", "commit", "-m", commit_message(run), "--", *files])
    if committed.code != 0:
        notices.tell(
            label, f"completo, pero el commit no pudo: {_last_line(committed)}", urgent=True
        )
        return False
    resolved = env.run(["git", "rev-parse", "--short", "HEAD"])
    if resolved.code != 0:
        notices.tell(label, f"git rev-parse no pudo: {_last_line(resolved)}", urgent=True)
    head = resolved.stdout.strip() if resolved.code == 0 else "sin hash"
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
    """Una ejecución del servicio. Sale con 0 si nada necesita al operador y con 1 si algo sí.

    El cierre del estudio, o el tope del bloque 52, se avisa como normal en cada ejecución y no
    cambia la salida."""
    with service_lock(env.lock) as held:
        notices = Notices(env)
        if not held:
            notices.log(
                "El temporizador no corrió",
                f"otra ejecución de la envoltura tiene {env.lock}: no se hace nada",
            )
            return 0
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
        if report.closing is not None:
            notices.tell(CLOSING_TITLE, report.closing, urgent=False)
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
    if status.code != 0:
        notices.tell(
            "Archivos sin versionar",
            f"git status no pudo, no se sabe si quedan: {_last_line(status)}",
            urgent=True,
        )
        return
    left = changed_paths(status.stdout)
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
    Un git que agota su tiempo sale en su fila, con «agotó»: aquí no se escribe ni se avisa.
    """
    branch = env.run(["git", "symbolic-ref", "--short", "HEAD"])
    status = _status(env)
    outside = [path for path in changed_paths(status.stdout) if not inside_study(path)]
    push = env.run(["git", "push", "--dry-run", "origin", BRANCH])
    inhibit = env.run([*INHIBIT, "true"])
    dry = env.run(_auto(env, "--dry-run"))
    if status.code != 0:
        tree = _last_line(status)
    else:
        tree = "limpio" if not outside else f"{len(outside)} cambios"
    rows = [
        (
            "rama",
            branch.code == 0 and branch.stdout.strip() == BRANCH,
            branch.stdout.strip() if branch.code == 0 else _last_line(branch),
        ),
        (f"árbol fuera de {STUDY_DIR}/", status.code == 0 and not outside, tree),
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
