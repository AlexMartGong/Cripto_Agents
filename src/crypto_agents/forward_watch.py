"""La comprobación semanal del estudio hacia adelante, desde GitHub Actions.

    python -m crypto_agents.forward_watch [--now INSTANTE] [--summary ARCHIVO]

La corre `.github/workflows/forward-watch.yml` los sábados a las 15:17 UTC, sobre el checkout de
`master`, sin secretos y sin red: lee lo que el temporizador de la laptop y el operador versionaron
en `data/forward/`. Si el temporizador deja de correr, o algo espera al operador y su aviso no
llegó, el job falla y GitHub manda un correo.

Lee el estudio solo con `forward_auto.study_state`, desde el bloque 1 (`first=FIRST_BLOCK`): ese
lote lo corre el operador, pero también se vigila. No repite ninguna regla del temporizador: los
bloques pendientes, los bloqueados, los vencidos, el cierre y la terminación son los de
`study_state`. Lo único propio es el margen de `GRACE` sobre la apertura de la ventana de un lote.

Falla, con código 1, si:

- algún bloque pendiente (sin preparar o incompleto, sin retener) tiene la ventana abierta hace
  más de `GRACE`;
- algún bloque espera al operador: retenido, inválido o con duplicados;
- algún bloque venció con el lote incompleto y sus líneas base sin completar.

`study_state` no devuelve los bloques posteriores al cierre del estudio, ni ninguno si el estudio
terminó, así que esos nunca fallan. Pasa, con código 0, en cualquier otro caso, y con el estudio
cerrado, agotado o terminado lo dice. Un estudio que no se puede leer sale con 2.

`--now` sustituye al reloj por un instante UTC: es la entrada `now` de `workflow_dispatch`, que
GitHub pasa vacía cuando nadie la escribe, y vacía es el reloj. El resumen va a la salida y, con
`--summary`, al final de ese archivo: el `GITHUB_STEP_SUMMARY` del job.

No importa el router: un test de arquitectura lo importa en un proceso nuevo.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from crypto_agents.audit import AuditError
from crypto_agents.context import utc_now
from crypto_agents.forward_analysis import study_paths
from crypto_agents.forward_auto import study_state
from crypto_agents.forward_data import FORWARD_DIR, STAGE1_DIR, ForwardDataError
from crypto_agents.forward_plan import MAX_BLOCKS, TARGET_BLOCKS
from crypto_agents.forward_record import FIRST_BLOCK, ForwardError, block_dates
from crypto_agents.selection import SelectionError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from crypto_agents.forward_analysis import StudyPaths
    from crypto_agents.forward_auto import BlockReading, StudyState

__all__ = [
    "GRACE",
    "READ_ERRORS",
    "Watch",
    "WatchEnvironment",
    "main",
    "parse_now",
    "render",
    "watch",
]

GRACE = timedelta(hours=24)
"""Lo que se le da al temporizador desde que se abre la ventana de un lote: corre a las 01:00Z, y
un lote que quedó para el día siguiente (3 o 6) vuelve a correr dentro de este margen."""

READ_ERRORS = (AuditError, ForwardDataError, ForwardError, SelectionError, OSError, ValueError)
"""Lo que levanta leer un estudio roto: un journal con media línea, un `forward.json` o un
manifiesto que no valida, un archivo que falta. `ValueError` cubre los errores de validación."""


class Watch(NamedTuple):
    """El estudio en un instante: el estado del temporizador y los lotes que van tarde."""

    state: StudyState
    late: tuple[BlockReading, ...]
    """Los pendientes con la ventana abierta hace más de `GRACE`."""

    on_time: tuple[BlockReading, ...]
    """Los pendientes con la ventana abierta hace `GRACE` o menos: el temporizador está a tiempo."""

    @property
    def failed(self) -> bool:
        return bool(self.late or self.state.blocked or self.state.expired)


def watch(paths: StudyPaths, stage1: Path, now: datetime) -> Watch:
    """El estudio en `now`, leído con `study_state` desde el bloque 1."""
    state = study_state(paths, stage1, now, first=FIRST_BLOCK)
    late = tuple(reading for reading in state.pending if now - reading.dates.ready > GRACE)
    on_time = tuple(reading for reading in state.pending if reading not in late)
    return Watch(state, late, on_time)


def parse_now(value: str) -> datetime | None:
    """El instante de `--now`: ISO 8601 en UTC, con `Z` o `+00:00`. Vacío es el reloj (`None`).

    Un instante sin zona o con otro desfase se rechaza: en un formulario de GitHub, una hora sin
    zona no dice de qué reloj es.
    """
    text = value.strip()
    if not text:
        return None
    try:
        instant = datetime.fromisoformat(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"no es un instante ISO 8601: {value!r}") from error
    if instant.utcoffset() != timedelta(0):
        raise argparse.ArgumentTypeError(f"tiene que ser UTC, con Z o +00:00: {value!r}")
    return instant.astimezone(UTC)


# ───────────────────────────────────────────── Salida ─────────────────────────────────────────────


def _label(reading: BlockReading) -> str:
    dates = reading.dates
    return f"bloque {dates.number} ({dates.start:%Y-%m-%d})"


def _hours(span: timedelta) -> str:
    return f"{span / timedelta(hours=1):.1f}".removesuffix(".0") + " h"


def _header(now: datetime) -> list[str]:
    first = block_dates(FIRST_BLOCK)
    return [
        f"## Comprobación semanal del estudio, a {now.isoformat(timespec='seconds')}",
        "",
        f"Del bloque {first.number} ({first.start:%Y-%m-%d}) al {MAX_BLOCKS}, leído de "
        "`data/forward/` en `master` con `forward_auto.study_state`.",
        "",
    ]


def render(found: Watch) -> list[str]:
    """El resumen: el resultado, el estado del estudio y una línea por bloque que cuenta."""
    state = found.state
    now = state.now
    problems = len(found.late) + len(state.blocked) + len(state.expired)
    verdict = (
        f"**falla**: {problems} bloque{'s' if problems != 1 else ''} necesita"
        f"{'n' if problems != 1 else ''} atención."
        if found.failed
        else "**pasa**: nada está retrasado ni espera al operador."
    )
    lines = [
        *_header(now),
        f"- Resultado: {verdict}",
        f"- Bloques no vacíos: {state.non_empty}; el estudio cierra con {TARGET_BLOCKS}.",
    ]
    if state.terminated is not None:
        lines.append(f"- El estudio terminó {state.terminated}: no queda nada que vigilar.")
    if state.closed_at is not None:
        lines.append(
            f"- El estudio cerró en el bloque {state.closed_at}: los posteriores no son del "
            "estudio y no se vigilan."
        )
    if state.exhausted:
        lines.append(
            f"- El bloque {MAX_BLOCKS} terminó sin {TARGET_BLOCKS} bloques no vacíos: el estudio "
            "ya no cierra."
        )
    if found.failed:
        lines += ["", "### Lo que falla", ""]
        lines += [
            f"- {_label(reading)}: {reading.status.value}, con la ventana abierta desde el "
            f"{reading.dates.ready.isoformat()}, hace {_hours(now - reading.dates.ready)}: más "
            f"de {_hours(GRACE)} sin completar su lote."
            for reading in found.late
        ]
        lines += [
            f"- {_label(reading)}: {reading.status.value}"
            + (f" ({reading.held_by})" if reading.held_by is not None else "")
            + f". Espera al operador; vence el {reading.dates.deadline.isoformat()}."
            for reading in state.blocked
        ]
        lines += [
            f"- {_label(reading)}: venció el {reading.dates.deadline.isoformat()} con el lote "
            f"{reading.status.value} y sus líneas base sin completar "
            f"(`forward baselines --block {reading.dates.start:%Y-%m-%d}`)."
            for reading in state.expired
        ]
    if found.on_time:
        lines += ["", "### A tiempo", ""]
        lines += [
            f"- {_label(reading)}: {reading.status.value}, con la ventana abierta hace "
            f"{_hours(now - reading.dates.ready)}; falla si sigue así el "
            f"{(reading.dates.ready + GRACE).isoformat()}."
            for reading in found.on_time
        ]
    if state.upcoming is not None:
        lines += [
            "",
            f"El siguiente: el lote del bloque {state.upcoming.number} "
            f"({state.upcoming.start:%Y-%m-%d}) se abre el {state.upcoming.ready.isoformat()}.",
        ]
    return lines


def _emit(lines: Sequence[str], summary: Path | None) -> None:
    text = "\n".join(lines) + "\n"
    print(text, end="", flush=True)
    if summary is not None:
        with summary.open("a", encoding="utf-8") as handle:
            handle.write(text)


# ─────────────────────────────────────────── La orden ─────────────────────────────────────────────


class WatchEnvironment(NamedTuple):
    """Lo que la comprobación lee fuera de sí misma. Las pruebas lo sustituyen entero."""

    clock: Callable[[], datetime] = utc_now
    paths: StudyPaths = study_paths(FORWARD_DIR)
    stage1: Path = STAGE1_DIR


def main(argv: Sequence[str] | None = None, env: WatchEnvironment | None = None) -> int:
    """Punto de entrada: 0 si pasa, 1 si falla una regla, 2 si el estudio no se puede leer."""
    parser = argparse.ArgumentParser(
        prog="python -m crypto_agents.forward_watch",
        description="La comprobación semanal del estudio hacia adelante.",
    )
    parser.add_argument(
        "--now",
        type=parse_now,
        default=None,
        help="instante UTC que sustituye al reloj (ISO 8601, con Z); vacío es el reloj",
    )
    parser.add_argument(
        "--summary", type=Path, default=None, help="añade el resumen al final de este archivo"
    )
    args = parser.parse_args(argv)
    environment = env if env is not None else WatchEnvironment()
    now: datetime = args.now if args.now is not None else environment.clock()
    try:
        found = watch(environment.paths, environment.stage1, now)
    except READ_ERRORS as error:
        _emit(
            [*_header(now), f"- Resultado: **falla**: el estudio no se pudo leer: {error}"],
            args.summary,
        )
        return 2
    _emit(render(found), args.summary)
    return 1 if found.failed else 0


if __name__ == "__main__":
    sys.exit(main())
