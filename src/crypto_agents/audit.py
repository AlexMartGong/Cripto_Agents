"""Auditoría de una corrida de ablación a partir de lo que dejó en disco.

La primera tabla de la ablación no se pudo auditar: los journals vivían en memoria
y murieron con el proceso, así que de 840 evaluaciones quedó una tabla agregada y
ninguna forma de preguntarle nada. Este módulo es la otra mitad del arreglo —la
corrida ahora escribe un directorio, y esto lo lee.

Un directorio de corrida contiene:

    meta.json       de qué plan salió, con qué argumentos, sobre qué commit
    <brazo>.jsonl   un `EvaluationRecord` por línea, uno por brazo

Tres propiedades:

- **Cada cifra cita su entrada.** El informe abre con el sha-256 del plan y el de
  cada journal leído. Una cifra sin el digest de su archivo no se puede reproducir,
  porque no hay forma de saber si el archivo de hoy es el de entonces.
- **Lo que no se puede determinar lo dice.** Un brazo que no escribió journal no es
  un brazo con cero evaluaciones, y una corrida servida entera desde la caché no
  tiene latencia cero: tiene latencia desconocida. Ambos salen como «no
  determinado: <por qué>».
- **No puede llamar a nadie.** No importa el router ni nada que lo traiga —ni
  `ablation.py`, que es quien escribe estos directorios—, así que auditar una
  corrida es gratis y repetible. `tests/test_architecture.py` lo impone.

El informe cuenta y no interpreta: qué significa una cifra se escribe a mano.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from pydantic import AwareDatetime, Field, ValidationError

from crypto_agents.journal import EvaluationRecord, JournalError, JsonlJournal
from crypto_agents.metrics import (
    attempt_counts,
    conviction_cross,
    dismissal_cross,
    invalidation_stats,
    live_latency,
    quota_by_locality,
    resume_delta,
    risk_flow,
    undecided_causes,
    validation_failure,
    worst_pair,
)
from crypto_agents.state import AgentRole, Backend, FrozenModel

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from crypto_agents.state import LLMCall

__all__ = [
    "META_FILE",
    "REFERENCE_ARM",
    "ArmJournal",
    "AuditError",
    "PlanKind",
    "RunDirectory",
    "RunMeta",
    "arm_journal_path",
    "chain_calls",
    "file_sha256",
    "main",
    "ratio",
    "read_meta",
    "read_run",
    "render_audit",
    "run_chain",
    "write_meta",
]

META_FILE = "meta.json"
REFERENCE_ARM = "full"
"""El brazo con las dos mesas en remoto: el único sobre el que se cruza acción y debate."""


class AuditError(RuntimeError):
    """El directorio no se puede leer como una corrida."""


class PlanKind(StrEnum):
    """De dónde salió el plan que recorrió la corrida."""

    MANIFEST = "manifest"
    """Una selección versionada de activaciones."""

    HISTORY = "history"
    """Un histórico contiguo, recorrido de principio a fin."""


class RunMeta(FrozenModel):
    """Lo que hace falta para saber qué corrida es esta sin preguntarle a nadie."""

    plan_kind: PlanKind
    plan_path: str = Field(min_length=1)
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    """Digest del manifiesto, o del histórico cuando no hay manifiesto."""

    argv: tuple[str, ...]
    fill: bool
    """Si la corrida podía llamar a proveedores. Sin esto, solo caché."""

    arms: tuple[str, ...] = Field(min_length=1)
    started_at: AwareDatetime
    git_commit: str | None = None
    """Commit sobre el que se corrió, o `None` si no se pudo preguntar a git."""

    git_dirty: bool | None = None
    """Si había cambios sin comprometer. Con ellos, el commit no describe el código."""

    resumed_from: str | None = None
    """Directorio de la pasada que esta reanuda, si la hay."""


def file_sha256(path: Path) -> str:
    """sha-256 del contenido de un archivo."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arm_journal_path(directory: Path, arm: str) -> Path:
    """Journal de un brazo dentro de un directorio de corrida."""
    return directory / f"{arm}.jsonl"


def write_meta(directory: Path, meta: RunMeta) -> Path:
    """Escribe `meta.json`. Se llama antes de la primera evaluación.

    Antes y no al final: una corrida interrumpida es justo la que hay que poder
    identificar, y si los metadatos se escribieran al terminar no los tendría.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / META_FILE
    path.write_text(meta.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def read_meta(directory: Path) -> RunMeta:
    """Lee `meta.json`. Sin él, el directorio no es una corrida."""
    path = directory / META_FILE
    try:
        return RunMeta.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise AuditError(f"{directory} no es una corrida: falta {META_FILE} ({error})") from error
    except ValidationError as error:
        raise AuditError(f"{path} no se pudo leer: {error}") from error


class ArmJournal(NamedTuple):
    """Lo que un brazo dejó escrito.

    `NamedTuple` y no `FrozenModel` por lo mismo que `ReplayPlan`: los registros ya
    vienen validados de `JsonlJournal`, y revalidar cientos de ellos al agruparlos
    es tiempo pagado por nada.
    """

    arm: str
    path: Path
    sha256: str | None
    """Digest del archivo, o `None` si el brazo no llegó a escribir."""

    records: tuple[EvaluationRecord, ...]


class RunDirectory(NamedTuple):
    """Una corrida leída entera."""

    path: Path
    meta: RunMeta
    arms: tuple[ArmJournal, ...]


def read_run(directory: Path) -> RunDirectory:
    """Lee los metadatos y el journal de cada brazo declarado.

    Los brazos salen de `meta.json` y no de listar el directorio: un brazo que la
    corrida declaró y no escribió tiene que aparecer como ausente, y listando
    archivos simplemente no aparecería.
    """
    meta = read_meta(directory)
    arms: list[ArmJournal] = []
    for name in meta.arms:
        path = arm_journal_path(directory, name)
        if not path.is_file():
            arms.append(ArmJournal(name, path, None, ()))
            continue
        try:
            records = tuple(JsonlJournal(path).read_all())
        except JournalError as error:
            raise AuditError(str(error)) from error
        arms.append(ArmJournal(name, path, file_sha256(path), records))
    return RunDirectory(directory, meta, tuple(arms))


def run_chain(directory: Path) -> list[RunDirectory]:
    """La corrida y todas las que reanuda, de la más reciente a la más antigua.

    Un eslabón que falta es un error y no el final de la cadena: quien la recorre
    lo hace para saber cuánto se gastó, y dar por cero lo que no se encuentra es
    creerse con más presupuesto del que hay.
    """
    chain: list[RunDirectory] = []
    seen: set[Path] = set()
    current: Path | None = directory
    while current is not None:
        resolved = current.resolve()
        if resolved in seen:
            raise AuditError(f"ciclo en la cadena de reanudación: {current} ya se visitó")
        seen.add(resolved)
        run = read_run(current)
        chain.append(run)
        current = Path(run.meta.resumed_from) if run.meta.resumed_from is not None else None
    return chain


def chain_calls(directory: Path) -> list[LLMCall]:
    """Todos los intentos de una corrida y de las que reanuda, de todos los brazos.

    Es lo que siembra el contador de cuota al reanudar. Se entregan todos, sin
    filtrar por ventana ni por backend: decidir cuáles siguen contando es de
    `QuotaLedger.seed()`, que es quien tiene el reloj.
    """
    return [
        call
        for run in run_chain(directory)
        for arm in run.arms
        for record in arm.records
        for call in record.calls
    ]


# ────────────────────────────────────────────── Informe ───────────────────────────────────────────


def _undetermined(reason: str) -> str:
    """Celda de lo que no se pudo calcular, con su causa."""
    return f"no determinado: {reason}"


NO_JOURNAL = _undetermined("el brazo no escribió journal")


def ratio(numerator: int, denominator: int) -> str:
    """Una tasa con sus dos números delante. Sin denominador, lo dice en vez de dar 0%."""
    if denominator == 0:
        return _undetermined("0 respondidas")
    return f"{numerator}/{denominator} ({numerator / denominator:.0%})"


def _ms(value: float) -> str:
    return f"{value:.0f} ms"


def _fraction(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def _table(
    header: Sequence[str],
    arms: Sequence[ArmJournal],
    row: Callable[[ArmJournal], Sequence[Sequence[str]]],
) -> list[str]:
    """Tabla por brazo. Un brazo sin journal ocupa una fila que lo dice.

    `row` devuelve las filas del brazo sin la columna del nombre; puede ser más de
    una —las causas de aborto— o ninguna, y entonces el brazo sale con un guion.
    """
    lines = [
        "| brazo | " + " | ".join(header) + " |",
        "| --- |" + " --- |" * len(header),
    ]
    for arm in arms:
        if arm.sha256 is None:
            lines.append(f"| `{arm.arm}` | {NO_JOURNAL} |" + " |" * (len(header) - 1))
            continue
        rows = row(arm) or [["—"] * len(header)]
        lines.extend(f"| `{arm.arm}` | " + " | ".join(cells) + " |" for cells in rows)
    lines.append("")
    return lines


def _header(run: RunDirectory) -> list[str]:
    meta = run.meta
    if meta.git_commit is None:
        commit = _undetermined("git no respondió al arrancar")
    else:
        state = {
            True: "con cambios sin comprometer",
            False: "árbol limpio",
            None: "estado desconocido",
        }
        commit = f"`{meta.git_commit}` ({state[meta.git_dirty]})"
    lines = [
        f"# Auditoría de la corrida `{run.path}`",
        "",
        f"Reproducir: `python -m crypto_agents.audit {run.path}`",
        "",
        f"- plan ({meta.plan_kind.value}): `{meta.plan_path}`, sha-256 `{meta.plan_sha256}`",
        f"- inicio: {meta.started_at.isoformat()}",
        f"- commit: {commit}",
        f"- podía llamar a proveedores (`--fill`): {'sí' if meta.fill else 'no'}",
        f"- argumentos: `{' '.join(meta.argv) or '(ninguno)'}`",
        f"- reanuda: {f'`{meta.resumed_from}`' if meta.resumed_from else 'no'}",
        "",
        "## Entradas",
        "",
    ]
    lines.extend(
        _table(
            ["archivo", "sha-256", "evaluaciones"],
            run.arms,
            lambda arm: [[f"`{arm.path.name}`", f"`{arm.sha256}`", str(len(arm.records))]],
        )
    )
    return lines


def _attempts(arms: Sequence[ArmJournal]) -> list[str]:
    def totals(arm: ArmJournal) -> list[list[str]]:
        counts = attempt_counts(arm.records)
        rate = validation_failure(arm.records)
        worst = worst_pair(arm.records)
        worst_cell = (
            "—"
            if worst is None
            else f"{worst.role.value}/{worst.backend.value} "
            + ratio(worst.stats.invalid, worst.stats.answered)
        )
        return [
            [
                str(counts.total),
                str(counts.live),
                str(counts.cache_hits),
                str(counts.valid),
                str(counts.invalid),
                str(counts.cached_invalid),
                ratio(rate.numerator, rate.denominator),
                worst_cell,
            ]
        ]

    lines = ["## 1. Intentos", ""]
    lines.extend(
        _table(
            [
                "total",
                "vivos",
                "caché",
                "válidos",
                "inválidos",
                "caché e inválido",
                "fallo validación (inválidas/respondidas)",
                "peor par",
            ],
            arms,
            totals,
        )
    )
    lines.extend(
        _table(
            [role.value for role in AgentRole],
            arms,
            lambda arm: [
                [str(attempt_counts(arm.records).by_role.get(role, 0)) for role in AgentRole]
            ],
        )
    )
    lines.extend(
        _table(
            [backend.value for backend in Backend],
            arms,
            lambda arm: [
                [str(attempt_counts(arm.records).by_backend.get(item, 0)) for item in Backend]
            ],
        )
    )
    return lines


def _latency(arms: Sequence[ArmJournal]) -> list[str]:
    def row(arm: ArmJournal) -> list[list[str]]:
        stats = live_latency(arm.records)
        if stats is None:
            return [[_undetermined("sin llamadas vivas"), "", "", ""]]
        return [[str(stats.n), _ms(stats.mean_ms), _ms(stats.median_ms), _ms(stats.p95_ms)]]

    return [
        "## 2. Latencia, solo llamadas vivas",
        "",
        *_table(["n", "media", "mediana", "p95"], arms, row),
    ]


def _quota(arms: Sequence[ArmJournal]) -> list[str]:
    def row(arm: ArmJournal) -> list[list[str]]:
        split = quota_by_locality(arm.records)
        return [[f"{split.remote:.1f}", f"{split.local:.1f}"]]

    return ["## 3. Cuota", "", *_table(["remota", "local"], arms, row)]


def _undecided(arms: Sequence[ArmJournal]) -> list[str]:
    def row(arm: ArmJournal) -> list[list[str]]:
        return [
            [node, kind.value, str(count)]
            for (node, kind), count in undecided_causes(arm.records).items()
        ]

    return [
        "## 4. Evaluaciones sin decisión",
        "",
        *_table(["nodo", "causa", "evaluaciones"], arms, row),
    ]


def _risk(arms: Sequence[ArmJournal]) -> list[str]:
    def row(arm: ArmJournal) -> list[list[str]]:
        flow = risk_flow(arm.records)
        vetoes = ", ".join(f"{rule} {count}" for rule, count in flow.vetoes.items()) or "0"
        return [
            [str(flow.actionable), str(flow.holds), str(flow.orders), vetoes, str(flow.unexecuted)]
        ]

    return [
        "## 5. Del decisor a la orden",
        "",
        *_table(
            ["accionables", "hold", "órdenes", "vetos por regla", "aprobadas sin orden"], arms, row
        ),
    ]


def _invalidation(arms: Sequence[ArmJournal]) -> list[str]:
    def row(arm: ArmJournal) -> list[list[str]]:
        stats = invalidation_stats(arm.records)
        if stats.n == 0:
            return [[_undetermined("sin propuestas accionables"), "", "", "", ""]]
        return [
            [
                str(stats.n),
                str(stats.wrong_side),
                _fraction(stats.p10),
                _fraction(stats.median),
                _fraction(stats.p90),
            ]
        ]

    return [
        "## 6. Invalidación respecto al cierre",
        "",
        "Distancia: `|invalidación - cierre| / cierre`. Lado equivocado: `>=` cierre en un "
        "`buy`, `<=` en un `sell`.",
        "",
        *_table(["n", "lado equivocado", "p10", "mediana", "p90"], arms, row),
    ]


def _debate(arms: Sequence[ArmJournal]) -> list[str]:
    lines = [f"## 7. `{REFERENCE_ARM}`: acción contra las mesas", ""]
    reference = next((arm for arm in arms if arm.arm == REFERENCE_ARM), None)
    if reference is None:
        return [*lines, _undetermined(f"la corrida no incluye el brazo `{REFERENCE_ARM}`"), ""]
    if reference.sha256 is None:
        return [*lines, NO_JOURNAL, ""]

    signs = {-1: "bear > bull", 0: "iguales", 1: "bull > bear"}
    lines.extend(["| acción | convicción | evaluaciones |", "| --- | --- | --- |"])
    lines.extend(
        f"| {action.value} | {signs[sign]} | {count} |"
        for (action, sign), count in conviction_cross(reference.records).items()
    )
    lines.extend(["", "| acción | mesa descartada | evaluaciones |", "| --- | --- | --- |"])
    lines.extend(
        f"| {action.value} | {side.value if side is not None else 'ninguna'} | {count} |"
        for (action, side), count in dismissal_cross(reference.records).items()
    )
    lines.append("")
    return lines


def _resume(run: RunDirectory, previous: RunDirectory) -> list[str]:
    before = {arm.arm: arm for arm in previous.arms}

    def row(arm: ArmJournal) -> list[list[str]]:
        earlier = before.get(arm.arm)
        if earlier is None or earlier.sha256 is None:
            return [[_undetermined("el brazo no tiene journal en la pasada anterior"), ""]]
        delta = resume_delta(earlier.records, arm.records)
        return [[str(delta.undecided_before), str(delta.rescued)]]

    return [
        f"## Reanudación de `{previous.path}`",
        "",
        "Un intento inválido no se cachea: la reanudación vuelve a llamar en vivo a las "
        "evaluaciones que abortaron. «Rescatadas» son las que antes no decidieron y ahora sí.",
        "",
        *_table(["sin decisión antes", "rescatadas"], run.arms, row),
    ]


def render_audit(run: RunDirectory, previous: RunDirectory | None = None) -> str:
    """Informe en Markdown de una corrida. Con `previous`, también lo que cambió al reanudar.

    Las cifras son de `run` y solo de `run`: las llamadas vivas de la pasada
    anterior se leen auditando su directorio, para que cada número cite un único
    conjunto de archivos.
    """
    lines = _header(run)
    for section in (_attempts, _latency, _quota, _undecided, _risk, _invalidation, _debate):
        lines.extend(section(run.arms))
    if previous is not None:
        lines.extend(_resume(run, previous))
    return "\n".join(lines).rstrip("\n") + "\n"


# ───────────────────────────────────────────── Comando ────────────────────────────────────────────


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada: `python -m crypto_agents.audit <directorio>`."""
    parser = argparse.ArgumentParser(
        prog="python -m crypto_agents.audit",
        description="Imprime las métricas de una corrida de ablación leyendo su directorio.",
    )
    parser.add_argument("directory", type=Path, help="directorio de la corrida, con meta.json")
    args = parser.parse_args(argv)

    try:
        chain = run_chain(args.directory)
    except AuditError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(render_audit(chain[0], chain[1] if len(chain) > 1 else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
