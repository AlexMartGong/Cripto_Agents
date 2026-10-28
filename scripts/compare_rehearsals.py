"""Compara dos directorios de `forward rehearse` y dice si son el mismo ensayo.

    uv run python scripts/compare_rehearsals.py <directorio A> <directorio B>

Dos ensayos del mismo código sobre la misma caché tienen que dejar los mismos archivos, con las
mismas líneas en el mismo orden. Solo se deja pasar lo que cambia de una corrida a otra sin que
cambie el ensayo:

- en `meta.json`, `started_at`, `git_commit` y `argv` (la hora, el commit y las rutas pasadas);
- en cada journal, `calls[].at`: el reloj del router, que en un replay es la hora de pared.

Cualquier otra diferencia —un archivo de más o de menos, otro número de líneas, otro valor— se
imprime y sale con 1. Sale con 2 si un directorio no existe. Lee JSON y nada más: no importa el
paquete, así que compara igual dos ensayos de commits distintos.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Sequence

META_FILE = "meta.json"
ALLOWED_META = frozenset({"started_at", "git_commit", "argv"})
"""Lo que puede cambiar en `meta.json`: cuándo empezó, en qué commit y con qué rutas."""

ALLOWED_JOURNAL = frozenset({"calls[].at"})
"""Lo que puede cambiar en un journal: el reloj del router en cada llamada."""


class FileComparison(NamedTuple):
    """Un archivo de los dos ensayos: cuántas líneas tiene en cada uno y qué difiere."""

    name: str
    lines: tuple[int, int]
    same_bytes: bool
    allowed: dict[str, int]
    """Diferencias permitidas, por campo: cuántas veces aparece cada una."""

    other: list[str]
    """Diferencias no permitidas, una por línea de texto."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _differences(left: object, right: object, path: str, found: list[tuple[str, str]]) -> None:
    """Las rutas de los campos que difieren, con las listas sin índice (`calls[].at`)."""
    if type(left) is not type(right):
        found.append((path, f"{left!r} ≠ {right!r}"))
    elif isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                found.append((f"{path}.{key}".lstrip("."), "falta en uno de los dos"))
            else:
                _differences(left[key], right[key], f"{path}.{key}", found)
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            found.append((f"{path}[]", f"{len(left)} ≠ {len(right)} elementos"))
            return
        for one, other in zip(left, right, strict=True):
            _differences(one, other, f"{path}[]", found)
    elif left != right:
        found.append((path, f"{left!r} ≠ {right!r}"))


def _split(
    found: list[tuple[str, str]], allowed: frozenset[str], where: str
) -> tuple[dict[str, int], list[str]]:
    """Las diferencias permitidas, contadas por campo, y las demás. Un campo se permite por su
    ruta entera (`calls[].at`) o por su clave de arriba (`argv`, con todo lo que cuelga de ella)."""
    kept: Counter[str] = Counter()
    other: list[str] = []
    for path, detail in found:
        field = path.lstrip(".")
        top = field.split(".", 1)[0].removesuffix("[]")
        if field in allowed or top in allowed:
            kept[top if top in allowed else field] += 1
        else:
            other.append(f"{where}: {field}: {detail}")
    return dict(kept), other


def compare_file(name: str, left: Path, right: Path) -> FileComparison:
    """Compara un archivo de los dos ensayos según su clase: `meta.json`, journal u otro."""
    same = _sha256(left) == _sha256(right)
    if name == META_FILE:
        found: list[tuple[str, str]] = []
        _differences(
            json.loads(left.read_text("utf-8")), json.loads(right.read_text("utf-8")), "", found
        )
        allowed, other = _split(found, ALLOWED_META, name)
        return FileComparison(name, (1, 1), same, allowed, other)
    left_lines = left.read_text("utf-8").splitlines()
    right_lines = right.read_text("utf-8").splitlines()
    counts = (len(left_lines), len(right_lines))
    if not name.endswith(".jsonl"):
        other = [] if same else [f"{name}: el contenido difiere"]
        return FileComparison(name, counts, same, {}, other)
    if counts[0] != counts[1]:
        return FileComparison(name, counts, same, {}, [f"{name}: {counts[0]} ≠ {counts[1]} líneas"])
    allowed_total: Counter[str] = Counter()
    other_total: list[str] = []
    for number, (one, two) in enumerate(zip(left_lines, right_lines, strict=True), start=1):
        found = []
        _differences(json.loads(one), json.loads(two), "", found)
        allowed, other = _split(found, ALLOWED_JOURNAL, f"{name}:{number}")
        allowed_total.update(allowed)
        other_total += other
    return FileComparison(name, counts, same, dict(allowed_total), other_total)


def compare(left: Path, right: Path) -> tuple[list[FileComparison], list[str]]:
    """Cada archivo de los dos ensayos, y lo que falta en uno de ellos."""
    names_left = {path.name for path in left.iterdir() if path.is_file()}
    names_right = {path.name for path in right.iterdir() if path.is_file()}
    missing = [f"{name}: solo en {left}" for name in sorted(names_left - names_right)]
    missing += [f"{name}: solo en {right}" for name in sorted(names_right - names_left)]
    files = [
        compare_file(name, left / name, right / name) for name in sorted(names_left & names_right)
    ]
    return files, missing


def render(
    left: Path, right: Path, files: Sequence[FileComparison], missing: Sequence[str]
) -> list[str]:
    lines = [
        f"A: {left}",
        f"B: {right}",
        "",
        "| archivo | líneas A | líneas B | bytes idénticos | diferencias permitidas | otras |",
        "| --- | ---: | ---: | --- | --- | ---: |",
    ]
    for item in files:
        allowed = ", ".join(f"{field} ({count})" for field, count in sorted(item.allowed.items()))
        lines.append(
            f"| {item.name} | {item.lines[0]} | {item.lines[1]} | "
            f"{'sí' if item.same_bytes else 'no'} | {allowed or 'ninguna'} | {len(item.other)} |"
        )
    other = [*missing, *(entry for item in files for entry in item.other)]
    lines.append("")
    if other:
        lines += [
            f"**DIFIEREN** en {len(other)} puntos fuera de lo permitido:",
            *(f"- {entry}" for entry in other),
        ]
    else:
        lines.append(
            "Iguales salvo lo permitido: started_at, git_commit y argv en meta.json, y calls[].at "
            "en los journals."
        )
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compara dos directorios de `forward rehearse`.")
    parser.add_argument("left", type=Path, help="el primer ensayo")
    parser.add_argument("right", type=Path, help="el segundo ensayo")
    args = parser.parse_args(argv)
    for directory in (args.left, args.right):
        if not directory.is_dir():
            print(f"error: no existe el directorio {directory}", file=sys.stderr)
            return 2
    files, missing = compare(args.left, args.right)
    print("\n".join(render(args.left, args.right, files, missing)))
    return 1 if missing or any(item.other for item in files) else 0


if __name__ == "__main__":
    sys.exit(main())
