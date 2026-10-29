"""El script que compara dos ensayos de `forward rehearse`: `scripts/compare_rehearsals.py`.

Deja pasar `started_at`, `git_commit` y `argv` en `meta.json` y `calls[].at` en los journals, y
nada más. Cada regla con su mutación.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import scripts.compare_rehearsals as module
from scripts.compare_rehearsals import ALLOWED_JOURNAL, ALLOWED_META, main

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

META: dict[str, object] = {
    "started_at": "2026-10-28T22:09:43Z",
    "git_commit": "cc42962",
    "argv": ["rehearse", "--out", "a"],
    "hostname": "zephyrus-G15",
    "arms": ["solo", "rule_trend"],
}
RECORD: dict[str, object] = {
    "run_id": "c03656b8-fa32-5130-8472-b94192733cb3",
    "proposed": {"action": "hold"},
    "calls": [{"at": "2026-10-28T22:09:44Z", "prompt_digest": "abc", "latency_ms": 0.0}],
}


def run_dir(
    root: Path, name: str, meta: Mapping[str, object], records: Sequence[Mapping[str, object]]
) -> Path:
    directory = root / name
    directory.mkdir()
    (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    lines = "".join(json.dumps(record) + "\n" for record in records)
    (directory / "solo.jsonl").write_text(lines, encoding="utf-8")
    return directory


def changed(record: Mapping[str, object], **fields: object) -> dict[str, object]:
    return {**record, **fields}


def assert_the_rules(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Igual salvo lo permitido sale con 0; cualquier otra diferencia, con 1."""
    left = run_dir(tmp_path, "a", META, [RECORD, RECORD])
    later_call = {"at": "2026-10-28T22:11:52Z", "prompt_digest": "abc", "latency_ms": 0.0}
    right_meta: dict[str, object] = {
        **META,
        "started_at": "x",
        "git_commit": "ed84077",
        "argv": ["otro"],
    }
    right = run_dir(tmp_path, "b", right_meta, [changed(RECORD, calls=[later_call])] * 2)
    assert main([str(left), str(right)]) == 0
    out = capsys.readouterr().out
    assert "| solo.jsonl | 2 | 2 | no | calls[].at (2) | 0 |" in out
    assert "| meta.json | 1 | 1 | no | argv (1), git_commit (1), started_at (1) | 0 |" in out

    cases: dict[str, tuple[Mapping[str, object], Sequence[Mapping[str, object]]]] = {
        "otra acción": (META, [changed(RECORD, proposed={"action": "buy"}), RECORD]),
        "otro digest": (META, [changed(RECORD, calls=[{**later_call, "prompt_digest": "x"}])] * 2),
        "otra máquina": ({**META, "hostname": "otra"}, [RECORD, RECORD]),
        "una línea menos": (META, [RECORD]),
    }
    for name, (meta, records) in cases.items():
        other = run_dir(tmp_path, name, meta, records)
        assert main([str(left), str(other)]) == 1, name
        assert "**DIFIEREN**" in capsys.readouterr().out, name

    extra = run_dir(tmp_path, "con un archivo más", META, [RECORD, RECORD])
    (extra / "report.md").write_text("tabla\n", encoding="utf-8")
    assert main([str(left), str(extra)]) == 1
    assert "report.md: solo en" in capsys.readouterr().out


def test_only_the_clocks_the_commit_and_the_argv_may_differ(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert_the_rules(tmp_path, capsys)


def test_a_missing_directory_is_a_usage_error(tmp_path: Path) -> None:
    assert main([str(tmp_path / "no"), str(tmp_path)]) == 2


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("ALLOWED_JOURNAL", ALLOWED_JOURNAL | {"proposed.action"}),
        ("ALLOWED_JOURNAL", ALLOWED_JOURNAL | {"calls[].prompt_digest"}),
        ("ALLOWED_META", ALLOWED_META | {"hostname"}),
    ],
    ids=["la acción", "el digest", "la máquina"],
)
def test_mutation_a_difference_let_through_is_caught(
    name: str,
    value: frozenset[str],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, name, value)
    with pytest.raises(AssertionError):
        assert_the_rules(tmp_path, capsys)
