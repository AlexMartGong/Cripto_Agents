"""Pruebas de la auditoría de una corrida de ablación.

Lo que se comprueba aquí es que un directorio de corrida se puede leer entero sin
más contexto que él mismo, y que cada cifra del informe trae al lado el digest del
archivo del que salió. Las métricas en sí se prueban en `tests/test_metrics.py`.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from crypto_agents.audit import (
    AuditError,
    PlanKind,
    RoleMeta,
    RunMeta,
    arm_journal_path,
    chain_calls,
    main,
    read_meta,
    read_run,
    render_audit,
    run_chain,
    write_meta,
)
from crypto_agents.journal import JsonlJournal
from crypto_agents.state import Action, AgentRole, Backend, Billing, FailureKind
from tests.test_metrics import (
    OPEN_GATE,
    failed,
    proposal,
    record,
    snapshot,
    timed,
    unbalanced_calls,
)

if TYPE_CHECKING:
    from pathlib import Path

    from crypto_agents.journal import EvaluationRecord

STARTED = datetime(2026, 10, 1, 17, 0, tzinfo=UTC)
PLAN_SHA = "b" * 64


def meta(arms: tuple[str, ...] = ("full", "solo"), resumed_from: Path | None = None) -> RunMeta:
    """Metadatos de una corrida de prueba."""
    return RunMeta(
        plan_kind=PlanKind.MANIFEST,
        plan_path="data/ablation_selection.json",
        plan_sha256=PLAN_SHA,
        argv=("--manifest", "data/ablation_selection.json", "--fill"),
        fill=True,
        arms=arms,
        started_at=STARTED,
        git_commit="d9f422c",
        git_dirty=False,
        resumed_from=None if resumed_from is None else str(resumed_from),
    )


def write_arm(directory: Path, arm: str, records: list[EvaluationRecord]) -> Path:
    """Journal de un brazo dentro del directorio de corrida."""
    journal = JsonlJournal(arm_journal_path(directory, arm))
    for item in records:
        journal.write(item)
    return journal.path


def full_records() -> list[EvaluationRecord]:
    """Una evaluación decidida con llamadas vivas y una que abortó en el decisor."""
    return [
        record(
            snapshot=snapshot(),
            activation=OPEN_GATE,
            proposal=proposal(Action.BUY, 95.0),
            calls=unbalanced_calls(),
        ),
        record(
            activation=OPEN_GATE,
            calls=(timed(50.0, role=AgentRole.DECIDER, failure=FailureKind.SCHEMA),),
            errors=failed("decide", "decider: 2 intento(s) sin salida válida; último error: x"),
        ),
    ]


# ───────────────────────────────────────────── Metadatos ──────────────────────────────────────────


def test_meta_round_trips_through_its_file(tmp_path: Path) -> None:
    """Lo que la corrida escribió al empezar es lo que la auditoría lee después."""
    write_meta(tmp_path, meta())
    assert read_meta(tmp_path) == meta()


def test_a_meta_written_before_billing_was_recorded_still_loads(tmp_path: Path) -> None:
    """Un `meta.json` de ayer sigue siendo una corrida, y no dice cómo se pagó."""
    write_meta(tmp_path, meta())
    path = tmp_path / "meta.json"
    data = json.loads(path.read_text("utf-8"))
    data.pop("billing", None)
    path.write_text(json.dumps(data), encoding="utf-8")

    assert read_meta(tmp_path).billing is None


def test_the_billing_round_trips_through_meta(tmp_path: Path) -> None:
    write_meta(tmp_path, meta().model_copy(update={"billing": Billing.PAYG}))
    assert read_meta(tmp_path).billing is Billing.PAYG


NEW_META_FIELDS = ("arm_roles", "kill_switch", "quota_window", "horizon")


def test_a_meta_written_before_the_role_fields_still_loads(tmp_path: Path) -> None:
    """Un `meta.json` de ayer sigue siendo una corrida, sin roles ni horizonte registrados."""
    write_meta(tmp_path, meta())
    path = tmp_path / "meta.json"
    data = json.loads(path.read_text("utf-8"))
    for field in NEW_META_FIELDS:
        data.pop(field, None)
    path.write_text(json.dumps(data), encoding="utf-8")

    loaded = read_meta(tmp_path)

    assert [getattr(loaded, field) for field in NEW_META_FIELDS] == [None, None, None, None]


def test_the_role_fields_round_trip_through_meta(tmp_path: Path) -> None:
    roles = {
        "full": {
            AgentRole.DECIDER: RoleMeta(
                model="glm-5.2",
                backend=Backend.OPENAI,
                temperature=0.0,
                quota_per_window=880,
                quota_weight=1.0,
            )
        }
    }
    written = meta().model_copy(
        update={
            "arm_roles": roles,
            "kill_switch": False,
            "quota_window": timedelta(hours=5),
            "horizon": 6,
        }
    )
    write_meta(tmp_path, written)

    loaded = read_meta(tmp_path)

    assert loaded == written
    assert json.loads((tmp_path / "meta.json").read_text("utf-8"))["quota_window"] == "PT5H"


def test_a_directory_without_meta_is_not_a_run(tmp_path: Path) -> None:
    """Sin `meta.json` no se sabe de qué plan salieron las cifras: no se informa nada."""
    with pytest.raises(AuditError, match=r"meta\.json"):
        read_run(tmp_path)


# ───────────────────────────────────────── Lectura del directorio ─────────────────────────────────


def test_a_run_is_read_with_the_digest_of_every_journal(tmp_path: Path) -> None:
    """Cada brazo trae sus registros y el sha-256 del archivo de donde salieron."""
    write_meta(tmp_path, meta())
    path = write_arm(tmp_path, "full", full_records())

    run = read_run(tmp_path)
    full = next(arm for arm in run.arms if arm.arm == "full")

    assert len(full.records) == 2
    assert full.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_an_arm_that_never_wrote_is_reported_as_missing_not_as_empty(tmp_path: Path) -> None:
    """Un brazo declarado sin archivo no es «cero evaluaciones»: no llegó a correr."""
    write_meta(tmp_path, meta())
    write_arm(tmp_path, "full", full_records())

    solo = next(arm for arm in read_run(tmp_path).arms if arm.arm == "solo")

    assert solo.sha256 is None
    assert solo.records == ()


# ───────────────────────────────────────── Cadena de reanudación ──────────────────────────────────


def chained(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Tres pasadas: la tercera reanuda la segunda, que reanuda la primera."""
    first, second, third = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    write_meta(first, meta())
    write_meta(second, meta(resumed_from=first))
    write_meta(third, meta(resumed_from=second))
    write_arm(first, "full", [record(calls=(timed(10.0, backend=Backend.OPENAI),))])
    write_arm(second, "full", [record(calls=(timed(20.0, backend=Backend.OPENAI),))])
    write_arm(second, "solo", [record(calls=(timed(30.0, backend=Backend.OPENAI),))])
    write_arm(third, "full", [record(calls=(timed(40.0, backend=Backend.OPENAI),))])
    return first, second, third


def test_the_chain_walks_back_through_every_resumed_run(tmp_path: Path) -> None:
    """Dos interrupciones en menos de una ventana dejan gasto vigente en ambos directorios."""
    first, second, third = chained(tmp_path)

    assert [run.path for run in run_chain(third)] == [third, second, first]
    assert sorted(call.latency_ms for call in chain_calls(third)) == [10.0, 20.0, 30.0, 40.0]


def test_a_chain_that_loops_is_refused(tmp_path: Path) -> None:
    """Un `resumed_from` que vuelve sobre sí mismo no termina nunca: se rechaza."""
    first, second = tmp_path / "a", tmp_path / "b"
    write_meta(first, meta(resumed_from=second))
    write_meta(second, meta(resumed_from=first))

    with pytest.raises(AuditError, match="ciclo"):
        run_chain(second)


def test_a_chain_with_a_missing_link_is_refused(tmp_path: Path) -> None:
    """Sin el directorio previo no se sabe cuánto se gastó: no se da por cero."""
    write_meta(tmp_path / "b", meta(resumed_from=tmp_path / "borrado"))

    with pytest.raises(AuditError, match="borrado"):
        run_chain(tmp_path / "b")


# ───────────────────────────────────────────── Informe ────────────────────────────────────────────


def test_the_report_cites_the_digest_of_every_input(tmp_path: Path) -> None:
    """Cada cifra se puede reproducir: el informe nombra el plan y cada archivo leído."""
    write_meta(tmp_path, meta())
    path = write_arm(tmp_path, "full", full_records())

    rendered = render_audit(read_run(tmp_path))

    assert PLAN_SHA in rendered
    assert hashlib.sha256(path.read_bytes()).hexdigest() in rendered
    assert f"python -m crypto_agents.audit {tmp_path}" in rendered
    assert "d9f422c" in rendered


def test_the_report_shows_the_weighted_rate_next_to_the_worst_pair(tmp_path: Path) -> None:
    """2 inválidas de 11 respondidas, y aparte el peor par con su nombre y sus intentos.

    `unbalanced_calls` deja 1 de 10; la segunda evaluación añade un fallo del
    decisor: 2 de 11, 18%. El peor par es decider/openai, 1 de 1.
    """
    write_meta(tmp_path, meta(arms=("full",)))
    write_arm(tmp_path, "full", full_records())

    rendered = render_audit(read_run(tmp_path))

    assert "2/11 (18%)" in rendered
    assert "decider/openai 1/1 (100%)" in rendered
    assert "peor par" in rendered


def test_what_cannot_be_determined_says_so_and_why(tmp_path: Path) -> None:
    """Un brazo sin journal y una latencia sin llamadas vivas no salen como ceros."""
    write_meta(tmp_path, meta())
    write_arm(tmp_path, "full", [record(calls=(timed(0.0, cache_hit=True),))])

    rendered = render_audit(read_run(tmp_path))

    assert "no determinado: el brazo no escribió journal" in rendered
    assert "no determinado: sin llamadas vivas" in rendered
    assert "no determinado: sin propuestas accionables" in rendered


def test_a_resumed_run_reports_the_evaluations_it_rescued(tmp_path: Path) -> None:
    """La reanudación da una segunda tirada, y el informe dice a cuántas."""
    aborted = record(activation=OPEN_GATE)
    decided = record(run_id=aborted.run_id, proposal=proposal(Action.HOLD))
    first, second = tmp_path / "a", tmp_path / "b"
    write_meta(first, meta(arms=("full",)))
    write_meta(second, meta(arms=("full",), resumed_from=first))
    write_arm(first, "full", [aborted])
    write_arm(second, "full", [decided])

    chain = run_chain(second)
    rendered = render_audit(chain[0], previous=chain[1])

    assert "| `full` | 1 | 1 |" in rendered
    assert str(first) in rendered


def test_the_command_prints_the_report_of_a_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`python -m crypto_agents.audit <dir>` no necesita configuración ni red."""
    write_meta(tmp_path, meta(arms=("full",)))
    write_arm(tmp_path, "full", full_records())

    assert main([str(tmp_path)]) == 0
    assert "## 1. Intentos" in capsys.readouterr().out


def test_the_command_fails_naming_what_is_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Un directorio que no es una corrida sale con 1 y dice por qué."""
    assert main([str(tmp_path / "nada")]) == 1
    assert "meta.json" in capsys.readouterr().err
