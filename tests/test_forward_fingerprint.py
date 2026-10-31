"""La huella del decisor de `solo`: `crypto_agents/forward_fingerprint.py`.

Lo que pide el bloque, cada parte con su mutación:

- **El orden de las claves es el del texto crudo** de la caché, no uno reordenado.
- **El tok/s es tokens de salida entre segundos de latencia**, a mano sobre la primera llamada de la
  etapa 1.
- **Los aciertos de caché no entran en la velocidad.**
- **Nada se escribe fuera de `fingerprints/`**: ni en el journal, ni en la caché, ni en las pasadas.
- **Las huellas retroactivas versionadas dan las cifras del registro**: la etapa 1 en el orden del
  esquema 140 de 140, de 48.4 a 65.2 tok/s, y de 50.0 a 59.2 en la fila de `lectura.md`; el
  bloque 1 en orden alfabético 41 de 42, de 83.6 a 107.6 con mediana 99.2. La mitad que no depende
  de la caché se recalcula aquí desde los journals versionados.

Cada motivo de una llamada sin orden, y lo que hace la huella retroactiva con lo que ya existe, con
un lote sustituido y con una huella que no se pudo calcular. Las pruebas del lote están en
`tests/test_forward.py`.
"""

from __future__ import annotations

import shutil
import statistics
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

import crypto_agents.forward_fingerprint as module
from crypto_agents.audit import arm_journal_path, file_sha256
from crypto_agents.cache import CacheEntry, JsonFileResponseCache, cache_key
from crypto_agents.forward_data import STAGE1_DIR
from crypto_agents.forward_fingerprint import (
    FINGERPRINT_DIR,
    STAGE1_FINGERPRINT,
    DeciderFingerprint,
    KeyOrder,
    Speed,
    fingerprint,
    key_order,
    retroactive,
    tokens_per_second,
)
from crypto_agents.forward_record import SUPERSEDED_FILE, TARGET_ARM, SupersededMark
from crypto_agents.journal import JsonlJournal
from crypto_agents.state import AgentRole, FailureKind, Proposal
from tests.test_forward import ROOT
from tests.test_net_outcomes import mutated
from tests.test_paired_dispersion import snapshot

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from crypto_agents.journal import EvaluationRecord
    from crypto_agents.state import LLMCall

STAGE1_JOURNAL = arm_journal_path(ROOT / STAGE1_DIR, TARGET_ARM)
VERSIONED = ROOT / "data" / "forward" / FINGERPRINT_DIR
BLOCK_ONE = VERSIONED / "2026-10-22" / "20261030T000327Z.json"
BLOCK_ONE_JOURNAL = ROOT / "data/forward/batches/2026-10-22/20261030T000327Z/solo.jsonl"

SCHEMA_ORDER = ("action", "confidence", "size_fraction", "invalidation_price", "rationale")
"""El orden en que el esquema de `Proposal` declara los argumentos."""

ALPHABETICAL = ("action", "confidence", "invalidation_price", "rationale", "size_fraction")

UNSORTED = (
    '{"rationale": "r", "action": "hold", "size_fraction": 0.0, "confidence": 0.5, '
    '"invalidation_price": null}'
)
"""Un texto con las claves en un orden que no es ni el del esquema ni el alfabético."""

WINDOW = (
    datetime(2026, 10, 9, 10, 19, 42, tzinfo=UTC),
    datetime(2026, 10, 9, 10, 24, 51, tzinfo=UTC),
)
"""La fila «etapa 1, mismos minutos que la segunda tanda» de `lectura.md`: las llamadas cuyo `at`
cae entre las 10:19:42 y las 10:24:50, los dos segundos incluidos."""


def stage1_records(count: int) -> list[EvaluationRecord]:
    return JsonlJournal(STAGE1_JOURNAL).read_all()[:count]


def write_journal(path: Path, records: Sequence[EvaluationRecord]) -> Path:
    journal = JsonlJournal(path)
    for record in records:
        journal.write(record)
    return path


def store(cache: Path, call: LLMCall, raw: str) -> None:
    """La entrada que el router habría dejado para esa llamada, con `raw` como texto."""
    entry = CacheEntry(
        backend=call.backend,
        model=call.model,
        role=AgentRole.DECIDER,
        prompt_digest=call.prompt_digest,
        schema_name=Proposal.__name__,
        structured_output=call.structured_output,
        raw=raw,
    )
    JsonFileResponseCache(cache).set(entry.key, entry.model_dump_json())


def key_of(call: LLMCall) -> str:
    return cache_key(call.backend, call.model, call.prompt_digest, Proposal, call.structured_output)


# ───────────────────────────────────────── El orden de claves ─────────────────────────────────────


def assert_the_order_is_the_raw_texts(tmp_path: Path) -> None:
    (record,) = stage1_records(1)
    journal = write_journal(tmp_path / "solo.jsonl", [record])
    store(tmp_path / "caché", record.calls[0], UNSORTED)

    found = fingerprint(journal, tmp_path / "caché")

    keys = ("rationale", "action", "size_fraction", "confidence", "invalidation_price")
    assert [call.key_order for call in found.calls] == [keys]
    assert found.summary is not None
    assert found.summary.orders == (KeyOrder(keys=keys, calls=1),)


def test_the_key_order_is_the_raw_texts(tmp_path: Path) -> None:
    assert_the_order_is_the_raw_texts(tmp_path)


def test_mutation_sorted_keys_are_caught(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mutant = mutated(
        key_order,
        "return tuple(key for key, _ in parsed)",
        "return tuple(sorted(key for key, _ in parsed))",
        module,
    )
    monkeypatch.setattr(module, "key_order", mutant)
    with pytest.raises(AssertionError):
        assert_the_order_is_the_raw_texts(tmp_path)


def test_the_key_order_keeps_duplicates_and_refuses_what_is_not_an_object() -> None:
    assert key_order('{"b": 1, "a": {"z": 0, "y": 1}, "b": 2}') == ("b", "a", "b")
    for raw in ("[1, 2]", '"hold"', "3"):
        with pytest.raises(ValueError, match="no un objeto"):
            key_order(raw)
    with pytest.raises(ValueError):
        key_order("{")


# ──────────────────────────────────────────── La velocidad ────────────────────────────────────────


def assert_the_speed_is_tokens_over_seconds() -> None:
    """La primera llamada de la etapa 1: 2 328 tokens en 41.44977 s, 56.16 tok/s."""
    (record,) = stage1_records(1)
    call = record.calls[0]
    assert (call.completion_tokens, call.prompt_tokens) == (2328, 1151)
    assert call.latency_ms == pytest.approx(41449.7693)

    speed = module.tokens_per_second
    assert speed(call) == pytest.approx(56.1644, abs=1e-4)
    assert speed(call.model_copy(update={"completion_tokens": None})) is None
    assert speed(call.model_copy(update={"latency_ms": 0.0})) is None


def test_the_speed_is_tokens_over_seconds() -> None:
    assert_the_speed_is_tokens_over_seconds()


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("(call.latency_ms / 1000)", "call.latency_ms"),
        ("return call.completion_tokens /", "return call.prompt_tokens /"),
    ],
    ids=["milisegundos", "tokens de entrada"],
)
def test_mutation_another_speed_is_caught(
    old: str, new: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "tokens_per_second", mutated(tokens_per_second, old, new, module))
    with pytest.raises(AssertionError):
        assert_the_speed_is_tokens_over_seconds()


def assert_cache_hits_stay_out_of_the_speed(tmp_path: Path) -> None:
    """Un acierto de caché con latencia: sus 10 000 tok/s serían el máximo si entrara."""
    first, second = stage1_records(2)
    hit = second.calls[0].model_copy(
        update={"cache_hit": True, "latency_ms": 1000.0, "completion_tokens": 10_000}
    )
    records = [first, second.model_copy(update={"calls": (hit,)})]
    journal = write_journal(tmp_path / "solo.jsonl", records)
    for record in records:
        store(tmp_path / "caché", record.calls[0], UNSORTED)

    found = fingerprint(journal, tmp_path / "caché")

    assert found.summary is not None
    assert [call.tokens_per_second for call in found.calls][1] == 10_000.0
    assert (found.summary.calls, found.summary.live_calls) == (2, 1)
    alone = tokens_per_second(first.calls[0])
    assert alone is not None
    assert found.summary.speed == Speed(calls=1, minimum=alone, median=alone, maximum=alone)


def test_cache_hits_stay_out_of_the_speed(tmp_path: Path) -> None:
    assert_cache_hits_stay_out_of_the_speed(tmp_path)


def test_mutation_cache_hits_in_the_speed_are_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mutant = mutated(
        module.summarise,
        "live = [call for call in calls if not call.cache_hit]",
        "live = list(calls)",
        module,
    )
    monkeypatch.setattr(module, "summarise", mutant)
    with pytest.raises(AssertionError):
        assert_cache_hits_stay_out_of_the_speed(tmp_path)


# ──────────────────────────────────────── Sin orden, y por qué ────────────────────────────────────


def test_each_call_without_an_order_says_why(tmp_path: Path) -> None:
    records = stage1_records(5)
    calls = [record.calls[0] for record in records]
    lost = calls[0].model_copy(
        update={
            "valid": False,
            "failure_kind": FailureKind.TRANSPORT,
            "failure_message": "InternalServerError: Unknown Error",
            "completion_tokens": None,
        }
    )
    records[0] = records[0].model_copy(update={"calls": (lost,)})
    cache = tmp_path / "caché"
    other = calls[1].model_copy(update={"prompt_digest": calls[2].prompt_digest})
    entry = CacheEntry(
        backend=other.backend,
        model=other.model,
        role=AgentRole.DECIDER,
        prompt_digest=other.prompt_digest,
        schema_name=Proposal.__name__,
        structured_output=other.structured_output,
        raw=UNSORTED,
    )
    JsonFileResponseCache(cache).set(key_of(calls[1]), entry.model_dump_json())
    JsonFileResponseCache(cache).set(key_of(calls[3]), '{"no": "es un sobre"}')
    store(cache, calls[4], "[1, 2]")
    journal = write_journal(tmp_path / "solo.jsonl", records)

    found = fingerprint(journal, cache)

    reasons = [call.key_order_error for call in found.calls]
    assert reasons == [
        "el intento no trajo texto (transport)",
        f"la entrada {key_of(calls[1])} no es la de esta llamada",
        f"falta la entrada {key_of(calls[2])} en la caché",
        f"la entrada {key_of(calls[3])} no es un sobre de caché",
        f"el texto de la entrada {key_of(calls[4])} no es un objeto JSON: es un list de JSON, no "
        "un objeto",
    ]
    assert found.summary is not None
    assert (found.summary.without_order, found.summary.orders) == (5, ())
    assert found.calls[0].tokens_per_second is None


def test_a_fingerprint_that_cannot_be_computed_says_why_and_creates_nothing(
    tmp_path: Path,
) -> None:
    journal = write_journal(tmp_path / "solo.jsonl", stage1_records(1))
    missing = fingerprint(tmp_path / "otro.jsonl", tmp_path)
    assert missing.error == f"no existe el journal {tmp_path / 'otro.jsonl'}"
    no_cache = fingerprint(journal, tmp_path / "caché")
    assert no_cache.error == f"no existe la caché {tmp_path / 'caché'}"
    assert not (tmp_path / "caché").exists(), "leer la huella creó la caché"
    broken = tmp_path / "roto.jsonl"
    broken.write_text("{\n", encoding="utf-8")
    unreadable = fingerprint(broken, tmp_path)
    assert unreadable.error is not None
    assert "roto.jsonl" in unreadable.error
    for found in (missing, no_cache, unreadable):
        assert (found.calls, found.summary, found.journal_sha256) == ((), None, None)


# ─────────────────────────────────────── La huella retroactiva ────────────────────────────────────


def study_tree(tmp_path: Path) -> tuple[Path, Path]:
    """Un estudio en `tmp_path`: la etapa 1 y la pasada del bloque 1 copiadas, y dos cachés con
    el texto de las dos primeras llamadas de cada una. Devuelve la raíz y la pasada."""
    stage1 = tmp_path / "etapa1"
    stage1.mkdir()
    shutil.copy(STAGE1_JOURNAL, stage1 / STAGE1_JOURNAL.name)
    batch = tmp_path / "estudio" / "batches" / "2026-10-22"
    directory = batch / BLOCK_ONE_JOURNAL.parent.name
    shutil.copytree(BLOCK_ONE_JOURNAL.parent, directory)
    shutil.copytree(
        BLOCK_ONE_JOURNAL.parent.parent / "20261030T000029Z", batch / "20261030T000029Z"
    )
    for journal, cache in ((STAGE1_JOURNAL, "caché-etapa1"), (BLOCK_ONE_JOURNAL, "caché")):
        for record in JsonlJournal(journal).read_all()[:2]:
            store(tmp_path / cache, record.calls[0], UNSORTED)
    return tmp_path / "estudio", directory


def run_retroactive(tmp_path: Path) -> module.Retroactive:
    return retroactive(
        tmp_path / "estudio" / FINGERPRINT_DIR,
        tmp_path / "etapa1",
        tmp_path / "caché-etapa1",
        tmp_path / "estudio" / "batches",
        tmp_path / "caché",
    )


def assert_only_the_fingerprints_are_written(tmp_path: Path) -> None:
    """Todo el árbol, byte a byte: lo único nuevo está bajo `fingerprints/`, y nada cambia."""
    study_tree(tmp_path)
    before = snapshot(tmp_path)

    found = run_retroactive(tmp_path)

    after = snapshot(tmp_path)
    prints = f"estudio/{FINGERPRINT_DIR}/"
    assert {name: digest for name, digest in after.items() if not name.startswith(prints)} == before
    assert sorted(name for name in after if name not in before) == [
        f"{prints}2026-10-22/20261030T000327Z.json",
        f"{prints}{STAGE1_FINGERPRINT}",
    ]
    assert [path.relative_to(tmp_path).as_posix() for path in found.written] == [
        f"{prints}{STAGE1_FINGERPRINT}",
        f"{prints}2026-10-22/20261030T000327Z.json",
    ]


def test_only_the_fingerprints_are_written(tmp_path: Path) -> None:
    assert_only_the_fingerprints_are_written(tmp_path)


def test_mutation_a_fingerprint_inside_the_pass_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mutant = mutated(
        module.pass_fingerprint_path,
        'return root / pass_dir.parent.name / f"{pass_dir.name}.json"',
        'return pass_dir / "huella.json"',
        module,
    )
    monkeypatch.setattr(module, "pass_fingerprint_path", mutant)
    with pytest.raises(AssertionError):
        assert_only_the_fingerprints_are_written(tmp_path)


def test_the_retroactive_fingerprint_is_incomplete_without_the_whole_cache(tmp_path: Path) -> None:
    """Las cachés del árbol solo tienen dos entradas por journal: sale con 1 y lo dice."""
    study_tree(tmp_path)
    found = run_retroactive(tmp_path)
    assert found.code == 1
    assert len(found.lines) == 2
    assert "140 llamadas vivas de 140" in found.lines[0]
    assert "138 sin orden" in found.lines[0]
    assert "42 llamadas vivas de 42" in found.lines[1]
    assert "40 sin orden" in found.lines[1]


def test_the_retroactive_fingerprint_never_rewrites(tmp_path: Path) -> None:
    study_tree(tmp_path)
    run_retroactive(tmp_path)
    before = snapshot(tmp_path)

    again = run_retroactive(tmp_path)

    assert snapshot(tmp_path) == before
    assert (again.code, again.written) == (0, ())
    assert [line.endswith("ya existe, no se reescribe") for line in again.lines] == [True, True]


def test_the_retroactive_fingerprint_skips_a_superseded_batch(tmp_path: Path) -> None:
    _, directory = study_tree(tmp_path)
    mark = SupersededMark(
        at=datetime(2026, 10, 30, 1, tzinfo=UTC), by="20261030T010000Z", reasons=("simulada",)
    )
    (directory / SUPERSEDED_FILE).write_text(mark.model_dump_json(), encoding="utf-8")

    found = run_retroactive(tmp_path)

    assert [path.name for path in found.written] == [STAGE1_FINGERPRINT]
    assert "tiene pasadas sustituidas" in found.lines[1]


def test_a_fingerprint_that_could_not_be_computed_exits_one(tmp_path: Path) -> None:
    study_tree(tmp_path)
    shutil.rmtree(tmp_path / "caché-etapa1")
    found = run_retroactive(tmp_path)
    assert found.code == 1
    assert "no se pudo calcular: no existe la caché" in found.lines[0]


# ─────────────────────────────────────── Las huellas versionadas ──────────────────────────────────


def versioned(path: Path) -> DeciderFingerprint:
    return DeciderFingerprint.model_validate_json(path.read_text(encoding="utf-8"))


def assert_the_journal_half(found: DeciderFingerprint, journal: Path, cache: str) -> None:
    """Lo que no depende de la caché, recalculado desde el journal versionado que cita."""
    assert ROOT / found.journal == journal
    assert (found.journal_sha256, found.cache, found.error) == (file_sha256(journal), cache, None)
    rows = [
        (record.run_id, call.at, call.cache_hit, call.completion_tokens, call.latency_ms)
        for record in JsonlJournal(journal).read_all()
        for call in record.calls
        if call.role is AgentRole.DECIDER
    ]
    assert [
        (call.run_id, call.at, call.cache_hit, call.completion_tokens, call.latency_ms)
        for call in found.calls
    ] == rows
    for call in found.calls:
        assert call.completion_tokens is not None
        assert call.tokens_per_second == call.completion_tokens / (call.latency_ms / 1000)


def figures(speeds: Sequence[float]) -> tuple[int, float, float, float]:
    return (
        len(speeds),
        round(min(speeds), 1),
        round(max(speeds), 1),
        round(statistics.median(speeds), 1),
    )


def test_the_stage1_fingerprint_is_the_registers() -> None:
    found = versioned(VERSIONED / STAGE1_FINGERPRINT)
    assert_the_journal_half(found, STAGE1_JOURNAL, "var/ablation-cache")
    assert tuple(Proposal.model_fields) == SCHEMA_ORDER
    summary = found.summary
    assert summary is not None
    assert (summary.calls, summary.live_calls, summary.without_order) == (140, 140, 0)
    assert summary.orders == (KeyOrder(keys=SCHEMA_ORDER, calls=140),)
    assert summary.speed is not None
    speed = summary.speed
    assert (speed.calls, round(speed.minimum, 1), round(speed.maximum, 1)) == (140, 48.4, 65.2)
    assert round(speed.median, 1) == 55.4
    window = [
        call.tokens_per_second
        for call in found.calls
        if WINDOW[0] <= call.at < WINDOW[1] and call.tokens_per_second is not None
    ]
    assert figures(window) == (7, 50.0, 59.2, 56.3)


def test_the_block_one_fingerprint_is_the_registers() -> None:
    found = versioned(BLOCK_ONE)
    assert_the_journal_half(found, BLOCK_ONE_JOURNAL, "var/forward/cache")
    assert tuple(sorted(SCHEMA_ORDER)) == ALPHABETICAL
    summary = found.summary
    assert summary is not None
    assert (summary.calls, summary.live_calls, summary.without_order) == (42, 42, 0)
    assert summary.orders == (
        KeyOrder(keys=ALPHABETICAL, calls=41),
        KeyOrder(
            keys=("action", "confidence", "invalidation_price", "size_fraction", "rationale"),
            calls=1,
        ),
    )
    assert summary.speed is not None
    speed = summary.speed
    assert (speed.calls, round(speed.minimum, 1), round(speed.maximum, 1)) == (42, 83.6, 107.6)
    assert round(speed.median, 1) == 99.2


def test_the_pass_stopped_by_the_revised_candles_has_no_fingerprint() -> None:
    """La primera pasada del bloque 1 se detuvo antes de llamar a nadie: no tiene huella."""
    assert (BLOCK_ONE_JOURNAL.parent.parent / "20261030T000029Z").is_dir()
    assert not (BLOCK_ONE.parent / "20261030T000029Z.json").exists()
    assert [path.name for path in BLOCK_ONE.parent.iterdir()] == [BLOCK_ONE.name]
