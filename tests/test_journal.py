"""Pruebas del registro estructurado de evaluaciones."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from crypto_agents.journal import (
    EvaluationRecord,
    InMemoryJournal,
    JournalError,
    JsonlJournal,
    build_record,
)
from crypto_agents.state import (
    Action,
    ActivationCheck,
    AgentRole,
    Backend,
    Decision,
    ExecutionMode,
    FailureKind,
    IndicatorSet,
    LLMCall,
    MarketSnapshot,
    NodeError,
    OrderIntent,
    RiskVerdict,
    Side,
    StructuredOutputMode,
    TradingState,
)

if TYPE_CHECKING:
    from pathlib import Path

NOW = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)
RUN_ID = uuid4()

SNAPSHOT = MarketSnapshot(
    run_id=RUN_ID,
    exchange="binance",
    symbol="BTC/USDT",
    timeframe="1h",
    timestamp=NOW,
    close=100.0,
    candles_digest="a" * 64,
    candles_count=300,
)
ORDER = OrderIntent(
    symbol="BTC/USDT",
    side=Action.BUY,
    size_fraction=0.05,
    reference_price=100.0,
    invalidation_price=99.0,
    mode=ExecutionMode.PAPER,
)


def call(cache_hit: bool = False, weight: float = 1.0) -> LLMCall:
    """Registro de llamada a modelo."""
    return LLMCall(
        role=AgentRole.STRUCTURE,
        backend=Backend.OLLAMA,
        model="qwen3:8b",
        structured_output=StructuredOutputMode.JSON_SCHEMA,
        quota_weight=weight,
        prompt_digest="b" * 64,
        cache_hit=cache_hit,
        valid=True,
        latency_ms=120.0,
        at=NOW,
    )


def full_state() -> TradingState:
    """Estado de una evaluación que llegó hasta la orden."""
    return TradingState(
        snapshot=SNAPSHOT,
        activation=ActivationCheck(
            should_run=True, triggers=["ma_cross_bullish"], reason="1 regla disparó"
        ),
        decision=Decision(
            action=Action.BUY,
            confidence=0.8,
            size_fraction=0.5,
            invalidation_price=99.0,
            rationale="Estructura, impulso y volumen coinciden en direccion alcista.",
            dismissed_side=Side.BEAR,
            dismissal_reason="Su contraargumento depende de un nivel que ya se perdio.",
        ),
        risk=RiskVerdict(
            approved=True, final_size_fraction=0.05, applied_limits=("max_position_fraction",)
        ),
        order=ORDER,
        calls=[call(weight=2.0), call(cache_hit=True, weight=2.0)],
    )


def record_from(state: TradingState) -> EvaluationRecord:
    """Registro construido desde un estado."""
    return build_record(state, RUN_ID, "BTC/USDT", "1h", NOW, order=state.order)


# ─────────────────────────────────────────── Contenido ────────────────────────────────────────────


def test_record_carries_the_whole_evaluation() -> None:
    """Entrada, decisión, veredicto de riesgo, orden y llamadas, en un objeto."""
    record = record_from(full_state())
    assert record.snapshot == SNAPSHOT
    assert record.decision is not None
    assert record.risk is not None
    assert record.order == ORDER
    assert len(record.calls) == 2


def test_record_reports_quota_ignoring_cache_hits() -> None:
    """La cuota del registro es la que se pagó de verdad."""
    assert record_from(full_state()).quota_used == 2.0


def test_record_knows_whether_it_traded() -> None:
    """Distingue una evaluación que operó de una que solo miró."""
    assert record_from(full_state()).traded is True
    assert record_from(TradingState()).traded is False


def test_aborted_evaluation_still_produces_a_record() -> None:
    """Una evaluación que murió pronto deja registro: es la que se querrá auditar."""
    state = TradingState(
        activation=ActivationCheck(should_run=False, triggers=[], reason="ninguna regla disparó"),
        errors=[NodeError(node="prepare_market_data", message="sin velas", at=NOW)],
    )
    record = record_from(state)
    assert record.decision is None
    assert record.order is None
    assert record.errors[0].node == "prepare_market_data"


# ─────────────────────────────────────────── Almacenes ────────────────────────────────────────────


def test_memory_journal_accumulates() -> None:
    """El journal en memoria acumula registros en orden."""
    journal = InMemoryJournal()
    journal.write(record_from(full_state()))
    journal.write(record_from(TradingState()))
    assert len(journal) == 2
    assert journal.records[0].traded is True


def test_jsonl_journal_round_trips(tmp_path: Path) -> None:
    """Una línea por evaluación, releída con el mismo esquema que la produjo."""
    journal = JsonlJournal(tmp_path / "evaluaciones.jsonl")
    journal.write(record_from(full_state()))
    journal.write(record_from(TradingState()))

    records = journal.read_all()
    assert len(records) == 2
    assert records[0].order == ORDER
    assert records[0].risk is not None
    assert records[0].risk.applied_limits == ("max_position_fraction",)
    assert records[1].traded is False


def test_jsonl_journal_writes_one_line_per_evaluation(tmp_path: Path) -> None:
    """El formato es apto para grep: nada de JSON multilínea."""
    path = tmp_path / "evaluaciones.jsonl"
    journal = JsonlJournal(path)
    journal.write(record_from(full_state()))
    journal.write(record_from(full_state()))

    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_jsonl_journal_creates_its_directory(tmp_path: Path) -> None:
    """El directorio de destino se crea solo."""
    journal = JsonlJournal(tmp_path / "anidado" / "log.jsonl")
    journal.write(record_from(TradingState()))
    assert journal.path.is_file()


def test_empty_journal_reads_as_empty(tmp_path: Path) -> None:
    """Leer un journal que aún no existe devuelve una lista vacía, no un error."""
    assert JsonlJournal(tmp_path / "todavia-no.jsonl").read_all() == []


@pytest.mark.parametrize("broken", ["{esto no es json", '{"run_id": "no-es-un-uuid"}'])
def test_an_unreadable_line_is_named_by_file_and_number(tmp_path: Path, broken: str) -> None:
    """Un journal a medias dice dónde se rompe, sea JSON inválido o un registro que no valida.

    Quien lo lee al arrancar decide con él cuánta cuota queda: un error de Pydantic
    sin archivo ni línea no le dice a nadie qué mirar, y saltarse la línea dejaría
    el recuento corto sin avisar.
    """
    path = tmp_path / "evaluaciones.jsonl"
    journal = JsonlJournal(path)
    journal.write(record_from(full_state()))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(broken + "\n")

    with pytest.raises(JournalError, match=r"evaluaciones\.jsonl.*línea 2"):
        journal.read_all()


def test_a_journal_written_before_the_failure_kinds_still_loads(tmp_path: Path) -> None:
    """Una línea con el fallo en su forma anterior se lee, y `validation` es `schema`.

    Es el archivo entero el que tiene que seguir cargando, no solo el modelo: el
    runner lo relee al arrancar para sembrar la cuota y se niega a empezar si no
    puede.
    """
    path = tmp_path / "antiguo.jsonl"
    journal = JsonlJournal(path)
    journal.write(
        record_from(full_state()).model_copy(update={"calls": (call(), call(cache_hit=True))})
    )
    line = json.loads(path.read_text(encoding="utf-8"))
    for item in line["calls"]:
        del item["failure_kind"], item["failure_message"]
        item["failure"] = None
    line["calls"][0] |= {"valid": False, "failure": {"kind": "validation", "message": "sin campos"}}
    path.write_text(json.dumps(line) + "\n", encoding="utf-8")

    [record] = journal.read_all()

    assert [item.failure_kind for item in record.calls] == [FailureKind.SCHEMA, None]
    assert record.calls[0].failure_message == "sin campos"


# ─────────────────────────────────────────── Indicadores ──────────────────────────────────────────


def test_the_record_keeps_the_indicators_the_decider_saw() -> None:
    """El ATR de la vela evaluada tiene que sobrevivir al proceso.

    Sin él, `solo` y las líneas base —que no consolidan evidencia— no dejan en el
    journal con qué reconstruir un stop común, y la puntuación común no se podría
    repetir sobre un directorio releído tres días después.
    """
    indicators = IndicatorSet(values={"ATRr_14": 2.5, "EMA_20": 100.0})
    state = full_state().model_copy(update={"indicators": indicators})

    assert record_from(state).indicators == indicators


def test_a_journal_without_indicators_still_loads(tmp_path: Path) -> None:
    """Una línea escrita antes de que el registro llevara indicadores se lee, con `None`.

    Es el archivo entero el que tiene que seguir cargando: el runner lo relee al
    arrancar para sembrar la cuota y se niega a empezar si no puede.
    """
    path = tmp_path / "antiguo.jsonl"
    journal = JsonlJournal(path)
    journal.write(record_from(full_state()))
    line = json.loads(path.read_text(encoding="utf-8"))
    line.pop("indicators", None)
    path.write_text(json.dumps(line) + "\n", encoding="utf-8")

    [record] = journal.read_all()

    assert record.indicators is None


def test_indicators_round_trip_through_the_jsonl_file(tmp_path: Path) -> None:
    """Escribir y releer no pierde ni cambia un valor."""
    indicators = IndicatorSet(values={"ATRr_14": 2.5, "EMA_20": 100.0})
    journal = JsonlJournal(tmp_path / "evaluaciones.jsonl")
    journal.write(record_from(full_state().model_copy(update={"indicators": indicators})))

    [record] = journal.read_all()

    assert record.indicators == indicators
