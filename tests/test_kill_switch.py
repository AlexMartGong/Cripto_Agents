"""Pruebas del interruptor de parada.

El criterio de la fase: un decisor que emite `buy` con confianza y tamaño máximos
no consigue sacar una orden si la parada está puesta. Se prueba sobre el grafo
real, no sobre `apply_risk` a solas, porque lo que hay que demostrar es que no
existe camino al mercado que lo esquive.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from crypto_agents.execution import PaperExecutor
from crypto_agents.graph import AgentContext, PipelineVariant, build_graph
from crypto_agents.journal import EvaluationRecord, InMemoryJournal
from crypto_agents.risk import (
    AnyKillSwitch,
    FileKillSwitch,
    RiskLimits,
    StaticKillSwitch,
    apply_risk,
)
from crypto_agents.state import Action, Decision, OrderIntent, Side, TradingState
from tests.conftest import HEALTHY
from tests.test_graph import make_context

if TYPE_CHECKING:
    from pathlib import Path

NOW = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)


def submitted(context: AgentContext) -> list[OrderIntent]:
    """Órdenes que llegaron al ejecutor. Estrecha el protocolo al falso de pruebas."""
    executor = context.executor
    assert isinstance(executor, PaperExecutor)
    return executor.submitted


def journalled(context: AgentContext) -> list[EvaluationRecord]:
    """Registros escritos durante la evaluación."""
    journal = context.journal
    assert isinstance(journal, InMemoryJournal)
    return journal.records


MAXIMAL_BUY = Decision(
    action=Action.BUY,
    confidence=1.0,
    size_fraction=1.0,
    invalidation_price=99.0,
    rationale="Toda la evidencia apunta en la misma direccion sin contraparte creible.",
    dismissed_side=Side.BEAR,
    dismissal_reason="Su contraargumento depende de un nivel que ya se perdio.",
).model_dump_json()


# ─────────────────────────────────── Los interruptores ────────────────────────────────────────────


def test_the_file_switch_is_engaged_only_while_the_file_exists(tmp_path: Path) -> None:
    """Poner y quitar el centinela es todo el mecanismo."""
    sentinel = tmp_path / "STOP"
    switch = FileKillSwitch(sentinel)

    assert switch.engaged() is False
    sentinel.write_text("parado", encoding="utf-8")
    assert switch.engaged() is True
    sentinel.unlink()
    assert switch.engaged() is False


def test_the_file_switch_is_not_cached_between_calls(tmp_path: Path) -> None:
    """Se relee en cada consulta.

    Un valor cacheado en un runner de velas de 4h no se enteraría del archivo hasta
    el siguiente arranque, que es exactamente cuando el interruptor no sirve.
    """
    sentinel = tmp_path / "STOP"
    switch = FileKillSwitch(sentinel)
    assert switch.engaged() is False

    sentinel.write_text("parado", encoding="utf-8")
    assert switch.engaged() is True, "no releyó el archivo"


def test_an_unreadable_sentinel_engages_the_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fallo de E/S: se para. Un kill switch que falla abierto no es un kill switch."""
    switch = FileKillSwitch("/ruta/que/no/importa")

    def explode(self: object) -> bool:
        raise OSError("permiso denegado")

    monkeypatch.setattr("pathlib.Path.exists", explode)
    assert switch.engaged() is True


def test_any_switch_engages_when_either_does() -> None:
    """Configuración y archivo conviven sin que uno tenga prioridad sobre el otro."""
    assert AnyKillSwitch(StaticKillSwitch(False), StaticKillSwitch(False)).engaged() is False
    assert AnyKillSwitch(StaticKillSwitch(True), StaticKillSwitch(False)).engaged() is True
    assert AnyKillSwitch(StaticKillSwitch(False), StaticKillSwitch(True)).engaged() is True
    assert AnyKillSwitch().engaged() is False


# ───────────────────────────── El gate lo respeta, y nadie lo esquiva ─────────────────────────────


@pytest.mark.asyncio
async def test_the_kill_switch_stops_a_maximal_buy(tmp_path: Path) -> None:
    """Criterio de aceptación de la fase.

    El decisor emite la decisión más convincente que el contrato permite —comprar,
    confianza 1.0, todo el capital— y no sale nada. La convicción de un modelo no
    es una entrada del gate de riesgo.
    """
    sentinel = tmp_path / "STOP"
    sentinel.write_text("parado", encoding="utf-8")
    context, _ = make_context(overrides={"decider": MAXIMAL_BUY})
    context = replace(context, kill_switch=FileKillSwitch(sentinel))

    graph = build_graph()
    raw = await graph.ainvoke(
        TradingState(), context=context, config={"configurable": {"thread_id": "stop"}}
    )
    result = TradingState.model_validate(raw)

    assert result.decision is not None, "el decisor sí decidió"
    assert result.decision.action is Action.BUY
    assert result.decision.confidence == 1.0
    assert result.risk is not None
    assert result.risk.approved is False
    assert result.risk.veto_rule == "kill_switch"
    assert result.risk.final_size_fraction == 0.0
    assert result.order is None
    assert submitted(context) == []


@pytest.mark.asyncio
async def test_the_stopped_evaluation_is_still_journaled(tmp_path: Path) -> None:
    """Una parada deja rastro: es lo que se querrá auditar cuando falten órdenes."""
    sentinel = tmp_path / "STOP"
    sentinel.write_text("parado", encoding="utf-8")
    context, _ = make_context(overrides={"decider": MAXIMAL_BUY})
    context = replace(context, kill_switch=FileKillSwitch(sentinel))

    graph = build_graph()
    await graph.ainvoke(
        TradingState(), context=context, config={"configurable": {"thread_id": "stop-journal"}}
    )

    record = journalled(context)[0]
    assert record.risk is not None
    assert record.risk.veto_rule == "kill_switch"
    assert record.order is None


@pytest.mark.asyncio
@pytest.mark.parametrize("variant", list(PipelineVariant))
async def test_no_pipeline_variant_escapes_the_kill_switch(
    variant: PipelineVariant, tmp_path: Path
) -> None:
    """Ninguna forma del grafo tiene un camino al mercado que salte el gate."""
    sentinel = tmp_path / "STOP"
    sentinel.write_text("parado", encoding="utf-8")
    context, _ = make_context(overrides={"decider": MAXIMAL_BUY})
    context = replace(context, kill_switch=FileKillSwitch(sentinel))

    raw = await build_graph(variant).ainvoke(
        TradingState(), context=context, config={"configurable": {"thread_id": f"stop-{variant}"}}
    )
    result = TradingState.model_validate(raw)

    assert result.order is None
    assert submitted(context) == []


@pytest.mark.asyncio
async def test_without_the_sentinel_the_same_decision_produces_an_order(tmp_path: Path) -> None:
    """El control: sin parada, esa misma decisión sí opera.

    Sin este caso, los de arriba pasarían igual si el pipeline estuviera roto por
    cualquier otra razón.
    """
    context, _ = make_context(overrides={"decider": MAXIMAL_BUY})
    context = replace(context, kill_switch=FileKillSwitch(tmp_path / "AUSENTE"))

    raw = await build_graph().ainvoke(
        TradingState(), context=context, config={"configurable": {"thread_id": "sin-stop"}}
    )
    result = TradingState.model_validate(raw)

    assert result.order is not None
    assert result.risk is not None
    assert result.risk.approved is True


# ──────────────────────────────── El gate sigue siendo puro ───────────────────────────────────────


def test_apply_risk_still_reads_a_plain_boolean() -> None:
    """El I/O queda en el nodo; la regla de veto no toca el disco."""
    verdict = apply_risk(
        Decision.model_validate_json(MAXIMAL_BUY),
        HEALTHY,
        RiskLimits(kill_switch=True),
        NOW,
        100.0,
    )
    assert verdict.approved is False
    assert verdict.veto_rule == "kill_switch"
