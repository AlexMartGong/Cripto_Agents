"""Pruebas del punto de entrada operativo.

Nada aquí arranca el bucle ni toca la red: se comprueba que los subcomandos de
vigilar y detener hacen lo que dicen, y que el código de salida sirve para
encadenarlos en un script.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from crypto_agents import cli
from crypto_agents.bootstrap import build_router, journal_calls
from crypto_agents.cli import main
from crypto_agents.doctor import CheckResult, CheckStatus
from crypto_agents.journal import EvaluationRecord, JsonlJournal
from crypto_agents.quota import QuotaExhaustedError
from crypto_agents.risk import FileKillSwitch
from crypto_agents.settings import ConfigError, load_settings
from crypto_agents.state import (
    Action,
    AgentRole,
    Backend,
    Decision,
    LLMCall,
    NodeError,
    Proposal,
    Side,
)
from tests.conftest import CHEAP, SCARCE, role_map

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable
    from pathlib import Path

    from crypto_agents.settings import Settings


def _moment() -> datetime:
    """Instante actual real.

    El CLI usa el reloj del sistema y no admite uno inyectado, así que los
    registros se sellan ahora: con una hora fija, la alerta de cuota dependería de
    a qué hora del día se corran las pruebas, porque la ventana es deslizante.
    """
    return datetime.now(UTC)


@pytest.fixture(autouse=True)
def _clean_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Aísla del entorno real del desarrollador, variables y archivo.

    El `chdir` no es cosmético: a diferencia del resto de la suite, estas pruebas
    entran por `main()`, y el CLI sí pide el `.env` por ruta relativa. Corriendo en
    la raíz del repositorio, el archivo real de quien opera se cuela en los dos
    casos que comprueban qué pasa cuando falta configuración —y el de `run` no
    falla: arranca el bucle de verdad y se duerme hasta el siguiente cierre de
    vela, colgando la suite entera.
    """
    for key in list(os.environ):
        if key.startswith("CA_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Configuración mínima válida apuntando a un directorio temporal."""
    roles = {
        role.value: {
            "primary": {
                "backend": "ollama",
                "model": choice.primary.model,
                "family": choice.primary.family,
                "structured_output": "json_schema",
                "quota_per_window": 100,
            }
        }
        for role, choice in role_map(primary=CHEAP).items()
    }
    monkeypatch.setenv("CA_ROLES", json.dumps(roles))
    monkeypatch.setenv("CA_OLLAMA__HOST", "http://localhost:11434")
    monkeypatch.setenv("CA_OPERATIONS__JOURNAL_PATH", str(tmp_path / "journal.jsonl"))
    monkeypatch.setenv("CA_OPERATIONS__CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CA_OPERATIONS__KILL_SWITCH_FILE", str(tmp_path / "STOP"))
    return tmp_path


def write_records(path: Path, count: int = 3, traded: bool = True) -> None:
    """Journal con algunas evaluaciones."""
    journal = JsonlJournal(path)
    for index in range(count):
        journal.write(
            EvaluationRecord(
                run_id=uuid4(),
                at=_moment(),
                symbol="BTC/USDT" if index % 2 == 0 else "ETH/USDT",
                timeframe="4h",
                decision=Decision(
                    action=Action.BUY,
                    confidence=0.7,
                    size_fraction=0.2,
                    invalidation_price=99.0,
                    rationale="La estructura acompana y el impulso no esta agotado todavia.",
                    dismissed_side=Side.BEAR,
                    dismissal_reason="Su nivel de referencia ya se perdio en la vela previa.",
                )
                if traded
                else None,
                errors=()
                if traded
                else (NodeError(node="structure", message="cuota agotada", at=_moment()),),
                calls=(
                    LLMCall(
                        role=AgentRole.STRUCTURE,
                        backend=Backend.OLLAMA,
                        model=CHEAP.model,
                        structured_output=CHEAP.structured_output,
                        quota_weight=1.0,
                        prompt_digest="a" * 64,
                        valid=True,
                        latency_ms=1.0,
                        at=_moment(),
                    ),
                ),
            )
        )


# ────────────────────────────────────────── stop y resume ─────────────────────────────────────────


def test_stop_creates_the_sentinel(workspace: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Detener es escribir un archivo; funciona sin proceso corriendo."""
    assert main(["stop"]) == 0

    sentinel = workspace / "STOP"
    assert sentinel.exists()
    assert FileKillSwitch(sentinel).engaged() is True
    assert "parada activada" in capsys.readouterr().out


def test_resume_removes_it(workspace: Path) -> None:
    """Reanudar es borrarlo. No arranca nada por sí solo: deja de vetar."""
    main(["stop"])
    assert main(["resume"]) == 0
    assert not (workspace / "STOP").exists()


def test_resume_without_a_stop_is_not_an_error(workspace: Path) -> None:
    """Idempotente: quitar una parada que no existe no debe fallar un script."""
    del workspace
    assert main(["resume"]) == 0


def test_resume_warns_when_config_still_stops_everything(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Quitar el archivo no basta si la parada también está en configuración.

    Sin este aviso, alguien retiraría el centinela, vería «parada retirada» y
    esperaría órdenes que nunca van a salir.
    """
    del workspace
    monkeypatch.setenv("CA_RISK__KILL_SWITCH", "true")
    assert main(["resume"]) == 1
    assert "sigue activo por configuración" in capsys.readouterr().err


def test_stop_creates_the_parent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workspace: Path
) -> None:
    """El centinela funciona en la primera ejecución, sin preparar nada antes."""
    del workspace
    monkeypatch.setenv("CA_OPERATIONS__KILL_SWITCH_FILE", str(tmp_path / "nuevo" / "STOP"))
    assert main(["stop"]) == 0
    assert (tmp_path / "nuevo" / "STOP").exists()


# ─────────────────────────────────────────────  status ────────────────────────────────────────────


def test_status_reports_the_stop_state(workspace: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Lo primero que se pregunta: ¿está parado?"""
    del workspace
    main(["status"])
    assert "parada        : no" in capsys.readouterr().out

    main(["stop"])
    capsys.readouterr()
    main(["status"])
    assert "parada        : SÍ" in capsys.readouterr().out


def test_status_counts_the_journal(workspace: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Cuántas evaluaciones hay y cuántas operaron."""
    write_records(workspace / "journal.jsonl", count=3)
    main(["status"])

    output = capsys.readouterr().out
    assert "3 evaluaciones" in output
    assert "decididas   : 3" in output


# ──────────────────────────────────────────── alerts ──────────────────────────────────────────────


def test_alerts_exits_zero_when_quiet(workspace: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Sin alertas, código 0: se puede encadenar en un script."""
    write_records(workspace / "journal.jsonl", count=1)
    assert main(["alerts"]) == 0
    assert "sin alertas" in capsys.readouterr().out


def test_alerts_exits_one_when_something_fires(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Con alertas, código 1, y cada línea dice qué la disparó."""
    write_records(workspace / "journal.jsonl", count=90)
    assert main(["alerts"]) == 1
    assert "quota_low" in capsys.readouterr().out


def test_alerts_under_payg_print_not_applicable_and_no_quota_alert(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Los mismos 90 registros que disparan `quota_low` con Go no disparan nada con `payg`."""
    monkeypatch.setenv("CA_BILLING", "payg")
    write_records(workspace / "journal.jsonl", count=90)
    assert main(["alerts"]) == 0
    output = capsys.readouterr().out
    assert "cuota: no aplica (payg)" in output
    assert "quota_low" not in output


# ───────────────────────────────────────────── query ──────────────────────────────────────────────


def test_query_filters_by_symbol(workspace: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Un mercado a la vez."""
    write_records(workspace / "journal.jsonl", count=4)
    main(["query", "--symbol", "ETH/USDT"])
    assert "2 de 4 evaluaciones" in capsys.readouterr().out


def test_query_groups_by_abort_cause(workspace: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Por qué no se operó es la mitad de las preguntas."""
    write_records(workspace / "journal.jsonl", count=2, traded=False)
    main(["query", "--group-by-cause"])
    assert "structure" in capsys.readouterr().out


def test_query_rejects_an_unknown_action(workspace: Path) -> None:
    """Un filtro mal escrito falla en el parser, no devuelve vacío en silencio."""
    del workspace
    with pytest.raises(SystemExit):
        main(["query", "--action", "comprar"])


# ───────────────────────────────────────────── doctor ─────────────────────────────────────────────


def _results(*statuses: CheckStatus) -> tuple[CheckResult, ...]:
    """Resultados de mentira, uno por estado pedido."""
    return tuple(
        CheckResult(name=f"comprobacion-{index}", status=status, detail="detalle")
        for index, status in enumerate(statuses)
    )


def test_doctor_exits_zero_when_everything_checks_out(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Con las cuatro comprobaciones en verde, el comando sale 0 y las nombra."""
    del workspace
    results = _results(CheckStatus.OK, CheckStatus.OK)
    monkeypatch.setattr(cli, "run_checks", _stub_checks(results))

    assert main(["doctor"]) == 0
    assert "comprobacion-0" in capsys.readouterr().out


def test_doctor_exits_one_when_a_check_fails(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Sale 1 para que un script pueda negarse a arrancar el bucle detrás."""
    del workspace
    results = _results(CheckStatus.OK, CheckStatus.FAIL)
    monkeypatch.setattr(cli, "run_checks", _stub_checks(results))

    assert main(["doctor"]) == 1
    captured = capsys.readouterr()
    assert "FALLA" in captured.out
    assert "no está listo para operar" in captured.err


def _stub_checks(results: tuple[CheckResult, ...]) -> Callable[[Settings], Awaitable[object]]:
    """Sustituto de `run_checks` que no toca nada de fuera."""

    async def stub(settings: Settings) -> tuple[CheckResult, ...]:
        del settings
        return results

    return stub


# ─────────────────────────────────────────── Arranque ─────────────────────────────────────────────


def test_a_missing_configuration_names_what_is_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Sin `CA_ROLES` el comando dice cuál falta, en vez de volcar un error de Pydantic."""
    monkeypatch.setenv("CA_OPERATIONS__JOURNAL_PATH", "/tmp/no-existe.jsonl")
    assert main(["status"]) == 1
    assert "CA_ROLES" in capsys.readouterr().err


def test_run_without_symbols_refuses_to_start(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """El bucle no arranca sin saber qué evaluar, y lo dice nombrando la variable."""
    del workspace
    assert main(["run"]) == 1
    assert "CA_RUNNER__SYMBOLS" in capsys.readouterr().err


# ─────────────────────────────── La cuota sobrevive a un reinicio ─────────────────────────────────
# El contador vive en memoria y la ventana del proveedor no. Un bucle que se reinicia
# a mitad de ventana con el contador a cero cree tener el presupuesto entero.


def spend(path: Path, choice_calls: int) -> None:
    """Journal con llamadas remotas del decisor hechas ahora mismo."""
    JsonlJournal(path).write(
        EvaluationRecord(
            run_id=uuid4(),
            at=_moment(),
            symbol="BTC/USDT",
            timeframe="4h",
            calls=tuple(
                LLMCall(
                    role=AgentRole.DECIDER,
                    backend=SCARCE.backend,
                    model=SCARCE.model,
                    structured_output=SCARCE.structured_output,
                    quota_weight=SCARCE.quota_weight,
                    prompt_digest="b" * 64,
                    valid=True,
                    latency_ms=1.0,
                    at=_moment(),
                )
                for _ in range(choice_calls)
            ),
        )
    )


@pytest.mark.asyncio
async def test_the_router_starts_knowing_what_the_journal_says_was_spent(tmp_path: Path) -> None:
    """Dos llamadas de peso 2 contra una ventana de 4: tras reiniciar no cabe ni una más.

    Sin sembrar, el router llamaría al proveedor creyéndose con la ventana entera.
    El `base_url` apunta a un puerto local cerrado: si la siembra se pierde, la
    prueba falla por un rechazo de conexión y no sale de la máquina.
    """
    settings = load_settings(
        roles=role_map(SCARCE),
        openai={"api_key": "sk-test", "base_url": "http://127.0.0.1:9"},
        operations={
            "journal_path": str(tmp_path / "journal.jsonl"),
            "cache_dir": str(tmp_path / "cache"),
        },
    )
    spend(tmp_path / "journal.jsonl", 2)

    router = build_router(settings, seed_from=journal_calls(settings))

    with pytest.raises(QuotaExhaustedError):
        await router.invoke(AgentRole.DECIDER, "una pregunta nueva", Proposal)


def test_run_seeds_the_router_from_the_journal_before_the_first_evaluation(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`crypto-agents run` le entrega al router lo registrado, sin que nadie lo pida.

    Es el cableado: `build_router` sabe sembrar, pero si el comando no le pasa las
    llamadas el reinicio sigue arrancando a cero. El doble corta ahí para que el
    bucle no llegue a empezar.
    """
    monkeypatch.setenv("CA_RUNNER__SYMBOLS", '["BTC/USDT"]')
    write_records(workspace / "journal.jsonl", count=3)
    received: list[LLMCall] = []

    def capture(settings: Settings, seed_from: Iterable[LLMCall] = ()) -> object:
        del settings
        received.extend(seed_from)
        raise ConfigError("la prueba corta aquí")

    monkeypatch.setattr(cli, "build_router", capture)

    assert main(["run"]) == 1
    assert len(received) == 3


def test_run_refuses_to_start_on_a_journal_it_cannot_read(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Sin poder leer lo gastado no se arranca dándolo por cero, y se dice qué línea."""
    monkeypatch.setenv("CA_RUNNER__SYMBOLS", '["BTC/USDT"]')
    write_records(workspace / "journal.jsonl", count=1)
    with (workspace / "journal.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{a medias\n")

    assert main(["run"]) == 1
    assert "línea 2" in capsys.readouterr().err
