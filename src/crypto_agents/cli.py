"""Punto de entrada operativo: `crypto-agents`.

Seis subcomandos, y ninguno de ellos decide nada: vigilar y detener es todo lo que
hace falta poder hacer desde fuera mientras el sistema corre.

    status    configuración resuelta y si la operación está detenida
    stop      pone el centinela: no sale ninguna orden más
    resume    lo quita
    alerts    avisos sobre lo registrado en el journal
    query     consulta el journal por símbolo, acción, backend o causa de aborto
    run       arranca el bucle sobre los símbolos configurados

`stop` y `resume` no hablan con el proceso que está corriendo: escriben y borran
un archivo. Eso es deliberado — funcionan aunque el proceso esté colgado, aunque
no haya proceso todavía, y sobreviven a un reinicio.

Se usa argparse y no una biblioteca de CLI: son seis subcomandos sin adornos y no
justifican una dependencia más.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import TYPE_CHECKING

from crypto_agents.alerts import AlertThresholds, evaluate_alerts
from crypto_agents.bootstrap import (
    NOTIONAL_ACCOUNT,
    build_executor,
    build_kill_switch,
    build_router,
    open_journal,
)
from crypto_agents.context import AgentContext, utc_now
from crypto_agents.execution import ExecutionMode
from crypto_agents.graph import build_graph
from crypto_agents.journal import JsonlJournal
from crypto_agents.market import CcxtMarketClient, MarketDataError
from crypto_agents.metrics import summarise
from crypto_agents.queries import abort_cause, by_abort_cause, filter_records
from crypto_agents.runner import Runner
from crypto_agents.settings import ConfigError, Settings, load_settings
from crypto_agents.state import Action, Backend

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from uuid import UUID

    from crypto_agents.journal import EvaluationRecord
    from crypto_agents.llm import ModelRouter

__all__ = ["main"]


def _load() -> Settings:
    """Configuración, o `ConfigError` nombrando lo que falta."""
    return load_settings()


def _records(settings: Settings) -> list[EvaluationRecord]:
    """Todo lo registrado hasta ahora."""
    return JsonlJournal(settings.operations.journal_path).read_all()


# ─────────────────────────────────────────── Subcomandos ──────────────────────────────────────────


def cmd_status(settings: Settings, args: argparse.Namespace) -> int:
    """Estado operativo: parada, modo de ejecución y recuento del journal."""
    del args
    switch = build_kill_switch(settings)
    sentinel = settings.operations.kill_switch_file
    records = _records(settings)

    print(f"parada        : {'SÍ' if switch.engaged() else 'no'}")
    print(f"  centinela   : {sentinel} ({'existe' if sentinel.exists() else 'ausente'})")
    print(f"  por config  : {'sí' if settings.risk.kill_switch else 'no'}")
    print(f"ejecución     : {settings.execution.mode.value}")
    print(f"exchange      : {settings.exchange.exchange_id} (sandbox={settings.exchange.sandbox})")
    print(f"journal       : {settings.operations.journal_path} ({len(records)} evaluaciones)")
    if records:
        summary = summarise(records)
        print(f"  decididas   : {summary.decided}")
        print(f"  operadas    : {summary.traded}")
        print(f"  cuota usada : {summary.quota_used:.1f}")
    return 0


def cmd_stop(settings: Settings, args: argparse.Namespace) -> int:
    """Pone el centinela. A partir de aquí el gate de riesgo veta toda orden."""
    del args
    path = settings.operations.kill_switch_file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"detenido a las {utc_now().isoformat()}\n", encoding="utf-8")
    print(f"parada activada: {path}")
    print("ninguna orden saldrá al mercado hasta que se ejecute `crypto-agents resume`")
    return 0


def cmd_resume(settings: Settings, args: argparse.Namespace) -> int:
    """Quita el centinela. No reanuda nada por sí solo: deja de vetar."""
    del args
    path = settings.operations.kill_switch_file
    if not path.exists():
        print(f"no había parada activa ({path})")
    else:
        path.unlink()
        print(f"parada retirada: {path}")

    if settings.risk.kill_switch:
        print("aviso: CA_RISK__KILL_SWITCH sigue activo por configuración", file=sys.stderr)
        return 1
    return 0


def cmd_alerts(settings: Settings, args: argparse.Namespace) -> int:
    """Avisos sobre lo registrado. Sale con 1 si hay alguno, para poder encadenarlo."""
    records = _records(settings)
    thresholds = AlertThresholds(
        quota_remaining_fraction=args.quota_fraction,
        veto_window=args.veto_window,
        veto_repeats=args.veto_repeats,
        validation_failure_rate=args.failure_rate,
    )
    alerts = evaluate_alerts(records, settings, utc_now(), thresholds)
    if not alerts:
        print(f"sin alertas sobre {len(records)} evaluaciones")
        return 0

    for alert in alerts:
        print(f"[{alert.kind.value}] {alert.subject}: {alert.detail}")
    return 1


def cmd_query(settings: Settings, args: argparse.Namespace) -> int:
    """Consulta el journal con los filtros pedidos."""
    records = _records(settings)
    selected = filter_records(
        records,
        symbol=args.symbol,
        action=Action(args.action) if args.action else None,
        backend=Backend(args.backend) if args.backend else None,
        cause=args.cause,
    )

    if args.group_by_cause:
        for cause, group in by_abort_cause(selected).items():
            print(f"{cause:24s} {len(group)}")
        return 0

    print(f"{len(selected)} de {len(records)} evaluaciones")
    for record in selected[-args.limit :]:
        proposed = record.proposed
        action = proposed.action.value if proposed is not None else "—"
        print(
            f"{record.at.isoformat()}  {record.symbol:12s} {action:5s} "
            f"{'ORDEN' if record.traded else '     '}  {abort_cause(record)}"
        )
    return 0


async def _run(settings: Settings) -> int:
    """Bucle de operación sobre los símbolos configurados."""
    config = settings.runner
    if config is None:
        raise ConfigError("falta CA_RUNNER__SYMBOLS: el bucle no sabe qué evaluar")

    router = build_router(settings)
    journal = open_journal(settings)
    kill_switch = build_kill_switch(settings)
    account = settings.account if settings.account is not None else NOTIONAL_ACCOUNT
    market = CcxtMarketClient(settings.exchange)
    executor = build_executor(settings)

    def build_context(symbol: str, run_id: UUID, shared: ModelRouter) -> AgentContext:
        """Un contexto por símbolo, todos sobre el mismo router y el mismo journal."""
        return AgentContext(
            settings=settings,
            router=shared,
            market=market,
            account=account,
            run_id=run_id,
            symbol=symbol,
            timeframe=config.timeframe,
            executor=executor,
            journal=journal,
            kill_switch=kill_switch,
        )

    runner = Runner(
        config=config,
        router=router,
        build_context=build_context,
        graph=build_graph(),
        journal=journal,
        clock=utc_now,
    )
    print(f"bucle iniciado: {', '.join(config.symbols)} en {config.timeframe}", file=sys.stderr)
    print("para detener órdenes: crypto-agents stop", file=sys.stderr)
    try:
        await runner.run_forever()
    except KeyboardInterrupt:
        runner.stop()
        print("apagando; el ciclo en curso termina antes de salir", file=sys.stderr)
    finally:
        await market.close()
    return 0


def cmd_run(settings: Settings, args: argparse.Namespace) -> int:
    """Arranca el bucle."""
    del args
    if settings.execution.mode is ExecutionMode.LIVE:
        print("modo LIVE: las órdenes llegan al exchange", file=sys.stderr)
    return asyncio.run(_run(settings))


# ────────────────────────────────────────────── Entrada ───────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    """Los seis subcomandos."""
    parser = argparse.ArgumentParser(prog="crypto-agents", description=__doc__.split("\n")[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status", help="estado operativo").set_defaults(handler=cmd_status)
    subparsers.add_parser("stop", help="detiene toda orden").set_defaults(handler=cmd_stop)
    subparsers.add_parser("resume", help="retira la parada").set_defaults(handler=cmd_resume)
    subparsers.add_parser("run", help="arranca el bucle").set_defaults(handler=cmd_run)

    alerts = subparsers.add_parser("alerts", help="avisos sobre el journal")
    alerts.add_argument("--quota-fraction", type=float, default=0.2)
    alerts.add_argument("--veto-window", type=int, default=10)
    alerts.add_argument("--veto-repeats", type=int, default=3)
    alerts.add_argument("--failure-rate", type=float, default=0.2)
    alerts.set_defaults(handler=cmd_alerts)

    query = subparsers.add_parser("query", help="consulta el journal")
    query.add_argument("--symbol")
    query.add_argument("--action", choices=[item.value for item in Action])
    query.add_argument("--backend", choices=[item.value for item in Backend])
    query.add_argument("--cause", help="nodo que abortó la evaluación, o `completed`")
    query.add_argument("--group-by-cause", action="store_true")
    query.add_argument("--limit", type=int, default=20)
    query.set_defaults(handler=cmd_query)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada del ejecutable `crypto-agents`."""
    args = _build_parser().parse_args(argv)
    try:
        settings = _load()
        handler: Callable[[Settings, argparse.Namespace], int] = args.handler
        return handler(settings, args)
    except (ConfigError, MarketDataError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
