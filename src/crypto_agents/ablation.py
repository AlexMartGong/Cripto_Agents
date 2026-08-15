"""Ablación: ¿los seis modelos deciden mejor que uno?

Corre la misma ventana histórica bajo varias formas del pipeline y compara. Todo
lo que cambia entre brazos es la forma del grafo y el reparto rol → modelo; el
histórico, el gate de activación, el gate de riesgo y la construcción de la orden
son idénticos. Si cada brazo tuviera su propio camino hasta el mercado, la tabla
mediría el arnés.

Qué mide cada brazo, y contra qué pregunta:

- `full` es el pipeline que opera. Es la referencia, no la respuesta.
- `no_debate` quita las dos mesas: ¿aporta algo el debate, o el decisor ya tenía
  todo lo que necesitaba en los veredictos técnicos?
- `bull_only` quita la contraparte: ¿aporta la mesa bajista, o solo ruido?
- `solo` es un modelo y una llamada. Es el brazo incómodo: si iguala a `full`, la
  arquitectura de seis modelos es cara y bonita, no buena.
- `local_technicals` y `local_bull` mueven roles concretos al respaldo local: ¿en
  qué roles se puede gastar menos sin perder decisión?

La coincidencia se mide sobre la acción, no sobre la confianza: dos decisores que
compran con 0.7 y 0.6 de confianza tomaron la misma decisión, y tratar esa
diferencia como desacuerdo inventaría señal donde no la hay.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import Field

from crypto_agents.activation import ActivationConfig
from crypto_agents.cache import JsonFileResponseCache
from crypto_agents.context import AgentContext, utc_now
from crypto_agents.execution import PaperExecutor
from crypto_agents.graph import PipelineVariant, build_graph
from crypto_agents.indicators import DEFAULT_PRESET, IndicatorPreset
from crypto_agents.journal import InMemoryJournal
from crypto_agents.llm import build_backends
from crypto_agents.market import MarketDataError, read_ohlcv_csv
from crypto_agents.metrics import RunSummary, summarise
from crypto_agents.outcomes import OutcomeStats, score_outcomes
from crypto_agents.quota import QuotaLedger
from crypto_agents.replay import (
    HistoricalMarketClient,
    ReplayCacheMissError,
    ReplaySettings,
    replay,
    replay_router,
)
from crypto_agents.risk import AccountState
from crypto_agents.settings import DEFAULT_ENV_FILE, ConfigError, RoleConfig, load_settings
from crypto_agents.state import Action, AgentRole, Backend, FrozenModel

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime
    from uuid import UUID

    from crypto_agents.cache import ResponseCache
    from crypto_agents.journal import EvaluationRecord
    from crypto_agents.llm import ChatBackend, ModelRouter
    from crypto_agents.quota import Clock
    from crypto_agents.settings import ModelChoice, Settings

__all__ = [
    "ARMS",
    "AblationArm",
    "ArmResult",
    "agreement",
    "arm_settings",
    "build_arm_result",
    "decision_actions",
    "main",
    "render_report",
    "run_arm",
]


class AblationArm(FrozenModel):
    """Una configuración a comparar."""

    name: str = Field(min_length=1)
    variant: PipelineVariant
    question: str = Field(min_length=10)
    """Qué pregunta responde este brazo. Si no responde ninguna, sobra."""

    local_roles: frozenset[AgentRole] = frozenset()
    """Roles que se sirven desde el respaldo local en vez del primario remoto."""


ARMS: tuple[AblationArm, ...] = (
    AblationArm(
        name="full",
        variant=PipelineVariant.FULL,
        question="referencia: tres técnicos, dos mesas y decisor",
    ),
    AblationArm(
        name="no_debate",
        variant=PipelineVariant.NO_DEBATE,
        question="¿aporta algo el debate sobre los veredictos técnicos?",
    ),
    AblationArm(
        name="bull_only",
        variant=PipelineVariant.BULL_ONLY,
        question="¿aporta la contraparte bajista, o basta una mesa?",
    ),
    AblationArm(
        name="solo",
        variant=PipelineVariant.SOLO,
        question="¿un modelo y una llamada igualan a seis modelos y seis llamadas?",
    ),
    AblationArm(
        name="local_technicals",
        variant=PipelineVariant.FULL,
        question="¿se pueden servir las tres lecturas técnicas en local sin perder decisión?",
        local_roles=frozenset({AgentRole.STRUCTURE, AgentRole.MOMENTUM, AgentRole.VOLUME}),
    ),
    AblationArm(
        name="local_bull",
        variant=PipelineVariant.FULL,
        question="¿aguanta una mesa en local contra una remota?",
        local_roles=frozenset({AgentRole.BULL}),
    ),
)
"""Los seis brazos, en el orden en que se leen en el reporte."""


class ArmResult(FrozenModel):
    """Lo que produjo un brazo sobre el histórico."""

    arm: AblationArm
    summary: RunSummary
    outcomes: OutcomeStats
    actions: tuple[Action | None, ...]
    """Acción por evaluación, en orden. `None` donde el brazo no llegó a decidir."""

    wall_clock_seconds: float = Field(ge=0.0)
    models: Mapping[AgentRole, str] = Field(default_factory=dict)

    @property
    def mean_latency_ms(self) -> float | None:
        """Latencia media por llamada que llegó a un proveedor."""
        stats = self.summary.calls
        attempts = sum(item.attempts for item in stats.values())
        if attempts == 0:
            return None
        return self.wall_clock_seconds * 1000.0 / attempts


def decision_actions(records: Sequence[EvaluationRecord]) -> tuple[Action | None, ...]:
    """Acción de cada evaluación, en orden, con `None` donde no hubo decisión."""
    return tuple(
        record.proposed.action if record.proposed is not None else None for record in records
    )


def agreement(left: Sequence[Action | None], right: Sequence[Action | None]) -> float | None:
    """Fracción de evaluaciones en que dos brazos tomaron la misma acción.

    Se comparan por posición, que es la misma vela en ambos brazos porque los dos
    recorren el mismo histórico. Cuenta también los acuerdos en «no decidí»: que
    dos configuraciones se callen a la vez es una coincidencia real.
    """
    if len(left) != len(right) or not left:
        return None
    return sum(1 for a, b in zip(left, right, strict=True) if a == b) / len(left)


def arm_settings(settings: Settings, arm: AblationArm) -> Settings:
    """Configuración del brazo: los roles marcados se sirven desde su respaldo local.

    Se intercambia el primario por el respaldo en vez de tocar el router: así el
    brazo local recorre exactamente el mismo código que el remoto, y la diferencia
    medida es el modelo y no el camino.
    """
    if not arm.local_roles:
        return settings

    roles = dict(settings.roles)
    for role in sorted(arm.local_roles):
        config = roles[role]
        if config.fallback is None:
            raise ConfigError(
                f"el brazo {arm.name!r} pide {role.value} en local, "
                f"pero ese rol no declara respaldo"
            )
        roles[role] = RoleConfig(primary=config.fallback)
    return settings.model_copy(update={"roles": roles})


async def run_arm(
    arm: AblationArm,
    rows: Sequence[Sequence[float]],
    settings: Settings,
    config: ReplaySettings,
    account: AccountState,
    cache: ResponseCache,
    clock: Clock,
    fill_with: Mapping[Backend, ChatBackend] | None = None,
    horizon: int = 6,
    preset: IndicatorPreset = DEFAULT_PRESET,
    activation: ActivationConfig | None = None,
) -> ArmResult:
    """Corre un brazo completo sobre el histórico y lo consolida.

    El preset es un parámetro y no una constante porque decide cuántas velas de
    warm-up se comen antes de la primera evaluación: con el de producción son 400,
    y sobre un histórico de 500 quedan menos de cien evaluaciones. Todos los brazos
    reciben el mismo, que es lo que los hace comparables.
    """
    gate = activation if activation is not None else ActivationConfig(preset=preset)
    tuned = arm_settings(settings, arm)
    ledger = QuotaLedger(tuned, clock)
    router = replay_router(tuned, ledger, cache, clock, fill_with)
    graph = build_graph(arm.variant)
    journal = InMemoryJournal()

    def build_context(
        symbol: str,
        run_id: UUID,
        router_: ModelRouter,
        market: HistoricalMarketClient,
        moment: datetime,
    ) -> AgentContext:
        return AgentContext(
            settings=tuned,
            router=router_,
            market=market,
            account=account,
            run_id=run_id,
            symbol=symbol,
            timeframe=config.timeframe,
            executor=PaperExecutor(),
            journal=journal,
            candle_limit=config.candle_limit,
            preset=preset,
            activation=gate,
            clock=lambda: moment,
            now=moment,
        )

    started = time.perf_counter()
    records = await replay(rows, config, router, build_context, graph)
    elapsed = time.perf_counter() - started

    return build_arm_result(
        arm,
        records,
        rows,
        wall_clock_seconds=elapsed,
        horizon=horizon,
        models={role: tuned.role_config(role).primary for role in AgentRole},
    )


def build_arm_result(
    arm: AblationArm,
    records: Sequence[EvaluationRecord],
    rows: Sequence[Sequence[float]],
    wall_clock_seconds: float,
    horizon: int = 6,
    models: Mapping[AgentRole, ModelChoice] | None = None,
) -> ArmResult:
    """Consolida una corrida de un brazo."""
    return ArmResult(
        arm=arm,
        summary=summarise(records),
        outcomes=score_outcomes(records, rows, horizon),
        actions=decision_actions(records),
        wall_clock_seconds=wall_clock_seconds,
        models={role: choice.model for role, choice in (models or {}).items()},
    )


# ────────────────────────────────────────────── Reporte ───────────────────────────────────────────


def _pct(value: float | None) -> str:
    """Porcentaje, o un guion si no hay dato."""
    return "—" if value is None else f"{value:.0%}"


def _num(value: float | None, digits: int = 2) -> str:
    """Número, o un guion si no hay dato."""
    return "—" if value is None else f"{value:.{digits}f}"


def render_report(results: Sequence[ArmResult], reference: str = "full") -> str:
    """Tabla comparativa en Markdown.

    Deliberadamente sin veredicto automático: qué significa que `solo` iguale a
    `full` es una conclusión que hay que escribir a mano, y una plantilla que la
    generase sola invitaría a no leerla.
    """
    if not results:
        return "# Ablación\n\nSin corridas.\n"

    baseline = next((item for item in results if item.arm.name == reference), results[0])
    lines = [
        "| brazo | evals | decididas | acciones | órdenes | coincidencia con "
        f"`{baseline.arm.name}` | cuota | latencia media | fallo validación |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        summary = result.summary
        mix = ", ".join(
            f"{action.value} {count}" for action, count in sorted(summary.actions.items())
        )
        failures = [stats.failure_rate for stats in summary.calls.values() if stats.answered > 0]
        lines.append(
            f"| `{result.arm.name}` | {summary.evaluations} | {summary.decided} | "
            f"{mix or '—'} | {summary.traded} | "
            f"{_pct(agreement(baseline.actions, result.actions))} | "
            f"{_num(summary.quota_used, 1)} | "
            f"{_num(result.mean_latency_ms, 0)} ms | "
            f"{_pct(max(failures) if failures else None)} |"
        )

    lines.append("")
    lines.append("| brazo | órdenes resueltas | invalidadas | aciertos | retorno medio |")
    lines.append("| --- | --- | --- | --- | --- |")
    for result in results:
        outcomes = result.outcomes
        lines.append(
            f"| `{result.arm.name}` | {outcomes.resolved} | {outcomes.invalidated} | "
            f"{_pct(outcomes.win_rate)} | {_pct(outcomes.mean_return)} |"
        )

    lines.append("")
    lines.append("| brazo | qué pregunta responde |")
    lines.append("| --- | --- |")
    lines.extend(f"| `{result.arm.name}` | {result.arm.question} |" for result in results)
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────── Comando ──────────────────────────────────────────────


DEFAULT_HISTORY = Path("tests/data/btcusdt_4h.csv")
DEFAULT_REPORT = Path("docs/ablation.md")
NOTIONAL_ACCOUNT = AccountState(equity=10_000.0, day_start_equity=10_000.0)
"""Cuenta nocional del arnés.

Es una elección del experimento y no del sistema: todos los tamaños del pipeline
son fracciones, así que el nivel absoluto solo afecta a los vetos de drawdown, que
en una ablación sin equity viva nunca se disparan. Se declara para que nadie lea
los retornos como si vinieran de una cuenta real.
"""


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Argumentos del comando."""
    parser = argparse.ArgumentParser(
        prog="python -m crypto_agents.ablation",
        description="Corre la misma ventana histórica bajo varias formas del pipeline.",
    )
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--warmup", type=int, default=DEFAULT_PRESET.min_bars)
    parser.add_argument("--evaluations", type=int, default=25)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--cache", type=Path, default=Path("var/ablation-cache"))
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--arms",
        default=",".join(arm.name for arm in ARMS),
        help="brazos a correr, separados por comas",
    )
    parser.add_argument(
        "--fill",
        action="store_true",
        help="permite llamar a los proveedores para llenar la caché; sin esto, solo caché",
    )
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> str:
    """Corre los brazos pedidos y devuelve el reporte."""
    settings = load_settings(DEFAULT_ENV_FILE)
    rows = read_ohlcv_csv(args.history)
    config = ReplaySettings(
        symbol=args.symbol,
        timeframe=args.timeframe,
        warmup_bars=args.warmup,
        max_evaluations=args.evaluations,
    )
    cache = JsonFileResponseCache(args.cache)
    clock = utc_now
    fill_with = build_backends(settings) if args.fill else None

    wanted = [name.strip() for name in args.arms.split(",") if name.strip()]
    by_name = {arm.name: arm for arm in ARMS}
    unknown = [name for name in wanted if name not in by_name]
    if unknown:
        raise ConfigError(f"brazos desconocidos: {', '.join(unknown)}")

    results: list[ArmResult] = []
    for name in wanted:
        arm = by_name[name]
        print(f"· {arm.name}: {arm.question}", file=sys.stderr)
        results.append(
            await run_arm(
                arm,
                rows,
                settings,
                config,
                NOTIONAL_ACCOUNT,
                cache,
                clock,
                fill_with,
                args.horizon,
            )
        )
    return render_report(results)


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada: `python -m crypto_agents.ablation`."""
    args = _parse_args(argv)
    try:
        report = asyncio.run(_run(args))
    except (ConfigError, MarketDataError, ReplayCacheMissError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"tabla escrita en {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
