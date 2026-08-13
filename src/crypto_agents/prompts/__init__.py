"""Carga y renderizado de prompts.

Los prompts viven en archivos `.md` junto a este módulo, no incrustados en el
código: así se versionan, se leen en un diff y se pueden revisar sin abrir la
lógica del grafo. Un cambio en un `.md` cambia el prompt renderizado y por tanto
su digest, lo que invalida la caché de respuestas por sí solo.

Se usa `string.Template` (`$variable`) y no `str.format`: los prompts describen
JSON y las llaves chocarían con la sintaxis de formato.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from crypto_agents.state import (
        DebateBrief,
        Dimension,
        IndicatorSet,
        MarketSnapshot,
        Side,
        TechnicalVerdict,
    )

__all__ = [
    "PROMPT_DIR",
    "PromptError",
    "debate_prompt",
    "decision_prompt",
    "format_brief",
    "format_indicators",
    "format_verdicts",
    "load_template",
    "render",
    "technical_prompt",
]

PROMPT_DIR = Path(__file__).parent


class PromptError(RuntimeError):
    """Falta una plantilla o una variable que la plantilla exige."""


@cache
def load_template(name: str) -> Template:
    """Carga `<name>.md` del directorio de prompts."""
    path = PROMPT_DIR / f"{name}.md"
    if not path.is_file():
        raise PromptError(f"no existe la plantilla {name!r} en {PROMPT_DIR}")
    return Template(path.read_text(encoding="utf-8"))


def render(name: str, /, **values: object) -> str:
    """Renderiza una plantilla. Una variable sin valor es un error, no un hueco."""
    template = load_template(name)
    try:
        return template.substitute({key: str(value) for key, value in values.items()})
    except KeyError as error:
        raise PromptError(f"falta la variable {error.args[0]!r} para la plantilla {name!r}") from (
            error
        )


# ───────────────────────────────────────── Formateo de evidencia ──────────────────────────────────


def format_indicators(indicators: IndicatorSet) -> str:
    """Lista de indicadores en orden estable, tal como el agente debe citarlos."""
    return "\n".join(
        f"- `{name}`: {indicators.values[name]:.4f}" for name in sorted(indicators.values)
    )


def format_triggers(triggers: Sequence[str]) -> str:
    """Disparadores del gate, o una nota si la evaluación se forzó."""
    if not triggers:
        return "- (evaluación forzada, sin disparadores)"
    return "\n".join(f"- `{trigger}`" for trigger in triggers)


def format_verdicts(verdicts: Sequence[TechnicalVerdict]) -> str:
    """Veredictos con sus observaciones, con los ids visibles para poder citarlos."""
    blocks: list[str] = []
    for verdict in sorted(verdicts, key=lambda item: item.dimension.value):
        lines = [
            f"### {verdict.dimension.value}",
            f"- sesgo: **{verdict.bias.value}** (confianza {verdict.confidence:.2f})",
            f"- invalidación: {verdict.invalidation}",
            "- observaciones:",
        ]
        lines.extend(
            f"  - `{observation.id}` [{observation.supports.value}] {observation.text} "
            f"(cita: {', '.join(observation.cites)})"
            for observation in verdict.observations
        )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def format_brief(brief: DebateBrief | None) -> str:
    """Alegato de una mesa, o una nota si no llegó a producirse."""
    if brief is None:
        return "(esta mesa no produjo alegato)"
    lines = [
        f"- tesis: {brief.thesis}",
        f"- convicción: {brief.conviction:.2f}",
        f"- contraargumento más fuerte: {brief.strongest_counterargument}",
        "- afirmaciones:",
    ]
    lines.extend(
        f"  - [{claim.strength.value}] {claim.text} (apoyo: {', '.join(claim.grounded_in)})"
        for claim in brief.claims
    )
    return "\n".join(lines)


# ──────────────────────────────────────── Prompts por agente ──────────────────────────────────────


def technical_prompt(
    dimension: Dimension,
    snapshot: MarketSnapshot,
    indicators: IndicatorSet,
    triggers: Sequence[str],
) -> str:
    """Prompt de un agente técnico. Recibe indicadores, nunca velas."""
    return render(
        "technical",
        dimension=dimension.value,
        exchange=snapshot.exchange,
        symbol=snapshot.symbol,
        timeframe=snapshot.timeframe,
        timestamp=snapshot.timestamp.isoformat(),
        close=f"{snapshot.close:.8g}",
        triggers=format_triggers(triggers),
        indicators=format_indicators(indicators),
    )


def debate_prompt(
    side: Side,
    snapshot: MarketSnapshot,
    indicators: IndicatorSet,
    verdicts: Sequence[TechnicalVerdict],
) -> str:
    """Prompt de una mesa. Ambas mesas reciben exactamente la misma evidencia."""
    return render(
        "debate",
        side=side.value,
        symbol=snapshot.symbol,
        timeframe=snapshot.timeframe,
        close=f"{snapshot.close:.8g}",
        indicators=format_indicators(indicators),
        verdicts=format_verdicts(verdicts),
    )


def decision_prompt(
    snapshot: MarketSnapshot,
    indicators: IndicatorSet,
    verdicts: Sequence[TechnicalVerdict],
    bull: DebateBrief | None,
    bear: DebateBrief | None,
) -> str:
    """Prompt del decisor: evidencia común más los dos alegatos."""
    return render(
        "decider",
        symbol=snapshot.symbol,
        timeframe=snapshot.timeframe,
        close=f"{snapshot.close:.8g}",
        indicators=format_indicators(indicators),
        verdicts=format_verdicts(verdicts),
        bull=format_brief(bull),
        bear=format_brief(bear),
    )
