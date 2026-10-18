"""El análisis del estudio hacia adelante: la regla de veredicto, las negativas y el ensayo.

Lo que `docs/forward.md`, apartado 6, y el bloque del análisis piden, con su mutación donde la hay:

- **La regla de veredicto**, en cada borde. La mutación «empate antes que supera» es equivalente
  mientras el empate exija que el IC 95 % contenga el 0: para eso está la cláusula. Se mata el
  cambio de orden sin la cláusula, y una prueba deja escrito que el cambio de orden solo no cambia
  ningún veredicto.
- **La coherencia con la planificación**: el empate es exactamente la región `(a, b)` de
  `forward_plan.tie_region`, hasta el último ulp.
- **Las negativas y las causas de «no determinado»**, con un espía en la puntuación que no se
  puede llamar.
- **De punta a punta**: un estudio sintético con 50 bloques no vacíos y uno vacío, que llega al
  veredicto conocido de antemano.
- **El ensayo sobre la etapa 1**: las cuatro filas publicadas, cifra por cifra, sin veredicto.
"""

from __future__ import annotations

import inspect
import math
import statistics
import textwrap
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from typing import TYPE_CHECKING, NamedTuple

import pytest

import crypto_agents.forward_analysis as module
from crypto_agents import criteria
from crypto_agents.ablation import build_run_meta, open_run_directory
from crypto_agents.activation_sweep import DEFAULT_SYMBOLS, history_path
from crypto_agents.audit import PlanKind, arm_journal_path, file_sha256
from crypto_agents.design_effect import pool_run_id
from crypto_agents.dispersion import Z_INTERVAL
from crypto_agents.forward import BatchEnvironment, forward_arms, main, write_forward_meta
from crypto_agents.forward_data import (
    STAGE1_DIR,
    StudyPopulation,
    block_manifest,
    manifest_path,
    ohlcv_lines,
    write_block_manifest,
)
from crypto_agents.forward_plan import DELTA, FORWARD_ARMS, Z_EQUIVALENCE, tie_region
from crypto_agents.forward_record import (
    FIRST_BLOCK,
    REGISTERED_PATH,
    STEP,
    BatchMeta,
    BatchOutcome,
    batch_directory,
    block_dates,
)
from crypto_agents.funding import FUNDING_DIR
from crypto_agents.journal import EvaluationRecord, JsonlJournal
from crypto_agents.llm import InvalidModelOutputError
from crypto_agents.metrics import ClusteredError, ClusteredPair, PairedDifference
from crypto_agents.paired_dispersion import read_stage
from crypto_agents.selection import HISTORY_DIR, PlannedEvaluation
from crypto_agents.state import (
    Action,
    AgentRole,
    Backend,
    FailureKind,
    IndicatorSet,
    LLMCall,
    MarketSnapshot,
    NodeError,
    Proposal,
    StructuredOutputMode,
)
from tests.conftest import DATA_DIR
from tests.test_criteria import provider_lost
from tests.test_forward import registered, study_settings
from tests.test_net_outcomes import mutated

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from crypto_agents.paired_dispersion import StageRun

ROOT = DATA_DIR.parent.parent
Verdict = module.Verdict

# ───────────────────────────────────────── La regla de veredicto ──────────────────────────────────


def interval(mean: float, stderr: float) -> PairedDifference:
    """Un intervalo como lo arma `metrics.clustered_paired_difference`: `media ± 1.96 · EE`."""
    return PairedDifference(
        n=2449, mean=mean, stderr=stderr, low=mean - 1.96 * stderr, high=mean + 1.96 * stderr
    )


EDGE_ERROR = 0.0015
"""Con este EE el IC 95 % de las medias del borde del IC 90 % contiene el 0: el borde lo decide
el IC 90 % y no `supera`."""

ZERO_ERROR = 0.001
"""Con este EE una media de 1.96·EE deja el IC 95 % con su extremo inferior exactamente en 0, y el
IC 90 % dentro de ±δ."""


def assert_the_hand_cases(verdict: Callable[[PairedDifference | None, float], Verdict]) -> None:
    """Cada caso calculado a mano con δ = 0.40 %.

    - Sin intervalo: no determinado.
    - 0.21 % con EE 0.10 %: IC 95 % [0.014 %, 0.406 %], entero por encima de 0, e IC 90 %
      [0.0455 %, 0.3745 %], dentro de ±δ. Supera: la cláusula del IC 95 % es la que impide que
      sea empate.
    - -0.30 % con EE 0.10 %: IC 95 % entero por debajo de 0. Peor.
    - 0.01 % con EE 0.05 %: IC 90 % [-0.072 %, 0.092 %] e IC 95 % con el 0. Empate, y también con
      la regla de la ablación (semiancho 0.098 % ≤ δ).
    - 0.36 % (0.9δ) con semiancho 0.39 %: IC 95 % [-0.03 %, 0.75 %], con el 0 y semiancho menor que
      δ, así que la ablación diría empate. Pero el IC 90 % llega a 0.68 %, fuera de δ: no
      concluyente.
    - 0.05 % con EE 0.19 %: el IC 90 % llega a 0.3626 % < δ, empate; con 1.96 llegaría a 0.4224 %.
    - El extremo del IC 90 % exactamente en ±δ no es empate; una ulp hacia dentro, sí.
    - El IC 95 % con su extremo exactamente en 0 no es supera ni peor ni, con la cláusula
      estricta, empate: no concluyente.
    """
    assert verdict(None, DELTA) is Verdict.UNDETERMINED
    assert verdict(interval(0.0021, 0.001), DELTA) is Verdict.ABOVE
    assert verdict(interval(-0.003, 0.001), DELTA) is Verdict.BELOW

    both = interval(0.0001, 0.0005)
    assert verdict(both, DELTA) is Verdict.TIE
    assert criteria.classify(both, DELTA) is criteria.Verdict.TIE

    ablation_only = interval(0.0036, 0.0039 / 1.96)
    assert ablation_only.low < 0.0 < ablation_only.high
    assert criteria.classify(ablation_only, DELTA) is criteria.Verdict.TIE
    assert verdict(ablation_only, DELTA) is Verdict.INCONCLUSIVE

    assert verdict(interval(0.0005, 0.0019), DELTA) is Verdict.TIE

    margin = Z_EQUIVALENCE * EDGE_ERROR
    for edge in (DELTA - margin, -DELTA + margin):
        assert interval(edge, EDGE_ERROR).low < 0.0 < interval(edge, EDGE_ERROR).high
        assert verdict(interval(edge, EDGE_ERROR), DELTA) is Verdict.INCONCLUSIVE
        assert verdict(interval(math.nextafter(edge, 0.0), EDGE_ERROR), DELTA) is Verdict.TIE

    for sign in (1.0, -1.0):
        touching = interval(sign * 1.96 * ZERO_ERROR, ZERO_ERROR)
        assert 0.0 in (touching.low, touching.high)
        assert verdict(touching, DELTA) is Verdict.INCONCLUSIVE


def test_the_hand_cases_hold_for_the_real_rule() -> None:
    """El control de las mutaciones: sin error puesto, la comprobación pasa."""
    assert_the_hand_cases(module.forward_verdict)


RULE = module.forward_verdict
CHECKS_FIRST = """    if interval.low > 0.0:
        return Verdict.ABOVE
    if interval.high < 0.0:
        return Verdict.BELOW
    margin = Z_EQUIVALENCE * interval.stderr
    inside = -delta + margin < interval.mean < delta - margin
    if inside and interval.low < 0.0 < interval.high:
        return Verdict.TIE
"""
TIE_FIRST = """    margin = Z_EQUIVALENCE * interval.stderr
    inside = -delta + margin < interval.mean < delta - margin
    if inside and interval.low < 0.0 < interval.high:
        return Verdict.TIE
    if interval.low > 0.0:
        return Verdict.ABOVE
    if interval.high < 0.0:
        return Verdict.BELOW
"""


@pytest.mark.parametrize(
    ("old", "new"),
    [
        pytest.param(
            "margin = Z_EQUIVALENCE * interval.stderr",
            "margin = 1.96 * interval.stderr",
            id="1.96 en lugar de 1.645 en el empate",
        ),
        pytest.param(
            "inside = -delta + margin < interval.mean < delta - margin",
            "inside = -delta + margin <= interval.mean <= delta - margin",
            id="bordes del IC 90 no estrictos",
        ),
        pytest.param("if interval.low > 0.0:", "if interval.low >= 0.0:", id="supera en el 0"),
        pytest.param(
            "if inside and interval.low < 0.0 < interval.high:",
            "if inside:",
            id="sin la cláusula del IC 95",
        ),
        pytest.param(
            CHECKS_FIRST,
            TIE_FIRST.replace("if inside and interval.low < 0.0 < interval.high:", "if inside:"),
            id="empate antes que supera, sin la cláusula",
        ),
    ],
)
def test_mutation_of_the_rule_is_caught(old: str, new: str) -> None:
    mutant = mutated(RULE, old, new, module)
    with pytest.raises(AssertionError):
        assert_the_hand_cases(mutant)


ERRORS = (0.0005, 0.001, 0.0011, 0.0015, 0.0019, 0.0025, 0.003)
"""EE de la prueba de coherencia. Con 0.0011, el límite de la región es 1.96·EE; con 0.0015, el
de δ - 1.645·EE; con 0.0025 y 0.003, 1.645·EE > δ y no hay empate posible."""


def means(error: float) -> list[float]:
    """Medias que pisan cada borde de la región, sus vecinas a una ulp y una rejilla regular."""
    points = {0.0, DELTA, -DELTA, Z_INTERVAL * error, -Z_INTERVAL * error}
    region = tie_region(error, DELTA)
    if region is not None:
        points |= set(region)
    points |= {math.nextafter(point, side) for point in set(points) for side in (-1.0, 1.0)}
    points |= {step * error / 7 for step in range(-40, 41)}
    return sorted(points)


def assert_coherent(verdict: Callable[[PairedDifference | None, float], Verdict]) -> None:
    """El veredicto es empate si y solo si la media cae en `(a, b)` de `tie_region`."""
    assert any(tie_region(error, DELTA) is None for error in ERRORS)
    assert any(tie_region(error, DELTA) is not None for error in ERRORS)
    for error in ERRORS:
        region = tie_region(error, DELTA)
        for mean in means(error):
            expected = region is not None and region[0] < mean < region[1]
            assert (verdict(interval(mean, error), DELTA) is Verdict.TIE) == expected, (
                error,
                mean,
            )


def test_the_tie_is_exactly_the_planned_region() -> None:
    assert_coherent(module.forward_verdict)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("margin = Z_EQUIVALENCE * interval.stderr", "margin = 1.96 * interval.stderr"),
        (
            "inside = -delta + margin < interval.mean < delta - margin",
            "inside = -delta + margin <= interval.mean <= delta - margin",
        ),
        (
            "if inside and interval.low < 0.0 < interval.high:",
            "if inside and interval.low <= 0.0 <= interval.high:",
        ),
    ],
    ids=["1.96 en el empate", "bordes del IC 90 no estrictos", "IC 95 cerrado"],
)
def test_mutation_a_tie_off_the_planned_region_is_caught(old: str, new: str) -> None:
    mutant = mutated(RULE, old, new, module)
    with pytest.raises(AssertionError):
        assert_coherent(mutant)


def test_the_order_alone_is_an_equivalent_mutant() -> None:
    """Con la cláusula del IC 95 %, supera y empate son disjuntos: evaluar el empate primero no
    cambia ningún veredicto. Por eso la mutación del orden que se mata es la que además la quita."""
    swapped = mutated(RULE, CHECKS_FIRST, TIE_FIRST, module)
    for error in ERRORS:
        for mean in means(error):
            assert swapped(interval(mean, error), DELTA) is RULE(interval(mean, error), DELTA)


def clustered(mean: float, stderr: float) -> ClusteredPair:
    found = interval(mean, stderr)
    error = ClusteredError(n=found.n, clusters=50, naive=stderr, clustered=stderr)
    return ClusteredPair(naive=found, error=error, interval=found)


def assert_the_primary_uses_the_study_delta() -> None:
    """0.05 % con EE 0.19 % es empate con δ = 0.40 %. Con el δ de la ablación, 0.20 %, el IC 90 %
    [-0.263 %, +0.363 %] no cabe en ±δ: no concluyente."""
    assert module.primary_verdict(clustered(0.0005, 0.0019)) is Verdict.TIE
    assert module.primary_verdict(None) is Verdict.UNDETERMINED


def test_the_primary_verdict_reads_the_study_delta() -> None:
    assert_the_primary_uses_the_study_delta()


def test_mutation_the_delta_of_the_ablation_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "DELTA", criteria.DELTA)
    with pytest.raises(AssertionError):
        assert_the_primary_uses_the_study_delta()


def test_the_rule_rejects_a_delta_that_is_not_positive() -> None:
    with pytest.raises(ValueError, match="positivo"):
        module.forward_verdict(interval(0.0, 0.001), 0.0)


def test_the_90_interval_printed_is_the_mean_plus_minus_1645() -> None:
    low, high = module.interval_90(interval(0.001, 0.002))
    assert low == pytest.approx(0.001 - 1.645 * 0.002)
    assert high == pytest.approx(0.001 + 1.645 * 0.002)


def assert_the_missing_limit() -> None:
    """5 de 100 es el 5.00 %: no supera el límite. 501 de 10 000 es el 5.01 %: sí."""
    assert not module.missing_exceeds(5, 100)
    assert module.missing_exceeds(501, 10_000)
    assert not module.missing_exceeds(0, 0)


def test_the_missing_limit_is_exact() -> None:
    assert Fraction(5, 100) == module.MISSING_LIMIT
    assert_the_missing_limit()


def test_mutation_a_limit_that_counts_its_edge_is_caught() -> None:
    mutant = mutated(
        module.missing_exceeds,
        "Fraction(missing, evaluations) > MISSING_LIMIT",
        "Fraction(missing, evaluations) >= MISSING_LIMIT",
        module,
    )
    with pytest.raises(AssertionError):
        assert not mutant(5, 100)


# ──────────────────────────────────────── Un estudio sintético ────────────────────────────────────

PERIOD = timedelta(hours=4)
OPENING = FIRST_BLOCK - timedelta(days=1)
"""La primera vela del histórico sintético. Corto: no hace falta el warm-up, porque la población
la da la prueba."""

SYMBOL = "BTC/USDT"
RETURNS = (0.012, 0.008)
"""Retorno de una compra en las activaciones pares e impares: 1.2 % y 0.8 %."""

UPSTREAM = ("accounts/fireworks/models/glm-5p3", "fireworks")
LATE = FIRST_BLOCK + 60 * STEP
"""Un instante en que ya venció el plazo de todos los lotes del estudio sintético."""

EARLY_BLOCK = 51
"""El último bloque de los estudios de 51: con `EARLY` su plazo todavía no venció."""


def early(blocks: int = EARLY_BLOCK) -> datetime:
    return block_dates(FIRST_BLOCK + (blocks - 1) * STEP).ready


def entry_of(position: int) -> PlannedEvaluation:
    """La activación del bloque `position` (desde 0): una vela un día después de su inicio."""
    opens = FIRST_BLOCK + position * STEP + timedelta(days=1)
    return PlannedEvaluation(
        symbol=SYMBOL,
        timeframe="4h",
        index=(opens - OPENING) // PERIOD,
        at=opens,
        candles_digest="0" * 64,
        triggers=("range_breakout_up",),
    )


def history(blocks: int, exits: Mapping[int, float]) -> list[list[float]]:
    """Velas de 4h hasta el cierre del último bloque más el horizonte. Rangos en [99, 101]: el stop
    común (100 ∓ 2·2.0) no se toca y cada posición sale al cierre del horizonte."""
    ready = block_dates(FIRST_BLOCK + (blocks - 1) * STEP).ready
    count = (ready - OPENING) // PERIOD
    return [
        [
            (OPENING + index * PERIOD).timestamp() * 1000,
            100.0,
            101.0,
            99.0,
            exits.get(index, 100.0),
            1.0,
        ]
        for index in range(count)
    ]


def decider_call(at: datetime, **changes: object) -> LLMCall:
    model, endpoint = UPSTREAM
    call = LLMCall(
        role=AgentRole.DECIDER,
        backend=Backend.OPENAI,
        model="glm-5.2",
        structured_output=StructuredOutputMode.FUNCTION_CALLING,
        quota_weight=1.0,
        prompt_digest="b" * 64,
        cache_hit=False,
        valid=True,
        latency_ms=10.0,
        at=at,
        upstream_model=model,
        upstream_endpoint=endpoint,
    )
    return call.model_copy(update=changes)


def record(
    entry: PlannedEvaluation,
    action: Action | None,
    calls: Sequence[LLMCall] = (),
    errors: Sequence[NodeError] = (),
) -> EvaluationRecord:
    """Lo que un brazo deja de una evaluación, con el `run_id` y el instante del replay."""
    at = entry.at + PERIOD
    run_id = pool_run_id(entry.symbol, entry.timeframe, at)
    proposal = None
    if action is not None:
        proposal = Proposal(
            action=action,
            confidence=0.5,
            size_fraction=0.0 if action is Action.HOLD else 0.05,
            invalidation_price={Action.BUY: 96.0, Action.SELL: 104.0, Action.HOLD: None}[action],
            rationale="Decisión sintética para el análisis.",
        )
    return EvaluationRecord(
        run_id=run_id,
        at=at,
        symbol=entry.symbol,
        timeframe=entry.timeframe,
        snapshot=MarketSnapshot(
            run_id=run_id,
            exchange="binance",
            symbol=entry.symbol,
            timeframe=entry.timeframe,
            timestamp=entry.at,
            close=100.0,
            candles_digest="a" * 64,
            candles_count=500,
        ),
        indicators=IndicatorSet(values={"ATRr_14": 2.0}),
        proposal=proposal,
        calls=tuple(calls),
        errors=tuple(errors),
    )


def decided(entry: PlannedEvaluation) -> dict[str, list[EvaluationRecord]]:
    """Los cinco brazos de una evaluación completa: `solo` compra, `rule_trend` mantiene."""
    at = entry.at + PERIOD
    return {
        "solo": [record(entry, Action.BUY, (decider_call(at),))],
        "always_buy": [record(entry, Action.BUY)],
        "always_sell": [record(entry, Action.SELL)],
        "random_uniform": [record(entry, Action.HOLD)],
        "rule_trend": [record(entry, Action.HOLD)],
    }


def missing(entry: PlannedEvaluation) -> dict[str, list[EvaluationRecord]]:
    """Una faltante: el proveedor se llevó la evaluación de `solo` (un 404 real de Zen)."""
    arms = decided(entry)
    lost = provider_lost("decide", FailureKind.TRANSPORT)
    arms["solo"] = [record(entry, None, lost.calls, lost.errors)]
    return arms


def invalid(entry: PlannedEvaluation) -> dict[str, list[EvaluationRecord]]:
    """Una respuesta inválida tras los dos intentos: es del modelo, vale 0 y no es faltante."""
    at = entry.at + PERIOD
    message = str(InvalidModelOutputError(AgentRole.DECIDER, 2, "action: Field required"))
    failed = decider_call(
        at, valid=False, failure_kind=FailureKind.SCHEMA, failure_message="action: Field required"
    )
    arms = decided(entry)
    arms["solo"] = [
        record(
            entry, None, (failed, failed), (NodeError(node="decide_solo", message=message, at=at),)
        )
    ]
    return arms


def foreign_cache(entry: PlannedEvaluation) -> dict[str, list[EvaluationRecord]]:
    """Un acierto de caché con otro backend: invalida el lote (apartado 5)."""
    at = entry.at + PERIOD
    arms = decided(entry)
    hit = decider_call(at, cache_hit=True, backend=Backend.OLLAMA, latency_ms=0.0)
    arms["solo"] = [record(entry, Action.BUY, (hit,))]
    return arms


type Arms = Callable[[PlannedEvaluation], dict[str, list[EvaluationRecord]]]


class Study(NamedTuple):
    root: Path
    entries: dict[datetime, tuple[PlannedEvaluation, ...]]
    differences: list[float]
    """`solo` - `rule_trend` por evaluación, a mano, en el orden de los bloques."""


def population_of(
    entries: Mapping[datetime, tuple[PlannedEvaluation, ...]],
) -> Callable[..., StudyPopulation]:
    """La población que da la prueba: la que el estudio sintético usó para sus manifiestos."""

    def population(histories: Mapping[str, object], starts: Sequence[datetime]) -> StudyPopulation:
        return StudyPopulation(
            blocks={start: entries.get(start, ()) for start in starts},
            activations=dict.fromkeys(histories, len(entries)),
        )

    return population


def write_pass(
    root: Path,
    entry_start: datetime,
    manifest_file: Path,
    arms: Mapping[str, Sequence[EvaluationRecord]],
    stamp: str = "20261030T000000Z",
) -> Path:
    """Una pasada que corrió: su `meta.json`, su `forward.json` con lo registrado y sus journals."""
    directory = batch_directory(root / "batches", entry_start) / stamp
    dates = block_dates(entry_start)
    sha = file_sha256(manifest_file)
    count = len(arms["solo"])
    open_run_directory(
        directory,
        build_run_meta(
            study_settings(),
            PlanKind.MANIFEST,
            manifest_file,
            sha,
            ("batch",),
            True,
            forward_arms(),
            dates.ready,
            None,
            6,
            3 * count * 0.01509,
        ),
    )
    write_forward_meta(
        directory,
        BatchMeta(
            block=dates,
            outcome=BatchOutcome.RAN,
            invoked_at=dates.ready,
            hostname="prueba",
            git_commit=None,
            git_dirty=None,
            uv_lock_sha256=None,
            python_version="3.12",
            measured_path=REGISTERED_PATH,
            decider=registered(),
            evaluations=count,
            manifest_path=str(manifest_file),
            manifest_sha256=sha,
            spend_cap_usd=3 * count * 0.01509,
            spent_before_usd=0.0,
        ),
    )
    for arm, records in arms.items():
        journal = JsonlJournal(arm_journal_path(directory, arm))
        for item in records:
            journal.write(item)
    return directory


def build_study(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    blocks: int = 51,
    empty: Sequence[int] = (10,),
    special: Mapping[int, Arms] | None = None,
    skipped: Sequence[int] = (),
) -> Study:
    """Un estudio de `blocks` bloques desde el bloque 1, con una activación de BTC/USDT en cada
    bloque salvo los de `empty` (posiciones desde 0). Cada lote corre en una sola pasada; los de
    `skipped` no corren. `special` cambia lo que dejan los brazos en un bloque."""
    special = special or {}
    entries: dict[datetime, tuple[PlannedEvaluation, ...]] = {}
    exits: dict[int, float] = {}
    differences: list[float] = []
    filled = 0
    for position in range(blocks):
        start = FIRST_BLOCK + position * STEP
        if position in empty:
            entries[start] = ()
            continue
        entry = entry_of(position)
        entries[start] = (entry,)
        gain = RETURNS[filled % 2]
        exits[entry.index + 6] = 100.0 * (1.0 + gain)
        made = special.get(position, decided)(entry)
        differences.append(gain if made["solo"][0].proposed is not None else 0.0)
        filled += 1

    rows = history(blocks, exits)
    for symbol in DEFAULT_SYMBOLS:
        path = history_path(symbol, "4h", root / "history")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(ohlcv_lines(rows))
    histories = {symbol: rows for symbol in DEFAULT_SYMBOLS}
    population = population_of(entries)(histories, list(entries))

    for position in range(blocks):
        start = FIRST_BLOCK + position * STEP
        if not entries[start] or position in skipped:
            continue
        manifest = block_manifest(histories, population, start, block_dates(start).ready)
        assert manifest is not None
        file = manifest_path(start, root / "manifests")
        write_block_manifest(manifest, file)
        write_pass(root, start, file, special.get(position, decided)(entries[start][0]))

    monkeypatch.setattr(module, "study_population", population_of(entries))
    return Study(root, entries, differences)


class Spy:
    """La puntuación que no se puede llamar: lo que pase por aquí antes de una negativa es un
    retorno calculado que el análisis no tenía permiso de calcular."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[str] = []
        for name in ("paired_returns", "score_run"):
            monkeypatch.setattr(module, name, self.forbidden(name))

    def forbidden(self, name: str) -> Callable[..., object]:
        def call(*args: object, **kwargs: object) -> object:
            self.calls.append(name)
            raise AssertionError(f"{name} se llamó antes de una negativa")

        return call


def run(study: Study, now: datetime = LATE) -> module.AnalysisResult:
    return module.analyze(study.root, now, ("deadbeef", False), stage1=ROOT / STAGE1_DIR)


def text(result: module.AnalysisResult) -> str:
    return "\n".join(result.lines)


# ──────────────────────────────────────────── De punta a punta ────────────────────────────────────


def test_a_closed_study_reaches_the_verdict_known_in_advance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """51 bloques, el 11 vacío: el estudio cierra en el 51 con 50 no vacíos.

    `solo` compra y gana 1.2 % y 0.8 % alternos; `rule_trend` mantiene y vale 0. En el bloque 3
    `solo` pierde la evaluación por el proveedor (faltante) y en el 4 responde mal (inválida): las
    dos valen 0. Una evaluación por bloque y un bloque por clúster: G = 50 y el error agrupado es el
    ingenuo. La media es ~0.96 % con un IC 95 % muy por encima de 0: supera.
    """
    study = build_study(tmp_path, monkeypatch, special={2: missing, 3: invalid})
    diffs = study.differences

    result = run(study)

    found = text(result)
    assert result.code == 0, found
    assert "**supera**" in found
    mean = statistics.fmean(diffs)
    stderr = statistics.stdev(diffs) / math.sqrt(len(diffs))
    low, high = mean - 1.96 * stderr, mean + 1.96 * stderr
    assert low > 0
    assert f"| 50 | 0 | {mean:+.3%} | {stderr:.3%} |" in found
    assert f"[{low:+.3%}, {high:+.3%}]" in found
    assert f"| `{UPSTREAM[0]}` por `{UPSTREAM[1]}` | 48 |" in found
    assert f"| {module.NO_CALL} | 1 |" in found
    assert f"| {module.INVALID_RESPONSE} | 1 |" in found
    assert "faltantes: 1 de 50 (2.00%)" in found
    assert "respuestas inválidas de `solo`: 1" in found
    assert "bloques: 51, 50 no vacíos y 1 vacíos (los bloques 11)" in found
    assert "Reproduce: `python -m crypto_agents.forward analyze" in found
    manifest = manifest_path(FIRST_BLOCK, tmp_path / "manifests")
    assert f"`{manifest}`, sha-256 `{file_sha256(manifest)}`" in found
    assert "commit del análisis: `deadbeef` (árbol limpio)" in found
    assert (
        f"cambios de upstream del decisor, desde la etapa 1 (`{UPSTREAM[0]}` por `{UPSTREAM[1]}`):"
        " ninguno." in found
    ), "un bloque en que el decisor no respondió no es un cambio de upstream"


def test_a_new_upstream_is_counted_and_reported_apart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Desde el bloque 21 sirve otra ruta: es un cambio, y el reparto por upstream la separa en
    sus 31 evaluaciones, de los bloques 21 a 51, frente a las 19 de antes (el 11 está vacío)."""
    other = "accounts/fireworks/models/glm-5p4"

    def moved(entry: PlannedEvaluation) -> dict[str, list[EvaluationRecord]]:
        arms = decided(entry)
        arms["solo"] = [
            record(entry, Action.BUY, (decider_call(entry.at + PERIOD, upstream_model=other),))
        ]
        return arms

    study = build_study(tmp_path, monkeypatch, special=dict.fromkeys(range(20, 51), moved))

    found = text(run(study))

    assert (
        f"bloque 21: antes `{UPSTREAM[0]}` por `fireworks`, ahora `{other}` por `fireworks`"
        in found
    )
    assert f"| `{other}` por `fireworks` | 31 |" in found
    assert f"| `{UPSTREAM[0]}` por `fireworks` | 19 |" in found


def test_a_study_with_every_batch_complete_needs_no_expired_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Con todos los lotes válidos y cerrados, el análisis no tiene que esperar a ningún plazo."""
    study = build_study(tmp_path, monkeypatch)
    result = run(study, early())
    assert result.code == 0, text(result)
    assert "**supera**" in text(result)


# ──────────────────────────────────────── Negativas y «no determinado» ───────────────────────────


def assert_unscored(spy: Spy, result: module.AnalysisResult) -> None:
    assert spy.calls == []
    assert "No se puntuó ninguna evaluación" in text(result)


def test_forty_nine_filled_blocks_are_refused_without_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = build_study(tmp_path, monkeypatch, blocks=49, empty=())
    spy = Spy(monkeypatch)

    result = run(study)

    assert result.code == 1
    assert "sigue abierto" in text(result)
    assert "49 no vacíos" in text(result)
    assert_unscored(spy, result)


def test_block_52_without_fifty_filled_is_undetermined_without_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = build_study(tmp_path, monkeypatch, blocks=52, empty=(0, 20, 40))
    spy = Spy(monkeypatch)

    result = run(study)

    assert result.code == 0
    assert "**no determinado**: al terminar el bloque 52 hay 49 bloques no vacíos" in text(result)
    assert_unscored(spy, result)


def rewrite_meta(directory: Path, **changes: object) -> None:
    path = directory / "forward.json"
    meta = BatchMeta.model_validate_json(path.read_text("utf-8")).model_copy(update=changes)
    write_forward_meta(directory, meta)


def only_pass(root: Path, position: int) -> Path:
    (directory,) = sorted(
        batch_directory(root / "batches", FIRST_BLOCK + position * STEP).iterdir()
    )
    return directory


@pytest.mark.parametrize(
    "change",
    [
        {"measured_path": REGISTERED_PATH.model_copy(update={"schema_sha256": "0" * 64})},
        {"decider": registered().model_copy(update={"model": "glm-9.9"})},
    ],
    ids=["un sha-256 del camino medido", "la configuración del decisor"],
)
def test_a_pass_that_called_with_an_unregistered_path_is_undetermined(
    change: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = build_study(tmp_path, monkeypatch)
    rewrite_meta(only_pass(tmp_path, 20), **change)
    spy = Spy(monkeypatch)

    result = run(study)

    assert result.code == 0
    found = text(result)
    assert "**no determinado**: el estudio terminó antes de cerrar, en el bloque 21" in found
    assert "distintos de los registrados" in found
    assert_unscored(spy, result)


@pytest.mark.parametrize(
    ("outcome", "cause"),
    [
        (BatchOutcome.TERMINATED, "ya no está en el catálogo"),
        (BatchOutcome.GAP_PERSISTENT, "hueco persistente"),
    ],
)
def test_a_study_that_ended_early_is_undetermined(
    outcome: BatchOutcome, cause: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """El estudio termina en el bloque 31, con 30 bloques no vacíos detrás: nunca llegará a 50.

    La pasada que lo termina no corrió y solo dejó su `forward.json`. Con la regla de cierre
    delante, el análisis diría «sigue abierto» y se negaría para siempre: la terminación se lee
    primero, y es «no determinado» con sus contadores.
    """
    study = build_study(tmp_path, monkeypatch, blocks=31, empty=(), skipped=(30,))
    start = FIRST_BLOCK + 30 * STEP
    ended = batch_directory(tmp_path / "batches", start) / "20270601T000000Z"
    before = BatchMeta.model_validate_json((only_pass(tmp_path, 29) / "forward.json").read_text())
    write_forward_meta(
        ended, before.model_copy(update={"outcome": outcome, "block": block_dates(start)})
    )
    spy = Spy(monkeypatch)

    result = run(study)

    assert result.code == 0
    found = text(result)
    assert "**no determinado**: el estudio terminó antes de cerrar, en el bloque 31" in found
    assert cause in found
    assert "- evaluaciones: 30" in found
    assert_unscored(spy, result)


def test_a_run_id_decided_in_two_passes_stops_the_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = build_study(tmp_path, monkeypatch)
    start = FIRST_BLOCK + 5 * STEP
    write_pass(
        tmp_path,
        start,
        manifest_path(start, tmp_path / "manifests"),
        decided(study.entries[start][0]),
        stamp="20261201T000000Z",
    )
    spy = Spy(monkeypatch)

    result = run(study)

    assert result.code == 1
    assert "dos pasadas" in text(result)
    assert_unscored(spy, result)


def test_a_manifest_that_is_not_the_population_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = build_study(tmp_path, monkeypatch)
    start = FIRST_BLOCK + 7 * STEP
    other = study.entries[start][0].model_copy(update={"triggers": ("regime_change_above_trend",)})
    monkeypatch.setattr(
        module, "study_population", population_of({**study.entries, start: (other,)})
    )
    spy = Spy(monkeypatch)

    result = run(study)

    assert result.code == 1
    assert "el manifiesto del bloque 8" in text(result)
    assert "no es la población recalculada" in text(result)
    assert_unscored(spy, result)


def test_a_baseline_without_a_record_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def without_trend(entry: PlannedEvaluation) -> dict[str, list[EvaluationRecord]]:
        return {**decided(entry), "rule_trend": []}

    study = build_study(tmp_path, monkeypatch, special={12: without_trend})
    spy = Spy(monkeypatch)

    result = run(study)

    assert result.code == 1
    assert "bloque 13, `rule_trend`: 1 run_id del manifiesto sin registro" in text(result)
    assert_unscored(spy, result)


def test_an_invalid_batch_at_its_deadline_is_undetermined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = build_study(tmp_path, monkeypatch, special={6: foreign_cache})
    spy = Spy(monkeypatch)

    result = run(study)

    assert result.code == 0
    found = text(result)
    assert "**no determinado**: un lote inválido al vencer su plazo: bloque 7" in found
    assert "aciertos de caché con otro backend" in found
    assert_unscored(spy, result)


def test_an_invalid_batch_before_its_deadline_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = build_study(tmp_path, monkeypatch, special={EARLY_BLOCK - 1: foreign_cache})
    spy = Spy(monkeypatch)

    result = run(study, early())

    assert result.code == 1
    assert f"el lote del bloque {EARLY_BLOCK} no está completo" in text(result)
    assert_unscored(spy, result)


def test_a_missing_evaluation_before_its_deadline_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = build_study(tmp_path, monkeypatch, special={EARLY_BLOCK - 1: missing})
    spy = Spy(monkeypatch)

    result = run(study, early())

    assert result.code == 1
    assert f"el lote del bloque {EARLY_BLOCK} no está completo" in text(result)
    assert_unscored(spy, result)


def test_a_batch_that_never_ran_is_undetermined_once_its_deadline_passed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = build_study(tmp_path, monkeypatch, skipped=(25,))
    spy = Spy(monkeypatch)

    result = run(study)

    assert result.code == 0
    assert "un lote inválido al vencer su plazo: bloque 26 (su lote no corrió)" in text(result)
    assert_unscored(spy, result)


def test_missing_above_five_percent_is_undetermined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """3 faltantes de 50 evaluaciones son el 6 %."""
    study = build_study(tmp_path, monkeypatch, special=dict.fromkeys((1, 2, 3), missing))
    spy = Spy(monkeypatch)

    result = run(study)

    assert result.code == 0
    assert "las faltantes son 3 de 50 evaluaciones (6.00%), más del 5%" in text(result)
    assert_unscored(spy, result)


def test_a_directory_that_is_not_a_study_exits_two(tmp_path: Path) -> None:
    result = module.analyze(tmp_path / "no-existe", LATE, (None, None), stage1=ROOT / STAGE1_DIR)
    assert result.code == 2
    assert result.lines[0].startswith("error:")


# ──────────────────────────────────────── El ensayo sobre la etapa 1 ──────────────────────────────


def rehearse() -> module.AnalysisResult:
    return module.rehearse_analysis(
        ("deadbeef", False),
        stage1=ROOT / STAGE1_DIR,
        history=ROOT / HISTORY_DIR,
        funding=ROOT / FUNDING_DIR,
    )


VERDICT_WORDS = ("supera", "peor", "empate", "concluyente", "determinado", "justifica")


def test_the_rehearsal_reproduces_the_four_published_rows() -> None:
    result = rehearse()

    found = text(result)
    assert result.code == 0, found
    assert "coinciden 44 de 44 cifras" in found
    assert "| solo | rule_trend | IC 95% | [-0.424%, +0.333%] | [-0.424%, +0.333%] | sí |" in found
    assert not [word for word in VERDICT_WORDS if word in found.lower()]


def test_mutation_a_scoring_over_five_candles_breaks_the_rehearsal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    def five(
        directory: Path, plan: Path | None, history_dir: Path, horizon: int | None
    ) -> StageRun:
        return read_stage(directory, plan, history_dir, horizon)._replace(horizon=5)

    monkeypatch.setattr(module, "read_stage", five)
    result = rehearse()
    assert result.code == 1
    assert "**NO**" in text(result)


def test_the_rehearsal_never_reaches_the_verdict_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("el ensayo no aplica la regla de veredicto")

    for name in ("forward_verdict", "primary_verdict"):
        monkeypatch.setattr(module, name, forbidden)
    assert rehearse().code == 0

    source = textwrap.dedent(inspect.getsource(module.rehearse_analysis))
    for name in ("forward_verdict", "primary_verdict", "classify", "Verdict", "VERDICT_LABEL"):
        assert name not in source


def test_the_rehearsal_reads_only_a_stage1_with_the_published_digests(tmp_path: Path) -> None:
    copy = tmp_path / "stage1"
    copy.mkdir()
    for path in (ROOT / STAGE1_DIR).iterdir():
        (copy / path.name).write_bytes(path.read_bytes())
    (copy / "solo.jsonl").write_bytes((copy / "solo.jsonl").read_bytes() + b"\n")

    result = module.rehearse_analysis((None, None), stage1=copy)

    assert result.code == 1
    assert "no es la etapa 1" in text(result)


def test_the_published_rows_are_the_ones_in_the_ablation_document() -> None:
    """Las cifras copiadas son las de las filas `| solo | <línea base> |` del documento, sin la
    columna del veredicto."""
    document = (ROOT / "docs" / "ablation.md").read_text("utf-8")
    section = document.split("### La etapa 1 releída con el intervalo de la enmienda 4", 1)[1]
    section = section.split("\n## ", 1)[0]
    for reference, figures in module.PUBLISHED_STAGE1.items():
        (line,) = [row for row in section.splitlines() if row.startswith(f"| solo | {reference} |")]
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        assert tuple(cells[2:11] + cells[12:14]) == figures, reference
    assert next(iter(module.PUBLISHED_STAGE1)) == module.REFERENCE == "rule_trend"


# ─────────────────────────────────────────────── El comando ───────────────────────────────────────


@pytest.mark.parametrize(
    "argv",
    [
        ["analyze", "data/forward", "--force"],
        ["analyze", "data/forward", "--delta", "0.002"],
        ["analyze"],
        ["rehearse-analysis", "data/forward"],
    ],
)
def test_no_argument_skips_a_refusal(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        main(argv)


class Refusing:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *args: object) -> object:
        self.calls += 1
        raise AssertionError("el análisis no construye backends ni lee los ajustes")


def test_analyze_builds_no_backend_and_reads_no_settings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    backends, settings = Refusing(), Refusing()
    env = BatchEnvironment(clock=lambda: LATE, settings=settings, backends=backends)  # type: ignore[arg-type]

    code = main(["analyze", str(tmp_path / "no-existe")], env=env)

    assert code == 2
    assert backends.calls == settings.calls == 0
    assert "no existe el directorio del estudio" in capsys.readouterr().err


def test_the_arms_of_the_analysis_are_the_batch_ones() -> None:
    assert module.ARM == "solo"
    assert module.REFERENCE == "rule_trend"
    assert ("solo", *module.DESCRIPTIVE_REFERENCES, "rule_trend") == (
        "solo",
        "always_buy",
        "always_sell",
        "random_uniform",
        "rule_trend",
    )
    assert set(FORWARD_ARMS) == {"solo", *criteria.BASELINE_ARMS}


# ──────────────────────────────────────────── El registro ─────────────────────────────────────────


def test_the_register_records_the_merge_of_the_candle_rules() -> None:
    register = (ROOT / "docs" / "forward.md").read_text("utf-8").split("## Registro del estudio")[1]
    entry = register.split("Las reglas de velas del apartado 4 entran en master", 1)[1]
    for fragment in (
        "bffd24e",
        "e1317e3571100a8e0ab74bd704525e5b91cef63d",
        "ce3afe2f172f152ec91ca87a836e5359bd081837",
        "f7214a181de9e763052740476f2f0e3eccb45bd4",
        "2026-10-17T22:33:16-06:00",
        "2026-10-18T04:33:16Z",
        "pr_merge",
        "/activity",
        "2026-10-22T00:00Z",
    ):
        assert fragment in entry, fragment
    assert datetime(2026, 10, 18, 4, 33, 16, tzinfo=UTC) < FIRST_BLOCK
