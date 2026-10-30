"""La huella del decisor de `solo`: a qué velocidad contesta y en qué orden da sus argumentos.

Desde el bloque 1 la pasarela no manda las cabeceras de upstream (registro, 2026-10-30), y el cambio
en cómo se sirve `glm-5.2` se vio por dos huellas de cómo se genera la respuesta:

- **la velocidad**, tokens de salida por segundo de pared (`completion_tokens / latency_ms`), que
  está en los journals versionados;
- **el orden de las claves de los argumentos**, que solo está en el texto crudo de cada respuesta,
  es decir en la caché de `var/`, que no se versiona.

Este módulo las deja escritas en `data/forward/fingerprints/`: `stage1.json` para la etapa 1 y
`<bloque>/<pasada>.json` para cada pasada del estudio. Nunca escribe dentro de `batches/` ni de
`data/stage1/`, y no compara nada: los órdenes van tal cual, sin decir cuál es el del esquema. El
archivo no lleva ninguna hora, así que volver a calcularlo da los mismos bytes.

Lo usan dos caminos:

- el lote (`forward.run_pass`), al final de cada pasada que llamó al decisor, desde la caché que
  usó esa pasada. Un fallo al calcularla queda escrito en el archivo con su motivo y nunca cambia el
  resultado ni el código de salida del lote;
- `python -m crypto_agents.forward fingerprint`, la huella retroactiva de la etapa 1 y de las
  pasadas anteriores a este código (`retroactive`).

Lee los journals y la caché sin tocarlos. No importa el router ni nada que lo traiga: un test de
arquitectura lo importa en un proceso nuevo.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, Self
from uuid import UUID

from pydantic import AwareDatetime, Field, ValidationError, model_validator

from crypto_agents.audit import arm_journal_path, file_sha256
from crypto_agents.cache import CacheEntry, JsonFileResponseCache, cache_key
from crypto_agents.forward_record import TARGET_ARM, batch_passes, superseded
from crypto_agents.journal import JournalError, JsonlJournal
from crypto_agents.state import AgentRole, FailureKind, FrozenModel, Proposal

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from crypto_agents.journal import EvaluationRecord
    from crypto_agents.state import LLMCall

__all__ = [
    "FINGERPRINT_DIR",
    "STAGE1_CACHE",
    "STAGE1_FINGERPRINT",
    "CallFingerprint",
    "DeciderFingerprint",
    "FingerprintSummary",
    "KeyOrder",
    "Retroactive",
    "Speed",
    "block_fingerprints",
    "call_fingerprint",
    "decider_called",
    "failed_fingerprint",
    "fingerprint",
    "fingerprint_line",
    "key_order",
    "pass_fingerprint_path",
    "retroactive",
    "summarise",
    "tokens_per_second",
    "write_fingerprint",
]

FINGERPRINT_DIR = "fingerprints"
"""El directorio de las huellas dentro del estudio: `data/forward/fingerprints/`."""

STAGE1_FINGERPRINT = "stage1.json"
"""La huella de la etapa 1, en la raíz de `fingerprints/`."""

STAGE1_CACHE = Path("var/ablation-cache")
"""La caché que llenó la etapa 1, la que lee `forward rehearse`. No se versiona."""

_NO_CONTENT = frozenset({FailureKind.TIMEOUT, FailureKind.TRANSPORT})
"""Los fallos del proveedor: el intento no trajo texto, así que no dejó nada en la caché."""


class CallFingerprint(FrozenModel):
    """Una llamada del decisor: lo que dice el journal y el orden de las claves de su texto."""

    run_id: UUID
    at: AwareDatetime
    """`LLMCall.at`: cuándo terminó la llamada."""

    cache_hit: bool
    completion_tokens: int | None = Field(ge=0)
    latency_ms: float = Field(ge=0.0)
    tokens_per_second: float | None = Field(ge=0.0)
    key_order: tuple[str, ...] | None
    """Las claves de primer nivel de los argumentos, en el orden en que llegaron."""

    key_order_error: str | None
    """Por qué no hay orden: el intento no trajo texto, o la caché no tiene la entrada."""

    @model_validator(mode="after")
    def _order_or_reason(self) -> Self:
        if (self.key_order is None) == (self.key_order_error is None):
            raise ValueError("una llamada lleva su orden de claves o el motivo de que no lo tenga")
        return self


class Speed(FrozenModel):
    """El tok/s de las llamadas vivas que lo tienen."""

    calls: int = Field(ge=1)
    minimum: float = Field(ge=0.0)
    median: float = Field(ge=0.0)
    maximum: float = Field(ge=0.0)


class KeyOrder(FrozenModel):
    """Un orden de claves y cuántas llamadas lo tienen."""

    keys: tuple[str, ...]
    calls: int = Field(ge=1)


class FingerprintSummary(FrozenModel):
    """El resumen de una huella: velocidad de las vivas y llamadas por orden de claves."""

    calls: int = Field(ge=0)
    live_calls: int = Field(ge=0)
    speed: Speed | None
    """Sobre las llamadas vivas con tokens y latencia. Los aciertos de caché no entran: su latencia
    es la de leer un archivo, no la del modelo."""

    orders: tuple[KeyOrder, ...]
    """Sobre todas las llamadas, de la más frecuente a la menos y, a igual número, por claves."""

    without_order: int = Field(ge=0)


class DeciderFingerprint(FrozenModel):
    """La huella del decisor de un journal de `solo`, o el motivo de que no se pudiera calcular."""

    journal: str
    journal_sha256: str | None
    """El sha-256 del journal leído; `None` si no se pudo leer."""

    cache: str
    """La caché de la que salió el texto crudo. No se versiona: el orden de claves es lo que queda
    de ella aquí."""

    error: str | None = None
    calls: tuple[CallFingerprint, ...] = ()
    summary: FingerprintSummary | None = None

    @model_validator(mode="after")
    def _summary_or_error(self) -> Self:
        if (self.error is None) == (self.summary is None):
            raise ValueError("una huella lleva su resumen o el motivo de que no se calculara")
        return self


def tokens_per_second(call: LLMCall) -> float | None:
    """Tokens de salida por segundo de pared. `None` sin tokens o sin latencia que dividir."""
    if call.completion_tokens is None or call.latency_ms <= 0:
        return None
    return call.completion_tokens / (call.latency_ms / 1000)


class _Pairs(list[tuple[str, object]]):
    """Los pares de un objeto JSON tal como llegaron, duplicados incluidos."""


def key_order(raw: str) -> tuple[str, ...]:
    """Las claves de primer nivel de un objeto JSON, en el orden del texto. `ValueError` si el
    texto no es un objeto JSON."""
    parsed = json.loads(raw, object_pairs_hook=_Pairs)
    if not isinstance(parsed, _Pairs):
        raise ValueError(f"es un {type(parsed).__name__} de JSON, no un objeto")
    return tuple(key for key, _ in parsed)


def _order_from_cache(call: LLMCall, cache: JsonFileResponseCache) -> tuple[str, ...] | str:
    """El orden de claves del texto que la caché guarda para esa llamada, o por qué no lo hay."""
    if call.failure_kind is not None and call.failure_kind in _NO_CONTENT:
        return f"el intento no trajo texto ({call.failure_kind.value})"
    key = cache_key(call.backend, call.model, call.prompt_digest, Proposal, call.structured_output)
    payload = cache.get(key)
    if payload is None:
        return f"falta la entrada {key} en la caché"
    try:
        entry = CacheEntry.model_validate_json(payload)
    except ValidationError:
        return f"la entrada {key} no es un sobre de caché"
    if not entry.answers(
        call.backend, call.model, call.prompt_digest, Proposal, call.structured_output
    ):
        return f"la entrada {key} no es la de esta llamada"
    try:
        return key_order(entry.raw)
    except ValueError as error:
        return f"el texto de la entrada {key} no es un objeto JSON: {error}"


def call_fingerprint(run_id: UUID, call: LLMCall, cache: JsonFileResponseCache) -> CallFingerprint:
    """La huella de una llamada del decisor, con el orden de claves leído de `cache`."""
    order = _order_from_cache(call, cache)
    return CallFingerprint(
        run_id=run_id,
        at=call.at,
        cache_hit=call.cache_hit,
        completion_tokens=call.completion_tokens,
        latency_ms=call.latency_ms,
        tokens_per_second=tokens_per_second(call),
        key_order=None if isinstance(order, str) else order,
        key_order_error=order if isinstance(order, str) else None,
    )


def summarise(calls: Sequence[CallFingerprint]) -> FingerprintSummary:
    """La velocidad de las vivas y cuántas llamadas tiene cada orden de claves."""
    live = [call for call in calls if not call.cache_hit]
    speeds = [call.tokens_per_second for call in live if call.tokens_per_second is not None]
    counted = Counter(call.key_order for call in calls if call.key_order is not None)
    return FingerprintSummary(
        calls=len(calls),
        live_calls=len(live),
        speed=Speed(
            calls=len(speeds),
            minimum=min(speeds),
            median=statistics.median(speeds),
            maximum=max(speeds),
        )
        if speeds
        else None,
        orders=tuple(
            KeyOrder(keys=keys, calls=count)
            for keys, count in sorted(counted.items(), key=lambda item: (-item[1], item[0]))
        ),
        without_order=sum(1 for call in calls if call.key_order is None),
    )


def decider_called(records: Iterable[EvaluationRecord]) -> bool:
    """Si algún registro lleva una llamada del decisor, viva o de caché."""
    return any(call.role is AgentRole.DECIDER for record in records for call in record.calls)


def failed_fingerprint(journal: Path, cache: Path, reason: str) -> DeciderFingerprint:
    """La huella que no se pudo calcular: el archivo lo dice, con el motivo."""
    return DeciderFingerprint(
        journal=str(journal), journal_sha256=None, cache=str(cache), error=reason
    )


def fingerprint(journal: Path, cache: Path) -> DeciderFingerprint:
    """La huella de las llamadas del decisor de un journal de `solo`, en su orden.

    El texto crudo sale de `cache`, que se lee sin tocarla. Un journal o una caché que no existen,
    o un journal que no se puede leer, dan una huella con el motivo y sin llamadas.
    """
    if not journal.is_file():
        return failed_fingerprint(journal, cache, f"no existe el journal {journal}")
    if not cache.is_dir():
        # Comprobado antes de construirla: `JsonFileResponseCache` crea el directorio que le falte.
        return failed_fingerprint(journal, cache, f"no existe la caché {cache}")
    try:
        records = JsonlJournal(journal).read_all()
        sha256 = file_sha256(journal)
    except (JournalError, OSError) as error:
        return failed_fingerprint(journal, cache, str(error))
    store = JsonFileResponseCache(cache)
    calls = tuple(
        call_fingerprint(record.run_id, call, store)
        for record in records
        for call in record.calls
        if call.role is AgentRole.DECIDER
    )
    return DeciderFingerprint(
        journal=str(journal),
        journal_sha256=sha256,
        cache=str(cache),
        calls=calls,
        summary=summarise(calls),
    )


def pass_fingerprint_path(root: Path, pass_dir: Path) -> Path:
    """`fingerprints/<bloque>/<pasada>.json` para la pasada `batches/<bloque>/<pasada>/`."""
    return root / pass_dir.parent.name / f"{pass_dir.name}.json"


def block_fingerprints(root: Path, block: str) -> list[Path]:
    """Las huellas de las pasadas de un bloque (`<AAAA-MM-DD>`), ordenadas."""
    directory = root / block
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir() if path.is_file())


def write_fingerprint(path: Path, found: DeciderFingerprint) -> Path:
    """Escribe la huella. Nunca sobrescribe: un archivo que ya existe es `FileExistsError`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(found.model_dump_json(indent=2) + "\n")
    return path


def _keys(order: KeyOrder) -> str:
    return f"«{', '.join(order.keys)}» en {order.calls}"


def fingerprint_line(found: DeciderFingerprint, path: Path) -> str:
    """La línea del resumen del lote: tok/s de las vivas y llamadas por orden de claves."""
    summary = found.summary
    if summary is None:
        return f"- huella del decisor ({path}): no se pudo calcular: {found.error}"
    speed = (
        f"{summary.speed.minimum:.1f} a {summary.speed.maximum:.1f} tok/s, mediana "
        f"{summary.speed.median:.1f}"
        if summary.speed is not None
        else "sin tok/s: ninguna llamada viva con tokens"
    )
    orders = "; ".join(_keys(order) for order in summary.orders) or "ninguno"
    missing = f"; {summary.without_order} sin orden" if summary.without_order else ""
    return (
        f"- huella del decisor ({path}): {summary.live_calls} llamadas vivas de {summary.calls}; "
        f"{speed}; orden de los argumentos: {orders}{missing}"
    )


# ───────────────────────────────────────── La retroactiva ─────────────────────────────────────────


class Retroactive(NamedTuple):
    """Lo que imprime `forward fingerprint` y con qué código sale."""

    code: int
    lines: tuple[str, ...]
    written: tuple[Path, ...]


def retroactive(
    root: Path, stage1: Path, stage1_cache: Path, batches: Path, study_cache: Path
) -> Retroactive:
    """La huella de la etapa 1 y la de cada pasada que llamó al decisor y todavía no la tiene.

    - La etapa 1 sale de `stage1/solo.jsonl` y de su caché; las pasadas, de la caché del estudio.
    - Un lote con una pasada sustituida se salta entero: su repetición corrió con una caché propia
      (`forward.pass_cache`), y aquí no se adivina cuál.
    - Nunca sobrescribe. De una huella que ya existe solo dice que existe.
    - Una pasada sin journal de `solo`, o cuyo journal no trae llamadas del decisor, no tiene
      huella.
    - Sale con 1 si alguna de las que escribe no se pudo calcular o tiene llamadas sin orden.

    Escribe solo dentro de `root`.
    """
    lines: list[str] = []
    written: list[Path] = []
    incomplete = False

    def write(path: Path, found: DeciderFingerprint) -> None:
        nonlocal incomplete
        write_fingerprint(path, found)
        written.append(path)
        lines.append(fingerprint_line(found, path))
        incomplete = incomplete or found.summary is None or found.summary.without_order > 0

    target = root / STAGE1_FINGERPRINT
    if target.exists():
        lines.append(f"- {target}: ya existe, no se reescribe")
    else:
        write(target, fingerprint(arm_journal_path(stage1, TARGET_ARM), stage1_cache))
    blocks = sorted(path for path in batches.iterdir() if path.is_dir()) if batches.is_dir() else []
    for block in blocks:
        chain = batch_passes(block, include_superseded=True)
        if any(superseded(path) is not None for path in chain):
            lines.append(
                f"- {block}: tiene pasadas sustituidas, y su repetición no usó la caché del "
                "estudio: se salta"
            )
            continue
        for directory in chain:
            target = pass_fingerprint_path(root, directory)
            if target.exists():
                lines.append(f"- {target}: ya existe, no se reescribe")
                continue
            journal = arm_journal_path(directory, TARGET_ARM)
            if not journal.is_file():
                continue
            found = fingerprint(journal, study_cache)
            if found.summary is not None and not found.calls:
                continue
            write(target, found)
    return Retroactive(1 if incomplete else 0, tuple(lines), tuple(written))
