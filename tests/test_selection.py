"""Pruebas de la selección estratificada y del manifiesto.

Lo que se comprueba aquí no es que la selección sea «buena» —eso lo dirá la tabla—
sino que sea la que dice ser: repartida entre símbolos y tramos, reproducible sin
volver a elegir, y verificable contra el histórico que la produjo.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from functools import cache
from pathlib import Path

import pytest

from crypto_agents import selection
from crypto_agents.activation import ActivationConfig
from crypto_agents.market import candles_digest, to_dataframe
from crypto_agents.replay import HistoricalMarketClient
from crypto_agents.selection import (
    DEFAULT_CANDLE_LIMIT,
    DEFAULT_MANIFEST,
    HISTORY_DIR,
    PlannedEvaluation,
    SelectionError,
    SelectionManifest,
    confirm_activation,
    load_manifest,
    load_selection_histories,
    replay_window,
    select_activations,
    verify_histories,
    window_digest,
    write_manifest,
)
from tests.conftest import PRESET, raw_ohlcv
from tests.test_upstream import mutated_method

GATE = ActivationConfig(preset=PRESET)
SYMBOLS = ("ADA/USDT", "BTC/USDT", "ETH/USDT")


def series(count: int = 400, phase: float = 0.0, drift: float = 0.05) -> list[list[float]]:
    """Serie determinista larga, con oscilación suficiente para que el gate abra."""
    closes = [100.0 + 8.0 * math.sin(index / 7.0 + phase) + drift * index for index in range(count)]
    volumes = [1000.0 + 200.0 * math.sin(index / 5.0 + phase) for index in range(count)]
    return raw_ohlcv(closes, volumes=volumes)


def histories(count: int = 400) -> dict[str, list[list[float]]]:
    """Tres series distintas entre sí, una por símbolo."""
    return {symbol: series(count, phase=float(position)) for position, symbol in enumerate(SYMBOLS)}


def manifest_of(
    target: int = 30, strata: int = 3, seed: int = 7, count: int = 400
) -> SelectionManifest:
    """Un manifiesto pequeño sobre las series sintéticas."""
    return select_activations(
        histories(count),
        timeframe="1h",
        target=target,
        strata=strata,
        seed=seed,
        horizon=3,
        candle_limit=DEFAULT_CANDLE_LIMIT,
        config=GATE,
        preset=PRESET,
        now=datetime(2026, 8, 15, tzinfo=UTC),
    )


# ─────────────────────────── La ventana es la que verá el replay ──────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("index", [120, 200, 399])
async def test_the_confirming_window_is_the_one_the_replay_will_see(index: int) -> None:
    """Confirmar con otra ventana sería confirmar otra cosa.

    `replay_window` reproduce a mano lo que hacen juntos `HistoricalMarketClient`
    y `drop_forming_candle`, porque este módulo no puede importar `replay.py` —que
    trae el router— sin poder llamar a un proveedor. Que la copia siga siendo fiel
    lo fija esta prueba y no la buena voluntad de quien la escribió.
    """
    rows = series(400)
    limit = 50
    market = HistoricalMarketClient(rows, index, limit)
    delivered = await market.fetch_ohlcv("BTC/USDT", "1h", limit)
    forming = index + 1 < len(rows)

    window = replay_window(rows, index, limit)
    # El exchange entrega la vela en formación y el pipeline la descarta; solo la
    # última vela de la serie llega sin una siguiente detrás.
    assert delivered[:-1] == window if forming else delivered == window
    assert window[-1][0] == rows[index][0]
    assert window_digest(rows, index, limit) == candles_digest(to_dataframe(window))


def test_a_bar_outside_the_series_is_an_error_not_an_empty_window() -> None:
    """Un índice que no existe significa que el histórico cambió, y hay que decirlo."""
    with pytest.raises(SelectionError, match="fuera de la serie"):
        replay_window(series(120), 500, DEFAULT_CANDLE_LIMIT)


def test_a_candidate_is_confirmed_against_the_truncated_window() -> None:
    """El barrido propone sobre la serie entera; la corrida verá 500 velas.

    EMA y ADX son recursivos, así que las dos vistas pueden discrepar en la misma
    vela. Lo que entra en el manifiesto es lo que abre el gate en la segunda.
    """
    rows = series(400)
    confirmed = [
        index
        for index in range(PRESET.min_bars, 399)
        if confirm_activation(rows, index, 60, GATE, PRESET) is not None
    ]
    assert confirmed, "el gate no abrió en ninguna vela con la ventana del replay"
    for index in confirmed:
        triggers = confirm_activation(rows, index, 60, GATE, PRESET)
        assert triggers, "una activación confirmada tiene que traer sus disparadores"


# ──────────────────────────────────── Reparto de la selección ─────────────────────────────────────


def test_the_selection_is_spread_across_symbols_and_strata() -> None:
    """150 activaciones contiguas de un símbolo son un régimen, no una muestra.

    Es el motivo entero de estratificar: repartida, la tabla mide si la ventaja de
    un brazo sobrevive al cambio de régimen; concentrada, mide un tramo de dos
    meses y lo llama arquitectura.
    """
    manifest = manifest_of(target=30, strata=3)

    assert len(manifest.entries) == 30
    assert manifest.by_symbol == dict.fromkeys(SYMBOLS, 10)
    assert set(manifest.by_stratum) == {0, 1, 2}, "algún tramo se quedó sin representación"
    assert all(count >= 9 for count in manifest.by_stratum.values())
    assert sum(manifest.by_stratum.values()) == 30


def test_the_same_seed_selects_the_same_activations() -> None:
    """Sin esto, el manifiesto no sería reproducible ni siquiera antes de escribirlo."""
    first = manifest_of(seed=7)
    second = manifest_of(seed=7)
    assert [entry.at for entry in first.entries] == [entry.at for entry in second.entries]


def test_a_different_seed_selects_differently() -> None:
    """Si la semilla no cambiara nada, declararla sería decorativo."""
    first = manifest_of(seed=7)
    other = manifest_of(seed=99)
    assert [entry.index for entry in first.entries] != [entry.index for entry in other.entries]


def test_every_selected_bar_has_room_to_be_scored() -> None:
    """Una activación sin horizonte por delante no puntúa: no aporta a la tabla."""
    manifest = manifest_of(target=30, strata=3, count=400)
    for entry in manifest.entries:
        assert entry.index + manifest.horizon <= 399


def test_asking_for_more_than_the_history_holds_fails_loudly() -> None:
    """Un manifiesto corto en silencio sería una tabla con menos material del pedido."""
    with pytest.raises(SelectionError, match="confirmar"):
        manifest_of(target=10_000, strata=3)


def test_a_stratum_without_activations_is_refused() -> None:
    """Rellenarlo con el tramo vecino dejaría de estar repartido en el tiempo."""
    with pytest.raises(SelectionError, match="tramos sin ninguna activación"):
        select_activations(
            {"BTC/USDT": series(400)},
            timeframe="1h",
            target=10,
            strata=200,
            seed=7,
            horizon=3,
            config=GATE,
            preset=PRESET,
            now=datetime(2026, 8, 15, tzinfo=UTC),
        )


# ─────────────────────────── El manifiesto sobrevive a una descarga ───────────────────────────────


def test_a_manifest_survives_a_round_trip_through_disk(tmp_path: Path) -> None:
    """Lo que reproduce una corrida es la lista escrita, no la semilla."""
    manifest = manifest_of()
    path = tmp_path / "selection.json"
    write_manifest(path, manifest)

    assert load_manifest(path) == manifest


def test_verification_passes_against_the_history_that_produced_it() -> None:
    """El caso normal: mismas velas, mismos digests, nada que decir."""
    data = histories()
    manifest = manifest_of()
    verify_histories(manifest, data)


def test_a_revised_candle_is_caught_before_anything_is_spent() -> None:
    """Un exchange que revisa una vela convierte la tabla en incomparable.

    Se comprueba antes de la primera llamada: enterarse al empezar cuesta un
    segundo, y enterarse al terminar cuesta la corrida entera.
    """
    manifest = manifest_of()
    data = histories()
    touched = manifest.entries[0]
    rows = [list(row) for row in data[touched.symbol]]
    rows[touched.index][4] += 1.0
    data[touched.symbol] = rows

    with pytest.raises(SelectionError, match="la ventana cambió"):
        verify_histories(manifest, data)


def test_a_missing_history_is_named() -> None:
    """Faltando una serie, el error dice cuál en vez de fallar al indexar."""
    manifest = manifest_of()
    with pytest.raises(SelectionError, match="faltan históricos"):
        verify_histories(manifest, {})


def test_an_entry_outside_a_shorter_history_is_named() -> None:
    """Un histórico recortado no puede evaluar la vela que el manifiesto pide."""
    manifest = manifest_of()
    data = {symbol: rows[:200] for symbol, rows in histories().items()}
    with pytest.raises(SelectionError, match="no existe en un histórico"):
        verify_histories(manifest, data)


def test_the_manifest_refuses_entries_without_their_series() -> None:
    """Una entrada que no se puede resolver es un manifiesto roto, no una advertencia."""
    manifest = manifest_of()
    intruder = PlannedEvaluation(
        symbol="DOGE/USDT",
        timeframe="1h",
        index=300,
        at=manifest.entries[0].at,
        candles_digest="0" * 64,
    )
    payload = manifest.model_dump(mode="json")
    payload["entries"] = [*payload["entries"], intruder.model_dump(mode="json")]

    with pytest.raises(ValueError, match="sin serie declarada"):
        SelectionManifest.model_validate(payload)


# ─────────────────────────── La selección mayor contiene a la versionada ──────────────────────────
# La enmienda 3 deja abierta una etapa con n mayor. Crecer de 140 a 280 solo conserva lo ya medido
# si las 140 del manifiesto siguen dentro: de lo contrario la selección nueva es otra muestra y lo
# pagado no se puede reutilizar.

ROOT = Path(selection.__file__).parents[2]
COMMITTED = ROOT / DEFAULT_MANIFEST


@cache
def committed() -> SelectionManifest:
    return load_manifest(COMMITTED)


def reselected(target: int) -> SelectionManifest:
    """Otra selección con la semilla y las series del manifiesto versionado, y solo otro target."""
    manifest = committed()
    return select_activations(
        load_selection_histories(manifest, ROOT / HISTORY_DIR),
        timeframe=manifest.timeframe,
        target=target,
        strata=manifest.strata,
        seed=manifest.seed,
        horizon=manifest.horizon,
        candle_limit=manifest.candle_limit,
        now=manifest.created_at,
    )


def assert_it_contains_the_committed_selection(target: int) -> SelectionManifest:
    """Cada entrada del manifiesto está en la mayor, igual: vela, tramo, digest y disparos.

    Devuelve la selección mayor: calcularla son segundos y quien la quiera mirar no la repite.
    """
    selected = reselected(target)
    larger = {(entry.symbol, entry.index): entry for entry in selected.entries}
    assert len(larger) == target
    missing = [
        f"{entry.symbol} #{entry.index}"
        for entry in committed().entries
        if larger.get((entry.symbol, entry.index)) != entry
    ]
    assert not missing, (
        f"la selección de {target} no contiene {len(missing)} de las 140 del manifiesto: "
        f"{', '.join(missing[:5])}"
    )
    return selected


@pytest.mark.parametrize("target", [280, 420])
def test_a_larger_selection_contains_the_committed_one(target: int) -> None:
    """Con la semilla y los históricos del manifiesto, 280 y 420 contienen sus 140.

    Sobre las series reales y no sobre una sintética: lo que se quiere saber es si *este*
    manifiesto se puede ampliar. El relleno de celdas cortas no llega a activarse aquí —la celda
    más pequeña tiene 106 candidatas y 420 pide 12—, y eso también lo dice esta prueba: una
    celda rellenada llevaría más entradas que su cuota.
    """
    larger = assert_it_contains_the_committed_selection(target)

    cells: dict[tuple[str, int], int] = {}
    for entry in larger.entries:
        cells[(entry.symbol, entry.stratum)] = cells.get((entry.symbol, entry.stratum), 0) + 1
    per_cell = target // len(committed().series) // committed().strata
    assert set(cells.values()) == {per_cell}, "una celda se rellenó desde otra"


def test_mutation_a_shuffle_that_depends_on_the_target_breaks_the_nesting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """La mutación: cada celda se baraja con un generador sembrado con la cuota del símbolo.

    Sigue siendo determinista y sigue repartiendo por símbolo y tramo —nada más lo delata—, pero
    el orden de cada celda ya depende de cuánto se pide, y lo elegido con 280 deja de incluir lo
    elegido con 140.
    """
    by_target = mutated_method(
        selection._select_from_series,
        "rng.shuffle(bucket)",
        "random.Random(quota).shuffle(bucket)",
        vars(selection),
    )
    monkeypatch.setattr(selection, "_select_from_series", by_target)
    with pytest.raises(AssertionError, match="no contiene"):
        assert_it_contains_the_committed_selection(280)


# ──────────────────────── El manifiesto versionado no se pisa por defecto ─────────────────────────


def never_computed(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Sustituye el cálculo por un testigo: una negativa que llega después ya gastó el minuto."""
    reached: list[str] = []

    async def witness(_args: object) -> SelectionManifest:
        reached.append("_run")
        return committed()

    monkeypatch.setattr(selection, "_run", witness)
    return reached


def test_without_out_the_command_refuses_before_computing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--out` valía `data/ablation_selection.json`: relanzar el comando sustituía el plan."""
    monkeypatch.chdir(tmp_path)
    reached = never_computed(monkeypatch)

    assert selection.main([]) == 1

    assert "falta --out" in capsys.readouterr().err
    assert reached == []
    assert list(tmp_path.iterdir()) == [], "se escribió algo sin que nadie dijera dónde"


def test_an_out_that_exists_is_refused_and_left_as_it_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    existing = tmp_path / "seleccion.json"
    existing.write_text("el plan que algo ya usó\n", encoding="utf-8")
    reached = never_computed(monkeypatch)

    assert selection.main(["--out", str(existing)]) == 1

    assert "ya existe y no se pisa" in capsys.readouterr().err
    assert reached == []
    assert existing.read_text(encoding="utf-8") == "el plan que algo ya usó\n"


def test_an_out_that_does_not_exist_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "nueva" / "seleccion.json"
    reached = never_computed(monkeypatch)

    assert selection.main(["--out", str(target)]) == 0, capsys.readouterr().err

    assert reached == ["_run"]
    assert load_manifest(target) == committed()


def test_the_committed_manifest_is_still_where_the_readers_look_for_it() -> None:
    """Que el comando ya no lo escriba por defecto no mueve de dónde se lee."""
    assert DEFAULT_MANIFEST.as_posix() == "data/ablation_selection.json"
    assert COMMITTED.is_file()
