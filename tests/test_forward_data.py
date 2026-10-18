"""Los datos del estudio hacia adelante: la etapa 1 versionada, las series extendidas, la población.

Cuatro cosas, cada una con su mutación:

- **La etapa 1 versionada es la publicada.** Un byte cambiado en la copia se detecta al leerla.
- **Las series extendidas solo crecen por la derecha.** Sobre los archivos reales de
  `data/history/` y `data/funding/` y un exchange simulado:
  - las filas cerradas salen byte a byte;
  - la vela que `data/history/` guardó en formación se sustituye por la cerrada;
  - una vela o una liquidación revisadas, un hueco o una serie corta detienen la extensión.
- **La población de un bloque es la de `forward_plan`.** Sobre una serie diaria sintética, el
  manifiesto del bloque contiene exactamente lo que `design_effect.pool_records` deja en ese bloque.
- **El manifiesto del bloque se lee por las puertas de la ablación**, y reanudar no lo cambia.
"""

from __future__ import annotations

import math
import shutil
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

import crypto_agents.design_effect as design_module
import crypto_agents.forward as forward_module
import crypto_agents.forward_data as module
from crypto_agents.activation_sweep import DEFAULT_SYMBOLS, history_path
from crypto_agents.audit import file_sha256, read_run
from crypto_agents.design_effect import pool_records
from crypto_agents.forward_data import (
    PUBLISHED_FUNDING_SHA256,
    PUBLISHED_HISTORY_SHA256,
    SERIES_START,
    STAGE1_DIR,
    STAGE1_FILES,
    CandleRef,
    ForwardDataError,
    HistoryExtension,
    MissingCandlesError,
    RevisedCandlesError,
    block_manifest,
    block_start_of,
    extend_funding,
    extend_history,
    manifest_path,
    ohlcv_lines,
    read_revisions,
    study_population,
    verify_stage1,
    write_block_manifest,
)
from crypto_agents.forward_plan import batch_spend_cap, spend_cap
from crypto_agents.funding import FUNDING_DIR, funding_path, read_funding_csv
from crypto_agents.market import read_ohlcv_csv, write_ohlcv_csv
from crypto_agents.selection import (
    DEFAULT_CANDLE_LIMIT,
    DEFAULT_HORIZON,
    DEFAULT_SEED,
    HISTORY_DIR,
    SelectionManifest,
    confirm_activation,
    load_manifest,
    select_activations,
    verify_histories,
    window_digest,
)
from tests.conftest import DATA_DIR, PRESET, raw_ohlcv
from tests.test_net_outcomes import mutated
from tests.test_selection import GATE

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

ROOT = DATA_DIR.parent.parent
DOCS = ROOT / "docs" / "forward.md"
FOUR_HOURS = timedelta(hours=4)
UNTIL = datetime(2026, 8, 17, tzinfo=UTC)
"""Un día después del histórico publicado: seis velas nuevas tras la fila 4 380."""

PUBLISHED_ROWS = 4380
FORMING = datetime(2026, 8, 15, 20, tzinfo=UTC)
"""La apertura de la fila 4 380 de `data/history/`, guardada en formación."""


def ms(moment: datetime) -> float:
    return float(int(moment.timestamp() * 1000))


# ────────────────────────────────────────── La etapa 1 ────────────────────────────────────────────


def test_the_stage1_copy_is_the_published_one() -> None:
    verify_stage1(ROOT / STAGE1_DIR)
    for name, digest in STAGE1_FILES.items():
        assert file_sha256(ROOT / STAGE1_DIR / name) == digest
    text = DOCS.read_text("utf-8")
    assert all(f"`{digest}`" in text for digest in STAGE1_FILES.values())
    assert set(STAGE1_FILES) == {
        "meta.json",
        "solo.jsonl",
        "always_buy.jsonl",
        "always_sell.jsonl",
        "random_uniform.jsonl",
        "rule_trend.jsonl",
    }


def test_a_changed_byte_or_a_missing_file_in_the_copy_is_caught(tmp_path: Path) -> None:
    copy = tmp_path / "stage1"
    shutil.copytree(ROOT / STAGE1_DIR, copy)
    verify_stage1(copy)

    journal = copy / "solo.jsonl"
    content = bytearray(journal.read_bytes())
    content[100] ^= 1
    journal.write_bytes(bytes(content))
    with pytest.raises(ForwardDataError, match=r"solo\.jsonl"):
        verify_stage1(copy)

    shutil.copy(ROOT / STAGE1_DIR / "solo.jsonl", journal)
    (copy / "rule_trend.jsonl").unlink()
    with pytest.raises(ForwardDataError, match="falta"):
        verify_stage1(copy)


def test_the_copy_reads_as_a_run_with_full_as_an_empty_arm() -> None:
    """`full.jsonl` no se versiona: el brazo se lee vacío y sin journal; los otros, con 140."""
    run = read_run(ROOT / STAGE1_DIR)
    arms = {arm.arm: arm for arm in run.arms}

    assert arms["full"].sha256 is None
    assert arms["full"].records == ()
    for name in ("solo", "always_buy", "always_sell", "random_uniform", "rule_trend"):
        assert len(arms[name].records) == 140


# ───────────────────────────────────── Las series publicadas ──────────────────────────────────────


def test_the_published_digests_are_the_ones_of_the_files_and_of_their_documents() -> None:
    forward = DOCS.read_text("utf-8")
    readme = (ROOT / FUNDING_DIR / "README.md").read_text("utf-8")
    assert set(PUBLISHED_HISTORY_SHA256) == set(PUBLISHED_FUNDING_SHA256) == set(DEFAULT_SYMBOLS)
    for symbol in DEFAULT_SYMBOLS:
        history = ROOT / history_path(symbol, "4h", HISTORY_DIR)
        funding = ROOT / funding_path(symbol)
        assert file_sha256(history) == PUBLISHED_HISTORY_SHA256[symbol]
        assert f"`{PUBLISHED_HISTORY_SHA256[symbol]}`" in forward
        assert file_sha256(funding) == PUBLISHED_FUNDING_SHA256[symbol]
        assert f"{PUBLISHED_FUNDING_SHA256[symbol]}  {funding.name}" in readme


def test_the_register_records_the_stored_row_saved_while_forming() -> None:
    """La fila 4 380 de las siete series es la vela de `FORMING`. El registro la anota como vela
    revisada con el valor guardado, con el mismo formato que prepara el lote."""
    register = " ".join(DOCS.read_text("utf-8").split("## Registro del estudio", 1)[1].split())
    for symbol in DEFAULT_SYMBOLS:
        rows = read_ohlcv_csv(ROOT / history_path(symbol, "4h", HISTORY_DIR))
        assert len(rows) == PUBLISHED_ROWS
        assert rows[-1][0] == ms(FORMING)
        assert f"guardado: {forward_module._values(tuple(rows[-1][1:]))}" in register


def test_the_line_format_is_the_one_of_the_published_history_and_write_ohlcv_csv_is_not(
    tmp_path: Path,
) -> None:
    """`ohlcv_lines` reproduce byte a byte las siete series. `write_ohlcv_csv` escribe `\\r\\n`, y
    con ella extender reescribiría lo publicado con otros bytes."""
    for symbol in DEFAULT_SYMBOLS:
        source = ROOT / history_path(symbol, "4h", HISTORY_DIR)
        content = source.read_bytes()
        header = content[: content.index(b"\n") + 1]
        assert header + ohlcv_lines(read_ohlcv_csv(source)) == content

        rewritten = tmp_path / source.name
        write_ohlcv_csv(rewritten, read_ohlcv_csv(source))
        assert rewritten.read_bytes() != content


# ───────────────────────────────────────── Las velas ──────────────────────────────────────────────


def exchange_rows(symbol: str, until: datetime = UNTIL) -> list[list[float]]:
    """Lo que el exchange simulado devuelve hoy: lo publicado con la fila 4 380 ya cerrada, como el
    exchange real el 2026-10-17, y velas nuevas hasta la que cierra en `until`."""
    published = read_ohlcv_csv(ROOT / history_path(symbol, "4h", HISTORY_DIR))
    rows = [list(row) for row in published[: PUBLISHED_ROWS - 1]]
    forming = list(published[PUBLISHED_ROWS - 1])
    rows.append([forming[0], forming[1], forming[2], forming[3] * 0.99, forming[4] * 0.995, 3e6])
    moment = FORMING + FOUR_HOURS
    level = rows[-1][4]
    while moment + FOUR_HOURS <= until:
        rows.append([ms(moment), level, level * 1.01, level * 0.99, level * 1.002, 12345.5])
        level *= 1.002
        moment += FOUR_HOURS
    return rows


class FakeExchange:
    """El exchange simulado: devuelve `[inicio, fin)` de cada serie y anota cada petición."""

    def __init__(self, rows: Mapping[str, list[list[float]]]) -> None:
        self.rows = rows
        self.asked: list[tuple[str, datetime, datetime]] = []

    async def __call__(
        self, symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> list[list[float]]:
        assert timeframe == "4h"
        self.asked.append((symbol, start, end))
        return [list(row) for row in self.rows[symbol] if ms(start) <= row[0] < ms(end)]


def exchange(until: datetime = UNTIL) -> FakeExchange:
    return FakeExchange({symbol: exchange_rows(symbol, until) for symbol in DEFAULT_SYMBOLS})


FORMING_REFS = tuple(CandleRef(symbol, FORMING) for symbol in DEFAULT_SYMBOLS)
"""Las siete velas de la fila 4 380, que el operador acepta con el flag."""

ACCEPTED_AT = datetime(2026, 10, 30, tzinfo=UTC)
BLOCK = datetime(2026, 10, 22, tzinfo=UTC)


async def extend(
    fake: FakeExchange,
    directory: Path,
    until: datetime = UNTIL,
    accept: Sequence[CandleRef] = FORMING_REFS,
    now: datetime | None = None,
) -> HistoryExtension:
    """Por el módulo, para que una mutación de `extend_history` se vea. Sin `now`, el reloj marca
    `until`: la vela que cierra en `until` acaba de cerrar."""
    return await module.extend_history(
        until,
        fake,
        now=until if now is None else now,
        accept=accept,
        accepted_at=ACCEPTED_AT,
        block=BLOCK,
        directory=directory,
        published=ROOT / HISTORY_DIR,
        revisions=directory / "revisions.json",
    )


async def assert_a_revised_candle_without_its_flag_stops_everything(directory: Path) -> None:
    """La fila 4 380 cerrada es una vela revisada en las siete series: sin el flag, la extensión se
    detiene nombrándolas todas, con el valor guardado y el devuelto, y no escribe nada."""
    fake = exchange()
    with pytest.raises(RevisedCandlesError) as caught:
        await extend(fake, directory, accept=())

    revised = {item.symbol: item for item in caught.value.revised}
    assert set(revised) == set(DEFAULT_SYMBOLS)
    for symbol, item in revised.items():
        published = read_ohlcv_csv(ROOT / history_path(symbol, "4h", HISTORY_DIR))[-1]
        assert item.opens == FORMING
        assert item.stored == tuple(published[1:])
        assert item.returned == tuple(fake.rows[symbol][PUBLISHED_ROWS - 1][1:])
    assert not directory.exists() or list(directory.iterdir()) == []


@pytest.mark.asyncio
async def test_a_revised_candle_without_its_flag_stops_everything(tmp_path: Path) -> None:
    await assert_a_revised_candle_without_its_flag_stops_everything(tmp_path / "series")


@pytest.mark.asyncio
async def test_mutation_a_revision_accepted_without_its_flag_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mutant = mutated(extend_history, "if key in wanted:", "if True:", module)
    monkeypatch.setattr(module, "extend_history", mutant)
    with pytest.raises((AssertionError, pytest.fail.Exception)):
        await assert_a_revised_candle_without_its_flag_stops_everything(tmp_path / "series")


@pytest.mark.asyncio
async def test_accepting_the_revision_keeps_the_stored_rows_byte_for_byte(tmp_path: Path) -> None:
    fake = exchange()
    extension = await extend(fake, tmp_path)

    assert {item.symbol for item in extension.accepted} == set(DEFAULT_SYMBOLS)
    for symbol in DEFAULT_SYMBOLS:
        published = ROOT / history_path(symbol, "4h", HISTORY_DIR)
        written = history_path(symbol, "4h", tmp_path)
        content = written.read_bytes()
        assert content.startswith(published.read_bytes())
        rows = read_ohlcv_csv(written)
        assert rows[:PUBLISHED_ROWS] == read_ohlcv_csv(published)
        assert rows[PUBLISHED_ROWS:] == fake.rows[symbol][PUBLISHED_ROWS:]
        assert len(rows) == PUBLISHED_ROWS + 6
        assert b"\r" not in content
    ledger = read_revisions(tmp_path / "revisions.json")
    assert ledger.accepted == extension.accepted
    assert all(item.accepted_at == ACCEPTED_AT and item.block == BLOCK for item in ledger.accepted)
    assert {start for _, start, _ in fake.asked} == {SERIES_START}
    assert {end for _, _, end in fake.asked} == {UNTIL}


async def assert_an_accepted_revision_needs_no_flag_again(directory: Path) -> None:
    await extend(exchange(), directory)
    ledger = (directory / "revisions.json").read_bytes()
    later = UNTIL + timedelta(days=1)

    again = await extend(exchange(later), directory, later, accept=())

    assert again.accepted == ()
    assert (directory / "revisions.json").read_bytes() == ledger


@pytest.mark.asyncio
async def test_an_accepted_revision_is_accepted_again_without_the_flag(tmp_path: Path) -> None:
    await assert_an_accepted_revision_needs_no_flag_again(tmp_path)


@pytest.mark.asyncio
async def test_mutation_an_extension_that_forgets_the_accepted_revisions_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mutant = mutated(extend_history, "previous = known.get(key)", "previous = None", module)
    monkeypatch.setattr(module, "extend_history", mutant)
    with pytest.raises(RevisedCandlesError):
        await assert_an_accepted_revision_needs_no_flag_again(tmp_path)


@pytest.mark.asyncio
async def test_an_accepted_candle_revised_again_stops_the_extension(tmp_path: Path) -> None:
    await extend(exchange(), tmp_path)
    before = history_path("BTC/USDT", "4h", tmp_path).read_bytes()
    fake = exchange()
    fake.rows["BTC/USDT"][PUBLISHED_ROWS - 1][5] += 1.0

    with pytest.raises(RevisedCandlesError) as caught:
        await extend(fake, tmp_path, accept=())
    assert [item.symbol for item in caught.value.revised] == ["BTC/USDT"]
    assert history_path("BTC/USDT", "4h", tmp_path).read_bytes() == before


@pytest.mark.asyncio
async def test_an_old_candle_revised_or_gone_is_a_revision_too(tmp_path: Path) -> None:
    fake = exchange()
    fake.rows["ETH/USDT"][1000][4] += 0.01
    del fake.rows["SOL/USDT"][2000]

    with pytest.raises(RevisedCandlesError) as caught:
        await extend(fake, tmp_path)
    revised = {(item.symbol, item.opens): item for item in caught.value.revised}
    assert set(revised) == {
        ("ETH/USDT", datetime.fromtimestamp(fake.rows["ETH/USDT"][1000][0] / 1000, UTC)),
        (
            "SOL/USDT",
            datetime.fromtimestamp(fake.rows["SOL/USDT"][2000][0] / 1000, UTC) - FOUR_HOURS,
        ),
    }
    assert [item.returned for item in revised.values() if item.symbol == "SOL/USDT"] == [None]


@pytest.mark.asyncio
async def test_a_flag_that_names_a_candle_that_is_not_revised_is_refused(tmp_path: Path) -> None:
    stray = CandleRef("ETH/USDT", datetime(2025, 1, 1, tzinfo=UTC))
    with pytest.raises(ForwardDataError, match="no devuelve distintas: ETH/USDT 2025-01-01"):
        await extend(exchange(), tmp_path, accept=(*FORMING_REFS, stray))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_a_missing_new_candle_stops_the_extension_and_writes_nothing(tmp_path: Path) -> None:
    fake = exchange()
    gone = fake.rows["SOL/USDT"].pop(PUBLISHED_ROWS + 3)
    last = fake.rows["ADA/USDT"].pop()

    with pytest.raises(MissingCandlesError) as caught:
        await extend(fake, tmp_path)
    assert {(item.symbol, ms(item.opens)) for item in caught.value.missing} == {
        ("SOL/USDT", gone[0]),
        ("ADA/USDT", last[0]),
    }
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_extending_again_verifies_everything_and_only_appends(tmp_path: Path) -> None:
    first = await extend(exchange(), tmp_path)
    before = {
        symbol: history_path(symbol, "4h", tmp_path).read_bytes() for symbol in DEFAULT_SYMBOLS
    }

    again = await extend(exchange(), tmp_path, accept=())
    assert [item.sha256 for item in again.files] == [item.sha256 for item in first.files]

    later = UNTIL + timedelta(days=1)
    await extend(exchange(later), tmp_path, later, accept=())
    for symbol in DEFAULT_SYMBOLS:
        content = history_path(symbol, "4h", tmp_path).read_bytes()
        assert content.startswith(before[symbol])
        assert len(read_ohlcv_csv(history_path(symbol, "4h", tmp_path))) == PUBLISHED_ROWS + 12

    # Un lote de un bloque anterior que corre después no recorta: verifica hasta donde llega.
    fake = exchange(later)
    await extend(fake, tmp_path, UNTIL, accept=())
    assert {end for _, _, end in fake.asked} == {later}


@pytest.mark.asyncio
async def test_an_edited_extended_file_or_published_file_is_refused(tmp_path: Path) -> None:
    await extend(exchange(), tmp_path)
    target = history_path("XRP/USDT", "4h", tmp_path)
    target.write_bytes(target.read_bytes().replace(b"1723766400000,", b"1723766400001,", 1))
    with pytest.raises(ForwardDataError, match="no empieza por los bytes publicados"):
        await extend(exchange(), tmp_path, accept=())

    published = tmp_path / "publicado"
    shutil.copytree(ROOT / HISTORY_DIR, published)
    source = history_path("DOGE/USDT", "4h", published)
    source.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ForwardDataError, match="el histórico publicado no se toca"):
        await extend_history(
            UNTIL, exchange(), now=UNTIL, directory=tmp_path / "otro", published=published
        )


def closes_written(directory: Path) -> list[datetime]:
    """El cierre de cada vela que la extensión dejó escrita en `directory`."""
    closes: list[datetime] = []
    for symbol in DEFAULT_SYMBOLS:
        path = history_path(symbol, "4h", directory)
        if path.is_file():
            closes.extend(
                datetime.fromtimestamp(row[0] / 1000, UTC) + FOUR_HOURS
                for row in read_ohlcv_csv(path)
            )
    return closes


async def assert_only_closed_candles_are_written(directory: Path) -> None:
    """Con el reloj inyectado, la extensión no escribe ninguna vela que cierre después de `now`, y
    la última que escribe es la que cierra en `until`.

    Una hora antes de `UNTIL`, la vela que cierra en `UNTIL` está en formación y el exchange
    simulado la devuelve, como el real devolvió la fila 4 380. Extender hasta ella se niega antes de
    descargar y no escribe nada. En el borde exacto, con el reloj en `UNTIL`, esa vela ya cerró y es
    la última escrita en las siete series.
    """
    forming_at = UNTIL - timedelta(hours=1)
    fake = exchange()
    refusal = ""
    try:
        await extend(fake, directory, UNTIL, now=forming_at)
    except ForwardDataError as error:
        refusal = str(error)
    late = [close for close in closes_written(directory) if close > forming_at]
    assert late == [], f"escribió {len(late)} velas que cierran después de {forming_at}"
    assert "sigue en formación" in refusal
    assert fake.asked == [], "no descarga hasta una vela que no cerró"

    try:
        extension = await extend(exchange(), directory, UNTIL, now=UNTIL)
    except ForwardDataError as error:
        raise AssertionError(f"en el borde la vela ya cerró: {error}") from error
    assert max(closes_written(directory)) == UNTIL
    for symbol in DEFAULT_SYMBOLS:
        rows = read_ohlcv_csv(history_path(symbol, "4h", directory))
        assert datetime.fromtimestamp(rows[-1][0] / 1000, UTC) + FOUR_HOURS == UNTIL
    assert {item.last + FOUR_HOURS for item in extension.files} == {UNTIL}


@pytest.mark.asyncio
async def test_the_extension_writes_closed_candles_only_and_ends_at_until(tmp_path: Path) -> None:
    await assert_only_closed_candles_are_written(tmp_path / "forward")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "new",
    [
        pytest.param("if until - timedelta(hours=4) > now:", id="juzga por la apertura"),
        pytest.param("if False:", id="sin comprobar el reloj"),
        pytest.param("if until >= now:", id="el borde exacto en formación"),
    ],
)
async def test_mutation_admitting_the_forming_candle_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, new: str
) -> None:
    """Las dos primeras admiten la vela en formación, como se admitió la fila 4 380; la tercera
    niega la que cierra justo en `now`, que ya cerró."""
    mutant = mutated(module.extend_history, "if until > now:", new, module)
    monkeypatch.setattr(module, "extend_history", mutant)
    with pytest.raises(AssertionError):
        await assert_only_closed_candles_are_written(tmp_path / "forward")


@pytest.mark.asyncio
async def test_mutation_rewriting_the_stored_rows_with_write_ohlcv_csv_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """La extensión que reescribe el archivo entero con `write_ohlcv_csv` cambia los bytes de lo
    publicado: la comprobación del prefijo la detiene antes de escribir nada que valga."""
    monkeypatch.setattr(module, "write_ohlcv_csv", write_ohlcv_csv, raising=False)
    mutant = mutated(
        extend_history,
        "content = stored_bytes + ohlcv_lines(appended)",
        "write_ohlcv_csv(target, [*stored, *appended])\n        content = target.read_bytes()",
        module,
    )
    with pytest.raises(ForwardDataError, match="no empieza por la publicada"):
        await mutant(
            UNTIL,
            exchange(),
            now=UNTIL,
            accept=FORMING_REFS,
            accepted_at=ACCEPTED_AT,
            block=BLOCK,
            directory=tmp_path,
            published=ROOT / HISTORY_DIR,
            revisions=tmp_path / "revisions.json",
        )


# ───────────────────────────────────────── El funding ─────────────────────────────────────────────


def settlements(symbol: str, until: datetime) -> bytes:
    """Lo publicado y liquidaciones nuevas cada 8 h hasta `until`, con su jitter de milisegundos."""
    content = (ROOT / funding_path(symbol)).read_bytes()
    moment = datetime(2026, 8, 16, 8, tzinfo=UTC)
    lines = []
    while moment <= until:
        lines.append(f"{int(moment.timestamp() * 1000) + 13},{1e-05!r}\n")
        moment += timedelta(hours=8)
    return content + "".join(lines).encode()


class FakeFunding:
    def __init__(self, until: datetime) -> None:
        self.series = {symbol: settlements(symbol, until) for symbol in DEFAULT_SYMBOLS}
        self.asked: list[tuple[datetime, datetime]] = []

    async def __call__(self, start: datetime, end: datetime) -> dict[str, bytes]:
        self.asked.append((start, end))
        return dict(self.series)


@pytest.mark.asyncio
async def test_the_funding_grows_only_on_the_right_byte_for_byte(tmp_path: Path) -> None:
    until = datetime(2026, 8, 18, tzinfo=UTC)
    fake = FakeFunding(until)
    files = await extend_funding(until, fake, directory=tmp_path, published=ROOT / FUNDING_DIR)

    for item in files:
        content = funding_path(item.symbol, tmp_path).read_bytes()
        assert content.startswith((ROOT / funding_path(item.symbol)).read_bytes())
        assert item.rows == 2191 + 6
        assert read_funding_csv(funding_path(item.symbol, tmp_path)).times[-1] == ms(
            datetime(2026, 8, 18, tzinfo=UTC)
        )
    assert fake.asked == [(SERIES_START, until)]


@pytest.mark.asyncio
async def test_a_revised_or_jittered_settlement_stops_the_funding(tmp_path: Path) -> None:
    until = datetime(2026, 8, 18, tzinfo=UTC)
    fake = FakeFunding(until)
    lines = fake.series["BNB/USDT"].split(b"\n")
    stamp, rate = lines[2].split(b",")
    lines[2] = stamp + b"," + repr(float(rate) + 1e-06).encode()
    fake.series["BNB/USDT"] = b"\n".join(lines)
    with pytest.raises(ForwardDataError, match=r"BNB/USDT: .*línea 3"):
        await extend_funding(until, fake, directory=tmp_path, published=ROOT / FUNDING_DIR)

    await extend_funding(
        until, FakeFunding(until), directory=tmp_path, published=ROOT / FUNDING_DIR
    )
    later = FakeFunding(until + timedelta(days=1))
    old = f"{int(datetime(2026, 8, 17, 8, tzinfo=UTC).timestamp() * 1000) + 13},"
    new = f"{int(datetime(2026, 8, 17, 8, tzinfo=UTC).timestamp() * 1000) + 21},"
    later.series["BTC/USDT"] = later.series["BTC/USDT"].replace(old.encode(), new.encode(), 1)
    with pytest.raises(ForwardDataError, match="BTC/USDT: el funding descargado no repite"):
        await extend_funding(
            until + timedelta(days=1), later, directory=tmp_path, published=ROOT / FUNDING_DIR
        )


@pytest.mark.asyncio
async def test_a_funding_already_longer_is_verified_and_not_cut(tmp_path: Path) -> None:
    later = datetime(2026, 8, 20, tzinfo=UTC)
    await extend_funding(
        later, FakeFunding(later), directory=tmp_path, published=ROOT / FUNDING_DIR
    )
    fake = FakeFunding(later)
    await extend_funding(
        datetime(2026, 8, 18, tzinfo=UTC), fake, directory=tmp_path, published=ROOT / FUNDING_DIR
    )
    assert fake.asked[0][1] > later
    assert read_funding_csv(funding_path("ETH/USDT", tmp_path)).times[-1] == ms(later)


# ─────────────────────────────────────── La población ─────────────────────────────────────────────

DAILY = timedelta(days=1)
DAY_ZERO = datetime(2025, 1, 2, tzinfo=UTC)
HORIZON = 3
WINDOW = 60


def daily_series(phase: float, count: int = 400) -> list[list[float]]:
    closes = [100.0 + 8.0 * math.sin(index / 7.0 + phase) + 0.05 * index for index in range(count)]
    volumes = [1000.0 + 200.0 * math.sin(index / 5.0 + phase) for index in range(count)]
    return raw_ohlcv(closes, start=DAY_ZERO, step=DAILY, volumes=volumes)


def daily_histories() -> dict[str, list[list[float]]]:
    return {symbol: daily_series(float(position)) for position, symbol in enumerate(("S0", "S1"))}


def daily_manifest(histories: Mapping[str, list[list[float]]]) -> SelectionManifest:
    return select_activations(
        histories,
        timeframe="1d",
        target=20,
        strata=3,
        seed=7,
        horizon=HORIZON,
        candle_limit=WINDOW,
        config=GATE,
        preset=PRESET,
        now=DAY_ZERO,
    )


def covered_starts(histories: Mapping[str, list[list[float]]]) -> list[datetime]:
    """Los bloques cuyo horizonte cubre la serie, desde el primero que la toca."""
    last = min(rows[-1][0] for rows in histories.values())
    covered = datetime.fromtimestamp(last / 1000, UTC) + DAILY
    starts = []
    moment = block_start_of(DAY_ZERO)
    while moment + timedelta(days=7) + HORIZON * DAILY <= covered:
        starts.append(moment)
        moment += timedelta(days=7)
    return starts


def population(histories: Mapping[str, list[list[float]]]) -> module.StudyPopulation:
    return module.study_population(
        histories,
        covered_starts(histories),
        timeframe="1d",
        horizon=HORIZON,
        candle_limit=WINDOW,
        config=GATE,
        preset=PRESET,
    )


def assert_the_blocks_are_the_pool_of_forward_plan() -> None:
    """Bloque a bloque, las entradas son las del pool de `forward_plan`, con sus disparadores y su
    ventana."""
    histories = daily_histories()
    pool = pool_records(daily_manifest(histories), histories, preset=PRESET, config=GATE)
    found = population(histories)

    assert len(found.blocks) > 40
    for start, entries in found.blocks.items():
        expected = {
            (bar.symbol, bar.candle) for bar in pool.bars if block_start_of(bar.at) == start
        }
        assert {(entry.symbol, entry.index) for entry in entries} == expected
        for entry in entries:
            assert entry.triggers == pool.confirmed[(entry.symbol, entry.index)]
            assert entry.candles_digest == window_digest(
                histories[entry.symbol], entry.index, WINDOW
            )
            assert entry.stratum == 0
    assert sum(len(entries) for entries in found.blocks.values()) == sum(
        1 for bar in pool.bars if block_start_of(bar.at) in found.blocks
    )


def test_the_population_of_each_block_is_the_pool_of_forward_plan() -> None:
    assert_the_blocks_are_the_pool_of_forward_plan()


def refuse_one(monkeypatch: pytest.MonkeyPatch) -> tuple[str, int]:
    """Una activación confirmada de un bloque interior que, desde ahora, el gate no confirma.

    En `design_effect` y aquí a la vez, como `test_forward_plan` hace con su escenario: en la serie
    diaria la ventana del replay confirma todas, y sin una que no confirme la mutación que se salta
    la confirmación no tendría nada que añadir.
    """
    histories = daily_histories()
    inside = covered_starts(histories)[1:]
    chosen = next(
        (entry.symbol, entry.index)
        for start, entries in population(histories).blocks.items()
        if start in inside
        for entry in entries
    )
    original = confirm_activation

    def refusing(rows: list[list[float]], index: int, *rest: object) -> tuple[str, ...] | None:
        if rows == histories[chosen[0]] and index == chosen[1]:
            return None
        return original(rows, index, *rest)  # type: ignore[arg-type]

    monkeypatch.setattr(module, "confirm_activation", refusing)
    monkeypatch.setattr(design_module, "confirm_activation", refusing)
    return chosen


def test_the_scenario_has_what_the_mutations_need(monkeypatch: pytest.MonkeyPatch) -> None:
    """Las mutaciones de abajo solo se notan si, dentro de los bloques cubiertos, hay una activación
    que el gate no confirma y velas confirmadas que abren en un bloque y cierran en el siguiente."""
    histories = daily_histories()
    inside = set(covered_starts(histories)[1:])
    boundary = [
        entry
        for start, entries in population(histories).blocks.items()
        if start in inside
        for entry in entries
        if block_start_of(entry.at) != start
    ]
    assert boundary

    chosen = refuse_one(monkeypatch)
    entries = {
        (entry.symbol, entry.index)
        for block in population(histories).blocks.values()
        for entry in block
    }
    assert chosen not in entries
    assert_the_blocks_are_the_pool_of_forward_plan()


def test_mutation_a_block_assigned_by_the_opening_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mutant = mutated(
        study_population,
        "block_of(bar.at + period, CLUSTER_DAYS)",
        "block_of(bar.at, CLUSTER_DAYS)",
        module,
    )
    monkeypatch.setattr(module, "study_population", mutant)
    with pytest.raises(AssertionError):
        assert_the_blocks_are_the_pool_of_forward_plan()


def test_mutation_a_population_without_confirmation_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refuse_one(monkeypatch)
    mutant = mutated(
        study_population,
        "if triggers is None:\n                continue",
        "if triggers is None:\n                triggers = bar.triggers",
        module,
    )
    monkeypatch.setattr(module, "study_population", mutant)
    with pytest.raises(AssertionError):
        assert_the_blocks_are_the_pool_of_forward_plan()


def test_a_block_whose_horizon_is_not_covered_is_refused() -> None:
    histories = daily_histories()
    last = covered_starts(histories)[-1] + timedelta(days=7)
    with pytest.raises(ForwardDataError, match="necesita hasta"):
        study_population(
            histories, [last], timeframe="1d", horizon=HORIZON, config=GATE, preset=PRESET
        )
    with pytest.raises(ForwardDataError, match="no es el inicio de un bloque"):
        study_population(histories, [last + DAILY], timeframe="1d", horizon=HORIZON)


# ─────────────────────────────────────── El manifiesto ────────────────────────────────────────────


def test_the_block_manifest_is_its_whole_population_and_reads_through_the_ablation_doors() -> None:
    histories = daily_histories()
    found = population(histories)
    start = next(moment for moment, entries in found.blocks.items() if len(entries) > 1)
    created = datetime(2026, 10, 30, tzinfo=UTC)

    manifest = block_manifest(
        histories,
        found,
        start,
        created,
        timeframe="1d",
        horizon=HORIZON,
        candle_limit=WINDOW,
        preset=PRESET,
    )

    assert manifest is not None
    assert manifest.entries == found.blocks[start]
    assert manifest.target == len(manifest.entries)
    assert manifest.strata == 1
    assert manifest.seed == DEFAULT_SEED
    assert manifest.warmup_bars == PRESET.min_bars
    assert manifest.created_at == created
    assert [ref.symbol for ref in manifest.series] == ["S0", "S1"]
    assert [ref.activations for ref in manifest.series] == [
        found.activations["S0"],
        found.activations["S1"],
    ]
    verify_histories(manifest, histories)
    assert batch_spend_cap(manifest) == spend_cap(len(manifest.entries))


def test_an_empty_block_has_no_manifest() -> None:
    histories = daily_histories()
    found = population(histories)
    start = next(iter(found.blocks))
    emptied = module.StudyPopulation(
        blocks={**found.blocks, start: ()}, activations=found.activations
    )
    assert block_manifest(histories, emptied, start, DAY_ZERO) is None


def test_the_production_parameters_are_the_ones_of_the_ablation_manifest() -> None:
    """Lo que el lote compara antes de llamar: con las constantes de hoy, los de la etapa 1."""
    reference = load_manifest(ROOT / "data" / "ablation_selection.json")
    assert reference.candle_limit == DEFAULT_CANDLE_LIMIT
    assert reference.horizon == DEFAULT_HORIZON
    assert reference.seed == DEFAULT_SEED
    assert {ref.symbol for ref in reference.series} == set(DEFAULT_SYMBOLS)


def test_writing_the_manifest_again_keeps_it_and_another_population_is_refused(
    tmp_path: Path,
) -> None:
    histories = daily_histories()
    found = population(histories)
    start = next(moment for moment, entries in found.blocks.items() if len(entries) > 1)

    def manifest_at(created: datetime) -> SelectionManifest:
        made = block_manifest(
            histories,
            found,
            start,
            created,
            timeframe="1d",
            horizon=HORIZON,
            candle_limit=WINDOW,
            preset=PRESET,
        )
        assert made is not None
        return made

    path = manifest_path(start, tmp_path)
    first = write_block_manifest(manifest_at(DAY_ZERO), path)
    written = path.read_bytes()

    again = write_block_manifest(manifest_at(DAY_ZERO + timedelta(days=30)), path)
    assert again == first
    assert path.read_bytes() == written
    assert path.name == f"{start:%Y-%m-%d}.json"

    other = manifest_at(DAY_ZERO).model_copy(update={"entries": first.entries[1:]})
    with pytest.raises(ForwardDataError, match="no es la suya"):
        write_block_manifest(other, path)


def test_the_extended_files_are_the_ones_the_last_batch_recorded() -> None:
    """`sha256sum` sobre las series extendidas da lo que dice el `forward.json` del último lote.

    Antes del primer lote no hay ni lotes ni series extendidas: la prueba exige eso mismo, y no
    pasa en vacío si aparece una serie sin el lote que la escribió.
    """
    from crypto_agents.forward import FORWARD_META_FILE, BatchMeta

    records = sorted((ROOT / module.BATCH_DIR).glob(f"*/*/{FORWARD_META_FILE}"))
    if not records:
        assert not (ROOT / module.FORWARD_HISTORY_DIR).exists()
        assert not (ROOT / module.FORWARD_FUNDING_DIR).exists()
        return
    last = BatchMeta.model_validate_json(records[-1].read_text("utf-8"))
    assert last.history
    assert last.funding
    for item in (*last.history, *last.funding):
        assert file_sha256(ROOT / item.path) == item.sha256, item.path
