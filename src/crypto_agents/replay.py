"""Replay del grafo sobre una ventana histórica de velas.

Convierte el journal en algo evaluable: se recorre un histórico vela a vela, se
ejecuta el grafo entero en cada cierre y se resuelve cada llamada desde la caché
cuando el digest del prompt coincide.

Dos garantías, y las dos son de construcción, no de intención:

- **No se mira adelante.** El cliente histórico entrega la ventana que termina en
  la vela evaluada *más la siguiente, todavía en formación*, exactamente como la
  entregaría el exchange. Que la vela en formación se descarte es trabajo de
  `drop_forming_candle`, y el replay la incluye a propósito para que ese descarte
  ocurra de verdad en cada iteración en vez de que el arnés se la esconda.
- **No se gasta cuota por accidente.** El backend por defecto se niega a hablar
  con ningún proveedor: un hueco de caché aborta la corrida nombrando el prompt
  que falta. Rellenar la caché sobre un histórico nuevo es un modo aparte y
  explícito, porque un replay de 500 velas son 3000 llamadas.

Determinismo. Cuatro cosas cambian entre dos corridas si no se atan:

1. `run_id`, que aquí es un UUID5 del instante evaluado en vez de un UUID4.
2. El reloj, que se ata al instante evaluado, así que los `at` dejan de depender
   del reloj real.
3. `latency_ms`, que solo es reproducible en un acierto de caché, donde vale
   cero. Un replay con caché completa es determinista incluido el tiempo; uno con
   caché fría no lo es nunca.
4. El orden de `calls`, que el abanico paralelo escribe por orden de resolución.
   Con la caché caliente no hay punto de espera y el orden acaba siendo estable,
   así que esto no afecta al criterio de dos corridas calientes; en cuanto las
   llamadas tardan de verdad —una pasada de relleno, o producción— los tres nodos
   técnicos terminan en cualquiera de los seis órdenes posibles. `run_digest()`
   canonicaliza por eso: para que una corrida caliente y una que no lo estaba
   sigan siendo comparables.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Self
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import Field, model_validator

from crypto_agents.cache import ReadOnlyResponseCache
from crypto_agents.journal import build_record
from crypto_agents.llm import BackendNotCalledError, ModelRouter
from crypto_agents.market import MarketDataError, timeframe_to_timedelta, to_dataframe
from crypto_agents.state import Backend, FrozenModel, TradingState

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from datetime import datetime

    from langgraph.graph.state import CompiledStateGraph

    from crypto_agents.cache import ResponseCache
    from crypto_agents.context import AgentContext
    from crypto_agents.journal import EvaluationRecord
    from crypto_agents.llm import ChatBackend
    from crypto_agents.quota import Clock, QuotaLedger
    from crypto_agents.selection import PlannedEvaluation
    from crypto_agents.settings import ModelChoice, Settings
    from crypto_agents.state import LLMOutput

__all__ = [
    "REPLAY_NAMESPACE",
    "RUN_DIGEST_VERSION",
    "CacheOnlyBackend",
    "HistoricalMarketClient",
    "ReplayCacheMissError",
    "ReplayContextFactory",
    "ReplaySettings",
    "replay",
    "replay_router",
    "replay_run_id",
    "replay_selection",
    "run_digest",
]

REPLAY_NAMESPACE = uuid5(NAMESPACE_URL, "https://crypto-agents/replay")
"""Raíz de los identificadores de replay. Fija, para que los ids no cambien nunca."""


RUN_DIGEST_VERSION = b"replay-v2"
"""Etiqueta que entra en `run_digest()`. Cambia cuando cambia la forma de lo que se hashea.

v2: el fallo de un `LLMCall` pasó de un objeto anidado a `failure_kind` y
`failure_message`, así que el JSON de cada llamada es otro aunque no haya fallado.
Sin subir la etiqueta, un digest guardado antes y uno calculado ahora diferirían
igual, pero nada diría que es por la forma y no por las decisiones.
"""


class ReplayCacheMissError(BackendNotCalledError):
    """El replay pidió algo que no está en la caché y no tiene permiso para llamar.

    Hereda de `BackendNotCalledError` para que el router la deje pasar entera. Si
    saliera envuelta en `ModelCallError`, un hueco de caché se leería como un
    fallo del proveedor y el mensaje que nombra modelo, esquema y prompt —lo
    único que dice qué hay que rellenar— quedaría enterrado.
    """

    def __init__(self, model: str, schema: str, prompt: str) -> None:
        self.model = model
        self.schema = schema
        self.prompt = prompt
        super().__init__(
            f"falta en caché: modelo {model}, esquema {schema}, prompt de {len(prompt)} caracteres"
        )


class CacheOnlyBackend:
    """Backend que nunca llama a un proveedor.

    Es el valor por defecto del replay. Un hueco de caché tiene que doler: la
    alternativa es que una reejecución sobre un histórico largo se convierta en
    miles de llamadas de pago sin que nadie lo haya pedido.
    """

    async def complete(self, choice: ModelChoice, prompt: str, schema: type[LLMOutput]) -> str:
        """Siempre falla, nombrando lo que se pidió."""
        raise ReplayCacheMissError(choice.model, schema.__name__, prompt)


def replay_router(
    settings: Settings,
    ledger: QuotaLedger,
    cache: ResponseCache,
    clock: Clock,
    fill_with: Mapping[Backend, ChatBackend] | None = None,
) -> ModelRouter:
    """Router para un replay: solo caché salvo que se pida explícitamente lo contrario.

    Sin `fill_with` ningún backend puede hablar con un proveedor y la caché se lee
    sin modificarla, así que reejecutar un histórico no puede costar dinero ni
    cambiar lo que se reejecuta. `fill_with` es la primera pasada sobre un histórico
    nuevo, la que llena la caché.

    Lo que el replay reproduce es todo intento que produjo contenido, válido o no:
    el inválido se revalida, da el mismo error, el mismo prompt de reintento y el
    mismo digest, y la conversación entera sale de la caché. Lo que no reproduce es
    lo que nunca tuvo contenido que guardar —un rechazo o un plazo vencido del
    proveedor— ni una evaluación que corrió degradada al respaldo: ahí falta la
    entrada, y falla con `ReplayCacheMissError` nombrándola.
    """
    backends: dict[Backend, ChatBackend] = dict(fill_with) if fill_with is not None else {}
    for backend in Backend:
        backends.setdefault(backend, CacheOnlyBackend())
    # Sin permiso para llamar tampoco hay permiso para tocar la caché: el router
    # descarta las entradas que ya no validan, y un replay que borra lo que lee
    # deja de poder repetirse sobre lo mismo.
    store = cache if fill_with is not None else ReadOnlyResponseCache(cache)
    return ModelRouter(settings, ledger, backends, clock, store)


class HistoricalMarketClient:
    """Ventana de un histórico ya descargado. No toca la red.

    Devuelve las velas hasta `cursor` incluida más la siguiente, que es la que el
    exchange entregaría todavía en formación.
    """

    def __init__(self, rows: Sequence[Sequence[float]], cursor: int, limit: int) -> None:
        self._rows = rows
        self._cursor = cursor
        self._limit = limit
        self.calls: list[tuple[str, str, int]] = []

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> list[list[float]]:
        """Ventana que termina en el cursor, con la vela en formación al final."""
        self.calls.append((symbol, timeframe, limit))
        end = min(self._cursor + 2, len(self._rows))
        start = max(0, end - min(limit, self._limit))
        return [list(row) for row in self._rows[start:end]]


class ReplaySettings(FrozenModel):
    """Qué trozo del histórico se recorre y con qué ventana."""

    symbol: str = Field(min_length=1)
    timeframe: str = Field(min_length=2)
    warmup_bars: int = Field(ge=2)
    """Velas que se reservan antes de la primera evaluación.

    Por debajo del warm-up del preset, `IndicatorSet` rechaza los NaN residuales y
    todas las evaluaciones iniciales abortarían en `prepare_market_data`.
    """

    candle_limit: int = Field(default=500, ge=2)
    max_evaluations: int | None = Field(default=None, ge=1)
    """Corta el recorrido. Sirve para acotar una prueba, no para producción."""

    @model_validator(mode="after")
    def _timeframe_is_parseable(self) -> Self:
        """Un timeframe inválido debe fallar al configurar, no en la primera vela."""
        try:
            timeframe_to_timedelta(self.timeframe)
        except MarketDataError as error:
            raise ValueError(str(error)) from error
        return self


type ReplayContextFactory = Callable[
    [str, UUID, "ModelRouter", HistoricalMarketClient, "datetime"], AgentContext
]
"""Construye el contexto de una vela: símbolo, id, router, mercado y el instante evaluado."""


def replay_run_id(symbol: str, timeframe: str, moment: datetime) -> UUID:
    """Identificador reproducible de una evaluación del histórico.

    Un `uuid4()` haría que dos replays del mismo histórico difirieran en cada
    registro sin que hubiera cambiado ninguna decisión.
    """
    return uuid5(REPLAY_NAMESPACE, f"{symbol}|{timeframe}|{moment.isoformat()}")


async def replay(
    rows: Sequence[Sequence[float]],
    config: ReplaySettings,
    router: ModelRouter,
    build_context: ReplayContextFactory,
    graph: CompiledStateGraph[TradingState, AgentContext, TradingState, TradingState],
) -> list[EvaluationRecord]:
    """Recorre el histórico evaluando cada vela cerrada a partir del warm-up.

    El `ModelRouter` es uno solo para todo el recorrido, igual que en el runner:
    el presupuesto de una corrida es el de la corrida, no el de cada vela.

    Los registros devueltos son los mismos que el nodo journal escribió por su
    cuenta: como el reloj está atado al instante evaluado, ambos coinciden campo a
    campo en vez de diferir en el `at`.
    """
    frame = to_dataframe(rows)
    period = timeframe_to_timedelta(config.timeframe)
    records: list[EvaluationRecord] = []

    last = len(frame) - 1
    for cursor in range(config.warmup_bars - 1, last):
        if config.max_evaluations is not None and len(records) >= config.max_evaluations:
            break

        moment = frame.index[cursor].to_pydatetime() + period
        run_id = replay_run_id(config.symbol, config.timeframe, moment)
        market = HistoricalMarketClient(rows, cursor, config.candle_limit)
        context = build_context(config.symbol, run_id, router, market, moment)

        raw = await graph.ainvoke(
            TradingState(),
            context=context,
            config={"configurable": {"thread_id": f"replay:{run_id}"}},
        )
        state = TradingState.model_validate(raw)
        records.append(
            build_record(
                state,
                run_id=run_id,
                symbol=config.symbol,
                timeframe=config.timeframe,
                at=moment,
                order=state.order,
            )
        )

    return records


async def replay_selection(
    entries: Sequence[PlannedEvaluation],
    histories: Mapping[str, Sequence[Sequence[float]]],
    candle_limit: int,
    router: ModelRouter,
    build_context: ReplayContextFactory,
    graph: CompiledStateGraph[TradingState, AgentContext, TradingState, TradingState],
) -> list[EvaluationRecord]:
    """Evalúa exactamente las velas que un manifiesto declara, en varios símbolos.

    Es el mismo recorrido que `replay()` sin la parte contigua: cada entrada trae
    su serie y su posición, así que la corrida salta entre símbolos y entre tramos
    del histórico. Eso es lo que permite comparar seis pipelines bajo regímenes de
    mercado distintos en vez de bajo el que tocara al final de la serie.

    No comprueba aquí que la ventana sea la esperada: `verify_histories()` lo hace
    antes, sobre todas las entradas a la vez, para que un histórico revisado se
    detecte antes de la primera llamada y no a mitad de la tercera hora.
    """
    records: list[EvaluationRecord] = []
    for entry in entries:
        rows = histories[entry.symbol]
        period = timeframe_to_timedelta(entry.timeframe)
        moment = entry.at + period
        run_id = replay_run_id(entry.symbol, entry.timeframe, moment)
        market = HistoricalMarketClient(rows, entry.index, candle_limit)
        context = build_context(entry.symbol, run_id, router, market, moment)

        raw = await graph.ainvoke(
            TradingState(),
            context=context,
            config={"configurable": {"thread_id": f"replay:{run_id}"}},
        )
        state = TradingState.model_validate(raw)
        records.append(
            build_record(
                state,
                run_id=run_id,
                symbol=entry.symbol,
                timeframe=entry.timeframe,
                at=moment,
                order=state.order,
            )
        )
    return records


def run_digest(records: Sequence[EvaluationRecord]) -> str:
    """Huella canónica de una corrida completa.

    Ordena las llamadas de cada registro por `(rol, digest de prompt)` antes de
    hashear. El abanico paralelo las escribe en orden de resolución: mientras todo
    sale de la caché no hay espera y el orden es estable, pero en cuanto las
    llamadas tardan de verdad los tres técnicos pueden terminar en cualquier
    orden. Sin canonicalizar, comparar una corrida caliente contra una que no lo
    estaba daría una diferencia que no es una diferencia de decisión.
    """
    digest = hashlib.sha256()
    digest.update(RUN_DIGEST_VERSION)
    for record in records:
        payload = record.model_dump(mode="json")
        payload["calls"] = sorted(
            payload["calls"], key=lambda call: (call["role"], call["prompt_digest"])
        )
        digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()
