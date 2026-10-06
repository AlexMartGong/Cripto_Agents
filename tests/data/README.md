# Histórico fijo para el replay

`btcusdt_4h.csv` — 500 velas de 4h de `BTC/USDT` en binance, de `2025-01-01T00:00:00Z` a
`2025-03-25T04:00:00Z`.

Ninguna prueba descarga nada: este archivo es la única entrada de mercado real del repositorio y se
versiona precisamente para que el replay sea reproducible entre máquinas y entre corridas.

## Procedencia

Descargado una sola vez con un rango fijo, no con «las últimas N velas»: `since` anclado hace que
repetir la descarga devuelva exactamente las mismas filas.

```python
import ccxt.async_support as ccxt

exchange = ccxt.binance({"enableRateLimit": True})
since = exchange.parse8601("2025-01-01T00:00:00Z")
rows = await exchange.fetch_ohlcv("BTC/USDT", timeframe="4h", since=since, limit=500)
```

Los valores se escriben con `repr(float(...))` para que el ida y vuelta por CSV no pierda ningún bit
del `float64` original.

## Digest

```
3460640ac8fb6ee9ad903c27d5c534e17b9954707c9d4a528bc7537e5142fae2
```

Es el `candles_digest()` del propio proyecto sobre el DataFrame normalizado.
`tests/test_replay.py` lo comprueba: si alguien edita el CSV, el replay dejaría de ser comparable
con corridas anteriores y el test lo dice en vez de que las métricas cambien en silencio.

# Journal escrito antes del bloque T5

`journal_pre_t5.jsonl` — tres `EvaluationRecord`, uno por línea: una `Decision` con `hold`, una
`Decision` con `buy` y una `Proposal` con `hold`.

En el bloque T5 `dismissed_side`, `dismissal_reason` e `invalidation_price` dejaron de tener valor
por defecto (son requeridos y anulables: ver `Proposal` y `Decision` en `state.py`). Un journal ya
escrito tiene que seguir cargando, y estas tres líneas son lo que lo prueba
(`tests/test_decision_contract.py`).

## Procedencia

No están escritas a mano: las serializó `EvaluationRecord.model_dump_json()` con el código del
commit `6e736aa`, el último antes del cambio, con el árbol limpio. No hay líneas reales que usar en
su lugar: a esa fecha ningún journal de `var/` llevaba una `decision` o una `proposal` no nulas
(los del sondeo de Zen las dejan en `null`, y la primera ablación no dejó journal).

`model_dump_json()` ya escribía los tres campos con `null`, así que las claves están. Una línea a
la que le falten —escrita por otra herramienta, o a mano— no carga con el contrato nuevo.

## Digest

```
30ddc376c57f488d3abc8ee29c03c3d5fbba84d92a39a452269ea0967802aac6  journal_pre_t5.jsonl
```

Es el sha-256 de los bytes del archivo (`sha256sum`). La prueba lo comprueba antes de leerlo, para
que nadie «arregle» las líneas y deje de probar lo que prueban.

# `LLMCall` escrito antes del bloque T6

`llmcall_pre_t6.jsonl` — un `EvaluationRecord` con cinco `LLMCall`: uno válido y vivo con sus tres
contadores, uno inválido de esquema, un acierto de caché, un rechazo del proveedor y uno local.

En el bloque T6 `LLMCall` ganó `upstream_model` y `upstream_endpoint` (quién contestó según las
cabeceras de la pasarela). Una línea ya escrita no los lleva y tiene que seguir cargando, con los
dos en `None`; y `run_digest()` los omite cuando valen `None`, así que el digest de una corrida sin
cabeceras no se mueve y `RUN_DIGEST_VERSION` sigue en `replay-v3`. `tests/test_upstream.py`
comprueba las dos cosas contra este archivo.

## Procedencia

No está escrito a mano: lo serializó `EvaluationRecord.model_dump_json()` con el código del commit
`60b09bc`, el último antes del cambio, con el árbol limpio y antes de tocar `state.py`. Los valores
son verosímiles (los ids, modos y mensajes son los de `var/zen-probe/`), pero el registro es
sintético: ningún journal real reúne los cinco casos en una evaluación.

## Digest

```
a56531255afe9169dec72a29df7a8a4447ad08cea70582f1cd6bfd6108cb98e6  llmcall_pre_t6.jsonl
```

Es el sha-256 de los bytes del archivo (`sha256sum`). El `run_digest()` de ese registro, calculado
con el mismo código de `60b09bc`:

```
e71e27a74d16f85201e03f5caa08aecb7c9d7032b14e4a80ce11c8b7a7cc6910
```
