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
