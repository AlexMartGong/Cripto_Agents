# Respuestas grabadas de perpetuos USDT

`binanceusdm.json` y `bybit.json` — una captura única, pública y de solo lectura, hecha el
`2026-10-02` a las 22:08 hora local (`2026-10-03T04:08Z`) desde el escritorio (sin claves, sin proxy, una petición por cosa, sin reintentos).
Ninguna prueba toca la red: estos archivos son lo que `tests/test_perp_probe.py` reproduce.

## Qué hay dentro

Respuestas **ya interpretadas por ccxt** (`ccxt 4.5.73`), no HTTP crudo. Las pruebas ejercitan el
código de este proyecto, no el analizador de ccxt.

| Clave | Contenido |
| --- | --- |
| `precisionMode` | `4` (`ccxt.TICK_SIZE`) en los dos exchanges |
| `markets` | `market` completo de los 7 símbolos del universo, más dos señuelos reales: `BTC/USDC:USDC` (perpetuo liquidado en USDC) y un futuro con vencimiento de `BTC/USDT:USDT-…` |
| `tickers` | `symbol`, `timestamp`, `datetime`, `last`, `baseVolume`, `quoteVolume` de cada símbolo. Bybit devuelve `timestamp: null` |
| `funding_btc` | `fetch_funding_rate_history("BTC/USDT:USDT", since, limit=200, params={"until": until})` |
| `window` | `since` y `until` de esa petición |

El rango de funding está anclado (`2025-09-01T00:00:00Z` – `2025-09-09T00:00:00Z`) para que repetir
la captura devuelva las mismas filas: 25, porque **ambos extremos son inclusivos** en los dos
exchanges. Binance trae jitter de milisegundos en las marcas (`…00:00:00.001`); Bybit no.

Lo que **no** es real, y se construye en línea en las pruebas rotulado como sintético: series con
intervalo de 4 h o de 1 h, huecos, páginas llenas y respuestas con cabeceras. Ninguno de los dos
exchanges tenía hoy un tramo así en BTC.

## Digests

```
638e36b5fceab8b86ba11ac35c7e1a8868863924fbd0273ac378e89c0f88d3b7  binanceusdm.json
a8e3cacfe710f17fc711a01fd68919f76388e701aaa8e8c8e7e5d9f1a3a90bb2  bybit.json
```

`tests/test_perp_probe.py` los comprueba: editar un fixture a mano haría las pruebas no comparables
con la captura sin que nada lo avise.
