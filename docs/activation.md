# Barrido del gate de activación

**Cuántas evaluaciones rinde una ventana histórica, qué reglas disparan y si 4h da material.**
Cero llamadas a modelo: el barrido no importa `crypto_agents.llm` y un test de arquitectura lo
fija, que es lo que permite repetir la tabla cuantas veces haga falta.

## Cómo se corre

```bash
uv run python -m crypto_agents.activation_sweep            # usa el caché de var/history/
uv run python -m crypto_agents.activation_sweep --refresh  # vuelve a descargar
```

Las velas se cachean en `var/history/` y **no se versionan**: son ~10 MB de salida de ejecución.
Lo versionado es este documento, con el rango y el digest sha-256 de cada serie — que es lo que
permite saber si dos tablas miraron los mismos datos sin cargar los datos.

El barrido llama a `evaluate_activation`, no a una copia de las reglas: una reimplementación
mediría este archivo en vez del gate. Los indicadores se calculan una vez por serie y después se
corta, lo cual es exacto porque `tests/test_lookahead.py` fija que truncar por la derecha no cambia
la barra `i`; sin esa garantía habría que recalcular el preset 153 000 veces.

## Las tres respuestas

**1. 4h es viable, y por bastante.** Dos años dan 3 980 evaluaciones por símbolo tras descontar el
warm-up de 400 velas, de las que el gate abre entre 617 y 736 — **4 740 activaciones en 4h sobre los
siete símbolos**. La ablación no está esperando material; hay tres órdenes de magnitud más del que
puede pagar.

**2. Ninguna de las cuatro reglas está muerta, y los nueve triggers disparan.** Pero el reparto es
muy desigual: `range_breakout` produce el 55% de los triggers y `volatility_jump` el 7%. En 4h,
`volatility_jump` llega a bajar a 20 disparos en dos años para SOL/USDT — sigue viva, pero cualquier
conclusión sobre esa regla en 4h descansa sobre una muestra pequeña.

**3. Cambiar a 1h no sube la tasa, multiplica las velas.** 15.5–18.5% en 4h contra 17.0–18.8% en 1h:
la tasa del gate es prácticamente invariante al timeframe. Lo que 1h aporta es 4.3× más barras
(17 120 evaluaciones por símbolo contra 3 980), no un gate más generoso. Si hace falta más material
la palanca es el timeframe o los símbolos, no aflojar los umbrales.

## Lo que esto dice de la corrida que motivó el barrido

Que el gate no disparara en 4h en ninguno de los siete símbolos en un instante concreto **no era
evidencia de nada**. Con una tasa del ~17%, la probabilidad de que ninguno de los siete abra en un
cierre dado es 0.83⁷ ≈ **27%**: algo más de uno de cada cuatro cierres de vela. Es el
comportamiento esperado de un gate que existe para ser selectivo.

## Simetría direccional

Las reglas no están sesgadas hacia un lado, con una excepción explicable:

| par de triggers | alcista | bajista | sesgo |
| --- | ---: | ---: | --- |
| `ma_cross_bullish` / `_bearish` | 1 443 | 1 446 | ninguno |
| `regime_change_above_trend` / `below_trend` | 2 892 | 2 893 | ninguno |
| `regime_change_trending` / `ranging` | 2 419 | 2 417 | ninguno |
| `range_breakout_up` / `down` | 7 194 | 6 645 | +8% al alza |

Los tres primeros son cruces, y un cruce en un sentido obliga a uno en el otro antes de repetirse:
su simetría es aritmética, no un hallazgo. El +8% de `range_breakout` sí es del mercado — la ventana
2024-2026 subió— y es la única asimetría que hay que recordar al leer resultados de la ablación
sobre estos datos.

## Lo que esto dimensiona

El techo de cuota lo fija el decisor: 880 llamadas por ventana de 5 h. Cada activación que llega a
decidir gasta **una llamada de decisor por brazo**, y la ablación tiene seis brazos. Con eso, una
ventana de cuota cubre unas **146 activaciones barridas por los seis brazos** — y el histórico
ofrece 4 740 solo en 4h. El límite de la ablación es el presupuesto, no los datos.

## Activaciones por símbolo y timeframe

| símbolo | tf | velas | evaluadas | activaciones | tasa | `ma_cross` | `range_breakout` | `volatility_jump` | `regime_change` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC/USDT | 4h | 4380 | 3980 | 672 | 16.9% | 73 | 377 | 28 | 272 |
| ETH/USDT | 4h | 4380 | 3980 | 617 | 15.5% | 81 | 359 | 29 | 227 |
| SOL/USDT | 4h | 4380 | 3980 | 689 | 17.3% | 72 | 396 | 20 | 279 |
| BNB/USDT | 4h | 4380 | 3980 | 691 | 17.4% | 86 | 365 | 47 | 278 |
| XRP/USDT | 4h | 4380 | 3980 | 736 | 18.5% | 77 | 375 | 68 | 305 |
| ADA/USDT | 4h | 4380 | 3980 | 699 | 17.6% | 87 | 406 | 51 | 257 |
| DOGE/USDT | 4h | 4380 | 3980 | 636 | 16.0% | 69 | 345 | 73 | 236 |
| BTC/USDT | 1h | 17520 | 17120 | 3069 | 17.9% | 314 | 1622 | 326 | 1258 |
| ETH/USDT | 1h | 17520 | 17120 | 2917 | 17.0% | 351 | 1431 | 279 | 1269 |
| SOL/USDT | 1h | 17520 | 17120 | 3046 | 17.8% | 324 | 1681 | 149 | 1279 |
| BNB/USDT | 1h | 17520 | 17120 | 3210 | 18.8% | 337 | 1719 | 211 | 1361 |
| XRP/USDT | 1h | 17520 | 17120 | 3022 | 17.7% | 342 | 1583 | 308 | 1204 |
| ADA/USDT | 1h | 17520 | 17120 | 2969 | 17.3% | 338 | 1626 | 184 | 1175 |
| DOGE/USDT | 1h | 17520 | 17120 | 2945 | 17.2% | 338 | 1554 | 207 | 1221 |

## Activaciones por trigger

| trigger | 4h | 1h | total |
| --- | ---: | ---: | ---: |
| `ma_cross_bearish` | 275 | 1171 | 1446 |
| `ma_cross_bullish` | 270 | 1173 | 1443 |
| `range_breakout_down` | 1221 | 5424 | 6645 |
| `range_breakout_up` | 1402 | 5792 | 7194 |
| `regime_change_above_trend` | 482 | 2410 | 2892 |
| `regime_change_below_trend` | 485 | 2408 | 2893 |
| `regime_change_ranging` | 441 | 1976 | 2417 |
| `regime_change_trending` | 446 | 1973 | 2419 |
| `volatility_jump` | 316 | 1664 | 1980 |

Los nueve triggers dispararon al menos una vez.

## Procedencia

Warm-up del preset: 400 velas por serie, descontadas de `evaluadas`.

| símbolo | tf | desde | hasta | velas | digest sha-256 |
| --- | --- | --- | --- | ---: | --- |
| BTC/USDT | 4h | 2024-08-16 | 2026-08-15 | 4380 | `42959301d3e25c7cc9d423d8083cd95a6242bbefa1c807de430011039fc91156` |
| ETH/USDT | 4h | 2024-08-16 | 2026-08-15 | 4380 | `18c3ad835b1cbe7e7b25eca6c3f554cc69cff835b6658438ba174485587f37d3` |
| SOL/USDT | 4h | 2024-08-16 | 2026-08-15 | 4380 | `2566b778d980c31be8647acbbbfcb8c2bbc4dda91c002a3a2481fa8b6bbb4d42` |
| BNB/USDT | 4h | 2024-08-16 | 2026-08-15 | 4380 | `61f45d7695abba80fd94020cacd5454b644b73510221946ce924e40d72a6304d` |
| XRP/USDT | 4h | 2024-08-16 | 2026-08-15 | 4380 | `4d278d8462982ff0582a071a307235cca6c7d8e508e309944789c39d5f2af075` |
| ADA/USDT | 4h | 2024-08-16 | 2026-08-15 | 4380 | `7a37d0f9af95a334383581696ae21ddf9543238d39e568369571a1d68b6847ae` |
| DOGE/USDT | 4h | 2024-08-16 | 2026-08-15 | 4380 | `02bd09d321a0441e95ae5c8277130e59e26b21637f61fc35f95d36361533e5cb` |
| BTC/USDT | 1h | 2024-08-15 | 2026-08-15 | 17520 | `29cf11d915dae3d9f609cacd394fe1a1deb70398d735f945552686b85ea24a1e` |
| ETH/USDT | 1h | 2024-08-15 | 2026-08-15 | 17520 | `48b66e151c0db2654c24744b08ece53f500f54c00d04421be1b5c0e7cd90936f` |
| SOL/USDT | 1h | 2024-08-15 | 2026-08-15 | 17520 | `3e2651936e830048989fa4267887b8912ca389cb071647061dccd1fda39b5b23` |
| BNB/USDT | 1h | 2024-08-15 | 2026-08-15 | 17520 | `f7f341dd989f2b4b4c8121e739b0ee3b87d5231f21de5f45a0ae23fc77bbd835` |
| XRP/USDT | 1h | 2024-08-15 | 2026-08-15 | 17520 | `71a538a61c9fc82d852640c09cf5862aa662536c9e566a5fb5b012c909d86b96` |
| ADA/USDT | 1h | 2024-08-15 | 2026-08-15 | 17520 | `a55c76129bb21d148030f1c4f2265f562198b9f94f45f7995ffb611e0708b1f0` |
| DOGE/USDT | 1h | 2024-08-15 | 2026-08-15 | 17520 | `7430b43b23987629650741a52cae5292a990667b6b5c2b10efac2e346d585346` |
