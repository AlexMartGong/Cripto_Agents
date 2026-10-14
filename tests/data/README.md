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

# Journals del sondeo de decisores sin mesas (bloque T6)

`zen_probe/20261006T235701Z/` — `meta.json`, `glm-5.2@solo.jsonl` y `glm-5.2@no_debate.jsonl`: 12
`EvaluationRecord` por journal, uno por activación, con un `LLMCall` cada uno (24 llamadas, todas
válidas al primer intento y con los tres contadores).

Sirven para que una cifra de `tests/test_estimate.py` salga de filas y no de una constante: el
coste por llamada del decisor de `solo` (0.01456 USD) y de `no_debate` (0.02014 USD) se calcula de
estas filas con la tabla de precios, y es lo que `estimate --source decider@solo=…` tiene que dar.

## Procedencia

Copia byte a byte de `var/zen-probe/20261006T235701Z/`, el sondeo real contra OpenCode Zen del
2026-10-06 (23:57:01Z a 00:11:05Z, árbol limpio en `b99aa9a`, `glm-5.2` servido por `fireworks` como
`accounts/fireworks/models/glm-5p3` en las 24 llamadas). No se copiaron `content.jsonl`,
`findings.json` ni `report.md`: `read_run()` solo necesita el `meta.json` y los journals.

Un journal lleva `LLMCall`, no contenido: no hay prompts, ni respuestas, ni claves. `meta.json`
lleva la `base_url` pública de la pasarela (`https://opencode.ai/zen/v1`), sin credenciales.

## Digest

```
349f08fbf806d31757c3313c9ffb49f06ebfe539dbfe3e8ef8f335e3ff9f1064  zen_probe/20261006T235701Z/glm-5.2@solo.jsonl
aa2f11f13bbe60139a0d25282d877bb49d718cfda2e16b8f71d8d1a640ee230a  zen_probe/20261006T235701Z/glm-5.2@no_debate.jsonl
55e766cefe4efab478261b0aa00dac8eb35e833bdff9e86dc743fad51bdeb47d  zen_probe/20261006T235701Z/meta.json
```

Son los sha-256 de los bytes (`sha256sum`), los mismos que tienen los archivos de `var/`. La prueba
comprueba los dos journals antes de leerlos.

# Retornos por evaluación de la etapa 1 (enmienda 4)

`stage1_paired_returns.jsonl` — 140 líneas, una por evaluación de la etapa 1
(`var/ablation/20261009T045033Z`, que no está versionado). Cada una lleva el `run_id`, el símbolo,
el instante evaluado (`at`, el cierre de la vela) y el retorno bruto por evaluación de los seis
brazos con el stop común: `full`, `solo`, `always_buy`, `always_sell`, `random_uniform` y
`rule_trend`.

Es lo que permite una prueba versionada de que los nueve veredictos de la etapa 1 no cambian con
el intervalo de la enmienda 4 (`tests/test_criteria.py`). No es un journal: no lleva decisiones,
llamadas, prompts ni respuestas, solo lo que `criteria` resta y el instante que decide el clúster.
La prueba no vuelve a puntuar nada —eso necesita los journals—: agrupa y juzga estos vectores con
`criteria.compare_returns`, y comprueba que la diferencia media y el error ingenuo de cada par son
los que `docs/ablation.md` ya tiene publicados. Comprueba también el suelo de esa enmienda: el
error usado no baja del ingenuo en ninguno de los nueve pares, y es el agrupado solo en `solo`
contra `rule_trend`.

## Procedencia

Derivado una sola vez, el 2026-10-12, en la laptop, con el código de este bloque sobre el commit
`05bb914` (la puntuación, `criteria.paired_returns` y `outcomes.score_run`, no cambió en él):

```python
import json, sys
from pathlib import Path

from crypto_agents.criteria import paired_returns
from crypto_agents.paired_dispersion import read_stage
from crypto_agents.selection import HISTORY_DIR

ARMS = ("full", "solo", "always_buy", "always_sell", "random_uniform", "rule_trend")

stage = read_stage(Path("var/ablation/20261009T045033Z"), None, HISTORY_DIR, None)
columns, records = {}, None
for arm in ARMS:
    aligned = paired_returns(
        arm, "solo" if arm == "full" else "full", stage.arms, stage.histories, stage.horizon
    )
    assert not isinstance(aligned, str) and aligned.unpaired == 0
    records = records or aligned.arm_records
    assert [r.run_id for r in aligned.arm_records] == [r.run_id for r in records]
    columns[arm] = aligned.arm_returns
with Path("tests/data/stage1_paired_returns.jsonl").open("w", encoding="utf-8") as out:
    for position, record in enumerate(records):
        line = {
            "run_id": str(record.run_id),
            "symbol": record.symbol,
            "at": record.at.isoformat().replace("+00:00", "Z"),
            "returns": {arm: columns[arm][position] for arm in ARMS},
        }
        out.write(json.dumps(line, ensure_ascii=False) + "\n")
```

`read_stage` comprueba antes que la corrida es válida y que el plan es el suyo. El orden de las
líneas es el de `paired_returns`: por `run_id`. Los journals de los que sale:

```
5d7f1a63f3aaf1c8469a572aa3aaff6184e671ff67698a1d3632a15645ada33f  full.jsonl
1180554e2c19aab20d623f2112547aae8f930ad934d3fc17ca65fd5c70cc4b17  solo.jsonl
77fe118e5efb5714384c3d3485baf31a9c80bdbded2c34f2b775c9667fddd1ff  always_buy.jsonl
ba1643fb2ce45114f728418e8894eb46b55f125fe2cef1fe0211e273746bc1d9  always_sell.jsonl
607b738ea2ee764d4ee8433adc0cac748f738761dfcea418f1a6690cd6ff6198  random_uniform.jsonl
3b8d58f9ce1b22699a7f0d92b2dd47e91235c3db9f0ef3b53d3bed215b19ab93  rule_trend.jsonl
```

Son los mismos sha-256 que «Etapa 1 — resultado» cita en `docs/ablation.md`.

## Digest

```
b521801e0545d3965d3f96c0fba32debd92b30988af6d31e3cfa1d034e55841f  stage1_paired_returns.jsonl
```

El sha-256 de los bytes (`sha256sum`). `tests/test_criteria.py` lo comprueba antes de leerlo.

