# Lectura de la tanda 20261030T051122Z

Escrita a mano después de la corrida. Las cifras salen de `meta.json`, `responses.jsonl` y
`calls.jsonl` de este directorio; `report.md` es la salida del script tal cual.

## Qué corrió

- Siete llamadas entre 05:11:22Z y 05:14:48Z, en la laptop (`zephyrus-G15`). Las siete
  contestaron con HTTP 200 y validaron a la primera.
- Gasto: 0.0916 USD en el extremo alto, de un tope de 0.20 USD. No se negó ninguna llamada y
  ninguna llegó sin tokens. `report.md` lo redondea a «0.1 USD»; la cifra exacta está en
  `meta.json`.
- El commit es `ffc94b0`, con el árbol sucio: el script todavía no estaba versionado. Lo que
  identifica al código que corrió es su sha-256, `efb9adbc…36be`. Es el de
  `scripts/probe_upstream.py` en el commit que añade este directorio; se comprueba con
  `sha256sum`.
- La petición de la variante a es la del estudio. Su envoltura es la registrada, `f0727ca0…6928`,
  y su prompt es el de la primera evaluación de la etapa 1 (`prompt_digest` `f1ec963b…fcea`).

## Qué dijo cada respuesta

**Cabeceras.** Las siete llamadas trajeron las mismas diez, y ninguna más:

- `date`, `content-type`, `transfer-encoding`, `connection` y `content-encoding`;
- `cf-placement: remote-IAD`, `server: cloudflare` y `cf-ray`;
- `x-opencode-log-id`, distinto en cada llamada;
- `x-zen-model`, que repite el id pedido.

Ninguna trajo `x-opencode-upstream-model-id` ni `x-opencode-endpoint-id`, tampoco la de
`minimax-m3`.

**Cuerpo.**

- `model` repite el id pedido.
- No hay clave `system_fingerprint`.
- Las claves del cuerpo llegan en el mismo orden en las siete llamadas.
- El `id` tiene dos formas: `chatcmpl-` seguido de 8 caracteres hexadecimales en las 6 de
  `glm-5.2`, y `chatcmpl-` seguido de 32 en la de `minimax-m3`.

**Orden de los argumentos.** El esquema los envía como `action, confidence, size_fraction,
invalidation_price, rationale`.

- `glm-5.2` los devolvió en orden alfabético en 6 de 6 llamadas: `action, confidence,
  invalidation_price, rationale, size_fraction`.
- `minimax-m3` los devolvió en el orden del esquema.

**Velocidad del decisor.** Son tokens de salida por segundo de pared, con
`completion_tokens / latency_ms`:

| qué | llamadas | hora UTC | tok/s |
| --- | ---: | --- | --- |
| esta tanda, variante a | 3 | 05:11–05:13 | 98.7 a 105.2 |
| esta tanda, variante b | 3 | 05:13–05:15 | 96.6 a 105.6 |
| lote del bloque 1 (`solo.jsonl` de `20261030T000327Z`) | 42 | 00:05–00:26 | 83.6 a 107.6, mediana 99.2 |
| etapa 1 (`data/stage1/solo.jsonl`) | 140 | 10:19–12:35 | 48.4 a 65.2, mediana 55.4 |
| etapa 1, la llamada con este mismo prompt | 1 | 10:19:42 | 56.2 (2 328 tokens en 41.4 s) |

`minimax-m3` contestó a 180.9 tok/s.

## Las cuatro explicaciones

1. **Otro upstream sin esas cabeceras, y más rápido. Se refuerza por la velocidad, no por las
   cabeceras.**
   - Con el mismo prompt, `glm-5.2` contesta hoy a unos 100 tok/s, como en el bloque 1. En la
     etapa 1 contestaba a 55.
   - Pero que falten las cabeceras ya no indica que el upstream sea otro: le faltan también a
     `minimax-m3` (punto 2).
   - Quedan dos huellas que se pueden comparar en llamadas futuras: la forma del `id` y el orden
     alfabético de los argumentos. Ninguna dice qué upstream es.
   - No queda decidido.
2. **Zen dejó de mandarlas para todos los modelos. Se refuerza.**
   - `minimax-m3` trajo las dos cabeceras en su `Ping` del bloque T5 y en 12 de 12 llamadas del
     bloque T6, en los dos casos por `fireworks`. Hoy no trae ninguna.
   - Las otras dos cabeceras de la pasarela que ya se vieron en el bloque T4,
     `x-opencode-log-id` y `x-zen-model`, siguen llegando.
   - Son dos modelos a una hora: decir «todos» es extrapolar.
3. **Algo de nuestra petición decide la ruta o las cabeceras. Se descarta para el User-Agent y la
   sesión.**
   - La variante b, con un User-Agent fijo y una sesión estable, dio las mismas cabeceras, la
     misma velocidad, la misma forma de `id` y el mismo orden de argumentos que la a.
   - La variante a es la petición registrada del estudio, la misma envoltura que en la etapa 1.
4. **La hora del día. Esta tanda no la decide.**
   - Las 05:11Z, como las 00:05Z del bloque 1, caen fuera de la ventana de la etapa 1, que fue
     de 10:19Z a 12:35Z.
   - La segunda tanda, idéntica, está programada para las 10:20Z.
   - Si `glm-5.2` vuelve a unos 55 tok/s, la velocidad depende de la hora. Si sigue cerca de 100,
     cambió con independencia de la hora.
   - Si a esa hora vuelven las cabeceras, el punto 2 cae: las cabeceras dependerían de la hora.

## Lo que no se miró

`var/ablation-cache` guarda el texto crudo de cada respuesta de la etapa 1. El orden de las claves
en esas respuestas diría si el upstream de entonces, `glm-5p3` por `fireworks`, ya devolvía los
argumentos ordenados. No se leyó, porque el punto 5 del bloque excluye esa caché.
