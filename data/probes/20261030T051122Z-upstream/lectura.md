# Lectura de las tandas 20261030T051122Z y 20261030T102001Z

Escrita a mano después de las corridas. La primera parte se escribió tras la tanda de las 05:11Z.
Lo demás se escribió tras la de las 10:20Z, que también reescribió «Las cuatro explicaciones»; la
versión anterior está en `63a2b94`.

Fuentes:

- Las cifras de cada tanda salen de `meta.json`, `responses.jsonl` y `calls.jsonl` de su
  directorio: este y `../20261030T102001Z-upstream/`. `report.md` es la salida del script tal cual.
- Las de la etapa 1 salen de `data/stage1/solo.jsonl`.
- Las del bloque 1 salen de `data/forward/batches/2026-10-22/20261030T000327Z/solo.jsonl`.
- Las velocidades son tokens de salida por segundo de pared: `completion_tokens / latency_ms`.

## Primera tanda (05:11Z)

### Qué corrió

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

### Qué dijo cada respuesta

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

**Velocidad del decisor.**

| qué | llamadas | hora UTC | tok/s |
| --- | ---: | --- | --- |
| esta tanda, variante a | 3 | 05:11–05:13 | 98.7 a 105.2 |
| esta tanda, variante b | 3 | 05:13–05:15 | 96.6 a 105.6 |
| lote del bloque 1 (`solo.jsonl` de `20261030T000327Z`) | 42 | 00:05–00:26 | 83.6 a 107.6, mediana 99.2 |
| etapa 1 (`data/stage1/solo.jsonl`) | 140 | 10:19–12:35 | 48.4 a 65.2, mediana 55.4 |
| etapa 1, la llamada con este mismo prompt | 1 | 10:19:42 | 56.2 (2 328 tokens en 41.4 s) |

`minimax-m3` contestó a 180.9 tok/s.

## Segunda tanda (10:20Z)

### Qué corrió

- Siete llamadas entre 10:20:01Z y 10:23:21Z, en la laptop. Fue el mismo plan: tres de la
  variante a, tres de la b y una de la c. Las siete contestaron con HTTP 200 y validaron a la
  primera.
- Gasto: 0.0917 USD en el extremo alto, de un tope de 0.20 USD. No se negó ninguna llamada y
  ninguna llegó sin tokens.
- El commit es `63a2b94`, con el árbol limpio. Corrió desde una copia fijada en ese commit,
  lanzada por un temporizador transitorio de usuario. El sha-256 del script es el de la primera
  tanda, `efb9adbc…36be`: las dos tandas corrieron el mismo código.
- La envoltura de la variante a es la registrada, `f0727ca0…6928`, y el prompt es el mismo de la
  primera tanda.

### Qué dijo cada respuesta

Lo mismo que en la primera tanda:

- Las mismas diez cabeceras, y ninguna más. Ni `x-opencode-upstream-model-id` ni
  `x-opencode-endpoint-id`, tampoco en la de `minimax-m3`.
- En el cuerpo, `model` repite el id pedido y no hay `system_fingerprint`. Las claves llegan en
  el mismo orden en las siete.
- El `id` tiene las mismas dos formas: `chatcmpl-` y 8 hexadecimales en las 6 de `glm-5.2`, y
  `chatcmpl-` y 32 en la de `minimax-m3`.
- `glm-5.2` devolvió los argumentos en orden alfabético en 6 de 6. `minimax-m3`, en el del
  esquema.

Lo único que cambió es el sufijo de `cf-ray`, que nombra el centro de datos de Cloudflare que
recibió la petición:

- en la primera tanda fue `IAH` en las 6 de `glm-5.2` y `ATL` en la de `minimax-m3`;
- en la segunda fue `DFW` en la variante a y en la c, e `IAH` en la b.

## La velocidad, con las dos tandas

Todas son llamadas a `glm-5.2` con el prompt de `solo`. Las dos tandas usan siempre el mismo
prompt; la etapa 1 y el bloque 1, uno por evaluación.

| qué | día (UTC) | hora UTC | llamadas | tok/s |
| --- | --- | --- | ---: | --- |
| etapa 1, la llamada con este mismo prompt | vie 2026-10-09 | 10:19:42 | 1 | 56.2 |
| etapa 1, mismos minutos que la segunda tanda | vie 2026-10-09 | 10:19:42–10:24:50 | 7 | 50.0 a 59.2, mediana 56.3 |
| etapa 1, la hora 10Z | vie 2026-10-09 | 10:19–10:59 | 46 | 48.4 a 65.2, mediana 56.2 |
| etapa 1, entera | vie 2026-10-09 | 10:19–12:35 | 140 | 48.4 a 65.2, mediana 55.4 |
| bloque 1 | vie 2026-10-30 | 00:05–00:26 | 42 | 83.6 a 107.6, mediana 99.2 |
| primera tanda | vie 2026-10-30 | 05:11–05:15 | 6 | 96.6 a 105.6, mediana 100.9 |
| segunda tanda | vie 2026-10-30 | 10:20–10:23 | 6 | 97.1 a 108.9, mediana 105.4 |

**Frente a la etapa 1, a la misma hora.**

- La segunda tanda cayó en el mismo día de la semana y en los mismos minutos del día que las
  primeras llamadas de la etapa 1, tres semanas después.
- La llamada de la etapa 1 con este prompt terminó a las 10:19:42Z, a 56.2 tok/s. La primera de
  la segunda tanda terminó a las 10:20:27Z, a 108.9.
- Los rangos no se solapan: la llamada más lenta de la segunda tanda (97.1) va 1.6 veces más
  rápido que la más rápida de esos minutos de la etapa 1 (59.2).

**Frente a la primera tanda.**

- Los rangos se solapan: 97.1 a 108.9 contra 96.6 a 105.6.
- Con 6 llamadas por tanda, la diferencia de medianas (105.4 contra 100.9) no se puede leer como
  un cambio.
- `minimax-m3` contestó a 188.7 tok/s, contra 180.9 a las 05:14Z.

## El `id` del cuerpo

**Lo que hay.**

- Las 12 llamadas a `glm-5.2` traen `chatcmpl-` y 8 caracteres hexadecimales.
- Las 2 de `minimax-m3` traen `chatcmpl-` y 32. Los dos tienen forma de UUID versión 4 sin
  guiones: el carácter 13 es `4` y el 17 está entre `8` y `b`.
- Las dos formas se mantienen a dos horas del día, con dos sesiones y con centros de datos de
  entrada distintos.

**Qué implica.**

- **La pasarela no fija el `id`.** Reescribe `model` con el id pedido, pero deja `id` como le
  llega. Dos formas en dos modelos quieren decir que el `id` lo generan dos sistemas distintos,
  los que producen cada respuesta. En el bloque T4 ya era así: tres endpoints dieron tres formas
  (CLAUDE.md, «Block T4»).
- **Es lo único del upstream que queda en la respuesta.** Las cabeceras ya no llegan, y `model`
  repite el id pedido. Pero el journal del estudio no guarda el `id`, así que solo un sondeo lo
  puede vigilar.
- **Dos sistemas sin cabeceras apuntan a la pasarela.** A los dos les faltan las mismas dos
  cabeceras. Un cambio en la pasarela, que es común a los dos, lo explica de una vez; un cambio en
  los upstreams necesitaría dos. Eso refuerza la explicación 2.

**Qué no implica.**

- **No sabemos si las formas cambiaron desde los bloques T5 y T6.** En T5 (el `Ping` de las
  06:14Z) y en T6 (12 de 12 llamadas) los dos modelos llegaban por `fireworks`, según la
  cabecera. Pero ninguno de los dos bloques guardó el `id` del cuerpo:
  - el `raw/<id>.json` del T5 solo guardó el estado y dos cabeceras;
  - `LLMCall` no tiene campo para el `id`.
- **Ya entonces eran dos rutas distintas dentro de `fireworks`.** `glm-5.2` era un modelo,
  `accounts/fireworks/models/glm-5p3`. `minimax-m3` era un router de otra cuenta,
  `accounts/anomalyinc/routers/zen-minimax-m3`. Dos formas de `id` cabrían aun si los dos
  siguieran por `fireworks`.
- **Solo en un caso se sabría algo.** Si en T5 y T6 los dos hubieran tenido la misma forma, al
  menos uno habría dejado ese camino. No se sabe si la tenían.
- **La forma no nombra el upstream.** CLAUDE.md lo dice desde el bloque T4: una forma es una
  conjetura, y la cabecera, una afirmación.

## El T5 y la hora de la primera tanda

**Cuándo midió el T5.**

- Según CLAUDE.md («Block T5 results»), el T5 midió `glm-5.2` a 54–64 tok/s en 24 llamadas
  válidas. Fue la corrida `20261006T063107Z`, entre las 06:31:07Z y las 07:01:17Z del martes
  2026-10-06, en el escritorio.
- Esa corrida no registró el upstream: `LLMCall` ganó los dos campos en el bloque T6. Que fuera
  por `fireworks` sale de otra corrida, un solo `Ping` a las 06:14:05Z (`20261006T061405Z`),
  17 minutos antes.

**¿Sirve para comparar con la primera tanda (05:11Z)? Como indicio, no como control.**

- La hora no es la misma: el T5 empezó 1 h 20 min más tarde en el día.
- El día y la máquina son otros: martes contra viernes, escritorio contra laptop. La velocidad
  se mide con el reloj de pared, así que incluye la red de cada máquina.
- El prompt es otro. El T5 usó `decider.md` con los alegatos de las dos mesas: 3 044 y 3 283
  tokens de prompt de media, contra 1 151. Sus respuestas también eran más largas: 4 494 y 4 837
  tokens de media, contra 1 760 a 4 994 en las tandas.
- El T5 tenía dos trabajos del decisor a la vez; las tandas, uno detrás de otro.
- Lo que sí dice: a una hora temprana en UTC, y por `fireworks`, `glm-5.2` iba a velocidades de
  la etapa 1, no de la primera tanda. Va en la misma dirección que la segunda tanda, que es el
  control que faltaba: la misma petición, el mismo día de la semana y los mismos minutos que la
  etapa 1.

**El T6, a la hora del bloque 1.**

- El T6 corrió entre el 2026-10-06T23:57:01Z y el 2026-10-07T00:11:05Z, en el escritorio, y
  registró `fireworks` en las 24 llamadas del decisor.
- CLAUDE.md no da su velocidad por llamada. El cociente de sus medias da unos 54 tok/s en `solo`
  (3 084 tokens en 56.8 s) y unos 57 en `no_debate` (4 000 en 70.2 s). Un cociente de medias no es
  un rango por llamada.
- Es la medida por `fireworks` más cercana a la hora del bloque 1 (00:05–00:26Z), que fue a unos
  99.

## El orden de las claves en la etapa 1

**Ya se conoce.** La investigación anterior a este sondeo (2026-10-30, antes de la primera tanda)
leyó el texto crudo de las respuestas en las cachés de `var/`, que no están versionadas:

- etapa 1 (`var/ablation-cache`): `action, confidence, size_fraction, invalidation_price,
  rationale`, el orden del esquema, en 140 de 140;
- bloque 1 (`var/forward/cache`): orden alfabético en 41 de 42. La que falta trae `action,
  confidence, invalidation_price, size_fraction, rationale`.

No se volvió a leer para este commit.

**Qué cambia con eso.**

- Con la envoltura registrada, la misma de la etapa 1, `glm-5.2` pasó del orden del esquema (140
  de 140) al alfabético. Desde el bloque 1 son 53 de 54: 41 del bloque 1 y 12 de las tandas.
- Una capa que reordenara la salida daría 54 de 54. La excepción del bloque 1 la descarta (lo
  leyó la investigación anterior). El orden sale de cómo se genera la respuesta.
- `minimax-m3` mantiene el orden del esquema en 2 de 2. No se sabe en qué orden contestaba antes
  a esta petición.

## Las cuatro explicaciones, con las dos tandas

1. **Otro upstream, o el mismo servido de otra manera, más rápido. Se refuerza. No queda
   decidida.**
   - Tras la primera tanda se reforzaba solo por la velocidad. Ahora:
     - la velocidad no depende de la hora (punto 4);
     - el orden de los argumentos también cambió: el del esquema en 140 de 140 en la etapa 1, el
       alfabético en 53 de 54 desde el bloque 1.
   - Son dos huellas de cómo se genera la respuesta, y cambiaron a la vez con la misma petición.
   - Que falten las cabeceras ya no forma parte de esta explicación: lo explica el punto 2.
   - Lo que no cambió no separa nada. Las 12 llamadas dan 1 151 `prompt_tokens`, y las que no
     acertaron la caché, 50 `cached_tokens`. La llamada de la etapa 1 con este prompt dio lo
     mismo: 1 151 y 50.
   - Queda abierto si es otro proveedor o el mismo con otra configuración. Ninguna huella nombra
     el upstream.
2. **Zen dejó de mandarlas para todos los modelos. Se refuerza.**
   - A las 10:20Z tampoco llegaron. Faltan en las 14 llamadas de las dos tandas, de dos modelos.
     `x-opencode-log-id` y `x-zen-model` llegaron en las 14.
   - Los dos modelos responden desde dos generadores de `id` distintos, y a ninguno le llegan.
     Una causa en la capa común, la pasarela, lo explica con un solo cambio.
   - La primera lectura decía que, si a esa hora volvían las cabeceras, este punto caía. No
     volvieron.
   - Siguen siendo dos modelos: decir «todos» es extrapolar.
3. **Algo de nuestra petición decide la ruta o las cabeceras. Se descarta en lo que se pudo
   variar.**
   - La variante b, con un User-Agent fijo y una sesión estable, dio lo mismo que la a en las dos
     tandas: las cabeceras, la forma del `id`, el orden de los argumentos y la velocidad.
     - A las 05:11Z: a, 98.7 a 105.2; b, 96.6 a 105.6.
     - A las 10:20Z: a, 104.5 a 108.9; b, 97.1 a 106.3.
   - El centro de datos de Cloudflare que recibió la petición cambió (`IAH`, `DFW`, `ATL`) sin que
     cambiara nada más.
   - La envoltura de la variante a es la registrada, la de la etapa 1.
   - No se varió la cuenta ni la máquina.
4. **La hora del día. Se descarta.**
   - La primera lectura fijó el criterio. Si `glm-5.2` volvía a unos 55 tok/s, la velocidad
     dependía de la hora. Si seguía cerca de 100, había cambiado con independencia de ella.
   - Siguió cerca de 100: 97.1 a 108.9 entre las 10:20Z y las 10:23Z de un viernes. En los mismos
     minutos del viernes de la etapa 1 fue de 50.0 a 59.2.
   - A esa hora también siguen el orden alfabético y la falta de cabeceras.
   - El T5 (06:31Z) y el T6 (23:57Z), por `fireworks`, dieron velocidades de la etapa 1 cerca de
     las horas de la primera tanda y del bloque 1. Van en la misma dirección, aunque ninguno es un
     control.
   - No descarta una variación de un día a otro: las dos tandas son del mismo día, y la etapa 1 es
     de otro.

**Lo que sigue abierto.**

- Qué upstream sirve hoy a `glm-5.2`. La respuesta ya no lo dice, y las huellas no lo nombran.
- Cuándo cambió. No hay ninguna llamada viva registrada en el repositorio entre el fin de la
  etapa 1 (2026-10-09T12:35Z) y el bloque 1 (2026-10-30T00:05Z).
- Si otros modelos también perdieron las cabeceras: solo se preguntó a dos.
