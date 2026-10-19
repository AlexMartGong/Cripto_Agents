# Datos del estudio hacia adelante de `solo`

Lo que escribe cada lote de `python -m crypto_agents.forward batch --block <inicio>`
(`docs/forward.md`, apartado 4), y la corrida solo de líneas base de
`python -m crypto_agents.forward baselines --block <inicio>` (apartado 5). Tras completar cada lote, todo se versiona y se sube (asiento del
2026-10-17 del «Registro del estudio»). El comando no hace commits: al terminar imprime la lista de
archivos que hay que versionar.

```
history/<símbolo>_4h.csv            velas de Binance spot, extendidas
funding/<símbolo>.csv               funding de los perpetuos de Binance USDⓈ-M, extendido
revisions.json                      velas revisadas que el operador aceptó con el flag
manifests/<AAAA-MM-DD>.json         manifiesto de cada bloque no vacío, por su inicio
batches/<AAAA-MM-DD>/<UTC>/         una pasada del lote de ese bloque:
    forward.json                    el meta del lote, también en una pasada que se niega
                                    o que se detiene por una vela revisada o que falta
    meta.json                       el meta de la corrida (`audit.RunMeta`), si corrió
    solo.jsonl, always_buy.jsonl…   un journal por brazo, si corrió
    upstream.md                     asiento preparado si cambió el upstream del decisor
    superseded.json                 la marca de una pasada sustituida, si lo está
```

La caché del estudio vive en `var/forward/cache/` y no se versiona: los journals guardan cada
llamada. La repetición de un lote sustituido usa la suya, nueva,
`var/forward/cache/reruns/<AAAA-MM-DD>/<pasada que sustituye>/`.

**La reanudación es por journal, no por caché.**
- Una pasada nueva del mismo lote se salta los `run_id` que ya tienen registro completo en los
  journals de las pasadas anteriores: una decisión, o una respuesta inválida tras los dos intentos.
- Solo vuelve a evaluar lo que se perdió por el proveedor, un 402 o el tope, y la caché sirve los
  reintentos de eso.
- Un `run_id` completo en dos pasadas detiene el lote.
- Por eso el journal de una pasada solo lleva lo que esa pasada evaluó. El estado del lote es la
  unión de sus pasadas no sustituidas (`forward.batch_records`), no el último journal.

**Un lote inválido se repite entero** con `batch --block <inicio> --supersede` (apartado 5).
- Se niega si el lote no es inválido: uno incompleto se reanuda.
- Justo antes de la primera llamada deja un `superseded.json` en cada pasada que corrió, con el
  instante, la pasada que la sustituye y por qué el lote era inválido. No reescribe nada de lo que
  ya tenían.
- Las sustituidas quedan fuera del lote: ni sus registros ni sus duplicados cuentan. Su gasto
  sigue contando para el tope, y el análisis las sigue leyendo para ver si una llamó con el camino
  medido cambiado, porque sustituirla no deshace esa llamada.

**Las líneas base se completan después del plazo** con `baselines --block <inicio>` (apartado 5).
- Es para un lote interrumpido o que no corrió: sin ellas, el análisis no puede puntuar las
  faltantes de `solo` contra `rule_trend`, y se niega.
- Es una pasada como las demás, con `forward.json` («líneas base»), `meta.json` y los journals de
  las cuatro líneas base que faltaban. No lleva `solo`.
- No construye backends ni lee el catálogo. Repite las comprobaciones que fijan las líneas base
  (etapa 1, atadura, recuentos del pool y manifiesto), corre con un router solo de caché sobre una
  caché vacía y comprueba al terminar que ningún registro trae una llamada.

## Procedencia de las velas

- **Descarga:** el camino de `activation_sweep.load_series`, es decir `CcxtMarketClient("binance")`
  y `market.download_history`. Es lectura pública, sin credenciales y solo con los mercados spot,
  y es el camino que descargó `data/history/` (procedencia en `docs/activation.md`).
- **Rango:** cada extensión vuelve a descargar la serie entera, desde 2024-08-16T00:00Z hasta la
  vela que cierra 24 h después del cierre del bloque.
- **Comparación:** cada vela guardada se compara con la descargada.
- **Formato:** byte a byte el de `data/history/`. Es `timestamp,open,high,low,close,volume`, con el
  timestamp en milisegundos como entero, el `repr` de cada valor y `\n` como fin de línea.
  `market.write_ohlcv_csv` escribe `\r\n` y no se usa: reescribiría lo publicado con otros bytes.

## Procedencia del funding

- **Descarga:** la misma que `data/funding/` (su README):
  `perp_probe --exchange binanceusdm --start 2024-08-16 --end <cierre del bloque + 1 día>`, con
  los dos extremos incluidos.
- **Formato:** cada archivo son los bytes que escribe `perp_probe.series_csv`:
  `timestamp_ms,funding_rate`, ordenado y con el `repr` de cada tasa.
- Lo que deja la descarga en `var/forward/perp/` no se versiona.

## Reglas

Las aplica `crypto_agents/forward_data.py` y las fijan sus pruebas en `tests/test_forward_data.py`
y `tests/test_forward.py`.

- **Solo crecen por la derecha.** Lo guardado nunca se reescribe: una extensión añade filas al
  final de los bytes del archivo.
- **Las velas empiezan byte a byte por las 4 380 filas de `data/history/`**, y el funding por las
  2 191 de `data/funding/`.
- **Solo se escriben velas cerradas.** La extensión recibe el instante actual y, si la vela que
  cierra en `until` no ha cerrado todavía, se niega antes de descargar. El exchange devuelve esa
  vela en formación, y así se guardó la fila 4 380. En el lote no llega a pasar, porque corre
  desde el cierre del bloque + 24 h, que es su `until`.
- **Vela revisada** (`docs/forward.md`, apartado 4). El exchange devuelve distinta, o ya no
  devuelve, una vela guardada.
  - El lote se detiene con el código 5. No escribe ninguna serie y deja preparados el asiento del
    registro (símbolo, apertura, valor guardado y devuelto) y el comando con el flag.
  - Con el asiento en el registro, se vuelve con
    `--accept-revised SÍMBOLO@APERTURA` (por ejemplo `ADA/USDT@2026-08-15T20:00Z`), una vez por
    vela. La serie del estudio conserva el valor guardado.
  - La aceptación queda en `revisions.json`, con el valor guardado, el devuelto, la hora y el
    bloque. Los lotes siguientes la aceptan solos mientras el exchange devuelva ese mismo valor. Si
    la cambia otra vez, es una revisión nueva y se detiene.
  - Un flag que nombra una vela que no está revisada se rechaza.
  - La primera: la fila 4 380 de las siete series, la vela que abre el 2026-08-15T20:00Z, se guardó
    en formación. El registro la anota con sus valores, y el lote del bloque 1 se detendrá en ella.
- **Vela que falta.** Una vela nueva que la descarga no trae detiene el lote con el código 6, sin
  escribir nada, y la pasada registra en su `forward.json` cuándo faltaba.
  - Hay que volver a correr el lote 24 h después.
  - Si sigue faltando cuando ya pasaron 24 h desde que se vio faltar por primera vez, el hueco es
    persistente: código 7, y el estudio termina en ese bloque (apartados 4 y 5).
- **Una liquidación de funding revisada** detiene la extensión con un error, incluido otro jitter
  en la marca. Las reglas de la vela revisada y del hueco son de las velas.
- **Otros errores que detienen la extensión sin escribir nada:**
  - una vela fuera de la rejilla de 4h;
  - un archivo extendido que no empieza por los bytes publicados;
  - un archivo publicado que ya no tiene su sha-256.
- **No se recorta.** Si un archivo ya llega más lejos, porque corrió antes el lote de un bloque
  posterior, se verifica hasta donde llega.

## Dónde queda la procedencia de cada extensión

Los digests cambian con cada lote, así que no hay aquí una tabla fija como la de
`data/funding/README.md`. El `forward.json` de cada pasada lleva, para cada archivo después de
extenderlo, la ruta, su sha-256, sus filas y su última marca. Lleva también la hora, la máquina,
el commit y si el árbol estaba limpio, y el sha-256 de `uv.lock` y la versión de Python.
`sha256sum` sobre los archivos tiene que dar lo que dice el `forward.json` del último lote.

## El manifiesto de un bloque

- **Formato.** Es un `SelectionManifest`, así que lo leen sin cambios `verify_histories`,
  `plan_from_manifest`, `audit.load_plan` y `forward_plan.batch_spend_cap`.
- **Contenido.** Lleva toda la población del bloque (apartado 2), sin muestreo. La población son
  las activaciones del barrido que cumplen tres cosas:
  - tienen el horizonte por delante;
  - el cierre de su vela cae en el bloque;
  - el gate las confirma con la ventana de 500 velas.
- **Parámetros.** Los parámetros son los de `data/ablation_selection.json`:
  - los siete símbolos y 4h;
  - `candle_limit` 500, horizonte 6 y warm-up 400;
  - la semilla 20260815.
- `strata` vale 1 y `target` el número de entradas. Cada entrada lleva el digest de su ventana.
- **Bloque vacío.** Un bloque sin activaciones no tiene manifiesto: `SelectionManifest` exige al
  menos una entrada. Lo registra el `forward.json` de su lote, con 0 evaluaciones.
