# Estudio hacia adelante de `solo` — prerregistro

Escrito el 2026-10-15 y cambiado el 2026-10-16, el 2026-10-17 y el 2026-10-18 (lo que cambió está
en el apartado 9), antes de que exista ningún dato del estudio. Fija la pregunta, la población, lo
que queda congelado, cómo se corre cada lote, cuándo un lote vale, cómo se analiza y qué significa
cada veredicto. **Estado: el del «Registro del estudio», al final.** Los comandos de lote y de
análisis cumplen lo que exige el apartado 10.

Cada cifra de este documento sale de uno de cuatro sitios, citado a su lado:
- el cálculo de planificación del apartado 9;
- la comprobación del prompt y de la petición del apartado 3;
- una sección publicada de `docs/ablation.md`;
- una medida única, con su fecha y lo que la hizo, donde se cita (apartado 10, punto 9).

Los dos comandos no llaman a ningún modelo:

```bash
python -m crypto_agents.forward_plan var/ablation/20261009T045033Z
python -m crypto_agents.forward_render var/ablation/20261009T045033Z
```

Nada de lo que fija este documento se cambia una vez abierto el bloque 1. Lo único que se le añade
después es el «Registro del estudio», al final, con entradas fechadas.

## 1. Pregunta y comparación principal

**Sobre activaciones que no existían cuando se escribió esto, ¿la decisión de `solo` rinde por
evaluación más que la regla de tendencia `rule_trend`, lo mismo o menos?**

- **Comparación principal:** `solo` − `rule_trend`, la diferencia pareada evaluación por evaluación
  del retorno **bruto**, con el stop común (`stops.py`, 2·ATR) y un horizonte de 6 velas de 4h.
  La puntuación es la del criterio 3 de la ablación: `criteria.paired_returns`, que empareja por
  `run_id` y puntúa con `criteria.SCORING = Scoring.COMMON_STOP` (`criteria.py:180`).
- **δ = 0.40 %** de retorno bruto por evaluación. Es el margen por debajo del cual dos brazos se
  consideran iguales.
  - Se fijó el 2026-10-15 por lo que se podía detectar en 50 semanas: alrededor de un 80 % de
    probabilidad de «supera» con `sigma_D` puntual, según la estimación del operador previa a este
    documento. El cálculo del apartado 9, con el efecto de diseño de planificación, da 87.6 %
    (sección 5).
  - **No es un umbral de relevancia económica.** Lo fijó lo que el estudio puede distinguir, no lo
    que valdría una ventaja.
  - Con δ = 0.20 %, el de la ablación, la probabilidad de «supera» sería del 34.4 % (apartado 9,
    sección 5).
- **Descriptivas:** `solo` frente a las otras tres líneas base, el retorno neto, el stop propio y
  el reparto por upstream del decisor (apartado 6). Ninguna entra en el veredicto.

Por qué un estudio nuevo y no otra etapa: la ablación cerró sin detectar señal, y la enmienda 4
descartó una etapa 2 sobre el pool histórico. Su n efectivo llega como mucho a 492.4 con n = 3 605,
que no alcanza las 952.0 que pedía la potencia del 80 % contra `rule_trend` con el δ de la ablación,
0.20 % (`docs/ablation.md`, «Enmienda 4», punto 2). Un estudio hacia adelante añade evaluaciones de
un periodo que el pool no tiene.

El brazo es `solo` porque la ablación lo conservó por la regla por defecto del criterio 7
(`docs/ablation.md`, «Conclusión»). Contra `rule_trend` es el par de la etapa 1 con menor `sigma_D`:
2.203 %, con un rango bootstrap de 1.717 % a 2.670 % («Recálculo de dispersión», sección 1).

## 2. Población y bloques

**Símbolos, temporalidad y gate:** los de la etapa 1.
- Los siete del manifiesto `data/ablation_selection.json`: ADA, BNB, BTC, DOGE, ETH, SOL y XRP,
  contra USDT.
- Velas de 4h.
- El gate `activation.DEFAULT_CONFIG`, el preset `indicators.DEFAULT_PRESET` y `candle_limit` 500.

**Bloques.**
- Duran 7 días UTC, alineados al epoch, y empiezan los jueves a las 00:00 UTC. Son exactamente los
  clústeres de la enmienda 4 (`criteria.CLUSTER_DAYS = 7`, `criteria.py:188`; `metrics.block_of`).
- A cada evaluación la asigna su instante evaluado: el cierre de su vela, que es su entrada y lo
  que lleva `EvaluationRecord.at`.
- Un instante en el borde exacto abre el bloque nuevo. Una vela que cierra el jueves a las 00:00
  pertenece al bloque que empieza entonces.

**Población de un bloque.** Todas las activaciones que cumplen las cuatro condiciones siguientes.
Sin muestreo.
- Las da el barrido (`activation_sweep.activated_bars`) sobre el histórico extendido, que empieza
  como `data/history/` en 2024-08-16.
- Su instante evaluado cae en el bloque.
- Tienen 6 velas por delante.
- El gate las confirma también con la ventana de 500 velas que verá el replay
  (`selection.confirm_activation`).

Es la definición del pool sobre el que se planificó: las 4 626 activaciones confirmadas de
`docs/ablation.md`, «Recálculo de dispersión», sección 5. El barrido arranca en el mismo inicio para
que los indicadores recursivos de las velas nuevas se calculen igual que los del pool.
`tests/test_lookahead.py` fija que añadir velas por la derecha no cambia ninguna barra anterior.

**Inicio.**
- Regla: el primer bloque es el primero que empieza **estrictamente después** del merge de este
  documento en master, y nunca antes de 2026-10-22T00:00Z.
- El merge se hace con `git merge --no-ff`, para que exista un commit de merge, y se sube a GitHub
  antes de 2026-10-22T00:00Z.
- La primera entrada del «Registro del estudio» guarda tres cosas: el hash del commit de merge, su
  fecha de committer y la fecha en que GitHub muestra el push.
- Las fechas de un commit las pone quien commitea; la de GitHub es la evidencia externa. El
  instante del merge es el del commit de merge en master, y si su fecha de committer y la de GitHub
  caen en bloques distintos, manda la de GitHub.
- Un merge justo en el borde de un bloque no lo abre: el estudio empieza en el siguiente.
- `forward_plan.first_block` es esa regla, y `tests/test_forward_plan.py` la fija, borde incluido.

**Cierre.**
- El estudio cierra cuando acumula **50 bloques no vacíos**: bloques con al menos una activación
  de la población.
- Si al terminar el **bloque 52** no ha llegado a 50, el estudio queda «no determinado».
- Los bloques vacíos se cuentan y se reportan, pero no suman para cerrar.
- Hay **un único análisis**, al cerrar (apartado 6).
- `forward_plan.closing_block` es esta regla, y `tests/test_forward_plan.py` la fija: con un
  bloque vacío el estudio cierra en el 51, con dos en el 52 y con tres queda «no determinado».

**Por qué cerrar así no es mirar los datos por el camino.**
- La regla lee una sola cosa: cuántas activaciones de la población tuvo cada bloque. Eso lo
  deciden las velas y el gate antes de que `solo` decida nada, así que no depende de ninguna
  decisión ni de ningún retorno de `solo`.
- Queda escrita antes del primer dato del estudio.
- Existe por el intervalo: cada bloque es un clúster y el intervalo exige 50 (apartado 6). Con 50
  bloques fijos, una sola semana vacía habría dejado el estudio sin intervalo.

**Con el merge antes de 2026-10-22T00:00Z** (apartado 9, sección 7):

| | inicio | fin | su lote, desde | plazo de su lote |
| --- | --- | --- | --- | --- |
| bloque 1 | 2026-10-22T00:00Z | 2026-10-29T00:00Z | 2026-10-30T00:00Z | 2026-11-12T00:00Z |
| bloque 50: fin esperado, si ningún bloque sale vacío | 2027-09-30T00:00Z | 2027-10-07T00:00Z | 2027-10-08T00:00Z | 2027-10-21T00:00Z |
| bloque 52: fin más tardío posible | 2027-10-14T00:00Z | 2027-10-21T00:00Z | 2027-10-22T00:00Z | 2027-11-04T00:00Z |

Si el merge llega más tarde, cada fecha se corre las semanas que haga falta, con la misma regla.

## 3. Lo que queda congelado

Todo lo que sigue forma el **camino medido**. Qué termina el estudio y qué no:
- **Lo termina un lote que llama a un modelo con el camino medido cambiado**, en el bloque de ese
  lote.
- **No lo termina una comprobación previa a la primera llamada que falla**, si el camino registrado
  se restaura antes de llamar a ningún modelo. Las comprobaciones son las del apartado 10, punto 4,
  y un lote no llama a nadie hasta que pasan todas. El «Registro del estudio» anota la fecha, la
  causa y la restauración, y el lote conserva su plazo de 14 días.
- **Durante el estudio no se actualizan dependencias.** Las fija el `uv.lock` del commit del merge.
- **El `meta` de cada lote registra el sha-256 de `uv.lock` y la versión de Python.** Un cambio en
  ellos no termina el estudio por sí solo: deciden las comprobaciones de comportamiento. Si pasan,
  el lote corre; si alguna falla, se restaura como cualquier otra.
- Las comprobaciones de comportamiento miran las 140 de la etapa 1 y el pool histórico. Un cambio
  que solo se notara en velas nuevas no lo verían.
- Una corrección que no toque ninguna pieza —el arnés, los reportes, la descarga, las alertas— no
  lo termina. Se documenta en el «Registro del estudio» con fecha, commit, qué cambió y por qué.

Ninguna pieza de las tablas cambió de valor entre el commit de la etapa 1 (`b9676eb`) y este
documento.
- `state.py`, `llm.py`, `prompts/`, `activation.py`, `indicators.py`, `stops.py`, `baselines.py`,
  `replay.py` y `settings.py` son idénticos.
- En `selection.py` y `criteria.py` cambió el código de alrededor, no estas constantes.

**El decisor**

La configuración registrada sale del `meta.json` de la etapa 1
(`var/ablation/20261009T045033Z/meta.json`, sha-256
`a9d4b2c3ff84a5c895eae2afa366e58bc89838758a0a9471cf5b2f9214aae06c`), que es el registro de lo que
corrió `solo`.
- Esos valores llegaron al `meta.json` desde `.env`, que es del operador y no está versionado.
  `.env` es el lugar del que el comando los lee, no la fuente de lo registrado.
- `max_attempts` no está en ese `meta.json`. Se cita del código, y el bloque de implementación lo
  añade al `meta` de cada lote (apartado 10, punto 4).

| pieza | valor registrado | fuente |
| --- | --- | --- |
| id del modelo | `glm-5.2` | `meta.json`: `arm_roles.solo.decider.model` |
| backend | `openai` | `meta.json`: `arm_roles.solo.decider.backend` |
| `base_url` | `https://opencode.ai/zen/v1` | `meta.json`: `base_url` |
| salida estructurada | `function_calling` | `meta.json`: `arm_roles.solo.decider.structured_output` |
| temperatura | 0.0 | `meta.json`: `arm_roles.solo.decider.temperature` |
| timeout | 240 s | `meta.json`: `openai_timeout_seconds` |
| intentos por llamada | 2 | no está en el `meta.json`: `llm.py:890` (`ModelRouter`, `max_attempts`) |
| plantilla de reintento | `_RETRY_TEMPLATE` | `llm.py:123`; sha-256 del texto en UTF-8 `588bf07e71d7a92a4d098a9659a3a9c4e05dc3347743ccc1f077e634a8e39890` |
| esquema de la respuesta | `Proposal` | `state.py:401`; sha-256 de `json.dumps(Proposal.model_json_schema(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)` en UTF-8 `4970a9ec09cfc93e89e3058561325e55368f5b87f3f154d5f4e686a1f21e5bb7` |

**Comprobación por lote.** Antes de su primera llamada, cada lote compara su `meta.json` con los
valores registrados de esta tabla, `max_attempts` incluido. Si alguno no coincide, el lote es
inválido y no llama a nadie. Le aplica la regla de lote inválido del apartado 5: se corrige sin
tocar el camino medido, devolviendo la configuración a lo registrado, y se corre entero antes de su
plazo. Como no llamó a ningún modelo, no termina el estudio.

**El prompt de `solo`**

El texto de la plantilla se fija por sus bytes. El código que la rellena se fija por lo que
produce, no por su código fuente: un cambio de espaciado no cambia el prompt, y un cambio en una
cadena de formato sí.

| pieza | fuente | lo que la fija |
| --- | --- | --- |
| plantilla | `src/crypto_agents/prompts/decider_solo.md`, sin cambios desde `48dfca6`, anterior a la etapa 1 | sha-256 del archivo, `fe622bc357f853db3e5b61bc06ff6312f47d1dfff4cc2eff06db15d20822f2de` |
| renderizado | `solo_prompt` (`prompts/__init__.py:183`), `format_indicators` (`:76`) y `format_triggers` (`:83`) | `tests/data/solo_prompt_digests.jsonl`, sha-256 `8d1063441c43d49969eba2911eb9ffa8593040eefaff112f464bf6ee5d67644b`: las dos primeras entradas de cada símbolo del manifiesto de la etapa 1, 14 en total, con el `prompt_digest` que registró el journal de `solo`. El test `tests/test_forward_render.py::test_the_fourteen_prompts_render_as_the_journal_recorded` las reconstruye desde `data/history/` y el manifiesto, sin modelos, y exige el mismo digest. |

- **Qué rompe la prueba y qué no.** Cada caso tiene su mutación en `tests/test_forward_render.py`.
  - La rompen una frase cambiada en la plantilla, `format_indicators` con 3 decimales,
    `format_triggers` sin las comillas invertidas y el cierre en coma fija en `solo_prompt`.
  - No la rompe el código de las tres funciones reescrito solo en espaciado. Esa reescritura sí
    cambia el sha-256 de su código fuente, que es lo que este apartado fijaba antes.
  - Ni la preparación ni el renderizado usan el router. El contexto de la prueba lleva uno que
    levanta ante cualquier atributo, y una `prepare_evaluation` que lee la configuración a través
    de él rompe la prueba.
- **Las 140 de la etapa 1, una vez.** `python -m crypto_agents.forward_render
  var/ablation/20261009T045033Z` terminó con **código 0** el 2026-10-16 en la laptop, sin llamar a
  ningún modelo:
  - 140 evaluaciones de `solo` en el journal (`solo.jsonl`, sha-256
    `1180554e2c19aab20d623f2112547aae8f930ad934d3fc17ca65fd5c70cc4b17`);
  - 140 reconstruidas, 140 coinciden, 0 distintas, 0 sin reconstruir y 0 de más.
- **Por lote.** Cada lote repite la comprobación antes de su primera llamada (apartado 10, punto 4).

**Lo que viaja al modelo en la llamada del decisor**

`llm.prompt_digest` (`llm.py:261`) es `sha256(prompt.encode("utf-8"))` del texto que el router
entrega al backend en ese intento (`llm.py:926` y `:964`).
- En el primer intento, ese texto es el prompt renderizado.
- En el reintento, es el prompt seguido de `_RETRY_TEMPLATE` con el error (`llm.py:1008`).
- No cubre nada más: ni la herramienta, ni el modelo, ni los parámetros.
- La clave de la caché añade el backend, el modelo, el **nombre** del esquema y el modo
  (`cache.py:51`). Tampoco cubre el contenido de la herramienta.

La petición se capturó con la configuración registrada de arriba. Salió de
`OpenAIBackend.complete`, con el `ChatOpenAI` y el SDK reales, hacia un transporte simulado, así
que nada salió de la máquina. Lo que lleva su cuerpo JSON:

| pieza | en la petición | qué la cubre |
| --- | --- | --- |
| mensaje de sistema | ninguno | la envoltura |
| mensaje del usuario | uno solo, con el prompt; en el reintento, el prompt y la plantilla de reintento con el error | su `prompt_digest`, en cada `LLMCall` del journal; la plantilla y el renderizado, en la tabla de arriba; `_RETRY_TEMPLATE`, en la tabla «El decisor» |
| herramienta | `tools`: una función `Proposal`, con el docstring de la clase como descripción y el esquema convertido por LangChain como parámetros | el lado de Python, el sha-256 del esquema de `Proposal` («El decisor»); lo que sale por el cable, la envoltura |
| elección de herramienta | `tool_choice` = la función `Proposal`; `parallel_tool_calls` = `false` | la envoltura |
| modelo y temperatura | `model` = `glm-5.2`; `temperature` = 0.0 | la tabla «El decisor» y la envoltura |
| el resto de parámetros | `stream` = `false`. No van `max_tokens`, `seed`, `top_p` ni `response_format`. | la envoltura: una clave de más o de menos la cambia |

**La envoltura** es lo único que no cubría ningún digest registrado, y se registra aquí.
- Es el cuerpo JSON de esa petición con el contenido de cada mensaje vaciado, en JSON canónico:
  `json.dumps(…, sort_keys=True, separators=(",", ":"), ensure_ascii=False)` en UTF-8.
- Su sha-256 es `f0727ca0e3c51a242b7351e215a0c6596d1f9e7b777f2efda9d402f7ba3a6928`
  (`forward_render.SOLO_REQUEST_ENVELOPE_SHA256`). No depende del prompt.
- La fija `tests/test_forward_render.py::test_the_envelope_of_the_registered_decider_is_the_registered_one`.
  Un mensaje de sistema, otro modo, otro modelo y otra temperatura la cambian, cada uno con su
  prueba.
- La invocación de arriba también la comprueba, con el decisor que lee de `.env`, y comprueba
  además que el único mensaje es el prompt cuyo digest registró el journal. Las dos cosas
  coinciden.

**Las cabeceras no llegan al modelo.** Las lee la pasarela, y ninguna se registra.
- `authorization` lleva la clave.
- `x-opencode-session` es un id aleatorio por proceso (`llm.py:684`) que la caché no ve.
- `user-agent` y `x-stainless-*` dicen:
  - la versión del SDK de OpenAI, la de Python y el sistema operativo;
  - el timeout de lectura, 240 s, que ya está en la tabla «El decisor»;
  - `x-stainless-retry-count: 0`.
- Las bibliotecas que arman la petición las fija `uv.lock`, cuyo sha-256 registra el `meta` de
  cada lote (arriba). Si una versión nueva cambiara lo que se manda al modelo, cambiaría la
  envoltura.

**El mercado, la puntuación y las líneas base**

Cada fila dice qué comprobación la fija, además de citar su fuente. Hay dos clases de pieza:
- **Los parámetros** —símbolos, temporalidad, `candle_limit`, horizonte, warm-up y semilla— los
  lleva el manifiesto de cada bloque. El lote los compara con los de `data/ablation_selection.json`
  antes de su primera llamada (apartado 10, punto 4). Las otras comprobaciones los leen del
  manifiesto y del `meta.json` de la etapa 1, así que no fijan la constante del código que escribe
  el manifiesto nuevo.
- **El código** —gate, indicadores, stop común, puntuación y líneas base— lo fijan las
  comprobaciones de comportamiento del apartado 10, punto 4, sobre la etapa 1 y el pool:
  - los recuentos del pool (apartado 9, sección 1);
  - la atadura de las líneas base con su journal (la misma sección);
  - los retornos de las líneas base contra los que publicó la etapa 1 (la misma sección);
  - el renderizado del prompt de `solo` (arriba).

| pieza | valor | fuente | qué la fija |
| --- | --- | --- | --- |
| símbolos y temporalidad | 7 símbolos, 4h | `data/ablation_selection.json`, sha-256 `73f87870cf24d06c341ff75fa72f164f9cd5202623cf0d76babbdab4a2803f23`, campo `series` | la comparación del manifiesto del bloque. El sha-256 del manifiesto de la etapa 1 lo compara `read_stage` con el `plan_sha256` de su `meta.json` (`audit.py:396`). |
| gate | `breakout_lookback` 20, `atr_jump_ratio` 1.5, `atr_jump_lookback` 20, `adx_trend_threshold` 25.0 | `activation.py:39-49` (`ActivationConfig`, `DEFAULT_CONFIG`) | los recuentos del pool: 4 740 activaciones del barrido y 112 sin confirmar. Y el renderizado: los disparadores de cada prompt entran en su digest. |
| indicadores | RSI 14, EMA 20/50/200, ATR 14, ADX 14, Bollinger 20, media de volumen 20; warm-up 400 velas | `indicators.py:40-55` y `:103` (`DEFAULT_PRESET`); `min_bars`, `indicators.py:78` | el renderizado: cada prompt lleva los valores de los indicadores con cuatro decimales. La atadura: el ATR da el stop, y el ADX y las medias, la regla de tendencia. Los recuentos: el gate los lee. El warm-up, la comparación del manifiesto (`warmup_bars`). |
| `candle_limit` | 500 | `selection.py:122` | la comparación del manifiesto. Con 500, los recuentos (las 112 sin confirmar y las 4 626 confirmadas dependen de esa ventana), el renderizado y la atadura, que calculan los indicadores sobre ella. |
| horizonte | 6 velas | `selection.py:118`; `horizon` del `meta.json` de la etapa 1 | la comparación del manifiesto. Con 6, los recuentos (4 737 con el horizonte por delante) y los retornos publicados. |
| stop común | entrada ∓ 2·ATR | `stops.py:38` (`COMMON_STOP_ATR_MULTIPLE`) | la atadura: el stop que declara cada línea base, en 560 evaluaciones. Y los retornos publicados. |
| puntuación | stop común, bruto | `criteria.py:180` (`SCORING`) | los retornos publicados: 560 de 560 iguales. La atadura con el journal no la ve, porque puntúa sus dos lados con la misma función. |
| regla de tendencia | ADX mínimo 25.0 | `baselines.py:55` (`TREND_ADX_MIN`) | la atadura: la acción de `rule_trend` en las 140. |
| tamaño y confianza de las líneas base | 0.05 y 0.5 | `baselines.py:58` y `:64` | ninguna comprobación por lote: no entran en el retorno bruto por evaluación ni en el veredicto. Las fija `tests/test_baselines.py::test_every_parameter_was_frozen_before_any_result_existed`. |
| semilla de `random_uniform` | 20260815 | `selection.py:110` (`DEFAULT_SEED`) y el campo `seed` del manifiesto de la etapa 1; la acción sale de `baselines.random_action` (`baselines.py:80`) con el `run_id` de `replay.replay_run_id` (`replay.py:234`) | la comparación del manifiesto. Con ella, la atadura: la acción de `random_uniform` en las 140 fija `random_action` y `replay_run_id`. |

El upstream que sirve el id del decisor **no** se congela. No depende de nadie de aquí, y su
política está en el apartado 5.

## 4. Procedimiento de cada lote

Un lote es el bloque de una semana pasado por el arnés de la ablación como un replay. No se usa el
runner en vivo.

**Cuándo.**
- Desde el cierre del bloque más el horizonte (6 velas de 4h, 24 h), es decir el viernes a las
  00:00 UTC. A esa hora todas las activaciones del bloque tienen sus 6 velas por delante.
- Plazo: 14 días tras el cierre del bloque.
- Con el inicio del apartado 2:
  - el lote del bloque 1 se corre desde 2026-10-30T00:00Z y vence el 2026-11-12T00:00Z;
  - el del bloque 50, desde 2027-10-08T00:00Z, y vence el 2027-10-21T00:00Z;
  - el del bloque 52, si el estudio llega a él, desde 2027-10-22T00:00Z, y vence el
    2027-11-04T00:00Z.

**Velas.**
- La misma fuente y el mismo tipo de mercado que `data/history/`: **Binance spot**, no perpetuos.
- Lectura pública sin credenciales, con `CcxtMarketClient("binance")`, que carga solo los mercados
  spot (`SPOT_ONLY`, `market.py:59`), y `market.download_history` (`market.py:252`). Es el camino
  de `activation_sweep.load_series`, que descargó `data/history/` (procedencia en
  `docs/activation.md`).
- El histórico extendido vive aparte de `data/history/`, cuyos sha-256 están publicados en
  `docs/ablation.md` y no se tocan.
- Se extiende solo por la derecha:
  - sus primeras 4 380 filas por serie tienen que ser byte a byte las de `data/history/`;
  - cubre hasta la vela que cierra 24 h después del cierre del bloque.
- **Vela revisada.** Si el exchange devuelve distinta una vela ya guardada, la extensión se
  detiene.
  - El operador lo anota en el «Registro del estudio»: el símbolo, la apertura de la vela, el
    valor guardado y el valor devuelto.
  - La extensión continúa con el valor guardado, mediante un flag explícito que nombra esa vela.
  - La serie del estudio es la que se guardó.
- **Hueco persistente.** Si una vela nueva falta y sigue faltando al volver a descargar 24 h
  después, el estudio termina en ese bloque, con lo que dice el apartado 5 sobre un estudio que
  termina antes de cerrar. En los dos años de `data/history/` no hubo ningún hueco.
- Las velas entre el 2026-08-16 y el inicio del bloque 1 se descargan porque son ventana de los
  primeros bloques, pero no se evalúa nada antes del bloque 1.

**Funding, para el neto descriptivo.**
- La misma descarga que `data/funding/` (su README): `perp_probe --exchange binanceusdm --start …
  --end <cierre del bloque + 1 día>`, con el extremo incluido, de modo que la liquidación posterior
  a la última salida del bloque está.
- Vive aparte de `data/funding/`, y se extiende también solo por la derecha.
- Un funding que falta deja el neto de esa evaluación «no determinado» (`funding.py`). No toca el
  bruto ni el veredicto.

**Manifiesto del bloque.**
- Antes de la primera llamada se escribe y se versiona un manifiesto por bloque. Lleva **todas**
  las activaciones de la población del apartado 2, cada una con su símbolo, su vela, su instante,
  el digest de su ventana del replay y sus disparadores.
- Lleva también la referencia y el sha-256 de cada serie extendida, y la semilla 20260815 de
  `random_uniform`.
- El replay comprueba los digests de las ventanas (`selection.verify_histories`) antes de pagar
  nada.

**Brazos, llenado, tope y reanudación.**
- Los brazos son `solo` y las cuatro líneas base, que no llaman a ningún modelo.
- Se corre con `--fill`, con facturación por uso (`CA_BILLING=payg`, el `billing` del `meta.json`
  de la etapa 1) y con caché propia del estudio.
- **Tope de gasto por lote:** 3 × las evaluaciones de su manifiesto × 0.01509 USD
  (`forward_plan.batch_spend_cap`).
  - 0.01509 USD por evaluación es la cota superior de `solo` en la etapa 1, y queda registrada
    (`forward_plan.USD_PER_EVALUATION`; apartado 9, sección 6).
  - Se calcula antes de la primera llamada del lote, y es **acumulado por lote**: una reanudación
    recibe el tope menos lo que ya gastaron las pasadas anteriores de ese lote, leído de sus
    journals.
  - En el pool, el lote medio (48.98 evaluaciones) tendría un tope de 2.22 USD; el del p90 (70),
    3.17; y el más grande (98), 4.44 (apartado 9, sección 6).
  - **Por qué proporcional.** Un tope fijo de 2.22 USD cubre unas 147 evaluaciones
    (2.22 / 0.01509). Una semana con más activaciones dejaría faltantes, y se concentrarían en las
    semanas más volátiles, que no faltan al azar. Con el tope proporcional, el coste medido de un
    lote es un tercio de su tope, sea del tamaño que sea.
- **La cota del gasto del estudio que dan estas reglas** es la suma de los topes de sus lotes:
  3 × las evaluaciones del estudio × 0.01509 USD. Con las 2 449 esperadas, **110.87 USD** (apartado
  9, sección 6).
  - Solo los bloques no vacíos llaman a algún modelo.
  - Una reanudación comparte el tope de su lote.
  - Lo único que el tope no corta es la invocación que lo cruza, que termina.
- Una reanudación vuelve a llamar solo a lo que no tiene entrada en caché: las evaluaciones perdidas
  por un fallo del proveedor, un 402 o el tope. Un fallo de contenido se repite desde la caché.

**Durante el estudio no se calcula ningún retorno de `solo`.** Hasta que el estudio cierra
(apartado 2), de un lote solo se miran sus contadores de validez y su número de evaluaciones.

## 5. Validez

**Las condiciones del criterio 6** (`criteria.py`, docstring del módulo) se aplican a cada lote,
con una diferencia: tres de ellas reanudan el lote en lugar de invalidarlo.

| condición del criterio 6 | en el estudio |
| --- | --- |
| el decisor perdió evaluaciones por cuota | invalida el lote |
| acierto de caché con un backend que no es el declarado | invalida el lote |
| un veto que no sea `invalid_stop_side` | invalida el lote |
| evaluación perdida por saldo insuficiente (402) | **no invalida**: se reanuda |
| evaluación perdida por un fallo del proveedor (transporte o plazo) | **no invalida**: se reanuda |
| evaluación cortada por el tope de gasto | **no invalida**: se reanuda |

- **Una configuración del decisor distinta de la registrada** en el `meta.json` del lote
  (apartado 3) invalida también el lote.
- **Un lote inválido** se corrige sin tocar el camino medido, la corrección se documenta en el
  registro, y el lote se vuelve a correr entero antes de su plazo. Si al vencer el plazo sigue
  inválido, el estudio es «no determinado».
  - Las pasadas del lote inválido quedan marcadas como sustituidas y se excluyen: sus registros no
    entran en el lote. Sustituirlas no deshace lo que termina el estudio (apartado 3). La
    repetición entera es una pasada nueva.
  - Los duplicados, un `run_id` con registro completo en dos pasadas, solo se buscan entre las
    pasadas no sustituidas.
- **Faltantes.** Una evaluación de `solo` que al vencer el plazo sigue sin decisión, por cualquier
  causa ajena a la respuesta del modelo —un fallo del proveedor, un 402, el tope, una pasada
  interrumpida o un lote que no se corrió—, cuenta como «faltante». Vale 0, sin posición, suma al
  5 % y se reporta. Las líneas base se puntúan como siempre.
- **Los registros de las líneas base se pueden completar después del plazo**, con una corrida solo
  de líneas base que no llama a ningún modelo.
- **Si las faltantes superan el 5 % de las evaluaciones del estudio, el estudio es «no
  determinado».**
- **Una respuesta inválida del decisor tras sus dos intentos** es una evaluación sin decisión que
  vale 0, como en la etapa 1. Se reporta y no cuenta para el 5 %: es del modelo, no del proveedor.
- **Bloque completo.** El lote del bloque es válido y está cerrado: todas sus activaciones tienen
  decisión de `solo` o una respuesta inválida, o venció su plazo.

**Política de upstream.**
- Cada llamada registra quién la sirvió (`LLMCall.upstream_model` y `upstream_endpoint`). En la
  etapa 1, `glm-5.2` lo sirvió `accounts/fireworks/models/glm-5p3` en sus 279 intentos
  (`docs/ablation.md`, «Quién respondió por el decisor»).
- **Si el upstream cambia, el estudio continúa.** El cambio se anota en el registro con el bloque y
  la primera llamada que lo muestra, y el resultado se reporta también separado por upstream, como
  dato descriptivo.
- **Si el id `glm-5.2` desaparece del catálogo de la pasarela** al empezar un lote —la comprobación
  `gateway` de `doctor`—, el estudio termina en ese bloque.

**Un estudio que termina antes de cerrar no da veredicto.** Termina así por la retirada del id, por
un hueco persistente en las velas (apartado 4) o por un lote que llamó a un modelo con el camino
medido cambiado (apartado 3). El análisis exige 50 bloques no vacíos completos (apartados 2 y 6),
así que de un estudio terminado antes solo se publican sus contadores de validez y su número de
evaluaciones.

## 6. Análisis

**Cuándo.**
- Una sola vez, cuando está completo el lote del bloque en que el estudio cierra (apartado 2): el
  50 si ningún bloque sale vacío, el 52 como mucho.
- Con el inicio del apartado 2, a más tardar al vencer el plazo de ese lote: el 2027-10-21T00:00Z
  si cierra en el bloque 50, el 2027-11-04T00:00Z si llega al 52.
- Antes no se calcula ningún retorno de `solo`.

**Con qué comando.** `python -m crypto_agents.forward analyze <directorio del estudio>`, que
escribe el bloque siguiente. Se niega a dar veredictos con menos de 50 bloques no vacíos
completos.

**El vector.**
- El retorno bruto de cada evaluación de `solo` y de `rule_trend`, emparejados por `run_id`.
- Las faltantes de `solo` valen 0.
- La resta es `metrics.paired_differences`.

**El intervalo** es el de la enmienda 4, con su función:
- `metrics.clustered_paired_difference` (`metrics.py:1128`), con un clúster por bloque de 7 días
  (`metrics.block_of`).
- El IC 95 % es `media ± 1.96 · EE usado`, con `EE usado = max(EE ingenuo, EE agrupado CR1)`.
- El IC 90 % es `media ± 1.645 · EE usado`, con el mismo error.
- **Con menos de 50 clústeres no hay intervalo** (`metrics.MIN_CLUSTERS = 50`, `metrics.py:1003`).

Cada bloque del estudio es exactamente un clúster, y un bloque vacío no aporta ninguno.
- Por eso el estudio cierra con 50 bloques no vacíos (apartado 2): al cerrar, G = 50, justo el
  umbral.
- Con 50 bloques fijos, una sola semana vacía habría dejado G = 49 y ningún intervalo. Con la regla
  de cierre, hacen falta tres semanas vacías en 52 para que falte.
- En el pool, ninguno de los 94 bloques completos estuvo vacío; el que menos tuvo, 15 activaciones
  (apartado 9, sección 3). Eso dice cómo fue el pasado, no cómo será el estudio.

**Veredictos de la comparación principal**, en este orden:

| veredicto | cuándo |
| --- | --- |
| no determinado | al terminar el bloque 52 no hay 50 bloques no vacíos, no hay intervalo (menos de 50 clústeres), las faltantes superan el 5 %, el estudio terminó antes de cerrar o un lote quedó inválido al vencer su plazo |
| **supera** | el IC 95 % queda entero por encima de 0 |
| **peor** | el IC 95 % queda entero por debajo de 0 |
| **empate** | el IC 90 % queda entero dentro de (−δ, +δ) **y** el IC 95 % incluye el 0 |
| **no concluyente** | cualquier otro caso |

δ = 0.40 % (0.004 como fracción). Los bordes son estrictos: un extremo del IC 90 % igual a ±δ no
cabe dentro.

**Por qué el empate no es el de la ablación.**
- La regla de la ablación (`criteria.classify`, `criteria.py:308`) llama «empate» a un IC 95 % que
  incluye el 0 con un semiancho de δ o menos.
- Ese intervalo puede llegar casi a 2δ por un lado: centrado en 0.9δ con semiancho δ va de −0.1δ a
  1.9δ, y es «empate» aunque la diferencia real compatible con los datos llegue a 1.9δ.
- Aquí el empate es una prueba de equivalencia estándar: dos pruebas de una cola al 5 %, que es el
  IC 90 % dentro de (−δ, +δ). Acota la diferencia real a ±δ, que es lo que la palabra promete.
- Exigir además que el IC 95 % incluya el 0 deja «supera» y «empate» excluyentes.
- La sustitución vale solo para este estudio. `criteria.classify`, `criteria.DELTA` y las reglas de
  la ablación no se tocan.

**Lo que significa cada veredicto**, escrito ahora:
- **supera**: en estas 50 semanas con activaciones, con este decisor, esta pasarela y este periodo,
  el retorno bruto por evaluación de `solo` (stop común, 6 velas) superó al de `rule_trend`, con el
  IC 95 % de la diferencia por encima de 0. El tamaño es el que diga el intervalo y nada más.
- **peor**: lo mismo con el signo contrario. `solo` rindió menos que la regla de tendencia.
- **empate**: la diferencia real está dentro de ±0.40 % por evaluación con un 90 % de confianza.
  Por encima de ese margen, `solo` no aporta nada sobre la regla de tendencia. **Un empate no
  descarta una ventaja real menor de 0.40 %.** Una ventaja de ese tamaño, en bruto por evaluación,
  podría tener valor económico, y este estudio no la distingue de cero.
- **no concluyente**: los datos no permiten ninguna de las tres afirmaciones anteriores.
- **no determinado**: el estudio no llegó a producir un intervalo que se pueda leer, por una de las
  causas de la tabla.

**Análisis descriptivos.** Ninguno entra en el veredicto ni lleva palabra de veredicto. Se publican
con su diferencia, su intervalo de la enmienda 4 y su n:
- `solo` frente a `always_buy`, `always_sell` y `random_uniform`. No se corrigen por multiplicidad;
  por eso son descriptivos.
- El retorno neto de la comparación principal: `outcomes.net_return` (`outcomes.py:426`) con
  `settings.DEFAULT_COSTS` y el funding extendido; las parejas sin funding se cuentan y se quitan.
- La comparación principal puntuada con el stop que declara cada brazo (stop propio).
- La comparación principal separada por el upstream que sirvió la llamada válida del decisor.
- Los contadores: evaluaciones por bloque, faltantes, respuestas inválidas, la mezcla de acciones de
  `solo` y los cambios de upstream.

## 7. Lo que el estudio no dirá

- **Es un estudio y un periodo.** Cincuenta semanas con activaciones en siete pares contra USDT y en
  4h. Con el inicio del apartado 2 van de 2026-10-22 a 2027-10-07 si ninguna sale vacía, y hasta
  2027-10-21 como mucho. Otro año, otros símbolos u otra temporalidad son otra pregunta.
- **Mide sobre precios de Binance spot, no de perpetuos.** Las velas son de spot (apartado 4), como
  en la etapa 1.
  - Operar perpetuos añade la base entre el precio del perpetuo y el de spot, y el funding.
  - El veredicto es bruto y no lleva ninguno de los dos.
  - El funding entra solo en el neto descriptivo. Ahí se aplica funding de perpetuos sobre precios
    de spot, que es una mezcla, igual que en la etapa 1.
- **Un «supera» no responde si operar es rentable.** Es bruto, sin comisiones, deslizamiento ni
  funding en el veredicto. No mide el tamaño de las posiciones, la exposición simultánea ni la
  cuenta. El neto descriptivo es una aproximación (`outcomes.net_return`), no una cuenta de
  resultados.
- **No dice nada de otros modelos ni de otros canales.** Ni de `full`, ni de otro decisor, ni de
  `glm-5.2` servido por otro proveedor o por otra pasarela, ni del runner en vivo, que el estudio no
  usa. Tampoco reabre la pregunta de arquitectura de la ablación.

## 8. Supuestos no medidos

- **El efecto de diseño es el de un sustituto.** 8.321 es el máximo sobre las seis diferencias
  entre dos líneas base (apartado 9, sección 2). `solo` discrepa de una línea base donde su decisor
  quiere, no donde una regla fija. Que su diferencia dependa en el tiempo igual es un supuesto.
- **`sigma_D` sale de las 140 evaluaciones de la etapa 1.** Su rango es el de un bootstrap, no un
  intervalo de confianza, y la potencia se mueve mucho dentro de él (apartado 9, sección 5).
- **El régimen de mercado puede cambiar.** Las activaciones por semana, el efecto de diseño y
  `sigma_D` son de 2024-10 a 2026-08. Otro régimen da otra densidad de activaciones y otra
  dispersión.
- **El proveedor puede actualizar los pesos del modelo bajo el mismo nombre de upstream.** La
  cabecera dice qué ruta respondió, no qué pesos; ese cambio no deja rastro en el journal.
- **Las probabilidades son de planificación.** Tratan el error estándar como conocido y usan la
  aproximación normal; en el estudio el error agrupado se estima con sus 50 clústeres. El coste y
  la pared son los de la etapa 1 por evaluación, escalados en línea recta.

## 9. Cálculo de planificación

`python -m crypto_agents.forward_plan var/ablation/20261009T045033Z` termina con código 0. Lo corrió
la laptop el 2026-10-16, dos veces y sin llamar a ningún modelo, y las dos salidas son idénticas
byte a byte (`cmp`). Reutiliza sin copiarlas las piezas de `design_effect` y `paired_dispersion`:
- el pool y sus series;
- el CR1 y el suelo `max(1, deff)` de la enmienda 4;
- la potencia de «supera» (`design_effect.planned_power`);
- `sigma_D`, el coste y la pared de la etapa 1.

Lo nuevo es el recuento por bloque, la probabilidad de «empate» con la regla del apartado 6, los
recuentos del pool, la comparación de las líneas base con los retornos publicados de la etapa 1 y
el tope proporcional.

Lo que fijó este documento antes de los números:
- el deff de planificación es el **máximo** de las seis diferencias entre líneas base sobre el pool
  entero con L = 7, con suelo `max(1, deff)`;
- solo cuentan los bloques completos, con los vacíos como 0;
- el n del estudio es 50 × la media por bloque; desde el cambio del cierre que se cuenta abajo, por
  bloque no vacío;
- las probabilidades se dan con δ = 0.40 %, el registrado, y con 0.20 %, 0.30 % y 0.50 % como
  referencia.

Lo que salió:

- **La sección 1 cuadra entera.**
  - Los recuentos del pool son los registrados: 4 740, 112, 4 737 y 4 626.
  - En las 140 de la etapa 1, las cuatro líneas base dan la acción, el stop y el retorno de su
    journal, y el retorno publicado en `tests/data/stage1_paired_returns.jsonl`.
- **El deff de planificación es 8.321**, de `always_sell - rule_trend`, con G = 96. Es el mismo
  máximo que publicó «Efecto de diseño (previo a la enmienda 4)», sección 3, y el deff usado es
  8.321.
- **Las semanas.** 94 bloques completos, del 2024-10-24 al que empieza el 2026-08-06, ninguno
  vacío. Media de 48.98 activaciones por bloque, la misma por bloque no vacío, p10 de 30, p90 de 70,
  mínimo 15 y máximo 98.
- **El n.** 50 bloques no vacíos dan 2 449 evaluaciones esperadas y un n efectivo de 294.3, 5.9 por
  bloque.
- **La comparación principal con δ = 0.40 %.**
  - P(«supera» | Δ = δ) = 87.6 % con `sigma_D` puntual, entre 72.9 % y 97.9 % sobre su rango
    bootstrap.
  - P(«empate» | Δ = 0) = 85.9 %, entre 64.5 % y 95.0 %.
- **El coste.**
  - El estudio esperado, 50 bloques no vacíos: 36.96 USD y 39.5 horas de pared, con la pared que
    midió la etapa 1.
  - El tope de un lote es 3 × sus evaluaciones × 0.01509 USD: 2.22 USD el lote medio, 3.17 el del
    p90 y 4.44 el más grande del pool.
  - La cota que dan las reglas: 3 × las 2 449 evaluaciones esperadas × 0.01509 USD = 110.87 USD. Lo
    único que el tope no corta es la invocación que lo cruza.

Nada de eso hace inviable el diseño.

**Lo que cambió después de ver estas cifras**, antes de cualquier dato del estudio. El 2026-10-16
el operador cambió:
- **El cierre** (apartado 2): de exactamente 50 bloques a 50 bloques no vacíos, con el 52 como
  máximo. Lo motivó una consecuencia de las reglas, no una cifra: con 50 bloques fijos, una sola
  semana vacía dejaba el estudio sin intervalo. Con él, el n es 50 × la media por bloque no vacío,
  que en este pool sin bloques vacíos es la misma cifra, así que ninguna probabilidad cambió.
- **La fuente de la configuración registrada del decisor** (apartado 3): el `meta.json` de la
  etapa 1 en lugar de `.env`, comprobada en cada lote.
- **Cómo se fija el renderizado del prompt de `solo`** (apartado 3): por su salida, con 14 digests
  del journal de la etapa 1 y su prueba, en lugar del sha-256 del código fuente de las tres
  funciones, que un cambio de espaciado rompía. La comprobación sobre las 140 de la etapa 1
  coincidió en las 140.
- **Lo que viaja al modelo además del prompt** (apartado 3): la envoltura de la petición del
  decisor, con su sha-256, comprobada en cada lote. Ningún digest registrado la cubría.
- **El tope de gasto por lote** (apartado 4): de 3 × el coste semanal esperado, 2.22 USD fijos, a
  3 × las evaluaciones del manifiesto del lote × 0.01509 USD. Con un tope fijo, una semana con más
  de unas 147 activaciones dejaba faltantes en las semanas más volátiles, que no faltan al azar. La
  cota del estudio pasó de 111.00 a 110.87 USD.
- **Dos comprobaciones más en cada lote** (apartados 3 y 10): los recuentos del pool y la atadura
  de las líneas base. La atadura compara además con los retornos publicados de la etapa 1, porque
  contra el journal no ve un cambio en la puntuación. Con ellas, cada fila de la tabla «El mercado,
  la puntuación y las líneas base» dice qué comprobación la fija.
- **Qué termina el estudio** (apartado 3): un lote que llama a un modelo con el camino medido
  cambiado. Una comprobación previa que falla y se restaura antes de llamar no lo termina. Durante
  el estudio no se actualizan dependencias, y el `meta` de cada lote registra `uv.lock` y la
  versión de Python.
- **De dónde sale δ** (apartado 1) y **qué deja abierto un empate** (apartado 6): se escribió, sin
  cambiar ninguno de los dos.
- **El instante del merge** (apartado 2): `git merge --no-ff`, subido a GitHub antes de
  2026-10-22T00:00Z, con la fecha de GitHub como evidencia externa.
- **De dónde leen las comprobaciones por lote** (apartado 10, punto 9): de una copia versionada de
  la etapa 1 en `data/stage1/`, no de `var/`.
- **Dos reglas para las velas** (apartado 4), añadidas el 2026-10-17, antes del bloque 1, porque el
  documento no decía qué pasaba en esos dos casos:
  - la vela revisada, que se anota en el registro y sigue con el valor guardado mediante un flag
    que la nombra;
  - el hueco persistente, que termina el estudio si una vela nueva sigue faltando al volver a
    descargar 24 h después.
- **Tres reglas para los lotes** (apartado 5), añadidas el 2026-10-18, antes del bloque 1, porque
  el documento no decía qué pasaba con un lote no corrido, interrumpido o repetido:
  - una evaluación de `solo` sin decisión al vencer el plazo, por cualquier causa ajena a la
    respuesta del modelo, es faltante: vale 0 y suma al 5 %;
  - las líneas base se pueden completar después del plazo, con una corrida que no llama a ningún
    modelo;
  - las pasadas de un lote inválido quedan sustituidas y se excluyen, la repetición entera es una
    pasada nueva y los duplicados solo se buscan entre pasadas no sustituidas.

δ, la regla de empate, el límite del 5 % de faltantes, el plazo de 14 días y el cierre no cambiaron.

Lo que sigue es la salida del comando, sin editar, con los títulos un nivel más abajo.

Reproduce: `python -m crypto_agents.forward_plan var/ablation/20261009T045033Z`

- commit de la corrida: `b9676eb91627a3d53b0e950d5866c3e0af06f671` (árbol limpio)
- plan: `data/ablation_selection.json` (manifest), sha-256 `73f87870cf24d06c341ff75fa72f164f9cd5202623cf0d76babbdab4a2803f23`
- `full.jsonl` sha-256 `5d7f1a63f3aaf1c8469a572aa3aaff6184e671ff67698a1d3632a15645ada33f`
- `solo.jsonl` sha-256 `1180554e2c19aab20d623f2112547aae8f930ad934d3fc17ca65fd5c70cc4b17`
- `always_buy.jsonl` sha-256 `77fe118e5efb5714384c3d3485baf31a9c80bdbded2c34f2b775c9667fddd1ff`
- `always_sell.jsonl` sha-256 `ba1643fb2ce45114f728418e8894eb46b55f125fe2cef1fe0211e273746bc1d9`
- `random_uniform.jsonl` sha-256 `607b738ea2ee764d4ee8433adc0cac748f738761dfcea418f1a6690cd6ff6198`
- `rule_trend.jsonl` sha-256 `3b8d58f9ce1b22699a7f0d92b2dd47e91235c3db9f0ef3b53d3bed215b19ab93`
- histórico `ADA/USDT`: `data/history/adausdt_4h.csv`, sha-256 `93b6462816a9f606450e5b5a1440b1b2cac46ef0ede9a56b1700d26c1b58255d`
- histórico `BNB/USDT`: `data/history/bnbusdt_4h.csv`, sha-256 `65ede8a9b4500197986bc5e1f922c5f59a6f33f39b359e0550dc6be080cb8375`
- histórico `BTC/USDT`: `data/history/btcusdt_4h.csv`, sha-256 `d1b7e107de43199c7b80e51afd66c660f1f89dc98c2d0f639c580f325b225ad4`
- histórico `DOGE/USDT`: `data/history/dogeusdt_4h.csv`, sha-256 `5c264c914a0db15d5563e7dc47f3db9fb4fbbde8268364ea11baf319ec4ecfff`
- histórico `ETH/USDT`: `data/history/ethusdt_4h.csv`, sha-256 `5ea175a55e90112f6f10510e4fe67d328dfd2795f6a9571e49e6f29ac34c5404`
- histórico `SOL/USDT`: `data/history/solusdt_4h.csv`, sha-256 `56517823a5b88b279ed3c15f2d78d491f15ddfa0fcf1280f17e19e436f1dfc3e`
- histórico `XRP/USDT`: `data/history/xrpusdt_4h.csv`, sha-256 `e907bb3e44bc5587b0736e6cc4604f15fae36b3dc46cc185bf3566cb1662348a`
- δ registrado del estudio = 0.004 (0.400% de retorno bruto por evaluación); de referencia, 0.200%, 0.300%, 0.500%. Horizonte 6 velas de 4h, stop común, por `criteria.paired_returns`.
- bloques: 7 días UTC alineados al epoch, que empiezan los jueves a las 00:00 UTC; son los clústeres de la enmienda 4 (`criteria.CLUSTER_DAYS`). A cada evaluación la asigna el cierre de su vela; un instante en el borde abre el bloque nuevo.
- `random_uniform` con la semilla del manifiesto de la corrida, 20260815
- `sigma_D` de `paired_dispersion`: bootstrap de 10000 remuestreos, semilla 20261011

### 1. Atadura con la corrida y con lo publicado

Tres comprobaciones del mercado y de la puntuación, sin modelos. Si una no cuadra, el comando sale con 1.

**Los recuentos del pool** sobre los históricos de arriba: las activaciones del barrido (`activation_sweep.activated_bars`), las que el gate no abre con la ventana del replay (`selection.confirm_activation`), las que tienen el horizonte por delante (`selection.stratify`) y, de estas, las confirmadas. Tienen que ser las registradas (`forward_plan.POOL_COUNTS`).

| | activaciones del barrido | sin confirmar | con el horizonte por delante | confirmadas |
| --- | ---: | ---: | ---: | ---: |
| estos históricos | 4740 | 112 | 4737 | 4626 |
| registrados | 4740 | 112 | 4737 | 4626 |

- coinciden: sí

**Las líneas base contra su journal.** Las cuatro deciden sobre cada activación confirmada del pool con las funciones de `baselines.py` (`design_effect.pool_records`). En las evaluaciones que la corrida ya hizo tienen que dar la acción, el stop y el retorno de su journal (`design_effect.stage_ties`). El retorno de los dos lados lo calcula la puntuación de hoy.

| línea base | evaluaciones en la corrida | de ellas, en el pool | acción distinta | stop distinto | retorno distinto | coincide |
| --- | --- | --- | --- | --- | --- | --- |
| always_buy | 140 | 140 | 0 | 0 | 0 | sí |
| always_sell | 140 | 140 | 0 | 0 | 0 | sí |
| random_uniform | 140 | 140 | 0 | 0 | 0 | sí |
| rule_trend | 140 | 140 | 0 | 0 | 0 | sí |

**Las líneas base contra los retornos publicados.** La tabla anterior no ve un cambio en la puntuación, que mueve los dos lados a la vez. Esta puntúa el pool con `criteria.paired_returns` y lo compara, evaluación por evaluación, con el retorno bruto que publicó la etapa 1: `tests/data/stage1_paired_returns.jsonl`, sha-256 `b521801e0545d3965d3f96c0fba32debd92b30988af6d31e3cfa1d034e55841f`.

| línea base | evaluaciones publicadas | de ellas, en el pool | retorno distinto | coincide |
| --- | --- | --- | --- | --- |
| always_buy | 140 | 140 | 0 | sí |
| always_sell | 140 | 140 | 0 | sí |
| random_uniform | 140 | 140 | 0 | sí |
| rule_trend | 140 | 140 | 0 | sí |

Reproduce: `python -m crypto_agents.forward_plan var/ablation/20261009T045033Z`

### 2. Efecto de diseño del pool entero con L = 7

La regla, fijada antes de los números: el deff de planificación es el **máximo** del deff de las seis diferencias entre líneas base sobre las 4626 activaciones confirmadas del pool, con CR1 y un clúster por bloque de 7 días; el **deff usado** es `max(1, deff)`, el suelo de la enmienda 4 (`design_effect.used_deff`). Las diferencias entre líneas base son un sustituto de `solo` menos una línea base, que no se puede calcular sobre el pool sin pagar.

| serie | L (días) | n | G | EE ingenuo | EE agrupado | deff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| always_buy - always_sell | 7 | 4626 | 96 | 0.117% | 0.314% | 7.239 |
| always_buy - random_uniform | 7 | 4626 | 96 | 0.076% | 0.162% | 4.623 |
| always_buy - rule_trend | 7 | 4626 | 96 | 0.068% | 0.151% | 4.973 |
| always_sell - random_uniform | 7 | 4626 | 96 | 0.076% | 0.170% | 4.992 |
| always_sell - rule_trend | 7 | 4626 | 96 | 0.074% | 0.213% | 8.321 |
| random_uniform - rule_trend | 7 | 4626 | 96 | 0.061% | 0.110% | 3.232 |

- deff de planificación: 8.321, de `always_sell - rule_trend`
- deff usado: 8.321

Reproduce: `python -m crypto_agents.forward_plan var/ablation/20261009T045033Z`

### 3. Activaciones confirmadas por bloque

Bloques completos: los de 7 días que caen enteros en el tramo en que las siete series pueden dar una evaluación —desde el cierre de la primera vela tras el warm-up hasta el de la última con el horizonte por delante—, del 2024-10-24T00:00Z al que empieza el 2026-08-06T00:00Z. Un bloque completo sin activaciones cuenta 0 en la media por bloque y queda fuera de la media por bloque no vacío. Fuera quedan los bloques parciales de los bordes, con sus activaciones: 2024-10-17T00:00Z (15), 2026-08-13T00:00Z (7). p10 y p90 por rango más cercano.

| bloques completos | vacíos | activaciones | media por bloque | media por bloque no vacío | p10 | p90 | mínimo | máximo |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 94 | 0 | 4604 | 48.98 | 48.98 | 30 | 70 | 15 | 98 |

Reproduce: `python -m crypto_agents.forward_plan var/ablation/20261009T045033Z`

### 4. n efectivo

n efectivo = evaluaciones / deff usado. El estudio cierra con 50 bloques no vacíos, así que las evaluaciones esperadas son 50 x la media por bloque no vacío, redondeado al entero: es el n de la sección 5.

| | evaluaciones | deff usado | n efectivo |
| --- | ---: | ---: | ---: |
| por bloque no vacío | 48.98 | 8.321 | 5.9 |
| a los 50 bloques no vacíos | 2449 | 8.321 | 294.3 |

Reproduce: `python -m crypto_agents.forward_plan var/ablation/20261009T045033Z`

### 5. Probabilidad de cada veredicto para `solo` contra cada línea base

Con `EE = sigma_D·√(deff usado / n)` y la media observada normal alrededor de la diferencia real Δ:

- **supera si Δ = δ**: P(IC 95 % entero por encima de 0) = `Φ(δ/EE - 1.96)`, `design_effect.planned_power`.
- **empate si Δ = 0**: P(IC 90 % dentro de (-δ, +δ) y IC 95 % con el 0) = P(a < media < b), con `a = max(-δ + 1.645·EE, -1.96·EE)` y `b = min(δ - 1.645·EE, 1.96·EE)`; 0 si `a ≥ b`.
- `sigma_D` es la de `paired_dispersion`: puntual y los dos extremos de su rango bootstrap. n = 2449 y el deff usado de la sección 2.
- La comparación principal es `solo` - `rule_trend`; las demás son descriptivas.

| comparación | brazo | referencia | sigma_D | valor | supera si Δ = δ, δ = 0.200% | empate si Δ = 0, δ = 0.200% | supera si Δ = δ, δ = 0.300% | empate si Δ = 0, δ = 0.300% | supera si Δ = δ, δ = 0.400% (registrado) | empate si Δ = 0, δ = 0.400% (registrado) | supera si Δ = δ, δ = 0.500% | empate si Δ = 0, δ = 0.500% |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| principal | solo | rule_trend | extremo bajo | 1.717% | 51.5% | 27.6% | 85.0% | 82.4% | 97.9% | 95.0% | 99.9% | 95.0% |
| principal | solo | rule_trend | puntual | 2.203% | 34.4% | 0.0% | 64.7% | 51.1% | 87.6% | 85.9% | 97.3% | 95.0% |
| principal | solo | rule_trend | extremo alto | 2.670% | 25.0% | 0.0% | 48.7% | 22.3% | 72.9% | 64.5% | 89.5% | 88.3% |
| descriptiva | solo | always_buy | extremo bajo | 3.354% | 17.4% | 0.0% | 33.5% | 0.0% | 53.4% | 31.2% | 72.5% | 63.9% |
| descriptiva | solo | always_buy | puntual | 4.087% | 13.1% | 0.0% | 24.2% | 0.0% | 38.9% | 2.7% | 55.5% | 35.0% |
| descriptiva | solo | always_buy | extremo alto | 4.765% | 10.7% | 0.0% | 18.9% | 0.0% | 30.2% | 0.0% | 43.6% | 12.3% |
| descriptiva | solo | always_sell | extremo bajo | 3.687% | 15.2% | 0.0% | 28.6% | 0.0% | 46.1% | 17.1% | 64.3% | 50.4% |
| descriptiva | solo | always_sell | puntual | 4.440% | 11.8% | 0.0% | 21.2% | 0.0% | 33.9% | 0.0% | 48.9% | 22.6% |
| descriptiva | solo | always_sell | extremo alto | 5.126% | 9.8% | 0.0% | 17.0% | 0.0% | 26.7% | 0.0% | 38.7% | 2.3% |
| descriptiva | solo | random_uniform | extremo bajo | 3.572% | 15.9% | 0.0% | 30.2% | 0.0% | 48.4% | 21.8% | 67.1% | 55.1% |
| descriptiva | solo | random_uniform | puntual | 4.312% | 12.2% | 0.0% | 22.2% | 0.0% | 35.6% | 0.0% | 51.2% | 26.9% |
| descriptiva | solo | random_uniform | extremo alto | 4.963% | 10.2% | 0.0% | 17.8% | 0.0% | 28.2% | 0.0% | 40.8% | 6.7% |

Reproduce: `python -m crypto_agents.forward_plan var/ablation/20261009T045033Z` (sigma_D: `python -m crypto_agents.paired_dispersion var/ablation/20261009T045033Z`)

### 6. Evaluaciones, coste y pared

- USD por evaluación medido: 0.01509, la cota superior de `solo` en la corrida (`paired_dispersion`, sección 6); las cuatro líneas base valen 0.
- USD por evaluación registrado para el tope: 0.01509 (`forward_plan.USD_PER_EVALUATION`).
- segundos de pared por evaluación: 58.1, los de `solo` en la corrida (`paired_dispersion`, sección 7), en la máquina que la corrió.
- **tope de gasto de un lote**: 3 x las evaluaciones de su manifiesto x 0.01509 USD, calculado antes de su primera llamada y acumulado entre sus reanudaciones (`forward_plan.batch_spend_cap`). Un bloque vacío no llama a nadie.
- **cota del gasto del estudio que dan las reglas**: la suma de los topes de sus lotes, 3 x las evaluaciones del estudio x 0.01509 USD; con las 2449 esperadas, 110.87 USD. Lo único que el tope no corta es la invocación que lo cruza, que termina.
- la tabla da, para el lote medio, el p90 y el máximo de los bloques completos del pool, y para el estudio esperado (50 bloques no vacíos a su media), el coste medido y el tope. El tope de la última fila es la cota del estudio.

| | evaluaciones | USD | tope | horas de pared |
| --- | ---: | ---: | ---: | ---: |
| lote medio | 48.98 | 0.74 | 2.22 | 0.8 |
| lote p90 | 70 | 1.06 | 3.17 | 1.1 |
| lote máximo del pool | 98 | 1.48 | 4.44 | 1.6 |
| el estudio esperado, 50 bloques no vacíos | 2449 | 36.96 | 110.87 | 39.5 |

Reproduce: `python -m crypto_agents.forward_plan var/ablation/20261009T045033Z` (coste y pared: `python -m crypto_agents.paired_dispersion var/ablation/20261009T045033Z`)

### 7. Calendario

- regla: el primer bloque es el que empieza estrictamente después del merge de `docs/forward.md` en master, y nunca antes de 2026-10-22T00:00Z. Un merge justo en el borde de un bloque abre el siguiente.
- cierre: el estudio cierra en el bloque en que acumula 50 bloques no vacíos; si al terminar el bloque 52 no los acumula, queda «no determinado».
- con el merge antes de 2026-10-22T00:00Z: bloque 1 = [2026-10-22T00:00Z, 2026-10-29T00:00Z). Fin esperado, si ningún bloque sale vacío: bloque 50 = [2027-09-30T00:00Z, 2027-10-07T00:00Z). Fin más tardío posible: bloque 52 = [2027-10-14T00:00Z, 2027-10-21T00:00Z).
- el lote de un bloque se corre desde su cierre más 6 velas de 4h: el del bloque 1, desde 2026-10-30T00:00Z; el del bloque 50, desde 2027-10-08T00:00Z; el del bloque 52, desde 2027-10-22T00:00Z.
- plazo de un lote: 14 días tras el cierre de su bloque. El del bloque 1 vence el 2026-11-12T00:00Z; el del bloque 50, el 2027-10-21T00:00Z; el del bloque 52, el 2027-11-04T00:00Z.

Reproduce: `python -m crypto_agents.forward_plan var/ablation/20261009T045033Z`

### Lo que estas cifras suponen

- El deff es el de un sustituto: diferencias entre dos políticas sin modelo. Que `solo` menos una línea base dependa en el tiempo igual es un supuesto.
- `sigma_D` sale de las 140 evaluaciones de la etapa 1 y su rango es el de un bootstrap, no un intervalo de confianza.
- Las activaciones por bloque, el deff y `sigma_D` son del periodo del histórico; el régimen de mercado del estudio puede ser otro.
- Las probabilidades tratan el error estándar como conocido y usan la aproximación normal; en el estudio el EE agrupado se estima con sus 50 clústeres.
- El coste y la pared son los de la etapa 1 por evaluación, escalados en línea recta.
- Ninguna cifra de este reporte es una media de retorno.

## 10. Requisitos para el bloque de implementación

El bloque siguiente escribe los lotes y el análisis, y cumple lo siguiente. Cada punto lleva su
test.

1. **Un comando de lote**: `python -m crypto_agents.forward batch --block <inicio del bloque>`.
   1. Extiende y verifica el histórico y el funding del apartado 4.
   2. Escribe el manifiesto del bloque con toda la población.
   3. Hace las comprobaciones previas del punto 4, sin llamar a ningún modelo.
   4. Comprueba que `glm-5.2` está en el catálogo de la pasarela.
   5. Corre `solo` y las cuatro líneas base con `--fill` y el tope del lote,
      `forward_plan.batch_spend_cap` de su manifiesto, acumulado entre reanudaciones.
   6. Reanuda.
   7. No puntúa ni imprime ningún retorno: solo contadores de validez y número de evaluaciones.
      Un test lo fija.
2. **δ registrado por estudio.**
   - El δ de este estudio, 0.004, vive con el estudio.
   - `criteria.DELTA = 0.002` es el de la ablación, y ni él ni `criteria.classify` se tocan.
   - La regla de empate del apartado 6 es una función propia, con el IC 90 %.
   - Se reutiliza el intervalo de `metrics.clustered_paired_difference`.
3. **Cerrar con la regla del apartado 2.** Antes de calcular nada, el comando de análisis cuenta
   los bloques no vacíos con `forward_plan.closing_block`.
   - Sin 50 bloques no vacíos completos, no da veredicto.
   - Si el bloque 52 terminó sin ellos, el estudio es «no determinado».
4. **El `meta` de cada lote** registra:
   - el hostname, el commit y si el árbol estaba limpio;
   - el sha-256 de `uv.lock` y la versión de Python (apartado 3);
   - los sha-256 del camino medido del apartado 3, calculados como allí se dice: la plantilla, el
     esquema de `Proposal`, la plantilla de reintento y la envoltura de la petición del decisor;
   - la configuración del decisor que leyó: los campos de la tabla «El decisor» del apartado 3,
     `max_attempts` incluido, que el `meta.json` de la etapa 1 no lleva;
   - las evaluaciones de su manifiesto y su tope;
   - el resultado de cada comprobación previa.

   Antes de la primera llamada, cada lote hace cinco comprobaciones, sin modelos y sobre archivos
   versionados (punto 9).
   - **Los sha-256.** Comprueba que son los registrados aquí.
   - **El renderizado y la petición.** Repite la comprobación del apartado 3 con
     `forward_render.check_run` sobre las 140 de la etapa 1, con el decisor que leyó: los 140
     digests tienen que coincidir y la envoltura tiene que ser la registrada.
   - **La atadura de las líneas base.** Las cuatro líneas base, sobre las 140 de la etapa 1, dan
     la misma acción, el mismo stop y el mismo retorno que sus journals, y el retorno publicado en
     `tests/data/stage1_paired_returns.jsonl` (sha-256
     `b521801e0545d3965d3f96c0fba32debd92b30988af6d31e3cfa1d034e55841f`). Es la sección 1 de
     `forward_plan`, reutilizada y no copiada: `design_effect.stage_ties` y
     `forward_plan.published_ties`.
   - **Los recuentos del pool** sobre `data/history/`, el original y no el extendido: 4 740
     activaciones, 112 sin confirmar, 4 737 con el horizonte por delante y 4 626 confirmadas
     (`forward_plan.pool_counts` contra `forward_plan.POOL_COUNTS`).
   - **La configuración.** Compara su `meta.json` con la configuración registrada del apartado 3,
     y su manifiesto con los parámetros de `data/ablation_selection.json`: los siete símbolos, 4h,
     `candle_limit` 500, horizonte 6, warm-up 400 y semilla 20260815.

   Si falla una de las cuatro primeras, el lote se niega a correr y dice que el camino medido
   cambió. Si la configuración no coincide, el lote es inválido, no llama a nadie y le aplica la
   regla de lote inválido del apartado 5. En los dos casos no se ha llamado a ningún modelo: si el
   camino registrado se restaura, el estudio sigue, y el registro anota la fecha, la causa y la
   restauración (apartado 3). Un lote que llama a un modelo con el camino medido cambiado termina
   el estudio.
5. **El histórico y el funding extendidos.**
   - Viven aparte de `data/history/` y `data/funding/`, se versionan, y solo crecen por la derecha.
   - Sus primeras filas son byte a byte las publicadas.
   - Su procedencia queda escrita como la de `data/funding/README.md`.
6. **El upstream y el registro.** El comando de lote informa de los upstreams del decisor que vio.
   Un cambio respecto al lote anterior queda en el «Registro del estudio».
7. **El análisis** produce exactamente lo del apartado 6:
   - el veredicto de la comparación principal;
   - los descriptivos sin palabra de veredicto;
   - las faltantes y su proporción;
   - el reparto por upstream.
8. **Una sola construcción del contexto.** Antes del primer lote se unifican en una función las
   tres copias que arman el `AgentContext` de una entrada del plan:
   - `ablation.dry_run`;
   - `zen_probe.prepare_activations`;
   - `forward_render.solo_context`.

   Su prueba de regresión es la comprobación de las 140 del apartado 3:
   `python -m crypto_agents.forward_render` sobre la etapa 1 tiene que seguir terminando con
   código 0 y 140 de 140. Hoy lee `var/ablation/20261009T045033Z`; con el punto 9, `data/stage1/`.
9. **Las comprobaciones por lote leen archivos versionados, no `var/`.**
   `var/ablation/20261009T045033Z` no está versionado y solo existe en la laptop. Antes del primer
   lote se versiona en `data/stage1/` una copia, con los mismos bytes, de los seis archivos que
   leen las comprobaciones:

   | archivo | bytes | sha-256 |
   | --- | ---: | --- |
   | `meta.json` | 10 082 | `a9d4b2c3ff84a5c895eae2afa366e58bc89838758a0a9471cf5b2f9214aae06c` |
   | `solo.jsonl` | 354 173 | `1180554e2c19aab20d623f2112547aae8f930ad934d3fc17ca65fd5c70cc4b17` |
   | `always_buy.jsonl` | 184 032 | `77fe118e5efb5714384c3d3485baf31a9c80bdbded2c34f2b775c9667fddd1ff` |
   | `always_sell.jsonl` | 184 160 | `ba1643fb2ce45114f728418e8894eb46b55f125fe2cef1fe0211e273746bc1d9` |
   | `random_uniform.jsonl` | 179 543 | `607b738ea2ee764d4ee8433adc0cac748f738761dfcea418f1a6690cd6ff6198` |
   | `rule_trend.jsonl` | 172 096 | `3b8d58f9ce1b22699a7f0d92b2dd47e91235c3db9f0ef3b53d3bed215b19ab93` |

   - **El tamaño.** Los cinco journals suman 1 074 004 bytes, y con `meta.json`, 1 084 086: menos
     de 5 MB. Medido con `du -b` el 2026-10-16 en la laptop.
   - **Los sha-256 son los publicados.** Los de los journals están en «Etapa 1 — resultado» de
     `docs/ablation.md` y en el apartado 9; el de `meta.json`, en el apartado 3.
   - **No llevan credenciales.** Medido el 2026-10-16 en la laptop con un script propio, no
     versionado, que solo imprime recuentos:
     - ninguno de los siete archivos de la corrida contiene el valor de `CA_OPENAI__API_KEY`,
       `CA_EXCHANGE__API_KEY` ni `CA_EXCHANGE__API_SECRET` de `.env`;
     - ninguno contiene `authorization`, `bearer`, `api_key`, `secret`, `password`, `cookie`,
       `x-api-key` ni `x-opencode-session`, sin distinguir mayúsculas;
     - `meta.json` lleva `argv`, el `base_url` público y el commit, y ninguna ruta del equipo.
   - **`full.jsonl` no hace falta.** Ninguna comprobación lo lee, y `read_stage` lee un brazo sin
     journal como vacío. Son 1 826 279 bytes que no se versionan.
   - **La copia funciona.** Comprobado el 2026-10-16 sobre una copia de los seis archivos en un
     directorio aparte:
     - `read_stage` la lee, con `full` vacío y los otros cinco con 140 evaluaciones;
     - `python -m crypto_agents.forward_render` sale con código 0 y 140 de 140;
     - con un script propio, no versionado, la puntuación de hoy da el retorno publicado de `solo`
       y de las cuatro líneas base en 700 de 700.
   - Cada comprobación lee de `data/stage1/` y comprueba antes los sha-256 de lo que lee.
   - En el bloque de este documento no se versiona nada: solo se decide.

## Registro del estudio

### 2026-10-17 — Merge de este documento e inicio del estudio

- **Commit de merge en master:** `ce3afe2f172f152ec91ca87a836e5359bd081837`. Tiene dos padres:
  `af85dd4bda46b0ee9ed67aa5c0530382f72a5497` (master) y
  `015f546de37024acc3d1631ccc9076e097fdef80` (la rama del documento).
- **Fecha de committer:** 2026-10-17T11:12:43-06:00, es decir 2026-10-17T17:12:43Z. También la
  puso GitHub: el merge es un `pr_merge` hecho en la web, y su committer es
  `GitHub <noreply@github.com>`.
- **Push en GitHub:** 2026-10-17T17:12:44Z.
  - Es el evento `pr_merge` de `refs/heads/master`, con `before` `af85dd4…` y `after`
    `ce3afe2…`, en `GET https://api.github.com/repos/AlexMartGong/Cripto_Agents/activity`.
  - Se leyó el 2026-10-17 en la laptop con `curl`, sin autenticar, porque `gh` no está instalado
    allí. El operador aprobó esta fuente: el endpoint `/activity` con `curl`.
- **Inicio resultante:** las dos fechas caen en el bloque [2026-10-15T00:00Z, 2026-10-22T00:00Z).
  El bloque 1 es el siguiente, [2026-10-22T00:00Z, 2026-10-29T00:00Z). Es lo que da
  `forward_plan.first_block`, y lo fija `forward.FIRST_BLOCK`.

### 2026-10-17 — Los artefactos de cada lote se versionan

- **Qué cambia.** Tras completar cada lote se versionan y se suben sus artefactos:
  - el manifiesto del bloque;
  - el `meta` de cada pasada;
  - los journals;
  - el histórico y el funding extendidos.
- Es una corrección que no toca ninguna pieza del camino medido (apartado 3). Va en el mismo commit
  que añade el comando de lote, `python -m crypto_agents.forward batch`.
- **Por qué.** Sin esos archivos en el repositorio, el análisis dependería de un directorio que
  solo existe en la laptop, como pasaba con la etapa 1 hasta `data/stage1/`.

### 2026-10-17 — La última vela de `data/history/` se guardó en formación: velas revisadas

- **Qué se encontró.** Se descargaron las siete series desde 2024-08-16 hasta la vela que cierra
  el 2026-10-17T20:00Z, en la laptop, con lectura pública y sin modelos, y se compararon con
  `data/history/`.
  - Las filas 1 a 4 379 coinciden exactas en las siete series.
  - La fila 4 380 es distinta en las siete. Es la vela que abre el 2026-08-15T20:00Z.
  - Se guardó cuando aún estaba en formación: `download_history` se pidió hasta un instante
    anterior a su cierre.
- **Es una vela revisada** (apartado 4, «Vela revisada»): una vela ya guardada que el exchange
  devuelve distinta. Se anota aquí y la serie del estudio es la que se guardó.
  - El histórico extendido conserva byte a byte las 4 380 filas de `data/history/`.
  - El primer lote se detendrá en estas siete velas con el código 5, y se continúa con
    `--accept-revised` para cada una. La aceptación queda en `data/forward/revisions.json`, con el
    valor guardado y el devuelto.
  - Lo decidió el operador el 2026-10-17, antes del bloque 1. Antes había elegido tomar la vela
    cerrada del exchange, y lo cambió al escribir la regla del apartado 4: con ella no hay
    excepción a las 4 380 filas.
- **Las siete velas.** Los valores devueltos son los de la descarga del 2026-10-17.
  - **ADA/USDT**, vela que abre el 2026-08-15T20:00Z:
    - guardado: apertura 0.1781, máximo 0.1783, mínimo 0.1772, cierre 0.1778, volumen 2490116.8;
    - devuelto: apertura 0.1781, máximo 0.1783, mínimo 0.1758, cierre 0.1765, volumen 7057240.3.
  - **BNB/USDT**, vela que abre el 2026-08-15T20:00Z:
    - guardado: apertura 611.41, máximo 611.67, mínimo 609.3, cierre 609.45, volumen 4102.006;
    - devuelto: apertura 611.41, máximo 611.67, mínimo 607.5, cierre 607.76, volumen 7581.756.
  - **BTC/USDT**, vela que abre el 2026-08-15T20:00Z:
    - guardado: apertura 63126.07, máximo 63168.95, mínimo 63076.96, cierre 63156.34, volumen
      290.08793;
    - devuelto: apertura 63126.07, máximo 63175.0, mínimo 63076.96, cierre 63086.01, volumen
      407.21827.
  - **DOGE/USDT**, vela que abre el 2026-08-15T20:00Z:
    - guardado: apertura 0.06995, máximo 0.07009, mínimo 0.06987, cierre 0.06991, volumen
      10323024.0;
    - devuelto: apertura 0.06995, máximo 0.07009, mínimo 0.06951, cierre 0.06961, volumen
      30219590.0.
  - **ETH/USDT**, vela que abre el 2026-08-15T20:00Z:
    - guardado: apertura 1884.55, máximo 1885.8, mínimo 1882.8, cierre 1885.03, volumen 4345.5732;
    - devuelto: apertura 1884.55, máximo 1885.8, mínimo 1881.98, cierre 1882.64, volumen 7295.586.
  - **SOL/USDT**, vela que abre el 2026-08-15T20:00Z:
    - guardado: apertura 75.57, máximo 75.74, mínimo 75.42, cierre 75.59, volumen 57552.37;
    - devuelto: apertura 75.57, máximo 75.74, mínimo 75.31, cierre 75.35, volumen 85072.665.
  - **XRP/USDT**, vela que abre el 2026-08-15T20:00Z:
    - guardado: apertura 1.004, máximo 1.0068, mínimo 1.0032, cierre 1.0051, volumen 1750224.9;
    - devuelto: apertura 1.004, máximo 1.0068, mínimo 1.0012, cierre 1.0028, volumen 3009949.6.
- **Lo que implica.**
  - `data/history/` no cambia, y los recuentos del pool se siguen comprobando sobre él.
  - Ninguna entrada de la etapa 1 llega a esa fila: la más tardía es la fila 4 371, y su horizonte
    termina en la 4 377.
  - Las ventanas de 500 velas de los bloques 1 a 3, hasta alrededor del 2026-11-07, contienen esa
    vela tal como se guardó, en formación.
- **El funding no tiene el problema.** Las siete series descargadas hasta el 2026-10-17T08:00Z
  empiezan byte a byte por las de `data/funding/`.

### 2026-10-17 — Lo que cambia la fila 4 380 guardada en formación

Dos comprobaciones sobre esa vela, en la laptop, sin llamar a ningún modelo y sin calcular ningún
retorno de `solo`.

- **Horizontes que la tocan.** Una evaluación tiene la fila 4 380 en su horizonte de 6 velas si está
  en la fila 4 374 (índice 4 373).
  - Ninguna de las 140 entradas de `data/ablation_selection.json` está ahí.
  - Del pool de 4 626 activaciones confirmadas, solo una: SOL/USDT, fila 4 374 (abre el
    2026-08-14T20:00Z, entrada el 2026-08-15T00:00Z), con `regime_change_above_trend`. Entra con la
    vela guardada en los recuentos del pool y en el cálculo del apartado 9.
- **Indicadores y gate con la vela guardada y con la cerrada.**
  - **Con qué datos.** Las siete series se extendieron con `forward_data.extend_history` hasta la
    vela que cierra el 2026-10-18T00:00Z. La descarga fue a las 2026-10-18T03:46Z, con lectura
    pública, en un directorio aparte y no en `data/forward/`.
  - Los siete archivos tienen 4 758 filas. Sus sha-256:
    - ADA/USDT `9774bbca5e30b21edf26859dbbb3474eb0a53cd1dfe7b18299103f3575197e83`;
    - BNB/USDT `fad5432255c876b563a9005819d8748e405f5ab06bcd3b208af072230e70a3f7`;
    - BTC/USDT `aaa1b30c2be2f08bec4b48c159a5d64700b5c7ba49910755d9c4ae4232c0e167`;
    - DOGE/USDT `7ffae97517c22a025435c9b6a7bd64f0a05a31d3620e393b42a92b4eb196dea4`;
    - ETH/USDT `0a1634eb0e761ae0d2353916d1c02cd49c1ad7f0a4754ebad189538796cf2d91`;
    - SOL/USDT `fcc7a6b9d63acc24159b5ef011ab6e3f5a9cacd72374ee69b5024c25bf9d35df`;
    - XRP/USDT `c64b8bf852f9aa4903980fa8d83d00fb41309c2b556e884ada9abfd9f70c4860`.
  - **Qué se calculó.** Para cada barra, de la fila 4 380 (la vela misma, que cierra el
    2026-08-16T00:00Z) a la 4 758, se calculó lo mismo que `nodes.prepare_evaluation`:
    - la ventana del replay, 499 velas cerradas (`selection.replay_window`);
    - `indicators.enrich`, `to_indicator_set` y `prompts.format_indicators`;
    - `activation.evaluate_activation` con `DEFAULT_CONFIG`.
  - Se calculó dos veces: con la fila guardada y con la cerrada, la devuelta en el asiento
    anterior. Las dos series no difieren en nada más.

| símbolo | barras | difiere algo | difiere el gate | última barra que difiere (cierre) | ΔEMA_200 desde la fila 4 679 |
| --- | ---: | ---: | ---: | --- | ---: |
| ADA/USDT | 379 | 135 | 1 | 2026-10-04T04:00Z | −3.3 × 10⁻⁷ |
| BNB/USDT | 379 | 379 | 0 | 2026-10-18T00:00Z, la última | −0.000425 |
| BTC/USDT | 379 | 379 | 0 | 2026-10-18T00:00Z, la última | −0.0177 |
| DOGE/USDT | 379 | 146 | 5 | 2026-09-19T00:00Z | −7.5 × 10⁻⁸ |
| ETH/USDT | 379 | 379 | 0 | 2026-10-18T00:00Z, la última | −0.000601 |
| SOL/USDT | 379 | 345 | 4 | 2026-10-17T20:00Z | −0.0000603 |
| XRP/USDT | 379 | 149 | 0 | 2026-09-27T12:00Z | −5.8 × 10⁻⁷ |
| **total** | **2 653** | **1 912** | **10** | | |

- **El gate difiere en 10 barras, todas con cierre hasta el 2026-08-19T20:00Z.**
  - ADA/USDT, en la vela misma: con la guardada no abre; con la cerrada abre por
    `range_breakout_down`.
  - DOGE/USDT, cinco barras con cierre entre el 2026-08-17T04:00Z y el 2026-08-19T20:00Z:
    - en cuatro abre con la guardada por un `regime_change` y no abre con la cerrada;
    - en una pasa al revés.
  - SOL/USDT, cuatro barras con cierre entre el 2026-08-17T00:00Z y el 2026-08-17T16:00Z: abre con
    las dos velas, con otros disparadores.
  - Después de esa fecha, el gate decide lo mismo en todas las barras medidas.
- **Los indicadores no dejan de diferir con los datos de hoy.** En BNB, BTC y ETH difiere también
  la última barra disponible.
  - **La razón es la EMA_200.**
    - Es lineal en el cierre, y pandas-ta la siembra con la media de las primeras 200 velas de la
      ventana.
    - Desde la fila 4 679, la vela cae en esa siembra. A partir de ahí, la diferencia en la última
      fila es constante: Δcierre × 2.515 × 10⁻⁴.
    - Calculada con la respuesta al impulso de `ta.ema`, coincide en todas las barras, con un error
      relativo de 6 × 10⁻⁹ o menos.
  - **Lo que se ve en el prompt.** `format_indicators` imprime cuatro decimales.
    - En BTC, ETH y BNB la diferencia es de 0.0001 o más, así que la EMA_200 impresa difiere en cada
      barra hasta que la vela sale de la ventana.
    - En SOL difiere según el redondeo: en 345 de 379 barras.
    - En ADA, DOGE y XRP la diferencia es menor que 10⁻⁶ y casi nunca cambia el valor impreso.
  - El resto de los indicadores de la última barra difiere en menos de 10⁻⁶, y esa diferencia sigue
    bajando.
- **En el estudio.** La vela está en la ventana de 499 de las barras hasta la fila 4 878, que
  cierra el 2026-11-07T00:00Z.
  - Son 97 barras del estudio por símbolo, de la fila 4 782 (cierre 2026-10-22T00:00Z) a la 4 878.
  - Cubren los bloques 1 y 2 enteros y las 13 primeras barras del bloque 3.
  - En esas barras, el prompt de `solo` de una activación de BTC, ETH o BNB lleva la EMA_200
    calculada con la vela guardada. Es lo que dice el apartado 4: la serie del estudio es la que se
    guardó.
  - Si el gate abriría distinto con la vela cerrada en alguna de esas barras, no se sabe hasta que
    existan sus velas. En las 2 653 barras medidas solo difirió en los cuatro primeros días.

### 2026-10-17 — Las reglas de velas del apartado 4 entran en master

- **Qué entró.** Las dos reglas de velas del apartado 4, «Vela revisada» y «Hueco persistente»,
  son el commit `bffd24e776e5…`. Entraron en master con el merge del PR #22.
  - `git merge-base --is-ancestor bffd24e e1317e3` sale con 0: `bffd24e` es antecesor del merge.
  - Los tres commits del PR, del más antiguo al último: `bffd24e`, `874f7ba` y `f7214a1`.
- **Commit de merge en master:** `e1317e3571100a8e0ab74bd704525e5b91cef63d`. Tiene dos padres:
  `ce3afe2f172f152ec91ca87a836e5359bd081837` (master) y
  `f7214a181de9e763052740476f2f0e3eccb45bd4` (la rama del PR).
- **Fecha de committer:** 2026-10-17T22:33:16-06:00, es decir 2026-10-18T04:33:16Z. También la
  puso GitHub: el merge es un `pr_merge` hecho en la web, y su committer es
  `GitHub <noreply@github.com>`.
- **Push en GitHub:** 2026-10-18T04:33:16Z.
  - Es el evento `pr_merge` de `refs/heads/master`, con `before` `ce3afe2…` y `after`
    `e1317e3…`, en `GET https://api.github.com/repos/AlexMartGong/Cripto_Agents/activity`.
  - Se leyó con `curl` sin autenticar, como el primer asiento, en la laptop. La lectura fue a las
    2026-10-18T05:55Z, el 2026-10-17 en hora local.
- **Antes del bloque 1.** Las dos fechas son anteriores a 2026-10-22T00:00Z, así que las dos
  reglas rigen desde el lote del bloque 1.

### 2026-10-19 — El código del análisis entra en master y queda congelado

- **Qué entró.** El análisis del apartado 6, `python -m crypto_agents.forward analyze`, y su ensayo
  sobre la etapa 1, `forward rehearse-analysis`. Entraron en master con el merge del PR #23.
  - Los dos commits del PR, del más antiguo al último: `021294d3cf4b…`, el análisis y su ensayo, y
    `200cb0000d15…`, lo que aplica las reglas de lotes del apartado 5 (`forward baselines`,
    `batch --supersede` y las faltantes en el análisis).
  - `git merge-base --is-ancestor 200cb00 5b74701` sale con 0: `200cb00` es antecesor del merge.
- **Commit de merge en master:** `5b74701b3fe187bb76e3ccc0307de3c9f4d38175`. Tiene dos padres:
  `e1317e3571100a8e0ab74bd704525e5b91cef63d` (master) y
  `200cb0000d157e5f40d3bf2d45d7b90fd7392ee9` (la rama del PR).
- **Fecha de committer:** 2026-10-19T16:15:53-06:00, es decir 2026-10-19T22:15:53Z. También la
  puso GitHub: el merge es un `pr_merge` hecho en la web, y su committer es
  `GitHub <noreply@github.com>`.
- **Push en GitHub:** 2026-10-19T22:15:53Z.
  - Es el evento `pr_merge` de `refs/heads/master`, con `before` `e1317e3…` y `after`
    `5b74701…`, en `GET https://api.github.com/repos/AlexMartGong/Cripto_Agents/activity`.
  - Se leyó con `curl` sin autenticar, como los asientos anteriores, en la laptop, a las
    2026-10-19T23:06Z.
- **Antes del bloque 1.** Las dos fechas son anteriores a 2026-10-22T00:00Z.
- **Queda congelado desde `5b74701`.** El código del análisis es:
  - `src/crypto_agents/forward_analysis.py`;
  - los 28 módulos del paquete que carga al importarse, sin el router: `activation`,
    `activation_sweep`, `audit`, `baselines`, `consumption`, `criteria`, `design_effect`,
    `dispersion`, `execution`, `forward_data`, `forward_plan`, `forward_record`, `funding`,
    `indicators`, `journal`, `market`, `metrics`, `outcomes`, `paired_dispersion`, `perp_probe`,
    `queries`, `quota`, `risk`, `runner`, `selection`, `settings`, `state` y `stops`;
  - el despacho de `analyze` y `rehearse-analysis` en `forward.py`.

  Cualquier cambio posterior a ese código se anota aquí, con su commit y su motivo. En `1e9ac51`,
  el merge siguiente, es el mismo: entre los dos commits solo cambia `docs/forward.md`. Sobre
  `1e9ac51`, en la laptop, el 2026-10-19, `forward rehearse-analysis` sale con 0, 44 de 44.
- **La cabecera del documento.** El 2026-10-19, antes de que abra el bloque 1, cambiaron dos
  frases:
  - «Estado: prerregistrado y no implementado.» pasó a «Estado: el del «Registro del estudio», al
    final.»;
  - «Los comandos de lote y de análisis los escribe el bloque siguiente, que cumple lo que exige el
    apartado 10.» pasó a «Los comandos de lote y de análisis cumplen lo que exige el apartado 10.».

  Es un cambio de redacción, no de regla: las dos frases anteriores dejaron de ser ciertas al entrar
  este código, y las nuevas no caducan.

### 2026-10-19 — Las reglas de lotes del apartado 5 entran en master

- **Qué entró.** Las tres reglas de lotes del apartado 5 —la faltante, las líneas base completadas
  después del plazo y las pasadas sustituidas— son el commit `91d74510a90b…`. Entraron en master
  con el merge del PR #24.
  - `git merge-base --is-ancestor 91d7451 1e9ac51` sale con 0: `91d7451` es antecesor del merge.
  - El PR tiene un solo commit, `91d7451`, y solo toca `docs/forward.md`. El código que aplica las
    reglas entró antes, con el PR #23 (`200cb00`, asiento anterior).
- **Commit de merge en master:** `1e9ac51fcafbf8724802d8c993917ef02ba5599a`. Tiene dos padres:
  `5b74701b3fe187bb76e3ccc0307de3c9f4d38175` (master) y
  `91d74510a90b09164715d6d7119074e5915662c9` (la rama del PR).
- **Fecha de committer:** 2026-10-19T16:33:19-06:00, es decir 2026-10-19T22:33:19Z. También la
  puso GitHub: el merge es un `pr_merge` hecho en la web, y su committer es
  `GitHub <noreply@github.com>`.
- **Push en GitHub:** 2026-10-19T22:33:19Z.
  - Es el evento `pr_merge` de `refs/heads/master`, con `before` `5b74701…` y `after`
    `1e9ac51…`, en `GET https://api.github.com/repos/AlexMartGong/Cripto_Agents/activity`.
  - Se leyó con `curl` sin autenticar, en la misma lectura que el asiento anterior.
- **Antes del bloque 1.** Las dos fechas son anteriores a 2026-10-22T00:00Z, así que las tres
  reglas rigen desde el lote del bloque 1.

### 2026-10-29 — El temporizador de los lotes entra en master

- **Qué entró.** Desde el bloque 2, los lotes los lanza un temporizador de systemd de usuario en la
  laptop, `deploy/crypto-agents-forward.timer`. Entró en master con el merge del PR #26.
  - Corre cada día a las 19:00 de Jalisco (`OnCalendar=*-*-* 19:00:00`), con `Persistent=true`:
    una ejecución que se pierde con la laptop apagada se hace al volver a encenderla.
  - Lanza la envoltura, `python -m crypto_agents.forward_service`:
    - antes, comprueba la rama y el árbol, espera la red y hace `git pull --ff-only`;
    - durante, corre `python -m crypto_agents.forward auto` con el inhibidor de suspensión;
    - después de cada lote completo, versiona exactamente los archivos que el lote listó bajo
      `data/forward/`, con un commit de mensaje fijo, y los sube con una clave de despliegue propia
      del servicio.
  - `auto` corre cada bloque pendiente con el mismo código que `batch --block`, desde el bloque 2
    (`forward_auto.AUTO_FIRST_BLOCK`). El lote del bloque 1 lo corre el operador.
  - Los cuatro commits del PR, del más antiguo al último: `cc42962`, `0c956e1`, `61addb9` y
    `74a8320`. `git merge-base --is-ancestor 74a8320 cd6a6d6` sale con 0.
- **Siguen siendo manuales:**
  - lo que sigue a los códigos 1, 5, 4 y 7: `auto` se detiene y avisa. El 5 se continúa con
    `batch --block … --accept-revised …` (apartado 4), y el 4 y el 7 terminan el estudio
    (apartado 5);
  - repetir entero un lote inválido, con `batch --block … --supersede` (apartado 5);
  - completar las líneas base después del plazo, con `forward baselines --block …`;
  - el análisis, `forward analyze`, que se corre una vez, al cerrar (apartado 6).
- **`forward baselines` registra como vacío un bloque vacío cuyo lote no corrió**, campo por campo
  como lo registra el lote, salvo cuándo y dónde corrió.
  - En la rama de `analyze` de un estudio terminado, ese registro añade la fila del bloque y cambia
    la línea de resumen. No cambia el veredicto.
  - `analyze` da la misma salida en sus cuatro ramas lo escriba el lote o `forward baselines`,
    también cuando `forward baselines` prepara el bloque tarde, sobre series alargadas y con una
    vela aceptada.
- **Es una corrección del arnés, no del camino medido** (apartado 3).
  - `git diff --stat ed84077 cd6a6d6` sale vacío sobre los 29 módulos congelados del asiento del
    2026-10-19.
  - `forward rehearse` sobre `ed84077` y sobre `74a8320`, comparados con
    `scripts/compare_rehearsals.py`, sale igual salvo lo permitido: `started_at`, `git_commit` y
    `argv` en `meta.json`, y `calls[].at` en los journals. El script sale con 0, y los journals de
    las cuatro líneas base son idénticos byte a byte.
  - Los dos ensayos corrieron en la laptop, solo con caché, y dieron 140 de 140 en los cinco brazos
    sin llamadas vivas. La tabla está en la descripción del PR #26.
- **Commit de merge en master:** `cd6a6d655bfe4e6799c7c57db9ab71b95b410cf9`. Tiene dos padres:
  `ed8407786d29239efc1f7b7b3897f0c325518a69` (master) y
  `74a83205eb1daba98dc0dae8f4b641b75d4e981a` (la rama del PR).
- **Push en GitHub:** 2026-10-29T05:43:37Z.
  - Es el evento `pr_merge` de `refs/heads/master`, con `before` `ed84077…` y `after`
    `cd6a6d6…`, en `GET https://api.github.com/repos/AlexMartGong/Cripto_Agents/activity`.
  - Se leyó con `curl` sin autenticar, en la laptop, en una sola lectura para este merge y para los
    del PR #27 y el PR #28.

### 2026-10-29 — La comprobación semanal del estudio entra en master

- **Qué entró.** Una comprobación semanal en GitHub Actions, `.github/workflows/forward-watch.yml`,
  que corre `python -m crypto_agents.forward_watch` sobre `master` cada sábado y a mano
  (`workflow_dispatch`). Entró en master con el merge del PR #27.
  - Solo vigila. Lee el estudio y no corre ningún lote ni escribe nada; el flujo tiene
    `permissions: contents: read` y ningún secreto.
  - Falla, y así le llega un correo al operador, si la ventana de un lote pendiente lleva más de
    24 h abierta, si un bloque está retenido, es inválido o tiene duplicados, o si un bloque venció
    sin sus líneas base completas.
  - El PR trae también la preparación de la laptop para el temporizador, en `deploy/README.md`,
    escrita y no ejecutada.
  - Los cuatro commits del PR, del más antiguo al último: `36bf0e1`, `caea897`, `1ab2862` y
    `8fd5437`. `git merge-base --is-ancestor 8fd5437 32850c3` sale con 0.
- **Lo único que cambia de `forward_auto.py` es `study_state`:**
  - gana un argumento `first`, que por defecto es `AUTO_FIRST_BLOCK`. Las dos llamadas del
    temporizador no lo pasan, así que el temporizador sigue empezando en el bloque 2. Solo
    `forward_watch` pasa `FIRST_BLOCK`, para vigilar también el bloque 1;
  - gana una comprobación de longitud: antes de cada bloque comprueba que la lista del cierre tiene
    una entrada por cada bloque anterior. Una lista mal contada es un error, no un estudio abierto.
- **No toca el código congelado.** `git diff --stat ed84077 32850c3` sale vacío sobre los 29
  módulos congelados.
- **Commit de merge en master:** `32850c37b557a13ae13d5892d543cb45b05606a7`. Tiene dos padres:
  `cd6a6d655bfe4e6799c7c57db9ab71b95b410cf9` (master) y
  `8fd54371cbbec3872a8a49350ce8e90fa1741cd5` (la rama del PR).
- **Push en GitHub:** 2026-10-29T19:37:06Z.
  - Es el evento `pr_merge` de `refs/heads/master`, con `before` `cd6a6d6…` y `after`
    `32850c3…`, en `GET https://api.github.com/repos/AlexMartGong/Cripto_Agents/activity`.
  - Se leyó en la misma lectura que el asiento anterior.

### 2026-10-30 — Velas revisadas por el exchange, lote del bloque 1

- **La pasada se detuvo.** La primera pasada del lote del bloque 1,
  `data/forward/batches/2026-10-22/20261030T000029Z/`, salió con el código 5: el exchange devolvió
  distintas las siete velas que abren el 2026-08-15T20:00Z, la fila 4 380 que se guardó en
  formación (asiento del 2026-10-17).
  - Su `forward.json` registra `outcome` «vela revisada» y las siete velas en `revised`, con el
    valor guardado y el devuelto.
  - No escribió ninguna serie y no llamó a ningún modelo.
- **Las siete velas**, con los valores de ese `forward.json`:
  - **ADA/USDT**, vela que abre el 2026-08-15T20:00Z:
    - guardado: apertura 0.1781, máximo 0.1783, mínimo 0.1772, cierre 0.1778, volumen 2490116.8;
    - devuelto: apertura 0.1781, máximo 0.1783, mínimo 0.1758, cierre 0.1765, volumen 7057240.3.
  - **BNB/USDT**, vela que abre el 2026-08-15T20:00Z:
    - guardado: apertura 611.41, máximo 611.67, mínimo 609.3, cierre 609.45, volumen 4102.006;
    - devuelto: apertura 611.41, máximo 611.67, mínimo 607.5, cierre 607.76, volumen 7581.756.
  - **BTC/USDT**, vela que abre el 2026-08-15T20:00Z:
    - guardado: apertura 63126.07, máximo 63168.95, mínimo 63076.96, cierre 63156.34, volumen
      290.08793;
    - devuelto: apertura 63126.07, máximo 63175.0, mínimo 63076.96, cierre 63086.01, volumen
      407.21827.
  - **DOGE/USDT**, vela que abre el 2026-08-15T20:00Z:
    - guardado: apertura 0.06995, máximo 0.07009, mínimo 0.06987, cierre 0.06991, volumen
      10323024.0;
    - devuelto: apertura 0.06995, máximo 0.07009, mínimo 0.06951, cierre 0.06961, volumen
      30219590.0.
  - **ETH/USDT**, vela que abre el 2026-08-15T20:00Z:
    - guardado: apertura 1884.55, máximo 1885.8, mínimo 1882.8, cierre 1885.03, volumen 4345.5732;
    - devuelto: apertura 1884.55, máximo 1885.8, mínimo 1881.98, cierre 1882.64, volumen 7295.586.
  - **SOL/USDT**, vela que abre el 2026-08-15T20:00Z:
    - guardado: apertura 75.57, máximo 75.74, mínimo 75.42, cierre 75.59, volumen 57552.37;
    - devuelto: apertura 75.57, máximo 75.74, mínimo 75.31, cierre 75.35, volumen 85072.665.
  - **XRP/USDT**, vela que abre el 2026-08-15T20:00Z:
    - guardado: apertura 1.004, máximo 1.0068, mínimo 1.0032, cierre 1.0051, volumen 1750224.9;
    - devuelto: apertura 1.004, máximo 1.0068, mínimo 1.0012, cierre 1.0028, volumen 3009949.6.

  Son los mismos valores, guardados y devueltos, que anotó el asiento del 2026-10-17.
- **Se continuó con el valor guardado.** La segunda pasada,
  `data/forward/batches/2026-10-22/20261030T000327Z/`, se lanzó con `--accept-revised` para cada una
  de las siete velas.
  - Su `forward.json` las lleva en `accepted`, y `data/forward/revisions.json` las guarda con el
    valor guardado y el devuelto.
  - La serie del estudio es la que se guardó.
- **El lote corrió sobre el commit que registra su `forward.json`.** Las dos pasadas registran
  `git_commit` `32850c37b557a13ae13d5892d543cb45b05606a7`, el merge del PR #27.
- **Este asiento se escribió después de relanzar.** El apartado 4 pone el asiento antes de que la
  extensión continúe, y el lote lo dice al detenerse: «Con el asiento en el registro, se continúa
  con el valor guardado».
  - La pasada 1 se detuvo a las 2026-10-30T00:00:29Z. Es el `invoked_at` de su `forward.json`: la
    pasada que se detiene por velas revisadas lo toma al detenerse.
  - La pasada 2 empezó a las 2026-10-30T00:02:51Z. Es el `accepted_at` de las siete velas en su
    `forward.json`, la hora con que empieza a extender las series. Su `invoked_at`,
    2026-10-30T00:03:27Z, es de después de extenderlas.
  - Los siete valores ya estaban en el registro, desde el asiento del 2026-10-17.

### 2026-10-30 — Bloque 1: el upstream del decisor no se puede verificar

- **Qué se vio.** Las 42 llamadas al decisor de `solo` del lote del bloque 1 fueron vivas, y
  ninguna trajo `x-opencode-upstream-model-id` ni `x-opencode-endpoint-id`. Las 42 tienen
  `upstream_model` y `upstream_endpoint` nulos en
  `data/forward/batches/2026-10-22/20261030T000327Z/solo.jsonl`.
  - En la etapa 1, las 140 llamadas del decisor de `solo` las sirvió
    `accounts/fireworks/models/glm-5p3` por `fireworks` (`data/stage1/solo.jsonl`).
  - El código que lee las cabeceras no cambió entre `b9676eb`, el commit de la etapa 1, y
    `32850c3`, el del lote: `git diff --stat b9676eb 32850c3 -- src/crypto_agents/llm.py uv.lock`
    sale vacío.
  - La primera llamada viva del bloque, la que lo muestra (apartado 5), es la de ADA/USDT con
    `run_id` `e27cb5e3-451c-543c-b567-7086df0a646f`, cuya vela evaluada cierra el
    2026-10-22T04:00Z. Terminó a las 2026-10-30T00:05:28Z, según el `solo.jsonl` del lote.
- **El `upstream.md` del lote dice «solo en aciertos de caché, sin llamada viva».** Es un error del
  generador: comparaba el par sin etiquetar con la etiqueta «sin cabecera». Las 42 fueron vivas y
  ninguna fue un acierto de caché. Lo corrige el PR #28 (asiento siguiente).
- **El sondeo**, `scripts/probe_upstream.py`, fuera del estudio, en dos tandas:
  `data/probes/20261030T051122Z-upstream/` y `data/probes/20261030T102001Z-upstream/`. La lectura
  de las dos está en `lectura.md`, en el directorio de la primera.
  - A `minimax-m3` también le faltan las dos cabeceras. La pérdida apunta a la pasarela, común a
    los dos modelos, y no dice nada del proveedor de `glm-5.2`.
  - Un User-Agent fijo y una sesión estable no cambian nada: ni las cabeceras, ni la velocidad, ni
    el orden de los argumentos.
- **`glm-5.2` se sirve de otra manera que en la etapa 1**, con la petición registrada: el
  `forward.json` de la pasada lleva el camino medido registrado, envoltura incluida, y la variante
  a de las tandas manda esa misma envoltura, `f0727ca0…6928`.
  - Va a 97–109 tok/s el viernes 2026-10-30 de 10:20 a 10:23Z (`calls.jsonl` de la segunda tanda),
    frente a 50–59 en la misma ventana del viernes de la etapa 1 (`data/stage1/solo.jsonl`). Los
    rangos no se solapan.
  - Devuelve los argumentos en orden alfabético en 53 de 54 llamadas desde el bloque 1 (41 de las
    42 del lote y las 12 de las tandas), frente al orden del esquema en 140 de 140 en la etapa 1.
    Las cifras del lote y de la etapa 1 salen del texto crudo de las cachés de `var/`, que no se
    versionan (`lectura.md`).
- **No se sabe si cambió el proveedor, la pila o la versión del modelo.** El cambio ocurrió entre
  la última llamada viva de la etapa 1, a las 2026-10-09T12:35Z (`data/stage1/solo.jsonl`), y la
  primera del bloque 1, a las 2026-10-30T00:05Z (el `solo.jsonl` del lote). En el repositorio no
  hay ninguna llamada viva registrada entre las dos.
- **El estudio continúa** (apartado 5, «Política de upstream»), y el resultado se reporta también
  separado por upstream, como dato descriptivo.
- **El `forward.json` de la pasada `20261030T000327Z` dice `git_dirty: true`.** Lo dan archivos sin
  versionar en `data/forward/`:
  - el `forward.json` de la pasada anterior, `20261030T000029Z`;
  - lo que la propia pasada escribió antes de su `forward.json`: las series extendidas, el
    manifiesto del bloque y `revisions.json`.

  En `32850c3`, `git_dirty` es `git status --porcelain` entero, así que cuenta todo eso. El código
  era el del commit registrado: la pasada 1, sobre el mismo commit y antes de escribir nada,
  registró `git_dirty: false`.

### 2026-10-30 — El sondeo del upstream y «no verificable» entran en master

- **Qué entró.** Con el merge del PR #28:
  - **El sondeo de diagnóstico**, `scripts/probe_upstream.py`, y sus dos tandas en `data/probes/`.
    Está fuera del estudio: no es un módulo del paquete, no lee `data/forward/` ni las cachés del
    estudio y de la etapa 1, y ninguna de sus llamadas entra en un lote.
  - **«Sin cabecera» frente a «sin cabecera» se informa como «no verificable».** El resumen del
    lote no dice «sin cambios», no se escribe `upstream.md`, y `auto` lo da como aviso normal. Un
    upstream identificable nuevo sigue siendo un cambio, con su `upstream.md` y su aviso urgente,
    también después de un lote sin cabecera.
  - **El asiento de upstream que prepara el lote** compara con la etiqueta que da
    `metrics.upstream_distribution`, y da de cada par nuevo sus llamadas vivas y sus aciertos de
    caché, contados.
  - **El `git_dirty` de `forward.json` excluye `data/forward/`**, versionado o no. El de
    `meta.json` sigue siendo el estado crudo del árbol, porque lo imprime código congelado
    (`audit`).
  - Los tres commits del PR, del más antiguo al último: `63a2b94`, `d8d9e82` y `f10d90f`.
    `git merge-base --is-ancestor f10d90f 537609f` sale con 0.
- **No toca el camino medido ni el código congelado.** `git diff --stat 32850c3 537609f` sale vacío
  sobre `src/crypto_agents/llm.py`, `src/crypto_agents/prompts/` y `uv.lock`, y sobre los 29
  módulos congelados.
- **El ensayo sale igual con los tres PR dentro.** `forward rehearse` sobre `ed84077` y sobre
  `537609f`, que ya incluye los PR #26, #27 y #28, comparados con `scripts/compare_rehearsals.py`,
  sale igual salvo lo permitido: `started_at`, `git_commit` y `argv` en `meta.json`, y `calls[].at`
  en los journals. El script sale con 0, y los journals de las cuatro líneas base son idénticos
  byte a byte.
  - Los dos ensayos corrieron en la laptop, solo con caché, y dieron 140 de 140 en los cinco brazos
    sin llamadas vivas. Su único acceso a la red fue el `GET /models` gratuito.
  - La salida del script y los comandos están en la descripción del PR #29.
- **Commit de merge en master:** `537609fcc2e25201919ad5945be9a4598a25282a`. Tiene dos padres:
  `ffc94b045d1fc1436be892c9e6ef7e545e9d93fa` (master, el commit de los artefactos del lote del
  bloque 1) y `f10d90fd3f71b7a4f15e04d1c2a25b3225127e9f` (la rama del PR).
- **Push en GitHub:** 2026-10-30T17:44:24Z.
  - Es el evento `pr_merge` de `refs/heads/master`, con `before` `ffc94b0…` y `after`
    `537609f…`, en `GET https://api.github.com/repos/AlexMartGong/Cripto_Agents/activity`.
  - Se leyó en la misma lectura que los asientos del PR #26 y del PR #27.

### 2026-10-31 — La huella del decisor entra en master

- **Qué entró.** Con el merge del PR #30:
  - **`src/crypto_agents/forward_fingerprint.py`**, libre del router. De cada llamada al decisor de
    un journal de `solo` toma dos huellas: el tok/s, que está en los journals versionados, y el
    orden de las claves de los argumentos, que solo está en el texto crudo de la caché. No compara
    nada: los órdenes van tal cual.
  - **Cada pasada que llama al decisor escribe su huella** en
    `data/forward/fingerprints/<bloque>/<pasada>.json`, desde la caché que usó esa pasada, y la
    incluye en los archivos a versionar. Un fallo al calcularla o al escribirla no cambia el
    resultado ni el código de salida del lote.
  - **El comando `python -m crypto_agents.forward fingerprint`** escribe la huella retroactiva de
    la etapa 1 y de las pasadas que todavía no la tienen. Toma el cerrojo que comparten `batch`,
    `baselines` y `auto`: con el cerrojo tomado sale con 1 y no escribe nada.
  - Los tres commits del PR, del más antiguo al último: `f2a0d48`, `c1230a9` y `c9606cd`.
    `git merge-base --is-ancestor c9606cd d527fd2` sale con 0.
- **Las huellas retroactivas**, versionadas en el PR, reproducen las cifras del registro:
  - la etapa 1, en `data/forward/fingerprints/stage1.json`: orden del esquema en 140 de 140, de
    48.4 a 65.2 tok/s;
  - el bloque 1, en `data/forward/fingerprints/2026-10-22/20261030T000327Z.json`: orden alfabético
    en 41 de 42, de 83.6 a 107.6 tok/s.

  Las cachés de `var/` no cambiaron durante la corrida retroactiva: sus 1 023 archivos conservaron
  el sha-256 y el mtime (descripción del PR #30).
- **Precisión sobre el asiento del 2026-10-30 «Bloque 1: el upstream del decisor no se puede
  verificar».**
  - Su «50–59 en la misma ventana» corresponde a las 7 llamadas de la etapa 1 con `at` entre
    10:19:42 y 10:24:50 (mediana 56.3), las de la fila de `lectura.md`.
  - Contando solo las que terminaron entre 10:20:00 y 10:23:59, son 5, de 52.5 a 59.2.
  - La conclusión no cambia: ninguna se solapa con 97–109, el tok/s de `glm-5.2` en el
    `calls.jsonl` de la segunda tanda. El asiento anterior no se edita.
- **No toca el camino medido ni lo congelado.** `git diff --stat ed84077 d527fd2` sale vacío sobre
  los 29 módulos congelados, `src/crypto_agents/llm.py`, `src/crypto_agents/prompts/`,
  `tests/data/` y `data/stage1/`.
  - `forward rehearse` no escribe huellas: el ensayo de `d527fd2` dejó el árbol limpio y
    `data/forward/fingerprints/` solo con sus dos archivos versionados.
  - `forward rehearse` sobre `ed84077` y sobre `d527fd2`, comparados con
    `scripts/compare_rehearsals.py`, sale igual salvo lo permitido: `started_at`, `git_commit` y
    `argv` en `meta.json`, y `calls[].at` en los journals. El script sale con 0, y los journals de
    las cuatro líneas base son idénticos byte a byte.
  - Los dos ensayos corrieron en la laptop, solo con caché, y dieron 140 de 140 en los cinco brazos
    sin llamadas vivas. Su único acceso a la red fue el `GET /models` gratuito, y dejaron los 1 023
    archivos de las cachés con el mismo sha-256 y el mismo mtime.
  - La salida del script y los comandos están en la descripción del PR #31.
- **Commit de merge en master:** `d527fd2d204a24d85c2db98a37702085357cc285`. Tiene dos padres:
  `5c53cb3fe4777e5d02ff8ac2c71aa3d007a9f46b` (master, el merge del PR #29) y
  `c9606cd30a4464dd3e8aa3c16a40c2d0adacc702` (la rama del PR).
- **Push en GitHub:** 2026-10-31T00:45:44Z.
  - Es el evento `pr_merge` de `refs/heads/master`, con `before` `5c53cb3…` y `after`
    `d527fd2…`, en `GET https://api.github.com/repos/AlexMartGong/Cripto_Agents/activity`.
  - Se leyó con `curl` sin autenticar, en la laptop, como los asientos anteriores.
