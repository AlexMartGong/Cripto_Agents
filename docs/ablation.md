# Ablación: ¿los seis modelos deciden mejor que uno?

**Estado: el arnés está construido y probado; hubo una primera corrida con modelos reales, descartada (ver la enmienda posterior), y la segunda está pendiente.**
Este documento contiene el método, lo que ya se puede afirmar y una conclusión sin escribir. La
tabla de resultados la genera el comando en el directorio de la corrida (`report.md`) y se copia a
mano a la sección marcada más abajo: el comando ya no escribe en este archivo.

## Cómo se corre

```bash
uv run python -m crypto_agents.ablation --dry-run --manifest data/ablation_selection.json
uv run python -m crypto_agents.ablation --fill --max-usd X --manifest data/ablation_selection.json
uv run python -m crypto_agents.ablation --manifest data/ablation_selection.json
```

Con pago por uso `--fill` exige `--max-usd`: la cuota declarada no frena un rol remoto y sin tope
lo único que para la corrida es el saldo. La tabla queda en `report.md` dentro del directorio de la
corrida; `--out` la manda a otro sitio, y un archivo que ya existe no se pisa.

Sin `--fill` el replay es solo-caché: se niega a llamar a ningún proveedor y no modifica la caché.
Lo que eso garantiza es acotado: un replay solo-caché de una corrida con journal no cuesta dinero,
y `tests/test_replay.py` fija que reproduce, sin llamar a nadie ni tocar la caché, una evaluación
que necesitó reintentos. No garantiza que cualquier reejecución de la tabla sea gratis —`--fill`
paga— ni que la reejecución termine. Reproduce todo intento que produjo contenido, también los
inválidos con su reintento, pero
no lo que nunca tuvo respuesta que guardar: una evaluación que murió por un rechazo o un plazo
vencido del proveedor, o que corrió degradada al respaldo local, no está en la caché y la corrida se
detiene con `ReplayCacheMissError` nombrando el modelo y el prompt que faltan. Opciones útiles: `--arms full,solo` para un subconjunto,
`--horizon` para el plazo con el que se puntúan las órdenes, y `--dry-run`, que cuenta la factura
sin emitir una sola llamada.

## Sobre qué se corre

Sin `--manifest` el comando recorre `tests/data/btcusdt_4h.csv` de principio a fin: 500 velas menos
400 de warm-up son 99 evaluaciones posibles y **15 activaciones**. Seis formas de pipeline sobre 15
decisiones no se separan, así que ese modo sirve para probar el arnés, no para responder la
pregunta.

Lo que se compara de verdad es una **selección estratificada**, construida con
`python -m crypto_agents.selection` y versionada en `data/ablation_selection.json`:

| | |
| --- | --- |
| Histórico | 7 símbolos, 4h, 2 años — 4 380 velas cada uno, en `data/history/` |
| Activaciones disponibles | 617–736 por símbolo (4 740 en total) |
| Seleccionadas | **140**: 20 por símbolo, 4 en cada uno de 5 tramos temporales |
| Semilla | 20260815, escrita en el manifiesto |
| Horizonte | 6 velas; ninguna activación entra sin esas velas por delante |

El tope son 140 y no más porque los seis brazos llegan al decisor y ninguno reutiliza la caché del
otro —su prompt lleva dentro los alegatos—, así que 880 llamadas por ventana son 146 activaciones.
140 deja 40 llamadas para reintentos.

Que estén repartidas es el punto, no un detalle: 140 activaciones seguidas de un solo símbolo son un
régimen de mercado, y la tabla estaría midiendo qué pipeline le sienta mejor a dos meses concretos.
Cada entrada del manifiesto lleva además el digest de la ventana exacta que verá la evaluación, y el
comando lo comprueba antes de la primera llamada: si el exchange revisa una vela, se sabe al empezar
y no en una tabla que ya no compara con la anterior.

## Los diez brazos

| brazo | forma del pipeline | qué pregunta responde |
| --- | --- | --- |
| `full` | 3 técnicos → 2 mesas → decisor | referencia, no respuesta |
| `no_debate` | 3 técnicos → decisor | ¿aporta algo el debate? |
| `bull_only` | 3 técnicos → 1 mesa → decisor | ¿aporta la contraparte, o solo ruido? |
| `solo` | indicadores → 1 llamada | ¿uno iguala a seis? |
| `local_technicals` | `full`, técnicos en local | ¿se puede gastar menos en las lecturas? |
| `local_bull` | `full`, mesa alcista en local | ¿aguanta una mesa en local? |

Todo lo que cambia entre brazos es la forma del grafo y el reparto rol → modelo. El histórico, el
gate de activación, el gate de riesgo y la construcción de la orden son idénticos, y hay un test por
variante que lo comprueba: si cada brazo tuviera su propio camino al mercado, la tabla mediría el
arnés en vez de los modelos.

La coincidencia se mide sobre la **acción**, no sobre la confianza. Dos decisores que compran con
0.7 y con 0.6 tomaron la misma decisión; tratar esa diferencia como desacuerdo inventaría señal.

### Cuatro líneas base, sin modelo

Un brazo que no se compara con nada no demuestra nada: que `full` gane dinero en una ventana
alcista no dice si decide o si el mercado subía. Se añaden cuatro brazos que cuestan **cero
llamadas** —`baselines.py` no puede importar el router— y que recorren la misma cola que los demás:
emiten un `Proposal`, y de ahí en adelante van decisor → riesgo → ejecución → journal.

| brazo | qué decide | qué pregunta responde |
| --- | --- | --- |
| `always_buy` | compra en cada activación | ¿qué da la dirección más simple? |
| `always_sell` | vende en cada activación | su espejo: cuánto de lo anterior es deriva del mercado |
| `random_uniform` | buy, sell o hold con igual probabilidad | ¿supera un modelo a tirar un dado con el mismo stop? |
| `rule_trend` | pila de medias estricta y ADX ≥ 25 | ¿supera un modelo a una regla de tendencia de manual? |

Corren **detrás del gate de activación**, como todos los brazos. Miden si un modelo aporta algo
*donde el gate decide mirar*; no miden si gate más modelo bate a comprar y mantener.

**Regla de `rule_trend`, congelada el 2026-10-01, antes de que existiera ninguna salida de la
ablación:**

- **Compra** si `close > EMA_20 > EMA_50 > EMA_200` y `ADX_14 ≥ 25`.
- **Vende** si `close < EMA_20 < EMA_50 < EMA_200` y `ADX_14 ≥ 25`.
- **Hold** en cualquier otro caso.

Las desigualdades son estrictas —un empate no es una pila—; el umbral de ADX no, porque el de Wilder
incluye el 25. Los periodos son los del preset y el 25 es el umbral que el gate ya usa. No hay
ningún parámetro buscado: cada número es una convención, y `tests/test_baselines.py` falla si
alguien edita uno. Cambiarlo después de ver resultados es ajustar la regla al dato, y el commit que
lo haga tiene que defenderlo como tal.

`random_uniform` toma `random.Random(sha256("<semilla del manifiesto>|<run_id>"))`: reproducible,
independiente del proceso y del orden en que se recorra el plan. Sin manifiesto usa la semilla de la
selección por defecto.

**Stop común.** Las cuatro declaran como `invalidation_price` el mismo stop: `entry ∓ 2·ATR`, con el
ATR de la vela evaluada (`stops.py`). Va en ATR y no en un porcentaje fijo porque la corrida salta
entre siete símbolos: un 2% es ancho en BTC a 4h y estrecho en SOL. El tamaño (`0.05`) y la
confianza (`0.5`) son relleno que `Proposal` exige; ninguna métrica los lee.

## Cómo se puntúa el resultado

Desde la vela de la orden se camina hacia adelante hasta el horizonte (6 velas por defecto, un día
en 4h):

- Si el precio toca el `invalidation_price` **que declaró el propio decisor**, la operación cuenta
  como invalidada y sale a ese nivel. Se comprueba contra el rango de la vela, no contra su cierre:
  un stop se toca intradía, y puntuar por cierres regalaría operaciones que el mercado no habría
  perdonado.
- Si sobrevive, sale al cierre del horizonte.
- Si el histórico se acaba antes, queda sin resolver y no cuenta ni a favor ni en contra.

Sin comisiones ni slippage. Los criterios se evalúan sobre ese retorno bruto. `audit` y `criteria`
publican además un retorno neto de perpetuos (comisión, slippage y funding), solo descriptivo. Con
pocas órdenes por brazo, el error de muestreo domina cualquier diferencia de retorno, por eso la
tabla publica el denominador junto a la tasa.

### Tres puntuaciones, para separar la dirección del stop

El resultado de un brazo mezcla dos cosas: si acertó la dirección y dónde puso el nivel donde su
tesis se rompe. Un decisor que compra con el stop pegado al precio sale invalidado por ruido aunque
tuviera razón. Cada brazo se puntúa de tres formas sobre las mismas velas:

| puntuación | salida | sobre qué |
| --- | --- | --- |
| stop propio | el `invalidation_price` que declaró el decisor; la de siempre, intacta | órdenes |
| stop común | `entry ∓ 2·ATR`, el mismo para todos los brazos | propuestas accionables |
| cierre del horizonte | sin stop: el cierre de la vela `i + 6` | propuestas accionables |

Las dos últimas puntúan la **propuesta**, no la orden. Una propuesta con el stop declarado en el lado
equivocado se veta, y por eso desaparece de la puntuación propia; si las otras dos también la
dejaran fuera, el stop del decisor volvería a filtrar qué direcciones se miden. La diferencia entre
unas y otras es el número de `invalid_stop_side` del embudo. El riesgo no filtra nada más: en la
ablación los demás vetos son inalcanzables.

Cada puntuación da un **retorno por posición** (media de las resueltas) y uno **por evaluación**
(todas las evaluaciones del plan, con 0 donde no hay posición). Es retorno bruto por unidad
nocional: no se pondera por tamaño, porque ponderar mezclaría el criterio de tamaño del modelo con
la dirección. El segundo permite emparejar: la **diferencia contra `solo`** se calcula evaluación
por evaluación, con error estándar de las diferencias y un intervalo normal al 95%, desde n = 30.

**Lo que esa tabla no corrige.** Son unas 27 diferencias al 95% (nueve brazos por tres
puntuaciones) sin corrección por comparaciones múltiples: con 27, una o dos excluirán el cero solo
por azar. Una diferencia suelta con el cero fuera del intervalo no es un hallazgo; una que se repite
en las tres puntuaciones y contra las líneas base, quizá.

Cada orden se puntúa contra la serie de su propio símbolo: `score_outcomes()` recibe los históricos
indexados por símbolo, porque con siete series en juego un solo `rows` puntuaría la orden de ETH
contra las velas de BTC.

## Lo que ya se puede afirmar sin correr nada

**El coste está medido y no depende del mercado.** El brazo `solo` gasta exactamente **una llamada
por evaluación contra las seis** del pipeline completo, y hay un test que lo fija. Eso es un factor
6 en cuota y en latencia. La consecuencia es incómoda y conviene decirla por adelantado: para
justificar el pipeline completo no basta con que decida *distinto*; tiene que decidir **mejor** lo
bastante como para pagar seis veces el coste. Un empate en calidad es una derrota para `full`.

**Candidatos locales, verificados contra el registro de Ollama en agosto de 2026:**

| modelo | tamaño | entradas | veredicto |
| --- | --- | --- | --- |
| `qwen3:8b` | 5.2 GB | solo texto | cabe con hueco para KV cache |
| `qwen3:4b` | 2.5 GB | solo texto | dos residentes a la vez |
| `granite4.1:8b` | 5.3 GB | solo texto, Q4_K_M | cabe |
| `granite4.1:3b` | 2.1 GB | solo texto | cabe |
| `gemma4:e2b` | 7.2 GB | texto, imagen, **audio** | descartado |
| `gemma4:e4b` | 9.6 GB | texto, imagen, audio | no cabe |

Las Gemma 4 E son multimodales: cargan encoders que este sistema no usa, y `e2b` son 7.2 GB contra
~7.4 GB libres, sin sitio para la KV cache. Quedan fuera por incumplir el criterio text-only.

**Sobre el formato JSON.** Ollama restringe la generación al esquema token a token, así que JSON
malformado es mecánicamente imposible en cualquiera de estos modelos. Los fallos de validación que
mide la columna correspondiente no son de sintaxis sino de los validadores que ningún JSON Schema
expresa: que las observaciones sean de la dimensión pedida, que los ids no se repitan, que una
decisión accionable traiga invalidación. Eso sí depende del modelo, y es lo que la tabla medirá.

## Resultados

<!-- Sustituir por la salida de `python -m crypto_agents.ablation`. -->

Pendiente de la segunda corrida. El camino ya está despejado: los seis modelos responden con el
prompt y el esquema reales. Los 880 del decisor son un ritmo por ventana de 5 h, no un total: el
límite que aprieta es el mensual compartido de la suscripción, y el consumo real se mide en el
journal.

El re-sondeo del 15 de agosto de 2026 dejó dos cosas escritas antes de correr:

- **Cinco de seis modelos no fallan de contenido.** 12 de 12 válidos en `mimo-v2.5`, `hy3`,
  `minimax-m3` y `deepseek-v4-flash`, 4 de 4 en `kimi-k2.6`. El único que reintenta es el decisor:
  2 de 13 intentos devolvieron `invalidation_price` como cadena, y el reintento con el error
  adjunto validó las dos veces.
- **`bull` cambió de modelo por disponibilidad, no por calidad.** `qwen3.7-plus` respondía 0/10 con
  503 y la familia qwen entera con él. `kimi-k2.6` la sustituye y mantiene seis familias distintas
  entre los seis primarios.

Nota: ese re-sondeo es anterior a 95a8b0f y solo contaba fallos de esquema, por lo que no es
comparable con la tasa de fallo de validación actual.

## Enmienda posterior a la primera corrida
La primera corrida (26-28 sep 2026) se descarta: no persistió journals, su caché mezcló
respuestas de backends distintos, puntuó como ganancia órdenes con el stop del lado equivocado
y no se puede reproducir porque los reintentos quedaron bajo claves inalcanzables. Su tabla se
conserva abajo como registro, no como evidencia. Los criterios originales no se evaluaron sobre
ella.

Escrita antes de la segunda corrida. Además de los criterios originales:
1. `full` justifica su coste frente a otro brazo solo si la diferencia pareada del retorno por
   evaluación, con parada común, es positiva y su intervalo al 95% excluye 0. Un intervalo que
   incluye 0 es un empate, y un empate es derrota para `full`. Se aplica contra `solo`,
   `no_debate` y `bull_only`.
2. `local_technicals` no pierde decisión si decide dentro de 5 puntos de `full` y su tasa media
   de fallo de validación no supera la de `full`. (El 5 es arbitrario, fijado aquí.)
3. Ningún brazo tiene señal si su retorno por evaluación no supera con intervalo al 95% al mejor
   de always_buy, always_sell y random_uniform. Si ninguno lo supera, se publica que la ablación
   no detecta señal y la pregunta de arquitectura queda sin responder, no resuelta.
4. Los muestreos usan temperatura 0; la coincidencia entre brazos mide diferencia de decisiones,
   no de calidad, y no se usa para concluir.
5. En la ablación el gate de riesgo opera con una cuenta congelada: solo `invalid_stop_side`
   puede vetar. La ablación no ejercita drawdown, cooldown ni exposición.

## Enmienda 2 (previa a la segunda corrida)
Escrita tras revisar la enmienda anterior, antes de ver ningún resultado de la segunda corrida.
Sustituye los criterios 1, 3 y 4 de la enmienda anterior; mantiene el 2 y el 5 con los ajustes
indicados, y añade el 6, el 7 y el 8. El historial de git conserva el texto previo.

1. Para cada comparación de `full` contra `solo`, `no_debate` y `bull_only`, sea D la diferencia
   pareada del retorno por evaluación con parada común (`paired_difference`, IC 95%).
   - `full` justifica su coste en esa comparación solo si D > 0 y el IC excluye 0.
   - Es empate (derrota para `full`) si el IC incluye 0 y su semiancho es <= δ.
   - Es no concluyente si el IC incluye 0 y su semiancho es > δ: el diseño no distingue, y
     no se lee ni como empate ni como derrota.
   - Es peor si el IC queda entero por debajo de 0: `full` decide peor que el brazo
     comparado; se lee como derrota para `full`, más fuerte que el empate. (Añadido
     antes de la segunda corrida, al detectar que criteria.py ya emitía este veredicto.)
   δ = 0.20 % por evaluación, fijado antes de ver resultados, a partir de comisión taker de ida y
   vuelta de perpetuos (~0.10–0.11 %), funding (0.003–0.014 % de media por 24 h, medido en
   binanceusdm por el sondeo del 2026-10-03; p95 0.03 % en seis de siete símbolos, 0.04 % en BNB)
   y slippage (0.02–0.05 % por lado).
   Una evaluación sin decisión o sin orden puntúa 0, igual que `hold`.
2. (Se mantiene.) La tasa de fallo de validación es `validation_failure` (inválidas sobre
   respondidas, ponderada por intentos), desglosada por `failure_kind`.
3. Un brazo tiene señal solo si la diferencia pareada de su retorno por evaluación supera, con
   IC 95% excluyendo 0, a cada una de las cuatro líneas base: `always_buy`, `always_sell`,
   `random_uniform` y `rule_trend`. Si ningún brazo lo hace, se publica que la ablación no
   detecta señal y la pregunta de arquitectura queda sin responder. No se corrige la
   multiplicidad (hasta 10 brazos x 4 líneas base): un positivo aislado es una hipótesis a
   replicar con otra semilla de selección, no un hallazgo.
   Este criterio gobierna la pregunta de señal. La viñeta de «Conclusión» sobre
   `random_uniform` y `rule_trend` con el cierre del horizonte queda como lectura
   descriptiva: si discrepa de este criterio, manda este. (Añadido antes de la segunda
   corrida, al detectar la ambigüedad.)
4. La coincidencia de acción entre brazos solo concluye en un sentido: los criterios originales
   (>90%) siguen vigentes como señal de redundancia. Una coincidencia baja no indica que el brazo
   más caro decida mejor y no justifica su coste.
5. (Se mantiene.) En la ablación el gate de riesgo opera con una cuenta congelada: solo
   `invalid_stop_side` puede vetar. Cualquier otro veto en la corrida la invalida (criterio 6).
6. (Nuevo.) Una corrida es inválida y no se interpreta si ocurre cualquiera de estas: hay
   evaluaciones perdidas por cuota del decisor; hay entradas de caché atribuidas a un backend
   distinto del declarado para el brazo; hay un veto distinto de `invalid_stop_side`; hay
   evaluaciones perdidas por saldo agotado (el `402 Insufficient account funds` del pago por
   uso), en cualquier nodo y en cualquier brazo (añadida antes de la segunda corrida, al
   implementar el guarda de saldo en T3); hay evaluaciones perdidas por un fallo del proveedor
   —un rechazo antes de producir contenido o un plazo vencido (`failure_kind` `transport` o
   `timeout`)—, en cualquier nodo y en cualquier brazo, juzgadas sobre el último eslabón de una
   corrida reanudada igual que las de saldo (añadida antes de la segunda corrida, en T5, tras 15
   rechazos 404 seguidos de deepseek-v4-flash). Las evaluaciones perdidas por fallos de
   validación del propio modelo no invalidan: son parte de lo que se mide.
7. (Nuevo.) Potencia. Con n = 140 y σ = 4.37 % (cota: la mayor de las desviaciones de always_buy
   y always_sell sobre el pool de 4 740 activaciones), el efecto mínimo detectable de la
   diferencia pareada es 1.03 % por evaluación con ρ = 0.5 y 1.46 % con ρ = 0 (IC 95 %,
   potencia 80 %). El veredicto «empate» exige n >= (1.96 x 4.37 / δ)^2 (ρ = 0.5). Con n = 140
   los veredictos esperables del criterio 1 son «justifica» o «no concluyente»; «no concluyente»
   no afirma que el brazo caro sea peor ni que empate. La regla por defecto es conservar el
   brazo más simple hasta que una corrida con n suficiente diga otra cosa.
8. (Nuevo.) Orden de las etapas. Etapa 1 (n = 140, brazos LLM): cribado operativo; se interpreta
   por fallo de validación, evaluaciones perdidas, llamadas, consumo y validez de la corrida
   (criterio 6), no por retorno. Etapa 2: `solo` con n mayor frente a las cuatro líneas base
   (criterio 3). `full` frente a `solo` (criterio 1) solo se mide con n mayor si `solo` mostró
   señal en la etapa 2 o el cribado muestra una diferencia que lo justifique, porque el n
   alcanzable lo limita el saldo de pago por uso
   (corregido antes de la segunda corrida, tras confirmar con soporte que la evaluación va por pago por uso), no el tiempo.

## Enmienda 3 (previa a la etapa 1)
Escrita el 2026-10-07, antes de cualquier llamada de la etapa 1 y antes de ver ningún resultado.
Cambia qué brazos corre la etapa 1 y añade una regla de decisión y una condición al criterio 6. No
toca δ, n, el manifiesto ni el texto de las enmiendas 1 y 2; de los criterios 2 a 8 toca solo lo
que se dice abajo, y lo dice.

**Por qué.** La potencia de la etapa 1, tal como la imprime `python -m crypto_agents.dispersion`
sobre el pool de 4 740 activaciones (no se copia ninguna media de retorno: el módulo no calcula
ninguna):

| n | rho | sigma_dif | semiancho IC95% | detectable |
| ---: | ---: | ---: | ---: | ---: |
| 140 | 0.0 | 6.17% | 1.02% | 1.46% |
| 140 | 0.5 | 4.37% | 0.72% | 1.03% |
| 280 | 0.0 | 6.17% | 0.72% | 1.03% |
| 280 | 0.5 | 4.37% | 0.51% | 0.73% |
| 420 | 0.0 | 6.17% | 0.59% | 0.84% |
| 420 | 0.5 | 4.37% | 0.42% | 0.60% |

Con n = 140 lo detectable es entre cinco y siete veces δ = 0.20 %, y el semiancho no baja de δ ni
con n = 420: el veredicto «empate» no está al alcance de ninguno de los tres tamaños. Seis brazos
con modelo sobre 140 activaciones cuestan entre 27.46 y 30.76 USD (estimación del bloque T6) para
un criterio 1 que casi seguro dirá «no concluyente» tres veces. Se corre lo que contesta la
pregunta central, `full` frente a `solo`, y el resto solo si esa respuesta lo pide.

Una nota sobre esa tabla, que es una cota y no una predicción: con el stop común, dos brazos que
proponen la misma acción en una activación tienen exactamente el mismo retorno en ella, así que su
diferencia es cero. La dispersión de la diferencia pareada depende entonces de en cuántas
activaciones discrepan `full` y `solo`, no de un ρ barrido entre 0 y 0.5. Tras la etapa 1 se
recalcula con el acuerdo medido, y esa cifra sustituye a la tabla de arriba para decidir el n de
cualquier etapa posterior.

1. **Etapa 1.** Corre `full`, `solo` y las cuatro líneas base (`always_buy`, `always_sell`,
   `random_uniform`, `rule_trend`) sobre el manifiesto actual, n = 140
   (`data/ablation_selection.json`, sha-256
   `73f87870cf24d06c341ff75fa72f164f9cd5202623cf0d76babbdab4a2803f23`). `no_debate` y `bull_only`
   no se corren salvo por la regla del punto 3. Sustituye a «brazos LLM» en el criterio 8.
2. **Mapa de roles de la etapa 1**, fijado aquí y registrado por la corrida en su `meta.json`:

   | rol | modelo | familia | salida estructurada |
   | --- | --- | --- | --- |
   | structure | `glm-5.3-flash` | zhipu | json_schema |
   | momentum | `deepseek-v4-pro` | deepseek | json_schema |
   | volume | `glm-5.3-flash` | zhipu | json_schema |
   | bull | `qwen3.8-max` | qwen | json_schema |
   | bear | `minimax-m3` | minimax | json_schema |
   | decider | `glm-5.2` | zhipu | function_calling |

   Todos contra OpenCode Zen, pago por uso, con el plazo del proveedor en 240 s. Lo que este mapa
   no tiene y el sistema pedía: tres de los seis roles quedan en familia zhipu, y dos de las tres
   lecturas técnicas salen del mismo modelo, que es el único candidato cuya evidencia leyeron las
   mesas y el decisor en todos los sondeos. La pasarela, además, sirve `glm-5.2` como
   `accounts/fireworks/models/glm-5p3`. La restricción dura (bull ≠ bear) se cumple.

   El bear se queda en `minimax-m3` tras comprobar la asimetría entre mesas que su precio hacía
   temer (cuesta de 12 a 22 veces menos por alegato que el bull). Sobre 12 activaciones, de forma
   descriptiva y sin pruebas, la mesa escueta resultó ser el bull: menos claims (3.42 frente a
   4.58), contraargumento la mitad de largo, menos observaciones citadas. Otro bear no corrige eso.
   El bull se eligió con esa tabla delante; está en `CLAUDE.md`, bloque T7.
3. **Regla de decisión**, escrita antes de los datos. Se lee el veredicto que `python -m
   crypto_agents.criteria <directorio> --delta 0.002` imprime en la fila `| full | solo |` del
   criterio 1, y se hace esto:

   | veredicto | qué se hace |
   | --- | --- |
   | `justifica` | Se corren `no_debate` y `bull_only` sobre el mismo manifiesto y se evalúan con el criterio 1. |
   | `empate` o `peor` | Derrota de `full` (criterio 1): el pipeline es `solo`. |
   | `no concluyente` | Ni derrota ni empate, como dice el criterio 1. Se conserva `solo` por la regla por defecto del criterio 7. Es el resultado que la tabla de arriba hace más probable. |
   | `no determinado: <por qué>` | La etapa 1 queda abierta: el informe dice por qué no hubo intervalo y no se elige pipeline. |

   Tras `empate`, `peor` o `no concluyente`, la etapa 2 sigue el criterio 8 sin cambios.
4. **Excepción explícita y limitada al criterio 8.** El criterio 8 dice que la etapa 1 se interpreta
   por fallo de validación, evaluaciones perdidas, llamadas, consumo y validez, «no por retorno».
   La regla del punto 3 lee un veredicto por retorno. Es la única lectura por retorno de la etapa 1
   y tiene un solo uso, el de esa tabla. No se publica como hallazgo sobre la arquitectura: con
   n = 140, `justifica` es una hipótesis que la etapa siguiente tiene que sostener.
5. **Brazos locales aplazados.** `local_technicals` y `local_bull` no se corren en la etapa 1 ni
   por la regla del punto 3. Con pago por uso el respaldo local de un rol remoto no se activa por
   cuota —el proveedor no publica ninguna y el router ya no aplica la declarada—, y nunca se
   activó por un rechazo del proveedor: hoy esos dos brazos miden un camino que la operación no
   recorre. El criterio 2 queda sin evaluar hasta que se corran, no resuelto.
6. **Criterio 6, sexta condición.** Una corrida es inválida si hay evaluaciones cortadas por el
   tope de gasto (`--max-usd`), en cualquier nodo y en cualquier brazo, juzgadas sobre el último
   eslabón de una corrida reanudada igual que las de saldo. Con pago por uso el tope es el freno
   que se declara antes de llegar al saldo, y lo que corta no se midió.
7. **Nada más cambia.** δ = 0.20 %, n = 140 y el manifiesto quedan como están; los criterios 2 a 7
   y el orden de las etapas del 8, también, salvo lo dicho en los puntos 1, 4 y 6.

Lo que cambió en el arnés antes de la etapa 1, que no es un criterio pero mueve lo que se mide:
el reintento de un rol en `json_schema` llevaba, con el fallo en un campo anidado, un error que el
modelo no había cometido, y los dos intentos se perdían seguros. Cinco de los seis roles van en
ese modo y el decisor no, así que una evaluación perdida así puntuaba 0 en `full` y nunca en
`solo`. Está corregido (bloque T7), y los sondeos anteriores no se rehicieron.

## Etapa 1 — resultado
Escrita el 2026-10-11, con la etapa 1 terminada. Solo añade: el texto de los criterios y de las tres
enmiendas queda como estaba. Cada cifra de esta sección está copiada de la salida de uno de estos
dos comandos, que no llaman a ningún modelo; la única excepción es el n del apartado de potencia,
que va con su cuenta.

```bash
python -m crypto_agents.audit var/ablation/20261009T045033Z
python -m crypto_agents.criteria var/ablation/20261009T045033Z --delta 0.002
```

### Identidad de la corrida

De `python -m crypto_agents.audit var/ablation/20261009T045033Z`:

- directorio: `var/ablation/20261009T045033Z` (no versionado), tipo `ablation`, sin reanudar
- inicio: 2026-10-09T04:50:33.337548+00:00
- commit: `b9676eb91627a3d53b0e950d5866c3e0af06f671` (árbol limpio)
- plan (manifest): `data/ablation_selection.json`, sha-256
  `73f87870cf24d06c341ff75fa72f164f9cd5202623cf0d76babbdab4a2803f23`
- argumentos: `--manifest data/ablation_selection.json --arms
  full,solo,always_buy,always_sell,random_uniform,rule_trend --fill --max-usd 14`
- base_url: `https://opencode.ai/zen/v1`; timeout de OpenAI: 240 s; tope de gasto: 14 USD
- máquina: la laptop, según el operador. `meta.json` no la registra.

| brazo | archivo | sha-256 | evaluaciones |
| --- | --- | --- | --- |
| `full` | `full.jsonl` | `5d7f1a63f3aaf1c8469a572aa3aaff6184e671ff67698a1d3632a15645ada33f` | 140 |
| `solo` | `solo.jsonl` | `1180554e2c19aab20d623f2112547aae8f930ad934d3fc17ca65fd5c70cc4b17` | 140 |
| `always_buy` | `always_buy.jsonl` | `77fe118e5efb5714384c3d3485baf31a9c80bdbded2c34f2b775c9667fddd1ff` | 140 |
| `always_sell` | `always_sell.jsonl` | `ba1643fb2ce45114f728418e8894eb46b55f125fe2cef1fe0211e273746bc1d9` | 140 |
| `random_uniform` | `random_uniform.jsonl` | `607b738ea2ee764d4ee8433adc0cac748f738761dfcea418f1a6690cd6ff6198` | 140 |
| `rule_trend` | `rule_trend.jsonl` | `3b8d58f9ce1b22699a7f0d92b2dd47e91235c3db9f0ef3b53d3bed215b19ab93` | 140 |

El mapa de roles que el audit lee de `meta.json` es el del punto 2 de la enmienda 3, rol por rol.

### Validez (criterio 6)

De `python -m crypto_agents.criteria var/ablation/20261009T045033Z --delta 0.002`, que termina con
código 0. Las seis condiciones de invalidez están en cero en los seis brazos: la corrida es válida y
se interpreta.

| brazo | decisor perdido por cuota | saldo insuficiente (402) | fallo del proveedor | tope de gasto | caché de otro backend | vetos ajenos | avisos |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full | 0 | 0 | 0 | 0 | 0 | 0 | — |
| solo | 0 | 0 | 0 | 0 | 0 | 0 | — |
| always_buy | 0 | 0 | 0 | 0 | 0 | 0 | — |
| always_sell | 0 | 0 | 0 | 0 | 0 | 0 | — |
| random_uniform | 0 | 0 | 0 | 0 | 0 | 0 | — |
| rule_trend | 0 | 0 | 0 | 0 | 0 | 0 | — |

### Criterio 1: `full` frente a `solo`

Del mismo comando (`criteria … --delta 0.002`): δ = 0.002 (fracción de retorno por evaluación),
horizonte 6 velas, puntuación con stop común, diferencias pareadas por `run_id`, IC normal al 95%.
Las dos últimas columnas son el neto descriptivo y no entran en el veredicto.

| brazo | referencia | n | diferencia media | EE | IC 95% | semiancho | veredicto | neto (descriptivo, no entra en los criterios de la enmienda 2): n | neto (descriptivo, no entra en los criterios de la enmienda 2): diferencia media [IC 95%] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | solo | 140 | +0.017% | 0.192% | [-0.360%, +0.393%] | 0.377% | no concluyente | 140 | -0.010% [-0.385%, +0.366%] |

Las filas `| full | no_debate |` y `| full | bull_only |` salen `no determinado`: esos brazos no se
corrieron y no tienen evaluaciones.

### Criterio 3: cada brazo contra las cuatro líneas base

Del mismo comando (`criteria … --delta 0.002`):

| brazo | referencia | n | diferencia media | EE | IC 95% | semiancho | veredicto | neto (descriptivo, no entra en los criterios de la enmienda 2): n | neto (descriptivo, no entra en los criterios de la enmienda 2): diferencia media [IC 95%] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | always_buy | 140 | -0.432% | 0.418% | [-1.252%, +0.388%] | 0.820% | no concluyente | 140 | -0.404% [-1.225%, +0.416%] |
| full | always_sell | 140 | +0.301% | 0.361% | [-0.407%, +1.009%] | 0.708% | no concluyente | 140 | +0.312% [-0.396%, +1.020%] |
| full | random_uniform | 140 | -0.044% | 0.396% | [-0.820%, +0.732%] | 0.776% | no concluyente | 140 | -0.067% [-0.843%, +0.708%] |
| full | rule_trend | 140 | -0.029% | 0.223% | [-0.467%, +0.409%] | 0.438% | no concluyente | 140 | -0.116% [-0.553%, +0.321%] |
| solo | always_buy | 140 | -0.449% | 0.345% | [-1.126%, +0.228%] | 0.677% | no concluyente | 140 | -0.395% [-1.071%, +0.282%] |
| solo | always_sell | 140 | +0.284% | 0.375% | [-0.451%, +1.020%] | 0.735% | no concluyente | 140 | +0.322% [-0.413%, +1.057%] |
| solo | random_uniform | 140 | -0.061% | 0.364% | [-0.775%, +0.654%] | 0.714% | no concluyente | 140 | -0.058% [-0.771%, +0.655%] |
| solo | rule_trend | 140 | -0.046% | 0.186% | [-0.411%, +0.319%] | 0.365% | no concluyente | 140 | -0.106% [-0.473%, +0.260%] |

- `full`: supera a 0 de 4 líneas base.
- `solo`: supera a 0 de 4 líneas base.

Ningún brazo supera a ninguna línea base con el IC excluyendo 0. Por el criterio 3 se publica eso:
la ablación no detecta señal y la pregunta de arquitectura queda sin responder, no resuelta.

### Decisión

El veredicto de la fila `| full | solo |` es `no concluyente`. La regla del punto 3 de la enmienda
3, escrita antes de los datos, dice qué se hace con él: no se lee como ningún otro veredicto del
criterio 1, y se conserva `solo` por la regla por defecto del criterio 7 (el brazo más simple hasta que una
corrida con n suficiente diga otra cosa). `no_debate` y `bull_only` no se corren: solo `justifica`
los pedía. La etapa 2 sigue el criterio 8 sin cambios.

La decisión no dice nada sobre cuál de los dos brazos decide mejor: el diseño no los distingue a
este n, y eso es lo único que la fila afirma.

### Lo que este n podía decir

Con la regla del criterio 1 el intervalo solo cuenta como igualdad si su semiancho es ≤ δ. El
semiancho observado es 0.377% con n = 140 y δ es 0.20 %; como el semiancho cae con la raíz de n,
llegar a δ con este mismo error estándar exigía

    n ≈ 140 · (0.377 / 0.20)² ≈ 500

Así que a n = 140 la etapa solo podía terminar en «justifica», en «no concluyente» o, con el IC
entero por debajo de cero, en «peor». El cuarto veredicto no estaba al alcance. Es lo que la
enmienda 3 anticipó con una cota; esta cuenta usa el error estándar medido.

### Quién respondió por el decisor

De `python -m crypto_agents.audit var/ablation/20261009T045033Z`, sección «Quién respondió».
`modelo pedido` es el id de la configuración; las otras dos columnas son lo que la pasarela declaró
en cada respuesta.

| rol | modelo pedido | upstream_model | upstream_endpoint | intentos | de caché |
| --- | --- | --- | --- | --- | --- |
| decider | `glm-5.2` | `accounts/fireworks/models/glm-5p3` | `fireworks` | 279 | 0 |
| structure | `glm-5.3-flash` | `accounts/fireworks/models/glm-5p3-flash` | `fireworks` | 73 | 0 |
| structure | `glm-5.3-flash` | `z-ai/glm-5.3-flash` | `relace-glm5.3flash` | 68 | 0 |
| volume | `glm-5.3-flash` | `accounts/fireworks/models/glm-5p3-flash` | `fireworks` | 74 | 0 |
| volume | `glm-5.3-flash` | `z-ai/glm-5.3-flash` | `relace-glm5.3flash` | 69 | 0 |

- **El decisor se pidió como `glm-5.2` y lo sirvió `accounts/fireworks/models/glm-5p3` en sus 279
  intentos**: 139 en `full` y 140 en `solo`, un solo upstream. Los dos brazos comparados leyeron al
  mismo modelo; lo que el nombre de la configuración dice de él no es lo que respondió.
- **`glm-5.3-flash` salió por dos upstreams** en structure y en volume, que solo existen en `full`.
  El reparto de arriba es del audit. Que la proporción entre los dos cambió a lo largo de la corrida
  no lo tabula el audit: se lee de `LLMCall.at` y `LLMCall.upstream_endpoint` en `full.jsonl`, y
  aquí no se da ninguna cifra de ello.

### Una ejecución descartada: `--delta 0.20`

Antes de la de arriba se ejecutó `python -m crypto_agents.criteria var/ablation/20261009T045033Z
--delta 0.20`. `--delta` es una fracción de retorno, no un porcentaje: 0.20 es δ = 20 %, no el δ
de la enmienda 2 (0.20 %, que se escribe `--delta 0.002`). Con ese valor todo semiancho queda
por debajo de δ, y la fila `| full | solo |` y las ocho del criterio 3 imprimen «empate». Las cifras
de cada fila son las mismas; solo cambia la palabra. Esa salida se descarta: no evalúa el criterio
que se fijó antes de los datos. El comando la reproduce, y hoy nada lo impide; el guarda va en otro
bloque.

### Defecto conocido: los tramos de la tabla descriptiva

La tabla «Diferencia pareada descriptiva» de `criteria` agrupa `full` menos `solo` por símbolo y por
tramo del manifiesto. Por tramo imprime esto:

| referencia | grupo | n | diferencia media |
| --- | --- | --- | --- |
| solo | tramo 0 | 1 | +0.000% |
| solo | tramo sin tramo | 139 | +0.017% |

El manifiesto reparte las 140 activaciones en 5 tramos de 28 (4 por símbolo y tramo, siete
símbolos), así que deberían ser cinco filas. Es un defecto de la tabla, conocido, y se corrige en
otro bloque. No toca nada de lo anterior: esa tabla no lleva veredicto ni intervalo, y los criterios
1, 3 y 6 no leen el tramo.

## Recálculo de dispersión (enmienda 3)
Escrita el 2026-10-11, con la etapa 1 terminada. Solo añade: el texto de los criterios, de las tres
enmiendas y de «Etapa 1 — resultado» queda como estaba, y «Conclusión» sigue sin escribir.

La enmienda 3 dejó dicho, antes de los datos, qué había que hacer con su tabla de potencia:

> Una nota sobre esa tabla, que es una cota y no una predicción: con el stop común, dos brazos que
> proponen la misma acción en una activación tienen exactamente el mismo retorno en ella, así que su
> diferencia es cero. La dispersión de la diferencia pareada depende entonces de en cuántas
> activaciones discrepan `full` y `solo`, no de un ρ barrido entre 0 y 0.5. Tras la etapa 1 se
> recalcula con el acuerdo medido, y esa cifra sustituye a la tabla de arriba para decidir el n de
> cualquier etapa posterior.

Esto es ese recálculo. Lo hace un comando que lee el directorio de la corrida y no llama a ningún
modelo. Termina con código 0, que es lo que da cuando `sigma_D/√n` reproduce el EE de `criteria` en
todos los pares y ninguna diferencia es distinta de cero entre dos brazos con la misma acción:

```bash
python -m crypto_agents.paired_dispersion var/ablation/20261009T045033Z --delta 0.002
```

Tres cosas que el comando hace y la enmienda no pedía, para que no pasen por parte de la promesa:

- La enmienda habla de `full` frente a `solo`. Las ocho filas contra las líneas base se añaden
  porque son las comparaciones del criterio 3, que es lo que mide la etapa 2 del criterio 8.
- `sigma_D` se mide directamente sobre las 140 diferencias pareadas, las mismas que resta
  `criteria`; no se deduce del acuerdo. El acuerdo va al lado, con cuántas evaluaciones discrepan
  y cuántas diferencias no son cero, que es lo que la nota de la enmienda afirmaba.
- La nota pedía sustituir una tabla de dos columnas, semiancho y detectable. La sección 3 las
  lleva las dos. El detectable es del diseño —qué diferencia real distingue un n—, no una
  potencia para la diferencia observada, que no se calcula en ningún sitio.

Lo que la tabla de la enmienda decía y lo que salió, para `full` frente a `solo` (las dos primeras
filas son de la enmienda 3; la tercera, de la sección 3 de abajo):

| | sigma de la diferencia | semiancho con n = 140 | con n = 280 | con n = 420 |
| --- | ---: | ---: | ---: | ---: |
| cota de la enmienda 3, ρ = 0 | 6.17% | 1.02% | 0.72% | 0.59% |
| cota de la enmienda 3, ρ = 0.5 | 4.37% | 0.72% | 0.51% | 0.42% |
| medido en la etapa 1 | 2.27% | 0.38% | 0.27% | 0.22% |

El «n ≈ 500» del apartado «Lo que este n podía decir» es el 496.4 de la sección 2, que al múltiplo
de 35 son 525.

El reporte da n, viabilidad y coste. Qué etapa se corre, y con qué n, no lo decide él ni esta
sección.

Lo que sigue es la salida del comando, sin editar, con los títulos un nivel más abajo.

Reproduce: `python -m crypto_agents.paired_dispersion var/ablation/20261009T045033Z --delta 0.002`

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
- δ = 0.002 (0.200% de retorno por evaluación) · horizonte 6 velas · puntuación: stop común, retorno bruto · diferencias pareadas por `run_id`, las mismas que resta `criteria`
- todo n se redondea hacia arriba al múltiplo de 35: 7 símbolos por 5 tramos del manifiesto, que tiene 140 evaluaciones
- bootstrap: 10000 remuestreos de las evaluaciones, con reemplazo y dentro de cada símbolo, semilla 20261011, percentiles 2.5 y 97.5 por rango más cercano

### 1. Acuerdo de acción y dispersión de la diferencia pareada

`sigma_D` es la desviación estándar muestral (n - 1) de la diferencia `brazo - referencia` por evaluación. El acuerdo cuenta como categorías buy, sell, hold y «no decidió». Con el stop común dos brazos con la misma acción rinden lo mismo, así que una diferencia solo puede ser distinta de 0 donde discrepan: la columna «≠ 0 con la misma acción» tiene que ser 0. `sigma_D/√n` tiene que ser el EE que imprime `criteria`.

La enmienda 3 promete este recálculo para `full` frente a `solo`. Las filas contra las líneas base se añaden porque son las comparaciones del criterio 3, que es lo que la etapa 2 del criterio 8 mide.

| brazo | referencia | n | acuerdo de acción | discrepan | diferencias ≠ 0 | ≠ 0 con la misma acción | sigma_D | rango bootstrap de sigma_D | sigma_D/√n | EE de criteria | reproduce |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | solo | 140 | 102/140 (73%) | 38 | 38 | 0 | 2.273% | [1.586%, 2.876%] | 0.192% | 0.192% | sí |
| full | always_buy | 140 | 48/140 (34%) | 92 | 92 | 0 | 4.951% | [4.029%, 5.827%] | 0.418% | 0.418% | sí |
| full | always_sell | 140 | 69/140 (49%) | 71 | 71 | 0 | 4.274% | [3.414%, 5.052%] | 0.361% | 0.361% | sí |
| full | random_uniform | 140 | 43/140 (31%) | 97 | 97 | 0 | 4.686% | [3.843%, 5.447%] | 0.396% | 0.396% | sí |
| full | rule_trend | 140 | 52/140 (37%) | 88 | 87 | 0 | 2.642% | [2.154%, 3.081%] | 0.223% | 0.223% | sí |
| solo | always_buy | 140 | 44/140 (31%) | 96 | 96 | 0 | 4.087% | [3.354%, 4.765%] | 0.345% | 0.345% | sí |
| solo | always_sell | 140 | 46/140 (33%) | 94 | 94 | 0 | 4.440% | [3.687%, 5.126%] | 0.375% | 0.375% | sí |
| solo | random_uniform | 140 | 37/140 (26%) | 103 | 103 | 0 | 4.312% | [3.572%, 4.963%] | 0.364% | 0.364% | sí |
| solo | rule_trend | 140 | 74/140 (53%) | 66 | 66 | 0 | 2.203% | [1.717%, 2.670%] | 0.186% | 0.186% | sí |

Reproduce: `python -m crypto_agents.paired_dispersion var/ablation/20261009T045033Z --delta 0.002` (los EE: `python -m crypto_agents.criteria var/ablation/20261009T045033Z --delta 0.002`)

### 2. n que pide cada par

- **n para semiancho ≤ δ**: `(1.96·sigma_D/δ)²`. Desde ese n el intervalo, si incluye 0, es un «empate» y no un «no concluyente».
- **n para potencia 80%**: `((1.96 + z)·sigma_D/δ)²`, con z el cuantil normal de 80%. Es el n con el que una diferencia real de δ daría «justifica» con probabilidad 80%. No es la potencia para la diferencia observada, que no se calcula.
- Cada n va con `sigma_D` puntual y con los dos extremos de su rango bootstrap.

| brazo | referencia | sigma_D | valor | n para semiancho ≤ δ | al múltiplo de 35 | n para potencia 80% | al múltiplo de 35 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full | solo | extremo bajo | 1.586% | 241.7 | 245 | 493.8 | 525 |
| full | solo | puntual | 2.273% | 496.4 | 525 | 1014.2 | 1015 |
| full | solo | extremo alto | 2.876% | 794.5 | 805 | 1623.4 | 1645 |
| full | always_buy | extremo bajo | 4.029% | 1558.9 | 1575 | 3185.0 | 3220 |
| full | always_buy | puntual | 4.951% | 2354.3 | 2380 | 4810.4 | 4830 |
| full | always_buy | extremo alto | 5.827% | 3261.2 | 3290 | 6663.2 | 6685 |
| full | always_sell | extremo bajo | 3.414% | 1119.6 | 1120 | 2287.6 | 2310 |
| full | always_sell | puntual | 4.274% | 1754.0 | 1785 | 3583.7 | 3605 |
| full | always_sell | extremo alto | 5.052% | 2451.1 | 2485 | 5008.0 | 5040 |
| full | random_uniform | extremo bajo | 3.843% | 1418.6 | 1435 | 2898.4 | 2905 |
| full | random_uniform | puntual | 4.686% | 2108.8 | 2135 | 4308.6 | 4340 |
| full | random_uniform | extremo alto | 5.447% | 2849.3 | 2870 | 5821.7 | 5845 |
| full | rule_trend | extremo bajo | 2.154% | 445.7 | 455 | 910.7 | 945 |
| full | rule_trend | puntual | 2.642% | 670.3 | 700 | 1369.5 | 1400 |
| full | rule_trend | extremo alto | 3.081% | 911.9 | 945 | 1863.1 | 1890 |
| solo | always_buy | extremo bajo | 3.354% | 1080.3 | 1085 | 2207.3 | 2240 |
| solo | always_buy | puntual | 4.087% | 1604.2 | 1610 | 3277.7 | 3290 |
| solo | always_buy | extremo alto | 4.765% | 2180.8 | 2205 | 4455.8 | 4480 |
| solo | always_sell | extremo bajo | 3.687% | 1305.9 | 1330 | 2668.1 | 2695 |
| solo | always_sell | puntual | 4.440% | 1892.9 | 1925 | 3867.6 | 3885 |
| solo | always_sell | extremo alto | 5.126% | 2523.5 | 2555 | 5156.0 | 5180 |
| solo | random_uniform | extremo bajo | 3.572% | 1225.5 | 1260 | 2503.9 | 2520 |
| solo | random_uniform | puntual | 4.312% | 1785.6 | 1820 | 3648.4 | 3675 |
| solo | random_uniform | extremo alto | 4.963% | 2365.4 | 2380 | 4833.0 | 4865 |
| solo | rule_trend | extremo bajo | 1.717% | 283.2 | 315 | 578.7 | 595 |
| solo | rule_trend | puntual | 2.203% | 466.0 | 490 | 952.0 | 980 |
| solo | rule_trend | extremo alto | 2.670% | 684.6 | 700 | 1398.9 | 1400 |

Reproduce: `python -m crypto_agents.paired_dispersion var/ablation/20261009T045033Z --delta 0.002`

### 3. Semiancho y diferencia detectable con n = 140, 280, 420

La tabla de `python -m crypto_agents.dispersion` que la enmienda 3 copió, con la `sigma_D` medida de cada par en lugar de una cota y un rho barrido: semiancho `1.96·sigma_dif/√n`, detectable `(1.96 + z)·sigma_dif/√n`.

| brazo | referencia | n | sigma_dif | semiancho IC95% | detectable |
| --- | --- | ---: | ---: | ---: | ---: |
| full | solo | 140 | 2.27% | 0.38% | 0.54% |
| full | solo | 280 | 2.27% | 0.27% | 0.38% |
| full | solo | 420 | 2.27% | 0.22% | 0.31% |
| full | always_buy | 140 | 4.95% | 0.82% | 1.17% |
| full | always_buy | 280 | 4.95% | 0.58% | 0.83% |
| full | always_buy | 420 | 4.95% | 0.47% | 0.68% |
| full | always_sell | 140 | 4.27% | 0.71% | 1.01% |
| full | always_sell | 280 | 4.27% | 0.50% | 0.72% |
| full | always_sell | 420 | 4.27% | 0.41% | 0.58% |
| full | random_uniform | 140 | 4.69% | 0.78% | 1.11% |
| full | random_uniform | 280 | 4.69% | 0.55% | 0.78% |
| full | random_uniform | 420 | 4.69% | 0.45% | 0.64% |
| full | rule_trend | 140 | 2.64% | 0.44% | 0.63% |
| full | rule_trend | 280 | 2.64% | 0.31% | 0.44% |
| full | rule_trend | 420 | 2.64% | 0.25% | 0.36% |
| solo | always_buy | 140 | 4.09% | 0.68% | 0.97% |
| solo | always_buy | 280 | 4.09% | 0.48% | 0.68% |
| solo | always_buy | 420 | 4.09% | 0.39% | 0.56% |
| solo | always_sell | 140 | 4.44% | 0.74% | 1.05% |
| solo | always_sell | 280 | 4.44% | 0.52% | 0.74% |
| solo | always_sell | 420 | 4.44% | 0.42% | 0.61% |
| solo | random_uniform | 140 | 4.31% | 0.71% | 1.02% |
| solo | random_uniform | 280 | 4.31% | 0.51% | 0.72% |
| solo | random_uniform | 420 | 4.31% | 0.41% | 0.59% |
| solo | rule_trend | 140 | 2.20% | 0.36% | 0.52% |
| solo | rule_trend | 280 | 2.20% | 0.26% | 0.37% |
| solo | rule_trend | 420 | 2.20% | 0.21% | 0.30% |

Reproduce: `python -m crypto_agents.paired_dispersion var/ablation/20261009T045033Z --delta 0.002`

### 4. Viabilidad y coste de cada n redondeado

- **por celda**: n entre las 35 celdas (símbolo, tramo).
- **reutilizando las 140**: contra la celda más pequeña del pool, DOGE/USDT tramo 4, con 103 activaciones confirmadas.
- **sin ellas**: contra la celda con menos activaciones que la corrida no usó, DOGE/USDT tramo 4, con 99.
- **n adicional**: n menos las 140 ya corridas.
- **USD**: n por el coste por evaluación de los brazos con modelo del par (sección 6); las líneas base no llaman a ningún modelo. **Horas**: n por la pared por evaluación de esos brazos, uno detrás de otro (sección 7).

| brazo | referencia | objetivo | sigma_D | n | por celda | reutilizando las 140 | sin ellas | n adicional | USD del n | USD del adicional | horas de pared del n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | solo | semiancho ≤ δ | extremo bajo | 245 | 7 | cabe (7 ≤ 103) | cabe (7 ≤ 99) | 105 | 16.25 | 6.96 | 13.5 |
| full | solo | semiancho ≤ δ | puntual | 525 | 15 | cabe (15 ≤ 103) | cabe (15 ≤ 99) | 385 | 34.81 | 25.53 | 29.0 |
| full | solo | semiancho ≤ δ | extremo alto | 805 | 23 | cabe (23 ≤ 103) | cabe (23 ≤ 99) | 665 | 53.38 | 44.10 | 44.4 |
| full | solo | potencia 80% | extremo bajo | 525 | 15 | cabe (15 ≤ 103) | cabe (15 ≤ 99) | 385 | 34.81 | 25.53 | 29.0 |
| full | solo | potencia 80% | puntual | 1015 | 29 | cabe (29 ≤ 103) | cabe (29 ≤ 99) | 875 | 67.30 | 58.02 | 56.0 |
| full | solo | potencia 80% | extremo alto | 1645 | 47 | cabe (47 ≤ 103) | cabe (47 ≤ 99) | 1505 | 109.08 | 99.80 | 90.8 |
| full | always_buy | semiancho ≤ δ | extremo bajo | 1575 | 45 | cabe (45 ≤ 103) | cabe (45 ≤ 99) | 1435 | 80.67 | 73.50 | 61.5 |
| full | always_buy | semiancho ≤ δ | puntual | 2380 | 68 | cabe (68 ≤ 103) | cabe (68 ≤ 99) | 2240 | 121.90 | 114.73 | 93.0 |
| full | always_buy | semiancho ≤ δ | extremo alto | 3290 | 94 | cabe (94 ≤ 103) | cabe (94 ≤ 99) | 3150 | 168.51 | 161.34 | 128.6 |
| full | always_buy | potencia 80% | extremo bajo | 3220 | 92 | cabe (92 ≤ 103) | cabe (92 ≤ 99) | 3080 | 164.93 | 157.75 | 125.8 |
| full | always_buy | potencia 80% | puntual | 4830 | 138 | no cabe (138 > 103) | no cabe (138 > 99) | 4690 | 247.39 | 240.22 | 188.7 |
| full | always_buy | potencia 80% | extremo alto | 6685 | 191 | no cabe (191 > 103) | no cabe (191 > 99) | 6545 | 342.40 | 335.23 | 261.2 |
| full | always_sell | semiancho ≤ δ | extremo bajo | 1120 | 32 | cabe (32 ≤ 103) | cabe (32 ≤ 99) | 980 | 57.37 | 50.19 | 43.8 |
| full | always_sell | semiancho ≤ δ | puntual | 1785 | 51 | cabe (51 ≤ 103) | cabe (51 ≤ 99) | 1645 | 91.43 | 84.26 | 69.8 |
| full | always_sell | semiancho ≤ δ | extremo alto | 2485 | 71 | cabe (71 ≤ 103) | cabe (71 ≤ 99) | 2345 | 127.28 | 120.11 | 97.1 |
| full | always_sell | potencia 80% | extremo bajo | 2310 | 66 | cabe (66 ≤ 103) | cabe (66 ≤ 99) | 2170 | 118.32 | 111.15 | 90.3 |
| full | always_sell | potencia 80% | puntual | 3605 | 103 | cabe (103 ≤ 103) | no cabe (103 > 99) | 3465 | 184.64 | 177.47 | 140.9 |
| full | always_sell | potencia 80% | extremo alto | 5040 | 144 | no cabe (144 > 103) | no cabe (144 > 99) | 4900 | 258.14 | 250.97 | 196.9 |
| full | random_uniform | semiancho ≤ δ | extremo bajo | 1435 | 41 | cabe (41 ≤ 103) | cabe (41 ≤ 99) | 1295 | 73.50 | 66.33 | 56.1 |
| full | random_uniform | semiancho ≤ δ | puntual | 2135 | 61 | cabe (61 ≤ 103) | cabe (61 ≤ 99) | 1995 | 109.35 | 102.18 | 83.4 |
| full | random_uniform | semiancho ≤ δ | extremo alto | 2870 | 82 | cabe (82 ≤ 103) | cabe (82 ≤ 99) | 2730 | 147.00 | 139.83 | 112.1 |
| full | random_uniform | potencia 80% | extremo bajo | 2905 | 83 | cabe (83 ≤ 103) | cabe (83 ≤ 99) | 2765 | 148.79 | 141.62 | 113.5 |
| full | random_uniform | potencia 80% | puntual | 4340 | 124 | no cabe (124 > 103) | no cabe (124 > 99) | 4200 | 222.29 | 215.12 | 169.6 |
| full | random_uniform | potencia 80% | extremo alto | 5845 | 167 | no cabe (167 > 103) | no cabe (167 > 99) | 5705 | 299.38 | 292.21 | 228.4 |
| full | rule_trend | semiancho ≤ δ | extremo bajo | 455 | 13 | cabe (13 ≤ 103) | cabe (13 ≤ 99) | 315 | 23.30 | 16.13 | 17.8 |
| full | rule_trend | semiancho ≤ δ | puntual | 700 | 20 | cabe (20 ≤ 103) | cabe (20 ≤ 99) | 560 | 35.85 | 28.68 | 27.4 |
| full | rule_trend | semiancho ≤ δ | extremo alto | 945 | 27 | cabe (27 ≤ 103) | cabe (27 ≤ 99) | 805 | 48.40 | 41.23 | 36.9 |
| full | rule_trend | potencia 80% | extremo bajo | 945 | 27 | cabe (27 ≤ 103) | cabe (27 ≤ 99) | 805 | 48.40 | 41.23 | 36.9 |
| full | rule_trend | potencia 80% | puntual | 1400 | 40 | cabe (40 ≤ 103) | cabe (40 ≤ 99) | 1260 | 71.71 | 64.54 | 54.7 |
| full | rule_trend | potencia 80% | extremo alto | 1890 | 54 | cabe (54 ≤ 103) | cabe (54 ≤ 99) | 1750 | 96.80 | 89.63 | 73.9 |
| solo | always_buy | semiancho ≤ δ | extremo bajo | 1085 | 31 | cabe (31 ≤ 103) | cabe (31 ≤ 99) | 945 | 16.37 | 14.26 | 17.5 |
| solo | always_buy | semiancho ≤ δ | puntual | 1610 | 46 | cabe (46 ≤ 103) | cabe (46 ≤ 99) | 1470 | 24.30 | 22.18 | 26.0 |
| solo | always_buy | semiancho ≤ δ | extremo alto | 2205 | 63 | cabe (63 ≤ 103) | cabe (63 ≤ 99) | 2065 | 33.27 | 31.16 | 35.6 |
| solo | always_buy | potencia 80% | extremo bajo | 2240 | 64 | cabe (64 ≤ 103) | cabe (64 ≤ 99) | 2100 | 33.80 | 31.69 | 36.1 |
| solo | always_buy | potencia 80% | puntual | 3290 | 94 | cabe (94 ≤ 103) | cabe (94 ≤ 99) | 3150 | 49.65 | 47.54 | 53.1 |
| solo | always_buy | potencia 80% | extremo alto | 4480 | 128 | no cabe (128 > 103) | no cabe (128 > 99) | 4340 | 67.61 | 65.49 | 72.3 |
| solo | always_sell | semiancho ≤ δ | extremo bajo | 1330 | 38 | cabe (38 ≤ 103) | cabe (38 ≤ 99) | 1190 | 20.07 | 17.96 | 21.5 |
| solo | always_sell | semiancho ≤ δ | puntual | 1925 | 55 | cabe (55 ≤ 103) | cabe (55 ≤ 99) | 1785 | 29.05 | 26.94 | 31.1 |
| solo | always_sell | semiancho ≤ δ | extremo alto | 2555 | 73 | cabe (73 ≤ 103) | cabe (73 ≤ 99) | 2415 | 38.56 | 36.44 | 41.2 |
| solo | always_sell | potencia 80% | extremo bajo | 2695 | 77 | cabe (77 ≤ 103) | cabe (77 ≤ 99) | 2555 | 40.67 | 38.56 | 43.5 |
| solo | always_sell | potencia 80% | puntual | 3885 | 111 | no cabe (111 > 103) | no cabe (111 > 99) | 3745 | 58.63 | 56.51 | 62.7 |
| solo | always_sell | potencia 80% | extremo alto | 5180 | 148 | no cabe (148 > 103) | no cabe (148 > 99) | 5040 | 78.17 | 76.06 | 83.6 |
| solo | random_uniform | semiancho ≤ δ | extremo bajo | 1260 | 36 | cabe (36 ≤ 103) | cabe (36 ≤ 99) | 1120 | 19.01 | 16.90 | 20.3 |
| solo | random_uniform | semiancho ≤ δ | puntual | 1820 | 52 | cabe (52 ≤ 103) | cabe (52 ≤ 99) | 1680 | 27.46 | 25.35 | 29.4 |
| solo | random_uniform | semiancho ≤ δ | extremo alto | 2380 | 68 | cabe (68 ≤ 103) | cabe (68 ≤ 99) | 2240 | 35.92 | 33.80 | 38.4 |
| solo | random_uniform | potencia 80% | extremo bajo | 2520 | 72 | cabe (72 ≤ 103) | cabe (72 ≤ 99) | 2380 | 38.03 | 35.92 | 40.7 |
| solo | random_uniform | potencia 80% | puntual | 3675 | 105 | no cabe (105 > 103) | no cabe (105 > 99) | 3535 | 55.46 | 53.35 | 59.3 |
| solo | random_uniform | potencia 80% | extremo alto | 4865 | 139 | no cabe (139 > 103) | no cabe (139 > 99) | 4725 | 73.42 | 71.30 | 78.5 |
| solo | rule_trend | semiancho ≤ δ | extremo bajo | 315 | 9 | cabe (9 ≤ 103) | cabe (9 ≤ 99) | 175 | 4.75 | 2.64 | 5.1 |
| solo | rule_trend | semiancho ≤ δ | puntual | 490 | 14 | cabe (14 ≤ 103) | cabe (14 ≤ 99) | 350 | 7.39 | 5.28 | 7.9 |
| solo | rule_trend | semiancho ≤ δ | extremo alto | 700 | 20 | cabe (20 ≤ 103) | cabe (20 ≤ 99) | 560 | 10.56 | 8.45 | 11.3 |
| solo | rule_trend | potencia 80% | extremo bajo | 595 | 17 | cabe (17 ≤ 103) | cabe (17 ≤ 99) | 455 | 8.98 | 6.87 | 9.6 |
| solo | rule_trend | potencia 80% | puntual | 980 | 28 | cabe (28 ≤ 103) | cabe (28 ≤ 99) | 840 | 14.79 | 12.68 | 15.8 |
| solo | rule_trend | potencia 80% | extremo alto | 1400 | 40 | cabe (40 ≤ 103) | cabe (40 ≤ 99) | 1260 | 21.13 | 19.01 | 22.6 |

Reproduce: `python -m crypto_agents.paired_dispersion var/ablation/20261009T045033Z --delta 0.002`

### 5. Pool de activaciones por celda

4740 activaciones del barrido sobre 7 series; 4737 con el horizonte por delante, repartidas en tramos por `selection.stratify`; 4626 que el gate abre también con la ventana del replay (`selection.confirm_activation`), que son las que una selección puede entregar. «En la corrida» son las entradas del manifiesto.

| símbolo | tramo | candidatas | confirmadas | en la corrida | libres |
| --- | ---: | ---: | ---: | ---: | ---: |
| ADA/USDT | 0 | 154 | 151 | 4 | 147 |
| ADA/USDT | 1 | 135 | 131 | 4 | 127 |
| ADA/USDT | 2 | 133 | 131 | 4 | 127 |
| ADA/USDT | 3 | 137 | 136 | 4 | 132 |
| ADA/USDT | 4 | 139 | 134 | 4 | 130 |
| BNB/USDT | 0 | 172 | 165 | 4 | 161 |
| BNB/USDT | 1 | 136 | 132 | 4 | 128 |
| BNB/USDT | 2 | 135 | 135 | 4 | 131 |
| BNB/USDT | 3 | 107 | 107 | 4 | 103 |
| BNB/USDT | 4 | 141 | 132 | 4 | 128 |
| BTC/USDT | 0 | 148 | 143 | 4 | 139 |
| BTC/USDT | 1 | 147 | 143 | 4 | 139 |
| BTC/USDT | 2 | 126 | 121 | 4 | 117 |
| BTC/USDT | 3 | 128 | 127 | 4 | 123 |
| BTC/USDT | 4 | 123 | 118 | 4 | 114 |
| DOGE/USDT | 0 | 140 | 139 | 4 | 135 |
| DOGE/USDT | 1 | 135 | 134 | 4 | 130 |
| DOGE/USDT | 2 | 131 | 130 | 4 | 126 |
| DOGE/USDT | 3 | 124 | 124 | 4 | 120 |
| DOGE/USDT | 4 | 106 | 103 | 4 | 99 |
| ETH/USDT | 0 | 124 | 123 | 4 | 119 |
| ETH/USDT | 1 | 123 | 118 | 4 | 114 |
| ETH/USDT | 2 | 120 | 120 | 4 | 116 |
| ETH/USDT | 3 | 127 | 119 | 4 | 115 |
| ETH/USDT | 4 | 123 | 118 | 4 | 114 |
| SOL/USDT | 0 | 140 | 140 | 4 | 136 |
| SOL/USDT | 1 | 140 | 134 | 4 | 130 |
| SOL/USDT | 2 | 139 | 139 | 4 | 135 |
| SOL/USDT | 3 | 127 | 125 | 4 | 121 |
| SOL/USDT | 4 | 141 | 135 | 4 | 131 |
| XRP/USDT | 0 | 160 | 157 | 4 | 153 |
| XRP/USDT | 1 | 172 | 168 | 4 | 164 |
| XRP/USDT | 2 | 150 | 147 | 4 | 143 |
| XRP/USDT | 3 | 119 | 118 | 4 | 114 |
| XRP/USDT | 4 | 135 | 129 | 4 | 125 |

Reproduce: `python -m crypto_agents.paired_dispersion var/ablation/20261009T045033Z --delta 0.002`

### 6. Coste por evaluación medido en la corrida

De `consumption.consume`. La cota superior cobra la entrada nueva a `cache_write` donde el modelo lo tiene y todo el prompt como nuevo donde el proveedor no informó `cached_tokens`. Facturación payg, precios de la página al 2026-10-04.

| brazo | evaluaciones | llamadas | sin medir | coste medido USD | cota superior USD | USD por evaluación |
| --- | --- | --- | --- | --- | --- | --- |
| full | 140 | 841 | 108 | ≥ 6.8991 a 7.0268† | 7.1707 | 0.05122 |
| solo | 140 | 140 | 0 | 2.1127 | 2.1127 | 0.01509 |
| always_buy | 140 | 0 | 0 | 0.0000 | 0.0000 | 0.00000 |
| always_sell | 140 | 0 | 0 | 0.0000 | 0.0000 | 0.00000 |
| random_uniform | 140 | 0 | 0 | 0.0000 | 0.0000 | 0.00000 |
| rule_trend | 140 | 0 | 0 | 0.0000 | 0.0000 | 0.00000 |

Reproduce: `python -m crypto_agents.paired_dispersion var/ablation/20261009T045033Z --delta 0.002` (el consumo: `python -m crypto_agents.consumption var/ablation/20261009T045033Z`)

### 7. Pared observada en la corrida

Entre la primera y la última `LLMCall.at` viva de cada brazo, dividida entre sus evaluaciones. `at` es el final de cada llamada: falta la latencia de la primera. Observada en esta corrida y en la máquina que la corrió.

| brazo | evaluaciones | llamadas vivas | primera `LLMCall.at` | última `LLMCall.at` | pared (h) | s por evaluación |
| --- | --- | --- | --- | --- | --- | --- |
| full | 140 | 841 | 2026-10-09T04:50:47+00:00 | 2026-10-09T10:19:01+00:00 | 5.47 | 140.7 |
| solo | 140 | 140 | 2026-10-09T10:19:42+00:00 | 2026-10-09T12:35:16+00:00 | 2.26 | 58.1 |
| always_buy | 140 | 0 | — | — | no determinado: sin llamadas | no determinado: sin llamadas |
| always_sell | 140 | 0 | — | — | no determinado: sin llamadas | no determinado: sin llamadas |
| random_uniform | 140 | 0 | — | — | no determinado: sin llamadas | no determinado: sin llamadas |
| rule_trend | 140 | 0 | — | — | no determinado: sin llamadas | no determinado: sin llamadas |

Reproduce: `python -m crypto_agents.paired_dispersion var/ablation/20261009T045033Z --delta 0.002`

### Lo que estas cifras suponen

- `sigma_D` sale de las mismas 140 evaluaciones cuyo veredicto ya se leyó. Es una muestra: su rango dice cuánto se mueve, y el n se mueve con su cuadrado.
- El rango es el de los percentiles 2.5 y 97.5 de `sigma_D` al remuestrear, no un intervalo de confianza al 95%. Con muchas diferencias exactamente 0 —las evaluaciones en que los dos brazos coinciden— y colas largas, ese rango se queda corto.
- El error estándar `sigma_D/√n` supone evaluaciones independientes. Horizontes que se solapan y símbolos que se mueven juntos reducen el n efectivo: los n de arriba son optimistas en eso.
- El coste y la pared son los de esta corrida por evaluación, escalados en línea recta con n: una estimación, no una cota de otra etapa.
- «Cabe» exige que ninguna celda quede corta. Una selección con una celda corta se rellena desde otros tramos del mismo símbolo y deja de estar repartida en el tiempo.
- Ninguna cifra de este reporte es una media de retorno.

## Conclusión

**Sin escribir, porque no se ha medido.** Escribirla aquí antes de tener la tabla sería exactamente
el vicio que esta fase existe para evitar: justificar una arquitectura con intuición.

Los criterios están fijados de antemano para que la conclusión no se pueda acomodar a lo que salga:

- Si `solo` coincide con `full` por encima del **90%** de las evaluaciones y no lo hace peor en
  resultado, el pipeline de seis modelos no está pagando su coste. Se dice y se actúa en
  consecuencia.
- Si `no_debate` coincide con `full` por encima del **90%**, el debate no está aportando y las dos
  mesas sobran. Es la mitad del coste del sistema.
- Si `bull_only` coincide con `full`, la mesa bajista no está cambiando decisiones y la regla de
  familias distintas entre mesas protege algo que no existe.
- Si `local_technicals` coincide con `full` y su tasa de fallo de validación es comparable, las
  lecturas técnicas se sirven en local y el presupuesto remoto se reserva para el decisor.

- Si un brazo con modelos no supera a `random_uniform` y a `rule_trend` en la puntuación de cierre
  del horizonte —su diferencia pareada con el cero dentro del intervalo—, no hay evidencia de que
  su dirección aporte nada. Que además gane contra `always_buy` solo dice que el mercado subía.

Una arquitectura de seis modelos que no supera a uno es cara y bonita, no buena.

### Tabla de la primera corrida (descartada: registro, no evidencia)

| brazo | evals | decididas | acciones | órdenes | coincidencia con `full` | cuota | latencia media | fallo validación |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `full` | 140 | 109 | buy 40, hold 7, sell 62 | 102 | 100% | 160.0 | 56314 ms | 58% |
| `no_debate` | 140 | 138 | buy 36, hold 71, sell 31 | 67 | 47% | 7.0 | 74728 ms | 0% |
| `bull_only` | 140 | 107 | buy 51, hold 25, sell 31 | 82 | 54% | 107.0 | 72817 ms | 60% |
| `solo` | 140 | 140 | buy 51, hold 37, sell 52 | 103 | 60% | 4.0 | 67718 ms | 0% |
| `local_technicals` | 140 | 52 | buy 16, hold 15, sell 21 | 37 | 39% | 296.0 | 41244 ms | 66% |
| `local_bull` | 140 | 80 | buy 24, hold 7, sell 49 | 73 | 56% | 327.0 | 44186 ms | 51% |

| brazo | órdenes resueltas | invalidadas | aciertos | retorno medio |
| --- | --- | --- | --- | --- |
| `full` | 102 | 42 | 34% | -0% |
| `no_debate` | 67 | 27 | 34% | -1% |
| `bull_only` | 82 | 29 | 39% | -0% |
| `solo` | 103 | 43 | 38% | -0% |
| `local_technicals` | 37 | 19 | 32% | -1% |
| `local_bull` | 73 | 29 | 38% | -0% |

| brazo | qué pregunta responde |
| --- | --- |
| `full` | referencia: tres técnicos, dos mesas y decisor |
| `no_debate` | ¿aporta algo el debate sobre los veredictos técnicos? |
| `bull_only` | ¿aporta la contraparte bajista, o basta una mesa? |
| `solo` | ¿un modelo y una llamada igualan a seis modelos y seis llamadas? |
| `local_technicals` | ¿se pueden servir las tres lecturas técnicas en local sin perder decisión? |
| `local_bull` | ¿aguanta una mesa en local contra una remota? |
