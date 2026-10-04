# Ablación: ¿los seis modelos deciden mejor que uno?

**Estado: el arnés está construido y probado; hubo una primera corrida con modelos reales, descartada (ver la enmienda posterior), y la segunda está pendiente.**
Este documento contiene el método, lo que ya se puede afirmar y una conclusión sin escribir. La
tabla de resultados la genera el comando y sustituye a la sección marcada más abajo.

## Cómo se corre

```bash
uv run python -m crypto_agents.ablation --dry-run --manifest data/ablation_selection.json
uv run python -m crypto_agents.ablation --fill --manifest data/ablation_selection.json
uv run python -m crypto_agents.ablation --manifest data/ablation_selection.json
```

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
   distinto del declarado para el brazo; hay un veto distinto de `invalid_stop_side`. Las
   evaluaciones perdidas por fallos de validación del propio modelo no invalidan: son parte de lo
   que se mide.
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
   alcanzable lo limita el consumo del pool de la suscripción, no el tiempo.

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
