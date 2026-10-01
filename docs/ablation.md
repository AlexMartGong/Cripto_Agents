# Ablación: ¿los seis modelos deciden mejor que uno?

**Estado: el arnés está construido y probado; la corrida con modelos reales no se ha hecho.**
Este documento contiene el método, lo que ya se puede afirmar y una conclusión sin escribir. La
tabla de resultados la genera el comando y sustituye a la sección marcada más abajo.

## Cómo se corre

```bash
uv run python -m crypto_agents.ablation --dry-run --manifest data/ablation_selection.json
uv run python -m crypto_agents.ablation --fill --manifest data/ablation_selection.json
uv run python -m crypto_agents.ablation --manifest data/ablation_selection.json
```

Sin `--fill` el replay es solo-caché: se niega a llamar a ningún proveedor y no modifica la caché,
así que reejecutar la tabla no puede costar dinero. Lo que no garantiza es que la reejecución
termine. Reproduce todo intento que produjo contenido, también los inválidos con su reintento, pero
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

## Los seis brazos

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

Sin comisiones ni slippage. Con pocas órdenes por brazo, el error de muestreo domina cualquier
diferencia de retorno, por eso la tabla publica el denominador junto a la tasa.

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

Pendiente de la corrida. El camino ya está despejado: los seis modelos responden con el prompt y el
esquema reales, y los seis brazos caben en la cuota del decisor (840 de 880 por ventana).

El re-sondeo del 15 de agosto de 2026 dejó dos cosas escritas antes de correr:

- **Cinco de seis modelos no fallan de contenido.** 12 de 12 válidos en `mimo-v2.5`, `hy3`,
  `minimax-m3` y `deepseek-v4-flash`, 4 de 4 en `kimi-k2.6`. El único que reintenta es el decisor:
  2 de 13 intentos devolvieron `invalidation_price` como cadena, y el reintento con el error
  adjunto validó las dos veces.
- **`bull` cambió de modelo por disponibilidad, no por calidad.** `qwen3.7-plus` respondía 0/10 con
  503 y la familia qwen entera con él. `kimi-k2.6` la sustituye y mantiene seis familias distintas
  entre los seis primarios.

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
