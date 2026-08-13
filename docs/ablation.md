# Ablación: ¿los seis modelos deciden mejor que uno?

**Estado: el arnés está construido y probado; la corrida con modelos reales no se ha hecho.**
Este documento contiene el método, lo que ya se puede afirmar y una conclusión sin escribir. La
tabla de resultados la genera el comando y sustituye a la sección marcada más abajo.

## Cómo se corre

```bash
uv run python -m crypto_agents.ablation --fill        # primera pasada: llena la caché, paga
uv run python -m crypto_agents.ablation               # a partir de ahí: gratis y reproducible
```

Sin `--fill` el replay es solo-caché y se niega a llamar a ningún proveedor, así que reejecutar la
tabla no puede costar dinero por descuido. Opciones útiles: `--arms full,solo` para un subconjunto,
`--evaluations` para acotar la ventana, `--horizon` para el plazo con el que se puntúan las órdenes.

Sobre el histórico versionado (500 velas de 4h) y con el preset de producción, las primeras 400
velas se van en warm-up: quedan **99 evaluaciones posibles**. El valor por defecto son 25.

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

Pendiente. Bloqueado por credenciales del gateway: `CA_OPENAI__API_KEY` y `CA_OPENAI__BASE_URL`
siguen siendo marcadores en `.env.example`, y los seis identificadores de modelo remoto no están
verificados contra la lista del gateway. Los brazos `full`, `local_technicals` y `local_bull`
comparan remoto contra local y no se pueden correr sin eso. Los brazos `no_debate`, `bull_only` y
`solo` sí podrían correrse solo con modelos locales, y responderían la pregunta principal —¿la
arquitectura aporta?— aunque no la de local contra remoto.

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

Una arquitectura de seis modelos que no supera a uno es cara y bonita, no buena.
