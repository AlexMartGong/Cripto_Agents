# `usage` real de la pasarela

`gateway_usage.json` — el `token_usage` que devolvió la pasarela OpenAI-compatible a 12 llamadas,
hechas el `2026-10-03` hacia las `04:55Z` desde el escritorio: dos llamadas idénticas, una detrás de
otra, a cada uno de los seis modelos remotos, con el modo de salida estructurada que declara su rol.

El prompt era un esquema `Ping` más ~1.4 k tokens de relleno estático, para que el prefijo fuera
cacheable. Las llamadas salieron por `OpenAIBackend` (la misma pila y la misma cabecera de sesión
que en producción) pero **no por el router**: no dejaron `LLMCall` ni cuentan en ningún journal, y
sí contaron contra la ventana del proveedor.

Solo se guardan los contadores: ni claves, ni URL de la pasarela, ni cabeceras, ni el texto de la
respuesta. Es lo que hay en `response_metadata["token_usage"]` del mensaje de LangChain, es decir el
`usage` del proveedor antes de que LangChain lo procese.

## Qué dijo la medición

- **`prompt_tokens` incluye los cacheados.** La segunda llamada repite el mismo `prompt_tokens`
  (1 602 en MiMo-V2.5) con 1 536 en `cached_tokens`: el total no baja cuando hay caché.
- **`completion_tokens` incluye el razonamiento.** `total_tokens = prompt + completion` y
  `reasoning_tokens` queda dentro de `completion_tokens` (Kimi K2.6: 519 de salida para un `ok=true`,
  512 de razonamiento).
- **En frío, cuatro modelos informan `cached_tokens: 0` y DeepSeek V4 Flash informa `null`**, con la
  clave presente. En caliente trae la cifra (1 536). `null` es «no informado», no cero.
- LangChain convierte un contador ausente en `0` al construir `usage_metadata`
  (`prompt_tokens or 0`): por eso los adaptadores leen el `usage` crudo y no `usage_metadata`.

## Digest

```
447132dde424f2b9b76aea2321b665bdc1866a17225f22e978b4552837e85dea  gateway_usage.json
```

`tests/test_llm.py` lo comprueba.
