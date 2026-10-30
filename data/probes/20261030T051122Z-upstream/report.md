# Sondeo del upstream del decisor — 2026-10-30T05:11:22Z

- commit: `ffc94b045d1fc1436be892c9e6ef7e545e9d93fa` (sucio: True); script sha-256 `efb9adbce1a994bb2f64b84a0a67a639644db6304004427166949032836c36be`; máquina `zephyrus-G15`
- base_url: `https://opencode.ai/zen/v1`; un intento por llamada, sin caché; tope 0.20 USD, gastado 0.1 USD en el extremo alto (0 sin tokens, 0 negadas)
- envoltura de la variante a: `f0727ca0e3c51a242b7351e215a0c6596d1f9e7b777f2efda9d402f7ba3a6928` (la registrada)
- prompt: el de `bef5177f-eb34-55cd-b0a6-1261366bd515` (ADA/USDT, evaluado 2024-10-23T12:00:00Z), `prompt_digest` `f1ec963b791e5a33ef99e4dabe8c03e719400eca1d71db668c4fa3cd4401fcea`

## La llamada de la etapa 1 con el mismo prompt

| hora UTC | latency_ms | completion_tokens | tok/s | upstream_model | upstream_endpoint |
| --- | --- | --- | --- | --- | --- |
| 2026-10-09T10:19:42.898913Z | 41449.8 | `2328` | 56.2 | `accounts/fireworks/models/glm-5p3` | `fireworks` |

## Resumen

| # | variante | modelo | hora UTC | estado | tok/s | cabeceras `x-opencode-*` | `model` del cuerpo | `id` del cuerpo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | a | `glm-5.2` | 2026-10-30T05:11:42.295856Z | `200` | 103.1 | `x-opencode-log-id` | `glm-5.2` | `chatcmpl-c21bdeda` |
| 2 | a | `glm-5.2` | 2026-10-30T05:12:12.372680Z | `200` | 98.7 | `x-opencode-log-id` | `glm-5.2` | `chatcmpl-af5f2a37` |
| 3 | a | `glm-5.2` | 2026-10-30T05:12:45.910931Z | `200` | 105.2 | `x-opencode-log-id` | `glm-5.2` | `chatcmpl-2584e135` |
| 4 | b | `glm-5.2` | 2026-10-30T05:13:23.538061Z | `200` | 96.6 | `x-opencode-log-id` | `glm-5.2` | `chatcmpl-10f496fa` |
| 5 | b | `glm-5.2` | 2026-10-30T05:14:05.553680Z | `200` | 98.4 | `x-opencode-log-id` | `glm-5.2` | `chatcmpl-ed5358dc` |
| 6 | b | `glm-5.2` | 2026-10-30T05:14:39.460616Z | `200` | 105.6 | `x-opencode-log-id` | `glm-5.2` | `chatcmpl-a1d4cd43` |
| 7 | c | `minimax-m3` | 2026-10-30T05:14:48.422226Z | `200` | 180.9 | `x-opencode-log-id` | `minimax-m3` | `chatcmpl-c2666e39923249e0a605beb83df46391` |

Cabeceras `opencode` vistas en alguna llamada: `x-opencode-log-id`.

## Una tabla por llamada

### Llamada 1 — variante a, `glm-5.2`

| campo | valor |
| --- | --- |
| variante | `a` |
| modelo pedido | `glm-5.2` |
| hora UTC (fin de la llamada) | 2026-10-30T05:11:42.295856Z |
| estado HTTP | `200` |
| cuerpo: `model` | `glm-5.2` |
| cuerpo: `id` | `chatcmpl-c21bdeda` |
| cuerpo: `system_fingerprint` | — |
| cuerpo: orden de las claves | `id`, `object`, `created`, `model`, `choices`, `usage` |
| argumentos: orden de las claves | `action`, `confidence`, `invalidation_price`, `rationale`, `size_fraction` |
| completion_tokens | `1760` |
| latency_ms | 17074.3 |
| tok/s | 103.1 |
| router: `upstream_model` | — |
| router: `upstream_endpoint` | — |
| válida | `True` |
| fallo | — |
| petición: sha-256 de la envoltura | `f0727ca0e3c51a242b7351e215a0c6596d1f9e7b777f2efda9d402f7ba3a6928` |
| petición: User-Agent | `AsyncOpenAI/Python 2.54.0` |
| petición: `x-opencode-session` | `crypto-agents-f6406173` |
| petición: el prompt es el de la etapa 1 | `True` |
| cabecera `date` | `Fri, 30 Oct 2026 05:11:42 GMT` |
| cabecera `content-type` | `application/json` |
| cabecera `transfer-encoding` | `chunked` |
| cabecera `connection` | `keep-alive` |
| cabecera `cf-placement` | `remote-IAD` |
| cabecera `x-opencode-log-id` | `01a18b38-4e12-7ace-ace7-43c654e02301` |
| cabecera `x-zen-model` | `glm-5.2` |
| cabecera `content-encoding` | `gzip` |
| cabecera `server` | `cloudflare` |
| cabecera `cf-ray` | `a527eb4ffbc74fba-IAH` |

### Llamada 2 — variante a, `glm-5.2`

| campo | valor |
| --- | --- |
| variante | `a` |
| modelo pedido | `glm-5.2` |
| hora UTC (fin de la llamada) | 2026-10-30T05:12:12.372680Z |
| estado HTTP | `200` |
| cuerpo: `model` | `glm-5.2` |
| cuerpo: `id` | `chatcmpl-af5f2a37` |
| cuerpo: `system_fingerprint` | — |
| cuerpo: orden de las claves | `id`, `object`, `created`, `model`, `choices`, `usage` |
| argumentos: orden de las claves | `action`, `confidence`, `invalidation_price`, `rationale`, `size_fraction` |
| completion_tokens | `2969` |
| latency_ms | 30076.5 |
| tok/s | 98.7 |
| router: `upstream_model` | — |
| router: `upstream_endpoint` | — |
| válida | `True` |
| fallo | — |
| petición: sha-256 de la envoltura | `f0727ca0e3c51a242b7351e215a0c6596d1f9e7b777f2efda9d402f7ba3a6928` |
| petición: User-Agent | `AsyncOpenAI/Python 2.54.0` |
| petición: `x-opencode-session` | `crypto-agents-f6406173` |
| petición: el prompt es el de la etapa 1 | `True` |
| cabecera `date` | `Fri, 30 Oct 2026 05:12:12 GMT` |
| cabecera `content-type` | `application/json` |
| cabecera `transfer-encoding` | `chunked` |
| cabecera `connection` | `keep-alive` |
| cabecera `cf-placement` | `remote-IAD` |
| cabecera `x-opencode-log-id` | `01a18b38-9017-732a-a8ee-004c9d41e584` |
| cabecera `x-zen-model` | `glm-5.2` |
| cabecera `content-encoding` | `gzip` |
| cabecera `server` | `cloudflare` |
| cabecera `cf-ray` | `a527ebb998b24fba-IAH` |

### Llamada 3 — variante a, `glm-5.2`

| campo | valor |
| --- | --- |
| variante | `a` |
| modelo pedido | `glm-5.2` |
| hora UTC (fin de la llamada) | 2026-10-30T05:12:45.910931Z |
| estado HTTP | `200` |
| cuerpo: `model` | `glm-5.2` |
| cuerpo: `id` | `chatcmpl-2584e135` |
| cuerpo: `system_fingerprint` | — |
| cuerpo: orden de las claves | `id`, `object`, `created`, `model`, `choices`, `usage` |
| argumentos: orden de las claves | `action`, `confidence`, `invalidation_price`, `rationale`, `size_fraction` |
| completion_tokens | `3527` |
| latency_ms | 33537.9 |
| tok/s | 105.2 |
| router: `upstream_model` | — |
| router: `upstream_endpoint` | — |
| válida | `True` |
| fallo | — |
| petición: sha-256 de la envoltura | `f0727ca0e3c51a242b7351e215a0c6596d1f9e7b777f2efda9d402f7ba3a6928` |
| petición: User-Agent | `AsyncOpenAI/Python 2.54.0` |
| petición: `x-opencode-session` | `crypto-agents-f6406173` |
| petición: el prompt es el de la etapa 1 | `True` |
| cabecera `date` | `Fri, 30 Oct 2026 05:12:45 GMT` |
| cabecera `content-type` | `application/json` |
| cabecera `transfer-encoding` | `chunked` |
| cabecera `connection` | `keep-alive` |
| cabecera `cf-placement` | `remote-IAD` |
| cabecera `x-opencode-log-id` | `01a18b39-0594-73fa-9519-5187621cfcdd` |
| cabecera `x-zen-model` | `glm-5.2` |
| cabecera `content-encoding` | `gzip` |
| cabecera `server` | `cloudflare` |
| cabecera `cf-ray` | `a527ec7599894fba-IAH` |

### Llamada 4 — variante b, `glm-5.2`

| campo | valor |
| --- | --- |
| variante | `b` |
| modelo pedido | `glm-5.2` |
| hora UTC (fin de la llamada) | 2026-10-30T05:13:23.538061Z |
| estado HTTP | `200` |
| cuerpo: `model` | `glm-5.2` |
| cuerpo: `id` | `chatcmpl-10f496fa` |
| cuerpo: `system_fingerprint` | — |
| cuerpo: orden de las claves | `id`, `object`, `created`, `model`, `choices`, `usage` |
| argumentos: orden de las claves | `action`, `confidence`, `invalidation_price`, `rationale`, `size_fraction` |
| completion_tokens | `3636` |
| latency_ms | 37626.8 |
| tok/s | 96.6 |
| router: `upstream_model` | — |
| router: `upstream_endpoint` | — |
| válida | `True` |
| fallo | — |
| petición: sha-256 de la envoltura | `f0727ca0e3c51a242b7351e215a0c6596d1f9e7b777f2efda9d402f7ba3a6928` |
| petición: User-Agent | `crypto-agents-sonda/1` |
| petición: `x-opencode-session` | `crypto-agents-sonda-upstream` |
| petición: el prompt es el de la etapa 1 | `True` |
| cabecera `date` | `Fri, 30 Oct 2026 05:13:23 GMT` |
| cabecera `content-type` | `application/json` |
| cabecera `transfer-encoding` | `chunked` |
| cabecera `connection` | `keep-alive` |
| cabecera `cf-placement` | `remote-IAD` |
| cabecera `x-opencode-log-id` | `01a18b39-8937-79ed-97b4-9aa07c616201` |
| cabecera `x-zen-model` | `glm-5.2` |
| cabecera `content-encoding` | `gzip` |
| cabecera `server` | `cloudflare` |
| cabecera `cf-ray` | `a527ed483ce520b0-IAH` |

### Llamada 5 — variante b, `glm-5.2`

| campo | valor |
| --- | --- |
| variante | `b` |
| modelo pedido | `glm-5.2` |
| hora UTC (fin de la llamada) | 2026-10-30T05:14:05.553680Z |
| estado HTTP | `200` |
| cuerpo: `model` | `glm-5.2` |
| cuerpo: `id` | `chatcmpl-ed5358dc` |
| cuerpo: `system_fingerprint` | — |
| cuerpo: orden de las claves | `id`, `object`, `created`, `model`, `choices`, `usage` |
| argumentos: orden de las claves | `action`, `confidence`, `invalidation_price`, `rationale`, `size_fraction` |
| completion_tokens | `4134` |
| latency_ms | 42015.3 |
| tok/s | 98.4 |
| router: `upstream_model` | — |
| router: `upstream_endpoint` | — |
| válida | `True` |
| fallo | — |
| petición: sha-256 de la envoltura | `f0727ca0e3c51a242b7351e215a0c6596d1f9e7b777f2efda9d402f7ba3a6928` |
| petición: User-Agent | `crypto-agents-sonda/1` |
| petición: `x-opencode-session` | `crypto-agents-sonda-upstream` |
| petición: el prompt es el de la etapa 1 | `True` |
| cabecera `date` | `Fri, 30 Oct 2026 05:14:05 GMT` |
| cabecera `content-type` | `application/json` |
| cabecera `transfer-encoding` | `chunked` |
| cabecera `connection` | `keep-alive` |
| cabecera `cf-placement` | `remote-IAD` |
| cabecera `x-opencode-log-id` | `01a18b3a-1b95-7b26-8e86-72f072dc284d` |
| cabecera `x-zen-model` | `glm-5.2` |
| cabecera `content-encoding` | `gzip` |
| cabecera `server` | `cloudflare` |
| cabecera `cf-ray` | `a527ee32683220b0-IAH` |

### Llamada 6 — variante b, `glm-5.2`

| campo | valor |
| --- | --- |
| variante | `b` |
| modelo pedido | `glm-5.2` |
| hora UTC (fin de la llamada) | 2026-10-30T05:14:39.460616Z |
| estado HTTP | `200` |
| cuerpo: `model` | `glm-5.2` |
| cuerpo: `id` | `chatcmpl-a1d4cd43` |
| cuerpo: `system_fingerprint` | — |
| cuerpo: orden de las claves | `id`, `object`, `created`, `model`, `choices`, `usage` |
| argumentos: orden de las claves | `action`, `confidence`, `invalidation_price`, `rationale`, `size_fraction` |
| completion_tokens | `3582` |
| latency_ms | 33906.6 |
| tok/s | 105.6 |
| router: `upstream_model` | — |
| router: `upstream_endpoint` | — |
| válida | `True` |
| fallo | — |
| petición: sha-256 de la envoltura | `f0727ca0e3c51a242b7351e215a0c6596d1f9e7b777f2efda9d402f7ba3a6928` |
| petición: User-Agent | `crypto-agents-sonda/1` |
| petición: `x-opencode-session` | `crypto-agents-sonda-upstream` |
| petición: el prompt es el de la etapa 1 | `True` |
| cabecera `date` | `Fri, 30 Oct 2026 05:14:39 GMT` |
| cabecera `content-type` | `application/json` |
| cabecera `transfer-encoding` | `chunked` |
| cabecera `connection` | `keep-alive` |
| cabecera `cf-placement` | `remote-IAD` |
| cabecera `x-opencode-log-id` | `01a18b3a-bfb6-70cd-914f-b59662767136` |
| cabecera `x-zen-model` | `glm-5.2` |
| cabecera `content-encoding` | `gzip` |
| cabecera `server` | `cloudflare` |
| cabecera `cf-ray` | `a527ef38f94320b0-IAH` |

### Llamada 7 — variante c, `minimax-m3`

| campo | valor |
| --- | --- |
| variante | `c` |
| modelo pedido | `minimax-m3` |
| hora UTC (fin de la llamada) | 2026-10-30T05:14:48.422226Z |
| estado HTTP | `200` |
| cuerpo: `model` | `minimax-m3` |
| cuerpo: `id` | `chatcmpl-c2666e39923249e0a605beb83df46391` |
| cuerpo: `system_fingerprint` | — |
| cuerpo: orden de las claves | `id`, `object`, `created`, `model`, `choices`, `usage` |
| argumentos: orden de las claves | `action`, `confidence`, `size_fraction`, `invalidation_price`, `rationale` |
| completion_tokens | `1621` |
| latency_ms | 8961.3 |
| tok/s | 180.9 |
| router: `upstream_model` | — |
| router: `upstream_endpoint` | — |
| válida | `True` |
| fallo | — |
| petición: sha-256 de la envoltura | `633a0112f7292ecaa04f5b386e2fb31037928eea684d18cdb2e5e900027af620` |
| petición: User-Agent | `AsyncOpenAI/Python 2.54.0` |
| petición: `x-opencode-session` | `crypto-agents-f24a565e` |
| petición: el prompt es el de la etapa 1 | `True` |
| cabecera `date` | `Fri, 30 Oct 2026 05:14:48 GMT` |
| cabecera `content-type` | `application/json` |
| cabecera `transfer-encoding` | `chunked` |
| cabecera `connection` | `keep-alive` |
| cabecera `cf-placement` | `remote-IAD` |
| cabecera `x-opencode-log-id` | `01a18b3b-44c1-7dd4-9ee5-c2fe4b7a0be5` |
| cabecera `x-zen-model` | `minimax-m3` |
| cabecera `content-encoding` | `gzip` |
| cabecera `server` | `cloudflare` |
| cabecera `cf-ray` | `a527f00dea85fca4-ATL` |

