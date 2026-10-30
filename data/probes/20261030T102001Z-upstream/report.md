# Sondeo del upstream del decisor — 2026-10-30T10:20:01Z

- commit: `63a2b94fccd04c79db5532caa6521124ce50d2b5` (sucio: False); script sha-256 `efb9adbce1a994bb2f64b84a0a67a639644db6304004427166949032836c36be`; máquina `zephyrus-G15`
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
| 1 | a | `glm-5.2` | 2026-10-30T10:20:27.085051Z | `200` | 108.9 | `x-opencode-log-id` | `glm-5.2` | `chatcmpl-c676a998` |
| 2 | a | `glm-5.2` | 2026-10-30T10:20:46.848211Z | `200` | 107.2 | `x-opencode-log-id` | `glm-5.2` | `chatcmpl-27c466dc` |
| 3 | a | `glm-5.2` | 2026-10-30T10:21:31.622437Z | `200` | 104.5 | `x-opencode-log-id` | `glm-5.2` | `chatcmpl-3a66b87f` |
| 4 | b | `glm-5.2` | 2026-10-30T10:21:55.853748Z | `200` | 97.1 | `x-opencode-log-id` | `glm-5.2` | `chatcmpl-5d428ee6` |
| 5 | b | `glm-5.2` | 2026-10-30T10:22:42.821630Z | `200` | 106.3 | `x-opencode-log-id` | `glm-5.2` | `chatcmpl-4c213b13` |
| 6 | b | `glm-5.2` | 2026-10-30T10:23:04.162432Z | `200` | 104.4 | `x-opencode-log-id` | `glm-5.2` | `chatcmpl-71c34798` |
| 7 | c | `minimax-m3` | 2026-10-30T10:23:21.737303Z | `200` | 188.7 | `x-opencode-log-id` | `minimax-m3` | `chatcmpl-75734ef013ff4d73a52b9f6a41c80125` |

Cabeceras `opencode` vistas en alguna llamada: `x-opencode-log-id`.

## Una tabla por llamada

### Llamada 1 — variante a, `glm-5.2`

| campo | valor |
| --- | --- |
| variante | `a` |
| modelo pedido | `glm-5.2` |
| hora UTC (fin de la llamada) | 2026-10-30T10:20:27.085051Z |
| estado HTTP | `200` |
| cuerpo: `model` | `glm-5.2` |
| cuerpo: `id` | `chatcmpl-c676a998` |
| cuerpo: `system_fingerprint` | — |
| cuerpo: orden de las claves | `id`, `object`, `created`, `model`, `choices`, `usage` |
| argumentos: orden de las claves | `action`, `confidence`, `invalidation_price`, `rationale`, `size_fraction` |
| completion_tokens | `2510` |
| latency_ms | 23049.8 |
| tok/s | 108.9 |
| router: `upstream_model` | — |
| router: `upstream_endpoint` | — |
| válida | `True` |
| fallo | — |
| petición: sha-256 de la envoltura | `f0727ca0e3c51a242b7351e215a0c6596d1f9e7b777f2efda9d402f7ba3a6928` |
| petición: User-Agent | `AsyncOpenAI/Python 2.54.0` |
| petición: `x-opencode-session` | `crypto-agents-477a373a` |
| petición: el prompt es el de la etapa 1 | `True` |
| cabecera `date` | `Fri, 30 Oct 2026 10:20:27 GMT` |
| cabecera `content-type` | `application/json` |
| cabecera `transfer-encoding` | `chunked` |
| cabecera `connection` | `keep-alive` |
| cabecera `cf-placement` | `remote-IAD` |
| cabecera `x-opencode-log-id` | `01a18c52-e106-7826-ae7e-43c5c8e0fef5` |
| cabecera `x-zen-model` | `glm-5.2` |
| cabecera `content-encoding` | `gzip` |
| cabecera `server` | `cloudflare` |
| cabecera `cf-ray` | `a529af6e4f4a4aa7-DFW` |

### Llamada 2 — variante a, `glm-5.2`

| campo | valor |
| --- | --- |
| variante | `a` |
| modelo pedido | `glm-5.2` |
| hora UTC (fin de la llamada) | 2026-10-30T10:20:46.848211Z |
| estado HTTP | `200` |
| cuerpo: `model` | `glm-5.2` |
| cuerpo: `id` | `chatcmpl-27c466dc` |
| cuerpo: `system_fingerprint` | — |
| cuerpo: orden de las claves | `id`, `object`, `created`, `model`, `choices`, `usage` |
| argumentos: orden de las claves | `action`, `confidence`, `invalidation_price`, `rationale`, `size_fraction` |
| completion_tokens | `2118` |
| latency_ms | 19762.8 |
| tok/s | 107.2 |
| router: `upstream_model` | — |
| router: `upstream_endpoint` | — |
| válida | `True` |
| fallo | — |
| petición: sha-256 de la envoltura | `f0727ca0e3c51a242b7351e215a0c6596d1f9e7b777f2efda9d402f7ba3a6928` |
| petición: User-Agent | `AsyncOpenAI/Python 2.54.0` |
| petición: `x-opencode-session` | `crypto-agents-477a373a` |
| petición: el prompt es el de la etapa 1 | `True` |
| cabecera `date` | `Fri, 30 Oct 2026 10:20:46 GMT` |
| cabecera `content-type` | `application/json` |
| cabecera `transfer-encoding` | `chunked` |
| cabecera `connection` | `keep-alive` |
| cabecera `cf-placement` | `remote-IAD` |
| cabecera `x-opencode-log-id` | `01a18c53-3a84-7f17-a06a-c5b1544154ec` |
| cabecera `x-zen-model` | `glm-5.2` |
| cabecera `content-encoding` | `gzip` |
| cabecera `server` | `cloudflare` |
| cabecera `cf-ray` | `a529affd7fc94aa7-DFW` |

### Llamada 3 — variante a, `glm-5.2`

| campo | valor |
| --- | --- |
| variante | `a` |
| modelo pedido | `glm-5.2` |
| hora UTC (fin de la llamada) | 2026-10-30T10:21:31.622437Z |
| estado HTTP | `200` |
| cuerpo: `model` | `glm-5.2` |
| cuerpo: `id` | `chatcmpl-3a66b87f` |
| cuerpo: `system_fingerprint` | — |
| cuerpo: orden de las claves | `id`, `object`, `created`, `model`, `choices`, `usage` |
| argumentos: orden de las claves | `action`, `confidence`, `invalidation_price`, `rationale`, `size_fraction` |
| completion_tokens | `4679` |
| latency_ms | 44773.9 |
| tok/s | 104.5 |
| router: `upstream_model` | — |
| router: `upstream_endpoint` | — |
| válida | `True` |
| fallo | — |
| petición: sha-256 de la envoltura | `f0727ca0e3c51a242b7351e215a0c6596d1f9e7b777f2efda9d402f7ba3a6928` |
| petición: User-Agent | `AsyncOpenAI/Python 2.54.0` |
| petición: `x-opencode-session` | `crypto-agents-477a373a` |
| petición: el prompt es el de la etapa 1 | `True` |
| cabecera `date` | `Fri, 30 Oct 2026 10:21:31 GMT` |
| cabecera `content-type` | `application/json` |
| cabecera `transfer-encoding` | `chunked` |
| cabecera `connection` | `keep-alive` |
| cabecera `cf-placement` | `remote-IAD` |
| cabecera `x-opencode-log-id` | `01a18c53-87ad-77ec-abc4-a073df22f599` |
| cabecera `x-zen-model` | `glm-5.2` |
| cabecera `content-encoding` | `gzip` |
| cabecera `server` | `cloudflare` |
| cabecera `cf-ray` | `a529b078fc1a4aa7-DFW` |

### Llamada 4 — variante b, `glm-5.2`

| campo | valor |
| --- | --- |
| variante | `b` |
| modelo pedido | `glm-5.2` |
| hora UTC (fin de la llamada) | 2026-10-30T10:21:55.853748Z |
| estado HTTP | `200` |
| cuerpo: `model` | `glm-5.2` |
| cuerpo: `id` | `chatcmpl-5d428ee6` |
| cuerpo: `system_fingerprint` | — |
| cuerpo: orden de las claves | `id`, `object`, `created`, `model`, `choices`, `usage` |
| argumentos: orden de las claves | `action`, `confidence`, `invalidation_price`, `rationale`, `size_fraction` |
| completion_tokens | `2352` |
| latency_ms | 24231.0 |
| tok/s | 97.1 |
| router: `upstream_model` | — |
| router: `upstream_endpoint` | — |
| válida | `True` |
| fallo | — |
| petición: sha-256 de la envoltura | `f0727ca0e3c51a242b7351e215a0c6596d1f9e7b777f2efda9d402f7ba3a6928` |
| petición: User-Agent | `crypto-agents-sonda/1` |
| petición: `x-opencode-session` | `crypto-agents-sonda-upstream` |
| petición: el prompt es el de la etapa 1 | `True` |
| cabecera `date` | `Fri, 30 Oct 2026 10:21:55 GMT` |
| cabecera `content-type` | `application/json` |
| cabecera `transfer-encoding` | `chunked` |
| cabecera `connection` | `keep-alive` |
| cabecera `cf-placement` | `remote-IAD` |
| cabecera `x-opencode-log-id` | `01a18c54-3746-7a38-aec1-1d712b1e63ad` |
| cabecera `x-zen-model` | `glm-5.2` |
| cabecera `content-encoding` | `gzip` |
| cabecera `server` | `cloudflare` |
| cabecera `cf-ray` | `a529b191e9128f0d-IAH` |

### Llamada 5 — variante b, `glm-5.2`

| campo | valor |
| --- | --- |
| variante | `b` |
| modelo pedido | `glm-5.2` |
| hora UTC (fin de la llamada) | 2026-10-30T10:22:42.821630Z |
| estado HTTP | `200` |
| cuerpo: `model` | `glm-5.2` |
| cuerpo: `id` | `chatcmpl-4c213b13` |
| cuerpo: `system_fingerprint` | — |
| cuerpo: orden de las claves | `id`, `object`, `created`, `model`, `choices`, `usage` |
| argumentos: orden de las claves | `action`, `confidence`, `invalidation_price`, `rationale`, `size_fraction` |
| completion_tokens | `4994` |
| latency_ms | 46967.5 |
| tok/s | 106.3 |
| router: `upstream_model` | — |
| router: `upstream_endpoint` | — |
| válida | `True` |
| fallo | — |
| petición: sha-256 de la envoltura | `f0727ca0e3c51a242b7351e215a0c6596d1f9e7b777f2efda9d402f7ba3a6928` |
| petición: User-Agent | `crypto-agents-sonda/1` |
| petición: `x-opencode-session` | `crypto-agents-sonda-upstream` |
| petición: el prompt es el de la etapa 1 | `True` |
| cabecera `date` | `Fri, 30 Oct 2026 10:22:42 GMT` |
| cabecera `content-type` | `application/json` |
| cabecera `transfer-encoding` | `chunked` |
| cabecera `connection` | `keep-alive` |
| cabecera `cf-placement` | `remote-IAD` |
| cabecera `x-opencode-log-id` | `01a18c54-9557-7bbf-85eb-7a5bfe724d92` |
| cabecera `x-zen-model` | `glm-5.2` |
| cabecera `content-encoding` | `gzip` |
| cabecera `server` | `cloudflare` |
| cabecera `cf-ray` | `a529b2286fc58f0d-IAH` |

### Llamada 6 — variante b, `glm-5.2`

| campo | valor |
| --- | --- |
| variante | `b` |
| modelo pedido | `glm-5.2` |
| hora UTC (fin de la llamada) | 2026-10-30T10:23:04.162432Z |
| estado HTTP | `200` |
| cuerpo: `model` | `glm-5.2` |
| cuerpo: `id` | `chatcmpl-71c34798` |
| cuerpo: `system_fingerprint` | — |
| cuerpo: orden de las claves | `id`, `object`, `created`, `model`, `choices`, `usage` |
| argumentos: orden de las claves | `action`, `confidence`, `invalidation_price`, `rationale`, `size_fraction` |
| completion_tokens | `2227` |
| latency_ms | 21340.5 |
| tok/s | 104.4 |
| router: `upstream_model` | — |
| router: `upstream_endpoint` | — |
| válida | `True` |
| fallo | — |
| petición: sha-256 de la envoltura | `f0727ca0e3c51a242b7351e215a0c6596d1f9e7b777f2efda9d402f7ba3a6928` |
| petición: User-Agent | `crypto-agents-sonda/1` |
| petición: `x-opencode-session` | `crypto-agents-sonda-upstream` |
| petición: el prompt es el de la etapa 1 | `True` |
| cabecera `date` | `Fri, 30 Oct 2026 10:23:04 GMT` |
| cabecera `content-type` | `application/json` |
| cabecera `transfer-encoding` | `chunked` |
| cabecera `connection` | `keep-alive` |
| cabecera `cf-placement` | `remote-IAD` |
| cabecera `x-opencode-log-id` | `01a18c55-4cca-7634-9a78-8a583cfc63c1` |
| cabecera `x-zen-model` | `glm-5.2` |
| cabecera `content-encoding` | `gzip` |
| cabecera `server` | `cloudflare` |
| cabecera `cf-ray` | `a529b34deef38f0d-IAH` |

### Llamada 7 — variante c, `minimax-m3`

| campo | valor |
| --- | --- |
| variante | `c` |
| modelo pedido | `minimax-m3` |
| hora UTC (fin de la llamada) | 2026-10-30T10:23:21.737303Z |
| estado HTTP | `200` |
| cuerpo: `model` | `minimax-m3` |
| cuerpo: `id` | `chatcmpl-75734ef013ff4d73a52b9f6a41c80125` |
| cuerpo: `system_fingerprint` | — |
| cuerpo: orden de las claves | `id`, `object`, `created`, `model`, `choices`, `usage` |
| argumentos: orden de las claves | `action`, `confidence`, `size_fraction`, `invalidation_price`, `rationale` |
| completion_tokens | `3316` |
| latency_ms | 17574.6 |
| tok/s | 188.7 |
| router: `upstream_model` | — |
| router: `upstream_endpoint` | — |
| válida | `True` |
| fallo | — |
| petición: sha-256 de la envoltura | `633a0112f7292ecaa04f5b386e2fb31037928eea684d18cdb2e5e900027af620` |
| petición: User-Agent | `AsyncOpenAI/Python 2.54.0` |
| petición: `x-opencode-session` | `crypto-agents-4429349e` |
| petición: el prompt es el de la etapa 1 | `True` |
| cabecera `date` | `Fri, 30 Oct 2026 10:23:21 GMT` |
| cabecera `content-type` | `application/json` |
| cabecera `transfer-encoding` | `chunked` |
| cabecera `connection` | `keep-alive` |
| cabecera `cf-placement` | `remote-IAD` |
| cabecera `x-opencode-log-id` | `01a18c55-a088-7064-ae3c-be992eae712d` |
| cabecera `x-zen-model` | `minimax-m3` |
| cabecera `content-encoding` | `gzip` |
| cabecera `server` | `cloudflare` |
| cabecera `cf-ray` | `a529b3d3ef0046b8-DFW` |

