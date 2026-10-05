# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Permanent rules

These hold in every phase. Each one is enforced by a test in `tests/test_architecture.py`; if you
change the design, the test must keep passing or you must argue why the rule no longer applies.

1. **An LLM never computes a number.** Agents interpret values that arrive already calculated. If a
   model returns an RSI, that is a hallucination. Every numeric field on an `LLMOutput` subclass is
   bounded to `[0,1]` — that makes it a judgement, not a measurement. The single exception is
   `Decision.invalidation_price`, allowlisted explicitly because a price level is a judgement about
   where the thesis breaks.
2. **An LLM is never the last step before an order.** The decider proposes, the risk gate disposes.
   `build_order()` reads its size from `RiskVerdict.final_size_fraction` and never from
   `Decision.size_fraction`.
3. **Every LLM claim cites evidence by id.** `Claim.grounded_in` and `Observation.cites` require at
   least one entry. Without grounding the debate drifts into free opinion within a few iterations.
4. **Every model call is recorded with its prompt digest.** `ModelRouter` is the only path to a
   provider, and it emits an `LLMCall` per attempt. Without the digest there is no backtesting and no
   audit of why the system decided to buy something.
5. **Write the test before calling a phase done.** With non-deterministic agents, "it ran fine" is not
   evidence of anything.

Also: before writing code for a phase, present a short plan (files, main signatures, open decisions)
and wait for approval. One commit per phase, no `--no-verify`. Say so before implementing a design
you would not defend.

## Repository state

All nine phases are implemented. `src/crypto_agents/` holds the package; `tests/` mirrors it.

| Module | Role |
| --- | --- |
| `state.py` | Data contract between every node. Imports nothing else from the package — it is the root of the dependency graph. |
| `settings.py` | `pydantic-settings` config, `CA_` prefix. `load_settings()` fails at startup naming the missing variables. Holds `Billing` (`go`/`payg`) and the dated `PriceTable`. |
| `quota.py` | Sliding-window quota ledger per `(role, model)`, injected clock. Holds a window, not `Settings`: the candidates come from the caller, which is what lets one counter serve several role maps. `seed()` rebuilds the window from journaled calls after a restart. |
| `llm.py` | `ChatBackend` protocol, OpenAI/Ollama adapters, and `ModelRouter` — resolve by budget, cache, validate, retry, record. |
| `cache.py` | Response cache keyed by `(backend, model, prompt digest, schema, mode)`. Each entry is an envelope naming who answered what, and holds every attempt that produced content, valid or not. |
| `market.py` | Two ccxt clients — reading (no credentials, production) and trading (credentials, sandbox) — OHLCV normalisation, reproducible candle digest. |
| `indicators.py` | pandas-ta preset producing a validated `IndicatorSet`. |
| `activation.py` | Four pure gate rules; no state between runs. |
| `prompts/` | Versioned `.md` templates plus loader. |
| `nodes.py`, `graph.py`, `context.py` | LangGraph nodes, compiled graph, execution context. |
| `risk.py` | Deterministic vetoes and caps. |
| `execution.py` | Order construction, paper executor, live executor gated behind three conditions. |
| `journal.py` | Structured record of every evaluation, JSONL or in memory. |
| `runner.py` | Candle-close schedule, multi-symbol cycle, bounded concurrency, clean shutdown. |
| `replay.py` | Historical replay over committed candles: cache-only by default, deterministic ids, canonical run digest. |
| `ablation.py` | Pipeline variants compared over one plan — a manifest or a contiguous history; `python -m crypto_agents.ablation` renders the table, `--dry-run` prices it first. Every run writes a directory: one JSONL journal per arm plus `meta.json`. |
| `audit.py` | Reads a run directory and prints what happened in it, each figure next to the digest of the file it came from. `python -m crypto_agents.audit <dir>`. No model calls. Also owns `load_plan()` / `resolve_horizon()` (the one door to a run's candles, shared with `criteria`) and the net-return section, and `RunKind` (`ablation` \| `probe`): `RunMeta.kind`, an old `meta.json` loads as `ablation`, `zen_probe` writes `probe`, `criteria` refuses a probe directory and `audit`/`consumption` read it. |
| `outcomes.py` | Labels each order against later candles: invalidation hit first, or the close at the horizon. Three scorings — declared stop, common stop, horizon close — and the per-evaluation vector. `net_return()` adds a descriptive net return on top of any of them; the gross path is untouched. |
| `stops.py` | The one function that builds the common stop (`entry ∓ 2·ATR`). Imports only `state`. |
| `baselines.py` | Four decision policies that call no model: always buy, always sell, uniform random, trend rule. Cannot import the router. |
| `alerts.py` | Quota running out, repeated vetoes, validation failures, skipped cycles, evaluations lost to insufficient funds (402). Pure over journal records. Under `billing = payg` the quota alert is silent and `cli alerts` prints `cuota: no aplica (payg)` instead of a percentage against the sentinel. |
| `queries.py` | Journal filters by symbol, action, backend and abort cause. |
| `doctor.py` | Startup checks: gateway catalog, Ollama tags, VRAM split, exchange and credentials. |
| `bootstrap.py`, `cli.py` | Composition root and the `crypto-agents` entry point. |
| `activation_sweep.py` | The four gate rules over a committed history, no model calls. Sizes the ablation. |
| `selection.py` | Stratified selection of activations across symbols and time spans, and the versioned manifest the ablation runs over. No model calls. |
| `metrics.py` | Aggregations over a run — the funnel, action mix, vetoes by rule, quota by role and backend — and the audit of one: attempts, live latency, abort causes, stop side, action against the desks. Pure over `EvaluationRecord`. `AbortKind.INSUFFICIENT_FUNDS` reads the 402 from the message (`Error code: 402` / `Insufficient account funds`, before the generic transport marker); `provider_rejections()` groups rejections by (model, role, code, body) so a 410 and a 503 never share a row. |
| `dispersion.py` | Standard deviation of the per-evaluation return over the whole 4h activation pool (`always_buy`/`always_sell`, common stop) and the detectable paired difference for n = 140/280/420. Carries no mean on purpose: no model field, no printed figure, no file written. No model calls. `python -m crypto_agents.dispersion`. |
| `perp_probe.py` | Public read-only probe of USDT perpetuals on `binanceusdm` / `bybit`: contract limits, 24 h volume, funding (the last 730 days, or an explicit `--start`/`--end` range) normalised to 24 h, connectivity, `exchange.has`. Imports nothing from the package. No keys, no orders, no model calls. `python -m crypto_agents.perp_probe --exchange binanceusdm`. |
| `funding.py` | Reads the versioned `data/funding/` series and answers one question: the sum of funding rates over `(entry, exit]`, or `None` if the series cannot guarantee it is all there. Stdlib plus `perp_probe`; no network, no credentials. |
| `consumption.py` | Cost of each call in USD and its share of the subscription pool, from the tokens the provider reported; per-arm and per-role report; declared `quota_per_window` against the page's estimate. Measures only. `call_cost_range_usd()` gives `(low, high)` per call: the high end charges the non-cached prompt at `cache_write` where the model has one, and a call with `cached_tokens: null` runs from all-cached to all-new; `format_cost()` marks an interval with `†`. `python -m crypto_agents.consumption <run dir>` / `--quotas`. |
| `zen_probe.py` | Probe of OpenCode Zen (pay as you go): `/models` catalog, structured-output mode per id, `x-opencode-session` with and without, 12 real technical verdicts per (candidate, dimension) and a chained desk/decider stage that measures their tokens. Everything through `ModelRouter`, no cache, no fallback; writes a run directory (`var/zen-probe/<UTC start>/`) that `audit` and `consumption` read. Refuses `billing != payg` and any `/zen/go` base_url before building a backend. `--desks --technicals-from <probe dir> --max-usd X` probes the desks instead: four bull candidates, `minimax-m3` as bear and `glm-5.2` as decider over the *same* technical evidence (produced once per activation by the producer the current rule picks from the previous probe), the decider once per (activation, bull with a valid brief) in its own `decider+<bull>` arm, and a hard `SpendGuard` cap (no rigorous cost bound exists before calling: the repo sets no `max_tokens`). `python -m crypto_agents.zen_probe --machine desktop\|laptop [--dry-run]`. |
| `estimate.py` | USD estimate of stage 1: the `--dry-run` counts over `data/ablation_selection.json` times the measured cost per call of a probe directory. No token is estimated; local calls and baselines are 0 by rule; structure and volume are a range between candidates that answered; one top-up (`PriceTable.topup_charge`). Labelled as an estimate. `--desks <dir>` gives one line per bull candidate with its own conditioned decider; `--balance X` (read from the console, never fetched) answers `PASA`/`NO PASA` against cost x `LAUNCH_MARGIN` at the high end, exit 1 when nothing passes. `python -m crypto_agents.estimate <probe dir>`. |
| `criteria.py` | Mechanical evaluator of the amendment's criteria over a run directory: `full` against `solo`/`no_debate`/`bull_only` and every arm against each of the four baselines (paired difference, 95% CI, verdict from a mandatory `--delta`), the run-validity guards (decider lost to quota, an evaluation lost to insufficient funds at any node, cache hit from another backend, any veto but `invalid_stop_side`: `CORRIDA INVÁLIDA`, exit 1, no verdicts), and the peak 5 h window usage per role (`no aplica (payg)` instead of a share when the run was paid per use). No model calls. `python -m crypto_agents.criteria <run dir> --delta X`. Carries two extra columns per comparison with the net-return paired difference, labelled descriptive; they enter no verdict. |

Pipeline, one evaluation = one symbol at one moment:

```
START -> prepare -> (no triggers) ------------------------------> journal -> END
                 -> structure ┐
                    momentum  ├-> consolidate -> (no evidence) -> journal -> END
                    volume    ┘               -> bull ┐
                                                 bear ┴-> decide -> risk -> execute -> journal
```

Every exit path goes through the journal, aborted runs included: an evaluation that did not trade is
exactly the one worth auditing.

`Runner` drives that pipeline once per symbol per candle close. It owns the schedule, not the graph:

```
next_close(now) -> sleep (in poll_seconds slices, so stop() is noticed)
                -> run_cycle: every symbol, max_concurrent at a time
                -> overran the next close? journal the skip, jump forward
```

Three invariants, each with a test in `tests/test_runner.py`:

- **One `QuotaLedger` for every symbol**, enforced by the signature: `Runner` holds the
  `ModelRouter` and hands it to the context factory, so a per-symbol ledger has nowhere to come
  from. With N private ledgers the aggregate spend would be N times the declared budget while each
  one believed it was inside its limit.
- **Falling behind never accumulates.** Missed closes are journaled as skipped and the schedule
  jumps to the next one. Chaining late cycles makes the system trade on stale candles believing it
  is current.
- **Nothing ends unrecorded.** `stop()` lets the in-flight cycle finish; a failure before the graph
  even starts still writes a record signed `node="runner"`, because a gap in the journal is
  indistinguishable from a symbol nobody asked for.

### A provider that misbehaves is the case the journal exists for

Five failure modes reach a node, and they are not the same failure. `FailureKind` is closed —
`schema`, `context`, `timeout`, `transport` — and `LLMCall.failure_kind` is `None` exactly when the
attempt is `valid`:

| What broke | Type | `LLMCall.failure_kind` | Retried | Cached |
| --- | --- | --- | --- | --- |
| Content arrived and did not match the schema | `InvalidModelOutputError` | `schema`, one row per attempt | yes, with the error attached | yes |
| Content matched the schema and contradicted its context | `InvalidModelOutputError` | `context`, one row per attempt | yes, with the error attached | yes |
| The provider did not answer in time | `ModelCallError` | `timeout` | no | no |
| The provider never returned content (4xx, 5xx, DNS, credentials) | `ModelCallError` | `transport` | no | no |
| Neither model fit the window | `QuotaExhaustedError` | no row — nothing was spent | no | — |

The first two measure the model, the next two the provider. Journal lines written when the failure
was a nested `failure: {kind, message}` still load: `validation` reads as `schema`, which is all it
could be then.

**Context validation lives in the router, not in the node.** What no JSON Schema can express — a
brief citing an observation id nobody emitted, a verdict citing an indicator that does not exist, a
desk answering as the other one — used to be checked by the node *after* the router had declared
the attempt valid. The call was journaled `valid=True`, the answer was cached, and the evaluation
aborted without the model ever seeing what was wrong. Now the node hands the router a `ContextCheck`
— a pure function returning a list of errors — and a context failure is an invalid attempt like any
other: recorded, charged, retried with the error attached. Schema and context share the same two
attempts. The nodes keep their checks as assertions; one firing is a bug and is journaled as a
`NodeError` prefixed `validación de contexto saltada`.

The retry template is unchanged and still says "esquema" for a context error; rewording it would be
changing a prompt. The error that follows it names what happened.

Four decisions hold this together:

- **The call is inside the `try`.** It used to be outside, so a provider failure meant a request the
  gateway had already seen and no row anywhere: rule 4 broke precisely when the provider
  misbehaved, which is when the record is the only evidence of what happened.
- **The router wraps everything a backend raises.** It is the only module allowed to import a
  provider, so it is also the only place that can turn an httpx, openai or ollama exception into one
  type a node can catch without importing any of the three. Before that, a transport error took the
  whole graph down and the evaluation left no candle, no gate and no calls — one line signed by the
  runner, which cannot distinguish a broken market from a broken provider.
- **The attempts travel inside the exception.** `QuotaLedger` already has them, but it lives in
  memory and dies with the process. `_model_failure()` copies them into the state so they reach the
  file, where somebody will look three days later.
- **A transport failure is not retried.** The retry template corrects a validation error; against a
  400 the second attempt is the first one with extra text, and it costs quota with certainty.

`BackendNotCalledError` is the deliberate hole in the wrapping: `CacheOnlyBackend` raises it to say
it refused to call, and the router re-raises it untouched. Dressed as transport, a replay cache miss
would send someone to check the network instead of filling the cache, and the message naming model,
schema and prompt — the only thing that says what to fill — would be buried.

`metrics.backend_stats()` counts content failures (`invalid`, with `context` as a sub-count) apart
from provider failures (`transport`, `timeout`), and `failure_rate` divides by `answered`, not by
`attempts`. A model id the gateway does not serve and a model that hallucinates
fields are one number apart otherwise, and the obvious reading of that number — "this model cannot
follow the schema" — is false for the first one.

## Environment

Dependencies are managed with **uv** (lockfile committed, Python 3.12+). Never pip or poetry.

```bash
uv sync                        # install/refresh .venv from uv.lock
uv add <pkg>                   # runtime dep; --dev for tooling
uv run ruff check .            # lint
uv run ruff format .           # format (line-length 100)
uv run mypy                    # strict, over src/ and tests/
uv run pytest                  # 1536 tests
```

All four must exit 0 before a phase is done.

### The two machines

Development happens on the desktop; the operating system-under-test is hosted on the laptop.

| | Desktop — develops | Laptop — hosts |
| --- | --- | --- |
| CPU | Ryzen 7 5800X | Ryzen 9 5900HS |
| RAM | 16 GiB | 37.5 GiB |
| GPU | RTX 3070 Ti, 8 GB | RTX 3080 Mobile, 8 GB |
| Free VRAM, desktop session loaded | 7.0 GiB (measured 2026-08-14) | not measured yet |

Same VRAM on both, so the local model is sized once and the answer transfers. **Throughput does
not.** The 3070 Ti runs without the Max-Q thermal envelope, so any local latency measured on the
desktop is optimistic for the laptop by an unmeasured factor. Every latency figure in this file
names the machine it was taken on; one that does not is unusable, because there is no way to tell
which of the two it describes.

## Conventions

- Identifiers, module names and field names in English. Docstrings and comments in Spanish.
- Full type hints. `from __future__ import annotations` in every module — enforced by ruff's
  `required-imports`.
- Pydantic v2 only (`model_validator`, `ConfigDict`, `Field`). Never v1 syntax or
  `langchain_core.pydantic_v1`.
- Config through `pydantic-settings`. No `os.environ` reads scattered in modules.
- No `dict[str, Any]` as an escape for a field you did not want to model.
- Anything a model produces inherits `LLMOutput` (`extra="forbid"`, `frozen=True`).
- Injected clocks everywhere time matters (`QuotaLedger`, `AgentContext.clock`, `drop_forming_candle`).
  No `datetime.now()` inside logic.

## Replay and backtesting

`replay()` walks a committed history candle by candle, running the whole graph at each close and
resolving every model call from the cache. Two properties are structural, not intentional:

- **Cache-only by default.** `replay_router()` wires a `CacheOnlyBackend` into every slot unless you
  pass `fill_with`. A cache miss raises `ReplayCacheMissError` naming the model, schema and prompt
  instead of quietly spending: a 500-candle replay is 3000 calls. Filling the cache for a new
  history is the explicit `fill_with` mode, paid once.
- **The forming candle is handed over, not hidden.** `HistoricalMarketClient` returns the window up
  to the evaluated candle *plus the next one*, exactly as the exchange would, so
  `drop_forming_candle` does real work on every iteration. Hiding it in the harness would leave the
  one function that prevents look-ahead untested across an entire backtest.

### What has to be pinned for a replay to be reproducible

Four things differ between two runs unless they are tied down. The first three are handled; the
fourth is why the digest exists.

| Source | How it is pinned |
| --- | --- |
| `run_id` | `replay_run_id()` — a UUID5 of `symbol\|timeframe\|instant`, not a UUID4 |
| Wall clock | The injected clock is the evaluated instant, so every `at` is a function of the data |
| `latency_ms` | Only reproducible on a cache hit, where it is `0.0` |
| `calls` ordering | `run_digest()` sorts by `(role, prompt_digest)` before hashing |

The latency row is the one that decides the acceptance criterion: **a replay with a full cache is
deterministic including timings; one with a cold cache never is.** Determinism is asserted between
two warm runs, and `tests/test_replay.py` also pins the opposite — a cold run and a warm run agree
on every decision while differing in digest, so nobody goes hunting for a phantom bug.

On ordering: while everything comes from the cache there is no await point and the fan-out resolves
in a stable order, so the canonicalisation is not what makes two warm runs match. It earns its keep
comparing a warm run against one that was not — with real latency the three technical nodes finish
in any of the six orders.

### Look-ahead

`tests/test_lookahead.py` computes the preset over the full series and over the series truncated
right after bar `i`, and requires bar `i` to be identical. Truncation is only ever on the right:
EMA and ADX are recursive, so cutting the *start* legitimately changes bar `i`. It covers all ten
preset columns, all four gate rules individually, and both the short test preset and the production
`IndicatorPreset()` / `DEFAULT_CONFIG`. Run against the real history rather than a synthetic series,
because a hand-made shape can hide the error exactly where it is probed. Nothing looks ahead today.

The history lives in `tests/data/` with its provenance and digest; `tests/test_replay.py` asserts
the file still hashes to the recorded value, so an edit cannot silently make two backtests
incomparable.


## Perpetuals probe

`python -m crypto_agents.perp_probe --exchange binanceusdm|bybit` reads public endpoints only and
writes `var/perp/<exchange>/<UTC start>/` (not versioned, never reused): one CSV per symbol, `spec.json`
and `meta.json`. It reports; it does not choose an exchange, and it prints no strategy return.

- **The funding interval is inferred, not assumed.** Each period covers `(ts[k-1], ts[k]]`; its interval
  is the gap to the previous row, accepted only on the {1, 2, 4, 8} h grid within 60 s. The first row
  and any off-grid gap are excluded and counted. A missing record inside a 4 h stretch looks like a
  valid 8 h gap and cannot be detected. `tests/test_perp_probe.py` carries a hand-computed series with
  8 h, 4 h and 1 h stretches, and a test that a normaliser fixed at 8 h would give another figure.
- **Pagination does not depend on each exchange's bounds semantics.** Binance returns the oldest rows
  from `since`; Bybit, given `until`, the newest of the window, so paging forward would skip data
  silently. Every request is a closed window, a page with `limit` rows or more is ambiguous and is
  split, windows overlap by one edge and are deduplicated, and 600 requests per series is a hard stop.
  The initial window is `limit - 2` periods: a closed window of `limit` periods holds `limit + 1` rows
  and always came back "full" (217 Bybit requests instead of 84, same series digest).
- **One request per window, no retries.** HTTP 429 or 418 aborts the whole probe, since binance bans
  the IP if pressed; any other failure leaves that symbol `no determinado` and continues.
- **Every figure cites a digest.** The series digest is the sha-256 of the CSV bytes, so `sha256sum`
  verifies it; contract and volume rows cite the sha-256 of the canonical JSON of their entry.
- **No headers, no query.** `instrument()` wraps `exchange.fetch` and hooks `on_rest_response` to get
  status and latency per HTTP request, forwards the header arguments untouched and keeps method, host
  and path only.
- **`exchange.has` is what ccxt implements, not what an account may use.** `createOrder` with
  `reduceOnly` has no flag of its own; `createReduceOnlyOrder` is the closest.
- **Bybit's `limits.cost.min` is `None` for all seven symbols.** The minimum notional (5 USDT) is in
  `info.lotSizeFilter.minNotionalValue`, which the probe does not read: it asked for `market['limits']`.

`tests/test_architecture.py` pins that the module imports nothing from `crypto_agents` (relative
imports included), never names a credential field or reads the environment (docstrings excluded), and
builds ccxt with only `enableRateLimit` and, for Bybit, `fetchMarkets: {types: [linear]}` — no keys, no
proxy, no sandbox, no open session.

Fixtures in `tests/data/perp/` are one public capture made on 2026-10-02 22:08 local, 2026-10-03T04:08Z (provenance and digests in its
README). They are ccxt-level responses, not raw HTTP; the 4 h / 1 h / gap cases are synthetic and say so.

## Token usage and pool consumption

`LLMCall` carries `prompt_tokens`, `cached_tokens` and `completion_tokens` (`int | None`, `>= 0`). They
are what the provider reported, never an estimate: no counter, `None`. Old journal lines load.
`consumption.py` turns them into dollars and a share of the OpenCode Go pool; it changes neither
`QuotaLedger`, nor the retry policy, nor the role → model map.

Measured against the gateway on 2026-10-03 (12 calls, `tests/data/usage/`, README has the details):

- **`prompt_tokens` includes the cached ones; `completion_tokens` includes the reasoning.** So a call
  costs `(prompt - cached)·input + cached·cached_price + completion·output`. Charging the cached tokens
  as input is the mutation `tests/test_consumption.py` pins.
- **LangChain turns a missing counter into `0`** (`_create_usage_metadata`: `prompt_tokens or 0`), so the
  adapters read `response_metadata["token_usage"]`, the provider's own dict. An architecture test
  forbids `usage_metadata` in `llm.py`.
- **DeepSeek V4 Flash reports `cached_tokens: null` on a cold call** and the number on a warm one; the
  other models report `0`. `null` is not zero: that call has no exact cost, only a ceiling (all input
  charged as new), which the report labels as such.
- A `ValidationError` escaping LangChain loses the message and with it the usage, though the attempt was
  billed: `None`, counted as unmeasured.

Rules, each with a test:

- **Free is free by rule.** A cache hit and a local call cost `0.0`, whatever tokens they carry. Under
  `payg` `pool_fraction` is always `None`: there is no pool.
- **A call without tokens does not cost 0.** It is counted apart (`sin medir`), the figure of its arm becomes a
  lower bound (`≥`), and a provider that never answered (transport, timeout) has its own column.
- **The peak is decided by `LLMCall.at` in UTC**, `[01:00, 04:00)` and `[06:00, 10:00)`, Monday to Friday
  (`PriceTable.is_peak`). Valid for live calls only: a replayed cache hit carries the evaluated instant.
- **Prices live in `settings.py`** (`DEFAULT_PRICING`, copied from the page on 2026-10-02, with the page's
  requests-per-5-h estimates). `consumption.py` may not carry a number beyond 0, 1, 100 and 1e6.
- **`run_digest` omits the token fields when they are `None`**, so digests of runs without tokens did not
  change and `replay-v3` did not move; a run with measured tokens hashes them, like latency.
- **`ChatBackend.complete()` returns `str | Completion`.** Real adapters return `Completion(text, usage)`;
  fakes that return text mean "no usage". `as_completion()` is the one place that normalises it — `doctor`
  needed it too.
- **`RunMeta.billing`** is written by the ablation; a run without it needs `--billing` on the command, and a flag
  that contradicts `meta.json` is refused. `--dry-run` prices the pool with the page's estimates, labelled
  "estimación de la página, no medida".

`python -m crypto_agents.consumption --quotas` against the shipped config lists two disagreements with the
page, and corrects nothing: `momentum` (63 300 declared, 31 650 effective with weight 2.0, against 13 000)
and `bull` (4 300 against 1 150).

## Net return on perpetuals (Q1)

Orders will go to Binance USDⓈ-M perpetuals, and the scoring above is gross. `outcomes.net_return()`
adds a **descriptive** net return; amendment 2's criteria and δ stay on the gross one, and a test pins
that gross vectors and verdicts are identical with no costs, with absurd costs and with no funding.

`net = gross − 2·(taker_fee + slippage) − side · Σ funding`, `side` = +1 for a long (pays a positive
rate) and −1 for a short (collects it). Costs live in `settings.CostModel` with an `as_of` date, and
nowhere else (`tests/test_architecture.py`): `taker_fee` 0.0005 per side is confirmed on the account
(VIP 0, no BNB discount, 2026-10-03); `slippage` 0.0002 per side is an **unmeasured assumption**.
`audit` and `criteria` use `DEFAULT_COSTS` and print it in their header.

- **The window is `(entry, exit]`.** Entry is the close of the evaluated candle, so a settlement at that
  exact instant is not paid; one at the exit instant is. A stop is hit *inside* a candle: the settlement
  at that candle's open is paid, the one at its close is not, and one strictly inside it makes the order
  `no determinado`. Nothing after the exit is read.
- **Binance timestamps carry 0–26 ms of jitter, always late** (`08:00:00.013`, 42% of rows). Compared raw
  with an entry at `08:00:00.000` that settlement would count as "strictly after", i.e. the one that is
  *not* paid. `funding.py` snaps each mark to the nearest minute on load.
- **Missing is not zero.** A window needs a row at or before the entry, a row at or after the exit and no
  off-grid gap between them (`perp_probe.interval_hours`); otherwise `None`. The per-evaluation net keeps
  that `None`, the paired difference drops those pairs and prints how many (`dropped`), and no figure is
  ever filled with 0. Known limit, inherited from the probe: a lost row inside a 4 h stretch looks like a
  valid 8 h gap and cannot be detected. The committed series are 8 h throughout.
- **The funding is a plain sum of rates.** The real charge is on the notional at each settlement's mark
  price; rescaling by the move since entry is a second-order effect and the scoring does not do it.
- **`data/funding/`** holds the seven series, `btcusdt.csv` etc., the exact bytes `perp_probe` wrote for
  `--start 2024-08-16 --end 2026-08-16T00:00:00` (the history's range plus the settlement at the last
  candle's close). The README carries provenance and sha-256, and `tests/test_funding.py` checks them.
  Over every real 4h candle, both sides and all six stop exits, none of 244 944 positions comes out
  undetermined.
- **The three mutations the net cannot survive** — short sign not inverted, the entry-instant settlement
  included, missing funding assumed 0 — are tested by rewriting the production source with the bug in
  and requiring the hand-computed check to fail (`tests/test_net_outcomes.py`), plus a fourth for exiting
  a stop at the candle close.

## Operating it

```bash
crypto-agents status     # is it stopped? how many evaluations, how many traded
crypto-agents doctor     # can this thing start at all? exit 1 if not
crypto-agents stop       # no order leaves the system until resumed
crypto-agents resume     # removes the sentinel
crypto-agents alerts     # exit code 1 when something fires, so scripts can chain it
crypto-agents query --symbol BTC/USDT --action buy --group-by-cause
crypto-agents run        # the candle-close loop over CA_RUNNER__SYMBOLS
```

### `doctor` asks the outside world, not the configuration

Five checks, each against the real thing, exit 1 if any fails:

```
gateway   OK     6/6 ids presentes en https://opencode.ai/zen/go/v1
modes     OK     bear json_mode (1.7 s), bull json_schema (3.3 s), decider function_calling (1.9 s), momentum json_mode (2.3 s), structure json_schema (1.3 s), volume json_schema (2.4 s)
ollama    OK     qwen3:8b descargado en http://localhost:11434
vram      OK     qwen3:8b 5.6 GiB, 100% GPU con num_ctx=4096 (5.6 s)
exchange  OK     binance lee 500 velas de BTC/USDT en https://api.binance.com/api/v3; órdenes contra https://testnet.binance.vision/api/v3 (sandbox=true) con credenciales válidas
```

The gateway check is the one that pays for the command. A model id that the gateway does not serve
loads fine and fails on the first call — halfway through an evaluation that already paid for the
calls before it. It costs one request to the catalog to know instead, and the failure names the role
and the id (`decider → glm-9.9`) rather than raising a traceback: every probe converts its exception
into a `FAIL` with the message inside, because a ccxt stack trace does not tell anyone which
variable to change.

Two design points worth keeping:

- **The probes live in `llm.py` and `market.py`, not in `doctor.py`.** `tests/test_architecture.py`
  forbids any module but the router from importing a provider. Reaching the catalogs with a
  hand-rolled `httpx` call would pass that test while defeating it, so `OpenAIBackend` and
  `OllamaBackend` grew `available_models()` and the door stays where it was.
- **Sandbox is checked by consequence, not by intention.** `CA_EXCHANGE__SANDBOX=true` is what you
  asked for; `api_base_url` is what ccxt will actually call. `doctor` builds a second *trading*
  client with sandbox off and fails if the two URLs match, which is exchange-agnostic — no substring
  matching on "testnet".
- **The exchange check asks for a full window, not for two candles.** Two candles prove the exchange
  answers and nothing else. The question that matters is whether this source carries enough history
  for the preset, and `doctor` fails naming both numbers when it does not.

The local probe does not go through `ModelRouter`, so it emits no `LLMCall`. Rule 4 exists so that
no call *belonging to an evaluation* goes unrecorded; this one belongs to none, is local and free,
and its record is the line printed. Probing a remote model this way would break that argument and
has to be routed instead — which is exactly what `modes` does.

### `modes` costs money, and that is the cheaper half of the bill

The structured-output mode is the second thing configuration cannot tell you. A model that does not
support the declared mode does not fail: it answers through a channel the adapter is not reading,
and the symptom is an empty response blamed on the model. So `check_modes` asks each remote model,
with the real `Ping` schema, whether the declared mode produces output.

Four properties, each with a test:

- **It goes through `ModelRouter`.** These calls cost quota, so the argument that excuses the local
  probe does not cover them: every attempt emits its `LLMCall` and the ledger counts it.
  `crypto-agents doctor` is no longer free — six remote calls per run, and up to two more per role
  that fails on content while the working mode is found.
- **The six roles are probed at once.** Sequentially the check costs the sum of six latencies;
  concurrently it costs the worst of them. Measured on the desktop against the gateway with the
  `Ping` schema: 12.9 s of probes collapse into 3.3 s, and the whole command finishes in 15.8 s.
  (The 5.7–56.3 s figures elsewhere in this file are the `technical.md` prompt, which is 1500
  characters against this one's 60 — the same models, a different question.) It is safe because they are six independent HTTP requests against the same gateway,
  with no GPU in the path and no shared ledger: quota is kept per (role, model) pair and each task
  owns its own. This is the opposite of `run_checks`, which stays sequential because the VRAM probe
  loads a model onto the GPU and its latency is the measurement.
- **A transport rejection is named as one, and stops the search.** With six probes in flight a
  gateway rate limit is exactly the false negative that would send someone to change
  `STRUCTURED_OUTPUT` when the right move is to wait. If the provider never generated content the
  declared mode was not disproved, so hunting for an alternative would be two more rejections and
  two wasted calls per role.
- **The probe router has no cache and no retry.** A cached entry would make the second run of
  `doctor` answer yes about a provider that is switched off. A retry measures whether the model
  corrects itself when shown the error, which is a different question and costs double; every
  evaluation calls it once, so the probe calls it once.
- **The probe settings have no fallback.** `resolve()` degrades to the local model the moment the
  remote does not fit the window, and then the probe would report that the remote's mode works when
  the answer came from a different model entirely.
- **Running out of budget is not reported as a broken mode.** The search is cut short and says so.
  Reporting "no mode works" there sends someone to change a configuration that may be correct.

A failure names the role and the line to write, not just the diagnosis:

```
modes  FALLA  decider → glm-5.2: json_schema no dio salida válida; function_calling sí → CA_ROLES__DECIDER__PRIMARY__STRUCTURED_OUTPUT=function_calling
```

### Reading the market and placing orders are two clients

One client held both roles and only one of them wants credentials. Both consequences were measured
on the first real run:

- **With the keys set, ccxt signs the public endpoints too.** Production answers
  `-2008 Invalid Api-Key ID` to a candle read that returns 500 rows without them.
- **Testnet is not a data source.** It answers perfectly and returns 58 4h candles against a preset
  that needs 400, so with `sandbox` governing the read no evaluation could ever complete.

So `CcxtMarketClient` takes an `exchange_id`, not an `ExchangeSettings`. The guarantee is the
signature: there is no parameter through which a key could arrive, and no `sandbox` to honour. It
always reads production. Accepting the settings object and ignoring three of its fields would be a
configuration that says one thing and does another — the same objection that keeps Ollama from
declaring a structured-output mode it does not implement.

`CcxtTradingClient` is the one that signs, and the only one `CA_EXCHANGE__SANDBOX` still applies to.
**That variable now governs where orders go, not where candles come from.** Its only use today is
`doctor`'s credential check, since a real send also requires `CA_EXECUTION__MODE=live`.

Public ingestion depends on no flag and no script. `CcxtMarketClient` is constructed in exactly four
places — `bootstrap.py`, `cli.py`, `doctor.py` and `activation_sweep.py` — and every one of them
passes an `exchange_id` and nothing else, because the signature accepts nothing else. There is no
environment variable that can point it anywhere but production: `settings.py` is the only module
that reads configuration at all, and the reader never receives it.

The reader also loads **spot markets only** (`SPOT_ONLY`). `load_markets()` runs before the first
candle and binance loads three universes in parallel — spot, linear futures, inverse futures — so
one unreachable host kills the whole read. `dapi.binance.com` returned `RequestTimeout` twice in a
row from the desktop while spot answered fine; in the runner that is one evaluation lost per cycle
for a market nobody asked about.

Verified against the live exchange with testnet keys in `.env`, three runs in a row: 500 candles
from `https://api.binance.com/api/v3`, orders pointed at `https://testnet.binance.vision/api/v3`.

### The kill switch is a risk-gate concern, not a runner concern

`risk_gate` consults `AgentContext.kill_switch` on **every** evaluation and, when engaged, hands
`apply_risk()` limits with `kill_switch=True`. Three consequences, each with a test:

- **No path to the market escapes it.** Had the runner checked instead, an evaluation already in
  flight would still place its order, and any other entry — a replay with execution, a manual run —
  would bypass it entirely. Every graph variant routes decide → risk → execute, so the gate is the
  one place all of them share.
- **`apply_risk()` stays pure.** The I/O lives in the node; the veto rule still reads a boolean.
- **Nothing is cached.** A value read once at startup would never notice the sentinel appearing,
  which is precisely when the switch is supposed to work.

`FileKillSwitch` treats an unreadable sentinel as engaged. A kill switch that fails open is not a
kill switch: a broken permission or a full disk should stop trading, not wave it through.
`AnyKillSwitch` combines the file with `CA_RISK__KILL_SWITCH`, so `resume` warns — and exits 1 — when
removing the file leaves the config one still on.

### Alerts read the journal, not live memory

`alerts.py` is pure over `EvaluationRecord`s. Quota consumption is rebuilt from `LLMCall.at` and
`quota_weight` rather than read off the live `QuotaLedger`, so the same alert evaluates identically
in-process and over yesterday's file. An alert that only existed in memory could not be audited
afterwards, which is exactly when someone asks why nobody warned.

Two thresholds exist to keep the alerts worth reading: a validation-failure rate needs
`min_attempts` before it is reported (1 of 1 is 100% and means nothing), and the repeated-veto alert
excludes `kill_switch`, whose repetition is its job once you engage it.

## The activation gate, measured

`python -m crypto_agents.activation_sweep` runs the four rules over two years of candles for seven
symbols in 4h and 1h — 153 000 bars, zero model calls, under a minute. The report lives in
`docs/activation.md`; the candles are cached in `var/history/` and are not committed, so what makes
two tables comparable is the per-series sha-256 digest written into the report.

Three answers it produced:

- **4h is viable.** 3 980 evaluations per symbol after the 400-bar warm-up, and the gate opens on
  15.5–18.5% of them: 4 740 activations in 4h across the seven symbols. The ablation is not waiting
  for material — under the **Go subscription** its limit was the decider's 880 calls per window.
  **All six arms reach the decider** (`decide`, `decide_without_debate` and `decide_solo` are one
  node each, and the two local arms are `full` with a role swapped), and none of them reuses
  another's decider entry, so the Go ceiling was 880 / 6 ≈ 146 activations and the selection
  committed **140**, leaving 40 calls for retries. **That derivation is Go's, not the system's:**
  the batch evaluation now goes through Zen (pay as you go), which publishes no request limit (see
  "Pay as you go (Zen)"). **n = 140 stays**, because stage 1 is an operational screen (amendment 2,
  criterion 8) and not because a ceiling imposes it.
- **No rule is dead, but the split is lopsided.** `range_breakout` produces 55% of the triggers and
  `volatility_jump` 7% — as few as 20 firings in two years for SOL/USDT in 4h. Any claim about that
  rule at 4h rests on a small sample.
- **1h does not raise the rate, it multiplies the bars.** 17.0–18.8% against 15.5–18.5%: the gate's
  rate is nearly timeframe-invariant. What 1h buys is 4.3× more bars, not a looser gate.

Which retires the observation that started it: with a ~17% rate, seven symbols staying quiet at one
candle close has probability 0.83⁷ ≈ 27%. That is one close in four, not a symptom.

Two properties make the number trustworthy, each with a test: the sweep calls `evaluate_activation`
rather than a copy of the rules, so the table measures the gate and not the harness; and the module
cannot import `crypto_agents.llm`, so re-running it can never cost money.

## Ablation

`python -m crypto_agents.ablation` runs one history under four shapes of the pipeline and prints the
comparison. The report lives in `docs/ablation.md`, criteria written before the numbers so the
conclusion cannot be fitted to whatever comes out.

`PipelineVariant` selects the shape; `build_graph(variant)` builds it. What never changes is the
deterministic tail — decider → risk → execute → journal — and there is a test per variant that says
so. An ablation that altered it would measure the harness, and would also open a path to the market
that skips the risk gate.

Two contract consequences worth knowing before adding a variant:

- **`Proposal` is the base, `Decision` extends it.** `Decision` demands `dismissed_side` for any
  non-hold action, which is right when desks exist and impossible when they do not. Rather than
  soften the rule that the operating pipeline depends on, the variants without desks emit a
  `Proposal` and write it to `TradingState.proposal`. `apply_risk()` and `build_order()` read
  `state.proposed`, so every arm reaches the market through identical code.
- **Counting decisions means counting `proposed`, not `decision`.** `summarise()` learned this the
  hard way: reading only `decision` reported zero decisions for exactly the arms that exist to be
  compared against the full pipeline, which would have argued for the opposite of the truth.

One number is settled without running anything: `solo` spends **one call per evaluation against
six**, fixed by a test. To justify the full pipeline it is not enough that it decide *differently* —
it has to decide better by enough to pay six times the cost. A tie is a loss for the architecture.

### `--dry-run` counts the bill before opening it

`python -m crypto_agents.ablation --dry-run` walks the history running only the deterministic layer
and prints calls per role and per arm. It exists because the ablation is the one command whose cost
is paid in hours: at ~90 s of wall clock per evaluation, finding out mid-run that an arm cannot start
costs the hours already spent, and one minute of CPU buys the answer instead.

Four properties, each with a test:

- **It cannot call anybody.** The context it builds carries the replay router with no `fill_with`,
  so every slot is a `CacheOnlyBackend`. A single call would raise `ReplayCacheMissError` rather than
  quietly spend, which is also what makes "it emitted zero calls" testable instead of asserted.
- **It runs `prepare_evaluation()`, the same function the `prepare` node runs.** The gate's rate is
  what multiplies everything else, so a copy of those five lines here would budget the harness.
- **Exact is separated from upper bound.** The three technical prompts and `decide_solo`'s are pure
  functions of snapshot, indicators and triggers, so their digest — and therefore their cache key —
  is computable before spending anything. Everything downstream of a verdict only admits "how many
  times it could run". Rendering the two as one number would invite reading the sum as the invoice.
- **The aggregate tables add up what reaches the provider, not what the arms ask.** `QuotaLine`
  (quota per role, pool consumption) carries `exact_calls` — the arms' `to_pay`, so a hit from the
  same run or an earlier one does not count — apart from `bound_calls`, the `≤` of what follows a
  verdict. Three arms sharing technicals add them once; the rows per arm still say 140 three times,
  because three arms ask. The total is labelled an upper bound at one attempt per call, is given in
  the monthly pool too (`five_hour_share` from `settings.pricing`), and carries `sin reintentos`: the
  dry-run models no retries. Over `data/ablation_selection.json` with an empty cache it was 164.29% of
  a 5 h window while counting hits; it is 149.90% (≈ 29.98% of the monthly pool) without them.

The nodes that spend are read off the compiled graph, not from a hand-written table: a new model node
in a variant that nobody declared makes `llm_nodes()` fail rather than under-count.

What the first dry-run over `tests/data/btcusdt_4h.csv` found, and both were blockers:

- **`local_bull` cannot run at all.** The arm asks for `bull` on its local fallback and `bull`
  declared none — the config mapped a fallback only for the three technical roles. It fails in
  `arm_settings()` naming the arm, which is the right place, but it fails on arm six after five
  arms have already been paid for. The dry-run reports it in a second. Fixed: `bull` declares
  `qwen3:8b`, and `tests/test_settings.py` now reads the shipped template and fails if any arm asks
  for a role in local that the template leaves without a fallback.
- **The committed history caps the comparison at 15 activations.** 500 rows minus a 400-bar warm-up
  leaves 100 evaluable candles, and the gate opens on 15 of them — 15%, consistent with the sweep's
  15.5–18.5% at 4h. Six pipeline shapes over 15 decisions do not separate. Fixed by the selection
  below: 500 candles were never the available history, only one page of the API.

Cost is far lower than the serial estimate for the same reason: only an evaluation that opens the
gate calls anybody. 25 evaluations produce 4 activations, so the arm pays 4 pipelines, not 25.

Cross-arm reuse is real and the dry-run shows it. `no_debate` and `bull_only` pay **zero** technical
calls — same prompt, same model as `full`, so the cache answers all 45. `local_technicals` pays its
three again, because the key includes the model and a local 8B's answer is not the remote's. The
decider never dedupes: its prompt carries the briefs, and every arm feeds it something different.

### What the ablation runs over: a selection, not a page of the API

500 candles were never "the history": they are what one `fetch_ohlcv` returns. `download_history()`
pages with `since`, so two years of 4h is 4 380 candles per symbol and ~630 activations each — 4 740
across the seven. Material is not the constraint. Under Go the decider's 880 calls per window was,
and six arms reach it (880 / 6 ≈ 146), so the comparison committed to **140 activations**; under Zen
there is no such ceiling and 140 stays as the size of the stage-1 operational screen.

Which 140 decides what the table measures. 140 contiguous activations of one symbol are a market
regime, and comparing six pipelines under one regime answers a question nobody asked: the advantage
that matters is the one that survives a change of regime. `python -m crypto_agents.selection` spreads
them — 20 per symbol, and within each symbol 4 per each of 5 equal time spans — and writes
`data/ablation_selection.json`, which is committed along with `data/history/*_4h.csv`.

Four properties, each with a test in `tests/test_selection.py`:

- **The seed picks, the split doesn't depend on it.** Quotas per symbol and per span are arithmetic;
  `random.Random(seed)` only decides *which* activation inside a cell. A different seed selects
  differently — otherwise declaring it would be decoration — and the same seed reselects identically.
- **Every entry is confirmed against the window the replay will see.** The sweep computes indicators
  over the whole series and truncates; an evaluation computes them over the 500 candles the exchange
  returns. EMA and ADX are recursive, so the two can disagree on the same bar. A candidate enters
  only if the gate also opens on the truncated window — otherwise the manifest would promise 140
  activations and the table would measure fewer.
- **Each entry carries the digest of that window.** `verify_histories()` runs before the first call,
  so a candle the exchange revised is an error at second zero rather than a table that silently stops
  comparing with the previous one. `plan_from_manifest()` will not hand back a plan without it.
- **What reproduces a run is the list, not the seed.** The manifest is versioned whole — symbol, bar,
  timestamp, triggers, digest — so re-downloading the history cannot quietly reselect.

`ReplayPlan` is what run and dry-run both consume, built either from a manifest or from a contiguous
history (`plan_from_history`). One code path, so the budget and the table cannot end up describing
different walks. `score_outcomes()` takes histories keyed by symbol for the same reason: with seven
series in play, one `rows` argument would score ETH's order against BTC's candles.

Measured on the committed selection: 140 evaluations, 140 activations, 0 prepare failures, two
dry-runs agreeing on all 140 exact prompt digests, and the decider at **840 calls over the six
arms**, against Go's 880 per window. At one attempt every role fits; retries are not in that count
(see the re-probe). The quota table now carries a "con reintentos" column (decider × 1.2 = 1 008):
against Go's 880 the decider fits at one attempt and **not** with retries, which is what
`fits_with_retries` says and why a Go figure left in a `payg` `.env` shows `**NO**`.

### One ledger for every arm

`run_arm` used to build its own `QuotaLedger`, so six counters each believed they were inside the
budget while the provider saw the sum. That is the same defect the runner had across symbols, back
through a different door, and it stays invisible until the history is big enough to exhaust a
window — which is to say, halfway through a run measured in hours.

The ledger is now built once in `_run` and handed to every arm. Two doors are shut by signature
rather than by discipline, each with a test in `tests/test_architecture.py`:

- **`run_arm` takes the ledger as a required parameter and constructs none.** No default, so
  forgetting it is a type error and not a silent second budget.
- **`QuotaLedger` no longer holds `Settings`.** It takes a window and a clock; `resolve(role,
  choices)` receives the candidates from whoever calls, which is the router holding *that arm's*
  settings. This is what makes one counter serve six different role maps at all: with the config
  inside, a shared ledger would resolve `local_technicals`'s roles through the base map and hand
  back the remote model, so the arm would stop being the arm it claims to be without failing.

What the counter refuses, it refuses loudly: the arm that finds the window empty aborts and the
evaluation still reaches the journal with `cuota agotada` in its errors, which is what distinguishes
"there was no budget left" from "this arm never decided". `tests/test_ablation.py` grants a budget
for three arms, runs six, and pins both halves — the first three decide, the rest abort recorded,
and the aggregate never exceeds what was declared.

So `QuotaLine` in the dry-run is no longer a curiosity nobody can see: it is exactly what the shared
ledger will count, and therefore the precondition for launching. The per-arm quota column is gone —
six rows each saying "fits" against a budget that exists once is the same lie in table form.

### A run that leaves nothing behind cannot be audited

The first complete ablation — six arms, 140 evaluations, a day and a half of wall clock — left a
table and nothing else. `run_arm` defaulted to an `InMemoryJournal` and the command never passed
another, so every `LLMCall` and every `NodeError` died with the process. The table said 31
evaluations of `full` did not decide and there was no way to ask why. It was discarded.

Each piece was tested. The wiring was not: `run_arm` accepted a journal, a test passed one, and
`_run` passed none. Rule 5 covers this case too — "it ran fine" included the journal.

A run now writes `var/ablation/<UTC start>/`:

```
meta.json       plan sha-256, argv, --fill, arms, start, git commit and dirty flag, resumed_from
<arm>.jsonl     one EvaluationRecord per line, written as each evaluation ends
```

Four properties, each with a test:

- **`run_arm` takes the journal as a required parameter**, and `ablation.py` does not name
  `InMemoryJournal`. The architecture test also requires `journal=` on the `AgentContext` that
  `run_arm` builds: the dataclass defaults to an in-memory one, so omitting it would be silent again.
- **`meta.json` is written before the first call.** The run that most needs identifying is the one
  that was interrupted.
- **A directory is never reused.** `run_id` in a replay is a deterministic UUID5, so a second pass
  appended to the same file leaves two lines with one id and no way to tell them apart. A resume
  writes a new directory and points at the old one.
- **`git_dirty` travels with the commit.** With uncommitted changes the hash names code that is not
  the code that ran.

`python -m crypto_agents.audit <dir>` reads it back. The module imports neither the router nor
anything that brings it — not `ablation.py` either, which is why `RunMeta` lives in `audit.py` and
the ablation imports it, not the other way round. Three rules hold in every table:

- **A rate never appears without its two numbers.** The arm's validation failure is invalid over
  answered, summed across pairs — the attempt-weighted mean. The first table published the *maximum*
  over (role, backend) pairs under the heading "fallo validación"; that figure still exists, in its
  own column, called "peor par", with its attempts beside it.
- **Latency is over live calls only**, and an arm served entirely from cache has no latency rather
  than zero. The old column was the arm's wall clock divided by live attempts, so a warm cache made
  it mean nothing.
- **What cannot be computed says `no determinado: <why>`.** An arm with no journal is not an arm
  with zero evaluations.

The ablation's own table renders the same three columns through the same functions, so the table
and the audit of its directory cannot disagree about what a rate is.

The attempts table breaks failures down by `FailureKind`, always all four, zeros included.

### The comparison table carries its uncertainty

`render_report()` is five tables, one per question: calls and cost, decider to order, evaluations
without a decision, order results, and what each arm asks. Everything in it is read from the pure
functions in `metrics.py`; the table computes nothing. Three rules, each with a test in
`tests/test_ablation.py`:

- **A cache hit is a call, not quota and not latency.** `llamadas` counts every `LLMCall` row;
  `cuota remota`/`cuota local` and the four latency figures (n, mean, median, p95) read live calls
  only. The test feeds seven rows, three of them hits, and fails if any of the three cells counts one.
- **No fraction without its two numbers.** Every column that is a part of a whole renders `k/n (p%)`
  or `no determinado: <why>` — funnel stages each over the previous one, `invalidadas` over resolved
  orders, validation over answered. A structural test walks the table and checks the shape of every
  such cell, so dropping a denominator from one fails it.
- **Hit rate and mean return carry their error.** Hits are `k/n` with a Wilson 95% interval
  (`wilson_interval`); the return is `mean ± standard error, n=` with two decimals of percentage —
  `.0%` turned +0.4% into "0%" — and with a single order the error says `no determinado`, not zero.
  `return_stats` uses the sample deviation (n - 1).

Two coincidences with `full` sit side by side because they answer different questions. `agreement`
counts a shared "did not decide" as agreeing and is unchanged; `agreement_decided` keeps only the
evaluations where both arms decided and says how many that is. `OutcomeStats` keeps the total and
not the individual returns, so the standard error needs `outcomes.resolved_returns()`, which repeats
`score_outcomes`' filter instead of touching it; `tests/test_outcomes.py` ties the two together.

### Baselines, one common stop and three scorings

An arm compared with nothing proves nothing, and its result mixes direction with the stop each
decider declares. Four `PipelineVariant`s that call no model — `always_buy`, `always_sell`,
`random_uniform`, `rule_trend` — are appended to `ARMS` (indices 0–5 unchanged), and
`outcomes.py` scores every arm three ways. Rules, each with a test:

- **A baseline costs zero by construction.** `baselines.py` and `stops.py` cannot import the router,
  graph, nodes, replay or quota (`test_architecture.py`), and the arm test runs cache-only on an empty
  cache: one model call would raise `ReplayCacheMissError`. Their nodes are in `DETERMINISTIC_NODES`,
  so the dry-run prices them at nothing instead of failing on an undeclared node.
- **The deterministic tail does not change.** Every variant has the same `risk → execute → journal`
  edges and only a decider reaches `risk` (`test_the_deterministic_tail_is_the_same_in_every_variant`).
- **One function builds the stop.** The baselines declare it and the scoring rebuilds it;
  `COMMON_STOP_ATR_MULTIPLE` may be named only in `stops.py`. For a baseline `OWN_STOP == COMMON_STOP`
  exactly, which breaks the moment the two formulas drift. It is ATR and not a fixed percentage
  because the run jumps across seven symbols; that is why `EvaluationRecord` now carries
  `indicators` (also for `solo` and the baselines, which produce no `evidence`), and
  `RUN_DIGEST_VERSION` went to `replay-v3`. A journal without the field still loads; `COMMON_STOP`
  counts those records as `unscorable` instead of inventing an ATR.
- **Declared stop untouched.** `score_record`, `score_outcomes` and `resolved_returns` are as they
  were; `OWN_STOP` delegates to the first. `COMMON_STOP` and `HORIZON_CLOSE` score the *proposal*
  (buy or sell, entry at `snapshot.close`), not the order, so a stop on the wrong side — which the gate
  vetoes — cannot filter which directions get measured.
- **Frozen before results.** `TREND_ADX_MIN = 25.0`, `COMMON_STOP_ATR_MULTIPLE = 2.0`,
  `BASELINE_SIZE = 0.05` and `BASELINE_CONFIDENCE = 0.5` are conventions, pinned by tests, and
  `rule_trend` is written down in `docs/ablation.md` with its date. Changing one after looking at a
  table is fitting the rule to the data.
- **No look-ahead.** The rule and the stop are checked on the real history truncated on the right,
  like `test_lookahead`; the scoring is checked by cutting the series right after `i + horizon` and
  by replacing everything past it with impossible candles.
- **The random line is reproducible.** `random_uniform` draws from `sha256(seed|run_id)`, not
  `hash()`; the seed is the manifest's, carried by `ReplayPlan.seed` into `AgentContext.seed`.

`metrics.paired_difference` compares an arm with `solo` evaluation by evaluation: mean, the
standard error of the differences and a normal 95% interval, `None` below n = 30. About 27 such
differences are published with no multiplicity correction, and the report says one or two will exclude
zero by chance.

Abort causes are grouped by node and by a closed `AbortKind`, read from the message text because
`NodeError` carries nothing else. `tests/test_metrics.py` produces each message with the real
exception or the real graph, so rewording one breaks a test instead of sending the cause to `other`.

### Resuming is not a second copy of the first pass

Two things differ, and both are now visible:

- **The ledger is seeded.** `QuotaLedger` lives in memory; the provider's window does not.
  `--resume-from <dir>` walks the chain of `resumed_from` and calls `ledger.seed()` with every call
  found, keeping the remote ones whose `at` is still inside the window. It refuses a directory that
  ran a different plan: other evaluations are not a resume. `crypto-agents run` does the same from
  the configured journal at startup, and refuses to start on a journal it cannot read — starting
  unseeded is starting with a budget the gateway does not agree with.
- **Only provider failures get a second draw.** Every attempt that produced content is cached, the
  invalid ones too, so a resume replays a content failure from the cache: same error, same abort,
  nothing spent. What is called live again is what has no entry — an evaluation that died on a
  transport rejection, a timeout or an exhausted window. The audit reports per arm how many were
  undecided before and decided now, which is that number.

The seed is a **lower bound**. It knows the journal it is given: what `doctor` or another command
spent against the same provider is not in that file.

Two clocks meet in one record under `--fill`: `LLMCall.at` is wall time — the router's clock, which
is what makes seeding possible — while `NodeError.at` and the record's own `at` are the evaluated
instant.

### The ablation vetoes for one reason only

With `NOTIONAL_ACCOUNT` three of the four vetoes are unreachable: drawdown is zero against a limit
above zero, there is no `last_loss_at` for the cooldown, and the context is built with the default
`StaticKillSwitch(False)` and never consults `var/STOP`. With zero open exposure a non-zero size
always leaves headroom, so the cap never turns into a veto either. The account is one frozen object
for all 140 evaluations; nothing updates it.

So `orders == buy + sell` was an identity in the first table, and it hid a defect: nothing between
the decider and the market looked at which side of the close the `invalidation_price` sat on. A
`buy` invalidated *above* its entry is "stopped out" on the first candle that touches that price —
at a profit — and `outcomes.py` scored it as a win.

Three layers now, one definition (`state.stop_on_wrong_side`: `>=` the reference on a buy, `<=` on a
sell, equality included because a stop at the entry leaves no room):

- **`invalid_stop_side` is the first veto in `VETOES`**, judged against `snapshot.close` — the same
  price `build_order` writes as `reference_price`. It runs before the kill switch on purpose: the
  other rules describe the state of the system, this one describes the proposal, and evaluated first
  an incoherent proposal is journaled as such even when nothing would have traded anyway. The
  consequence is that with the switch engaged, wrong-sided proposals group under their own rule and
  not under `kill_switch`, and the repeated-veto alert — which excludes only `kill_switch` — can fire
  while the system is stopped.
- **`OrderIntent` refuses to be built with the stop on the wrong side.** The veto is the normal path
  and leaves a record; this is what remains if some future path reaches `build_order` without
  passing the gate. It also means a journal line carrying such an order no longer loads.
- **`score_record` raises `OutcomeError`** rather than invent a result. Unreachable by construction
  — the test builds the order with `model_construct` — so if it fires it is a bug, and it surfaces
  as a traceback, not as a row in the table.

`apply_risk()` takes the reference price as a required argument. With a default, a caller that
forgot it would stop checking the side without anything failing, which is how it was before the rule
existed. The audit's `wrong_side` count over proposals uses the same function, so it equals the
number of `invalid_stop_side` vetoes.

### The cache does not cross from one model to another

The router used to look up every candidate of a role and return the first entry it found. With the
primary's entry missing and the fallback's present, the role received the fallback's answer as a
cache hit — recorded under the fallback's backend, costing no quota. In the ablation that is `full`
serving what an 8B local model answered to `local_technicals`, with nothing in the table to say so.
The existing test ran the arms remote-then-local, the one order in which it cannot happen.

One key, one model, and the backend is part of the key:

- **The primary's key is read first, without asking the ledger.** A hit spends nothing, and
  `resolve()` raises when the window is empty — a warm rerun would fail on quota without going to
  call anybody.
- **The fallback's key is read only when the ledger actually degraded on that attempt.** That is the
  model that was going to answer anyway.
- **What is stored is a `CacheEntry`**: backend, model, role, prompt digest, schema, mode and the raw
  text. The key is a sha-256; without the envelope, 2 106 files from the first run could not be
  attributed to any role or model. On read the envelope must match the question asked — a file
  copied under another key is discarded, not used.

One consequence is accepted: **an evaluation that ran degraded during `--fill` is not replayable
cache-only.** The replay starts with a clean ledger, resolves the primary, does not find it and
raises `ReplayCacheMissError` naming the model. It used to replay thanks to the cross-read this
removes. Unlikely in the ablation — the decider has no fallback and the other windows are in the
thousands — possible in a replay of the runner.

### Every attempt is in the cache, so a replay finishes

The cache used to hold only answers that validated. The retry went to a key that included the
error text of an attempt that was never stored, so the second half of the conversation could not be
asked of the cache: a cache-only replay of any evaluation that had needed one retry raised
`ReplayCacheMissError`, and under `--fill` it paid for the failing call again.

Now each attempt that produced content is stored under the digest of its own prompt, with its raw
text and how it was judged. Read back, the text is validated again — schema and context — and three
things can happen:

- **It passes.** It is the answer.
- **It fails and was stored as invalid.** A replayed attempt: the same error, therefore the same
  retry prompt and the same digest, where the next entry is waiting. It is journaled with
  `cache_hit=True` and `valid=False`, and spends no quota.
- **It fails and was stored as valid.** The schema changed since. A stale entry: discarded under
  `--fill`, left alone in a replay, and a miss in both.

Three properties, each with a test:

- **The key did not change.** A first-try answer is stored exactly where it was; the invalid attempt
  and its retry each sit under their own prompt digest. No attempt number in the key.
- **A replay does not modify the cache.** Without `fill_with`, `replay_router()` hands the router a
  `ReadOnlyResponseCache`. The router discards stale entries, and a replay that deletes what it
  reads cannot be repeated over the same thing.
- **Provider failures are not cached.** There is no text to store, and a cached 503 would be a 503
  forever. An evaluation that died on one is the case a replay still cannot reproduce.

The chain depends on the error text. If a validator's message changes — a Pydantic upgrade can do
it — the retry prompt is another prompt and the chain breaks at that point, loudly.

`run_digest()` is tagged `replay-v2`: the serialised shape of every `LLMCall` changed with the flat
failure fields, so a digest computed before and one computed now differ regardless of decisions.

## Pay as you go (Zen)

Support said the batch evaluation must not run on the Go subscription but on OpenCode Zen, pay as you
go (`https://opencode.ai/zen/v1`, `/chat/completions`). In `.env` that is `CA_OPENAI__BASE_URL` and
`CA_BILLING=payg`. `zen_probe` refuses anything else before it builds a backend, and `meta.json`
records the endpoint (`RunMeta.base_url`: scheme, host[:port] and path through `settings.public_url`,
never credentials or query — a validator on `RunMeta` enforces it even if a caller forgets).

- **Prices** live in `settings.py`, dated per page (`PriceTable.as_of(billing)`): Go 2026-10-02, Zen
  2026-10-04. The four `payg` rows that were already there matched the Zen page and did not change; six
  candidates for structure/volume were added. `qwen3.8-max` has a cached-write price (2.50 USD/Mtok),
  modelled since block T3 as `PriceRow.cache_write`: the provider's `usage` carries a
  `cache_write_tokens` counter that `LLMCall` does not keep, so that model's cost is an interval, not
  a figure (see "What pay as you go must not pretend to measure"). `kimi-k3` was added in T3.
- **Quotas.** Zen publishes no request limit (its pricing page lists none). A Go figure left on a
  `payg` role makes the ledger degrade a remote role to the local model, or abort the decider, for a
  limit the provider does not impose. Remote roles declare `ZEN_UNPUBLISHED_QUOTA = 100 000` and weight
  1.0 — a sentinel with the same status as the 10 000 of the local fallbacks, ~100× the decider's
  worst case (840 calls, 1 008 with `DECIDER_ATTEMPTS = 1.2`). Local fallbacks and the forced-local
  arms are untouched. `tests/test_zen_payg.py` pins it over the real manifest: every remote role fits
  with room, the ledger does not degrade any, and the decider left at 880 fails. `consumption
  --quotas` under `payg` says "no publicado por Zen" instead of comparing with Go's estimates.
- **Top-up.** The page says "4.4% + 0.30 USD per transaction"; `PriceTable.topup_charge` reads it as
  `credit × 1.044 + 0.30`. It is a reading, not a receipt: the first real top-up says whether it is
  that.

### What pay as you go must not pretend to measure (block T3)

The sentinel quota and the dollar cap change what several reports can honestly say. Each rule has a
test, and the first three carry a mutation test (the production source rewritten with the bug in,
the hand-computed check required to fail).

- **No percentage against the sentinel.** Under `billing = payg` the `quota_low` alert is silent and
  `crypto-agents alerts` prints `cuota: no aplica (payg)`; `criteria`'s 5 h peak table keeps the
  measured peaks (calls, summed weights) and prints `no aplica (payg)` in the share column, with no
  `AVISO`. A run that did not record its billing (`meta.billing is None`) keeps the old comparison.
- **Running out of balance is an invalid run.** The ledger no longer brakes anything, so the cap is the
  balance, and a `402 Insufficient account funds` used to be one more `transport` failure that lost the
  evaluation quietly. `AbortKind.INSUFFICIENT_FUNDS` reads it from the message (the production test
  builds it with the real `openai.APIStatusError`, body captured on 2026-10-05); `LLMCall.failure_kind`
  stays `transport` because `FailureKind` is closed. `criteria` counts it at **any** node and arm (the
  balance belongs to the account): `CORRIDA INVÁLIDA`, exit 1, no verdicts. This is a fourth
  condition of criterion 6 in code; the text of amendment 2 still lists three and was not touched.
  `alerts` reports it whatever the billing.
- **A probe is not a run.** `RunMeta.kind`; `criteria` refuses any link of the chain that is a probe.
- **Cache write is an interval.** `PriceRow.cache_write` (payg only, never below the input price,
  `qwen3.8-max` 2.50 on the page of 2026-10-04). The provider reports cache *reads* but not *writes*,
  so the cost of those calls is `[no write charged, non-cached prompt at cache_write]`, marked `†`.
  `cached_tokens: null` (DeepSeek V4 Flash) is the same kind of interval, from all-cached to all-new:
  it is a bound, not an estimate, and it is what lets `estimate` give momentum a cost range.
- **`LAUNCH_MARGIN = 1.5`** lives in `ablation.py` beside `DECIDER_ATTEMPTS`. `estimate --balance X`
  passes only if `X >= cost x 1.5` at the high end of the range with the decider at 1.2 attempts; an
  undetermined cost never passes. `X` is what the console shows: nothing queries the network.
- **The desks probe's cost bound is a cap, not an estimate.** The repo sets no `max_tokens` and the
  desks' prompts depend on verdicts that do not exist before the technicals answer, so no bound can
  be computed before calling without estimating tokens (forbidden). `--max-usd` is mandatory; the
  guard sums the high end of what the provider declared and refuses to open a new invocation at the
  cap. It can overshoot by the invocations already in flight, and calls without tokens do not count
  towards it; the report says both.

### Probes against Zen (desktop; directories under `var/zen-probe/`, not versioned)

Three runs, each a new directory. **Without balance (02:51Z, `20261005T025129Z`)** nothing could be
measured: the five ids absent from `GET /models` answered `403 Model access is disabled` and the five
listed ones `402 Insufficient account funds`. **With balance (04:14Z, `20261005T041459Z`)** three
candidates answered and the three disabled models of the role map stayed disabled. **With those models
enabled (04:33Z, `20261005T043359Z`, tree clean at `e3dfc31`, ≈ 0.15 USD)**, the one this section
tabulates:

| answer | ids |
| --- | --- |
| responds in the declared mode | `glm-5.2`, `minimax-m3`, `deepseek-v4-flash` (`Ping`; momentum also 12 verdicts), `deepseek-v4.1-flash`, `deepseek-v4-pro`, `glm-5.3-flash` (12 verdicts per dimension) |
| `410 Upstream request failed: Endpoint is unavailable.` | `kimi-k2.6`, `kimi-k2.7-code` (both moonshot, both listed in `/models`; `kimi-k2.7-code` also 410 in the 04:14Z run) |
| absent from `/models` → `403 Model access is disabled` | `minimax-m2.7`, `qwen3.8-max` |

Real technical prompts, 12 verdicts per arm through `ModelRouter` (no cache, no fallback,
`max_attempts=2`). Every arm: 12/12 valid, 0 retries, 0 failures of any `FailureKind`. Latency is the
mean over live calls **on the desktop**; tokens are the means the provider reported (prompt / cached /
completion):

| arm | latency | tokens | USD per verdict |
| --- | ---: | --- | ---: |
| `deepseek-v4.1-flash` structure | 17.4 s | 1072 / 472 / 976 | 0.00135 |
| `deepseek-v4.1-flash` volume | 16.7 s | 980 / 423 / 1169 | 0.00157 |
| `deepseek-v4-pro` structure | 12.2 s | 521 / 0 / 473 | 0.00255 |
| `deepseek-v4-pro` volume | 13.8 s | 521 / 0 / 454 | 0.00249 |
| `glm-5.3-flash` structure | 20.9 s | 525 / 0 / 1884 | ≥ 0.00102 (7 of 12 measured) |
| `glm-5.3-flash` volume | 31.2 s | 524 / 0 / 2983 | ≥ 0.00157 (10 of 12 measured) |
| `deepseek-v4-flash` momentum | 7.8 s | 604 / — / 2822 | undetermined; ≤ 0.00088 (see below) |

- **The docs page is not the catalog, and the catalog is not availability.** The Zen docs page lists
  all ten ids (`mimo-v2.5` only as `mimo-v2.5-free`, `hy3` not at all). `/models` listed 44 ids and
  grew to 47 when the three models were enabled, so the listing followed the workspace's model access.
  But a listed id is not a serving one: both moonshot ids are listed and answer 410, the same
  "endpoint unavailable" shape as Go's qwen 503. One account, one day.
- **`x-opencode-session` is not required.** With and without it `deepseek-v4.1-flash` answers the
  same `Ping` (and, before the balance, the same 402). The adapter keeps sending it: it costs nothing
  and Go does require it.
- **Some responses carry no `usage` at all.** 7 of 24 `glm-5.3-flash` calls came back valid with
  `prompt`, `cached` and `completion` tokens all `None` (0 of 24 in the 04:14Z run), so their cost is
  unknown and that model's figures are lower bounds. Why is not determined.
- **`deepseek-v4-flash` reports `cached_tokens: null` on 12 of 12** (as on the 2026-10-03 fixture), so
  momentum has no exact cost, only the ceiling `consumption` prints: 0.0105 USD for the 12 calls, i.e.
  ≤ 0.00088 each and ≤ 0.12 for the 140 paid ones.
- **The same prompt is counted twice as large by one model.** `prompt_tokens` was 1072 for
  `deepseek-v4.1-flash` and 521 for `deepseek-v4-pro` over the same 12 structure prompts. Costs are
  computed from what each provider reported, so the figures above are what Zen would bill.
- **What is still undetermined, and why.** The chained stage did not run: `kimi-k2.6` (bull) is
  unavailable, and the stage needs bull, bear and decider to build the decider's real prompt, so bull,
  bear and decider have no measured cost and the stage-1 total and the top-up stay `no determinado`.
  Determined, over 140 paid calls each: structure 0.14 to 0.36 USD and volume 0.22 to 0.35 USD between
  the cheapest and the dearest candidate that answered (the low end is `glm-5.3-flash`, a lower
  bound). The `--dry-run` counts over the manifest are 140 paid calls each for structure, momentum and
  volume, 420 for bull and for bear, 840 for the decider.

Re-run `python -m crypto_agents.zen_probe --machine desktop` when bull answers (or the role map points
at a bull model that does); it writes a new directory and `python -m crypto_agents.estimate <dir>`
prices the stage. The probe's `Ping` and verdict calls are recorded like any other (rule 4).

### Block T3 probes against Zen (desktop; `var/zen-probe/`, not versioned)

Two runs of `zen_probe --desks --technicals-from var/zen-probe/20261005T043359Z --max-usd 2`, tree
clean at `d07334e`, manifest `data/ablation_selection.json` sha-256
`73f87870cf24d06c341ff75fa72f164f9cd5202623cf0d76babbdab4a2803f23`, both on the desktop: `20261005T062628Z`
(06:26Z) and `20261005T063458Z` (06:34Z). The cap's guard saw 0.0066 and 0.0294 USD at the high end and
refused nothing. The second run repeated the first after its first failure; it was authorised under the
same cap, and the first one had cost under a cent.

**The desks were not measured.** `deepseek-v4-flash` — the momentum producer the task fixed —
answered the mode `Ping` with `404 {'status': 404, 'message': 'Cannot find any route matching [POST]
https://opencode.ai/zen/v1/chat/completions'}` in both runs (06:26:30Z and ~06:35Z), after answering
at 04:33Z. Without a momentum verdict there is no common evidence, so all 12 activations were skipped
for every desk and **bull, bear and the decider have no rows**. Everything that depends on them —
valid/12 per bull, the conditioned decider, tokens, USD, the per-bull totals, the top-up and the
`--balance` verdict (no balance was given either) — is `no determinado`, not zero. Choosing another
momentum producer is choosing a model, which the task forbade; the choice was left to the operator,
who closed the block with partial results. Rerun when the route is back (a new directory).

What did answer — mode `Ping`, one attempt each, run `20261005T063458Z` (`Ping` rows sha-256 in the
table below; latency on the desktop):

| id | role in the probe | mode confirmed | tokens prompt / cached / completion | USD |
| --- | --- | --- | --- | --- |
| `kimi-k2.6` | bull | **none**: `410 {'error': {'type': 'server_error', 'message': 'Upstream request failed: Endpoint is unavailable.'}}` | — | 0 (no answer) |
| `kimi-k3` | bull | `json_schema` (2.5 s) | 270 / 256 / 67 | 0.0011 |
| `qwen3.8-max` | bull | `json_schema` (1.3 s) | 68 / 67 / 54 | 0.0003 to 0.0003† |
| `deepseek-v4-pro` | bull | `json_schema` (1.4 s) | 21 / 0 / 6 | 0.0001 |
| `minimax-m3` | bear | `json_mode` (0.9 s) | 144 / 143 / 7 | 0.0000 |
| `glm-5.2` | decider | `function_calling` (8.7 s) | 232 / 231 / 524 | 0.0024 |
| `glm-5.3-flash` | structure, volume | `json_schema` (2.7 s) | 29 / 0 / 87 | 0.0000 |
| `deepseek-v4-flash` | momentum | **none**: the 404 above (548 ms) | — | 0 (no answer) |

† interval for cache write; at that size both ends round to the same four decimals.

The 410 and the 404 are different things and stay in separate rows of the report: a model the gateway
lists and does not serve (`kimi-k2.6`, which also answered 410 in `20261005T043359Z`), and a route the gateway could not
find for an id that served the same endpoint two hours earlier. A mode `Ping` is one attempt with no
retry, by design, so neither says whether `deepseek-v4-flash` is gone or flapping.

The technical producers did run — once per activation, as the desks would have read them. The rule
picks `glm-5.3-flash` for structure and volume (12 valid verdicts each in `20261005T043359Z`, tied with
two others; the cheapest output price breaks the tie). Per (model, role), valid / attempts / retries per
verdict, latency on the desktop, tokens (prompt / cached / completion) are the means the provider
reported:

| run | model · role | valid | schema · context · timeout · transport | attempts · retries/verdict | latency | tokens | USD |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `…062628Z` | `glm-5.3-flash` structure, `json_schema` | 12/12 | 0 · 0 · 0 · 0 | 12 · 0.00 | 23.5 s | 520 / 0 / 1578 | ≥ 0.0017 (10 of 12 without usage) |
| `…062628Z` | `glm-5.3-flash` volume, `json_schema` | 12/12 | 1 · 0 · 0 · 0 | 13 · 0.08 | 36.4 s | 520 / 0 / 1491 | ≥ 0.0016 (11 of 13 without usage) |
| `…063458Z` | `glm-5.3-flash` structure, `json_schema` | 12/12 | 0 · 1 · 0 · 0 | 13 · 0.08 | 18.2 s | 529 / 0 / 1864 | ≥ 0.0111 (2 of 13 without usage) |
| `…063458Z` | `glm-5.3-flash` volume, `json_schema` | 11/12 | 0 · 0 · 1 · 0 | 12 · 0.00 | 34.7 s | 525 / 0 / 2707 | ≥ 0.0143 (1 of 12 without usage; 1 timeout) |

`context` is a desk answering as the other one or citing evidence nobody emitted: the model's fault, not
the provider's, and it is its own column; here the single `context` failure is a *technical* verdict,
not a desk. The timeout lost its activation (a timeout is not retried). **`glm-5.3-flash` returned no
`usage` on 21 of 25 calls in the first run and on 3 of 25 in the second**: absent usage is not a
property of the model id, and every USD above is a lower bound for that reason. Sources — the
`LLMCall` rows are the lines of these files, and `python -m crypto_agents.consumption <dir>` recomputes
each figure:

| file | sha-256 |
| --- | --- |
| `20261005T062628Z/glm-5.3-flash@structure.jsonl` (12 rows) | `fb24eb3b2e3af1b0d4714eb41682eb53342a4b50e3d1be4c397553bd40bb130f` |
| `20261005T062628Z/glm-5.3-flash@volume.jsonl` (13 rows) | `300a374b1db5bf8fbf09da34cc385de1b9759847e43915ac07e69c2f08c926b5` |
| `20261005T063458Z/glm-5.3-flash@structure.jsonl` (13 rows) | `9c2a2ea3b8ea7a66fc6e471c134b2abab5056c8f5eac663fbb9693f009d8153a` |
| `20261005T063458Z/glm-5.3-flash@volume.jsonl` (12 rows) | `88232dc4d97cc97e63b9030f64b05f40155c0afa877cbe6ecdb99e8bd244a73b` |
| `20261005T062628Z/deepseek-v4-flash@ping.jsonl` (404) | `12e34248ac1ed205e6c7aef2037046ab2f0663c1e3ede8086444c07e58752a7c` |
| `20261005T063458Z/deepseek-v4-flash@ping.jsonl` (404) | `63d4b096a491b546c792a30e337572d069c3c8ac81c8bd5154411ee820d55357` |
| `20261005T063458Z/kimi-k2.6@ping.jsonl` (410) | `1526015f99055529b637ef61859077dd18e195a8f87cd09a62ddd1efbe45b34b` |
| `20261005T063458Z/kimi-k3@ping.jsonl` | `939a0a3e18b3cb6abe9fab24a93cdad231869c281d7f53a947d3ed3ab7ac4847` |
| `20261005T063458Z/qwen3.8-max@ping.jsonl` | `1381f831960ff1a524fd8468bed47e9aa977cddf6acc85480b97e8793e4da919` |
| `20261005T063458Z/deepseek-v4-pro@ping.jsonl` | `44a999f01fbe5c0a20fdcd7a8df18724a7c4051608984c8e8c687bb11142f109` |
| `20261005T063458Z/minimax-m3@ping.jsonl` | `4cf68aba2f40852a10bff80d8c7a2c0f97e9280363230cb8f424c387152ab252` |
| `20261005T063458Z/glm-5.2@ping.jsonl` | `f85a6c7e9536dcbf9dda79885f5a01b12d6dc747a3d2bdf805f20b71573dba5f` |
| `20261005T063458Z/glm-5.3-flash@ping.jsonl` | `bc4a93ae8a5aad7c20e80dbc4dfbf68693087b39f1828ea7601142fafbf97ec0` |

Families of the bull candidates against the bear (`minimax-m3`, minimax) and the decider (`glm-5.2`,
zhipu), from the code that will run them: `kimi-k2.6` and `kimi-k3` moonshot, `qwen3.8-max` qwen,
`deepseek-v4-pro` deepseek — **all four disjoint from both**, so the hard constraint `bull != bear`
holds for any of them. Outside that constraint, `deepseek-v4-pro` shares a family with `momentum`
(`deepseek-v4-flash`), and `qwen3.8-max` with the local fallback of `bull` (`qwen3:8b`).

**Why the same prompt counts 1072 prompt tokens for one model and 521 for another.** Both used
`json_schema`, and `LLMCall.prompt_digest` — the digest of the prompt text alone, schema not included —
is identical row by row. In `20261005T043359Z`, `deepseek-v4.1-flash@structure.jsonl`
(`ff2d2a5a…01c6`) against `deepseek-v4-pro@structure.jsonl` (`6d33be00…d46d87`): `prompt_tokens`
1071/520, 1070/519, 1067/516, 1068/517, … the difference is **551 in 12 of 12 rows**, whatever the prompt.
In `…volume.jsonl` (`0a47693d…c8383` against `ba19149b…c762`) it is 551 in 10 of 12 rows and **0 in the
last two** (518 against 518, 519 against 519). In the `Ping` the difference is 152 (173 against 21) for
a 405-character schema, against 551 for the 1 625-character `TechnicalVerdict` schema. Reading: the
extra tokens are a fixed block that grows with the response schema, not with the prompt — the
`response_format` counted as input by one serving path and not by the other — and it is not a property
of the model id: the last two `deepseek-v4.1-flash` volume calls took 4.2 s and 3.0 s against ~18 s,
produced 453 and 530 completion tokens against 929 to 1 828, cached 0, and counted like
`deepseek-v4-pro`. The cached counts fit too: `deepseek-v4.1-flash` reports 438 to 526 cached tokens
on warm calls, a stable prefix about the size of that block, while `deepseek-v4-pro` reports 0. What
the rows cannot say is whether Zen **bills** those tokens: costs are computed from what the provider
reported, which is not the invoice. The first top-up will say.

**Estimate** (`python -m crypto_agents.estimate var/zen-probe/20261005T043359Z [--desks …063458Z]`; every
line labelled an estimate; the measured cost per call is the mean of the rows of the cited files).
Determined, over 140 paid calls each: structure 0.00102 to 0.00255 per call = **0.14 to 0.36 USD**
(`glm-5.3-flash` to `deepseek-v4-pro`; the low end is a lower bound, 5 of 12 calls without usage);
volume 0.00157 to 0.00249 = **0.22 to 0.35 USD**; momentum `deepseek-v4-flash` **0.11 to 0.12 USD**
(0.00081 to 0.00087 per call) — an interval, since every call reports `cached_tokens: null`: the low
end charges the whole prompt at the cache price, the high end all of it as new, which the
2026-10-03 and T figures could only call "≤ 0.00088". Not determined: bull (420 calls), bear (420) and
the decider (840 at 1.2 attempts), hence every per-arm total, the stage-1 total, the top-up and the
balance check. With `--desks`, each of the four bull candidates gets its line and every line says
`ningún alegato válido`. Every figure above comes from these files (full digests; the manifest is the
one cited at the top):

| file in `var/zen-probe/20261005T043359Z/` | sha-256 |
| --- | --- |
| `deepseek-v4-flash@momentum.jsonl` | `7fc89f90bf36bc019024a12db0d003865be2e1016001128a1049a67600cc1659` |
| `deepseek-v4.1-flash@structure.jsonl` | `ff2d2a5a084427721e3df0cfdaeccbb9f7f1a1747ed69c2a28c5ac38185c01c6` |
| `deepseek-v4.1-flash@volume.jsonl` | `0a47693dcf9f502e4ad63564ab4070e57480de8e1ccc07c0bc56c943f8edc383` |
| `deepseek-v4-pro@structure.jsonl` | `6d33be009e60aca4774270df21ff4e573db2b2035fe7a894370f55ed77d46d87` |
| `deepseek-v4-pro@volume.jsonl` | `ba19149b80c12107b1a5c7c1615c6c657fddf99b71b9101faaee7fccea45c762` |
| `glm-5.3-flash@structure.jsonl` | `9adfa6504fb57ea62ff3a24a9216d8f27019ad69b677f86b43d38d27aa7b72e2` |
| `glm-5.3-flash@volume.jsonl` | `fb35858bbefc967a13791acd78f31fa4818146455db1b2c226970643bf545c55` |

`docs/ablation.md`, "Resultados" (not edited): it is the placeholder of the **second** run, not the
first run's table (that one is further down, discarded), and its text is **Go's**: "los 880 del decisor
son un ritmo por ventana de 5 h", the subscription's shared monthly limit, and the 2026-08-15 re-probe
of the six Go models. It says nothing about Zen, pay as you go or a stage-1 screen.

## Gotchas found the hard way

- **A provider client's default timeout is not a decision anybody made, and both defaults are
  unusable.** Measured against the installed versions: `ChatOpenAI(...)` without `timeout` resolves
  to the OpenAI SDK's `Timeout(connect=5, read=600, write=600, pool=600)`, so one hung role holds ten
  minutes per attempt against an evaluation that costs ~90 s complete; and `AsyncClient(host=...)`
  gives httpx `Timeout(None)`, meaning a wedged Ollama never returns at all and the run hangs with
  nothing to read. Both are now declared — `CA_OPENAI__TIMEOUT_SECONDS`, `CA_OLLAMA__TIMEOUT_SECONDS`
  — and `tests/test_architecture.py` parses `llm.py` to refuse any provider client built without one.
- **`max_retries` defaults to 2 in the OpenAI SDK, which breaks rule 4 where nothing can see it.**
  One `complete()` becomes up to three billed requests and exactly one `LLMCall`. The other two exist
  only on the invoice. Retrying belongs to the router, which attaches the validation error to the
  prompt and writes a row per attempt, so the client is now built with `max_retries=0` and the same
  architecture test enforces it.
- **OpenCode Go rejects any call without `x-opencode-session`.** After five idle weeks every role
  failed `doctor`'s `modes` check as a transport rejection, while the catalog still listed 6/6 ids.
  The body was `400 MissingSessionID`, and `doctor` does not print it: only a raw request showed it.
  The gateway asks for a stable id per conversation. `OpenAIBackend` generates one per backend
  (i.e. per process) and sends it from both the chat client and the catalog probe. Carrying it per
  evaluation would change `ChatBackend.complete()`'s contract. The cache key ignores it, so replay
  is unaffected. Its docs also describe the plan as meant for coding agents, which this is not.
- **Testnet answers perfectly and is still not a data source.** It returns 58 4h candles against a
  preset that needs 400, so with `sandbox` governing the read no evaluation could ever complete. The
  fix was structural, not a longer request: `CcxtMarketClient` takes an `exchange_id` and has no
  parameter through which `sandbox` or a credential could arrive. `CA_EXCHANGE__SANDBOX` now governs
  only where orders go.
- **With the keys set, ccxt signs the public endpoints too.** Production answers `-2008 Invalid
  Api-Key ID` to a candle read that returns 500 rows without them. Reading the market and placing
  orders are two clients precisely because only one of them wants credentials, and the guarantee is
  the signature rather than the discipline of not passing them.
- **The structured-output mode is a property of the model, not of the client.** A model that does not
  support the declared mode does not fail: it answers through a channel the adapter is not reading,
  and the symptom is an empty response that gets blamed on the model. glm-5.2's empty answer on the
  first real run was `json_schema`, not glm-5.2. Which is also why the first table of six models is
  contaminated: three of those failures were `ValidationError` raised inside `complete()`, where the
  raw text never reached the router and the documented retry never ran. `crypto-agents doctor` asks
  each remote model with the real `Ping` schema instead of trusting the declaration.
- **A model's JSON often arrives wrapped, and the wrapper is what reaches the validator.** Two
  shapes, both measured against the gateway: minimax-m3 prefixes a `<think>…</think>` block of
  thousands of characters in all three modes, and mimo-v2.5 fences its answer in ```` ```json ````
  under `function_calling`. The verdict inside is correct; what pydantic sees starts with `<` or
  with a backtick and dies at column 1, and the retry attaches an error describing nothing the model
  did — so it reasons out loud again and the second call buys the same result. `json_payload()`
  strips both, but only when the text does not already parse and only when the extraction does:
  editing something that already validates is the one failure this function must never cause, since
  the result would still be valid JSON with altered content.
- **`with_structured_output` defaults to `method="function_calling"`, so the model's answer is not
  in `content`.** It arrives as a tool call, and `raw.content` is `""`. Reading only `content`
  returned an empty string for every call, which the router then dutifully retried against a
  validation error listing every field as missing — a correction that describes nothing the model
  did. The raw output lives in `raw.tool_calls[0]["args"]`, or in `raw.invalid_tool_calls[0]["args"]`
  when it was not even JSON, and `raw_text()` reads them in that order. `include_raw=True` does not
  help by itself: it wraps the parser in `with_fallbacks(exception_key="parsing_error")`, so a
  parsing failure is captured into a key nobody was reading.
- **LangGraph resolves node annotations at runtime** via `get_type_hints`. Types used in node
  signatures cannot live under `TYPE_CHECKING`, or the graph fails with `NameError`. This is why
  `AgentContext` lives in `context.py` rather than `graph.py` — `graph.py` imports the nodes, so the
  other direction would be circular.
- **The graph is async end to end.** Use `ainvoke`; `invoke` raises
  `No synchronous function provided to "prepare"`.
- **`ainvoke` returns only the channels some node wrote.** On an aborted path `result["decision"]`
  raises `KeyError` rather than returning `None`. Revalidate through `TradingState` to get defaults.
- **ruff does not follow inheritance across modules.** `crypto_agents.state.FrozenModel` is listed in
  `runtime-evaluated-base-classes`; without it, field types of models inheriting it get moved into
  `TYPE_CHECKING` and Pydantic cannot build the class.
- **A perfectly flat candle series produces `ADX = NaN`** (no directional movement), which
  `IndicatorSet` rejects. Use `drifting_closes()` for a quiet-but-valid baseline in tests;
  `flat_closes()` is only safe when the test stops before indicators.
- **Parallel fan-out without a reducer raises** `InvalidUpdateError` in langgraph 1.2. The silent
  loss happens in a *sequential* chain, which keeps only the last write. Both cases are covered in
  `tests/test_state.py`.
- **A `pattern` in a JSON Schema can kill the Ollama server.** Its grammar compiler (0.18.0) takes a
  SIGSEGV inside cgo on two constructs: the non-capturing group `(?:...)` and the shorthand class
  `\d`. It is not a rejected request — the whole service dies and systemd restarts it, so the next
  call fails too with `model runner has unexpectedly stopped`. `OBSERVATION_ID_PATTERN` is written
  as `^(structure|momentum|volume)-[0-9]+$` for exactly this reason. Any new `Field(pattern=...)` on
  an `LLMOutput` has to stay in that subset; Python validation cannot tell the difference, so
  nothing else will warn you.
- **`ollama ps` is the only honest VRAM check.** It reports `SIZE` and a `PROCESSOR` split: anything
  other than `100% GPU` means layers spilled to CPU and latency is about to multiply. `num_ctx`
  drives that as much as the weights do, since the KV cache grows with it.
- **A Hypothesis strategy that can generate impossible states produces false counterexamples.**
  `accounts` used to draw `last_loss_at` up to a week *after* the evaluated instant. With
  `cooldown_after_loss_minutes=0` the cooldown rule is disabled, so the order goes through and the
  elapsed time comes out negative — a failure that says nothing about the gate, because an account
  cannot have lost money in the future. Bound generated timestamps by what the system can actually
  reach. It surfaced ~500 examples in, so it looked like a regression from an unrelated change.
- **Ollama's default `keep_alive` is 5 minutes**, shorter than any candle the runner watches, so
  without `CA_OLLAMA__KEEP_ALIVE` every cycle would pay the reload. It is passed per call, not set
  on the server.
- **`load_settings()` reads no file unless handed a path.** `Settings` used to declare
  `env_file=".env"`, which resolves against the working directory, so every call anywhere in the
  suite picked up whatever `.env` the machine happened to have. The suite was green only on a
  machine with no configuration — on the one that operates, the CLI test for a missing config
  failed, and `test_run_without_symbols_refuses_to_start` started a real `Runner` and slept until
  the next candle close, hanging the whole run with no output. Now only `cli.py` and `ablation.py` ask
  for `DEFAULT_ENV_FILE`, and `tests/test_cli.py` chdirs into `tmp_path` because it is the one file
  that enters through `main()`. Two tests in `test_settings.py` pin both directions.
- **The qwen repeated across four roles is deliberate, not an oversight.** The local fallback is a
  qwen and `bull` now declares it too, so an evaluation where those four roles are all out of budget
  runs `structure`, `momentum`, `volume` and `bull` on the same model — four of six roles. It is
  accepted: the hard constraint is `bull != bear` and it still holds, so the debate stays diverse
  exactly where a shared family would collapse it, and the three technical readings are of three
  different dimensions rather than three opinions on one question. The cost is real and bounded —
  those readings correlate in the degraded case, which is one more reason `LLMCall.backend` is on
  every row. What is not accepted is putting a qwen on `bear`, and `Settings` refuses it.

## Configuration

`.env.example` is the template; copy it to `.env` (gitignored) and fill in the gateway key. Every
role maps to a distinct family — the hard constraint is only between the two desks, but three
technical agents on one model would make the same mistake three times.

| Role | Model | Family | `structured_output` | `quota_per_window` | `quota_weight` | Fallback |
| --- | --- | --- | --- | --- | --- | --- |
| structure | MiMo-V2.5 | xiaomi | json_schema | 30 100 | 1.0 | local |
| momentum | DeepSeek V4 Flash | deepseek | json_mode | 63 300 | 2.0 | local |
| volume | Hy3 | tencent | json_schema | 4 300 | 1.0 | local |
| bull | Kimi K2.6 | moonshot | json_schema | 4 300 | 1.0 | local |
| bear | MiniMax M3 | minimax | json_mode | 3 200 | 1.0 | none |
| decider | GLM-5.2 | zhipu | function_calling | 880 | 1.0 | never |

The `quota_per_window` and `quota_weight` columns are **Go figures** (the page of the subscription,
2026-10-02), and the models are the map in `.env` today. Under `payg` they do not apply: Zen publishes
no request limit, so each remote role declares `ZEN_UNPUBLISHED_QUOTA` (100 000) and weight 1.0 — the
2.0 on `deepseek-v4-flash` is a Go pool weight — see "Pay as you go (Zen)". The role map itself is not
changed by that block.

`structured_output` has no default, on purpose: a default is the implicit constant this field exists
to remove, moved from LangChain into the configuration. **All six values are now measured**, not
declared — see the re-probe below. Local fallbacks only accept `json_schema`: `OllamaBackend` constrains generation by passing the
schema in `format` and exposes no tools, so `Settings` rejects anything else rather than accepting a
declaration it would ignore.

### The six models, re-probed on the real prompt (desktop, 2026-08-15)

The first table of six models was contaminated: three of those failures were `ValidationError` raised
inside `complete()`, where the raw text never reached the router and the documented retry never ran.
This run goes through `ModelRouter` with the production prompts and schemas, no cache and **no
fallback declared** — a role that degraded to the local model would report the local model's numbers.

| Role | Model | Mode | Attempts | Transport | Content | Valid | Retries/verdict | Mean latency |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| structure | `mimo-v2.5` | json_schema | 12 | 0 | 0 | 12 | 0.00 | 5.9 s |
| momentum | `deepseek-v4-flash` | json_mode | 12 | 0 | 0 | 12 | 0.00 | 11.9 s |
| volume | `hy3` | json_schema | 12 | 0 | 0 | 12 | 0.00 | 24.4 s |
| bear | `minimax-m3` | json_mode | 12 | 0 | 0 | 12 | 0.00 | 22.4 s |
| decider | `glm-5.2` | function_calling | 13 | 0 | 2 | 11 | 0.18 | 15.2 s |
| bull | `kimi-k2.6` | json_schema | 4 | 0 | 0 | 4 | 0.00 | 32.1 s |
| ~~bull~~ | `qwen3.7-plus` | json_schema | 5 | **4** | 0 | 1 | — | 45.0 s |

Three readings:

- **minimax-m3, mimo-v2.5 and glm-5.2 were fine all along.** Five of six models produce zero invalid
  content on the real prompt in their declared mode. What the old table measured was the missing
  retry, not the models.
- **The decider is the one that needs the retry, and it works.** Twice out of thirteen attempts
  glm-5.2 returned `invalidation_price` as a string and omitted `dismissed_side`; the router attached
  the error and the second attempt validated. Budget ~1.2 decider calls per evaluation, not 1.0.
  The dry-run's static count does not model retries: 840 declared calls become ~1 000 in practice,
  which only fits because a run of this size spans more than one 5-hour window.
- **Latency is not evenly spread.** `hy3` and `minimax-m3` cost four times `mimo-v2.5`. With the
  technical fan-out and the two desks each running concurrently, one full evaluation is
  ~max(6, 12, 24) + ~max(32, 22) + ~15 ≈ 70 s, consistent with the ~90 s used for planning.

**`json_mode` is not a downgrade here.** `deepseek-v4-flash` does not accept `json_schema`, so the
schema is not enforced server-side — and it still validated 12 of 12 with zero retries, because the
router validates against Pydantic and retries with the error attached. Replacing it was considered
and rejected: it yields 63 300 / 2.0 = 31 650 effective calls per window against `qwen3.8-max`'s 160,
so the swap would move the system ceiling from the decider (880) to momentum (80 effective) — and
`qwen3.8-max` answers 0/10 anyway.

### A model id in the catalog is not a model that answers

`doctor` reports `gateway OK 6/6 ids presentes` and every id is genuinely listed. `qwen3.7-plus` still
answers **0/10** with `503 Upstream request failed: Endpoint is unavailable`, in 0.2 s, in all three
structured-output modes. So does the rest of the family — `qwen3.7-max`, `qwen3.6-plus`,
`qwen3.8-max` — with a one-minute window in the middle where two of them answered a Ping and then
went back down.

Two things follow. The catalog check is worth what it costs and no more: it catches a typo in an id,
not a provider that is down, and only a real call distinguishes them — which is what `modes` does.
And **a transport rejection is not a mode problem**: identical 503s across `json_schema`, `json_mode`
and `function_calling` say the endpoint is unavailable, and changing `STRUCTURED_OUTPUT` in response
would be changing a line that was already correct.

`bull` moved to `kimi-k2.6` (moonshot) for that reason, keeping six roles in six distinct primary
families. Its `quota_per_window` is a declaration the gateway does not publish; the probe only says
it answers. If the ablation starts degrading at `bull`, that number is the first suspect.

All six remote models come from one OpenAI-compatible gateway. `Settings.openai` holds a single
`api_key` and `base_url`, so moving one model to a different provider means moving those two fields
onto `ModelChoice` — a change to the configuration contract, not a change to `.env`. Model ids are
the gateway's own, without a provider prefix: it serves `glm-5.2`, not `zhipu/glm-5.2`. That, and
whether each model answers in its declared mode, is what `doctor` checks before the first paid
call.

The local fallback is `qwen3:8b` (5.2 GB on disk, 6.0 GB resident, `100% GPU` at `num_ctx=4096`
against ~7.0 GiB free): one resident model, no second local model alongside it. Four roles now
declare it — the three technical ones and `bull` — so a fully degraded evaluation runs four of six
roles on one model. The hard constraint still holds: `bull`'s families are {moonshot, qwen} and
`bear`'s is {minimax}, disjoint.

`bull`'s fallback exists so the ablation's `local_bull` arm can run at all. **It does not rescue a
provider outage**: `resolve()` degrades on exhausted quota, never on a transport rejection, so a 503
aborts the evaluation with or without a fallback declared.

### The ceiling on evaluations per window

**Go subscription only.** The 880 below is derived from the `quota_per_window` of the Go page. Zen
publishes no request limit, so under `payg` there is no such ceiling to derive: the only caps are the
account balance and the monthly limit set in the Zen workspace, both in dollars.

    ceiling = min over roles of  sum over role_choices(role) of  quota_per_window / quota_weight

With the map above the ceiling is **880 complete evaluations per 5-hour window** — one every ~20 s,
fixed by the decider, the only role with no fallback. Three things this number is not:

- It counts **complete** evaluations. One that dies at the activation gate costs nothing; one that
  dies in `consolidate_evidence` costs three technical calls and no decider call. Only evaluations
  that reach `decide` consume the budget that sets the ceiling.
- It is an **upper bound**. `ModelRouter` records every attempt, so a retry costs quota, and the
  decider is the role most likely to retry: `Decision` carries the strictest validator in the
  contract. The real figure has to be measured, not derived.
- It is **not the binding constraint** in operation. Ten symbols on 4h candles spend ~13 evaluations
  per window against a ceiling of 880. VRAM and local inference latency bind long before quota does.

The `role_choices` sum matters even though it changes nothing today: drop a fallback onto the
decider and a formula that only reads the primary would understate the ceiling.

### Degrading to the local backend

Out of budget, `QuotaLedger.resolve(role, choices)` drops to the role's local fallback instead of
failing — `choices` being `settings.role_choices(role)` as the router holds them, primary first.
`resolve()` runs inside the retry loop, so a single verdict can start remote and finish local —
which is why every `LLMCall` records its own `backend` and its own `valid`. Without those two
columns the journal mixes a large remote model with a small local one on the same line, and a
fallback that needs three attempts per verdict looks exactly as cheap as one that gets it right
first time. `metrics.backend_stats()` is what reads them back, denominator included.

The decider is the exception, enforced by a `Settings` validator: it takes no fallback at all. Out
of budget, the evaluation aborts and the journal records why. Degrading it would change who makes
the final call without that appearing anywhere until the order was already placed.

**Measured on the desktop**, RTX 3070 Ti with the desktop session loaded, `num_ctx=4096`. Nothing
has been measured on the laptop yet; with its Max-Q envelope these are a floor for it, not a
forecast.

`qwen3:8b`, the configured fallback, over the real `technical.md` prompt (1548 characters) with
`TechnicalVerdict` as the grammar, three calls each, all six valid on the first attempt:

| What | Time |
| --- | --- |
| Warm call, thinking on — Ollama's default | 9.8 s (8.9 - 10.8) |
| Warm call, `think: false` | 5.4 s (4.9 - 6.1) |

`OllamaBackend` does not pass `think`, so today every local call pays the reasoning tokens: 1851
characters of thinking per verdict that nothing downstream reads, since `response.message.thinking`
is discarded and only `content` is returned. The waste is 4.4 s per call — ~13 s per evaluation with
all three technical agents degraded, on top of the 16 s the verdicts themselves cost.

`llama3:latest`, the previous fallback, over a ~375-token technical prompt — kept because it is what
the parallelism finding below rests on:

| What | Time |
| --- | --- |
| First call, weights not resident | 6.45 s |
| Warm call | 3.4 - 4.5 s |
| Three technical agents, sequential | 11.6 s |
| The same three via `asyncio.gather` | 10.3 s |

A 1.13x speedup from parallelism is a measurement of no parallelism: **Ollama serializes on the
GPU** (`OLLAMA_NUM_PARALLEL:1` at these VRAM levels), so the graph's technical fan-out becomes a
queue. On `qwen3:8b` that means budgeting ~29 s of wall clock for that stage when all three fall
back to local, or ~16 s with thinking off, against a 4h candle — three orders of magnitude of
headroom, so the runner's skip path should never fire on local latency alone. It exists for a hung
provider, not for this.

Model choice is deliberately left in `.env`. Criteria for the candidate, to be settled with an
ablation and not by taste: GGUF **text-only** (a multimodal variant loads a vision encoder and
spends over 1 GB of VRAM on it even for pure-text calls), resident alongside the desktop in ~7 GB,
and reliable under a JSON Schema grammar. `qwen3:8b` meets all three on the desktop and is what is
configured; the ablation's `local_technicals` arm is what decides whether it is good enough, not the
fact that it fits.

**Check the tag, not the name.** `qwen3.5:latest` sounds like the newer, better choice and is 8.9 GB
resident — `ollama ps` reports `28%/72% CPU/GPU` and a one-line answer takes 30.8 s against
`qwen3:8b`'s 5.4 s on the real prompt. Nothing in the system notices: the call succeeds, it is
merely six times slower, so this only ever shows up as latency nobody can explain.
