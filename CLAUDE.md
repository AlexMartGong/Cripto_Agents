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
| `state.py` | Data contract between every node. Imports nothing else from the package — it is the root of the dependency graph. `LLMCall` carries, per attempt, who answered upstream of the gateway (`upstream_model`, `upstream_endpoint`), `None` when no header said so. |
| `settings.py` | `pydantic-settings` config, `CA_` prefix. `load_settings()` fails at startup naming the missing variables. Holds `Billing` (`go`/`payg`) and the dated `PriceTable`. Since block T7 `billing` also decides what brakes a run: the window quota under `go`, a dollar cap under `payg`. |
| `quota.py` | Sliding-window quota ledger per `(role, model)`, injected clock. Holds a window, not `Settings`: the candidates come from the caller, which is what lets one counter serve several role maps. `seed()` rebuilds the window from journaled calls after a restart. Under `payg` the router does not ask it about a remote role (block T7): it still records every call, it no longer decides. |
| `spend.py` | `SpendGuard`, the dollar cap shared by the probes, the ablation and the runner, and `spend_guard()`, the one door through which the last two build it (refuses `billing != payg` and a remote role whose model has no `payg` price row). Measures what was already spent with `consumption.consume`; cannot import the router. |
| `llm.py` | `ChatBackend` protocol, OpenAI/Ollama adapters, and `ModelRouter` — resolve by budget, cache, validate, retry, record. The only module that reads response headers, and it reads two (`x-opencode-upstream-model-id`, `x-opencode-endpoint-id`) from the response of that very call; `Completion.upstream` carries two names, never the header dictionary. `ModelRouter._choose` is the one place where `payg` lifts the quota of a remote role, and `_admit` asks the spend guard once per invocation, before the first live remote attempt; a refusal is `SpendCapReachedError`. On the `ValidationError` path of `json_schema` the usage and the whole content are read from the response that hangs from the exception (block T7). |
| `cache.py` | Response cache keyed by `(backend, model, prompt digest, schema, mode)`. Each entry is an envelope naming who answered what, and holds every attempt that produced content, valid or not. The envelope also keeps the upstream the gateway declared, outside the key, so a hit can say whose text it returns. |
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
| `ablation.py` | Pipeline variants compared over one plan — a manifest or a contiguous history; `python -m crypto_agents.ablation` renders the table, `--dry-run` prices it first. Every run writes a directory: one JSONL journal per arm plus `meta.json`. `--fill` requires `--max-usd` under `payg` and refuses it under `go`; one `SpendGuard` for every arm, as with the ledger. `--out` has no default: the table goes to `report.md` inside the run directory and an existing file is never overwritten. |
| `audit.py` | Reads a run directory and prints what happened in it, each figure next to the digest of the file it came from. `python -m crypto_agents.audit <dir>`. No model calls. Also owns `load_plan()` / `resolve_horizon()` (the one door to a run's candles, shared with `criteria`) and the net-return section, and `RunKind` (`ablation` \| `probe`): `RunMeta.kind`, an old `meta.json` loads as `ablation`, `zen_probe` writes `probe`, `criteria` refuses a probe directory and `audit`/`consumption` read it. Its last table says who answered each (role, requested model) according to the gateway, over the run and arm by arm, with `sin cabecera` apart from `sin respuesta`. `RoleMeta` also carries `family` and `structured_output`, and `RunMeta` the OpenAI timeout and the spend cap: the report has a `Mapa de roles` table per arm, and a `meta.json` written before those fields loads and prints `no registrado`. |
| `outcomes.py` | Labels each order against later candles: invalidation hit first, or the close at the horizon. Three scorings — declared stop, common stop, horizon close — and the per-evaluation vector. `net_return()` adds a descriptive net return on top of any of them; the gross path is untouched. |
| `stops.py` | The one function that builds the common stop (`entry ∓ 2·ATR`). Imports only `state`. |
| `baselines.py` | Four decision policies that call no model: always buy, always sell, uniform random, trend rule. Cannot import the router. |
| `alerts.py` | Quota running out, repeated vetoes, validation failures, skipped cycles, evaluations lost to insufficient funds (402). Pure over journal records. Under `billing = payg` the quota alert is silent and `cli alerts` prints `cuota: no aplica (payg)` instead of a percentage against the sentinel. |
| `queries.py` | Journal filters by symbol, action, backend and abort cause. |
| `doctor.py` | Startup checks: gateway catalog, Ollama tags, VRAM split, exchange and credentials. |
| `bootstrap.py`, `cli.py` | Composition root and the `crypto-agents` entry point. `crypto-agents run --max-usd X` hands the router a spend cap for that process (`payg` only). |
| `activation_sweep.py` | The four gate rules over a committed history, no model calls. Sizes the ablation. |
| `selection.py` | Stratified selection of activations across symbols and time spans, and the versioned manifest the ablation runs over. No model calls. `--out` is mandatory and an existing file is refused before anything is computed. Selections are nested: with the manifest's seed and histories, a larger target contains the smaller one. |
| `metrics.py` | Aggregations over a run — the funnel, action mix, vetoes by rule, quota by role and backend — and the audit of one: attempts, live latency, abort causes, stop side, action against the desks. Pure over `EvaluationRecord`. `AbortKind.INSUFFICIENT_FUNDS` reads the 402 from the message (`Error code: 402` / `Insufficient account funds`, before the generic transport marker); `provider_rejections()` groups rejections by (model, role, code, body) so a 410 and a 503 never share a row. `upstream_distribution()` counts who answered per (role, requested model); `schema_faults()` reads from a schema failure message whether the model left a key out (`Field required`) or sent it empty on an action that needs it (`exige:`). `AbortKind.SPEND_CAP` reads `tope de gasto alcanzado` from the message of `SpendCapReachedError`. |
| `dispersion.py` | Standard deviation of the per-evaluation return over the whole 4h activation pool (`always_buy`/`always_sell`, common stop) and the detectable paired difference for n = 140/280/420. Carries no mean on purpose: no model field, no printed figure, no file written. No model calls. `python -m crypto_agents.dispersion`. |
| `perp_probe.py` | Public read-only probe of USDT perpetuals on `binanceusdm` / `bybit`: contract limits, 24 h volume, funding (the last 730 days, or an explicit `--start`/`--end` range) normalised to 24 h, connectivity, `exchange.has`. Imports nothing from the package. No keys, no orders, no model calls. `python -m crypto_agents.perp_probe --exchange binanceusdm`. |
| `funding.py` | Reads the versioned `data/funding/` series and answers one question: the sum of funding rates over `(entry, exit]`, or `None` if the series cannot guarantee it is all there. Stdlib plus `perp_probe`; no network, no credentials. |
| `consumption.py` | Cost of each call in USD and its share of the subscription pool, from the tokens the provider reported; per-arm and per-role report; declared `quota_per_window` against the page's estimate. Measures only. `call_cost_range_usd()` gives `(low, high)` per call: the high end charges the non-cached prompt at `cache_write` where the model has one, and a call with `cached_tokens: null` runs from all-cached to all-new; `format_cost()` marks an interval with `†`. `python -m crypto_agents.consumption <run dir>` / `--quotas`. |
| `zen_probe.py` | Probe of OpenCode Zen (pay as you go): `/models` catalog, structured-output mode per id, `x-opencode-session` with and without, 12 real technical verdicts per (candidate, dimension) and a chained desk/decider stage that measures their tokens. Everything through `ModelRouter`, no cache, no fallback; writes a run directory (`var/zen-probe/<UTC start>/`) that `audit` and `consumption` read. Refuses `billing != payg` and any `/zen/go` base_url before building a backend. `--desks --technicals-from <probe dir> --max-usd X` probes the desks instead: two bull candidates (`kimi-k3`, `qwen3.8-max`; `deepseek-v4-pro` left in block T5, it is the momentum model), `minimax-m3` as bear and `glm-5.2` as decider over the *same* technical evidence, produced once per activation by the first producer of an ordered list that gives a valid verdict (structure and volume: the one the current rule picks from the previous probe; momentum: `deepseek-v4-pro`, the role map's model and today the only one in `MOMENTUM_PRODUCERS`), the decider once per (activation, bull with a valid brief) in its own `decider+<bull>` arm, and a hard `SpendGuard` cap (no rigorous cost bound exists before calling: the repo sets no `max_tokens`). It also writes `content.jsonl` — the verdicts, briefs and decisions the models answered, which the journal does not keep — and `--compare-with <probe dir>` reports, before the first decider call, how many `prompt_digest`s match another probe, and pairs the decider's outcome by activation in the report. Two more probes read a desks probe's `content.jsonl` instead of asking for the inputs again: `--deciders-from <dir> --max-usd X` measures `decide_solo` and `decide_without_debate` (`Proposal`) over its activations and its technical evidence, and `--bear-from <dir> --bear-mode <mode> --max-usd X` asks the bear the same prompt in another structured-output mode without touching `.env`; both refuse before the first call unless the recomputed `prompt_digest`s are the ones the source asked, and both report the upstream of every attempt. The technical probe (no `--desks`) reads the present models from the role map (`PRESENT_ROLES`) and pings an id once even when it is both present and a candidate. `python -m crypto_agents.zen_probe --machine desktop\|laptop [--dry-run]`. |
| `estimate.py` | USD estimate of stage 1: the `--dry-run` counts over `data/ablation_selection.json` times the measured cost per call of a probe directory. No token is estimated; local calls and baselines are 0 by rule; structure and volume are a range between candidates that answered; one top-up (`PriceTable.topup_charge`). Labelled as an estimate. Each role is budgeted with the attempts per verdict its probe measured (live attempts over invocations), not with constants; `DECIDER_ATTEMPTS` and 1.0 remain as a labelled fallback for a role with no rows. `--desks <dir>` gives one line per bull candidate with its own conditioned decider; `--source ROLE=DIR` (repeatable) names the probe a role's cost and attempts come from; `--balance X` (read from the console, never fetched) answers `PASA`/`NO PASA` against cost x `LAUNCH_MARGIN` at the high end, exit 1 when nothing passes. A table per arm and role (paid calls, measured attempts, USD per attempt, USD range, source file) adds up to the total the verdict is judged on; the bull lines come from `candidates.BULL_CANDIDATES`, not from the arms a directory happens to hold. `--source decider@ARM=DIR` gives one arm's decider its own measurement (the arm `<model>@ARM` of a deciders probe); a decider priced with the prompt of `full` in an arm that sends a shorter one is labelled an estimate, never a bound. `python -m crypto_agents.estimate <probe dir>`. `--arms a,b,…` budgets only those arms (the count is redone, not trimmed). structure and volume stop being a range when the role map declares a model the probe measured in that role. |
| `candidates.py` | The lists of models proposed for a role that has no model yet: `CANDIDATES` (structure, volume) and `BULL_CANDIDATES`. Imports nothing from the package, so `estimate` can read the same list `zen_probe` probes without importing a module that calls models. |
| `criteria.py` | Mechanical evaluator of the amendment's criteria over a run directory: `full` against `solo`/`no_debate`/`bull_only` and every arm against each of the four baselines (paired difference, 95% CI, verdict from a mandatory `--delta`), the run-validity guards (decider lost to quota, an evaluation lost to insufficient funds at any node, an evaluation lost to a provider failure — transport or timeout — at any node, cache hit from another backend, any veto but `invalid_stop_side`: `CORRIDA INVÁLIDA`, exit 1, no verdicts; on a resumed chain they read the last link only, `final_records`), and the peak 5 h window usage per role (`no aplica (payg)` instead of a share when the run was paid per use). No model calls. `python -m crypto_agents.criteria <run dir> --delta X`. Carries two extra columns per comparison with the net-return paired difference, labelled descriptive; they enter no verdict. Since block T7 a sixth guard: an evaluation cut by the spend cap, at any node and arm. |

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

Six failure modes reach a node, and they are not the same failure. `FailureKind` is closed —
`schema`, `context`, `timeout`, `transport` — and `LLMCall.failure_kind` is `None` exactly when the
attempt is `valid`:

| What broke | Type | `LLMCall.failure_kind` | Retried | Cached |
| --- | --- | --- | --- | --- |
| Content arrived and did not match the schema | `InvalidModelOutputError` | `schema`, one row per attempt | yes, with the error attached | yes |
| Content matched the schema and contradicted its context | `InvalidModelOutputError` | `context`, one row per attempt | yes, with the error attached | yes |
| The provider did not answer in time | `ModelCallError` | `timeout` | no | no |
| The provider never returned content (4xx, 5xx, DNS, credentials) | `ModelCallError` | `transport` | no | no |
| Neither model fit the window | `QuotaExhaustedError` | no row — nothing was spent | no | — |
| The dollar cap was reached (`payg`) | `SpendCapReachedError` | no row — nothing was spent; it carries the attempts already replayed from the cache | no | — |

The first two measure the model, the next two the provider, and the last two whoever set the
budget: the provider's window, or the operator's dollars. Journal lines written when the failure
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
uv run pytest                  # 1862 tests
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
- A `ValidationError` escaping LangChain loses the message, and until block T7 the usage with it:
  the attempt was billed and journaled `None`, counted as unmeasured. The HTTP response still hangs
  from the exception. Block T6 reads the two upstream headers from it; block T7 reads the usage and
  the whole content too (`usage_from_rejected_parse`, `content_from_rejected_parse`). What stays
  unmeasured is a response whose body cannot be read or carries no counters.

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
  change and `replay-v3` did not move; a run with measured tokens hashes them, like latency. The two
  upstream fields of block T6 follow the same rule.
- **`ChatBackend.complete()` returns `str | Completion`.** Real adapters return `Completion(text, usage)`;
  fakes that return text mean "no usage". `as_completion()` is the one place that normalises it — `doctor`
  needed it too.
- **`RunMeta.billing`** is written by the ablation; a run without it needs `--billing` on the command, and a flag
  that contradicts `meta.json` is refused. `--dry-run` prices the pool with the page's estimates, labelled
  "estimación de la página, no medida".

`python -m crypto_agents.consumption --quotas` against the shipped config lists no disagreement with
the page since block T7, and corrects nothing. `momentum` and `bull` are reported as not comparable
(`sin estimado`): `deepseek-v4-pro` (since T5) and `qwen3.8-max` (since T7) are ids the Go page does
not estimate, and their 100 000 is a declaration. Until then they were the two disagreements:
`deepseek-v4-flash` (63 300 declared, 31 650 effective with weight 2.0, against 13 000) and
`kimi-k2.6` (4 300 against 1 150).

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
crypto-agents run --max-usd 5   # the same, with a dollar cap for this process (payg only)
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
them — 20 per symbol, and within each symbol 4 per each of 5 equal time spans — and wrote
`data/ablation_selection.json`, which is committed along with `data/history/*_4h.csv`.

Six properties, each with a test in `tests/test_selection.py`:

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
- **A larger selection contains the committed one** (block T7). Each cell is shuffled by a generator
  that advances with the size of the cell, not with the target, and a prefix of that order is taken.
  With the manifest's seed and histories, a target of 280 or 420 contains its 140, entry for entry,
  so growing n keeps everything already paid for. The fill of a short cell
  (`selection.py:392-401`) would not break it — it still takes prefixes — but it does not even
  trigger here: the smallest cell holds 106 candidates and 420 asks for 12. The mutation that seeds
  each cell's shuffle with the quota is pinned.
- **The command does not write the committed manifest by default** (block T7). `--out` used to
  default to `data/ablation_selection.json`, so running the command again — with another
  `--target`, say — replaced the plan the ablation runs over. Now `--out` is mandatory and an
  existing file is refused before anything is computed.

`ReplayPlan` is what run and dry-run both consume, built either from a manifest or from a contiguous
history (`plan_from_history`). One code path, so the budget and the table cannot end up describing
different walks. `score_outcomes()` takes histories keyed by symbol for the same reason: with seven
series in play, one `rows` argument would score ETH's order against BTC's candles.

Measured on the committed selection: 140 evaluations, 140 activations, 0 prepare failures, two
dry-runs agreeing on all 140 exact prompt digests, and the decider at **840 calls over the six
arms**, against Go's 880 per window. At one attempt every role fits; retries are not in that count
(see the re-probe). The quota table now carries a "con reintentos" column (decider × 1.2 = 1 008):
against Go's 880 the decider fits at one attempt and **not** with retries, which is what
`fits_with_retries` says. Under `go` that cell reads `**NO**`. Under `payg` it reads `no aplica
(payg)` since block T7 (`QuotaLine.limits`): the router no longer applies a remote role's quota
there, so the dry-run does not judge against a limit that does not brake.

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
meta.json       plan sha-256, argv, --fill, arms, start, git commit and dirty flag, resumed_from,
                and since block T7 the role map of each arm, the OpenAI timeout and the spend cap
<arm>.jsonl     one EvaluationRecord per line, written as each evaluation ends
report.md       the comparison table, unless --out sent it elsewhere (block T7)
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
  **Superseded in block T7:** under `payg` the router no longer applies the quota of a remote role
  at all, whatever it declares, so the sentinel stopped being what prevents the degradation. See
  "Block T7" below.
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
  balance belongs to the account): `CORRIDA INVÁLIDA`, exit 1, no verdicts. This is the fourth
  condition of criterion 6; the text of amendment 2 lists it since block T4, added before the second
  run and marked as such there. `alerts` reports it whatever the billing.
- **A provider failure is an invalid run too** (block T5; the fifth condition, listed in amendment 2
  before the second run and marked as such there). An evaluation lost to `AbortKind.TRANSPORT` or
  `AbortKind.TIMEOUT` at **any** node invalidates. `resolve()` never degrades on transport, so a dead
  route on a technical node leaves every arm that uses it without evidence, and the four conditions
  would have called that run valid: `deepseek-v4-flash` answered 404 fifteen times in a row while
  `.env` still pointed momentum at it. The first error of the record rules, as everywhere
  (`metrics.undecided_causes`): an evaluation that exhausted its attempts on validation in one node and
  also met a rejection in another counts as validation, which is the model's. The mutation that looks
  only at the decider nodes is pinned through `main()`. One T3 test changed meaning with it: a 402
  misread as generic transport used to leave the run *valid*; now the run is invalid either way and
  what `AbortKind.INSUFFICIENT_FUNDS` protects is the name of the reason — a balance is not rescued by
  resuming, a rejection may be.
- **A resume is judged by its last link** (block T4; `criteria.final_records`). A resume walks the whole
  plan again, so its directory is the final state of every evaluation: one lost to a 402 (or to the
  decider's quota, or to a provider failure) in the first pass and decided in the second does not
  invalidate; one still undecided in the last pass does. `tests/test_criteria.py` runs both over a
  two-directory chain, for the four kinds of loss (402, decider quota, a 404 at momentum, a timeout
  at the decider), and the mutation that adds up the links makes the rescued chain invalid. This was already
  how `main` behaved; it had no name and no test. Only the peak of the 5 h window adds the whole chain
  up. **Known limit, not fixed:** it trusts the last link to be complete. A resume that was interrupted,
  or launched with fewer `--arms`, does not carry what it did not run again, and neither the guards nor
  the verdicts see it; fixing it means judging by the latest record of each (arm, evaluation) across
  the chain.
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

**Every mode `Ping` of both runs** (added in block T4, correcting the block's summary, which named only
`deepseek-v4-flash` as failing at 06:35Z: `kimi-k2.6` failed too, in both runs). One attempt per id.
`valid` and `failure_kind` are the `LLMCall` fields, the code is read from `failure_message`, latency is
on the desktop, and the last column is the sha-256 of `<run>/<id>@ping.jsonl` (one row each). In each
run six answered and the same two did not:

| run | id | role | mode | valid | failure_kind | code | at | latency | sha-256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `…062628Z` | `kimi-k2.6` | bull | json_schema | **no** | transport | 410 | 06:26:30Z | 1.4 s | `2fd5b511b1174fba70deb51cba138eec43fbad62ff53abb339aa559278abd2cf` |
| `…062628Z` | `kimi-k3` | bull | json_schema | yes | — | — | 06:26:32Z | 2.4 s | `23f52bf22a464cd9b6aee404b0618a0e857b7bc764a389cbb75cd3530c27c161` |
| `…062628Z` | `qwen3.8-max` | bull | json_schema | yes | — | — | 06:26:31Z | 1.7 s | `ae9def1829ca0a12ba0a4baf1d4cfc875b61197de2b951ba7c8deec30486ffd8` |
| `…062628Z` | `deepseek-v4-pro` | bull | json_schema | yes | — | — | 06:26:31Z | 1.4 s | `e3439036b89bf678097d2715e4f99f4ffccc4953b91facce6b95ee5ed95b7f35` |
| `…062628Z` | `minimax-m3` | bear | json_mode | yes | — | — | 06:26:30Z | 1.2 s | `fad834a41ac5c38d06fdd7ee42b0b6e322936aa0d120f9c02905bb2033012898` |
| `…062628Z` | `glm-5.2` | decider | function_calling | yes | — | — | 06:26:32Z | 2.7 s | `df40c2b66c9a9a0bb3e960ccbe5ab236c289fbfac6c6fcc8b4b6b7c74a99a62a` |
| `…062628Z` | `glm-5.3-flash` | structure | json_schema | yes | — | — | 06:26:31Z | 2.0 s | `cfad0d71cc579d36a5c2182b7372b3b5b7666c14b29cfd67df4a60d060bbbd93` |
| `…062628Z` | `deepseek-v4-flash` | momentum | json_mode | **no** | transport | 404 | 06:26:30Z | 0.5 s | `12e34248ac1ed205e6c7aef2037046ab2f0663c1e3ede8086444c07e58752a7c` |
| `…063458Z` | `kimi-k2.6` | bull | json_schema | **no** | transport | 410 | 06:35:00Z | 1.4 s | `1526015f99055529b637ef61859077dd18e195a8f87cd09a62ddd1efbe45b34b` |
| `…063458Z` | `kimi-k3` | bull | json_schema | yes | — | — | 06:35:02Z | 2.5 s | `939a0a3e18b3cb6abe9fab24a93cdad231869c281d7f53a947d3ed3ab7ac4847` |
| `…063458Z` | `qwen3.8-max` | bull | json_schema | yes | — | — | 06:35:00Z | 1.3 s | `1381f831960ff1a524fd8468bed47e9aa977cddf6acc85480b97e8793e4da919` |
| `…063458Z` | `deepseek-v4-pro` | bull | json_schema | yes | — | — | 06:35:01Z | 1.4 s | `44a999f01fbe5c0a20fdcd7a8df18724a7c4051608984c8e8c687bb11142f109` |
| `…063458Z` | `minimax-m3` | bear | json_mode | yes | — | — | 06:35:00Z | 0.9 s | `4cf68aba2f40852a10bff80d8c7a2c0f97e9280363230cb8f424c387152ab252` |
| `…063458Z` | `glm-5.2` | decider | function_calling | yes | — | — | 06:35:08Z | 8.7 s | `f85a6c7e9536dcbf9dda79885f5a01b12d6dc747a3d2bdf805f20b71573dba5f` |
| `…063458Z` | `glm-5.3-flash` | structure | json_schema | yes | — | — | 06:35:02Z | 2.7 s | `bc4a93ae8a5aad7c20e80dbc4dfbf68693087b39f1828ea7601142fafbf97ec0` |
| `…063458Z` | `deepseek-v4-flash` | momentum | json_mode | **no** | transport | 404 | 06:35:00Z | 0.5 s | `63d4b096a491b546c792a30e337572d069c3c8ac81c8bd5154411ee820d55357` |

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

### Block T4: the desks, measured (desktop; `var/zen-probe/`, not versioned)

One run of `zen_probe --machine desktop --desks --technicals-from var/zen-probe/20261005T043359Z
--max-usd 2`: `20261005T233053Z`, 23:30:53Z to 23:55:46Z, tree clean at `269d538`, same manifest
(sha-256 `73f87870cf24d06c341ff75fa72f164f9cd5202623cf0d76babbdab4a2803f23`). The cap's guard saw
**1.4216 USD** at the high end and refused nothing, so every candidate was asked on all 12 activations
and the `k/12` table of a cut run was not printed. Two `glm-5.3-flash` calls came back without `usage`
and are not in that figure.

**The evidence has more than one possible producer** (`evidence_arm`). Per dimension there is an ordered
list, each activation uses the first producer that gives a valid verdict, and the result is one tuple
per activation, the same object for the three bull desks and the bear. Structure and volume carry the
one producer the T3 rule picks; momentum carries `MOMENTUM_PRODUCERS`, fixed by whoever orders the
probe. A producer whose `Ping` was rejected before generating content is still asked on every
activation in its declared mode (`producer_mode`: the rejection did not disprove the mode, and a route
can come back); one that answered garbage in every mode, or timed out, is dropped. The mutation that
always uses the first of the list is pinned in `tests/test_zen_probe.py`.

| dimension | producers, in order | who produced the 12 |
| --- | --- | --- |
| structure | `glm-5.3-flash` | `glm-5.3-flash` 12 |
| volume | `glm-5.3-flash` | `glm-5.3-flash` 12 |
| momentum | `deepseek-v4-flash` → `deepseek-v4-pro` | `deepseek-v4-flash` 0, `deepseek-v4-pro` 12 |

`deepseek-v4-flash` answered `404 … Cannot find any route matching [POST] …/chat/completions` to its
`Ping` and to each of the 12 activations: 13 attempts between 23:30:54Z and 23:34:35Z, 0.3 s each. With
the two `Ping`s of T3 that is 15 rejections in a row since 06:26:30Z, after 13 valid answers at
04:34–04:35Z. Over that span it is not flapping. It is still the model the role map declares.

**`deepseek-v4-pro` is both a bull candidate and the author of every momentum verdict of this run.** That
candidate argued over a verdict written by its own model in 12 of 12 activations; the other two read
the same verdicts from another family. The report says so under "Familias".

Per (model, role): valid over asked, failures by `FailureKind`, attempts and retries per verdict, mean
latency **on the desktop**, mean tokens the provider reported (prompt / cached / completion), USD of the
arm's rows:

| arm | mode | valid | schema · context · timeout · transport | attempts · retries/verdict | latency | tokens | USD |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `glm-5.3-flash@structure` | json_schema | 12/12 | 0 · 0 · 0 · 0 | 12 · 0.00 | 13.2 s | 524 / 0 / 1857 | ≥ 0.0111 (1 of 12 without usage) |
| `glm-5.3-flash@volume` | json_schema | 12/12 | 0 · 0 · 0 · 0 | 12 · 0.00 | 15.9 s | 524 / 0 / 2162 | ≥ 0.0128 (1 of 12 without usage) |
| `deepseek-v4-flash@momentum` | json_mode | 0/12 | 0 · 0 · 0 · 12 | 12 · 0.00 | 0.3 s | — | 0 (no answer) |
| `deepseek-v4-pro@momentum` | json_schema | 12/12 | 0 · 0 · 0 · 0 | 12 · 0.00 | 19.5 s | 525 / 0 / 484 | 0.0312 |
| `kimi-k3@bull` | json_schema | 12/12 | 0 · 0 · 0 · 0 | 12 · 0.00 | 19.8 s | 2331 / 0 / 1260 | 0.3107 |
| `qwen3.8-max@bull` | json_schema | 12/12 | 0 · 0 · 0 · 0 | 12 · 0.00 | 24.7 s | 1820 / 0 / 2589 | 0.2301 to 0.2410† |
| `deepseek-v4-pro@bull` | json_schema | 12/12 | 0 · 0 · 0 · 0 | 12 · 0.00 | 15.4 s | 1689 / 0 / 490 | 0.0557 |
| `minimax-m3@bear` | json_mode | 12/12 | 10 · 0 · 0 · 0 | 22 · 0.83 | 4.8 s | 1679 / 853 / 485 | 0.0194 |
| `glm-5.2@decider+kimi-k3` | function_calling | 7/12 | 8 · 0 · 2 · 0 | 17 · 0.42 | 55.7 s | 3203 / 1956 / 3084 | 0.2373 |
| `glm-5.2@decider+qwen3.8-max` | function_calling | 10/12 | 8 · 0 · 0 · 1 | 19 · 0.58 | 43.8 s | 2961 / 1343 / 2684 | 0.2597 |
| `glm-5.2@decider+deepseek-v4-pro` | function_calling | 4/12 | 13 · 0 · 2 · 0 | 19 · 0.58 | 50.3 s | 3024 / 1851 / 2721 | 0.2396 |

† interval for cache write. By role, mode `Ping`s included (`python -m crypto_agents.consumption
var/zen-probe/20261005T233053Z`): structure ≥ 0.0111, volume ≥ 0.0128, momentum 0.0312, bull 0.5988 to
0.6097†, bear 0.0194, decider 0.7374 — **≥ 1.4107 USD, 1.4216 at the high end**. A lower bound: two
calls carry no usage, and the five decider attempts that ended in a timeout or a 5xx carry no tokens
either, so whether they were billed is not known.

- **The three bull candidates did what was asked**: 12 of 12 valid briefs at the first attempt, no
  context failure. They differ in cost per brief — 0.0259 (`kimi-k3`), 0.0192 to 0.0201† (`qwen3.8-max`),
  0.0046 (`deepseek-v4-pro`) — and in how the same prompt is counted: identical `prompt_digest` row by
  row, 2331 / 1820 / 1689 prompt tokens.
- **The decider lost 15 of its 36 evaluations.** 55 attempts: 12 evaluations valid at the first attempt,
  9 more after the retry, 10 lost with both attempts invalid, 4 lost to the 120 s timeout
  (`CA_OPENAI__TIMEOUT_SECONDS`), 1 to `InternalServerError: Unknown Error` after 92 s. **All 29 schema
  failures are the same one**: `la acción buy|sell exige: dismissed_side` — `glm-5.2` proposes a trade and
  omits the field `Decision` demands. On Go on 2026-08-15 that was 2 of 13 attempts; here it is 29 of
  55, and 1.53 attempts per evaluation against the 1.2 of `DECIDER_ATTEMPTS`. Valid decisions took 960
  to 7 483 completion tokens (median 2 650) and the slowest valid one 109.9 s, so the 120 s limit sits
  inside the distribution of valid answers, not beyond it. The valid counts per bull (7, 10 and 4 of 12)
  are not a property of the bull: 12 activations each, and the field that fails is the decider's own.
  Nothing was changed in response: not the prompt, not the timeout, not `DECIDER_ATTEMPTS`. Block T5
  found what the schema was telling the model — the field was outside `required` in the tool it reads
  — and changed the contract; see "Block T5" below for what that confirms and what it does not.
- **The bear needs its retry almost every time.** 10 of 12 first attempts were invalid, all
  `claims.N.grounded_in: Input should be a valid array`, and the retry fixed all ten: 22 attempts for 12
  briefs, against 12 of 12 with no retry on Go on 2026-08-15. `estimate` budgeted one attempt per bear
  call until block T5; it now reads the 22 for 12 from this directory.

Sources (`<arm>.jsonl` in `var/zen-probe/20261005T233053Z/`; rows are `LLMCall` rows):

| file | rows | sha-256 |
| --- | --- | --- |
| `glm-5.3-flash@structure.jsonl` | 12 | `c0e3a778b3d0bae81bf3ab5bf2620a3881755b313e6fcb6fc946b239ce4c32ef` |
| `glm-5.3-flash@volume.jsonl` | 12 | `f3d49c715dbbe084dd160fe437bc6f86a4b8a6285a79a5f3d9eaa19da0de646c` |
| `deepseek-v4-flash@momentum.jsonl` | 12 | `a0483c2b1cdddb9d3fb9cf3219bd86e28d2ca5a2dee32b57e127df78a70ad97a` |
| `deepseek-v4-pro@momentum.jsonl` | 12 | `089aaabc21705ef2b525fa040878bbcb3d0617349a4df973d6294ef8462662da` |
| `kimi-k3@bull.jsonl` | 12 | `3f12291b7760646eba2c57ea1f0fb33e7f59e3ddae1fae36fba37927ddb91c5e` |
| `qwen3.8-max@bull.jsonl` | 12 | `c95617a3546e0252cd4ef3bfb41f0b8e4f9be27b2afb6337b2e46e9150b44536` |
| `deepseek-v4-pro@bull.jsonl` | 12 | `d0f2f028f35ec6c29947a5c32007ec99e61df340fbeee82508e29e6b351b98fa` |
| `minimax-m3@bear.jsonl` | 22 | `50257005c23252379f4e50bb71a6049a6d9be19d2793747c1fdeba7cf673ba1b` |
| `glm-5.2@decider+kimi-k3.jsonl` | 17 | `ccfb24cbe45ee0e32b1e14a7084dd16ba9bd694bf688c45ef45f6ff96d65a371` |
| `glm-5.2@decider+qwen3.8-max.jsonl` | 19 | `6dfc7114d145261ca25c6a429f56160043826e315cae022abc6556140968bb66` |
| `glm-5.2@decider+deepseek-v4-pro.jsonl` | 19 | `8ff8d74c11e0142c939e61a3f4f50fdfe971a19718c8e9f65e0be3eb90c2bd5e` |
| `kimi-k3@ping.jsonl` | 1 | `4038c2491a2f9947e52c1c4f9d475bfab45b48c10759941a9c64cd4f748e1df3` |
| `qwen3.8-max@ping.jsonl` | 1 | `2b68b47500e1b5557120fc42ee23fb5e90c3c94d7c5e073498912d011219e7f5` |
| `deepseek-v4-pro@ping.jsonl` | 1 | `df7cb56718b783b2094e497cf84c635a24fc828f2f89d97a5f6d148486355a02` |
| `minimax-m3@ping.jsonl` | 1 | `c2012f1749c76f2e946fe67c35ed44601fbf5737857862dbd1e7997d72cff167` |
| `glm-5.2@ping.jsonl` | 1 | `b48be18cc440ea9db4cfe79b54e011a17652ff8cfbaeb57f3d8583cf7ab5cfd2` |
| `glm-5.3-flash@ping.jsonl` | 1 | `c3450c219816bad978ee931da7ec5fa1675f718ed21933314f826be135097898` |
| `deepseek-v4-flash@ping.jsonl` | 1 (404) | `34baaca01ad6b8dd916b5198d1d7edf8965202dfc256f94a5fdfb52c64c0102f` |

**Calls that arrived without `prompt_tokens` or `completion_tokens`**, over the eight directories of
`var/zen-probe/` (`…024946Z`, `…025129Z`, `…041459Z`, `…043359Z`, `…062628Z`, `…063458Z`, `…233053Z`,
`…233218Z`), read with `audit.read_run`. "Answered" is a live attempt that produced content; a rejection
or a timeout brings no usage because there was no response:

| model | attempts | no response | answered | without `prompt_tokens` | without `completion_tokens` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `deepseek-v4-flash` | 31 | 18 | 13 | 0 | 0 |
| `deepseek-v4-pro` | 80 | 2 | 78 | 0 | 0 |
| `deepseek-v4.1-flash` | 58 | 4 | 54 | 0 | 0 |
| `glm-5.2` | 62 | 8 | 54 | 0 | 0 |
| `glm-5.3-flash` | 130 | 3 | 127 | **33** | **33** |
| `kimi-k2.6` | 6 | 6 | 0 | — | — |
| `kimi-k2.7-code` | 4 | 4 | 0 | — | — |
| `kimi-k3` | 16 | 0 | 16 | 0 | 0 |
| `minimax-m2.7` | 4 | 4 | 0 | — | — |
| `minimax-m3` | 29 | 2 | 27 | 0 | 0 |
| `qwen3.8-max` | 19 | 4 | 15 | 0 | 0 |

Only `glm-5.3-flash`, and always both counters together: 33 of 127 answered (26%), by directory 0 of 25
(`…041459Z`), 7 of 25 (`…043359Z`), 21 of 26 (`…062628Z`), 3 of 25 (`…063458Z`), 2 of 25 (`…233053Z`),
0 of 1 (`…233218Z`). One of the 33 is an invalid attempt, where the adapter loses the usage by itself;
the other 32 validated. A different gap, not counted above: `cached_tokens` alone is `null` on 13 of 13
`deepseek-v4-flash` answers and on 2 `glm-5.3-flash` ones.

**What identifies the upstream.** Three `Ping`s to models of three families, through `ModelRouter`, with
a backend that inherits `OpenAIBackend` and adds an httpx response hook — a scratch script, not repo
code; `var/zen-probe/20261005T233218Z` (`kind=probe`, tree clean at `269d538`, 23:32:18Z), raw captures in
its `raw/<id>.json` with the generated text replaced by its length and nothing of the request stored.
Cost: `deepseek-v4-pro` 0.0001 + `kimi-k3` 0.0011 USD measured, `glm-5.3-flash` ≤ 0.0000 (a ceiling: its
`cached_tokens` was not reported) — rows of `glm-5.3-flash@ping.jsonl`
`5d0f1b72cd7edcbdcf34347c7c24708a14e9f4519cc10fcca78c1d596c55f3d1`, `deepseek-v4-pro@ping.jsonl`
`cfb6d49a19f9d02cefc544cc9fc4cb08632f2044c5f1d30fd5e0805e862e1363` and `kimi-k3@ping.jsonl`
`eb48f6ade7b5f58bf6e7adf47a9ff103ff651593e0d62c5c466d59e550257510`, one row each.

| id asked | `x-opencode-endpoint-id` | `x-opencode-upstream-model-id` | body `id` | `usage.prompt_tokens_details` |
| --- | --- | --- | --- | --- |
| `glm-5.3-flash` | `relace-glm5.3flash` | `z-ai/glm-5.3-flash` | 32 hex characters, no prefix | `{}` |
| `deepseek-v4-pro` | `together` | `deepseek-ai/DeepSeek-V4-Pro-0813` | `239ce7ad-aws_ue1` | `{cached_tokens: 0}` |
| `kimi-k3` | `inferact` | `moonshotai/Kimi-K3` | `chatcmpl-…` | `{audio_tokens: 0, cached_tokens: 256}` |

- **The response headers name the upstream; the body does not.** `x-opencode-endpoint-id` and
  `x-opencode-upstream-model-id` are in all three, with `x-zen-model` (the id asked) and
  `x-opencode-log-id` (a UUID per request). No body has a `provider` field or a `system_fingerprint`;
  `model` echoes the id asked. The body `id` has a different shape per upstream, but a shape is a
  guess where the header is a statement.
- **LangChain does not pass the headers on by default.** `response_metadata` keeps `id`, `model_name`,
  `system_fingerprint`, `service_tier` and `token_usage`; the headers arrive there only with
  `include_response_headers=True`. Keeping the endpoint on each call would be that flag plus a field.
  `LLMCall` was not touched: the decision is pending. (Taken in block T6: two fields, and that flag.)
- **What three responses cannot say** is whether one id is served by more than one endpoint, which is
  what would explain both the 551-token block and the calls without `usage`. That needs the header on
  many calls of the same id. This one `Ping` did add a third shape of `usage` for `glm-5.3-flash`: all
  counters (its `Ping` of `…063458Z`, 29 / 0 / 87), no `cached_tokens` key (this one), and no usage at
  all (the 33 above).

**Estimate with a balance** (`python -m crypto_agents.estimate var/zen-probe/20261005T043359Z --desks
var/zen-probe/20261005T233053Z --balance 24.66`; every figure an estimate, exit 1 because no line
passes). Structure + volume 0.36 to 0.71, momentum 0.11 to 0.12 and bear 0.37 are the same in every
line; the decider is the one measured over that bull's brief, at 1.2 attempts:

| bull | valid briefs | bull | decider x1.2 | total USD | top-up charged | balance required (x1.5) | 24.66 USD |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `kimi-k3` | 12 | 10.87 | 15.95 | 27.67 to 28.02 | 29.19 to 29.55 | 41.50 to 42.03 | NO PASA |
| `qwen3.8-max` | 12 | 8.05 to 8.43 | 14.54 | 23.44 to 24.17 | 24.77 to 25.54 | 35.16 to 36.26 | NO PASA |
| `deepseek-v4-pro` | 12 | 1.95 | 14.21 | 17.01 to 17.36 | 18.05 to 18.42 | 25.51 to 26.04 | NO PASA |

The cost per call is the mean of the measured rows of the files in the sources table above, plus, for
the technical roles, those of `20261005T043359Z` cited in the T3 section; the counts are the `--dry-run`
over the manifest. Three things that estimate assumes and this probe measured otherwise, none of them
corrected:

- **The decider at 1.2 attempts.** The three arms took 17, 19 and 19 attempts for 12 evaluations (1.42,
  1.58, 1.58), so the decider line, which is more than half of every total, is low by that ratio.
- **The bear at one attempt.** It took 22 for 12 (1.83).
- **Momentum on `deepseek-v4-flash`**, priced from its 12 verdicts of 04:33Z. It has not answered since
  06:26Z. The producer that did answer, `deepseek-v4-pro`, cost 0.0312 USD for 12 verdicts.

Block T5 removed the three assumptions: `estimate` reads the attempts per verdict from the probe each
role comes from, and momentum is the role map's model, `deepseek-v4-pro`. The table above is what
the old constants gave and is kept as that.

### Block T5: the decider's contract (phase 0, no spend)

Three findings of the T4 run drove the block: the decider lost 15 of 36 evaluations on one schema
failure, `.env` still sent momentum to a route that had answered 404 fifteen times, and criterion 6
would have called a run that lost its evidence to that route valid. Phase 0 read what was already on
disk before any code was written.

**(a) The schema said "optional".** pydantic 2.13.4, langchain-openai 1.5.0, langchain-core 1.5.4.
`Decision.model_json_schema()["required"]` and `Proposal`'s were both `action, confidence,
size_fraction, rationale`; `dismissed_side`, `dismissal_reason` and `invalidation_price` sat in
`properties` with `"default": null`. What the provider receives under `function_calling` is
`OpenAIBackend.complete` → `structured_runnable` → `ChatOpenAI.with_structured_output(schema,
method="function_calling", include_raw=True)` → `bind_tools([schema], tool_choice=…,
parallel_tool_calls=False)`, and the bound tool is identical to `convert_to_openai_tool(Decision)`:
same four names in `required`. `strict` does not travel, so `required` is something the model
reads, not something the upstream enforces.

**(b) Every schema failure was the same field.** `LLMCall.failure_message` of the 34 invalid decider
attempts (of 55) in `20261005T233053Z`:

| failure_kind | action | fields named | +deepseek-v4-pro | +kimi-k3 | +qwen3.8-max | total |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| schema | sell | `dismissed_side` | 7 | 3 | 6 | 16 |
| schema | buy | `dismissed_side` | 6 | 5 | 2 | 13 |
| timeout | — | `APITimeoutError: Request timed out.` | 2 | 2 | 0 | 4 |
| transport | — | `InternalServerError: Unknown Error` | 0 | 0 | 1 | 1 |

Per evaluation: schema → schema 10, schema → valid 9, valid 12, timeout 4, transport 1. **None of
the 29 names `dismissal_reason` or `invalidation_price`.** The raw text cannot be read anywhere: the
probe ran with `cache=None` and its journal holds `LLMCall` rows, not content.

**(c) The prompt already stated the rule.** `prompts/decider.md`, lines 41 to 49: each of the three
fields is described as a value "o `null`", and then "Si tu acción es `"buy"` o `"sell"`, entonces
`invalidation_price` debe tener valor, `size_fraction` debe ser mayor que 0, y tanto
`dismissed_side` como `dismissal_reason` deben estar rellenos." No template embeds the schema.
`decider_solo.md` and `decider_no_debate.md` say `invalidation_price` is "Obligatorio si actúas" and
never say "o `null`".

**(d) The bear, diagnosed and not fixed.** `minimax-m3` under `json_mode` receives no schema at all,
only `prompts/debate.md`, whose lines 31 to 33 describe `grounded_in` as "al menos un id de
observación, copiado **literalmente** de los veredictos de arriba (por ejemplo `"structure-1"`)":
it never says the field is a list and its example is a bare string. The ten invalid first attempts
are `claims.N.grounded_in: Input should be a valid array` on every claim of the brief (five errors
in nine attempts, four in one). Which type arrived is not recorded: the message carries no
`input_type` and there is no raw text.

**(e) The timeout cut the distribution.** Valid decider calls on the desktop: n = 21, median 40.9 s,
p95 101.5 s (nearest rank), maximum 109.9 s. `glm-5.2` produced 62 to 71 output tokens per second,
so 120 s is 7.4 to 8.5 k tokens and the longest valid answer had 7 483. The four timeouts were first
attempts cut at 120.0 s, ending (`LLMCall.at` is the end of the call) at 23:45:40Z
(`decider+deepseek-v4-pro`, BTC/USDT 2025-06-18T20:00Z), 23:52:01Z (`decider+kimi-k3`, SOL/USDT
2025-06-09T20:00Z), 23:52:59Z (`decider+deepseek-v4-pro`, the same SOL activation) and 23:54:01Z
(`decider+kimi-k3`, SOL/USDT 2026-04-11T20:00Z).

**What that confirms and what it does not.** The gate the block set passes by its letter: the fields
were not in `required`, and every schema failure is `dismissed_side` on a buy or a sell. Two things
it does not ask:

- **The hypothesis is not isolated.** The three fields had the same shape in the schema and only one
  was lost: all 29 failing answers carried a `dismissal_reason`, and none failed on
  `invalidation_price`. "Optional" does not by itself explain why only `dismissed_side`; what sets
  it apart is being `anyOf[enum, null]`, and that did not change.
- **If the model writes an explicit `null`, the change fixes nothing.** Required and nullable still
  admits `null`. The old validator gave the same message for a missing key and for a null one, so T4
  cannot say which it was. The new contract can: a missing key fails as `dismissed_side: Field
  required`, before the validator runs, and an explicit null as the old `la acción sell exige:
  dismissed_side`.

**What changed** (`state.py`; no validator touched). `Proposal.invalidation_price`,
`Decision.dismissed_side` and `Decision.dismissal_reason` are required and nullable, with no default:
the field always exists and is `null` on a hold. `tests/test_decision_contract.py` captures the
request body through the real `ChatOpenAI` and a mock transport — it checks the tool that goes on
the wire, not only `model_json_schema()` — and the mutation that gives `dismissed_side` its default
back fails it. Consequences:

- **The decider's `prompt_digest` did not change**: it is the digest of the prompt text, and no
  template embeds the schema.
- **The cache key did not change either**, and that is worth knowing: `cache_key` takes the schema by
  *name*, so an entry written under the old contract would be served as the answer to the new one if
  it still validates. There is none on disk (`var/cache` and `var/ablation-cache` are empty,
  `var/ablation-cache-run1` is the 2 106 unattributed files of the discarded run). Not changed.
- **`run_digest` did not change and `RUN_DIGEST_VERSION` stays `replay-v3`**: `run_digest` hashes
  `record.model_dump(mode="json")`, which already wrote the three fields as `null`. A test pins that
  a record written before serialises to the same bytes.
- **A hold that says nothing no longer validates.** A raw `hold` without the three keys was valid and
  is now three `Field required` errors. That is what "the field always exists" means, and it is
  where a cache entry written before can stop being usable: a valid one becomes a stale entry (a
  miss), and an invalid one revalidates with another error text, so its retry has another digest
  and the chain breaks at that point, loudly.
- **Old journal lines load.** No line on disk carried a non-null `decision` or `proposal`, so
  `tests/data/journal_pre_t5.jsonl` holds three lines serialised by `6e736aa` before the change
  (provenance and sha-256 in `tests/data/README.md`). A line written by hand without the keys would
  not load.
- **`Proposal` changed too, unmeasured.** `Decision` inherits the field, so the desks probe exercises
  it with `decider.md`. `Proposal` with `decider_solo.md` and `decider_no_debate.md` — the two
  templates that never say "o `null`" — goes to the second run without having been asked once under
  the new contract. Twelve `decide_solo` calls would measure it; they were not in the block.

**The timeout is 240 s in the template, as a declaration.** The data say 120 s cut the distribution;
they cannot say 240 s is enough, because the four answers that were cut are censored. At the
measured throughput 240 s admits about 16 k output tokens. The defaults in `settings.py` and
`OpenAIBackend` stay at 120.

**The T4 run cannot be repeated over its own briefs.** `zen_probe` ran without a cache, `_decide`
discarded its output and every journal line has `evidence: null`, `briefs: []`, `decision: null`:
the verdicts and the briefs were never written. Only the `prompt_digest`s remain. A technical prompt
is a pure function of the candles, so its digest can be recomputed and compared without spending;
a desk's prompt carries the text of the verdicts and cannot. The desks probe now writes
`content.jsonl` so that this does not happen twice, and a test rebuilds the decider's prompt from it
and gets the journal's digest. It is not a cache — the architecture test that forbids one in
`zen_probe.py` is untouched — and nothing reads it yet.

**Left as it was, and said.** The technical probe (`zen_probe` without `--desks`) refuses to start
with momentum on `deepseek-v4-pro`: its `PRESENT` list is the role map of block T, and that id is
also one of its structure and volume candidates. It fails naming the role. The flexible evidence
producer stays in the probe and was not ported to the pipeline: in the ablation momentum is one
fixed model. The known limit of the resume judgement — it trusts the last link to be complete —
stands, and now covers the fifth condition as well. (The technical probe was fixed in block T6: it
reads the present models from the role map.)

### Block T5 results: the decider re-measured (desktop; `var/zen-probe/`, not versioned)

One run of `zen_probe --machine desktop --desks --technicals-from var/zen-probe/20261005T043359Z
--compare-with var/zen-probe/20261005T233053Z --max-usd 1.50`: `20261006T063107Z`, 06:31:07Z to
07:01:17Z, tree clean at `fdee241`, same manifest (sha-256
`73f87870cf24d06c341ff75fa72f164f9cd5202623cf0d76babbdab4a2803f23`), `.env` with momentum on
`deepseek-v4-pro` and `CA_OPENAI__TIMEOUT_SECONDS=240`. The `--dry-run` went first. The cap's guard saw
**1.2327 USD** at the high end and refused nothing; nothing was skipped. Eleven `glm-5.3-flash` calls
came back without `usage` and are not in that figure.

**It is not the T4 briefs.** They were never written, so the evidence and the briefs were asked again.
The comparison of `prompt_digest`s, printed before the first decider call (first attempt of each
invocation, paired by `run_id`):

| arm | invocations | paired with `20261005T233053Z` | same `prompt_digest` |
| --- | ---: | ---: | ---: |
| `glm-5.3-flash@structure` | 12 | 12 | 12 |
| `glm-5.3-flash@volume` | 12 | 12 | 12 |
| `deepseek-v4-pro@momentum` | 12 | 12 | 12 |
| `kimi-k3@bull` | 12 | 12 | 0 |
| `qwen3.8-max@bull` | 12 | 12 | 0 |
| `minimax-m3@bear` | 12 | 12 | 0 |

The 36 technical questions are byte for byte the ones T4 asked; no desk prompt is, because each
carries the text of verdicts that were answered anew. So what follows compares the same activation,
not the same question.

**Every mode `Ping`** (one attempt per id; all six answered in their declared mode):

| id | role | mode | valid | failure_kind | at | latency | sha-256 of `<id>@ping.jsonl` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `kimi-k3` | bull | json_schema | yes | — | 06:31:12Z | 4.8 s | `aea19881330d3a4063ccd07e757d736c2575b3bbb4c0583c9d4f4125a9fba7c9` |
| `qwen3.8-max` | bull | json_schema | yes | — | 06:31:11Z | 2.7 s | `624adb8050aa1c3ce122ce91ddd384630b5a5fe385dee047b3502e6a40030417` |
| `minimax-m3` | bear | json_mode | yes | — | 06:31:10Z | 1.2 s | `e8b39afb7aa522aa7e0c19642bffe60095314a8579e79592bd2fd1aceb06984f` |
| `glm-5.2` | decider | function_calling | yes | — | 06:31:21Z | 12.2 s | `ab40a3f01ad1de6eb79eaee894860819c50eeb2c28723c910eed89c43bc800e7` |
| `glm-5.3-flash` | structure | json_schema | yes | — | 06:31:11Z | 2.3 s | `667d71fa0007ab3c6d322f5196d72e72b4984a4ed1cdb89edd51e8304ce17645` |
| `deepseek-v4-pro` | momentum | json_schema | yes | — | 06:31:10Z | 1.4 s | `01ad806b0675d1ca33a78343efe8223522043cc0769b9e461cc60c0a99a17c8e` |

Per (model, role): valid over asked, failures by `FailureKind`, attempts and retries per verdict, mean
latency **on the desktop**, mean tokens the provider reported (prompt / cached / completion), USD of the
arm's rows:

| arm | mode | valid | schema · context · timeout · transport | attempts · retries/verdict | latency | tokens | USD |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `glm-5.3-flash@structure` | json_schema | 12/12 | 0 · 0 · 0 · 0 | 12 · 0.00 | 30.9 s | 525 / 0 / 2030 | ≥ 0.0066 (6 of 12 without usage) |
| `glm-5.3-flash@volume` | json_schema | 12/12 | 0 · 0 · 0 · 0 | 12 · 0.00 | 34.3 s | 525 / 9 / 2572 | ≥ 0.0095 (5 of 12 without usage) |
| `deepseek-v4-pro@momentum` | json_schema | 12/12 | 0 · 0 · 0 · 0 | 12 · 0.00 | 13.3 s | 525 / 0 / 514 | 0.0324 |
| `kimi-k3@bull` | json_schema | 12/12 | 0 · 0 · 0 · 0 | 12 · 0.00 | 20.5 s | 2360 / 0 / 1301 | 0.3191 |
| `qwen3.8-max@bull` | json_schema | 12/12 | 0 · 0 · 0 · 0 | 12 · 0.00 | 34.4 s | 1859 / 0 / 3001 | 0.2606 to 0.2718† |
| `minimax-m3@bear` | json_mode | 12/12 | 12 · 0 · 0 · 0 | 24 · 1.00 | 4.3 s | 1713 / 933 / 495 | 0.0212 |
| `glm-5.2@decider+kimi-k3` | function_calling | 12/12 | 0 · 0 · 0 · 0 | 12 · 0.00 | 73.7 s | 3283 / 569 / 4494 | 0.2847 |
| `glm-5.2@decider+qwen3.8-max` | function_calling | 12/12 | 0 · 0 · 0 · 0 | 12 · 0.00 | 81.0 s | 3044 / 1821 / 4837 | 0.2816 |

† interval for cache write. By role, mode `Ping`s included (`python -m crypto_agents.consumption
var/zen-probe/20261006T063107Z`): structure ≥ 0.0066, volume ≥ 0.0095, momentum 0.0325, bull 0.5820 to
0.5932†, bear 0.0212, decider 0.5696 — **≥ 1.2215 USD, 1.2327 at the high end**. A lower bound: eleven
calls carry no usage.

**The decider, per bull candidate.** One attempt is one `LLMCall` row; "asked" is an invocation:

| bull | valid decisions | Wilson 95% | attempts | attempts per decision asked | schema · context · timeout · transport | buy · sell · hold |
| --- | --- | --- | ---: | ---: | --- | --- |
| `kimi-k3` | 12/12 | 75.7% to 100% | 12 | 1.00 | 0 · 0 · 0 · 0 | 5 · 5 · 2 |
| `qwen3.8-max` | 12/12 | 75.7% to 100% | 12 | 1.00 | 0 · 0 · 0 · 0 | 5 · 5 · 2 |

No decider attempt was invalid, so the table of failure messages is empty: not one `exige:` and not
one `Field required`. Paired by activation against `20261005T233053Z` (the same `run_id`, other
briefs; the instant is the close of the evaluated candle):

| activation | `kimi-k3` in T4 | `kimi-k3` now | `qwen3.8-max` in T4 | `qwen3.8-max` now | action now (both) |
| --- | --- | --- | --- | --- | --- |
| ADA/USDT 2024-10-23T12:00Z | valid after the retry | valid, 1st | valid after the retry | valid, 1st | sell |
| ADA/USDT 2025-11-12T00:00Z | valid, 1st | valid, 1st | valid after the retry | valid, 1st | sell |
| BNB/USDT 2025-02-04T12:00Z | lost: schema | valid, 1st | lost: schema | valid, 1st | sell |
| BNB/USDT 2026-03-19T16:00Z | valid, 1st | valid, 1st | valid, 1st | valid, 1st | sell |
| BTC/USDT 2025-06-19T00:00Z | valid, 1st | valid, 1st | lost: transport | valid, 1st | hold |
| BTC/USDT 2026-07-21T16:00Z | lost: schema | valid, 1st | valid after the retry | valid, 1st | buy |
| DOGE/USDT 2025-10-01T12:00Z | valid, 1st | valid, 1st | valid, 1st | valid, 1st | buy |
| ETH/USDT 2024-11-27T20:00Z | valid after the retry | valid, 1st | valid, 1st | valid, 1st | buy |
| ETH/USDT 2025-12-16T08:00Z | valid, 1st | valid, 1st | valid after the retry | valid, 1st | sell |
| SOL/USDT 2025-06-10T00:00Z | lost: timeout | valid, 1st | valid, 1st | valid, 1st | buy |
| SOL/USDT 2026-04-12T00:00Z | lost: timeout | valid, 1st | valid after the retry | valid, 1st | hold |
| XRP/USDT 2025-07-17T20:00Z | lost: schema | valid, 1st | valid after the retry | valid, 1st | buy |

| bull | in T4 | now | activations |
| --- | --- | --- | ---: |
| `kimi-k3` | lost: schema | valid at the first attempt | 3 |
| `kimi-k3` | lost: timeout | valid at the first attempt | 2 |
| `kimi-k3` | valid at the first attempt | valid at the first attempt | 5 |
| `kimi-k3` | valid after the retry | valid at the first attempt | 2 |
| `qwen3.8-max` | lost: schema | valid at the first attempt | 1 |
| `qwen3.8-max` | lost: transport | valid at the first attempt | 1 |
| `qwen3.8-max` | valid at the first attempt | valid at the first attempt | 4 |
| `qwen3.8-max` | valid after the retry | valid at the first attempt | 6 |

What that says, and what it does not:

- **The schema failure did not happen once.** For these two bulls T4 had 12 of 21 answered first
  attempts fail on `dismissed_side`; here it is 0 of 24. With the field in `required`, `glm-5.2`
  emitted it every time.
- **Three of the 24 valid answers took longer than 120 s**: 125.3 s (ADA/USDT 2024-10-23, 7 516
  completion tokens), 143.4 s (ETH/USDT 2024-11-27, 8 324) and 159.1 s (BTC/USDT 2025-06-19, 9 617),
  all in `decider+qwen3.8-max`. Under T4's limit they would have been timeouts, so 24 of 24 depends on
  the contract **and** on the 240 s. Valid latencies: median 68.8 s, maximum 159.1 s, at 54 to 64
  output tokens per second.
- **It is not a controlled comparison.** Against T4 the contract, the timeout, the briefs, the hour
  and the number of concurrent decider jobs (three then, two now) all changed, and the old contract
  was not run beside the new one over these prompts. `content.jsonl` now holds what that control
  would need.
- **Whether T4's 29 failures were omissions or explicit nulls stays unknown**: with no failure, there
  is no message to read.
- **A hold kept its keys.** The four holds validated: two with the three fields explicitly `null`
  (SOL/USDT 2026-04-12, both bulls) and two that named a dismissed desk anyway (BTC/USDT 2025-06-19).
  `Proposal` with the solo and no-debate templates was still not asked.
- **The two deciders proposed the same action on 12 of 12 activations** (5 buy, 5 sell, 2 hold each).
  It is one decider model reading two bulls; nothing here says which bull is better, and the block
  does not choose one.
- **The decider answers longer than it did.** Mean completion tokens 4 494 and 4 837 against 3 084 and
  2 684 in T4, so an attempt costs 0.0237 and 0.0235 USD against 0.0158 and 0.0144 then. Per decision
  asked, with T4's retries in (and its timeouts carrying no tokens), that was 0.0198 and 0.0216: one
  attempt per decision instead of 1.42 to 1.58 did not make the decider cheaper.
- **The bear needed its retry every time**: 12 of 12 first attempts invalid, all
  `claims.N.grounded_in: Input should be a valid array`, all fixed by the retry — 24 attempts for 12
  briefs, against 22 in T4. Diagnosed in phase 0, not corrected.
- **`glm-5.3-flash` returned no `usage` on 11 of its 24 verdict calls** (6 of 12 structure, 5 of 12
  volume; its `Ping` did carry usage), always both counters together, all on valid verdicts. With the
  33 of 127 of the earlier directories that is 44 of 152 answered calls.

**What identifies the upstream of the two models that fail.** Two `Ping`s through `ModelRouter` with a
backend that inherits `OpenAIBackend` and adds an httpx response hook — a scratch script, not repo
code, the same shape as T4's; `var/zen-probe/20261006T061405Z` (`kind=probe`, tree clean at `fdee241`,
06:14:05Z). Only the status and two response headers are stored in `raw/<id>.json`; nothing of the
request. Both answered valid in their declared mode at the first attempt; 0.0037 USD at the high end
of a 0.01 cap.

| id asked | role · mode | `x-opencode-endpoint-id` | `x-opencode-upstream-model-id` | tokens | USD | sha-256 of `<id>@ping.jsonl` (1 row) |
| --- | --- | --- | --- | --- | --- | --- |
| `glm-5.2` | decider · function_calling | `fireworks` | `accounts/fireworks/models/glm-5p3` | 232 / 50 / 765 | 0.0036 | `b3c4f7dabf4509002ebb71a08703f3e5347beb61e129adb26828e1b8f6faffdc` |
| `minimax-m3` | bear · json_mode | `fireworks` | `accounts/anomalyinc/routers/zen-minimax-m3` | 144 / 114 / 7 | 0.0000 | `4201d50271f9c74e3eb2eaa53e3e649de3bb7b965561d16ded5c3676bfc37e7e` |

The header for `glm-5.2` reads `glm-5p3`. It is copied as it came; whether that is another model or
the name of a route is not something one response can say. Both ids are served through the same
endpoint, which T4's three (`relace-glm5.3flash`, `together`, `inferact`) were not.

**Estimate with a balance** (`python -m crypto_agents.estimate var/zen-probe/20261005T043359Z --desks
var/zen-probe/20261005T233053Z --source momentum=var/zen-probe/20261005T233053Z --source
decider=var/zen-probe/20261006T063107Z --balance 21.83`; every figure an estimate, exit 1 because no
line passes). Structure and volume from `20261005T043359Z`; momentum (`deepseek-v4-pro`, which that
probe did not measure for this role), bull and bear from `20261005T233053Z`; the decider from the
re-measurement. Structure + volume 0.36 to 0.71, momentum 0.36 and bear 0.68 are the same in every
line:

| bull | bull | decider | total USD | top-up charged | balance required (x1.5) | 21.83 USD | total in T4 | moved by |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `kimi-k3` | 10.87 | 19.93 | 32.20 to 32.55 | 33.92 to 34.28 | 48.31 to 48.82 | NO PASA | 27.67 to 28.02 | +4.53 |
| `qwen3.8-max` | 8.05 to 8.43 | 19.71 | 29.17 to 29.90 | 30.76 to 31.51 | 43.76 to 44.84 | NO PASA | 23.44 to 24.17 | +5.73 |

The T4 column is the table of commit `4f8c877`, computed with the constants. Attempts per verdict, as
the report prints them with their n and file: structure and volume 1.00 for each candidate (12 / 12
in `20261005T043359Z`), momentum 1.00 (12 / 12, `20261005T233053Z/deepseek-v4-pro@momentum.jsonl`), bull
1.00 (12 / 12 each), **bear 1.83** (22 / 12, `20261005T233053Z/minimax-m3@bear.jsonl`), decider 1.00
(12 / 12 in each `decider+<bull>` arm of `20261006T063107Z`).

- **The estimate went up, not down.** Of the +4.53 for `kimi-k3`: decider +3.98 (19.93 against 15.95
  at the old 1.2), bear +0.31 (0.68 against 0.37), momentum +0.24 to +0.25 (0.36 against 0.11 to
  0.12). For `qwen3.8-max`: decider +5.17 (19.71 against 14.54), the same bear and momentum. The decider line is 840 calls at the cost
  of a `full` decision, an upper bound for the three arms that send it a shorter prompt. (Not a
  bound: the decider's cost is mostly output, which a shorter prompt does not limit. Block T6
  measured two of those arms and relabelled the third an estimate.)
- **`deepseek-v4-pro` still gets a line** from the T4 desks directory, where it was a bull candidate;
  it has no decider in the re-measurement, so its total is `no determinado` and it does not pass.
  (Gone in block T6: the lines come from `BULL_CANDIDATES`.)
- **With everything from the re-measurement** (`--desks var/zen-probe/20261006T063107Z --source
  momentum=var/zen-probe/20261006T063107Z`): bear 0.74 at 2.00 attempts, momentum 0.38, bull 11.17 and
  9.12 to 9.51, totals 32.58 to 32.92 and 30.32 to 31.05, required 48.87 to 49.38 and 45.48 to 46.58.
  Two measurements of 12 whose totals differ by 0.38 and 1.15 USD; neither passes.
- **The balance was read from the console** (21.83 USD, given on 2026-10-06 after the two runs of this
  block); nothing queried the network.

Sources (`var/zen-probe/20261006T063107Z/`; rows are `LLMCall` rows):

| file | rows | sha-256 |
| --- | ---: | --- |
| `glm-5.3-flash@structure.jsonl` | 12 | `6be5bf3abc0e33d4b1d3bb79935dfdd250c3492f1b41c9ca87a4a69cede3f312` |
| `glm-5.3-flash@volume.jsonl` | 12 | `69bc2707d0aa02bc3bf469928af19f074588a51095b9d742a04a568ed99cea13` |
| `deepseek-v4-pro@momentum.jsonl` | 12 | `eb7b04b2c9dd88028ef2463fabff2e1618f8cb7ae1497d84713c3c09d79eabe4` |
| `kimi-k3@bull.jsonl` | 12 | `d74bacbc1dcab364ffe7014efacb2e4066d0fce23d3b3e9231f141f489c2eb43` |
| `qwen3.8-max@bull.jsonl` | 12 | `089a2ca18c80b431d90e3d1bc4e067323008f10ec5b275acf4e7bf3f963d3a3f` |
| `minimax-m3@bear.jsonl` | 24 | `a954e6ababaa151c7690288880269ddc1a0fefb078bd88d101573d2ed931583e` |
| `glm-5.2@decider+kimi-k3.jsonl` | 12 | `59e87ac5d3d4a02903a43308fa50bdade872fd198cb5cfaf4bde407612bd641f` |
| `glm-5.2@decider+qwen3.8-max.jsonl` | 12 | `ad5175b9a2099b2534aad77f4d5f051567691b9864aa86ca4c858ecc6517856c` |
| `content.jsonl` (12 lines: verdicts, briefs, decisions) | — | `4b235a938d93b47ba1fb2735a83bbd40549f36357779fc8af221086d7557a97a` |

The six `Ping` files are in the table above. Spent by the block, at the high end of what the provider
declared: 0.0037 + 1.2327 = **1.2364 USD** of the 1.51 authorised, a lower bound by the eleven calls
without usage.

### Block T6: who answered, and what T5 left unmeasured (phase 0, no spend)

T5 left five things open: `LLMCall.model` is the id that was asked and not who answered, the two
decider branches gave the same action mix, the new contract was never measured with
`decider_solo.md` or `decider_no_debate.md`, the template and `.env` disagree on the bear's mode,
and `estimate` gave totals per bull with no way to see where the money goes. Phase 0 read what was
on disk before any code was written.

**(a) The two decider branches did not read the same thing.** The gate was: if a decider
`prompt_digest` matches between `decider+kimi-k3` and `decider+qwen3.8-max` on any activation, or
the two bull briefs are the same text, the harness is broken and T5 did not measure what it says.
It passes. `var/zen-probe/20261006T063107Z`; digests cut to 12 characters, the full ones are the
first attempt of each line of `glm-5.2@decider+kimi-k3.jsonl` (`59e87ac5…641f`) and
`glm-5.2@decider+qwen3.8-max.jsonl` (`ad5175b9…856c`). "brief" is the sha-256 of the canonical JSON
(`sort_keys`, no spaces) of that bull's brief in `content.jsonl` (`4b235a93…a97a`): computed for
this table, not a figure the repo keeps.

| activation (close) | decider digest, +kimi-k3 | +qwen3.8-max | brief, kimi-k3 | brief, qwen3.8-max | action k · q |
| --- | --- | --- | --- | --- | --- |
| ADA/USDT 2024-10-23T12:00Z | `5db7d9643565` | `a80ffe4f187d` | `cc1796924957` | `05f4b6c0c962` | sell · sell |
| ADA/USDT 2025-11-12T00:00Z | `4dff1c9ccf74` | `106c0ed37f71` | `37d524031c46` | `3399b3f85d7d` | sell · sell |
| BNB/USDT 2025-02-04T12:00Z | `f8c945b98b52` | `cde84210ec8c` | `d9804c971bc0` | `83fb10985a62` | sell · sell |
| BNB/USDT 2026-03-19T16:00Z | `877173a768cf` | `87620ccc5e6f` | `e797f3d96930` | `99880e581bc0` | sell · sell |
| BTC/USDT 2025-06-19T00:00Z | `2a2745181415` | `337063f001a7` | `70583a2eecca` | `147a8cd7561f` | hold · hold |
| BTC/USDT 2026-07-21T16:00Z | `bd1f6fc440ee` | `f616afebf447` | `c0192db131d2` | `121035acc184` | buy · buy |
| DOGE/USDT 2025-10-01T12:00Z | `b52c7646e85b` | `4d6bedcd08e2` | `4fbbf012979d` | `c10d4daf3512` | buy · buy |
| ETH/USDT 2024-11-27T20:00Z | `f3fabd28f6b8` | `b6947c0f4d04` | `1d63a5c59f8e` | `45839a964e4a` | buy · buy |
| ETH/USDT 2025-12-16T08:00Z | `e998ef76af62` | `9db5ad4108d3` | `89d02fb16d50` | `c6dfc8386e0c` | sell · sell |
| SOL/USDT 2025-06-10T00:00Z | `0b62babdac79` | `a881a78fa67f` | `02f50493155c` | `a5201b86b3b8` | buy · buy |
| SOL/USDT 2026-04-12T00:00Z | `e86eda38f025` | `abeab1762a00` | `3fd61bd0fae8` | `7bfe0d17362b` | hold · hold |
| XRP/USDT 2025-07-17T20:00Z | `a211c614b919` | `88692afe6dae` | `167065faa28b` | `2c5bac3b8422` | buy · buy |

- **0 of 12 decider digests match between branches, and 0 of 12 briefs are the same** (the thesis
  differs in 12 of 12).
- **The stored content is what the decider read.** Rebuilding `decision_prompt()` from the evidence
  and the briefs of `content.jsonl` gives the digest of the journal in 12 of 12 for each branch;
  the same holds for the bear's prompt (12 of 12) and for the three technical prompts (12 of 12 per
  dimension). So the bull's brief is inside the prompt whose digest was journaled. `zen_probe` now
  makes this check itself before it asks anything over a stored `content.jsonl`.
- **The decisions are not copies.** The `rationale` differs in 12 of 12, `size_fraction` in 10 of
  12 (the other two are the holds, at 0), and `invalidation_price` in 6 of the 10 that carry one.
- **What stays open.** Both branches share the evidence and the bear by design, and the two bulls
  give almost the same conviction (within 0.02 in 10 of 12 activations). "The decider does not care
  which bull it reads" is still not separated from "the two bulls say the same thing".

**(b) T4 lost 7 evaluations on those two bulls, not 9.** A summary of block T5 spoke of "the 9
lost"; no versioned file ever carried that figure (not this file, not the T5 `report.md`, not a
commit message, not the description of PR #13), and the tables above in this file say 3 + 2 and
1 + 1. Read from `var/zen-probe/20261005T233053Z`, by `run_id`:

| arm | activation (close) | `run_id` | attempts | cause |
| --- | --- | --- | --- | --- |
| `decider+kimi-k3` | BNB/USDT 2025-02-04T12:00Z | `c03656b8-fa32-5130-8472-b94192733cb3` | schema, schema | `la acción sell exige: dismissed_side` |
| `decider+kimi-k3` | BTC/USDT 2026-07-21T16:00Z | `6ba860c0-f563-59e7-baf0-61e30e598784` | schema, schema | `la acción buy exige: dismissed_side` |
| `decider+kimi-k3` | SOL/USDT 2025-06-10T00:00Z | `978eb8e9-63ee-595a-81cc-545270a943b8` | timeout | `APITimeoutError: Request timed out.` |
| `decider+kimi-k3` | SOL/USDT 2026-04-12T00:00Z | `223d63e2-4d3a-5aa2-b65c-164fe86ed56a` | timeout | `APITimeoutError: Request timed out.` |
| `decider+kimi-k3` | XRP/USDT 2025-07-17T20:00Z | `8363c553-2ace-5cd6-a779-e9aeb422f03e` | schema, schema | `la acción buy exige: dismissed_side` |
| `decider+qwen3.8-max` | BNB/USDT 2025-02-04T12:00Z | `c03656b8-fa32-5130-8472-b94192733cb3` | schema, schema | `la acción sell exige: dismissed_side` |
| `decider+qwen3.8-max` | BTC/USDT 2025-06-19T00:00Z | `076fcce3-a339-5e47-b0df-0553679769c4` | transport | `InternalServerError: Unknown Error` |

5 + 2 = 7: four on the schema, two timeouts, one transport. That is 7 of 12 and 10 of 12 valid, as
measured. The third arm of that run (`decider+deepseek-v4-pro`) lost 8, and the 15 of 36 quoted
above is the three arms together.

**(c) Why the technical probe refused.** `_present()` raised unless the role map declared the id it
expected, and the expected ids were written by hand: `PRESENT` carried `deepseek-v4-flash` for
momentum, and `.env` declares `deepseek-v4-pro` since T5. It failed in `subjects_of()`, before the
`--dry-run`. Two more hand-written ids sat in `all_arms()` and `chain_stage()`. It would have
broken again the day `.env` picks a bull.

**(d) Where the `deepseek-v4-pro` line of `estimate` came from.** The list of bulls was read off the
`<model>@bull` arms present in the `--desks` directory. `20261005T233053Z` holds
`deepseek-v4-pro@bull.jsonl` from when it was a candidate; the decider source of T5 has no
`decider+deepseek-v4-pro`, hence a line reading `no determinado`.

**(e) The response headers reach the message, call by call.** Read in the installed code
(langchain-openai 1.5.0, langchain-core 1.5.4, openai 2.54.0), no network:

- `ChatOpenAI.include_response_headers` (`langchain_openai/chat_models/base.py:987`) makes
  `_agenerate` (`:1975-2036`) put `dict(raw_response.headers)` in the generation info, for the two
  branches this repo uses: `chat.completions.with_raw_response.parse` (`:1988`, taken by
  `json_schema` and `json_mode`, which carry a `response_format`) and `with_raw_response.create`
  (`:2016`, taken by `function_calling`).
- `langchain_core/language_models/chat_models.py:2184-2186` merges the generation info into
  `message.response_metadata`, and `with_structured_output(..., include_raw=True)`
  (`base.py:2586-2594`) returns that same message as `raw` in the three modes.
- So the headers travel inside the message their own response produced. No httpx hook is involved,
  and none could tell which call a response belongs to: that is how the headers were looked at by
  hand in T4 and T5, and it is the mutation the concurrency test kills.
- **One hole, in `json_schema` only.** With a Pydantic class as `response_format` the SDK validates
  inside `raw_response.parse()` (`openai/lib/_parsing/_completions.py:243-245`). Content that does
  not validate raises there and no message exists: the path `OpenAIBackend.complete` already caught,
  and where the usage is lost. LangChain hangs the HTTP response on the exception before re-raising
  (`base.py:2024-2026`), so the two headers are read from it. It is the response of that call.

**What changed.**

- **`LLMCall.upstream_model` and `LLMCall.upstream_endpoint`**, nullable, copied from
  `x-opencode-upstream-model-id` and `x-opencode-endpoint-id` without interpreting them. The reason
  is the header captured in `var/zen-probe/20261006T061405Z`: asked for `glm-5.2`, the gateway
  answered through `fireworks` with `accounts/fireworks/models/glm-5p3`. The id asked does not say
  who answered, and the route can change with the id unchanged; then two arms of a run compare
  different models and nothing records it. T4's upstream was never captured, which is why T5's 24
  of 24 against T4's 29 failures cannot separate four causes: the contract, the timeout, the briefs
  and the upstream model.
- **Only `llm.py` reads response headers, and it reads two.** `Completion.upstream` carries two
  names; the header dictionary never leaves the adapter, so there is no field through which a
  cookie or a credential could reach a journal. Both halves are architecture tests.
- **It is a property of the attempt, not of the evaluation**: a verdict can start on one route and
  retry on another. `None` means there was no header to read: a provider that sends none, a local
  backend, a rejection before content, or a line written before the field.
- **A provider failure carries none.** There is no `Completion`, and the router does not read the
  provider's exception for it. `audit` counts those as `sin respuesta`, apart from `sin cabecera`
  (it answered and did not say).
- **The usage is still lost on the `ValidationError` path.** Recovering it from the same response
  was left out of this block. It means an invalid attempt under `json_schema` says who answered and
  not what it cost. (Recovered in block T7.)
- **The cache stores it, outside the key.** A hit reports the upstream of the stored entry with
  `cache_hit=True`; an entry written before the field gives `None`.
- **`run_digest` omits the two fields when they are `None`**, like the tokens, so `replay-v3` did not
  move: `tests/data/llmcall_pre_t6.jsonl` (serialised by `60b09bc`) hashes to the digest it had.
- **`audit` prints who answered** per (role, requested model), over the run and arm by arm. Two rows
  for one pair in the first table say the route changed; the second says in which arm.
- **`zen_probe --deciders-from <dir>`** measures `decide_solo` and `decide_without_debate` over the
  activations of a desks probe: `solo` reads no evidence, `no_debate` reads the verdicts stored in
  that probe's `content.jsonl`. **`zen_probe --bear-from <dir> --bear-mode <mode>`** asks the bear
  the same prompt in another structured-output mode; `.env` is not touched. Both recompute the
  `prompt_digest`s and refuse before the first call unless they are the ones the source asked, both
  refuse a source that ran another plan, and neither asks for a technical verdict again. The arms
  are `<decider>@solo`, `<decider>@no_debate` and `<bear>@bear`; not `decider+…`, which `estimate`
  reads as a decider conditioned to a bull.
- **`metrics.schema_faults()` tells two failures apart**: `field: Field required` is a key the model
  left out, and `la acción buy exige: field` is a key that came empty on an action that needs it.
  T4 could not make that distinction; the contract of T5 is what makes it readable.
- **`estimate` breaks the total down by arm and role**: paid calls, measured attempts, the USD range
  and the file each figure comes from. The cells of an arm add up to that arm and all of them to
  the total; the x1.5 verdict is judged on the same total as before. The bull lines come from
  `candidates.BULL_CANDIDATES`.
- **A decider arm can bring its own measurement**: `--source decider@solo=<dir>` and
  `--source decider@no_debate=<dir>` read the arms `<model>@solo` and `<model>@no_debate` of a
  deciders probe, with their own cost per attempt and their own attempts, for that arm and no
  other. Only the decider admits a source per arm: it is the one role each arm asks a different
  question. Without one, the decider of `solo`, `no_debate` and `bull_only` is priced with the
  prompt of `full` and labelled `estimación con el prompt de full`. It was first labelled an upper
  bound, and it is not one: the decider's cost is mostly output, and a shorter prompt does not
  bound what the model writes. The total of a role is the sum of its cells, since a role can now
  cost differently per arm; with no source per arm every figure is what it was.
- **The technical probe reads the present models from the role map** (`PRESENT_ROLES`). An id that
  is both present and a candidate — `deepseek-v4-pro` carries momentum and is a candidate for
  structure and volume — is pinged once, in its role of the map, and keeps its candidate arms:
  dropping it from the list would be choosing a model from inside the probe.

**Not verified, and written down because it came out of (e).** `glm-5.3-flash` returned no `usage`
on 44 of 152 answered calls, all of them under `json_schema`, and 43 of those validated. The path
above loses the usage exactly when the content is not bare JSON and `json_payload()` rescues it
afterwards, which would produce that same row: valid, no counters. Nothing on disk confirms or
refutes it — the probes ran with no cache and kept no raw text. (Block T7 reproduced the mechanism
offline and reads the usage on that path; the next `glm-5.3-flash` call says which it was.)

**Seen and left alone.** The operator's `.env` (2026-10-06) still carries Go figures on a `payg`
map: momentum has weight 2.0 and 63 300 on `deepseek-v4-pro`, the decider has 880, and `bull` is
still `kimi-k2.6`, which answers 410. The probes do not notice (`_unmetered`); the second run will.

### Block T6 results: the deciders without desks, the bear's mode, and who answered (desktop; `var/zen-probe/`, not versioned)

Two runs over the `content.jsonl` of the T5 desks probe (`var/zen-probe/20261006T063107Z`,
`content.jsonl` sha-256 `4b235a938d93b47ba1fb2735a83bbd40549f36357779fc8af221086d7557a97a`), tree clean
at `b99aa9a`, same manifest (sha-256
`73f87870cf24d06c341ff75fa72f164f9cd5202623cf0d76babbdab4a2803f23`), `.env` with momentum on
`deepseek-v4-pro`, `CA_OPENAI__TIMEOUT_SECONDS=240` and the bear still on `json_mode`. The `--dry-run`
of each went first. Joint cap 0.75 USD, split 0.65 + 0.05:

| run | order | span (UTC) | cap | guard saw, high end | refused | calls without usage |
| --- | --- | --- | --- | --- | --- | --- |
| `20261006T235701Z` | `--deciders-from …063107Z --max-usd 0.65` | 23:57:01Z to 00:11:05Z | 0.65 | 0.4164 | 0 | 0 of 24 |
| `20261006T235710Z` | `--bear-from …063107Z --bear-mode json_schema --max-usd 0.05` | 23:57:10Z to 23:58:02Z | 0.05 | 0.0125 | 0 | 0 of 12 |

Spent: **0.4289 USD** of the 0.75 authorised, every call measured. Nothing was cut, so there is no
`k/N` table.

**The questions were the source's, checked before the first call.** Each prompt was rebuilt from
today's candles and what `content.jsonl` holds, and its digest compared with the first attempt of the
source journal. No technical verdict was asked again.

| run | input | activations | with that input in the source | same `prompt_digest` |
| --- | --- | ---: | ---: | ---: |
| `…235701Z` | structure | 12 | 12 | 12 |
| `…235701Z` | momentum | 12 | 12 | 12 |
| `…235701Z` | volume | 12 | 12 | 12 |
| `…235701Z` | decider+kimi-k3 | 12 | 12 | 12 |
| `…235701Z` | decider+qwen3.8-max | 12 | 12 | 12 |
| `…235710Z` | bear | 12 | 12 | 12 |

The two `decider+…` rows are the source decider's prompt rebuilt from the stored evidence and
briefs: they tie the *text* of the evidence `no_debate` read to what T5's decider read, where the
three technical rows only tie the question.

Per arm: valid over asked, failures by `FailureKind`, attempts and retries per verdict, mean latency
**on the desktop**, mean tokens the provider reported (prompt / cached / completion), USD of the arm's
rows:

| arm | mode | valid | schema · context · timeout · transport | attempts · retries/verdict | latency | tokens | USD |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `glm-5.2@solo` | function_calling | 12/12 | 0 · 0 · 0 · 0 | 12 · 0.00 | 56.8 s | 1155 / 552 / 3084 | 0.1747 |
| `glm-5.2@no_debate` | function_calling | 12/12 | 0 · 0 · 0 · 0 | 12 · 0.00 | 70.2 s | 2303 / 598 / 4000 | 0.2417 |
| `minimax-m3@bear` | json_schema | 12/12 | 0 · 0 · 0 · 0 | 12 · 0.00 | 4.3 s | 1662 / 210 / 495 | 0.0125 |

**The deciders without desks, under the contract of T5** (schema `Proposal`; one attempt is one
`LLMCall` row):

| arm | template | valid proposals | Wilson 95% | attempts | attempts per proposal asked | schema · context · timeout · transport | buy · sell · hold |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| `solo` | `decider_solo.md` | 12/12 | 75.7% to 100% | 12 | 1.00 | 0 · 0 · 0 · 0 | 5 · 3 · 4 |
| `no_debate` | `decider_no_debate.md` | 12/12 | 75.7% to 100% | 12 | 1.00 | 0 · 0 · 0 · 0 | 5 · 2 · 5 |

No attempt was invalid, so the table of failure messages is empty and so is the list of failing
ids. **The gate of the block: no attempt failed with `Field required` on `invalidation_price`,
`dismissed_side` or `dismissal_reason`** (0 of 24), and none with the validator's `exige:` either. No
template was touched.

By activation, next to what the full pipeline decided over the same evidence in T5 (both branches
gave the same action there):

| activation (close) | `full` in T5 | `solo` | `no_debate` |
| --- | --- | --- | --- |
| ADA/USDT 2024-10-23T12:00Z | sell | hold | hold |
| ADA/USDT 2025-11-12T00:00Z | sell | hold | hold |
| BNB/USDT 2025-02-04T12:00Z | sell | sell | sell |
| BNB/USDT 2026-03-19T16:00Z | sell | sell | sell |
| BTC/USDT 2025-06-19T00:00Z | hold | hold | hold |
| BTC/USDT 2026-07-21T16:00Z | buy | buy | buy |
| DOGE/USDT 2025-10-01T12:00Z | buy | buy | buy |
| ETH/USDT 2024-11-27T20:00Z | buy | buy | buy |
| ETH/USDT 2025-12-16T08:00Z | sell | sell | hold |
| SOL/USDT 2025-06-10T00:00Z | buy | buy | buy |
| SOL/USDT 2026-04-12T00:00Z | hold | hold | hold |
| XRP/USDT 2025-07-17T20:00Z | buy | buy | buy |

What that says, and what it does not:

- **The two templates that never say "o `null`" did not trip on the new contract.** 24 of 24 at the
  first attempt. The nine holds (four in `solo`, five in `no_debate`) all carry
  `invalidation_price: null` and `size_fraction: 0`: the key the contract made required is there.
- **It is 12 activations per arm.** 12 of 12 is compatible with a true rate anywhere from 75.7% up.
  It says the harness does not charge `solo` a retry on every call, which is what would have biased
  the central comparison; it does not say it never will.
- **`solo` agrees with `full` on 10 of 12 actions, `no_debate` on 9 of 12, and the two with each
  other on 11 of 12.** Where they differ, the arm without desks held and the full pipeline sold
  (both ADA activations; ETH 2025-12-16 for `no_debate` only). That is a description of twelve
  rows, not a result: no outcome was scored, and the block compares no arm.
- **One valid answer took longer than 120 s**: `no_debate` on ETH/USDT 2024-11-27, 134.0 s, 7 613
  completion tokens. Under the old limit it would have been a timeout. The rest: `solo` median
  51.2 s, maximum 96.2 s; `no_debate` median 52.3 s.
- **The decider came out cheaper without desks.** Per proposal: `solo` 0.01456 USD and `no_debate`
  0.02014, against 0.02372 and 0.02347 per decision of `full` in T5. Over 140 calls that is 2.04
  and 2.82 where `estimate` budgeted 3.32 (or 3.29) for each of those two arms. `estimate` now
  reads them (`--source decider@solo=…`, see the estimate below); `bull_only` is still unmeasured.

**The bear, same prompt, another mode.** Model, prompt and evidence are the source's; only
`structured_output` changes:

| | mode | first attempts valid | valid briefs | attempts | attempts per brief | schema · context · timeout · transport |
| --- | --- | --- | --- | ---: | ---: | --- |
| source (`…063107Z`) | json_mode | 0/12 | 12/12 | 24 | 2.00 | 12 · 0 · 0 · 0 |
| now (`…235710Z`) | json_schema | 12/12 | 12/12 | 12 | 1.00 | 0 · 0 · 0 · 0 |

- **12 of 12 valid at the first attempt** (Wilson 75.7% to 100%), against 0 of 12 (0% to 24.3%) with
  `json_mode` on the same twelve prompts. Every one of the twelve first attempts that failed in the
  source with `claims.N.grounded_in: Input should be a valid array` validated here; there is no
  failing id to list.
- **Per brief it costs 0.00104 USD against 0.00177** in the source (two attempts) and 0.00161 in T4:
  one call instead of two. The provider reported usage on 12 of 12, so the path where the SDK
  rejects the content and the usage is lost did not occur once.
- **The template stays on `json_schema`** — it already declared it — and a test now pins it as the
  measured mode. The operator's `.env` still says `json_mode`; the line to change by hand is
  `CA_ROLES__BEAR__PRIMARY__STRUCTURED_OUTPUT=json_schema`. Until then template and `.env` differ.
- **What it does not say.** Whether the schema was enforced upstream or the model simply followed
  it: the request carried it, and nothing here distinguishes the two. And it is Zen: `json_schema`
  for `minimax-m3` against Go has not been asked.

**Who answered**, over every call of the two runs, from `LLMCall.upstream_model` (`python -m
crypto_agents.audit <dir>` prints it):

| role | requested model | `upstream_model` | `upstream_endpoint` | attempts | without header | no response |
| --- | --- | --- | --- | ---: | ---: | ---: |
| decider | `glm-5.2` | `accounts/fireworks/models/glm-5p3` | `fireworks` | 24 | 0 | 0 |
| bear | `minimax-m3` | `accounts/anomalyinc/routers/zen-minimax-m3` | `fireworks` | 12 | 0 | 0 |

- **One upstream per requested model, on every attempt**, and the same two the single `Ping`s of
  `20261006T061405Z` showed. Over the fourteen minutes of the run the route did not move.
- **The header arrived in the two modes asked** (`function_calling` and `json_schema`).
- **It says nothing about T4.** Whether `glm-5.2` was `glm-5p3` on 2026-10-05, when it failed 29
  times on `dismissed_side`, was not recorded and cannot be recovered. From here on a change of
  route shows up as a second row.

**Estimate with a balance** (`python -m crypto_agents.estimate var/zen-probe/20261005T043359Z --desks
var/zen-probe/20261005T233053Z --source momentum=var/zen-probe/20261005T233053Z --source
decider=var/zen-probe/20261006T063107Z --source decider@solo=var/zen-probe/20261006T235701Z --source
decider@no_debate=var/zen-probe/20261006T235701Z --balance 21.40`; every figure an estimate, exit 1
because no line passes). The sources are T5's — structure and volume from `20261005T043359Z`,
momentum, bull and bear from `20261005T233053Z`, the decider of `full` from `20261006T063107Z` — plus
the two per arm. Structure + volume 0.36 to 0.71, momentum 0.36 and bear 0.68 are the same in every
line:

| bull | bull | decider | total USD | top-up charged | balance required (x1.5) | 21.40 USD | total without the per-arm sources |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `kimi-k3` | 10.87 | 18.14 | 30.42 to 30.76 | 32.06 to 32.42 | 45.63 to 46.14 | NO PASA | 32.20 to 32.55 |
| `qwen3.8-max` | 8.05 to 8.43 | 18.00 | 27.46 to 28.18 | 28.97 to 29.72 | 41.19 to 42.27 | NO PASA | 29.17 to 29.90 |

The last column is the same command without the two `decider@…` sources, which is T5's table to the
cent. By arm, with the measured attempts (the four baselines are 0.00 by rule):

| arm | `kimi-k3` | `qwen3.8-max` | what it pays |
| --- | --- | --- | --- |
| `full` | 7.90 to 8.24 | 6.92 to 7.39 | three technicals, both desks, decider |
| `local_technicals` | 7.17 | 6.20 to 6.32 | both desks, decider; the technicals are local |
| `bull_only` | 6.95 | 5.97 to 6.10 | bull, decider; the technicals come from the cache |
| `local_bull` | 3.55 | 3.51 | bear, decider; the bull is local |
| `no_debate` | 2.82 | 2.82 | decider, from its own rows |
| `solo` | 2.04 | 2.04 | decider, from its own rows |
| **total** | 30.42 to 30.76 | 27.46 to 28.18 | |

And the decider, the role that weighs most, cell by cell (`glm-5.2`, 1.00 attempts per call in
every arm):

| arm | paid calls | USD per attempt | USD | source of the cost |
| --- | --- | ---: | ---: | --- |
| `full` | ≤ 140 | 0.02372 · 0.02347 | 3.32 · 3.29 | `20261006T063107Z/glm-5.2@decider+<bull>.jsonl` |
| `local_technicals` | ≤ 140 | 0.02372 · 0.02347 | 3.32 · 3.29 | the same: it sends the prompt of `full` |
| `local_bull` | ≤ 140 | 0.02372 · 0.02347 | 3.32 · 3.29 | the same |
| `bull_only` | ≤ 140 | 0.02372 · 0.02347 | 3.32 · 3.29 | the same, labelled `estimación con el prompt de full` |
| `no_debate` | ≤ 140 | 0.02014 | 2.82 | `20261006T235701Z/glm-5.2@no_debate.jsonl`, 12 rows |
| `solo` | 140 | 0.01456 | 2.04 | `20261006T235701Z/glm-5.2@solo.jsonl`, 12 rows |

(Two figures in a cell are `kimi-k3` · `qwen3.8-max`: the decider of `full` is conditioned on the
bull whose brief it reads.)

- **The balance does not cover either line.** 21.40 USD is under the total itself, before any
  margin. To pass at x1.5 it would take 46.14 (`kimi-k3`) or 42.27 (`qwen3.8-max`) at the high end:
  24.74 or 20.87 more than there is.
- **Where the money is.** The decider is 18.14 of 30.42 to 30.76, and 18.00 of 27.46 to 28.18: six
  arms pay it. The bull is next (10.87, or 8.05 to 8.43, over three arms). Structure, volume,
  momentum and bear together are under 2 USD.
- **The two measured arms moved the total by under 2 USD.** `solo` went from 3.32 (or 3.29) to 2.04
  and `no_debate` to 2.82; nothing else changed.
- **`bull_only` is priced with the prompt of `full` and says so.** It was labelled an upper bound,
  and that was wrong: the decider's cost is mostly output — 3 084 to 4 837 completion tokens against
  1 155 to 3 283 of prompt in these runs — and a shorter prompt does not bound what the model
  writes. `solo` and `no_debate` came out cheaper; that is what their rows say, not something the
  prompt guaranteed.
- **The bear is still budgeted at 1.83 attempts**, from the `json_mode` rows of T4, because those are
  the sources asked for. With `--source bear=var/zen-probe/20261006T235710Z` (`json_schema`, 1.00
  attempts) the bear line is 0.44 instead of 0.68 and the totals 30.18 to 30.52 and 27.22 to 27.94:
  it changes nothing about the verdict.
- **The balance was read from the console** (21.40 USD, given on 2026-10-06 after the two runs of
  this block; it is the 21.83 of T5 less the 0.4289 spent). Nothing queried the network.

**No other header left the adapter.** Searched, with no spend, in every file of the two directories
(`content.jsonl`, `findings.json`, `meta.json`, `report.md` and the journals: six files and five), without
regard to case: `set-cookie` 0, `authorization` 0, `x-request-id` 0, against `upstream_model` 12 per
journal. Two tests now do it on purpose: a simulated response carrying those three headers goes
through the bear probe and the deciders probe under the real adapter (`ChatOpenAI` and the SDK over a
mock transport) and through a router with a cache on disk, in the three modes and on the path where
the SDK rejects the answer, and no file written holds a name or a value of the three.

**For the next block, not done here:** recovering the `usage` from `error.response` on the
`ValidationError` path of `json_schema`, with its mutation. Today that attempt says who answered
and not what it cost, and it would also say whether the 44 `glm-5.3-flash` calls without usage were
the adapter's loss or the provider's. (Done in block T7, C2.)

Sources (rows are `LLMCall` rows; `python -m crypto_agents.consumption <dir>` recomputes each USD):

| file | rows | sha-256 |
| --- | ---: | --- |
| `20261006T235701Z/glm-5.2@solo.jsonl` | 12 | `349f08fbf806d31757c3313c9ffb49f06ebfe539dbfe3e8ef8f335e3ff9f1064` |
| `20261006T235701Z/glm-5.2@no_debate.jsonl` | 12 | `aa2f11f13bbe60139a0d25282d877bb49d718cfda2e16b8f71d8d1a640ee230a` |
| `20261006T235701Z/content.jsonl` (12 lines: evidence read, proposals) | — | `9aa79f8f19cdf6262155ed1b538ae79b450d14731ef963bee489d5070fe2f063` |
| `20261006T235710Z/minimax-m3@bear.jsonl` | 12 | `4be3d89787ac2d4b18d73dff0c870cdcd9c376eb580b62c70fa236e99441e5dd` |
| `20261006T235710Z/content.jsonl` (12 lines: evidence read, briefs) | — | `4ad802b6e48ddd702e4b5962f9aaa843d114907b129fca88cee563a7a302cb04` |

`criteria` refuses both directories (`kind=probe`).

### Block T7: what brakes a run under pay as you go, the two desks, and amendment 3 (no spend)

The dispersion module said the six-arm stage 1 would cost 27 to 31 USD to come out «no
concluyente» almost for certain (detectable 1.03–1.46 % at n = 140 against δ = 0.20 %; the half-width
stays above δ at n = 420 too). Amendment 3 (`docs/ablation.md`, dated 2026-10-07, written before any
stage-1 call) cuts stage 1 to `full`, `solo` and the four baselines, and makes the rest depend on
one verdict. The block is what had to be true before launching that: nothing in it calls a model.

**Phase 0 (a): the desks are asymmetric the other way round.** The bear costs 12 to 22 times less
per brief than the bull, and the worry was a poor desk against a rich one. Read from
`var/zen-probe/20261006T063107Z/content.jsonl` (sha-256
`4b235a938d93b47ba1fb2735a83bbd40549f36357779fc8af221086d7557a97a`; both bulls, bear in `json_mode`)
and `20261006T235710Z/content.jsonl`
(`4ad802b6e48ddd702e4b5962f9aaa843d114907b129fca88cee563a7a302cb04`; bear in `json_schema`), the same
evidence in all 12 activations of both. **Descriptive, n = 12, no tests.** Each cell is `claims ·
weak/moderate/strong · conviction · len(thesis)/len(strongest_counterargument) · distinct ids in
grounded_in`:

| activation (close) | ids available | bull `qwen3.8-max` | bear `json_mode` (T5) | bear `json_schema` (T6) |
| --- | ---: | --- | --- | --- |
| ADA/USDT 2024-10-23T12:00Z | 14 | 4 · 1/3/0 · 0.34 · 130/140 · 5 | 5 · 1/3/1 · 0.62 · 207/351 · 7 | 4 · 1/2/1 · 0.62 · 196/356 · 7 |
| ADA/USDT 2025-11-12T00:00Z | 13 | 3 · 0/3/0 · 0.34 · 137/167 · 7 | 5 · 1/2/2 · 0.68 · 150/272 · 5 | 5 · 1/2/2 · 0.68 · 294/465 · 10 |
| BNB/USDT 2025-02-04T12:00Z | 14 | 4 · 2/2/0 · 0.32 · 205/160 · 7 | 5 · 0/3/2 · 0.78 · 183/291 · 9 | 5 · 0/2/3 · 0.78 · 205/334 · 10 |
| BNB/USDT 2026-03-19T16:00Z | 14 | 3 · 2/1/0 · 0.35 · 167/139 · 5 | 5 · 1/2/2 · 0.72 · 166/326 · 10 | 4 · 0/2/2 · 0.72 · 174/330 · 10 |
| BTC/USDT 2025-06-19T00:00Z | 13 | 4 · 3/1/0 · 0.34 · 171/191 · 5 | 5 · 0/3/2 · 0.62 · 190/300 · 8 | 5 · 0/3/2 · 0.62 · 192/333 · 6 |
| BTC/USDT 2026-07-21T16:00Z | 14 | 4 · 0/2/2 · 0.67 · 153/154 · 9 | 5 · 0/3/2 · 0.62 · 162/202 · 5 | 4 · 0/2/2 · 0.62 · 210/334 · 8 |
| DOGE/USDT 2025-10-01T12:00Z | 14 | 3 · 0/1/2 · 0.62 · 108/146 · 5 | 5 · 1/2/2 · 0.62 · 170/236 · 10 | 5 · 1/2/2 · 0.62 · 186/312 · 8 |
| ETH/USDT 2024-11-27T20:00Z | 12 | 3 · 0/1/2 · 0.68 · 136/153 · 6 | 5 · 1/2/2 · 0.62 · 242/287 · 9 | 4 · 1/2/1 · 0.42 · 222/251 · 8 |
| ETH/USDT 2025-12-16T08:00Z | 13 | 2 · 0/2/0 · 0.30 · 158/144 · 2 | 5 · 0/3/2 · 0.72 · 239/410 · 6 | 5 · 1/2/2 · 0.72 · 224/328 · 9 |
| SOL/USDT 2025-06-10T00:00Z | 14 | 3 · 0/1/2 · 0.64 · 139/138 · 3 | 4 · 0/2/2 · 0.62 · 275/267 · 8 | 5 · 1/2/2 · 0.62 · 209/309 · 9 |
| SOL/USDT 2026-04-12T00:00Z | 14 | 4 · 0/3/1 · 0.58 · 183/140 · 6 | 5 · 2/2/1 · 0.55 · 205/327 · 5 | 5 · 1/3/1 · 0.62 · 206/328 · 5 |
| XRP/USDT 2025-07-17T20:00Z | 14 | 4 · 0/1/3 · 0.78 · 136/151 · 4 | 5 · 1/2/2 · 0.45 · 227/295 · 10 | 4 · 1/2/1 · 0.45 · 205/352 · 9 |

| desk | claims: total (mean; range) | weak/moderate/strong | conviction: mean (median; range) | len thesis | len counter | distinct ids |
| --- | --- | --- | --- | --- | --- | --- |
| bull `qwen3.8-max` | 41 (3.42; 2–4) | 8/21/12 (20/51/29 %) | 0.497 (0.46; 0.30–0.78) | 152 (146; 108–205) | 152 (148; 138–191) | 5.33 (5; 2–9) |
| bear `minimax-m3` `json_mode` | 59 (4.92; 4–5) | 8/29/22 (14/49/37 %) | 0.635 (0.62; 0.45–0.78) | 201 (198; 150–275) | 297 (293; 202–410) | 7.67 (8; 5–10) |
| bear `minimax-m3` `json_schema` | 55 (4.58; 4–5) | 8/26/21 (15/47/38 %) | 0.624 (0.62; 0.42–0.78) | 210 (206; 174–294) | 336 (332; 251–465) | 8.25 (8.5; 5–10) |
| *context:* bull `kimi-k3` (T5) | 53 (4.42; 4–5) | 13/28/12 (25/53/23 %) | 0.516 (0.54; 0.30–0.78) | 251 (256; 205–303) | 456 (454; 332–572) | 6.92 (6.5; 4–10) |

Bull against bear per activation (lower / equal / higher than the bear):

| | `qwen3.8-max` vs `json_mode` | `qwen3.8-max` vs `json_schema` | *context:* `kimi-k3` vs `json_schema` |
| --- | --- | --- | --- |
| claims | 12 / 0 / 0 | 9 / 3 / 0 | 4 / 6 / 2 |
| len thesis | 10 / 0 / 2 | 11 / 1 / 0 | 3 / 0 / 9 |
| len counter | 12 / 0 / 0 | 12 / 0 / 0 | 2 / 0 / 10 |
| distinct ids | 9 / 0 / 3 | 10 / 0 / 2 | 7 / 4 / 1 |
| conviction | 6 / 1 / 5 | 7 / 1 / 4 | 7 / 1 / 4 |

The decider on those briefs, branch `decider+qwen3.8-max` of T5:

| activation | action | `dismissed_side` | coherent | bull conviction | bear conviction (`json_mode`) |
| --- | --- | --- | --- | ---: | ---: |
| ADA 2024-10-23 | sell | bull | yes | 0.34 | 0.62 |
| ADA 2025-11-12 | sell | bull | yes | 0.34 | 0.68 |
| BNB 2025-02-04 | sell | bull | yes | 0.32 | 0.78 |
| BNB 2026-03-19 | sell | bull | yes | 0.35 | 0.72 |
| BTC 2025-06-19 | hold | bull | n/a | 0.34 | 0.62 |
| BTC 2026-07-21 | buy | bear | yes | 0.67 | 0.62 |
| DOGE 2025-10-01 | buy | bear | yes | 0.62 | 0.62 |
| ETH 2024-11-27 | buy | bear | yes | 0.68 | 0.62 |
| ETH 2025-12-16 | sell | bull | yes | 0.30 | 0.72 |
| SOL 2025-06-10 | buy | bear | yes | 0.64 | 0.62 |
| SOL 2026-04-12 | hold | `null` | n/a | 0.58 | 0.55 |
| XRP 2025-07-17 | buy | bear | yes | 0.78 | 0.45 |

- **The cheap desk is the one that writes more.** More claims, a thesis about 35 % longer, a
  counterargument twice as long, about 50 % more observation ids cited. The terse desk is the bull
  `qwen3.8-max`.
- **The bull's price is not in the brief.** Mean visible text (thesis + counterargument + claims)
  against mean `completion_tokens` of T5 and T6: bull `qwen3.8-max` 763 characters for 3 001 tokens,
  bear `json_schema` 1 377 for 495, bull `kimi-k3` 1 583 for 1 301. What `qwen3.8-max` bills is
  reasoning that never reaches the `DebateBrief`.
- **The bear's conviction is almost fixed, and the decider's action follows the bull's** (n = 12, an
  observation, not a result). The bear says 0.62 in 6 of 12 under each mode, and the same figure in
  both runs in 10 of 12. The bull's is bimodal: 0.30–0.35 in six activations, 0.58–0.78 in the other
  six. Bull ≤ 0.35 → sell or hold (6 of 6); bull ≥ 0.58 → buy or hold (6 of 6). The decider dismisses
  the bull in 6 (5 sell, 1 hold), the bear in 5 (5 buy) and nobody in 1; coherent in 10 of 10
  actionable decisions. No activation has the decider selling against a convinced bull or buying
  against one that gave up. The branch `decider+kimi-k3` gives the same counts.
- These are surface measures — lengths and counts — and say nothing about the quality of an
  argument.

**Decisions taken with that in view (2026-10-07), none by the code.** Bear stays `minimax-m3` in
`json_schema`: the terse desk is the bull, so another bear would not fix the asymmetry, and
`kimi-k3` was not probed as bear. Bull is `qwen3.8-max`. structure and volume are `glm-5.3-flash`
in stage 1, which leaves three of six roles in the zhipu family and two of the three technical
readings on one model — the only candidate whose evidence the desks and the decider ever read.

**Found on the way, and it blocked stage 1.** The operator's `.env` still sent structure to
`mimo-v2.5` and volume to `hy3`, the Go ids. Neither was ever called on Zen (no row in the twelve
directories of `var/zen-probe/`), neither has a `payg` price row, and with them `full` would have
lost all 140 evaluations at structure. T6's "seen and left alone" did not list them.

**What brakes a run under `payg` (1-A).** `QuotaLedger.resolve()` knows nothing about billing, so
with `CA_BILLING=payg` and Go figures in `.env` the ledger applied 880 to the decider. That does not
bind stage 1 (at most 280 decider calls) and would bind any larger one.

- **The rule lives in the router.** `ModelRouter._choose`: under `payg` a remote primary is
  returned without asking the ledger. `_record()` still calls `ledger.record()`: the counter
  records, it does not brake. A local primary — the `local_*` arms — still goes through the ledger,
  and with `go` nothing changes. The ledger still holds no `Settings`; an architecture test pins
  that `llm.py` is the only module that asks it whom to call.
- **Consequence: under `payg` the local fallback of a remote role never activates by quota**, and it
  never did on a provider rejection. That is why amendment 3 defers the two local arms: today they
  measure a path the operation does not take.
- **What brakes is a dollar cap, and there is one class.** `SpendGuard` moved from `zen_probe.py`
  to `spend.py`, because the ablation cannot import the probe. `spend_guard()` is the one door
  through which the ablation and the runner build it: it refuses `billing != payg`, and it refuses
  a remote role whose model has no `payg` price row, because a call without a price adds nothing
  to the spend and the cap would be blind to that role.
- **The router asks once per invocation, right before the first live remote attempt** (`_admit`). A
  cache hit does not ask, so a resume over a warm cache cannot be cut; a retry of an invocation
  already open is not refused, or a paid invalid attempt would lose its correction; a local call
  never asks. Every live attempt enters the guard in `_record()`.
- **A refusal is `SpendCapReachedError`**, a `ModelInvocationError` so that the six nodes catch it
  untouched and so that it carries the attempts that invocation had already replayed from the
  cache. It never carries a paid one. The message reads `tope de gasto alcanzado`, which
  `AbortKind.SPEND_CAP` reads.
- **`ablation --fill` requires `--max-usd` under `payg`** and refuses it under `go`; without
  `--fill` it is refused too, there being nothing to cap. One guard for every arm, required by the
  signature of `run_arm` like the ledger. `crypto-agents run --max-usd X` is optional.
- **Criterion 6, sixth condition** (written in amendment 3): an evaluation cut by the cap, at any
  node and arm, invalidates the run, judged on the last link of a resumed chain like the others.
- **What the cap does not guarantee.** There is no cost bound before calling, so the invocation that
  crosses the cap finishes, like the ones in flight (up to 3 in the technical fan-out, 2 at the
  desks; in the runner, times `max_concurrent`). Calls without `usage` add nothing.
- **Two tests of block T changed meaning.** In `tests/test_zen_payg.py`, "the ledger never
  degrades a remote role" now also asks the router; "the decider left at the Go figure is caught"
  became three: the count's arithmetic is unchanged (880 < 1 008), under `payg` that figure no
  longer exhausts the decider, and under `go` it still does.
- **The dry-run prints `no aplica (payg)` where it printed `**NO**`** for a remote role
  (`QuotaLine.limits`): it does not judge against a limit that does not brake. A pair served
  locally is still judged against its quota.

**`RunMeta` keeps the effective role map.** `.env` is not versioned and is edited between runs.
`RoleMeta` gained `family` and `structured_output`, and `RunMeta` the OpenAI timeout and the cap;
they live in `arm_roles`, already per arm and per role as the router will call it, because the
local arms swap the primary. `build_run_meta()` came out of `_run` so that it can be tested without
launching anything; the mutation that skips a role is pinned.

**C2: the usage of a rejected parse.** Under `json_schema` the SDK validates inside `parse()` and
raises before a message exists, so the attempt was journaled with no counters although it was
billed. `usage_from_rejected_parse` reads them from the response LangChain hangs from the
exception. That path is also where content that is not bare JSON ends — a `<think>` block before a
valid verdict is rejected by the SDK, rescued by `json_payload`, and validates — which is the
"valid and no usage" row. Such a row now carries its usage.

**The retry carried an error the model had not made** (commit 1b). On that same path the adapter
recovered the text from the `input` Pydantic keeps in the error, the outermost one. That is the
whole answer only when the failure is at the root. Reproduced offline with the real adapter, the
real `ChatOpenAI` and SDK, and a mock transport, over `TechnicalVerdict`:

| what the model answered | what the adapter handed over | the `Error:` the retry carried | now |
| --- | --- | --- | --- |
| a nested field of the wrong type (`cites` a string) | `rsi_14` | `: Invalid JSON: expected value at line 1 column 1` | `observations.0.cites: Input should be a valid array` |
| a missing nested key | `observations[0]` alone | `id: Extra inputs are not permitted; …; dimension: Field required; bias: Field required` | `observations.0.cites: Field required` |
| a root number out of range (`confidence: 1.7`) | `''` | `: Invalid JSON: EOF while parsing a value at line 1 column 0` | `confidence: Input should be less than or equal to 1` |
| a root field of the wrong type | `ninguna` | `: Invalid JSON: expected ident at line 1 column 2` | `observations: Input should be a valid array` |
| a missing root key | the whole answer | `dimension: Field required` | the same |

- In four of five shapes both attempts failed for certain, and the cache kept the fragment.
- **It was a bias against `full` and not against `solo`**: five of the six roles run under
  `json_schema` and the decider (`function_calling`) does not, and an evaluation without a decision
  scores 0.
- `content_from_rejected_parse` reads the whole content from the same response; the old recovery
  stays as the fallback for a body that cannot be read. `_RETRY_TEMPLATE` did not change: what
  changed is what fills `{error}` on that path, and with it the digest of that retry prompt. No
  cache on disk held an entry to invalidate.
- The probes before this block were not repeated. The one invalid `json_schema` row they hold
  (`20261005T062628Z/glm-5.3-flash@volume.jsonl`, `observations.N.cites: Field required`, no
  counters) carries a message only the whole answer produces, so that content was not bare JSON.

**Two commands no longer overwrite a versioned file by default.** `ablation --out` defaulted to
`docs/ablation.md` and a run that was not a `--dry-run` replaced the whole file with the table:
launching stage 1 without `--out` would have removed the criteria and the three amendments from
the tree. The table now goes to `report.md` inside the run directory, and an `--out` that exists is
refused before the first call. `selection --out` is covered above.

**`estimate` can price a stage.** `--arms` redoes the count with those arms only, and structure and
volume are one model when the role map declares one the probe measured in that role; with an id
nobody measured they stay a range between candidates and a note says which.

**Known limits, not fixed.**

- **Go only: a paid attempt can stay out of the journal.** `resolve()` runs inside the retry loop.
  If the window is exhausted between attempt 1 (invalid, paid) and attempt 2 of a role with no
  fallback, `QuotaExhaustedError` leaves without the attempts and `_model_failure` writes
  `calls = []` (`nodes.py:191`). Under `payg` it cannot happen to a remote role.
- **The runner's cap is per process and is not seeded from the journal.** A restart starts from
  zero, so a process that keeps restarting can spend the cap each time. What survives restarts is
  the monthly limit set in the provider's workspace. A resumed ablation does not inherit the spend
  of the pass it resumes either: `--max-usd` is what that command may spend.
- **The ablation still does not refuse to run outside Zen.** `refuse_unless_zen` is the probe's.
  Under `go`, `--fill` runs with no cap, as before.
- **The stage-1 role map is not the template**, and nothing checks `.env` against the amendment:
  the run records the map it used in `meta.json`, and that is what there is to compare.

## Gotchas found the hard way

- **A provider client's default timeout is not a decision anybody made, and both defaults are
  unusable.** Measured against the installed versions: `ChatOpenAI(...)` without `timeout` resolves
  to the OpenAI SDK's `Timeout(connect=5, read=600, write=600, pool=600)`, so one hung role holds ten
  minutes per attempt against an evaluation that costs ~90 s complete; and `AsyncClient(host=...)`
  gives httpx `Timeout(None)`, meaning a wedged Ollama never returns at all and the run hangs with
  nothing to read. Both are now declared — `CA_OPENAI__TIMEOUT_SECONDS`, `CA_OLLAMA__TIMEOUT_SECONDS`
  — and `tests/test_architecture.py` parses `llm.py` to refuse any provider client built without one.
- **What Pydantic keeps in a `ValidationError` is not what the model said.** `errors()[i]["input"]`
  is the value that failed *at that location*: the whole answer for a failure at the root, one field
  or one sub-object for a nested one. Under `json_schema` the SDK validates inside `parse()`, and
  recovering the text from that `input` handed the router a fragment in four of five shapes; it
  validated the fragment, failed on something else, and the retry corrected an error the model had
  not made. The whole answer is in the body of the HTTP response that hangs from the exception,
  and so is the `usage`. The one test of that path used a missing root key, the single shape in
  which the fragment is the whole answer.
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
  qwen and `bull` now declares it too — since block T7 its primary is a qwen as well
  (`qwen3.8-max`) — so an evaluation where those four roles are all out of budget
  runs `structure`, `momentum`, `volume` and `bull` on the same model — four of six roles. That
  degraded case exists under the Go subscription only: under `payg` no remote role falls back to
  local by quota. It is
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
| momentum | DeepSeek V4 Pro | deepseek | json_schema | 100 000 (declared) | 1.0 | local |
| volume | Hy3 | tencent | json_schema | 4 300 | 1.0 | local |
| bull | Qwen3.8 Max | qwen | json_schema | 100 000 (declared) | 1.0 | local |
| bear | MiniMax M3 | minimax | json_schema | 3 200 | 1.0 | none |
| decider | GLM-5.2 | zhipu | function_calling | 880 | 1.0 | never |

The `quota_per_window` and `quota_weight` columns are **Go figures** (the page of the subscription,
2026-10-02) for every role but momentum and bull. Under `payg` they do not apply: Zen publishes no request
limit, so each remote role declares `ZEN_UNPUBLISHED_QUOTA` (100 000) and weight 1.0 — see "Pay as you
go (Zen)".

**Momentum moved to `deepseek-v4-pro` in block T5**, in the shipped template; `.env` is the operator's
and changes by hand. Until then it was `deepseek-v4-flash` (`json_mode`, 63 300 at weight 2.0, a Go
pool weight), which is the model the re-probe below measured on Go and the one that answered 404 on
Zen fifteen times in a row on 2026-10-05. `json_schema` for the new one is measured (12 of 12 valid at
the first attempt, `var/zen-probe/20261005T233053Z/deepseek-v4-pro@momentum.jsonl`, sha-256
`089aaabc21705ef2b525fa040878bbcb3d0617349a4df973d6294ef8462662da`). Its quota is a **declaration**:
neither Go nor Zen publishes a figure for that id, so the row carries the Zen sentinel, and a smaller
number would be an invented limit that degrades momentum to the local qwen. **The template is now a
hybrid**: it is the Go map (`CA_BILLING=go`, pinned by a test) carrying one id measured only on Zen,
with no Go price row in `settings.py`, no page estimate, and no evidence that Go serves it. Two more
places where the template and this table disagreed, found in T5: the template declared
`json_schema` for `bear` where the table and the measurements said `json_mode`, and it declared
`json_schema` for `deepseek-v4-flash` too.

**The bear's mode is `json_schema` since block T6, in the template and in this table.** The template
already declared it; what changed is that it is measured. On Zen, with the real prompt and the same
`prompt_digest` as the T5 desks probe, `minimax-m3` gave 12 of 12 valid briefs at the first attempt
under `json_schema` against 0 of 12 under `json_mode` (see "Block T6 results"). It makes the
template a hybrid on a second row: `json_mode` is what the re-probe below measured on **Go**, and
`json_schema` against Go has not been asked. `.env` is the operator's and changes by hand.

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

**`bull` moved again in block T7, to `qwen3.8-max`**, in the shipped template; the operator's `.env`
already had it. The 503 of the qwen family was Go's. On Zen `kimi-k2.6` is listed and answers `410
Endpoint is unavailable`, and `qwen3.8-max` gave 12 of 12 valid briefs at the first attempt under
`json_schema`, twice (`var/zen-probe/20261005T233053Z` and `20261006T063107Z`;
`20261006T063107Z/qwen3.8-max@bull.jsonl`, sha-256
`089a2ca18c80b431d90e3d1bc4e067323008f10ec5b275acf4e7bf3f963d3a3f`). Its quota is the Zen sentinel,
a declaration, like momentum's. The six primaries are still six families (xiaomi, deepseek, tencent,
qwen, minimax, zhipu), and `bull` is now the one role whose primary and fallback share a family. The
template is a hybrid on a third row: the Go map carrying two ids measured only on Zen and a bear
mode measured only there. **The stage-1 map of amendment 3 is not the template**: it also puts
`glm-5.3-flash` on structure and volume, where the template keeps `mimo-v2.5` and `hy3`, ids that
do not exist under pay as you go.

All six remote models come from one OpenAI-compatible gateway. `Settings.openai` holds a single
`api_key` and `base_url`, so moving one model to a different provider means moving those two fields
onto `ModelChoice` — a change to the configuration contract, not a change to `.env`. Model ids are
the gateway's own, without a provider prefix: it serves `glm-5.2`, not `zhipu/glm-5.2`. That, and
whether each model answers in its declared mode, is what `doctor` checks before the first paid
call.

The local fallback is `qwen3:8b` (5.2 GB on disk, 6.0 GB resident, `100% GPU` at `num_ctx=4096`
against ~7.0 GiB free): one resident model, no second local model alongside it. Four roles now
declare it — the three technical ones and `bull` — so a fully degraded evaluation runs four of six
roles on one model. The hard constraint still holds: `bull`'s family is {qwen} — primary and
fallback, since block T7 — and `bear`'s is {minimax}, disjoint.

`bull`'s fallback exists so the ablation's `local_bull` arm can run at all. **It does not rescue a
provider outage**: `resolve()` degrades on exhausted quota, never on a transport rejection, so a 503
aborts the evaluation with or without a fallback declared. Under `payg` it does not degrade on
quota either (block T7), so there the fallback is reached by that arm and by nothing else.

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

**Go subscription only, since block T7.** Under `payg` `ModelRouter._choose` does not ask the
ledger about a remote role, so nothing below happens there: the role stays on its primary and what
brakes is the dollar cap.

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
