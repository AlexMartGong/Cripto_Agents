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
| `settings.py` | `pydantic-settings` config, `CA_` prefix. `load_settings()` fails at startup naming the missing variables. |
| `quota.py` | Sliding-window quota ledger per `(role, model)`, injected clock. Holds a window, not `Settings`: the candidates come from the caller, which is what lets one counter serve several role maps. `seed()` rebuilds the window from journaled calls after a restart. |
| `llm.py` | `ChatBackend` protocol, OpenAI/Ollama adapters, and `ModelRouter` — resolve by budget, cache, validate, retry, record. |
| `cache.py` | Response cache keyed by `(model, prompt digest, schema)`. |
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
| `audit.py` | Reads a run directory and prints what happened in it, each figure next to the digest of the file it came from. `python -m crypto_agents.audit <dir>`. No model calls. |
| `outcomes.py` | Labels each order against later candles: invalidation hit first, or the close at the horizon. |
| `alerts.py` | Quota running out, repeated vetoes, validation failures, skipped cycles. Pure over journal records. |
| `queries.py` | Journal filters by symbol, action, backend and abort cause. |
| `doctor.py` | Startup checks: gateway catalog, Ollama tags, VRAM split, exchange and credentials. |
| `bootstrap.py`, `cli.py` | Composition root and the `crypto-agents` entry point. |
| `activation_sweep.py` | The four gate rules over a committed history, no model calls. Sizes the ablation. |
| `selection.py` | Stratified selection of activations across symbols and time spans, and the versioned manifest the ablation runs over. No model calls. |
| `metrics.py` | Aggregations over a run — the funnel, action mix, vetoes by rule, quota by role and backend — and the audit of one: attempts, live latency, abort causes, stop side, action against the desks. Pure over `EvaluationRecord`. |

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

Three failure modes reach a node, and they are not the same failure:

| What broke | Type | `LLMCall` row | Retried |
| --- | --- | --- | --- |
| The provider never returned content (4xx, 5xx, DNS, credentials) | `ModelCallError` | `failure.kind = transport` | no |
| Content arrived and did not validate | `InvalidModelOutputError` | `failure.kind = validation`, one per attempt | yes, with the error attached |
| Neither model fit the window | `QuotaExhaustedError` | none — nothing was spent | no |

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

`metrics.backend_stats()` counts the two kinds in separate columns, and `failure_rate` divides by
`answered`, not by `attempts`. A model id the gateway does not serve and a model that hallucinates
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
uv run pytest                  # 613 tests
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
  for material — its limit is the decider's 880 calls per window. **All six arms reach the decider**
  (`decide`, `decide_without_debate` and `decide_solo` are one node each, and the two local arms are
  `full` with a role swapped), and none of them reuses another's decider entry, so the ceiling is
  880 / 6 ≈ 146 activations. The selection commits **140**, which leaves 40 calls for retries.
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

Three properties, each with a test:

- **It cannot call anybody.** The context it builds carries the replay router with no `fill_with`,
  so every slot is a `CacheOnlyBackend`. A single call would raise `ReplayCacheMissError` rather than
  quietly spend, which is also what makes "it emitted zero calls" testable instead of asserted.
- **It runs `prepare_evaluation()`, the same function the `prepare` node runs.** The gate's rate is
  what multiplies everything else, so a copy of those five lines here would budget the harness.
- **Exact is separated from upper bound.** The three technical prompts and `decide_solo`'s are pure
  functions of snapshot, indicators and triggers, so their digest — and therefore their cache key —
  is computable before spending anything. Everything downstream of a verdict only admits "how many
  times it could run". Rendering the two as one number would invite reading the sum as the invoice.

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
across the seven. Material is not the constraint; the decider's 880 calls per window is, and six arms
reach it, so **140 activations** is what the comparison commits to.

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
arms**, against 880 per window. Every role fits; retries are not in that count (see the re-probe).

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
- **Failed evaluations get a second draw.** The cache is written after validation, so an invalid
  attempt is never cached and the retry is stored under a digest that includes the error text. On
  resume, exactly the evaluations that aborted are called live again. The audit reports per arm how
  many were undecided before and decided now; without that number, "decided" in a resumed run is not
  comparable with a single pass.

The seed is a **lower bound**. It knows the journal it is given: what `doctor` or another command
spent against the same provider is not in that file.

Two clocks meet in one record under `--fill`: `LLMCall.at` is wall time — the router's clock, which
is what makes seeding possible — while `NodeError.at` and the record's own `at` are the evaluated
instant.

### Why the ablation never vetoes

`orders == buy + sell` in every arm is an identity, not a finding. With `NOTIONAL_ACCOUNT` none of
the four ways to stop an order is reachable: drawdown is zero against a limit above zero, there is
no `last_loss_at` for the cooldown, the context is built with the default `StaticKillSwitch(False)`
and never consults `var/STOP`, and with zero open exposure a non-zero size always leaves headroom.
The account is one frozen object for all 140 evaluations; nothing updates it. The only limit that can
apply is the `max_position_fraction` cap, which shrinks the order and does not veto it.

`risk.py` has no rule about which side of the close an `invalidation_price` sits on, and
`outcomes.py` scores `low <= invalidation` on a long without checking. The audit counts wrong-side
stops per arm (`>=` close on a buy, `<=` on a sell); what to do about them is not decided here.

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
