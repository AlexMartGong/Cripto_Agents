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
| `quota.py` | Sliding-window quota ledger per `(role, model)`, injected clock. |
| `llm.py` | `ChatBackend` protocol, OpenAI/Ollama adapters, and `ModelRouter` — resolve by budget, cache, validate, retry, record. |
| `cache.py` | Response cache keyed by `(model, prompt digest, schema)`. |
| `market.py` | ccxt client, OHLCV normalisation, reproducible candle digest. |
| `indicators.py` | pandas-ta preset producing a validated `IndicatorSet`. |
| `activation.py` | Four pure gate rules; no state between runs. |
| `prompts/` | Versioned `.md` templates plus loader. |
| `nodes.py`, `graph.py`, `context.py` | LangGraph nodes, compiled graph, execution context. |
| `risk.py` | Deterministic vetoes and caps. |
| `execution.py` | Order construction, paper executor, live executor gated behind three conditions. |
| `journal.py` | Structured record of every evaluation, JSONL or in memory. |
| `runner.py` | Candle-close schedule, multi-symbol cycle, bounded concurrency, clean shutdown. |
| `replay.py` | Historical replay over committed candles: cache-only by default, deterministic ids, canonical run digest. |
| `ablation.py` | Pipeline variants compared over one history; `python -m crypto_agents.ablation` renders the table. |
| `outcomes.py` | Labels each order against later candles: invalidation hit first, or the close at the horizon. |
| `alerts.py` | Quota running out, repeated vetoes, validation failures, skipped cycles. Pure over journal records. |
| `queries.py` | Journal filters by symbol, action, backend and abort cause. |
| `bootstrap.py`, `cli.py` | Composition root and the `crypto-agents` entry point. |
| `metrics.py` | Aggregations over a run — the funnel, action mix, vetoes by rule, quota by role and backend. |

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

## Environment

Dependencies are managed with **uv** (lockfile committed, Python 3.12+). Never pip or poetry.

```bash
uv sync                        # install/refresh .venv from uv.lock
uv add <pkg>                   # runtime dep; --dev for tooling
uv run ruff check .            # lint
uv run ruff format .           # format (line-length 100)
uv run mypy                    # strict, over src/ and tests/
uv run pytest                  # 409 tests
```

All four must exit 0 before a phase is done.

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
crypto-agents stop       # no order leaves the system until resumed
crypto-agents resume     # removes the sentinel
crypto-agents alerts     # exit code 1 when something fires, so scripts can chain it
crypto-agents query --symbol BTC/USDT --action buy --group-by-cause
crypto-agents run        # the candle-close loop over CA_RUNNER__SYMBOLS
```

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

## Gotchas found the hard way

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

## Configuration

`.env.example` is the template; copy it to `.env` (gitignored) and fill in the gateway key. Every
role maps to a distinct family — the hard constraint is only between the two desks, but three
technical agents on one model would make the same mistake three times.

| Role | Model | Family | `quota_per_window` | `quota_weight` | Fallback |
| --- | --- | --- | --- | --- | --- |
| structure | MiMo-V2.5 | xiaomi | 30 100 | 1.0 | local |
| momentum | DeepSeek V4 Flash | deepseek | 63 300 | 2.0 | local |
| volume | Hy3 | tencent | 4 300 | 1.0 | local |
| bull | Qwen3.7 Plus | qwen | 4 300 | 1.0 | none |
| bear | MiniMax M3 | minimax | 3 200 | 1.0 | none |
| decider | GLM-5.2 | zhipu | 880 | 1.0 | never |

All six remote models come from one OpenAI-compatible gateway. `Settings.openai` holds a single
`api_key` and `base_url`, so moving one model to a different provider means moving those two fields
onto `ModelChoice` — a change to the configuration contract, not a change to `.env`. The local
fallback is `llama3:latest` (4.7 GB against ~7.6 GB of free VRAM): one resident model, no second
local model alongside it.

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

Out of budget, `QuotaLedger.resolve()` drops to the role's local fallback instead of failing.
`resolve()` runs inside the retry loop, so a single verdict can start remote and finish local —
which is why every `LLMCall` records its own `backend` and its own `valid`. Without those two
columns the journal mixes a large remote model with a small local one on the same line, and a
fallback that needs three attempts per verdict looks exactly as cheap as one that gets it right
first time. `metrics.backend_stats()` is what reads them back, denominator included.

The decider is the exception, enforced by a `Settings` validator: it takes no fallback at all. Out
of budget, the evaluation aborts and the journal records why. Degrading it would change who makes
the final call without that appearing anywhere until the order was already placed.

**Measured on this machine** (llama3:latest, 4.7 GB, `num_ctx=4096`, ~375-token technical prompt,
RTX 3070 Ti with the desktop loaded):

| What | Time |
| --- | --- |
| First call, weights not resident | 6.45 s |
| Warm call | 3.4 - 4.5 s |
| Three technical agents, sequential | 11.6 s |
| The same three via `asyncio.gather` | 10.3 s |

A 1.13x speedup from parallelism is a measurement of no parallelism: **Ollama serializes on the
GPU** (`OLLAMA_NUM_PARALLEL:1` at these VRAM levels), so the graph's technical fan-out becomes a
queue. Budget ~12 s of wall clock for that stage when all three fall back to local, against a 4h
candle — three orders of magnitude of headroom, so the runner's skip path should never fire on
local latency alone. It exists for a hung provider, not for this.

Model choice is deliberately left in `.env`. Criteria for the candidate, to be settled with an
ablation and not by taste: GGUF **text-only** (a multimodal variant loads a vision encoder and
spends over 1 GB of VRAM on it even for pure-text calls), resident alongside the desktop in ~7 GB,
and reliable under a JSON Schema grammar. `llama3:latest` is what the measurements above used; it is
a stand-in, not a recommendation.
