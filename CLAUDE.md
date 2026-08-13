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

All five phases are implemented. `src/crypto_agents/` holds the package; `tests/` mirrors it.

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

## Environment

Dependencies are managed with **uv** (lockfile committed, Python 3.12+). Never pip or poetry.

```bash
uv sync                        # install/refresh .venv from uv.lock
uv add <pkg>                   # runtime dep; --dev for tooling
uv run ruff check .            # lint
uv run ruff format .           # format (line-length 100)
uv run mypy                    # strict, over src/ and tests/
uv run pytest                  # 247 tests
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

## Pending configuration

The system cannot start until `CA_ROLES` is provided: for each of the six `AgentRole` values, a
model, its `family`, its `quota_per_window` and its `quota_weight` (`2.0` for double-usage models).
The bull and bear desks must not share a family — `Settings` refuses to load otherwise, because two
desks on one model have correlated errors and the debate stops adding information.
