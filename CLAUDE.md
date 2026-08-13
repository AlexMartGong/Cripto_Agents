# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository state

This is a fresh scaffold: the only committed files are `pyproject.toml`, `uv.lock`, and `.gitignore`. There is no source code, package layout, test suite, or lint config yet. Do not assume an architecture exists — read the tree before answering structural questions, and when adding the first modules, propose the layout rather than inferring one from this file.

## Environment

Dependencies are managed with **uv** (lockfile committed, `.venv/` present, Python 3.12+ required). Use uv rather than pip/poetry so `uv.lock` stays authoritative.

```bash
uv sync                  # install/refresh .venv from uv.lock
uv add <pkg>             # add dep, update pyproject.toml + uv.lock
uv run python -m <mod>   # run inside the project env (no manual activation)
```

There is no build, lint, or test command configured. `uv run pytest` etc. will not work until the corresponding dev dependency and config are added.

## Intended stack

The declared dependencies define what this project is for, even though nothing is wired up yet:

- **ccxt** — unified exchange API for crypto market data / order execution.
- **pandas-ta** (0.4.x beta fork) — technical indicators over OHLCV DataFrames. Pulls in numba; indicator calls are pandas-accessor based (`df.ta.*`).
- **langgraph** (1.x) — the agent orchestration layer: stateful graphs, not one-shot chains. Multi-agent trading/analysis workflows belong here.
- **langchain-openai** and **ollama** — two model backends side by side (hosted OpenAI-compatible and local Ollama). Keep model selection configurable rather than hardcoding one backend.
- **pydantic** / **pydantic-settings** — structured agent I/O schemas and env-var-driven config (`BaseSettings`). Exchange keys and model config should flow through pydantic-settings, not raw `os.environ` reads scattered in modules.

Both LangGraph 1.x and pydantic-settings 2.15 are recent majors — check current API shapes rather than relying on older-tutorial patterns.
