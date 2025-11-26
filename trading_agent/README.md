# TradingAgents Architecture Overview

This document explains how the trimmed TradingAgents codebase is organized after the recent LLM refactor. The goal is to highlight the data flow, the major components, and how optional language-model reasoning slots into the pipeline.

## High-Level Flow

1. **CLI entrypoints (`cli/main.py`)** – Typer commands (`weight`, `news-weight`, `weight-summary`) orchestrate requests from the terminal. Each command accepts `--llm/--no-llm` and an optional `--llm-model` override. The CLI prints Rich-formatted Markdown and surfaces whether an LLM response was successfully used (or why it fell back).
2. **Agents** – Lightweight classes under `tradingagents/` fetch data and produce structured reports:
   - `fundamental_agent.py` pulls Yahoo Finance fundamentals, renders neutral metric descriptions, and can let an LLM synthesise the rationale text.
   - `news_agent.py` collects recent headlines (Google News RSS with yfinance fallback), applies VADER sentiment scoring (no keyword lists), and can hand those to an LLM for tone-aware guidance.
   - `combined_weight_agent.py` fuses both reports and optionally calls the LLM to write the unified bullets.
3. **LLM bridge (`tradingagents/llm_client.py`)** – Central helper that routes prompts to OpenAI or Google Gemini based on the model name. It exposes convenience functions:
   - `summarise_fundamentals` – returns action-oriented bullets about fundamentals.
   - `summarise_news` – condenses headline sentiment into guidance.
   - `summarise_weight_points` – blends both streams for the combined agent.
   Each helper leverages a shared `generate_bullets` function, records the last error in `LAST_LLM_ERROR`, and normalises the raw text into bullet lists.
4. **Environment (`.env`)** – Stores API keys (not auto-loaded). Export the relevant key into your shell before running a command:
   ```zsh
   export GEMINI_API_KEY="..."          # or GOOGLE_API_KEY for Gemini
   export OPENAI_API_KEY="..."          # if using OpenAI models
   export TRADINGAGENTS_LLM_MODEL="gemini-2.0-flash"  # optional global override
   ```
5. **Data utilities (`market_data.py`, `dataloader/`, etc.)** – Provide historical pricing, correlation matrices, and cached datasets used by upstream workflows. The core CLI agents depend only on Yahoo Finance and live news, so these modules are optional for the LLM-driven summaries.
6. **Training and policy experiments (`env/`, `policy/`, `trainer/`, `model/`)** – Legacy reinforcement-learning scaffolding retained for research. They no longer participate in the default CLI flows but remain available for deeper experimentation.

## Key Modules

| Module | Responsibility |
| --- | --- |
| `cli/main.py` | Typer CLI, Rich output, LLM status reporting. |
| `tradingagents/fundamental_agent.py` | Pulls Yahoo Finance fundamentals, emits descriptive metric bullets, optional LLM rationale. |
| `tradingagents/news_agent.py` | Fetches headlines, scores them with VADER, optional LLM news summary. |
| `tradingagents/combined_weight_agent.py` | Merges fundamentals & news into one report, optional LLM synthesis. |
| `tradingagents/llm_client.py` | Routes prompts to Gemini or OpenAI, normalises bullet output, tracks errors. |
| `tradingagents/dataloader/` | Loads historical datasets for advanced scenarios. |
| `tradingagents/model/`, `policy/`, `trainer/` | Reinforcement-learning experiments (not used by the CLI). |

## LLM Integration Details

- Default model is taken from `TRADINGAGENTS_LLM_MODEL` or falls back to `gemini-2.0-flash`.
- Model routing logic:
  - Names starting with `gemini` or `flash-` invoke Google Gemini via `google-generativeai`.
  - Other names delegate to OpenAI’s Responses API.
- Every agent method records whether the LLM path produced content. When it fails (missing API key, model error, empty response), the CLI prints a yellow message with `llm_client.LAST_LLM_ERROR` so you can troubleshoot quickly.
- LLM helpers fall back to deterministic descriptions when a call fails, so the system still returns grounded output even without API keys.

## Typical Command Examples

```zsh
# Fundamentals-only review (LLM on):
python -m cli.main weight AAPL 0.08 --llm --llm-model gemini-2.0-flash

# News-flow review over the last 5 days (LLM on):
python -m cli.main news-weight AAPL 0.08 --lookback-days 5 --llm

# Blended summary from both agents, letting the LLM synthesise the final bullets:
python -m cli.main weight-summary AAPL 0.08 --llm
```

## Error Handling & Observability

- `llm_client.LAST_LLM_ERROR` stores the most recent provider error. The CLI prints it whenever an LLM call was requested but not used.
- LLM prompts include the underlying tables and summaries, so the generated bullets remain grounded in the fetched data.
- If you need to audit the deterministic fallbacks, inspect `_build_rationale` in `fundamental_agent.py` and `_build_opinion` in `news_agent.py`—they provide descriptive, data-backed summaries whenever `use_llm=False` or the LLM fails.

## Extending the System

1. **Add new data sources** by wrapping their fetch logic in the relevant agent and passing structured context into the LLM prompts.
2. **Support additional LLM providers** by extending `generate_bullets` with new routing branches and exposing the necessary environment variables.
3. **Surface configuration via CLI** by adding Typer options and passing them through to agent constructors.
4. **Instrument logging** by emitting structured logs around agent calls, using `LAST_LLM_ERROR` for quick diagnosis.

This modular layout keeps the deterministic pipelines intact while allowing optional LLM assistance wherever narrative judgement is desired.