# ======================================================================
# 🔹 ADDITIONAL CLIENT FUNCTIONS FOR TRADING AGENTS
# ======================================================================

import os
import sys
from typing import List, Optional

LAST_LLM_ERROR: Optional[str] = None  # for CLI error reporting


def _safe_configure_gemini():
    """Configures Gemini SDK and returns model handle."""
    global genai
    if genai is None:
        raise RuntimeError("google.generativeai is not installed.")
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY or GEMINI_API_KEY in environment.")
    genai.configure(api_key=api_key)
    return genai


def summarise_fundamentals(
    *,
    ticker: str,
    weight: float,
    as_of: str,
    metrics_table: str,
    metrics_summary: str,
    max_points: int = 4,
    model: Optional[str] = None,
) -> List[str]:
    """
    Summarise fundamental metrics using Gemini.
    Produces concise rationale points.
    """
    global LAST_LLM_ERROR
    try:
        genai_mod = _safe_configure_gemini()
        llm = genai_mod.GenerativeModel(model or "gemini-2.0-flash")
        prompt = (
            f"You are an equity analyst creating a short fundamental summary for {ticker}.\n"
            f"Portfolio Weight: {weight:.2%}\nAs of: {as_of}\n\n"
            f"Metrics Table:\n{metrics_table}\n\nSummary of Metrics:\n{metrics_summary}\n\n"
            f"Write {max_points} key bullet points summarizing financial health, valuation, "
            f"and growth outlook in crisp language."
        )
        response = llm.generate_content(prompt)
        text = getattr(response, "text", "")
        points = [line.strip(" -*•") for line in text.splitlines() if line.strip()]
        return points[:max_points]
    except Exception as e:
        LAST_LLM_ERROR = str(e)
        print(f"[WARN] summarise_fundamentals failed: {e}", file=sys.stderr)
        return []


def summarise_news(
    *,
    ticker: str,
    weight: float,
    as_of: str,
    lookback_days: int,
    article_summaries: str,
    net_sentiment: int,
    max_points: int = 4,
    model: Optional[str] = None,
) -> List[str]:
    """
    Summarise news sentiment and coverage.
    """
    global LAST_LLM_ERROR
    try:
        genai_mod = _safe_configure_gemini()
        llm = genai_mod.GenerativeModel(model or "gemini-2.0-flash")
        prompt = (
            f"You are a financial news analyst reviewing {ticker}.\n"
            f"Portfolio Weight: {weight:.2%}\nDate: {as_of}\nLookback: {lookback_days} days\n"
            f"Net Sentiment Score: {net_sentiment}\n\n"
            f"Recent Headlines:\n{article_summaries}\n\n"
            f"Summarize {max_points} insights capturing sentiment trends, risks, "
            f"and how news tone might affect allocation decisions."
        )
        response = llm.generate_content(prompt)
        text = getattr(response, "text", "")
        points = [line.strip(" -*•") for line in text.splitlines() if line.strip()]
        return points[:max_points]
    except Exception as e:
        LAST_LLM_ERROR = str(e)
        print(f"[WARN] summarise_news failed: {e}", file=sys.stderr)
        return []


def summarise_weight_points(
    *,
    ticker: str,
    weight: float,
    as_of: str,
    fundamental_points: List[str],
    news_points: List[str],
    metrics_table: str,
    news_table: str,
    max_points: int = 6,
    model: Optional[str] = None,
) -> List[str]:
    """
    Combine fundamental + news rationales into unified LLM summary.
    """
    global LAST_LLM_ERROR
    try:
        genai_mod = _safe_configure_gemini()
        llm = genai_mod.GenerativeModel(model or "gemini-2.0-flash")

        prompt = (
            f"As an investment strategist, create {max_points} concise, bullet-point insights "
            f"that integrate fundamental and news perspectives for {ticker}.\n\n"
            f"Portfolio Weight: {weight:.2%} (as of {as_of})\n\n"
            f"Fundamental Rationale Points:\n"
            + "\n".join(f"- {pt}" for pt in fundamental_points)
            + "\n\nNews-Based Points:\n"
            + "\n".join(f"- {pt}" for pt in news_points)
            + "\n\nFundamental Metrics:\n"
            + metrics_table
            + "\n\nRecent News Table:\n"
            + news_table
            + "\n\nWrite objective, data-grounded bullet points combining both analyses, "
            f"and avoid repetition."
        )

        response = llm.generate_content(prompt)
        text = getattr(response, "text", "")
        bullets = [line.strip(" -*•") for line in text.splitlines() if line.strip()]
        return bullets[:max_points]
    except Exception as e:
        LAST_LLM_ERROR = str(e)
        print(f"[WARN] summarise_weight_points failed: {e}", file=sys.stderr)
        return []
