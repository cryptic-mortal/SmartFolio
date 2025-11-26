from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown

from tradingagents import llm_client
from tradingagents.combined_weight_agent import WeightSynthesisAgent
from tradingagents.fundamental_agent import FundamentalWeightAgent
from tradingagents.news_agent import NewsWeightReviewAgent

console = Console()

app = typer.Typer(
    name="TradingAgents",
    help="Fundamental, news, and blended weight reviews.",
    add_completion=False,
)

@app.command()
def weight(
    ticker: str = typer.Argument(..., help="Ticker symbol, e.g. AAPL"),
    weight: float = typer.Argument(..., help="Portfolio weight between 0.0 and 1.0"),
    include_metrics: bool = typer.Option(
        True,
        "--include-metrics/--no-metrics",
        help="Include the fundamentals metrics table.",
    ),
    use_llm: bool = typer.Option(
        False,
        "--llm/--no-llm",
        help="Ask an LLM to draft the fundamental rationale when an API key is configured.",
    ),
    llm_model: Optional[str] = typer.Option(
        None,
        help="Override the model name when --llm is enabled (defaults to TRADINGAGENTS_LLM_MODEL or gemini-2.0-flash).",
    ),
    as_of: Optional[str] = typer.Option(
        None,
        help="Override the as-of date (YYYY-MM-DD).",
    ),
):
    """Generate a fundamentals rationale for the supplied weight."""

    agent = FundamentalWeightAgent()
    try:
        report = agent.generate_report(
            ticker,
            weight,
            as_of=as_of,
            use_llm=use_llm,
            llm_model=llm_model,
        )
    except ValueError as err:
        console.print(f"[red]{err}[/red]")
        raise typer.Exit(code=1) from err
    except Exception as err:  # noqa: BLE001
        console.print(f"[red]Fundamentals report failed: {err}[/red]")
        raise typer.Exit(code=1) from err

    console.print(Markdown(report.to_markdown(include_metrics=include_metrics)))

    if report.generated_via_llm:
        console.print("\n[dim]Fundamental rationale generated via LLM.[/dim]")
    elif use_llm:
        reason = llm_client.LAST_LLM_ERROR or "LLM call returned no content."
        console.print(f"\n[yellow]LLM path skipped: {reason}[/yellow]")


@app.command()
def news_weight(
    ticker: str = typer.Argument(..., help="Ticker symbol, e.g. AAPL"),
    weight: float = typer.Argument(..., help="Portfolio weight between 0.0 and 1.0"),
    lookback_days: int = typer.Option(7, help="Number of calendar days to scan for news."),
    max_articles: int = typer.Option(8, help="Maximum number of headlines to surface."),
    include_articles: bool = typer.Option(
        True,
        "--include-articles/--no-articles",
        help="Include the headline table.",
    ),
    use_llm: bool = typer.Option(
        False,
        "--llm/--no-llm",
        help="Ask an LLM to synthesise the news-based rationale when an API key is configured.",
    ),
    llm_model: Optional[str] = typer.Option(
        None,
        help="Override the model name when --llm is enabled (defaults to TRADINGAGENTS_LLM_MODEL or gemini-2.0-flash).",
    ),
    as_of: Optional[str] = typer.Option(
        None,
        help="Override the as-of date (YYYY-MM-DD).",
    ),
):
    """Evaluate the weight against recent headline tone."""

    agent = NewsWeightReviewAgent()
    try:
        report = agent.generate_report(
            ticker,
            weight,
            as_of=as_of,
            lookback_days=lookback_days,
            max_articles=max_articles,
            use_llm=use_llm,
            llm_model=llm_model,
        )
    except ValueError as err:
        console.print(f"[red]{err}[/red]")
        raise typer.Exit(code=1) from err
    except Exception as err:  # noqa: BLE001
        console.print(f"[red]News review failed: {err}[/red]")
        raise typer.Exit(code=1) from err

    console.print(Markdown(report.to_markdown(include_articles=include_articles)))

    if report.generated_via_llm:
        console.print("\n[dim]News rationale generated via LLM.[/dim]")
    elif use_llm:
        reason = llm_client.LAST_LLM_ERROR or "LLM call returned no content."
        console.print(f"\n[yellow]LLM path skipped: {reason}[/yellow]")


@app.command()
def weight_summary(
    ticker: str = typer.Argument(..., help="Ticker symbol, e.g. AAPL"),
    weight: float = typer.Argument(..., help="Portfolio weight between 0.0 and 1.0"),
    lookback_days: int = typer.Option(7, help="Days of news used by the news agent."),
    max_articles: int = typer.Option(8, help="Maximum headlines for the news agent."),
    include_components: bool = typer.Option(
        False,
        "--include-components/--summary-only",
        help="Append detailed agent outputs after the unified bullets.",
    ),
    include_metrics: bool = typer.Option(
        True,
        help="Include the fundamentals metrics table when components are shown.",
    ),
    include_articles: bool = typer.Option(
        True,
        help="Include the headline table when components are shown.",
    ),
    use_llm: bool = typer.Option(
        False,
        "--llm/--no-llm",
        help="Ask an LLM (OpenAI Responses API) to draft the unified summary when an API key is available.",
    ),
    llm_model: Optional[str] = typer.Option(
        None,
        help="Override the model name when --llm is enabled (defaults to gpt-4o-mini).",
    ),
    as_of: Optional[str] = typer.Option(
        None,
        help="Override the as-of date (YYYY-MM-DD).",
    ),
):
    """Blend fundamentals and news agents into a 5–6 point summary."""

    agent = WeightSynthesisAgent()
    try:
        report = agent.generate_report(
            ticker,
            weight,
            as_of=as_of,
            lookback_days=lookback_days,
            max_articles=max_articles,
            use_llm=use_llm,
            llm_model=llm_model,
        )
    except ValueError as err:
        console.print(f"[red]{err}[/red]")
        raise typer.Exit(code=1) from err
    except Exception as err:  # noqa: BLE001
        console.print(f"[red]Combined summary failed: {err}[/red]")
        raise typer.Exit(code=1) from err

    console.print(
        Markdown(
            report.to_markdown(
                include_components=include_components,
                include_metrics=include_metrics,
                include_articles=include_articles,
            )
        )
    )

    if report.generated_via_llm:
        console.print("\n[dim]Unified summary generated via LLM.[/dim]")
    elif use_llm:
        reason = llm_client.LAST_LLM_ERROR or "LLM call returned no content."
        console.print(f"\n[yellow]LLM path skipped: {reason}[/yellow]")


if __name__ == "__main__":
    app()
