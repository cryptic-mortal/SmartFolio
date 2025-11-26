from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pathway as pw

from tradingagents.fundamental_agent import FundamentalWeightAgent, WeightReport
from tradingagents.news_agent import (
    NewsArticle,
    NewsWeightReport,
    NewsWeightReviewAgent,
)
from tradingagents.llm_client import summarise_weight_points


@dataclass
class WeightSynthesisReport:
	ticker: str
	weight: float
	as_of: str
	lookback_days: int
	summary_points: List[str]
	fundamental_report: WeightReport
	news_report: NewsWeightReport
	generated_via_llm: bool = False

	def to_markdown(
		self,
		*,
		include_components: bool = False,
		include_metrics: bool = True,
		include_articles: bool = True,
	) -> str:
		header = (
			f"# Combined Weight Review: {self.ticker}\n\n"
			f"- **As of:** {self.as_of}\n"
			f"- **Assigned Weight:** {self.weight:.2%}\n"
			f"- **News Lookback:** {self.lookback_days} day(s)\n\n"
		)

		summary_section = "\n".join(f"- {point}" for point in self.summary_points)

		sections = [header, "## Unified Summary\n", summary_section, "\n"]

		if include_components:
			fund_markdown = _strip_top_heading(
				self.fundamental_report.to_markdown(include_metrics=include_metrics)
			)
			news_markdown = _strip_top_heading(
				self.news_report.to_markdown(include_articles=include_articles)
			)
			sections.extend(
				[
					"## Fundamental Agent Detail\n",
					fund_markdown + "\n\n",
					"## News Agent Detail\n",
					news_markdown + "\n",
				]
			)

		return "".join(sections)


class WeightSynthesisAgent:
	"""Coordinates fundamental and news agents to deliver a unified view via Pathway pipelines."""

	def __init__(self):
		self._fundamental_agent = FundamentalWeightAgent()
		self._news_agent = NewsWeightReviewAgent()

	def generate_report(
		self,
		ticker: str,
		weight: float,
		*,
		as_of: Optional[str] = None,
		lookback_days: int = 7,
		max_articles: int = 8,
		use_llm: bool = False,
		llm_model: Optional[str] = None,
	) -> WeightSynthesisReport:
		payload = _run_combined_pipeline(
			ticker=ticker.strip().upper(),
			weight=float(weight),
			as_of=as_of,
			lookback_days=int(lookback_days),
			max_articles=int(max_articles),
			use_llm=bool(use_llm),
			llm_model=llm_model,
			fundamental_agent=self._fundamental_agent,
			news_agent=self._news_agent,
		)
		return _payload_to_combined_report(payload)


def _run_combined_pipeline(
	*,
	ticker: str,
	weight: float,
	as_of: Optional[str],
	lookback_days: int,
	max_articles: int,
	use_llm: bool,
	llm_model: Optional[str],
	fundamental_agent: FundamentalWeightAgent,
	news_agent: NewsWeightReviewAgent,
) -> Dict[str, Any]:
	report = _build_combined_report(
		ticker=ticker,
		weight=weight,
		as_of=as_of,
		lookback_days=lookback_days,
		max_articles=max_articles,
		use_llm=use_llm,
		llm_model=llm_model,
		fundamental_agent=fundamental_agent,
		news_agent=news_agent,
	)
	payload = _combined_report_to_payload(report)
	table = pw.debug.table_from_rows(
		pw.schema_from_types(payload=pw.Json),
		[(pw.Json(payload),)],
	)
	keys, columns = pw.debug.table_to_dicts(table)
	if not keys:
		raise RuntimeError("Combined synthesis produced no output.")
	stored = columns["payload"][keys[0]]
	if isinstance(stored, pw.Json):
		return stored.value
	return stored


def _build_combined_report(
	*,
	ticker: str,
	weight: float,
	as_of: Optional[str],
	lookback_days: int,
	max_articles: int,
	use_llm: bool,
	llm_model: Optional[str],
	fundamental_agent: FundamentalWeightAgent,
	news_agent: NewsWeightReviewAgent,
) -> WeightSynthesisReport:
	if not ticker:
		raise ValueError("Ticker symbol cannot be empty")
	if not (0.0 <= weight <= 1.0):
		raise ValueError("Weight must be between 0.0 and 1.0 inclusive")
	if lookback_days <= 0:
		raise ValueError("lookback_days must be positive")
	if max_articles <= 0:
		raise ValueError("max_articles must be positive")

	fundamental_report = fundamental_agent.generate_report(
		ticker,
		weight,
		as_of=as_of,
	)
	news_report = news_agent.generate_report(
		ticker,
		weight,
		as_of=as_of,
		lookback_days=lookback_days,
		max_articles=max_articles,
	)

	summary_points = _synthesise_summary(fundamental_report, news_report)
	used_llm = False
	if use_llm:
		fund_markdown = fundamental_report.to_markdown(include_metrics=True)
		news_markdown = news_report.to_markdown(include_articles=True)
		llm_points = summarise_weight_points(
			ticker=fundamental_report.ticker,
			weight=weight,
			as_of=fundamental_report.as_of,
			fundamental_points=fundamental_report.rationale_points,
			news_points=news_report.points,
			metrics_table=fund_markdown,
			news_table=news_markdown,
			max_points=len(summary_points) or 6,
			model=llm_model,
		)
		if llm_points:
			cleaned_points = [str(point) for point in llm_points if point]
			if cleaned_points:
				limit = len(summary_points) or 6
				summary_points = cleaned_points[:limit]
				used_llm = True

	return WeightSynthesisReport(
		ticker=fundamental_report.ticker,
		weight=weight,
		as_of=fundamental_report.as_of,
		lookback_days=lookback_days,
		summary_points=summary_points,
		fundamental_report=fundamental_report,
		news_report=news_report,
		generated_via_llm=used_llm,
	)


def _combined_report_to_payload(report: WeightSynthesisReport) -> Dict[str, Any]:
	return {
		"ticker": report.ticker,
		"weight": float(report.weight),
		"as_of": report.as_of,
		"lookback_days": int(report.lookback_days),
		"summary_points": [str(point) for point in report.summary_points],
		"generated_via_llm": bool(report.generated_via_llm),
		"fundamental_report": _fundamental_report_to_dict(report.fundamental_report),
		"news_report": _news_report_to_dict(report.news_report),
	}


def _payload_to_combined_report(payload: Dict[str, Any]) -> WeightSynthesisReport:
	fundamental_payload = payload.get("fundamental_report", {})
	news_payload = payload.get("news_report", {})
	summary_points_raw = payload.get("summary_points", [])

	return WeightSynthesisReport(
		ticker=str(payload.get("ticker", "")),
		weight=float(payload.get("weight", 0.0)),
		as_of=str(payload.get("as_of", "")),
		lookback_days=int(payload.get("lookback_days", 0)),
		summary_points=[str(point) for point in summary_points_raw if point is not None],
		fundamental_report=_fundamental_report_from_dict(fundamental_payload),
		news_report=_news_report_from_dict(news_payload),
		generated_via_llm=bool(payload.get("generated_via_llm", False)),
	)


def _fundamental_report_to_dict(report: WeightReport) -> Dict[str, Any]:
	return {
		"ticker": report.ticker,
		"weight": float(report.weight),
		"as_of": report.as_of,
		"rationale_points": [str(point) for point in report.rationale_points],
		"metrics": {key: (float(value) if value is not None else None) for key, value in report.metrics.items()},
		"generated_via_llm": bool(report.generated_via_llm),
	}


def _fundamental_report_from_dict(payload: Dict[str, Any]) -> WeightReport:
	metrics_payload = payload.get("metrics", {})
	metrics: Dict[str, Optional[float]] = {}
	for key, value in metrics_payload.items():
		if value is None:
			metrics[key] = None
		else:
			try:
				metrics[key] = float(value)
			except (TypeError, ValueError):
				metrics[key] = None

	return WeightReport(
		ticker=str(payload.get("ticker", "")),
		weight=float(payload.get("weight", 0.0)),
		as_of=str(payload.get("as_of", "")),
		rationale_points=[str(point) for point in payload.get("rationale_points", []) if point is not None],
		metrics=metrics,
		generated_via_llm=bool(payload.get("generated_via_llm", False)),
	)


def _news_report_to_dict(report: NewsWeightReport) -> Dict[str, Any]:
	return {
		"ticker": report.ticker,
		"weight": float(report.weight),
		"as_of": report.as_of,
		"lookback_days": int(report.lookback_days),
		"judgement": report.judgement,
		"points": [str(point) for point in report.points],
		"articles": [_news_article_to_dict(article) for article in report.articles],
		"generated_via_llm": bool(report.generated_via_llm),
	}


def _news_report_from_dict(payload: Dict[str, Any]) -> NewsWeightReport:
	articles_payload = payload.get("articles", [])
	articles: List[NewsArticle] = []
	for item in articles_payload:
		if isinstance(item, dict):
			articles.append(_news_article_from_dict(item))

	return NewsWeightReport(
		ticker=str(payload.get("ticker", "")),
		weight=float(payload.get("weight", 0.0)),
		as_of=str(payload.get("as_of", "")),
		lookback_days=int(payload.get("lookback_days", 0)),
		judgement=str(payload.get("judgement", "")),
		points=[str(point) for point in payload.get("points", []) if point is not None],
		articles=articles,
		generated_via_llm=bool(payload.get("generated_via_llm", False)),
	)


def _news_article_to_dict(article: NewsArticle) -> Dict[str, Any]:
	return {
		"headline": article.headline,
		"published_at": article.published_at,
		"summary": article.summary,
		"source": article.source,
		"url": article.url,
		"sentiment": article.sentiment,
		"sentiment_score": int(article.sentiment_score),
	}


def _news_article_from_dict(payload: Dict[str, Any]) -> NewsArticle:
	score = payload.get("sentiment_score", 0)
	try:
		sentiment_score = int(score)
	except (TypeError, ValueError):
		sentiment_score = 0
	return NewsArticle(
		headline=str(payload.get("headline", "")),
		published_at=payload.get("published_at"),
		summary=payload.get("summary"),
		source=payload.get("source"),
		url=payload.get("url"),
		sentiment=str(payload.get("sentiment", "neutral")),
		sentiment_score=sentiment_score,
	)


def _synthesise_summary(
	fund_report: WeightReport,
	news_report: NewsWeightReport,
	*,
	min_points: int = 5,
	max_points: int = 6,
) -> List[str]:
	summary: List[str] = []
	seen = set()

	def add(point: Optional[str]) -> None:
		if not point:
			return
		normalised = " ".join(point.lower().split())
		if normalised in seen:
			return
		seen.add(normalised)
		summary.append(point)

	fund_points = fund_report.rationale_points or []
	news_points = news_report.points or []

	if fund_points:
		add(fund_points[0])
	if news_points:
		add(news_points[0])

	for point in fund_points[1:]:
		if len(summary) >= max_points:
			break
		add(point)

	for point in news_points[1:]:
		if len(summary) >= max_points:
			break
		add(point)

	if len(summary) < min_points:
		metrics_glance = _metrics_snapshot(fund_report.metrics)
		add(metrics_glance)

	if len(summary) < min_points:
		coverage_glance = _news_snapshot(news_report.articles)
		add(coverage_glance)

	while len(summary) < min_points:
		add("Vendor data remains sparse; maintain close monitoring before resizing.")

	return summary[:max_points]


def _metrics_snapshot(metrics: dict) -> Optional[str]:
	key_map = {
		"pe_ratio": ("P/E", lambda v: f"{v:.1f}×"),
		"roe": ("ROE", lambda v: f"{v:.1f}%"),
		"profit_margin": ("margin", lambda v: f"{v:.1f}%"),
		"revenue_growth": ("growth", lambda v: f"{v:.1f}%"),
		"debt_to_equity": ("D/E", lambda v: f"{v:.2f}×"),
	}
	parts = []
	for key in ["pe_ratio", "roe", "profit_margin", "revenue_growth", "debt_to_equity"]:
		value = metrics.get(key)
		if value is None:
			continue
		label, formatter = key_map[key]
		parts.append(f"{label} {formatter(float(value))}")
	if not parts:
		return None
	return "Fundamentals check-in: " + ", ".join(parts)


def _news_snapshot(articles: List[NewsArticle]) -> Optional[str]:
	total = len(articles)
	if total == 0:
		return None
	positives = sum(1 for article in articles if article.sentiment_score > 0)
	negatives = sum(1 for article in articles if article.sentiment_score < 0)
	neutrals = total - positives - negatives
	return (
		f"News cadence: {positives} positive / {negatives} negative / {neutrals} neutral headlines in scope."
	)


def _strip_top_heading(markdown: str) -> str:
	lines = markdown.strip().splitlines()
	if not lines:
		return markdown
	if lines[0].startswith("#"):
		lines = lines[1:]
	return "\n".join(lines).strip()