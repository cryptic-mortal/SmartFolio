#!/usr/bin/env python3
"""
Quick inspection tool for saved allocation CSVs.
Loads weights_{split}_*.csv produced by trainer/irl_trainer.py, then:
 - Plots per-ticker weight trajectories over steps
 - Computes turnover between consecutive steps to see persistence
 - Prints basic stats (mean, std, min, max) per ticker

Usage:
    python tools/weights_persistence_viz.py --weights-csv path/to/test_weights_YYYYMMDD_HHMMSS.csv \
        [--tickers-csv tickers.csv] [--top-k 10] [--plot]
"""

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def load_tickers(csv_path: str, expected_count: int | None = None) -> List[str]:
    if not csv_path:
        return []
    path = Path(csv_path).expanduser()
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
        if "ticker" in df.columns:
            tickers = df["ticker"].dropna().astype(str).tolist()
        else:
            tickers = df.iloc[:, 0].dropna().astype(str).tolist()
        if expected_count and len(tickers) != expected_count:
            print(f"[WARN] ticker count {len(tickers)} != expected {expected_count}; using generic names beyond provided list")
        return tickers
    except Exception as exc:
        print(f"[WARN] failed to load tickers from {path}: {exc}")
        return []


def turnover(weights: np.ndarray) -> np.ndarray:
    """L1 turnover between consecutive steps."""
    if weights.shape[0] < 2:
        return np.array([])
    diffs = np.abs(weights[1:] - weights[:-1]).sum(axis=1)
    return diffs


def summarize(weights_df: pd.DataFrame, top_k: int, tickers: List[str]) -> None:
    grouped = weights_df.groupby("ticker")["weight"]
    stats = grouped.agg(["mean", "std", "min", "max", "count"]).sort_values("mean", ascending=False)
    print("\nTop tickers by mean weight:")
    print(stats.head(top_k).round(6))

    # Persistence metrics
    print("\nPer-ticker persistence (Spearman rank corr vs. previous step):")
    steps = sorted(weights_df["step"].unique())
    ticker_list = tickers if tickers else sorted(weights_df["ticker"].unique())
    pivot = weights_df.pivot_table(index="step", columns="ticker", values="weight", aggfunc="mean").reindex(
        columns=ticker_list, fill_value=0
    ).fillna(0)
    from scipy.stats import spearmanr  # Local import to avoid hard dependency if unused
    corrs = []
    for i in range(len(steps) - 1):
        r, _ = spearmanr(pivot.iloc[i].values, pivot.iloc[i + 1].values)
        corrs.append(r)
    if corrs:
        corrs_np = np.array(corrs, dtype=np.float32)
        print(f"  Spearman rho w_t vs w_(t-1): mean={corrs_np.mean():.4f}, std={corrs_np.std():.4f}, "
              f"min={corrs_np.min():.4f}, max={corrs_np.max():.4f}")
    else:
        print("  Not enough steps for Spearman correlation.")


def rank_frequency(weights_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """
    Compute how often each ticker appears at each rank (0=highest) per step.
    Returns a DataFrame with rows=ranks, cols=tickers.
    """
    if weights_df.empty:
        return pd.DataFrame()
    # Pivot to step x ticker
    pivot = weights_df.pivot_table(index="step", columns="ticker", values="weight", aggfunc="mean").fillna(0)
    # Argsort descending to get ranks per step
    ranks = pivot.apply(lambda row: np.argsort(-row.values), axis=1, result_type="expand")
    # Map positions to ticker names
    ticker_list = list(pivot.columns)
    rank_counts = pd.DataFrame(0, index=range(min(top_k, len(ticker_list))), columns=ticker_list)
    for _, row in ranks.iterrows():
        ranked_indices = row.values.astype(int)
        for rank_pos, idx in enumerate(ranked_indices[: rank_counts.shape[0]]):
            ticker = ticker_list[idx]
            rank_counts.at[rank_pos, ticker] += 1
    return rank_counts


def plot(weights_df: pd.DataFrame, tickers: List[str], top_k: int, save_path: Path | None = None) -> None:
    if plt is None:
        print("matplotlib not available; skipping plots")
        return
    top_tickers = (
        weights_df.groupby("ticker")["weight"].mean().sort_values(ascending=False).head(top_k).index.tolist()
    )
    subset = weights_df[weights_df["ticker"].isin(top_tickers)]
    pivot = subset.pivot(index="step", columns="ticker", values="weight").fillna(0)

    pivot.plot(figsize=(12, 6))
    plt.xlabel("Step")
    plt.ylabel("Weight")
    plt.title(f"Top {top_k} ticker weights over time")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved weight trajectories plot to {save_path}")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser(description="Inspect allocation persistence from saved weights CSV.")
    ap.add_argument("--weights-csv", required=True, help="Path to weights CSV produced by trainer/irl_trainer.py")
    ap.add_argument("--tickers-csv", default=None, help="Optional tickers CSV (with 'ticker' column) for labels")
    ap.add_argument("--top-k", type=int, default=10, help="Number of tickers to highlight")
    ap.add_argument("--plot", action="store_true", help="Show/save plots (requires matplotlib)")
    ap.add_argument("--save-plot", default=None, help="Optional path to save the plot instead of showing")
    ap.add_argument("--top-k-overlap", type=int, default=10, help="K for top-K overlap and rank stats")
    args = ap.parse_args()

    weights_path = Path(args.weights_csv).expanduser()
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights CSV not found: {weights_path}")

    df = pd.read_csv(weights_path)
    if df.empty:
        raise ValueError("Weights CSV is empty.")

    tickers = load_tickers(args.tickers_csv, expected_count=df["ticker"].nunique() if "ticker" in df else None)

    # Ensure required columns
    required_cols = {"step", "ticker", "weight"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Weights CSV missing required columns {required_cols}. Found columns: {df.columns.tolist()}")

    summarize(df, args.top_k, tickers)

    # Rank frequency table
    rf = rank_frequency(df, args.top_k)
    if not rf.empty:
        rf_path = weights_path.with_name(weights_path.stem + "_rank_frequency.csv")
        rf.to_csv(rf_path, index_label="rank")
        print(f"\nSaved rank frequency table to {rf_path}")

    # Compute turnover
    steps = sorted(df["step"].unique())
    tickers_list = tickers if tickers else sorted(df["ticker"].unique())
    # Build a step-by-ticker matrix
    matrix = np.zeros((len(steps), len(tickers_list)), dtype=np.float32)
    ticker_to_idx = {t: i for i, t in enumerate(tickers_list)}
    step_to_idx = {s: i for i, s in enumerate(steps)}
    for _, row in df.iterrows():
        ti = ticker_to_idx.get(row["ticker"], None)
        si = step_to_idx.get(row["step"], None)
        if ti is None or si is None:
            continue
        matrix[si, ti] = row["weight"]

    to = turnover(matrix)
    if to.size > 0:
        print(f"\nTurnover stats (L1 change per step): mean={to.mean():.4f}, std={to.std():.4f}, "
              f"min={to.min():.4f}, max={to.max():.4f}")
    else:
        print("\nNot enough steps to compute turnover.")

    # Cosine similarity between consecutive steps
    if matrix.shape[0] >= 2:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
        normed = matrix / norms
        cos = (normed[1:] * normed[:-1]).sum(axis=1)
        print(f"\nCosine similarity w_t vs w_(t-1): mean={cos.mean():.4f}, std={cos.std():.4f}, "
              f"min={cos.min():.4f}, max={cos.max():.4f}")

        # Top-K overlap (Jaccard) per step
        k = min(args.top_k_overlap, matrix.shape[1])
        overlaps = []
        for i in range(matrix.shape[0] - 1):
            top_t = set(np.argsort(-matrix[i])[:k])
            top_prev = set(np.argsort(-matrix[i + 1])[:k])
            inter = len(top_t & top_prev)
            union = len(top_t | top_prev)
            overlaps.append(inter / union if union > 0 else 0.0)
        overlaps = np.array(overlaps)
        print(f"Top-{k} overlap (Jaccard) w_t vs w_(t-1): mean={overlaps.mean():.4f}, std={overlaps.std():.4f}, "
              f"min={overlaps.min():.4f}, max={overlaps.max():.4f}")
    else:
        print("\nNot enough steps for cosine/overlap metrics.")

    if args.plot:
        save_path = Path(args.save_plot).expanduser() if args.save_plot else None
        plot(df, tickers_list, args.top_k, save_path)


if __name__ == "__main__":
    main()
