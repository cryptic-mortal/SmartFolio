#!/usr/bin/env python3
"""
Analyze stability of top-N allocations across steps.

Inputs: weights CSV (from trainer/irl_trainer.py) with columns: step, ticker, weight.
Outputs: prints stability metrics and saves CSVs for:
  - Top-N weights per step
  - Position change frequencies
Metrics:
  - Top-N overlap (Jaccard) between consecutive steps
  - Position change rate for each ticker in top-N
  - Weight change stats for top-N tickers
  - Holding time in top-N (consecutive steps)
"""

import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd


def load_weights(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = {"step", "ticker", "weight"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Weights CSV missing required columns {required_cols}. Found: {df.columns.tolist()}")
    return df


def top_n_per_step(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Return subset with only top-n tickers per step (by weight)."""
    top_rows = []
    for step, grp in df.groupby("step"):
        top = grp.nlargest(n, "weight")
        top_rows.append(top)
    return pd.concat(top_rows, ignore_index=True)


def top_overlap(matrix: np.ndarray, n: int) -> np.ndarray:
    k = min(n, matrix.shape[1])
    overlaps = []
    for i in range(matrix.shape[0] - 1):
        top_t = set(np.argsort(-matrix[i])[:k])
        top_prev = set(np.argsort(-matrix[i + 1])[:k])
        inter = len(top_t & top_prev)
        union = len(top_t | top_prev)
        overlaps.append(inter / union if union > 0 else 0.0)
    return np.array(overlaps)


def position_changes(top_df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Compute how often tickers move positions within top-n between steps."""
    position_logs: Dict[str, List[int]] = {}
    for step, grp in top_df.groupby("step"):
        ordered = grp.sort_values("weight", ascending=False).reset_index(drop=True)
        for pos, row in ordered.iterrows():
            ticker = row["ticker"]
            position_logs.setdefault(ticker, []).append(pos)
    # Compute position change frequency
    changes = []
    for ticker, positions in position_logs.items():
        if len(positions) < 2:
            changes.append((ticker, 0.0))
            continue
        diffs = np.diff(positions)
        changes.append((ticker, np.mean(np.abs(diffs))))
    return pd.DataFrame(changes, columns=["ticker", "avg_abs_position_change"]).sort_values("avg_abs_position_change")


def holding_times(top_df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Compute average consecutive holding time in top-n for each ticker."""
    holds: Dict[str, List[int]] = {}
    for ticker, grp in top_df.groupby("ticker"):
        steps_sorted = sorted(grp["step"].unique())
        runs = []
        current_run = 1
        for i in range(1, len(steps_sorted)):
            if steps_sorted[i] == steps_sorted[i - 1] + 1:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        runs.append(current_run)
        holds[ticker] = runs
    rows = []
    for ticker, runs in holds.items():
        rows.append(
            {
                "ticker": ticker,
                "avg_consecutive_topN_steps": float(np.mean(runs)),
                "max_consecutive_topN_steps": max(runs),
                "entries_topN": len(runs),
            }
        )
    return pd.DataFrame(rows).sort_values("avg_consecutive_topN_steps", ascending=False)


def main():
    ap = argparse.ArgumentParser(description="Top-N allocation stability analysis.")
    ap.add_argument("--weights-csv", required=True, help="Path to weights CSV from evaluation")
    ap.add_argument("--top-n", type=int, default=20, help="Top-N tickers to analyze")
    ap.add_argument("--output-dir", default=None, help="Optional directory to write CSV outputs")
    args = ap.parse_args()

    weights_path = Path(args.weights_csv).expanduser()
    df = load_weights(weights_path)
    n = args.top_n

    top_df = top_n_per_step(df, n)

    steps = sorted(df["step"].unique())
    tickers = sorted(df["ticker"].unique())
    matrix = np.zeros((len(steps), len(tickers)), dtype=np.float32)
    t2i = {t: i for i, t in enumerate(tickers)}
    s2i = {s: i for i, s in enumerate(steps)}
    for _, row in df.iterrows():
        matrix[s2i[row["step"]], t2i[row["ticker"]]] = row["weight"]

    overlaps = top_overlap(matrix, n)
    if overlaps.size > 0:
        print(f"Top-{n} overlap (Jaccard) w_t vs w_(t-1): mean={overlaps.mean():.4f}, std={overlaps.std():.4f}, "
              f"min={overlaps.min():.4f}, max={overlaps.max():.4f}")
    else:
        print("Not enough steps for overlap metric.")

    pos_changes = position_changes(top_df, n)
    holds = holding_times(top_df, n)

    # Weight change stats for top-N
    weight_changes = []
    for ticker in top_df["ticker"].unique():
        series = top_df[top_df["ticker"] == ticker].sort_values("step")["weight"].values
        if len(series) < 2:
            continue
        diffs = np.diff(series)
        weight_changes.append(
            (ticker, float(np.mean(np.abs(diffs))), float(np.std(diffs)), float(np.max(np.abs(diffs)))))
    weight_changes_df = pd.DataFrame(weight_changes, columns=["ticker", "mean_abs_delta", "std_delta", "max_abs_delta"])\
        .sort_values("mean_abs_delta")

    out_dir = Path(args.output_dir) if args.output_dir else weights_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    top_out = out_dir / f"{weights_path.stem}_top{n}.csv"
    pos_out = out_dir / f"{weights_path.stem}_position_changes_top{n}.csv"
    hold_out = out_dir / f"{weights_path.stem}_holding_times_top{n}.csv"
    delta_out = out_dir / f"{weights_path.stem}_weight_changes_top{n}.csv"

    top_df.to_csv(top_out, index=False)
    pos_changes.to_csv(pos_out, index=False)
    holds.to_csv(hold_out, index=False)
    weight_changes_df.to_csv(delta_out, index=False)

    print(f"Saved top-{n} allocations to {top_out}")
    print(f"Saved top-{n} position change stats to {pos_out}")
    print(f"Saved top-{n} holding times to {hold_out}")
    print(f"Saved top-{n} weight change stats to {delta_out}")


if __name__ == "__main__":
    main()
