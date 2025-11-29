#!/usr/bin/env python3
"""
Generate a monthly allocation for PMS-style deployment.

Logic:
- Load the latest available data in the specified test window (e.g., month-end).
- Run a short rollout (deterministic) over that window with monthly rebalance logic.
- Take the final rebalance allocation (last weights) as the allocation for the next month.
- Save the allocation to CSV with a validity tag.

Example:
    python api/run_monthly_allocation.py \
      --model-path checkpoints/baseline.zip \
      --market custom \
      --horizon 1 \
      --relation-type hy \
      --start-date 2024-01-02 \
      --end-date 2024-01-31 \
      --output-dir ./logs/monthly_alloc
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from stable_baselines3 import PPO

from dataloader.data_loader import AllGraphDataSampler
from env.portfolio_env import StockPortfolioEnv
from trainer.irl_trainer import process_data
from utils.ticker_mapping import get_ticker_mapping_for_period

def parse_args():
    ap = argparse.ArgumentParser(description="Generate monthly allocation from PPO model.")
    ap.add_argument("--model-path", required=True, help="Path to PPO checkpoint (.zip)")
    ap.add_argument("--market", default="custom")
    ap.add_argument("--horizon", default="1")
    ap.add_argument("--relation-type", default="hy")
    ap.add_argument("--start-date", required=True, help="Start of inference window (e.g., month)")
    ap.add_argument("--end-date", required=True, help="End of inference window (e.g., month end)")
    ap.add_argument("--deterministic", action="store_true", help="Use deterministic actions")
    ap.add_argument("--output-dir", default="./logs/monthly_alloc", help="Where to save allocation CSV")
    ap.add_argument("--lookback", type=int, default=30)
    ap.add_argument("--risk-score", type=float, default=0.5)
    ap.add_argument("--ind-yn", action="store_true", default=True)
    ap.add_argument("--pos-yn", action="store_true", default=True)
    ap.add_argument("--neg-yn", action="store_true", default=True)
    return ap.parse_args()


def dataset_dir(args):
    return Path("dataset_default") / f"data_train_predict_{args.market}" / f"{args.horizon}_{args.relation_type}"


def load_window(args):
    data_dir = dataset_dir(args)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    ds = AllGraphDataSampler(
        base_dir=str(data_dir),
        date=True,
        test_start_date=args.start_date,
        test_end_date=args.end_date,
        mode="test",
    )
    if len(ds) == 0:
        raise ValueError(f"No data in window {args.start_date} to {args.end_date}")
    return DataLoader(ds, batch_size=len(ds), pin_memory=True)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = load_window(args)
    model = PPO.load(str(Path(args.model_path).expanduser()), env=None, device=device)

    final_weights = None
    ticker_map = get_ticker_mapping_for_period(
        args.market, args.start_date, args.end_date, base_dir="dataset_default"
    )

    for batch_idx, data in enumerate(loader):
        corr, ts_features, features, ind, pos, neg, labels, pyg_data, mask = process_data(data, device=device)
        args_stub = argparse.Namespace(
            risk_score=args.risk_score,
            ind_yn=args.ind_yn,
            pos_yn=args.pos_yn,
            neg_yn=args.neg_yn,
            lookback=args.lookback,
            input_dim=features.shape[-1],
        )
        df_benchmark = pd.read_csv(f"./dataset_default/index_data/{args.market}_index.csv")
        df_benchmark = df_benchmark[(df_benchmark['datetime'] >= args.start_date) &
                                    (df_benchmark['datetime'] <= args.end_date)]
        benchmark_return = df_benchmark['daily_return']
        env_test = StockPortfolioEnv(
            args=args_stub,
            corr=corr,
            ts_features=ts_features,
            features=features,
            ind=ind,
            pos=pos,
            neg=neg,
            returns=labels,
            pyg_data=pyg_data,
            benchmark_return=benchmark_return,
            mode="test",
            ind_yn=args.ind_yn,
            pos_yn=args.pos_yn,
            neg_yn=args.neg_yn,
            risk_profile={"risk_score": args.risk_score},
        )
        obs = env_test.reset()
        max_step = len(labels)
        for _ in range(max_step):
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, done, _ = env_test.step(action)
            if done:
                break
        weights_history = env_test.get_weights_history()
        if weights_history.size == 0:
            raise RuntimeError("No weights captured during rollout.")
        final_weights = weights_history[-1]
        # Map tickers if available
        date_keys = list(ticker_map.keys()) if ticker_map else []
        tickers = None
        if date_keys:
            # Use the last date in range
            last_date = sorted(date_keys)[-1]
            candidate = ticker_map.get(last_date)
            if candidate and len(candidate) == len(final_weights):
                tickers = candidate
        if tickers is None:
            tickers = [f"stock_{i}" for i in range(len(final_weights))]
        break  # only first batch

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "ticker": tickers,
        "weight": final_weights,
        "weight_pct": final_weights * 100,
        "valid_from": args.end_date,
        "valid_through": None,  # can fill with next rebalance date
        "risk_score": args.risk_score,
    })
    out_path = out_dir / f"allocation_{args.market}_{args.end_date}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved monthly allocation to {out_path}")


if __name__ == "__main__":
    main()
