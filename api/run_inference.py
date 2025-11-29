#!/usr/bin/env python3
"""
CLI inference runner to test the pipeline without hitting the FastAPI endpoints.

Example:
    python api/run_inference.py \
        --model-path checkpoints/baseline.zip \
        --market custom \
        --horizon 1 \
        --relation-type hy \
        --test-start-date 2024-01-02 \
        --test-end-date 2024-01-31 \
        --output-dir ./logs/api_cli
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
from gen_data import update_monthly_dataset


def parse_args():
    ap = argparse.ArgumentParser(description="Run PPO inference and save weights/metrics.")
    ap.add_argument("--model-path", required=True, help="Path to PPO checkpoint (.zip)")
    ap.add_argument("--market", default="custom")
    ap.add_argument("--horizon", default="1")
    ap.add_argument("--relation-type", default="hy")
    ap.add_argument("--test-start-date", required=True)
    ap.add_argument("--test-end-date", required=True)
    ap.add_argument("--deterministic", action="store_true", help="Use deterministic actions")
    ap.add_argument("--output-dir", default="./logs/api_cli", help="Where to save weights CSV")
    ap.add_argument("--lookback", type=int, default=30)
    ap.add_argument("--risk-score", type=float, default=0.5)
    ap.add_argument("--ind-yn", action="store_true", default=True)
    ap.add_argument("--pos-yn", action="store_true", default=True)
    ap.add_argument("--neg-yn", action="store_true", default=True)
    return ap.parse_args()


def dataset_dir(args):
    return Path("dataset_default") / f"data_train_predict_{args.market}" / f"{args.horizon}_{args.relation_type}"


def ensure_dataset(args):
    data_dir = dataset_dir(args)
    if data_dir.exists() and any(data_dir.glob("*.pkl")):
        return
    print(f"Dataset missing at {data_dir}, attempting to build via update_monthly_dataset...")
    updater_args = argparse.Namespace(
        market=args.market,
        horizon=int(args.horizon),
        relation_type=args.relation_type,
        lookback=args.lookback,
        threshold=0.5,
        n_clusters=8,
        disable_norm=False,
        dataset_root=None,
        corr_root=None,
        raw_path=None,
        tickers_file=None,
    )
    update_monthly_dataset.run(updater_args)
    if not data_dir.exists() or not any(data_dir.glob("*.pkl")):
        raise FileNotFoundError(f"Dataset still missing at {data_dir} after auto-build.")


def load_test_loader(args):
    ensure_dataset(args)
    data_dir = dataset_dir(args)
    test_dataset = AllGraphDataSampler(
        base_dir=str(data_dir),
        date=True,
        test_start_date=args.test_start_date,
        test_end_date=args.test_end_date,
        mode="test",
    )
    if len(test_dataset) == 0:
        raise ValueError(f"Empty test dataset for range {args.test_start_date} to {args.test_end_date}")
    return DataLoader(test_dataset, batch_size=len(test_dataset), pin_memory=True)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = load_test_loader(args)

    model_path = Path(args.model_path).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model = PPO.load(str(model_path), env=None, device=device)

    records = []
    all_weights_data = []

    for batch_idx, data in enumerate(test_loader):
        corr, ts_features, features, ind, pos, neg, labels, pyg_data, mask = process_data(data, device=device)
        args_stub = argparse.Namespace(
            risk_score=args.risk_score,
            ind_yn=args.ind_yn,
            pos_yn=args.pos_yn,
            neg_yn=args.neg_yn,
            lookback=args.lookback,
            input_dim=features.shape[-1],
        )
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
            benchmark_return=None,
            mode="test",
            ind_yn=args.ind_yn,
            pos_yn=args.pos_yn,
            neg_yn=args.neg_yn,
            risk_profile={"risk_score": args.risk_score},
        )

        obs_test = env_test.reset()
        max_step = len(labels)

        for _ in range(max_step):
            action, _states = model.predict(obs_test, deterministic=args.deterministic)
            obs_test, reward, done, info = env_test.step(action)
            if done:
                break

        metrics, benchmark_metrics = env_test.evaluate()
        metrics_record = {"batch": batch_idx}
        metrics_record.update(metrics)
        if benchmark_metrics:
            metrics_record.update({f"benchmark_{k}": v for k, v in benchmark_metrics.items()})
        records.append(metrics_record)

        weights_array = env_test.get_weights_history()
        if weights_array.size > 0:
            num_steps, num_stocks = weights_array.shape
            step_labels = getattr(env_test, "dates", list(range(num_steps)))
            for step_idx in range(num_steps):
                weights = weights_array[step_idx]
                step_value = step_labels[step_idx] if step_idx < len(step_labels) else step_idx
                for idx, weight in enumerate(weights):
                    if weight > 0.0001:
                        all_weights_data.append({
                            "run_id": f"cli_run_{batch_idx}",
                            "batch": batch_idx,
                            "date": None,
                            "step": step_value,
                            "ticker": f"stock_{idx}",
                            "weight": float(weight),
                            "weight_pct": float(weight * 100),
                        })

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_csv = out_dir / f"weights_{args.market}_{args.test_start_date}_{args.test_end_date}.csv"
    if all_weights_data:
        pd.DataFrame(all_weights_data).to_csv(weights_csv, index=False)
        print(f"Saved weights to {weights_csv}")
    else:
        print("No weights captured.")

    print("Metrics:", records)


if __name__ == "__main__":
    main()
