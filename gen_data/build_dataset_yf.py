import argparse
import os
import pickle
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pandas.tseries.offsets import MonthEnd
from torch.autograd import Variable
from torch_geometric.data import Data
from tqdm import tqdm

# Optional dependency; this file is meant to be run as a standalone data builder
try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None


FEATURE_COLS = ["close", "open", "high", "low", "prev_close", "volume"]
FEATURE_COLS_NORM = [f"{c}_normalized" for c in FEATURE_COLS]

# Output root to match what main.py expects
# main.py looks under dataset_default/data_train_predict_{market}/...
DATASET_DEFAULT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset_default"))
DATASET_CORR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "corr"))


def fetch_ohlcv_yf(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Download OHLCV from yfinance and return tall dataframe with required columns.

    Output columns: kdcode, dt, close, open, high, low, prev_close, volume
    """
    if yf is None:
        raise RuntimeError("yfinance is not installed. Please `pip install yfinance` and try again.")

    # yfinance returns a wide MultiIndex df when multiple tickers
    df = yf.download(tickers, start=start, end=end, auto_adjust=False, group_by="ticker", progress=False)
    print(df.columns)

    # Normalize to a tall dataframe
    if isinstance(df.columns, pd.MultiIndex):
        parts = []
        for t in tickers:
            if (t, "Close") not in df.columns:
                # ticker may have failed; skip
                continue
            sub = df[(t,)].copy()
            sub.columns = [c.lower() for c in sub.columns]
            sub["kdcode"] = t
            parts.append(sub.reset_index().rename(columns={"Date": "dt"}))
        tall = pd.concat(parts, ignore_index=True)
    else:
        # Single ticker case
        d = df.copy()
        d.columns = [c.lower() for c in d.columns]
        d["kdcode"] = tickers[0]
        tall = d.reset_index().rename(columns={"Date": "dt"})

    # Ensure required columns exist
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(tall.columns)
    if missing:
        raise ValueError(f"Missing required columns from yfinance output: {missing}")

    # Compute prev_close per kdcode
    tall = tall.sort_values(["kdcode", "dt"]).reset_index(drop=True)
    tall["prev_close"] = tall.groupby("kdcode")["close"].shift(1)

    # Drop first row per ticker where prev_close is NaN; keep consistent entries
    tall = tall.dropna(subset=["prev_close"]).copy()

    # Cast dt to string YYYY-MM-DD to match existing dataset files
    tall["dt"] = pd.to_datetime(tall["dt"]).dt.strftime("%Y-%m-%d")

    # Keep only needed columns
    tall = tall[["kdcode", "dt", "close", "open", "high", "low", "prev_close", "volume"]]
    return tall


def get_label(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    df = df.copy()
    df.set_index("kdcode", inplace=True)
    for code, group in df.groupby("kdcode"):
        group = group.set_index("dt").sort_index()
        group["return"] = group["close"].shift(-horizon) / group["close"] - 1
        df.loc[code, "label"] = group["return"].values
    df = df.dropna().reset_index()
    return df


def cal_rolling_mean_std(df: pd.DataFrame, cal_cols: List[str], lookback: int = 5) -> pd.DataFrame:
    df = df.sort_values(by=["kdcode", "dt"])  # sort by ticker, date
    for col in cal_cols:
        df[f"{col}_mean"] = df.groupby("kdcode")[col].transform(
            lambda x: x.rolling(window=lookback, min_periods=1).mean()
        )
        df[f"{col}_std"] = df.groupby("kdcode")[col].transform(
            lambda x: x.rolling(window=lookback, min_periods=1).std()
        )
    df = df.dropna().reset_index(drop=True)
    return df


def _zscore_safe(series: pd.Series) -> pd.Series:
    """Return z-score, guarding against zero or NaN std."""
    mean = series.mean()
    std = series.std()
    if pd.isna(std) or std < 1e-8:
        # If variance is zero, all values are identical; return zeros.
        return series.map(lambda _: 0.0)
    return (series - mean) / std


def group_and_norm(df: pd.DataFrame, base_cols: List[str], n_clusters: int) -> pd.DataFrame:
    result = []
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df = df.sort_values(by=["kdcode", "dt"])  # by ticker/date
    for date, group in df.groupby("dt"):
        group = group.copy()
        cluster_features = group[base_cols].fillna(0)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(cluster_features)
        group["cluster"] = kmeans.fit_predict(features_scaled)
        # Merge tiny clusters into nearest
        group_sizes = group["cluster"].value_counts()
        small_clusters = group_sizes[group_sizes < 2].index
        for cl in small_clusters:
            mask = group["cluster"] == cl
            cluster_data = group[mask]
            other_data = group[~mask]
            if other_data.empty or cluster_data.empty:
                continue
            distances = np.linalg.norm(other_data[base_cols].values[:, np.newaxis] - cluster_data[base_cols].values, axis=2)
            closest_cluster_indices = np.argmin(distances, axis=0)
            closest_clusters = other_data.iloc[closest_cluster_indices]["cluster"].values
            group.loc[mask, "cluster"] = closest_clusters
        # Z-score within cluster
        for f in FEATURE_COLS:
            group[f"{f}_normalized"] = group.groupby("cluster")[f].transform(_zscore_safe)
        result.append(group)
    return pd.concat(result)


def filter_code(df: pd.DataFrame) -> List[str]:
    dts = set(df["dt"])  # all dates
    valid_codes = df.groupby("kdcode")["dt"].apply(set)
    # tickers that appear on all dates
    return valid_codes[valid_codes.apply(lambda x: x == dts)].index.to_list()


def get_relation_dt(str_year: str, str_month: str, stock_trade_dt_s: List[str]) -> str:
    month_dts = [k for k in stock_trade_dt_s if k > f"{str_year}-{str_month}" and k < f"{str_year}-{str_month}-32"]
    relation_dt = pd.to_datetime(month_dts[0]) + MonthEnd(1)
    return relation_dt.strftime("%Y-%m-%d")


def gen_mats_by_threshold(corr_df: pd.DataFrame, threshold: float = 0.2):
    mat = corr_df.values
    pos_adj = (mat > threshold).astype(np.float32)
    neg_adj = (mat < -threshold).astype(np.float32)
    # remove self loops
    np.fill_diagonal(pos_adj, 0.0)
    np.fill_diagonal(neg_adj, 0.0)
    return pos_adj, neg_adj


def compute_monthly_corrs(df: pd.DataFrame, market: str, lookback_days: int = 20, out_root: Optional[str] = None):
    """Compute monthly correlation matrices on rolling window and save to dataset/corr/{market}/YYYY-MM-DD.csv.

    This replicates gen_data/generate_relation.py behavior.
    """
    out_root = out_root or DATASET_CORR_ROOT
    corr_path = os.path.join(out_root, market)
    os.makedirs(corr_path, exist_ok=True)

    df_local = df.copy()
    df_local["dt"] = pd.to_datetime(df_local["dt"], format="%Y-%m-%d")
    date_unique = df_local["dt"].dt.strftime("%Y-%m-%d").unique().tolist()
    date_unique.sort()

    # last trading day per calendar month in the range
    last_days = []
    calendar = sorted(set([d[:7] for d in date_unique]))
    for y_m in calendar:
        year, month = y_m.split("-")
        month_days = [k for k in date_unique if k > f"{year}-{month}" and k < f"{year}-{month}-32"]
        if not month_days:
            continue
        last_days.append(month_days[-1])

    codes = sorted(df_local.kdcode.unique().tolist())

    for end_date in tqdm(last_days, desc="Monthly corrs"):
        end_idx = date_unique.index(end_date)
        # For early months with insufficient lookback, use all available data
        start_idx = max(0, end_idx - (lookback_days - 1))
        start_date = date_unique[start_idx]
        window = df_local[(df_local["dt"] >= start_date) & (df_local["dt"] <= end_date)]
        # Build feature tensor: per code, stack features over time
        feat_dict = {}
        actual_lookback = end_idx - start_idx + 1
        for code in codes:
            sub = window[window["kdcode"] == code]
            y = sub[["close", "open", "high", "low", "prev_close", "volume"]].values
            # ensure complete window (or at least 2 days for correlation)
            if y.shape[0] == actual_lookback and y.shape[0] >= 2:
                feat_dict[code] = y.T  # shape [F, T]
        # Align codes for this window
        valid_codes = list(feat_dict.keys())
        if len(valid_codes) < 2:
            continue
        # Compute simple Pearson correlation between flattened feature windows
        X = np.stack([feat_dict[c].reshape(-1) for c in valid_codes], axis=0)  # [N, F*T]
        # standardize across features to avoid scale dominance
        X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
        corr = np.corrcoef(X)
        corr_df = pd.DataFrame(corr, index=valid_codes, columns=valid_codes).fillna(0)

        # Ensure diagonal is 1
        for i in range(len(valid_codes)):
            corr_df.iat[i, i] = 1.0

        # Save under last calendar day of that month
        rel_dt = pd.to_datetime(end_date) + MonthEnd(1)
        rel_dt_str = rel_dt.strftime("%Y-%m-%d")
        out_path = os.path.join(corr_path, f"{rel_dt_str}.csv")
        corr_df.to_csv(out_path)


def build_industry_matrix(market: str, codes: List[str], mode: str = "identity") -> np.ndarray:
    """Build an industry adjacency matrix for non-CN markets.

    mode:
      - identity: I (no industry edges)
      - full: fully connected (1 off-diagonal)
      - sector: try to group by yfinance Ticker.info["sector"]. Slower and optional.
    """
    n = len(codes)
    if mode == "identity":
        mat = np.eye(n, dtype=np.float32)
    elif mode == "full":
        mat = np.ones((n, n), dtype=np.float32)
        np.fill_diagonal(mat, 0.0)
    elif mode == "sector":
        if yf is None:
            raise RuntimeError("yfinance required for sector mode. Install yfinance.")
        sectors = {}
        for c in tqdm(codes, desc="Fetching sectors"):
            try:
                info = yf.Ticker(c).info  # network call per ticker
                sectors[c] = info.get("sector", None)
                print(f"Ticker: {c}, Sector: {sectors[c]}")
            except Exception:
                sectors[c] = None
        mat = np.zeros((n, n), dtype=np.float32)
        idx = {c: i for i, c in enumerate(codes)}
        for a in codes:
            for b in codes:
                if a == b:
                    mat[idx[a], idx[b]] = 1.0
                else:
                    mat[idx[a], idx[b]] = 1.0 if sectors.get(a) and sectors.get(a) == sectors.get(b) else 0.0
    else:
        raise ValueError(f"Unknown industry mode: {mode}")
    return mat


def save_daily_graph(dt: str,
                     df_all: pd.DataFrame,
                     relation_dt: str,
                     stock_trade_dt_s_all: List[str],
                     codes: List[str],
                     market: str,
                     horizon: int,
                     relation_type: str,
                     lookback: int = 20,
                     threshold: float = 0.2,
                     norm: bool = True,
                     industry_mat: Optional[np.ndarray] = None):
    """Construct and save one day's pickle in the format loaders expect, under dataset_default.
    """
    # time series window
    ts_start = stock_trade_dt_s_all[stock_trade_dt_s_all.index(dt) - (lookback - 1)]
    df_ts = df_all[(df_all["dt"] >= ts_start) & (df_all["dt"] <= dt)].copy()

    if industry_mat is None:
        # For CN markets, expect an industry matrix CSV under dataset_default
        # but for yfinance-built sets we pass industry_mat
        industry_mat = np.eye(len(codes), dtype=np.float32)
    ind = torch.from_numpy(industry_mat.astype(np.float32))

    # Read corr for relation_dt (auto-compute on the fly if missing)
    corr_csv = os.path.join(DATASET_CORR_ROOT, market, f"{relation_dt}.csv")
    if not os.path.exists(corr_csv):
        # Attempt to compute this month's correlation ad-hoc
        os.makedirs(os.path.dirname(corr_csv), exist_ok=True)
        # Determine the last trading day for this month (<= relation_dt)
        month_tag = relation_dt[:7]
        month_days = [d for d in stock_trade_dt_s_all if d.startswith(month_tag) and d <= relation_dt]
        if not month_days:
            # if no day in that month, pick the nearest previous trading day
            month_days = [d for d in stock_trade_dt_s_all if d <= relation_dt]
        if month_days:
            last_td = month_days[-1]
            end_idx = stock_trade_dt_s_all.index(last_td)
            # Use available data even if less than full lookback window
            start_idx = max(0, end_idx - (lookback - 1))
            start_date = stock_trade_dt_s_all[start_idx]
            actual_lookback = end_idx - start_idx + 1
            window = df_all[(df_all["dt"] >= start_date) & (df_all["dt"] <= last_td)]
            # Build per-code feature windows
            feat_dict = {}
            for code in codes:
                sub = window[window["kdcode"] == code]
                y = sub[["close", "open", "high", "low", "prev_close", "volume"]].values
                # Require at least 2 days for correlation computation
                if y.shape[0] == actual_lookback and y.shape[0] >= 2:
                    feat_dict[code] = y.reshape(-1)
            if len(feat_dict) >= 2:
                # Align order to codes subset with enough data
                valid_codes = [c for c in codes if c in feat_dict]
                X = np.stack([feat_dict[c] for c in valid_codes], axis=0)
                X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
                corr_mat = np.corrcoef(X)
                corr_df = pd.DataFrame(corr_mat, index=valid_codes, columns=valid_codes).fillna(0)
                for i in range(len(valid_codes)):
                    corr_df.iat[i, i] = 1.0
                corr_df.to_csv(corr_csv)
        if not os.path.exists(corr_csv):
            raise FileNotFoundError(f"Correlation CSV not found (and could not be auto-generated): {corr_csv}.")
    corr_df = pd.read_csv(corr_csv, index_col=0)
    # Ensure order aligns with current codes
    corr_df = corr_df.reindex(index=codes, columns=codes)
    corr_df = corr_df.fillna(0)

    pos_adj, neg_adj = gen_mats_by_threshold(corr_df, threshold)
    corr = torch.from_numpy(corr_df.values.astype(np.float32))
    pos = torch.from_numpy(pos_adj.astype(np.float32))
    neg = torch.from_numpy(neg_adj.astype(np.float32))

    ts_features = []
    features = []
    labels = []
    day_last_code = []

    cols = FEATURE_COLS_NORM if norm else FEATURE_COLS
    for code in codes:
        df_ts_code = df_ts[df_ts["kdcode"] == code]
        ts_array = df_ts_code[cols].values
        df_code_dt = df_ts_code[df_ts_code["dt"] == dt]
        array = df_code_dt[cols].values
        if ts_array.T.shape[1] == lookback and array.shape[0] == 1:
            ts_features.append(ts_array)
            features.append(array[0])  # Squeeze to [6] instead of [1, 6]
            label = df_ts_code.loc[df_ts_code["dt"] == dt]["label"].values
            labels.append(label[0])
            day_last_code.append([code, dt])

    ts_features = torch.from_numpy(np.array(ts_features)).float()
    features = torch.from_numpy(np.array(features)).float()  # Now [num_stocks, 6]
    labels = torch.tensor(labels, dtype=torch.float32)

    # Create pyg_data
    edge_index = torch.triu_indices(ind.size(0), ind.size(0), offset=1)
    pyg_data = Data(x=features, edge_index=edge_index)
    pyg_data.edge_attr = ind[edge_index[0], edge_index[1]]

    result = {
        "corr": Variable(corr),
        "ts_features": Variable(ts_features),
        "features": Variable(features),
        "industry_matrix": Variable(ind),
        "pos_matrix": Variable(pos),
        "neg_matrix": Variable(neg),
        "pyg_data": pyg_data,
        "labels": Variable(labels),
        "mask": [True] * len(labels),
    }

    # sanitize NaNs
    for k, v in list(result.items()):
        if isinstance(v, torch.Tensor):
            result[k] = torch.nan_to_num(v, nan=0.0)

    # Save
    save_dir = os.path.join(DATASET_DEFAULT_ROOT, f"data_train_predict_{market}", f"{horizon}_{relation_type}")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"{dt}.pkl"), "wb") as f:
        pickle.dump(result, f)

    # daily stock list (optional, mirrors existing structure)
    ds_dir = os.path.join(DATASET_DEFAULT_ROOT, f"daily_stock_{market}")
    os.makedirs(ds_dir, exist_ok=True)
    pd.DataFrame(columns=["kdcode", "dt"], data=day_last_code).to_csv(
        os.path.join(ds_dir, f"{dt}.csv"), index=False, encoding="utf_8_sig"
    )


def main():
    parser = argparse.ArgumentParser(description="Build SmartFolio-compatible dataset from yfinance")
    parser.add_argument("--market", default="custom", help="Market name tag to use in output paths (e.g., sp500, nd100, custom)")
    parser.add_argument("--tickers_file", default=None, help="CSV with a 'kdcode' or 'ticker' column listing symbols")
    parser.add_argument("--tickers", default=None, help="Comma-separated ticker list if no file is provided")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--relation_type", default="hy")
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--norm", dest="norm", action="store_true", help="Use normalized features (default)")
    parser.add_argument("--no-norm", dest="norm", action="store_false", help="Disable feature normalization")
    parser.set_defaults(norm=True)
    parser.add_argument("--industry_mode", default="identity", choices=["identity", "full", "sector"], help="How to build industry matrix for non-CN markets")
    args = parser.parse_args()

    # Resolve tickers
    if args.tickers_file:
        df_t = pd.read_csv(args.tickers_file)
        col = "kdcode" if "kdcode" in df_t.columns else ("ticker" if "ticker" in df_t.columns else None)
        if col is None:
            raise ValueError("tickers_file must have a 'kdcode' or 'ticker' column")
        tickers = sorted(df_t[col].dropna().astype(str).unique().tolist())
    elif args.tickers:
        tickers = sorted([t.strip() for t in args.tickers.split(",") if t.strip()])
    else:
        raise ValueError("Provide --tickers_file or --tickers")

    # 1) Download OHLCV
    print(f"Downloading OHLCV for {len(tickers)} tickers from {args.start} to {args.end}...")
    df_raw = fetch_ohlcv_yf(tickers, args.start, args.end)

    # Save the 'org' CSV for reference (mirrors existing naming convention)
    org_out = os.path.join(DATASET_DEFAULT_ROOT, f"{args.market}_org.csv")
    os.makedirs(DATASET_DEFAULT_ROOT, exist_ok=True)
    df_raw.to_csv(org_out, index=False)

    # --- Create an index CSV (equal-weighted average daily return) for model_predict ---
    # model_predict expects ./dataset/index_data/{market}_index_2024.csv with columns ['datetime','daily_return']
    try:
        idx_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "index_data"))
        os.makedirs(idx_dir, exist_ok=True)
        # Use prev_close from df_raw to compute per-ticker daily return
        df_idx = df_raw.copy()
        if "prev_close" in df_idx.columns:
            df_idx["daily_return"] = df_idx["close"] / df_idx["prev_close"] - 1
            df_idx_summary = df_idx.groupby("dt")["daily_return"].mean().reset_index()
            df_idx_summary = df_idx_summary.rename(columns={"dt": "datetime"})
            index_out = os.path.join(idx_dir, f"{args.market}_index.csv")
            df_idx_summary.to_csv(index_out, index=False)
        else:
            # If prev_close missing (shouldn't happen), skip index creation but warn
            print("Warning: prev_close missing in raw data; skipping index CSV creation.")
    except Exception as e:
        print(f"Warning: failed to create index CSV: {e}")

    # 2) Labels and preprocessing
    df_lbl = get_label(df_raw, horizon=args.horizon)
    df_roll = cal_rolling_mean_std(df_lbl, cal_cols=["close", "volume"], lookback=5)
    df_norm = group_and_norm(
        df_roll,
        base_cols=["close_mean", "close_std", "volume_mean", "volume_std"],
        n_clusters=4,
    )

    # dates and codes
    df_all = df_norm.copy()
    df_all = df_all[(df_all["dt"] >= args.start) & (df_all["dt"] <= args.end)].copy()
    stock_trade_dt_s_all = sorted(df_norm["dt"].unique().tolist())
    stock_trade_dt_s = sorted(df_all["dt"].unique().tolist())

    # Filter to stocks present across the (filtered) date range
    codes = filter_code(df_all)
    print(f"Universe size after filtering for complete histories: {len(codes)}")

    # 3) Monthly correlations (saved to dataset/corr/{market}/YYYY-MM-DD.csv)
    print("Computing monthly correlation matrices...")
    compute_monthly_corrs(df_norm, market=args.market, lookback_days=args.lookback, out_root=DATASET_CORR_ROOT)

    # 4) Industry adjacency
    print(f"Building industry matrix mode={args.industry_mode} ...")
    ind_mat = build_industry_matrix(args.market, codes, mode=args.industry_mode)
    ind_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset")), args.market)
    os.makedirs(ind_dir, exist_ok=True)
    np.save(os.path.join(ind_dir, "industry.npy"), ind_mat)

    # 5) Daily pickles
    print("Saving daily graph pickles...")
    for dt in tqdm(stock_trade_dt_s, desc="Daily pickles"):
        rel_dt = get_relation_dt(str_year=dt[:4], str_month=dt[5:7], stock_trade_dt_s=stock_trade_dt_s)
        save_daily_graph(
            dt=dt,
            df_all=df_norm,
            relation_dt=rel_dt,
            stock_trade_dt_s_all=stock_trade_dt_s_all,
            codes=codes,
            market=args.market,
            horizon=args.horizon,
            relation_type=args.relation_type,
            lookback=args.lookback,
            threshold=args.threshold,
            norm=args.norm,
            industry_mat=ind_mat,
        )

    print("Dataset build complete.")


if __name__ == "__main__":
    main()
