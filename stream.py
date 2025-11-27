import os
import json
import pickle
import threading
import time
import random
import argparse
import warnings
from typing import List, Optional, Dict, Mapping, Tuple, cast
import numpy as np
import pandas as pd
import torch
from threading import Lock
from pandas.tseries.offsets import MonthBegin, MonthEnd

warnings.filterwarnings("ignore", category=UserWarning)

# Import required functions from gen_data modules
try:
    from gen_data.build_dataset_yf import (
        fetch_ohlcv_yf,
        DATASET_DEFAULT_ROOT,
        DATASET_CORR_ROOT,
    )
    from gen_data.update_monthly_dataset import (
        run,
        _load_manifest,
        _dump_manifest,
        _find_latest_raw_snapshot,
        _load_snapshot,
        _determine_next_month,
        MANIFEST_NAME,
    )
except ImportError:
    print("Warning: Could not import gen_data modules. Ensure paths are correct.")

# Import main.py functions for direct fine-tuning
try:
    from main import fine_tune_month
except ImportError:
    print("Warning: Could not import fine_tune_month from main. Ensure main.py is in the same directory.")

"""
STREAM.PY - Two-Threaded Streaming Pipeline for Continuous Model Fine-Tuning

ARCHITECTURE:
=============
Two concurrent threads implement an incremental learning pipeline:

1. FETCH THREAD:
   - Continuously fetches raw OHLCV data from yfinance (30 days worth)
   - Uses raw yfinance.download() directly (fast, no per-day processing)
   - Accumulates raw dataframes into a buffer
   - After 30 days collected, processes all data ONCE (batch normalization)
   - Signals training thread when processed data is ready

2. TRAINING THREAD:
   - Waits for "data_ready_event" signal from fetch thread
   - Calls run() with fetched data to create monthly shards (gen_data/update_monthly_dataset.py):
     * Merges with raw snapshot
     * Computes labels (1-day return)
     * Calculates rolling statistics
     * Applies KMeans normalization
     * Creates PyTorch Geometric graph samples (nodes + edges)
     * Saves monthly shard
     * Updates manifest with new month
   - Clears memory (gc.collect(), torch.cuda.empty_cache())
   - Calls fine_tune_month() from main.py to fine-tune PPO model:
     * Finds latest unprocessed monthly shard from manifest
     * Creates AllGraphDataSampler for that month's data
     * Loads baseline checkpoint from args.baseline_checkpoint
     * Fine-tunes PPO for fine_tune_steps timesteps (reduced, e.g., 100)
     * Saves fine-tuned checkpoint with month slug
     * Updates manifest with processed flag

DATA FLOW:
==========
yfinance (raw, 1 day) → buffer (30 raw dfs) → process_ohlcv_data() 
  → run(fetched=df, fetched_end_date=...) 
  → monthly shard + manifest update
  → fine_tune_month() 
  → fine-tuned checkpoint

DEPENDENCIES:
=============
- baseline_checkpoint: Must exist at args.baseline_checkpoint before first fine_tune_month() call
- manifest.json: Created/updated by run(), read by fine_tune_month()
- monthly_shards: Created by run(), loaded by fine_tune_month()

MANIFEST STRUCTURE:
===================
{
  "last_trading_day": "YYYY-MM-DD",           # Updated by run()
  "monthly_shards": {
    "2024-11": "path/to/shard_2024-11.pkl",  # Created by run()
    ...
  },
  "last_fine_tuned_month": "2024-11",         # Updated by fine_tune_month()
  "last_checkpoint_path": "path/to/...zip",  # Updated by fine_tune_month()
  ...
}
"""

# Global variables
counter = 0
data_buffer = []  # Store the data for 30 days
lock = Lock()  # A lock to synchronize both threads
data_ready_event = threading.Event()  # Event to signal when data is ready for training
current_date = None  # Track current date being processed
tickers_list = []  # Store tickers
market = ""  # Store market identifier
args_global = None  # Store global args


def _load_manifest_local(path: str) -> Dict[str, object]:
    """Load manifest from file."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                raise RuntimeError(f"Manifest at {path} is corrupt. Please inspect or delete it.")
    return {
        "last_trading_day": None,
        "monthly_shards": {},
        "daily_index": {},
        "corr_matrices": [],
        "tickers": [],
        "raw_snapshot": None,
    }


def get_next_trading_day(start_date: str, df_raw: pd.DataFrame) -> str:
    """Get the next trading day from the dataset."""
    available_dates = sorted(df_raw["dt"].unique().tolist())
    current = pd.to_datetime(start_date)
    
    for date_str in available_dates:
        if pd.to_datetime(date_str) > current:
            return date_str
    
    # If no future date found, return the next day
    return (pd.to_datetime(start_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def fetch_one_day_data_raw(tickers: List[str], date: str) -> pd.DataFrame:
    """Fetch raw OHLCV data for one specific day using yfinance directly (fast)."""
    try:
        # Use yfinance directly without processing for speed
        date_next = (pd.to_datetime(date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        import yfinance as yf
        df = yf.download(tickers, start=date, end=date_next, auto_adjust=False, group_by="ticker", progress=False)
        if not df.empty:
            print(f"Fetched raw data for {len(tickers)} tickers on {date}")
            return df
        else:
            print(f"No data fetched for {date}")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching raw data for {date}: {e}")
        return pd.DataFrame()


def process_ohlcv_data(raw_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Process raw yfinance data to match fetch_ohlcv_yf output format.
    
    Mimics the data processing logic from fetch_ohlcv_yf.
    """
    if not raw_dfs:
        return pd.DataFrame()
    
    # Concatenate all raw dataframes
    df = pd.concat(raw_dfs, ignore_index=False)
    
    # Normalize to a tall dataframe
    if isinstance(df.columns, pd.MultiIndex):
        parts = []
        tickers = set()
        for col in df.columns:
            if isinstance(col, tuple):
                tickers.add(col[0])
        
        for t in sorted(tickers):
            if (t, "Close") not in df.columns:
                continue
            sub = df[(t,)].copy()
            sub.columns = [c.lower() for c in sub.columns]
            sub["kdcode"] = t
            parts.append(sub.reset_index().rename(columns={"Date": "dt"}))
        
        if parts:
            tall = pd.concat(parts, ignore_index=True)
        else:
            return pd.DataFrame()
    else:
        # Single ticker case
        d = df.copy()
        d.columns = [c.lower() for c in d.columns]
        # Infer ticker from index or use first ticker
        d["kdcode"] = d.get("kdcode", "UNKNOWN")
        tall = d.reset_index().rename(columns={"Date": "dt"})
    
    # Ensure required columns exist
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(tall.columns)
    if missing:
        print(f"Warning: Missing columns {missing}, attempting to continue...")
    
    # Compute prev_close per kdcode
    tall = tall.sort_values(["kdcode", "dt"]).reset_index(drop=True)
    tall["prev_close"] = tall.groupby("kdcode")["close"].shift(1)
    
    # Drop first row per ticker where prev_close is NaN
    tall = tall.dropna(subset=["prev_close"]).copy()
    
    # Cast dt to string YYYY-MM-DD
    tall["dt"] = pd.to_datetime(tall["dt"]).dt.strftime("%Y-%m-%d")
    
    # Keep only needed columns
    cols_to_keep = ["kdcode", "dt", "close", "open", "high", "low", "prev_close", "volume"]
    tall = tall[[c for c in cols_to_keep if c in tall.columns]]
    
    return tall


def _detect_input_dim_for_finetune(data_dir: str, fallback: int = 6) -> int:
    """Best-effort input_dim detection for fine-tuning."""
    try:
        sample_files = [f for f in os.listdir(data_dir) if f.endswith(".pkl")]
        if not sample_files:
            return fallback
        sample_path = os.path.join(data_dir, sample_files[0])
        with open(sample_path, "rb") as f:
            sample = pickle.load(f)
        feats = sample.get("features")
        if feats is None:
            return fallback
        try:
            shape = feats.shape
        except Exception:
            try:
                shape = feats.size()
            except Exception:
                shape = None
        if shape and len(shape) >= 2:
            return shape[-1]
    except Exception as e:
        print(f"Warning: input_dim auto-detection failed ({e}); using fallback={fallback}")
    return fallback


def fetch_data_thread(
    tickers: List[str],
    manifest_path: str,
    dataset_root: str,
    market_name: str,
):
    """Thread function to fetch 30 days of data incrementally (raw, then process)."""
    global counter, data_buffer, current_date
    
    # Load initial manifest to get the starting date
    manifest = _load_manifest_local(manifest_path)
    
    # Load raw snapshot to get available tickers and dates
    try:
        raw_snapshot = cast(Optional[str], manifest.get("raw_snapshot"))
        raw_path = _find_latest_raw_snapshot(dataset_root, market_name)
        df_raw = _load_snapshot(raw_path)
    except Exception as e:
        print(f"Error loading raw snapshot: {e}")
        return
    
    # Determine the next month to process
    next_month_start, next_month_end = _determine_next_month(manifest, df_raw)
    
    # Get the starting date from manifest or use next month start
    last_trading_day = manifest.get("last_trading_day")
    if last_trading_day:
        last_trading_day_str = cast(str, last_trading_day)
        current_date = pd.to_datetime(last_trading_day_str) + pd.Timedelta(days=1)
    else:
        current_date = next_month_start
    
    print(f"Starting data fetch from {current_date.strftime('%Y-%m-%d')}")
    
    # Fetch 30 days of raw data (FAST - no processing)
    counter = 0
    fetched_raw_data_list: List[pd.DataFrame] = []
    
    while counter < 6:
        date_str = current_date.strftime("%Y-%m-%d")
        
        # Fetch RAW data for this day (fast, minimal processing)
        day_raw_data = fetch_one_day_data_raw(tickers, date_str)
        
        if day_raw_data is not None and not day_raw_data.empty:
            fetched_raw_data_list.append(day_raw_data)
            counter += 1
            print(f"Counter: {counter}/6 - Date: {date_str}")
        else:
            print(f"Skipping {date_str} (no data available)")
        
        # Move to next day
        current_date += pd.Timedelta(days=1)
        
        # Apply random delay between fetches
        time.sleep(random.uniform(8, 12))  # Random delay between 8 and 12 seconds
    
    # Process all 30 days of fetched data (ONCE, after collection)
    print("Processing collected raw data...")
    data_buffer = process_ohlcv_data(fetched_raw_data_list)
    
    if not data_buffer.empty:
        print(f"Processed data. Total rows: {len(data_buffer)}")
    else:
        print("Warning: Processed data is empty")
    
    # Acquire lock and signal that data is ready
    with lock:
        data_ready_event.set()
        print("Data fetch and processing complete. Signaling training thread...")


def training_thread(args: argparse.Namespace):
    """Thread function to perform training once data is ready.
    
    Implements the fine_tune_month pipeline from main.py:
    1. Call run() to create monthly shards from fetched data
    2. Call fine_tune_month() to fine-tune PPO model on the latest shard
    """
    global data_buffer, counter
    
    # Wait for data to be ready
    print("Training thread waiting for data...")
    data_ready_event.wait()
    
    with lock:
        print("Training thread acquired lock. Starting run() with fetched data...")
        
        if isinstance(data_buffer, pd.DataFrame) and not data_buffer.empty:
            try:
                # Step 1: Get the end date from the fetched data
                fetched_end_date = data_buffer["dt"].max()
                print(f"Fetched data date range: {data_buffer['dt'].min()} to {fetched_end_date}")
                
                # Step 2: Call run() with fetched data to create monthly shards
                run(args, fetched=data_buffer, fetched_end_date=fetched_end_date)
                print("run() completed successfully! Monthly shards created.")
                
                # Step 3: Clear data buffer to free memory before fine-tuning
                data_buffer = pd.DataFrame()
                import gc
                gc.collect()  # Force garbage collection
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # Clear GPU cache
                print("Memory cleared before fine-tuning...")
                
                # Step 4: Call fine_tune_month directly from main.py
                print("Starting fine-tuning with HGAT policy...")
                
                try:
                    # Build manifest path
                    manifest_path = os.path.join(
                        args.dataset_root,
                        f"data_train_predict_{args.market}",
                        f"{args.horizon}_{args.relation_type}",
                        MANIFEST_NAME
                    )
                    data_dir = os.path.dirname(manifest_path)
                    
                    # Verify manifest exists
                    if not os.path.exists(manifest_path):
                        raise FileNotFoundError(f"Manifest not found at {manifest_path}")
                    
                    print(f"Using manifest at: {manifest_path}")
                    ft_device = getattr(args, "device", None) or ("cuda:0" if torch.cuda.is_available() else "cpu")
                    ft_save_dir = getattr(args, "save_dir", "./checkpoints")
                    ft_baseline = getattr(
                        args,
                        "baseline_checkpoint",
                        os.path.join(ft_save_dir, "baseline.zip"),
                    )
                    ft_input_dim = _detect_input_dim_for_finetune(
                        data_dir=data_dir,
                        fallback=getattr(args, "input_dim", 6),
                    )
                    fine_tune_args = argparse.Namespace(
                        market=args.market,
                        horizon=args.horizon,
                        relation_type=args.relation_type,
                        device=ft_device,
                        model_name=getattr(args, "model_name", "SmartFolio"),
                        save_dir=ft_save_dir,
                        baseline_checkpoint=ft_baseline,
                        resume_model_path=getattr(args, "resume_model_path", None),
                        fine_tune_steps=getattr(args, "fine_tune_steps", 5000),
                        seed=getattr(args, "seed", 123),
                        ind_yn=True,
                        pos_yn=True,
                        neg_yn=True,
                        input_dim=ft_input_dim,
                    )
                    
                    checkpoint = fine_tune_month(fine_tune_args, manifest_path=manifest_path)
                    
                    print(f"Fine-tuning completed successfully! Checkpoint: {checkpoint}")
                    
                except FileNotFoundError as e:
                    print(f"File error during fine-tuning: {e}")
                    import traceback
                    traceback.print_exc()
                except RuntimeError as e:
                    print(f"Runtime error during fine-tuning (likely no unprocessed shards): {e}")
                    import traceback
                    traceback.print_exc()
                except Exception as e:
                    print(f"Unexpected error during fine-tuning: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Step 5: Reset counter for next batch
                counter = 0
                data_ready_event.clear()
                
            except Exception as e:
                print(f"Error during training: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("No data available for training.")


def main(
    market_name: str,
    tickers_file: Optional[str] = None,
    dataset_root: Optional[str] = None,
    corr_root: Optional[str] = None,
    horizon: int = 1,
    relation_type: str = "corr",
    lookback: int = 20,
    threshold: float = 0.2,
    n_clusters: int = 8,
    disable_norm: bool = False,
):
    """Main function to start both threads."""
    global tickers_list, market, args_global
    
    market = market_name
    dataset_root = dataset_root or DATASET_DEFAULT_ROOT
    corr_root = corr_root or DATASET_CORR_ROOT
    
    # Determine data directory and manifest path
    data_dir = os.path.join(
        dataset_root,
        f"data_train_predict_{market_name}",
        f"{horizon}_{relation_type}",
    )
    os.makedirs(data_dir, exist_ok=True)
    manifest_path = os.path.join(data_dir, MANIFEST_NAME)
    
    # Load manifest and get tickers
    manifest = _load_manifest_local(manifest_path)
    try:
        raw_snapshot = cast(Optional[str], manifest.get("raw_snapshot"))
        raw_path = _find_latest_raw_snapshot(dataset_root, market_name)
        df_raw = _load_snapshot(raw_path)
    except Exception as e:
        print(f"Error loading raw snapshot: {e}")
        return
    
    if tickers_file and os.path.exists(tickers_file):
        tickers_list = [line.strip() for line in open(tickers_file, "r", encoding="utf-8") if line.strip()]
    else:
        tickers_list = sorted(df_raw["kdcode"].unique().tolist())
    
    print(f"Using {len(tickers_list)} tickers: {tickers_list[:5]}..." if len(tickers_list) > 5 else f"Using {len(tickers_list)} tickers: {tickers_list}")
    
    # Create args for training thread
    # Include only attributes required for data generation and fine-tuning
    args_global = argparse.Namespace(
        # Dataset building parameters (used by run())
        market=market_name,
        dataset_root=dataset_root,
        corr_root=corr_root,
        horizon=horizon,
        relation_type=relation_type,
        lookback=lookback,
        threshold=threshold,
        n_clusters=n_clusters,
        disable_norm=disable_norm,
        raw_path=None,
        tickers_file=tickers_file,
        # Fine-tuning parameters (minimal set actually used by fine_tune_month)
        seed=123,
        input_dim=6,  # Will be auto-detected from pickle files if needed
        save_dir="./checkpoints",
        model_name="SmartFolio",
        fine_tune_steps=5000,  # PPO learning timesteps for monthly updates
    )
    
    # Create threads
    fetch_thread = threading.Thread(
        target=fetch_data_thread,
        args=(tickers_list, manifest_path, dataset_root, market_name),
        daemon=False,
    )
    train_thread = threading.Thread(
        target=training_thread,
        args=(args_global,),
        daemon=False,
    )
    
    # Start the threads
    print("Starting fetch and training threads...")
    fetch_thread.start()
    train_thread.start()
    
    # Wait for both threads to complete
    fetch_thread.join()
    train_thread.join()
    
    print("Both threads completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream data fetching and model training with threading.")
    parser.add_argument("--market", required=True, help="Market identifier (e.g., 'us', 'cn').")
    parser.add_argument("--tickers-file", default=None, help="File with newline-delimited tickers.")
    parser.add_argument("--dataset-root", default=None, help="Root directory for datasets.")
    parser.add_argument("--corr-root", default=None, help="Root directory for correlation matrices.")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon for labels.")
    parser.add_argument("--relation-type", default="corr", help="Relation type for data organization.")
    parser.add_argument("--lookback", type=int, default=20, help="Lookback window for features.")
    parser.add_argument("--threshold", type=float, default=0.2, help="Threshold for adjacency matrices.")
    parser.add_argument("--n-clusters", type=int, default=8, help="Number of clusters for normalization.")
    parser.add_argument("--disable-norm", action="store_true", help="Disable feature normalization.")
    
    args = parser.parse_args()
    
    main(
        market_name=args.market,
        tickers_file=args.tickers_file,
        dataset_root=args.dataset_root,
        corr_root=args.corr_root,
        horizon=args.horizon,
        relation_type=args.relation_type,
        lookback=args.lookback,
        threshold=args.threshold,
        n_clusters=args.n_clusters,
        disable_norm=args.disable_norm,
    )
