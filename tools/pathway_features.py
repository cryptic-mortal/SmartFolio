import datetime
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import pathway as pw


def _build_schema(columns: List[str]) -> type:
    cols = {
        "kdcode": pw.column_definition(dtype=str),
        "dt": pw.column_definition(dtype=pw.DateTimeNaive),
    }
    cols.update({col: pw.column_definition(dtype=float) for col in columns})
    return pw.schema_builder(columns=cols, name="_RollingFeatureSchema")


def compute_rolling_mean_std_pathway(
    df: pd.DataFrame, cal_cols: List[str], lookback_days: int, *, max_rows: int = 50000
) -> pd.DataFrame:
    """
    Compute rolling mean/std per ticker using Pathway sliding windows.

    Args:
        df: DataFrame with at least ['kdcode', 'dt'] + cal_cols
        cal_cols: feature columns to aggregate
        lookback_days: trailing window length (inclusive) in days

    Returns:
        DataFrame with same rows as input and added {col}_mean / {col}_std columns.
    """
    if df.empty:
        return df

    if len(df) > max_rows:
        raise RuntimeError(
            f"Pathway rolling disabled: len(df)={len(df)} exceeds max_rows={max_rows}"
        )

    df_work = df.copy()
    original_dt_dtype = df_work["dt"].dtype
    df_work["dt"] = pd.to_datetime(df_work["dt"])

    # Keep only required columns in a deterministic order and normalize dtypes
    ordered_cols = ["kdcode", "dt"] + cal_cols
    df_ordered = df_work[ordered_cols].copy()
    for col in cal_cols:
        df_ordered[col] = pd.to_numeric(df_ordered[col], errors="coerce").astype(float).fillna(0.0)

    schema = _build_schema(cal_cols)
    rows = [
        tuple([row.kdcode, row.dt.to_pydatetime()] + [getattr(row, col) for col in cal_cols])
        for row in df_ordered.itertuples(index=False)
    ]

    table = pw.debug.table_from_rows(schema, rows)

    window = pw.temporal.intervals_over(
        at=table.dt,
        lower_bound=-datetime.timedelta(days=lookback_days - 1),
        upper_bound=datetime.timedelta(0),
    )
    grouped = table.windowby(table.dt, window=window, instance=table.kdcode)

    # Build reducers for each column
    reduce_kwargs = {
        "kdcode": pw.this._pw_instance,
        "dt": pw.this._pw_window_end,
    }
    for col in cal_cols:
        col_ref = getattr(pw.this, col)
        reduce_kwargs[f"{col}_values"] = pw.reducers.tuple(col_ref)

    aggregated = grouped.reduce(**reduce_kwargs)
    pdf = pw.debug.table_to_pandas(aggregated)

    # Compute mean/std from collected values (ddof=0); guard against empty tuples
    for col in cal_cols:
        values_col = f"{col}_values"
        pdf[f"{col}_mean"] = pdf[values_col].apply(
            lambda vals: float(np.array(vals).mean()) if len(vals) else np.nan
        )
        pdf[f"{col}_std"] = pdf[values_col].apply(
            lambda vals: float(np.array(vals).std(ddof=0)) if len(vals) else np.nan
        )
        pdf.drop(columns=[values_col], inplace=True)

    # Merge back to original rows
    merged = (
        df_work.drop(columns=[c for c in df_work.columns if c.endswith("_mean") or c.endswith("_std")])
        .merge(pdf, on=["kdcode", "dt"], how="left")
    )

    # Restore dt format if caller passed strings
    if pd.api.types.is_object_dtype(original_dt_dtype):
        merged["dt"] = merged["dt"].dt.strftime("%Y-%m-%d")
    return merged
