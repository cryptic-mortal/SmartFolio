import datetime
import json
import os
from typing import Dict, Iterable, List, Tuple

import pathway as pw


class _DailyFileSchema(pw.Schema):
    ts: pw.DateTimeNaive
    path: str


def _iter_daily_files(base_dir: str) -> Iterable[Tuple[datetime.datetime, str]]:
    """Yield parsed daily pickle files with a date prefix (YYYY-MM-DD)."""
    for name in os.listdir(base_dir):
        if not name.endswith(".pkl"):
            continue
        date_part = name[:10]
        try:
            ts = datetime.datetime.strptime(date_part, "%Y-%m-%d")
        except ValueError:
            continue
        yield (ts, os.path.join(base_dir, name))


def build_monthly_shards_with_pathway(
    base_dir: str,
    manifest_path: str,
    *,
    min_days: int = 10,
    cutoff_days: int | None = None,
) -> List[Dict[str, object]]:
    """
    Build monthly shards via Pathway tumbling windows and update manifest.

    Args:
        base_dir: directory containing daily *.pkl files (YYYY-MM-DD prefix).
        manifest_path: path to monthly_manifest.json to update.
        min_days: minimum files per month to keep shard.
        cutoff_days: optional temporal cutoff for late data; if provided, older
                     rows outside the cutoff are ignored (Pathway common_behavior).

    Returns:
        List of shard dicts written into manifest["monthly_shards"].
    """
    rows = list(_iter_daily_files(base_dir))
    if not rows:
        raise ValueError(f"No daily pickle files found under {base_dir}")

    table = pw.debug.table_from_rows(_DailyFileSchema, rows)

    behavior = None
    if cutoff_days is not None:
        behavior = pw.temporal.common_behavior(cutoff=datetime.timedelta(days=cutoff_days))

    monthly = (
        table.windowby(
            table.ts,
            window=pw.temporal.session(
                predicate=lambda a, b: a.year == b.year and a.month == b.month
            ),
            behavior=behavior,
        )
        .reduce(
            month_start=pw.reducers.min(table.ts),
            month_end=pw.reducers.max(table.ts),
            days=pw.reducers.count(),
            paths=pw.reducers.sorted_tuple(table.path),
        )
    )

    pdf = pw.debug.table_to_pandas(monthly)
    pdf["month"] = pdf["month_start"].dt.strftime("%Y-%m")

    shards: List[Dict[str, object]] = []
    for _, row in pdf.iterrows():
        if row["days"] < min_days:
            continue
        shards.append(
            {
                "month": row["month"],
                "month_start": row["month_start"].strftime("%Y-%m-%d"),
                "month_end": row["month_end"].strftime("%Y-%m-%d"),
                "shard_path": base_dir,
                "daily_paths": list(row["paths"]),
                "processed": False,
            }
        )

    shards.sort(key=lambda s: s["month"])

    manifest = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
        except Exception:
            manifest = {}

    manifest["monthly_shards"] = shards
    manifest["base_dir"] = base_dir
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    return shards

