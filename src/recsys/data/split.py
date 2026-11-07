from datetime import datetime, timezone
from pathlib import Path

import duckdb


def to_unix_seconds(date_str: str) -> int:
    """Turn '2018-01-01' into seconds since 1970-01-01, using UTC."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def time_split(ratings_path: Path, out_dir: Path, val_start: str, test_start: str) -> dict[str, int]:
    """Split ratings by time into train/val/test parquet files. Returns row counts."""
    val_ts = to_unix_seconds(val_start)
    test_ts = to_unix_seconds(test_start)

    # Each part is defined by a condition on the timestamp.
    parts = {
        "train": f"ts < {val_ts}",
        "val": f"ts >= {val_ts} AND ts < {test_ts}",
        "test": f"ts >= {test_ts}",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    counts = {}
    for name, condition in parts.items():
        out_path = out_dir / f"{name}.parquet"
        con.execute(f"""
            COPY (SELECT * FROM read_parquet('{ratings_path}') WHERE {condition})
            TO '{out_path}' (FORMAT parquet, COMPRESSION zstd)
        """)
        counts[name] = con.execute(f"SELECT COUNT(*) FROM '{out_path}'").fetchone()[0]
    con.close()
    return counts