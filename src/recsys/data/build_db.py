from pathlib import Path

import duckdb


def build_database(interim_dir: Path, db_path: Path) -> list[str]:
    """Create one view per parquet file in interim_dir. Returns the view names."""
    parquet_files = sorted(interim_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {interim_dir}. Run convert_to_parquet.py first.")

    con = duckdb.connect(str(db_path))
    view_names = []
    for pq in parquet_files:
        name = pq.stem  # "ratings.parquet" -> "ratings"
        con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{pq}')")
        view_names.append(name)
    con.close()
    return view_names