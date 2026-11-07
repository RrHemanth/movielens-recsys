from pathlib import Path

import duckdb

# For each table: (CSV file name, SQL that picks columns and sets their types).
# We read every CSV column as text (all_varchar) and cast it ourselves,
# so no type is ever guessed.
TABLES = {
    "ratings": ("ratings.csv", """
        SELECT
            CAST(userId AS INTEGER)    AS user_id,
            CAST(movieId AS INTEGER)   AS movie_id,
            CAST(rating AS FLOAT)      AS rating,
            CAST(timestamp AS BIGINT)  AS ts
        FROM read_csv('{csv}', header = true, all_varchar = true)
    """),
    "movies": ("movies.csv", """
        SELECT
            CAST(movieId AS INTEGER) AS movie_id,
            title,
            genres
        FROM read_csv('{csv}', header = true, all_varchar = true)
    """),
    "tags": ("tags.csv", """
        SELECT
            CAST(userId AS INTEGER)    AS user_id,
            CAST(movieId AS INTEGER)   AS movie_id,
            tag,
            CAST(timestamp AS BIGINT)  AS ts
        FROM read_csv('{csv}', header = true, all_varchar = true)
    """),
    "links": ("links.csv", """
        SELECT
            CAST(movieId AS INTEGER) AS movie_id,
            imdbId                   AS imdb_id,   -- kept as text to keep leading zeros
            CAST(tmdbId AS INTEGER)  AS tmdb_id
        FROM read_csv('{csv}', header = true, all_varchar = true)
    """),
    "genome_scores": ("genome-scores.csv", """
        SELECT
            CAST(movieId AS INTEGER)  AS movie_id,
            CAST(tagId AS INTEGER)    AS tag_id,
            CAST(relevance AS FLOAT)  AS relevance
        FROM read_csv('{csv}', header = true, all_varchar = true)
    """),
    "genome_tags": ("genome-tags.csv", """
        SELECT
            CAST(tagId AS INTEGER) AS tag_id,
            tag
        FROM read_csv('{csv}', header = true, all_varchar = true)
    """),
}


def convert_table(csv_path: Path, select_sql: str, out_path: Path) -> int:
    """Convert one CSV to parquet using the given SELECT. Returns the row count."""
    con = duckdb.connect()  # in-memory database, just used as a fast converter
    query = select_sql.format(csv=csv_path)
    con.execute(f"COPY ({query}) TO '{out_path}' (FORMAT parquet, COMPRESSION zstd)")
    n_rows = con.execute(f"SELECT COUNT(*) FROM '{out_path}'").fetchone()[0]
    con.close()
    return n_rows


def convert_all(raw_dir: Path, out_dir: Path) -> dict[str, int]:
    """Convert every table in TABLES. Returns {table_name: row_count}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    row_counts = {}
    for name, (csv_file, select_sql) in TABLES.items():
        print(f"Converting {csv_file}...")
        row_counts[name] = convert_table(raw_dir / csv_file, select_sql, out_dir / f"{name}.parquet")
    return row_counts