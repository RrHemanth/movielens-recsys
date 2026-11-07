import duckdb

from recsys.data.to_parquet import TABLES, convert_table


def column_types(parquet_path):
    """Return {column_name: type} for a parquet file."""
    rows = duckdb.sql(f"DESCRIBE SELECT * FROM '{parquet_path}'").fetchall()
    return {row[0]: row[1] for row in rows}


def test_ratings_columns_are_renamed_and_typed(tmp_path):
    csv_path = tmp_path / "ratings.csv"
    csv_path.write_text(
        "userId,movieId,rating,timestamp\n"
        "1,296,5.0,1147880044\n"
        "2,306,3.5,1147868817\n"
    )
    out_path = tmp_path / "ratings.parquet"

    n_rows = convert_table(csv_path, TABLES["ratings"][1], out_path)

    assert n_rows == 2
    assert column_types(out_path) == {
        "user_id": "INTEGER",
        "movie_id": "INTEGER",
        "rating": "FLOAT",
        "ts": "BIGINT",
    }
    rows = duckdb.sql(f"SELECT * FROM '{out_path}'").fetchall()
    assert rows == [(1, 296, 5.0, 1147880044), (2, 306, 3.5, 1147868817)]


def test_imdb_id_keeps_leading_zeros(tmp_path):
    csv_path = tmp_path / "links.csv"
    csv_path.write_text("movieId,imdbId,tmdbId\n1,0114709,862\n")
    out_path = tmp_path / "links.parquet"

    convert_table(csv_path, TABLES["links"][1], out_path)

    rows = duckdb.sql(f"SELECT imdb_id FROM '{out_path}'").fetchall()
    assert rows == [("0114709",)]