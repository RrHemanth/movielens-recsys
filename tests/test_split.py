import duckdb

from recsys.data.split import time_split, to_unix_seconds


def test_to_unix_seconds_matches_known_values():
    # These are the same numbers we used in the SQL exploration queries.
    assert to_unix_seconds("2018-01-01") == 1514764800
    assert to_unix_seconds("2019-01-01") == 1546300800


def test_ratings_at_the_cutoffs_go_to_the_right_split(tmp_path):
    val_ts = to_unix_seconds("2018-01-01")
    test_ts = to_unix_seconds("2019-01-01")

    # Make a tiny ratings file with 4 ratings right around the cutoffs.
    ratings_path = tmp_path / "ratings.parquet"
    con = duckdb.connect()
    con.execute(f"""
        COPY (
            SELECT * FROM (VALUES
                (1, 10, 4.0, {val_ts - 1}),   -- 1 second before 2018 -> train
                (1, 11, 3.0, {val_ts}),       -- exactly 2018-01-01   -> val
                (2, 12, 5.0, {test_ts - 1}),  -- 1 second before 2019 -> val
                (2, 13, 2.0, {test_ts})       -- exactly 2019-01-01   -> test
            ) AS t(user_id, movie_id, rating, ts)
        ) TO '{ratings_path}' (FORMAT parquet)
    """)

    out_dir = tmp_path / "out"
    counts = time_split(ratings_path, out_dir, "2018-01-01", "2019-01-01")

    # Every rating lands in exactly one split.
    assert counts == {"train": 1, "val": 2, "test": 1}

    # And it's the right rating in each split.
    def movies_in(split):
        rows = con.execute(f"SELECT movie_id FROM '{out_dir / split}.parquet' ORDER BY movie_id").fetchall()
        return [r[0] for r in rows]

    assert movies_in("train") == [10]
    assert movies_in("val") == [11, 12]
    assert movies_in("test") == [13]
    con.close()