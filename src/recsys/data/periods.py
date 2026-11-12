import pandas as pd

from recsys.data.split import to_unix_seconds

AS_OF_DATES = ["2017-01-01", "2018-01-01", "2019-01-01"]


def period_files(processed, as_of: str) -> list:
    """Parquet files that together contain every rating before `as_of`."""
    if as_of == "2019-01-01":
        return [processed / "train.parquet", processed / "val.parquet"]
    return [processed / "train.parquet"]


def load_period(processed, as_of: str):
    """(history, answers): all ratings before `as_of`, and the ratings in the year that starts at `as_of`."""
    as_of_ts = to_unix_seconds(as_of)
    if as_of == "2017-01-01":
        train = pd.read_parquet(processed / "train.parquet")
        return train[train["ts"] < as_of_ts], train[train["ts"] >= as_of_ts]   # 2017 sits inside train
    if as_of == "2018-01-01":
        return pd.read_parquet(processed / "train.parquet"), pd.read_parquet(processed / "val.parquet")
    if as_of == "2019-01-01":
        history = pd.concat([pd.read_parquet(f) for f in period_files(processed, as_of)], ignore_index=True)
        return history, pd.read_parquet(processed / "test.parquet")
    raise ValueError(f"Unsupported as-of date: {as_of}")