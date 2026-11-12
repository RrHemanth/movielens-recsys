import pandas as pd
import pytest

from recsys.data.periods import AS_OF_DATES, load_period
from recsys.data.split import to_unix_seconds


@pytest.fixture
def processed(tmp_path):
    def year_ts(year):
        return to_unix_seconds(f"{year}-06-01")
    rows = lambda years: pd.DataFrame({"user_id": 1, "movie_id": range(len(years)), "rating": 5.0,
                                        "ts": [year_ts(y) for y in years]})
    rows([2015, 2016, 2017]).to_parquet(tmp_path / "train.parquet")
    rows([2018]).to_parquet(tmp_path / "val.parquet")
    rows([2019]).to_parquet(tmp_path / "test.parquet")
    return tmp_path


@pytest.mark.parametrize("as_of", AS_OF_DATES)
def test_history_is_before_and_answers_are_the_following_year(processed, as_of):
    history, answers = load_period(processed, as_of)
    start = to_unix_seconds(as_of)
    end = to_unix_seconds(f"{int(as_of[:4]) + 1}-01-01")
    assert history["ts"].max() < start                                       # no peeking
    assert answers["ts"].between(start, end - 1).all() and len(answers) == 1   # exactly that year