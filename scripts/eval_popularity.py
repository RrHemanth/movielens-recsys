import time

import pandas as pd

from recsys.config import load_config
from recsys.evaluation.evaluate import evaluate
from recsys.models.popularity import PopularityRecommender


def load_genres(movies: pd.DataFrame) -> dict:
    """{movie_id: set of genres}. '(no genres listed)' becomes an empty set."""
    genres = {}
    for movie_id, genre_text in zip(movies["movie_id"], movies["genres"]):
        if genre_text == "(no genres listed)":
            genres[movie_id] = set()
        else:
            genres[movie_id] = set(genre_text.split("|"))
    return genres


def main():
    cfg = load_config()
    train = pd.read_parquet(cfg["paths"]["processed"] / "train.parquet")
    val = pd.read_parquet(cfg["paths"]["processed"] / "val.parquet")
    movies = pd.read_parquet(cfg["paths"]["interim"] / "movies.parquet")

    genres = load_genres(movies)
    titles = dict(zip(movies["movie_id"], movies["title"]))
    known_users = set(train["user_id"])
    n_catalog = train["movie_id"].nunique()
    val_users = val["user_id"].unique().tolist()
    threshold = cfg["split"]["relevant_threshold"]
    k = 10

    print(f"Train ratings: {len(train):,} | catalog: {n_catalog:,} movies | 2018 users: {len(val_users):,}")
    pd.set_option("display.float_format", "{:.4f}".format)

    for recent_days in [None, 365, 90]:
        name = "all-time" if recent_days is None else f"last {recent_days} days"
        start = time.time()

        model = PopularityRecommender(threshold=threshold, recent_days=recent_days).fit(train)
        recs = model.recommend(val_users, k=k)
        _, summary = evaluate(recs, val, known_users, genres, n_catalog, k=k, threshold=threshold)

        print(f"\n=== Popularity ({name}) | {time.time() - start:.1f}s ===")
        print("Top 5 movies:", [titles[m] for m in model.ranking[:5]])
        print(summary)


if __name__ == "__main__":
    main()