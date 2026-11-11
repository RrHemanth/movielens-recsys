import argparse

import pandas as pd

from recsys.config import load_config
from recsys.data.movies import load_genres
from recsys.data.split import to_unix_seconds
from recsys.features.reranker import FEATURES, add_features, add_labels


def main():
    parser = argparse.ArgumentParser(description="Add features and labels to a candidate shortlist.")
    parser.add_argument("--as-of", required=True, choices=["2017-01-01", "2018-01-01"])
    args = parser.parse_args()

    cfg = load_config()
    processed = cfg["paths"]["processed"]
    threshold = cfg["split"]["relevant_threshold"]
    as_of_ts = to_unix_seconds(args.as_of)

    train = pd.read_parquet(processed / "train.parquet")
    history = train[train["ts"] < as_of_ts]
    answers = train[train["ts"] >= as_of_ts] if args.as_of == "2017-01-01" else pd.read_parquet(processed / "val.parquet")
    movies = pd.read_parquet(cfg["paths"]["interim"] / "movies.parquet")

    cands = pd.read_parquet(processed / f"candidates_{args.as_of[:4]}.parquet")
    table = add_features(cands, history, movies, load_genres(movies), as_of_ts, threshold)
    table = add_labels(table, answers, threshold)

    out = processed / f"features_{args.as_of[:4]}.parquet"
    table.to_parquet(out)
    print(f"{len(table):,} rows -> {out.name}")
    print(f"Liked (label = 1): {table['label'].mean():.1%} of rows\n")

    pd.set_option("display.float_format", "{:.3f}".format)
    print("Average feature value, liked vs not liked:")
    print(table.groupby("label")[FEATURES].mean().T.rename(columns={0: "not liked", 1: "liked"}))


if __name__ == "__main__":
    main()