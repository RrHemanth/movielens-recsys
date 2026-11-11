import argparse
import time

import pandas as pd

from recsys.candidates import build_candidates, candidate_recall
from recsys.config import load_config
from recsys.data.split import to_unix_seconds
from recsys.spark import get_spark

ALS_PARAMS = {"rank": 64, "reg_param": 1.0, "alpha": 10.0, "max_iter": 10}   # chosen in Step 8


def main():
    parser = argparse.ArgumentParser(description="Build shortlists as of a pretend date and measure candidate recall.")
    parser.add_argument("--as-of", required=True, choices=["2017-01-01", "2018-01-01"])
    args = parser.parse_args()

    cfg = load_config()
    processed = cfg["paths"]["processed"]
    threshold = cfg["split"]["relevant_threshold"]
    as_of_ts = to_unix_seconds(args.as_of)

    # History users (returning) and the answer year
    if args.as_of == "2017-01-01":
        train = pd.read_parquet(processed / "train.parquet")
        history_users = set(train.loc[train["ts"] < as_of_ts, "user_id"])
        answers = train[train["ts"] >= as_of_ts]              # 2017 sits inside train.parquet
    else:
        history_users = set(pd.read_parquet(processed / "train.parquet", columns=["user_id"])["user_id"])
        answers = pd.read_parquet(processed / "val.parquet")  # 2018

    liked_users = set(answers.loc[answers["rating"] >= threshold, "user_id"])
    users = sorted(history_users & liked_users)
    print(f"As of {args.as_of}: {len(users):,} returning users who liked something in the answer year")

    spark = get_spark("candidates")
    start = time.time()
    cands = build_candidates(spark, [processed / "train.parquet"], args.as_of, users, ALS_PARAMS,
                             threshold=threshold, seed=cfg["seed"])
    spark.stop()

    out = processed / f"candidates_{args.as_of[:4]}.parquet"
    cands.to_parquet(out)
    print(f"Built {len(cands):,} candidates ({len(cands) / len(users):.0f} per user) "
          f"in {time.time() - start:.0f}s -> {out.name}")

    users_with_als = set(cands.loc[cands["from_als"], "user_id"])
    print(f"Users without an ALS vector (popular list only): {len(set(users) - users_with_als):,}\n")

    print("Candidate recall (average per user):")
    for name, subset in [("ALS top 100 only", cands[cands["from_als"]]),
                         ("Popular top 100 only", cands[cands["from_pop"]]),
                         ("Combined shortlist", cands)]:
        recall = candidate_recall(subset, answers, threshold, users)["recall"].mean()
        print(f"  {name:22s} {recall:.3f}")


if __name__ == "__main__":
    main()