import argparse
import time

from recsys.candidates import build_candidates, candidate_recall
from recsys.config import load_config
from recsys.data.periods import AS_OF_DATES, load_period, period_files
from recsys.spark import get_spark

ALS_PARAMS = {"rank": 64, "reg_param": 1.0, "alpha": 10.0, "max_iter": 10}   # chosen in Step 8


def main():
    parser = argparse.ArgumentParser(description="Build shortlists as of a pretend date and measure candidate recall.")
    parser.add_argument("--as-of", required=True, choices=AS_OF_DATES)
    args = parser.parse_args()

    cfg = load_config()
    processed = cfg["paths"]["processed"]
    threshold = cfg["split"]["relevant_threshold"]

    history, answers = load_period(processed, args.as_of)
    liked_users = set(answers.loc[answers["rating"] >= threshold, "user_id"])
    users = sorted(set(history["user_id"]) & liked_users)
    print(f"As of {args.as_of}: {len(users):,} returning users who liked something in the answer year")

    vectors_dir = processed / "als_2019" if args.as_of == "2019-01-01" else None
    spark = get_spark("candidates")
    start = time.time()
    cands = build_candidates(spark, period_files(processed, args.as_of), args.as_of, users, ALS_PARAMS,
                             threshold=threshold, seed=cfg["seed"], save_vectors_dir=vectors_dir)
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