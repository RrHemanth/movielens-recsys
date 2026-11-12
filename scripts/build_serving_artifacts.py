import json
import time
from datetime import UTC, datetime

import pandas as pd

from recsys.candidates import build_candidates
from recsys.config import load_config
from recsys.data.movies import load_genres
from recsys.data.periods import load_period, period_files
from recsys.data.split import to_unix_seconds
from recsys.features.reranker import add_features
from recsys.models.popularity import PopularityRecommender
from recsys.models.reranker import LightGBMReranker
from recsys.spark import get_spark
from recsys.tracking import git_info

AS_OF = "2019-01-01"                                                        # the snapshot we tested
ALS_PARAMS = {"rank": 64, "reg_param": 1.0, "alpha": 10.0, "max_iter": 10}  # chosen in Step 8
K = 10
N_POPULAR = 500
YEAR = 365 * 24 * 3600


def main():
    cfg = load_config()
    processed = cfg["paths"]["processed"]
    out_dir = cfg["paths"]["serving"]
    out_dir.mkdir(parents=True, exist_ok=True)
    threshold = cfg["split"]["relevant_threshold"]
    as_of_ts = to_unix_seconds(AS_OF)

    history, _ = load_period(processed, AS_OF)   # the answers are deliberately ignored: serving only knows the past
    movies = pd.read_parquet(cfg["paths"]["interim"] / "movies.parquet")
    users = sorted(set(history.loc[history["ts"] >= as_of_ts - YEAR, "user_id"]))
    print(f"Precomputing for {len(users):,} users active in the year before {AS_OF}")

    # 1. Shortlists (ALS on 2018 likes + 90-day popularity), saving the ALS vectors for fold-in
    start = time.time()
    spark = get_spark("serving")
    cands = build_candidates(spark, period_files(processed, AS_OF), AS_OF, users, ALS_PARAMS,
                             threshold=threshold, seed=cfg["seed"], save_vectors_dir=out_dir)
    spark.stop()

    # 2. Features, then the re-ranker (trained exactly as in the final test)
    table = add_features(cands, history, movies, load_genres(movies), as_of_ts, threshold)
    reranker = LightGBMReranker(objective="binary", seed=cfg["seed"]).fit(
        pd.read_parquet(processed / "features_2018.parquet"))
    recs = reranker.recommend(table, k=K)

    # 3. Save everything the API needs
    rows = [(u, rank, m) for u, movie_list in recs.items() for rank, m in enumerate(movie_list, start=1)]
    pd.DataFrame(rows, columns=["user_id", "rank", "movie_id"]).to_parquet(out_dir / "recommendations.parquet")
    popularity = PopularityRecommender(threshold=threshold, recent_days=90).fit(history)
    pd.DataFrame({"movie_id": popularity.ranking[:N_POPULAR]}).to_parquet(out_dir / "popular.parquet")
    movies[["movie_id", "title", "genres"]].to_parquet(out_dir / "movies.parquet")

    manifest = {
        "as_of": AS_OF,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_users_precomputed": len(recs),
        "k": K,
        "als_params": ALS_PARAMS,
        "reranker": "LightGBM binary, trained on features as of 2018-01-01 with 2018 answers",
        "fallback": "90-day popularity",
        **git_info(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Done in {time.time() - start:.0f}s. Saved to {out_dir}:")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name:28s} {f.stat().st_size / 1e6:6.1f} MB")

    # 4. Sanity check: what do two users actually get?
    titles = dict(zip(movies["movie_id"], movies["title"]))
    for u in users[:2]:
        print(f"\nUser {u}:", [titles[m] for m in recs.get(u, [])[:5]])


if __name__ == "__main__":
    main()