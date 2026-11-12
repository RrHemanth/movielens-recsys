import tempfile
from pathlib import Path

import mlflow
import pandas as pd

from recsys.config import load_config
from recsys.data.movies import load_genres
from recsys.data.periods import load_period
from recsys.evaluation.evaluate import evaluate
from recsys.evaluation.stats import bootstrap_ci, compare_paired
from recsys.models.cold_start import FoldInRecommender, onboarding_split, top_unseen
from recsys.models.popularity import PopularityRecommender
from recsys.models.reranker import LightGBMReranker
from recsys.models.similarity import load_vectors
from recsys.tracking import git_info, setup_mlflow

K = 10
N_ONBOARD = 5


def main():
    cfg = load_config()
    setup_mlflow(cfg["mlflow"]["experiment"])
    processed = cfg["paths"]["processed"]
    threshold = cfg["split"]["relevant_threshold"]

    history, test = load_period(processed, "2019-01-01")   # history = everything before 2019
    movies = pd.read_parquet(cfg["paths"]["interim"] / "movies.parquet")
    genres = load_genres(movies)
    n_catalog = history["movie_id"].nunique()
    history_users = set(history["user_id"])
    popularity = PopularityRecommender(threshold=threshold, recent_days=90).fit(history)   # Oct-Dec 2018

    results, per_user = {}, {}

    # ---------- Returning users: popularity vs ALS vs the re-ranker ----------
    shortlists = pd.read_parquet(processed / "features_2019.parquet")
    users = sorted(shortlists["user_id"].unique())
    returning_answers = test[test["user_id"].isin(history_users)]

    pop_recs = popularity.recommend(users, k=K)
    als_top = shortlists[shortlists["als_rank"] <= K].sort_values(["user_id", "als_rank"])
    als_lists = als_top.groupby("user_id")["movie_id"].apply(list).to_dict()
    als_recs = {u: als_lists.get(u, pop_recs[u]) for u in users}            # no ALS vector -> popularity
    reranker = LightGBMReranker(objective="binary", seed=cfg["seed"]).fit(
        pd.read_parquet(processed / "features_2018.parquet"))               # learns from 2018 answers
    rerank_recs = reranker.recommend(shortlists, k=K)

    for name, recs in [("popularity", pop_recs), ("als", als_recs), ("re-ranker", rerank_recs)]:
        scores, summary = evaluate(recs, returning_answers, history_users, genres, n_catalog, k=K, threshold=threshold)
        key = f"returning / {name}"
        per_user[key] = scores.set_index("user_id")["ndcg"].sort_index()
        results[key] = summary.loc["returning"]

    # ---------- New users: popularity vs fold-in, after 5 onboarding ratings ----------
    onboard, new_answers = onboarding_split(test[~test["user_id"].isin(history_users)], N_ONBOARD)
    new_users = sorted(new_answers["user_id"].unique())
    seen = onboard.groupby("user_id")["movie_id"].apply(set).to_dict()
    pop_new = {u: top_unseen(popularity.ranking, seen[u], K) for u in new_users}
    movie_ids, vectors = load_vectors(processed / "als_2019" / "movie_vectors.parquet")   # ALS on 2018 likes
    fold = FoldInRecommender(movie_ids, vectors, threshold).recommend(onboard, k=K)
    fold_new = {u: fold.get(u, pop_new[u]) for u in new_users}

    for name, recs in [("popularity", pop_new), ("fold-in", fold_new)]:
        scores, summary = evaluate(recs, new_answers, set(), genres, n_catalog, k=K, threshold=threshold)
        key = f"new / {name}"
        per_user[key] = scores.set_index("user_id")["ndcg"].sort_index()
        results[key] = summary.loc["new"]

    # ---------- Report ----------
    print("FINAL TEST: 2019 (run once; all choices frozen from validation)\n")
    print(f"  {'system':26s} {'users':>6s} {'NDCG@10 [95% CI]':>27s} {'recall':>7s} {'coverage':>9s} {'diversity':>9s}")
    for key, row in results.items():
        mean, low, high = bootstrap_ci(per_user[key].to_numpy())
        print(f"  {key:26s} {int(row['n_users']):6,d} {mean:9.4f} [{low:.4f}, {high:.4f}] "
              f"{row['recall']:7.4f} {row['coverage']:9.4f} {row['diversity']:9.4f}")

    print("\nPaired comparisons on 2019 (B - A, same users):")
    for a, b in [("returning / popularity", "returning / re-ranker"),
                 ("returning / als", "returning / re-ranker"),
                 ("returning / popularity", "returning / als"),
                 ("new / popularity", "new / fold-in")]:
        assert per_user[a].index.equals(per_user[b].index), "Compared on different users"
        r = compare_paired(per_user[a].to_numpy(), per_user[b].to_numpy())
        print(f"  {b:24s} vs {a:24s} {r['mean_diff']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] "
              f"p={r['p_value']:.4f}")

    with mlflow.start_run(run_name="FINAL TEST 2019"):
        mlflow.log_params({"eval_set": "test_2019", "k": K, "n_onboard": N_ONBOARD,
                           "returning_system": "shortlist (ALS a=10 r=1.0 + pop90) -> LightGBM binary",
                           "new_user_system": "fold-in (avg liked onboarding vectors), popularity fallback"})
        mlflow.set_tags(git_info())
        for key, row in results.items():
            prefix = key.replace(" / ", "_").replace("-", "")
            mlflow.log_metrics({f"{prefix}_{m}": float(row[m]) for m in ["ndcg", "recall", "coverage", "diversity"]})
        with tempfile.TemporaryDirectory() as tmp:
            for key, s in per_user.items():
                s.to_frame("ndcg").to_parquet(Path(tmp) / f"{key.replace(' / ', '_').replace('-', '')}.parquet")
            mlflow.log_artifacts(tmp)


if __name__ == "__main__":
    main()