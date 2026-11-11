import tempfile
from pathlib import Path

import mlflow
import pandas as pd

from recsys.config import load_config
from recsys.data.movies import load_genres
from recsys.evaluation.evaluate import evaluate
from recsys.evaluation.stats import compare_paired
from recsys.models.cold_start import FoldInRecommender, blend_ranks, onboarding_split, top_unseen
from recsys.models.popularity import PopularityRecommender
from recsys.models.similarity import load_vectors
from recsys.tracking import git_info, setup_mlflow, summary_to_metrics

N_ONBOARD = 5
K = 10
BLEND_DEPTH = 200


def main():
    cfg = load_config()
    setup_mlflow(cfg["mlflow"]["experiment"])
    processed = cfg["paths"]["processed"]
    threshold = cfg["split"]["relevant_threshold"]

    train = pd.read_parquet(processed / "train.parquet")
    val = pd.read_parquet(processed / "val.parquet")
    movies = pd.read_parquet(cfg["paths"]["interim"] / "movies.parquet")
    genres = load_genres(movies)
    n_catalog = train["movie_id"].nunique()

    # New users in 2018, split into onboarding (first 5 ratings) and answer key (the rest)
    new_val = val[~val["user_id"].isin(set(train["user_id"]))]
    onboard, answers = onboarding_split(new_val, N_ONBOARD)
    users = sorted(answers["user_id"].unique())
    seen = onboard.groupby("user_id")["movie_id"].apply(set).to_dict()
    print(f"New users in 2018: {new_val['user_id'].nunique():,} | with more than {N_ONBOARD} ratings: {len(users):,}")

    # Method A: popularity (90 days), skipping the onboarding movies
    popular = PopularityRecommender(threshold=threshold, recent_days=90).fit(train).ranking
    pop_recs = {u: top_unseen(popular, seen[u], K) for u in users}

    # Method B: average the vectors of liked onboarding movies (ALS trained on 2017 likes, saved in Step 9)
    movie_ids, vectors = load_vectors(processed / "als" / "movie_vectors.parquet")
    foldin = FoldInRecommender(movie_ids, vectors, threshold)
    taste_deep = foldin.recommend(onboard, k=BLEND_DEPTH)
    n_fallback = sum(1 for u in users if u not in taste_deep)
    foldin_recs = {u: taste_deep[u][:K] if u in taste_deep else pop_recs[u] for u in users}

    # Method C: blend the taste ranking with the popular ranking
    blend_recs = {}
    for u in users:
        if u in taste_deep:
            blend_recs[u] = blend_ranks(taste_deep[u], top_unseen(popular, seen[u], BLEND_DEPTH), K, BLEND_DEPTH)
        else:
            blend_recs[u] = pop_recs[u]
    print(f"Users with no liked onboarding movie that has a vector (get popularity): {n_fallback:,}\n")

    pd.set_option("display.float_format", "{:.4f}".format)
    ndcg = {}
    for name, recs in [("popularity", pop_recs), ("fold-in", foldin_recs), ("blend", blend_recs)]:
        per_user, summary = evaluate(recs, answers, set(), genres, n_catalog, k=K, threshold=threshold)
        ndcg[name] = per_user.set_index("user_id")["ndcg"].sort_index()
        with mlflow.start_run(run_name=f"cold start {name}"):
            mlflow.log_params({"model": f"cold_start_{name}", "n_onboard": N_ONBOARD, "k": K,
                               "blend_depth": BLEND_DEPTH, "eval_set": "val_new_users_after_onboarding"})
            mlflow.set_tags(git_info())
            mlflow.log_metrics(summary_to_metrics(summary))
            mlflow.log_metric("n_fallback_users", n_fallback)
            with tempfile.TemporaryDirectory() as tmp:
                per_user.to_parquet(Path(tmp) / "per_user_scores.parquet")
                mlflow.log_artifacts(tmp)
        print(f"=== {name} ===")
        print(summary.loc[["new"]])
        print()

    assert ndcg["popularity"].index.equals(ndcg["fold-in"].index), "Methods scored on different users"
    print("Paired comparisons on new users (B - A):")
    for a, b in [("popularity", "fold-in"), ("popularity", "blend"), ("fold-in", "blend")]:
        r = compare_paired(ndcg[a].to_numpy(), ndcg[b].to_numpy())
        print(f"  {b:10s} vs {a:10s} {r['mean_diff']:+.4f}  [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  "
              f"p={r['p_value']:.4f}  better {r['share_better']:.1%}  worse {r['share_worse']:.1%}")


if __name__ == "__main__":
    main()