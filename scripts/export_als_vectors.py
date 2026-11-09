import json

import pandas as pd

from recsys.config import load_config
from recsys.data.split import to_unix_seconds
from recsys.models.als import ALSRecommender
from recsys.models.similarity import load_vectors, most_similar, normalize
from recsys.spark import get_spark

# The chosen setting from Step 8's tuning.
SETTINGS = {"train_start_year": 2017, "rank": 64, "reg_param": 1.0, "alpha": 10.0, "max_iter": 10}
CHECK_TITLES = ["Toy Story (1995)", "Dark Knight, The (2008)", "Inception (2010)"]


def main():
    cfg = load_config()
    out_dir = cfg["paths"]["processed"] / "als"

    # 1. Train once and save the vectors
    spark = get_spark("als-export")
    model = ALSRecommender(
        rank=SETTINGS["rank"], reg_param=SETTINGS["reg_param"], alpha=SETTINGS["alpha"],
        max_iter=SETTINGS["max_iter"], threshold=cfg["split"]["relevant_threshold"],
        seed=cfg["seed"], train_start_ts=to_unix_seconds(f"{SETTINGS['train_start_year']}-01-01"),
    ).fit(spark.read.parquet(str(cfg["paths"]["processed"] / "train.parquet")))
    model.save_vectors(out_dir)
    spark.stop()
    (out_dir / "settings.json").write_text(json.dumps(SETTINGS, indent=2))  # record what produced the files
    print(f"Saved {len(model.movie_ids):,} movie vectors and {len(model.user_index):,} user vectors to {out_dir}")

    # 2. Sanity check: nearest neighbors of a few well-known movies (exact search)
    movies = pd.read_parquet(cfg["paths"]["interim"] / "movies.parquet")
    titles = dict(zip(movies["movie_id"], movies["title"]))
    title_to_id = {t: m for m, t in titles.items()}
    movie_ids, vectors = load_vectors(out_dir / "movie_vectors.parquet")
    position = {int(m): i for i, m in enumerate(movie_ids)}
    normed = normalize(vectors)

    for title in CHECK_TITLES:
        pos = position.get(int(title_to_id[title]))
        if pos is None:
            print(f"\n{title}: not in the ALS catalog")
            continue
        top, sims = most_similar(normed, pos, k=5)
        print(f"\nMost similar to {title}:")
        for p, s in zip(top, sims):
            print(f"  {s:.3f}  {titles[int(movie_ids[p])]}")


if __name__ == "__main__":
    main()