import faiss
import numpy as np

from recsys.config import load_config
from recsys.models.similarity import load_vectors, most_similar, normalize

K = 10
N_CHECK = 200


def main():
    cfg = load_config()
    _, vectors = load_vectors(cfg["paths"]["processed"] / "als" / "movie_vectors.parquet")
    x = np.ascontiguousarray(normalize(vectors))

    index = faiss.IndexFlatIP(x.shape[1])
    index.add(x)
    _, found = index.search(x[:N_CHECK], K + 1)

    n_differ, n_equally_good, n_tie_at_10th = 0, 0, 0
    for i in range(N_CHECK):
        numpy_top, numpy_sims = most_similar(x, i, K)
        faiss_top = [p for p in found[i] if p != i][:K]
        if set(numpy_top) == set(faiss_top):
            continue
        n_differ += 1

        sims = x @ x[i]                       # true similarity of every movie to movie i
        if abs(sims[faiss_top].sum() - numpy_sims.sum()) < 1e-4:
            n_equally_good += 1               # both lists have the same total similarity
        sims[i] = -np.inf
        ranked = np.sort(sims)[::-1]
        if ranked[K - 1] - ranked[K] < 1e-5:
            n_tie_at_10th += 1                # 10th and 11th place have the same score

    print(f"Lists differ for {n_differ} of {N_CHECK} movies")
    print(f"  ...of those, equally good (same total similarity): {n_equally_good}")
    print(f"  ...of those, a tie between 10th and 11th place:    {n_tie_at_10th}")

    # How common are near-identical "twin" movies across the whole catalog?
    scores, _ = index.search(x, 2)            # nearest neighbor besides itself
    twins = (scores[:, 1] > 0.9999).sum()
    print(f"\nMovies with a near-identical twin (cosine > 0.9999): {twins:,} of {len(x):,}")


if __name__ == "__main__":
    main()