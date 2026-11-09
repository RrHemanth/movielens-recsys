import time

import faiss
import numpy as np

from recsys.config import load_config
from recsys.models.similarity import drop_self, load_vectors, most_similar, neighbor_recall, normalize

K = 10
N_SECTIONS = 128   # nlist: how many "sections" the approximate index groups movies into


def main():
    cfg = load_config()
    movie_ids, vectors = load_vectors(cfg["paths"]["processed"] / "als" / "movie_vectors.parquet")
    x = np.ascontiguousarray(normalize(vectors))   # length-1 vectors: inner product = cosine
    n, dim = x.shape
    print(f"{n:,} movies, {dim} numbers each\n")

    # 1. Exact search: compare every movie against every movie
    exact_index = faiss.IndexFlatIP(dim)
    exact_index.add(x)
    start = time.time()
    _, found = exact_index.search(x, K + 1)
    exact_ms = (time.time() - start) * 1000 / n
    exact = drop_self(found, K)

    # Sanity check: FAISS exact search should agree with our NumPy version
    agree = np.mean([set(most_similar(x, i, K)[0]) == set(exact[i]) for i in range(200)])
    print(f"FAISS exact vs NumPy exact: same top-{K} for {agree:.0%} of 200 checked movies")

    # 2. Approximate search: group movies into sections, search only the closest few
    quantizer = faiss.IndexFlatIP(dim)
    approx_index = faiss.IndexIVFFlat(quantizer, dim, N_SECTIONS, faiss.METRIC_INNER_PRODUCT)
    approx_index.train(x)   # learn the sections (k-means)
    approx_index.add(x)

    print(f"\n{'search':>22} {'ms per movie':>13} {'speed-up':>9} {'neighbor recall':>16}")
    print(f"{'exact':>22} {exact_ms:13.4f} {1.0:8.1f}x {1.0:16.3f}")
    for nprobe in [1, 4, 16, 64]:
        approx_index.nprobe = nprobe   # how many sections to look inside
        start = time.time()
        _, found = approx_index.search(x, K + 1)
        ms = (time.time() - start) * 1000 / n
        recall = neighbor_recall(drop_self(found, K), exact)
        label = f"approx, {nprobe} of {N_SECTIONS} sections"
        print(f"{label:>22} {ms:13.4f} {exact_ms / ms:8.1f}x {recall:16.3f}")


if __name__ == "__main__":
    main()