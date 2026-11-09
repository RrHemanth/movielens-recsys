import numpy as np
import pandas as pd


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale every vector to length 1, so a dot product equals cosine similarity."""
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0   # avoid dividing by zero
    return vectors / lengths


def most_similar(normed: np.ndarray, index: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact search: positions and similarities of the k vectors most similar to `index`."""
    sims = normed @ normed[index]   # cosine similarity with every vector
    sims[index] = -np.inf           # don't return the movie itself
    top = np.argsort(-sims)[:k]
    return top, sims[top]


def load_vectors(path) -> tuple[np.ndarray, np.ndarray]:
    """Read a vectors file saved by ALSRecommender.save_vectors -> (ids, matrix)."""
    df = pd.read_parquet(path)
    return df.iloc[:, 0].to_numpy(), np.array(df["vector"].tolist(), dtype=np.float32)

def drop_self(neighbors: np.ndarray, k: int) -> np.ndarray:
    """Each row q lists neighbors of movie q; remove q itself and keep the first k."""
    result = np.full((len(neighbors), k), -1, dtype=np.int64)
    for q, row in enumerate(neighbors):
        row = row[row != q][:k]
        result[q, :len(row)] = row
    return result


def neighbor_recall(approx: np.ndarray, exact: np.ndarray) -> float:
    """Average fraction of the true (exact) neighbors that the approximate search found."""
    k = exact.shape[1]
    found = [len(set(a) & set(e)) / k for a, e in zip(approx, exact)]
    return float(np.mean(found))