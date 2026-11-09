import numpy as np
import pytest

from recsys.models.similarity import drop_self, most_similar, neighbor_recall, normalize

# Toy Story, Up, Heat -- the (fun, intense) example from Step 3.
VECTORS = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9]], dtype=np.float32)


def test_normalized_vectors_have_length_one():
    lengths = np.linalg.norm(normalize(VECTORS), axis=1)
    assert lengths == pytest.approx([1.0, 1.0, 1.0])


def test_toy_storys_neighbors():
    top, sims = most_similar(normalize(VECTORS), index=0, k=2)
    assert top.tolist() == [1, 2]                      # Up first, then Heat
    assert sims[0] == pytest.approx(0.99, abs=0.01)    # Toy Story vs Up
    assert sims[1] == pytest.approx(0.22, abs=0.01)    # Toy Story vs Heat

def test_drop_self_removes_the_query_itself():
    neighbors = np.array([[0, 5, 6], [7, 1, 8]])   # row 0 is movie 0, row 1 is movie 1
    assert drop_self(neighbors, k=2).tolist() == [[5, 6], [7, 8]]


def test_neighbor_recall():
    exact = np.array([[1, 2, 9], [4, 5, 6]])
    approx = np.array([[1, 2, 3], [4, 5, 6]])   # found 2 of 3, then 3 of 3
    assert neighbor_recall(approx, exact) == pytest.approx((2 / 3 + 1) / 2)