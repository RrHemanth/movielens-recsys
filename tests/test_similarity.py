import numpy as np
import pytest

from recsys.models.similarity import most_similar, normalize

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