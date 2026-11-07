import pytest

from recsys.evaluation.metrics import ndcg_at_k, recall_at_k

MAYA_LOVED = {"A", "C", "F"}


# --- Recall ---

def test_recall_mayas_first_list():
    assert recall_at_k(["A", "B", "C", "D", "E"], MAYA_LOVED, k=5) == pytest.approx(2 / 3)


def test_recall_ignores_order():
    # Same two hits, different positions -> same recall.
    assert recall_at_k(["B", "D", "E", "A", "C"], MAYA_LOVED, k=5) == pytest.approx(2 / 3)


def test_recall_only_counts_top_k():
    # "A" is a hit, but it's at position 6, outside the top 5.
    assert recall_at_k(["B", "D", "E", "G", "H", "A"], MAYA_LOVED, k=5) == 0.0


# --- NDCG: the three lists we worked out by hand ---

def test_ndcg_hits_at_1_and_3():
    assert ndcg_at_k(["A", "B", "C", "D", "E"], MAYA_LOVED, k=5) == pytest.approx(0.70, abs=0.01)


def test_ndcg_hits_at_4_and_5():
    assert ndcg_at_k(["B", "D", "E", "A", "C"], MAYA_LOVED, k=5) == pytest.approx(0.38, abs=0.01)


def test_ndcg_hits_at_2_and_5():
    assert ndcg_at_k(["B", "A", "D", "E", "F"], MAYA_LOVED, k=5) == pytest.approx(0.48, abs=0.01)


# --- NDCG: edge cases ---

def test_ndcg_perfect_list_scores_1():
    assert ndcg_at_k(["A", "C", "F", "B", "D"], MAYA_LOVED, k=5) == pytest.approx(1.0)


def test_ndcg_no_hits_scores_0():
    assert ndcg_at_k(["B", "D", "E", "G", "H"], MAYA_LOVED, k=5) == 0.0


def test_ndcg_more_loved_movies_than_slots():
    # User loved 8 movies, but the list has only 3 slots. Filling all 3 with hits is perfect.
    loved = {"A", "B", "C", "D", "E", "F", "G", "H"}
    assert ndcg_at_k(["A", "B", "C"], loved, k=3) == pytest.approx(1.0)


def test_user_with_no_loved_movies_raises():
    with pytest.raises(ValueError):
        ndcg_at_k(["A", "B"], set(), k=2)