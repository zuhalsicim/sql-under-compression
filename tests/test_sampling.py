import pytest

from sqlbench.sampling import choose_indices
from sqlbench.settings import FULL_QUESTIONS, SEED


def test_same_seed_gives_same_questions():
    assert choose_indices(1034, 200, seed=42) == choose_indices(1034, 200, seed=42)


def test_other_seed_gives_other_questions():
    assert choose_indices(1034, 200, seed=42) != choose_indices(1034, 200, seed=43)


def test_questions_are_different_sorted_and_in_range():
    indices = choose_indices(1034, 200, seed=42)
    assert len(indices) == 200
    assert len(set(indices)) == 200
    assert indices == sorted(indices)
    assert 0 <= indices[0] and indices[-1] < 1034


def test_sample_does_not_depend_on_python_version():
    # The Spider development set has 1034 questions. These numbers were
    # computed once with Python 3.13. If another Python version chose other
    # questions, results from different machines could not be compared.
    indices = choose_indices(1034, FULL_QUESTIONS, SEED)
    assert indices[:8] == [3, 6, 11, 27, 32, 44, 46, 48]
    assert indices[-3:] == [1016, 1020, 1024]
    assert sum(indices) == 100493


def test_asking_for_too_many_questions_fails():
    with pytest.raises(ValueError):
        choose_indices(10, 11, seed=1)
    with pytest.raises(ValueError):
        choose_indices(10, 0, seed=1)
