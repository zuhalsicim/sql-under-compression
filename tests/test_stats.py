import pytest

from sqlbench.stats import wilson_interval


def test_half_of_one_hundred():
    low, high = wilson_interval(50, 100)
    assert low == pytest.approx(0.4038, abs=0.0001)
    assert high == pytest.approx(0.5962, abs=0.0001)


def test_no_success_still_has_an_upper_bound():
    # For 0 successes the upper bound is z*z / (n + z*z).
    low, high = wilson_interval(0, 10)
    assert low == 0.0
    assert high == pytest.approx(1.96**2 / (10 + 1.96**2))


def test_all_successes_still_has_a_lower_bound():
    low, high = wilson_interval(10, 10)
    assert low == pytest.approx(10 / (10 + 1.96**2))
    assert high == 1.0


def test_interval_contains_the_share_and_shrinks_with_more_trials():
    low_small, high_small = wilson_interval(13, 20)
    low_large, high_large = wilson_interval(130, 200)
    assert low_small < 0.65 < high_small
    assert low_large < 0.65 < high_large
    assert high_large - low_large < high_small - low_small


def test_impossible_counts_fail():
    with pytest.raises(ValueError):
        wilson_interval(1, 0)
    with pytest.raises(ValueError):
        wilson_interval(11, 10)
