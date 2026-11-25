import pytest

from sqlbench.stats import sign_test, wilson_interval


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


def test_sign_test_on_the_three_comparisons_of_the_full_run():
    # Lost and gained questions against f16 in results/: q8_0, q5_k_m, q4_k_m.
    assert sign_test(lost=1, gained=2) == 1.0
    assert sign_test(lost=1, gained=6) == 0.125
    assert sign_test(lost=9, gained=5) == pytest.approx(0.424, abs=0.0005)


def test_sign_test_by_hand():
    # 5 questions differ, all in one direction. Fair coins do that with
    # chance 1/32. Both directions together: 2/32.
    assert sign_test(lost=5, gained=0) == 2 / 32
    # 1 lost, 6 gained: 0 or 1 heads in 7 tosses has chance (1 + 7) / 128, times 2.
    assert sign_test(lost=1, gained=6) == 2 * (1 + 7) / 128


def test_sign_test_without_any_difference():
    assert sign_test(lost=0, gained=0) == 1.0


def test_sign_test_with_an_even_split_is_never_above_one():
    assert sign_test(lost=3, gained=3) == 1.0
    assert sign_test(lost=1, gained=1) == 1.0


def test_sign_test_does_not_care_about_the_direction():
    assert sign_test(lost=9, gained=5) == sign_test(lost=5, gained=9)


def test_sign_test_gets_smaller_when_the_split_gets_more_uneven():
    assert sign_test(lost=10, gained=0) < sign_test(lost=9, gained=1) < sign_test(lost=8, gained=2)
    assert sign_test(lost=10, gained=0) == 2 / 1024


def test_sign_test_refuses_negative_counts():
    with pytest.raises(ValueError):
        sign_test(lost=-1, gained=3)
