"""The statistics: an interval for one accuracy, and a test for the difference between two."""

import math


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Return the Wilson score interval for a share `successes / trials`.

    With z = 1.96 this is the 95 % interval. Unlike the simple
    "share plus or minus" formula it never leaves the range 0 to 1 and it
    still gives a sensible width when the share is 0 or 1.
    """
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError(f"cannot compute an interval for {successes} out of {trials}")
    share = successes / trials
    denominator = 1 + z * z / trials
    centre = (share + z * z / (2 * trials)) / denominator
    half_width = z * math.sqrt(share * (1 - share) / trials + z * z / (4 * trials * trials)) / denominator
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def sign_test(lost: int, gained: int) -> float:
    """Return the p-value of the exact two-sided sign test.

    Two versions of a model answer the same questions. The test looks only at
    the questions where they differ: `lost` went from right to wrong, `gained`
    from wrong to right. If both versions were equally good, each of these
    questions would be a fair coin toss. The p-value is the chance that fair
    coin tosses split at least as unevenly as the answers did. By convention
    a value below 0.05 counts as a sign of a real difference.

    For right-or-wrong answers this is the same as the exact McNemar test.
    """
    if lost < 0 or gained < 0:
        raise ValueError(f"cannot test {lost} lost and {gained} gained questions")
    tosses = lost + gained
    if tosses == 0:
        return 1.0  # the two versions never differ, so nothing speaks for a difference
    smaller = min(lost, gained)
    # The chance of `smaller` or fewer heads in `tosses` fair tosses ...
    one_side = sum(math.comb(tosses, heads) for heads in range(smaller + 1)) / 2**tosses
    # ... and the same again for the other side, because "lost" and "gained" could be swapped.
    return min(1.0, 2 * one_side)
