"""The confidence interval for an accuracy."""

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
