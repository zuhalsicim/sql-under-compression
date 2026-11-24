"""Choose which questions are asked."""

import random


def choose_indices(total: int, count: int, seed: int) -> list[int]:
    """Return `count` different positions out of `total`, in ascending order.

    The same seed always gives the same positions. Ascending order keeps the
    questions in the order of the dataset, where questions about the same
    database follow each other.
    """
    if not 0 < count <= total:
        raise ValueError(f"cannot choose {count} questions out of {total}")
    return sorted(random.Random(seed).sample(range(total), count))
