"""Pure-Python helpers of the image benchmark. They need only the standard library."""

import csv
from pathlib import Path

from sqlbench.sampling import choose_indices


def read_prompts(tsv_path: str | Path, count: int, seed: int) -> list[str]:
    """Return `count` prompts from the PartiPrompts file, chosen with a fixed seed.

    The file is tab-separated and its first line names the columns.
    The prompt text is in the column "Prompt".
    """
    with open(tsv_path, encoding="utf-8", newline="") as file:
        prompts = [row["Prompt"] for row in csv.DictReader(file, delimiter="\t")]
    return [prompts[index] for index in choose_indices(len(prompts), count, seed)]


def spread(values: list[float]) -> float:
    """Return how far repeated measurements lie apart, as a share of their mean.

    0.05 means: the slowest and the fastest repeat differ by 5 % of the mean.
    """
    if not values:
        raise ValueError("no values")
    mean = sum(values) / len(values)
    return (max(values) - min(values)) / mean


def summarise(result: dict) -> dict:
    """Turn the per-image records of one setting into the numbers for the table."""
    first_pass = [row for row in result["images"] if row["repeat"] == 0]
    seconds = [row["seconds"] for row in first_pass]
    # For every prompt that was generated more than once: the spread of its times.
    by_prompt = {}
    for row in result["images"]:
        by_prompt.setdefault(row["prompt_number"], []).append(row["seconds"])
    spreads = [spread(times) for times in by_prompt.values() if len(times) > 1]
    return {
        "images": len(first_pass),
        "mean_seconds_per_image": sum(seconds) / len(seconds),
        "largest_spread_between_repeats": max(spreads) if spreads else None,
        # None if nothing was repeated.
        "repeats_gave_the_same_image": (
            all(row["same_as_first"] for row in result["images"] if row["repeat"] > 0) if spreads else None
        ),
    }


def make_table(results: dict[str, dict], scores: dict[str, dict]) -> str:
    """Return the Markdown table. `results` maps a setting to its result file, `scores` to its quality scores."""
    lines = [
        "| Setting | Seconds per image | Speed against base | Spread between repeats | Peak memory "
        "| CLIP score | SSIM to base image | LPIPS to base image |",
        "|---|---|---|---|---|---|---|---|",
    ]
    base = results.get("base")
    base_seconds = summarise(base)["mean_seconds_per_image"] if base and "error" not in base else None
    for name, result in results.items():
        if "error" in result:
            lines.append(f"| {name} | did not run: {result['error']} | | | | | | |")
            continue
        summary = summarise(result)
        seconds = summary["mean_seconds_per_image"]
        speed = f"{base_seconds / seconds:.2f} x" if base_seconds else ""
        largest = summary["largest_spread_between_repeats"]
        score = scores.get(name, {})
        lines.append(
            f"| {name} | {seconds:.1f} | {speed} "
            f"| {'' if largest is None else format(largest, '.1%')} "
            f"| {result['peak_memory_bytes'] / 1e9:.2f} GB "
            f"| {_number(score.get('clip_score'), '.2f')} "
            f"| {_number(score.get('ssim'), '.3f')} "
            f"| {_number(score.get('lpips'), '.3f')} |"
        )
    return "\n".join(lines) + "\n"


def _number(value: float | None, pattern: str) -> str:
    """Format a score, or say that it does not exist (the base model has no "similarity to base")."""
    return "" if value is None else format(value, pattern)
