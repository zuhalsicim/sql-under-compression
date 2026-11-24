"""Turn the result files of one run into a Markdown table."""

from sqlbench.settings import BASELINE, PRECISIONS
from sqlbench.stats import wilson_interval

HEADER = (
    "| Precision | File size | Peak memory | Tokens per second | Seconds per question "
    "| Execution accuracy (95 % interval) | Change against f16 |\n"
    "|---|---|---|---|---|---|---|\n"
)


def make_table(results: dict[str, dict]) -> str:
    """Return the results table. `results` maps a precision to its result file."""
    lines = [_row(precision, results.get(precision), results.get(BASELINE)) for precision in PRECISIONS]
    return HEADER + "\n".join(lines) + "\n"


def _row(precision: str, result: dict | None, baseline: dict | None) -> str:
    if result is None:
        return f"| {precision} | no result: this job did not finish | | | | | |"
    low, high = wilson_interval(result["correct"], result["questions"])
    share = result["correct"] / result["questions"]
    accuracy = f"{result['correct']}/{result['questions']} = {share:.1%} ({low:.1%} to {high:.1%})"
    return (
        f"| {precision} "
        f"| {result['file_size_bytes'] / 1e9:.2f} GB "
        f"| {result['peak_memory_bytes'] / 1e9:.2f} GB "
        f"| {result['tokens_per_second']:.1f} "
        f"| {result['mean_seconds_per_question']:.1f} "
        f"| {accuracy} "
        f"| {_change(precision, result, baseline)} |"
    )


def _change(precision: str, result: dict, baseline: dict | None) -> str:
    """Describe the accuracy of one precision against the baseline, question by question."""
    if precision == BASELINE:
        return "baseline"
    if baseline is None:
        return "no baseline"
    before = {row["index"]: row["correct"] for row in baseline["per_question"]}
    after = {row["index"]: row["correct"] for row in result["per_question"]}
    if before.keys() != after.keys():
        return "not comparable: different questions"
    lost = sum(1 for index in before if before[index] and not after[index])
    gained = sum(1 for index in before if after[index] and not before[index])
    points = 100 * (gained - lost) / len(before)
    return f"{points:+.1f} points (lost {lost}, gained {gained})"


def describe_runners(results: dict[str, dict]) -> str:
    """Return one line per machine, with the precisions measured on it."""
    precisions_on = {}
    for precision in PRECISIONS:
        if precision in results:
            runner = results[precision]["runner"]
            machine = (
                f"runner \"{runner['name']}\", {runner['cpu_model']}, {runner['cpu_count']} vCPU, "
                f"{runner['memory_bytes'] / 1e9:.1f} GB RAM, {runner['image']}"
            )
            precisions_on.setdefault(machine, []).append(precision)
    return "".join(f"- {', '.join(names)}: {machine}\n" for machine, names in precisions_on.items())
