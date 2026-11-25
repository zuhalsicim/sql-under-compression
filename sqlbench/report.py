"""Turn the result files of one run into a Markdown table."""

from sqlbench.settings import BASELINE, PRECISIONS
from sqlbench.stats import most_even_split_below, sign_test, wilson_interval

HEADER = (
    "| Precision | File size | Peak memory | Tokens per second | Seconds per question "
    "| Execution accuracy (95 % interval) | Change against f16 |\n"
    "|---|---|---|---|---|---|---|\n"
)


def make_table(results: dict[str, dict], accuracy_results: dict[str, dict] | None = None) -> str:
    """Return the results table. `results` maps a precision to its result file.

    File size, memory and speed always come from `results`. The two accuracy
    columns come from `accuracy_results` if it is given: a second run that
    asked more questions.
    """
    if accuracy_results is None:
        accuracy_results = results
    lines = [
        _row(precision, results.get(precision), accuracy_results.get(precision), accuracy_results.get(BASELINE))
        for precision in PRECISIONS
    ]
    return HEADER + "\n".join(lines) + "\n"


def _row(precision: str, result: dict | None, accuracy_result: dict | None, baseline: dict | None) -> str:
    if result is None or accuracy_result is None:
        return f"| {precision} | no result in this run | | | | | |"
    correct, questions = accuracy_result["correct"], accuracy_result["questions"]
    low, high = wilson_interval(correct, questions)
    accuracy = f"{correct}/{questions} = {correct / questions:.1%} ({low:.1%} to {high:.1%})"
    return (
        f"| {precision} "
        f"| {result['file_size_bytes'] / 1e9:.2f} GB "
        f"| {result['peak_memory_bytes'] / 1e9:.2f} GB "
        f"| {result['tokens_per_second']:.1f} "
        f"| {result['mean_seconds_per_question']:.1f} "
        f"| {accuracy} "
        f"| {_change(precision, accuracy_result, baseline)} |"
    )


def _change(precision: str, result: dict, baseline: dict | None) -> str:
    """Describe the accuracy of one precision against the baseline, question by question.

    The p-value comes from the sign test on the lost and gained questions.
    """
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
    return f"{points:+.1f} points (lost {lost}, gained {gained}, {_p_value(lost, gained)})"


def _p_value(lost: int, gained: int) -> str:
    """Return the p-value of the sign test as text. A very small value is not printed as 0.000."""
    p = sign_test(lost, gained)
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


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


def describe_memory(results: dict[str, dict]) -> str:
    """Return one line per precision that splits the memory held after generation into two parts."""
    lines = []
    for precision in PRECISIONS:
        if precision in results:
            memory = results[precision]["memory_after_generation"]
            lines.append(
                f"- {precision}: {memory['RssFile'] / 1e9:.2f} GB model file mapped into memory, "
                f"{memory['RssAnon'] / 1e9:.2f} GB allocated by the process\n"
            )
    return "".join(lines)


def describe_second_pass(results: dict[str, dict], second_passes: dict[str, dict]) -> str:
    """Return one line per precision that compares evaluate.py with the second pass through pruna."""
    lines = []
    for precision in PRECISIONS:
        if precision in results and precision in second_passes:
            first, second = results[precision], second_passes[precision]
            accuracy = second["execution_accuracy_with_pruna"]
            if not accuracy["works"]:
                lines.append(f"- {precision}: pruna's EvaluationAgent failed: {accuracy['error']}\n")
                continue
            questions = first["questions"]
            lines.append(
                f"- {precision}: evaluate.py {first['correct']} correct, "
                f"pruna {round(accuracy['value'] * questions)} correct, "
                f"{second['questions_with_a_different_outcome']} questions with a different outcome; "
                f"{second['seconds_for_all_questions_with_evaluate_py'] / questions:.1f} s per question "
                f"in the first pass, {second['seconds_for_all_questions_with_pruna'] / questions:.1f} s in the second\n"
            )
    return "".join(lines)


def describe_differences(results: dict[str, dict]) -> str:
    """Return one line per precision that says how its answers differ from the baseline's.

    The last sentence of a line says how clear a difference would have to be
    before the sign test could see it.
    """
    lines = []
    baseline = results.get(BASELINE)
    for precision in PRECISIONS:
        if precision == BASELINE or precision not in results or baseline is None:
            continue
        before = {row["index"]: row for row in baseline["per_question"]}
        after = {row["index"]: row for row in results[precision]["per_question"]}
        if before.keys() != after.keys():
            continue
        other_text = sum(1 for index in before if before[index]["sql"] != after[index]["sql"])
        lost = sum(1 for index in before if before[index]["correct"] and not after[index]["correct"])
        gained = sum(1 for index in before if after[index]["correct"] and not before[index]["correct"])
        split = most_even_split_below(lost + gained)
        if split:
            needed = f"the split would have to be at least {split[0]} to {split[1]} to fall below 0.05"
        else:
            needed = "no split can fall below 0.05"
        lines.append(
            f"- {precision}: {other_text} of {len(before)} answers have another SQL text than {BASELINE}. "
            f"{lost + gained} of them change the outcome: {lost} lost, {gained} gained, "
            f"{_p_value(lost, gained)}. With {lost + gained} such questions {needed}.\n"
        )
    return "".join(lines)


def describe_agreement(first: dict[str, dict], second: dict[str, dict]) -> str:
    """Return one line per precision that compares two runs on the questions both asked."""
    lines = []
    for precision in PRECISIONS:
        if precision in first and precision in second:
            one = {row["index"]: row for row in first[precision]["per_question"]}
            two = {row["index"]: row for row in second[precision]["per_question"]}
            shared = sorted(one.keys() & two.keys())
            same_answer = sum(1 for index in shared if one[index]["answer"] == two[index]["answer"])
            same_outcome = sum(1 for index in shared if one[index]["correct"] == two[index]["correct"])
            lines.append(
                f"- {precision}: {len(shared)} questions are in both runs. "
                f"{same_answer} got the same answer text, {same_outcome} got the same outcome.\n"
            )
    return "".join(lines)
