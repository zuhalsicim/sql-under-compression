from sqlbench.report import (
    describe_agreement,
    describe_differences,
    describe_memory,
    describe_runners,
    describe_second_pass,
    make_table,
)


def result(correct_flags, size=1_000_000_000):
    """A result file like evaluate.py writes it, with made-up numbers."""
    return {
        "questions": len(correct_flags),
        "correct": sum(correct_flags),
        "file_size_bytes": size,
        "peak_memory_bytes": 2_500_000_000,
        "tokens_per_second": 12.34,
        "mean_seconds_per_question": 5.67,
        "per_question": [{"index": i, "correct": flag} for i, flag in enumerate(correct_flags)],
        "runner": {
            "name": "Runner 1",
            "cpu_model": "Test CPU",
            "cpu_count": 2,
            "memory_bytes": 8_000_000_000,
            "image": "ubuntu24",
        },
    }


def test_table_has_one_row_per_precision_in_fixed_order():
    results = {name: result([True, False]) for name in ["q4_k_m", "f16", "q8_0", "q5_k_m"]}
    rows = make_table(results).splitlines()[2:]
    assert [row.split("|")[1].strip() for row in rows] == ["f16", "q8_0", "q5_k_m", "q4_k_m"]


def test_row_shows_the_measured_numbers():
    table = make_table({"f16": result([True, True, True, False], size=3_090_000_000)})
    row = table.splitlines()[2]
    assert "| 3.09 GB | 2.50 GB | 12.3 | 5.7 |" in row
    assert "3/4 = 75.0% (30.1% to 95.4%)" in row
    assert row.endswith("| baseline |")


def test_change_counts_lost_and_gained_questions():
    results = {
        "f16": result([True, True, True, False]),
        "q4_k_m": result([True, False, False, True]),
    }
    row = make_table(results).splitlines()[-1]
    assert row.endswith("| -25.0 points (lost 2, gained 1, p = 1.000) |")


def test_change_shows_a_small_p_value_for_a_one_sided_split():
    results = {
        "f16": result([True] * 8),
        "q4_k_m": result([False] * 8),
    }
    row = make_table(results).splitlines()[-1]
    assert row.endswith("| -100.0 points (lost 8, gained 0, p = 0.008) |")


def test_missing_precision_is_shown_as_missing():
    table = make_table({"f16": result([True])})
    assert "| q8_0 | no result in this run |" in table


def test_different_questions_are_not_compared():
    other = result([True, True])
    other["per_question"][0]["index"] = 99
    table = make_table({"f16": result([True, True]), "q8_0": other})
    assert "not comparable: different questions" in table


def test_precisions_measured_on_the_same_machine_share_a_line():
    other = result([True])
    other["runner"]["name"] = "Runner 2"
    text = describe_runners({"f16": result([True]), "q8_0": other, "q4_k_m": result([True])})
    assert text == (
        '- f16, q4_k_m: runner "Runner 1", Test CPU, 2 vCPU, 8.0 GB RAM, ubuntu24\n'
        '- q8_0: runner "Runner 2", Test CPU, 2 vCPU, 8.0 GB RAM, ubuntu24\n'
    )


def test_memory_lines_split_mapped_file_and_allocated_memory():
    one = result([True])
    one["memory_after_generation"] = {"RssFile": 1_000_000_000, "RssAnon": 930_000_000}
    assert describe_memory({"q4_k_m": one}) == (
        "- q4_k_m: 1.00 GB model file mapped into memory, 0.93 GB allocated by the process\n"
    )


def second_pass(share, different=0):
    """A file like evaluate_with_pruna.py writes it, with made-up numbers."""
    return {
        "execution_accuracy_with_pruna": {"works": True, "value": share},
        "questions_with_a_different_outcome": different,
        "seconds_for_all_questions_with_evaluate_py": 40.0,
        "seconds_for_all_questions_with_pruna": 42.0,
    }


def test_second_pass_line_compares_counts_and_time():
    text = describe_second_pass({"f16": result([True, True, True, False])}, {"f16": second_pass(0.75)})
    assert text == (
        "- f16: evaluate.py 3 correct, pruna 3 correct, 0 questions with a different outcome; "
        "10.0 s per question in the first pass, 10.5 s in the second\n"
    )


def test_second_pass_line_shows_an_error_of_pruna():
    failed = {"execution_accuracy_with_pruna": {"works": False, "error": "ValueError: no"}}
    text = describe_second_pass({"f16": result([True])}, {"f16": failed})
    assert text == "- f16: pruna's EvaluationAgent failed: ValueError: no\n"


def answers(rows):
    """A result file with an answer text and an outcome per question, with made-up content."""
    one = result([correct for _, correct in rows])
    for row, (text, _) in zip(one["per_question"], rows):
        row["answer"] = row["sql"] = text
    return one


def test_accuracy_columns_can_come_from_a_second_run():
    first = {"f16": result([True, True], size=3_000_000_000), "q8_0": result([True, True], size=1_500_000_000)}
    second = {"f16": result([True, True, True, False]), "q8_0": result([True, False, False, False])}
    rows = make_table(first, second).splitlines()[2:4]
    assert "| 3.00 GB |" in rows[0] and "3/4 = 75.0%" in rows[0]
    assert "| 1.50 GB |" in rows[1] and "1/4 = 25.0%" in rows[1]
    assert rows[1].endswith("| -50.0 points (lost 2, gained 0, p = 0.500) |")


def test_precision_missing_in_the_second_run_is_shown_as_missing():
    table = make_table({"f16": result([True]), "q8_0": result([True])}, {"f16": result([True])})
    assert "| q8_0 | no result in this run |" in table


def test_differences_count_other_text_and_changed_outcomes():
    results = {
        "f16": answers([("SELECT 1", True), ("SELECT 2", True), ("SELECT 3", False), ("SELECT 4", True)]),
        "q4_k_m": answers([("SELECT 1", True), ("SELECT two", True), ("SELECT three", True), ("SELECT four", False)]),
    }
    assert describe_differences(results) == (
        "- q4_k_m: 3 of 4 answers have another SQL text than f16. "
        "2 of them change the outcome: 1 lost, 1 gained, p = 1.000. "
        "With 2 such questions no split can fall below 0.05.\n"
    )


def test_differences_say_which_split_would_be_needed():
    f16 = answers([(f"SELECT {i}", True) for i in range(14)])
    q8_0 = answers([(f"SELECT {i} + 0", i >= 9) for i in range(14)])
    text = describe_differences({"f16": f16, "q8_0": q8_0})
    assert "14 of 14 answers have another SQL text than f16." in text
    assert "9 of them change the outcome: 9 lost, 0 gained, p = 0.004." in text
    assert "With 9 such questions the split would have to be at least 8 to 1 to fall below 0.05." in text


def test_agreement_looks_only_at_questions_in_both_runs():
    first = answers([("SELECT 1", True), ("SELECT 2", False)])
    second = answers([("SELECT 1", True), ("SELECT two", False), ("SELECT 3", True)])
    assert describe_agreement({"f16": first}, {"f16": second}) == (
        "- f16: 2 questions are in both runs. 1 got the same answer text, 2 got the same outcome.\n"
    )


def test_a_very_small_p_value_is_not_printed_as_zero():
    results = {"f16": result([True] * 20), "q4_k_m": result([False] * 20)}
    row = make_table(results).splitlines()[-1]
    assert row.endswith("| -100.0 points (lost 20, gained 0, p < 0.001) |")
