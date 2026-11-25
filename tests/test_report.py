from sqlbench.report import describe_memory, describe_runners, describe_second_pass, make_table


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
