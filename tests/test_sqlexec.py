import sqlite3

import pytest

from sqlbench import sqlexec
from sqlbench.sqlexec import check_answer, has_order_by, run_query, same_result


def test_run_query_returns_rows(db_path):
    assert run_query(db_path, "SELECT name, age FROM singer WHERE id = 2") == [("Ben", 41)]


def test_run_query_cannot_change_the_database(db_path):
    with pytest.raises(sqlite3.Error):
        run_query(db_path, "DELETE FROM singer")
    assert run_query(db_path, "SELECT count(*) FROM singer") == [(4,)]


def test_run_query_fails_on_wrong_sql(db_path):
    with pytest.raises(sqlite3.Error):
        run_query(db_path, "SELECT nothing FROM nowhere")


def test_run_query_reads_text_that_is_not_utf8(db_path):
    connection = sqlite3.connect(db_path)
    connection.execute("INSERT INTO singer (name, age) VALUES (CAST(x'c3' AS TEXT), 20)")
    connection.commit()
    connection.close()
    rows = run_query(db_path, "SELECT name FROM singer WHERE age = 20")
    assert len(rows) == 1


def test_run_query_stops_a_query_that_runs_too_long(db_path, monkeypatch):
    monkeypatch.setattr(sqlexec, "TIMEOUT_SECONDS", 0.2)
    endless = "WITH RECURSIVE numbers(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM numbers) SELECT count(*) FROM numbers"
    with pytest.raises(sqlite3.Error):
        run_query(db_path, endless)


def test_run_query_refuses_too_many_rows(db_path, monkeypatch):
    monkeypatch.setattr(sqlexec, "MAX_ROWS", 3)
    with pytest.raises(sqlite3.Error):
        run_query(db_path, "SELECT * FROM singer")


def test_has_order_by():
    assert has_order_by("SELECT name FROM singer ORDER BY age")
    assert has_order_by("select name from singer order  by age desc")
    assert not has_order_by("SELECT name FROM singer GROUP BY name")
    assert not has_order_by("SELECT reorder_bytes FROM t")


def test_row_order_is_ignored_when_not_ordered():
    assert same_result([(1,), (2,)], [(2,), (1,)], ordered=False)


def test_row_order_counts_when_ordered():
    assert not same_result([(1,), (2,)], [(2,), (1,)], ordered=True)
    assert same_result([(1,), (2,)], [(1,), (2,)], ordered=True)


def test_repeated_rows_count():
    assert not same_result([(1,), (1,), (2,)], [(1,), (2,)], ordered=False)
    assert not same_result([(1,), (1,), (2,)], [(1,), (2,), (2,)], ordered=False)


def test_column_order_counts():
    assert not same_result([("Ada", 30)], [(30, "Ada")], ordered=False)


def test_other_values_are_different():
    assert not same_result([(1,)], [(2,)], ordered=False)
    assert not same_result([(1,)], [], ordered=False)


def test_two_empty_results_are_the_same():
    assert same_result([], [], ordered=False)


def test_whole_number_as_float_equals_integer():
    assert same_result([(3.0,)], [(3,)], ordered=False)


def test_tiny_float_differences_are_ignored():
    assert same_result([(3.3333333333333335,)], [(3.333333333333333,)], ordered=False)
    assert not same_result([(3.33,)], [(3.34,)], ordered=False)


def test_check_answer_accepts_another_query_with_the_same_result(db_path):
    gold = "SELECT name FROM singer WHERE age > 40"
    predicted = "SELECT name FROM singer WHERE age >= 41 ORDER BY name DESC"
    assert check_answer(db_path, predicted, gold) == (True, "")


def test_check_answer_uses_row_order_only_if_gold_has_order_by(db_path):
    gold = "SELECT name FROM singer WHERE age > 40 ORDER BY age"
    assert check_answer(db_path, "SELECT name FROM singer WHERE age > 40 ORDER BY age DESC", gold) == (False, "")


def test_check_answer_sees_missing_distinct(db_path):
    assert check_answer(db_path, "SELECT name FROM singer", "SELECT DISTINCT name FROM singer") == (False, "")


def test_check_answer_reports_sql_errors(db_path):
    correct, note = check_answer(db_path, "SELECT title FROM singer", "SELECT name FROM singer")
    assert not correct
    assert note.startswith("SQLite error: no such column")


def test_check_answer_counts_text_that_sqlite_refuses_as_wrong(db_path):
    gold = "SELECT name FROM singer"
    assert check_answer(db_path, "SELECT 1; SELECT 2", gold)[0] is False
    assert check_answer(db_path, "SELECT 'a\0b'", gold)[0] is False
    assert check_answer(db_path, "DROP TABLE singer", gold)[0] is False
    assert run_query(db_path, "SELECT count(*) FROM singer") == [(4,)]


def test_check_answer_rejects_an_empty_answer(db_path):
    # Without this rule an empty answer would match a gold query with no rows.
    gold = "SELECT name FROM singer WHERE age > 100"
    assert check_answer(db_path, "", gold) == (False, "no SQL in the answer")


def test_check_answer_fails_loudly_if_the_gold_query_is_broken(db_path):
    with pytest.raises(sqlite3.Error):
        check_answer(db_path, "SELECT 1", "SELECT nothing FROM nowhere")
