"""Run SQL on a SQLite database and decide whether two results are the same."""

import re
import sqlite3
import time
from collections import Counter
from pathlib import Path

ORDER_BY = re.compile(r"\border\s+by\b", re.IGNORECASE)
MAX_ROWS = 100_000
TIMEOUT_SECONDS = 30


def open_read_only(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite file so that no query can change it."""
    connection = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)
    # Some Spider databases contain text that is not valid UTF-8.
    connection.text_factory = lambda data: data.decode("utf-8", errors="replace")
    return connection


def run_query(db_path: str | Path, sql: str) -> list[tuple]:
    """Run one query and return its rows.

    A query that runs too long or returns too many rows raises sqlite3.Error.
    """
    connection = open_read_only(db_path)
    # SQLite calls this function again and again while it works.
    # When the function returns True, SQLite stops the query.
    deadline = time.monotonic() + TIMEOUT_SECONDS
    connection.set_progress_handler(lambda: time.monotonic() > deadline, 100_000)
    try:
        rows = connection.execute(sql).fetchmany(MAX_ROWS + 1)
    finally:
        connection.close()
    if len(rows) > MAX_ROWS:
        raise sqlite3.DataError(f"more than {MAX_ROWS} rows")
    return rows


def has_order_by(sql: str) -> bool:
    """Return True if the query asks for a row order."""
    return ORDER_BY.search(sql) is not None


def same_result(predicted: list[tuple], gold: list[tuple], ordered: bool) -> bool:
    """Return True if both results hold the same rows.

    Columns must be in the same order. Rows must be in the same order only
    if `ordered` is True. Otherwise the rows are compared as a bag: order
    does not matter, but how often each row appears does.
    """
    predicted = [_round_floats(row) for row in predicted]
    gold = [_round_floats(row) for row in gold]
    if ordered:
        return predicted == gold
    return Counter(predicted) == Counter(gold)


def _round_floats(row: tuple) -> tuple:
    """Round floats so that 3.3333333333333335 and 3.333333333333333 count as equal."""
    return tuple(round(value, 6) if isinstance(value, float) else value for value in row)


def check_answer(db_path: str | Path, predicted_sql: str, gold_sql: str) -> tuple[bool, str]:
    """Return (correct, note). The note says why an answer could not be run."""
    if not predicted_sql:
        return False, "no SQL in the answer"
    gold_rows = run_query(db_path, gold_sql)
    try:
        predicted_rows = run_query(db_path, predicted_sql)
    except sqlite3.Error as error:
        return False, f"SQLite error: {error}"
    return same_result(predicted_rows, gold_rows, ordered=has_order_by(gold_sql)), ""
