"""Build the text that the model reads."""

from pathlib import Path

from sqlbench.sqlexec import open_read_only

SYSTEM_MESSAGE = "You are an SQLite expert. Answer with one SQLite query and nothing else."


def read_schema(db_path: str | Path) -> str:
    """Return the CREATE TABLE statements of a SQLite database as one text."""
    connection = open_read_only(db_path)
    try:
        rows = connection.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL "
            "ORDER BY rowid"
        ).fetchall()
    finally:
        connection.close()
    return "\n\n".join(row[0].strip() for row in rows)


def build_prompt(schema: str, question: str) -> str:
    """Return the full prompt in ChatML, the chat format Qwen models are trained on.

    The schema comes before the question. Questions about the same database
    then share the start of the prompt, and llama.cpp can reuse the work it
    already did for that start.
    """
    return (
        f"<|im_start|>system\n{SYSTEM_MESSAGE}<|im_end|>\n"
        f"<|im_start|>user\nDatabase schema:\n{schema}\n\nQuestion: {question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
