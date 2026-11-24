"""Cut the answer of the model down to one SQL statement."""

import re
import sqlite3

# The start of a ``` code block, with a language name such as "sql" if there is one.
FENCE_START = r"```(?:[a-zA-Z]+\n)?"
CODE_BLOCK = re.compile(FENCE_START + r"(.*?)```", re.DOTALL)
SELECT_WORD = re.compile(r"\bselect\b", re.IGNORECASE)
STARTS_WITH_WITH = re.compile(r"with\b", re.IGNORECASE)


def extract_sql(answer: str) -> str:
    """Return the first SQL statement in `answer`, or "" if there is none.

    Models often wrap the query in a ```sql code block, write a sentence
    before it, or add an explanation after it. This removes all of that.
    """
    block = CODE_BLOCK.search(answer)
    if block:
        text = block.group(1)
    else:
        text = re.sub(FENCE_START, "", answer)  # a code block that was never closed
    text = text.strip()

    # A query starts with SELECT, or with WITH if it defines a helper table first.
    if not STARTS_WITH_WITH.match(text):
        select = SELECT_WORD.search(text)
        if not select:
            return ""
        text = text[select.start():]

    # Cut at the first semicolon that really ends a statement.
    # A semicolon inside quotes does not end it.
    for position, character in enumerate(text):
        if character == ";" and sqlite3.complete_statement(text[: position + 1]):
            return text[:position].strip()
    return text.strip()
