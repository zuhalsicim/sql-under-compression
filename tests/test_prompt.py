from sqlbench.prompt import SYSTEM_MESSAGE, build_prompt, read_schema


def test_read_schema_returns_the_create_statements(db_path):
    schema = read_schema(db_path)
    assert schema.startswith("CREATE TABLE singer (")
    assert "CREATE TABLE concert (" in schema
    # AUTOINCREMENT makes SQLite add its own table "sqlite_sequence". It is left out.
    assert "sqlite_sequence" not in schema


def test_prompt_is_chatml_and_ends_where_the_model_answers():
    prompt = build_prompt("CREATE TABLE t (a INT)", "How many rows?")
    assert prompt == (
        f"<|im_start|>system\n{SYSTEM_MESSAGE}<|im_end|>\n"
        "<|im_start|>user\nDatabase schema:\nCREATE TABLE t (a INT)\n\nQuestion: How many rows?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def test_two_questions_about_one_database_share_the_start_of_the_prompt():
    schema = "CREATE TABLE t (a INT)"
    first = build_prompt(schema, "How many rows?")
    second = build_prompt(schema, "What is the largest a?")
    shared = first[: first.index("Question:")]
    assert schema in shared
    assert second.startswith(shared)
