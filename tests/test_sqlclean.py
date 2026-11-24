from sqlbench.sqlclean import extract_sql


def test_plain_query_loses_its_semicolon():
    assert extract_sql("SELECT name FROM singer;") == "SELECT name FROM singer"


def test_query_in_a_code_block():
    answer = "```sql\nSELECT name\nFROM singer;\n```"
    assert extract_sql(answer) == "SELECT name\nFROM singer"


def test_code_block_without_language_name():
    assert extract_sql("```\nSELECT 1\n```") == "SELECT 1"
    assert extract_sql("```SELECT 1```") == "SELECT 1"


def test_sentences_around_the_code_block_are_dropped():
    answer = "Here is the query:\n```sql\nSELECT count(*) FROM singer;\n```\nIt counts all singers."
    assert extract_sql(answer) == "SELECT count(*) FROM singer"


def test_sentence_before_a_query_without_code_block():
    answer = "The query with a join is: SELECT a FROM t JOIN u ON t.id = u.id"
    assert extract_sql(answer) == "SELECT a FROM t JOIN u ON t.id = u.id"


def test_explanation_after_the_semicolon_is_dropped():
    assert extract_sql("SELECT 1; This returns one.") == "SELECT 1"


def test_only_the_first_statement_is_kept():
    assert extract_sql("SELECT 1; SELECT 2;") == "SELECT 1"


def test_semicolon_inside_quotes_does_not_end_the_query():
    assert extract_sql("SELECT * FROM t WHERE name = 'a;b';") == "SELECT * FROM t WHERE name = 'a;b'"


def test_query_that_starts_with_with():
    answer = "WITH old AS (SELECT * FROM singer WHERE age > 40) SELECT count(*) FROM old;"
    assert extract_sql(answer) == answer[:-1]


def test_code_block_that_was_never_closed():
    assert extract_sql("```sql\nSELECT 1") == "SELECT 1"


def test_lower_case_query():
    assert extract_sql("select name from singer") == "select name from singer"


def test_answer_without_query():
    assert extract_sql("") == ""
    assert extract_sql("I cannot answer this question.") == ""
