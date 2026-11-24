import sqlite3

import pytest


@pytest.fixture
def db_path(tmp_path):
    """A small SQLite database with singers, two of them with the same name and age."""
    path = tmp_path / "concert.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE singer (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, age INTEGER);
        CREATE TABLE concert (id INTEGER PRIMARY KEY, singer_id INTEGER REFERENCES singer(id));
        INSERT INTO singer (name, age) VALUES ('Ada', 30), ('Ben', 41), ('Cem', 52), ('Ada', 30);
        INSERT INTO concert VALUES (1, 1), (2, 1), (3, 2);
        """
    )
    connection.close()
    return path
