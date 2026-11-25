import hashlib

from sqlbench.checksum import sha256_of_file


def test_checksum_of_a_small_file(tmp_path):
    path = tmp_path / "model.gguf"
    path.write_bytes(b"abc")
    # The SHA-256 of "abc" is a published test value.
    assert sha256_of_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_checksum_of_an_empty_file(tmp_path):
    path = tmp_path / "empty"
    path.write_bytes(b"")
    assert sha256_of_file(path) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_reading_in_pieces_gives_the_same_checksum(tmp_path):
    content = bytes(range(256)) * 1000  # 256,000 bytes, read in pieces of 1000 bytes
    path = tmp_path / "large"
    path.write_bytes(content)
    assert sha256_of_file(path, piece_bytes=1000) == hashlib.sha256(content).hexdigest()
    assert sha256_of_file(path, piece_bytes=7) == sha256_of_file(path)


def test_one_changed_byte_changes_the_checksum(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    first.write_bytes(b"x" * 5000)
    second.write_bytes(b"x" * 4999 + b"y")
    assert sha256_of_file(first) != sha256_of_file(second)
