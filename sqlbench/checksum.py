"""A checksum of a file, to tell whether two runs used exactly the same model file."""

import hashlib
from pathlib import Path


def sha256_of_file(path: str | Path, piece_bytes: int = 1024 * 1024) -> str:
    """Return the SHA-256 of a file as 64 hex characters.

    The file is read piece by piece, so a model file of several GB needs
    only one megabyte of memory at a time.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while piece := file.read(piece_bytes):
            digest.update(piece)
    return digest.hexdigest()
