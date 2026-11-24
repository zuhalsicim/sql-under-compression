"""Step 1: download the Spider development set and unpack what the benchmark needs.

Usage: python fetch_spider.py data

Afterwards the folder holds
    data/spider/dev.json                           the questions with their reference SQL
    data/spider/database/<name>/<name>.sqlite      one SQLite file per database
"""

import hashlib
import json
import sys
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

from sqlbench.settings import SPIDER_FILE, SPIDER_REPO, SPIDER_REVISION, SPIDER_SHA256


def main() -> None:
    target = Path(sys.argv[1])
    zip_path = hf_hub_download(SPIDER_REPO, SPIDER_FILE, repo_type="dataset", revision=SPIDER_REVISION)

    # Make sure the file is exactly the one this benchmark was written for.
    digest = hashlib.sha256(Path(zip_path).read_bytes()).hexdigest()
    if digest != SPIDER_SHA256:
        raise SystemExit(f"unexpected file content: sha256 is {digest}")

    with zipfile.ZipFile(zip_path) as archive:
        archive.extract("spider/dev.json", target)
        questions = json.loads((target / "spider" / "dev.json").read_text(encoding="utf-8"))
        databases = sorted({question["db_id"] for question in questions})
        for name in databases:
            archive.extract(f"spider/database/{name}/{name}.sqlite", target)

    print(f"{len(questions)} questions about {len(databases)} databases unpacked to {target}")


if __name__ == "__main__":
    main()
