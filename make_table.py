"""Step 5: read the result files of one run and print the results table.

Usage: python make_table.py results
"""

import json
import sys
from pathlib import Path

from sqlbench.report import describe_memory, describe_runners, describe_second_pass, make_table
from sqlbench.settings import PRECISIONS


def read_files(folder: Path, ending: str) -> dict[str, dict]:
    """Read "<precision><ending>" for every precision that has such a file."""
    files = {}
    for precision in PRECISIONS:
        path = folder / f"{precision}{ending}"
        if path.exists():
            files[precision] = json.loads(path.read_text(encoding="utf-8"))
    return files


def main() -> None:
    folder = Path(sys.argv[1])
    results = read_files(folder, ".json")
    print(make_table(results))
    print("Machines:\n")
    print(describe_runners(results))
    print("Memory after generation:\n")
    print(describe_memory(results))
    print("Second pass with pruna's EvaluationAgent:\n")
    print(describe_second_pass(results, read_files(folder, ".pruna.json")))


if __name__ == "__main__":
    main()
