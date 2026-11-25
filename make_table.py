"""Step 5: read the result files of one run and print the results table.

Usage: python make_table.py output                     (the files of a new run)
       python make_table.py results                    (the published run with 200 questions)
       python make_table.py results results-full-dev   (the table in the README: size, memory and
                                                        speed from the first folder, accuracy
                                                        from the second)
"""

import json
import sys
from pathlib import Path

from sqlbench.report import (
    describe_agreement,
    describe_differences,
    describe_memory,
    describe_runners,
    describe_second_pass,
    make_table,
)
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
    accuracy_folder = Path(sys.argv[2]) if len(sys.argv) > 2 else folder
    results = read_files(folder, ".json")
    accuracy_results = read_files(accuracy_folder, ".json")

    print(make_table(results, accuracy_results))
    print(f"Machines ({folder}):\n")
    print(describe_runners(results))
    print(f"Memory after generation ({folder}):\n")
    print(describe_memory(results))
    print(f"Second pass with pruna's EvaluationAgent ({folder}):\n")
    print(describe_second_pass(results, read_files(folder, ".pruna.json")))
    print(f"Answers compared with f16 ({accuracy_folder}):\n")
    print(describe_differences(accuracy_results))
    if accuracy_folder != folder:
        print(f"Machines ({accuracy_folder}):\n")
        print(describe_runners(accuracy_results))
        print(f"The same questions in both runs ({folder} and {accuracy_folder}):\n")
        print(describe_agreement(results, accuracy_results))


if __name__ == "__main__":
    main()
