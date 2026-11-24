"""Step 5: read the result files of one run and print the results table.

Usage: python make_table.py results
"""

import json
import sys
from pathlib import Path

from sqlbench.report import describe_runners, make_table
from sqlbench.settings import PRECISIONS


def main() -> None:
    folder = Path(sys.argv[1])
    results = {}
    for precision in PRECISIONS:
        path = folder / f"{precision}.json"
        if path.exists():
            results[precision] = json.loads(path.read_text(encoding="utf-8"))
    print(make_table(results))
    print("Machines:\n")
    print(describe_runners(results))


if __name__ == "__main__":
    main()
