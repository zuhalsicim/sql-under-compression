"""Compare two runs: did a precision give the same answers, and was the model file the same?

Usage: python compare_runs.py results results-q4-repeat/200-questions-machine-1
"""

import sys
from pathlib import Path

from make_table import read_files
from sqlbench.report import describe_agreement, describe_model_files, describe_runners


def main() -> None:
    first_folder, second_folder = Path(sys.argv[1]), Path(sys.argv[2])
    first, second = read_files(first_folder, ".json"), read_files(second_folder, ".json")
    print(f"{first_folder} against {second_folder}\n")
    print("Answers:\n")
    print(describe_agreement(first, second))
    print("Model files:\n")
    print(describe_model_files(first, second))
    print(f"Machines ({first_folder}):\n")
    print(describe_runners(first))
    print(f"Machines ({second_folder}):\n")
    print(describe_runners(second))


if __name__ == "__main__":
    main()
