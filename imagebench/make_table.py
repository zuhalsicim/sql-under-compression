"""Print the table of the image benchmark from the files of one run.

Usage: python -m imagebench.make_table output-image
"""

import json
import sys
from pathlib import Path

from imagebench.measure import make_table
from imagebench.settings import RUN


def main() -> None:
    out_dir = Path(sys.argv[1])
    results = {}
    for name in RUN:
        path = out_dir / f"{name}.json"
        if path.exists():
            results[name] = json.loads(path.read_text(encoding="utf-8"))
    scores_path = out_dir / "scores.json"
    scores = json.loads(scores_path.read_text(encoding="utf-8")) if scores_path.exists() else {}

    print(make_table(results, scores))
    for name, result in results.items():
        if "error" not in result:
            runner = result["runner"]
            print(
                f"Machine: runner \"{runner['name']}\", {runner['cpu_model']}, {runner['cpu_count']} vCPU, "
                f"{runner['memory_bytes'] / 1e9:.1f} GB RAM, {runner['image']}"
            )
            print(f"Image settings: {result['image_settings']}")
            break
    print()
    for name, result in results.items():
        if "error" not in result:
            notes = [f"warm-up {result['warm_up_seconds']:.1f} s"]
            if "smash_seconds" in result:
                notes.append(f"pruna smash {result['smash_seconds']:.1f} s")
            if "saved_size_bytes" in result:
                notes.append(f"saved size {result['saved_size_bytes'] / 1e9:.2f} GB")
            else:
                notes.append(f"could not be saved: {result.get('saved_size_error')}")
            blocked = sum(row["blocked_by_safety_checker"] for row in result["images"])
            notes.append(f"{blocked} images blocked by the safety checker")
            print(f"- {name}: " + ", ".join(notes))


if __name__ == "__main__":
    main()
