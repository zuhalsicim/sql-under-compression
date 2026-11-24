"""Step 2: compress the model with pruna and save it as a GGUF file.

Usage: python compress.py q4_k_m model

pruna's "llama_cpp" algorithm converts the Hugging Face model to the GGUF
format of llama.cpp (16-bit floats) and then quantizes it to the wanted
precision. For "f16" it only converts. Afterwards the folder holds
    model/model.gguf          the model
    model/smash_config.json   how pruna made it
"""

import json
import sys
import time
from pathlib import Path

import pruna
from pruna import SmashConfig, smash
from transformers import AutoModelForCausalLM, AutoTokenizer

from sqlbench.settings import MODEL_ID, MODEL_REVISION, PRECISIONS


def main() -> None:
    precision, target = sys.argv[1], Path(sys.argv[2])
    if precision not in PRECISIONS:
        raise SystemExit(f"precision must be one of {PRECISIONS}")

    # dtype="auto" keeps the 16-bit numbers of the download. The default
    # would turn them into 32-bit numbers and need twice the memory.
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)

    smash_config = SmashConfig(
        {"llama_cpp": {"quantization_method": precision, "n_gpu_layers": 0}},
        device="cpu",
    )
    # llama.cpp stores the tokenizer inside the GGUF file, so pruna needs it too.
    smash_config.add_tokenizer(tokenizer)

    start = time.perf_counter()
    smashed_model = smash(model=model, smash_config=smash_config)
    seconds = time.perf_counter() - start
    smashed_model.save_pretrained(str(target))

    info = {
        "precision": precision,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "pruna_version": pruna.__version__,
        "compression_seconds": round(seconds, 1),
        "file_size_bytes": (target / "model.gguf").stat().st_size,
    }
    (target / "compression.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
