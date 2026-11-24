"""Step 3: ask the model the questions, check its SQL, and measure memory and speed.

Usage: python evaluate.py model data/spider 200 results/q4_k_m.json

This script loads the GGUF file with llama-cpp-python only. It does not import
pruna or PyTorch, so the measured memory is the memory of the model and not
of the tools that made it. It runs on Linux only (it reads /proc).
"""

import json
import os
import sys
import time
from pathlib import Path

import llama_cpp

from sqlbench.prompt import build_prompt, read_schema
from sqlbench.sampling import choose_indices
from sqlbench.settings import MAX_TOKENS, N_CTX, SEED
from sqlbench.sqlclean import extract_sql
from sqlbench.sqlexec import check_answer, run_query
from sqlbench.stats import wilson_interval


def memory_status() -> dict:
    """Read from Linux how much RAM this process holds, in bytes.

    VmHWM    the most RAM the process has held so far (peak resident set size)
    VmRSS    the RAM it holds now
    RssFile  the part of VmRSS that is a file mapped into memory, here the model file
    RssAnon  the part of VmRSS that the process allocated itself
    """
    status = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        name, _, value = line.partition(":")
        if name in ("VmHWM", "VmRSS", "RssFile", "RssAnon"):
            status[name] = int(value.split()[0]) * 1024  # the line looks like "VmHWM:  123456 kB"
    return status


def runner_info() -> dict:
    """Describe the machine, because speed depends on it."""
    cpu_lines = Path("/proc/cpuinfo").read_text().splitlines()
    cpu_model = next(line.split(":", 1)[1].strip() for line in cpu_lines if line.startswith("model name"))
    memory_kilobytes = int(Path("/proc/meminfo").read_text().split()[1])  # first line is "MemTotal: <n> kB"
    return {
        "name": os.environ.get("RUNNER_NAME", "unknown"),  # GitHub gives every machine of a run its own name
        "cpu_model": cpu_model,
        "cpu_count": os.cpu_count(),
        "memory_bytes": memory_kilobytes * 1024,
        "image": f"{os.environ.get('ImageOS', 'unknown image')} {os.environ.get('ImageVersion', '')}".strip(),
    }


def main() -> None:
    model_dir, spider_dir = Path(sys.argv[1]), Path(sys.argv[2])
    question_count, result_path = int(sys.argv[3]), Path(sys.argv[4])
    model_path = model_dir / "model.gguf"

    all_questions = json.loads((spider_dir / "dev.json").read_text(encoding="utf-8"))
    indices = choose_indices(len(all_questions), question_count, SEED)

    def database_of(index: int) -> Path:
        name = all_questions[index]["db_id"]
        return spider_dir / "database" / name / f"{name}.sqlite"

    # Run every reference query once before the model is loaded.
    # A broken reference query then stops the script here and not after an hour.
    for index in indices:
        run_query(database_of(index), all_questions[index]["query"])
    memory_before_model = memory_status()

    threads = os.cpu_count()
    llm = llama_cpp.Llama(
        model_path=str(model_path),
        n_ctx=N_CTX,
        n_threads=threads,
        n_threads_batch=threads,
        n_gpu_layers=0,
        verbose=False,
    )
    llm("SELECT", max_tokens=1)  # one untimed call, so that the first question does not pay for loading

    # Part 1: generate all answers. Nothing the model wrote is run yet,
    # so the memory measured after this loop is the memory of generation alone.
    rows = []
    for index in indices:
        prompt = build_prompt(read_schema(database_of(index)), all_questions[index]["question"])
        llama_cpp.llama_perf_context_reset(llm.ctx)  # set llama.cpp's own counters to zero
        start = time.perf_counter()
        output = llm(prompt, max_tokens=MAX_TOKENS, temperature=0.0)  # temperature 0 = always the most likely token
        seconds = time.perf_counter() - start
        counters = llama_cpp.llama_perf_context(llm.ctx)
        rows.append(
            {
                "index": index,
                "db_id": all_questions[index]["db_id"],
                "answer": output["choices"][0]["text"],
                "finish_reason": output["choices"][0]["finish_reason"],
                "seconds": seconds,
                "prompt_tokens": output["usage"]["prompt_tokens"],
                "generated_tokens": output["usage"]["completion_tokens"],
                # llama.cpp's own counters. Reading the prompt and generating are
                # counted apart. Prompt tokens that were still in its cache from
                # the question before are not read again and not counted.
                "prompt_tokens_read": counters.n_p_eval,
                "prompt_seconds": counters.t_p_eval_ms / 1000,
                "generation_steps": counters.n_eval,  # one step makes one token
                "generation_seconds": counters.t_eval_ms / 1000,
            }
        )
        print(f"question {index}: {seconds:.1f} s, {rows[-1]['generated_tokens']} tokens", flush=True)
    memory_after_generation = memory_status()

    # Part 2: run the generated SQL and compare it with the reference result.
    for row in rows:
        row["sql"] = extract_sql(row["answer"])
        row["correct"], row["note"] = check_answer(
            database_of(row["index"]), row["sql"], all_questions[row["index"]]["query"]
        )

    correct = sum(row["correct"] for row in rows)
    low, high = wilson_interval(correct, len(rows))
    result = {
        "compression": json.loads((model_dir / "compression.json").read_text(encoding="utf-8")),
        "gguf_file_type": llm.metadata.get("general.file_type"),  # llama.cpp's number for the precision
        "file_size_bytes": model_path.stat().st_size,
        "questions": len(rows),
        "seed": SEED,
        "correct": correct,
        "accuracy": correct / len(rows),
        "accuracy_interval_95": [low, high],
        "peak_memory_bytes": memory_after_generation["VmHWM"],
        "peak_memory_before_model_bytes": memory_before_model["VmHWM"],
        "memory_after_generation": memory_after_generation,
        "mean_seconds_per_question": sum(row["seconds"] for row in rows) / len(rows),
        "tokens_per_second": sum(row["generation_steps"] for row in rows)
        / sum(row["generation_seconds"] for row in rows),
        "prompt_tokens_per_second": sum(row["prompt_tokens_read"] for row in rows)
        / sum(row["prompt_seconds"] for row in rows),
        "answers_cut_at_max_tokens": sum(row["finish_reason"] == "length" for row in rows),
        "settings": {"n_ctx": N_CTX, "max_tokens": MAX_TOKENS, "threads": threads, "temperature": 0.0},
        "runner": runner_info(),
        "llama_cpp_python_version": llama_cpp.__version__,
        "llama_cpp_system_info": llama_cpp.llama_print_system_info().decode(),
        "python_version": sys.version.split()[0],
        "per_question": rows,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_question"}, indent=2))


if __name__ == "__main__":
    main()
