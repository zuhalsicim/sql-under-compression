"""Step 4: compute the accuracy a second time, now with pruna's own evaluation tools.

Usage: python evaluate_with_pruna.py model data/spider results/q4_k_m.json results/q4_k_m.pruna.json

evaluate.py already measured everything. This script asks the same questions
again through pruna's EvaluationAgent with a custom metric, and compares the
outcome with evaluate.py question by question. It also tries pruna's memory
metric and pruna's timing metric and records either the value or the error.
"""

import json
import os
import sys
import time
from pathlib import Path

import pruna
from pruna import PrunaModel
from pruna.data.pruna_datamodule import PrunaDataModule
from pruna.evaluation.evaluation_agent import EvaluationAgent
from pruna.evaluation.metrics import LatencyMetric
from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.result import MetricResult

from evaluate import memory_status
from sqlbench.prompt import build_prompt, read_schema
from sqlbench.settings import MAX_TOKENS, N_CTX, SMOKE_QUESTIONS
from sqlbench.sqlclean import extract_sql
from sqlbench.sqlexec import check_answer


class ExecutionAccuracy(StatefulMetric):
    """A custom pruna metric: the share of questions answered with correct SQL."""

    metric_name = "execution_accuracy"
    call_type = "single"  # one model is scored on its own, not compared with another model

    def __init__(self) -> None:
        super().__init__(device="cpu")
        self.add_state("outcomes", [])  # pruna empties this list after compute()
        self.last_outcomes = []  # a copy that stays, for the comparison with evaluate.py

    def update(self, prompt: str, reference: tuple, output: dict) -> None:
        """pruna calls this after every question with the input, the reference and the model output."""
        database, gold_sql = reference
        sql = extract_sql(output["choices"][0]["text"])
        self.outcomes.append(check_answer(database, sql, gold_sql)[0])

    def compute(self) -> MetricResult:
        self.last_outcomes = list(self.outcomes)
        share = sum(self.outcomes) / len(self.outcomes)
        return MetricResult(self.metric_name, {}, share, higher_is_better=True)


def first_item(batch: list) -> tuple:
    """pruna hands the model one batch at a time. Here a batch is one question."""
    return batch[0]


def attempt(action) -> dict:
    """Run one check. Record its value, or its error instead of stopping."""
    try:
        return {"works": True, "value": action()}
    except Exception as error:
        return {"works": False, "error": f"{type(error).__name__}: {error}"}


def main() -> None:
    model_dir, spider_dir = Path(sys.argv[1]), Path(sys.argv[2])
    result_path, findings_path = Path(sys.argv[3]), Path(sys.argv[4])
    memory_after_imports = memory_status()["VmHWM"]

    # The same questions, prompts and reference queries as in evaluate.py, in the same order.
    all_questions = json.loads((spider_dir / "dev.json").read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    rows = result["per_question"]
    items = []
    for row in rows:
        question = all_questions[row["index"]]
        database = spider_dir / "database" / question["db_id"] / f"{question['db_id']}.sqlite"
        prompt = build_prompt(read_schema(database), question["question"])
        items.append((prompt, (str(database), question["query"])))
    datamodule = PrunaDataModule(items, items, items, collate_fn=first_item, dataloader_args={"batch_size": 1})

    # Load the saved model through pruna. Extra arguments go on to llama-cpp-python.
    threads = os.cpu_count()
    model = PrunaModel.from_pretrained(
        str(model_dir), n_ctx=N_CTX, n_threads=threads, n_threads_batch=threads, n_gpu_layers=0
    )
    # pruna calls model(prompt). These arguments are added to every call.
    model.inference_handler.model_args.update({"max_tokens": MAX_TOKENS, "temperature": 0.0})

    def run(metric) -> float:
        agent = EvaluationAgent(request=[metric], datamodule=datamodule, device="cpu")
        return agent.evaluate(model)[0].result

    # 1. Execution accuracy on all questions, as a custom metric inside pruna's EvaluationAgent.
    # This repeats the work of evaluate.py on the same machine, so the time it
    # takes also shows how steady the speed of the machine is.
    accuracy_metric = ExecutionAccuracy()
    start = time.perf_counter()
    accuracy = attempt(lambda: run(accuracy_metric))
    seconds_with_pruna = time.perf_counter() - start
    expected = [row["correct"] for row in rows]
    different = sum(a != b for a, b in zip(accuracy_metric.last_outcomes, expected))

    # 2. pruna's memory metric.
    memory = attempt(lambda: run("inference_memory"))

    # 3. pruna's timing metric on the first few questions: milliseconds per call, after one warm-up call.
    few = min(SMOKE_QUESTIONS, len(rows))
    latency = attempt(lambda: run(LatencyMetric(n_iterations=few, n_warmup_iterations=1, device="cpu")))

    findings = {
        "pruna_version": pruna.__version__,
        "questions": len(rows),
        "execution_accuracy_with_pruna": accuracy,
        "execution_accuracy_with_evaluate_py": sum(expected) / len(expected),
        "questions_scored_by_pruna": len(accuracy_metric.last_outcomes),
        "questions_with_a_different_outcome": different,
        "seconds_for_all_questions_with_pruna": seconds_with_pruna,
        "seconds_for_all_questions_with_evaluate_py": result["generation_seconds_all_questions"]
        + result["scoring_seconds_all_questions"],
        "inference_memory_with_pruna": memory,
        "latency_ms_with_pruna": latency,
        "latency_ms_with_evaluate_py": 1000 * sum(row["seconds"] for row in rows[:few]) / few,
        "latency_questions": few,
        "memory_after_importing_pruna_bytes": memory_after_imports,
    }
    findings_path.write_text(json.dumps(findings, indent=2), encoding="utf-8")
    print(json.dumps(findings, indent=2))


if __name__ == "__main__":
    main()
