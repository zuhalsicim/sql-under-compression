"""Step 4: try pruna's own evaluation tools on the llama.cpp model and write down what happens.

Usage: python pruna_check.py model data/spider results/q4_k_m.json results/q4_k_m.pruna_check.json

The numbers in the results table do not come from this script. The script is
the evidence for one sentence in the README: why the benchmark measures with
evaluate.py and not with pruna's EvaluationAgent. It asks the first few
questions that evaluate.py asked, this time through pruna, and records for
each pruna tool either its value or its error.
"""

import json
import os
import sys
from pathlib import Path

import pruna
from pruna import PrunaModel
from pruna.data.pruna_datamodule import PrunaDataModule
from pruna.evaluation.evaluation_agent import EvaluationAgent
from pruna.evaluation.metrics import LatencyMetric
from pruna.evaluation.metrics.metric_stateful import StatefulMetric
from pruna.evaluation.metrics.result import MetricResult

from evaluate import peak_memory_bytes
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
        self.add_state("outcomes", [])

    def update(self, prompt: str, reference: tuple, output: dict) -> None:
        """pruna calls this after every question with the input, the reference and the model output."""
        database, gold_sql = reference
        sql = extract_sql(output["choices"][0]["text"])
        self.outcomes.append(check_answer(database, sql, gold_sql)[0])

    def compute(self) -> MetricResult:
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
    memory_after_imports = peak_memory_bytes()

    # The same questions, prompts and reference queries as in evaluate.py.
    all_questions = json.loads((spider_dir / "dev.json").read_text(encoding="utf-8"))
    rows = json.loads(result_path.read_text(encoding="utf-8"))["per_question"][:SMOKE_QUESTIONS]
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

    findings = {
        "pruna_version": pruna.__version__,
        "questions": len(items),
        "memory_after_importing_pruna_bytes": memory_after_imports,
        # 1. A custom accuracy metric inside pruna's EvaluationAgent.
        "execution_accuracy_with_pruna": attempt(lambda: run(ExecutionAccuracy())),
        "execution_accuracy_with_evaluate_py": sum(row["correct"] for row in rows) / len(rows),
        # 2. pruna's memory metric.
        "inference_memory_with_pruna": attempt(lambda: run("inference_memory")),
        # 3. pruna's timing metric: milliseconds per call, with one warm-up call.
        "latency_ms_with_pruna": attempt(
            lambda: run(LatencyMetric(n_iterations=len(items), n_warmup_iterations=1, device="cpu"))
        ),
        "latency_ms_with_evaluate_py": 1000 * sum(row["seconds"] for row in rows) / len(rows),
    }
    findings_path.write_text(json.dumps(findings, indent=2), encoding="utf-8")
    print(json.dumps(findings, indent=2))


if __name__ == "__main__":
    main()
