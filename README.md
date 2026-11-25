# SQL under compression

When a small open language model is compressed with [pruna](https://github.com/PrunaAI/pruna),
how many correct SQL answers does it lose, and what does it save in file size, memory and speed?

## Results

Model: Qwen2.5-Coder-1.5B-Instruct. Questions: 200 from the Spider development set.
An answer is correct if its SQL, run on the SQLite database, returns the same result as the reference query.

| Precision | File size | Peak memory | Tokens per second | Seconds per question | Execution accuracy (95 % interval) | Change against f16 |
|---|---|---|---|---|---|---|
| f16 | 3.09 GB | 3.53 GB | 7.9 | 5.2 | 118/200 = 59.0% (52.1% to 65.6%) | baseline |
| q8_0 | 1.65 GB | 2.08 GB | 12.6 | 3.3 | 119/200 = 59.5% (52.6% to 66.1%) | +0.5 points (lost 1, gained 2, p = 1.000) |
| q5_k_m | 1.13 GB | 1.56 GB | 12.1 | 3.1 | 123/200 = 61.5% (54.6% to 68.0%) | +2.5 points (lost 1, gained 6, p = 0.125) |
| q4_k_m | 0.99 GB | 2.03 GB | 20.4 | 2.3 | 114/200 = 57.0% (50.1% to 63.7%) | -2.0 points (lost 9, gained 5, p = 0.424) |

All numbers come from one run: [Actions run 45508785082](https://github.com/zuhalsicim/sql-under-compression/actions/runs/45508785082), 24 to 25 November 2026, commit `f6682bf`.
The four precisions ran one after the other on one GitHub-hosted machine: AMD EPYC 9V45, 2 virtual CPUs, 8.3 GB RAM, no GPU, image `ubuntu26 20261116.202.1`.
The result files are in `results/`.

The last column is paired: each precision is compared with f16 on the same 200 questions, one by one.
`p` is from an exact two-sided sign test on the questions where the two differ: the chance of a split between lost and gained at least this uneven, if both versions were equally good.
The p-values were computed afterwards from the same result files.

What the numbers say, and what they do not say:

- The q4_k_m file is a third of the f16 file. On this machine the compressed versions generated 1.5 to 2.6 times as many tokens per second as f16.
- Accuracy stayed between 57 % and 61.5 %. With 200 questions each interval is about 13.5 points wide. So the run shows no loss from compression, and it cannot rule out a loss of a few points.
- The numbers do not say that q5_k_m is better than f16 or that q4_k_m is worse. Compression changed answers in both directions: q4_k_m lost 9 questions and gained 5. In the paired test no difference is below the usual line of 0.05 (p = 1.000, 0.125 and 0.424), so each of them can come from chance. That is not proof that the versions are equally good.
- Memory does not follow file size. q4_k_m needed more memory than q5_k_m, because llama.cpp kept a second, re-arranged copy of about two thirds of the q4_k_m weights in RAM (see Limits).
- Speed was measured once, on one shared machine. It is not a general statement about these files (see Limits).

## Method

### Model

[`Qwen/Qwen2.5-Coder-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct),
revision `2e1fd397ee46e1388853d2af2c993145b0f1098a`. It has 1.54 billion weights.
It is not gated and its licence is Apache-2.0.

### Compression

pruna 0.3.5 with its `llama_cpp` algorithm (`compress.py`).
The algorithm converts the model to the GGUF format of [llama.cpp](https://github.com/ggml-org/llama.cpp) with 16-bit floats.
Then it quantizes the weights to the wanted precision.
The benchmark sets two options: `llama_cpp_quantization_method` and `llama_cpp_n_gpu_layers = 0`.

| Precision | What it is | Bits per weight, as llama.cpp reports them |
|---|---|---|
| f16 | 16-bit floats. Converted, not quantized. This is the baseline. | 16.00 |
| q8_0 | 8-bit integers. | 8.50 |
| q5_k_m | llama.cpp's "Q5_K - Medium" mix. | 5.80 |
| q4_k_m | llama.cpp's "Q4_K - Medium" mix. | 5.08 |

The original model stores its weights as bfloat16. The f16 file is therefore already a conversion, but not a smaller one.
The original model itself was not evaluated.

### Questions and prompt

The questions come from the development set of [Spider](https://yale-lily.github.io/spider): 1034 questions about 20 SQLite databases.
The benchmark asks 200 of them, chosen at random with a fixed seed (42). Every precision gets the same 200. They cover 18 of the 20 databases.

The prompt has three parts: one fixed instruction, the `CREATE TABLE` statements of the database, and the question.
It contains no example rows and no example answers.
The model answers with at most 256 tokens. Decoding is greedy: the model always takes the most likely next token.
The exact text is built in `sqlbench/prompt.py`.

### What counts as correct

The answer of the model is cut down to its first SQL statement (`sqlbench/sqlclean.py`).
This statement and the reference query both run on the SQLite database (`sqlbench/sqlexec.py`).
The answer is correct if both return the same rows.

- Columns must come in the same order.
- Rows must come in the same order only if the reference query has `ORDER BY`.
- A row that appears twice must appear twice.
- Floats are compared after rounding to 6 decimal places.
- An answer that is empty, that SQLite cannot run, that runs longer than 30 seconds or that returns more than 100,000 rows is wrong.

This measure is called execution accuracy.

### What is measured, and with which tool

Compressed with pruna. Measured by `evaluate.py`, a script in this repository that loads the GGUF file with llama-cpp-python 0.3.36 and nothing else.

- **File size**: the size of the GGUF file that pruna saved.
- **Peak memory**: the largest amount of RAM the `evaluate.py` process held, read from Linux after the last question (`VmHWM`). It includes the model file, because llama.cpp maps the file into memory.
- **Tokens per second**: generated tokens divided by the time llama.cpp spent generating them, from llama.cpp's own counters. Reading the prompt is not included.
- **Seconds per question**: the time of one call, from sending the prompt to the end of the answer, averaged over the questions. See Limits.
- **Execution accuracy**: correct answers divided by questions, with the Wilson 95 % interval.
- **Change against f16**: the same questions are compared one by one. "Lost" are questions that f16 answered correctly and this precision did not. "Gained" is the reverse. `p` is the exact two-sided sign test on these two counts (`sign_test` in `sqlbench/stats.py`). Questions that both versions got right, or both got wrong, do not enter the test. If both versions were equally good, a question where they differ would be lost or gained with the same chance, like a coin toss. `p` is the chance that coin tosses split at least as unevenly as the two counts. For right-or-wrong answers this is the same as the exact McNemar test.

pruna has its own evaluation tools. `evaluate_with_pruna.py` tried them on the same model files in the same run:

- **Accuracy: works.** pruna's `EvaluationAgent` ran the llama.cpp model with a custom metric. It asked all 200 questions again for every precision. Every single question got the same outcome as in `evaluate.py`. So the accuracy column is confirmed by pruna's own tool.
- **Memory: does not work on a CPU.** pruna's memory metrics measure GPU memory. `inference_memory` stopped with the error `Memory metrics does not support device cpu. Must be one of ['cuda'].`
- **Speed: works in part.** pruna's `latency` metric ran and gives milliseconds per call. pruna 0.3.5 has no metric for generated tokens per second.
- **The measuring process must stay small.** A Python process that had imported pruna held 0.80 GB of RAM before any model was loaded. `evaluate.py` held 0.06 GB at that point.

That is why file size, memory and speed come from `evaluate.py` and not from pruna.

### Where it runs

GitHub Actions, `ubuntu-latest`, no GPU. This repository is private, so GitHub gives each job 2 virtual CPUs and 8 GB RAM.

- A push that changes code starts the smoke test: 5 questions, each precision on its own machine.
- The full run is started by hand: 200 questions, all four precisions one after the other on one machine.

The full run uses one machine because the machines differ.
In three smoke tests the same f16 file and the same 5 questions gave 4.3, 7.5 and 7.8 generated tokens per second on three machines with three different CPUs
(runs [45505572223](https://github.com/zuhalsicim/sql-under-compression/actions/runs/45505572223),
[45500240706](https://github.com/zuhalsicim/sql-under-compression/actions/runs/45500240706) and
[45508690256](https://github.com/zuhalsicim/sql-under-compression/actions/runs/45508690256)).
A comparison across machines would mix this with the effect of compression.

## Limits

- **One small model, one dataset, 200 questions.** Each accuracy interval is about 13.5 points wide. A real difference of a few points between two precisions cannot be seen with this sample. The sign test is weak for the same reason: few questions differ. With the 14 differing questions of q4_k_m, the split would have to be 12 to 2 or more uneven to fall below 0.05.
- **Speed is one run on one shared machine.** Other users of the same hardware can slow it down. Within the run the speed was steady: the second pass over the same questions (through pruna) took 5.1, 3.2, 3.1 and 2.4 seconds per question for f16, q8_0, q5_k_m and q4_k_m, against 5.2, 3.3, 3.1 and 2.3 in the first pass. On another machine the numbers will be different, and the order of q8_0 and q5_k_m may be different too.
- **Seconds per question depends on the order of the questions.** The questions are asked in the order of the dataset, database by database. llama.cpp keeps the last prompt in a cache, so it reads the schema once per database and not once per question. In the smoke test every one of the 5 questions is about another database. There f16 needed between 27.7 and 31.7 seconds per question.
- **Peak memory counts the model file that is mapped into memory.** The system can drop such pages when RAM gets short and read them again later. After the last question the memory was split like this (`RssFile` and `RssAnon` in the result files):
  - f16: 3.13 GB model file mapped into memory, 0.40 GB allocated by the process
  - q8_0: 1.68 GB model file mapped into memory, 0.40 GB allocated by the process
  - q5_k_m: 1.16 GB model file mapped into memory, 0.40 GB allocated by the process
  - q4_k_m: 1.00 GB model file mapped into memory, 1.03 GB allocated by the process
- **The extra memory of q4_k_m comes from llama.cpp.** While loading the q4_k_m file, llama.cpp reported `CPU_REPACK model buffer size = 596.53 MiB` next to `CPU_Mapped model buffer size = 927.31 MiB` (`results/q4_k_m.llama_load.txt`). For the other three files it reported only the mapped buffer. This run did not test whether the re-arranged copy is also the reason for the higher speed of q4_k_m.
- **The f16 file is the baseline, not the original model.** See Compression above.
- **Execution accuracy is not a perfect judge.** A wrong query can return the right rows by chance. A right query with another column order counts as wrong. The reference queries of Spider are used as they are.
- **The prompt shows the schema only.** The model does not see any stored values.
- **Answers may depend on the CPU.** In the smoke tests each file gave the same 5 answers on every machine it ran on. Whether all 200 answers would be the same on another CPU was not tested.

## How to reproduce

On GitHub: put this repository into your own account, open the Actions tab, choose "benchmark" and press "Run workflow".
The defaults start the full run. It took 1 hour and 49 minutes. The table appears in the summary of the run. The result files are in the artifact `results`.

On a Linux machine with Python 3.12:

```
pip install -r requirements.txt
python fetch_spider.py data
python compress.py q4_k_m model
python evaluate.py model data/spider 200 output/q4_k_m.json
python evaluate_with_pruna.py model data/spider output/q4_k_m.json output/q4_k_m.pruna.json
python make_table.py output
```

A new run writes to `output/`. The folder `results/` holds the published full run and stays as it is.
`python make_table.py results` prints the table above again.

These steps were only ever run on the GitHub machines described above.

The unit tests need only Python and pytest and run on any system:

```
pip install -r requirements-dev.txt
pytest
```

## Files

| File | What it does |
|---|---|
| `sqlbench/settings.py` | Every fixed choice: model, data, seed, precisions. |
| `sqlbench/sampling.py` | Chooses the 200 questions. |
| `sqlbench/prompt.py` | Reads the schema of a database and builds the prompt. |
| `sqlbench/sqlclean.py` | Cuts the answer of the model down to one SQL statement. |
| `sqlbench/sqlexec.py` | Runs SQL and decides whether two results are the same. |
| `sqlbench/stats.py` | The Wilson interval and the sign test. |
| `sqlbench/report.py` | Builds the results table from the result files. |
| `fetch_spider.py` | Step 1: downloads Spider. |
| `compress.py` | Step 2: compresses the model with pruna. |
| `evaluate.py` | Step 3: asks the questions and measures. |
| `evaluate_with_pruna.py` | Step 4: scores a second time with pruna's EvaluationAgent. |
| `make_table.py` | Step 5: prints the table. |
| `tests/` | Unit tests for the code in `sqlbench/`. |
| `requirements.txt`, `requirements-dev.txt` | The packages for the GitHub machine, and for the unit tests. |
| `.github/workflows/benchmark.yml` | Runs the unit tests and the five steps on GitHub. |
| `results/` | The result files of the full run behind the table, and the package versions it used. |

## Sources and licences

- Code in this repository: MIT, see `LICENSE`.
- Model: Qwen2.5-Coder-1.5B-Instruct by Alibaba Cloud, Apache-2.0. <https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct>
- Data: Spider by Yu et al., 2018 (<https://arxiv.org/abs/1809.08887>), CC BY-SA 4.0.
  Downloaded from the Hugging Face dataset repository [`xlangai/spider`](https://huggingface.co/datasets/xlangai/spider),
  revision `6232cc3fad6d54c62b3ba23a364083a98ff36a17`, file `data/spider.zip`.
  This is the last revision that holds the SQLite databases. Later revisions hold only the questions.
  This repository contains no Spider data. The result files hold the SQL the model wrote, but not the questions and not the reference queries.
- pruna 0.3.5: Apache-2.0. <https://github.com/PrunaAI/pruna>
- llama.cpp and llama-cpp-python 0.3.36: MIT.
