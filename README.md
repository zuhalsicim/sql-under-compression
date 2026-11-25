# SQL under compression

When a small open language model is compressed with [pruna](https://github.com/PrunaAI/pruna),
how many correct SQL answers does it lose, and what does it save in file size, memory and speed?

This repository holds a benchmark that answers this with measured numbers. It also holds one small experiment with an image model, further down.

## Results

Model: Qwen2.5-Coder-1.5B-Instruct. Questions: the Spider development set.
An answer is correct if its SQL, run on the SQLite database, returns the same result as the reference query.

Every answer is generated greedily: temperature 0, the model always takes the most likely next token (the setting is written in every result file).
The last column compares each precision with f16 on the same questions, one by one. Its `p` comes from an exact two-sided sign test on the questions whose outcome differs.

| Precision | File size | Peak memory | Tokens per second | Seconds per question | Execution accuracy (95 % interval) | Change against f16 |
|---|---|---|---|---|---|---|
| f16 | 3.09 GB | 3.53 GB | 7.9 | 5.2 | 647/1034 = 62.6% (59.6% to 65.5%) | baseline |
| q8_0 | 1.65 GB | 2.08 GB | 12.6 | 3.3 | 646/1034 = 62.5% (59.5% to 65.4%) | -0.1 points (lost 6, gained 5, p = 1.000) |
| q5_k_m | 1.13 GB | 1.56 GB | 12.1 | 3.1 | 653/1034 = 63.2% (60.2% to 66.0%) | +0.6 points (lost 14, gained 20, p = 0.392) |
| q4_k_m | 0.99 GB | 2.03 GB | 20.4 | 2.3 | 614/1034 = 59.4% (56.4% to 62.3%) | -3.2 points (lost 59, gained 26, p < 0.001) |

The table comes from two runs. `python make_table.py results results-full-dev` prints it.

- **File size, peak memory, tokens per second, seconds per question**: [run 45508785082](https://github.com/zuhalsicim/sql-under-compression/actions/runs/45508785082), 24 to 25 November 2026, commit `f6682bf`.
  200 questions, all four precisions one after the other on one GitHub-hosted machine: AMD EPYC 9V45, 2 virtual CPUs, 8.3 GB RAM, no GPU, image `ubuntu26 20261116.202.1`. The result files are in `results/`.
- **Execution accuracy and change against f16**: [run 45529640714](https://github.com/zuhalsicim/sql-under-compression/actions/runs/45529640714), 25 November 2026, commit `1df0342`.
  All 1034 questions of the development set, each precision on its own machine. The result files are in `results-full-dev/`.

More on the last column: the comparison uses the same 1034 questions for every precision. "Lost" are questions that f16 answered correctly and this precision did not, "gained" is the reverse.
`p` is the chance of a split between lost and gained at least this uneven, if both versions were equally good.

What the numbers say, and what they do not say:

- **Size and speed.** The q4_k_m file is a third of the f16 file. On one machine the compressed versions generated 1.5 to 2.6 times as many tokens per second as f16.
- **8 bits and the 5-bit mix cost no accuracy that this run can see.** q8_0 answered 646 of 1034 questions correctly and q5_k_m 653, against 647 for f16. Lost and gained questions are close to balanced: 6 to 5, and 14 to 20.
- **The 4-bit mix costs about 3 points.** q4_k_m answered 614 correctly. It lost 59 questions that f16 got right and gained 26. Fair coin tosses give a split this uneven less than once in a thousand tries (p < 0.001).
- **What the run could have seen.** Each interval is about 6 points wide. The paired test is finer. With the number of answers that really changed, a net loss of 9 questions (0.9 points) for q8_0, of 14 questions (1.4 points) for q5_k_m and of 21 questions (2.0 points) for q4_k_m would have fallen below 0.05. A smaller loss can hide in this run. So "no loss that this run can see" does not mean "no loss".
- **The answers of q4_k_m depend a little on which questions came before.** Of the 200 questions that both runs asked, q4_k_m gave 11 another answer text and 2 another outcome in the accuracy run. The other three precisions repeated all 200 answers exactly. Three more runs of q4_k_m alone looked for the cause (`results-q4-repeat/`). With the same questions in the same order, q4_k_m gave exactly the same answers again: 200 of 200 on two more machines, and 1034 of 1034. So the machine does not change the answers, and the model file, which is built anew in every run, does not either. The order of the questions does. The loss of 33 questions was measured with the same order for f16 and q4_k_m, and the repeat gave 614 of 1034 again. Why the order matters for q4_k_m was not measured (see Limits).
- **Memory does not follow file size.** q4_k_m needed more memory than q5_k_m, because llama.cpp kept a second, re-arranged copy of about two thirds of the q4_k_m weights in RAM (see Limits).
- **Speed was measured once, on one shared machine.** It is not a general statement about these files (see Limits).

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
The accuracy run asks all 1034. The run for size, memory and speed asks 200 of them, chosen at random with a fixed seed (42); these 200 cover 18 of the 20 databases. Within a run every precision gets the same questions.

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

pruna has its own evaluation tools. `evaluate_with_pruna.py` tried them on the same model files in the 200-question run:

- **Accuracy: works.** pruna's `EvaluationAgent` ran the llama.cpp model with a custom metric. It asked all 200 questions again for every precision. Every single question got the same outcome as in `evaluate.py`. So pruna's own tool confirms how `evaluate.py` scores. The accuracy run with 1034 questions used `evaluate.py` alone.
- **Memory: does not work on a CPU.** pruna's memory metrics measure GPU memory. `inference_memory` stopped with the error `Memory metrics does not support device cpu. Must be one of ['cuda'].`
- **Speed: works in part.** pruna's `latency` metric ran and gives milliseconds per call. pruna 0.3.5 has no metric for generated tokens per second.
- **The measuring process must stay small.** A Python process that had imported pruna held 0.80 GB of RAM before any model was loaded. `evaluate.py` held 0.06 GB at that point.

That is why file size, memory and speed come from `evaluate.py` and not from pruna.

### Where it runs

GitHub Actions, `ubuntu-latest`, no GPU. This repository is private, so GitHub gives each job 2 virtual CPUs and 8 GB RAM.

- A push that changes code starts the smoke test: 5 questions, each precision on its own machine.
- The run for size, memory and speed is started by hand: 200 questions, all four precisions one after the other on one machine, with the second pass through pruna.
- The accuracy run is started by hand: all 1034 questions, each precision on its own machine, without the second pass.

The run for speed uses one machine because the machines differ.
In three smoke tests the same f16 file and the same 5 questions gave 4.3, 7.5 and 7.8 generated tokens per second on three machines with three different CPUs
(runs [45505572223](https://github.com/zuhalsicim/sql-under-compression/actions/runs/45505572223),
[45500240706](https://github.com/zuhalsicim/sql-under-compression/actions/runs/45500240706) and
[45508690256](https://github.com/zuhalsicim/sql-under-compression/actions/runs/45508690256)).
A comparison across machines would mix this with the effect of compression.

## Limits

- **One small model, one dataset.** With 1034 questions each accuracy interval is about 6 points wide. The sign test sees a difference only if the differing questions split unevenly enough: at least 10 to 1 for the 11 differing questions of q8_0, at least 24 to 10 for the 34 of q5_k_m, at least 53 to 32 for the 85 of q4_k_m. So a loss of about 1 point for q8_0, or of 1.4 points for q5_k_m, could not have been seen. The observed split of q4_k_m was 59 to 26.
- **The first run had only 200 questions and could not see the loss of q4_k_m.** There f16, q8_0, q5_k_m and q4_k_m answered 118, 119, 123 and 114 of 200 correctly, and no difference was below 0.05 (p = 1.000, 0.125 and 0.424). `python make_table.py results` prints that table.
- **q4_k_m gives slightly different answers when the order of the questions changes.** The 200 questions of the first run are part of the 1034. f16, q8_0 and q5_k_m gave all 200 the same answer text in both runs. q4_k_m gave 189 the same text and 198 the same outcome. Three more runs with q4_k_m alone tested why (`results-q4-repeat/`, compared with `compare_runs.py`):
  - 200 questions on an AMD EPYC 7763: 200 of 200 answers are the same as in the first run, which ran on an AMD EPYC 9V45.
  - 200 questions on an AMD EPYC 9V74: 200 of 200 are the same.
  - 1034 questions on an AMD EPYC 9V74: 1034 of 1034 answers are the same as in the accuracy run (another machine, also an EPYC 9V74). Against the 200-question runs it shows 189 and 198 of 200 again.

  What this shows: with the same order the answers repeat exactly, on three kinds of CPU for the 200 questions. The 1034 questions were repeated only on the same kind of CPU. With another order 11 of 200 answers change, on the same kind of CPU.
  What it does not show: whether the weights in the file are identical from run to run. The SHA-256 of the file was different in all three runs, although the size was the same. pruna converts from a temporary folder, and the file stores the random name of that folder as the model name (three different names in the logs). A checksum of the whole file can therefore never be the same twice. The answers were the same anyway.
  A guess about the cause, not measured: llama.cpp keeps the last prompt in a cache, so the order of the questions decides how many tokens it reads in one batch, and the re-arranged copy of the q4_k_m weights (see the limit on memory below) may be computed with slightly different rounding for another batch size. The other three precisions have no such copy and repeated exactly.
  For the table this means: the 3 points of q4_k_m are reproducible for this order of questions. With another order a few outcomes change (2 of 200 here), so the exact size of the loss is a little less certain than the interval says.
- **Speed is one run on one shared machine.** Other users of the same hardware can slow it down. Within the run the speed was steady: the second pass over the same questions (through pruna) took 5.1, 3.2, 3.1 and 2.4 seconds per question for f16, q8_0, q5_k_m and q4_k_m, against 5.2, 3.3, 3.1 and 2.3 in the first pass. On another machine the numbers will be different, and the order of q8_0 and q5_k_m may be different too.
- **Seconds per question depends on the order of the questions.** The questions are asked in the order of the dataset, database by database. llama.cpp keeps the last prompt in a cache, so it reads the schema once per database and not once per question. In the smoke test every one of the 5 questions is about another database. There f16 needed between 27.7 and 31.7 seconds per question.
- **Peak memory counts the model file that is mapped into memory.** The system can drop such pages when RAM gets short and read them again later. After the last question the memory was split like this (`RssFile` and `RssAnon` in the result files):
  - f16: 3.13 GB model file mapped into memory, 0.40 GB allocated by the process
  - q8_0: 1.68 GB model file mapped into memory, 0.40 GB allocated by the process
  - q5_k_m: 1.16 GB model file mapped into memory, 0.40 GB allocated by the process
  - q4_k_m: 1.00 GB model file mapped into memory, 1.03 GB allocated by the process
- **The extra memory of q4_k_m comes from llama.cpp.** While loading the q4_k_m file, llama.cpp reported `CPU_REPACK model buffer size = 596.53 MiB` next to `CPU_Mapped model buffer size = 927.31 MiB` (`results/q4_k_m.llama_load.txt`). For the other three files it reported only the mapped buffer. This run did not test whether the re-arranged copy is also the reason for the higher speed of q4_k_m.
- **The f16 file is the baseline, not the original model.** See Compression above.
- **The model has very probably seen these questions before.** Spider is a public dataset from 2018, and its development set was very probably part of the model's training data. This benchmark did not check that. It can raise all four accuracies, so the absolute figures are not a measure of text-to-SQL skill. The point of the run is the comparison between precisions of the same model.
- **Execution accuracy is not a perfect judge.** A wrong query can return the right rows by chance. A right query with another column order counts as wrong. The reference queries of Spider are used as they are.
- **The prompt shows the schema only.** The model does not see any stored values.

## A second, small experiment: one image model

The same question for one image model: what does pruna change in speed, memory and image quality?
It ran on the same 2-CPU GitHub machines. There was no GPU. So it shows the method, and what pruna does on a CPU. It does not show the speeds a GPU user gets.

Model: [`nota-ai/bk-sdm-base`](https://huggingface.co/nota-ai/bk-sdm-base), a Stable Diffusion v1.4 with a smaller UNet. 256 x 256 pixels, 15 steps.
12 prompts from PartiPrompts. Every setting gets the same prompts and the same seeds.

| Setting | Seconds per image | Speed against base | Spread between repeats | Peak memory | CLIP score | SSIM to base image | LPIPS to base image |
|---|---|---|---|---|---|---|---|
| base | 41.7 | 1.00 x | 4.9% | 4.19 GB | 22.17 |  |  |
| deepcache_2 | 21.2 | 1.97 x | 5.7% | 4.21 GB | 23.06 | 0.412 | 0.508 |
| deepcache_3 | 18.6 | 2.25 x | 1.7% | 4.23 GB | 22.06 | 0.436 | 0.463 |

All numbers come from one run on one machine: [Actions run 45552987716](https://github.com/zuhalsicim/sql-under-compression/actions/runs/45552987716), commit `a77364d`, AMD EPYC 9V45, 2 virtual CPUs, 8.3 GB RAM.
The files are in `results-image/without-safety-checker/`. `python -m imagebench.make_table results-image/without-safety-checker base deepcache_2 deepcache_3` prints the table.

- **The speed-up is no finding on its own, its price is.** `deepcache` reuses part of the UNet's work from the step before: every second step (`deepcache_2`) or two of three steps (`deepcache_3`). That roughly halves the work by construction, and the time per image roughly halves with it, also on a CPU. The finding is what it costs: the images change clearly, while the CLIP score does not fall on 12 prompts. Memory and the size of the saved model do not change.
- **The images change clearly.** SSIM and LPIPS compare each image with the base model's image for the same prompt and seed. An SSIM near 0.4 means another picture, not a slightly noisier copy. "Changed" is not "worse".
- **The CLIP score does not fall.** It says how well image and prompt fit. With 12 prompts it cannot show a small loss.
- **Most of pruna's other algorithms refuse a CPU or do not fit this model.** `quanto` and `diffusers_int8` refuse a CPU. `stable_fast`, `x_fast` and `hqq_diffusers` stopped on missing packages. `torch_compile` with the openvino backend failed inside PyTorch. `half`, `fora`, `sage_attn` and `hyper` do not fit this model. `torch_dynamic` and `torch_compile` with the inductor backend ran and brought no gain: 4 % faster with other images and more memory, and 10 % slower. The error texts are in `results-image-first-try/`, the two measurements in `results-image/machines-2-and-3-from-the-job-logs.txt`.

How it is measured: `imagebench/generate.py` runs every setting in a process of its own and measures the time of every image and the peak memory (`VmHWM`). The first three prompts are generated twice; "spread" is the largest gap between two times for the same prompt, as a share of their mean. Every repeat gave exactly the same image. Quality is scored by pruna's `EvaluationAgent` with its built-in metrics `clip_score`, `ssim` and `lpips` (`imagebench/score.py`).

Limits: 12 prompts. 256 x 256 pixels on a model made for 512 x 512. One run on one shared machine. No GPU.
The safety checker of the pipeline is off for this run, because in the run before it replaced 5 of 12 harmless images of the base model with black ones, and black images cannot be scored.

## How to reproduce

On GitHub: put this repository into your own account, open the Actions tab, choose "benchmark" and press "Run workflow".
The defaults start the run for size, memory and speed (200 questions, one machine). It took 1 hour and 49 minutes.
For the accuracy run set `questions` to `1034`, `machines` to `["f16", "q8_0", "q5_k_m", "q4_k_m"]` and `second_pass` to `no`. It took 1 hour and 41 minutes on four machines at the same time.
The table appears in the summary of the run. The result files are in the artifact `results`.

On a Linux machine with Python 3.12:

```
pip install -r requirements.txt
python fetch_spider.py data
python compress.py q4_k_m model
python evaluate.py model data/spider 200 output/q4_k_m.json
python evaluate_with_pruna.py model data/spider output/q4_k_m.json output/q4_k_m.pruna.json
python make_table.py output
```

A new run writes to `output/`. The folders `results/` and `results-full-dev/` hold the two published runs and stay as they are.

These steps were only ever run on the GitHub machines described above.

The image experiment: choose "image" in the Actions tab and press "Run workflow". It took 27 minutes.

The unit tests need only Python and pytest and run on any system:

```
pip install -r requirements-dev.txt
pytest
pytest imagebench/tests
```

## Files

| File | What it does |
|---|---|
| `sqlbench/settings.py` | Every fixed choice: model, data, seed, precisions. |
| `sqlbench/sampling.py` | Chooses the questions. |
| `sqlbench/prompt.py` | Reads the schema of a database and builds the prompt. |
| `sqlbench/sqlclean.py` | Cuts the answer of the model down to one SQL statement. |
| `sqlbench/sqlexec.py` | Runs SQL and decides whether two results are the same. |
| `sqlbench/stats.py` | The Wilson interval and the sign test. |
| `sqlbench/checksum.py` | The SHA-256 of a model file. |
| `sqlbench/report.py` | Builds the results table from the result files. |
| `fetch_spider.py` | Step 1: downloads Spider. |
| `compress.py` | Step 2: compresses the model with pruna. |
| `evaluate.py` | Step 3: asks the questions and measures. |
| `evaluate_with_pruna.py` | Step 4: scores a second time with pruna's EvaluationAgent. |
| `make_table.py` | Step 5: prints the table. |
| `compare_runs.py` | Compares two runs: same answers, same model file? |
| `tests/` | Unit tests for the code in `sqlbench/`. |
| `requirements.txt`, `requirements-dev.txt` | The packages for the GitHub machine, and for the unit tests. |
| `.github/workflows/benchmark.yml` | Runs the unit tests and the five steps on GitHub. |
| `results/` | The result files of the 200-question run (size, memory, speed), and the package versions it used. |
| `results-full-dev/` | The result files of the accuracy run with all 1034 questions. |
| `results-q4-repeat/` | Three more runs of q4_k_m alone, made to find out why it did not repeat its answers. |
| `imagebench/` | The image experiment: `settings.py` (every fixed choice), `measure.py` (prompts, spread, table), `generate.py` (images, time, memory), `score.py` (quality with pruna's EvaluationAgent), `make_table.py`, `tests/`. |
| `.github/workflows/image.yml` | Runs the image experiment on GitHub, started by hand. |
| `results-image/` | The result files of the image experiment. |
| `results-image-first-try/` | Which pruna algorithms ran on a CPU in a first try, and the error texts of those that did not. |

## Sources and licences

- Code in this repository: MIT, see `LICENSE`.
- Model: Qwen2.5-Coder-1.5B-Instruct by Alibaba Cloud, Apache-2.0. <https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct>
- Data: Spider by Yu et al., 2018 (<https://arxiv.org/abs/1809.08887>), CC BY-SA 4.0.
  Downloaded from the Hugging Face dataset repository [`xlangai/spider`](https://huggingface.co/datasets/xlangai/spider),
  revision `6232cc3fad6d54c62b3ba23a364083a98ff36a17`, file `data/spider.zip`.
  This is the last revision that holds the SQLite databases. Later revisions hold only the questions.
  This repository contains no Spider data. The result files hold the SQL the model wrote, but not the questions and not the reference queries.
- Image model: BK-SDM-Base by Nota AI (Kim et al., <https://arxiv.org/abs/2305.15798>), CreativeML OpenRAIL-M, not gated. <https://huggingface.co/nota-ai/bk-sdm-base>, revision `0c03b7a0369b49f97d1acf7256d4ee55ced9b2e0`.
- Image prompts: PartiPrompts by Google, Apache-2.0. Downloaded from <https://huggingface.co/datasets/nateraw/parti-prompts>, revision `944b156abfdad7627c3221b5ec4f6a6fb060a197`. This repository contains no prompts and no images.
- pruna 0.3.5: Apache-2.0. <https://github.com/PrunaAI/pruna>
- llama.cpp and llama-cpp-python 0.3.36: MIT.
