"""Generate the images of one setting and measure time and memory.

Usage: python -m imagebench.generate deepcache_2 output-image

One call measures one setting in its own process, so the peak memory belongs
to this setting alone. It writes
    output-image/deepcache_2.json          times and memory, or the error if pruna refused
    output-image/deepcache_2/000.png ...   the images
It runs on Linux only (it reads /proc).
"""

import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import diffusers
import pruna
import torch
from diffusers import StableDiffusionPipeline
from huggingface_hub import hf_hub_download
from pruna import SmashConfig, smash

from imagebench.measure import read_prompts
from imagebench.settings import (
    CANDIDATES,
    EXTRA_REPEATS,
    GUIDANCE_SCALE,
    HEIGHT,
    MODEL_ID,
    MODEL_REVISION,
    PROMPT_COUNT,
    PROMPTS_FILE,
    PROMPTS_REPO,
    PROMPTS_REVISION,
    REPEATED_PROMPTS,
    SEED,
    STEPS,
    WIDTH,
)


def peak_memory_bytes() -> int:
    """Return the most RAM this process has held so far (VmHWM, the peak resident set size)."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024  # the line looks like "VmHWM:  123456 kB"
    raise RuntimeError("no VmHWM line")


def runner_info() -> dict:
    """Describe the machine, because speed depends on it."""
    cpu_lines = Path("/proc/cpuinfo").read_text().splitlines()
    return {
        "name": os.environ.get("RUNNER_NAME", "unknown"),
        "cpu_model": next(line.split(":", 1)[1].strip() for line in cpu_lines if line.startswith("model name")),
        "cpu_count": os.cpu_count(),
        "memory_bytes": int(Path("/proc/meminfo").read_text().split()[1]) * 1024,
        "image": f"{os.environ.get('ImageOS', 'unknown image')} {os.environ.get('ImageVersion', '')}".strip(),
    }


def folder_size(folder: Path) -> int:
    return sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())


def measure(name: str, out_dir: Path, result: dict) -> None:
    prompts_file = hf_hub_download(PROMPTS_REPO, PROMPTS_FILE, repo_type="dataset", revision=PROMPTS_REVISION)
    prompts = read_prompts(prompts_file, PROMPT_COUNT, SEED)

    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if CANDIDATES[name] is not None:
        start = time.perf_counter()
        pipe = smash(model=pipe, smash_config=SmashConfig(CANDIDATES[name], device="cpu"))
        result["smash_seconds"] = time.perf_counter() - start

    def make_image(number: int):
        """Generate the image for one prompt. The seed depends only on the prompt number."""
        generator = torch.Generator("cpu").manual_seed(SEED + number)
        start = time.perf_counter()
        output = pipe(
            prompts[number],
            height=HEIGHT,
            width=WIDTH,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE_SCALE,
            generator=generator,
        )
        seconds = time.perf_counter() - start
        blocked = bool(output.nsfw_content_detected and output.nsfw_content_detected[0])
        return output.images[0], seconds, blocked

    # One untimed image first. With a compiler this call includes the compilation.
    _, result["warm_up_seconds"], _ = make_image(0)
    print(f"{name}: warm-up took {result['warm_up_seconds']:.1f} s", flush=True)

    image_dir = out_dir / name
    image_dir.mkdir(parents=True, exist_ok=True)
    first_images = {}
    result["images"] = []
    for repeat in range(1 + EXTRA_REPEATS):
        # The first pass generates every prompt. Later passes repeat the first few.
        numbers = range(len(prompts)) if repeat == 0 else range(min(REPEATED_PROMPTS, len(prompts)))
        for number in numbers:
            image, seconds, blocked = make_image(number)
            if repeat == 0:
                image.save(image_dir / f"{number:03d}.png")
                first_images[number] = image.tobytes()
            result["images"].append(
                {
                    "prompt_number": number,
                    "repeat": repeat,
                    "seconds": seconds,
                    "blocked_by_safety_checker": blocked,
                    "same_as_first": image.tobytes() == first_images[number],
                }
            )
            print(f"{name}: prompt {number}, repeat {repeat}: {seconds:.1f} s", flush=True)
    result["peak_memory_bytes"] = peak_memory_bytes()

    # After the memory is read: how large is the model when it is saved?
    try:
        saved = out_dir / f"{name}-saved"
        pipe.save_pretrained(str(saved))
        result["saved_size_bytes"] = folder_size(saved)
        shutil.rmtree(saved)
    except Exception as error:
        result["saved_size_error"] = f"{type(error).__name__}: {error}"[:300]


def main() -> None:
    name, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "setting": name,
        "smash_config": CANDIDATES[name],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "image_settings": {"height": HEIGHT, "width": WIDTH, "steps": STEPS, "guidance_scale": GUIDANCE_SCALE},
        "pruna_version": pruna.__version__,
        "torch_version": torch.__version__,
        "diffusers_version": diffusers.__version__,
        "runner": runner_info(),
    }
    try:
        measure(name, out_dir, result)
    except Exception as error:  # pruna refused the setting, or it failed while running
        traceback.print_exc()
        result["error"] = f"{type(error).__name__}: {error}"[:600]
        result.setdefault("peak_memory_bytes", peak_memory_bytes())
    (out_dir / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "images"}, indent=2))


if __name__ == "__main__":
    main()
