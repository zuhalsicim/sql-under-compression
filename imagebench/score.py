"""Score the stored images with pruna's EvaluationAgent and its built-in image metrics.

Usage: python -m imagebench.score output-image

generate.py made the images. Here a small stand-in "model" hands them to
pruna's EvaluationAgent, so the agent scores them without generating again.
The base model comes first. pruna keeps its images and compares the images of
every later setting with them (this is what pruna calls a pairwise metric).

It writes output-image/scores.json.
"""

import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from huggingface_hub import hf_hub_download
from PIL import Image
from pruna.data.pruna_datamodule import PrunaDataModule
from pruna.evaluation.evaluation_agent import EvaluationAgent
from pruna.evaluation.metrics import TorchMetricWrapper
from torchvision.transforms.functional import pil_to_tensor

from imagebench.measure import read_prompts
from imagebench.settings import PROMPT_COUNT, PROMPTS_FILE, PROMPTS_REPO, PROMPTS_REVISION, RUN, SEED


class StoredImages:
    """Stands in for a pipeline: called with a batch of prompts, it returns the images made for them earlier."""

    def __init__(self, folder: Path) -> None:
        self.paths = sorted(folder.glob("*.png"))
        self.position = 0

    def __call__(self, prompts: list[str]) -> torch.Tensor:
        paths = self.paths[self.position : self.position + len(prompts)]
        self.position += len(prompts)
        return torch.stack([pil_to_tensor(Image.open(path).convert("RGB")) for path in paths])


def main() -> None:
    out_dir = Path(sys.argv[1])
    prompts_file = hf_hub_download(PROMPTS_REPO, PROMPTS_FILE, repo_type="dataset", revision=PROMPTS_REVISION)
    prompts = read_prompts(prompts_file, PROMPT_COUNT, SEED)

    # pruna's own data format for prompts: a dataset with a "text" column.
    dataset = Dataset.from_dict({"text": prompts})
    datamodule = PrunaDataModule.from_datasets(
        (dataset, dataset, dataset), "prompt_collate", dataloader_args={"batch_size": 4}
    )
    metrics = [
        TorchMetricWrapper("clip_score"),  # do image and prompt fit together?
        TorchMetricWrapper("ssim"),  # is the image built like the base model's image?
        TorchMetricWrapper("lpips"),  # does the image look like the base model's image?
    ]
    agent = EvaluationAgent(request=metrics, datamodule=datamodule, device="cpu")

    # Only settings that produced all images can be scored. The base model must come first.
    finished = [
        name
        for name in RUN
        if "error" not in json.loads((out_dir / f"{name}.json").read_text(encoding="utf-8"))
    ]
    finished.sort(key=lambda name: name != "base")
    scores = {}
    for name in finished:
        results = agent.evaluate(StoredImages(out_dir / name))
        scores[name] = {result.name: result.result for result in results}
        print(name, scores[name], flush=True)
    (out_dir / "scores.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
