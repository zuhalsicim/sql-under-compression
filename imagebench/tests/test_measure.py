import pytest

from imagebench.measure import make_table, read_prompts, spread, summarise


@pytest.fixture
def prompts_file(tmp_path):
    path = tmp_path / "PartiPrompts.tsv"
    lines = ["Prompt\tCategory\tChallenge\tNote"] + [f"prompt number {i}\tAnimals\tBasic\t" for i in range(50)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_read_prompts_gives_the_same_prompts_for_the_same_seed(prompts_file):
    first = read_prompts(prompts_file, 5, seed=42)
    assert first == read_prompts(prompts_file, 5, seed=42)
    assert first != read_prompts(prompts_file, 5, seed=43)
    assert len(set(first)) == 5
    assert all(prompt.startswith("prompt number ") for prompt in first)


def test_a_larger_sample_is_not_needed_to_keep_the_order(prompts_file):
    numbers = [int(prompt.split()[-1]) for prompt in read_prompts(prompts_file, 10, seed=42)]
    assert numbers == sorted(numbers)


def test_spread_is_the_range_as_a_share_of_the_mean():
    assert spread([10.0, 10.0]) == 0.0
    assert spread([9.0, 11.0]) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        spread([])


def image(number, repeat, seconds, same=True):
    return {
        "prompt_number": number,
        "repeat": repeat,
        "seconds": seconds,
        "blocked_by_safety_checker": False,
        "same_as_first": same,
    }


def result(images):
    """A result file like generate.py writes it, with made-up numbers."""
    return {"images": images, "peak_memory_bytes": 3_000_000_000}


def test_summarise_uses_the_first_pass_for_the_mean_and_the_repeats_for_the_spread():
    summary = summarise(result([image(0, 0, 10.0), image(1, 0, 20.0), image(0, 1, 12.0)]))
    assert summary["images"] == 2
    assert summary["mean_seconds_per_image"] == 15.0
    assert summary["largest_spread_between_repeats"] == pytest.approx(2.0 / 11.0)
    assert summary["repeats_gave_the_same_image"] is True


def test_summarise_sees_a_repeat_that_gave_another_image():
    summary = summarise(result([image(0, 0, 10.0), image(0, 1, 10.0, same=False)]))
    assert summary["repeats_gave_the_same_image"] is False


def test_summarise_without_repeats():
    summary = summarise(result([image(0, 0, 10.0)]))
    assert summary["largest_spread_between_repeats"] is None
    assert summary["repeats_gave_the_same_image"] is None


def test_table_compares_speed_with_the_base_model_and_shows_refused_settings():
    results = {
        "base": result([image(0, 0, 40.0), image(0, 1, 44.0)]),
        "deepcache_2": result([image(0, 0, 20.0), image(0, 1, 20.0)]),
        "quanto": {"error": "ValueError: needs cuda"},
    }
    scores = {"base": {"clip_score": 30.123}, "deepcache_2": {"clip_score": 29.5, "ssim": 0.71234, "lpips": 0.2}}
    rows = make_table(results, scores).splitlines()
    assert rows[2] == "| base | 40.0 | 1.00 x | 9.5% | 3.00 GB | 30.12 |  |  |"
    assert rows[3] == "| deepcache_2 | 20.0 | 2.00 x | 0.0% | 3.00 GB | 29.50 | 0.712 | 0.200 |"
    assert rows[4] == "| quanto | did not run: ValueError: needs cuda | | | | | | |"
