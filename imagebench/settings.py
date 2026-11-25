"""Every fixed choice of the image benchmark, in one place.

A run on GitHub takes its work from this file: RUN lists the settings it
measures and PROMPT_COUNT says how many prompts it uses.
"""

# The model: a Stable Diffusion v1.4 with a much smaller UNet (Kim et al., "BK-SDM").
MODEL_ID = "nota-ai/bk-sdm-tiny"
MODEL_REVISION = "0364108e53b7f7f4d2585e817a0b7a83dc261cfa"

# The prompts: PartiPrompts by Google, 1632 prompts, Apache-2.0.
PROMPTS_REPO = "nateraw/parti-prompts"
PROMPTS_REVISION = "944b156abfdad7627c3221b5ec4f6a6fb060a197"
PROMPTS_FILE = "PartiPrompts.tsv"

SEED = 42  # chooses the prompts; prompt number i is generated with seed SEED + i

# How an image is made. Every setting uses the same values.
HEIGHT = 256
WIDTH = 256
STEPS = 8
GUIDANCE_SCALE = 7.5

# Timing: the first few prompts are generated again, to see how much the time varies.
REPEATED_PROMPTS = 2
EXTRA_REPEATS = 0

# All pruna settings that were tried. None means: the model as it is.
# The value is what goes into pruna's SmashConfig.
CANDIDATES = {
    "base": None,
    # Cachers
    "deepcache_2": {"deepcache": {"interval": 2}},
    "deepcache_3": {"deepcache": {"interval": 3}},
    "fora": ["fora"],
    # Compilers
    "torch_compile_inductor": {"torch_compile": {"backend": "inductor"}},
    "torch_compile_openvino": {"torch_compile": {"backend": "openvino"}},
    "stable_fast": ["stable_fast"],
    "x_fast": ["x_fast"],
    # Quantizers
    "torch_dynamic": ["torch_dynamic"],
    "torchao": ["torchao"],
    "half": ["half"],
    "quanto": ["quanto"],
    "hqq_diffusers": ["hqq_diffusers"],
    "diffusers_int8": ["diffusers_int8"],
    "static_fp8_diffusers": ["static_fp8_diffusers"],
    # Others
    "qkv_diffusers": ["qkv_diffusers"],
    "hyper": ["hyper"],
    "sage_attn": ["sage_attn"],
}

# What the next run measures. This is the first try: every setting, one prompt.
RUN = list(CANDIDATES)
PROMPT_COUNT = 1
