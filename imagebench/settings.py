"""Every fixed choice of the image benchmark, in one place.

The workflow says which settings run together on one machine.
PROMPT_COUNT says how many prompts every setting gets.
"""

# The model: a Stable Diffusion v1.4 with a smaller UNet (Kim et al., "BK-SDM").
# The first try used the still smaller "nota-ai/bk-sdm-tiny". Its UNet has no
# middle block, and pruna's deepcache stops with an error on such a UNet.
MODEL_ID = "nota-ai/bk-sdm-base"
MODEL_REVISION = "0c03b7a0369b49f97d1acf7256d4ee55ced9b2e0"

# The prompts: PartiPrompts by Google, 1632 prompts, Apache-2.0.
PROMPTS_REPO = "nateraw/parti-prompts"
PROMPTS_REVISION = "944b156abfdad7627c3221b5ec4f6a6fb060a197"
PROMPTS_FILE = "PartiPrompts.tsv"

SEED = 42  # chooses the prompts; prompt number i is generated with seed SEED + i

# How an image is made. Every setting uses the same values.
HEIGHT = 256
WIDTH = 256
STEPS = 15
GUIDANCE_SCALE = 7.5

# Timing: the first few prompts are generated again, to see how much the time varies.
REPEATED_PROMPTS = 3
EXTRA_REPEATS = 1

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

PROMPT_COUNT = 12
