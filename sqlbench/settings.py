"""Every fixed choice of the benchmark, in one place."""

# The model. The revision is a commit of the model repository on Hugging Face.
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MODEL_REVISION = "2e1fd397ee46e1388853d2af2c993145b0f1098a"

# The data: the Spider zip file with questions and SQLite databases.
# The current revision of this repository holds only Parquet files without
# databases, so the benchmark uses the last revision that still has the zip.
SPIDER_REPO = "xlangai/spider"
SPIDER_REVISION = "6232cc3fad6d54c62b3ba23a364083a98ff36a17"
SPIDER_FILE = "data/spider.zip"
SPIDER_SHA256 = "5ddff97bb1d421282c593e8d30ce0ce107270f4dd4a21d60eba4bf287d5956b1"

# The four precisions. "f16" is the baseline, the others are compressed.
PRECISIONS = ["f16", "q8_0", "q5_k_m", "q4_k_m"]
BASELINE = "f16"

# Which questions are asked.
SEED = 42
FULL_QUESTIONS = 200
SMOKE_QUESTIONS = 5

# How the model is run.
N_CTX = 4096  # longest prompt plus answer the model can hold, in tokens
MAX_TOKENS = 256  # longest answer the model may write, in tokens
