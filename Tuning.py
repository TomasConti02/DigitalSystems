# ============================
# 🚀 Unsloth - Fast LLM Loader
# ============================

# 🔄 Detect environment (Colab or Local)
%%capture
import os

# Install dependencies based on the environment
if "COLAB_" not in "".join(os.environ.keys()):
    # ✅ Local machine (personal GPU with CUDA runtime)
    !pip install unsloth
else:
    # ✅ Google Colab or Kaggle
    !pip install --no-deps bitsandbytes accelerate xformers==0.0.29 peft trl triton
    !pip install --no-deps cut_cross_entropy unsloth_zoo
    !pip install sentencepiece protobuf datasets huggingface_hub hf_transfer
    !pip install --no-deps unsloth

# =============================
# 🔧 Load Model with Unsloth
# =============================

from unsloth import FastLanguageModel
import torch

# Maximum sequence length supported (RoPE scaling auto-handled)
max_seq_length = 2048

# Set dtype:
# - None = auto-detect
# - torch.float16 for Tesla T4 / V100
# - torch.bfloat16 for Ampere (A100, L4, etc.)
dtype = None

# Enable 4-bit quantization to save memory
load_in_4bit = True

# =======================================
# 📦 Supported Pre-Quantized 4bit Models
# =======================================

fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",
    "unsloth/Mistral-Small-Instruct-2409",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",
    "unsloth/Llama-3.2-1B-bnb-4bit",
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-3B-bnb-4bit",
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    "unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
]
# 🔗 More at: https://huggingface.co/unsloth

# =============================
# 🧠 Load the Chosen LLM Model
# =============================

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct",  # 🔁 You can change this
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_..."  # 🔐 Use this if the model is gated (like Llama-2)
)
