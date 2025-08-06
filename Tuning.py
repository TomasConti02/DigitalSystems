%%capture
# Suppresses the output of this cell. Useful for keeping the notebook clean during installations.

import os
if "COLAB_" not in "".join(os.environ.keys()):
    # If not running in Colab or Kaggle, install only the base Unsloth library.
    !pip install unsloth
else:
    # If running inside Colab or Kaggle, install all required dependencies manually.
    !pip install --no-deps bitsandbytes accelerate xformers==0.0.29 peft trl triton
    !pip install --no-deps cut_cross_entropy unsloth_zoo
    !pip install sentencepiece protobuf datasets huggingface_hub hf_transfer
    !pip install --no-deps unsloth

from unsloth import FastLanguageModel
import torch

# Configuration for model loading and quantization.
max_seq_length = 2048  # Maximum token sequence length. Longer contexts are supported.
dtype = None           # Automatically detect the optimal data type (e.g., float16 or bfloat16).
load_in_4bit = True    # Enable 4-bit quantization to reduce memory usage.

# Pre-defined 4-bit quantized models supported by Unsloth for efficient loading and training.
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
    "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
]

# Load the pre-trained model and tokenizer using Unsloth's optimized loading mechanism.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct",  # Choose any supported model.
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_..."  # Use this if accessing gated models that require authentication.
)

# Apply Parameter Efficient Fine-Tuning (PEFT) using LoRA.
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,  # LoRA rank; higher values add more trainable parameters.
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,           # Scaling factor for LoRA layers.
    lora_dropout = 0,          # Dropout rate; zero is optimal for most cases.
    bias = "none",             # Type of bias; "none" uses the least memory.
    use_gradient_checkpointing = "unsloth",  # Enable gradient checkpointing to save VRAM.
    random_state = 3407,       # Random seed for reproducibility.
    use_rslora = False,        # Disable Rank-Stabilized LoRA.
    loftq_config = None        # LoftQ configuration (unused here).
)

# Import libraries for dataset handling, chat formatting, and pretty printing.
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template
from pprint import pprint

# Load a custom dataset from Hugging Face or local file.
dataset = load_dataset(
    path="tomasconti/TestTuning",             # Dataset repo or local directory.
    data_files="formatted_blackhole_dataset_20.json",  # Data file containing formatted conversations.
    split="train"                             # Load the 'train' split.
)

# Apply a LLaMA-style chat template to the tokenizer.
tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.1",  # Format conversations for LLaMA 3.1 models.
)

# Format the dataset by applying the chat template to each conversation.
def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(
            convo,
            tokenize=False,             # Return raw text, not tokenized output.
            add_generation_prompt=False # Do not add generation tokens at the end.
        ) for convo in convos
    ]
    return {"text": texts}

# Map the formatting function across all dataset entries.
dataset = dataset.map(formatting_prompts_func, batched=True)

# Print the first 3 formatted examples for verification.
print("📄 Esempi formattati dal dataset:\n")
for i in range(3):
    pprint(dataset[i]["text"])
    print("\n" + "-" * 80 + "\n")
