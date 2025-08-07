%%capture
# Suppress output to keep the notebook clean during installation
import os

# Check if we're running in Colab or Kaggle
if "COLAB_" not in "".join(os.environ.keys()):
    # Local installation: install only the core Unsloth library
    !pip install unsloth
else:
    # Colab/Kaggle installation: install dependencies manually
    !pip install --no-deps bitsandbytes accelerate xformers==0.0.29 peft trl triton
    !pip install --no-deps cut_cross_entropy unsloth_zoo
    !pip install sentencepiece protobuf datasets huggingface_hub hf_transfer
    !pip install --no-deps unsloth
###################################################################################################
# Import Unsloth's model loader
from unsloth import FastLanguageModel
import torch

# Set up model loading configuration
max_seq_length = 2048     # Maximum context length for prompts
dtype = None              # Automatically selects float16 or bfloat16 depending on GPU support
load_in_4bit = True       # Load model in 4-bit quantization (saves VRAM)
###################################################################################################
# List of 4-bit quantized models supported by Unsloth
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
###################################################################################################
# Load a quantized 4-bit pre-trained model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct",  # Choose any model from the list above
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_..."  # Uncomment if your model requires authentication
)
###################################################################################################
# Apply Parameter-Efficient Fine-Tuning (PEFT) using LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,  # LoRA rank (higher = more capacity but slower)
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha = 16,           # Scaling factor for LoRA
    lora_dropout = 0,          # No dropout for training stability
    bias = "none",             # No additional bias parameters
    use_gradient_checkpointing = "unsloth",  # Reduces VRAM usage during training
    random_state = 3407,       # Seed for reproducibility
    use_rslora = False,        # Disable Rank-Stabilized LoRA
    loftq_config = None        # Not using LoftQ quantization
)
###################################################################################################
# Load and preprocess your dataset
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template
from pprint import pprint

# Load dataset from Hugging Face hub or local path
dataset = load_dataset(
    path = "tomasconti/TestTuning",                 # HF dataset repo or local directory
    data_files = "formatted_blackhole_dataset_20.json",  # JSON file with formatted conversations
    split = "train"                                 # Load the training split
)
###################################################################################################
# Apply the appropriate chat template for LLaMA 3.1
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",  # Format prompt/response with LLaMA 3.1 style tags
)
###################################################################################################
# Format each example into a complete prompt using the tokenizer
def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(
            convo,
            tokenize = False,              # Return string, not token IDs
            add_generation_prompt = False  # Don’t append assistant generation tokens
        ) for convo in convos
    ]
    return {"text": texts}
###################################################################################################
# Standardize ShareGPT-like format, then apply formatting to all examples
from unsloth.chat_templates import standardize_sharegpt

dataset = standardize_sharegpt(dataset)                     # Convert raw format to ShareGPT-style
dataset = dataset.map(formatting_prompts_func, batched=True)  # Apply prompt formatting
###################################################################################################
# Print 3 examples to verify the formatting
print("📄 Formatted dataset examples:\n")
for i in range(3):
    pprint(dataset[i]["text"])
    print("\n" + "-" * 80 + "\n")
###################################################################################################
# Import trainer and training arguments
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported

# Initialize the trainer for supervised fine-tuning
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",  # Field in dataset containing the prompt
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 2,         # Use 2 processes for preprocessing (adjust if needed)
    packing = False,              # Set to True if you want to pack short sequences together
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,  # Simulates effective batch size = 2 * 4 = 8
        warmup_steps = 5,
        # num_train_epochs = 1,    # Optional: can use `max_steps` instead
        max_steps = 60,           # Run training for 60 steps (good for testing)
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),  # Use fp16 unless bf16 is supported
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,        # Log every step
        optim = "adamw_8bit",     # Use 8-bit Adam optimizer
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",       # Set to "wandb" if using Weights & Biases
    ),
)
###################################################################################################
# Train only on assistant responses (ignores the user input parts)
from unsloth.chat_templates import train_on_responses_only

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",       # Marks user input
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",     # Marks assistant reply
)
###################################################################################################
# (Optional) Debug a specific sample by printing the decoded labels
space = tokenizer(" ", add_special_tokens = False).input_ids[0]

# Replace label -100 (masked) tokens with space for readability
tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[5]["labels"]])
###################################################################################################
# 🚀 Start training!
trainer_stats = trainer.train()
###################################################################################################
# === Import chat template for LLaMA 3.1-style conversation formatting ===
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel
from transformers import TextStreamer

# === Apply the LLaMA 3.1 chat template to the tokenizer ===
tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.1",  # Defines the conversation formatting style
)

# === Optimize the model for inference (faster generation) ===
FastLanguageModel.for_inference(model)

# === Define a sample user message (prompt) ===
messages = [
    {"role": "user", "content": "Continue the Fibonacci sequence: 1, 1, 2, 3, 5, 8,"},
]

# === Tokenize the chat-style input with generation flag ===
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,  # Adds "Assistant:" to indicate where generation starts
    return_tensors="pt",         # Return PyTorch tensors
).to("cuda")                     # Move input tensors to GPU

# === Option 1: Generate output and decode the full response ===
outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=64,       # Limit the number of new tokens generated
    use_cache=True,          # Speed up generation with caching
    temperature=1.5,         # High temp = more creativity
    min_p=0.1                # nucleus/top-p sampling for diversity
)

# === Decode and print the output text ===
response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print("Generated response:\n", response[0])


# === Option 2: Stream the output token by token (live printing) ===
text_streamer = TextStreamer(tokenizer, skip_prompt=True)
_ = model.generate(
    input_ids=inputs,
    streamer=text_streamer,  # Stream output live
    max_new_tokens=128,
    use_cache=True,
    temperature=1.5,
    min_p=0.1,
)


# === Save the fine-tuned model and tokenizer to disk ===
model.save_pretrained("MY_MODEL")         # Save model weights/config
tokenizer.save_pretrained("MY_MODEL")     # Save tokenizer with chat template
