"""ClimateCalendar — Gemma 4 E4B Fine-Tune Notebook
=================================================
Run on Kaggle with T4 GPU accelerator.
Trains LoRA adapter on punjab_agronomy.jsonl (799 examples).

Setup before running:
1. Create new Kaggle notebook
2. Settings -> Accelerator -> GPU T4
3. Settings -> Internet -> On
4. Add HUGGINGFACE_TOKEN as a Kaggle secret (Add-ons -> Secrets)

Cell-by-cell execution recommended on first run.
"""

# ============================================================
# CELL 1: Install Unsloth and dependencies
# ============================================================
# !pip install --upgrade pip
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps "trl<0.9.0" "peft" "accelerate" "bitsandbytes"
# !pip install datasets


# ============================================================
# CELL 2: Imports and environment check
# ============================================================
import json
import torch
from pathlib import Path

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ============================================================
# CELL 3: Load Gemma 4 E4B base model in 4-bit
# ============================================================
from unsloth import FastModel

# Gemma 4 E4B Instruction-Tuned, 4-bit quantized
MODEL_NAME = "unsloth/gemma-4-e4b-it-bnb-4bit"
MAX_SEQ_LENGTH = 2048

model, tokenizer = FastModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,  # Auto-detect
    load_in_4bit=True,
    full_finetuning=False,
)

print(f"Loaded {MODEL_NAME}")
print(f"Tokenizer vocab size: {len(tokenizer)}")


# ============================================================
# CELL 4: Configure LoRA adapter
# ============================================================
model = FastModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=32,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
    use_rslora=False,
    loftq_config=None,
)

trainable, total = model.get_nb_trainable_parameters()
print(f"Trainable parameters: {trainable:,} ({100 * trainable / total:.2f}% of {total:,})")


# ============================================================
# CELL 5: Load dataset from GitHub
# ============================================================
import urllib.request

DATASET_URL = "https://raw.githubusercontent.com/tkaushik015/climate-calendar/main/finetune/data/punjab_agronomy.jsonl"
LOCAL_PATH = "/kaggle/working/punjab_agronomy.jsonl"

urllib.request.urlretrieve(DATASET_URL, LOCAL_PATH)
print(f"Downloaded to {LOCAL_PATH}")

with open(LOCAL_PATH, encoding="utf-8") as f:
    raw_entries = [json.loads(line) for line in f if line.strip()]

print(f"Loaded {len(raw_entries)} entries")
print(f"Sample entry:\n{json.dumps(raw_entries[0], indent=2)}")


# ============================================================
# CELL 6: Format dataset with Gemma 4 chat template
# ============================================================
from datasets import Dataset

SYSTEM_PROMPT = (
    "You are ClimateCalendar, a Punjab-specific agronomy assistant for "
    "smallholder farmers. You answer questions using Punjab Agricultural "
    "University (PAU) Package of Practices recommendations. Always cite your "
    "source. Be specific with numbers (kg per acre, days, temperature ranges)."
)


def format_example(entry):
    """Apply Gemma 4 chat template."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": entry["instruction"]},
        {"role": "assistant", "content": entry["output"]},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


dataset = Dataset.from_list(raw_entries)
dataset = dataset.map(format_example)

print(f"Formatted {len(dataset)} examples")
print(f"Sample formatted text:\n{dataset[0]['text'][:500]}")


# ============================================================
# CELL 7: Train/eval split
# ============================================================
split = dataset.train_test_split(test_size=0.05, seed=42)
train_ds = split["train"]
eval_ds = split["test"]

print(f"Train: {len(train_ds)} examples")
print(f"Eval: {len(eval_ds)} examples")


# ============================================================
# CELL 8: Configure trainer
# ============================================================
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="/kaggle/working/checkpoints",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,  # Effective batch size = 8
    warmup_steps=10,
    num_train_epochs=2,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=42,
    report_to="none",
    save_strategy="epoch",
    save_total_limit=2,
    eval_strategy="epoch",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
    packing=False,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    args=training_args,
)

print("Trainer configured.")


# ============================================================
# CELL 9: Sanity check before training - generate from base model
# ============================================================
TEST_QUERIES = [
    "I'm a wheat farmer in Bathinda Punjab. What variety should I sow this November?",
    "How much paddy straw is produced in Punjab and what should I do with it?",
    "What is the ETL for cotton whitefly?",
]


def generate_response(model, tokenizer, query, max_new_tokens=256):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(outputs[0][inputs.shape[1] :], skip_special_tokens=True)
    return response


print("=" * 60)
print("BASE MODEL RESPONSES (before fine-tuning)")
print("=" * 60)
base_responses = {}
for query in TEST_QUERIES:
    print(f"\nQuery: {query}")
    response = generate_response(model, tokenizer, query)
    print(f"Response: {response}")
    base_responses[query] = response

# Save for comparison
with open("/kaggle/working/base_responses.json", "w", encoding="utf-8") as f:
    json.dump(base_responses, f, indent=2)


# ============================================================
# CELL 10: Train
# ============================================================
print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

torch.cuda.empty_cache()
mem_before = torch.cuda.memory_allocated() / 1e9
print(f"GPU memory before training: {mem_before:.2f} GB")

trainer_stats = trainer.train()

mem_after = torch.cuda.memory_allocated() / 1e9
print(f"GPU memory after training: {mem_after:.2f} GB")
print(f"\nTraining complete in {trainer_stats.metrics['train_runtime']:.1f} seconds")
print(f"Final train loss: {trainer_stats.metrics['train_loss']:.4f}")


# ============================================================
# CELL 11: Save LoRA adapter
# ============================================================
ADAPTER_DIR = "/kaggle/working/climate_calendar_lora"
model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print(f"LoRA adapter saved to {ADAPTER_DIR}")

# List saved files
for f in Path(ADAPTER_DIR).iterdir():
    size_mb = f.stat().st_size / 1e6
    print(f"  {f.name}: {size_mb:.2f} MB")


# ============================================================
# CELL 12: Generate from fine-tuned model
# ============================================================
print("\n" + "=" * 60)
print("FINE-TUNED MODEL RESPONSES (after training)")
print("=" * 60)

finetuned_responses = {}
for query in TEST_QUERIES:
    print(f"\nQuery: {query}")
    response = generate_response(model, tokenizer, query)
    print(f"Response: {response}")
    finetuned_responses[query] = response

with open("/kaggle/working/finetuned_responses.json", "w", encoding="utf-8") as f:
    json.dump(finetuned_responses, f, indent=2)


# ============================================================
# CELL 13: Side-by-side comparison
# ============================================================
print("\n" + "=" * 60)
print("A/B COMPARISON: Base vs Fine-Tuned")
print("=" * 60)

comparison = []
for query in TEST_QUERIES:
    base = base_responses[query]
    ft = finetuned_responses[query]

    print(f"\n{'=' * 60}")
    print(f"QUERY: {query}")
    print(f"{'=' * 60}")
    print(f"\n--- BASE Gemma 4 E4B ---")
    print(base[:500] + ("..." if len(base) > 500 else ""))
    print(f"\n--- FINE-TUNED ClimateCalendar ---")
    print(ft[:500] + ("..." if len(ft) > 500 else ""))

    comparison.append(
        {
            "query": query,
            "base_response": base,
            "finetuned_response": ft,
            "pau_citation_in_base": "PAU" in base,
            "pau_citation_in_finetuned": "PAU" in ft,
        }
    )

with open("/kaggle/working/ab_comparison.json", "w", encoding="utf-8") as f:
    json.dump(comparison, f, indent=2)

print("\nA/B comparison saved to /kaggle/working/ab_comparison.json")


# ============================================================
# CELL 14: (Optional) Push to Hugging Face Hub
# ============================================================
# Uncomment and run this cell only after confirming results look good.
# Requires HUGGINGFACE_TOKEN in Kaggle Secrets.

# from kaggle_secrets import UserSecretsClient
# user_secrets = UserSecretsClient()
# hf_token = user_secrets.get_secret("HUGGINGFACE_TOKEN")
#
# from huggingface_hub import login
# login(token=hf_token)
#
# HF_REPO_ID = "tkaushik015/climate-calendar-gemma4-e4b-lora"
# model.push_to_hub(HF_REPO_ID, token=hf_token)
# tokenizer.push_to_hub(HF_REPO_ID, token=hf_token)
# print(f"Pushed to https://huggingface.co/{HF_REPO_ID}")


# ============================================================
# CELL 15: Save merged model (optional, for Ollama later)
# ============================================================
# Merging produces a full model file (~5GB). Skip if disk space is a concern.
# Useful only when you're ready to convert to GGUF for Ollama.

# model.save_pretrained_merged(
#     "/kaggle/working/climate_calendar_merged",
#     tokenizer,
#     save_method="merged_16bit",
# )
# print("Merged model saved for GGUF conversion")
