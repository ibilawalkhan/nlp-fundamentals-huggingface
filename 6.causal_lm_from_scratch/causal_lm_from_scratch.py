"""
GOAL: Train a small GPT-2-style CAUSAL language model FROM SCRATCH
(random weights, no pretrained starting point) on the TinyStories
dataset, then generate text with it.

PIPELINE POSITION:
    Build a fresh, randomly-initialized
    model and teach it language from nothing but our training data.

KEY CONCEPTS:
    - "From scratch"  = random weights, no pretrained checkpoint.
    - Causal LM       = predict the NEXT token using only LEFT-side
                         context (unlike MLM, which sees both sides).
                         This is the GPT-style objective, and it's
                         why GPT-style models can generate text one
                         token at a time.
    - We reuse GPT-2's existing tokenizer here (training a brand new
      tokenizer is a separate skill - deliberately skipped for now).
    - We use a SMALL model config and a SMALL dataset subset, purely
      so this finishes in a reasonable time on a laptop CPU. This is
      a toy model to see the mechanism work, not a useful model.
"""

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    GPT2Config,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
import math


# ===========================================================
# 1. LOAD A SMALL DATASET
# ===========================================================
# TinyStories is a modern dataset made specifically for training
# small language models quickly - short, simple stories
print("[1/9] Loading TinyStories dataset (small subset)...")

raw_datasets = load_dataset("roneneldan/TinyStories")

# Downsample heavily - this is a "from scratch" toy run, not a real
# training job. Even this will take a while on CPU.
train_dataset = raw_datasets["train"].shuffle(seed=42).select(range(5_000))
eval_dataset = raw_datasets["validation"].shuffle(seed=42).select(range(500))

print(f"    Train: {len(train_dataset):,} stories | Eval: {len(eval_dataset):,} stories")
print(f"    Example story: {train_dataset[0]['text'][:150]}...")


# ===========================================================
# 2. TOKENIZER 
# ===========================================================
# Reused GPT-2's existing tokenizer here to keep focus on the 
# causal-LM training mechanism
print("\n[2/9] Loading GPT-2 tokenizer...")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token 


# ===========================================================
# 3. TOKENIZE
# ===========================================================
print("\n[3/9] Tokenizing dataset...")

context_length = 128


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=context_length)


tokenized_train = train_dataset.map(
    tokenize_function, batched=True, remove_columns=train_dataset.column_names
)
tokenized_eval = eval_dataset.map(
    tokenize_function, batched=True, remove_columns=eval_dataset.column_names
)
print("    Done.")


# ===========================================================
# 4. CHUNKING
# ===========================================================
# Stories have different lengths. Concatenate everything into
# one long stream, then slice into equal-size chunks
print("\n[4/9] Grouping into fixed-length chunks...")


def group_texts(examples):
    concatenated = sum(examples["input_ids"], [])
    total_length = (len(concatenated) // context_length) * context_length
    result = {
        "input_ids": [
            concatenated[i : i + context_length]
            for i in range(0, total_length, context_length)
        ]
    }
    result["labels"] = result["input_ids"].copy()
    return result


lm_train_dataset = tokenized_train.map(group_texts, batched=True)
lm_eval_dataset = tokenized_eval.map(group_texts, batched=True)

print(f"    Train chunks: {len(lm_train_dataset):,} | Eval chunks: {len(lm_eval_dataset):,}")


# ===========================================================
# 5. BUILD A SMALL MODEL FROM SCRATCH (random weights)
# ===========================================================
# THE key concept of this whole script: we define an ARCHITECTURE
# (GPT2Config) and build a model FROM THAT CONFIG, not from a
# pretrained checkpoint. This model currently knows NOTHING about
# language - every weight is randomly initialized.
#
# We use a much smaller config than real GPT-2 (which has 12+ layers,
# 768+ dimensions) purely so training is feasible on a CPU.
print("\n[5/9] Building a small GPT-2 architecture with RANDOM weights...")

config = GPT2Config(
    vocab_size=len(tokenizer),
    n_positions=context_length,
    n_ctx=context_length,
    n_embd=128,   # real GPT-2 uses 768+ - shrunk for CPU feasibility
    n_layer=4,    # real GPT-2 uses 12+
    n_head=4,     # real GPT-2 uses 12+
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

model = AutoModelForCausalLM.from_config(config)  # <-- random weights, NOT from_pretrained
num_params = sum(p.numel() for p in model.parameters())
print(f"    Model built with {num_params:,} randomly-initialized parameters.")


# ===========================================================
# 6. DATA COLLATOR - CAUSAL mode (mlm=False)
# ===========================================================
# mlm=False switches it to causal mode: no masking, the model just learns 
# to predict each next token given everything before it.
print("\n[6/9] Setting up causal-LM data collator...")

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# ===========================================================
# 7. TRAINING CONFIGURATION
# ===========================================================
print("\n[7/9] Configuring Trainer...")

training_args = TrainingArguments(
    output_dir="./gpt2-scratch-tinystories",
    overwrite_output_dir=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-4,
    num_train_epochs=3,   # from-scratch models need more passes than fine-tuning
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    logging_steps=50,
    fp16=False,           # CPU training
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_train_dataset,
    eval_dataset=lm_eval_dataset,
    data_collator=data_collator,
)


# ===========================================================
# 8. TRAIN, EVALUATE (PERPLEXITY), SAVE
# ===========================================================
if __name__ == "__main__":
    # Baseline perplexity BEFORE training - the model is random right
    # now, so expect a very high (bad) number here.
    print("\n[8/9] Evaluating BEFORE training (random weights baseline)...")
    eval_results = trainer.evaluate()
    print(f"    Perplexity before training: {math.exp(eval_results['eval_loss']):.2f}")
    print("    (This should be extremely high - the model knows nothing yet.)")

    print("\nStarting training from scratch (this will take a while)...")
    train_result = trainer.train()
    print("    Training finished.")
    print(f"    Final train loss: {train_result.training_loss:.4f}")

    print("\nEvaluating AFTER training...")
    eval_results = trainer.evaluate()
    print(f"    Perplexity after training: {math.exp(eval_results['eval_loss']):.2f}")
    print("    (Should be MUCH lower than the random-weights baseline above.)")

    save_path = "./gpt2-scratch-tinystories-final"
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\nModel saved to {save_path}")

    # -------------------------------------------------------
    # 9. GENERATE - see what the from-scratch model can produce
    # -------------------------------------------------------
    print("\n[9/9] Generating a sample story...")

    prompt = "Once upon a time"
    inputs = tokenizer(prompt, return_tensors="pt")
    output_ids = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        top_k=50,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print(f"    Prompt:    {prompt}")
    print(f"    Generated: {generated_text}")
    print(
        "\n    NOTE: with only 5,000 training stories, 3 epochs, and a tiny"
        " 4-layer model, expect fairly rough/incoherent text - this is a"
        " toy run to see the training mechanism work, not a usable model."
    )