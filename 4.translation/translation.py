"""
translation.py

GOAL: Fine-tune a Marian (English -> French) translation model on the
KDE4 dataset.

PIPELINE POSITION:
    KDE4 dataset (en/fr sentence pairs)
        -> Preprocessing (tokenize both languages)
        -> Marian encoder-decoder model (AutoModelForSeq2SeqLM)
        -> Fine-tune using Seq2SeqTrainer
        -> Evaluate with BLEU (sacrebleu)
        -> Save model
        -> Translate new sentences

KEY DIFFERENCE FROM NER / MLM:
    This is a SEQ2SEQ task - input and output are different sequences
    (English in, French out), not aligned token-by-token like NER,
    and not "fill in one word" like MLM.
"""

from huggingface_hub import hf_hub_download
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import evaluate
import numpy as np


# ===========================================================
# 1. LOAD DATASET
# ===========================================================
# The "kde4" loading script is deprecated,
# so we download the exact Parquet file directly and load it locally
# - this avoids repeated network calls to resolve the dataset config.
print("[1/9] Downloading KDE4 (en-fr) dataset file...")

local_path = hf_hub_download(
    repo_id="Helsinki-NLP/kde4",
    repo_type="dataset",
    filename="en-fr/train/0000.parquet",
    revision="refs/convert/parquet",
)

raw_datasets = load_dataset("parquet", data_files={"train": local_path})
print(f"    Loaded {raw_datasets['train'].num_rows:,} raw sentence pairs.")


# ===========================================================
# 2. CREATE TRAIN / VALIDATION SPLIT
# ===========================================================
# The dataset only ships a single "train" split, so we carve out
# our own validation set to check performance during training.
print("[2/9] Splitting into train/validation sets...")

split_datasets = raw_datasets["train"].train_test_split(train_size=0.9, seed=10)
split_datasets["validation"] = split_datasets.pop("test")

print(f"    Train:      {split_datasets['train'].num_rows:,} pairs")
print(f"    Validation: {split_datasets['validation'].num_rows:,} pairs")
print(f"    Example pair: {split_datasets['train'][0]['translation']}")


# ===========================================================
# 3. LOAD TOKENIZER
# ===========================================================
print("[3/9] Loading tokenizer for", "Helsinki-NLP/opus-mt-en-fr")

model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


# ===========================================================
# 4. PREPROCESSING - tokenize inputs AND targets
# ===========================================================
# CONCEPT: text_target=... tells the tokenizer to encode the target
# (French) text using the correct target-language tokenization rules,
# and store the result in "labels" instead of "input_ids".
#
# Skipping text_target and tokenizing French with the default (source
# language) settings produces broken, over-fragmented tokens 
max_length = 128


def preprocess_function(examples):
    inputs = [ex["en"] for ex in examples["translation"]]
    targets = [ex["fr"] for ex in examples["translation"]]
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=max_length, truncation=True
    )
    return model_inputs


print("[4/9] Tokenizing dataset (this may take a minute)...")

tokenized_datasets = split_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=split_datasets["train"].column_names,
)

print("    Done. Example tokenized labels (first row):")
print("   ", tokenized_datasets["train"][0]["labels"])


# ===========================================================
# 5. DOWNSAMPLE - use a smaller subset for fast local testing
# ===========================================================
# Full KDE4 training would take hours on a CPU. Use a small slice
# first to confirm everything runs end-to-end before committing to
# a full training run.
print("[5/9] Selecting a small subset for local testing...")

small_train_dataset = tokenized_datasets["train"].select(range(5_000))
small_eval_dataset = tokenized_datasets["validation"].select(range(500))

print(f"    Using {len(small_train_dataset):,} train / {len(small_eval_dataset):,} eval examples")


# ===========================================================
# 6. LOAD MODEL + DATA COLLATOR
# ===========================================================
# AutoModelForSeq2SeqLM = pretrained Marian encoder-decoder, ready
# for translation-style generation (not classification/token-labeling).
#
# DataCollatorForSeq2Seq handles padding for BOTH the input sequences
# and the label sequences, and takes care of setting up
# "decoder_input_ids" (what the decoder sees, shifted right) -
# infrastructure detail, not something you need to build yourself.
print("[6/9] Loading pretrained model + data collator...")

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


# ===========================================================
# 7. METRIC - BLEU (via sacrebleu)
# ===========================================================
# CONCEPT: BLEU measures translation quality by comparing generated
# translations against human reference translations 
# Higher BLEU = better translation.
print("[7/9] Setting up BLEU metric (sacrebleu)...")

metric = evaluate.load("sacrebleu")


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 with the pad token before
    # decoding, since -100 isn't a valid token id.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}


# ===========================================================
# 8. TRAINING CONFIGURATION
# ===========================================================
print("[8/9] Configuring Seq2SeqTrainer...")

args = Seq2SeqTrainingArguments(
    output_dir="marian-finetuned-kde4-en-to-fr",
    eval_strategy="epoch",       # evaluate (and print BLEU) each epoch
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,  # simulates a larger effective batch size
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=1,
    predict_with_generate=True,     # required for BLEU - actually generates text
    fp16=False,                     # CPU training - fp16 only helps on GPU
    push_to_hub=False,              # set True + login first if you want to upload
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)