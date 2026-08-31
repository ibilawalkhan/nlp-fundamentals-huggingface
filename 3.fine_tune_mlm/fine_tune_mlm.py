"""
fine_tune_mlm.py

GOAL: Fine-tune BERT using Masked Language Modeling (MLM) on IMDB movie
reviews, to adapt a generic pretrained BERT to the "movie review" domain.

PIPELINE POSITION:
    Generic pretrained BERT
        -> Fine-tune with MLM on IMDB text   (THIS SCRIPT)
        -> BERT that understands movie-review language better
        -> (later) fine-tune further for a specific task, e.g. sentiment

KEY IDEA:
    MLM = hide (mask) some words in a sentence, ask BERT to guess them.
    This is UNSUPERVISED - the "label" is just the original word that
    got hidden. No human-annotated labels needed (that's why we don't
    touch the IMDB sentiment `label` column anywhere in this script).
"""

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
import collections
import numpy as np
from transformers import default_data_collator


# ===========================================================
# 1. LOAD DATASET
# ===========================================================
# Three splits exist: train / test / unsupervised.
# We only care about the raw TEXT here, not the sentiment labels,
# because MLM doesn't need labels - the text itself is the training
# signal (predict the hidden words).
data_files = {
    "train": "https://huggingface.co/datasets/stanfordnlp/imdb/resolve/main/plain_text/train-00000-of-00001.parquet",
    "test": "https://huggingface.co/datasets/stanfordnlp/imdb/resolve/main/plain_text/test-00000-of-00001.parquet",
    "unsupervised": "https://huggingface.co/datasets/stanfordnlp/imdb/resolve/main/plain_text/unsupervised-00000-of-00001.parquet",
}

imdb_dataset = load_dataset("parquet", data_files=data_files)


# ===========================================================
# 2. TOKENIZER
# ===========================================================
model_checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def tokenize_function(examples):
    """
    Turn raw text into token IDs.

    We also save word_ids here (which token belongs to which original
    word). We need this LATER for whole-word masking, so a word like
    "playing" -> ["play", "##ing"] gets masked as a whole word, not
    as separate random tokens.
    """
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [
            result.word_ids(i) for i in range(len(result["input_ids"]))
        ]
    return result


tokenized_datasets = imdb_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text", "label"],  # we don't need these for MLM
)


# ===========================================================
# 3. CHUNKING - group all tokens into fixed-length blocks
# ===========================================================
# WHY: Reviews have wildly different lengths. Instead of padding
# every review to match the longest one (wasteful), we glue ALL
# tokenized reviews into one long stream, then slice that stream
# into equal-sized chunks. A chunk might contain the end of one
# review and the start of another - that's fine for MLM, since the
# task is just "predict masked tokens from local context."
chunk_size = 128


def group_texts(examples):
    """
    Concatenate all sequences in the batch into one long stream,
    then split into equal-size chunks of `chunk_size` tokens.
    Leftover tokens that don't fill a full chunk are dropped.
    """
    result = {}

    for key in examples:
        # Join all sequences for this key into one long list
        all_tokens = []
        for seq in examples[key]:
            all_tokens += seq

        # Trim to a multiple of chunk_size (drop the remainder)
        total_length = (len(all_tokens) // chunk_size) * chunk_size

        # Split into equal chunks
        result[key] = [
            all_tokens[i : i + chunk_size]
            for i in range(0, total_length, chunk_size)
        ]

    # For MLM, the "labels" are the ORIGINAL (unmasked) token ids.
    # Masking itself happens later, dynamically, during training -
    # not here. Right now we're just preparing the ground truth.
    result["labels"] = result["input_ids"].copy()

    return result


lm_datasets = tokenized_datasets.map(group_texts, batched=True)

# Sanity check: decode a chunk to see it's readable text
# print(tokenizer.decode(lm_datasets["train"][1]["input_ids"]))


# ===========================================================
# 4. MASKING - the actual MLM task
# ===========================================================
# DataCollatorForLanguageModeling randomly replaces ~15% of tokens
# with [MASK] EVERY TIME a batch is built (fresh randomization each
# training step, not a one-time preprocessing step). The model must
# predict what the masked tokens originally were.
#
# Tokens that are NOT masked get label -100 (ignored in the loss) -
# same -100 trick you already used for special tokens in NER.
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm_probability=0.15,
)


# ===========================================================
# 5. WHOLE WORD MASKING (an upgrade over standard masking)
# ===========================================================
# PROBLEM: BERT splits words into subwords, e.g. "playing" ->
# ["play", "##ing"]. If standard masking only hides "##ing", the
# model can cheat by guessing common suffixes from "play" alone,
# without really using sentence context.
#
# FIX: Mask ALL subword tokens of a word together, so the model is
# forced to use real context to recover the whole word.
#
# This is why we saved word_ids earlier - they tell us which tokens
# belong to the same original word.
wwm_probability = 0.2


def whole_word_masking_data_collator(features):
    for feature in features:
        word_ids = feature.pop("word_ids")

        # Map: original word index -> list of token indices for that word
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly choose ~20% of WHOLE WORDS to mask
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)  # -100 = ignored in loss

        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]          # keep true label
                input_ids[idx] = tokenizer.mask_token_id  # replace with [MASK]

        feature["labels"] = new_labels

    return default_data_collator(features)


# ===========================================================
# 6. DOWNSAMPLE - use a smaller subset so training is faster
#    while you're learning (pure practicality, not a new concept)
# ===========================================================
train_size = 10_000
test_size = int(0.1 * train_size)

downsampled_dataset = lm_datasets["train"].train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)


# ===========================================================
# 7. LOAD MODEL
# ===========================================================
# Same "pretrained BERT + task-specific head" idea as NER, but the
# head here predicts a word from BERT's vocabulary at each masked
# position, instead of one of 9 NER labels.
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)


# ===========================================================
# 8. TRAINING CONFIGURATION
# ===========================================================
batch_size = 64
# Log once per epoch
logging_steps = len(downsampled_dataset["train"]) // batch_size
model_name = model_checkpoint.split("/")[-1]

training_args = TrainingArguments(
    output_dir=f"{model_name}-finetuned-imdb",
    eval_strategy="epoch",       # evaluate at the end of every epoch
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    push_to_hub=False,           # set True + run notebook_login() first
                                  # if you want to upload to your HF account
    fp16=False,                   # mixed precision - faster on GPU;
                                  # set False if you're on CPU only
    logging_steps=logging_steps,
)


# ===========================================================
# 9. TRAINER
# ===========================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=downsampled_dataset["train"],
    eval_dataset=downsampled_dataset["test"],
    data_collator=data_collator,   # swap for whole_word_masking_data_collator
                                    # if you want to experiment with WWM
    processing_class=tokenizer,
)


# ===========================================================
# 10. TRAIN, EVALUATE, SAVE
# ===========================================================
if __name__ == "__main__":
    import math
    from transformers import pipeline

    # -------------------------------------------------------
    # 10a. PERPLEXITY BEFORE FINE-TUNING (baseline)
    # -------------------------------------------------------
    # MLM has no human labels, so we can't use precision/recall/F1
    # like we did for NER. Instead we use PERPLEXITY:
    #
    #     perplexity = e^(cross-entropy loss)
    #
    # Lower perplexity = the model is less "surprised" by real text
    # = it's a better language model for this domain.
    #
    # We check this BEFORE training so we have a baseline to compare
    # against after fine-tuning.
    eval_results = trainer.evaluate()
    print(f">>> Perplexity before fine-tuning: {math.exp(eval_results['eval_loss']):.2f}")

    # -------------------------------------------------------
    # 10b. TRAIN
    # -------------------------------------------------------
    trainer.train()

    # -------------------------------------------------------
    # 10c. PERPLEXITY AFTER FINE-TUNING
    # -------------------------------------------------------
    # This number should be noticeably LOWER than the baseline above.
    # That drop is the proof that domain adaptation worked - the
    # model got better at predicting movie-review-style text.
    eval_results = trainer.evaluate()
    print(f">>> Perplexity after fine-tuning: {math.exp(eval_results['eval_loss']):.2f}")

    # -------------------------------------------------------
    # 11. SAVE THE FINE-TUNED MODEL
    # -------------------------------------------------------
    trainer.save_model("./bert-mlm-imdb-final")
    tokenizer.save_pretrained("./bert-mlm-imdb-final")
    print("Model saved to ./bert-mlm-imdb-final")

    # -------------------------------------------------------
    # 12. QUICK SANITY CHECK - use the fine-tuned model
    # -------------------------------------------------------
    # Same idea as pipeline("token-classification") in your NER
    # script: load the saved model and try it on a real example.
    # If domain adaptation worked, top predictions should lean
    # toward movie-related words (e.g. "movie", "film", "story").
    mask_filler = pipeline(
        "fill-mask",
        model="./bert-mlm-imdb-final",
        tokenizer="./bert-mlm-imdb-final",
    )

    text = "This is a great [MASK]."
    preds = mask_filler(text)

    print("\n>>> Fill-mask predictions for:", text)
    for pred in preds:
        print(f">>> {pred['sequence']}")