"""
summarization.py

GOAL: Fine-tune mT5 (a multilingual seq2seq model) to generate short
review TITLES from long review BODIES, using English + Spanish Amazon
reviews.

PIPELINE POSITION:
    Amazon reviews (en + es)
        -> Combine into one multilingual dataset
        -> Preprocessing (tokenize long body -> input, short title -> label)
        -> mT5 encoder-decoder model (AutoModelForSeq2SeqLM)
        -> Compare against a "dumb" 3-sentence baseline (ROUGE)
        -> Fine-tune using Seq2SeqTrainer
        -> Evaluate with ROUGE
        -> Save model
        -> Generate a summary for a new review
"""

from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import evaluate
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True) 


# ===========================================================
# 1. LOAD MULTILINGUAL DATASETS (English + Spanish)
# ===========================================================
print("[1/10] Loading English and Spanish Amazon review datasets...")

english_dataset = load_dataset("goosmanlei/amazon_reviews_multi", "en")
spanish_dataset = load_dataset("goosmanlei/amazon_reviews_multi", "es")

print(f"    English train rows: {english_dataset['train'].num_rows:,}")
print(f"    Spanish train rows: {spanish_dataset['train'].num_rows:,}")


def show_samples(dataset, num_samples=3, seed=42):
    """Print a few random examples - useful sanity check after each
    dataset transformation."""
    sample = dataset["train"].shuffle(seed=seed).select(range(num_samples))
    for example in sample:
        print(f"   '>> Title: {example['review_title']}'")
        print(f"   '>> Review: {example['review_body'][:100]}...'")


# ===========================================================
# 2. FILTER TO BOOK REVIEWS ONLY + COMBINE LANGUAGES
# ===========================================================
# WHY filter to books: keeps the dataset focused on one domain,
# similar in spirit to why we fine-tuned on movie reviews for MLM.
print("\n[2/10] Filtering to book reviews and combining languages...")


def filter_books(example):
    return example["product_category"] in ("book", "digital_ebook_purchase")


english_books = english_dataset.filter(filter_books)
spanish_books = spanish_dataset.filter(filter_books)

books_dataset = DatasetDict()
for split in english_books.keys():
    books_dataset[split] = concatenate_datasets(
        [english_books[split], spanish_books[split]]
    )
    books_dataset[split] = books_dataset[split].shuffle(seed=42)

print(f"    Combined train rows: {books_dataset['train'].num_rows:,}")
print("    Sample after combining:")
show_samples(books_dataset)


# ===========================================================
# 3. FILTER OUT VERY SHORT TITLES
# ===========================================================
# A 1-2 word title gives the model almost nothing useful to learn
# from as a "summary" target.
print("\n[3/10] Removing examples with very short titles...")

books_dataset = books_dataset.filter(lambda x: len(x["review_title"].split()) > 2)
print(f"    Train rows after filtering: {books_dataset['train'].num_rows:,}")


# ===========================================================
# 4. DOWNSAMPLE - use a smaller subset for fast local testing
# ===========================================================
print("\n[4/10] Selecting a small subset for local testing...")

books_dataset["train"] = books_dataset["train"].shuffle(seed=42).select(range(1000))
books_dataset["validation"] = (
    books_dataset["validation"].shuffle(seed=42).select(range(200))
)

print(f"    Using {len(books_dataset['train'])} train / {len(books_dataset['validation'])} validation examples")


# ===========================================================
# 5. TOKENIZER + PREPROCESSING
# ===========================================================
# CONCEPT: same idea as text_target in Translation - the long review
# body is the input, the short review title is the label. Note the
# input and target use DIFFERENT max lengths, since reviews are long
# and titles are short.
print("\n[5/10] Loading tokenizer and preprocessing dataset...")

model_checkpoint = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

max_input_length = 128
max_target_length = 30


def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["review_body"], max_length=max_input_length, truncation=True
    )
    labels = tokenizer(
        examples["review_title"], max_length=max_target_length, truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_datasets = books_dataset.map(preprocess_function, batched=True)
print("    Done tokenizing.")


# ===========================================================
# 6. ROUGE METRIC + BASELINE
# ===========================================================
# CONCEPT: before trusting a fine-tuned model's ROUGE score, we need
# a baseline to compare against. Here the baseline is "just take the
# first 3 sentences of the review and call that the summary" - no AI
# involved at all. If the fine-tuned model can't beat this, something
# is wrong.
print("\n[6/10] Computing baseline ROUGE score (first-3-sentences heuristic)...")

rouge_score = evaluate.load("rouge")


def three_sentence_summary(text):
    return "\n".join(sent_tokenize(text)[:3])


def evaluate_baseline(dataset, metric):
    summaries = [three_sentence_summary(text) for text in dataset["review_body"]]
    return metric.compute(predictions=summaries, references=dataset["review_title"])


baseline_score = evaluate_baseline(books_dataset["validation"], rouge_score)
rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
baseline_dict = {rn: round(baseline_score[rn] * 100, 2) for rn in rouge_names}

print(f"    Baseline ROUGE scores: {baseline_dict}")
print("    (Our fine-tuned model should beat these numbers.)")


# ===========================================================
# 7. LOAD MODEL + DATA COLLATOR
# ===========================================================
print("\n[7/10] Loading pretrained mT5 model + data collator...")

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Remove the original text columns now that we have tokenized versions -
# the Trainer only wants numeric tensors, not raw strings.
tokenized_datasets = tokenized_datasets.remove_columns(
    books_dataset["train"].column_names
)


# ===========================================================
# 8. compute_metrics - ROUGE during training
# ===========================================================
def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 (ignored positions) with the pad token before decoding
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # ROUGE (specifically rougeLsum) expects one sentence per line
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = [
        "\n".join(sent_tokenize(label.strip())) for label in decoded_labels
    ]

    result = rouge_score.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    result = {key: value * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}


# ===========================================================
# 9. TRAINING CONFIGURATION
# ===========================================================
print("\n[8/10] Configuring Seq2SeqTrainer...")

batch_size = 1  
num_train_epochs = 1
logging_steps = len(tokenized_datasets["train"]) // batch_size
model_name = model_checkpoint.split("/")[-1]

args = Seq2SeqTrainingArguments(
    output_dir=f"{model_name}-finetuned-amazon-en-es",
    eval_strategy="epoch",
    learning_rate=5.6e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,  
    logging_steps=logging_steps,
    fp16=False,                  # CPU training - fp16 only helps on GPU
    push_to_hub=False,
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


# ===========================================================
# 10. TRAIN, EVALUATE, SAVE, GENERATE
# ===========================================================
if __name__ == "__main__":
    print("\n[9/10] Starting training...")
    train_result = trainer.train()
    print("    Training finished.")
    print(f"    Final train loss: {train_result.training_loss:.4f}")

    print("\nRunning final evaluation (this computes ROUGE)...")
    eval_results = trainer.evaluate()
    print("    Fine-tuned ROUGE scores:")
    for rn in rouge_names:
        print(f"      {rn}: {eval_results.get('eval_' + rn, 'N/A')}")
    print(f"    Baseline ROUGE scores for comparison: {baseline_dict}")

    save_path = "./mt5-summarizer-final"
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\nModel saved to {save_path}")

    # -------------------------------------------------------
    # Quick sanity check - summarize a validation review
    # -------------------------------------------------------
    print("\n[10/10] Sample generation check:")

    saved_tokenizer = AutoTokenizer.from_pretrained(save_path)
    saved_model = AutoModelForSeq2SeqLM.from_pretrained(save_path)

    text = books_dataset["validation"][0]["review_body"]
    inputs = saved_tokenizer(text, return_tensors="pt", max_length=64, truncation=True)
    output = saved_model.generate(**inputs, max_new_tokens=20)
    summary = saved_tokenizer.decode(output[0], skip_special_tokens=True)

    print(f"    Review:            {text[:200]}...")
    print(f"    Generated summary: {summary}")
    print(f"    Actual title:      {books_dataset['validation'][0]['review_title']}")