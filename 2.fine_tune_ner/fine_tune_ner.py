# ============================================================
# FINE-TUNING BERT FOR NER
# ============================================================

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    pipeline,
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

raw_datasets = load_dataset(
    "eriktks/conll2003",
    revision="refs/convert/parquet"
)

print("\n=== DATASET ===")
print(raw_datasets)


# ============================================================
# 2. GET NER LABEL NAMES
# ============================================================

ner_feature = raw_datasets["train"].features["ner_tags"]

label_names = ner_feature.feature.names

print("\n=== LABEL NAMES ===")
print(label_names)

print("Number of labels:", len(label_names))


# ============================================================
# 3. LOAD TOKENIZER
# ============================================================

model_checkpoint = "bert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(
    model_checkpoint
)


# ============================================================
# 4. ALIGN WORD-LEVEL NER LABELS WITH TOKENS
# ============================================================

def align_labels_with_tokens(labels, word_ids):

    new_labels = []

    current_word = None

    for word_id in word_ids:

        # Special tokens such as [CLS] and [SEP]
        if word_id is None:
            new_labels.append(-100)

        # First token of a word
        elif word_id != current_word:

            current_word = word_id

            label = labels[word_id]

            new_labels.append(label)

        # Additional subword token
        else:

            label = labels[word_id]

            # B- label → I- label
            if label % 2 == 1:
                label = label + 1

            new_labels.append(label)

    return new_labels


# ============================================================
# 5. TOKENIZE + ALIGN LABELS
# ============================================================

def tokenize_and_align_labels(examples):

    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
    )

    all_labels = examples["ner_tags"]

    new_labels = []

    for i, labels in enumerate(all_labels):

        word_ids = tokenized_inputs.word_ids(i)

        aligned_labels = align_labels_with_tokens(
            labels,
            word_ids
        )

        new_labels.append(aligned_labels)

    tokenized_inputs["labels"] = new_labels

    return tokenized_inputs


# ============================================================
# 6. PROCESS THE WHOLE DATASET
# ============================================================

tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)

print("\n=== TOKENIZED DATASET ===")
print(tokenized_datasets)


# ============================================================
# 7. CREATE THE MODEL
# ============================================================

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(label_names),
    id2label={
        i: label
        for i, label in enumerate(label_names)
    },
    label2id={
        label: i
        for i, label in enumerate(label_names)
    },
)

print("\n=== MODEL ===")
print("Model:", model_checkpoint)
print("Number of labels:", len(label_names))


# ============================================================
# 8. DATA COLLATOR
# ============================================================

data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer
)


# ============================================================
# 9. TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(
    output_dir="./bert-ner",

    # Training
    learning_rate=2e-5,
    num_train_epochs=3,
    per_device_train_batch_size=8,

    # Evaluation
    per_device_eval_batch_size=8,
    eval_strategy="epoch",

    # Save model
    save_strategy="epoch",

    # Keep the best checkpoint
    load_best_model_at_end=True,

    # Logging
    logging_steps=100,

    # Avoid unnecessary external reporting
    report_to="none",
)


# ============================================================
# 10. CREATE TRAINER
# ============================================================

trainer = Trainer(
    model=model,

    args=training_args,

    train_dataset=tokenized_datasets["train"],

    eval_dataset=tokenized_datasets["validation"],

    processing_class=tokenizer,

    data_collator=data_collator,
)


# ============================================================
# 11. START TRAINING
# ============================================================

print("\n=== STARTING TRAINING ===")

trainer.train()


# ============================================================
# 12. EVALUATE
# ============================================================

print("\n=== EVALUATION ===")

results = trainer.evaluate()

print(results)


# ============================================================
# 13. SAVE THE FINAL MODEL
# ============================================================

output_dir = "./bert-ner-final"

trainer.save_model(output_dir)

tokenizer.save_pretrained(output_dir)

print("\nModel saved to:", output_dir)


# ============================================================
# 14. LOAD THE FINE-TUNED MODEL WITH PIPELINE
# ============================================================

token_classifier = pipeline(
    "token-classification",
    model=output_dir,
    tokenizer=output_dir,
    aggregation_strategy="simple",
)


# ============================================================
# 15. TEST THE MODEL
# ============================================================

text = (
    "My name is John and I work at Google "
    "in New York."
)

predictions = token_classifier(text)


print("\n=== PREDICTIONS ===")

for prediction in predictions:

    print(
        f"{prediction['word']} "
        f"→ {prediction['entity_group']} "
        f"(score: {prediction['score']:.4f})"
    )