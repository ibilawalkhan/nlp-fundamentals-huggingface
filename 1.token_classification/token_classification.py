from datasets import load_dataset
from transformers import AutoTokenizer


# ============================================================
# 1. LOAD THE DATASET
# ============================================================

raw_datasets = load_dataset(
    "eriktks/conll2003",
    revision="refs/convert/parquet"
)

print("Dataset:")
print(raw_datasets)


# ============================================================
# 2. LOOK AT ONE EXAMPLE
# ============================================================

example = raw_datasets["train"][0]

print("\n--- ONE EXAMPLE ---")

print("ID:", example["id"])
print("Tokens:", example["tokens"])
print("POS Tags:", example["pos_tags"])
print("Chunk Tags:", example["chunk_tags"])
print("NER Tags:", example["ner_tags"])


# ============================================================
# 3. GET THE NER LABEL NAMES
# ============================================================

ner_feature = raw_datasets["train"].features["ner_tags"]

label_names = ner_feature.feature.names

print("\nNER label names:")
print(label_names)


# Show the relationship between words and NER labels
print("\n--- WORD → NER LABEL ---")

for word, label_id in zip(
    example["tokens"],
    example["ner_tags"]
):
    print(
        word,
        "→",
        label_id,
        "→",
        label_names[label_id]
    )


# ============================================================
# 4. LOAD THE BERT TOKENIZER
# ============================================================

model_checkpoint = "bert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(
    model_checkpoint
)


# ============================================================
# 5. TOKENIZE ONE EXAMPLE
# ============================================================

inputs = tokenizer(
    example["tokens"],
    is_split_into_words=True
)

print("\n--- TOKENIZED EXAMPLE ---")

print("Tokens:")
print(inputs.tokens())

print("\nInput IDs:")
print(inputs["input_ids"])


# ============================================================
# 6. SEE WHICH ORIGINAL WORD EACH TOKEN CAME FROM
# ============================================================

word_ids = inputs.word_ids()

print("\n--- TOKEN → ORIGINAL WORD ---")

for token, word_id in zip(inputs.tokens(), word_ids):
    print(token, "→ word_id:", word_id)


# ============================================================
# 7. FUNCTION TO ALIGN NER LABELS WITH TOKENIZER TOKENS
# ============================================================

def align_labels_with_tokens(labels, word_ids):
    """
    Align the original word-level NER labels
    with the tokens produced by the tokenizer.
    """

    new_labels = []

    current_word = None

    for word_id in word_ids:

        # Special tokens such as [CLS] and [SEP]
        if word_id is None:
            new_labels.append(-100)

        # First token of a new word
        elif word_id != current_word:
            current_word = word_id
            label = labels[word_id]

            new_labels.append(label)

        # Additional subword token of the same word
        else:
            label = labels[word_id]

            # B- label → I- label
            if label % 2 == 1:
                label = label + 1

            new_labels.append(label)

    return new_labels


# ============================================================
# 8. TEST LABEL ALIGNMENT ON ONE EXAMPLE
# ============================================================

aligned_labels = align_labels_with_tokens(
    example["ner_tags"],
    word_ids
)

print("\n--- ALIGNED LABELS ---")

for token, label_id in zip(
    inputs.tokens(),
    aligned_labels
):

    if label_id == -100:
        label_name = "IGNORED"
    else:
        label_name = label_names[label_id]

    print(
        token,
        "→",
        label_id,
        "→",
        label_name
    )


# ============================================================
# 9. APPLY TOKENIZATION + LABEL ALIGNMENT
#    TO THE WHOLE DATASET
# ============================================================

def tokenize_and_align_labels(examples):

    # Tokenize all sentences/words in this batch
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True
    )

    # Original word-level NER labels
    all_labels = examples["ner_tags"]

    new_labels = []

    # Process each example in the batch
    for i, labels in enumerate(all_labels):

        # Find which original word each token came from
        word_ids = tokenized_inputs.word_ids(i)

        # Align word-level labels with tokenizer tokens
        aligned_labels = align_labels_with_tokens(
            labels,
            word_ids
        )

        new_labels.append(aligned_labels)

    # Add aligned labels to the tokenized dataset
    tokenized_inputs["labels"] = new_labels

    return tokenized_inputs


# ============================================================
# 10. PROCESS THE WHOLE DATASET
# ============================================================

tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets["train"].column_names
)


# ============================================================
# 11. INSPECT THE RESULT
# ============================================================

print("\n--- TOKENIZED DATASET ---")

print(tokenized_datasets)

print("\n--- FIRST TOKENIZED EXAMPLE ---")

print(tokenized_datasets["train"][0])