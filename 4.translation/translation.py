from huggingface_hub import hf_hub_download
from datasets import load_dataset

local_path = hf_hub_download(
    repo_id="Helsinki-NLP/kde4",
    repo_type="dataset",
    filename="en-fr/train/0000.parquet",
    revision="refs/convert/parquet",
)

raw_datasets = load_dataset("parquet", data_files={"train": local_path})

split_datasets = raw_datasets["train"].train_test_split(train_size=0.9, seed=10)

split_datasets["validation"] = split_datasets.pop("test")

print("split_datasets", split_datasets)

print("split_datasets['train'][0][translation]", split_datasets["train"][0]["translation"])

# Preprocessing the datasets
from transformers import AutoTokenizer
model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, return_tensors="pt")

en_sentence = split_datasets["train"][0]["translation"]["en"]
fr_sentence = split_datasets["train"][0]["translation"]["fr"]

inputs = tokenizer(en_sentence, text_target=fr_sentence)

wrong_targets = tokenizer(fr_sentence)

print(tokenizer.convert_ids_to_tokens(wrong_targets["input_ids"]))
print(tokenizer.convert_ids_to_tokens(inputs["labels"]))

max_length = 128

def preprocess_function(examples):
    inputs = [ex["en"] for ex in examples["translation"]]
    targets = [ex["fr"] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=max_length, truncation=True)

    return model_inputs

tokenized_datasets = split_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=split_datasets["train"].column_names,
)

from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

batch = data_collator([tokenized_datasets["train"][i] for i in range(1, 3)])
print("batch.keys::::::::", batch.keys())

print("batch[labels]::::::::", batch["labels"])

print("batch[decoder_input_ids]::::::::", batch["decoder_input_ids"])

for i in range(1, 3):
    print(tokenized_datasets["train"][i]["labels"])

import evaluate

metric = evaluate.load("sacrebleu")

predictions = [
    "This plugin lets you translate web pages between several languages automatically."
]
references = [
    [
        "This plugin allows you to automatically translate web pages between several languages."
    ]
]
metric.compute(predictions=predictions, references=references)
print("metric.compute(predictions=predictions, references=references)", metric.compute(predictions=predictions, references=references))

import numpy as np

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)

    result = {"bleu": result["score"]}

    