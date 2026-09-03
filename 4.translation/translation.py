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