"""
test_translation.py

GOAL: Load the fine-tuned Marian translation model (saved by
translation.py) and test it on new English sentences, without
re-running the dataset loading / tokenization / training steps.

PIPELINE POSITION:
    translation.py      -> trains + saves model to ./marian-finetuned-kde4-final
    test_translation.py -> loads that saved model and uses it
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ===========================================================
# 1. LOAD THE FINE-TUNED MODEL
# ===========================================================
save_path = "./marian-finetuned-kde4-final" 

print(f"Loading fine-tuned model from {save_path} ...")
tokenizer = AutoTokenizer.from_pretrained(save_path)
model = AutoModelForSeq2SeqLM.from_pretrained(save_path)


# ===========================================================
# 2. TEST SENTENCES
# ===========================================================
# A mix of general and KDE4-technical sentences, so you can see
# whether fine-tuning shifted translations toward the dataset's
# style (e.g. "fils de discussion" instead of "threads").
test_sentences = [
    "Default to expanded threads",
    "Unable to import %1 using the OFX importer plugin. This file is not the correct format.",
    "Please enter your email address.",
    "The file could not be saved.",
]


# ===========================================================
# 3. TRANSLATE
# ===========================================================
def translate(text, max_length=128):
    inputs = tokenizer(text, return_tensors="pt")
    output_ids = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    for text in test_sentences:
        translation = translate(text)
        print(f"\n>>> Input:  {text}")
        print(f">>> Output: {translation}")