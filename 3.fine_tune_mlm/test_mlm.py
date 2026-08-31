"""
GOAL: Load the fine-tuned MLM model (saved by fine_tune_mlm.py) and test
it on new sentences, to SEE whether domain adaptation actually worked.

PIPELINE POSITION:
    fine_tune_mlm.py  -> trains + saves model to ./bert-mlm-imdb-final
    test_mlm.py       -> loads that saved model and uses it

This is the "Prediction" step at the end of the pipeline:
    Dataset -> Preprocessing -> Tokenizer -> Model -> Training ->
    Evaluation -> Save -> PREDICTION 

CONCEPT REMINDER:
    fill-mask = give the model a sentence with [MASK] in it,
    it predicts what word most likely goes there.

    If fine-tuning worked, predictions should lean toward
    movie-review vocabulary (e.g. "movie", "film", "actor")
    more than a generic pretrained BERT would.
"""

from transformers import pipeline

# ===========================================================
# 1. LOAD THE FINE-TUNED MODEL
# ===========================================================
# Point this at wherever fine_tune_mlm.py saved the model.
model_path = "./bert-mlm-imdb-final"

mask_filler = pipeline(
    "fill-mask",
    model=model_path,
    tokenizer=model_path,
)


# ===========================================================
# 2. TEST SENTENCES
# ===========================================================
# Each sentence needs exactly one [MASK] token.
# Try a mix: some clearly movie-related, some more neutral,
# to compare how domain-specific the predictions are.
test_sentences = [
    "This is a great [MASK].",
    "The acting in this movie was absolutely [MASK].",
    "I would not recommend this [MASK] to anyone.",
    "The [MASK] was delicious.",  # non-movie sentence, for contrast
]


# ===========================================================
# 3. RUN PREDICTIONS
# ===========================================================
if __name__ == "__main__":
    for text in test_sentences:
        print(f"\n>>> Input: {text}")
        preds = mask_filler(text)

        # Each pred contains: sequence (full filled sentence),
        # token_str (the predicted word), and score (confidence)
        for pred in preds:
            print(f"    {pred['score']:.3f}  ->  {pred['sequence']}")