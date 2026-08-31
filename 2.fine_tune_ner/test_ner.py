from transformers import pipeline


model_path = "./bert-ner-final"


token_classifier = pipeline(
    "token-classification",
    model=model_path,
    tokenizer=model_path,
    aggregation_strategy="simple",
)


texts = [
    "My name is John and I work at Google in New York.",
    "Microsoft opened a new office in Sydney.",
    "Barack Obama visited Paris.",
    "Elon Musk founded SpaceX.",
    "Apple opened a store in London",
    "I ate an apple in London."
]


for text in texts:

    print("\nTEXT:")
    print(text)

    predictions = token_classifier(text)

    print("ENTITIES:")

    for prediction in predictions:
        print(
            f"{prediction['word']} "
            f"→ {prediction['entity_group']} "
            f"(score: {prediction['score']:.4f})"
        )
