
from transformers import pipeline

# Diverse categories for flexible recognition
DEFAULT_CATEGORIES = [
    "a real dog",
    "a cartoon dog",
    "a real cat",
    "a cartoon cat",
    "a rabbit or bunny",
    "a cartoon rabbit",
    "a frog",
    "a cartoon frog",
    "a bird",
    "a horse",
    "a cow",
    "a turtle",
    "a fish",
    "a car",
    "a motorcycle",
    "a bicycle",
    "an airplane",
    "a building",
    "a flower",
    "a person",
    "a toy animal",
    "a stuffed animal"
]


def load_flexible_model():
    return pipeline(
        task="zero-shot-image-classification",
        model="openai/clip-vit-base-patch32",
        framework="pt",
        device=-1
    )


def recognize_flexible(
    image,
    classifier,
    categories=None
):
    if categories is None:
        categories = DEFAULT_CATEGORIES

    categories = [
        category.strip()
        for category in categories
        if category.strip()
    ]

    if len(categories) < 2:
        raise ValueError(
            "Please provide at least two categories."
        )

    if len(set(categories)) != len(categories):
        raise ValueError(
            "Categories must be unique."
        )

    predictions = classifier(
        image.convert("RGB"),
        candidate_labels=categories,
        hypothesis_template="This is an image of {}."
    )

    return [
        {
            "category": item["label"],
            "score": float(item["score"])
        }
        for item in predictions[:5]
    ]
