
import numpy as np
from PIL import Image

from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)


def load_model():
    return MobileNetV2(weights="imagenet")


def recognize_image(image, model):
    # Resize to MobileNetV2 input dimensions
    image = image.convert("RGB")
    resized = image.resize((224, 224))

    # Convert image into a numerical array
    image_array = np.asarray(
        resized, dtype=np.float32
    )

    # Add the batch dimension
    image_array = np.expand_dims(
        image_array, axis=0
    )

    # Apply model-specific preprocessing
    image_array = preprocess_input(image_array)

    # Make a real CNN prediction
    predictions = model.predict(
        image_array, verbose=0
    )

    # Return the top 5 ImageNet predictions
    results = decode_predictions(
        predictions, top=5
    )[0]

    return [
        {
            "category": name.replace("_", " ").title(),
            "confidence": float(score)
        }
        for _, name, score in results
    ]


def compare_cats_dogs(image, model):
    image = image.convert("RGB").resize((224, 224))

    array = np.asarray(image, dtype=np.float32)
    array = np.expand_dims(array, axis=0)
    array = preprocess_input(array)

    predictions = model.predict(array, verbose=0)

    # Check all ImageNet classes
    results = decode_predictions(
        predictions, top=1000
    )[0]

    cat_ids = {
        "n02123045", "n02123159",
        "n02123394", "n02123597",
        "n02124075"
    }

    cat_score = 0.0
    dog_score = 0.0

    for class_id, name, score in results:
        if class_id in cat_ids:
            cat_score += float(score)

        elif "n02085620" <= class_id <= "n02113978":
            dog_score += float(score)

    total = cat_score + dog_score

    if total < 0.10:
        return {
            "prediction": "Uncertain",
            "cat": 0.0,
            "dog": 0.0,
            "evidence": total
        }

    cat = cat_score / total
    dog = dog_score / total

    return {
        "prediction": "Cat" if cat > dog else "Dog",
        "cat": cat,
        "dog": dog,
        "evidence": total
    }


from pathlib import Path
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent.parent
CUSTOM_MODEL_PATH = BASE_DIR / "models" / "cats_vs_dogs.keras"


def load_custom_cnn():
    if not CUSTOM_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Custom CNN model not found. Train the model first."
        )

    return tf.keras.models.load_model(
        CUSTOM_MODEL_PATH,
        compile=False
    )


def predict_custom_cnn(image, model):
    # Match the input size used during training
    resized = image.convert("RGB").resize((128, 128))

    image_array = np.asarray(resized, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    # The saved model already contains Rescaling
    # so we do NOT divide pixels by 255 here.
    dog_score = float(
        model.predict(image_array, verbose=0)[0][0]
    )

    cat_score = 1.0 - dog_score

    return {
        "prediction": "Dog" if dog_score >= 0.5 else "Cat",
        "cat": cat_score,
        "dog": dog_score
    }
