
import json
import csv
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input,
    decode_predictions
)

from classifier import load_model, load_custom_cnn


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = (
    BASE_DIR / "dataset"
    / "cats_and_dogs_filtered"
    / "validation"
)
OUTPUT_DIR = BASE_DIR / "models"
OUTPUT_DIR.mkdir(exist_ok=True)

SEED = 42
BATCH_SIZE = 32

CLASS_NAMES = ["Cat", "Dog", "Uncertain"]

CAT_IDS = {
    "n02123045",
    "n02123159",
    "n02123394",
    "n02123597",
    "n02124075"
}

tf.keras.utils.set_random_seed(SEED)


# ==========================================
# 2. RECOVER THE EXACT TEST SPLIT
# ==========================================

print("\nLoading the reserved test images...")

test_reference = (
    tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.5,
        subset="validation",
        seed=SEED,
        image_size=(128, 128),
        batch_size=BATCH_SIZE,
        label_mode="binary"
    )
)

# These are the selected test files.
# Sorting ensures identical order for both models.
test_paths = sorted(test_reference.file_paths)

y_true = np.array([
    0 if Path(p).parent.name == "cats" else 1
    for p in test_paths
], dtype=np.int32)

print(f"Test images: {len(test_paths)}")
print(f"Cats: {np.sum(y_true == 0)}")
print(f"Dogs: {np.sum(y_true == 1)}")

if len(test_paths) != 500:
    raise RuntimeError(
        "Expected 500 test images. "
        "Check the dataset and split configuration."
    )


# ==========================================
# 3. IMAGE PREPROCESSING
# ==========================================

def create_dataset(paths, image_size, mobilenet=False):

    ds = tf.data.Dataset.from_tensor_slices(paths)

    def process(path):
        image_data = tf.io.read_file(path)

        image = tf.io.decode_image(
            image_data,
            channels=3,
            expand_animations=False
        )

        image.set_shape([None, None, 3])

        image = tf.image.resize(
            image,
            image_size,
            method="bilinear"
        )

        image = tf.cast(image, tf.float32)

        if mobilenet:
            image = preprocess_input(image)

        # Custom CNN already contains Rescaling
        return image

    ds = ds.map(
        process,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return ds.batch(BATCH_SIZE).prefetch(
        tf.data.AUTOTUNE
    )


# ==========================================
# 4. IMAGENET CAT/DOG CLASS MAPPING
# ==========================================

# Decode all 1000 class IDs once.
# Distinct scores preserve each original index.
reference_scores = np.arange(
    1000, dtype=np.float32
).reshape(1, 1000)

decoded = decode_predictions(
    reference_scores,
    top=1000
)[0]

# Results are ordered from index 999 to 0.
class_to_index = {
    class_id: 999 - position
    for position, (class_id, _, _) in enumerate(decoded)
}

cat_indices = [
    class_to_index[class_id]
    for class_id in CAT_IDS
]

dog_indices = [
    index
    for class_id, index in class_to_index.items()
    if "n02085620" <= class_id <= "n02113978"
]


# ==========================================
# 5. EVALUATION FUNCTIONS
# ==========================================

def evaluate_mobilenet(model, dataset):
    # Warm up the model before timing.
    dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
    model.predict(dummy, verbose=0)

    start = time.perf_counter()

    predictions = model.predict(
        dataset, verbose=1
    )

    elapsed = time.perf_counter() - start

    cat_scores = predictions[:, cat_indices].sum(axis=1)
    dog_scores = predictions[:, dog_indices].sum(axis=1)

    evidence = cat_scores + dog_scores

    labels = np.where(
        evidence < 0.10,
        2,  # Uncertain
        np.where(cat_scores > dog_scores, 0, 1)
    )

    return labels.astype(int), elapsed


def evaluate_custom(model, dataset):
    # Warm up the model before timing.
    dummy = np.zeros((1, 128, 128, 3), dtype=np.float32)
    model.predict(dummy, verbose=0)

    start = time.perf_counter()

    predictions = model.predict(
        dataset, verbose=1
    ).reshape(-1)

    elapsed = time.perf_counter() - start

    labels = (predictions >= 0.5).astype(int)

    return labels, elapsed


def calculate_metrics(y_true, y_pred, elapsed):

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[0, 1],
            zero_division=0
        )
    )

    matrix = np.zeros((2, 3), dtype=int)

    for actual, predicted in zip(y_true, y_pred):
        matrix[int(actual), int(predicted)] += 1

    rejected = int(np.sum(y_pred == 2))

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(np.mean(f1)),
        "rejected_images": rejected,
        "rejection_rate": float(
            rejected / len(y_true)
        ),
        "inference_seconds": float(elapsed),
        "seconds_per_image": float(
            elapsed / len(y_true)
        ),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "support": support.tolist(),
        "confusion_matrix": matrix.tolist()
    }


# ==========================================
# 6. LOAD BOTH CNN MODELS
# ==========================================

print("\nLoading MobileNetV2...")
mobilenet = load_model()

print("Loading your custom CNN...")
custom_cnn = load_custom_cnn()

mobilenet_ds = create_dataset(
    test_paths,
    (224, 224),
    mobilenet=True
)

custom_ds = create_dataset(
    test_paths,
    (128, 128),
    mobilenet=False
)


# ==========================================
# 7. RUN THE EVALUATION
# ==========================================

print("\nEvaluating MobileNetV2...")

pred_a, time_a = evaluate_mobilenet(
    mobilenet,
    mobilenet_ds
)

print("\nEvaluating Custom CNN...")

pred_b, time_b = evaluate_custom(
    custom_cnn,
    custom_ds
)

results = {
    "sample_count": len(test_paths),
    "class_names": CLASS_NAMES,
    "mobilenet": calculate_metrics(
        y_true, pred_a, time_a
    ),
    "custom_cnn": calculate_metrics(
        y_true, pred_b, time_b
    )
}


# ==========================================
# 8. PRINT RESULTS
# ==========================================

for name, data in [
    ("MobileNetV2", results["mobilenet"]),
    ("Custom CNN", results["custom_cnn"])
]:
    print("\n" + "=" * 45)
    print(name)
    print("=" * 45)

    print(f"Accuracy: {data['accuracy']:.2%}")
    print(f"Macro F1: {data['macro_f1']:.4f}")

    print(
        f"Rejected images: "
        f"{data['rejected_images']}"
    )

    print(
        f"Rejection rate: "
        f"{data['rejection_rate']:.2%}"
    )

    print(
        f"Inference time: "
        f"{data['inference_seconds']:.3f}s"
    )

    for i, category in enumerate(["Cat", "Dog"]):
        print(
            f"{category}: "
            f"Precision={data['precision'][i]:.3f}, "
            f"Recall={data['recall'][i]:.3f}, "
            f"F1={data['f1'][i]:.3f}"
        )


# ==========================================
# 9. SAVE RESULTS
# ==========================================

with open(
    OUTPUT_DIR / "model_comparison.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(results, file, indent=2)

with open(
    OUTPUT_DIR / "model_predictions.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "image",
        "actual",
        "mobilenet_prediction",
        "custom_cnn_prediction"
    ])

    for path, actual, a, b in zip(
        test_paths, y_true, pred_a, pred_b
    ):
        writer.writerow([
            Path(path).name,
            CLASS_NAMES[actual],
            CLASS_NAMES[a],
            CLASS_NAMES[b]
        ])


# ==========================================
# 10. CONFUSION MATRICES
# ==========================================

fig, axes = plt.subplots(
    1, 2, figsize=(13, 5)
)

for ax, title, key in [
    (axes[0], "MobileNetV2", "mobilenet"),
    (axes[1], "Custom CNN", "custom_cnn")
]:
    matrix = np.array(
        results[key]["confusion_matrix"]
    )

    ax.imshow(matrix, cmap="Blues")

    ax.set_title(title)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")

    ax.set_xticks(range(3))
    ax.set_xticklabels(CLASS_NAMES)

    ax.set_yticks(range(2))
    ax.set_yticklabels(["Cat", "Dog"])

    for i in range(2):
        for j in range(3):
            ax.text(
                j, i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=12
            )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "confusion_matrices.png",
    dpi=160
)

plt.close()

print("\nEvaluation completed successfully!")
print("Results saved in the models folder.")
