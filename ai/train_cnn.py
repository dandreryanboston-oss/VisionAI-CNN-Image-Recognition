
import json
from pathlib import Path

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

# =====================================
# 1. PROJECT CONFIGURATION
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset" / "cats_and_dogs_filtered"
TRAIN_DIR = DATASET_DIR / "train"
HELD_OUT_DIR = DATASET_DIR / "validation"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42

tf.keras.utils.set_random_seed(SEED)

print("Starting VisionAI CNN Training...")

# =====================================
# 2. LOAD DATASETS
# =====================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
    seed=SEED
)

# Split the original validation directory
# into separate validation and test sets.
validation_ds = tf.keras.utils.image_dataset_from_directory(
    HELD_OUT_DIR,
    validation_split=0.5,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    HELD_OUT_DIR,
    validation_split=0.5,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

print("Class names:", train_ds.class_names)
print("Label 0 = Cat, Label 1 = Dog")

# Improve data loading performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
validation_ds = validation_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# =====================================
# 3. BUILD OUR CUSTOM CNN
# =====================================

model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),

    # Image augmentation during training
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),

    # Normalize pixel values to 0-1
    layers.Rescaling(1.0 / 255),

    # First convolution block
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    # Second convolution block
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    # Third convolution block
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    # Fourth convolution block
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    # Classification layers
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    # Output: probability of Dog
    layers.Dense(1, activation="sigmoid")
])

# =====================================
# 4. COMPILE THE MODEL
# =====================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =====================================
# 5. TRAIN THE CNN
# =====================================

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        str(MODEL_DIR / "cats_vs_dogs.keras"),
        monitor="val_loss",
        save_best_only=True
    )
]

history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# =====================================
# 6. EVALUATE ON UNSEEN TEST IMAGES
# =====================================

test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1
)

print(f"Test accuracy: {test_accuracy:.2%}")
print(f"Test loss: {test_loss:.4f}")

# =====================================
# 7. SAVE METRICS AND TRAINING GRAPHS
# =====================================

metrics = {
    "class_names": ["cats", "dogs"],
    "test_accuracy": float(test_accuracy),
    "test_loss": float(test_loss),
    "history": {
        key: [float(v) for v in values]
        for key, values in history.history.items()
    }
}

with open(
    MODEL_DIR / "training_metrics.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(metrics, file, indent=2)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("CNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("CNN Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig(MODEL_DIR / "training_results.png")
plt.close()

print("Training completed!")
print("Model saved in models/cats_vs_dogs.keras")
print("Graphs saved in models/training_results.png")
