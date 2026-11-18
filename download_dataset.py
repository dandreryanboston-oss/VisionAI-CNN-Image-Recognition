
import os
import zipfile
import urllib.request
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent

# Dataset directory
DATASET_DIR = BASE_DIR / "dataset"
DATASET_DIR.mkdir(exist_ok=True)

# Official TensorFlow tutorial dataset
URL = (
    "https://storage.googleapis.com/tensorflow-1-public/"
    "course2/cats_and_dogs_filtered.zip"
)

ZIP_PATH = DATASET_DIR / "cats_and_dogs.zip"

print("Downloading Cats vs Dogs dataset...")

# Download only if not already downloaded
if not ZIP_PATH.exists():
    urllib.request.urlretrieve(URL, ZIP_PATH)
    print("Download completed!")
else:
    print("Dataset ZIP already downloaded.")

# Extract the dataset
print("Extracting images...")

with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
    zip_ref.extractall(DATASET_DIR)

print("Extraction completed!")

# Check image counts
root = DATASET_DIR / "cats_and_dogs_filtered"

for split in ["train", "validation"]:
    for category in ["cats", "dogs"]:
        folder = root / split / category

        images = [
            f for f in folder.iterdir()
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]

        print(
            f"{split}/{category}: {len(images)} images"
        )

print("\nDataset ready for CNN training!")
