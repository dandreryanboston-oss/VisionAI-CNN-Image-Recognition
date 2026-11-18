
import time
from transformers import pipeline

print("Starting CLIP test...", flush=True)

start = time.perf_counter()

print("Loading CLIP model...", flush=True)

classifier = pipeline(
    task="zero-shot-image-classification",
    model="openai/clip-vit-base-patch32",
    framework="pt",
    device=-1
)

print("CLIP loaded successfully!", flush=True)
print(
    f"Loading took {time.perf_counter() - start:.2f} seconds",
    flush=True
)
