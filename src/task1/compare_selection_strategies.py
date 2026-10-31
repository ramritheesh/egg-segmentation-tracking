
import cv2
import torch
import numpy as np
import pandas as pd

from pathlib import Path
from segment_anything import (
    sam_model_registry,
    SamAutomaticMaskGenerator
)

# Configuration
CHECKPOINT = "models/sam_vit_b.pth"
MANIFEST = "outputs/task2/split_manifest.csv"
OUTPUT = Path("outputs/task1_selection_experiment")

OUTPUT.mkdir(parents=True, exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Load only development egg images.
df = pd.read_csv(MANIFEST)
df = df[(df["subset"] == "dev") & (df["class"] == "egg")]

# Deterministic sample of 10 images.
sample = df.sample(n=10, random_state=42)

print("Selected development images:", len(sample))

# Load SAM once.
sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT)
sam.to(device=DEVICE)

generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=16,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=0,
    min_mask_region_area=100
)

records = []

for _, row in sample.iterrows():

    image_path = Path(row["image_path"])
    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not read:", image_path)
        continue

    height, width = image.shape[:2]
    image_area = height * width

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = generator.generate(image_rgb)

    # Apply the same filters as the baseline.
    valid = []

    for candidate in masks:

        area = candidate["area"]
        x, y, box_width, box_height = candidate["bbox"]

        area_ratio = area / image_area

        if area_ratio < 0.01 or area_ratio > 0.85:
            continue

        covers_full_frame = (
            x <= 1 and y <= 1
            and x + box_width >= width - 1
            and y + box_height >= height - 1
        )

        if covers_full_frame:
            continue

        valid.append(candidate)

    if not valid:
        print("No valid candidates:", image_path.name)
        continue

    strategies = {
        "largest_area": max(
            valid, key=lambda c: c["area"]
        ),
        "highest_predicted_iou": max(
            valid, key=lambda c: c["predicted_iou"]
        ),
        "combined_confidence": max(
            valid,
            key=lambda c: (
                c["predicted_iou"] + c["stability_score"]
            ) / 2
        )
    }

    for strategy_name, selected in strategies.items():

        mask = selected["segmentation"].astype(np.uint8) * 255

        output_path = (
            OUTPUT / strategy_name /
            f"{image_path.stem}.png"
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), mask)

        records.append({
            "filename": image_path.name,
            "strategy": strategy_name,
            "area": selected["area"],
            "predicted_iou": selected["predicted_iou"],
            "stability_score": selected["stability_score"],
            "bbox": str(selected["bbox"]),
            "mask_path": str(output_path)
        })

    print("Processed:", image_path.name)

pd.DataFrame(records).to_csv(
    OUTPUT / "selection_comparison.csv",
    index=False
)

print("\nExperiment completed.")
print("Results:", OUTPUT / "selection_comparison.csv")
