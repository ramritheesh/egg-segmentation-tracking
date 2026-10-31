
import cv2
import pandas as pd
import numpy as np

from pathlib import Path

METRICS = Path(
    "outputs/task2/evaluation/dev_metrics_individual.csv"
)

OUTPUT_DIR = Path(
    "outputs/task2/failure_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT = "experiment_4_grabcut_mask"

df = pd.read_csv(METRICS)

df = df[df["experiment"] == EXPERIMENT]

# Sort from worst IoU to best IoU
df = df.sort_values("IoU", ascending=True)

print("\nWorst 10 development examples:")

print(
    df[["filename", "IoU", "Dice", "Precision", "Recall"]]
    .head(10)
    .to_string(index=False)
)

print("\nSaving failure-case comparison images...")

for rank, (_, row) in enumerate(df.head(10).iterrows(), start=1):

    filename = row["filename"]

    manifest = pd.read_csv(
        "outputs/task2/split_manifest.csv"
    )

    matching = manifest[
        manifest["filename"] == filename
    ]

    if matching.empty:
        continue

    record = matching.iloc[0]

    image_path = Path(record["image_path"])
    reference_path = Path(record["mask_path"])

    relative_path = image_path.relative_to(
        Path("data/raw/egg.v1i.folder")
    )

    prediction_path = (
        Path("outputs/task2/dev_predictions")
        / EXPERIMENT
        / relative_path.with_suffix(".png")
    )

    image = cv2.imread(str(image_path))
    reference = cv2.imread(
        str(reference_path),
        cv2.IMREAD_GRAYSCALE
    )
    prediction = cv2.imread(
        str(prediction_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None or reference is None or prediction is None:
        continue

    # Visualize SAM and prediction boundaries
    comparison = image.copy()

    comparison[reference > 0] = (
        0.5 * comparison[reference > 0]
        + 0.5 * np.array([0, 255, 0])
    ).astype(np.uint8)

    comparison[prediction > 0] = (
        0.5 * comparison[prediction > 0]
        + 0.5 * np.array([0, 0, 255])
    ).astype(np.uint8)

    output_path = (
        OUTPUT_DIR / f"failure_{rank}_{filename}.jpg"
    )

    cv2.imwrite(str(output_path), comparison)

print("\nFailure analysis images saved to:", OUTPUT_DIR)
