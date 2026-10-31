
import cv2
import pandas as pd
import numpy as np
from pathlib import Path

manifest = pd.read_csv("outputs/task2/split_manifest.csv")
results = pd.read_csv(
    "outputs/task1_selection_experiment/selection_comparison.csv"
)

output = Path("outputs/task1_selection_experiment/visual_comparison")
output.mkdir(parents=True, exist_ok=True)

strategies = [
    "largest_area",
    "highest_predicted_iou",
    "combined_confidence"
]

for filename in results["filename"].unique():

    row = manifest[manifest["filename"] == filename].iloc[0]
    image = cv2.imread(row["image_path"])

    if image is None:
        continue

    image = cv2.resize(image, (320, 320))

    panels = []

    for strategy in strategies:

        record = results[
            (results["filename"] == filename) &
            (results["strategy"] == strategy)
        ].iloc[0]

        mask = cv2.imread(record["mask_path"], 0)
        mask = cv2.resize(mask, (320, 320))

        overlay = image.copy()
        foreground = mask > 0

        overlay[foreground] = (
            0.5 * overlay[foreground] +
            0.5 * np.array([0, 255, 0])
        ).astype(np.uint8)

        cv2.putText(
            overlay, strategy, (5, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (255, 255, 255), 1
        )

        panels.append(overlay)

    comparison = np.hstack(panels)

    cv2.imwrite(
        str(output / f"{Path(filename).stem}.jpg"),
        comparison
    )

print("Visual comparisons saved:", output)
