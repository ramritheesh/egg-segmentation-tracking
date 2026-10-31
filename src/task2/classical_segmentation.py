
import cv2
import numpy as np
from pathlib import Path


def hsv_component_segmentation(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError("Could not read image")

    height, width = image.shape[:2]

    # Convert image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Stricter threshold
    lower = np.array([0, 0, 150])
    upper = np.array([180, 100, 255])

    mask = cv2.inRange(hsv, lower, upper)

    # Remove noise
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Close small gaps
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Connected component analysis
    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    best_label = None
    largest_area = 0

    for label in range(1, count):

        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]

        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]

        area = stats[label, cv2.CC_STAT_AREA]

        # Reject very small regions
        if area < 0.01 * height * width:
            continue

        # Reject components touching image borders
        touches_border = (
            x == 0
            or y == 0
            or x + w >= width
            or y + h >= height
        )

        if touches_border:
            continue

        # Select largest remaining component
        if area > largest_area:
            largest_area = area
            best_label = label

    result = np.zeros((height, width), dtype=np.uint8)

    if best_label is not None:
        result[labels == best_label] = 255

    return result


# ==========================================
# TEST ON SAME IMAGE
# ==========================================

DATASET_DIR = Path("data/raw/egg.v1i.folder")

OUTPUT_DIR = Path("outputs/task2/experiment_2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

image_paths = sorted(
    (DATASET_DIR / "train" / "egg").glob("*.jpg")
)

image_path = image_paths[0]

mask = hsv_component_segmentation(image_path)

cv2.imwrite(
    str(OUTPUT_DIR / "hsv_component_mask.png"),
    mask
)

print("Experiment 2 completed!")
print("Input:", image_path)
print("Output:", OUTPUT_DIR / "hsv_component_mask.png")
