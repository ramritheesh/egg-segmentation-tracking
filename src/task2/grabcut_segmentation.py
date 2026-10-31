
import cv2
import numpy as np
from pathlib import Path


# ==========================================
# EXPERIMENT 3: GRABCUT SEGMENTATION
# ==========================================

def grabcut_segmentation(image_path):

    # Read image
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError("Could not read image")

    height, width = image.shape[:2]

    # Define rectangle around central object.
    # This is an initialization, not an egg detector.
    margin_x = int(width * 0.10)
    margin_y = int(height * 0.10)

    rect = (
        margin_x,
        margin_y,
        width - 2 * margin_x,
        height - 2 * margin_y
    )

    # Initialize GrabCut mask
    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    # Background and foreground models
    background_model = np.zeros(
        (1, 65),
        dtype=np.float64
    )

    foreground_model = np.zeros(
        (1, 65),
        dtype=np.float64
    )

    # Apply GrabCut
    cv2.grabCut(
        image,
        mask,
        rect,
        background_model,
        foreground_model,
        5,
        cv2.GC_INIT_WITH_RECT
    )

    # Extract foreground pixels
    result = np.where(
        (mask == cv2.GC_FGD) |
        (mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype(np.uint8)

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)

    result = cv2.morphologyEx(
        result,
        cv2.MORPH_OPEN,
        kernel
    )

    result = cv2.morphologyEx(
        result,
        cv2.MORPH_CLOSE,
        kernel
    )

    return result


# ==========================================
# TEST ON SAME IMAGE
# ==========================================

DATASET_DIR = Path("data/raw/egg.v1i.folder")

OUTPUT_DIR = Path("outputs/task2/experiment_3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

image_paths = sorted(
    (DATASET_DIR / "train" / "egg").glob("*.jpg")
)

if not image_paths:
    raise FileNotFoundError("No training images found")

image_path = image_paths[0]

result = grabcut_segmentation(image_path)

output_path = OUTPUT_DIR / "grabcut_mask.png"

cv2.imwrite(
    str(output_path),
    result
)

print("GrabCut experiment completed!")
print("Input:", image_path)
print("Output:", output_path)
