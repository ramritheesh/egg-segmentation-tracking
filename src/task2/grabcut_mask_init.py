
import cv2
import numpy as np
from pathlib import Path


def grabcut_with_mask_initialization(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError("Could not read image")

    height, width = image.shape[:2]

    # Initialize every pixel as probable background
    mask = np.full(
        (height, width),
        cv2.GC_PR_BGD,
        dtype=np.uint8
    )

    # Mark outer border as definite background
    border = int(min(height, width) * 0.05)

    mask[:border, :] = cv2.GC_BGD
    mask[-border:, :] = cv2.GC_BGD
    mask[:, :border] = cv2.GC_BGD
    mask[:, -border:] = cv2.GC_BGD

    # Mark central region as probable foreground
    x1 = int(width * 0.20)
    x2 = int(width * 0.80)

    y1 = int(height * 0.20)
    y2 = int(height * 0.80)

    mask[y1:y2, x1:x2] = cv2.GC_PR_FGD

    # Background and foreground models
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)

    # Run GrabCut using mask initialization
    cv2.grabCut(
        image,
        mask,
        None,
        background_model,
        foreground_model,
        5,
        cv2.GC_INIT_WITH_MASK
    )

    # Extract foreground
    result = np.where(
        (mask == cv2.GC_FGD) |
        (mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype(np.uint8)

    # Remove small noise
    kernel = np.ones((5, 5), np.uint8)

    result = cv2.morphologyEx(
        result,
        cv2.MORPH_OPEN,
        kernel
    )

    # Fill small gaps
    result = cv2.morphologyEx(
        result,
        cv2.MORPH_CLOSE,
        kernel
    )

    return result


# ==========================================
# TEST
# ==========================================

DATASET_DIR = Path("data/raw/egg.v1i.folder")

OUTPUT_DIR = Path("outputs/task2/experiment_4")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

image_paths = sorted(
    (DATASET_DIR / "train" / "egg").glob("*.jpg")
)

if not image_paths:
    raise FileNotFoundError("No training images found")

image_path = image_paths[0]

result = grabcut_with_mask_initialization(image_path)

output_path = OUTPUT_DIR / "grabcut_mask_init.png"

cv2.imwrite(str(output_path), result)

print("Experiment 4 completed!")
print("Input:", image_path)
print("Output:", output_path)
