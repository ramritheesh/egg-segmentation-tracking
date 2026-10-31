
import cv2
import numpy as np
import pandas as pd

from pathlib import Path


# ==========================================
# 1. CONFIGURATION
# ==========================================

MANIFEST = Path("outputs/task2/split_manifest.csv")

OUTPUT_DIR = Path("outputs/task2/dev_predictions")

DATASET_ROOT = Path("data/raw/egg.v1i.folder")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(MANIFEST)
dev_df = df[df["subset"] == "dev"].copy()

print("Development images:", len(dev_df))


# ==========================================
# 2. EXPERIMENT 1: HSV THRESHOLDING
# ==========================================

def hsv_segmentation(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 0, 100])
    upper = np.array([180, 100, 255])

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel
    )

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel
    )

    return mask


# ==========================================
# 3. EXPERIMENT 2: HSV + COMPONENTS
# ==========================================

def hsv_component_segmentation(image):

    height, width = image.shape[:2]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 0, 150])
    upper = np.array([180, 100, 255])

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel
    )

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel
    )

    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask, connectivity=8
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

        if area < 0.01 * height * width:
            continue

        touches_border = (
            x == 0 or y == 0
            or x + w >= width
            or y + h >= height
        )

        if touches_border:
            continue

        if area > largest_area:
            largest_area = area
            best_label = label

    result = np.zeros((height, width), dtype=np.uint8)

    if best_label is not None:
        result[labels == best_label] = 255

    return result


# ==========================================
# 4. EXPERIMENT 3: GRABCUT RECTANGLE
# ==========================================

def grabcut_rectangle(image):

    height, width = image.shape[:2]

    margin_x = int(width * 0.10)
    margin_y = int(height * 0.10)

    rect = (
        margin_x,
        margin_y,
        width - 2 * margin_x,
        height - 2 * margin_y
    )

    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        image,
        mask,
        rect,
        bg_model,
        fg_model,
        5,
        cv2.GC_INIT_WITH_RECT
    )

    result = np.where(
        (mask == cv2.GC_FGD) |
        (mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)

    result = cv2.morphologyEx(
        result, cv2.MORPH_OPEN, kernel
    )

    result = cv2.morphologyEx(
        result, cv2.MORPH_CLOSE, kernel
    )

    return result


# ==========================================
# 5. EXPERIMENT 4: GRABCUT MASK INITIALIZATION
# ==========================================

def grabcut_mask_initialization(image):

    height, width = image.shape[:2]

    mask = np.full(
        (height, width),
        cv2.GC_PR_BGD,
        dtype=np.uint8
    )

    border = int(min(height, width) * 0.05)

    mask[:border, :] = cv2.GC_BGD
    mask[-border:, :] = cv2.GC_BGD
    mask[:, :border] = cv2.GC_BGD
    mask[:, -border:] = cv2.GC_BGD

    x1 = int(width * 0.20)
    x2 = int(width * 0.80)

    y1 = int(height * 0.20)
    y2 = int(height * 0.80)

    mask[y1:y2, x1:x2] = cv2.GC_PR_FGD

    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        image,
        mask,
        None,
        bg_model,
        fg_model,
        5,
        cv2.GC_INIT_WITH_MASK
    )

    result = np.where(
        (mask == cv2.GC_FGD) |
        (mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)

    result = cv2.morphologyEx(
        result, cv2.MORPH_OPEN, kernel
    )

    result = cv2.morphologyEx(
        result, cv2.MORPH_CLOSE, kernel
    )

    return result


# ==========================================
# 6. REGISTER EXPERIMENTS
# ==========================================

EXPERIMENTS = {
    "experiment_1_hsv": hsv_segmentation,
    "experiment_2_hsv_components": hsv_component_segmentation,
    "experiment_3_grabcut_rectangle": grabcut_rectangle,
    "experiment_4_grabcut_mask": grabcut_mask_initialization
}


# ==========================================
# 7. RUN ALL EXPERIMENTS ON DEV SET
# ==========================================

for experiment_name, algorithm in EXPERIMENTS.items():

    print("\nRunning:", experiment_name)

    experiment_dir = OUTPUT_DIR / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    error_count = 0

    for index, (_, row) in enumerate(
        dev_df.iterrows(), start=1
    ):

        image_path = Path(row["image_path"])

        relative_path = image_path.relative_to(
            DATASET_ROOT
        )

        output_path = (
            experiment_dir / relative_path.with_suffix(".png")
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if output_path.exists():
            success_count += 1
            continue

        try:

            image = cv2.imread(str(image_path))

            if image is None:
                raise ValueError("Could not read image")

            prediction = algorithm(image)

            if not cv2.imwrite(
                str(output_path), prediction
            ):
                raise IOError("Could not save prediction")

            success_count += 1

        except Exception as error:

            error_count += 1

            print(
                f"Error on {image_path.name}: {error}"
            )

        if index % 10 == 0:
            print(
                f"{index}/{len(dev_df)} images processed"
            )

    print(
        f"{experiment_name}: "
        f"{success_count} successful, "
        f"{error_count} errors"
    )


print("\nAll development experiments finished!")
