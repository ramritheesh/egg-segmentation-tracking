
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from segment_anything import (
    sam_model_registry,
    SamAutomaticMaskGenerator
)

# --------------------------------
# 1. Configuration
# --------------------------------

DATASET_PATH = Path("data/raw/egg.v1i.folder")

CHECKPOINT = "models/sam_vit_b.pth"

OUTPUT_DIR = Path("outputs/task1_sam")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# --------------------------------
# 2. Select computation device
# --------------------------------

if torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print("Using device:", DEVICE)

# --------------------------------
# 3. Load pretrained SAM model
# --------------------------------

print("Loading SAM model...")

sam = sam_model_registry["vit_b"](
    checkpoint=CHECKPOINT
)

sam.to(device=DEVICE)

print("SAM model loaded!")

# --------------------------------
# 4. Create automatic mask generator
# --------------------------------

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=16,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=0,
    min_mask_region_area=100
)

# --------------------------------
# 5. Select one sample egg image
# --------------------------------

train_dir = DATASET_PATH / "train" / "egg"

image_paths = sorted(
    list(train_dir.glob("*.jpg"))
    + list(train_dir.glob("*.jpeg"))
    + list(train_dir.glob("*.png"))
)

if not image_paths:
    raise ValueError("No training images found!")

image_path = image_paths[0]

print("Processing:", image_path.name)

# --------------------------------
# 6. Read image
# --------------------------------

image_bgr = cv2.imread(str(image_path))

if image_bgr is None:
    raise ValueError("Unable to read image")

image_rgb = cv2.cvtColor(
    image_bgr,
    cv2.COLOR_BGR2RGB
)

# --------------------------------
# 7. Generate candidate masks
# --------------------------------

print("Generating segmentation masks...")

masks = mask_generator.generate(image_rgb)

print("Total masks generated:", len(masks))

# --------------------------------
# 8. Sort masks by area
# --------------------------------

masks = sorted(
    masks,
    key=lambda x: x["area"],
    reverse=True
)


# --------------------------------
# 9. Select the egg mask
# --------------------------------

height, width = image_rgb.shape[:2]

image_area = height * width

valid_candidates = []

for annotation in masks:

    area = annotation["area"]

    x, y, box_width, box_height = annotation["bbox"]

    area_ratio = area / image_area

    # Reject very small regions
    if area_ratio < 0.01:
        continue

    # Reject masks covering almost the whole image
    if area_ratio > 0.85:
        continue

    # Reject full-image bounding boxes
    covers_full_frame = (
        x <= 1
        and y <= 1
        and x + box_width >= width - 1
        and y + box_height >= height - 1
    )

    if covers_full_frame:
        continue

    valid_candidates.append(annotation)

if not valid_candidates:
    raise ValueError(
        "No suitable egg mask found!"
    )

# Select largest remaining candidate
egg_mask = max(
    valid_candidates,
    key=lambda x: x["area"]
)["segmentation"]

print(
    "Selected egg mask area:",
    int(egg_mask.sum())
)

# --------------------------------
# 10. Save binary egg mask
# --------------------------------

binary_mask = (
    egg_mask.astype(np.uint8) * 255
)

cv2.imwrite(
    str(OUTPUT_DIR / "egg_mask.png"),
    binary_mask
)

# --------------------------------
# 11. Create overlay visualization
# --------------------------------

overlay = image_rgb.copy()

green = np.array([0, 255, 0])

overlay[egg_mask] = (
    0.5 * overlay[egg_mask]
    + 0.5 * green
).astype(np.uint8)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Egg Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(overlay)
plt.title("Selected Egg Mask")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "selected_egg_mask.png",
    dpi=150
)

plt.show()

print("Egg mask selection completed!")



# --------------------------------
# 9. Select the egg mask
# --------------------------------

height, width = image_rgb.shape[:2]

image_area = height * width

valid_candidates = []

for annotation in masks:

    area = annotation["area"]

    x, y, box_width, box_height = annotation["bbox"]

    area_ratio = area / image_area

    # Reject very small regions
    if area_ratio < 0.01:
        continue

    # Reject masks covering almost the whole image
    if area_ratio > 0.85:
        continue

    # Reject full-image bounding boxes
    covers_full_frame = (
        x <= 1
        and y <= 1
        and x + box_width >= width - 1
        and y + box_height >= height - 1
    )

    if covers_full_frame:
        continue

    valid_candidates.append(annotation)

if not valid_candidates:
    raise ValueError(
        "No suitable egg mask found!"
    )

# Select largest remaining candidate
egg_mask = max(
    valid_candidates,
    key=lambda x: x["area"]
)["segmentation"]

print(
    "Selected egg mask area:",
    int(egg_mask.sum())
)

# --------------------------------
# 10. Save binary egg mask
# --------------------------------

binary_mask = (
    egg_mask.astype(np.uint8) * 255
)

cv2.imwrite(
    str(OUTPUT_DIR / "egg_mask.png"),
    binary_mask
)

# --------------------------------
# 11. Create overlay visualization
# --------------------------------

overlay = image_rgb.copy()

green = np.array([0, 255, 0])

overlay[egg_mask] = (
    0.5 * overlay[egg_mask]
    + 0.5 * green
).astype(np.uint8)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Egg Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(overlay)
plt.title("Selected Egg Mask")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "selected_egg_mask.png",
    dpi=150
)

plt.show()

print("Egg mask selection completed!")


# --------------------------------
# 10. Save individual candidate masks
# --------------------------------

mask_dir = OUTPUT_DIR / "candidate_masks"

mask_dir.mkdir(
    parents=True,
    exist_ok=True
)

for index, annotation in enumerate(masks):

    binary_mask = (
        annotation["segmentation"].astype(np.uint8)
        * 255
    )

    cv2.imwrite(
        str(mask_dir / f"mask_{index}.png"),
        binary_mask
    )


# --------------------------------
# 10. Inspect individual masks
# --------------------------------

print("\n========== SAM MASK DETAILS ==========")

for index, annotation in enumerate(masks):

    print(f"\nMask {index}")
    print("Area:", annotation["area"])
    print("Bounding box:", annotation["bbox"])
    print("Predicted IoU:", annotation["predicted_iou"])
    print("Stability:", annotation["stability_score"])

# --------------------------------
# 11. Display individual masks
# --------------------------------

fig, axes = plt.subplots(
    2, 3,
    figsize=(15, 10)
)

axes[0, 0].imshow(image_rgb)
axes[0, 0].set_title("Original Image")
axes[0, 0].axis("off")

for index, annotation in enumerate(masks):

    mask = annotation["segmentation"]

    row = (index + 1) // 3
    col = (index + 1) % 3

    axes[row, col].imshow(
        mask,
        cmap="gray"
    )

    axes[row, col].set_title(
        f"Mask {index} | Area: {annotation['area']}"
    )

    axes[row, col].axis("off")

# Hide unused subplot
axes[1, 2].axis("off")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "individual_masks.png",
    dpi=150
)

plt.show()

print("\nIndividual mask inspection completed!")

