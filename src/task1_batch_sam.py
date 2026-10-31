
import cv2
import torch
import numpy as np
import pandas as pd

from pathlib import Path
from segment_anything import (
    sam_model_registry,
    SamAutomaticMaskGenerator
)

# ==========================================
# 1. CONFIGURATION
# ==========================================

DATASET_PATH = Path("data/raw/egg.v1i.folder")
CHECKPOINT = Path("models/sam_vit_b.pth")

OUTPUT_DIR = Path("outputs/task1_sam_batch")

MASK_DIR = OUTPUT_DIR / "masks"
OVERLAY_DIR = OUTPUT_DIR / "overlays"
CANDIDATE_DIR = OUTPUT_DIR / "candidates"

RESULTS_CSV = OUTPUT_DIR / "sam_results.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

for directory in [MASK_DIR, OVERLAY_DIR, CANDIDATE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. DEVICE CONFIGURATION
# ==========================================

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

print("=" * 50)
print("EGG SEGMENTATION - SAM BATCH PROCESSING")
print("=" * 50)

print("Using device:", DEVICE)


# ==========================================
# 3. LOAD SAM MODEL ONCE
# ==========================================

print("\nLoading SAM ViT-B model...")

sam = sam_model_registry["vit_b"](
    checkpoint=str(CHECKPOINT)
)

sam.to(device=DEVICE)

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=16,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=0,
    min_mask_region_area=100
)

print("SAM model loaded successfully!")


# ==========================================
# 4. FIND ALL DATASET IMAGES
# ==========================================

image_paths = sorted([
    path
    for path in DATASET_PATH.rglob("*")
    if path.is_file()
    and path.suffix.lower() in IMAGE_EXTENSIONS
])

print("\nTotal images found:", len(image_paths))


# ==========================================
# 5. EGG MASK SELECTION FUNCTION
# ==========================================

def select_egg_mask(masks, image_shape):

    height, width = image_shape[:2]
    image_area = height * width

    valid_candidates = []

    for annotation in masks:

        area = annotation["area"]

        x, y, box_width, box_height = annotation["bbox"]

        area_ratio = area / image_area

        # Reject very small regions
        if area_ratio < 0.01:
            continue

        # Reject near-full-image regions
        if area_ratio > 0.85:
            continue

        # Reject full-frame bounding boxes
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
        return None, []

    # Baseline approach:
    # Select the largest plausible candidate.
    selected = max(
        valid_candidates,
        key=lambda item: item["area"]
    )

    return selected, valid_candidates


# ==========================================
# 6. LOAD PREVIOUS RESULTS FOR RESUMING
# ==========================================

if RESULTS_CSV.exists():

    old_df = pd.read_csv(RESULTS_CSV)

    # Keep the latest record for each image path.
    if "image_path" in old_df.columns:
        old_df = old_df.drop_duplicates(
            subset=["image_path"],
            keep="last"
        )

    records = old_df.to_dict("records")

    print("\nPrevious experiment records loaded:", len(records))

else:

    records = []

    print("\nNo previous CSV found. Starting fresh.")


# Create a lookup for previous records.
record_lookup = {
    record.get("image_path"): record
    for record in records
    if record.get("image_path")
}


# ==========================================
# 7. PROCESS ALL IMAGES WITH RESUME SUPPORT
# ==========================================

for index, image_path in enumerate(image_paths, start=1):

    relative_path = image_path.relative_to(DATASET_PATH)

    output_relative = relative_path.with_suffix(".png")

    mask_path = MASK_DIR / output_relative
    overlay_path = OVERLAY_DIR / output_relative

    image_key = str(relative_path)

    # --------------------------------------
    # SKIP ALREADY COMPLETED IMAGES
    # --------------------------------------

    if mask_path.exists() and overlay_path.exists():

        print(
            f"\n[{index}/{len(image_paths)}] "
            f"SKIPPING: {relative_path}"
        )

        # Keep an existing record if available.
        if image_key not in record_lookup:

            existing_mask = cv2.imread(
                str(mask_path),
                cv2.IMREAD_GRAYSCALE
            )

            if existing_mask is not None:

                area = int(np.count_nonzero(existing_mask))

                image_area = (
                    existing_mask.shape[0]
                    * existing_mask.shape[1]
                )

                record = {
                    "image_path": image_key,
                    "filename": image_path.name,
                    "split": relative_path.parts[0],
                    "class": (
                        relative_path.parts[1]
                        if len(relative_path.parts) > 1
                        else "unknown"
                    ),
                    "status": "success",
                    "candidate_count": np.nan,
                    "selected_area": area,
                    "selected_ratio": round(
                        area / image_area, 4
                    ),
                    "mask_path": str(mask_path),
                    "overlay_path": str(overlay_path)
                }

                records.append(record)
                record_lookup[image_key] = record

        continue

    print(
        f"\n[{index}/{len(image_paths)}] "
        f"Processing: {relative_path}"
    )

    try:

        # ----------------------------------
        # READ IMAGE
        # ----------------------------------

        image_bgr = cv2.imread(str(image_path))

        if image_bgr is None:
            raise ValueError("Could not read image")

        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------
        # GENERATE SAM MASKS
        # ----------------------------------

        print("Generating segmentation masks...")

        masks = mask_generator.generate(image_rgb)

        print("Total SAM masks:", len(masks))

        # ----------------------------------
        # SELECT EGG CANDIDATE
        # ----------------------------------

        selected, candidates = select_egg_mask(
            masks,
            image_rgb.shape
        )

        candidate_count = len(candidates)

        # ----------------------------------
        # CREATE OUTPUT DIRECTORIES
        # ----------------------------------

        mask_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        overlay_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        candidate_folder = (
            CANDIDATE_DIR
            / relative_path.parent
            / image_path.stem
        )

        candidate_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        # ----------------------------------
        # HANDLE NO CANDIDATE
        # ----------------------------------

        if selected is None:

            print("WARNING: No suitable egg candidate found")

            record = {
                "image_path": image_key,
                "filename": image_path.name,
                "split": relative_path.parts[0],
                "class": (
                    relative_path.parts[1]
                    if len(relative_path.parts) > 1
                    else "unknown"
                ),
                "status": "no_candidate",
                "candidate_count": candidate_count,
                "selected_area": 0,
                "selected_ratio": 0,
                "mask_path": "",
                "overlay_path": ""
            }

            record_lookup[image_key] = record

            # Remove any old record for this image.
            records = [
                r for r in records
                if r.get("image_path") != image_key
            ]

            records.append(record)

            # Save progress immediately.
            pd.DataFrame(records).to_csv(
                RESULTS_CSV,
                index=False
            )

            continue

        # ----------------------------------
        # SAVE SELECTED BINARY MASK
        # ----------------------------------

        egg_mask = selected["segmentation"]

        binary_mask = (
            egg_mask.astype(np.uint8) * 255
        )

        if not cv2.imwrite(str(mask_path), binary_mask):
            raise IOError("Failed to save binary mask")

        # ----------------------------------
        # CREATE AND SAVE OVERLAY
        # ----------------------------------

        overlay = image_rgb.copy()

        overlay[egg_mask] = (
            0.5 * overlay[egg_mask]
            + 0.5 * np.array([0, 255, 0])
        ).astype(np.uint8)

        overlay_bgr = cv2.cvtColor(
            overlay,
            cv2.COLOR_RGB2BGR
        )

        if not cv2.imwrite(str(overlay_path), overlay_bgr):
            raise IOError("Failed to save overlay")

        # ----------------------------------
        # SAVE ALL PLAUSIBLE CANDIDATES
        # ----------------------------------

        for candidate_index, candidate in enumerate(candidates):

            candidate_mask = (
                candidate["segmentation"].astype(np.uint8)
                * 255
            )

            cv2.imwrite(
                str(
                    candidate_folder
                    / f"candidate_{candidate_index}.png"
                ),
                candidate_mask
            )

        # ----------------------------------
        # CALCULATE MASK AREA
        # ----------------------------------

        area = int(egg_mask.sum())

        image_area = (
            image_rgb.shape[0]
            * image_rgb.shape[1]
        )

        area_ratio = area / image_area

        # ----------------------------------
        # RECORD RESULTS
        # ----------------------------------

        record = {
            "image_path": image_key,
            "filename": image_path.name,
            "split": relative_path.parts[0],
            "class": (
                relative_path.parts[1]
                if len(relative_path.parts) > 1
                else "unknown"
            ),
            "status": "success",
            "candidate_count": candidate_count,
            "selected_area": area,
            "selected_ratio": round(area_ratio, 4),
            "mask_path": str(mask_path),
            "overlay_path": str(overlay_path)
        }

        # Replace an older record for this image.
        records = [
            r for r in records
            if r.get("image_path") != image_key
        ]

        records.append(record)
        record_lookup[image_key] = record

        # Save after every successfully processed image.
        pd.DataFrame(records).to_csv(
            RESULTS_CSV,
            index=False
        )

        print(
            f"Selected area: {area} pixels | "
            f"Candidates: {candidate_count}"
        )

        print("Mask saved:", mask_path)

    except Exception as error:

        print("ERROR:", error)

        record = {
            "image_path": image_key,
            "filename": image_path.name,
            "split": relative_path.parts[0],
            "class": (
                relative_path.parts[1]
                if len(relative_path.parts) > 1
                else "unknown"
            ),
            "status": "error",
            "candidate_count": 0,
            "selected_area": 0,
            "selected_ratio": 0,
            "mask_path": "",
            "overlay_path": ""
        }

        records = [
            r for r in records
            if r.get("image_path") != image_key
        ]

        records.append(record)
        record_lookup[image_key] = record

        # Save progress even when an image fails.
        pd.DataFrame(records).to_csv(
            RESULTS_CSV,
            index=False
        )

    finally:

        if DEVICE == "mps":
            torch.mps.empty_cache()


# ==========================================
# 8. SAVE FINAL EXPERIMENT RESULTS
# ==========================================

df = pd.DataFrame(records)

df.to_csv(
    RESULTS_CSV,
    index=False
)


# ==========================================
# 9. PRINT FINAL SUMMARY
# ==========================================

print("\n" + "=" * 50)
print("SAM BATCH PROCESSING SUMMARY")
print("=" * 50)

print("Total dataset images:", len(image_paths))
print("Total recorded images:", len(df))

if not df.empty:

    print("\nProcessing status:")
    print(df["status"].value_counts())

    print("\nResults by split:")
    print(
        df.groupby(["split", "status"]).size()
    )

    successful = df[
        df["status"] == "success"
    ]

    if not successful.empty:

        print("\nAverage selected mask area:")

        print(
            successful["selected_area"].mean()
        )

print("\nResults CSV:", RESULTS_CSV)

print("\nSAM batch processing completed!")
