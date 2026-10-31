
import cv2
import numpy as np
import pandas as pd

from pathlib import Path


# =====================================================
# 1. CONFIGURATION
# =====================================================

MANIFEST = Path("outputs/task2/split_manifest.csv")

DATASET_ROOT = Path("data/raw/egg.v1i.folder")

CANDIDATE_ROOT = Path("outputs/task1_sam_batch/candidates")

# New output folder: previous experiments are preserved.
OUTPUT_ROOT = Path(
    "outputs/task1_selection_experiment/multi_egg_v4"
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# Candidate filtering parameters
MIN_AREA = 1500
MIN_AREA_RATIO = 0.005
MAX_AREA_RATIO = 0.35

MIN_ASPECT_RATIO = 0.55
MAX_ASPECT_RATIO = 1.80

MIN_EXTENT = 0.55
MIN_SOLIDITY = 0.85

DUPLICATE_OVERLAP_THRESHOLD = 0.75

BORDER_MARGIN = 2

# None means process all development images.
SAMPLE_SIZE = None

RANDOM_SEED = 42


# =====================================================
# 2. CALCULATE SHAPE METRICS
# =====================================================

def calculate_shape_metrics(mask):

    binary = (mask > 0).astype(np.uint8)

    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    if count <= 1:
        return None

    # Keep the largest connected component.
    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + np.argmax(component_areas)

    component = (
        labels == largest_label
    ).astype(np.uint8)

    area = int(component.sum())

    x = int(stats[largest_label, cv2.CC_STAT_LEFT])
    y = int(stats[largest_label, cv2.CC_STAT_TOP])

    width = int(stats[largest_label, cv2.CC_STAT_WIDTH])
    height = int(stats[largest_label, cv2.CC_STAT_HEIGHT])

    bbox_area = width * height

    aspect_ratio = width / max(height, 1)

    extent = area / max(bbox_area, 1)

    contours, _ = cv2.findContours(
        component,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    contour = max(
        contours,
        key=cv2.contourArea
    )

    contour_area = cv2.contourArea(contour)

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    solidity = (
        contour_area / hull_area
        if hull_area > 0 else 0
    )

    perimeter = cv2.arcLength(contour, True)

    circularity = (
        4 * np.pi * contour_area / (perimeter ** 2)
        if perimeter > 0 else 0
    )

    return {
        "area": area,
        "bbox": (x, y, width, height),
        "aspect_ratio": aspect_ratio,
        "extent": extent,
        "solidity": solidity,
        "circularity": circularity,
        "component_mask": component * 255
    }


# =====================================================
# 3. BORDER INFORMATION
# =====================================================

def touches_image_border(bbox, image_width, image_height):

    x, y, width, height = bbox

    margin = BORDER_MARGIN

    return (
        x <= margin
        or y <= margin
        or x + width >= image_width - margin
        or y + height >= image_height - margin
    )


# =====================================================
# 4. DUPLICATE OVERLAP
# =====================================================

def calculate_overlap_ratio(mask_a, mask_b):

    a = mask_a > 0
    b = mask_b > 0

    intersection = np.logical_and(a, b).sum()

    smaller_area = min(
        a.sum(),
        b.sum()
    )

    if smaller_area == 0:
        return 0

    # Intersection divided by the smaller mask area.
    # This also detects masks contained inside larger masks.
    return intersection / smaller_area


# =====================================================
# 5. FILTER SAM CANDIDATES
# =====================================================

def filter_candidates(
    candidate_paths,
    image_width,
    image_height
):

    image_area = image_width * image_height

    candidates = []

    rejection_counts = {
        "invalid": 0,
        "area": 0,
        "aspect_ratio": 0,
        "extent": 0,
        "solidity": 0
    }

    for candidate_path in candidate_paths:

        mask = cv2.imread(
            str(candidate_path),
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:
            rejection_counts["invalid"] += 1
            continue

        metrics = calculate_shape_metrics(mask)

        if metrics is None:
            rejection_counts["invalid"] += 1
            continue

        area = metrics["area"]

        area_ratio = area / image_area

        if (
            area < MIN_AREA
            or area_ratio < MIN_AREA_RATIO
            or area_ratio > MAX_AREA_RATIO
        ):
            rejection_counts["area"] += 1
            continue

        if not (
            MIN_ASPECT_RATIO
            <= metrics["aspect_ratio"]
            <= MAX_ASPECT_RATIO
        ):
            rejection_counts["aspect_ratio"] += 1
            continue

        if metrics["extent"] < MIN_EXTENT:
            rejection_counts["extent"] += 1
            continue

        if metrics["solidity"] < MIN_SOLIDITY:
            rejection_counts["solidity"] += 1
            continue

        # Do not automatically reject border candidates.
        border_flag = touches_image_border(
            metrics["bbox"],
            image_width,
            image_height
        )

        candidates.append({
            "path": candidate_path,
            "mask": metrics["component_mask"],
            "area": area,
            "bbox": metrics["bbox"],
            "aspect_ratio": metrics["aspect_ratio"],
            "extent": metrics["extent"],
            "solidity": metrics["solidity"],
            "circularity": metrics["circularity"],
            "touches_border": border_flag
        })

    return candidates, rejection_counts


# =====================================================
# 6. REMOVE DUPLICATE MASKS
# =====================================================

def remove_duplicate_masks(candidates):

    # Sort by area, largest first.
    candidates = sorted(
        candidates,
        key=lambda item: item["area"],
        reverse=True
    )

    selected = []

    duplicate_count = 0

    for candidate in candidates:

        duplicate = False

        for existing in selected:

            overlap = calculate_overlap_ratio(
                candidate["mask"],
                existing["mask"]
            )

            if overlap >= DUPLICATE_OVERLAP_THRESHOLD:

                duplicate = True
                duplicate_count += 1
                break

        if not duplicate:
            selected.append(candidate)

    return selected, duplicate_count


# =====================================================
# 7. CREATE OVERLAY
# =====================================================

def create_overlay(image, selected):

    overlay = image.copy()

    # Give each selected instance a different color.
    colors = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
        (0, 128, 255)
    ]

    for index, candidate in enumerate(selected):

        mask = candidate["mask"] > 0

        color = np.array(
            colors[index % len(colors)],
            dtype=np.float32
        )

        overlay[mask] = (
            0.5 * overlay[mask].astype(np.float32)
            + 0.5 * color
        ).astype(np.uint8)

    return overlay


# =====================================================
# 8. PROCESS ONE IMAGE
# =====================================================

def process_image(row):

    image_path = Path(row["image_path"])

    image = cv2.imread(str(image_path))

    if image is None:

        print("Could not read:", image_path)

        return [], None

    height, width = image.shape[:2]

    relative_path = image_path.relative_to(
        DATASET_ROOT
    )

    candidate_folder = (
        CANDIDATE_ROOT
        / relative_path.parent
        / image_path.stem
    )

    candidate_paths = sorted(
        candidate_folder.glob("candidate_*.png")
    )

    if not candidate_paths:

        print("No candidates:", image_path.name)

        return [], None

    candidates, rejection_counts = filter_candidates(
        candidate_paths,
        width,
        height
    )

    selected, duplicate_count = remove_duplicate_masks(
        candidates
    )

    image_output = OUTPUT_ROOT / image_path.stem

    individual_folder = (
        image_output / "individual_masks"
    )

    individual_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    combined_mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    records = []

    for index, candidate in enumerate(selected):

        egg_id = index + 1

        mask = candidate["mask"]

        mask_path = (
            individual_folder
            / f"egg_{egg_id}.png"
        )

        cv2.imwrite(
            str(mask_path),
            mask
        )

        combined_mask = cv2.bitwise_or(
            combined_mask,
            mask
        )

        records.append({

            "filename": image_path.name,
            "egg_id": egg_id,
            "area": candidate["area"],
            "aspect_ratio": round(
                candidate["aspect_ratio"], 4
            ),
            "extent": round(
                candidate["extent"], 4
            ),
            "solidity": round(
                candidate["solidity"], 4
            ),
            "circularity": round(
                candidate["circularity"], 4
            ),
            "touches_border": candidate["touches_border"],
            "mask_path": str(mask_path)

        })

    # Save combined mask.
    cv2.imwrite(
        str(image_output / "combined_mask.png"),
        combined_mask
    )

    # Save colored instance overlay.
    overlay = create_overlay(
        image,
        selected
    )

    cv2.imwrite(
        str(image_output / "overlay.jpg"),
        overlay
    )

    summary = {
        "filename": image_path.name,
        "input_candidates": len(candidate_paths),
        "filtered_candidates": len(candidates),
        "selected_candidates": len(selected),
        "duplicates_removed": duplicate_count,
        **rejection_counts
    }

    print(
        f"{image_path.name}: "
        f"{len(selected)} candidates selected"
    )

    return records, summary


# =====================================================
# 9. MAIN
# =====================================================

def main():

    print("=" * 60)
    print("MULTI-EGG SEGMENTATION - DEVELOPMENT EXPERIMENT")
    print("=" * 60)

    df = pd.read_csv(MANIFEST)

    dev = df[
        (df["subset"] == "dev")
        & (df["class"] == "egg")
    ].copy()

    if dev.empty:
        raise ValueError("No development egg images found.")

    dev = dev.sample(
        frac=1,
        random_state=RANDOM_SEED
    )

    if SAMPLE_SIZE is not None:
        dev = dev.head(SAMPLE_SIZE)

    print("Development images:", len(dev))

    all_records = []
    all_summaries = []

    for _, row in dev.iterrows():

        records, summary = process_image(row)

        all_records.extend(records)

        if summary is not None:
            all_summaries.append(summary)

    # Save individual egg information.
    result_columns = [
        "filename",
        "egg_id",
        "area",
        "aspect_ratio",
        "extent",
        "solidity",
        "circularity",
        "touches_border",
        "mask_path"
    ]

    results = pd.DataFrame(
        all_records,
        columns=result_columns
    )

    results.to_csv(
        OUTPUT_ROOT / "multi_egg_results.csv",
        index=False
    )

    # Save image-level experiment statistics.
    summary_df = pd.DataFrame(all_summaries)

    summary_df.to_csv(
        OUTPUT_ROOT / "experiment_summary.csv",
        index=False
    )

    # Print summary.
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETED")
    print("=" * 60)

    print("Images processed:", len(all_summaries))
    print("Total selected candidates:", len(results))

    print(
        "Images with zero selected candidates:",
        int((summary_df["selected_candidates"] == 0).sum())
        if not summary_df.empty else 0
    )

    print("\nCandidate count distribution:")

    if not summary_df.empty:
        print(
            summary_df["selected_candidates"]
            .value_counts()
            .sort_index()
            .to_string()
        )

    print("\nTotal rejection statistics:")

    rejection_columns = [
        "invalid",
        "area",
        "aspect_ratio",
        "extent",
        "solidity",
        "duplicates_removed"
    ]

    if not summary_df.empty:
        print(
            summary_df[rejection_columns]
            .sum()
            .to_string()
        )

    print("\nResults saved to:", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
