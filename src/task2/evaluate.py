import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
MANIFEST = Path("outputs/task2/split_manifest.csv")
DATASET_ROOT = Path("data/raw/egg.v1i.folder")

TASK1_SCRIPT_DIR = Path("src/task1").resolve()
sys.path.insert(0, str(TASK1_SCRIPT_DIR))

import multi_egg_segmentation_v5 as task1_v5

OUTPUT_ROOT = Path("outputs/task2/final_evaluation")
REFERENCE_ROOT = OUTPUT_ROOT / "task1_v5_references"
PREDICTION_ROOT = OUTPUT_ROOT / "experiment_4_grabcut_predictions"
VISUAL_ROOT = OUTPUT_ROOT / "visual_comparisons"

for folder in (OUTPUT_ROOT, REFERENCE_ROOT, PREDICTION_ROOT, VISUAL_ROOT):
    folder.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Task 2 final method: exact Experiment 4 implementation
# from run_dev_experiments.py / grabcut_mask_init.py
# ---------------------------------------------------------
def grabcut_mask_initialization(image):
    height, width = image.shape[:2]

    mask = np.full((height, width), cv2.GC_PR_BGD, dtype=np.uint8)

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
        image, mask, None, bg_model, fg_model, 5,
        cv2.GC_INIT_WITH_MASK
    )

    result = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255, 0
    ).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    return result


def calculate_metrics(prediction, reference):
    prediction = prediction > 0
    reference = reference > 0

    tp = np.logical_and(prediction, reference).sum()
    fp = np.logical_and(prediction, ~reference).sum()
    fn = np.logical_and(~prediction, reference).sum()
    union = np.logical_or(prediction, reference).sum()

    iou = tp / union if union else 1.0
    dice_denominator = prediction.sum() + reference.sum()
    dice = (2 * tp / dice_denominator) if dice_denominator else 1.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    return {
        "IoU": float(iou),
        "Dice": float(dice),
        "Precision": float(precision),
        "Recall": float(recall),
        "TP_pixels": int(tp),
        "FP_pixels": int(fp),
        "FN_pixels": int(fn),
    }


def save_visual_comparison(image, reference, prediction, output_path):
    # White = overlap, green = prediction-only, red = reference-only.
    ref = reference > 0
    pred = prediction > 0

    comparison = image.copy()
    overlap = ref & pred
    pred_only = pred & ~ref
    ref_only = ref & ~pred

    comparison[overlap] = (
        0.45 * comparison[overlap] + 0.55 * np.array([255, 255, 255])
    ).astype(np.uint8)
    comparison[pred_only] = (
        0.45 * comparison[pred_only] + 0.55 * np.array([0, 255, 0])
    ).astype(np.uint8)
    comparison[ref_only] = (
        0.45 * comparison[ref_only] + 0.55 * np.array([0, 0, 255])
    ).astype(np.uint8)

    cv2.imwrite(str(output_path), comparison)


def main():
    manifest = pd.read_csv(MANIFEST)
    eval_df = manifest[
        (manifest["subset"] == "eval") & (manifest["class"] == "egg")
    ].copy()

    if eval_df.empty:
        raise RuntimeError("No evaluation egg images found in split manifest.")

    print("=" * 64)
    print("FINAL EVALUATION: TASK 1 v5 vs TASK 2 EXPERIMENT 4")
    print("=" * 64)
    print("Evaluation egg images:", len(eval_df))
    print("Reference: Task 1 v5 SAM candidate masks (pseudo-labels)")
    print("Prediction: Experiment 4 GrabCut mask initialization")

    # Generate Task 1 v5 references for evaluation images.
    # The v5 filter parameters are unchanged; only output path is redirected.
    task1_v5.OUTPUT_ROOT = REFERENCE_ROOT

    rows = []
    missing = []

    for index, (_, row) in enumerate(eval_df.iterrows(), start=1):
        image_path = Path(row["image_path"])
        image = cv2.imread(str(image_path))
        if image is None:
            missing.append((image_path.name, "image_unreadable"))
            continue

        # Create the Task 1 v5 combined binary reference mask.
        _, summary = task1_v5.process_image(row)

        image_output = REFERENCE_ROOT / image_path.stem
        reference_path = image_output / "combined_mask.png"
        reference = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)

        if reference is None:
            missing.append((image_path.name, "reference_missing"))
            continue

        prediction = grabcut_mask_initialization(image)
        prediction_path = PREDICTION_ROOT / f"{image_path.stem}.png"
        cv2.imwrite(str(prediction_path), prediction)

        if reference.shape != prediction.shape:
            missing.append((image_path.name, "dimension_mismatch"))
            continue

        metrics = calculate_metrics(prediction, reference)
        metrics["filename"] = image_path.name
        metrics["reference_foreground_pixels"] = int((reference > 0).sum())
        metrics["prediction_foreground_pixels"] = int((prediction > 0).sum())
        metrics["v5_reference_instances"] = (
            int(summary["selected_candidates"]) if summary else 0
        )
        rows.append(metrics)

        save_visual_comparison(
            image,
            reference,
            prediction,
            VISUAL_ROOT / f"{image_path.stem}_comparison.jpg"
        )

        print(
            f"[{index}/{len(eval_df)}] {image_path.name}: "
            f"IoU={metrics['IoU']:.4f}, "
            f"Dice={metrics['Dice']:.4f}, "
            f"v5 instances={metrics['v5_reference_instances']}"
        )

    results = pd.DataFrame(rows)
    individual_path = OUTPUT_ROOT / "eval_metrics_individual.csv"
    results.to_csv(individual_path, index=False)

    metric_columns = ["IoU", "Dice", "Precision", "Recall"]
    if results.empty:
        raise RuntimeError(
            "No images were evaluated. Check missing paths and candidate masks."
        )

    summary = results[metric_columns].mean().to_frame("mean").T
    summary.insert(0, "method", "experiment_4_grabcut_mask")
    summary.insert(1, "evaluated_images", len(results))
    summary_path = OUTPUT_ROOT / "eval_metrics_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 64)
    print("EVALUATION SUMMARY")
    print("=" * 64)
    print(summary.to_string(index=False))
    print("\nEvaluated images:", len(results), "/", len(eval_df))
    print("Missing / skipped:", len(missing))
    if missing:
        print("\nMissing details:")
        for filename, reason in missing:
            print("-", filename, reason)

    print("\nSaved individual metrics:", individual_path)
    print("Saved summary:", summary_path)
    print("Saved reference masks:", REFERENCE_ROOT)
    print("Saved predictions:", PREDICTION_ROOT)
    print("Saved visual comparisons:", VISUAL_ROOT)
    print("\nNote: metrics measure agreement with SAM v5 pseudo-labels,")
    print("not agreement with human-annotated ground truth.")


if __name__ == "__main__":
    main()
