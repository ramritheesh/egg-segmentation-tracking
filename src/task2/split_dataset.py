
import hashlib
import random
from pathlib import Path

import pandas as pd


# ==========================================
# 1. CONFIGURATION
# ==========================================

DATASET_DIR = Path("data/raw/egg.v1i.folder")
SAM_MASK_DIR = Path("outputs/task1_sam_batch/masks")

OUTPUT_DIR = Path("outputs/task2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "split_manifest.csv"

RANDOM_SEED = 42
EVAL_RATIO = 0.20

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ==========================================
# 2. HELPER: CALCULATE IMAGE HASH
# ==========================================

def calculate_sha256(image_path):

    sha256 = hashlib.sha256()

    with open(image_path, "rb") as file:

        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


# ==========================================
# 3. FIND DATASET IMAGES
# ==========================================

image_paths = sorted([
    path
    for path in DATASET_DIR.rglob("*")
    if path.is_file()
    and path.suffix.lower() in IMAGE_EXTENSIONS
])

print("Total images found:", len(image_paths))


# ==========================================
# 4. CREATE IMAGE RECORDS
# ==========================================

records = []

for image_path in image_paths:

    relative_path = image_path.relative_to(DATASET_DIR)

    if len(relative_path.parts) < 2:
        print("Skipping unexpected path:", relative_path)
        continue

    original_split = relative_path.parts[0]
    class_name = relative_path.parts[1]

    # Match the corresponding SAM mask
    mask_relative_path = relative_path.with_suffix(".png")
    mask_path = SAM_MASK_DIR / mask_relative_path

    if not mask_path.exists():
        print("WARNING: Missing SAM mask:", mask_path)
        continue

    image_hash = calculate_sha256(image_path)

    records.append({
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "filename": image_path.name,
        "original_split": original_split,
        "class": class_name,
        "sha256": image_hash
    })


df = pd.DataFrame(records)

print("Images with available SAM masks:", len(df))


# ==========================================
# 5. GROUP EXACT DUPLICATES
# ==========================================

# Identical image contents must remain in the same subset.
# Group within each class to retain class-wise splitting.

rng = random.Random(RANDOM_SEED)

df["subset"] = ""

for class_name in sorted(df["class"].unique()):

    class_df = df[df["class"] == class_name]

    unique_groups = list(
        class_df["sha256"].unique()
    )

    rng.shuffle(unique_groups)

    total_groups = len(unique_groups)

    if total_groups < 2:

        print(
            f"WARNING: Class '{class_name}' has fewer "
            "than 2 unique groups. Keeping it in dev."
        )

        eval_groups = []

    else:

        eval_count = round(
            total_groups * EVAL_RATIO
        )

        eval_count = max(
            1,
            min(eval_count, total_groups - 1)
        )

        eval_groups = unique_groups[:eval_count]

    # Assign all exact duplicates to the same subset
    eval_condition = (
        df["class"].eq(class_name)
        & df["sha256"].isin(eval_groups)
    )

    dev_condition = (
        df["class"].eq(class_name)
        & ~df["sha256"].isin(eval_groups)
    )

    df.loc[eval_condition, "subset"] = "eval"
    df.loc[dev_condition, "subset"] = "dev"


# ==========================================
# 6. VERIFY NO EXACT HASH LEAKAGE
# ==========================================

dev_hashes = set(
    df.loc[df["subset"] == "dev", "sha256"]
)

eval_hashes = set(
    df.loc[df["subset"] == "eval", "sha256"]
)

overlap = dev_hashes.intersection(eval_hashes)

if overlap:
    raise RuntimeError(
        "Data leakage detected: duplicate hashes across subsets!"
    )

print("\nExact duplicate leakage check: PASSED")


# ==========================================
# 7. SAVE SPLIT MANIFEST
# ==========================================

df.to_csv(
    OUTPUT_CSV,
    index=False
)

print("\nSplit manifest saved:", OUTPUT_CSV)


# ==========================================
# 8. PRINT SUMMARY
# ==========================================

print("\n========== DATASET SPLIT SUMMARY ==========")

print("\nImages by subset:")
print(df["subset"].value_counts())

print("\nClass distribution:")
print(
    pd.crosstab(
        df["class"],
        df["subset"]
    )
)

print("\nOriginal split distribution:")
print(
    pd.crosstab(
        df["original_split"],
        df["subset"]
    )
)

print("\nTotal images:", len(df))

print(
    "Development ratio:",
    round((df["subset"] == "dev").mean() * 100, 2),
    "%"
)

print(
    "Evaluation ratio:",
    round((df["subset"] == "eval").mean() * 100, 2),
    "%"
)

print("\nTask 2 dataset split completed!")
