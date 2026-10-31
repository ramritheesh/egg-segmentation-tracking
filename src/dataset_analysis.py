
import cv2
import pandas as pd

from pathlib import Path

# Dataset directory
DATASET_PATH = Path("data/raw/egg.v1i.folder")

# Image extensions
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".webp"
}

# Find all images
image_paths = [
    path for path in DATASET_PATH.rglob("*")
    if path.is_file()
    and path.suffix.lower() in IMAGE_EXTENSIONS
]

print("TOTAL IMAGES:", len(image_paths))

records = []

for path in image_paths:

    # Read image
    image = cv2.imread(str(path))

    if image is None:
        print("Cannot read:", path)
        continue

    # Get image dimensions
    height, width, channels = image.shape

    records.append({
        "filename": path.name,
        "split": path.relative_to(DATASET_PATH).parts[0],
        "class": path.parent.name,
        "width": width,
        "height": height,
        "channels": channels,
        "size_kb": round(
            path.stat().st_size / 1024, 2
        )
    })

# Create DataFrame
df = pd.DataFrame(records)

print("\nIMAGE DETAILS:")
print(df.head(10))

print("\nDIMENSION SUMMARY:")
print(df[["width", "height"]].describe())

print("\nIMAGE BY SPLIT AND CLASS:")
print(df.groupby(["split", "class"]).size())

# Save results
Path("outputs").mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    "outputs/dataset_analysis.csv",
    index=False
)

print("\nAnalysis saved successfully!")
