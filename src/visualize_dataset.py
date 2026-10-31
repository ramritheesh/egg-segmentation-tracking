
import cv2
import matplotlib.pyplot as plt

from pathlib import Path

DATASET_PATH = Path("data/raw/egg.v1i.folder")

# Collect images from all splits
image_paths = []

for split in ["train", "valid", "test"]:
    split_path = DATASET_PATH / split

    for path in split_path.rglob("*"):
        if path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            image_paths.append(path)

# Display first 12 images
sample_images = image_paths[:12]

fig, axes = plt.subplots(3, 4, figsize=(14, 10))

for ax in axes.flat:
    ax.axis("off")

for i, path in enumerate(sample_images):

    image = cv2.imread(str(path))

    if image is None:
        continue

    # OpenCV BGR to Matplotlib RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    ax = axes.flat[i]
    ax.imshow(image)
    ax.set_title(
        f"{path.parent.parent.name}/{path.parent.name}",
        fontsize=9
    )
    ax.axis("off")

plt.tight_layout()

Path("outputs").mkdir(exist_ok=True)

plt.savefig(
    "outputs/egg_samples.png",
    dpi=150
)

plt.show()

print("Visualization saved!")
