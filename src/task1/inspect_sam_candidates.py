
import cv2
import torch
import pandas as pd

from pathlib import Path
from segment_anything import (
    sam_model_registry,
    SamAutomaticMaskGenerator
)

# Configuration
IMAGE_PATH = Path(
    "data/raw/egg.v1i.folder/valid/egg/"
    "not_damaged_6_jpg.rf.1b79fa07c23ef838f19c9a226c9022f5.jpg"
)

CHECKPOINT = "models/sam_vit_b.pth"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Load image
image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(IMAGE_PATH)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load SAM
sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT)
sam.to(device=DEVICE)

generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=16,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=0,
    min_mask_region_area=100
)

# Generate candidates
masks = generator.generate(image_rgb)

print("\nSAM CANDIDATE CONFIDENCE SCORES\n")

for i, candidate in enumerate(masks):
    print(
        f"Candidate {i}: "
        f"Area={candidate['area']}, "
        f"Predicted IoU={candidate['predicted_iou']:.4f}, "
        f"Stability={candidate['stability_score']:.4f}, "
        f"BBox={candidate['bbox']}"
    )

print("\nTotal candidates:", len(masks))
