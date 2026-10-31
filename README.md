# Egg Segmentation and Tracking

**Computer Vision Assignment | K.V.RAM RITHEESH**

A computer vision project focused on egg segmentation, classical image processing, quantitative evaluation, and individual egg tracking in video.

## Project Overview

This project implements four major tasks:

1. **SAM-Based Segmentation:** Use the pretrained Segment Anything Model (SAM) to generate egg segmentation masks.
2. **Classical Image Segmentation:** Implement and compare multiple traditional image-processing techniques against SAM-generated pseudo-labels.
3. **Quantitative Evaluation:** Evaluate classical segmentation predictions using IoU, Dice, Precision, and Recall, with qualitative failure analysis.
4. **Video Segmentation and Tracking:** Detect eggs in a video and track detected objects using centroid-based tracking.

## Project Links

- **GitHub Repository:** https://github.com/ramritheesh/egg-segmentation-tracking
- **Technical Report:** [Egg Segmentation and Tracking Report](Egg_Segmentation_Tracking_Report.pdf)
- **Annotated Tracking Video:** [Watch the output video on Google Drive](https://drive.google.com/file/d/1CU-npwmSGBq3-55IV46EynXyTkwMeDe0/view?usp=sharing)
- **Dataset:** [Roboflow Egg Dataset — Version 1](https://universe.roboflow.com/fire-bf7qe/egg-vzhtr/dataset/1)
- **Source Video:** [Raw and Boiled Egg Spinning – Cool Science Experiment](https://www.youtube.com/watch?v=Y4JyV94y2_A)

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python 3.13 |
| Computer Vision | OpenCV |
| Deep Learning | PyTorch |
| Segmentation Model | Segment Anything Model (SAM) |
| Image Processing | HSV Thresholding, Morphology, GrabCut |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib |
| Video Processing | FFmpeg, OpenCV |
| Version Control | Git and GitHub |

## Dataset Information

The project uses the Roboflow Egg Dataset, Version 1.

- Total images: 110
- Egg images: 108
- Unlabeled images: 2
- Image resolution: 640 × 640 pixels

### Dataset Split

The dataset was divided into development and evaluation subsets using an approximately 4:1 ratio.

| Subset | Egg Images | Unlabeled Images | Total |
|---|---:|---:|---:|
| Development | 86 | 1 | 87 |
| Evaluation | 22 | 1 | 23 |
| Total | 108 | 2 | 110 |

An exact SHA-256 duplicate check found no exact duplicate leakage between the subsets. Near-duplicate leakage was not tested.

The split manifest is stored at:

`outputs/task2/split_manifest.csv`

## Project Structure

```text
egg-segmentation-tracking/
│
├── README.md
├── Egg_Segmentation_Tracking_Report.pdf
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── dataset_analysis.py
│   ├── visualize_dataset.py
│   ├── task1_sam.py
│   ├── task1_batch_sam.py
│   │
│   ├── task1/
│   │   ├── multi_egg_segmentation.py
│   │   ├── multi_egg_segmentation_v5.py
│   │   ├── compare_selection_strategies.py
│   │   ├── inspect_sam_candidates.py
│   │   └── visualize_selection_comparison.py
│   │
│   ├── task2/
│   │   ├── split_dataset.py
│   │   ├── classical_segmentation.py
│   │   ├── grabcut_segmentation.py
│   │   ├── grabcut_mask_init.py
│   │   ├── run_dev_experiments.py
│   │   ├── evaluate.py
│   │   └── analyze_failures.py
│   │
│   └── task4/
│       ├── detect_eggs.py
│       └── track_eggs.py
│
├── data/
│   ├── raw/
│   └── video/
│
├── models/
│   └── sam_vit_b.pth
│
└── outputs/
    ├── task1_sam_batch/
    ├── task2/
    └── task4/
```

*Note: The dataset, model checkpoint, and generated outputs are stored locally and are not included in the GitHub repository.*

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/ramritheesh/egg-segmentation-tracking.git

cd egg-segmentation-tracking
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

Activate it:

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip

pip install -r requirements.txt
```

Install FFmpeg and yt-dlp if you need to download or process the video:

**macOS**

```bash
brew install ffmpeg yt-dlp
```

### 4. Download the dataset

Download the dataset from:

https://universe.roboflow.com/fire-bf7qe/egg-vzhtr/dataset/1

Extract it into the following location:

```text
data/raw/egg.v1i.folder/
```

The expected dataset structure is:

```text
data/raw/egg.v1i.folder/
├── train/
├── valid/
└── test/
```

### 5. Download the SAM checkpoint

Download the SAM ViT-B pretrained checkpoint from the official Segment Anything repository:

https://github.com/facebookresearch/segment-anything

Place the checkpoint in:

```text
models/sam_vit_b.pth
```

The model checkpoint is not included in this repository because of its large file size.

## Task 1 — SAM-Based Egg Segmentation

### Objective

Use a pretrained segmentation model to identify and segment eggs in the dataset images.

### Implementation

The project uses the pretrained SAM ViT-B model to generate candidate masks.

The automatic mask generator uses the following configuration:

| Parameter | Value |
|---|---:|
| Points per side | 16 |
| Predicted IoU threshold | 0.86 |
| Stability score threshold | 0.92 |
| Crop layers | 0 |
| Minimum mask region area | 100 |

A multi-egg candidate-selection pipeline was implemented to select plausible egg instances from the generated masks.

The selection pipeline was refined from v4 to v5.

| Version | Selected candidates | Zero-detection images |
|---|---:|---:|
| v4 | 201 | 2 |
| v5 | 203 | 1 |

Version 5 recovered a plausible candidate in `not_damaged_35`. The image `not_damaged_124` still had no reliable selected candidate.

### Run Task 1

```bash
python3 src/task1_batch_sam.py
```

### Outputs

Generated outputs are stored under:

```text
outputs/task1_sam_batch/
```

The outputs include:

- Candidate segmentation masks
- Selected segmentation masks
- Combined masks
- Visualization overlays
- SAM processing results

**Note:** SAM-generated masks are treated as pseudo-labels in this project. They are not manually verified ground-truth annotations.

## Task 2 — Classical Image Segmentation

### Objective

Implement traditional image-processing methods to segment eggs and compare their results with SAM-generated reference masks.

### Implemented Methods

**Experiment 1 — HSV Thresholding**

- Convert images from BGR to HSV.
- Apply HSV thresholding.
- Use morphological operations to refine the mask.

**Experiment 2 — HSV with Connected Components**

- Apply stricter HSV thresholding.
- Extract connected components.
- Filter components based on area, dimensions, and border conditions.

**Experiment 3 — GrabCut with Rectangle Initialization**

- Initialize GrabCut using a rectangular region.
- Apply iterative foreground and background classification.
- Generate the segmentation mask.

**Experiment 4 — GrabCut with Mask Initialization**

- Initialize probable foreground and background regions.
- Apply GrabCut using mask initialization.
- Refine the output using morphological opening and closing.

### Run Development Experiments

```bash
python3 src/task2/run_dev_experiments.py
```

### Development Results

| Method | IoU | Dice | Precision | Recall |
|---|---:|---:|---:|---:|
| HSV Thresholding | 0.2986 | 0.4121 | 0.3201 | 0.7514 |
| HSV Components | 0.1611 | 0.1881 | 0.2410 | 0.1747 |
| GrabCut Rectangle | 0.3291 | 0.4109 | 0.3784 | 0.5285 |
| GrabCut Mask | 0.3388 | 0.4127 | 0.3821 | 0.5142 |

**Selected method:** GrabCut with mask initialization (Experiment 4), because it achieved the highest development IoU and Dice.

The development metrics use the earlier single-mask SAM references. The final evaluation uses the v5 multi-instance SAM references, so the two sets of results are not directly interchangeable.

## Task 3 — Quantitative Evaluation

### Objective

Evaluate the selected classical segmentation method against SAM v5 pseudo-reference masks on the held-out evaluation images.

### Evaluation Metrics

- **Intersection over Union (IoU):** Measures overlap between the predicted mask and reference mask.
- **Dice Coefficient:** Measures similarity between the predicted and reference regions.
- **Precision:** Measures the proportion of predicted foreground pixels that overlap the reference.
- **Recall:** Measures the proportion of reference foreground pixels captured by the prediction.

### Run Final Evaluation

```bash
python3 src/task2/evaluate.py
```

### Final Evaluation Results

The final evaluation covered all 22 evaluation images.

| Metric | Result |
|---|---:|
| Mean IoU | 0.473918 |
| Mean Dice | 0.591723 |
| Mean Precision | 0.600877 |
| Mean Recall | 0.685487 |
| Images evaluated | 22 / 22 |

### Qualitative Analysis

High-agreement examples:

- `not_damaged_159` — IoU: 0.9915
- `not_damaged_22` — IoU: 0.9443
- `not_damaged_162` — IoU: 0.8139

Low-agreement examples:

- `not_damaged_12` — IoU: 0.0000
- `not_damaged_34` — IoU: 0.0000
- `not_damaged_125` — IoU: 0.1279
- `not_damaged_128` — IoU: 0.1860
- `not_damaged_121` — IoU: 0.2201

Failure analysis identified possible SAM reference-mask failures, segmentation mismatches, and over-segmentation.

**Important evaluation limitation:** These metrics measure agreement with SAM v5 pseudo-labels, not accuracy against human-annotated ground truth. In particular, `not_damaged_34` had an empty SAM reference mask while GrabCut predicted a foreground region that visual inspection suggested contained an egg.

### Evaluation Outputs

```text
outputs/task2/final_evaluation/
```

This directory contains:

- Task 1 v5 reference masks
- Experiment 4 prediction masks
- Visual comparison images
- Individual evaluation metrics
- Summary evaluation metrics

## Task 4 — Video Egg Detection and Tracking

### Objective

Apply the final detection and tracking pipeline to the supplied egg video and track detected eggs individually.

### Video Information

| Property | Value |
|---|---|
| Resolution | 1280 × 720 |
| Frame rate | 24 FPS |
| Duration | 60.021 seconds |
| Total frames | 1,440 |

### Detection Method

The video detection pipeline uses:

- HSV color thresholding
- Morphological opening and closing
- Contour detection
- Minimum contour area filtering
- Bounding-box width and height filtering
- Aspect-ratio filtering

### Tracking Method

The baseline tracker uses centroid-distance matching.

Each detected egg is assigned a tracking ID. Existing tracks are matched to new detections based on the distance between their centroids, with a temporary disappearance allowance.

The baseline processed all 1,440 frames and assigned 10 track IDs.

A second motion-prediction experiment assigned 15 IDs and did not improve the baseline. The simpler baseline was retained.

### Run Video Tracking

Ensure the input video is available at:

```text
data/video/egg_tracking_video.mp4
```

Run:

```bash
python3 src/task4/track_eggs.py
```

### Output Files

```text
outputs/task4/
```

Main baseline outputs:

- `egg_tracking_output.mp4`
- `egg_tracking_log.csv`

Experimental V2 outputs:

- `egg_tracking_output_v2.mp4`
- `egg_tracking_log_v2.csv`

### Watch the Tracking Output

[**Open Annotated Egg Tracking Video**](https://drive.google.com/file/d/1CU-npwmSGBq3-55IV46EynXyTkwMeDe0/view?usp=sharing)

### Tracking Limitations

- Hands repeatedly occlude eggs.
- Occlusion can cause temporary detection failures.
- A physical egg may receive a new ID after reappearing.
- Centroid-based matching can fail when detections shift or overlap.
- Ten assigned IDs do not represent ten unique physical eggs.

No manually annotated video tracks or identity-switch benchmark was available. Therefore, tracking identity accuracy has not been quantitatively validated.

## Reproducibility — Main Commands

After completing the dataset and checkpoint setup, the main experiments can be run using:

```bash
# Task 1: SAM batch segmentation
python3 src/task1_batch_sam.py

# Task 2: Development experiments
python3 src/task2/run_dev_experiments.py

# Task 3: Final evaluation
python3 src/task2/evaluate.py

# Task 4: Video detection and tracking
python3 src/task4/track_eggs.py
```

The scripts may require the dataset, model checkpoint, and video to be placed in the expected local paths described above.

## Limitations and Future Improvements

Potential improvements include:

1. Create manually annotated segmentation masks for a representative subset.
2. Improve SAM candidate selection to reduce false positives and missed eggs.
3. Test near-duplicate leakage across dataset splits.
4. Improve classical segmentation under challenging lighting conditions.
5. Improve egg detection under hand occlusion.
6. Evaluate Kalman filtering and Hungarian assignment for more robust tracking.
7. Measure ID switches, track fragmentation, and IDF1 using manually annotated video tracks.

## Technical Report

The complete technical report documents:

- Project methodology
- Dataset split
- SAM segmentation implementation
- Classical segmentation experiments
- Quantitative evaluation results
- Qualitative failure analysis
- Video tracking implementation
- Limitations and future improvements

**[Open the Technical Report PDF](Egg_Segmentation_Tracking_Report.pdf)**

## AI Usage Disclosure

Generative AI assistance was used for code drafting, debugging guidance, experiment interpretation, and report preparation. The implementation, experimental outputs, and conclusions were reviewed as part of the project workflow.

## Author

**K.V.RAM RITHEESH**

GitHub: [ramritheesh](https://github.com/ramritheesh)

---

*This project was developed as a computer vision assignment covering segmentation, image processing, evaluation, and object tracking.*