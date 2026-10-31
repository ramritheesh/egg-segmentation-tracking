# Egg Segmentation and Tracking using Computer Vision

A computer vision project for detecting, segmenting, and tracking individual eggs using a pretrained Segment Anything Model (SAM), classical image processing, and OpenCV.

## 1. Project Overview

This project consists of four major tasks:

1. **SAM-based segmentation:** Generate egg segmentation masks using the pretrained SAM ViT-B model.
2. **Classical segmentation:** Implement and compare four image-processing approaches against SAM-generated reference masks.
3. **Quantitative evaluation:** Evaluate the selected classical segmentation method using IoU, Dice, Precision, and Recall.
4. **Video tracking:** Detect eggs in a video, assign tracking IDs, and generate an annotated output video.

## 2. Technologies Used

- Python
- OpenCV
- PyTorch
- Segment Anything Model (SAM)
- NumPy
- Pandas
- GrabCut
- HSV Color Thresholding
- FFmpeg
- yt-dlp

## 3. Dataset

Dataset: [Roboflow Egg Dataset](https://universe.roboflow.com/fire-bf7qe/egg-vzhtr/dataset/1)

| Description | Count |
|---|---:|
| Total images | 110 |
| Egg images | 108 |
| Unlabeled images | 2 |
| Development images | 87 |
| Evaluation images | 23 |

The dataset was divided into development and evaluation subsets using an approximately 4:1 split.

An exact SHA-256 duplicate check found no exact duplicate leakage between the subsets.

## 4. Project Structure

```text
egg-segmentation-tracking/
│
├── data/
│   ├── raw/
│   └── video/
│       └── egg_tracking_video.mp4
│
├── models/
│   └── sam_vit_b.pth
│
├── src/
│   ├── task1_batch_sam.py
│   ├── task1/
│   │   ├── multi_egg_segmentation.py
│   │   └── multi_egg_segmentation_v5.py
│   │
│   ├── task2/
│   │   ├── run_dev_experiments.py
│   │   ├── evaluate.py
│   │   └── classical_segmentation.py
│   │
│   └── task4/
│       ├── detect_eggs.py
│       └── track_eggs.py
│
├── outputs/
│   ├── task1_sam_batch/
│   ├── task2/
│   │   ├── evaluation/
│   │   └── final_evaluation/
│   └── task4/
│       ├── egg_tracking_output.mp4
│       └── egg_tracking_log.csv
│
├── requirements.txt
└── README.md
```

## 5. Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd egg-segmentation-tracking
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Download the pretrained SAM ViT-B checkpoint and place it inside:

```text
models/sam_vit_b.pth
```

Ensure the dataset is extracted into the expected directory:

```text
data/raw/egg.v1i.folder/
```

Install FFmpeg and yt-dlp if video acquisition is required.

## 6. Running the Project

### Task 1: SAM Segmentation

```bash
python3 src/task1_batch_sam.py
```

This generates candidate masks, selected masks, and segmentation overlays.

### Task 2: Classical Segmentation Experiments

```bash
python3 src/task2/run_dev_experiments.py
```

Four methods were implemented:

1. HSV thresholding.
2. HSV thresholding with connected-component filtering.
3. GrabCut with rectangle initialization.
4. GrabCut with mask initialization.

### Task 3: Quantitative Evaluation

```bash
python3 src/task2/evaluate.py
```

This generates individual and summary evaluation metrics and visual comparisons.

### Task 4: Video Segmentation and Tracking

```bash
python3 src/task4/track_eggs.py
```

The video pipeline uses HSV thresholding, morphological operations, contour filtering, and centroid-based matching to assign persistent track IDs.

## 7. Experimental Results

### Final Evaluation Results

The selected GrabCut mask-initialization method was evaluated on 22 evaluation images.

| Metric | Result |
|---|---:|
| Mean IoU | 47.39% |
| Dice Score | 59.17% |
| Precision | 60.09% |
| Recall | 68.55% |

**Important:** These metrics measure agreement with SAM v5-generated pseudo-labels, not human-annotated ground truth.

### Video Tracking Results

| Parameter | Result |
|---|---:|
| Video resolution | 1280 × 720 |
| Frame rate | 24 FPS |
| Duration | 60.02 seconds |
| Frames processed | 1,440 |
| Baseline track IDs assigned | 10 |

The baseline generated an annotated video and a CSV containing frame numbers, track IDs, centroid coordinates, and mask areas.

## 8. Limitations

- SAM-generated masks are pseudo-labels and may contain false positives or missed objects.
- Classical segmentation is sensitive to lighting, reflections, and background conditions.
- Hand occlusion can interrupt egg detection.
- Centroid-based tracking can produce fragmented IDs when eggs disappear and reappear.
- The number of assigned track IDs does not necessarily represent the number of distinct physical eggs.
- Video identity accuracy has not been validated against manually annotated tracking ground truth.

## 9. Future Improvements

- Create manually annotated segmentation masks for more reliable evaluation.
- Improve egg detection under partial occlusion.
- Explore Kalman filtering and Hungarian assignment for tracking.
- Evaluate identity switches and track fragmentation using annotated video sequences.
- Improve segmentation robustness under changing illumination.

## 10. Output Files

Important generated artifacts:

- SAM segmentation masks and overlays: `outputs/task1_sam_batch/`
- Development metrics: `outputs/task2/evaluation/`
- Final evaluation metrics: `outputs/task2/final_evaluation/`
- Annotated video: `outputs/task4/egg_tracking_output.mp4`
- Tracking log: `outputs/task4/egg_tracking_log.csv`

**Video source:** [Raw and Boiled Egg Spinning Experiment](https://www.youtube.com/watch?v=Y4JyV94y2_A)

## 11. AI Use Disclosure

Generative AI assistance was used for code drafting, debugging guidance, experiment interpretation, and report preparation. Implementation, execution, output inspection, and final verification were performed as part of the project workflow.

---

**Author:** [K.V.RAM RITHEESH]

**Project:** Egg Segmentation and Tracking using Computer Vision