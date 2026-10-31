
import cv2
import numpy as np
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================

VIDEO_PATH = "data/video/egg_tracking_video.mp4"
OUTPUT_PATH = "outputs/task4/egg_detection_test.jpg"

# White-object HSV threshold
LOWER_WHITE = np.array([0, 0, 145])
UPPER_WHITE = np.array([180, 110, 255])

# Detection filters
MIN_AREA = 10000
MIN_WIDTH = 100
MIN_HEIGHT = 80

MIN_ASPECT_RATIO = 0.55
MAX_ASPECT_RATIO = 1.8

# Frame to inspect (milliseconds)
FRAME_TIME_MS = 10000


# ==========================================
# DETECT EGGS IN ONE FRAME
# ==========================================

def detect_eggs(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Detect bright, low-saturation regions
    mask = cv2.inRange(
        hsv,
        LOWER_WHITE,
        UPPER_WHITE
    )

    # Remove noise
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Fill small gaps
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Find external contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    detections = []

    for contour in contours:

        area = cv2.contourArea(contour)

        # Filter small regions
        if area < MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Filter small bounding boxes
        if w < MIN_WIDTH or h < MIN_HEIGHT:
            continue

        aspect_ratio = w / float(h)

        # Filter unusual shapes
        if not (
            MIN_ASPECT_RATIO
            <= aspect_ratio
            <= MAX_ASPECT_RATIO
        ):
            continue

        moments = cv2.moments(contour)

        if moments["m00"] == 0:
            continue

        cx = int(
            moments["m10"] / moments["m00"]
        )

        cy = int(
            moments["m01"] / moments["m00"]
        )

        detections.append({
            "contour": contour,
            "bbox": (x, y, w, h),
            "centroid": (cx, cy),
            "area": area,
            "aspect_ratio": aspect_ratio
        })

    return detections, mask


# ==========================================
# MAIN
# ==========================================

def main():

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    # Move to the selected timestamp
    cap.set(
        cv2.CAP_PROP_POS_MSEC,
        FRAME_TIME_MS
    )

    ret, frame = cap.read()

    cap.release()

    if not ret:
        raise RuntimeError(
            "Could not read the selected video frame"
        )

    detections, mask = detect_eggs(frame)

    result = frame.copy()

    # Draw detections
    for index, detection in enumerate(detections, start=1):

        contour = detection["contour"]
        x, y, w, h = detection["bbox"]
        cx, cy = detection["centroid"]

        # Green contour
        cv2.drawContours(
            result,
            [contour],
            -1,
            (0, 255, 0),
            3
        )

        # Blue bounding box
        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        # Centroid
        cv2.circle(
            result,
            (cx, cy),
            6,
            (0, 0, 255),
            -1
        )

        # Detection label
        cv2.putText(
            result,
            f"Egg {index}",
            (x, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        print(
            f"Egg {index}: "
            f"Area={detection['area']:.0f}, "
            f"Centroid={detection['centroid']}"
        )

    # Add total count
    cv2.putText(
        result,
        f"Detected Eggs: {len(detections)}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    # Save output
    Path(OUTPUT_PATH).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(
        OUTPUT_PATH,
        result
    )

    print("\n================================")
    print("EGG DETECTION SUMMARY")
    print("================================")
    print("Detected objects:", len(detections))
    print("Saved output:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
