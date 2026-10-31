
import cv2
import numpy as np
import csv
from pathlib import Path

# =====================================================
# CONFIGURATION
# =====================================================

VIDEO_PATH = "data/video/egg_tracking_video.mp4"

OUTPUT_DIR = Path("outputs/task4")
OUTPUT_VIDEO = OUTPUT_DIR / "egg_tracking_output_v2.mp4"
OUTPUT_CSV = OUTPUT_DIR / "egg_tracking_log_v2.csv"

# HSV threshold for white eggs
LOWER_WHITE = np.array([0, 0, 145])
UPPER_WHITE = np.array([180, 110, 255])

# Detection filters
MIN_AREA = 10000
MIN_WIDTH = 100
MIN_HEIGHT = 80

MIN_ASPECT_RATIO = 0.55
MAX_ASPECT_RATIO = 1.8

# Tracking parameters
MAX_DISTANCE = 180
MAX_MISSED_FRAMES = 48

# Ignore upper part of video
ROI_TOP_RATIO = 0.35

# Limit velocity prediction
MAX_PREDICTION_DISTANCE = 250


# =====================================================
# MOTION-AWARE TRACKER
# =====================================================

class EggTracker:

    def __init__(self):

        self.next_id = 1
        self.tracks = {}

    def predict_position(self, track):

        history = track["history"]

        current_x, current_y = track["centroid"]

        # If only one observation exists, use last position
        if len(history) < 2:
            return current_x, current_y

        previous_x, previous_y = history[-2]
        last_x, last_y = history[-1]

        # Estimate velocity
        velocity_x = last_x - previous_x
        velocity_y = last_y - previous_y

        # Predict position after missed frames
        steps = track["missed"] + 1

        predicted_x = current_x + velocity_x * steps
        predicted_y = current_y + velocity_y * steps

        # Prevent excessive extrapolation
        dx = predicted_x - current_x
        dy = predicted_y - current_y

        distance = np.sqrt(dx * dx + dy * dy)

        if distance > MAX_PREDICTION_DISTANCE:
            scale = MAX_PREDICTION_DISTANCE / distance

            predicted_x = current_x + dx * scale
            predicted_y = current_y + dy * scale

        return int(predicted_x), int(predicted_y)

    def update(self, detections):

        # Mark tracks as temporarily missing
        for track in self.tracks.values():
            track["missed"] += 1

        possible_matches = []

        # Compare detections with predicted positions
        for track_id, track in self.tracks.items():

            predicted_x, predicted_y = self.predict_position(track)

            for detection_index, detection in enumerate(detections):

                new_x, new_y = detection["centroid"]

                distance = np.sqrt(
                    (predicted_x - new_x) ** 2 +
                    (predicted_y - new_y) ** 2
                )

                # Allow a larger search radius for temporarily missing tracks
                allowed_distance = min(
                    MAX_DISTANCE + track["missed"] * 3,
                    300
                )

                if distance <= allowed_distance:

                    possible_matches.append(
                        (distance, track_id, detection_index)
                    )

        # Global greedy one-to-one assignment:
        # closest available pairs are matched first.
        possible_matches.sort(key=lambda item: item[0])

        used_tracks = set()
        used_detections = set()

        for distance, track_id, detection_index in possible_matches:

            if track_id in used_tracks:
                continue

            if detection_index in used_detections:
                continue

            detection = detections[detection_index]
            track = self.tracks[track_id]

            track["centroid"] = detection["centroid"]
            track["bbox"] = detection["bbox"]
            track["contour"] = detection["contour"]
            track["area"] = detection["area"]

            track["missed"] = 0

            # Keep recent position history
            track["history"].append(detection["centroid"])

            if len(track["history"]) > 2:
                track["history"].pop(0)

            used_tracks.add(track_id)
            used_detections.add(detection_index)

        # Create tracks for unmatched detections
        for detection_index, detection in enumerate(detections):

            if detection_index in used_detections:
                continue

            track_id = self.next_id
            self.next_id += 1

            self.tracks[track_id] = {

                "centroid": detection["centroid"],

                "bbox": detection["bbox"],

                "contour": detection["contour"],

                "area": detection["area"],

                "missed": 0,

                "history": [
                    detection["centroid"]
                ]
            }

        # Remove expired tracks
        expired_tracks = []

        for track_id, track in self.tracks.items():

            if track["missed"] > MAX_MISSED_FRAMES:
                expired_tracks.append(track_id)

        for track_id in expired_tracks:
            del self.tracks[track_id]

        return self.tracks


# =====================================================
# EGG DETECTION
# =====================================================

def detect_eggs(frame):

    height, width = frame.shape[:2]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(
        hsv,
        LOWER_WHITE,
        UPPER_WHITE
    )

    # Ignore upper portion of video
    roi_top = int(height * ROI_TOP_RATIO)

    mask[:roi_top, :] = 0

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    detections = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if w < MIN_WIDTH or h < MIN_HEIGHT:
            continue

        aspect_ratio = w / float(h)

        if not (
            MIN_ASPECT_RATIO
            <= aspect_ratio
            <= MAX_ASPECT_RATIO
        ):
            continue

        moments = cv2.moments(contour)

        if moments["m00"] == 0:
            continue

        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        detections.append({

            "contour": contour,

            "bbox": (x, y, w, h),

            "centroid": (cx, cy),

            "area": area
        })

    return detections


# =====================================================
# VISUALIZATION
# =====================================================

def get_track_color(track_id):

    colors = [
        (0, 255, 0),
        (255, 100, 0),
        (0, 200, 255),
        (255, 0, 255),
        (255, 255, 0),
        (0, 100, 255)
    ]

    return colors[(track_id - 1) % len(colors)]


def draw_tracks(frame, tracks):

    output = frame.copy()

    active_count = 0

    for track_id, track in tracks.items():

        # Do not draw missing tracks
        if track["missed"] > 0:
            continue

        active_count += 1

        x, y, w, h = track["bbox"]

        cx, cy = track["centroid"]

        color = get_track_color(track_id)

        # Draw segmentation contour
        cv2.drawContours(
            output,
            [track["contour"]],
            -1,
            color,
            3
        )

        # Draw bounding box
        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

        # Draw centroid
        cv2.circle(
            output,
            (cx, cy),
            6,
            (0, 0, 255),
            -1
        )

        # Persistent ID
        cv2.putText(
            output,
            f"Egg ID: {track_id}",
            (x, max(y - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    cv2.putText(
        output,
        f"Active Eggs: {active_count}",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    return output, active_count


# =====================================================
# MAIN VIDEO PROCESSING
# =====================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {VIDEO_PATH}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if fps <= 0:
        fps = 24.0

    print("====================================")
    print("MOTION-AWARE EGG TRACKING V2")
    print("====================================")

    print("Resolution:", width, "x", height)
    print("FPS:", fps)
    print("Total frames:", total_frames)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError(
            "Could not create output video"
        )

    tracker = EggTracker()

    frame_number = 0
    csv_rows = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        # Detect eggs
        detections = detect_eggs(frame)

        # Update motion-aware tracker
        tracks = tracker.update(detections)

        # Draw annotations
        output, active_count = draw_tracks(
            frame,
            tracks
        )

        writer.write(output)

        # Save tracking information
        for track_id, track in tracks.items():

            if track["missed"] > 0:
                continue

            cx, cy = track["centroid"]

            csv_rows.append([
                frame_number,
                track_id,
                cx,
                cy,
                track["area"]
            ])

        if frame_number % 120 == 0:

            print(
                f"Processed {frame_number}/{total_frames} "
                f"frames | Active eggs: {active_count}"
            )

    cap.release()
    writer.release()

    # Save CSV
    with open(
        OUTPUT_CSV,
        "w",
        newline=""
    ) as file:

        csv_writer = csv.writer(file)

        csv_writer.writerow([
            "frame",
            "egg_id",
            "centroid_x",
            "centroid_y",
            "area"
        ])

        csv_writer.writerows(csv_rows)

    print("\n====================================")
    print("V2 TRACKING COMPLETED")
    print("====================================")

    print("Processed frames:", frame_number)

    print("Created video:", OUTPUT_VIDEO)

    print("Created CSV:", OUTPUT_CSV)

    print("Unique IDs assigned:", tracker.next_id - 1)


if __name__ == "__main__":
    main()
