"""
L08 – CV Applications: Real-Time Object Detection with YOLO11
==============================================================
Uses ultralytics YOLO11-nano for live or video-file detection.
Draws bounding boxes, labels, and confidence scores; saves an
annotated sample frame to figures/.

Usage
-----
    python yolo_realtime.py                       # webcam (source 0)
    python yolo_realtime.py --source video.mp4
    python yolo_realtime.py --source 0 --conf 0.5
"""

import argparse
import os

import cv2
from ultralytics import YOLO

SEED = 1337

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def draw_detections(frame, results, conf_thr: float):
    """Draw bounding boxes + labels on *frame* in-place."""
    for r in results:
        for box in r.boxes:
            if float(box.conf) < conf_thr:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls)
            label = f"{r.names[cls_id]} {float(box.conf):.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame


def main():
    parser = argparse.ArgumentParser(description="YOLO11 real-time detection")
    parser.add_argument("--source", default="0",
                        help="Webcam index or video path (default: 0)")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="Confidence threshold")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    model = YOLO("yolo11n.pt")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    saved = False
    print("Press 'q' to quit.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, verbose=False)
        frame = draw_detections(frame, results, args.conf)

        if not saved:
            path = os.path.join(FIG_DIR, "yolo_sample.png")
            cv2.imwrite(path, frame)
            print(f"[saved] {path}")
            saved = True

        cv2.imshow("YOLO Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
