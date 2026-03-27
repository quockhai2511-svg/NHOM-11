import cv2
import numpy as np
import argparse
from ultralytics import YOLO

# Classes vehicle của COCO
VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck
BOX_COLOR = (0, 255, 0)


# ───────────────── Tracker ─────────────────
class CentroidTracker:

    def __init__(self, max_disappeared=30, max_distance=80):

        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.counted_ids = set()

        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):

        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, obj_id):

        del self.objects[obj_id]
        del self.disappeared[obj_id]

    def update(self, centroids):

        if len(centroids) == 0:

            for obj_id in list(self.disappeared.keys()):

                self.disappeared[obj_id] += 1

                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)

            return self.objects

        input_centroids = np.array(centroids)

        if len(self.objects) == 0:

            for c in input_centroids:
                self.register(c)

            return self.objects

        obj_ids = list(self.objects.keys())
        obj_centroids = list(self.objects.values())

        D = np.linalg.norm(np.array(obj_centroids)[:, None] - input_centroids, axis=2)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):

            if row in used_rows or col in used_cols:
                continue

            if D[row, col] > self.max_distance:
                continue

            obj_id = obj_ids[row]

            self.objects[obj_id] = input_centroids[col]
            self.disappeared[obj_id] = 0

            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(len(obj_ids))) - used_rows
        unused_cols = set(range(len(input_centroids))) - used_cols

        for row in unused_rows:

            obj_id = obj_ids[row]
            self.disappeared[obj_id] += 1

            if self.disappeared[obj_id] > self.max_disappeared:
                self.deregister(obj_id)

        for col in unused_cols:
            self.register(input_centroids[col])

        return self.objects


# ───────────────── HUD ─────────────────
def draw_hud(frame, total, fps):

    overlay = frame.copy()

    cv2.rectangle(overlay, (10, 10), (230, 80), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame,
                f"FPS: {fps:.1f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1)

    cv2.putText(frame,
                f"TOTAL: {total}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2)


# ───────────────── MAIN ─────────────────
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="output.mp4")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--line-ratio", type=float, default=0.5)

    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input)

    if not cap.isOpened():

        print("Không mở được video")
        return

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30

    COUNT_LINE_Y = int(H * args.line_ratio)

    out = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (W, H)
    )

    print("Loading YOLO model...")
    model = YOLO(args.model)

    tracker = CentroidTracker()

    total_count = 0
    frame_idx = 0

    tick = cv2.getTickCount()
    fps_display = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_idx += 1

        results = model(frame, conf=args.conf, verbose=False)[0]

        detections = []

        for box in results.boxes:

            cls = int(box.cls[0])

            if cls not in VEHICLE_CLASS_IDS:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
        

            detections.append((cx, cy, x1, y1, x2, y2))

        centroids = [(d[0], d[1]) for d in detections]

        tracked = tracker.update(centroids)

        # ───── Đếm xe ─────
        for obj_id, centroid in tracked.items():

            cy = int(centroid[1])

            if cy > COUNT_LINE_Y and obj_id not in tracker.counted_ids:

                tracker.counted_ids.add(obj_id)
                total_count += 1

        # ───── Vẽ bbox + ID ─────
        for cx, cy, x1, y1, x2, y2 in detections:

            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

            cv2.circle(frame, (cx, cy), 4, BOX_COLOR, -1)

            vehicle_id = None
            min_dist = 50

            for obj_id, centroid in tracked.items():

                dist = np.linalg.norm(np.array([cx, cy]) - np.array(centroid))

                if dist < min_dist:

                    min_dist = dist
                    vehicle_id = obj_id

            if vehicle_id is not None:

                cv2.putText(frame,
                            f"ID {vehicle_id}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2)

        # ───── Vẽ line ─────
        # cv2.line(frame, (0, COUNT_LINE_Y), (W, COUNT_LINE_Y), (0, 255, 255), 2)

        cv2.putText(frame,
                    "COUNT LINE",
                    (10, COUNT_LINE_Y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2)

        # ───── FPS ─────
        if frame_idx % 30 == 0:

            tock = cv2.getTickCount()

            fps_display = 30 / ((tock - tick) / cv2.getTickFrequency())

            tick = tock

        draw_hud(frame, total_count, fps_display)

        out.write(frame)

        cv2.imshow("Vehicle Counter", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Total vehicles:", total_count)


if __name__ == "__main__":
    main()