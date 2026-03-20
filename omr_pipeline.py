import cv2
import numpy as np
import os
import csv
from ultralytics import YOLO

# ==========================================
# SETTINGS
# ==========================================
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

ROW_MODEL_PATH = BASE_DIR / "sheet_model" / "best.pt"
BUBBLE_MODEL_PATH = BASE_DIR / "bubble_model" / "best.pt"
DEBUG_FOLDER = r"C:\Users\user\Desktop\Testing\debug_answers6.0"

# Bubble model class ids
FILLED_CLASS_ID = 0
UNFILLED_CLASS_ID = 1
QUESTION_CLASS_ID = 2

OPTIONS = ["A", "B", "C", "D"]

# Row model settings
ROW_CONF = 0.25

# Bubble model settings
BUBBLE_CONF = 0.06
BUBBLE_IMGSZ = 1440


# ==========================================
# HELPERS
# ==========================================
def ensure_folder(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# ==========================================
# DETECT ANSWER FROM A CROPPED ROW IMAGE
# ==========================================
def detect_answer_from_row_image(model, row_img, debug_folder=None, row_index=None, save_debug=False):
    if row_img is None or row_img.size == 0:
        return "READ_ERROR"

    h, w = row_img.shape[:2]

    results = model.predict(
        row_img,
        conf=BUBBLE_CONF,
        imgsz=BUBBLE_IMGSZ,
        verbose=False
    )

    detections = []

    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            return "NO_DETECTION"

        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        for box, cls, conf in zip(boxes, classes, confs):
            x1, y1, x2, y2 = box
            detections.append({
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "cx": float((x1 + x2) / 2),
                "cy": float((y1 + y2) / 2),
                "class": int(cls),
                "conf": float(conf)
            })

    if len(detections) == 0:
        return "NO_DETECTION"

    detections = sorted(detections, key=lambda d: d["cx"])

    question_boxes = [d for d in detections if d["class"] == QUESTION_CLASS_ID]

    if len(question_boxes) > 0:
        qbox = sorted(question_boxes, key=lambda d: d["cx"])[0]
        option_start = int(qbox["x2"]) + 3
    else:
        option_start = int(w * 0.20)

    option_end = w - 3

    if option_end <= option_start:
        return "INVALID_REGION"

    region_width = option_end - option_start
    slot_width = region_width / 4.0

    filled_detections = [d for d in detections if d["class"] == FILLED_CLASS_ID]

    detected_options = []

    for d in filled_detections:
        cx = d["cx"]

        if cx < option_start or cx > option_end:
            continue

        idx = int((cx - option_start) / slot_width)

        if idx < 0:
            idx = 0
        if idx > 3:
            idx = 3

        detected_options.append(OPTIONS[idx])

    unique_options = []
    for op in detected_options:
        if op not in unique_options:
            unique_options.append(op)

    if len(unique_options) == 0:
        final_answer = "NO_ANSWER"
    elif len(unique_options) == 1:
        final_answer = unique_options[0]
    else:
        final_answer = "INVALID_ANSWER"

    if save_debug and debug_folder:
        ensure_folder(debug_folder)
        debug_img = row_img.copy()

        for d in detections:
            x1, y1, x2, y2 = int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])
            cls_id = d["class"]

            if cls_id == FILLED_CLASS_ID:
                color = (0, 255, 0)
                label = "filled"
            elif cls_id == UNFILLED_CLASS_ID:
                color = (255, 255, 0)
                label = "unfilled"
            else:
                color = (0, 0, 255)
                label = "question"

            cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 1)
            cv2.putText(
                debug_img,
                label,
                (x1, max(10, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1
            )

        cv2.line(debug_img, (option_start, 0), (option_start, h), (255, 0, 0), 1)
        cv2.line(debug_img, (option_end, 0), (option_end, h), (255, 0, 0), 1)

        for i in range(4):
            x1 = int(option_start + i * slot_width)
            x2 = int(option_start + (i + 1) * slot_width)
            xm = (x1 + x2) // 2

            cv2.line(debug_img, (x1, 0), (x1, h), (200, 200, 200), 1)
            cv2.putText(
                debug_img,
                OPTIONS[i],
                (xm - 5, h - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1
            )

        cv2.putText(
            debug_img,
            f"Ans: {final_answer}",
            (5, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

        debug_name = f"row_{row_index:03d}.jpg" if row_index is not None else "debug_row.jpg"
        out_path = os.path.join(debug_folder, debug_name)
        cv2.imwrite(out_path, debug_img)

    return final_answer


# ==========================================
# MAIN PIPELINE FUNCTION FOR API
# ==========================================
def process_omr_image_file(
    image_path,
    row_model,
    bubble_model,
    output_csv=None,
    debug_folder=None,
    save_debug=False
):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    results = row_model.predict(image_path, conf=ROW_CONF, verbose=False)

    boxes = []

    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue

        xyxy = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy()

        for box, c in zip(xyxy, cls):
            class_name = row_model.names[int(c)]

            if str(class_name).lower() != "row":
                continue

            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            boxes.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "cx": cx,
                "cy": cy
            })

    if len(boxes) == 0:
        raise RuntimeError("No row detections found.")

    boxes = sorted(boxes, key=lambda b: b["cx"])

    n_cols = 4
    columns = np.array_split(boxes, n_cols)
    columns = [list(col) for col in columns]
    columns = sorted(columns, key=lambda col: np.mean([b["cx"] for b in col]) if len(col) > 0 else 0)

    ordered_boxes = []
    for col in columns:
        col_sorted = sorted(col, key=lambda b: b["cy"])
        ordered_boxes.extend(col_sorted)

    all_results = []

    for idx, b in enumerate(ordered_boxes, start=1):
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.shape[1], x2)
        y2 = min(img.shape[0], y2)

        row_crop = img[y1:y2, x1:x2]

        answer = detect_answer_from_row_image(
            bubble_model,
            row_crop,
            debug_folder=debug_folder,
            row_index=idx,
            save_debug=save_debug
        )

        all_results.append({
            "question": idx,
            "answer": answer
        })

    if output_csv:
        with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Question", "Answer"])
            for row in all_results:
                writer.writerow([row["question"], row["answer"]])

    return {
        "total_rows_detected": len(ordered_boxes),
        "results": all_results,
        "csv_path": output_csv
    }