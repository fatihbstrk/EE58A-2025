import logging
from ultralytics import YOLO
import cv2
from pathlib import Path
import json
import torch
import contextlib
import io

# ─── Suppress Ultralytics logger (info/debug) ───
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# Load YOLOv8 model quietly
model_path = "trained_models/yolov8_best.pt"
yolo_model = YOLO(model_path, verbose=False)
yolo_model.to("cuda" if torch.cuda.is_available() else "cpu")

def detect_components(image_path, output_dir):
    """
    Run YOLOv8-based component detection on image_path,
    save outputs into output_dir, and return a list of detections.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # ─── SILENT INFERENCE ───
    # redirect both stdout & stderr so absolutely nothing prints
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        results = yolo_model(
            img,
            verbose=False,   # no model banner
            show=False,      # no OpenCV window
            save=False,      # don't write any images
            save_txt=False   # don't write .txt outputs
        )[0]

    # ─── post‐process as before ───
    boxes   = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)
    names   = yolo_model.model.names

    vis_img        = img.copy()
    detection_data = []

    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = map(int, box)
        label = names[cls]
        detection_data.append({
            "label": label,
            "bbox": [x1, y1, x2, y2]
        })
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(
            vis_img, label, (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2
        )

    # Save outputs
    cv2.imwrite(str(output_dir / "detected_components.png"), vis_img)
    with open(output_dir / "components.json", "w") as f:
        json.dump(detection_data, f, indent=4)

    return detection_data
