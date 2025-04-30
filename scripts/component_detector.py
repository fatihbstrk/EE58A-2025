from ultralytics import YOLO
import cv2
from pathlib import Path
import json
import torch


# Load YOLOv8 model and set device to GPU (if available)
model_path = "trained_models/yolov8_best.pt"
yolo_model = YOLO(model_path)
yolo_model.to("cuda" if torch.cuda.is_available() else "cpu")

def detect_components(image_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Run YOLO detection
    results = yolo_model(img)[0]

    boxes = results.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
    classes = results.boxes.cls.cpu().numpy().astype(int)
    names = yolo_model.model.names

    # Draw on image
    vis_img = img.copy()
    detection_data = []

    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = map(int, box)
        label = names[cls]
        detection_data.append({
            "label": label,
            "bbox": [x1, y1, x2, y2]  # Save the coordinates
        })
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(vis_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Save output image
    cv2.imwrite(str(output_dir / "detected_components.png"), vis_img)

    # Save detection as JSON with bounding boxes
    with open(output_dir / "components.json", "w") as f:
        json.dump(detection_data, f, indent=4)

    return detection_data

