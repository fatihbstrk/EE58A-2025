import easyocr
import json
from pathlib import Path

# Initialize OCR Reader
reader = easyocr.Reader(['en'], gpu=True)

# In ocr_utils.py:
def extract_text_with_positions(image_path, save_path=None, conf_threshold=0.3):
    # Convert image_path to string if it's a Path object
    if isinstance(image_path, Path):
        image_path = str(image_path)

    # Read text from image using EasyOCR
    results = reader.readtext(image_path)
    
    parsed_results = []
    for bbox, text, conf in results:
        if conf >= conf_threshold:
            # bbox is a list of 4 corner points: [[x1, y1], [x2, y2], ...]
            flat_bbox = [int(coord) for point in bbox for coord in point]
            parsed_results.append({
                "text": str(text),
                "confidence": float(conf),
                "bbox": flat_bbox
            })

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(parsed_results, f, indent=4)

    return parsed_results



