import json
from pathlib import Path

def combine_yolo_ocr_data(img_id, proc_dir):
    """
    Combine YOLO detection and OCR results for a given image.
    Args:
        img_id: The image ID to combine data for (e.g., the filename without extension).
        proc_dir: Directory where the processed image data (OCR and YOLO results) are saved.
    Returns:
        A dictionary containing combined YOLO and OCR data or None if data not found.
    """
    ocr_json_path = proc_dir / img_id / "ocr_output.json"
    yolo_json_path = proc_dir / img_id / "components.json"
    
    if not ocr_json_path.exists() or not yolo_json_path.exists():
        print(f"❌ Data not found for {img_id}")
        return None
    
    # Load OCR data
    with open(ocr_json_path, "r") as f:
        ocr_data = json.load(f)
    
    # Load YOLO data
    with open(yolo_json_path, "r") as f:
        yolo_data = json.load(f)
    
    # Standardize bounding box key names for consistency
    for component in yolo_data:
        component['bbox'] = component.pop('box')  # Rename 'box' to 'bbox' for consistency

    # Prepare data for assigning texts to components
    combined_data = {
        'texts': ocr_data,  # OCR results with text and bounding boxes
        'components': yolo_data  # YOLO results with labels and bounding boxes
    }
    
    return combined_data
