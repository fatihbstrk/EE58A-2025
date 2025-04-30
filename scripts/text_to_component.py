import numpy as np

# Set a higher default threshold for assignment (adjust as needed)
DEFAULT_DISTANCE_THRESHOLD = 100

# Dictionary mapping component types to possible keywords.
# Consider adding "VDD" if it should map to Voltage Source.
COMPONENT_KEYWORDS = {
    "MOSFET": ["W=", "M", "MOSFET", "M1", "M2"],
    "Capacitor": ["F", "Cap", "C", "Capacitor"],
    "Resistor": ["R", "Resistor", "Ohm"],
    "Voltage Source": ["Vin", "Vout", "DC", "Source", "VDD"],
}

def calculate_distance(bbox1, bbox2):
    """
    Calculate the Euclidean distance between the centers of two bounding boxes.
    Supports OCR boxes (8 points) and YOLO boxes (4 points).
    """
    # Determine center for bbox1
    if len(bbox1) == 8:  # OCR bounding box: [x1, y1, x2, y2, x3, y3, x4, y4]
        x1, y1, x2, y2, x3, y3, x4, y4 = bbox1
        center1 = ((x1 + x3) / 2, (y1 + y3) / 2)
    else:  # YOLO bounding box: [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox1
        center1 = ((x1 + x2) / 2, (y1 + y2) / 2)

    # Determine center for bbox2 (YOLO boxes assumed to be 4 points)
    x1, y1, x2, y2 = bbox2
    center2 = ((x1 + x2) / 2, (y1 + y2) / 2)

    return np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

def identify_text_type(text):
    """
    Identify the component type for an OCR text entry using the keyword dictionary.
    Returns the component type (e.g., 'MOSFET') if found, else "Unknown".
    """
    lower_text = text.lower()
    for component_type, keywords in COMPONENT_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in lower_text:
                return component_type
    return "Unknown"

def assign_text_to_component(ocr_texts, component_bboxes, distance_threshold=DEFAULT_DISTANCE_THRESHOLD):
    """
    Assign OCR texts to detected components based on semantic clues and proximity.
    
    For each OCR text:
      - Identify its component type using known keywords.
      - For each YOLO-detected component whose label contains the text type (case-insensitive),
        compute the center-to-center distance.
      - If the closest component is within the distance threshold, assign the text.
    
    Returns a list of assignments. Debug prints are included to trace distances.
    """
    assignments = []
    
    for text_entry in ocr_texts:
        text_content = text_entry['text']
        text_type = identify_text_type(text_content)
        
        # Skip texts that do not match any known component type
        if text_type == "Unknown":
            print(f"Skipping text '{text_content}' as its type is Unknown.")
            continue
        
        closest_component = None
        min_distance = float('inf')
        
        for component in component_bboxes:
            # Match component label and text type in a case-insensitive manner
            if text_type.lower() in component['label'].lower():
                distance = calculate_distance(text_entry['bbox'], component['bbox'])
                print(f"Comparing text '{text_content}' (type {text_type}) with component '{component['label']}' yields distance {distance:.2f}.")
                if distance < min_distance:
                    min_distance = distance
                    closest_component = component
        
        # Only assign if a matching component was found within the threshold
        if closest_component and min_distance <= distance_threshold:
            assignment = {
                "component": closest_component["label"],
                "text": text_content,
                "confidence": text_entry["confidence"],
                "text_bbox": text_entry["bbox"],
                "component_bbox": closest_component["bbox"],
                "text_type": text_type,
                "distance": min_distance
            }
            assignments.append(assignment)
            print(f"Assigned text '{text_content}' to component '{closest_component['label']}' with distance {min_distance:.2f}.")
        else:
            print(f"No close enough component found for text '{text_content}' (type {text_type}).")
            
    return assignments
