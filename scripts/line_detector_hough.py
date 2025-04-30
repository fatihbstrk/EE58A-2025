import cv2
import numpy as np
import json
from pathlib import Path

def detect_lines_hough(image_path, rho=1, theta=np.pi/180, threshold=50, min_line_length=20, max_line_gap=5):
    """Detect lines using Hough Transform and label them with a net_id."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Optional: slight blur to reduce noise
    blurred = cv2.GaussianBlur(img, (3, 3), 0)

    # Canny edge detection
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # Hough Line Transform
    raw_lines = cv2.HoughLinesP(edges, rho, theta, threshold,
                                minLineLength=min_line_length, maxLineGap=max_line_gap)

    # Label lines with net_id
    labeled_lines = []
    if raw_lines is not None:
        for idx, line in enumerate(raw_lines):
            x1, y1, x2, y2 = line[0]
            # Forcefully convert numpy int to Python int
            labeled_lines.append({
                "net_id": f"net_{idx}",
                "line": [int(x1), int(y1), int(x2), int(y2)]  # Convert NumPy ints to Python int
            })
    
    return labeled_lines

def save_lines_json(lines, save_path):
    """Save detected lines with net IDs to JSON."""
    with open(save_path, "w") as f:
        json.dump(lines, f, indent=2)

def visualize_lines(image_path, lines, save_path):
    """Visualize detected lines."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Draw each line from labeled_lines
    for entry in lines:
        x1, y1, x2, y2 = entry["line"]
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

    # Save the image with overlays
    cv2.imwrite(str(save_path), img)
