import cv2
import numpy as np
from pathlib import Path

def detect_node_dots(image_path, min_area=3, max_area=30, circularity_thresh=(0.6, 1.2)):
    """Detect small round dark blobs (node dots) from a schematic image.
    
    Args:
        image_path (str or Path): Path to the schematic image.
        min_area (int): Minimum contour area to consider.
        max_area (int): Maximum contour area to consider.
        circularity_thresh (tuple): Acceptable (min, max) circularity values.
        
    Returns:
        List of (x, y) coordinates of detected dots.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    
    # Threshold
    _, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dots = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area <= area <= max_area):
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * (area / (perimeter * perimeter))

        if circularity_thresh[0] <= circularity <= circularity_thresh[1]:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                dots.append((cx, cy))

    return dots

def visualize_dots(image_path, dots, save_path):
    """Draw detected dots on image."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    for (x, y) in dots:
        cv2.circle(img, (x, y), 6, (0, 0, 255), 2)

    cv2.imwrite(str(save_path), img)
