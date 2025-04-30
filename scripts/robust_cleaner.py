import cv2
import numpy as np
from pathlib import Path

def robust_clean_image(image_path, min_area=50, aspect_ratio_thresh=(0.1, 10.0)):
    """Robustly remove large text/symbols from schematic while preserving wires and node dots.
    
    Args:
        image_path (str or Path): Path to raw schematic image.
        min_area (int): Minimum area to consider for removal.
        aspect_ratio_thresh (tuple): Acceptable (min, max) aspect ratio (width/height) range.
    
    Returns:
        cleaned_img (numpy array): Cleaned schematic image.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    # Minimal Gaussian blur
    blurred = cv2.GaussianBlur(img, (3, 3), 0.5)

    # Stronger adaptive thresholding
    binary = cv2.adaptiveThreshold(blurred, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 
                                   blockSize=11, C=1)

    # Morphological closing to connect wires better
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create mask
    mask = np.zeros_like(binary)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h if h != 0 else 0

        if aspect_ratio_thresh[0] <= aspect_ratio <= aspect_ratio_thresh[1]:
            # Remove it
            cv2.drawContours(mask, [cnt], -1, 255, -1)

    # Remove unwanted blobs
    cleaned = cv2.bitwise_not(cv2.bitwise_or(binary, mask))

    return cleaned

def save_cleaned_image(cleaned_img, save_path):
    """Save cleaned schematic image."""
    cv2.imwrite(str(save_path), cleaned_img)
