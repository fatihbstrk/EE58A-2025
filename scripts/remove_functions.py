import cv2
import numpy as np

def remove_components(image: np.ndarray, dim_matrix: np.ndarray, ratio: float = 0.85):
    """
    Removes detected component regions by replacing them with white pixels inside the bounding box.
    
    Args:
        image (np.ndarray): Original image.
        dim_matrix (np.ndarray): Array with component bounding box dimensions 
                                 in the form [x1, y1, x2, y2] for each box.
        ratio (float): The fraction of the bounding box to keep as margin.
    
    Returns:
        img_removed (np.ndarray): Image with components removed.
    """
    img_removed = image.copy()
    for dim in dim_matrix:
        start = (int(dim[0]), int(dim[1]))
        end = (int(dim[2]), int(dim[3]))
        width = end[0] - start[0]
        height = end[1] - start[1]
        
        # Calculate the amount to reduce (to leave a margin around the component)
        reduction_w = int(width * (1 - ratio) / 2)
        reduction_h = int(height * (1 - ratio) / 2)
        new_start = (start[0] + reduction_w, start[1] + reduction_h)
        new_end = (end[0] - reduction_w, end[1] - reduction_h)
        
        # Replace the region inside the bounding box with white pixels (255)
        img_removed[new_start[1]:new_end[1], new_start[0]:new_end[0]] = 255
    return img_removed

def remove_text(img, ocr_data):
    """
    Removes text from the image using OCR bounding boxes.

    Args:
        img (numpy.ndarray): The input image.
        ocr_data (list): List of OCR text regions with bounding boxes.

    Returns:
        numpy.ndarray: The image with text regions removed.
    """
    img_copy = img.copy()
    for item in ocr_data:
        if "bbox" in item:
            # Convert 8-point polygon bbox to bounding rectangle
            x_coords = item["bbox"][::2]
            y_coords = item["bbox"][1::2]
            x1, y1 = min(x_coords), min(y_coords)
            x2, y2 = max(x_coords), max(y_coords)
            # Fill the text region with white
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 255, 255), thickness=-1)
    return img_copy

