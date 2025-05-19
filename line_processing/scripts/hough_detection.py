# scripts/hough_detection.py

import cv2
import numpy as np

def detect_lines(
    image,
    rho: float = 1,
    theta: float = np.pi / 180,
    threshold: int = 15,
    min_line_length: int = 10,
    max_line_gap: int = 15
):
    """
    Perform Probabilistic Hough Transform on a binary image.

    Args:
        image: single-channel (binary) image (numpy array).
        rho: distance resolution in pixels of the Hough grid.
        theta: angle resolution in radians of the Hough grid.
        threshold: minimum number of intersections to detect a line.
        min_line_length: minimum length of line to be accepted.
        max_line_gap: maximum gap between segments to link them.
    Returns:
        lines: List of segments, each as [x1, y1, x2, y2].
    """
    # Ensure image is uint8
    img = image
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    lines = cv2.HoughLinesP(
        img,
        rho=rho,
        theta=theta,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )
    if lines is None:
        return []

    # Flatten to a Python list of [x1,y1,x2,y2]
    return [line[0].tolist() for line in lines]
