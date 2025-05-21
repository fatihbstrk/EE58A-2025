# terminal_detection.py

import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple

Point = Tuple[int,int]

def detect_terminals_midpoint(
    cleaned:       np.ndarray,
    components_json: Path,
    wire_thresh_block: int = 11,
    wire_thresh_C:     int = 2
) -> List[Point]:
    """
    For each bbox in components.json, check its four edge‐midpoints
    and retain those that land on a wire in the thresholded cleaned image.
    """
    # 1) threshold the cleaned image to isolate wires
    blur = cv2.GaussianBlur(cleaned, (9,9), 0)
    th   = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        wire_thresh_block, wire_thresh_C
    )

    # 2) load bboxes
    entries = json.load(open(components_json))
    terminals: List[Point] = []

    for e in entries:
        x1,y1,x2,y2 = e["bbox"]
        # compute midpoints
        mids = [
            (int(round((x1 + x2)/2)), int(y1)),      # top
            (int(round((x1 + x2)/2)), int(y2)),      # bottom
            (int(x1), int(round((y1 + y2)/2))),      # left
            (int(x2), int(round((y1 + y2)/2))),      # right
        ]
        # keep those that lie on a wire pixel
        for (x,y) in mids:
            if 0 <= x < th.shape[1] and 0 <= y < th.shape[0]:
                if th[y, x] > 0:
                    terminals.append((x, y))

    return terminals
