# net_labeling.py

import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Dict

Point = Tuple[int,int]

def make_wire_mask(
    cleaned: np.ndarray,
    block_size: int = 11,
    C: int = 2
) -> np.ndarray:
    """
    Adaptive-threshold the cleaned image to isolate wire pixels (white on black).
    """
    blur = cv2.GaussianBlur(cleaned, (9,9), 0)
    th   = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size, C
    )
    return th

def label_nets(
    wire_mask: np.ndarray
) -> Tuple[np.ndarray, int]:
    """
    Connected-component label the wire mask.
    Returns (label_img, num_labels), where label_img[y,x] ∈ [0..num_labels].
    Background is 0, nets are 1..num_labels.
    """
    # Use 8-connectivity
    num_labels, label_img = cv2.connectedComponents(wire_mask, connectivity=8)
    # label_img: 0=background, 1..num_labels-1 = net IDs
    return label_img, num_labels-1

def map_terminals_to_nets(
    terminals: List[Point],
    label_img: np.ndarray
) -> List[int]:
    """
    For each terminal (x,y), read label_img[y,x] to get its net ID (0 means no net).
    """
    term2net: List[int] = []
    h, w = label_img.shape
    for x,y in terminals:
        if 0 <= x < w and 0 <= y < h:
            term2net.append(int(label_img[y, x]))
        else:
            term2net.append(0)
    return term2net

def save_net_overlays(
    img_dir: Path,
    wire_mask: np.ndarray,
    label_img: np.ndarray
):
    """
    Save debug images:
      - wire_mask.png (binary)
      - nets_colored.png (each net in a random color)
    """
    lr = img_dir / "line_results"
    # wire mask
    cv2.imwrite(str(lr/"wire_mask.png"), wire_mask)

    # colorize labels
    h, w = label_img.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    rng = np.random.RandomState(0)
    # assign random color for each net ID (1..N)
    colors = [(0,0,0)] + [tuple(rng.randint(50,255,3).tolist()) for _ in range(label_img.max())]
    for net_id, col in enumerate(colors):
        canvas[label_img == net_id] = col

    cv2.imwrite(str(lr/"nets_colored.png"), canvas)

def save_mapping_info(
    img_dir: Path,
    comp_json: List[Dict],
    terminals: List[Point],
    term2net: List[int]
):
    """
    For each detected terminal, find its component and which side (pin),
    then write out component_net_map.json.
    """
    mapping = []

    # helper to compute distance from (tx,ty) to each side of a bbox
    for (tx, ty), nid in zip(terminals, term2net):
        best = None  # (dist, component_label, pin_index, side)
        for comp in comp_json:
            lbl = comp["label"]
            x1,y1,x2,y2 = comp["bbox"]

            # only consider if the terminal is roughly adjacent to that edge span
            candidates = []
            if x1 <= tx <= x2:
                candidates.append(("top",    abs(ty - y1),    0))
                candidates.append(("bottom", abs(ty - y2),    1))
            if y1 <= ty <= y2:
                candidates.append(("left",   abs(tx - x1),    2))
                candidates.append(("right",  abs(tx - x2),    3))

            for side, dist, pin_idx in candidates:
                if best is None or dist < best[0]:
                    best = (dist, lbl, pin_idx, side)

        if best is None:
            # didn’t match any component—skip or log
            continue

        _, comp_label, pin_idx, side = best
        mapping.append({
            "component": comp_label,
            "pin_index": pin_idx,
            "pin_side":  side,
            "x":         tx,
            "y":         ty,
            "net_id":    nid
        })

    out = img_dir/"line_results"/"component_net_map.json"
    out.write_text(json.dumps(mapping, indent=2))
