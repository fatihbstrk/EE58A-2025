# component_mapping.py

import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

Point = Tuple[int,int]
Component = Dict  # expects {"label":str, "bbox":[x1,y1,x2,y2]}

def map_terminals_to_components(
    comp_defs:   List[Component],
    terminals:   List[Point],
    term2net:    List[int],
    tol:         int = 3
) -> List[Dict]:
    """
    Assign each terminal to the nearest component edge (top/bottom/left/right),
    within a tolerance on the span.

    Returns a list of entries:
      {component, pin_index, pin_side, x, y, net_id}
    """
    comp_net_map = []

    for (tx, ty), nid in zip(terminals, term2net):
        best = None  # (distance, comp_label, pin_idx, side)

        for comp in comp_defs:
            lbl = comp["label"]
            x1, y1, x2, y2 = comp["bbox"]

            # Top edge?
            if x1 - tol <= tx <= x2 + tol:
                d = abs(ty - y1)
                best = min(best, (d, lbl, 0, "top")) if best else (d, lbl, 0, "top")

            # Bottom edge?
            if x1 - tol <= tx <= x2 + tol:
                d = abs(ty - y2)
                best = min(best, (d, lbl, 1, "bottom")) if best else (d, lbl, 1, "bottom")

            # Left edge?
            if y1 - tol <= ty <= y2 + tol:
                d = abs(tx - x1)
                best = min(best, (d, lbl, 2, "left")) if best else (d, lbl, 2, "left")

            # Right edge?
            if y1 - tol <= ty <= y2 + tol:
                d = abs(tx - x2)
                best = min(best, (d, lbl, 3, "right")) if best else (d, lbl, 3, "right")

        if best:
            _, comp_label, pin_idx, side = best
            comp_net_map.append({
                "component": comp_label,
                "pin_index": pin_idx,
                "pin_side":  side,
                "x":         tx,
                "y":         ty,
                "net_id":    nid
            })

    return comp_net_map


def save_component_net_map(
    img_dir:      Path,
    comp_net_map: List[Dict],
    cleaned_img:  Path
):
    """
    Save `component_net_map.json` in `img_dir/line_results/` and
    draw a debug overlay mapping.png on `cleaned_img`.
    """
    lr = img_dir / "line_results"
    # 1) JSON
    out_json = lr / "component_net_map.json"
    out_json.write_text(json.dumps(comp_net_map, indent=2))

    # 2) Overlay
    clean = cv2.imread(str(cleaned_img), cv2.IMREAD_GRAYSCALE)
    ov    = cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)
    for entry in comp_net_map:
        x, y, nid = entry["x"], entry["y"], entry["net_id"]
        cv2.circle(ov, (x,y), 4, (0,255,0), -1)
        cv2.putText(ov, str(nid), (x+3,y-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)

    cv2.imwrite(str(lr / "mapping.png"), ov)
