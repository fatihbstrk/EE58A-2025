# intersections.py

import numpy as np
from typing import List, Tuple, Optional

Segment = Tuple[int, int, int, int]
Point   = Tuple[int, int]

def intersection(
    line1: Segment,
    line2: Segment
) -> Optional[Point]:
    """
    Compute intersection of two segments.
    Returns (x,y) if they cross within both segments, else None.
    """
    x1,y1,x2,y2 = line1
    x3,y3,x4,y4 = line2

    a1 = y2-y1; b1 = x1-x2; c1 = a1*x1 + b1*y1
    a2 = y4-y3; b2 = x3-x4; c2 = a2*x3 + b2*y3

    det = a1*b2 - a2*b1
    if det == 0:
        return None

    x0 = (b2*c1 - b1*c2) / det
    y0 = (a1*c2 - a2*c1) / det

    if (
        min(x1,x2) <= x0 <= max(x1,x2) and
        min(y1,y2) <= y0 <= max(y1,y2) and
        min(x3,x4) <= x0 <= max(x3,x4) and
        min(y3,y4) <= y0 <= max(y3,y4)
    ):
        return (int(round(x0)), int(round(y0)))
    return None

def segmented_intersections(
    horizontals: List[Segment],
    verticals:   List[Segment]
) -> List[Point]:
    """
    Return unique intersection points between two segment lists.
    """
    pts = []
    for h in horizontals:
        for v in verticals:
            p = intersection(h, v)
            if p is not None:
                pts.append(p)
    # remove duplicates
    return list(set(pts))
