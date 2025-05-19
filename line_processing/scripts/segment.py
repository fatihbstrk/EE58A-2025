# line_processing/scripts/segment.py

import math
from typing import List, Tuple

# Define a segment type for clarity
Segment = Tuple[int, int, int, int]

def split_by_orientation(
    segments: List[Segment],
    angle_thresh: float = 45.0
) -> Tuple[List[Segment], List[Segment]]:
    """
    Split line segments into horizontals and verticals.

    Args:
        segments: List of (x1, y1, x2, y2) tuples.
        angle_thresh: threshold in degrees;
                      |angle| <= angle_thresh → horizontal; else vertical.

    Returns:
        horizontals, verticals: two lists of segments.
    """
    horizontals = []
    verticals   = []

    for x1, y1, x2, y2 in segments:
        dy = y2 - y1
        dx = x2 - x1
        angle = abs(math.degrees(math.atan2(dy, dx)))
        if angle <= angle_thresh:
            horizontals.append((x1, y1, x2, y2))
        else:
            verticals.append((x1, y1, x2, y2))

    return horizontals, verticals
