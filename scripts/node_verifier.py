import numpy as np

def count_lines_touching_dot(dot, lines, radius=10):
    """Count how many lines touch a node dot within the specified radius."""
    cx, cy = dot
    count = 0

    for entry in lines:  # Iterate over each line's dictionary entry
        x1, y1, x2, y2 = entry["line"]  # Access the line coordinates inside the dictionary

        # Check both endpoints for proximity to the dot
        dist1 = np.linalg.norm(np.array([cx, cy]) - np.array([x1, y1]))
        dist2 = np.linalg.norm(np.array([cx, cy]) - np.array([x2, y2]))

        if dist1 <= radius or dist2 <= radius:
            count += 1

    return count


def verify_dots_with_lines(dots, lines, min_connections=3, radius=10):
    """Verify dots by checking if they are connected to enough lines.
    
    Args:
        dots: List of (x, y) dot coordinates.
        lines: List of lines [(x1, y1, x2, y2)].
        min_connections: Minimum number of lines to accept a dot.
        radius: Max endpoint distance to consider a line touching a dot.
    
    Returns:
        verified_dots: List of accepted (x, y) dots.
    """
    verified = []

    for dot in dots:
        touching_lines = count_lines_touching_dot(dot, lines, radius)
        if touching_lines >= min_connections:
            verified.append(dot)

    return verified
