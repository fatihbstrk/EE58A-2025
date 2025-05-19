# node_clustering.py

import numpy as np
from typing import List, Tuple
from scipy.cluster.hierarchy import fclusterdata

Point = Tuple[int, int]

def cluster_nodes(
    points: List[Point],
    threshold: float = 5.0
) -> List[Point]:
    """
    Cluster raw intersection points into consolidated node positions.

    Args:
        points: list of (x,y) intersections.
        threshold: maximum distance (in pixels) for points to be in same cluster.

    Returns:
        List of cluster centroids as (x,y) ints.
    """
    # Handle trivial cases
    if len(points) == 0:
        return []
    if len(points) == 1:
        return points.copy()

    arr = np.array(points)
    # Use flat cluster: any two within threshold become same cluster
    labels = fclusterdata(arr, t=threshold, criterion='distance', metric='euclidean')
    clusters = {}
    for (x, y), lbl in zip(arr, labels):
        clusters.setdefault(lbl, []).append((x, y))

    # Compute centroids
    centroids = []
    for pts in clusters.values():
        xs, ys = zip(*pts)
        cx = int(round(np.mean(xs)))
        cy = int(round(np.mean(ys)))
        centroids.append((cx, cy))

    return centroids
