import math
import json

# -------------------------
# Helper Functions
# -------------------------

def point_to_line_distance(px, py, line):
    """Compute shortest distance from a point to a line segment."""
    x1, y1, x2, y2 = line
    num = abs((y2 - y1)*px - (x2 - x1)*py + x2*y1 - y2*x1)
    den = math.hypot(x2 - x1, y2 - y1)
    return num / den if den != 0 else float('inf')

def estimate_terminals(label, bbox):
    """Estimate component terminal positions based on bounding box and type."""
    x1, y1, x2, y2 = bbox
    mid_x = (x1 + x2) // 2
    mid_y = (y1 + y2) // 2

    if label == "MOSFET":
        return {
            "gate": (x1, mid_y),
            "drain": (mid_x, y1),
            "source": (mid_x, y2)
        }
    elif label in ["Resistor", "Voltage_Source", "Current_Source", "Inductor", "Diode"]:
        return {
            "T1": (x1, mid_y),
            "T2": (x2, mid_y)
        }
    elif label == "Capacitor":
        return {
            "T1": (mid_x, y1),
            "T2": (mid_x, y2)
        }
    elif label == "Ground":
        return {
            "T1": (mid_x, y2)
        }
    else:
        # Generic fallback
        return {
            "T1": (x1, y1),
            "T2": (x2, y2)
        }

def save_terminal_mapping(mapping, out_path):
    """Save terminal-to-net mapping as a JSON file."""
    with open(out_path, 'w') as f:
        json.dump(mapping, f, indent=2)

import cv2
import numpy as np

import cv2
import numpy as np

# Function to visualize components with net labeling
def visualize_components_and_nets(image_path, components, terminal_mapping, save_path):
    """Visualize components with bounding boxes and net labels."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Draw bounding boxes and labels
    for comp_id, comp in enumerate(components):  # Iterate through components
        label = comp["label"]
        bbox = comp["bbox"]
        net_mapping = terminal_mapping.get(comp_id, {})

        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box

        # Label component with its label
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw terminal and net labels
        for terminal, net_id in net_mapping.items():
            # Display terminal label
            cv2.putText(img, f"{terminal}", (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            # Display net label
            cv2.putText(img, f"{net_id}", (x1, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # Save the output image
    cv2.imwrite(str(save_path), img)

# Function to match terminals to nets (already present in your project)
def match_terminals_to_nets(components, labeled_lines):
    """
    Match terminals to nets using line proximity.
    Args:
        components (list): A list of component details including bbox and label.
        labeled_lines (list): List of detected lines with associated net IDs.
    Returns:
        mapping (dict): A dictionary with component IDs and their terminals' net IDs.
    """
    matches = {}
    for i, comp in enumerate(components):
        label = comp.get("label", "C")
        comp_id = f"{label[0]}{i+1}"
        bbox = comp["bbox"]

        terminals = estimate_terminals(label, bbox)  # You would have this function defined elsewhere

        term_net_map = {}
        for pin, pt in terminals.items():
            min_dist = float('inf')
            closest_net = "unconnected"
            for entry in labeled_lines:
                line = entry["line"]
                net_id = entry["net_id"]
                dist = point_to_line_distance(*pt, line)  # This should be defined elsewhere
                if dist < min_dist:
                    min_dist = dist
                    closest_net = net_id
            term_net_map[pin] = closest_net

        matches[comp_id] = term_net_map

    return matches
import cv2
import json

import cv2
import json

def visualize_hough_lines_and_nets(image_path, lines_data, save_path):
    """Visualize the Hough lines and label them with net IDs."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Draw lines from Hough transform data and label with net IDs
    for line_data in lines_data:
        net_id = line_data["net_id"]
        line = line_data["line"]
        x1, y1, x2, y2 = line

        # Draw the line on the image (color it in blue)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Label the line with its net ID
        label_position = (x1, y1 - 10)  # Label above the starting point of the line
        cv2.putText(img, net_id, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Save the output image
    cv2.imwrite(str(save_path), img)

