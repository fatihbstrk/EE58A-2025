import os
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import random
import numpy as np
from pathlib import Path

def list_images(folder_path, exts=[".png", ".jpg", ".jpeg"]):
    """
    Lists image files in a given folder, filtered by extension.

    Args:
        folder_path (str or Path): Path to the folder.
        exts (list): List of extensions to include.

    Returns:
        list: Sorted list of image file paths as Path objects.
    """
    folder = Path(folder_path)
    image_files = [p for p in folder.glob("*") if p.suffix.lower() in exts]
    print(f"Debug: Found {len(image_files)} images in {folder_path}")
    return sorted(image_files)

# Test the list_images function
PROC_DIR = Path("data/processed")
image_files = list_images(PROC_DIR)
print(f"Found {len(image_files)} images.")


def create_output_folders(image_paths, output_root):
    """
    Create output folders for each image based on its imgid (filename stem with '_preprocessed' stripped).
    """
    from pathlib import Path

    for path in image_paths:
        img_id = Path(path).stem.replace("_preprocessed", "")  # 🧼 strip "_preprocessed"
        folder_path = output_root / img_id
        folder_path.mkdir(parents=True, exist_ok=True)




def load_image(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    return img

def save_image(image, output_path):
    cv2.imwrite(str(output_path), image)

def preview_image(img, title="Image"):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.title(title)
    plt.show()


def preview_random_images(image_files, num_images=10):
    sample_files = random.sample(image_files, min(num_images, len(image_files)))
    for img_path in sample_files:
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title(img_path.name)
        plt.axis("off")
        plt.show()


def save_image(image, save_path):
    """
    Save an image to a specified path.
    """
    cv2.imwrite(save_path, image)


import cv2
import numpy as np

def preprocess_image(image):
    """
    Preprocess the image by converting it to grayscale and applying Gaussian blur.
    Args:
        image (numpy.ndarray): The input image.
    
    Returns:
        numpy.ndarray: The preprocessed image (grayscale and blurred).
    """
    # Convert to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur to remove noise
    #blurred_img = cv2.GaussianBlur(gray_img, (3, 3), 0.5)
    return gray_img
