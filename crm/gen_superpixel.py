# I acknowledge the use of ChatGPT (version GPT-4o, OpenAI, https://chatgpt.com/) for assistance in debugging and
# writing docstrings.

import os
import torch
from skimage.segmentation import slic
from skimage.color import rgb2lab
from glob import glob
import numpy as np
from PIL import Image


def generate_superpixels(image_dir, save_dir, n_segments=100, compactness=10):
    """
    Generates and saves superpixel segmentation maps for a directory of images.

    Args:
        image_dir (str): Path to the directory containing input `.jpg` images.
        save_dir (str): Directory to save the resulting superpixel `.pt` files.
        n_segments (int, optional): Number of superpixels to generate per image. Default is 100.
        compactness (float, optional): Balances color proximity and space proximity.
        Higher values make superpixels more spatially compact. Default is 10.
    """

    if not os.path.exists(image_dir):
        raise RuntimeError(f"Image directory '{image_dir}' not found.")

    os.makedirs(save_dir, exist_ok=True)
    image_paths = glob(os.path.join(image_dir, "*.jpg"))

    for path in image_paths:
        filename = os.path.basename(path).replace(".jpg", ".pt")
        save_path = os.path.join(save_dir, filename)

        if os.path.exists(save_path):
            continue  # Skip if already exists

        image = Image.open(path).convert("RGB")
        image = np.array(image)
        lab_img = rgb2lab(image)
        segments = slic(lab_img, n_segments=n_segments, compactness=compactness, start_label=0)
        torch.save(torch.tensor(segments, dtype=torch.int64), save_path)

    print("Superpixel generation complete.")

