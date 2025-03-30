import os
import torch
from skimage.segmentation import slic
from skimage.color import rgb2lab
from glob import glob
import numpy as np
from PIL import Image

def generate_superpixels(image_dir, save_dir, n_segments=100, compactness=10):
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

