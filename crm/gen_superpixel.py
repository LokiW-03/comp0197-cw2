import os
import torch
from skimage.segmentation import slic
from skimage.color import rgb2lab
from glob import glob
import numpy as np
from PIL import Image

def generate_superpixels(image_dir, save_dir, n_segments=100, compactness=10):
    os.makedirs(save_dir, exist_ok=True)
    image_paths = glob(os.path.join(image_dir, "*.jpg"))

    for path in image_paths:
        print(f"Processing: {os.path.basename(path)}")
        image = Image.open(path).convert("RGB")
        image = np.array(image)
        lab_img = rgb2lab(image)  # improves segmentation quality
        segments = slic(lab_img, n_segments=n_segments, compactness=compactness, start_label=0)
        filename = os.path.basename(path).replace(".jpg", ".pt")
        torch.save(torch.tensor(segments, dtype=torch.int64), os.path.join(save_dir, filename))

if __name__ == "__main__":
    generate_superpixels(
        image_dir="./data/oxford-iiit-pet/images",
        save_dir="./superpixels",
        n_segments=100,
        compactness=10
    )
