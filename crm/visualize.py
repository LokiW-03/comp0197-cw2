# I acknowledge the use of ChatGPT (version GPT-4o, OpenAI, https://chatgpt.com/) for assistance in debugging and
# writing docstrings.

import torchvision.transforms.functional as TF
import torch

from torchvision.utils import make_grid

def visualize_recon_grid(originals: torch.Tensor, reconstructions: torch.Tensor, path: str, nrow=1):
    """
    Visualizes and saves a side-by-side grid of original and reconstructed images.

    Args:
        originals (torch.Tensor): A batch of original images. Shape (N, 3, H, W), normalized using ImageNet stats.
        reconstructions (torch.Tensor): A batch of reconstructed images. Shape (N, 3, H, W), with values in [-1, 1].
        path (str): The file path where the visualized grid image will be saved.
        nrow (int, optional): Number of image pairs (original + reconstruction) per row in the output grid.
                              The default is 1, resulting in a vertical column of image pairs.
    """

    recon = (reconstructions + 1) / 2.0  # [-1,1] → [0,1]
    orig = originals.clone()
    mean = torch.tensor([0.485, 0.456, 0.406], device=orig.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=orig.device).view(1, 3, 1, 1)
    orig = torch.clamp(orig * std + mean, 0, 1)

    pairs = [img for pair in zip(orig, recon) for img in pair]
    interleaved = torch.stack(pairs)  # Shape: (2N, 3, H, W)

    grid = make_grid(interleaved, nrow=nrow*2, padding=4)
    TF.to_pil_image(grid).save(path)
    print(f"Saved recon grid: {path}")
