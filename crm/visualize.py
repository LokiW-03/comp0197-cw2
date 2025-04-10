from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
import torch

def visualize_recon_grid(originals: torch.Tensor, reconstructions: torch.Tensor, path: str, nrow=1):

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
