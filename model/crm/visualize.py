from PIL import Image
import torch
import torchvision.transforms.functional as TF
import os

def visualize_reconstruction(output_tensor, output_path="reconstructed.png", title="Reconstructed Image"):
    """
    Save a reconstructed image tensor (assumed in [-1, 1] range) as a PNG using PIL.

    Args:
        output_tensor (torch.Tensor): Tensor of shape [B, 3, H, W]
        output_path (str): Path to save the image.
        title (str): Optional title for logging purposes.
    """
    img = output_tensor.detach().cpu().squeeze(0)  # [3, H, W]
    img = (img + 1) / 2  # [0, 1]
    img = (img * 255).clamp(0, 255).byte()  # [0, 255]

    pil_img = TF.to_pil_image(img)
    pil_img.save(output_path)
    print(f"{title} saved to {output_path}")
