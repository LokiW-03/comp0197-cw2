import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch

def visualize_batch(x, y, nrows=4, ncols=4):
    """
    Visualize a batch of images with their corresponding masks overlaid.
    
    Parameters:
    - x: torch.Tensor of shape [B, 3, H, W] representing the images.
    - y: torch.Tensor of shape [B, 1, H, W] representing the segmentation masks.
         The mask values are assumed to be in the set {0, 1, 2}.
    - nrows: Number of rows in the grid (default 4 for 16 images).
    - ncols: Number of columns in the grid (default 4 for 16 images).
    
    The function creates a custom colormap for the mask:
      - Class 0: fully transparent (so the underlying image shows)
      - Class 1: red with 50% opacity
      - Class 2: green with 50% opacity
    """
    # Create a custom colormap for the masks
    mask_cmap = ListedColormap([
        (0, 0, 0, 0),    # Class 0: fully transparent
        (1, 0, 0, 0.5),  # Class 1: red with 50% transparency
        (0, 1, 0, 0.5)   # Class 2: green with 50% transparency
    ])
    
    batch_size = x.shape[0]
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes = axes.flatten()

    for idx in range(batch_size):
        # Convert tensor image to numpy (assuming image is in [0,1] range)
        img = x[idx].permute(1, 2, 0).cpu().numpy()
        # Squeeze the mask channel
        mask = y[idx].squeeze(0).cpu().numpy()
        
        axes[idx].imshow(img)
        # Overlay the mask; interpolation='none' avoids smoothing the discrete classes
        axes[idx].imshow(mask, cmap=mask_cmap, interpolation='none')
        axes[idx].axis('off')

    # Hide any unused subplots if batch size < nrows*ncols
    for idx in range(batch_size, len(axes)):
        axes[idx].axis('off')
        
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from model.data import trainset, testset
    from torch.utils.data import DataLoader
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--pseudo_path', type=str, default='./pseudo_masks.pt', help='Path to pseudo masks')

    trainloader = DataLoader(trainset, batch_size=16, shuffle=True)
    testloader = DataLoader(testset, batch_size=16, shuffle=False)

    arg = parser.parse_args()

    print("trainset sample:")
    x, y = next(iter(trainloader))
    visualize_batch(x, y)
    print("testset sample:")
    x, y = next(iter(testloader))
    print("pseudo mask sample:")
    from cam.load_pseudo import load_pseudo
    pseudo_loader = load_pseudo(arg.pseudo_path, batch_size=16, shuffle=True)
    x, y = next(iter(pseudo_loader))
    visualize_batch(x, y)
