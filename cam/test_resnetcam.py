import torch
from PIL import Image
import numpy as np
from resnetcam import ResNet50_CAM, GradCAMpp
from __init__ import NUM_CLASSES, TMP_OUTPUT_PATH
from preprocessing import unnormalize
from torch.utils.data import DataLoader

def test_gradcampp(
    dataloader: DataLoader, 
    model_path: str, 
    save_path: str, 
    num_samples: int = 16,
    nrow: int = 4
):
    """
    Batch test and save result grid using DataLoader
    
    Args:
        dataloader (DataLoader): Test data loader (returns tensor or tuple)
        model_path (str): Model weights path
        save_path (str): Grid save path
        num_samples (int): Maximum number of samples to visualize (default 16)
        nrow (int): Number of rows in grid (default 4)
    """
    # Load model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = ResNet50_CAM(NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model = model.to(device).eval()
    
    # Store results
    all_combined = []
    processed = 0
    
    for batch in dataloader:
        # Parse batch data (supports tuple or pure tensor)
        if isinstance(batch, (list, tuple)):
            inputs = batch[0].to(device)  # Assume input is first element of tuple
        else:
            inputs = batch.to(device)
        
        # Iterate through each sample in batch
        for i in range(inputs.size(0)):
            if processed >= num_samples:
                break
            
            # Get single sample
            input_tensor = inputs[i].unsqueeze(0)  # (1, C, H, W)
            
            # Generate CAM
            gradcam = GradCAMpp(model)
            cam = gradcam.generate_cam(input_tensor)
            
            # Denormalize and convert to uint8
            img_np = unnormalize(input_tensor.cpu())  # (H, W, 3) [0,1]
            img_uint8 = (img_np * 255).astype(np.uint8)
            
            # Generate heatmap
            heatmap = jet_colormap(cam)  # (H, W, 3) uint8
            
            # Generate superimposed image
            superimposed = (heatmap * 0.4 + img_uint8 * 0.6).clip(0, 255).astype(np.uint8)
            
            # Horizontal concatenation
            combined = np.concatenate([img_uint8, heatmap, superimposed], axis=1)  # (H, 3W, 3)
            
            # Convert to tensor and store
            combined_tensor = torch.from_numpy(combined).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, 3W)
            all_combined.append(combined_tensor)
            
            processed += 1
        
        if processed >= num_samples:
            break
    
    # Generate final grid
    if len(all_combined) > 0:
        batch = torch.cat(all_combined, dim=0)  # (N, 3, H, 3W)
        ncol = int(np.ceil(len(all_combined) / nrow))
        visualize_grid(batch, save_path, nrow=nrow, ncol=ncol)
        print(f"Saved visualization grid with {len(all_combined)} samples")
    else:
        print("No samples processed")

def visualize_grid(images: torch.Tensor, path: str, nrow: int=4, ncol: int=4):
    """
    Save image grid to file (compatible with single or multiple images)
    
    Args:
        images: torch.Tensor, image tensor (N, C, H, W), value range [0, 255], uint8 type
        path: str, save path
        nrow: int, number of rows in grid
        ncol: int, number of columns in grid
    """
    N, C, H, W = images.shape
    assert nrow * ncol == N, f"nrow({nrow})*ncol({ncol}) must equal N({N})"
    
    # Build grid
    grid = images.view(nrow, ncol, C, H, W)
    grid = grid.permute(2, 0, 3, 1, 4)  # (C, nrow, H, ncol, W)
    grid = grid.reshape(C, nrow * H, ncol * W)
    grid = grid.permute(1, 2, 0).cpu().numpy()  # (H_total, W_total, C)
    
    # Save image
    Image.fromarray(grid).save(path)
    print(f"Grid image saved at {path}")


# Avoid using plt.cm.jet
def jet_colormap(cam: np.ndarray) -> np.ndarray:
    """
    Manual implementation of JET colormap (equivalent to plt.cm.jet)
    
    Args:
        cam: np.ndarray, input heatmap (H, W), value range [0,1]
        
    Returns:
        heatmap: np.ndarray, RGB image (H, W, 3), value range [0,255], uint8
    """
    # Define JET color lookup table (from matplotlib's JET LUT)
    jet_lut = np.array([
        [0.0, 0.0, 0.5],     # Dark blue (0.0)
        [0.0, 0.0, 1.0],     # Blue
        [0.0, 0.5, 1.0],     # Sky blue
        [0.0, 1.0, 1.0],     # Cyan
        [0.5, 1.0, 0.5],     # Yellow-green
        [1.0, 1.0, 0.0],     # Yellow
        [1.0, 0.5, 0.0],     # Orange
        [1.0, 0.0, 0.0],     # Red
        [0.5, 0.0, 0.0]      # Dark red (1.0)
    ], dtype=np.float32)
    
    # Map cam values to LUT indices (0~1 → 0~7)
    indices = np.clip(cam, 0.0, 1.0) * (len(jet_lut) - 1)
    lower = np.floor(indices).astype(int)
    upper = np.ceil(indices).astype(int)
    ratio = indices - lower
    
    # Interpolate RGB values
    lower_colors = jet_lut[lower]
    upper_colors = jet_lut[upper.clip(max=len(jet_lut)-1)]
    interpolated = (1 - ratio[..., None]) * lower_colors + ratio[..., None] * upper_colors
    
    # Convert to 0-255 range and adjust channel order to RGB
    heatmap = (interpolated * 255).astype(np.uint8)
    return heatmap[..., ::-1]

if __name__ == "__main__":
    from dataset.oxfordpet import download_pet_dataset
    train_loader, test_loader = download_pet_dataset()
    test_gradcampp(
        dataloader=test_loader,
        model_path="resnet50_pet_cam.pth",
        save_path=f"{TMP_OUTPUT_PATH}/cam_grid.jpg",
        num_samples=16,
        nrow=4
    )
