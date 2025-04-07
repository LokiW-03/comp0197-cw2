import torch
from PIL import Image
import random

torch.manual_seed(42)
random.seed(42)

def overlay_mask_on_image(image, mask):
    """
    image: [3, H, W] float tensor in [0,1] or similar range
    mask:  [H, W]    long tensor of class indices
    Returns a Pillow Image with the overlay applied.
    """
    # Move data to CPU (in case it's on GPU)
    image = image.cpu()
    mask = mask.cpu()
    
    # [3, H, W] -> [H, W, 3]
    image = image.permute(1, 2, 0)
    
    # Scale to [0, 255] and convert to uint8
    image = (image * 255.0).clamp(0, 255).to(torch.uint8)

    # Define class colors ([R, G, B]) for each class
    colors = torch.tensor([
        [255,   0,   0],  # pet: red
        [  0, 255,   0],  # background: green
        [  0,   0, 255],  # border: blue
    ], dtype=torch.uint8)

    # Create color_mask for each pixel using class index in `mask`
    color_mask = colors[mask]  # [H, W, 3]

    # Blend image (50%) and color_mask (50%)
    overlay = (image.float() * 0.5 + color_mask.float() * 0.5).to(torch.uint8)

    # Convert to Pillow
    overlay_pil = Image.fromarray(overlay.numpy())
    return overlay_pil

def tensor_to_pil(image):
    """
    Convert a [3, H, W] float tensor (in [0,1] or similar) to a Pillow Image.
    """
    image = image.cpu()
    # [3, H, W] -> [H, W, 3]
    image = image.permute(1, 2, 0).clamp(0, 1)
    
    # Scale to [0,255], convert to uint8
    image = (image * 255.0).byte()
    return Image.fromarray(image.numpy())

def visualise_fs_segmentation(model, testset, device):
    """
    Runs 8 examples through the model and draws them on a single 4x4 grid:
    - 2 images per example: original (left) + overlay (right)
    - 4 rows of examples (2 examples per row) = 8 examples total = 16 cells
    """
    model.eval()
    
    # Pick 8 random indices from the dataset
    total_samples = len(testset)
    samples = random.sample(range(total_samples), 8)

    # First, load and process all images & overlays so we know the size
    original_list = []
    overlay_list = []
    
    for idx in samples:
        # Load one sample
        img, _ = testset[idx]

        # Convert original image to PIL
        original_pil = tensor_to_pil(img)

        # Forward pass for prediction
        img_batch = img.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(img_batch).argmax(dim=1).squeeze(0)
        
        # Create overlay
        overlay_pil = overlay_mask_on_image(img, pred)

        original_list.append(original_pil)
        overlay_list.append(overlay_pil)
    
    # Assume all images have the same size
    width, height = original_list[0].size
    
    # We have 8 examples => 8 * 2 = 16 total cells in a 4x4 grid
    grid_rows = 4
    grid_cols = 4
    
    # Each cell is (width, height). So total size:
    combined_width = grid_cols * width
    combined_height = grid_rows * height
    
    # Create a big blank canvas
    combined = Image.new("RGB", (combined_width, combined_height))
    
    # Place each example pair (original + overlay) into the grid
    # We'll do 2 examples per row. Each example uses 2 columns:
    # row = i // 2, col_offset = (i % 2) * 2
    for i in range(len(samples)):
        row = i // 2
        col_offset = (i % 2) * 2

        # Original goes in (row, col_offset)
        x_orig = col_offset * width
        y_orig = row * height
        
        # Overlay goes in (row, col_offset + 1)
        x_ovly = (col_offset + 1) * width
        y_ovly = row * height
        
        combined.paste(original_list[i], (x_orig, y_orig))
        combined.paste(overlay_list[i], (x_ovly, y_ovly))

    # Finally, save one image with all 8 pairs in a 4x4 grid
    combined.save("all_examples_grid.png")
    print("Saved grid of 8 examples as 'all_examples_grid.png'")
