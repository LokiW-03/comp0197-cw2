# I acknowledge the use of ChatGPT (version GPT-4o, OpenAI, https://chatgpt.com/) for assistance in debugging and
# writing docstrings.

import torch
from PIL import Image, ImageDraw
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

def visualise_fs_segmentation(model, testset, device, model_name):
    """
    Runs 8 examples through the model and draws them on a single 8x3 grid:
    - 3 images per example: original (left), ground truth (middle), prediction (right)
    - 8 rows (one per example), 3 columns = 24 cells total
    """
    model.eval()
    
    # Pick 8 random indices from the dataset
    total_samples = len(testset)
    samples = random.sample(range(total_samples), 8)

    # We'll store these images for each sample
    original_list = []
    gt_list = []
    pred_list = []
    
    for idx in samples:
        # Load one sample (assumes testset returns (img, mask))
        img, mask = testset[idx]
        mask = mask.squeeze()

        # Convert original image to PIL
        original_pil = tensor_to_pil(img)

        # Forward pass for prediction
        img_batch = img.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(img_batch).argmax(dim=1).squeeze(0)
        
        # Create ground-truth overlay
        gt_overlay_pil = overlay_mask_on_image(img, mask)
        # Create prediction overlay
        pred_overlay_pil = overlay_mask_on_image(img, pred)

        original_list.append(original_pil)
        gt_list.append(gt_overlay_pil)
        pred_list.append(pred_overlay_pil)
    
    # Assume all images have the same size
    width, height = original_list[0].size
    
    # 8 rows, 3 columns
    grid_rows = 8
    grid_cols = 3
    
    # Each cell is (width, height). So total size:
    combined_width = grid_cols * width
    combined_height = grid_rows * height
    
    # Create a big blank canvas for the montage
    combined = Image.new("RGB", (combined_width, combined_height))
    
    # Place each example (original, ground truth, prediction) in its row
    for i in range(len(samples)):
        row = i  # each example is on its own row
        # Column 0: original
        x_0 = 0 * width
        y_0 = row * height
        
        # Column 1: true target overlay
        x_1 = 1 * width
        y_1 = row * height
        
        # Column 2: prediction overlay
        x_2 = 2 * width
        y_2 = row * height
        
        combined.paste(original_list[i], (x_0, y_0))
        combined.paste(gt_list[i], (x_1, y_1))
        combined.paste(pred_list[i], (x_2, y_2))

    # Now, add a header region above the grid to draw text
    header_height = 60  # space for two lines of text
    montage_with_header = Image.new("RGB", (combined_width, combined_height + header_height), color=(0, 0, 0))

    # Draw text on the new image
    draw = ImageDraw.Draw(montage_with_header)
    draw.text((10, 10), "Fully supervised montage", fill=(255, 255, 255))
    draw.text((10, 30), "(image, mask, prediction)", fill=(255, 255, 255))

    # Paste the montage below the header text
    montage_with_header.paste(combined, (0, header_height))

    # Save the final montage
    montage_with_header.save(f"montage_fully_supervised_{model_name}.png")
    print(f"Saved grid of 8 examples as 'montage_fully_supervised_{model_name}.png'")
