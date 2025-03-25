import torch
from PIL import Image
import os
from typing import Callable
from torchvision import transforms
from torch.utils.data import DataLoader
from __init__ import *
from preprocessing import unnormalize
from dataset.oxfordpet import download_pet_dataset
from visualize import visualize_cam


def generate_pseudo_masks(
    dataloader: DataLoader,
    model: torch.nn.Module,
    cam_generator: Callable,
    save_path: str = './pseudo_masks.pt',
    threshold_low: float = 0.3,
    threshold_high: float = 0.7,
    device: str = 'auto'
):
    """
    Enhanced pseudo mask generation function
    
    Args:
        dataloader: DataLoader that returns (images, _, image_paths)
        model: Pretrained model instance (weights must be loaded)
        cam_generator: Factory function that returns CAM instance, signature: (model) -> CAM_instance
        save_path: Path to save the generated masks
        threshold_low: Lower threshold for contour class
        threshold_high: Higher threshold for foreground class
        device: Auto-detect/manually specify device ('cuda', 'mps', 'cpu')
    """
    
    # Move model to device and set to eval mode
    model = model.to(device).eval()
    
    # Initialize CAM generator
    gradcam = cam_generator(model)  # Create instance using factory pattern
    
    # Verify CAM class interface
    if not hasattr(gradcam, 'generate_cam'):
        raise ValueError("CAM class must implement generate_cam() method")
    
    # TODO: Should be aligned with the original dataset
    mask_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).long() - 1)
    ])
    
    all_pseudo_masks = []
    image_paths = []
    sample_mask_images = []
    
    for batch in dataloader:
        # Parse batch data (assumes dataloader returns (img, _, paths))
        inputs = batch[0].to(device)
        paths = batch[2]  # Third element is image path
        
        # Generate CAM heatmap
        cams = gradcam.generate_cam(inputs)  # (B, H, W)
        
        # Generate pseudo masks
        for cam, img_path in zip(cams, paths):
            # Threshold processing to generate three-class mask
            pseudo_mask = torch.zeros_like(cam)
            pseudo_mask[cam >= threshold_high] = 255   # Foreground
            pseudo_mask[(cam >= threshold_low) & (cam < threshold_high)] = 128  # Contour
            
            # Convert to PIL Image and apply standard transforms
            pil_mask = Image.fromarray(pseudo_mask.cpu().numpy(), mode='L')
            processed_mask = mask_transform(pil_mask)
            if len(sample_mask_images) < NUM_SAMPLES:
                sample_mask_images.append(pil_mask)
            all_pseudo_masks.append(processed_mask)
            image_paths.append(img_path)
    
    # Save in compressed format
    torch.save({
        'pairs': [
            {
                'image': unnormalize(img.cpu()),
                'mask': processed_mask,
                'path': img_path
            }
            for img, processed_mask, img_path in zip(inputs, all_pseudo_masks, image_paths)
        ]
    }, save_path)
    
    print(f"Generated {len(all_pseudo_masks)} pseudo masks saved at {save_path}")

    # save sample mask images
    for i, mask in enumerate(sample_mask_images):
        os.makedirs(f'{TMP_OUTPUT_PATH}/masks', exist_ok=True)
        mask.save(f'{TMP_OUTPUT_PATH}/masks/{os.path.basename(image_paths[i]).split(".")[0]}.png')
    
    # save sample mask images
    visualize_cam(inputs[:NUM_SAMPLES], cams[:NUM_SAMPLES], f'{TMP_OUTPUT_PATH}/cam_grid.jpg')
    
    return save_path


# Usage example
if __name__ == "__main__":
    # ---------- User-defined section ----------
    from resnetcam import ResNet50_CAM, GradCAMpp
    
    # 1. Model initialization
    model = ResNet50_CAM(num_classes=37)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    model.load_state_dict(torch.load(f"{MODEL_SAVE_PATH}/resnet50_pet_cam.pth", map_location=device, weights_only=True))
    
    # 2. Data initialization
    _, test_loader = download_pet_dataset(with_paths=True)
    
    # 3. Call function
    generate_pseudo_masks(
        dataloader=test_loader,
        model=model,
        cam_generator=lambda model: GradCAMpp(model),  # Instance factory
        save_path=f'{MODEL_SAVE_PATH}/resnet50_pet_cam_pseudo.pt',
        device=device
    )