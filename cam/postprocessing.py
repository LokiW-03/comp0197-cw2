#postprocessing.py
import torch
import argparse
from typing import Callable
from torchvision import transforms
from torch.utils.data import DataLoader
from common import *
from preprocessing import unnormalize
from data_utils.data import download_pet_dataset
from visualize import visualize_cam
from model.resnet_drs import ResNet50_CAM_DRS
from model.resnet_gradcampp import ResNet50_CAM, GradCAMpp
from crm import CRM_MODEL_SAVE_PATH


def generate_pseudo_masks(
    dataloader: DataLoader,
    model: torch.nn.Module,
    cam_generator: Callable,
    save_path: str = './pseudo_masks.pt',
    threshold_low: float = 0.15,
    threshold_high: float = 0.45,
    device: torch.device = torch.device('cpu'),
    side_effect: bool = True
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

    all_pseudo_masks = []
    all_images = []
    image_paths = []
    sample_mask_images = []
    
    for batch in dataloader:
        # Parse batch data (assumes dataloader returns (img, _, paths))
        inputs = batch[0].to(device)
        all_images.append(inputs)
        paths = batch[2]  # Third element is image path
        
        # Generate CAM heatmap
        cams, _ = gradcam.generate_cam(inputs, all_classes=False, resize=True)  # (B, H, W)
        cams = (cams - cams.min()) / (cams.max() - cams.min() + 1e-8)
        # Generate pseudo masks
        for cam, img_path in zip(cams, paths):
            # Threshold processing to generate three-class mask
            pseudo_mask = torch.zeros_like(cam)

            # Aligned with the testsets
            pseudo_mask[cam >= threshold_high] = 0 # Foreground
            pseudo_mask[(cam >= threshold_low) & (cam < threshold_high)] = 2 # Contour
            pseudo_mask[cam < threshold_low] = 1 # Background
            
            # Convert to PIL Image and apply standard transforms
            processed_mask = torch.nn.functional.interpolate(
                pseudo_mask.unsqueeze(0).unsqueeze(0).float(),
                size=(224, 224),
                mode='nearest'
            ).squeeze().to(torch.long)
            
            if side_effect and len(sample_mask_images) < NUM_SAMPLES:
                vis_mask = torch.zeros_like(pseudo_mask, dtype=torch.uint8)
                vis_mask[pseudo_mask == 0] = 255 # Foreground = White
                vis_mask[pseudo_mask == 1] = 0   # Background = Black
                vis_mask[pseudo_mask == 2] = 128 # Contour = Gray
                
                to_pil_transform = transforms.ToPILImage(mode='L')
                pil_mask_for_saving = to_pil_transform(vis_mask)

                sample_mask_images.append(pil_mask_for_saving)

            all_pseudo_masks.append(processed_mask)
            image_paths.append(img_path)
    
    print(f"Generated {len(all_pseudo_masks)} pseudo masks")
    all_images = torch.cat(all_images, dim=0)
    print(f"Input images: {len(all_images)}")
    # Save in compressed format
    torch.save({
        'pairs': [
            {
                'image': unnormalize(img.cpu()),
                'mask': processed_mask,
                'path': img_path
            }
            for img, processed_mask, img_path in zip(all_images, all_pseudo_masks, image_paths)
        ]
    }, save_path)

    print(max(all_pseudo_masks[0].flatten()), min(all_pseudo_masks[0].flatten()))
    
    print(f"Generated {len(all_pseudo_masks)} pseudo masks saved at {save_path}")

    if side_effect:
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


    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet',
                        choices=['resnet', 'resnet_crm', 'resnet_drs'],)
    args = parser.parse_args()

    num_classes=37
    cam_threshold = CAM_THRESHOLD.get(args.model, [0.15, 0.45])

    if args.model == 'resnet':
        model = ResNet50_CAM(num_classes)
        model_save_path = f"{MODEL_SAVE_PATH}/resnet50_pet_cam.pth"
        cam_generator = lambda model: GradCAMpp(model)
        pseudo_save_path = f"{MODEL_SAVE_PATH}/resnet50_pet_cam_pseudo.pt"

    elif args.model == 'resnet_crm':
        model = ResNet50_CAM(num_classes)
        model_save_path = f"{CRM_MODEL_SAVE_PATH}/resnet_pet_gradcampp_crm.pth"
        cam_generator = lambda model: GradCAMpp(model)
        pseudo_save_path = f"{CRM_MODEL_SAVE_PATH}/resnet_pet_gradcampp_crm_pseudo.pt"

    elif args.model == 'resnet_drs':
        model = ResNet50_CAM_DRS(num_classes)
        model_save_path = f"{CRM_MODEL_SAVE_PATH}/resnet_drs_pet_gradcampp_crm.pth"
        cam_generator = lambda model: GradCAMpp(model)
        pseudo_save_path = f"{CRM_MODEL_SAVE_PATH}/resnet_drs_pet_gradcampp_crm_pseudo.pt" 

    else:
        raise ValueError(f"Invalid model: {args.model}")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    
    model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
    
    # 2. Data initialization
    train_loader, test_loader = download_pet_dataset(with_paths=True)
    
    generate_pseudo_masks(train_loader, model, cam_generator, pseudo_save_path, device=device,
                          threshold_low=cam_threshold[0], threshold_high=cam_threshold[1])