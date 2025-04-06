from cam.resnet_gradcampp import ResNet50_CAM
from torchcam.methods import GradCAMpp, CAM, GradCAM, ScoreCAM, SmoothGradCAMpp

import torch
from torchvision.transforms import Resize, InterpolationMode
from cam.dataset.oxfordpet import download_pet_dataset
from cam.preprocessing import unnormalize


# Use opensource libraries to cross-validate
def postprocessing(cam, threshold_low, threshold_high):
    pseudo_mask = torch.zeros_like(cam)
    pseudo_mask[cam >= threshold_high] = 0 # Foreground
    pseudo_mask[cam < threshold_low] = 1 # Background
    pseudo_mask[(cam >= threshold_low) & (cam < threshold_high)] = 2 # Contour
    
    return pseudo_mask

if __name__ == "__main__":
    IMAGE_SIZE = 224
    MODEL_PATH = "cam/saved_models/resnet50_pet_cam.pth"
    CAM_TYPE = "gradcampp"
    SAVE_PATH = f"cam/saved_models/resnet50_{CAM_TYPE}_pseudo.pt"
    device = torch.device("cuda" if torch.cuda.is_available() 
                        else "mps" if torch.backends.mps.is_available()
                        else "cpu")
    print(device)

    cam_func = CAM
    if CAM_TYPE == "gradcampp":
        cam_func = GradCAMpp 
    elif CAM_TYPE == "scorecam":
        cam_func = ScoreCAM
    elif CAM_TYPE == "smoothgradcampp":
        cam_func = SmoothGradCAMpp
    elif CAM_TYPE == "gradcam":
        cam_func = GradCAM

    resnet50 = ResNet50_CAM(num_classes=37)
    resnet50.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    resnet50.eval()
    resnet50.to(device)
    cam_extractor = cam_func(resnet50, target_layer=resnet50.target_layer)
    train_loader, test_loader = download_pet_dataset(with_paths=False)

    resize = Resize((IMAGE_SIZE, IMAGE_SIZE), 
                            interpolation=InterpolationMode.BILINEAR)

    all_pseudo_masks = []
    all_images = []
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        out = resnet50(x)
        # print(x.shape, out.shape) # torch.Size([64, 3, 224, 224]) torch.Size([64, 37])
        class_idxs = out.argmax(dim=1)
        # print(class_idxs.shape) # torch.Size([64])
        cam = cam_extractor(class_idxs.tolist(), out, normalized=True)
        cam = cam[0]
        cam = resize(cam.cpu()).to(device)
        pseudo_mask = postprocessing(cam, 0.2, 0.275)
        processed_mask = torch.nn.functional.interpolate(
                pseudo_mask.unsqueeze(1).float(),
                size=(224, 224),
                mode='nearest'
            ).squeeze().to(torch.long)
        all_pseudo_masks.append(processed_mask)
        all_images.append(x)
    
    all_pseudo_masks = torch.cat(all_pseudo_masks, dim=0)
    all_images = torch.cat(all_images, dim=0)
    # save as dataset
    torch.save({
        'pairs': [
            {
                'image': unnormalize(img.cpu()),
                'mask': processed_mask,
            }
            for img, processed_mask in zip(all_images, all_pseudo_masks)
        ]
    }, SAVE_PATH)