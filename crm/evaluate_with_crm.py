import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from cam.resnet_gradcampp import ResNet50_CAM, GradCAMpp
from cam.efficientnet_scorecam import EfficientNetB4_CAM, ScoreCAM
from crm.reconstruct_net import ReconstructNet
from crm.vgg_loss import VGGLoss
from crm.visualize import visualize_recon_grid
from crm import IMG_SIZE, CRM_MODEL_SAVE_PATH, NUM_CLASSES

def evaluate_crm(model_name='resnet', save_dir='crm_eval_outputs'):
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    testset = datasets.OxfordIIITPet(
        root='./data', split='test', target_types='category',
        download=True, transform=transform
    )
    test_loader = DataLoader(testset, batch_size=8, shuffle=False)

    # Load models
    if model_name == 'resnet':
        cam_model = ResNet50_CAM(NUM_CLASSES).to(device)
        cam_model.load_state_dict(torch.load(f"{CRM_MODEL_SAVE_PATH}/resnet_pet_gradcampp_crm.pth", map_location=device, weights_only=True))
        cam_generator = GradCAMpp(cam_model)
        recon_model = ReconstructNet(NUM_CLASSES).to(device)
        recon_model.load_state_dict(torch.load(f"{CRM_MODEL_SAVE_PATH}/reconstruct_net_resnet.pth", map_location=device, weights_only=True))
    else:
        cam_model = EfficientNetB4_CAM(NUM_CLASSES).to(device)
        cam_model.load_state_dict(torch.load(f"{CRM_MODEL_SAVE_PATH}/efficientnet_pet_scorecam_crm.pth", map_location=device, weights_only=True))
        cam_generator = ScoreCAM(cam_model)
        recon_model = ReconstructNet(NUM_CLASSES).to(device)
        recon_model.load_state_dict(torch.load(f"{CRM_MODEL_SAVE_PATH}/reconstruct_net_eff.pth", map_location=device, weights_only=True))

    cam_model.eval()
    recon_model.eval()

    vgg_loss_fn = VGGLoss(device)

    total_vgg = 0.0
    correct = 0
    total = 0

    sample_images = []
    sample_recons = []

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            logits = cam_model(images)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            cams, _ = cam_generator.generate_cam(images, all_classes=True, resize=False)
            recon = recon_model(cams)
            recon = F.interpolate(recon, size=images.shape[-2:], mode='bilinear', align_corners=False)

            # VGG loss over full batch
            vgg_loss = vgg_loss_fn(images, recon).item()
            total_vgg += vgg_loss * images.size(0)

            # Save 5 samples for visualization
            if len(sample_images) < 5:
                needed = 5 - len(sample_images)
                sample_images.extend(images[:needed])
                sample_recons.extend(recon[:needed])

    # Final metrics
    avg_vgg_loss = total_vgg / total
    acc = correct / total

    print("=== Evaluation Results ===")
    print(f"Classification Accuracy (full test set): {acc:.4f}")
    print(f"Average VGG Loss (full test set): {avg_vgg_loss:.4f}")

    # Visualization
    if sample_images:
        stacked_input = torch.stack(sample_images)
        stacked_recon = torch.stack(sample_recons)
        save_path = f"{save_dir}/recon_comparison.jpg"
        visualize_recon_grid(stacked_input, stacked_recon, save_path, nrow=1, ncol=5)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'efficientnet'])
    args = parser.parse_args()

    evaluate_crm(args.model)
