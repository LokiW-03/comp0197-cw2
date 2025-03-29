import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image
from cam.resnet_gradcampp import ResNet50_CAM, GradCAMpp
from cam.efficientnet_scorecam import EfficientNetB4_CAM, ScoreCAM
from crm.reconstruct_net import ReconstructNet
from crm.vgg_loss import VGGLoss
from cam import IMAGE_SIZE

MODEL_SAVE_PATH = "crm_models"

def evaluate_crm(model_name='resnet', save_dir='crm_eval_outputs'):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/reconstructed", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
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
        cam_model = ResNet50_CAM(37).to(device)
        cam_model.load_state_dict(torch.load(f"{MODEL_SAVE_PATH}/resnet_pet_gradcampp_crm.pth", map_location=device, weights_only=True))
        cam_generator = GradCAMpp(cam_model)
        recon_model = ReconstructNet(37).to(device)
        recon_model.load_state_dict(torch.load(f"{MODEL_SAVE_PATH}/reconstruct_net_resnet.pth", map_location=device, weights_only=True))
    else:
        cam_model = EfficientNetB4_CAM(37).to(device)
        cam_model.load_state_dict(torch.load(f"{MODEL_SAVE_PATH}/efficientnet_pet_scorecam_crm.pth", map_location=device, weights_only=True))
        cam_generator = ScoreCAM(cam_model)
        recon_model = ReconstructNet(37).to(device)
        recon_model.load_state_dict(torch.load(f"{MODEL_SAVE_PATH}/reconstruct_net_eff.pth", map_location=device, weights_only=True))

    cam_model.eval()
    recon_model.eval()

    vgg_loss_fn = VGGLoss(device)
    total_vgg = 0.0
    correct = 0
    total = 0
    count = 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = cam_model(images)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        cams, _ = cam_generator.generate_cam(images, all_classes=True, resize=False)

        with torch.no_grad():
            recon = recon_model(cams)
            recon = F.interpolate(recon, size=images.shape[-2:], mode='bilinear', align_corners=False)

        for i in range(images.size(0)):
            vgg = vgg_loss_fn(images[i:i+1], recon[i:i+1]).item()
            total_vgg += vgg
            count += 1

            save_image(recon[i], f"{save_dir}/reconstructed/recon_{count}.png")

        if count >= 32:  # limit visualizations for speed
            break

    print(f"Evaluation Results on {count} samples:")
    print(f"Avg VGG Loss: {total_vgg / count:.4f}")
    print(f"Classification Accuracy: {correct / total:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'efficientnet'])
    args = parser.parse_args()

    evaluate_crm(args.model)