import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

from crm.reconstruct_net import ReconstructNet
from crm.vgg_loss import VGGLoss, MaskedVGGLoss
from cam.efficientnet_scorecam import EfficientNetB4_CAM, ScoreCAM
from cam.resnet_gradcampp import ResNet50_CAM, GradCAMpp

BATCH_SIZE = 32
NUM_CLASSES = 37
NUM_EPOCHS = 10
MODEL_SAVE_PATH = "crm_models/"
LR = 0.1

def train(model_name: str = 'resnet'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create the model save directory if it doesn't exist
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    # Set up transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    trainset = datasets.OxfordIIITPet(
        root='./data', split='trainval', target_types='category',
        download=True, transform=transform
    )
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    if model_name == 'resnet':
        model = ResNet50_CAM(NUM_CLASSES).to(device)
        cam_generator = GradCAMpp(model)
    elif model_name == 'efficientnet':
        model = EfficientNetB4_CAM(NUM_CLASSES).to(device)
        cam_generator = ScoreCAM(model)
    recon_net = ReconstructNet(input_channel=NUM_CLASSES).to(device)

    optimizer = optim.Adam(
        list(model.parameters()) + list(recon_net.parameters()),
        lr=LR
    )

    cls_loss_fn = nn.CrossEntropyLoss()
    rec_loss_fn = VGGLoss(device)
    scaler = GradScaler(enabled=(device_type == 'cuda'))

    for epoch in range(NUM_EPOCHS):
        model.train()
        recon_net.train()
        total_cls_loss, total_rec_loss = 0.0, 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()

            with autocast(device_type=device_type, enabled=(device_type == 'cuda')):
                cams, logits = cam_generator.generate_cam(images, all_classes=True, resize=False)
                cls_loss = cls_loss_fn(logits, labels)
                recon = recon_net(cams)
                recon = F.interpolate(recon, size=images.shape[-2:], mode='bilinear', align_corners=False)
                # Create a binary mask from the CAMs (using the max activation across classes)
                # mask = (cams.max(dim=1, keepdim=True)[0] > 0.5).float()
                rec_loss = rec_loss_fn(recon, images)
                loss = cls_loss + rec_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_cls_loss += cls_loss.item()
            total_rec_loss += rec_loss.item()

        print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] "
              f"CLS Loss: {total_cls_loss:.4f} | REC Loss: {total_rec_loss:.4f}")

    if model_name == 'resnet':
        torch.save(model.state_dict(), f"{MODEL_SAVE_PATH}/resnet_pet_gradcampp_crm.pth")
        torch.save(recon_net.state_dict(), f"{MODEL_SAVE_PATH}/reconstruct_net_resnet.pth")
    elif model_name == 'efficientnet':
        torch.save(model.state_dict(), f"{MODEL_SAVE_PATH}/efficientnet_pet_scorecam_crm.pth")
        torch.save(recon_net.state_dict(), f"{MODEL_SAVE_PATH}/reconstruct_net_eff.pth")
    print("Training complete. Models saved.")

if __name__ == "__main__":
    train()
