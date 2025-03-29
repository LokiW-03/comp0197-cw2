import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

from cam.efficientnet_scorecam import EfficientNetB4_CAM, ScoreCAM
from cam.resnet_gradcampp import ResNet50_CAM, GradCAMpp
from crm.reconstruct_net import ReconstructNet
from crm import CRM_MODEL_SAVE_PATH, BATCH_SIZE, NUM_CLASSES, NUM_EPOCHS, LR, RESNET_PATH, EFFNET_PATH, IMG_SIZE


def train(model_name: str = 'resnet'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(CRM_MODEL_SAVE_PATH, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
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
        state_dict = torch.load(RESNET_PATH, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        cam_generator = GradCAMpp(model)
    else:
        model = EfficientNetB4_CAM(NUM_CLASSES).to(device)
        state_dict = torch.load(EFFNET_PATH, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        cam_generator = ScoreCAM(model)

    recon_net = ReconstructNet(input_channel=NUM_CLASSES).to(device)

    # Combined optimizer for both CAM model + ReconstructNet initially
    optimizer = optim.Adam(
        list(model.parameters()) + list(recon_net.parameters()),
        lr=LR
    )

    cls_loss_fn = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=(device_type == 'cuda'))
    model.train()
    recon_net.train()
    
    for epoch in range(NUM_EPOCHS):
        # Freeze classifier after epoch 10
        if epoch == 10:
            print("Freezing classifier parameters from epoch 10 onward...")
            for param in model.parameters():
                param.requires_grad = True  # Leave this ON for CAM/GradCAM to work
            model.eval()  # Stops dropout/batchnorm updates
            optimizer = optim.Adam(recon_net.parameters(), lr=LR)

        total_cls_loss, total_rec_loss = 0.0, 0.0
        correct_preds, total_preds = 0, 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with autocast(device_type=device_type, enabled=(device_type == 'cuda')):
                cams, logits = cam_generator.generate_cam(images, all_classes=True, resize=False)
                cls_loss = cls_loss_fn(logits, labels)
                recon = recon_net(cams)
                rec_loss = F.l1_loss(recon, images)
                loss = cls_loss + rec_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_cls_loss += cls_loss.item() * images.size(0)
            total_rec_loss += rec_loss.item() * images.size(0)

            _, predicted = torch.max(logits, dim=1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

        avg_cls_loss = total_cls_loss / len(train_loader.dataset)
        avg_rec_loss = total_rec_loss / len(train_loader.dataset)
        train_acc = correct_preds / total_preds * 100.00

        print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] "
              f"CLS Loss: {avg_cls_loss:.4f} | "
              f"REC Loss: {avg_rec_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}%")

    # Save weights
    if model_name == 'resnet':
        torch.save(model.state_dict(), f"{CRM_MODEL_SAVE_PATH}/resnet_pet_gradcampp_crm.pth")
        torch.save(recon_net.state_dict(), f"{CRM_MODEL_SAVE_PATH}/reconstruct_net_resnet.pth")
    else:
        torch.save(model.state_dict(), f"{CRM_MODEL_SAVE_PATH}/efficientnet_pet_scorecam_crm.pth")
        torch.save(recon_net.state_dict(), f"{CRM_MODEL_SAVE_PATH}/reconstruct_net_eff.pth")

    print("Training complete. Models saved.")


if __name__ == "__main__":
    train()
