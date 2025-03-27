import torch
import torch.nn as nn
import torch.optim as optim

from .reconstruct_net import ReconstructNet
from .vgg_loss import VGGLoss
from cam.resnet_gradcampp import ResNet50_CAM, GradCAMpp
from cam.dataset.oxfordpet import download_pet_dataset
from cam import NUM_CLASSES, LR, NUM_EPOCHS, MODEL_SAVE_PATH


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet50_CAM(NUM_CLASSES).to(device)
    cam_generator = GradCAMpp(model)
    recon_net = ReconstructNet(input_channel=NUM_CLASSES).to(device)

    optimizer = optim.Adam(
        list(model.parameters()) + list(recon_net.parameters()),
        lr=LR
    )

    cls_loss_fn = nn.CrossEntropyLoss()

    # if vggloss performs poorly, try l1
    rec_loss_fn = VGGLoss(device)
    # rec_loss_fn = nn.L1Loss()

    train_loader, _ = download_pet_dataset()

    for epoch in range(NUM_EPOCHS):
        model.train()
        recon_net.train()
        total_cls_loss, total_rec_loss = 0.0, 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Get multi-class CAMs - (B, C, H, W), (B, C)
            cams, logits = cam_generator.generate_cam(images, all_classes=True)

            cls_loss = cls_loss_fn(logits, labels)
            
            # Reconstruct image from CAM
            recon = recon_net(cams)
            rec_loss = rec_loss_fn(recon, images)

            loss = cls_loss + rec_loss
            loss.backward()
            optimizer.step()

            total_cls_loss += cls_loss.item()
            total_rec_loss += rec_loss.item()

        print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] "
              f"CLS Loss: {total_cls_loss:.4f} | REC Loss: {total_rec_loss:.4f}")

    # Save checkpoints
    torch.save(model.state_dict(), f"{MODEL_SAVE_PATH}/resnet50_pet_cam_crm.pth")
    torch.save(recon_net.state_dict(), f"{MODEL_SAVE_PATH}/reconstruct_net.pth")
    print("Training complete. Models saved.")

if __name__ == "__main__":
    train()
