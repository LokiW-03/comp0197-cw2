import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F

from cam.dataset.oxfordpet_paths import OxfordIIITPetWithPaths
from cam.efficientnet_scorecam import EfficientNetB4_CAM, ScoreCAM
from cam.resnet_gradcampp import ResNet50_CAM, GradCAMpp
from cam.resnet_drs import ResNet50_CAM_DRS

from crm import CRM_MODEL_SAVE_PATH, BATCH_SIZE, NUM_CLASSES, NUM_EPOCHS, CLS_LR, REC_LR ,IMG_SIZE
from crm.reconstruct_net import ReconstructNet
from crm.crm_loss import VGGLoss, alignment_loss
from crm.oxfordpet_superpixel import OxfordPetSuperpixels
from crm.gen_superpixel import generate_superpixels


def train(model_name: str = 'resnet', 
          cls_lr: float = CLS_LR,
          rec_lr: float = 2e-3,
          vgg_weight: float = 2,
          align_weight: float = 0.3,
          num_epochs: int = NUM_EPOCHS):

    loss_history = {
        "cls": [],
        "rec": [],
        "align": [],
        "total": [],
        "acc": []
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(CRM_MODEL_SAVE_PATH, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    base_dataset = OxfordIIITPetWithPaths(
        root='./data', split='trainval', target_types='category',
        download=True, transform=None
    )

    superpixel_dir = "./superpixels"
    image_dir = "./data/oxford-iiit-pet/images"

    # generate superpixel if not already
    if not os.path.exists(superpixel_dir):
        print(f"Generating superpixels...")
        generate_superpixels(image_dir=image_dir, save_dir=superpixel_dir)

    trainset = OxfordPetSuperpixels(
        base_dataset=base_dataset,
        superpixel_dir="./superpixels",
        transform=transform
    )

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    if model_name == 'resnet':
        model = ResNet50_CAM(NUM_CLASSES).to(device)
        cam_generator = GradCAMpp(model)
        # Freeze bottom layer parameters, only train last two layers
        for name, param in model.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False

    elif model_name == 'resnet_drs':
        model = ResNet50_CAM_DRS(NUM_CLASSES).to(device)
        cam_generator = GradCAMpp(model)
        # Freeze bottom layer parameters, only train last two layers
        for name, param in model.named_parameters():
            if "resnet.layer4" not in name and "resnet.fc" not in name:
                param.requires_grad = False  

    else:
        model = EfficientNetB4_CAM(NUM_CLASSES).to(device)
        cam_generator = ScoreCAM(model)
        # Freeze all feature blocks except the last one
        # Assuming model.effnet.features is a nn.Sequential, freeze all blocks except the last
        for i, block in enumerate(model.effnet.features):
            if i < len(model.effnet.features) - 1:
                for param in block.parameters():
                    param.requires_grad = False

    recon_net = ReconstructNet(input_channel=NUM_CLASSES).to(device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cls_lr)

    rec_optimizer = optim.Adam(list(recon_net.parameters()), lr=rec_lr)

    cls_loss_fn = nn.CrossEntropyLoss()
    rec_loss_fn = VGGLoss(device)
    align_loss_fn = nn.MSELoss()
    scaler = GradScaler(enabled=(device_type == 'cuda'))

    model.train()
    recon_net.train()

    for epoch in range(num_epochs):
        total_cls_loss, total_rec_loss, total_align_loss = 0.0, 0.0, 0.0
        correct_preds, total_preds = 0, 0

        for images, labels, sp in train_loader:
            images, labels, sp = images.to(device), labels.to(device), sp.to(device)

            optimizer.zero_grad()

            with autocast(device_type=device_type, enabled=(device_type == 'cuda')):
                cams, logits = cam_generator.generate_cam(images, all_classes=True, resize=False)
                cls_loss = cls_loss_fn(logits, labels)
                recon = recon_net(cams)
                rec_loss = F.l1_loss(recon, images) + vgg_weight * rec_loss_fn(recon, images) 
                labels_onehot = F.one_hot(labels, num_classes=NUM_CLASSES).float()
                align_loss_val = alignment_loss(cams, sp, labels_onehot, align_loss_fn)
                loss = cls_loss + rec_loss + align_weight * align_loss_val
                # loss = cls_loss + rec_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.step(rec_optimizer)
            scaler.update()

            total_cls_loss += cls_loss.item() * images.size(0)
            total_rec_loss += rec_loss.item() * images.size(0)
            total_align_loss += align_loss_val.item() * images.size(0)

            _, predicted = torch.max(logits, dim=1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

        avg_cls_loss = total_cls_loss / len(train_loader.dataset)
        avg_rec_loss = total_rec_loss / len(train_loader.dataset)
        avg_align_loss = total_align_loss / len(train_loader.dataset)
        train_acc = correct_preds / total_preds * 100.00

        print(f"[Epoch {epoch+1:02}/{num_epochs}] "
              f"CLS Loss: {avg_cls_loss:.4f} | "
              f"REC Loss: {avg_rec_loss:.4f} | "
              # f"ALIGN Loss: {avg_align_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}%")

        total_loss = avg_cls_loss + avg_rec_loss + align_weight * avg_align_loss
        loss_history["cls"].append(avg_cls_loss)
        loss_history["rec"].append(avg_rec_loss)
        loss_history["align"].append(avg_align_loss)
        loss_history["total"].append(total_loss)
        loss_history["acc"].append(train_acc)

    if model_name == 'resnet':
        torch.save(model.state_dict(), f"{CRM_MODEL_SAVE_PATH}/resnet_pet_gradcampp_crm.pth")
        torch.save(recon_net.state_dict(), f"{CRM_MODEL_SAVE_PATH}/reconstruct_net_resnet.pth")
    
    elif model_name == 'resnet_drs':
        torch.save(model.state_dict(), f"{CRM_MODEL_SAVE_PATH}/resnet_drs_pet_gradcampp_crm.pth")
        torch.save(recon_net.state_dict(), f"{CRM_MODEL_SAVE_PATH}/reconstruct_net_resnet_drs.pth")

    else:
        torch.save(model.state_dict(), f"{CRM_MODEL_SAVE_PATH}/efficientnet_pet_scorecam_crm.pth")
        torch.save(recon_net.state_dict(), f"{CRM_MODEL_SAVE_PATH}/reconstruct_net_eff.pth")

    graph_dir = "./graph"
    os.makedirs(graph_dir, exist_ok=True)
    filename = os.path.join(graph_dir, f"{model_name}_loss_history.pt")
    torch.save(loss_history, filename)
    print("Saved loss history to ./graph/{model_name}_loss_history.pt")

    print("Training complete. Models saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'efficientnet', 'resnet_drs'])
    parser.add_argument('--cls_lr', type=float, default=CLS_LR, help="Classifier learning rate")
    parser.add_argument('--rec_lr', type=float, default=REC_LR, help="Reconstruction network learning rate")
    parser.add_argument('--vgg_weight', type=float, default=0.3, help="Weight for VGG loss")
    parser.add_argument('--align_weight', type=float, default=0.3, help="Weight for alignment loss")
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help="Number of training epochs")
    args = parser.parse_args()

    train(model_name=args.model, 
        cls_lr=args.cls_lr, 
        rec_lr=args.rec_lr, 
        vgg_weight=args.vgg_weight,
        align_weight=args.align_weight,
        num_epochs=args.epochs)
