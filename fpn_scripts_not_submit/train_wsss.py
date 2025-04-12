# train_wsss.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import time

# --- Model and Data Imports ---
from cam.load_pseudo import load_pseudo
from supervised.train import compute_test_metrics_fn
from data_utils.data import testset
from fpn_scripts_not_submit.fpn import FPN
from model.baseline_segnet import SegNet
from model.baseline_unet import UNet
from model.segnext import SegNeXt

# --- Constants ---
NUM_CLASSES = 3 # Foreground, Background, Contour/Boundary

def train_segmentation_model(
    seg_model_name="fpn",
    encoder_name="resnet34",
    pseudo_mask_path="cam/saved_models/resnet50_pet_cam_pseudo.pt",
    data_root='./data',
    save_dir='./saved_models_wsss',
    epochs=10,
    batch_size=16,
    lr=1e-4,
    t_max_factor=1.0, # Multiplier for T_MAX calculation (1.0 = standard)
    eta_min=1e-6,
    device: torch.device = torch.device('cpu'),
):
    """
    Trains the FPN segmentation model using pseudo-masks.

    Args:
        seg_model_name (str): Name of the segmentation model ('fpn', 'segnext', 'segnet', 'unet').
        encoder_name (str): Backbone encoder for FPN ('resnet34', 'resnet50', etc.).
        pseudo_mask_path (str): Path to the .pt file containing pseudo mask dataloader info.
        data_root (str): Root directory for the OxfordIIITPet dataset.
        save_dir (str): Directory to save trained models.
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size.
        lr (float): Learning rate for the optimizer.
        t_max_factor (float): Multiplier for CosineAnnealingLR T_MAX.
        eta_min (float): Minimum learning rate for CosineAnnealingLR.
        device: Auto-detect/manually specify device ('cuda', 'mps', 'cpu').
    """

    # --- Create Save Directory ---
    os.makedirs(save_dir, exist_ok=True)
    model_save_path_base = os.path.join(save_dir, f"fpn_{encoder_name}_wsss")

    # --- DataLoaders ---
    print("Loading pseudo-mask training data...")
    pseudo_loader = load_pseudo(
        pseudo_mask_path,
        batch_size=batch_size,
        shuffle=True,
        device=device
    )
    
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=15,         # Adjust based on system
        pin_memory=(device.type == 'cuda')
    )
    print(f"Training samples (pseudo-masks): {len(pseudo_loader.dataset)}")
    print(f"Test samples (ground truth): {len(testset)}")

    # --- Model Initialization ---
    print(f"Initializing {seg_model_name.upper()} model with {encoder_name}(device.type == 'cuda' backbone...")
    
    if seg_model_name == 'segnext':
        model = SegNeXt(num_classes=3).to(device)
        print('Using SegNeXt model')
    elif seg_model_name == 'segnet':
        model = SegNet().to(device)
        print('Using SegNet model')
    elif seg_model_name == 'unet':
        print('Using UNet model')
        model = UNet(3, 3).to(device)
    elif seg_model_name == "efficientunet":
        print('Using EfficientUNet model')
    elif seg_model_name == "fpn":
        model = FPN(
            encoder_name=encoder_name,
            in_channels=3,
            classes=NUM_CLASSES,
            pretrained=True
        ).to(device)
    else:
        raise ValueError(f"Unknown model name: {seg_model_name}.")

    # --- Loss Function ---
    loss_fn = nn.CrossEntropyLoss()
    print(f"Using CrossEntropyLoss.")

    # --- Optimizer ---
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    print(f"Using AdamW optimizer with LR={lr}.")

    # --- Learning Rate Scheduler ---
    steps_per_epoch = len(pseudo_loader)
    t_max_effective = int(steps_per_epoch * epochs * t_max_factor)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=t_max_effective,
        eta_min=eta_min
    )
    print(f"Using CosineAnnealingLR scheduler with T_max={t_max_effective}, eta_min={eta_min}.")

    # --- Training Loop ---
    best_test_iou = 0.0
    start_time_total = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        model.train()
        train_loss_accum = 0.0
        processed_samples = 0

        for i, (images, pseudo_masks) in enumerate(pseudo_loader):
            # Ensure masks are LongTensor and remove channel dim if necessary
            pseudo_masks = pseudo_masks.squeeze(1).long() # Shape (B, H, W)

            optimizer.zero_grad()
            outputs = model(images)  # Forward pass -> (B, C, H, W)
            loss = loss_fn(outputs, pseudo_masks)
            loss.backward()
            optimizer.step()         # Update weights
            scheduler.step()         # Update learning rate (per step)

            train_loss_accum += loss.item() * images.size(0)
            processed_samples += images.size(0)

        avg_train_loss_epoch = train_loss_accum / processed_samples
        print(f"Epoch {epoch+1} Average Training Loss (Pseudo Masks): {avg_train_loss_epoch:.4f}")

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        # --- Evaluation Phase ---
        print("Evaluating on Training Set (Pseudo Masks)...")
        print("Evaluating on Training Set...")
        train_metrics = compute_test_metrics_fn(model, pseudo_loader, loss_fn, device, num_classes = NUM_CLASSES)
        print(f"Epoch {epoch+1} Train Metrics:")
        print(f"  Loss: {train_metrics['loss']:.4f}")
        print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"  Precision: {train_metrics['precision']:.4f}")
        print(f"  Recall: {train_metrics['recall']:.4f}")
        print(f"  Mean IoU: {train_metrics['iou']:.4f}")
        print(f"  Dice Score: {train_metrics['dice']:.4f}")
        
        print("----------")
        print("Evaluating on Test Set...")
        test_metrics = compute_test_metrics_fn(model, test_loader, loss_fn, device, num_classes=NUM_CLASSES)
        print(f"Epoch {epoch+1} Test Metrics:")
        print(f"  Loss: {test_metrics['loss']:.4f}")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall: {test_metrics['recall']:.4f}")
        print(f"  Mean IoU: {test_metrics['iou']:.4f}")
        print(f"  Dice Score: {test_metrics['dice']:.4f}")
        
        print("----------")
        print(f"Epoch Duration: {epoch_duration:.2f} seconds")

        # --- Save Best Model ---
        current_iou = test_metrics['iou']
        if current_iou > best_test_iou:
            best_test_iou = current_iou
            save_path = f"{model_save_path_base}_best_iou.pth"
            torch.save(model.state_dict(), save_path)
            print(f"*** New best model saved with Test IoU: {best_test_iou:.4f} to {save_path} ***")

        # --- Save Checkpoint Periodically (Optional) ---
        # if (epoch + 1) % 10 == 0: # Save every 10 epochs
        #     ckpt_path = f"{model_save_path_base}_epoch_{epoch+1}.pth"
        #     torch.save({
        #         'epoch': epoch + 1,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'scheduler_state_dict': scheduler.state_dict(),
        #         'best_test_iou': best_test_iou,
        #     }, ckpt_path)
        #     print(f"Checkpoint saved to {ckpt_path}")

    end_time_total = time.time()
    total_duration = end_time_total - start_time_total
    print(f"\n--- Training Finished ---")
    print(f"Total Training Time: {total_duration / 60:.2f} minutes")
    print(f"Best Test Mean IoU achieved: {best_test_iou:.4f}")
    print(f"Final model state dict saved to: {model_save_path_base}_final.pth")
    torch.save(model.state_dict(), f"{model_save_path_base}_final.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FPN Segmentation Model with Pseudo Masks (WSSS)")

    # --- Model Arguments ---
    parser.add_argument('--encoder_name', type=str, default='resnet34', choices=['resnet34', 'resnet50'], help='Backbone encoder for FPN')

    # --- Data Arguments ---
    parser.add_argument('--pseudo_mask_path', type=str, default='cam/saved_models/resnet50_pet_cam_pseudo.pt', help='Path to the saved pseudo mask dataloader file (.pt)')
    parser.add_argument('--data_root', type=str, default='./data', help='Root directory for OxfordIIITPet dataset')

    # --- Training Hyperparameters ---
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--t_max_factor', type=float, default=1.0, help='Multiplier for CosineAnnealingLR T_max (relative to total steps)')
    parser.add_argument('--eta_min', type=float, default=1e-6, help='Minimum learning rate for CosineAnnealingLR')

    # --- Infrastructure Arguments ---
    parser.add_argument('--save_dir', type=str, default='./saved_models_wsss', help='Directory to save trained models')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    train_segmentation_model(
        encoder_name=args.encoder_name,
        pseudo_mask_path=args.pseudo_mask_path,
        data_root=args.data_root,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        t_max_factor=args.t_max_factor,
        eta_min=args.eta_min,
        device=device,
    )
    