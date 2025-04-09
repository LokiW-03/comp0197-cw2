# train.py (original script modified)
import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model_utils import EffUnetWrapper
from data_utils import PetsDataset, IGNORE_INDEX
from losses import PartialCrossEntropyLoss, CombinedLoss
from model_utils import SegNeXtWrapper
import numpy as np # Needed for metric calculation maybe
import torchmetrics # Added for metric calculation

# --- Configuration ---
DEFAULT_DATA_DIR = './data'
DEFAULT_WEAK_LABEL_PATH = './weak_labels/weak_labels_train.pkl'
DEFAULT_CHECKPOINT_DIR = './checkpoints'
DEFAULT_NUM_CLASSES = 2 # BG + Pet

def setup_arg_parser():
    parser = argparse.ArgumentParser(description='Train WSSS Model on Pets Dataset')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR, help='Dataset directory')
    parser.add_argument('--weak_label_path', type=str, default=DEFAULT_WEAK_LABEL_PATH, help='Path to pre-generated weak labels')
    parser.add_argument('--supervision_mode', type=str, required=True,
                        choices=['full', 'points', 'scribbles', 'boxes'],
                        help='Type of supervision to use for training')
    parser.add_argument('--backbone', type=str, default='efficientnet-b0', help='EffUnet backbone')
    parser.add_argument('--img_size', type=int, default=256, help='Image size for training (square)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--checkpoint_dir', type=str, default=DEFAULT_CHECKPOINT_DIR, help='Directory to save checkpoints')
    parser.add_argument('--run_name', type=str, required=True, help='Unique name for this training run (used for saving checkpoints)')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of dataloader workers')
    parser.add_argument('--lambda_seg', type=float, default=1.0, help='Weight for segmentation loss in hybrid mode')
    parser.add_argument('--augment', action='store_true', help='Enable basic data augmentation')
    parser.add_argument('--num_classes', type=int, default=DEFAULT_NUM_CLASSES, help='Number of output classes (e.g., 2 for BG+Pet)')


    return parser

# Modified train_one_epoch to calculate and return train loss, accuracy, and pet_iou
def train_one_epoch(model, loader, optimizer, loss_fn, device, mode, num_classes):
    model.train()
    total_loss = 0.0
    num_batches = len(loader)

    # Initialize metrics
    train_accuracy = torchmetrics.Accuracy(
        task="multiclass", num_classes=num_classes, ignore_index=IGNORE_INDEX, average='macro'
        ).to(device)
    train_iou = torchmetrics.JaccardIndex(
        task="multiclass", num_classes=num_classes, ignore_index=IGNORE_INDEX, average='none' # Get IoU per class
        ).to(device)

    print(f"Starting training epoch...") # Simple progress indicator
    for i, (images, targets, _) in enumerate(loader): # Ignore GT mask during weak training unless needed
        images = images.to(device)

        # Targets here are the weak labels (points, scribbles, boxes pseudo-masks, or full masks)
        # Need targets as LongTensor for loss and metrics
        if isinstance(targets, torch.Tensor):
            targets_device = targets.to(device).long() # Ensure LongTensor
        elif isinstance(targets, dict):
            # This script shouldn't hit this based on its modes, but handle defensively
            # Assuming the relevant target for segmentation is a tensor if dict is used
            if 'segmentation' in targets:
                 targets_device = targets['segmentation'].to(device).long()
            else:
                 # Heuristic: try to find a tensor target
                 found_target = None
                 for k, v in targets.items():
                     if isinstance(v, torch.Tensor):
                         found_target = v.to(device).long()
                         print(f"Warning: Using target['{k}'] for training metrics as default 'segmentation' not found.")
                         break
                 if found_target is None:
                     print(f"Error: Cannot determine target tensor for metrics in training batch {i}. Skipping metrics update.")
                     targets_device = None # Signal to skip metrics
                 else:
                     targets_device = found_target

        else:
            raise TypeError("Invalid target type")

        optimizer.zero_grad()
        outputs = model(images) # Shape: (B, C, H, W)

        # Calculate loss based on mode
        if targets_device is not None:
            if mode in ['points', 'scribbles', 'boxes', 'full']: # Assume segmentation output
                 loss = loss_fn(outputs, targets_device) # CE/PartialCE expects long targets
            else:
                 raise ValueError(f"Unknown mode {mode} for loss calculation")

            if torch.isnan(loss) or torch.isinf(loss):
                 print(f"Warning: NaN or Inf loss encountered in batch {i}. Skipping backward pass.")
                 # Optional: Add more debugging here (print inputs/outputs)
                 continue # Skip this batch

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Calculate metrics
            try:
                preds = torch.argmax(outputs, dim=1) # Shape: (B, H, W)
                train_accuracy.update(preds, targets_device)
                train_iou.update(preds, targets_device)
            except Exception as e:
                print(f"Warning: Error updating training metrics in batch {i}: {e}")

        else:
             # Skip batch if target couldn't be determined
             print(f"Skipping training batch {i} due to missing target for metrics.")
             continue


        if (i + 1) % 50 == 0: # Print progress every 50 batches
             print(f"  Batch {i+1}/{num_batches}, Current Avg Loss: {total_loss / (i+1):.4f}")

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # Compute final epoch metrics
    epoch_train_acc = 0.0
    epoch_train_pet_iou = 0.0
    try:
        epoch_train_acc = train_accuracy.compute().item()
        iou_per_class = train_iou.compute()
        if num_classes > 1 and len(iou_per_class) > 1:
            epoch_train_pet_iou = iou_per_class[1].item() # Assuming class 1 is 'pet'
        elif num_classes == 1:
             epoch_train_pet_iou = iou_per_class[0].item() # Or handle single class case if needed
        else: # Handle cases where IoU might not be computed correctly (e.g., all ignored pixels)
             epoch_train_pet_iou = 0.0

    except Exception as e:
        print(f"Error computing final training metrics: {e}")
    finally:
        train_accuracy.reset()
        train_iou.reset()

    print(f"Training epoch finished. Average Loss: {avg_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, Train Pet IoU: {epoch_train_pet_iou:.4f}")
    return avg_loss, epoch_train_acc, epoch_train_pet_iou

# Modified validate_one_epoch to calculate and return val loss, accuracy, and pet_iou
def validate_one_epoch(model, loader, loss_fn, device, mode, num_classes):
    model.eval()
    total_loss = 0.0
    num_batches = len(loader)

    # Initialize metrics
    val_accuracy = torchmetrics.Accuracy(
        task="multiclass", num_classes=num_classes, ignore_index=IGNORE_INDEX, average='macro'
        ).to(device)
    val_iou = torchmetrics.JaccardIndex(
        task="multiclass", num_classes=num_classes, ignore_index=IGNORE_INDEX, average='none' # Get IoU per class
        ).to(device)

    print(f"Starting validation epoch...")
    with torch.no_grad():
        # Validation uses the ground truth mask (third item from dataset)
        for i, (images, _, gt_masks) in enumerate(loader):
            images = images.to(device)
            gt_masks = gt_masks.to(device).long() # Ensure LongTensor for GT

            # --- Model Forward Pass ---
            outputs = model(images) # Shape: (B, C, H, W)

            # --- Loss Calculation (using the same loss_fn as training for consistency, but on GT) ---
            try:
                # For validation, always calculate loss against the ground truth mask
                loss = loss_fn(outputs, gt_masks)

                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                else:
                    print(f"Warning: NaN/Inf validation loss in batch {i}")

            except Exception as e:
                 print(f"Error during validation loss calculation in batch {i}: {e}")
                 # Continue validation if one batch fails

            # --- Metric Calculation (Accuracy and IoU against GT) ---
            try:
                preds = torch.argmax(outputs, dim=1) # Shape: (B, H, W)
                val_accuracy.update(preds, gt_masks)
                val_iou.update(preds, gt_masks)
            except Exception as e:
                 print(f"Error during validation metric calculation in batch {i}: {e}")


    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # Compute final epoch metrics
    epoch_val_acc = 0.0
    epoch_val_pet_iou = 0.0
    try:
        epoch_val_acc = val_accuracy.compute().item()
        iou_per_class = val_iou.compute()
        if num_classes > 1 and len(iou_per_class) > 1:
            epoch_val_pet_iou = iou_per_class[1].item() # Assuming class 1 is 'pet'
        elif num_classes == 1:
             epoch_val_pet_iou = iou_per_class[0].item()
        else:
             epoch_val_pet_iou = 0.0
    except Exception as e:
        print(f"Error computing final validation metrics: {e}")
    finally:
        val_accuracy.reset()
        val_iou.reset()

    print(f"Validation epoch finished. Average Loss: {avg_loss:.4f}, Val Acc: {epoch_val_acc:.4f}, Val Pet IoU: {epoch_val_pet_iou:.4f}")
    return avg_loss, epoch_val_acc, epoch_val_pet_iou


def main():
    torch.manual_seed(42)
    parser = setup_arg_parser()
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): # Check for MPS availability
        device = torch.device('mps')
        print("Warning: MPS support is experimental. Consider CUDA if available.")
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # --- Create Datasets and Dataloaders ---
    img_size_tuple = (args.img_size, args.img_size)
    train_dataset = PetsDataset(args.data_dir, split='train', supervision_mode=args.supervision_mode,
                                weak_label_path=args.weak_label_path, img_size=img_size_tuple, augment=args.augment)
    # Validation dataset always uses 'full' supervision mode internally to load GT masks for evaluation
    val_dataset = PetsDataset(args.data_dir, split='val', supervision_mode='full', # Load GT for validation
                              weak_label_path=args.weak_label_path, img_size=img_size_tuple, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True if device != torch.device('cpu') else False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True if device != torch.device('cpu') else False)

    # --- Initialize Model ---
    model_mode = 'segmentation' # This script focuses on segmentation output

    # Use num_classes from args
    num_output_classes = args.num_classes # e.g., 2 for Background + Pet
    print(f"Initializing model with {num_output_classes} output classes.")
    # model = EffUnetWrapper(backbone=args.backbone, num_classes=num_output_classes, mode=model_mode)
    # model.to(device)

    model = SegNeXtWrapper(num_classes=num_output_classes, mode='single') # Assuming 'single' is equiv to segmentation
    model.to(device)

    # --- Define Loss Function ---
    if args.supervision_mode in ['points', 'scribbles']:
        loss_fn = PartialCrossEntropyLoss(ignore_index=IGNORE_INDEX)
    elif args.supervision_mode in ['boxes', 'full']:
        # For boxes pseudo-mask and full GT mask
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    else:
        # Should not happen due to argparse choices, but safeguard
        raise ValueError(f"Supervision mode {args.supervision_mode} not implemented for loss")
    loss_fn.to(device) # Move loss function to device if it has parameters/buffers

    # --- Optimizer ---
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    # Optional: Learning rate scheduler (e.g., CosineAnnealingLR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)


    # --- Training Loop ---
    # Initialize based on validation accuracy
    best_val_acc = 0.0 # Changed from best_val_loss
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path_base = os.path.join(args.checkpoint_dir, f"{args.run_name}")

    print(f"\nStarting training run: {args.run_name}")
    print(f"Supervision: {args.supervision_mode}, Batch Size: {args.batch_size}, LR: {args.lr}, Epochs: {args.epochs}, Num Classes: {num_output_classes}")


    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")

        try:
            # Get train metrics
            train_loss, train_acc, train_pet_iou = train_one_epoch(
                model, train_loader, optimizer, loss_fn, device, args.supervision_mode, num_output_classes
            )
            # Get validation metrics
            val_loss, val_acc, val_pet_iou = validate_one_epoch(
                model, val_loader, loss_fn, device, args.supervision_mode, num_output_classes
            )

            scheduler.step() # Step the scheduler each epoch
            current_lr = scheduler.get_last_lr()[0]

            # ***** MODIFIED PRINT STATEMENT *****
            print(f"Epoch {epoch+1} Summary: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train Pet IoU: {train_pet_iou:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Pet IoU: {val_pet_iou:.4f} | "
                  f"LR: {current_lr:.6f}")

            # --- Save checkpoint based on best validation accuracy ---
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_path = f"{checkpoint_path_base}_best_acc.pth" # Changed filename
                # Save metrics along with model state
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'train_pet_iou': train_pet_iou,
                    'val_loss': val_loss,
                    'val_acc': val_acc,       # Save the current val_acc
                    'val_pet_iou': val_pet_iou,
                    'best_val_acc': best_val_acc, # Save the best val_acc achieved so far
                    'args': args # Save config too
                }, save_path)
                print(f"Checkpoint saved: Validation accuracy improved to {best_val_acc:.4f}. Saved to {save_path}")
            # ---------------------------------------------------------

            # Optional: Save latest checkpoint every N epochs or at the end
            if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
                latest_save_path = f"{checkpoint_path_base}_latest.pth"
                model.cpu() # Move to CPU for saving
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss, # Save current metrics for latest
                    'train_acc': train_acc,
                    'train_pet_iou': train_pet_iou,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_pet_iou': val_pet_iou,
                    'best_val_acc': best_val_acc, # Keep track of best acc in latest too
                    'args': args
                }, latest_save_path)
                model.to(device) # Move back to device
                print(f"Latest checkpoint saved to {latest_save_path}")

        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Exiting.")
            break # Exit loop gracefully

        except Exception as e:
            print(f"\nAn error occurred during epoch {epoch+1}: {e}")
            import traceback
            traceback.print_exc() # Print traceback for debugging
            break


    print("\n------------------------------------")
    print("Training finished.")
    print(f"Best Validation Accuracy achieved: {best_val_acc:.4f}")
    print(f"Best model saved to: {checkpoint_path_base}_best_acc.pth (if accuracy improved)")
    print(f"Latest model saved to: {checkpoint_path_base}_latest.pth")
    print("\nRECOMMENDATION: Load the '_best_acc.pth' checkpoint and evaluate it on the separate TEST set for final performance.")


if __name__ == '__main__':
    # Make sure to install torchmetrics: pip install torchmetrics
    main()