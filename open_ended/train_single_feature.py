# train.py (original script modified)
import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from open_ended.model_utils import EffUnetWrapper, SegNetWrapper
from open_ended.data_utils import PetsDataset, IGNORE_INDEX
from open_ended.losses import PartialCrossEntropyLoss, CombinedLoss
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

def train_one_epoch(model, loader, optimizer, loss_fn, device, mode, num_classes):
    model.train()
    total_loss = 0.0
    num_batches = len(loader)

    # Initialize metrics (to be computed against GT)
    train_accuracy = torchmetrics.Accuracy(
        task="multiclass", num_classes=num_classes, ignore_index=IGNORE_INDEX, average='macro'
    ).to(device)
    train_iou = torchmetrics.JaccardIndex(
        task="multiclass", num_classes=num_classes, ignore_index=IGNORE_INDEX, average='none'
    ).to(device)

    print(f"Starting training epoch (Metrics calculated against GT)...")
    # Expecting (images, supervision_target, gt_mask) from the DataLoader
    for i, batch_data in enumerate(loader):
        if len(batch_data) != 3:
             print(f"Warning: Unexpected data format from loader in batch {i}. Expected 3 items, got {len(batch_data)}. Skipping batch.")
             continue
        # Unpack all three components
        images, targets, gt_masks = batch_data

        images = images.to(device)
        # --- Use GT Mask for metrics ---
        gt_masks_device = gt_masks.to(device).long() # Ensure GT is LongTensor on device

        # --- Prepare Supervision Target for LOSS ---
        # This logic handles Tensor or Dict targets for the loss function
        targets_for_loss = None
        if isinstance(targets, torch.Tensor):
            # Expected case for single supervision modes if dataset is correct
            targets_for_loss = targets.to(device).long()
        elif isinstance(targets, dict):
             # Handle dict if dataset returns it unexpectedly or for specific loss fns
             if mode == 'points' and 'points' in targets:
                 targets_for_loss = targets['points'].to(device).long()
             elif mode == 'scribbles' and 'scribbles' in targets:
                 targets_for_loss = targets['scribbles'].to(device).long()
             elif mode == 'boxes' and 'boxes' in targets:
                 targets_for_loss = targets['boxes'].to(device).long()
             # Add more specific handling if needed, or a fallback
             else:
                 # Fallback logic (less ideal) - try finding any tensor
                 found_target = None
                 for k, v in targets.items():
                     if isinstance(v, torch.Tensor):
                         found_target = v.to(device).long()
                         print(f"Warning: Using target['{k}'] heuristically for LOSS calculation in mode '{mode}'.")
                         break
                 if found_target is None:
                      print(f"Error: Cannot determine suitable target tensor for LOSS in batch {i}. Skipping loss.")
                      targets_for_loss = None
                 else:
                      targets_for_loss = found_target
        else:
             print(f"Error: Invalid target type for loss: {type(targets)}. Skipping batch {i}.")
             continue # Skip if target type is wrong

        # --- Training Step ---
        optimizer.zero_grad()
        outputs = model(images) # Shape: (B, C, H, W) - Assuming single tensor output

        # Check model output type
        if not isinstance(outputs, torch.Tensor):
             print(f"Error: Model output is not a Tensor. Got {type(outputs)}. Skipping batch {i}.")
             continue

        # Calculate Loss (using the prepared targets_for_loss)
        loss = torch.tensor(0.0).to(device) # Initialize loss
        if targets_for_loss is not None:
             try:
                  # Use the appropriate target for the loss function
                  loss = loss_fn(outputs, targets_for_loss)

                  if torch.isnan(loss) or torch.isinf(loss):
                      print(f"Warning: NaN or Inf loss encountered in batch {i}. Skipping backward pass.")
                      continue

                  loss.backward() # Backpropagate only if loss is valid
                  optimizer.step()
                  total_loss += loss.item()

             except Exception as e:
                  print(f"Error during loss calculation/backward in batch {i}: {e}")
                  import traceback
                  traceback.print_exc()
                  # Don't update metrics if loss failed
                  continue # Skip metric update for this batch
        else:
            # If targets_for_loss is None, skip backward/step and metric update
            print(f"Skipping backward/step and metrics for batch {i} due to missing loss target.")
            continue


        # --- Calculate Metrics (using GT masks) ---
        try:
            with torch.no_grad():
                preds = torch.argmax(outputs, dim=1) # Shape: (B, H, W)
                # Update metrics using predictions and GROUND TRUTH masks
                train_accuracy.update(preds, gt_masks_device)
                train_iou.update(preds, gt_masks_device)
        except Exception as e:
            print(f"Warning: Error updating training metrics in batch {i}: {e}")
            # Continue training even if metrics fail for a batch


        if (i + 1) % 50 == 0:
             # Optionally compute/print current metrics here if needed (can be slow)
             # current_acc = train_accuracy.compute().item()
             print(f"  Batch {i+1}/{num_batches}, Current Avg Loss: {total_loss / (i+1):.4f}") # Use i+1 for avg loss calc


    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # Compute final epoch metrics from GT comparison
    epoch_train_acc = 0.0
    epoch_train_pet_iou = 0.0
    try:
        epoch_train_acc = train_accuracy.compute().item()
        iou_per_class = train_iou.compute()
        if num_classes > 1 and iou_per_class.numel() > 1 :
             # Add check for NaN/Inf IoU
             pet_iou_tensor = iou_per_class[1]
             if torch.isinf(pet_iou_tensor) or torch.isnan(pet_iou_tensor):
                 epoch_train_pet_iou = 0.0
                 print("Warning: Pet IoU calculation resulted in NaN or Inf for training epoch.")
             else:
                 epoch_train_pet_iou = pet_iou_tensor.item() # Assuming class 1 is 'pet'
        elif num_classes == 1 and iou_per_class.numel() > 0:
             pet_iou_tensor = iou_per_class[0]
             if torch.isinf(pet_iou_tensor) or torch.isnan(pet_iou_tensor):
                 epoch_train_pet_iou = 0.0
                 print("Warning: Single class IoU calculation resulted in NaN or Inf for training epoch.")
             else:
                 epoch_train_pet_iou = iou_per_class[0].item()
        else:
             epoch_train_pet_iou = 0.0
             print(f"Warning: Could not compute valid Pet IoU for training epoch (num_classes={num_classes}, iou_tensor_size={iou_per_class.numel()}).")


    except Exception as e:
        print(f"Error computing final training metrics: {e}")
    finally:
        train_accuracy.reset()
        train_iou.reset()

    print(f"Training epoch finished. Avg Loss: {avg_loss:.4f}, Train Acc (GT): {epoch_train_acc:.4f}, Train Pet IoU (GT): {epoch_train_pet_iou:.4f}")
    # Return metrics calculated against GT
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

    model = SegNetWrapper(num_classes=num_output_classes, mode='single') # Assuming 'single' is equiv to segmentation
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
    best_val_iou = 0.0 # Changed from best_val_loss
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
            if val_pet_iou > best_val_iou:
                best_val_iou = val_acc
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
                    'best_val_iou': best_val_iou, # Save the best iou achieved so far
                    'args': args # Save config too
                }, save_path)
                print(f"Checkpoint saved: Validation accuracy improved to {best_val_iou:.4f}. Saved to {save_path}")
            # ---------------------------------------------------------

            # Optional: Save latest checkpoint every N epochs or at the end
            if (epoch + 1) % 3 == 0 or epoch == args.epochs - 1:
                latest_save_path = f"{checkpoint_path_base}_{epoch}.pth"
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
                    'best_val_acc': best_val_iou, # Keep track of best acc in latest too
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
    print(f"Best Validation IOU achieved: {best_val_iou:.4f}")
    print(f"Best model saved to: {checkpoint_path_base}_best_acc.pth (if accuracy improved)")
    print(f"Latest model saved to: {checkpoint_path_base}_latest.pth")
    print("\nRECOMMENDATION: Load the '_best_acc.pth' checkpoint and evaluate it on the separate TEST set for final performance.")


if __name__ == '__main__':
    # Make sure to install torchmetrics: pip install torchmetrics
    main()