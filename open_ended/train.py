# I acknowledge the use of ChatGPT (version GPT-4o, OpenAI, https://chatgpt.com/) for assistance in debugging and
# writing docstrings.

# train_hybrid_feature.py (modified script)
import os
import argparse
import torch
import torch.optim as optim
import torchmetrics # Added for metric calculation
import time
import traceback
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from model.segnet_wrapper import SegNetWrapper
from data_utils.data_util import PetsDataset, IGNORE_INDEX
from open_ended.losses import CombinedLoss



# --- Configuration ---
DEFAULT_DATA_DIR = './data/oxford-iiit-pet'
DEFAULT_WEAK_LABEL_PATH = './weak_labels/weak_labels_train.pkl'
DEFAULT_CHECKPOINT_DIR = './checkpoints'
DEFAULT_NUM_CLASSES = 2 # IoU should have 2 classes, foreground, background

# ***** HELPER FUNCTION for formatting time *****
def format_time(seconds):
    """
    Converts seconds to HH:MM:SS format.

    Args:
        seconds: time in second format
    """

    seconds = int(round(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
# ********************************************


def setup_arg_parser():
    """Set up argparser for running this file"""
    parser = argparse.ArgumentParser(description='Train WSSS Model on Pets Dataset')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR, help='Dataset directory')
    parser.add_argument('--weak_label_path', type=str, default=DEFAULT_WEAK_LABEL_PATH, help='Path to pre-generated weak labels')
    parser.add_argument('--supervision_mode', type=str, required=True,
                        choices=['full', 'points', 'scribbles', 'boxes', 'hybrid_points_scribbles', 'hybrid_points_boxes', 'hybrid_scribbles_boxes', 'hybrid_points_scribbles_boxes'],
                        help='Type of supervision to use for training')
    parser.add_argument('--img_size', type=int, default=256, help='Image size for training (square)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--checkpoint_dir', type=str, default=DEFAULT_CHECKPOINT_DIR, help='Directory to save checkpoints')
    parser.add_argument('--run_name', type=str, required=True, help='Unique name for this training run (used for saving checkpoints)')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of dataloader workers')
    parser.add_argument('--augment', action='store_true', help='Enable basic data augmentation')
    parser.add_argument('--num_classes', type=int, default=DEFAULT_NUM_CLASSES, help='Number of output classes (e.g., 2 for BG+Pet)')

    return parser


def train_one_epoch(model, loader, optimizer, loss_fn, device, num_classes):
    """
    Performs a single training epoch for a semantic segmentation model using weak labels for loss
    and ground truth masks for metrics.

    Args:
        model (torch.nn.Module): The segmentation model to train.
        loader (DataLoader): DataLoader providing batches of (image, weak_target, ground_truth_mask).
        optimizer (torch.optim.Optimizer): Optimizer used to update model weights.
        loss_fn (callable): Loss function that supports weak supervision inputs.
        device (torch.device): Device to run the training on (CPU or CUDA).
        num_classes (int): Number of segmentation classes.

    Returns:
        tuple of metrics:
            - avg_loss (float): Average training loss across the epoch.
            - epoch_train_acc (float): Training accuracy calculated using ground truth masks.
            - epoch_train_avg_iou (float): Mean IoU across all classes using ground truth masks.
    """

    model.train()
    total_loss = 0.0
    num_batches = len(loader)
    epoch_batches_start_time = time.time()

    # --- Initialize metrics (similar to train.py) ---
    train_accuracy = torchmetrics.Accuracy(
        task="multiclass", num_classes=num_classes, ignore_index=IGNORE_INDEX, average='macro'
    ).to(device)
    train_iou = torchmetrics.JaccardIndex(
        task="multiclass", num_classes=num_classes, ignore_index=IGNORE_INDEX, average='none' # Get IoU per class
    ).to(device)
    # ------------------------------------------------

    print(f"Starting training epoch (metrics calculated against GT)...")
    # Dataset returns (image, weak_target, gt_mask)
    # We use weak_target for loss, gt_mask for metrics
    for i, batch_data in enumerate(loader):
        batch_start_time = time.time()

        if len(batch_data) != 3:
             print(f"Warning: Unexpected data format from loader in batch {i}. Expected 3 items, got {len(batch_data)}. Skipping batch.")
             continue
        images, targets, gt_masks = batch_data # Unpack all three

        images = images.to(device)
        # Ensure GT masks are suitable for metrics (LongTensor)
        gt_masks_device = gt_masks.to(device).long()

        # Move weak targets to device based on targets type
        if isinstance(targets, torch.Tensor):
            targets_device = targets.to(device).long()
            # print(f"Warning: Training with Tensor targets in hybrid script (Batch {i}). Ensure loss function handles this.")
        elif isinstance(targets, dict):
            targets_device = {k: v.to(device) for k, v in targets.items()}
        else:
            print(f"Error: Invalid weak target type received: {type(targets)}. Skipping batch {i}.")
            continue # Skip batch

        optimizer.zero_grad()

        try:
            # --- Model Forward Pass ---
            outputs = model(images) # Expect dict {'segmentation':...} or tensor

            # --- Extract Segmentation Logits ---
            # Handle potential dict or tensor output from model consistently
            seg_logits = None
            if isinstance(outputs, dict):
                seg_logits = outputs.get('segmentation')
                if seg_logits is None:
                     print(f"Error: Model output dictionary missing 'segmentation' key in training batch {i}. Skipping.")
                     continue
            elif isinstance(outputs, torch.Tensor):
                seg_logits = outputs
            else:
                 print(f"Error: Unexpected model output type in training batch {i}: {type(outputs)}. Skipping.")
                 continue

            # --- Loss Calculation (using weak labels) ---
            # Pass the original outputs (dict or tensor) and weak targets (targets_device) to loss_fn
            loss = loss_fn(outputs, targets_device)

            if torch.isnan(loss) or torch.isinf(loss):
                 print(f"Warning: NaN or Inf loss encountered in training batch {i}. Skipping backward pass.")
                 continue

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # --- Metric Calculation (using GT masks) ---
            with torch.no_grad(): # Ensure metrics don't track gradients
                 preds = torch.argmax(seg_logits, dim=1) # Get predictions from logits
                 # Update metrics using predictions and the GROUND TRUTH mask
                 train_accuracy.update(preds, gt_masks_device)
                 train_iou.update(preds, gt_masks_device)
            # --------------------------------------------

            # --- Optional: Batch timing and ETA within epoch ---
            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            batches_elapsed_time = batch_end_time - epoch_batches_start_time
            batches_remaining = num_batches - (i + 1)
            avg_batch_time = batches_elapsed_time / (i + 1) if (i + 1) > 0 else 0
            eta_epoch_seconds = batches_remaining * avg_batch_time if avg_batch_time > 0 else 0
            # ---------------------------------------------------

            if (i + 1) % 50 == 0: # Print progress every 50 batches
                 eta_epoch_str = format_time(eta_epoch_seconds)
                 # Calculate current average accuracy for interim reporting (optional)
                 current_acc = train_accuracy.compute().item() # Compute on the fly
                 print(f"  Train Batch {i+1}/{num_batches}, Loss: {total_loss / (i+1):.4f}, Current Train Acc (GT): {current_acc:.4f}, Batch Time: {batch_duration:.2f}s, Epoch ETA: {eta_epoch_str}")

        except Exception as e:
            print(f"Error during training batch {i}: {e}")
            traceback.print_exc()
            continue # Skip batch on error

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # --- Compute final epoch metrics ---
    epoch_train_acc = 0.0
    epoch_train_avg_iou = 0.0
    try:
        epoch_train_acc = train_accuracy.compute().item()
        # Handle potential issues with IoU calculation (e.g., division by zero if no true positives/union)

        # Inside train_one_epoch after computing iou_per_class
        iou_per_class = train_iou.compute()
        # Compute average IoU across all classes
        if iou_per_class.numel() > 0:
            avg_iou = torch.mean(iou_per_class)
            if torch.isnan(avg_iou) or torch.isinf(avg_iou):
                epoch_train_avg_iou = 0.0
                print("Warning: Average IoU is NaN or Inf for training.")
            else:
                epoch_train_avg_iou = avg_iou.item()
        else:
            epoch_train_avg_iou = 0.0
            print("Warning: No IoU values computed for training.")
        
        
    except Exception as e:
        print(f"Error computing final training metrics: {e}")
        traceback.print_exc()
    finally:
        # Reset metrics for the next epoch
        train_accuracy.reset()
        train_iou.reset()
    # ---------------------------------

    # --- Updated Print Statement & Return Value ---
    print(f"Training epoch finished. Average Loss: {avg_loss:.4f}, Train Acc (GT): {epoch_train_acc:.4f}, Train Avg IoU (GT): {epoch_train_avg_iou:.4f}")
    return avg_loss, epoch_train_acc, epoch_train_avg_iou # Return metrics


# Modified validate_one_epoch to calculate and return val loss, accuracy, and pet_iou
def validate_one_epoch(model, loader, device, num_classes):
    """
    Evaluates the model for one epoch on the validation dataset using ground truth masks.

    Args:
        model (torch.nn.Module): Trained segmentation model to validate.
        loader (DataLoader): DataLoader providing batches of (image, _, ground_truth_mask).
        device (torch.device): Device to run validation on (CPU or CUDA).
        num_classes (int): Number of segmentation classes.

    Returns:
        tuple of metrics:
            - avg_loss (float): Average validation loss across all batches.
            - epoch_val_acc (float): Validation accuracy using ground truth.
            - epoch_val_avg_iou (float): Mean IoU across all classes using ground truth.
    """

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

    # Use standard CrossEntropyLoss for validation against GT masks
    val_loss_fn = CrossEntropyLoss(ignore_index=IGNORE_INDEX).to(device)

    print(f"Starting validation epoch...")
    with torch.no_grad():
        # Validation uses the ground truth mask (third item from dataset)
        for i, (images, _, gt_masks) in enumerate(loader):
            images = images.to(device)
            gt_masks = gt_masks.to(device).long() # Ensure LongTensor for GT

            # --- Model Forward Pass ---
            outputs = model(images) # Expect dict or tensor

            # --- Loss Calculation ---
            seg_logits_for_loss = None
            # Extract segmentation logits consistently, assuming 'segmentation' key if dict
            if isinstance(outputs, dict):
                seg_logits_for_loss = outputs.get('segmentation')
            elif isinstance(outputs, torch.Tensor): # Handle case where model directly outputs seg logits
                seg_logits_for_loss = outputs
            else:
                 print(f"Warning: Unexpected model output type during validation: {type(outputs)}")

            if seg_logits_for_loss is not None:
                try:
                    # Calculate validation loss against the ground truth mask
                    loss = val_loss_fn(seg_logits_for_loss, gt_masks)

                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        total_loss += loss.item()
                    else:
                        print(f"Warning: NaN/Inf validation loss in batch {i}")
                except Exception as e:
                     print(f"Error during validation loss calculation in batch {i}: {e}")
            else:
                print(f"Warning: Could not find 'segmentation' output or suitable tensor in model output during validation batch {i}")

            # --- Metric Calculation (Accuracy and IoU against GT) ---
            try:
                 seg_logits_for_metric = seg_logits_for_loss # Use the same logits used for loss
                 if seg_logits_for_metric is not None:
                      preds = torch.argmax(seg_logits_for_metric, dim=1) # Shape: (B, H, W)
                      val_accuracy.update(preds, gt_masks)
                      val_iou.update(preds, gt_masks)
                 # else: # Warning already printed above if logits are None
                 #    pass
            except Exception as e:
                 print(f"Error during validation metric calculation in batch {i}: {e}")

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # --- Compute and Log Final Metrics ---
    epoch_val_acc = 0.0
    epoch_val_avg_iou = 0.0
    try:
        epoch_val_acc = val_accuracy.compute().item()
        
        final_iou_per_class = val_iou.compute()
        if final_iou_per_class.numel() > 0:
            avg_iou = torch.mean(final_iou_per_class)
            if torch.isnan(avg_iou) or torch.isinf(avg_iou):
                epoch_val_avg_iou = 0.0
                print("Warning: Average IoU is NaN or Inf for validation.")
            else:
                epoch_val_avg_iou = avg_iou.item()
        else:
            epoch_val_avg_iou = 0.0
            print("Warning: No IoU values computed for validation.")

        print(f"Validation epoch finished. Average Loss: {avg_loss:.4f}, Val Acc: {epoch_val_acc:.4f}, Val Avg IoU: {epoch_val_avg_iou:.4f}")
    except Exception as e:
        print(f"Error computing final validation metrics: {e}")
        print(f"Validation epoch finished. Average Loss: {avg_loss:.4f}, Val Acc: Calculation Error, Val Avg IoU: Calculation Error")
    finally:
        val_accuracy.reset()
        val_iou.reset()

    # Return all calculated validation metrics
    return avg_loss, epoch_val_acc, epoch_val_avg_iou


def main():
    """Main function for training"""
    torch.manual_seed(42)
    parser = setup_arg_parser()
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Warning: MPS support is experimental. Consider CUDA if available.")
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # --- Create Datasets and Dataloaders ---
    img_size_tuple = (args.img_size, args.img_size)
    # Training dataset uses the specified hybrid/weak mode
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
    # Determine model mode based on supervision type
    if args.supervision_mode in ['hybrid_points_scribbles', 'hybrid_points_boxes', 'hybrid_scribbles_boxes', 'hybrid_points_scribbles_boxes']:
        model_mode = 'hybrid' # Expects dict output?
    else:
        # If running non-hybrid modes through this script, assume 'single' output
        model_mode = 'single' # Assumed equivalent to 'segmentation' for SegNeXtWrapper
    num_output_classes = args.num_classes
    print(f"Initializing model in '{model_mode}' mode with {num_output_classes} output classes for segmentation head.")

    model = SegNetWrapper(num_classes=num_output_classes, mode=model_mode) # Pass num_classes here
    model.to(device)

    # --- Define Loss Function ---
    if args.supervision_mode == 'full':
        loss_fn = CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    else:
        loss_fn = CombinedLoss(ignore_index=IGNORE_INDEX, mode=args.supervision_mode)

    loss_fn.to(device) # Move loss function to device

    # --- Optimizer ---
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)


    # --- Training Loop ---
    # Initialize based on validation accuracy
    best_val_iou = 0.0 # Changed from best_val_iou
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path_base = os.path.join(args.checkpoint_dir, f"{args.run_name}")

    print(f"\nStarting training run: {args.run_name}")
    print(f"Supervision: {args.supervision_mode}, Batch Size: {args.batch_size}, LR: {args.lr}, Epochs: {args.epochs}, Num Classes: {num_output_classes}")

    # ***** ADDED: Time tracking variables *****
    training_start_time = time.time()
    # ****************************************

    for epoch in range(args.epochs):
        # ***** ADDED: Epoch start time *****
        epoch_start_time = time.time()
        # ***********************************

        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")

        try:
            # Get train loss (only loss is returned now)
            train_loss, train_acc, train_avg_iou = train_one_epoch(
                model, train_loader, optimizer, loss_fn, device, num_output_classes
            )
            # Get validation metrics
            val_loss, val_acc, val_avg_iou = validate_one_epoch(
                model, val_loader, device, num_output_classes
            )
            scheduler.step() # Step the scheduler each epoch
            current_lr = scheduler.get_last_lr()[0]

            # ***** MODIFIED PRINT STATEMENT *****
            print(f"Epoch {epoch+1} Summary: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train Avg IoU: {train_avg_iou:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Avg IoU: {val_avg_iou:.4f} | "
                  f"LR: {current_lr:.6f}")

            # --- Save checkpoint based on best validation accuracy ---
            if val_avg_iou > best_val_iou:
                best_val_iou = val_avg_iou
                save_path = f"{checkpoint_path_base}_best_acc.pth" # Changed filename
                model.cpu() # Move to CPU before saving
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss, # Save train loss
                    'val_loss': val_loss,     # Save current val loss
                    'val_acc': val_acc,       # Save current val accuracy
                    'val_avg_iou': val_avg_iou,# Save current val Avg IoU
                    'best_val_acc': best_val_iou, # Save the best accuracy achieved
                    'args': args
                }, save_path)
                model.to(device) # Move back to device
                print(f"Checkpoint saved: Validation IOU improved to {best_val_iou:.4f} (Loss: {val_loss:.4f}, Avg IoU: {val_avg_iou:.4f}). Saved to {save_path}")
            # ********************************************************

            # Optional: Save latest checkpoint
            if (epoch + 1) % 3 == 0 or epoch == args.epochs - 1:
                latest_save_path = f"{checkpoint_path_base}_{epoch}.pth"
                model.cpu()
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss, # Save current train loss
                    'val_loss': val_loss,     # Save current val metrics
                    'val_acc': val_acc,
                    'val_avg_iou': val_avg_iou,
                    'best_val_acc': best_val_iou, # Keep track of best acc
                    'args': args
                }, latest_save_path)
                model.to(device)
                print(f"Latest checkpoint saved to {latest_save_path}")

            # ***** ADDED: Time Estimation Logic *****
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            total_elapsed_time = epoch_end_time - training_start_time
            average_epoch_time = total_elapsed_time / (epoch + 1)
            remaining_epochs = args.epochs - (epoch + 1)
            estimated_remaining_time = remaining_epochs * average_epoch_time

            print(f"Epoch {epoch+1} duration: {format_time(epoch_duration)}")
            if remaining_epochs > 0:
                 print(f"Estimated time remaining: {format_time(estimated_remaining_time)}")
            # ***************************************

        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Exiting.")
            break

        except Exception as e:
            print(f"\nAn error occurred during epoch {epoch+1}: {e}")
            import traceback
            traceback.print_exc()
            break


    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    print("\n------------------------------------")
    print("Training finished.")
    print(f"Total Training Time: {format_time(total_training_time)}")
    # ************************************
    print(f"Best Validation IOU achieved: {best_val_iou:.4f}") # Changed metric name
    print(f"Best model saved to: {checkpoint_path_base}_best_acc.pth (if accuracy improved)") # Changed filename
    print(f"Latest model saved to: {checkpoint_path_base}_latest.pth")


if __name__ == '__main__':
    main()