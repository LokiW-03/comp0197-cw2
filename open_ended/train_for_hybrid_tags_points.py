# train.py (for hybrid tags and points)
import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from model import EffUnetWrapper
from data_utils import PetsDataset, IGNORE_INDEX
from losses import PartialCrossEntropyLoss, CombinedLoss
import numpy as np
import torchmetrics
# ***** ADDED IMPORT *****
import time
# ***********************

# --- Configuration ---
DEFAULT_DATA_DIR = './data'
DEFAULT_WEAK_LABEL_PATH = './weak_labels/weak_labels_train.pkl'
DEFAULT_CHECKPOINT_DIR = './checkpoints'
DEFAULT_NUM_CLASSES = 2 # 0=Background, 1=Pet for CrossEntropyLoss/IoU calculation

# ***** HELPER FUNCTION for formatting time *****
def format_time(seconds):
    """Converts seconds to HH:MM:SS format."""
    seconds = int(round(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
# ********************************************

def setup_arg_parser():
    parser = argparse.ArgumentParser(description='Train WSSS Model on Pets Dataset')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR, help='Dataset directory')
    parser.add_argument('--weak_label_path', type=str, default=DEFAULT_WEAK_LABEL_PATH, help='Path to pre-generated weak labels')
    parser.add_argument('--supervision_mode', type=str, required=True,
                        choices=['full', 'tags', 'points', 'scribbles', 'boxes', 'hybrid_tags_points','hybrid_points_scribbles', 'hybrid_points_boxes', 'hybrid_scribbles_boxes', 'hybrid_points_scribbles_boxes'],
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

def train_one_epoch(model, loader, optimizer, loss_fn, device, mode):
    model.train()
    total_loss = 0.0
    num_batches = len(loader)
    epoch_batches_start_time = time.time() # Time batch processing within epoch

    print(f"Starting training epoch...")
    # The third item returned by dataset is GT mask - ignored during weak training
    for i, (images, targets, _) in enumerate(loader):
        batch_start_time = time.time()
        images = images.to(device)

        # Move targets to device based on mode
        if mode in ['hybrid_tags_points', 'hybrid_points_scribbles', 'hybrid_points_boxes', 'hybrid_scribbles_boxes', 'hybrid_points_scribbles_boxes']:
            targets_device = {k: v.to(device) for k, v in targets.items()}
        elif isinstance(targets, torch.Tensor):
             targets_device = targets.to(device)
        else:
             print(f"Warning: Unexpected target type in training batch {i}: {type(targets)}")
             continue

        optimizer.zero_grad()
        outputs = model(images)

        # Calculate loss based on mode
        try:
            if mode in ['hybrid_tags_points', 'hybrid_points_scribbles', 'hybrid_points_boxes', 'hybrid_scribbles_boxes', 'hybrid_points_scribbles_boxes']:
                loss = loss_fn(outputs, targets_device)
            elif mode == 'tags':
                 cls_output = outputs['classification'] if isinstance(outputs, dict) else outputs
                 loss = loss_fn(cls_output, targets_device.float())
            elif mode in ['points', 'scribbles', 'boxes', 'full']:
                 seg_output = outputs['segmentation'] if isinstance(outputs, dict) else outputs
                 loss = loss_fn(seg_output, targets_device.long())
            else:
                 raise ValueError(f"Unknown mode {mode} for loss calculation")

            if torch.isnan(loss) or torch.isinf(loss):
                 print(f"Warning: NaN or Inf loss encountered in training batch {i}. Skipping backward pass.")
                 continue

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

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
                 print(f"  Train Batch {i+1}/{num_batches}, Current Avg Loss: {total_loss / (i+1):.4f}, Batch Time: {batch_duration:.2f}s, Epoch ETA: {eta_epoch_str}")

        except Exception as e:
            print(f"Error during training batch {i}: {e}")
            continue # Skip batch on error

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    print(f"Training epoch finished. Average Loss: {avg_loss:.4f}")
    #return avg_loss


def validate_one_epoch(model, loader, device, mode, num_classes):
    model.eval()
    total_loss = 0.0
    num_batches = len(loader)

    val_iou = torchmetrics.JaccardIndex(
        task="multiclass", num_classes=num_classes, ignore_index=IGNORE_INDEX, average='none'
        ).to(device)


    val_loss_fn = CrossEntropyLoss(ignore_index=IGNORE_INDEX).to(device)
    print(f"Starting validation epoch...")
    with torch.no_grad():
        for i, (images, _, gt_masks) in enumerate(loader):
            images = images.to(device)
            gt_masks = gt_masks.to(device).long()

            outputs = model(images)

            # --- Loss Calculation ---
            #loss = torch.tensor(0.0, device=device)
            seg_logits_for_loss = None
            
            # Extract segmentation logits consistently
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
                print(f"Warning: Could not find 'segmentation' output in model dictionary during validation batch {i}")

            # --- Metric Calculation ---
            try:
                 seg_logits_for_metric = None
                 if isinstance(outputs, dict):
                     seg_logits_for_metric = outputs.get('segmentation', None)
                 elif isinstance(outputs, torch.Tensor) and mode != 'tags':
                     seg_logits_for_metric = outputs

                 if seg_logits_for_metric is not None:
                      preds = torch.argmax(seg_logits_for_metric, dim=1)
                      val_iou.update(preds, gt_masks)
            except Exception as e:
                 print(f"Error during validation metric calculation in batch {i}: {e}")

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # --- Compute and Log Metrics ---
    pet_iou = 0.0
    try:
        final_iou_per_class = val_iou.compute()
        if num_classes > 1 and len(final_iou_per_class) > 1:
             pet_iou = final_iou_per_class[1].item()
        elif num_classes == 1:
             pet_iou = final_iou_per_class[0].item()
        print(f"Validation epoch finished. Average Loss: {avg_loss:.4f}, Pet IoU: {pet_iou:.4f}")
    except Exception as e:
        print(f"Error computing final validation metrics: {e}")
        print(f"Validation epoch finished. Average Loss: {avg_loss:.4f}, Pet IoU: Calculation Error")

    val_iou.reset()
    return avg_loss, pet_iou


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    num_output_classes = args.num_classes

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
    train_dataset = PetsDataset(args.data_dir, split='train', supervision_mode=args.supervision_mode,
                                weak_label_path=args.weak_label_path, img_size=img_size_tuple, augment=args.augment,
                                num_classes=num_output_classes)
    val_dataset = PetsDataset(args.data_dir, split='val', supervision_mode=args.supervision_mode,
                              weak_label_path=args.weak_label_path, img_size=img_size_tuple, augment=False,
                              num_classes=num_output_classes) # Also pass to val dataset


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True if device != torch.device('cpu') else False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True if device != torch.device('cpu') else False)

    # --- Initialize Model ---
    model_mode = 'segmentation'
    if args.supervision_mode == 'tags':
        model_mode = 'hybrid' # Keep segmentation head for potential eval
    elif args.supervision_mode in ['hybrid_tags_points', 'hybrid_points_scribbles', 'hybrid_points_boxes', 'hybrid_scribbles_boxes', 'hybrid_points_scribbles_boxes']:
        model_mode = 'hybrid'
    num_output_classes = args.num_classes
    print(f"Initializing model with {num_output_classes} output classes for segmentation head.")
    model = EffUnetWrapper(backbone=args.backbone, num_classes=num_output_classes, mode=model_mode)
    model.to(device)

    # --- Define Loss Function ---
    if args.supervision_mode == 'tags':
        # Using CombinedLoss even for tags, assuming model has both heads ('hybrid' mode)
        # If you strictly want only BCE on classification, adjust model mode and loss
        loss_fn = torch.nn.BCEWithLogitsLoss() # Set lambda_seg=0? or use BCE?
        # Alternative if only BCE desired and model mode is classification:
        # loss_fn = torch.nn.BCEWithLogitsLoss()
    elif args.supervision_mode in ['points', 'scribbles']:
        loss_fn = PartialCrossEntropyLoss(ignore_index=IGNORE_INDEX)
    elif args.supervision_mode in ['boxes', 'full']:
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    elif args.supervision_mode in ['hybrid_tags_points', 'hybrid_points_scribbles', 'hybrid_points_boxes', 'hybrid_scribbles_boxes', 'hybrid_points_scribbles_boxes']:
        loss_fn = CombinedLoss(lambda_seg=args.lambda_seg, ignore_index=IGNORE_INDEX)
    else:
        raise ValueError(f"Supervision mode {args.supervision_mode} not implemented for loss")

    # --- Optimizer ---
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)


    # --- Training Loop ---
    best_val_iou = 0.0
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path_base = os.path.join(args.checkpoint_dir, f"{args.run_name}")

    print(f"\nStarting training run: {args.run_name}")
    print(f"Supervision: {args.supervision_mode}, Batch Size: {args.batch_size}, LR: {args.lr}, Epochs: {args.epochs}")

    # ***** ADDED: Time tracking variables *****
    training_start_time = time.time()
    # ****************************************

    for epoch in range(args.epochs):
        # ***** ADDED: Epoch start time *****
        epoch_start_time = time.time()
        # ***********************************

        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")

        try:
            train_one_epoch(model, train_loader, optimizer, loss_fn, device, args.supervision_mode)
            val_loss, val_iou = validate_one_epoch(model, val_loader, device, args.supervision_mode, num_output_classes)
            scheduler.step()

            # ***** SAVE CHECKPOINT Logic (Based on Validation IoU) *****
            if val_iou > best_val_iou:
                best_val_iou = val_iou
                save_path = f"{checkpoint_path_base}_best_iou.pth"
                model.cpu()
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'val_iou': best_val_iou,
                    'args': args
                }, save_path)
                model.to(device)
                print(f"Checkpoint saved: Validation IoU improved to {best_val_iou:.4f} (Loss: {val_loss:.4f}). Saved to {save_path}")
            # ********************************************************

            # Optional: Save latest checkpoint
            if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
                latest_save_path = f"{checkpoint_path_base}_latest.pth"
                model.cpu()
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'val_iou': val_iou,
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


    # ***** ADDED: Final time printout *****
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    print("\n------------------------------------")
    print("Training finished.")
    print(f"Total Training Time: {format_time(total_training_time)}")
    # ************************************
    print(f"Best Validation Pet IoU achieved: {best_val_iou:.4f}")
    print(f"Best model saved to: {checkpoint_path_base}_best_iou.pth (if IoU improved)")
    print(f"Latest model saved to: {checkpoint_path_base}_latest.pth")
    print("\nRECOMMENDATION: Load the '_best_iou.pth' checkpoint and evaluate it on the separate TEST set for final performance.")


if __name__ == '__main__':
    # Make sure to install torchmetrics: pip install torchmetrics
    main()