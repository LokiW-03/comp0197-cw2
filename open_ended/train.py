import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import EffUnetWrapper
from data_utils import PetsDataset, IGNORE_INDEX
from losses import PartialCrossEntropyLoss, CombinedLoss
import numpy as np # Needed for metric calculation maybe

# --- Configuration ---
DEFAULT_DATA_DIR = './data'
DEFAULT_WEAK_LABEL_PATH = './weak_labels/weak_labels_train.pkl'
DEFAULT_CHECKPOINT_DIR = './checkpoints'
DEFAULT_NUM_CLASSES = 1 # Binary Pet vs Background

def setup_arg_parser():
    parser = argparse.ArgumentParser(description='Train WSSS Model on Pets Dataset')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR, help='Dataset directory')
    parser.add_argument('--weak_label_path', type=str, default=DEFAULT_WEAK_LABEL_PATH, help='Path to pre-generated weak labels')
    parser.add_argument('--supervision_mode', type=str, required=True,
                        choices=['full', 'tags', 'points', 'scribbles', 'boxes', 'hybrid_tags_points'],
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

    return parser

def train_one_epoch(model, loader, optimizer, loss_fn, device, mode):
    model.train()
    total_loss = 0.0
    num_batches = len(loader)

    print(f"Starting training epoch...") # Simple progress indicator
    for i, (images, targets, _) in enumerate(loader): # Ignore GT mask during weak training unless needed
        images = images.to(device)

        # Move targets to device based on mode
        if mode in ['hybrid_tags_points', 'hybrid_points_scribbles', 'hybrid_points_boxes', 'hybrid_scribbles_boxes', 'hybrid_points_scribbles_boxes']:
            targets_device = {k: v.to(device) for k, v in targets.items()}
        elif isinstance(targets, torch.Tensor):
             targets_device = targets.to(device)
        # Add handling if targets is unexpected type
        else:
            raise TypeError('Unexpected target type.')

        optimizer.zero_grad()
        outputs = model(images)

        # Calculate loss based on mode
        if mode in ['hybrid_tags_points', 'hybrid_points_scribbles', 'hybrid_points_boxes', 'hybrid_scribbles_boxes', 'hybrid_points_scribbles_boxes']:
            loss = loss_fn(outputs, targets_device) # Combined loss expects dicts
        elif mode == 'tags': # Assumes classification output
             loss = loss_fn(outputs, targets_device.float()) # BCE expects float targets
        elif mode in ['points', 'scribbles', 'boxes', 'full']: # Assume segmentation output
             loss = loss_fn(outputs, targets_device.long()) # CE/PartialCE expects long targets
        else:
             raise ValueError(f"Unknown mode {mode} for loss calculation")

        if torch.isnan(loss) or torch.isinf(loss):
             print(f"Warning: NaN or Inf loss encountered in batch {i}. Skipping backward pass.")
             # Optional: Add more debugging here (print inputs/outputs)
             continue # Skip this batch

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (i + 1) % 50 == 0: # Print progress every 50 batches
             print(f"  Batch {i+1}/{num_batches}, Current Avg Loss: {total_loss / (i+1):.4f}")

    avg_loss = total_loss / num_batches
    print(f"Training epoch finished. Average Loss: {avg_loss:.4f}")
    return avg_loss

def validate_one_epoch(model, loader, loss_fn, device, mode):
    model.eval()
    total_loss = 0.0
    num_batches = len(loader)
    print(f"Starting validation epoch...")
    with torch.no_grad():
        for i, (images, targets, _) in enumerate(loader): # Ignore GT mask here too
            images = images.to(device)

            # Move targets to device
            if mode == 'hybrid_tags_points':
                 targets_device = {k: v.to(device) for k, v in targets.items()}
            elif isinstance(targets, torch.Tensor):
                 targets_device = targets.to(device)

            outputs = model(images)

            # Calculate loss based on mode (using same loss as training for consistency)
            try:
                if mode in ['hybrid_tags_points', 'hybrid_points_scribbles', 'hybrid_points_boxes', 'hybrid_scribbles_boxes', 'hybrid_points_scribbles_boxes']:
                    loss = loss_fn(outputs, targets_device)
                elif mode == 'tags':
                    loss = loss_fn(outputs, targets_device.float())
                elif mode in ['points', 'scribbles', 'boxes', 'full']:
                    loss = loss_fn(outputs, targets_device.long())
                else:
                    raise ValueError(f"Unknown mode {mode} for loss calculation")


                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                else:
                    print(f"Warning: NaN/Inf validation loss in batch {i}")


            except Exception as e:
                 print(f"Error during validation loss calculation in batch {i}: {e}")
                 # Continue validation if one batch fails

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    print(f"Validation epoch finished. Average Loss: {avg_loss:.4f}")
    return avg_loss


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): # Check for MPS availability
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # --- Create Datasets and Dataloaders ---
    img_size_tuple = (args.img_size, args.img_size)
    train_dataset = PetsDataset(args.data_dir, split='train', supervision_mode=args.supervision_mode,
                                weak_label_path=args.weak_label_path, img_size=img_size_tuple, augment=args.augment)
    val_dataset = PetsDataset(args.data_dir, split='val', supervision_mode=args.supervision_mode, # Val uses same 'mode' for consistency if needed, but loads GT
                              weak_label_path=args.weak_label_path, img_size=img_size_tuple, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # --- Initialize Model ---
    if args.supervision_mode == 'tags':
        model_mode = 'classification'
    elif args.supervision_mode in ['hybrid_tags_points', 'hybrid_points_scribbles', 'hybrid_points_boxes', 'hybrid_scribbles_boxes', 'hybrid_points_scribbles_boxes']:
        model_mode = 'hybrid'
    else:
        model_mode = 'segmentation' # Default

    # +1 for background class if using standard CE loss expecting class indices 0...N
    # If binary (num_classes=1) with BCE or Dice, keep num_classes=1
    # Let's use standard CE, so need 2 classes: 0=background, 1=pet
    num_output_classes = 2 # Background + Pet
    model = EffUnetWrapper(backbone=args.backbone, num_classes=num_output_classes, mode=model_mode)
    model.to(device)

    # --- Define Loss Function ---
    if args.supervision_mode == 'tags':
        # Assumes binary presence/absence for each class (num_classes should match)
        loss_fn = torch.nn.BCEWithLogitsLoss()
    elif args.supervision_mode in ['points', 'scribbles']:
        loss_fn = PartialCrossEntropyLoss(ignore_index=IGNORE_INDEX)
    elif args.supervision_mode in ['boxes', 'full']:
        # For boxes pseudo-mask and full GT mask
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    elif args.supervision_mode in ['hybrid_tags_points', 'hybrid_points_scribbles', 'hybrid_points_boxes', 'hybrid_scribbles_boxes', 'hybrid_points_scribbles_boxes']:
        loss_fn = CombinedLoss(lambda_seg=args.lambda_seg, ignore_index=IGNORE_INDEX)
    else:
        raise ValueError(f"Supervision mode {args.supervision_mode} not implemented for loss")

    # --- Optimizer ---
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    # Optional: Learning rate scheduler (e.g., CosineAnnealingLR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)


    # --- Training Loop ---
    best_val_loss = float('inf')
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path_base = os.path.join(args.checkpoint_dir, f"{args.run_name}")

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        try:
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, args.supervision_mode)
            val_loss = validate_one_epoch(model, val_loader, loss_fn, device, args.supervision_mode)

            scheduler.step() # Step the scheduler each epoch

            # Save checkpoint if validation loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = f"{checkpoint_path_base}_best.pth"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': best_val_loss,
                    'args': args # Save config too
                }, save_path)
                print(f"Checkpoint saved with val_loss: {best_val_loss:.4f} to {save_path}")
            
            # Optional: Save latest checkpoint every N epochs
            if (epoch + 1) % 10 == 0:
                latest_save_path = f"{checkpoint_path_base}_latest.pth"
                model.cpu() # Move to CPU for saving
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss, # Save current val loss for latest
                    'args': args
                }, latest_save_path)
                model.to(device) # Move back to device
                print(f"Latest checkpoint saved to {latest_save_path}")
                
        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Exiting.")
            break # Exit loop gracefully
        
        except Exception as e:
            print(f"\nAn error occurred during epoch {epoch+1}: {e}")
            break


    print("Training finished.")


if __name__ == '__main__':
    main()