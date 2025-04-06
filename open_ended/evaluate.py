import os
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from model import EffUnetWrapper
from data_utils import PetsDataset, IGNORE_INDEX
from tqdm import tqdm # Using tqdm here for convenience, replace with print if needed

DEFAULT_DATA_DIR = './data'


# Simple IoU calculation
def calculate_iou(pred, target, num_classes, ignore_index=IGNORE_INDEX):
    """Calculates IoU for each class."""
    pred = pred.view(-1)
    target = target.view(-1)

    # Remove ignored pixels
    valid = (target != ignore_index)
    pred = pred[valid]
    target = target[valid]

    iou_per_class = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds[target_inds]).long().sum().item() # Cast to long before sum
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        if union == 0:
            # If there is no GT or prediction, IoU is technically undefined or 1 if intersection is also 0.
            # Common practice: score 1 if intersection is 0, else 0. Or NaN. Let's use NaN and handle later.
             iou = float('nan')
        else:
            iou = float(intersection) / union
        iou_per_class.append(iou)

    return np.array(iou_per_class)


def evaluate_model(model, loader, device, num_classes, supervision_mode):
    model.eval()
    total_iou = np.zeros(num_classes)
    images_processed = 0

    print("Starting evaluation...")
    with torch.no_grad():
        # Wrap loader with tqdm or use simple print
        # for images, _, gt_masks in tqdm(loader, desc="Evaluating"):
        for i, (images, _, gt_masks) in enumerate(loader): # Always use GT mask for evaluation
            images = images.to(device)
            gt_masks = gt_masks.to(device) # Shape (B, H, W)

            outputs = model(images)

            # Get segmentation prediction based on mode
            if supervision_mode == 'tags':
                # Need to generate CAMs or handle classification output appropriately
                # Simplification: Assume evaluate.py is only run for segmentation modes
                # Or implement CAM generation here if needed.
                # For now, raise error if trying to evaluate tags mode directly for segmentation
                raise NotImplementedError("Evaluation for 'tags' mode requires CAM generation logic.")
            elif supervision_mode in ['hybrid_tags_points', 'hybrid_points_scribbles', 'hybrid_points_boxes', 'hybrid_scribbles_boxes', 'hybrid_points_scribbles_boxes']:
                 seg_logits = outputs['segmentation'] # Shape (B, C, H, W)
            else: # 'full', 'points', 'scribbles', 'boxes'
                 seg_logits = outputs # Shape (B, C, H, W)

            # Convert logits to predictions
            # For binary (num_classes=2), argmax gives 0 or 1
            predictions = torch.argmax(seg_logits, dim=1) # Shape (B, H, W)

            # Calculate IoU for each image in the batch
            for i in range(images.size(0)):
                iou_scores = calculate_iou(predictions[i], gt_masks[i], num_classes, IGNORE_INDEX)
                # Accumulate IoU scores, handling NaNs
                # If a class wasn't present/predicted (NaN), don't include it in the sum for *that image*
                valid_iou_mask = ~np.isnan(iou_scores)
                if np.any(valid_iou_mask): # Only add if there are valid classes
                    total_iou[valid_iou_mask] += iou_scores[valid_iou_mask]
                    # How to count images for averaging? Count if *any* class had valid IoU?
                    # Let's increment count per image processed that yielded *some* valid IoU score
                    # This might slightly bias mIoU if some images have only NaN classes.
                    # Alternative: track counts per class.
            images_processed += images.size(0) # Count total images processed for averaging later if needed

            if (i + 1) % 20 == 0: # Print progress
                 print(f"  Evaluated batch {i+1}/{len(loader)}")


    # Calculate mean IoU across all images/batches
    # This simple average might be skewed if class distribution is uneven.
    # A better approach is macro-average: average per-class IoUs calculated over the whole dataset.
    # Let's refine the calculation:
    # Need intersection and union sums per class over the *entire dataset*

    # --- Recalculate mIoU properly ---
    print("Recalculating mIoU over full dataset...")
    intersection_per_class = np.zeros(num_classes)
    union_per_class = np.zeros(num_classes)
    model.eval()
    with torch.no_grad():
        for images, _, gt_masks in tqdm(loader, desc="Final mIoU Calc"): # Use tqdm here for clarity
            images = images.to(device)
            gt_masks = gt_masks.to(device)
            outputs = model(images)
            if supervision_mode in ['hybrid_tags_points', 'hybrid_points_scribbles', 'hybrid_points_boxes', 'hybrid_scribbles_boxes', 'hybrid_points_scribbles_boxes']:
                seg_logits = outputs['segmentation']
            else:
                seg_logits = outputs
            predictions = torch.argmax(seg_logits, dim=1)

            pred_flat = predictions.view(-1)
            gt_flat = gt_masks.view(-1)
            valid = (gt_flat != IGNORE_INDEX)
            pred_valid = pred_flat[valid]
            gt_valid = gt_flat[valid]

            for cls in range(num_classes):
                 pred_inds = (pred_valid == cls)
                 target_inds = (gt_valid == cls)
                 intersection = (pred_inds & target_inds).long().sum().item()
                 union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
                 intersection_per_class[cls] += intersection
                 union_per_class[cls] += union

    iou_per_class = intersection_per_class / (union_per_class + 1e-8) # Add epsilon for stability
    mean_iou = np.nanmean(iou_per_class) # Use nanmean to ignore potential NaN for absent classes

    print("\n--- Evaluation Results ---")
    for cls in range(num_classes):
        print(f"Class {cls} IoU: {iou_per_class[cls]:.4f}")
    print(f"Mean IoU (mIoU): {mean_iou:.4f}")
    print("--------------------------")

    return mean_iou, iou_per_class

def main():
    parser = argparse.ArgumentParser(description='Evaluate WSSS Model')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR, help='Dataset directory')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the trained model checkpoint (.pth file)')
    parser.add_argument('--img_size', type=int, default=256, help='Image size used during training')
    parser.add_argument('--batch_size', type=int, default=16, help='Evaluation batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of dataloader workers')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Load Checkpoint and Config ---
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu') # Load to CPU first
    train_args = checkpoint['args'] # Get saved training args
    print("Loaded training configuration:", train_args)

    # --- Initialize Model ---
    model_mode = 'segmentation' # Default
    if train_args.supervision_mode == 'tags':
        model_mode = 'classification'
    elif train_args.supervision_mode in ['hybrid_tags_points', 'hybrid_points_scribbles', 'hybrid_points_boxes', 'hybrid_scribbles_boxes', 'hybrid_points_scribbles_boxes']:
        model_mode = 'hybrid'

    num_output_classes = 2 # Background + Pet (consistent with training)
    model = EffUnetWrapper(backbone=train_args.backbone, num_classes=num_output_classes, mode=model_mode)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'N/A')} with val_loss {checkpoint.get('val_loss', 'N/A'):.4f}")

    # --- Create Test Dataloader ---
    img_size_tuple = (args.img_size, args.img_size)
    # Important: Use 'full' mode for dataset loader during evaluation to get GT masks
    test_dataset = PetsDataset(args.data_dir, split='test', supervision_mode='full',
                               img_size=img_size_tuple, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # --- Evaluate ---
    # Pass the original supervision mode from training args to handle model output correctly
    evaluate_model(model, test_loader, device, num_output_classes, train_args.supervision_mode)

if __name__ == '__main__':
    main()