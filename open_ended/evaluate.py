# evaluate.py

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchmetrics

# Import your existing modules/dataset/models
from open_ended.data_utils import PetsDataset, IGNORE_INDEX
from model.baseline_segnet import SegNet  # Make sure this path matches your repo structure

def load_segnet_checkpoint(checkpoint_path, num_classes=2, device='cpu'):
    """
    Loads a SegNet model from a given checkpoint path, attempting to handle 
    multiple checkpoint formats (similar to your visualization script).
    """
    import pickle
    import logging

    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Instantiate your SegNet with the expected # of classes
    model = SegNet(in_channels=3, output_num_classes=num_classes)
    
    # --- Attempt to load checkpoint using 'weights_only' first ---
    try:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            logging.info("Attempted load with weights_only=True.")
        except (RuntimeError, pickle.UnpicklingError, AttributeError) as e_safe:
            logging.warning(f"Could not load {checkpoint_path} with weights_only=True ({e_safe}). "
                            f"Falling back to weights_only=False.")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            logging.warning("Warning: loading with weights_only=False can execute arbitrary code if the file is untrusted.")
    except Exception as e:
        logging.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
        raise e

    # --- Extract the actual model state_dict ---
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # E.g. a typical training checkpoint with 'model_state_dict' key
        logging.info("Checkpoint contains 'model_state_dict' key.")
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        # E.g. a training checkpoint that used 'state_dict'
        logging.info("Checkpoint contains 'state_dict' key.")
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and all(
        isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in checkpoint.items()
    ):
        # Possibly the checkpoint is just the raw state_dict already
        logging.info("Checkpoint looks like it is a raw state_dict (all keys map to Tensors).")
        state_dict = checkpoint
    else:
        msg_keys = list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint)
        logging.error(f"Unexpected checkpoint structure. Keys/Type: {msg_keys}")
        raise TypeError("Could not interpret checkpoint format. "
                        "Expected 'model_state_dict' or 'state_dict' dict, "
                        "or a raw state_dict (key->Tensor).")

    # --- Remove any known prefixes from keys (e.g., "module.", "seg_model.") ---
    adapted_state_dict = {}
    for k, v in state_dict.items():
        new_key = k
        if k.startswith("module."):              # from multi-GPU training
            new_key = k[len("module."):]
        elif k.startswith("seg_model."):         # from a wrapper
            new_key = k[len("seg_model."):]
        adapted_state_dict[new_key] = v

    # --- Load into the model with strict=False, to ignore non-matching keys ---
    missing_keys, unexpected_keys = model.load_state_dict(adapted_state_dict, strict=False)
    if missing_keys:
        logging.warning(f"Missing keys in the model structure: {missing_keys}")
    if unexpected_keys:
        logging.warning(f"Unexpected keys in state_dict (ignored): {unexpected_keys}")

    model.to(device)
    model.eval()
    
    return model


def evaluate_model_on_test(model, test_loader, device, num_classes):
    """
    Runs inference on the entire test set and computes:
      - test accuracy (macro)
      - test IoU (per-class or just for 'Pet' class)
    
    Returns:
      (test_accuracy, test_iou_list) 
        test_accuracy: float
        test_iou_list: torch.Tensor or float, containing IoU for each class or just class 1.
    """
    # We can optionally track the average CE loss for information:
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    # Set up metrics
    # For a 2-class problem [Background=0, Pet=1], we can do:
    accuracy_metric = torchmetrics.Accuracy(
        task="multiclass",
        num_classes=num_classes,
        ignore_index=IGNORE_INDEX,
        average='macro'
    ).to(device)

    iou_metric = torchmetrics.JaccardIndex(
        task="multiclass",
        num_classes=num_classes,
        ignore_index=IGNORE_INDEX,
        average='none'  # We can get per-class IoU (index 0 = BG, 1 = Pet)
    ).to(device)

    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, (images, _, gt_masks) in enumerate(test_loader):
            images = images.to(device)
            gt_masks = gt_masks.to(device).long()

            outputs = model(images)  # shape: (B, num_classes, H, W)
            loss = criterion(outputs, gt_masks)
            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
            num_batches += 1

            preds = torch.argmax(outputs, dim=1)  # shape: (B, H, W)
            accuracy_metric.update(preds, gt_masks)
            iou_metric.update(preds, gt_masks)

    # Compute final metrics
    avg_loss = total_loss / max(num_batches, 1)
    accuracy = accuracy_metric.compute().item()  # returns a scalar
    iou_per_class = iou_metric.compute()         # returns a tensor of size [num_classes]
    
    # Clean up metric states
    accuracy_metric.reset()
    iou_metric.reset()

    return avg_loss, accuracy, iou_per_class


def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple SegNet models on the Pets test set.")
    parser.add_argument('--data_dir', type=str, default='./data',
                        help="Root directory of the Oxford Pets dataset.")
    PROJECT_ROOT = 'open_ended'
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    
    

    MODEL_PATHS = [
        'checkpoints_single/segnet_point_run1_best_acc.pth',
        'checkpoints_single/segnet_scatter_run1_best_acc.pth',
        'checkpoints_single/segnet_boxes_run1_best_acc.pth',
        'checkpoints_hybrid/segnet_hybrid_point_scatter_run1_best_acc.pth',
        'checkpoints_hybrid/segnet_hybrid_point_boxes_run1_best_acc.pth',
        'checkpoints_hybrid/segnet_hybrid_scatter_boxes_run1_best_acc.pth',
        'checkpoints_hybrid/segnet_hybrid_point_scatter_boxes_run1_best_acc.pth',
    ]
    
    
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size for test DataLoader.")
    parser.add_argument('--num_workers', type=int, default=2,
                        help="Number of workers for DataLoader.")
    parser.add_argument('--num_classes', type=int, default=2,
                        help="Number of output classes (e.g. 2 for BG + Pet).")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device to use (cuda, cpu, etc.).")
    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        print("Warning: CUDA not available. Falling back to CPU.")
        args.device = "cpu"

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # --- Create the test dataset ---
    # For test, we load the ground truth so set 'supervision_mode' to 'full'
    # so that __getitem__ returns the actual GT masks for evaluation.
    test_dataset = PetsDataset(
        data_dir=args.data_dir,
        split='test',
        supervision_mode='full',
        weak_label_path=None,   # We don't need weak labels for test
        img_size=(256, 256),    # You can adjust if your models expect a certain size
        augment=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda')
    )

    # --- Evaluate each model on the test set ---
    results = []
    for model_path in MODEL_PATHS:
        if not os.path.isfile(model_path):
            print(f"\n[ERROR] Model checkpoint not found at: {model_path}")
            continue

        segnet_model = load_segnet_checkpoint(model_path, num_classes=args.num_classes, device=device)
        test_loss, test_acc, test_iou_per_class = evaluate_model_on_test(segnet_model, test_loader, device, args.num_classes)

        if args.num_classes == 2:
            # iou_per_class = [IoU_BG, IoU_Pet]
            iou_bg = test_iou_per_class[0].item()
            iou_pet = test_iou_per_class[1].item()
            print(f"\nModel: {model_path}")
            print(f"  Test Loss       : {test_loss:.4f}")
            print(f"  Test Accuracy   : {test_acc:.4f}")
            print(f"  Test IoU (BG)   : {iou_bg:.4f}")
            print(f"  Test IoU (Pet)  : {iou_pet:.4f}")
        else:
            # If you have multiple classes, print them all:
            print(f"\nModel: {model_path}")
            print(f"  Test Loss      : {test_loss:.4f}")
            print(f"  Test Accuracy  : {test_acc:.4f}")
            print(f"  Test IoU per class:")
            for class_idx, class_iou in enumerate(test_iou_per_class):
                print(f"    Class {class_idx}: {class_iou.item():.4f}")

        results.append((model_path, test_loss, test_acc, test_iou_per_class))

    # Optionally, you can add logic here to pick the "best" model across all tested ones,
    # or store the metrics in a file, etc.
    print("\nEvaluation finished for all provided model checkpoint.")


if __name__ == "__main__":
    main()
