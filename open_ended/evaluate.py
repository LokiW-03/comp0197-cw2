# evaluate.py

import os
import argparse
import torch
import torch.nn as nn
import torchmetrics
import logging

from torch.utils.data import DataLoader
from open_ended.data_utils import PetsDataset, IGNORE_INDEX
from model.baseline_segnet import SegNet

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_segnet_checkpoint(checkpoint_path, num_classes=2, device='cpu'):
    """
    Loads a SegNet model from a given checkpoint path, attempting to handle
    multiple checkpoint formats (similar to your visualization script).
    """
    import pickle

    print(f"Loading checkpoint: {checkpoint_path}") # Keep this print for clarity during run

    # Instantiate your SegNet with the expected # of classes
    model = SegNet(in_channels=3, output_num_classes=num_classes)

    # --- Attempt to load checkpoint using 'weights_only' first ---
    try:
        try:
            # Use weights_only=None to let torchauto-detect if safe
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=None)
            # logging.info("Attempted load with weights_only=True.") # Commented out as weights_only=None handles this better
        except (RuntimeError, pickle.UnpicklingError, AttributeError, EOFError) as e_safe:
            logging.warning(f"Could not load {checkpoint_path} safely ({e_safe}). "
                            f"Falling back to legacy loading (weights_only=False).")
            # Explicitly use weights_only=False as fallback
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            logging.warning("Warning: loading with weights_only=False can execute arbitrary code if the file is untrusted.")
        except Exception as e_other:
             # Catch other potential loading errors
             logging.error(f"An unexpected error occurred during torch.load: {e_other}")
             raise e_other

    except FileNotFoundError:
        logging.error(f"Checkpoint file not found at {checkpoint_path}")
        raise
    except Exception as e:
        logging.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
        raise e

    # --- Extract the actual model state_dict ---
    state_dict = None
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            logging.info("Checkpoint contains 'model_state_dict' key.")
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            logging.info("Checkpoint contains 'state_dict' key.")
            state_dict = checkpoint["state_dict"]
        elif all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in checkpoint.items()):
            logging.info("Checkpoint looks like it is a raw state_dict (all keys map to Tensors).")
            state_dict = checkpoint
        else:
             msg_keys = list(checkpoint.keys())
             logging.error(f"Unexpected checkpoint dictionary structure. Keys: {msg_keys}")
             raise TypeError("Could not interpret checkpoint format. Expected dict with 'model_state_dict'/'state_dict' or raw state_dict.")
    elif isinstance(checkpoint, torch.nn.Module): # Sometimes the whole model is saved
         logging.info("Checkpoint appears to be a saved torch.nn.Module object. Extracting state_dict.")
         state_dict = checkpoint.state_dict()
    elif isinstance(checkpoint, tuple) and len(checkpoint) > 0 and isinstance(checkpoint[0], dict):
        # Handle cases where checkpoint might be a tuple (e.g., from specific saving patterns)
        logging.warning("Checkpoint is a tuple, attempting to extract state_dict from the first element.")
        if "model_state_dict" in checkpoint[0]:
            state_dict = checkpoint[0]["model_state_dict"]
        elif "state_dict" in checkpoint[0]:
            state_dict = checkpoint[0]["state_dict"]
        else:
            logging.error("Could not find state_dict in the first element of the checkpoint tuple.")
            raise TypeError("Unsupported checkpoint tuple structure.")
    else:
        logging.error(f"Unexpected checkpoint type: {type(checkpoint)}")
        raise TypeError("Could not interpret checkpoint format. Expected a dictionary or nn.Module.")

    if state_dict is None:
        raise ValueError("Could not extract state_dict from the checkpoint.")

    # --- Remove any known prefixes from keys (e.g., "module.", "seg_model.") ---
    adapted_state_dict = {}
    prefix_to_remove = ""
    # Check for common prefixes only once
    first_key = next(iter(state_dict.keys()), None)
    if first_key:
        if first_key.startswith("module."):
            prefix_to_remove = "module."
            logging.info(f"Removing prefix '{prefix_to_remove}' from state_dict keys.")
        elif first_key.startswith("seg_model."):
            prefix_to_remove = "seg_model."
            logging.info(f"Removing prefix '{prefix_to_remove}' from state_dict keys.")
        # Add more prefixes if needed:
        # elif first_key.startswith("another_prefix."):
        #     prefix_to_remove = "another_prefix."
        #     logging.info(f"Removing prefix '{prefix_to_remove}' from state_dict keys.")

    if prefix_to_remove:
        for k, v in state_dict.items():
            if k.startswith(prefix_to_remove):
                adapted_state_dict[k[len(prefix_to_remove):]] = v
            else:
                # Keep keys that didn't have the prefix (might indicate an issue?)
                adapted_state_dict[k] = v
                logging.warning(f"Key '{k}' did not start with expected prefix '{prefix_to_remove}'. Kept as is.")
    else:
        adapted_state_dict = state_dict # No prefix found, use original state_dict

    # --- Load into the model with strict=False, to ignore non-matching keys ---
    try:
        missing_keys, unexpected_keys = model.load_state_dict(adapted_state_dict, strict=False)
        if missing_keys:
            logging.warning(f"Missing keys when loading state_dict into the model structure: {missing_keys}")
        if unexpected_keys:
            logging.warning(f"Unexpected keys found in state_dict (were ignored): {unexpected_keys}")
        if not missing_keys and not unexpected_keys:
            logging.info("State_dict loaded successfully with no missing or unexpected keys.")
    except Exception as e:
        logging.error(f"Error loading state_dict into model: {e}")
        # Optionally re-raise or attempt strict=True loading for debugging
        # model.load_state_dict(adapted_state_dict, strict=True)
        raise e

    model.to(device)
    model.eval()

    return model


def evaluate_model_on_test(model, test_loader, device, num_classes):
    """
    Runs inference on the entire test set and computes:
      - test average loss
      - test accuracy (macro)
      - test IoU (per-class)

    Returns:
      (test_loss, test_accuracy, test_iou_list)
        test_loss: float
        test_accuracy: float
        test_iou_list: torch.Tensor, containing IoU for each class.
    """
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

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
        average='none'  # Get per-class IoU
    ).to(device)

    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, (images, _, gt_masks) in enumerate(test_loader):
            images = images.to(device)
            gt_masks = gt_masks.to(device).long() # Ensure masks are LongTensor

            outputs = model(images)
            loss = criterion(outputs, gt_masks)

            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
            else:
                 logging.warning(f"NaN or Inf loss detected in batch {batch_idx}. Skipping loss accumulation for this batch.")

            num_batches += 1

            preds = torch.argmax(outputs, dim=1)
            # Ensure predictions and ground truth have compatible shapes and types
            accuracy_metric.update(preds, gt_masks)
            iou_metric.update(preds, gt_masks)

    # Compute final metrics
    avg_loss = total_loss / max(num_batches, 1) # Avoid division by zero
    accuracy = accuracy_metric.compute().item()
    iou_per_class = iou_metric.compute() # Returns a tensor

    # Clean up metric states
    accuracy_metric.reset()
    iou_metric.reset()

    return avg_loss, accuracy, iou_per_class


def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple SegNet models on the Pets test set.")
    parser.add_argument('--data_dir', type=str, default='./data',
                        help="Root directory of the Oxford Pets dataset.")
    # --- MODIFICATION: Add argument for model paths ---
    parser.add_argument('--model_paths', type=str, nargs='+', required=True,
                        help="List of paths to the model checkpoint files (.pth) to evaluate.")
    # --- END MODIFICATION ---
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size for test DataLoader.")
    parser.add_argument('--num_workers', type=int, default=2,
                        help="Number of workers for DataLoader.")
    parser.add_argument('--num_classes', type=int, default=2,
                        help="Number of output classes (e.g. 2 for BG + Pet).")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device to use (cuda, cpu, mps).")
    parser.add_argument('--img_height', type=int, default=256, help="Image height for resizing.")
    parser.add_argument('--img_width', type=int, default=256, help="Image width for resizing.")

    args = parser.parse_args()

    # --- Device Setup ---
    if args.device == 'cuda' and not torch.cuda.is_available():
        logging.warning("CUDA selected but not available. Falling back to CPU.")
        args.device = "cpu"
    elif args.device == 'mps' and not torch.backends.mps.is_available():
         logging.warning("MPS selected but not available. Falling back to CPU.")
         args.device = "cpu"

    device = torch.device(args.device)
    logging.info(f"Using device: {device}")

    # --- Create the test dataset ---
    test_dataset = PetsDataset(
        data_dir=args.data_dir,
        split='test',
        supervision_mode='full', # Need ground truth for evaluation
        weak_label_path=None,
        img_size=(args.img_height, args.img_width),
        augment=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda') # Pin memory only if using CUDA
    )
    logging.info(f"Test dataset loaded with {len(test_dataset)} samples.")


    # --- Evaluate each model on the test set ---
    results = {} # Use a dictionary to store results keyed by model path
    # --- MODIFICATION: Use args.model_paths instead of hardcoded list ---
    for model_path in args.model_paths:
        print("-" * 50) # Separator for different models
        if not os.path.isfile(model_path):
            logging.error(f"Model checkpoint not found, skipping: {model_path}")
            continue

        try:
            # Load the model
            segnet_model = load_segnet_checkpoint(
                model_path,
                num_classes=args.num_classes,
                device=device
            )

            # Evaluate the model
            test_loss, test_acc, test_iou_per_class = evaluate_model_on_test(
                segnet_model,
                test_loader,
                device,
                args.num_classes
            )

            # Store and print results
            model_results = {
                'loss': test_loss,
                'accuracy': test_acc,
                'iou_per_class': test_iou_per_class.cpu().numpy().tolist() # Convert tensor to list for easier handling/saving
            }
            results[model_path] = model_results

            print(f"\nResults for Model: {os.path.basename(model_path)}") # Print basename for brevity
            print(f"  Test Loss       : {test_loss:.4f}")
            print(f"  Test Accuracy   : {test_acc:.4f}")
            print(f"  Test IoU per class:")
            for class_idx, class_iou in enumerate(model_results['iou_per_class']):
                 class_name = f"Class {class_idx}"
                 if args.num_classes == 2:
                     class_name = "Background" if class_idx == 0 else "Pet"
                 print(f"    {class_name:<12}: {class_iou:.4f}")

            # Optional: Clear GPU memory if evaluating many large models
            del segnet_model
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        except Exception as e:
            logging.error(f"Failed to evaluate model {model_path}: {e}")
            # Optionally continue to the next model or re-raise the exception
            # raise e # Uncomment this line to stop execution on error
            continue # Continue with the next model path

    print("-" * 50)
    logging.info("Evaluation finished for all provided model checkpoints.")

    # --- Optional: Post-processing or saving results ---
    # You could save the 'results' dictionary to a JSON file,
    # find the best model based on a specific metric, etc.
    # Example: Find model with best Pet IoU (assuming num_classes=2)
    if results and args.num_classes == 2:
        best_pet_iou = -1.0
        best_model_path = None
        for path, metrics in results.items():
            pet_iou = metrics['iou_per_class'][1] # Index 1 is 'Pet'
            if pet_iou > best_pet_iou:
                best_pet_iou = pet_iou
                best_model_path = path

        if best_model_path:
            print("\n--- Best Performing Model (based on Pet IoU) ---")
            print(f"  Path: {os.path.basename(best_model_path)}")
            print(f"  Pet IoU: {best_pet_iou:.4f}")
            print(f"  Accuracy: {results[best_model_path]['accuracy']:.4f}")
            print(f"  Loss: {results[best_model_path]['loss']:.4f}")


if __name__ == "__main__":
    main()