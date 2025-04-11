#lossess.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from pkg_resources import require
from torch.nn import CrossEntropyLoss


class PartialCrossEntropyLoss(nn.Module):
    """
    Calculates Cross Entropy Loss only on pixels with valid labels (not ignore_index).
    """
    def __init__(self, ignore_index=255):
        super().__init__()
        # Use reduction='none' to apply ignore_index correctly per pixel
        self.base_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
        self.ignore_index = ignore_index

    def forward(self, input_logits, target_sparse):
        """
        Args:
            input_logits (torch.Tensor): Predicted logits (B, C, H, W).
            target_sparse (torch.Tensor): Target labels (B, H, W), containing class indices
                                          or ignore_index.
        """
        # Calculate loss per pixel
        pixel_losses = self.base_loss(input_logits, target_sparse) # Shape (B, H, W)

        # Create a mask for valid pixels (where target is not ignore_index)
        valid_pixel_mask = (target_sparse != self.ignore_index)

        # Sum loss only over valid pixels
        total_loss = pixel_losses[valid_pixel_mask].sum()

        # Normalize by the number of valid pixels to get mean loss
        num_valid_pixels = valid_pixel_mask.sum()

        if num_valid_pixels > 0:
            mean_loss = total_loss / num_valid_pixels
        else:
            # Avoid division by zero if no valid pixels in batch (should not happen often)
            mean_loss = total_loss # Or return 0.0 * total_loss to keep grad graph
            # Alternatively return torch.tensor(0.0, device=input_logits.device, requires_grad=True)

        return mean_loss


# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

class CombinedLoss(nn.Module):
    """
    Calculates loss for segmentation tasks using various supervision types.
    Handles:
    - Full supervision (standard CrossEntropy).
    - Weak supervision (points, scribbles, boxes) by applying CrossEntropy
      only on labeled pixels provided in a target dictionary.
    - Hybrid supervision by summing weighted losses from multiple weak types.
    """
    def __init__(self, lambda_seg=1.0, ignore_index=255, mode='full'):
        """
        Args:
            lambda_seg (float): Default weight for segmentation loss components.
                                 Can be overridden by specific weights per mode if needed.
            ignore_index (int): Index in the target mask to ignore during loss calculation.
            mode (str): The supervision mode (e.g., 'full', 'points', 'hybrid_points_boxes').
                        Used to determine which keys to expect/process in the target dict.
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.mode = mode
        self.lambda_seg = lambda_seg # Base weight

        # --- Determine which loss components are active based on mode ---
        self.active_modes = []
        if mode == 'full':
            self.active_modes = ['segmentation'] # Expect a single tensor target
        elif mode == 'points':
            self.active_modes = ['points']
        elif mode == 'scribbles':
            self.active_modes = ['scribbles']
        elif mode == 'boxes':
            self.active_modes = ['boxes']
        elif mode == 'hybrid_points_scribbles':
            self.active_modes = ['points', 'scribbles']
        elif mode == 'hybrid_points_boxes':
            self.active_modes = ['points', 'boxes']
        elif mode == 'hybrid_scribbles_boxes':
            self.active_modes = ['scribbles', 'boxes']
        elif mode == 'hybrid_points_scribbles_boxes':
            self.active_modes = ['points', 'scribbles', 'boxes']
        else:
            raise ValueError(f"Unsupported supervision mode for CombinedLoss: {mode}")

        # --- Define the base loss criterion ---
        # We use ONE CrossEntropyLoss instance for all segmentation-based losses.
        # It inherently handles ignoring unlabeled pixels via ignore_index.
        self.segmentation_criterion = CrossEntropyLoss(ignore_index=self.ignore_index)

        # --- Optional: Define weights per component if needed ---
        # For now, use the single lambda_seg for all components.
        self.loss_weights = {key: self.lambda_seg for key in self.active_modes}
        # Example for different weights:
        # self.loss_weights = {'points': 1.0, 'boxes': 0.8}.get(key, self.lambda_seg)

        print(f"Initialized CombinedLoss for mode '{mode}'. Active components: {self.active_modes} with weights {self.loss_weights}")


    def forward(self, outputs, targets):
        """
        Calculates the combined loss.

        Args:
            outputs (torch.Tensor or dict): Model outputs. Expected to contain segmentation logits.
                                             If dict, assumes key 'segmentation'.
            targets (torch.Tensor or dict): Ground truth or weak supervision targets.
                                            - Tensor (B, H, W) for 'full' mode.
                                            - Dict {key: Tensor(B, H, W)} for weak/hybrid modes.

        Returns:
            torch.Tensor: The calculated total loss.
        """
        total_loss = 0.0
        num_loss_components = 0

        # --- Get Segmentation Logits ---
        # Assumes model output is either the logits tensor directly
        # or a dict containing logits under the key 'segmentation'.
        if isinstance(outputs, dict):
            seg_logits = outputs.get('segmentation')
            if seg_logits is None:
                raise ValueError("Model output dictionary does not contain 'segmentation' key.")
        elif isinstance(outputs, torch.Tensor):
            seg_logits = outputs
        else:
            raise TypeError(f"Unsupported model output type: {type(outputs)}")

        # --- Calculate Loss based on Target Type ---
        if isinstance(targets, dict):
            # --- Weak / Hybrid Supervision Mode ---
            for key in self.active_modes:
                if key in targets:
                    key_targets = targets[key] # Shape (B, H, W), long with ignore_index

                    # Ensure target is LongTensor for CrossEntropyLoss
                    if key_targets.dtype != torch.long:
                        key_targets = key_targets.long()

                    # --- Calculate loss for this component ---
                    # We use the SAME criterion for points, scribbles, boxes, etc.
                    # because the target mask itself defines which pixels contribute.
                    try:
                        # Ensure logits and targets have compatible shapes B,C,H,W and B,H,W
                        if seg_logits.ndim != 4 or key_targets.ndim != 3:
                             raise ValueError(f"Dimension mismatch: Logits {seg_logits.shape}, Targets {key_targets.shape}")
                        if seg_logits.shape[0] != key_targets.shape[0] or \
                           seg_logits.shape[2:] != key_targets.shape[1:]:
                             raise ValueError(f"Shape mismatch: Logits {seg_logits.shape}, Targets {key_targets.shape}")

                        # Calculate the standard CrossEntropyLoss using the specific weak mask
                        loss_component = self.segmentation_criterion(seg_logits, key_targets)

                        # Check for NaN/Inf loss
                        if not torch.isnan(loss_component) and not torch.isinf(loss_component):
                            weight = self.loss_weights.get(key, self.lambda_seg) # Get weight for this key
                            total_loss += loss_component * weight
                            num_loss_components += 1
                            # print(f"  Loss component '{key}': {loss_component.item():.4f} (Weight: {weight})") # Debug print
                        else:
                            print(f"Warning: NaN or Inf loss detected for component '{key}'. Skipping.")

                    except Exception as e:
                         print(f"Error calculating loss for component '{key}': {e}")
                         print(f"Logits shape: {seg_logits.shape}, dtype: {seg_logits.dtype}")
                         print(f"Targets ('{key}') shape: {key_targets.shape}, dtype: {key_targets.dtype}")
                         # Optionally re-raise or continue
                         continue

            if num_loss_components == 0 and targets: # Dict wasn't empty, but no keys matched/valid loss calculated
                print(f"Warning: No valid loss components were calculated for non-empty target dictionary in mode '{self.mode}'. Target keys: {list(targets.keys())}. Returning zero loss.")
                # Return a zero loss tensor that requires gradients if necessary
                return torch.tensor(0.0, device=seg_logits.device, requires_grad=True)


        elif isinstance(targets, torch.Tensor):
            # --- Full Supervision Mode ---
            # Ensure target is LongTensor
            if targets.dtype != torch.long:
                targets = targets.long()

            try:
                 if seg_logits.ndim != 4 or targets.ndim != 3:
                      raise ValueError(f"Dimension mismatch: Logits {seg_logits.shape}, Targets {targets.shape}")
                 if seg_logits.shape[0] != targets.shape[0] or \
                    seg_logits.shape[2:] != targets.shape[1:]:
                      raise ValueError(f"Shape mismatch: Logits {seg_logits.shape}, Targets {targets.shape}")

                 loss_component = self.segmentation_criterion(seg_logits, targets)

                 if not torch.isnan(loss_component) and not torch.isinf(loss_component):
                      total_loss = loss_component * self.loss_weights.get('segmentation', self.lambda_seg) # Use 'segmentation' weight or default
                      num_loss_components += 1
                 else:
                      print("Warning: NaN or Inf loss detected for full supervision loss. Returning zero loss.")
                      return torch.tensor(0.0, device=seg_logits.device, requires_grad=True)

            except Exception as e:
                 print(f"Error calculating loss for full supervision: {e}")
                 print(f"Logits shape: {seg_logits.shape}, dtype: {seg_logits.dtype}")
                 print(f"Targets shape: {targets.shape}, dtype: {targets.dtype}")
                 # Return zero loss tensor
                 return torch.tensor(0.0, device=seg_logits.device, requires_grad=True)

        else:
            raise TypeError(f"Unsupported target type in CombinedLoss: {type(targets)}")

        # Avoid division by zero if no components were added
        # Although previous checks should return 0 loss already in that case.
        # if num_loss_components > 0:
        #     return total_loss / num_loss_components # Average loss ? Or just sum? Usually SUM.
        # else:
        #     return torch.tensor(0.0, device=seg_logits.device, requires_grad=True)

        return total_loss # Return the weighted sum