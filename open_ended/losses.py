# I acknowledge the use of ChatGPT (version GPT-4o, OpenAI, https://chatgpt.com/) for assistance in debugging and
# writing docstrings.

#lossess.py
import torch
import torch.nn as nn


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
            mean_loss = total_loss

        return mean_loss


class CombinedLoss(nn.Module):
    """
    Combines segmentation losses with uncertainty-based adaptive weighting.
    Supports boxes (full CE), scribbles/points (partial CE).
    """
    def __init__(self, mode, ignore_index=255):
        super().__init__()
        self.segmentation_loss_fn = PartialCrossEntropyLoss(ignore_index=ignore_index)
        self.cross_entropy_loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

        # Define valid supervision types per mode
        self.mode_to_key = {
            'points': ['points'],
            'scribbles': ['scribbles'],
            'boxes': ['boxes'],
            'hybrid_points_scribbles': ['scribbles', 'points'],
            'hybrid_points_boxes': ['points', 'boxes'],
            'hybrid_scribbles_boxes': ['scribbles', 'boxes'],
            'hybrid_points_scribbles_boxes': ['points', 'scribbles', 'boxes']
        }
        self.required_keys = self.mode_to_key[mode]
        self.ignore_index = ignore_index

        # Initialize learnable log-variance parameters
        for key in self.required_keys:
            self.register_parameter(f'log_var_{key}', nn.Parameter(torch.zeros(1)))

    def forward(self, model_output, targets):
        loss_dict = {}
        total_loss = 0.0

        # Compute individual losses
        for key in self.required_keys:
            if key in ['points', 'scribbles']:
                loss = self.segmentation_loss_fn(model_output, targets[key])
            elif key == 'boxes':
                loss = self.cross_entropy_loss_fn(model_output, targets[key])
            loss_dict[key] = loss

        # Apply uncertainty-based weighting (Kendall et al.)
        for key in self.required_keys:
            log_var = getattr(self, f'log_var_{key}')
            loss = loss_dict[key]

            # L = 1/(2σ²)*loss + log(σ) = 1/(2exp(log_var)) * loss + 0.5*log_var
            precision = torch.exp(-log_var)
            weighted_loss = 0.5 * precision * loss + 0.5 * log_var
            total_loss += weighted_loss

        return total_loss