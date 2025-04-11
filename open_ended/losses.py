#lossess.py
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
            mean_loss = total_loss # Or return 0.0 * total_loss to keep grad graph
            # Alternatively return torch.tensor(0.0, device=input_logits.device, requires_grad=True)

        return mean_loss


class CombinedLoss(nn.Module):
    """
    Combines Classification Loss (BCE) and Partial Segmentation Loss (Partial CE).
    """
    def __init__(self, mode, lambda_seg=1.0, ignore_index=255):
        super().__init__()
        # For binary classification (pet present/absent) or multi-label
        self.classification_loss_fn = nn.BCEWithLogitsLoss()
        # For sparse segmentation labels (points)
        self.segmentation_loss_fn = PartialCrossEntropyLoss(ignore_index=ignore_index)
        self.cross_entropy_loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.lambda_seg = lambda_seg # Weight for the segmentation loss
        self.mode = mode
        self.mode_to_key = {
            'points': ['points'],
            'scribbles': ['scribbles'],
            'boxes': ['boxes'],
            'hybrid_points_scribbles': ['scribbles', 'points'],
            'hybrid_points_boxes': ['points', 'boxes'],
            'hybrid_scribbles_boxes': ['scribbles', 'boxes'],
            'hybrid_points_scribbles_boxes': ['points', 'scribbles', 'boxes']
        }
        self.ignore_index = ignore_index

    def forward(self, model_output, targets):
        """
        Args:
            model_output (dict): {'segmentation': logits_seg, 'classification': logits_cls}
            targets (dict): {'tags': target_tags, 'points': target_points_sparse}
        """
        # Classification Loss (Tags)
        #cls_logits = model_output['classification'] # Shape (B, C)
        #tag_targets = targets['tags'] # Shape (B, C), float for BCE
        #loss_cls = self.classification_loss_fn(cls_logits, tag_targets)

        required_keys = self.mode_to_key.get(self.mode, [])
        loss_list = []

        for key in required_keys:
            if key in ['points', 'scribbles']:
                # Segmentation Loss (Points)
                seg_logits = model_output # Shape (B, C, H, W)
                key_targets = targets[key] # Shape (B, H, W), long with ignore_index
                loss_list.append(self.segmentation_loss_fn(seg_logits, key_targets))
            elif key == "boxes":
                seg_logits = model_output
                key_targets = targets[key]
                loss_list.append(self.cross_entropy_loss_fn(seg_logits, key_targets))
            else:
                raise ValueError("Invalid key.")

        # Combine losses
        total_loss = sum([loss * self.lambda_seg for loss in loss_list])

        # Optional: return individual losses for logging
        return total_loss #, loss_cls, loss_seg