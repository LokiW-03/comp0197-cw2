# I acknowledge the use of ChatGPT (version GPT-4o, OpenAI, https://chatgpt.com/) for assistance in debugging and
# writing docstrings.

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Multi-class Dice Loss for semantic segmentation tasks.

    Args:
        eps (float): Small constant to avoid division by zero.
        ignore_index (int or None): If set, pixels with this label index will be ignored
                                    in the loss calculation. (Advanced usage.)
    """
    def __init__(self, eps=1e-6, ignore_index=None):
        super(DiceLoss, self).__init__()
        self.eps = eps
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        """
        Args:
            logits (torch.Tensor): Network outputs of shape (N, C, H, W).
            target (torch.Tensor): Ground-truth labels of shape (N, H, W), where
                                   each value is in [0, C-1].
        Returns:
            torch.Tensor: Scalar Dice loss.
        """
        # 1) Convert logits -> probabilities with softmax
        probs = torch.softmax(logits, dim=1)  # (N, C, H, W)

        # 2) One-hot encode the ground truth
        #    result shape => (N, C, H, W)
        num_classes = logits.shape[1]
        target_one_hot = F.one_hot(target, num_classes=num_classes)  # (N, H, W, C)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # => (N, C, H, W)

        # 3) Optional: create a mask to ignore a specific label (if ignore_index is set)
        if self.ignore_index is not None:
            # Mask out pixels where target == ignore_index
            # => shape: (N, 1, H, W), True where not ignoring
            valid_mask = (target != self.ignore_index).unsqueeze(1)
            # We'll apply this mask to both predictions and target
            probs = probs * valid_mask
            target_one_hot = target_one_hot * valid_mask

        # 4) Flatten spatial dims => (N, C, H*W)
        probs_flat = probs.view(probs.size(0), probs.size(1), -1)
        target_flat = target_one_hot.view(target_one_hot.size(0), target_one_hot.size(1), -1)

        # 5) Compute intersection and denominator
        intersection = (probs_flat * target_flat).sum(dim=2)  # (N, C)
        denominator = (probs_flat + target_flat).sum(dim=2) + self.eps  # (N, C)

        # 6) Dice coefficient per class (N, C)
        dice_per_class = 2.0 * intersection / denominator

        # 7) Average Dice across classes => (N,)
        dice_per_sample = dice_per_class.mean(dim=1)

        # 8) Final average Dice across the batch => scalar
        mean_dice = dice_per_sample.mean()

        # Return Dice loss = 1 - mean(Dice)
        return 1.0 - mean_dice


class CombinedCELDiceLoss(nn.Module):
    def __init__(self, dice_weight=1.0):
        """
        dice_weight: scale factor for the Dice component relative to CE.
                     If dice_weight=1.0, total_loss = CE + Dice.
        """
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
        self.dice_weight = dice_weight
        self.dice_loss = DiceLoss()

    def forward(self, logits, labels):
        """
        logits: (N, C, H, W) - raw output from segmentation model
        labels: (N, H, W) - ground-truth class indices
        """
        # 1) Standard CrossEntropy
        ce_val = self.ce_loss(logits, labels)

        # 2) Dice loss
        dice_val = self.dice_loss(logits, labels)

        # 3) Weighted sum
        loss_val = ce_val + self.dice_weight * dice_val
        return loss_val
