#segnet_wrapper.py
import torch.nn as nn
from model.baseline_segnet import SegNet


class SegNetWrapper(nn.Module):
    """
    Wrapper that uses the SegNet architecture internally, but allows switching
    between 'single' and 'hybrid' output formats:

      * 'single' -> returns only the segmentation tensor
      * 'hybrid' -> returns a dictionary: {'segmentation': seg_logits}

    No classification head is used here.
    """
    def __init__(self, num_classes=2, mode='single'):
        """
        Args:
            num_classes (int): number of segmentation classes (e.g. 2 => BG + Foreground).
            mode (str): 'single' or 'hybrid'.
        """
        super().__init__()
        self.mode = mode
        self.num_classes = num_classes
        # Use the SegNet model you defined above
        self.seg_model = SegNet(output_num_classes=num_classes)

    def forward(self, x):
        seg_logits = self.seg_model(x)  # [B, C, H, W]

        if self.mode == 'single':
            return seg_logits
        elif self.mode == 'hybrid':
            return seg_logits
        else:
            raise ValueError(f"Unknown mode {self.mode}, must be 'single' or 'hybrid'.")
