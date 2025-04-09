# models/fpn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet34_Weights # Modern weights API
from torchvision.models.feature_extraction import create_feature_extractor
from typing import Optional, Dict

# Keep the ConvReLU helper class
class ConvReLU(nn.Sequential):
    """Standard Conv -> BN (optional) -> ReLU block."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
        use_batchnorm: bool = True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batchnorm,
        )
        relu = nn.ReLU(inplace=False)
        layers = [conv, relu]
        if use_batchnorm:
            layers.insert(1, nn.BatchNorm2d(out_channels))
        super().__init__(*layers)


class FPN(nn.Module):
    """
    Feature Pyramid Network (FPN) for Semantic Segmentation using a
    Torchvision ResNet-34 backbone with pretrained weights.

    Args:
        in_channels (int): Number of input channels (e.g., 3 for RGB images).
            Must be 3 if using standard ImageNet pretrained weights. Defaults to 3.
        classes (int): Number of output segmentation classes. Defaults to 1.
        pretrained (bool): Whether to use ImageNet pretrained weights for the
            ResNet-34 backbone. Defaults to True.
        pyramid_channels (int): Number of channels in the FPN lateral and
            top-down layers. Defaults to 256.
        segmentation_channels (int): Number of channels in the segmentation
            head before the final classification layer. Defaults to 128.
        final_upsampling (int): The factor by which to upsample the final
            logits map. Should typically match the stride of the P2 feature map (4).
            Defaults to 4.
        dropout (float): Dropout probability applied in the segmentation head.
            Defaults to 0.2.
        use_batchnorm (bool): Whether to use BatchNorm in the ConvReLU blocks
            within the FPN layers and segmentation head. Defaults to True.
    """
    def __init__(
        self,
        in_channels: int = 3, # Must be 3 for standard pretrained weights
        classes: int = 1,
        pretrained: bool = True,
        pyramid_channels: int = 256,
        segmentation_channels: int = 128,
        final_upsampling: int = 4,
        dropout: float = 0.2,
        use_batchnorm: bool = True,
    ):
        super().__init__()

        if pretrained and in_channels != 3:
            print("Warning: Pretrained weights require in_channels=3. Setting pretrained=False.")
            pretrained = False

        self.classes = classes
        self.final_upsampling = final_upsampling

        # --- Load Pretrained ResNet-34 Backbone ---
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet34(weights=weights)

        # --- Create Feature Extractor ---
        # Define the nodes (layers) from ResNet-34 to extract features from.
        # These correspond typically to the outputs of stage 1, 2, 3, 4 -> C2, C3, C4, C5 for FPN
        # Common layer names in torchvision ResNet: 'layer1', 'layer2', 'layer3', 'layer4'
        return_nodes = {
            # node_name: user_defined_feature_name
            'layer1': 'c2', # Output of ResNet layer 1 (BasicBlock x3, stride 4) -> 64 channels
            'layer2': 'c3', # Output of ResNet layer 2 (BasicBlock x4, stride 8) -> 128 channels
            'layer3': 'c4', # Output of ResNet layer 3 (BasicBlock x6, stride 16) -> 256 channels
            'layer4': 'c5', # Output of ResNet layer 4 (BasicBlock x3, stride 32) -> 512 channels
        }
        self.encoder = create_feature_extractor(backbone, return_nodes=return_nodes)

        # --- Define Encoder Output Channels ---
        # These are fixed for ResNet-34 stages corresponding to C2, C3, C4, C5
        encoder_channels = [64, 128, 256, 512]

        # --- FPN Layers (Lateral, Smoothing) ---
        self.lateral_c2 = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=1)
        self.lateral_c3 = nn.Conv2d(encoder_channels[1], pyramid_channels, kernel_size=1)
        self.lateral_c4 = nn.Conv2d(encoder_channels[2], pyramid_channels, kernel_size=1)
        self.lateral_c5 = nn.Conv2d(encoder_channels[3], pyramid_channels, kernel_size=1)

        self.smooth_p4 = ConvReLU(pyramid_channels, pyramid_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.smooth_p3 = ConvReLU(pyramid_channels, pyramid_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.smooth_p2 = ConvReLU(pyramid_channels, pyramid_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)

        # --- Segmentation Head ---
        self.seg_head_conv = ConvReLU(
            pyramid_channels * 4, # Concatenated P2, P3, P4, P5
            segmentation_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.dropout = nn.Dropout2d(p=dropout, inplace=False)
        self.final_conv = nn.Conv2d(segmentation_channels, classes, kernel_size=1)

        # --- Initialize ONLY FPN Layers ---
        # The encoder part uses pretrained weights (if loaded) or torchvision's default init.
        self.initialize_fpn_layers()

    def initialize_fpn_layers(self):
        """Initializes weights ONLY for FPN layers and segmentation head."""
        for m in [
            self.lateral_c2, self.lateral_c3, self.lateral_c4, self.lateral_c5,
            self.smooth_p2, self.smooth_p3, self.smooth_p4,
            self.seg_head_conv, self.final_conv
        ]:
            for inner_m in m.modules():
                if isinstance(inner_m, nn.Conv2d):
                    nn.init.kaiming_uniform_(inner_m.weight, mode="fan_in", nonlinearity="relu")
                    if inner_m.bias is not None:
                        nn.init.constant_(inner_m.bias, 0)
                elif isinstance(inner_m, nn.BatchNorm2d):
                    nn.init.constant_(inner_m.weight, 1)
                    nn.init.constant_(inner_m.bias, 0)

    def _upsample_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Upsamples x to size of y and adds them."""
        _, _, H, W = y.size()
        upsampled_x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return upsampled_x + y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the FPN model with Torchvision ResNet encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, H, W).

        Returns:
            torch.Tensor: Output logits tensor of shape (batch_size, classes, H, W).
        """
        # --- Encoder ---
        # features is a Dict[str, torch.Tensor]: {'c2': tensor, 'c3': tensor, ...}
        features: Dict[str, torch.Tensor] = self.encoder(x)
        c2, c3, c4, c5 = features['c2'], features['c3'], features['c4'], features['c5']

        # --- Top-down Pathway & Lateral Connections ---
        p5 = self.lateral_c5(c5)
        p4_merged = self._upsample_add(p5, self.lateral_c4(c4))
        p4 = self.smooth_p4(p4_merged)
        p3_merged = self._upsample_add(p4, self.lateral_c3(c3))
        p3 = self.smooth_p3(p3_merged)
        p2_merged = self._upsample_add(p3, self.lateral_c2(c2))
        p2 = self.smooth_p2(p2_merged)

        # --- Segmentation Head ---
        _, _, h_p2, w_p2 = p2.size()
        p3_upsampled = F.interpolate(p3, size=(h_p2, w_p2), mode='bilinear', align_corners=False)
        p4_upsampled = F.interpolate(p4, size=(h_p2, w_p2), mode='bilinear', align_corners=False)
        p5_upsampled = F.interpolate(p5, size=(h_p2, w_p2), mode='bilinear', align_corners=False)

        pyramid_features_cat = torch.cat([p2, p3_upsampled, p4_upsampled, p5_upsampled], dim=1)

        segmentation_features = self.seg_head_conv(pyramid_features_cat)
        segmentation_features = self.dropout(segmentation_features)
        logits = self.final_conv(segmentation_features) # Output at P2 resolution (H/4, W/4)

        # --- Final Upsampling ---
        if self.final_upsampling > 1:
            logits = F.interpolate(
                logits,
                scale_factor=self.final_upsampling,
                mode='bilinear',
                align_corners=False
            )

        return logits