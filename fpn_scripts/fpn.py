# models/fpn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet34_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from typing import Dict

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


from torchvision.models import ResNet34_Weights, ResNet50_Weights
# --- Helper dictionary for encoder properties ---
# Maps encoder name to: (torchvision_model_fn, weights_enum, return_nodes, C2-C5_channels)
ENCODER_INFO = {
    "resnet34": (
        models.resnet34,
        ResNet34_Weights.IMAGENET1K_V1,
        {'layer1': 'c2', 'layer2': 'c3', 'layer3': 'c4', 'layer4': 'c5'},
        [64, 128, 256, 512], # Channels for C2, C3, C4, C5
    ),
    "resnet50": (
        models.resnet50,
        ResNet50_Weights.IMAGENET1K_V2, # Use V2 for ResNet50
        {'layer1': 'c2', 'layer2': 'c3', 'layer3': 'c4', 'layer4': 'c5'},
        # Note: Bottleneck blocks change channel dims for ResNet50
        [256, 512, 1024, 2048], # Channels for C2, C3, C4, C5
    ),
}

class FPN(nn.Module):
    """
    Feature Pyramid Network (FPN) for Semantic Segmentation using a
    configurable Torchvision backbone with pretrained weights.

    Args:
        encoder_name (str): Name of the Torchvision backbone to use
                            (e.g., "resnet34", "resnet50").
        in_channels (int): Number of input channels (e.g., 3 for RGB images).
                            Must be 3 if using standard ImageNet pretrained weights.
        classes (int): Number of output segmentation classes.
        pretrained (bool): Whether to use ImageNet pretrained weights for the
                            backbone. Defaults to True.
        pyramid_channels (int): Number of channels in the FPN lateral and
                            top-down layers. Defaults to 256.
        segmentation_channels (int): Number of channels in the segmentation
                                    head. Defaults to 128.
        final_upsampling (int): Upsampling factor for final logits. Defaults to 4.
        dropout (float): Dropout probability in segmentation head. Defaults to 0.2.
        use_batchnorm (bool): Whether to use BatchNorm in FPN/Head layers. Defaults to True.
    """
    def __init__(
        self,
        encoder_name: str = "resnet34",
        in_channels: int = 3,
        classes: int = 1,
        pretrained: bool = True,
        pyramid_channels: int = 256,
        segmentation_channels: int = 128,
        final_upsampling: int = 4,
        dropout: float = 0.2,
        use_batchnorm: bool = True,
    ):  
        super().__init__()

        if encoder_name not in ENCODER_INFO:
            raise ValueError(f"Encoder '{encoder_name}' not supported. Available: {list(ENCODER_INFO.keys())}")

        if pretrained and in_channels != 3:
            print(f"Warning: Pretrained weights requested for encoder '{encoder_name}' but in_channels={in_channels} != 3. Forcing pretrained=False.")
            pretrained = False

        self.classes = classes
        self.final_upsampling = final_upsampling

        # --- Dynamically Load Encoder Info ---
        model_fn, weights_enum, return_nodes, encoder_channels = ENCODER_INFO[encoder_name]

        # --- Load Backbone ---
        weights = weights_enum if pretrained else None
        backbone = model_fn(weights=weights)

        # --- Create Feature Extractor ---
        self.encoder = create_feature_extractor(backbone, return_nodes=return_nodes)

        # --- FPN Layers (using dynamically determined encoder_channels) ---
        self.lateral_c2 = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=1) # C2 channels
        self.lateral_c3 = nn.Conv2d(encoder_channels[1], pyramid_channels, kernel_size=1) # C3 channels
        self.lateral_c4 = nn.Conv2d(encoder_channels[2], pyramid_channels, kernel_size=1) # C4 channels
        self.lateral_c5 = nn.Conv2d(encoder_channels[3], pyramid_channels, kernel_size=1) # C5 channels

        self.smooth_p4 = ConvReLU(pyramid_channels, pyramid_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.smooth_p3 = ConvReLU(pyramid_channels, pyramid_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.smooth_p2 = ConvReLU(pyramid_channels, pyramid_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)

        # --- Segmentation Head ---
        self.seg_head_conv = ConvReLU(
            pyramid_channels * len(encoder_channels), # Number of pyramid levels used
            segmentation_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.dropout = nn.Dropout2d(p=dropout, inplace=False)
        self.final_conv = nn.Conv2d(segmentation_channels, classes, kernel_size=1)

        # --- Initialize ONLY FPN Layers ---
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
        Forward pass through the FPN model with configurable Torchvision encoder.
        """
        # --- Encoder ---
        features: Dict[str, torch.Tensor] = self.encoder(x)
        # Extract features based on the keys defined in return_nodes (c2, c3, c4, c5)
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
        segmentation_features = self.dropout(segmentation_features) # Dropout applied here
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
