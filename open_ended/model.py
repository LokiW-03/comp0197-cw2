#model.py
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

# TODO: What if we opt for a different segmentation model? How easy it is to make this change? Does this model need to match the model used in MRP?
# TODO: Can we use the model built in /model/baseline_<model name>.py?
class EffUnetWrapper(nn.Module):
    """
    Wrapper around segmentation-models-pytorch Unet with EfficientNet backbone.
    Handles different output requirements for various supervision modes.
    """
    def __init__(self, backbone='efficientnet-b0', num_classes=1, mode='segmentation'):
        """
        Args:
            backbone (str): Name of the EfficientNet backbone.
            num_classes (int): Number of output classes (1 for binary pet segmentation).
            mode (str): 'segmentation', 'classification', or 'hybrid'.
                        Determines the output head(s).
        """
        super().__init__()
        self.mode = mode
        self.num_classes = num_classes # For Pets binary, this is 1 (pet class)

        # Use SMP's Unet - assumes binary segmentation (num_classes=1 requires sigmoid activation)
        # For multi-class (like breeds), use num_classes > 1 and remove activation='sigmoid'
        self.unet = smp.Unet(
            encoder_name=backbone,
            encoder_weights="imagenet", # Use pretrained weights
            in_channels=3,
            classes=num_classes, # Output channels for segmentation map
            activation=None # Output logits for BCEWithLogitsLoss or CrossEntropyLoss
            # activation='sigmoid' if using DiceLoss or BCELoss directly
        )

        if mode == 'classification' or mode == 'hybrid':
            # Add a classification head
            # Get the number of features from the encoder output before decoder
            # This depends on the backbone, check SMP docs or probe the model
            # For efficientnet-b0 with Unet, the bottleneck features might be around 1280
            # Let's assume a reasonable number, may need adjustment
            encoder_out_channels = self.unet.encoder.out_channels[-1] # Get channels before decoder
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Linear(encoder_out_channels, num_classes) # Output logits

    def forward(self, x):
        """ Forward pass returning dictionary based on mode. """
        features = self.unet.encoder(x) # Get encoder features
        decoder_output = self.unet.decoder(*features) # Get decoder output (logits)
        segmentation_logits = self.unet.segmentation_head(decoder_output) # [B, C, H, W]

        output = {}
        if self.mode == 'segmentation' or self.mode == 'hybrid':
             # For binary case (num_classes=1), output is [B, 1, H, W]
             # For multi-class (num_classes=N), output is [B, N, H, W]
             output['segmentation'] = segmentation_logits

        if self.mode == 'classification' or self.mode == 'hybrid': #TODO: Do we need classification? Remove if not
            # Use features from the deepest encoder stage for classification
            pooled_features = self.global_pool(features[-1]) # Pool features from last encoder stage
            pooled_features = torch.flatten(pooled_features, 1) # Flatten
            classification_logits = self.classifier(pooled_features) # [B, C]
            output['classification'] = classification_logits

        # If mode was just 'segmentation' or 'classification', return only that tensor
        if self.mode == 'segmentation':
             return output['segmentation']
        elif self.mode == 'classification':
             return output['classification']
        else: # Hybrid mode
             return output # Return dictionary