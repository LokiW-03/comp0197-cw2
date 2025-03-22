import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models


class UNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, skip_out_channels):
        super(UNetDecoder, self).__init__()
        # Define upsampling layers (ConvTranspose2d)
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.conv = nn.Conv2d(skip_out_channels + out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, skip_connection):
        # Upsample the feature map using ConvTranspose2d
        x = self.upconv(x)  # Upsample the feature map using transpose convolution

        # Concatenate the upsampled output with the skip connection from the encoder
        x = torch.cat([x, skip_connection], dim=1) # Concatenate along the channel dimension

        # Apply convolution after concatenation
        x = self.conv(x)

        return x

class EfficientUNet(nn.Module):
    def __init__(self):
        super(EfficientUNet, self).__init__()
        self.encoder = models.efficientnet_b0(weights="DEFAULT")

        # Encode layer from efficient net
        self.encoder_stem = self.encoder.features[:1]  # First layer
        self.encoder_blocks = self.encoder.features[1:8]

        # Decode layer
        self.upconv1 = UNetDecoder(320, 192, 112)
        self.upconv2 = UNetDecoder(192, 112, 40)
        self.upconv3 = UNetDecoder(112, 80, 24)
        self.upconv4 = UNetDecoder(80, 40, 16)
        self.upconv5 = nn.ConvTranspose2d(40, 24, kernel_size=2, stride=2, padding=0)

        # Final convolution to match the number of output classes
        self.final_conv = nn.Conv2d(24, 3, kernel_size=3, padding=1)


    def forward(self, x):
        # Encoder: Extract features
        x = self.encoder_stem(x)  # First stem layer
        encoder_outputs = []

        for idx, block in enumerate(self.encoder_blocks):
            x = block(x)
            encoder_outputs.append(x)
            #print(f"Shape of encoder output at block {idx}: {x.shape}") # Store encoder outputs for skip connections in upsampling

        # Decoder: Upsample and concatenate with encoder outputs
        x = self.upconv1(x, encoder_outputs[-3])
        x = self.upconv2(x, encoder_outputs[-5])
        x = self.upconv3(x, encoder_outputs[-6])
        x = self.upconv4(x, encoder_outputs[-7])
        x = self.upconv5(x)

        # Final layer to output segmentation map with 3 classes
        x = self.final_conv(x)

        return x