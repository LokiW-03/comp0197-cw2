import torch
import torch.nn as nn
from torchvision import models


# Unet code partially adopted from https://medium.com/@fernandopalominocobo/mastering-u-net-a-step-by-step-guide-to-segmentation-from-scratch-with-pytorch-6a17c5916114

class UpSampleDecoder(nn.Module):
    """
    A decoder block that performs upsampling using transposed convolution,
    concatenates the result with a corresponding skip connection, and applies a convolutional layer.

    Args:
        in_channels (int): Number of input channels to the upsampling layer.
        out_channels (int): Number of output channels after upsampling and convolution.
        skip_out_channels (int): Number of channels in the skip connection from the encoder.

    Methods:
        forward(x, skip_connection):
            Upsamples the input, concatenates with a skip connection, and applies a convolution.

    """

    def __init__(self, in_channels, out_channels, skip_out_channels):
        super(UpSampleDecoder, self).__init__()
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
    """
    A lightweight semantic segmentation model based on EfficientNet-B0 as the encoder
    and unet-style decoder for spatially-precise segmentation and efficiency.

    Methods:
        forward(x):
            Passes input through the encoder and decoder to produce a segmentation map.
    """

    def __init__(self):
        super(EfficientUNet, self).__init__()
        self.encoder = models.efficientnet_b0(weights="DEFAULT")

        # Encode layer from efficient net
        self.encoder_stem = self.encoder.features[:1]  # First layer
        self.encoder_blocks = self.encoder.features[1:8]

        # Decode layer
        self.upconv1 = UpSampleDecoder(320, 192, 112)
        self.upconv2 = UpSampleDecoder(192, 112, 40)
        self.upconv3 = UpSampleDecoder(112, 80, 24)
        self.upconv4 = UpSampleDecoder(80, 40, 16)
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

        # Decoder: Upsample and concatenate with encoder outputs
        x = self.upconv1(x, encoder_outputs[-3])
        x = self.upconv2(x, encoder_outputs[-5])
        x = self.upconv3(x, encoder_outputs[-6])
        x = self.upconv4(x, encoder_outputs[-7])
        x = self.upconv5(x)

        # Final layer to output segmentation map with 3 classes
        x = self.final_conv(x)

        return x