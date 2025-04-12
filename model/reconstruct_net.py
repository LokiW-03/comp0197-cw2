# I acknowledge the use of ChatGPT (version GPT-4o, OpenAI, https://chatgpt.com/) for assistance in debugging and
# writing docstrings.

import torch
import torch.nn as nn

"""
This implementation of ReconstructNet is adapted from the decoder architecture 
used in the paper
"Spatial Structure Constraints for Weakly Supervised Semantic Segmentation"
https://arxiv.org/abs/2401.11122

The code structure and modules (e.g., Conv2dBlock, ResBlock) were adopted from the paper's repository:
https://github.com/NUST-Machine-Intelligence-Laboratory/SSC

Modifications were made for simplification, clarity, and alignment with our project requirements.
"""


class LayerNorm(nn.Module):
    """
    Applies layer normalization over an entire input tensor, normalizing across all spatial and channel dimensions.
    Useful for stabilizing training in convolutional networks.

    Args:
        num_features (int): Number of features/channels in the input.
        eps (float): Small epsilon added to standard deviation for numerical stability. Default is 1e-5.
        affine (bool): If True, includes learnable scaling (gamma) and shifting (beta) parameters. Default is True.

    Methods:
        forward(x):
            Applies layer normalization to the input tensor.
    """

    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.contiguous().view(x.size(0), -1).mean(1).view(*shape)
        std = x.contiguous().view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class Conv2dBlock(nn.Module):
    """
    A configurable 2D convolution block with optional padding, normalization, and activation.
    Commonly used as a basic building block in convolutional networks.

    Args:
        input_dim (int): Number of input channels.
        output_dim (int): Number of output channels.
        kernel_size (int): Size of the convolution kernel.
        stride (int): Convolution stride. Default is 1.
        padding (int): Padding size. Default is 0.
        norm (str): Normalization type. Use 'ln' for LayerNorm or None. Default is 'ln'.
        activation (str): Activation type. Supports 'relu', 'tanh', or None. Default is 'relu'.
        pad_type (str): Padding type, either 'zero' or 'reflect'. Default is 'zero'.
        bias (bool): Whether to include a bias term in the convolution. Default is True.

    Methods:
        forward(x):
            Applies padding, convolution, optional normalization, and activation.

    """

    def __init__(self, input_dim, output_dim, kernel_size, stride=1,
                 padding=0, norm='ln', activation='relu', pad_type='zero', bias=True):

        super(Conv2dBlock, self).__init__()
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)

        if norm == 'ln':
            self.norm = LayerNorm(output_dim)
        else:
            self.norm = None

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = None

        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class ResBlock(nn.Module):
    """
    Residual block composed of two convolutional blocks with LayerNorm and ReLU.
    Adds the input to the output of the block to form a residual connection.

    Args:
        dim (int): Number of input and output channels.

    Methods:
        forward(x):
            Applies two convolutional layers with a skip connection.

    """

    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            Conv2dBlock(dim, dim, 3, 1, 1, norm='ln', activation='relu'),
            Conv2dBlock(dim, dim, 3, 1, 1, norm='ln', activation=None)
        )

    def forward(self, x):
        return x + self.block(x)


class ReconstructNet(nn.Module):
    """
    Reconstructs RGB image from CAM-like input tensor.
    Args:
        input_channel (int): Number of input channels in CAM (default: 37)
    Input: [B, input_channel, H, W]
    Output: [B, 3, H_out, W_out] in range [-1, 1]
    """
    def __init__(self, input_channel=37):
        super(ReconstructNet, self).__init__()

        # Conv Block x1
        # 3×3 Conv, LayerNorm, Relu
        self.conv_block = Conv2dBlock(input_channel, 256, 3, 1, 1, norm='ln', activation='relu')
        
        # ResBlock x1 
        # 3×3 Conv, LayerNorm, Relu
        # 3×3 Conv, LayerNorm
        self.res_block = ResBlock(256)

        # Up-sample Blocks x4
        # 4×4 ConvTranspose, InstanceNorm, Relu 
        # 3×3 Conv, LayerNorm, Relu
        self.upsample_1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            Conv2dBlock(256, 256, 3, 1, 1, norm='ln', activation='relu')
        )

        self.upsample_2 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            Conv2dBlock(256, 128, 3, 1, 1, norm='ln', activation='relu')
        )

        self.upsample_3 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            Conv2dBlock(128, 64, 3, 1, 1, norm='ln', activation='relu')
        )

        self.upsample_4 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            Conv2dBlock(64, 32, 3, 1, 1, norm='ln', activation='relu')
        )

        self.upsample_5 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            Conv2dBlock(32, 16, 3, 1, 1, norm='ln', activation='relu')
        )

        # Last Conv Block
        # 3×3 Conv,Tanh
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(16, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        
        x = self.conv_block(x)
        x = self.res_block(x)

        x = self.upsample_1(x)
        x = self.upsample_2(x)
        x = self.upsample_3(x)
        x = self.upsample_4(x)
        x = self.upsample_5(x)

        x = self.conv_block_2(x)
        return x
