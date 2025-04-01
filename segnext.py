
"""
Architecture Explanation: We define the MSCAN_Backbone with four stages. The first stage (stem) uses two 3x3 strided conv layers to reduce the input size to 1/4 with an embedding dimension (e.g., 64 channels).

Subsequent stages begin with a 3x3 stride-2 convolution (overlapping patch embed) to downsample and increase the channels, then apply a sequence of MSCABlock modules. Each MSCABlock contains a LargeKernelAttention (with multi-scale depthwise conv kernels 5,7,11,21) and a ConvFeedForward (1x1 conv layers expanding/contracting channels) with residual connections.

We include DropPath (stochastic depth) and layer scale parameters as in the official implementation for stability in deep networks.

The SegNeXt class ties everything together: it uses the backbone to extract features, upsamples and concatenates them (multi-level feature fusion), then applies fusion_conv and a final classifier conv. This is analogous to the official decoder that “collects multi-level features from different stages” and applies a global context module. Here we use a simple 1x1 conv in place of the Hamburger attention for simplicity. The model outputs a tensor of shape (B, num_classes, H, W) with raw logits per class for each pixel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Convolutional MLP block used in each MSCA block (1x1 convs with expansion)
class ConvFeedForward(nn.Module):
    def __init__(self, dim, expansion_ratio=4, drop=0.0):
        super().__init__()
        hidden_dim = int(dim * expansion_ratio)
        self.fc1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# Large Kernel Attention module (MSCA attention part)
class LargeKernelAttention(nn.Module):
    """Spatial attention with multi-scale large kernel convolutions."""
    def __init__(self, dim):
        super().__init__()
        # Depthwise convolutions with large kernels (decomposed for efficiency)
        self.dw_conv5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)  # 5x5 depthwise
        self.dw_conv7_1 = nn.Conv2d(dim, dim, kernel_size=(1,7), padding=(0,3), groups=dim)  # 7x7 via 1x7
        self.dw_conv7_2 = nn.Conv2d(dim, dim, kernel_size=(7,1), padding=(3,0), groups=dim)  # 7x7 via 7x1
        self.dw_conv11_1 = nn.Conv2d(dim, dim, kernel_size=(1,11), padding=(0,5), groups=dim) # 11x11 via strips
        self.dw_conv11_2 = nn.Conv2d(dim, dim, kernel_size=(11,1), padding=(5,0), groups=dim)
        self.dw_conv21_1 = nn.Conv2d(dim, dim, kernel_size=(1,21), padding=(0,10), groups=dim) # 21x21 via strips
        self.dw_conv21_2 = nn.Conv2d(dim, dim, kernel_size=(21,1), padding=(10,0), groups=dim)
        self.point_conv = nn.Conv2d(dim, dim, kernel_size=1)  # 1x1 conv to mix channels after spatial conv
    def forward(self, x):
        # x: [B, C, H, W]
        identity = x
        # Apply base depthwise convolution
        attn = self.dw_conv5(x)
        # Multi-scale large kernel conv branches
        attn_branch0 = self.dw_conv7_2(self.dw_conv7_1(attn))
        attn_branch1 = self.dw_conv11_2(self.dw_conv11_1(attn))
        attn_branch2 = self.dw_conv21_2(self.dw_conv21_1(attn))
        # Sum the multi-scale convolution outputs
        attn = attn + attn_branch0 + attn_branch1 + attn_branch2
        attn = self.point_conv(attn)
        # Multiply attention weights with input (gating mechanism)
        out = identity * attn
        return out

# MSCA Block: combines LargeKernelAttention and ConvFeedForward with residuals
class MSCABlock(nn.Module):
    def __init__(self, dim, expansion_ratio=4, drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = LargeKernelAttention(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.ffn = ConvFeedForward(dim, expansion_ratio, drop=drop)
        # Stochastic depth drop-path
        self.drop_path = (DropPath(drop_path) if drop_path > 0.0 else nn.Identity())
        # Layer scale (learnable rescaling for stability)
        self.layer_scale_1 = nn.Parameter(1e-2 * torch.ones(dim), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(1e-2 * torch.ones(dim), requires_grad=True)
    def forward(self, x):
        # x shape [B, C, H, W]
        # Attention branch with residual
        out = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        # Feed-forward branch with residual
        out = out + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.ffn(self.norm2(out)))
        return out

# Stochastic depth (DropPath) implementation for drop_path in blocks
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        # Compute drop mask
        keep_prob = 1 - self.drop_prob
        # Work with shape [batch, 1, 1, 1] to broadcast
        random_tensor = keep_prob + torch.rand((x.shape[0], 1, 1, 1), dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x / keep_prob * random_tensor
        return output

# Encoder Backbone: Multi-Scale Convolutional Attention Network (MSCAN)
class MSCAN_Backbone(nn.Module):
    def __init__(self, in_channels=3, embed_dims=[64, 128, 256, 512], depths=[3, 4, 6, 3], drop_rate=0.0, drop_path_rate=0.1):
        """
        embed_dims: list of feature dimensions for the 4 stages.
        depths: number of MSCA blocks in each stage.
        """
        super().__init__()
        self.num_stages = len(embed_dims)
        # Stem convolution (stage 0) to downsample input to 1/4 size
        stem_out = embed_dims[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_out//2, kernel_size=3, stride=2, padding=1),  # stride 2
            nn.BatchNorm2d(stem_out//2), nn.GELU(),
            nn.Conv2d(stem_out//2, stem_out, kernel_size=3, stride=2, padding=1),  # another stride 2
            nn.BatchNorm2d(stem_out), nn.GELU()
        )
        # Stages 1-4
        total_blocks = sum(depths)
        current_block_idx = 0
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            stage_blocks = []
            if i == 0:
                # Stage 1 already got stem conv, will apply blocks now
                in_ch = embed_dims[0]
            else:
                # Downsample layer for stage i (overlap patch embedding)
                in_ch = embed_dims[i-1]
                out_ch = embed_dims[i]
                stage_blocks.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1))
                stage_blocks.append(nn.BatchNorm2d(out_ch))
                stage_blocks.append(nn.GELU())
            # Add MSCA blocks for this stage
            for j in range(depths[i]):
                block_dim = embed_dims[i]  # number of channels in this stage
                # Calculate drop_path probability for this block (linearly scaled through stages)
                dpr = drop_path_rate * (current_block_idx / (total_blocks - 1))
                stage_blocks.append(MSCABlock(block_dim, expansion_ratio=4, drop=drop_rate, drop_path=dpr))
                current_block_idx += 1
            # Combine all parts for this stage into a Sequential module
            self.stages.append(nn.Sequential(*stage_blocks))
    def forward(self, x):
        features = []
        # Stage 0 (stem)
        x = self.stem(x)  # downsample to 1/4
        # Stage 1
        x = self.stages[0](x)
        features.append(x)
        # Stages 2-4
        for i in range(1, self.num_stages):
            x = self.stages[i](x)
            features.append(x)
        # features will be [stage1, stage2, stage3, stage4] feature maps
        return features

# Complete SegNeXt model with encoder and decoder
class SegNeXt(nn.Module):
    def __init__(self, num_classes=3, backbone_dims=[64,128,256,512], backbone_depths=[3,4,6,3]):
        super().__init__()
        self.backbone = MSCAN_Backbone(in_channels=3, embed_dims=backbone_dims, depths=backbone_depths)
        # Decoder: fuse multi-scale features
        # We will upsample all features to the resolution of stage1 feature map and concatenate
        fusion_in_channels = sum(backbone_dims)  # total channels when all features are concatenated
        self.fusion_conv = nn.Conv2d(fusion_in_channels, 256, kernel_size=1)  # reduce fused channels
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)  # output segmentation logits
    def forward(self, x):
        B, C, H, W = x.shape
        feats = self.backbone(x)  # list of feature maps from 4 stages
        # Upsample all features to the size of the first stage (1/4 of input)
        # Stage 1 feature is feats[0] with spatial size (H/4, W/4)
        up_feats = [feats[0]]
        target_size = feats[0].shape[2:]  # (H/4, W/4)
        for f in feats[1:]:
            up_feats.append(F.interpolate(f, size=target_size, mode='bilinear', align_corners=False))
        fused = torch.cat(up_feats, dim=1)  # concatenate along channel axis
        fused = self.fusion_conv(fused)
        fused = F.gelu(fused)  # activation (optional)
        out = self.classifier(fused)
        # Upsample output to original image size
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out
