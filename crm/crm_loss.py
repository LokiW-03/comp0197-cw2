# I acknowledge the use of ChatGPT (version GPT-4o, OpenAI, https://chatgpt.com/) for assistance in debugging and
# writing docstrings.

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import VGG19_Weights
import torch.nn.functional as F

"""
This implementation of the loss functions is adapted from the loss network used in the paper
"Spatial Structure Constraints for Weakly Supervised Semantic Segmentation"
https://arxiv.org/abs/2401.11122

The code structure and modules (e.g. VGGLoss, alignment loss) were adopted from the paper's repository:
https://github.com/NUST-Machine-Intelligence-Laboratory/SSC

Modifications were made for simplification, clarity, and alignment with our project requirements.
"""

class VGGLoss(nn.Module):
    """
    Computes perceptual loss using a pretrained VGG19 network.

    Args:
        device (torch.device): Device to load the VGG model on (CPU or CUDA).
    """

    def __init__(self, device):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().to(device)
        self.criterion = nn.L1Loss()
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)

    def forward(self, x, y, weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]):
        bs = x.size(0)
        while x.size()[3] > 1024:
            x, y = self.downsample(x), self.downsample(y)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss
    
class MaskedVGGLoss(nn.Module):
    """
    Computes a masked version of the perceptual loss using a pretrained VGG19 network.

    Only considers features in valid regions defined by the mask, allowing partial-region
    comparison. Useful for tasks like inpainting, segmentation, or selective region evaluation.

    Args:
        device (torch.device): Device to load the VGG model on (CPU or CUDA).
    """

    def __init__(self, device):
        super(MaskedVGGLoss, self).__init__()
        self.vgg = Vgg19().to(device)
        self.criterion = nn.L1Loss(reduction='none')
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)

    def forward(self, x, y, mask, weights=[1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]):
        while x.size(3) > 1024:
            x, y, mask = self.downsample(x), self.downsample(y), self.downsample(mask)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        total_loss = 0
        for i in range(len(x_vgg)):
            # Downsample the mask to match the spatial dimensions of the current feature map.
            m = F.interpolate(mask, size=x_vgg[i].shape[2:], mode='nearest')
            
            diff = self.criterion(x_vgg[i], y_vgg[i].detach())
            masked_diff = diff * m
            loss_i = masked_diff.sum() / (m.sum() + 1e-8)
            total_loss += weights[i] * loss_i
        return total_loss

class Vgg19(nn.Module):
    """
    VGG19 feature extractor.

    Loads a pretrained VGG19 model and slices it into 5 sequential blocks, each representing
    different levels of abstraction in the image. Used for computing perceptual similarity
    or extracting deep features.

    Args:
        requires_grad (bool): If False, disables gradient computation for VGG parameters.
    """

    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
    
    
def alignment_loss(pred, sp, label, criterion):

    """
    Computes alignment loss between predicted CAMs and superpixel regions.

    This loss function encourages the model's CAM outputs to align with superpixel
    boundaries by averaging CAM activations within each superpixel and minimizing
    the difference between original and averaged activations.

    Args:
        pred (Tensor): Predicted CAMs of shape [B, C, H, W].
        sp (Tensor): Superpixel segmentation map of shape [B, H', W'].
        label (Tensor): One-hot encoded label of shape [B, C].
        criterion (callable): A PyTorch loss function (e.g., L1Loss).

    Returns:
        Tensor: Scalar alignment loss.
    """

    superpixel = sp.float()

    # downsample superpixel to match CAM
    superpixel = F.interpolate(superpixel.unsqueeze(1), scale_factor=0.5, mode='nearest').squeeze(1)
    _, h2, w2 = superpixel.shape

    # prepare and reshape
    n, c, _, _ = pred.shape

    sp_num = int(superpixel.max().item()) + 1
    superpixel = superpixel.to(torch.int64)
    sp_flat = F.one_hot(superpixel, num_classes=sp_num)
    sp_flat = sp_flat.permute(0, 3, 1, 2)

    # match CAM with SP shape
    pred = F.interpolate(pred, size=(h2, w2), mode='bilinear')

    # compute avg activation per SP
    sp_flat = sp_flat.reshape(n, sp_num, -1)
    score_flat = pred.reshape(n, c, -1) * label.reshape(n, c, 1)
    sp_flat_avg = sp_flat / (sp_flat.sum(dim=2, keepdim=True) + 1e-5)
    sp_flat_avg = sp_flat_avg.float()
    score_sum = torch.matmul(score_flat, sp_flat.transpose(1, 2).float())
    
    score_sum = score_sum.reshape(n, c, sp_num, 1)
    sp_flat_avg = sp_flat_avg.unsqueeze(1).repeat(1, c, 1, 1)

    # distribute Averaged CAM Scores
    final_averaged_value = (sp_flat_avg * score_sum).sum(2)
    final_averaged_value = final_averaged_value * label.reshape(n, c, 1)

    # normalize and return loss for relevant class channel
    original_value = score_flat / (score_flat.max(dim=2, keepdim=True)[0] + 1e-5)
    final_averaged_value = final_averaged_value / (final_averaged_value.max(dim=2, keepdim=True)[0] + 1e-5)
    
    return criterion(original_value[label == 1], final_averaged_value[label == 1])