import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import VGG19_Weights
import torch.nn.functional as F


class VGGLoss(nn.Module):
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