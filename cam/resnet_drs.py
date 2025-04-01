import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class DRS(nn.Module):
    def __init__(self, delta=0.55):
        super(DRS, self).__init__()
        self.delta = delta
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        x_max = self.global_max_pool(x)
        x = torch.min(x, x_max * self.delta)
        return x


class ResNet50_CAM_DRS(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Replace final FC layer for custom classification
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

        # Insert DRS after layer2
        self.drs = DRS(delta=0.55)

        # Hook last conv layer for CAM
        self.target_layer = self.resnet.layer4[-1].conv3
        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_full_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.drs(x)                 
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        return x
