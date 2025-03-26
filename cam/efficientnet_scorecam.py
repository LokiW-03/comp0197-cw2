# efficientnet_scorecam.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from torchvision.transforms import Resize, InterpolationMode
from __init__ import IMAGE_SIZE  # Make sure IMAGE_SIZE is defined

##########################################
# EfficientNet-B4 with CAM Hooks
##########################################
class EfficientNetB4_CAM(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB4_CAM, self).__init__()
        # Load pretrained EfficientNet-B4
        self.effnet = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        
        # Disable in-place operations in SiLU activations
        self._disable_inplace(self.effnet)
        
        # Replace classifier for the new task
        in_features = self.effnet.classifier[1].in_features
        self.effnet.classifier[1] = nn.Linear(in_features, num_classes)
        
        # Choose a target layer for CAM – we select the last convolution of the last feature block.
        self.target_layer = self.effnet.features[-1][-1]
        
        # Placeholders for hooks
        self.activations = None
        self.gradients = None
        
        # Register hooks to capture forward activations and backward gradients.
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_full_backward_hook(self.save_gradients)
    
    def _disable_inplace(self, model):
        """Disable in-place operations for all SiLU activations to avoid hook conflicts."""
        for module in model.modules():
            if isinstance(module, nn.SiLU):
                module.inplace = False

    def save_activations(self, module, input, output):
        self.activations = output.detach().clone()

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach().clone()
    
    def forward(self, x):
        return self.effnet(x)

##########################################
# Score-CAM Implementation
##########################################
class ScoreCAM:
    def __init__(self, model):
        """
        Initialize ScoreCAM with a model that must have hooks set up to record
        activations (e.g. an instance of EfficientNetB4_CAM).
        """
        self.model = model
        self.model.eval()
    
    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate Score-CAM visualization.
        
        Args:
            input_tensor: Tensor of shape (B, C, H, W).
            target_class: Tensor of shape (B,) containing class indices.
                          If None, the predicted class is used.
        
        Returns:
            Tensor of shape (B, H, W) containing the normalized CAM.
        """
        B, C, H, W = input_tensor.shape
        
        # Forward pass through the model
        output = self.model(input_tensor)  # (B, num_classes)
        if target_class is None:
            target_class = torch.argmax(output, dim=1)
        
        # Get activations from the hooked target layer
        activations = self.model.activations  # (B, k, h, w)
        if activations is None:
            raise RuntimeError("Activations not captured; ensure hook registration is correct.")
        k = activations.shape[1]
        
        # Upsample activation maps to match the input image size
        upsampled_maps = F.interpolate(activations, size=(H, W), mode='bilinear', align_corners=False)
        
        # Normalize each activation map to [0, 1]
        eps = 1e-8
        upsampled_maps_flat = upsampled_maps.view(B, k, -1)
        maps_min = upsampled_maps_flat.min(dim=2, keepdim=True)[0].view(B, k, 1, 1)
        maps_max = upsampled_maps_flat.max(dim=2, keepdim=True)[0].view(B, k, 1, 1)
        norm_maps = (upsampled_maps - maps_min) / (maps_max - maps_min + eps)
        
        # Create masked inputs by elementwise multiplication of the input with each normalized map.
        masked_inputs = input_tensor.unsqueeze(1) * norm_maps.unsqueeze(2)
        masked_inputs = masked_inputs.view(B * k, C, H, W)
        
        # Forward pass for masked images (with no gradient computation)
        with torch.no_grad():
            masked_output = self.model(masked_inputs)  # (B*k, num_classes)
            masked_output = F.softmax(masked_output, dim=1)
        
        # Retrieve scores for the target class for each masked input.
        scores = []
        for i in range(B):
            sample_scores = masked_output[i * k: (i + 1) * k, target_class[i]]
            scores.append(sample_scores.unsqueeze(0))
        scores = torch.cat(scores, dim=0).view(B, k, 1, 1)
        
        # Compute the weighted sum of the normalized maps.
        cam = torch.sum(norm_maps * scores, dim=1)  # (B, H, W)
        cam = F.relu(cam)
        
        # Normalize the CAM for each image
        cam_out = []
        for i in range(B):
            cam_i = cam[i]
            cam_min = cam_i.min()
            cam_max = cam_i.max()
            cam_i = (cam_i - cam_min) / (cam_max - cam_min + eps)
            cam_out.append(cam_i)
        cam_out = torch.stack(cam_out)
        
        return cam_out
