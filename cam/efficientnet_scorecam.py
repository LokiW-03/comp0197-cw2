# efficientnet_scorecam.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from torchvision.transforms import Resize, InterpolationMode
from common import IMAGE_SIZE  # Ensure IMAGE_SIZE is defined

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
        
        # Target layer for CAM (last conv of the last feature block)
        self.target_layer = self.effnet.features[-1][-1]
        
        # Hook placeholders
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_full_backward_hook(self.save_gradients)
    
    def _disable_inplace(self, model):
        """Disable in-place operations in SiLU"""
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
# Modified Score-CAM for All Classes
##########################################
class ScoreCAM:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        # Initialize a resize transform for the final CAM output.
        self.resize = Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BILINEAR)
    
    def generate_cam(self, input_tensor, all_classes=True, resize=True):
        """
        Generate CAM for all classes.
        
        Args:
            input_tensor: (B, C, H, W)
            all_classes: Whether to generate CAMs for all classes.
            resize: If True, the output CAMs are resized to (IMAGE_SIZE, IMAGE_SIZE).
            
        Returns:
            Tuple: (cams, logits)
                - cams: Tensor of shape (B, num_classes, H_final, W_final) 
                - logits: Tensor of shape (B, num_classes)
        """
        if not all_classes:
            # previous implementation
            # TODO: merge with the all_classes implementation
            return self._generate_single_cam(input_tensor)
        
        B, C, H, W = input_tensor.shape
        
        # Forward pass to get logits
        logits = self.model(input_tensor)  # (B, num_classes)
        num_classes = logits.size(1)
        
        # Get activation maps
        activations = self.model.activations  # (B, k, h, w)
        if activations is None:
            raise RuntimeError("Activations not captured")
        B, k, H, W = activations.shape
        
        # Normalize activation maps at their native resolution.
        flat_activations = activations.view(B, k, -1)
        min_vals = flat_activations.min(dim=2, keepdim=True)[0].view(B, k, 1, 1)
        max_vals = flat_activations.max(dim=2, keepdim=True)[0].view(B, k, 1, 1)
        norm_maps = (activations - min_vals) / (max_vals - min_vals + 1e-8)  # (B, k, H, W)
        
        # Downsample the input to the activation map resolution.
        input_down = F.interpolate(input_tensor, size=(H, W), mode='bilinear', align_corners=False)
        
        # Initialize score_maps tensor
        score_maps = torch.zeros(B, k, num_classes, device=input_tensor.device)
        
        # Process activation maps in batches
        # TODO: Optimize this part when using GPU
        batch_size = 8  # Adjust this value based on GPU memory
        num_batches = (k + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, k)
            current_norm_maps = norm_maps[:, start:end, :, :]  # (B, m, H, W)
            m = current_norm_maps.size(1)
            
            # Generate masked inputs
            masked_inputs = input_down.unsqueeze(1) * current_norm_maps.unsqueeze(2)
            masked_inputs = masked_inputs.view(B * m, C, H, W)
            
            with torch.no_grad():
                masked_logits = self.model(masked_inputs)  # (B*m, num_classes)
                masked_scores = F.softmax(masked_logits, dim=1)  # (B*m, num_classes)
            
            # Save results
            score_maps[:, start:end, :] = masked_scores.view(B, m, num_classes)
        
        # Calculate final CAM
        cam = torch.einsum("bkhw,bkc->bchw", norm_maps, score_maps)  # (B, C, H, W)
        cam = F.relu(cam)
        
        # Normalize
        flat_cam = cam.view(B, num_classes, -1)
        cam_min = flat_cam.min(dim=2, keepdim=True)[0].unsqueeze(-1)
        cam_max = flat_cam.max(dim=2, keepdim=True)[0].unsqueeze(-1)
        norm_cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        
        if resize:
            norm_cam = self.resize(norm_cam)
        return norm_cam, logits
        
    def _generate_single_cam(self, input_tensor, target_class=None, resize=True):
        """
        Generate Score-CAM visualization for a single target class,
        delaying the upscaling until the final CAM is computed.
        
        Args:
            input_tensor: Tensor of shape (B, C, H, W).
            target_class: Tensor of shape (B,) containing class indices.
                        If None, uses predicted class.
            resize: If True, upscales the final CAM to (IMAGE_SIZE, IMAGE_SIZE).
        
        Returns:
            Tuple: (norm_cam, logits)
                - norm_cam: Tensor of shape (B, H_final, W_final) with values in [0,1]
                - logits: Tensor of shape (B, num_classes)
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
        B, k, H, W = activations.shape

        # Downsample the input to the native activation resolution.
        input_down = F.interpolate(input_tensor, size=(H, W), mode='bilinear', align_corners=False)
        
        # Normalize each activation map to [0, 1]
        eps = 1e-8
        flat_maps = activations.view(B, k, -1)
        maps_min = flat_maps.min(dim=2, keepdim=True)[0].view(B, k, 1, 1)
        maps_max = flat_maps.max(dim=2, keepdim=True)[0].view(B, k, 1, 1)
        norm_maps = (activations - maps_min) / (maps_max - maps_min + eps)  # (B, k, H, W)
        
        # Create masked inputs by elementwise multiplication of the input with each normalized map.
        masked_inputs = input_down.unsqueeze(1) * norm_maps.unsqueeze(2)  # (B, k, C, H, W)
        masked_inputs = masked_inputs.view(B * k, C, H, W)
        
        # Forward pass for masked images (with no gradient computation)
        with torch.no_grad():
            masked_output = self.model(masked_inputs)  # (B*k, num_classes)
            masked_output = F.softmax(masked_output, dim=1)
        
        # Retrieve scores for the target class for each masked input.
        scores = []
        for i in range(B):
            sample_scores = masked_output[i * k: (i + 1) * k, target_class[i]]  # (k,)
            scores.append(sample_scores.unsqueeze(0))
        scores = torch.cat(scores, dim=0).view(B, k, 1, 1)  # (B, k, 1, 1)

        # Compute the weighted sum of the normalized maps.
        cam = torch.sum(norm_maps * scores, dim=1)  # (B, H, W)
        cam = F.relu(cam)
        
        # Normalize the CAM for each image
        cam_flat = cam.view(B, -1)
        cam_min = cam_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1)
        cam_max = cam_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1)
        norm_cam = (cam - cam_min) / (cam_max - cam_min + eps)  # (B, H, W)

        # Delay the upsampling until now, if requested.
        if resize:
            norm_cam = self.resize(norm_cam)  # Upsamples to (B, IMAGE_SIZE, IMAGE_SIZE)

        return norm_cam, output
