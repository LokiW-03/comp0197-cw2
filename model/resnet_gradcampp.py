# resnetcam.py
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.transforms import Resize, InterpolationMode
from common import IMAGE_SIZE

class ResNet50_CAM(nn.Module):
    """
    ResNet50 model modified for Class Activation Mapping (CAM)
    - Uses pre-trained ResNet50 as backbone
    - Replaces final FC layer for custom classification
    - Registers hook to capture intermediate activations
    
    Output shapes:
    - forward: (B, num_classes) - Classification logits
    - activations: (B, 2048, H', W') - Feature maps from last conv layer
    """
    def __init__(self, num_classes):
        super().__init__()
        # Load pre-trained ResNet-50 with ImageNet weights
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Replace final FC layer with custom one for our number of classes
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        
        # Register hook for last conv layer to capture feature maps
        self.target_layer = self.resnet.layer4[-1].conv3  # Last conv layer in ResNet50
        self.activations = None
        self.gradients = None
        
        # register forward and backward hooks
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_full_backward_hook(self.save_gradients)
    
    def save_activations(self, output):
        """
        Hook function to save intermediate activations
        Args:
            output: Output from the layer, shape: (B, 2048, H', W')
        """
        self.activations = output  # Keep gradient information for backprop
    
    def save_gradients(self, grad_output):
        """
        Hook function to save gradients
        Args:
            grad_output: Gradient of the output
        """
        self.gradients = grad_output[0]
    
    def forward(self, x):
        """
        Forward pass of the model
        Args:
            x: Input tensor of shape (B, 3, H, W)
        Returns:
            Tensor of shape (B, num_classes) - Classification logits
        """
        return self.resnet(x)

class GradCAMpp:
    """
    Grad-CAM++ implementation for generating class activation maps
    - Extends standard Grad-CAM with higher-order derivatives
    - Provides better localization of object features
    - Supports batch processing and multiple classes
    
    Output shapes:
    - forward: ((B, num_classes, H, W), (B, num_classes)) - (CAMs, logits)
    - normalize: (B, H, W) - Normalized CAM values
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.num_classes = model.resnet.fc.out_features
        # Initialize resize transform for upsampling to input size
        self.resize = Resize((IMAGE_SIZE, IMAGE_SIZE), 
                            interpolation=InterpolationMode.BILINEAR)
    
    def generate_cam(self, x, all_classes=True, resize=True):
        """
        Generate class activation maps for input images
        Args:
            x: Input tensor of shape (B, 3, H, W)
            all_classes: Whether to generate CAMs for all classes
            resize: Whether to resize back to image shape
        Returns:
            tuple: (upsampled_cams, logits)
                - upsampled_cams: Tensor of shape (B, num_classes, H, W), if all_classes is False, (B, H, W) otherwise
                - logits: Tensor of shape (B, num_classes)
        """
        if not all_classes:
            return self._generate_single_cam(x)

        # Forward pass to get activation maps and logits
        logits = self.model(x)  # (B, num_classes)
        activations = self.model.activations  # (B, 2048, H', W')
        
        # Get dimensions for feature maps and batch size
        B, num_classes = logits.shape
        H_prime, W_prime = activations.shape[2], activations.shape[3]
        
        # Initialize tensor to store CAMs for all classes
        cams = torch.zeros(B, num_classes, H_prime, W_prime, device=x.device)
        
        # Calculate CAM for each class
        for class_idx in range(num_classes):
            # Compute gradients with respect to class score
            grads = torch.autograd.grad(
                outputs=logits[:, class_idx].sum(),
                inputs=activations,
                retain_graph=True,
                create_graph=False
            )[0]  # (B, 2048, H', W')
            
            # Grad-CAM++ core computation
            # 1. Compute alpha weights using squared gradients
            alpha = grads.pow(2).mean(dim=(2,3), keepdim=True)  # (B, 2048, 1, 1)
            # 2. Compute final weights using alpha and gradients
            weights = (alpha * grads).mean(dim=(2,3), keepdim=True)  # (B, 2048, 1, 1)
            # 3. Generate CAM by weighted sum of activations
            cam = (weights * activations).sum(dim=1)  # (B, H', W')
            
            # Post-processing steps
            cam = cam.view(B, H_prime, W_prime)  # Ensure correct shape
            cam = F.relu(cam)  # Remove negative values
            cam = self.normalize(cam)  # Normalize to [0,1]
            
            # Store CAM for current class
            cams[:, class_idx] = cam

        if resize:
            cams = self.resize(cams)
        
        # Upsample CAMs to match input image size
        return cams, logits  # ((B, num_classes, H, W), (B, num_classes))
    
    @staticmethod
    def normalize(cam):
        """
        Normalize CAM values to [0,1] range
        Args:
            cam: Input CAM tensor of shape (B, H, W)
        Returns:
            Tensor of shape (B, H, W) - Normalized CAM values in range [0,1]
        """
        B, H, W = cam.shape
        # Flatten for batch-wise normalization
        cam_flat = cam.view(B, -1)  # (B, H*W)
        # Compute min and max per batch
        cam_min = cam_flat.min(dim=1, keepdim=True)[0]  # (B, 1)
        cam_max = cam_flat.max(dim=1, keepdim=True)[0]  # (B, 1)
        # Normalize with small epsilon to avoid division by zero
        normalized = (cam_flat - cam_min) / (cam_max - cam_min + 1e-7)  # (B, H*W)
        # Restore original shape
        return normalized.view(B, H, W)  # (B, H, W)
    
    def _generate_single_cam(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM++ visualization for input tensor(s)
        
        Args:
            input_tensor: Input tensor of shape (B, C, H, W) or (C, H, W)
            target_class: Target class index. If None, uses predicted class
            
        Returns:
            cam: numpy array of shape (B, H, W) or (H, W) containing CAM values
        """
        # Handle single image input
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        batch_size = input_tensor.size(0)
        device = input_tensor.device
        
        # Forward propagation
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1)
        
        # Backward propagation to calculate gradients
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[torch.arange(batch_size, device=device), target_class] = 1
        output.backward(gradient=one_hot)
        
        # Get activations and gradients
        activations = self.model.activations.detach().cpu()
        gradients = self.model.gradients.detach().cpu()
        
        # Grad-CAM++
        alpha = torch.sum(gradients, dim=(2, 3), keepdim=True)
        weights = torch.mean(alpha * torch.relu(gradients), dim=(2, 3), keepdim=True)
        
        # Sum up the weighted activations
        cam = torch.sum(weights * activations, dim=1)  # (B, H, W)
        cam = torch.relu(cam)  # Remove negative responses
        
        # Normalize the cam
        cam -= cam.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        cam /= cam.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        
        # Resize the cam to the input size
        cam_resized = self.resize(cam.unsqueeze(1)).squeeze(1)  # (B, H, W)
        
        # Convert to numpy (only at the end)
        cam = cam_resized
        
        # Return single image if input was single image
        if batch_size == 1:
            cam = cam.squeeze(0)
            
        return cam, output