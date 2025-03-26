#resnetcam.py
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import Resize, InterpolationMode
from __init__ import IMAGE_SIZE

class ResNet50_CAM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # replace the last fully connected layer
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        
        # register hook for the target layer
        self.target_layer = self.resnet.layer4[-1].conv3
        self.activations = None
        self.gradients = None
        
        # register forward and backward hooks
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_full_backward_hook(self.save_gradients)
    
    def save_activations(self, module, input, output):
        self.activations = output
    
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def forward(self, x):
        return self.resnet(x)

class GradCAMpp:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.resize_transform = Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BILINEAR)
    
    def generate_cam(self, input_tensor, target_class=None):
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
        cam_resized = self.resize_transform(cam.unsqueeze(1)).squeeze(1)  # (B, H, W)
        
        # Convert to numpy (only at the end)
        cam = cam_resized
        
        # Return single image if input was single image
        if batch_size == 1:
            cam = cam.squeeze(0)
            
        return cam