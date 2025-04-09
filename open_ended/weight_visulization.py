import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
import logging
from PIL import Image
from torchvision import transforms

from model.baseline_segnet import SegNet

# Configure visual settings
plt.style.use('ggplot')
sns.set_palette("husl")
logging.basicConfig(level=logging.INFO)

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Metal GPU
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

class SegmentationVisualizer:
    def __init__(self, model_paths, sample_image_path, output_dir="results"):

        self.device = get_device()
        print(f"Using device: {self.device}")
        self.model_paths = model_paths
        self.output_dir = output_dir
        self.sample_image = self.load_sample_image(sample_image_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Visualization configuration
        self.layer_groups = {
            'encoder': ['stage1_encoder.0.0', 'stage3_encoder.6', 'stage5_encoder.3'],
            'decoder': ['stage1_decoder.2', 'stage3_decoder.5', 'stage5_decoder.3']
        }

    def load_sample_image(self, path, img_size=128):
        """Load and preprocess sample image"""
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        return transform(Image.open(path)).unsqueeze(0)

    def load_model(self, path):
        """Load trained SegNet model"""
        model = SegNet(in_channels=3, output_num_classes=2)
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        return model.eval()

    def visualize_weights(self, model, ann_type):
        """Visualization 1: Weight distributions and filters"""
        plt.figure(figsize=(15, 10))
        
        # First layer filters
        first_conv = model.stage1_encoder[0][0]
        weights = first_conv.weight.detach().cpu().numpy()
        
        # Plot filters
        plt.subplot(2, 2, 1)
        grid_size = int(np.sqrt(len(weights)))
        for i, w in enumerate(weights[:grid_size**2]):
            plt.subplot(grid_size, grid_size, i+1)
            plt.imshow(w.mean(0), cmap='viridis', interpolation='nearest')
            plt.axis('off')
        plt.suptitle(f"{ann_type} - First Layer Filters", y=0.95)
        
        # Weight distributions
        plt.subplot(2, 2, 2)
        all_weights = []
        for name, param in model.named_parameters():
            if 'weight' in name and 'conv' in name:
                all_weights.extend(param.detach().flatten().cpu().numpy())
        sns.histplot(all_weights, kde=True, bins=50)
        plt.title("All Conv Layer Weight Distribution")
        plt.xlim(-0.5, 0.5)
        
        # Layer-wise means
        plt.subplot(2, 1, 2)
        layer_means = []
        layer_names = []
        for name, param in model.named_parameters():
            if 'weight' in name and 'conv' in name:
                layer_means.append(param.detach().abs().mean().item())
                layer_names.append(name.split('.')[-2])
        sns.barplot(x=layer_names, y=layer_means)
        plt.title("Average Absolute Weight Magnitudes")
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"weights_{ann_type}.png"), dpi=300)
        plt.close()

    def visualize_activations(self, model, ann_type):
        """Visualization 2: Activation maps and predictions"""
        activation_maps = {}
        
        # Register hooks
        def get_activation(name):
            def hook(module, input, output):
                activation_maps[name] = output.detach()
            return hook
        
        hooks = []
        for layer in self.layer_groups['encoder'][:1] + self.layer_groups['decoder'][-1:]:
            module = dict(model.named_modules())[layer]
            hooks.append(module.register_forward_hook(get_activation(layer)))

        # Run inference
        with torch.no_grad():
            output = model(self.sample_image)
            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        # Remove hooks
        for h in hooks:
            h.remove()

        # Plot results
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        # Input image
        axs[0].imshow(self.sample_image.squeeze().permute(1,2,0).numpy()*0.5+0.5)
        axs[0].set_title("Input Image")
        axs[0].axis('off')
        
        # Prediction mask
        axs[1].imshow(pred_mask, cmap='tab20')
        axs[1].set_title(f"Prediction - {ann_type}")
        axs[1].axis('off')
        
        # Activation maps
        layer_name = self.layer_groups['encoder'][0]
        activations = activation_maps[layer_name].squeeze().mean(0).cpu().numpy()
        axs[2].imshow(activations, cmap='inferno')
        axs[2].set_title(f"{layer_name} Activations")
        axs[2].axis('off')

        plt.savefig(os.path.join(self.output_dir, f"activations_{ann_type}.png"), 
                   bbox_inches='tight', dpi=300)
        plt.close()

    def generate_report(self):
        """Generate complete visualization report"""
        logging.info("Starting visualization process...")
        
        for model_path in self.model_paths:
            ann_type = os.path.basename(model_path).split('_')[0]
            logging.info(f"Processing {ann_type} model...")
            
            model = self.load_model(model_path)
            
            # Generate both visualizations
            self.visualize_weights(model, ann_type)
            self.visualize_activations(model, ann_type)

        logging.info(f"Visualization report saved to {self.output_dir}")
        




if __name__ == "__main__":
    # Example usage
    MODEL_PATHS = [
        './open_ended/models_segnet_single',
        './checkpoints/scribbles_best.pth', 
        './checkpoints/boxes_best.pth',
        './checkpoints/hybrid_best.pth'
    ]
    
    SAMPLE_IMAGE_PATH = "./open_ended/data/images/beagle_9.jpg"
    
    visualizer = SegmentationVisualizer(
        model_paths=MODEL_PATHS,
        sample_image_path=SAMPLE_IMAGE_PATH,
        output_dir="comparison_report"
    )
    visualizer.generate_report()