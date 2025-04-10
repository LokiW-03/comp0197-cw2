
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
import logging
from PIL import Image
from torchvision import transforms
import math

# Assuming your SegNet model definition is in this path
# Make sure this import points to your actual model file
try:
    from model.baseline_segnet import SegNet
except ImportError:
    print("ERROR: Could not import SegNet from model.baseline_segnet.")
    print("Please ensure the 'baseline_segnet.py' file is in the 'model' directory relative to this script,")
    print("or adjust the import path accordingly.")
    exit()


# Configure visual settings
plt.style.use('ggplot')
# Using a categorical palette suitable for masks might be better than husl
# plt.set_cmap('tab10') # Example, set globally or per plot
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

# --- Helper Function for De-normalization ---
def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """De-normalizes a tensor image back to the [0, 1] range for display."""
    # Ensure tensor is on CPU for numpy conversion if needed, though ops work on device
    # Clone to avoid modifying the original tensor if it's used elsewhere
    tensor = tensor.clone()
    mean = torch.tensor(mean, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(1, 3, 1, 1)
    tensor.mul_(std).add_(mean) # In-place multiplication and addition
    return torch.clamp(tensor, 0, 1)
# --- End Helper ---

def get_device():
    # Prioritize MPS on Apple Silicon if available and PyTorch supports it well for the model
    # if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    #     print("MPS backend is available and built. Using MPS.")
    #     # Note: MPS support can still be experimental for some ops. Fallback if issues arise.
    #     return torch.device("mps")
    if torch.cuda.is_available():
        print("CUDA is available. Using CUDA.")
        return torch.device("cuda")
    else:
        print("CUDA not available. Using CPU.")
        return torch.device("cpu")

class SegmentationVisualizer:
    def __init__(self, model_paths, sample_image_path, output_dir="results"):

        self.device = get_device()
        print(f"Selected device: {self.device}")
        self.model_paths = model_paths
        self.output_dir = output_dir
        self.sample_image_path = sample_image_path
        # Load raw image for display later
        try:
            self.sample_image_raw = Image.open(sample_image_path).convert('RGB') # Ensure RGB
        except FileNotFoundError:
             logging.error(f"Sample image not found at: {sample_image_path}")
             raise
        except Exception as e:
             logging.error(f"Error opening sample image {sample_image_path}: {e}")
             raise

        # Determine image size from the first model's transform if needed, or set default
        self.img_size = 128 # Or another default/config value
        self.sample_image_tensor = self.load_and_preprocess_image(sample_image_path, self.img_size)
        # Resize the raw image used for display to match the tensor input size
        self.sample_image_raw = self.sample_image_raw.resize((self.img_size, self.img_size))

        os.makedirs(output_dir, exist_ok=True)

        # Define layers you want to compare activations for - VERIFY THESE NAMES
        self.comparison_layers = [
            'stage1_encoder.0', # Early encoder conv
            'stage3_encoder.6', # Mid encoder conv
            'stage5_encoder.3', # Late encoder conv (bottleneck?)
            'stage3_decoder.5', # Mid decoder conv
            'stage1_decoder.2'  # Late decoder conv
        ]
        # Store collected data here
        self.collected_data = {}
        logging.info(f"Visualizer initialized for {len(model_paths)} models. Output dir: {output_dir}")
        logging.info(f"Layers targeted for activation comparison: {self.comparison_layers}")


    def load_and_preprocess_image(self, path, img_size):
        """Load and preprocess sample image into a tensor."""
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(), # Converts to [0, 1] and CxHxW
            transforms.Normalize(mean=[0.485, 0.456, 0.406], # Standard ImageNet normalization
                               std=[0.229, 0.224, 0.225])
        ])
        try:
             # Re-open here to ensure consistency if called multiple times (though not current use case)
             with Image.open(path).convert('RGB') as img:
                   return transform(img).unsqueeze(0) # Add batch dimension
        except Exception as e:
             logging.error(f"Failed to load or transform image {path}: {e}")
             raise


    def load_model(self, path):
        """Load a trained SegNet model state_dict."""
        # Ensure the model architecture matches the saved weights
        # Adjust output_num_classes if your models were trained for different numbers
        model = SegNet(in_channels=3, output_num_classes=2)
        logging.info(f"Loading model state_dict from: {path}")
        try:
            # Use weights_only=True for security if the file ONLY contains weights
            checkpoint = torch.load(path, map_location='cpu')
            logging.info(f"Loaded state_dict successfully (weights_only=True).")
            # Check if checkpoint is the state_dict itself or wrapped in a dictionary
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                logging.info("Extracted state_dict from 'model_state_dict' key.")
            elif isinstance(checkpoint, dict):
                 state_dict = checkpoint # Assume it's the state_dict directly
                 logging.info("Checkpoint appears to be state_dict directly.")
            else:
                 logging.error("Loaded checkpoint format not recognized (expected dict).")
                 raise TypeError("Checkpoint format not recognized with weights_only=True.")

        except (RuntimeError, TypeError, KeyError) as e:
            logging.warning(f"Could not load {path} with weights_only=True ({e}). Trying with weights_only=False.")
            logging.warning("Loading with weights_only=False allows arbitrary code execution and is insecure if the source is untrusted.")
            try:
                checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                if 'model_state_dict' not in checkpoint:
                    logging.error("Checkpoint loaded with weights_only=False, but 'model_state_dict' key is missing.")
                    raise KeyError("Checkpoint does not contain 'model_state_dict'.")
                state_dict = checkpoint['model_state_dict']
                logging.info("Loaded state_dict successfully from 'model_state_dict' key (weights_only=False).")
            except Exception as fallback_e:
                logging.error(f"Failed to load model from {path} even with weights_only=False: {fallback_e}")
                raise fallback_e

        # Adjust keys: remove 'seg_model.' prefix if present in ALL keys (be careful)
        # It's safer to check if *any* key starts with it
        if any(key.startswith('seg_model.') for key in state_dict.keys()):
            logging.info("Detected 'seg_model.' prefix in state_dict keys. Attempting removal.")
            new_state_dict = {}
            corrected_keys = 0
            problematic_keys = 0
            for k, v in state_dict.items():
                 if k.startswith('seg_model.'):
                      new_key = k.replace('seg_model.', '', 1) # Replace only the first instance
                      new_state_dict[new_key] = v
                      corrected_keys +=1
                 else:
                      # If some keys have the prefix and others don't, it might indicate an issue
                      new_state_dict[k] = v
                      problematic_keys +=1
            if problematic_keys > 0:
                 logging.warning(f"{problematic_keys} keys did not have the 'seg_model.' prefix. Check model saving code.")
            state_dict = new_state_dict
            logging.info(f"Removed prefix from {corrected_keys} keys.")
        else:
            logging.info("No 'seg_model.' prefix detected in state_dict keys.")

        try:
             model.load_state_dict(state_dict)
             logging.info("State_dict loaded into model structure successfully.")
        except RuntimeError as load_e:
             logging.error(f"Error loading state_dict into model: {load_e}")
             logging.error("This often means the model architecture definition (SegNet) doesn't match the keys/weights in the loaded file.")
             raise load_e

        return model.eval().to(self.device) # Set to evaluation mode and move to device


    def _get_layer_display_name(self, layer_full_name):
        """Creates a shorter, more readable name for plotting."""
        parts = layer_full_name.split('.')
        if len(parts) >= 2:
            stage_part = parts[0].replace("stage", "S").replace("_encoder", " Enc").replace("_decoder", " Dec")
            layer_idx = parts[1]
            # You might want finer control, e.g., map specific indices to block names if known
            return f"{stage_part} ({layer_idx})"
        return layer_full_name # Fallback


    def collect_data_for_comparison(self):
        """Runs inference for each model and collects activations/predictions."""
        logging.info("Starting data collection for comparison plot...")
        self.collected_data = {} # Reset data

        if not self.model_paths:
            logging.warning("No model paths provided. Skipping data collection.")
            return

        input_tensor_device = self.sample_image_tensor.to(self.device)
        logging.info(f"Input tensor moved to: {input_tensor_device.device}")


        for model_idx, model_path in enumerate(self.model_paths):
            # Try to infer a meaningful name, fallback to index
            try:
                 basename = os.path.basename(model_path)
                 parts = basename.split('_')
                 if parts[1] == 'hybrid':
                     # Try to capture the specific hybrid combination
                     # This assumes format like segnet_hybrid_TYPE1_TYPE2_..._runX...
                     # Adjust indices based on your exact naming!
                     if len(parts) > 3:
                         model_type = f"Hybrid_{parts[2]}_{parts[3]}" # e.g., Hybrid_points_boxes
                         # Add more parts[N] if you have combinations like points_scribbles_boxes
                         if len(parts) > 4 and 'run' not in parts[4]:
                              model_type += f"_{parts[4]}"
                     else:
                         model_type = f"Hybrid_{model_idx}" # Fallback if name is short
                 elif len(parts) > 1:
                     model_type = parts[1] # Original logic for single types
                 else:
                     raise IndexError # Fallback to index if split fails
            except IndexError:
                 logging.warning(f"Could not infer detailed model type from path: {model_path}. Using index {model_idx}.")
                 model_type = f"Model_{model_idx+1}"

            logging.info(f"--- Processing model: {model_type} ({model_path}) ---")

            try:
                model = self.load_model(model_path)
                model_device = next(model.parameters()).device
                logging.info(f"Model loaded and moved to: {model_device}")

                if input_tensor_device.device != model_device:
                     logging.error(f"CRITICAL: Device mismatch! Input tensor on {input_tensor_device.device}, Model on {model_device}. Aborting for this model.")
                     # Decide how to handle: skip model, try moving tensor again, raise error?
                     # Skipping is safest if unsure why mismatch occurred.
                     continue # Skip to the next model

                activation_maps = {}
                hooks = []

                # --- Verify Layer Names Exist in this specific model instance ---
                available_modules = dict(model.named_modules())
                valid_comparison_layers = []
                for layer_name in self.comparison_layers:
                    if layer_name in available_modules:
                        valid_comparison_layers.append(layer_name)
                    else:
                        logging.warning(f"Layer '{layer_name}' not found in model {model_type}. It will be skipped for this model's activation plot.")
                logging.info(f"Found {len(valid_comparison_layers)} target layers in {model_type}: {valid_comparison_layers}")
                # -------------------------------------------------------------

                # Register hooks only for valid layers found in this model
                def get_activation(name):
                    # This function is created dynamically for each hook
                    # It captures the 'name' variable from the outer scope
                    def hook(module, input, output):
                        # Store activations on CPU to save GPU memory during collection
                        activation_maps[name] = output.detach().cpu()
                    return hook

                for layer_name in valid_comparison_layers:
                    module = available_modules[layer_name]
                    try:
                        hook_handle = module.register_forward_hook(get_activation(layer_name))
                        hooks.append(hook_handle)
                        # logging.debug(f"Registered hook for {layer_name}")
                    except Exception as hook_e:
                         logging.error(f"Failed to register hook for layer {layer_name} on model {model_type}: {hook_e}")


                logging.info(f"Running inference for {model_type}...")
                # Run inference
                with torch.no_grad(): # Ensure gradients are not computed
                     output = model(input_tensor_device)
                     # Assuming output is (Batch, Classes, H, W)
                     pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu() # Remove batch dim, move to CPU
                     logging.info(f"Inference complete. Output shape: {output.shape}, Pred mask shape: {pred_mask.shape}")


                # Remove hooks immediately after inference to free resources
                logging.debug(f"Removing {len(hooks)} hooks...")
                for h in hooks:
                    h.remove()
                hooks.clear() # Clear the list
                logging.debug("Hooks removed.")


                # Store data for this model
                self.collected_data[model_type] = {
                    'prediction': pred_mask,       # Store prediction tensor (CPU)
                    'activations': activation_maps, # Store dict of activation tensors (CPU)
                    'valid_layers': valid_comparison_layers # Store which layers were actually processed
                }
                logging.info(f"Successfully collected data for {model_type}")

            except Exception as e:
                logging.error(f"Failed to process model {model_type} from {model_path}: {e}", exc_info=True) # Log traceback
                # Store placeholder data or skip? Skipping might be cleaner for the plot.
                # self.collected_data[model_type] = {'error': str(e)} # Option to store error

            finally:
                # Clean up model and cache regardless of success/failure
                if 'model' in locals():
                    del model
                if self.device.type == 'cuda':
                     torch.cuda.empty_cache()
                     # logging.debug("CUDA cache cleared.")
                elif self.device.type == 'mps':
                     # torch.mps.empty_cache() # If available/needed
                     pass


        logging.info(f"Data collection finished. Collected data for {len(self.collected_data)} models.")


    def plot_comparison_grid(self):
        """Plots the collected activations and predictions in a grid."""
        if not self.collected_data:
            logging.error("No data collected or processed successfully. Cannot generate comparison plot.")
            return
        # Filter out models that might have failed during collection if errors weren't stored
        valid_model_types = [mt for mt, data in self.collected_data.items() if 'prediction' in data]
        if not valid_model_types:
             logging.error("No models had successful data collection. Cannot plot grid.")
             return

        num_models = len(valid_model_types)
        # Columns: Input Image + N comparison layers + Prediction Mask
        num_cols = 1 + len(self.comparison_layers) + 1

        fig, axs = plt.subplots(num_models, num_cols,
                                figsize=(num_cols * 2.5, num_models * 2.7), # Adjust size as needed
                                squeeze=False) # Ensure axs is always 2D array

        fig.suptitle("Model Comparison: Layer Activations and Predictions", fontsize=14, y=1.03)

        # --- Plot Input Image (Column 0) ---
        # Prepare input image for display (denormalize, permute dims)
        # Ensure sample_image_tensor is on CPU before denormalizing if needed, though denormalize handles device
        img_display_tensor = denormalize(self.sample_image_tensor.cpu()) # Denormalize on CPU
        img_display_np = img_display_tensor.squeeze(0).permute(1, 2, 0).numpy()

        for i, model_type in enumerate(valid_model_types):
            ax = axs[i, 0]
            ax.imshow(img_display_np)
            # Set Y-label as the model type name for the row
            ax.set_ylabel(model_type.replace('_', ' ').title(), # Nicer formatting
                          rotation=0, fontsize=10, labelpad=40, # Adjust labelpad for spacing
                          verticalalignment='center', horizontalalignment='right')
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0: # Set title only for the top-left plot
                ax.set_title("Input", fontsize=10)


        # --- Plot Activations (Columns 1 to N) ---
        for j, layer_name in enumerate(self.comparison_layers):
            col_idx = j + 1 # Start from column 1
            layer_display_name = self._get_layer_display_name(layer_name)

            for i, model_type in enumerate(valid_model_types):
                ax = axs[i, col_idx]
                model_data = self.collected_data[model_type]

                # Check if this specific layer's activation was successfully collected for this model
                if layer_name in model_data.get('activations', {}): # Use .get for safety
                    activation = model_data['activations'][layer_name].squeeze(0) # Remove potential batch dim if not already done
                    # Handle different activation tensor dimensions
                    if activation.dim() == 3: # Expected: (C, H, W)
                         activation_display = activation.mean(0).numpy() # Mean across channels
                    elif activation.dim() == 2: # Possible: (H, W)
                         activation_display = activation.numpy()
                    else:
                         logging.warning(f"Unexpected activation shape {activation.shape} for {layer_name} in {model_type}. Displaying zeros.")
                         # Get spatial dimensions from prediction mask as fallback size
                         h, w = model_data['prediction'].shape[-2:]
                         activation_display = np.zeros((h, w))

                    im = ax.imshow(activation_display, cmap='inferno', aspect='auto') # 'auto' or 'equal'
                    # Optional: Add colorbar for activation intensity (can make plot busy)
                    # fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                else:
                    # Indicate if activation for this layer is missing for this model
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=9, color='grey')
                    ax.set_facecolor('#f0f0f0') # Light grey background for missing data

                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0: # Set column title only for the top row
                    ax.set_title(layer_display_name, fontsize=10)


        # --- Plot Predictions (Last Column) ---
        pred_col_idx = num_cols - 1
        for i, model_type in enumerate(valid_model_types):
            ax = axs[i, pred_col_idx]
            pred_mask = self.collected_data[model_type]['prediction'].numpy()
            # Use a categorical colormap suitable for segmentation masks
            # 'tab10' provides 10 distinct colors. Adjust vmin/vmax if class indices differ.
            im = ax.imshow(pred_mask, cmap='tab10', vmin=0, vmax=9, aspect='auto') # 'tab10' is good for up to 10 classes
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title("Prediction", fontsize=10)
            # Optional: Add a colorbar for the prediction legend (might need separate handling)


        plt.tight_layout(rect=[0, 0.02, 1, 0.98]) # Adjust rect to make space for suptitle and ylabel
        # Fine-tune spacing if elements overlap
        plt.subplots_adjust(wspace=0.05, hspace=0.15) # Small width space, slightly more height space

        save_path = os.path.join(self.output_dir, "model_activation_comparison_grid.png")
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Comparison grid saved successfully to {save_path}")
        except Exception as save_e:
             logging.error(f"Failed to save comparison grid plot: {save_e}")

        plt.close(fig) # Close the figure to free memory


    def generate_comparison_report(self):
        """Generates the comparison plot ONLY."""
        logging.info("Starting comparison report generation...")

        # 1. Collect data for all models for the comparison grid
        self.collect_data_for_comparison()

        # 2. Generate the comparison grid plot
        self.plot_comparison_grid()

        logging.info(f"Comparison report generation finished. Output saved in {self.output_dir}")


if __name__ == "__main__":
    # --- Configuration ---
    # Ensure these paths are correct relative to where you run the script
    MODEL_PATHS = [
        './open_ended/models/segnet_single/segnet_points_run1_best_acc.pth',
        './open_ended/models/segnet_single/segnet_scribbles_run1_best_acc.pth',
        './open_ended/models/segnet_hybrid/segnet_hybrid_points_scribbles_run1_best_acc.pth',

        './open_ended/models/segnet_single/segnet_boxes_run1_best_acc.pth',
        './open_ended/models/segnet_hybrid/segnet_hybrid_points_boxes_run1_best_acc.pth',
        './open_ended/models/segnet_hybrid/segnet_hybrid_scribbles_boxes_run1_best_acc.pth',

        './open_ended/models/segnet_hybrid/segnet_hybrid_points_scribbles_boxes_run1_best_acc.pth',

    ]

    SAMPLE_IMAGE_PATH = "./open_ended/data/images/basset_hound_38.jpg" # Make sure this path is correct
    OUTPUT_DIR = "comparison_report_grid_only" # Specify output directory

    # --- Basic Checks ---
    print("--- Performing Pre-run Checks ---")
    abort = False
    if not os.path.exists(SAMPLE_IMAGE_PATH):
        print(f"ERROR: Sample image not found at: {SAMPLE_IMAGE_PATH}")
        abort = True
    else:
        print(f"Sample image found: {SAMPLE_IMAGE_PATH}")

    found_models = 0
    for i, p in enumerate(MODEL_PATHS):
        if not os.path.exists(p):
             print(f"ERROR: Model file {i+1} not found at: {p}")
             abort = True
        else:
             print(f"Model file {i+1} found: {p}")
             found_models +=1

    if abort:
        print("ERROR: Required file(s) missing. Please check paths. Aborting.")
        exit()
    elif found_models == 0:
        print("ERROR: No model files listed or found. Aborting.")
        exit()
    else:
         print(f"All {found_models} model files and sample image found. Proceeding...")
    print("---------------------------------")
    # --------------------

    try:
        visualizer = SegmentationVisualizer(
            model_paths=MODEL_PATHS,
            sample_image_path=SAMPLE_IMAGE_PATH,
            output_dir=OUTPUT_DIR
        )
        visualizer.generate_comparison_report()
        print(f"\n--- Script Finished ---")
        print(f"Comparison report generated in: {os.path.abspath(OUTPUT_DIR)}")

    except ImportError as e:
        # Catch potential import errors for SegNet if not caught earlier
        print(f"\n--- SCRIPT FAILED ---")
        print(f"Import Error: {e}")
        print("Please ensure your SegNet model definition is accessible.")
    except FileNotFoundError as e:
         print(f"\n--- SCRIPT FAILED ---")
         print(f"File Not Found Error: {e}")
         print("Please double-check the paths for models and the sample image.")
    except Exception as e:
         # Catch any other unexpected errors during visualization
         print(f"\n--- SCRIPT FAILED ---")
         print(f"An unexpected error occurred: {e}")
         import traceback
         print("\n--- Traceback ---")
         traceback.print_exc()
         print("-----------------")