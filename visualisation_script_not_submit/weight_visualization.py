# visualization_script.py (Modified with Annotation Scaling)

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import logging
from PIL import Image, ImageDraw # Added ImageDraw
from torchvision import transforms
import pickle # Added pickle to load weak labels
import traceback # For detailed error reporting

# Assuming your SegNet model definition is in this path
# Make sure this import point to your actual model file
try:
    # Make sure this import point to your actual model file
    # If your model class is directly in 'baseline_segnet.py':
    # from baseline_segnet import SegNet
    # If it's inside a 'model' folder relative to this script:
    from model.baseline_segnet import SegNet
    print("Successfully imported SegNet from model.baseline_segnet")
except ImportError as e:
    print(f"ImportError: {e}")
    print("ERROR: Could not import SegNet.")
    print("Attempted import from 'model.baseline_segnet'.")
    print("Please ensure:")
    print("  1. An '__init__.py' file exists in the 'model' directory.")
    print("  2. The 'baseline_segnet.py' file containing the SegNet class is inside the 'model' directory.")
    print("  3. The script is run from the directory *containing* the 'model' directory.")
    print("  4. There are no syntax errors in 'baseline_segnet.py'.")
    # Provide more context about the current working directory
    print(f"Current Working Directory: {os.getcwd()}")
    print(f"Checking for model directory existence: {'Exists' if os.path.isdir('model') else 'Not Found'}")
    if os.path.isdir('model'):
        print(f"Checking for model file existence: {'Exists' if os.path.isfile('model/baseline_segnet.py') else 'Not Found'}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    exit()


# Configure visual settings
plt.style.use('ggplot')
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

# --- Helper Function for De-normalization ---
def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """De-normalizes a tensor image back to the [0, 1] range for display."""
    tensor = tensor.clone()
    mean = torch.tensor(mean, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(1, 3, 1, 1)
    tensor.mul_(std).add_(mean)
    return torch.clamp(tensor, 0, 1)
# --- End Helper ---

# --- Annotation Drawing Helpers ---
# --- MODIFIED to handle scaling ---
def draw_point(draw_context, point_list, target_size, x_scale, y_scale, color="red", radius=3):
    """Draws scaled point (circles) on the PIL Draw context."""
    if not point_list: return
    target_w, target_h = target_size, target_size # Assuming square target

    for y_orig, x_orig in point_list:
        # Scale coordinates
        # Ensure original coords are treated as numbers before scaling
        x_scaled = int(float(x_orig) * x_scale)
        y_scaled = int(float(y_orig) * y_scale)

        # Clamp coordinates to be within target bounds
        x_scaled = max(0, min(target_w - 1, x_scaled))
        y_scaled = max(0, min(target_h - 1, y_scaled))

        # Ellipse takes bounding box [x0, y0, x1, y1] using scaled coords
        bbox = [x_scaled - radius, y_scaled - radius, x_scaled + radius, y_scaled + radius]
        draw_context.ellipse(bbox, fill=color, outline=color)

def draw_boxes(draw_context, boxes_list, target_size, x_scale, y_scale, color="lime", width=1):
    """Draws scaled bounding boxes on the PIL Draw context."""
    if not boxes_list: return
    target_w, target_h = target_size, target_size # Assuming square target

    for ymin_orig, xmin_orig, ymax_orig, xmax_orig in boxes_list:
        # Scale coordinates
        xmin_scaled = int(float(xmin_orig) * x_scale)
        ymin_scaled = int(float(ymin_orig) * y_scale)
        xmax_scaled = int(float(xmax_orig) * x_scale)
        ymax_scaled = int(float(ymax_orig) * y_scale)

        # Clamp coordinates
        xmin_scaled = max(0, min(target_w - 1, xmin_scaled))
        ymin_scaled = max(0, min(target_h - 1, ymin_scaled))
        xmax_scaled = max(0, min(target_w - 1, xmax_scaled))
        ymax_scaled = max(0, min(target_h - 1, ymax_scaled))

        # Ensure xmin <= xmax and ymin <= ymax after potential clamping/scaling edge cases
        if xmin_scaled > xmax_scaled: xmin_scaled, xmax_scaled = xmax_scaled, xmin_scaled
        if ymin_scaled > ymax_scaled: ymin_scaled, ymax_scaled = ymax_scaled, ymin_scaled

        # Avoid drawing zero-width/height boxes if scaling collapses them
        if xmin_scaled == xmax_scaled or ymin_scaled == ymax_scaled:
            # Optionally draw a point or skip
            # draw_context.point([xmin_scaled, ymin_scaled], fill=color) # Example: draw corner point
            continue # Skip degenerate boxes

        # Rectangle takes [x0, y0, x1, y1] using scaled coords
        draw_context.rectangle([xmin_scaled, ymin_scaled, xmax_scaled, ymax_scaled], outline=color, width=width)
# --- End Annotation Drawing Helpers ---


def get_device():
    if torch.cuda.is_available():
        print("CUDA is available. Using CUDA.")
        return torch.device("cuda")
    # elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    #     print("MPS backend is available. Using MPS.")
    #     return torch.device("mps") # Uncomment if preferred and stable
    else:
        print("CUDA/MPS not available. Using CPU.")
        return torch.device("cpu")

class SegmentationVisualizer:
    def __init__(self, model_paths, sample_image_path, weak_labels_path, output_dir="results"):

        self.device = get_device()
        print(f"Selected device: {self.device}")
        self.model_paths = model_paths
        self.output_dir = output_dir
        self.sample_image_path = sample_image_path
        self.sample_image_filename = os.path.basename(sample_image_path)
        self.weak_labels_path = weak_labels_path
        self.weak_labels = None
        self.sample_image_weak_labels = None

        # --- Load Weak Labels ---
        try:
            with open(self.weak_labels_path, 'rb') as f:
                self.weak_labels = pickle.load(f)
            logging.info(f"Successfully loaded weak labels from: {self.weak_labels_path}")
            if self.sample_image_filename not in self.weak_labels:
                 logging.warning(f"Sample image filename '{self.sample_image_filename}' not found as a key in the loaded weak labels file '{self.weak_labels_path}'. Annotation overlays might be empty.")
                 self.sample_image_weak_labels = {'tags': [], 'point': [], 'scatter': [], 'boxes': []}
            else:
                 self.sample_image_weak_labels = self.weak_labels[self.sample_image_filename]
                 logging.info(f"Found weak labels for sample image '{self.sample_image_filename}'.")
                 # --- Convert numpy types in annotations to standard python types ---
                 # This prevents potential issues with downstream processing or scaling if coords are numpy types
                 if 'point' in self.sample_image_weak_labels:
                      self.sample_image_weak_labels['point'] = [
                          (int(p[0]), int(p[1])) for p in self.sample_image_weak_labels['point']
                      ]
                 if 'scatter' in self.sample_image_weak_labels:
                     self.sample_image_weak_labels['scatter'] = [
                         (int(p[0]), int(p[1])) for p in self.sample_image_weak_labels['scatter']
                     ]
                 if 'boxes' in self.sample_image_weak_labels:
                      self.sample_image_weak_labels['boxes'] = [
                          (int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in self.sample_image_weak_labels['boxes']
                      ]
                 logging.info("Converted annotation coordinates to standard Python integers.")
                 #---------------------------------------------------------------------

        except FileNotFoundError:
            logging.error(f"Weak labels file not found at: {self.weak_labels_path}. Cannot display annotations.")
            raise
        except Exception as e:
            logging.error(f"Error loading or processing weak labels from {self.weak_labels_path}: {e}", exc_info=True)
            raise
        # ------------------------

        # Load raw image for display later
        try:
            # Load the full raw image first to get original dimensions
            self.sample_image_raw_full = Image.open(sample_image_path).convert('RGB')
            self.orig_w, self.orig_h = self.sample_image_raw_full.size # Get original size
            logging.info(f"Original sample image size: {self.orig_w}x{self.orig_h}")
        except FileNotFoundError:
             logging.error(f"Sample image not found at: {sample_image_path}")
             raise
        except Exception as e:
             logging.error(f"Error opening sample image {sample_image_path}: {e}")
             raise

        # Determine image size for processing and display
        self.img_size = 128 # Keep consistent size for model input and display

        # --- Calculate Scaling Factors ---
        self.x_scale = self.img_size / self.orig_w
        self.y_scale = self.img_size / self.orig_h
        logging.info(f"Annotation scaling factors: x={self.x_scale:.4f}, y={self.y_scale:.4f}")
        # --------------------------------

        # Now create the tensor using transforms (which includes resizing)
        self.sample_image_tensor = self.load_and_preprocess_image(sample_image_path, self.img_size)

        # Create the resized raw image for display using the full raw image
        self.sample_image_raw = self.sample_image_raw_full.resize(
            (self.img_size, self.img_size),
            Image.Resampling.LANCZOS # Use high-quality resampling
        )

        os.makedirs(output_dir, exist_ok=True)

        # Define layers you want to compare activations for
        self.comparison_layers = [
            'stage1_encoder.0', 'stage3_encoder.6', 'stage5_encoder.3',
            'stage3_decoder.5', 'stage1_decoder.2'
        ]
        self.collected_data = {}
        logging.info(f"Visualizer initialized for {len(model_paths)} models. Output dir: {output_dir}")
        logging.info(f"Target sample image: {self.sample_image_path}")
        logging.info(f"Weak labels source: {self.weak_labels_path}")
        logging.info(f"Image size for processing/display: {self.img_size}x{self.img_size}")
        logging.info(f"Layers targeted for activation comparison: {self.comparison_layers}")

    # load_and_preprocess_image remains the same as it uses transforms.Resize
    def load_and_preprocess_image(self, path, img_size):
        """Load and preprocess sample image into a tensor."""
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR), # Ensure consistent resize method
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        try:
             with Image.open(path).convert('RGB') as img:
                   return transform(img).unsqueeze(0)
        except Exception as e:
             logging.error(f"Failed to load or transform image {path}: {e}")
             raise

    # load_model remains the same
    def load_model(self, path):
        """Load a trained SegNet model state_dict."""
        model = SegNet(in_channels=3, output_num_classes=2) # Adjust num_classes if needed
        logging.info(f"Loading model state_dict from: {path}")
        try:
            try:
                checkpoint = torch.load(path, map_location='cpu', weights_only=True)
                logging.info("Attempted load with weights_only=True.")
            except (RuntimeError, pickle.UnpicklingError, AttributeError) as e_safe:
                 logging.warning(f"Could not load {path} with weights_only=True ({e_safe}). Falling back to weights_only=False.")
                 logging.warning("Loading with weights_only=False allows arbitrary code execution and is insecure if the source is untrusted.")
                 checkpoint = torch.load(path, map_location='cpu', weights_only=False) # Fallback

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                logging.info("Extracted state_dict from 'model_state_dict' key.")
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                logging.info("Extracted state_dict from 'state_dict' key.")
            elif isinstance(checkpoint, dict) and all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in checkpoint.items()):
                state_dict = checkpoint
                logging.info("Checkpoint appears to be state_dict directly.")
            else:
                 logging.error(f"Loaded checkpoint dictionary has unexpected structure. Keys: {list(checkpoint.keys())}")
                 raise TypeError("Checkpoint format not recognized.")

        except Exception as load_e:
            logging.error(f"Failed to load model checkpoint from {path}: {load_e}", exc_info=True)
            raise load_e

        if all(key.startswith('module.') for key in state_dict.keys()):
             logging.info("Detected 'module.' prefix likely from DataParallel. Removing prefix.")
             state_dict = {k.partition('.')[-1]: v for k, v in state_dict.items()}
        elif any(key.startswith('seg_model.') for key in state_dict.keys()):
            logging.info("Detected 'seg_model.' prefix in state_dict keys. Attempting removal.")
            new_state_dict = {}
            corrected_keys = 0; problematic_keys = 0
            for k, v in state_dict.items():
                 if k.startswith('seg_model.'):
                      new_key = k.replace('seg_model.', '', 1)
                      new_state_dict[new_key] = v
                      corrected_keys +=1
                 else:
                      new_state_dict[k] = v
                      problematic_keys +=1
            if problematic_keys > 0: logging.warning(f"{problematic_keys} keys did not have the 'seg_model.' prefix.")
            state_dict = new_state_dict
            logging.info(f"Removed prefix from {corrected_keys} keys.")
        else:
            logging.info("No common model state_dict prefixes ('module.' or 'seg_model.') detected.")

        try:
             missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
             if missing_keys:
                  param_keys = {name for name, _ in model.named_parameters()}
                  buffer_keys = {name for name, _ in model.named_buffers()}
                  model_state_keys = param_keys.union(buffer_keys)
                  truly_missing = [k for k in missing_keys if k in model_state_keys]
                  if truly_missing: logging.warning(f"Missing keys IN MODEL STRUCTURE: {truly_missing}")
                  else: logging.info(f"Ignored missing keys (likely not part of model state): {missing_keys}")
             if unexpected_keys: logging.warning(f"Unexpected keys in state_dict (ignored): {unexpected_keys}")
             logging.info("State_dict loaded into model structure (strict=False).")
        except RuntimeError as load_e:
             logging.error(f"Error loading state_dict into model: {load_e}")
             logging.error("Architecture mismatch likely. Check SegNet definition (output_num_classes?).")
             logging.error(f"Model expects keys like: {list(model.state_dict().keys())[:5]}...")
             logging.error(f"Loaded state_dict has keys like: {list(state_dict.keys())[:5]}...")
             raise load_e

        return model.eval().to(self.device)

    # _get_layer_display_name remains the same
    def _get_layer_display_name(self, layer_full_name):
        """Creates a shorter, more readable name for plotting."""
        parts = layer_full_name.split('.')
        if len(parts) >= 2:
            stage_part = parts[0].replace("stage", "S").replace("_encoder", " Enc").replace("_decoder", " Dec")
            layer_idx = parts[1]
            return f"{stage_part} ({layer_idx})"
        return layer_full_name

    # _get_model_type_from_path remains the same
    def _get_model_type_from_path(self, model_path, model_idx):
        """Infers model type (e.g., point, scatter, hybrid) from path."""
        try:
            basename = os.path.basename(model_path)
            basename_no_ext, _ = os.path.splitext(basename)
            parts = basename_no_ext.split('_')

            if len(parts) < 2: raise IndexError("Filename too short")

            model_arch = parts[0]
            type_indicator = parts[1]

            if type_indicator == 'single' and len(parts) > 2:
                 annotation_type = parts[2].lower()
                 if annotation_type in ['point', 'scatter', 'boxes']: return annotation_type.capitalize()
                 else: raise ValueError(f"Unknown single type: {parts[2]}")

            elif type_indicator == 'hybrid':
                 hybrid_types = []
                 known_types = {'point', 'scatter', 'boxes'}
                 for part in parts[2:]:
                      part_lower = part.lower()
                      if part_lower.startswith('run'): break
                      if part_lower in known_types: hybrid_types.append(part_lower.capitalize())
                 if hybrid_types:
                      hybrid_types.sort()
                      return f"Hybrid_{'_'.join(hybrid_types)}"
                 else: return f"Hybrid_{model_idx+1}" # Fallback

            else: # Fallback for other naming conventions
                 annotation_type = parts[1].lower()
                 if annotation_type in ['point', 'scatter', 'boxes']: return annotation_type.capitalize()
                 else: raise ValueError(f"Could not determine type from parts: {parts}")

        except (IndexError, ValueError) as e:
            logging.warning(f"Could not infer model type from path: {model_path} (Error: {e}). Using generic name.")
            return f"Model_{model_idx+1}"

    # collect_data_for_comparison remains the same
    def collect_data_for_comparison(self):
        """Runs inference for each model and collects activations/predictions."""
        logging.info("Starting data collection for comparison plot...")
        self.collected_data = {}
        if not self.model_paths:
            logging.warning("No model paths provided. Skipping data collection.")
            return
        try:
             input_tensor_device = self.sample_image_tensor.to(self.device)
             logging.info(f"Input tensor moved to: {input_tensor_device.device}")
        except Exception as e:
             logging.error(f"Failed to move input tensor to device {self.device}: {e}")
             raise

        for model_idx, model_path in enumerate(self.model_paths):
            model_type_key = self._get_model_type_from_path(model_path, model_idx)
            logging.info(f"--- Processing model: {model_type_key} ({model_path}) ---")
            try:
                model = self.load_model(model_path)
                model_device = next(model.parameters()).device
                logging.info(f"Model loaded and on device: {model_device}")
                if input_tensor_device.device != model_device:
                     logging.error(f"CRITICAL: Device mismatch! Input on {input_tensor_device.device}, Model on {model_device}. Skipping model.")
                     del model; torch.cuda.empty_cache(); continue

                activation_maps = {}; hooks = []
                available_modules = dict(model.named_modules())
                valid_comparison_layers = []
                for layer_name in self.comparison_layers:
                    if layer_name in available_modules: valid_comparison_layers.append(layer_name)
                    else: logging.warning(f"Layer '{layer_name}' not found in {model_type_key}. Skipping.")

                logging.info(f"Found {len(valid_comparison_layers)} target layers in {model_type_key}: {valid_comparison_layers}")

                def get_activation(name):
                    def hook(module, input, output):
                        activation_maps[name] = output.detach().cpu()
                    return hook

                for layer_name in valid_comparison_layers:
                    try:
                        hook_handle = available_modules[layer_name].register_forward_hook(get_activation(layer_name))
                        hooks.append(hook_handle)
                    except Exception as hook_e:
                         logging.error(f"Failed to register hook for {layer_name} on {model_type_key}: {hook_e}")

                logging.info(f"Running inference for {model_type_key}...")
                with torch.no_grad():
                     output = model(input_tensor_device)
                     pred_mask = torch.argmax(output, dim=1).squeeze(0)
                     logging.info(f"Inference complete. Output shape: {output.shape}, Pred mask shape: {pred_mask.shape}")

                logging.debug(f"Removing {len(hooks)} hooks...")
                for h in hooks: h.remove()
                hooks.clear()
                logging.debug("Hooks removed.")

                self.collected_data[model_type_key] = {
                    'prediction': pred_mask, 'activations': activation_maps,
                    'valid_layers': valid_comparison_layers
                }
                logging.info(f"Successfully collected data for {model_type_key}")

            except Exception as e:
                logging.error(f"Failed to process model {model_type_key} from {model_path}: {e}", exc_info=True)
            finally:
                if 'model' in locals(): del model
                if self.device.type == 'cuda': torch.cuda.empty_cache()

        logging.info(f"Data collection finished. Collected data for {len(self.collected_data)} models.")


    def plot_comparison_grid(self):
        """Plots the collected activations and predictions in a grid."""
        if not self.collected_data:
            logging.error("No data collected. Cannot generate comparison plot.")
            return
        valid_model_keys = list(self.collected_data.keys())
        if not valid_model_keys:
             logging.error("No models had successful data collection. Cannot plot grid.")
             return

        num_models = len(valid_model_keys)
        num_cols = 1 + 1 + len(self.comparison_layers) + 1
        fig, axs = plt.subplots(num_models, num_cols,
                                figsize=(num_cols * 2.5, num_models * 2.7), squeeze=False)
        fig.suptitle(f"Model Comparison [{self.sample_image_filename} @ {self.img_size}x{self.img_size}]", fontsize=14, y=1.04)

        # --- Plot Input Image (Column 0) ---
        try:
            img_display_tensor = denormalize(self.sample_image_tensor.cpu())
            img_display_np = img_display_tensor.squeeze(0).permute(1, 2, 0).numpy()
        except Exception as e:
             logging.error(f"Failed to denormalize input image: {e}")
             img_display_np = np.array(self.sample_image_raw) / 255.0 # Fallback

        for i, model_key in enumerate(valid_model_keys):
            ax = axs[i, 0]
            ax.imshow(img_display_np)
            row_label = model_key.replace('_', ' ').title()
            ax.set_ylabel(row_label, rotation=0, fontsize=9, labelpad=45,
                          verticalalignment='center', horizontalalignment='right')
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0: ax.set_title("Input", fontsize=10)

        # --- Plot Annotated Input Image (Column 1) ---
        weak_data_for_image = self.sample_image_weak_labels
        if not weak_data_for_image: # Should have been initialized to empty dict if key missing
            logging.warning(f"Weak label data for '{self.sample_image_filename}' is empty. Annotations column will be blank.")
            weak_data_for_image = {}

        print("\n--- Annotation Data Used for Plot (Scaled to 128x128) ---")

        # Define smaller radius/width suitable for 128x128 image
        point_radius = 3
        box_width = 1

        for i, model_key in enumerate(valid_model_keys):
            ax = axs[i, 1]
            annotated_img = self.sample_image_raw.copy()
            draw = ImageDraw.Draw(annotated_img)

            point = weak_data_for_image.get('point', [])
            scatter = weak_data_for_image.get('scatter', [])
            boxes = weak_data_for_image.get('boxes', [])

            print(f"Row {i+1} ({model_key}):")
            annotation_printed = False

            # --- MODIFIED calls to use scaled drawing functions ---
            if 'point' in model_key.lower():
                if point:
                    print(f"  Drawing point ({len(point)}): Original={point}")
                    draw_point(draw, point, self.img_size, self.x_scale, self.y_scale,
                                color="red", radius=point_radius) # Pass scales & target size
                    annotation_printed = True
                else: print(f"  point requested but none found for this image.")
            if 'scatter' in model_key.lower():
                if scatter:
                    print(f"  Drawing scatter ({len(scatter)}): Original={scatter}")
                    # Use draw_point for scatter, just different color/data
                    draw_point(draw, scatter, self.img_size, self.x_scale, self.y_scale,
                                color="blue", radius=point_radius) # Pass scales & target size
                    annotation_printed = True
                else: print(f"  scatter requested but none found for this image.")
            if 'boxes' in model_key.lower():
                 if boxes:
                     print(f"  Drawing Boxes ({len(boxes)}): Original={boxes}")
                     draw_boxes(draw, boxes, self.img_size, self.x_scale, self.y_scale,
                                color="lime", width=box_width) # Pass scales & target size
                     annotation_printed = True
                 else: print(f"  Boxes requested but none found for this image.")
            # --------------------------------------------------------

            if not annotation_printed:
                 print(f"  No specific annotation type ('point', 'scatter', 'boxes') identified in model key or no data found. Showing blank annotation.")

            ax.imshow(np.array(annotated_img))
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0: ax.set_title("Annotation", fontsize=10)

        print("---------------------------------------------------------\n")

        # --- Plot Activations (Columns 2 to N+1) ---
        for j, layer_name in enumerate(self.comparison_layers):
            col_idx = j + 2
            layer_display_name = self._get_layer_display_name(layer_name)
            for i, model_key in enumerate(valid_model_keys):
                ax = axs[i, col_idx]
                model_data = self.collected_data[model_key]
                if layer_name in model_data.get('activations', {}):
                    activation = model_data['activations'][layer_name].squeeze(0)
                    if activation.dim() == 3: activation_display = activation.mean(0).numpy()
                    elif activation.dim() == 2: activation_display = activation.numpy()
                    else: # Handle unexpected shapes
                         logging.warning(f"Unexpected activation shape {activation.shape} for {layer_name} in {model_key}. Displaying zeros.")
                         h, w = model_data.get('prediction', np.zeros((self.img_size, self.img_size))).shape[-2:] # Use pred size or default
                         activation_display = np.zeros((h, w))
                    im = ax.imshow(activation_display, cmap='inferno', aspect='auto')
                else:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=9, color='grey')
                    ax.set_facecolor('#f0f0f0')
                ax.set_xticks([]); ax.set_yticks([])
                if i == 0: ax.set_title(layer_display_name, fontsize=10)

        # --- Plot Predictions (Last Column) ---
        pred_col_idx = num_cols - 1
        for i, model_key in enumerate(valid_model_keys):
            ax = axs[i, pred_col_idx]
            if 'prediction' in self.collected_data[model_key]:
                pred_mask = self.collected_data[model_key]['prediction'].numpy()
                num_classes = 2 # Adjust if needed
                cmap = plt.get_cmap('viridis', num_classes)
                im = ax.imshow(pred_mask, cmap=cmap, vmin=0, vmax=num_classes - 1, aspect='auto')
            else:
                 ax.text(0.5, 0.5, 'Error', ha='center', va='center', fontsize=9, color='red')
                 ax.set_facecolor('#fcc')
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0: ax.set_title("Prediction", fontsize=10)

        plt.tight_layout(rect=[0, 0.01, 1, 0.96])
        plt.subplots_adjust(wspace=0.1, hspace=0.2)

        save_path = os.path.join(self.output_dir, "model_annotation_activation_comparison_grid.png")
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Comparison grid saved successfully to {save_path}")
        except Exception as save_e:
             logging.error(f"Failed to save comparison grid plot: {save_e}")

        plt.close(fig)


    def generate_comparison_report(self):
        """Generates the comparison plot ONLY."""
        logging.info("Starting comparison report generation...")
        self.collect_data_for_comparison()
        self.plot_comparison_grid()
        logging.info(f"Comparison report generation finished. Output saved in {self.output_dir}")


if __name__ == "__main__":
    # --- Configuration ---
    PROJECT_ROOT = './open_ended'
    MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    # Assuming weak_labels is directly under the project root or wherever it is located
    WEAK_LABELS_PATH = os.path.join(PROJECT_ROOT, "weak_labels/weak_labels_train.pkl") # More specific path

    MODEL_PATHS = [
        os.path.join(MODELS_DIR, 'segnet_single/segnet_point_run1_best_acc.pth'),
        os.path.join(MODELS_DIR, 'segnet_single/segnet_scatter_run1_best_acc.pth'),
        os.path.join(MODELS_DIR, 'segnet_single/segnet_boxes_run1_best_acc.pth'),
        os.path.join(MODELS_DIR, 'segnet_hybrid/segnet_hybrid_point_scatter_run1_best_acc.pth'),
        os.path.join(MODELS_DIR, 'segnet_hybrid/segnet_hybrid_point_boxes_run1_best_acc.pth'),
        os.path.join(MODELS_DIR, 'segnet_hybrid/segnet_hybrid_scatter_boxes_run1_best_acc.pth'),
        os.path.join(MODELS_DIR, 'segnet_hybrid/segnet_hybrid_point_scatter_boxes_run1_best_acc.pth'),
    ]
    SAMPLE_IMAGE_PATH = "./data/images/Abyssinian_210.jpg"
    OUTPUT_DIR = "comparison_report_with_annotations_scaled" # Changed output dir name

    # --- Basic Checks ---
    print("--- Performing Pre-run Checks ---")
    abort = False
    if not os.path.exists(SAMPLE_IMAGE_PATH): print(f"ERROR: Sample image not found: {SAMPLE_IMAGE_PATH}"); abort = True
    else: print(f"Sample image found: {SAMPLE_IMAGE_PATH}")
    if not os.path.exists(WEAK_LABELS_PATH): print(f"ERROR: Weak labels file not found: {WEAK_LABELS_PATH}"); abort = True
    else: print(f"Weak labels file found: {WEAK_LABELS_PATH}")

    found_models = 0; models_to_run = []
    for i, p in enumerate(MODEL_PATHS):
        norm_p = os.path.normpath(p)
        if not os.path.exists(norm_p): print(f"WARNING: Model file {i+1} not found: {norm_p}. Skipping.")
        else: print(f"Model file {i+1} found: {norm_p}"); models_to_run.append(norm_p); found_models +=1

    if abort: print("\nERROR: Required file(s) missing. Abortinhttps://file+.vscode-resource.vscode-cdn.net/Users/yuyu/Fork_git/comp0197-cw2/comparison_report_with_annotations_scaled/model_annotation_activation_comparison_grid.png?version%3D1744312950594g."); exit()
    elif found_models == 0: print("\nERROR: No valid model files found. Aborting."); exit()
    else: print(f"\nFound {found_models} model files and support files. Proceeding with {len(models_to_run)} models..."); MODEL_PATHS = models_to_run
    print("---------------------------------")
    # --------------------

    try:
        visualizer = SegmentationVisualizer(
            model_paths=MODEL_PATHS,
            sample_image_path=SAMPLE_IMAGE_PATH,
            weak_labels_path=WEAK_LABELS_PATH,
            output_dir=OUTPUT_DIR
        )
        visualizer.generate_comparison_report()
        print(f"\n--- Script Finished Successfully ---")
        print(f"Comparison report generated in: {os.path.abspath(OUTPUT_DIR)}")

    # Keep existing detailed error handling
    except ImportError as e: print(f"\n--- SCRIPT FAILED (Import Error) ---\n{e}\nSee SegNet import details above.");
    except FileNotFoundError as e: print(f"\n--- SCRIPT FAILED (File Not Found) ---\n{e}\nCheck config paths.");
    except KeyError as e: print(f"\n--- SCRIPT FAILED (Key Error) ---\n{e}\nCheck state_dict keys or weak label keys/content."); traceback.print_exc();
    except TypeError as e: print(f"\n--- SCRIPT FAILED (Type Error) ---\n{e}\nCheck data types (loading, activations, coordinates?)."); traceback.print_exc();
    except RuntimeError as e: print(f"\n--- SCRIPT FAILED (Runtime Error) ---\n{e}\nCheck CUDA memory, state_dict mismatch, device issues."); traceback.print_exc();
    except Exception as e: print(f"\n--- SCRIPT FAILED (Unexpected Error) ---\n{type(e).__name__} - {e}\n--- Traceback ---"); traceback.print_exc(); print("-----------------");