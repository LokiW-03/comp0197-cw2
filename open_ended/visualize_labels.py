# visualize_labels.py
import os
import pickle
import random
from PIL import Image, ImageDraw
import argparse
import numpy as np 

# --- Configuration ---
DEFAULT_WEAK_LABEL_FILE = './open_ended/weak_labels/weak_labels_train.pkl'
DEFAULT_IMAGE_DIR = './data/images' # Path to ORIGINAL images
DEFAULT_TRIMAP_DIR = './data/annotations/trimaps' # <<< ADDED: Path to ORIGINAL trimaps
DEFAULT_OUTPUT_DIR = './open_ended/visualization_output'
DEFAULT_SEED = 7

# --- Annotation Drawing Settings ---
BOX_COLOR = (255, 0, 0, 255)       # Red (RGBA)
POINT_COLOR = (0, 255, 0, 255)     # Lime Green (RGBA)
FG_SCRIBBLE_COLOR = (255, 255, 0, 255) # Yellow (RGBA)
BG_SCRIBBLE_COLOR = (255, 0, 255, 255) # Magenta (RGBA)
# <<< ADDED: Mask Overlay Settings >>>
MASK_OVERLAY_COLOR = (0, 0, 255, 100) # Semi-transparent Blue (R, G, B, Alpha)
DEFAULT_FOREGROUND_VALUE = 1 # <<< ADDED: The value in trimap indicating foreground
DEFAULT_TRIMAP_EXT = 'png'   # <<< ADDED: Default trimap extension

POINT_RADIUS = 3  # Radius for drawing points/scribble pixels
BOX_WIDTH = 2     # Line thickness for boxes
# --- End Configuration ---

# Standard EXIF Orientation tag ID
ORIENTATION_TAG_ID = 274 # Hex: 0x0112
DEFAULT_ORIENTATION = 1 # Value for standard, top-left orientation

def get_exif_orientation(image_path):
    """
    Reads an image file and returns its EXIF orientation value.

    Args:
        image_path (str): The path to the image file.

    Returns:
        int: The EXIF orientation value (typically 1-8).
             Returns DEFAULT_ORIENTATION (1) if the orientation tag is missing,
             if there's no EXIF data, or if an error occurs during EXIF reading.
        None: If the file cannot be found or opened as an image.
    """
    try:
        img = Image.open(image_path)

        # Use getexif() - returns an Exif object (dict-like) or None
        exif_data = img.getexif()

        if exif_data:
            # Look for the orientation tag, return DEFAULT_ORIENTATION if not found
            orientation = exif_data.get(ORIENTATION_TAG_ID, DEFAULT_ORIENTATION)
            return orientation
        else:
            # No EXIF dictionary found at all
            # print(f"INFO: No EXIF data found in {os.path.basename(image_path)}") # Optional: uncomment for info
            return DEFAULT_ORIENTATION

    except FileNotFoundError:
        print(f"ERROR: File not found: {image_path}")
        return None
    except Exception as e:
        # Catch other potential errors during file opening or EXIF parsing
        # (e.g., file is not a valid image, corrupted EXIF)
        print(f"WARNING: Error processing EXIF for {os.path.basename(image_path)}: {e}. Assuming default orientation.")
        return DEFAULT_ORIENTATION # Return default as a safe fallback

# <<< ADDED FUNCTION: To load and process the mask >>>
def load_and_get_mask(trimap_path, foreground_value):
    """
    Loads the trimap, converts to grayscale, and creates a binary mask
    based on the specified foreground value.
    Returns the binary mask (numpy array HxW, dtype=uint8) or None on error.
    """
    try:
        if not os.path.exists(trimap_path):
            print(f"Error: Trimap file not found at {trimap_path}")
            return None
        
        
        trimap_orientation = get_exif_orientation(trimap_path)
        print(trimap_orientation)
        trimap_pil = Image.open(trimap_path)
        # trimap_pil = ImageOps.exif_transpose(trimap_pil)
        
        trimap_pil = trimap_pil.convert('L') # Convert to grayscale *after* transposing
        original_size = trimap_pil.size # Get size *after* potential transpose
        trimap_np = np.array(trimap_pil)
        mask = (trimap_np == 1).astype(np.uint8)

        # trimap = Image.open(trimap_path).convert('L')
        # trimap_np = np.array(trimap)

        # # Create mask based on the specified foreground value
        # mask = (trimap_np == foreground_value).astype(np.uint8) # 0 or 1

        if not np.any(mask):
            print(f"Warning: No foreground pixels with value {foreground_value} found in trimap: {trimap_path}")
            # Return an empty mask of the correct shape instead of None
            # return np.zeros_like(trimap_np, dtype=np.uint8)
            # Or maybe signal caller this specific issue? For now, let's return it empty
            return mask # It's valid, just empty

        return mask
    except FileNotFoundError:
        # This case is handled above, but kept for safety
        print(f"Error: Trimap file not found: {trimap_path}")
        return None
    except Exception as e:
        print(f"Error processing trimap {trimap_path}: {e}")
        return None

# <<< MODIFIED FUNCTION: To accept and draw the mask >>>
def draw_annotations(image, annotations, mask_np=None):
    """
    Draws the mask overlay (if provided) and annotations onto the PIL image object.
    Assumes image is in RGB or RGBA mode.
    """
    # Ensure image is RGBA for overlay compositing
    if image.mode != 'RGBA':
        image = image.convert('RGBA')

    # --- Draw Mask Overlay (if mask is provided) ---
    if mask_np is not None and mask_np.shape[0] == image.height and mask_np.shape[1] == image.width:
        try:
            # Create an RGBA overlay image where mask=1 is the overlay color, else transparent
            overlay = np.zeros((image.height, image.width, 4), dtype=np.uint8)
            overlay[mask_np == 1] = MASK_OVERLAY_COLOR # Apply color where mask is 1

            # Convert overlay numpy array to PIL Image
            mask_overlay_pil = Image.fromarray(overlay, 'RGBA')

            # Composite the overlay onto the image using alpha blending
            # The mask_overlay_pil itself acts as the mask due to its alpha channel
            image.paste(mask_overlay_pil, (0, 0), mask_overlay_pil)

        except Exception as e:
            print(f"Warning: Failed to draw mask overlay: {e}")
    elif mask_np is not None:
         print(f"Warning: Mask dimensions ({mask_np.shape}) do not match image dimensions ({image.height}, {image.width}). Skipping mask overlay.")


    # --- Now draw other annotations ON TOP of the image + overlay ---
    draw = ImageDraw.Draw(image) # Draw directly on the (potentially overlaid) image

    # Helper to draw points/pixels as small ellipses centered at (x, y)
    def draw_point_marker(x, y, color, radius):
        bbox = [
            (x - radius, y - radius), # Top-left (xmin, ymin)
            (x + radius, y + radius)  # Bottom-right (xmax, ymax)
        ]
        # Use RGBA color for drawing
        draw.ellipse(bbox, fill=color, outline=color)

    # 1. Draw Bounding Boxes
    # Expected format: list of (xmin, ymin, xmax, ymax)
    if 'boxes' in annotations and annotations['boxes']:
        for box in annotations['boxes']:
            try:
                xmin, ymin, xmax, ymax = map(int, box)
                draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=BOX_COLOR[:3], width=BOX_WIDTH) # Use RGB for outline
            except (ValueError, TypeError) as e:
                print(f"Warning: Skipping invalid box format {box}: {e}")

    # 2. Draw 'points' annotations
    # Expected format: list of (x, y)
    if 'points' in annotations and annotations['points']:
        for point in annotations['points']:
            try:
                x, y = map(int, point)
                draw_point_marker(x, y, POINT_COLOR, POINT_RADIUS)
            except (ValueError, TypeError) as e:
                 print(f"Warning: Skipping invalid point format {point}: {e}")

    # 3. Draw 'scribbles' annotations
    # Expected format: dict with 'foreground'/'background' lists of (x, y)
    if 'scribbles' in annotations:
        scribble_data = annotations['scribbles']
        if 'foreground' in scribble_data and scribble_data['foreground']:
            for point in scribble_data['foreground']:
                try:
                    x, y = map(int, point)
                    draw_point_marker(x, y, FG_SCRIBBLE_COLOR, POINT_RADIUS)
                except (ValueError, TypeError) as e:
                    print(f"Warning: Skipping invalid fg scribble point format {point}: {e}")
        if 'background' in scribble_data and scribble_data['background']:
             for point in scribble_data['background']:
                try:
                    x, y = map(int, point)
                    draw_point_marker(x, y, BG_SCRIBBLE_COLOR, POINT_RADIUS)
                except (ValueError, TypeError) as e:
                    print(f"Warning: Skipping invalid bg scribble point format {point}: {e}")

    return image


def main(args):
    random.seed(args.seed)

    # --- Load Weak Labels ---
    if not os.path.exists(args.label_file):
        print(f"Error: Weak label file not found at {args.label_file}")
        return
    print(f"Loading weak labels from: {args.label_file}")
    try:
        with open(args.label_file, 'rb') as f:
            all_weak_labels = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle file {args.label_file}: {e}")
        return
    if not isinstance(all_weak_labels, dict) or not all_weak_labels:
        print("Error: Loaded labels data is not a non-empty dictionary.")
        return

    # --- Select Image ---
    available_images = list(all_weak_labels.keys())
    if not available_images:
        print("Error: No image keys found in the loaded labels.")
        return
    chosen_img_filename = random.choice(available_images)
    print(f"Randomly selected image: {chosen_img_filename}")

    # --- Load Original Image ---
    img_path = os.path.join(args.image_dir, chosen_img_filename)
    base_name = os.path.splitext(chosen_img_filename)[0]
    found_img = False
    if os.path.exists(img_path):
        found_img = True
    else:
        print(f"Warning: Image file not found directly at {img_path}. Trying common extensions...")
        for ext in ['.jpg', '.png', '.jpeg', '.bmp', '.gif']:
            potential_path = os.path.join(args.image_dir, base_name + ext)
            if os.path.exists(potential_path):
                img_path = potential_path
                print(f"Found image as: {img_path}")
                found_img = True
                break
    if not found_img:
        print(f"Error: Could not find image file corresponding to {chosen_img_filename} in {args.image_dir}")
        return
    try:
        image = Image.open(img_path).convert('RGB') # Start with RGB
    except Exception as e:
        print(f"Error opening image {img_path}: {e}")
        return

    # --- Load Corresponding Mask (from Trimap) ---
    mask_np = None # Initialize mask as None
    trimap_filename = base_name + '.' + args.trimap_ext
    trimap_path = os.path.join(args.trimap_dir, trimap_filename)
    print(f"Attempting to load mask from trimap: {trimap_path}")
    mask_np = load_and_get_mask(trimap_path, args.foreground_value)
    
    if mask_np is None:
        print(f"Warning: Could not load or process mask for {chosen_img_filename}. Proceeding without mask overlay.")
        # Proceeding, mask_np remains None

    # --- Get Annotations ---
    if chosen_img_filename not in all_weak_labels:
         print(f"Error: Key {chosen_img_filename} not found in labels dictionary.")
         return
    annotations = all_weak_labels[chosen_img_filename]
    if not isinstance(annotations, dict):
        print(f"Error: Annotation data for {chosen_img_filename} is not a dictionary.")
        return

    # --- Draw Annotations (and mask if loaded) ---
    print("Drawing annotations...")
    # Pass the loaded mask (or None) to the drawing function
    annotated_image = draw_annotations(image.copy(), annotations, mask_np)

    # --- Save Output ---
    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directory {args.output_dir}: {e}")
        return
    output_filename = f"annotated_{base_name}.png" # Save as PNG to preserve transparency
    output_path = os.path.join(args.output_dir, output_filename)
    try:
        # Convert back to RGB if no transparency was actually needed (e.g., no mask)
        # or just save as PNG which handles RGBA well. Let's stick with PNG.
        annotated_image.save(output_path)
        print("-" * 30)
        print("Annotation Summary:")
        print(f"  Image: {chosen_img_filename}")
        if mask_np is not None:
            print(f"  Mask Overlay: Drawn (using value {args.foreground_value} from {trimap_filename})")
        else:
            print(f"  Mask Overlay: Not drawn (trimap/mask issue)")
        print(f"  Bounding Boxes: {len(annotations.get('boxes', []))}")
        print(f"  Points: {len(annotations.get('points', []))}")
        print(f"  FG Scribble Points: {len(annotations.get('scribbles', {}).get('foreground', []))}")
        print(f"  BG Scribble Points: {len(annotations.get('scribbles', {}).get('background', []))}")
        print("-" * 30)
        print(f"Annotated image saved to: {output_path}")
    except Exception as e:
        print(f"Error saving annotated image to {output_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Weak Labels and Mask Overlay on a Random Image")
    parser.add_argument('--label_file', type=str, default=DEFAULT_WEAK_LABEL_FILE,
                        help='Path to the generated weak labels pickle file')
    parser.add_argument('--image_dir', type=str, default=DEFAULT_IMAGE_DIR,
                        help='Path to the directory containing the original images')
    # <<< ADDED Arguments for Mask Loading >>>
    parser.add_argument('--trimap_dir', type=str, default=DEFAULT_TRIMAP_DIR,
                        help='Path to the directory containing the original trimaps')
    parser.add_argument('--trimap_ext', type=str, default=DEFAULT_TRIMAP_EXT,
                        help='Extension of trimap files (e.g., png)')
    parser.add_argument('--foreground_value', type=int, default=DEFAULT_FOREGROUND_VALUE,
                        help='Pixel value in the trimap representing definite foreground')
    # <<< End Added Arguments >>>
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Directory to save the visualized output image')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help='Random seed for selecting the image')

    args = parser.parse_args()
    main(args)