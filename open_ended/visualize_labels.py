# visualize_labels.py
import os
import pickle
import random
from PIL import Image, ImageDraw
import argparse
import numpy as np # Only needed if drawing complex shapes, PIL handles tuples

# --- Configuration ---
DEFAULT_WEAK_LABEL_FILE = './weak_labels/weak_labels_train.pkl'
DEFAULT_IMAGE_DIR = './data/images' # Path to ORIGINAL images
DEFAULT_OUTPUT_DIR = './visualization_output'
DEFAULT_SEED = 7 # Use same seed for reproducibility if needed

# --- Annotation Drawing Settings ---
BOX_COLOR = (255, 0, 0)       # Red
POINT_COLOR = (0, 255, 0)     # Lime Green
SCATTER_COLOR = (0, 0, 255)   # Blue
FG_SCRIBBLE_COLOR = (255, 255, 0) # Yellow
BG_SCRIBBLE_COLOR = (255, 0, 255) # Magenta

POINT_RADIUS = 3  # Radius for drawing points/scatter/scribble pixels
BOX_WIDTH = 2     # Line thickness for boxes
# --- End Configuration ---

def draw_annotations(image, annotations):
    """Draws annotations onto the PIL image object."""
    draw = ImageDraw.Draw(image)

    # 1. Draw Bounding Boxes
    if 'boxes' in annotations and annotations['boxes']:
        for box in annotations['boxes']:
            # box format: (ymin, xmin, ymax, xmax)
            ymin, xmin, ymax, xmax = box
            # PIL draw rectangle uses [(x0, y0), (x1, y1)]
            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=BOX_COLOR, width=BOX_WIDTH)

    # Helper to draw points/pixels as small ellipses
    def draw_point_marker(y, x, color, radius):
         # PIL uses (x,y) coordinates
        bbox = [(x - radius, y - radius), (x + radius, y + radius)]
        draw.ellipse(bbox, fill=color, outline=color)

    # 2. Draw 'point' annotations
    if 'points' in annotations and annotations['points']:
        for (y, x) in annotations['points']:
            draw_point_marker(y, x, POINT_COLOR, POINT_RADIUS)


    # 4. Draw 'scribbles' annotations
    if 'scribbles' in annotations:
        # Draw foreground scribble points
        if 'foreground' in annotations['scribbles'] and annotations['scribbles']['foreground']:
            for (y, x) in annotations['scribbles']['foreground']:
                draw_point_marker(y, x, FG_SCRIBBLE_COLOR, POINT_RADIUS)
        # Draw background scribble points
        if 'background' in annotations['scribbles'] and annotations['scribbles']['background']:
             for (y, x) in annotations['scribbles']['background']:
                draw_point_marker(y, x, BG_SCRIBBLE_COLOR, POINT_RADIUS)

    return image


def main(args):
    # Set seed for reproducible random choice if desired
    random.seed(args.seed)

    # --- Load Data ---
    if not os.path.exists(args.label_file):
        print(f"Error: Weak label file not found at {args.label_file}")
        return

    print(f"Loading weak labels from: {args.label_file}")
    with open(args.label_file, 'rb') as f:
        all_weak_labels = pickle.load(f)

    if not all_weak_labels:
        print("Error: No labels found in the pickle file.")
        return

    # --- Select Image ---
    available_images = list(all_weak_labels.keys())
    if not available_images:
        print("Error: No image keys found in the loaded labels.")
        return

    chosen_img_filename = random.choice(available_images)
    print(f"Randomly selected image: {chosen_img_filename}")

    # --- Load Image ---
    img_path = os.path.join(args.image_dir, chosen_img_filename)
    if not os.path.exists(img_path):
        print(f"Error: Image file not found at {img_path}")
        # Attempt common extensions if original failed (e.g., if pkl key has .jpg but file is .png)
        base_name = os.path.splitext(chosen_img_filename)[0]
        found = False
        for ext in ['.jpg', '.png', '.jpeg']:
            potential_path = os.path.join(args.image_dir, base_name + ext)
            if os.path.exists(potential_path):
                img_path = potential_path
                print(f"Found image as: {img_path}")
                found = True
                break
        if not found:
            print(f"Error: Could not find image file corresponding to {chosen_img_filename} in {args.image_dir}")
            return

    try:
        image = Image.open(img_path).convert('RGB') # Ensure RGB for color drawing
    except Exception as e:
        print(f"Error opening image {img_path}: {e}")
        return

    # --- Get Annotations ---
    annotations = all_weak_labels[chosen_img_filename]

    # --- Draw Annotations ---
    print("Drawing annotations...")
    annotated_image = draw_annotations(image.copy(), annotations) # Draw on a copy

    # --- Save Output ---
    os.makedirs(args.output_dir, exist_ok=True)
    output_filename = f"annotated_{os.path.splitext(chosen_img_filename)[0]}.png"
    output_path = os.path.join(args.output_dir, output_filename)

    try:
        annotated_image.save(output_path)
        print("-" * 30)
        print("Annotation Summary:")
        print(f"  Image: {chosen_img_filename}")
        print(f"  Bounding Boxes: {len(annotations.get('boxes', []))}")
        print(f"  Points: {len(annotations.get('point', []))}")
        print(f"  FG Scribble Points: {len(annotations.get('scribbles', {}).get('foreground', []))}")
        print(f"  BG Scribble Points: {len(annotations.get('scribbles', {}).get('background', []))}")
        print("-" * 30)
        print(f"Annotated image saved to: {output_path}")
    except Exception as e:
        print(f"Error saving annotated image to {output_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Weak Labels on a Random Image")
    parser.add_argument('--label_file', type=str, default=DEFAULT_WEAK_LABEL_FILE,
                        help='Path to the generated weak labels pickle file')
    parser.add_argument('--image_dir', type=str, default=DEFAULT_IMAGE_DIR,
                        help='Path to the directory containing the original images')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Directory to save the visualized output image')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help='Random seed for selecting the image')

    args = parser.parse_args()
    main(args)