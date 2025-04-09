# weak_label_generator.py
import os
import pickle
from PIL import Image
import numpy as np
from skimage import measure, morphology
from tqdm import tqdm
import argparse
import glob

# --- Configuration ---
# Define DEFAULT number of points to generate for the 'points' supervision type
DEFAULT_NUM_POINTS_PER_OBJ = 10 # <--- Changed Default
# Define number of scribble points per object (for the 'scribbles' key)
DEFAULT_NUM_SCRIBBLE_POINTS_PER_OBJ = 20
# Define a fixed seed for reproducibility
RANDOM_SEED = 42
# --- End Configuration ---

def get_binary_mask_from_trimap(trimap_path):
    """Loads trimap and converts to binary mask (1=Foreground, 0=Background/Boundary)."""
    try:
        trimap = Image.open(trimap_path).convert('L')
        trimap_np = np.array(trimap)
        mask = (trimap_np == 1).astype(np.uint8)
        return mask
    except FileNotFoundError:
        print(f"Warning: Trimap not found {trimap_path}")
        return None
    except Exception as e:
        print(f"Error loading trimap {trimap_path}: {e}")
        return None

def get_tags(mask_np):
    """Returns list of unique classes present (just [1] if foreground exists). Deterministic."""
    if np.any(mask_np == 1):
        return [1]
    return []

# --- MODIFIED get_points Function ---
def get_points(mask_np, num_points_per_obj=DEFAULT_NUM_POINTS_PER_OBJ):
    """
    Generates a specified number of points sampled from each object mask.
    Replaces the previous centroid calculation. Deterministic due to global seeding.
    """
    points = []
    labels = measure.label(mask_np, connectivity=2)
    props = measure.regionprops(labels)
    # Sort props by label to ensure consistent order
    props = sorted(props, key=lambda p: p.label)

    for prop in props:
        if prop.label != 0: # Ignore background
            # Create a mask for the current object only
            obj_mask = (labels == prop.label)
            # Find coordinates *within the object mask* (not eroded)
            coords = np.argwhere(obj_mask) # List of [y, x]

            if len(coords) > 0:
                 # Sort coordinates to ensure consistent input order for sampling
                 coords = coords[np.lexsort((coords[:, 1], coords[:, 0]))] # Sort by y, then x

                 # Determine how many points to sample for this specific object
                 num_to_sample = min(num_points_per_obj, len(coords))

                 # Sample points randomly (uses globally seeded np.random)
                 indices = np.random.choice(len(coords), size=num_to_sample, replace=False)

                 # Sort indices to ensure consistent order of selected points
                 indices = np.sort(indices)

                 sampled_coords = coords[indices]
                 # Add points as (y, x) tuples
                 points.extend([(int(y), int(x)) for y, x in sampled_coords]) # Ensure ints

    return points
# --- End MODIFIED get_points Function ---

def get_scribbles(mask_np, num_points_per_obj=DEFAULT_NUM_SCRIBBLE_POINTS_PER_OBJ):
    """
    Generates scribble points by sampling from ERODED regions.
    Used for the 'scribbles' key. Made deterministic by using np.random seeded globally.
    """
    scribbles = []
    labels = measure.label(mask_np, connectivity=2)
    props = measure.regionprops(labels)
    props = sorted(props, key=lambda p: p.label)

    # Erosion for scribbles (distinct from get_points)
    erosion_size = max(1, int(min(mask_np.shape) * 0.01))
    selem = morphology.disk(erosion_size)

    for prop in props:
        if prop.label != 0:
            obj_mask = (labels == prop.label)
            eroded_mask = morphology.binary_erosion(obj_mask, footprint=selem)
            coords = np.argwhere(eroded_mask)

            if len(coords) > 0:
                 coords = coords[np.lexsort((coords[:, 1], coords[:, 0]))]
                 num_to_sample = min(num_points_per_obj, len(coords))
                 indices = np.random.choice(len(coords), size=num_to_sample, replace=False)
                 indices = np.sort(indices)
                 sampled_coords = coords[indices]
                 scribbles.extend([(int(y), int(x)) for y, x in sampled_coords]) # Ensure ints

    return scribbles


def get_bounding_boxes(mask_np):
    """Returns list of bounding boxes [(ymin, xmin, ymax, xmax)] for each object. Deterministic."""
    boxes = []
    labels = measure.label(mask_np, connectivity=2)
    props = measure.regionprops(labels)
    props = sorted(props, key=lambda p: p.label)
    for prop in props:
         if prop.label != 0:
              boxes.append(prop.bbox)
    return boxes


def main(args):
    # --- Seed for Determinism ---
    # Use the seed provided via args or the default
    np.random.seed(args.seed)
    print(f"Using fixed random seed: {args.seed} for reproducibility.")
    # ---------------------------

    print("Generating weak labels...")
    data_dir = args.data_dir
    output_file = args.output_file

    image_dir = os.path.join(data_dir, 'images')
    trimap_dir = os.path.join(data_dir, 'annotations', 'trimaps')
    image_files = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))

    num_images = len(image_files)
    num_train = int(num_images * 0.7)
    train_image_files = image_files[:num_train]
    print(f"Processing {len(train_image_files)} images for training weak labels.")

    all_weak_labels = {}

    for img_path in tqdm(train_image_files, desc="Processing images"):
        img_filename = os.path.basename(img_path)
        trimap_path = os.path.join(trimap_dir, img_filename.replace('.jpg', '.png'))

        binary_mask = get_binary_mask_from_trimap(trimap_path)

        if binary_mask is not None and np.any(binary_mask):
            try:
                 tags = get_tags(binary_mask)
                 # --- Call MODIFIED get_points with specified number ---
                 points = get_points(binary_mask, num_points_per_obj=args.num_points)
                 # --- Call get_scribbles with its own specified number ---
                 scribbles = get_scribbles(binary_mask, num_points_per_obj=args.num_scribble_points)
                 boxes = get_bounding_boxes(binary_mask)

                 all_weak_labels[img_filename] = {
                      'tags': tags,
                      'points': points,     # Now contains list of ~10 sampled points
                      'scribbles': scribbles, # Still contains list of ~20 eroded scribble points
                      'boxes': boxes
                 }
            except Exception as e:
                 print(f"Error processing {img_filename}: {e}")
                 all_weak_labels[img_filename] = {'tags': [], 'points': [], 'scribbles': [], 'boxes': []}
        else:
             all_weak_labels[img_filename] = {'tags': [], 'points': [], 'scribbles': [], 'boxes': []}

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'wb') as f:
        pickle.dump(all_weak_labels, f)

    print(f"Weak labels saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Weak Labels for Oxford Pets")
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to the root Oxford Pets dataset directory')
    parser.add_argument('--output_file', type=str, default='./weak_labels/weak_labels_train.pkl',
                        help='Path to save the generated weak labels pickle file')
    # --- Arguments to control point counts ---
    parser.add_argument('--num_points', type=int, default=DEFAULT_NUM_POINTS_PER_OBJ,
                        help="Number of points to generate per object for the 'points' key")
    parser.add_argument('--num_scribble_points', type=int, default=DEFAULT_NUM_SCRIBBLE_POINTS_PER_OBJ,
                        help="Number of points to generate per object for the 'scribbles' key")
    # ----------------------------------------
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help='Random seed for deterministic point generation')

    args = parser.parse_args()

    # No longer need to update global constants here, using args directly in main()

    main(args)