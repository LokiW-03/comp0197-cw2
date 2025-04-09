# weak_label_generator.py
import os
import pickle
from PIL import Image
import numpy as np # Ensure numpy is imported
from skimage import measure, morphology
from tqdm import tqdm
import argparse
import glob

# Define number of scribble points per object (adjust as needed)
NUM_SCRIBBLE_POINTS_PER_OBJ = 20
# Define a fixed seed for reproducibility
RANDOM_SEED = 42

def get_binary_mask_from_trimap(trimap_path):
    """Loads trimap and converts to binary mask (1=Foreground, 0=Background/Boundary)."""
    try:
        trimap = Image.open(trimap_path).convert('L')
        trimap_np = np.array(trimap)
        mask = (trimap_np == 1).astype(np.uint8) # Foreground is 1, others 0
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
        return [1] # Binary case: only foreground class
    return []

def get_points(mask_np):
    """Returns list of centroid coordinates [(y, x)] for each object. Deterministic."""
    points = []
    # measure.label and regionprops are deterministic
    labels = measure.label(mask_np, connectivity=2)
    props = measure.regionprops(labels)
    # Sort props by label to ensure consistent order if multiple objects exist
    props = sorted(props, key=lambda p: p.label)
    for prop in props:
        if prop.label != 0:
             y, x = prop.centroid
             # Conversion to int is deterministic truncation
             points.append((int(y), int(x)))
    return points

def get_scribbles(mask_np, num_points_per_obj=NUM_SCRIBBLE_POINTS_PER_OBJ):
    """
    Generates scribble points by sampling from eroded regions.
    Made deterministic by using np.random seeded globally.
    """
    scribbles = []
    labels = measure.label(mask_np, connectivity=2)
    props = measure.regionprops(labels)
    # Sort props by label to ensure consistent order
    props = sorted(props, key=lambda p: p.label)

    erosion_size = max(1, int(min(mask_np.shape) * 0.01))
    selem = morphology.disk(erosion_size) # Deterministic

    for prop in props:
        if prop.label != 0:
            obj_mask = (labels == prop.label)
            eroded_mask = morphology.binary_erosion(obj_mask, footprint=selem) # Deterministic
            coords = np.argwhere(eroded_mask) # Deterministic, but order might vary slightly based on impl.

            if len(coords) > 0:
                 # Sort coordinates to ensure consistent input order for sampling, just in case.
                 # Sorting by y then x.
                 coords = coords[np.lexsort((coords[:, 1], coords[:, 0]))]

                 num_to_sample = min(num_points_per_obj, len(coords))

                 indices = np.random.choice(len(coords), size=num_to_sample, replace=False)

                 # Sort indices to ensure the order of points added to scribbles is consistent
                 indices = np.sort(indices)

                 sampled_coords = coords[indices]
                 # Add points; order is now determined by sorted indices.
                 scribbles.extend([(y, x) for y, x in sampled_coords])

    return scribbles


def get_bounding_boxes(mask_np):
    """Returns list of bounding boxes [(ymin, xmin, ymax, xmax)] for each object. Deterministic."""
    boxes = []
    labels = measure.label(mask_np, connectivity=2)
    props = measure.regionprops(labels)
    # Sort props by label to ensure consistent order
    props = sorted(props, key=lambda p: p.label)
    for prop in props:
         if prop.label != 0:
              # prop.bbox is deterministic
              boxes.append(prop.bbox)
    return boxes


def main(args):
    # --- Seed for Determinism ---
    # Set a fixed seed for NumPy's random number generator.
    # This ensures that operations like np.random.choice (used in get_scribbles)
    # produce the same results every time the script is run with the same inputs.
    np.random.seed(RANDOM_SEED)
    print(f"Using fixed random seed: {RANDOM_SEED} for reproducibility.")
    # ---------------------------

    print("Generating weak labels...")
    data_dir = args.data_dir
    output_file = args.output_file

    image_dir = os.path.join(data_dir, 'images')
    trimap_dir = os.path.join(data_dir, 'annotations', 'trimaps')

    # Use sorted glob to ensure consistent file processing order
    image_files = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))

    # Determine Training Split Files (consistent split logic)
    num_images = len(image_files)
    num_train = int(num_images * 0.7)
    train_image_files = image_files[:num_train]
    print(f"Processing {len(train_image_files)} images for training weak labels.")

    all_weak_labels = {}

    # Use sorted list train_image_files ensures deterministic iteration order
    for img_path in tqdm(train_image_files, desc="Processing images"):
        img_filename = os.path.basename(img_path)
        trimap_path = os.path.join(trimap_dir, img_filename.replace('.jpg', '.png'))

        binary_mask = get_binary_mask_from_trimap(trimap_path)

        if binary_mask is not None and np.any(binary_mask):
            try:
                 # Call deterministic functions (assuming np.random is seeded)
                 tags = get_tags(binary_mask)
                 points = get_points(binary_mask)
                 scribbles = get_scribbles(binary_mask, num_points_per_obj=args.num_scribble_points) # Pass arg here
                 boxes = get_bounding_boxes(binary_mask)

                 all_weak_labels[img_filename] = {
                      'tags': tags,
                      'points': points,
                      'scribbles': scribbles,
                      'boxes': boxes
                 }
            except Exception as e:
                 print(f"Error processing {img_filename}: {e}")
                 all_weak_labels[img_filename] = {'tags': [], 'points': [], 'scribbles': [], 'boxes': []}
        else:
             all_weak_labels[img_filename] = {'tags': [], 'points': [], 'scribbles': [], 'boxes': []}

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save the generated labels using pickle
    # Pickle is deterministic for the same object structure and data
    with open(output_file, 'wb') as f:
        pickle.dump(all_weak_labels, f)

    print(f"Weak labels saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Weak Labels for Oxford Pets")
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to the root Oxford Pets dataset directory')
    parser.add_argument('--output_file', type=str, default='./weak_labels/weak_labels_train.pkl',
                        help='Path to save the generated weak labels pickle file')
    # Add argument for num_scribble_points
    parser.add_argument('--num_scribble_points', type=int, default=NUM_SCRIBBLE_POINTS_PER_OBJ,
                        help='Number of scribble points to generate per object')
    # Optional: Add argument for seed
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help='Random seed for deterministic scribble generation')

    args = parser.parse_args()

    # Update global constants if seed/points provided via args
    RANDOM_SEED = args.seed
    NUM_SCRIBBLE_POINTS_PER_OBJ = args.num_scribble_points

    main(args)