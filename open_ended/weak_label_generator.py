#weak_label_generator.py
import os
import pickle
from PIL import Image
import numpy as np
from skimage import measure, morphology
from tqdm import tqdm # Using tqdm here for convenience, replace with print if needed
import argparse
import glob

# Define number of scribble points per object (adjust as needed)
NUM_SCRIBBLE_POINTS_PER_OBJ = 20

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
    """Returns list of unique classes present (just [1] if foreground exists)."""
    if np.any(mask_np == 1):
        return [1] # Binary case: only foreground class
    return []

def get_points(mask_np):
    """Returns list of centroid coordinates [(y, x)] for each object."""
    points = []
    labels = measure.label(mask_np, connectivity=2) # Find connected components
    props = measure.regionprops(labels)
    for prop in props:
        if prop.label != 0: # Ignore background label if generated
             # Centroid returns (row, col) which corresponds to (y, x)
             y, x = prop.centroid
             points.append((int(y), int(x)))
    return points

def get_scribbles(mask_np, num_points_per_obj=NUM_SCRIBBLE_POINTS_PER_OBJ):
    """Generates scribble points by sampling from eroded regions."""
    scribbles = []
    labels = measure.label(mask_np, connectivity=2)
    props = measure.regionprops(labels)

    # Determine erosion structure size (can be adjusted)
    erosion_size = max(1, int(min(mask_np.shape) * 0.01)) # 1% of min dimension
    selem = morphology.disk(erosion_size) # Morphological element

    for prop in props:
        if prop.label != 0:
            # Create a mask for the current object only
            obj_mask = (labels == prop.label)
            # Erode the object mask
            eroded_mask = morphology.binary_erosion(obj_mask, footprint=selem)
            # Find coordinates within the eroded region
            coords = np.argwhere(eroded_mask) # List of [y, x]

            if len(coords) > 0:
                 # Sample points randomly from the eroded coordinates
                 num_to_sample = min(num_points_per_obj, len(coords))
                 indices = np.random.choice(len(coords), size=num_to_sample, replace=False)
                 sampled_coords = coords[indices]
                 scribbles.extend([(y, x) for y, x in sampled_coords]) # Add as (y, x) tuples

    return scribbles


def get_bounding_boxes(mask_np):
    """Returns list of bounding boxes [(ymin, xmin, ymax, xmax)] for each object."""
    boxes = []
    labels = measure.label(mask_np, connectivity=2)
    props = measure.regionprops(labels)
    for prop in props:
         if prop.label != 0:
              # bbox returns (min_row, min_col, max_row, max_col) -> (ymin, xmin, ymax, xmax)
              boxes.append(prop.bbox)
    return boxes


def main(args):
    print("Generating weak labels...")
    data_dir = args.data_dir
    output_file = args.output_file

    image_dir = os.path.join(data_dir, 'images')
    trimap_dir = os.path.join(data_dir, 'annotations', 'trimaps')
    image_files = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))

    # --- Determine Training Split Files ---
    # Using the same split logic as in Dataset class for consistency
    num_images = len(image_files)
    num_train = int(num_images * 0.7)
    train_image_files = image_files[:num_train]
    print(f"Processing {len(train_image_files)} images for training weak labels.")
    # ---

    all_weak_labels = {}

    for img_path in tqdm(train_image_files, desc="Processing images"):
        img_filename = os.path.basename(img_path)
        trimap_path = os.path.join(trimap_dir, img_filename.replace('.jpg', '.png'))

        binary_mask = get_binary_mask_from_trimap(trimap_path)

        if binary_mask is not None and np.any(binary_mask): # Process only if mask is valid and has foreground
            try:
                 tags = get_tags(binary_mask)
                 points = get_points(binary_mask)
                 scribbles = get_scribbles(binary_mask)
                 boxes = get_bounding_boxes(binary_mask)

                 all_weak_labels[img_filename] = {
                      'tags': tags,
                      'points': points, # List of (y, x)
                      'scribbles': scribbles, # List of (y, x)
                      'boxes': boxes # List of (minY, minX, maxY, maxX)
                 }
            except Exception as e:
                 print(f"Error processing {img_filename}: {e}")
                 # Store empty labels for this image if processing failed
                 all_weak_labels[img_filename] = {
                      'tags': [], 'points': [], 'scribbles': [], 'boxes': []
                 }
        else:
             # Store empty labels if mask is empty or invalid
            all_weak_labels[img_filename] = {
                'tags': [], 'points': [], 'scribbles': [], 'boxes': []
            }


    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save the generated labels
    with open(output_file, 'wb') as f:
        pickle.dump(all_weak_labels, f)

    print(f"Weak labels saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Weak Labels for Oxford Pets")
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to the root Oxford Pets dataset directory')
    parser.add_argument('--output_file', type=str, default='./weak_labels/weak_labels_train.pkl',
                        help='Path to save the generated weak labels pickle file')
    args = parser.parse_args()
    main(args)