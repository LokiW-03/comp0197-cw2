import os
import pickle
from PIL import Image
import numpy as np
from skimage import measure, morphology
from tqdm import tqdm
import argparse
import glob
import random
import logging # Import the logging module

# --- Configuration ---
DEFAULT_NUM_POINT_PER_OBJ = 10 # Changed back to 10 as per original code before user changed it
DEFAULT_NUM_SCATTER_POINT_PER_OBJ = 20
DEFAULT_NUM_SCRIBBLES_PER_TYPE = 3
DEFAULT_SCRIBBLE_LENGTH = 15
DEFAULT_SCRIBBLE_EROSION_FACTOR = 0.02
RANDOM_SEED = 42
# --- End Configuration ---

# --- Logger Setup ---
log = logging.getLogger(__name__)
# --- End Logger Setup ---

def get_binary_mask_from_trimap(trimap_path):
    """Loads trimap and converts to binary mask (1=Foreground, 0=Background/Boundary)."""
    try:
        trimap = Image.open(trimap_path).convert('L')
        trimap_np = np.array(trimap)
        unique_vals = np.unique(trimap_np)
        log.debug(f"(get_binary_mask): Trimap unique values in {os.path.basename(trimap_path)}: {unique_vals}")

        # !! CRITICAL: ADJUST THIS VALUE if your foreground is not 1 !!
        # Common values: 1 (Oxford default), 255 (White)
        foreground_value = 1 # Keep assumption or make configurable? Assuming 1 for now.
        mask = (trimap_np == foreground_value).astype(np.uint8)

        log.debug(f"(get_binary_mask): Mask sum (pixels=={foreground_value}): {np.sum(mask)}")
        return mask
    except FileNotFoundError:
        # Use logging.warning for non-critical issues that don't stop execution
        log.warning(f"Trimap not found {trimap_path}")
        return None
    except Exception as e:
        # Use logging.error for errors during processing a single file
        log.error(f"Error loading trimap {trimap_path}: {e}")
        return None

def get_tags(mask_np):
    """Returns list of unique classes present (just [1] if foreground exists). Deterministic."""
    if np.any(mask_np == 1): # Assumes mask uses 1 for foreground
        return [1]
    return []

def get_point(mask_np, num_point_per_obj=DEFAULT_NUM_POINT_PER_OBJ):
    """Generates a specified number of point sampled from each object mask."""
    point = []
    labels = measure.label(mask_np, connectivity=2)
    props = measure.regionprops(labels)
    props = sorted(props, key=lambda p: p.label)
    log.debug(f"(get_point): Found {len([p for p in props if p.label != 0])} potential FG objects.")

    for prop in props:
        if prop.label != 0:
            obj_mask = (labels == prop.label)
            coords = np.argwhere(obj_mask)
            log.debug(f"(get_point): Object {prop.label} - Original mask sum: {np.sum(obj_mask)}, Found {len(coords)} coordinates.")

            if len(coords) > 0:
                coords = coords[np.lexsort((coords[:, 1], coords[:, 0]))]
                num_to_sample = min(num_point_per_obj, len(coords))
                log.debug(f"(get_point): Object {prop.label} - Attempting to sample {num_to_sample} points.")

                if num_to_sample > 0:
                    indices = np.random.choice(len(coords), size=num_to_sample, replace=False)
                    indices = np.sort(indices)
                    sampled_coords = coords[indices]
                    point.extend([(int(y), int(x)) for y, x in sampled_coords])
                else:
                    log.debug(f"(get_point): Object {prop.label} - num_to_sample is 0, skipping sampling.")

    log.debug(f"(get_point): Generated {len(point)} total points.")
    return point

def get_scatter(mask_np, num_point_per_obj=DEFAULT_NUM_SCATTER_POINT_PER_OBJ):
    """Generates SCATTERED points by sampling from ERODED regions."""
    scatter = []
    labels = measure.label(mask_np, connectivity=2)
    props = measure.regionprops(labels)
    props = sorted(props, key=lambda p: p.label)

    erosion_size = max(1, int(min(mask_np.shape) * 0.01)) # Relatively small erosion
    selem = morphology.disk(erosion_size)
    log.debug(f"(get_scatter): Using erosion size: {erosion_size}")

    for prop in props:
        if prop.label != 0:
            obj_mask = (labels == prop.label)
            eroded_mask = morphology.binary_erosion(obj_mask, footprint=selem)
            coords = np.argwhere(eroded_mask)
            log.debug(f"(get_scatter): Object {prop.label} - Original mask sum: {np.sum(obj_mask)}, Eroded mask sum: {np.sum(eroded_mask)}, Found {len(coords)} eroded coordinates.")

            if len(coords) > 0:
                coords = coords[np.lexsort((coords[:, 1], coords[:, 0]))]
                num_to_sample = min(num_point_per_obj, len(coords))
                log.debug(f"(get_scatter): Object {prop.label} - Attempting to sample {num_to_sample} scatter points.")

                if num_to_sample > 0:
                    indices = np.random.choice(len(coords), size=num_to_sample, replace=False)
                    indices = np.sort(indices)
                    sampled_coords = coords[indices]
                    scatter.extend([(int(y), int(x)) for y, x in sampled_coords])
                else:
                    log.debug(f"(get_scatter): Object {prop.label} - num_to_sample is 0, skipping sampling.")

    log.debug(f"(get_scatter): Generated {len(scatter)} total scatter points.")
    return scatter

def get_scribbles(mask_np,
                  num_scribbles_per_type=DEFAULT_NUM_SCRIBBLES_PER_TYPE,
                  scribble_length=DEFAULT_SCRIBBLE_LENGTH,
                  erosion_factor=DEFAULT_SCRIBBLE_EROSION_FACTOR):
    """
    Generates foreground and background scribbles as one long freehand path
    per connected object region. This ensures a single big scribble for
    each FG or BG object, rather than multiple short strokes.
    """
    scribbles = {'foreground': [], 'background': []}
    h, w = mask_np.shape
    erosion_radius = max(1, int(min(h, w) * erosion_factor))
    selem = morphology.disk(erosion_radius)
    log.debug(f"(get_scribbles): Using erosion radius: {erosion_radius}")

    def generate_single_type_scribbles(target_mask, target_value, num_paths, path_length):
        """
        For each connected component in target_mask, we generate exactly ONE
        random-walk scribble path, with an extended path length so that it
        covers most of the object. We keep the rest of the logic as-is,
        but remove repeated path generation.
        """
        scribble_points = []
        labels = measure.label(target_mask, connectivity=2)
        props = measure.regionprops(labels)
        type_str = ('background', 'foreground')[target_value]

        log.debug(f"(get_scribbles - {type_str}): Found {len(props)} connected regions total.")

        for prop in props:
            if prop.label == 0:
                continue

            obj_mask = (labels == prop.label)
            # Erode the object mask so we don't start right at the edge
            eroded_mask = morphology.binary_erosion(obj_mask, footprint=selem)
            start_coords_list = np.argwhere(eroded_mask)

            log.debug(f"(get_scribbles - {type_str}): Object {prop.label} - Original mask sum: {np.sum(obj_mask)}, "
                      f"Eroded mask sum: {np.sum(eroded_mask)}, potential starts: {len(start_coords_list)}")

            if len(start_coords_list) == 0:
                # fallback to original mask if erosion kills everything
                log.warning(f"(get_scribbles - {type_str}): Erosion removed all. Using original object mask.")
                start_coords_list = np.argwhere(obj_mask)
                if len(start_coords_list) == 0:
                    log.warning(f"(get_scribbles - {type_str}): Cannot generate scribble for empty object region.")
                    continue

            # Sort the start coords for determinism, then pick one randomly
            start_coords_list = start_coords_list[np.lexsort((start_coords_list[:,1], start_coords_list[:,0]))]
            start_y, start_x = random.choice(start_coords_list)

            # Make the scribble path length significantly larger
            # so we get a "very long freehand stroke."
            extended_path_length = path_length * 20

            current_path = []
            visited_in_this_path = set()
            current_pos = (start_y, start_x)

            log.debug(f"(get_scribbles - {type_str}): Generating one scribble for object {prop.label} "
                      f"with path_length={extended_path_length} starting at {current_pos}")

            for step_num in range(extended_path_length):
                # Only check in-bounds and object membership,
                # do *not* check if we have already visited the pixel
                if 0 <= current_pos[0] < h and 0 <= current_pos[1] < w and obj_mask[current_pos[0], current_pos[1]]:
                    current_path.append(current_pos)

                    # STILL gather neighbors
                    neighbors = []
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            ny = current_pos[0] + dy
                            nx = current_pos[1] + dx
                            if 0 <= ny < h and 0 <= nx < w and obj_mask[ny, nx]:
                                neighbors.append((ny, nx))

                    if not neighbors:
                        # If truly no valid neighbors remain, we must stop
                        break

                    current_pos = random.choice(neighbors)

                else:
                    # out-of-bounds or off-mask
                    break

            if current_path:
                scribble_points.extend(current_path)
                log.debug(f"(get_scribbles - {type_str}): Object {prop.label}, scribble path length={len(current_path)}")
            else:
                log.debug(f"(get_scribbles - {type_str}): Object {prop.label} scribble was empty.")

        return [(int(y), int(x)) for y, x in scribble_points]

    # Foreground scribbles
    fg_mask = (mask_np == 1)
    if np.any(fg_mask):
        log.debug("--- Generating Foreground Scribbles ---")
        scribbles['foreground'] = generate_single_type_scribbles(
            target_mask=fg_mask,
            target_value=1,
            num_paths=num_scribbles_per_type,  # argument retained, but effectively unused now
            path_length=scribble_length
        )
    else:
        log.debug("--- Skipping Foreground Scribbles (No FG pixels in mask) ---")

    # Background scribbles
    bg_mask = (mask_np == 0)
    if np.any(bg_mask):
        log.debug("--- Generating Background Scribbles ---")
        scribbles['background'] = generate_single_type_scribbles(
            target_mask=bg_mask,
            target_value=0,
            num_paths=num_scribbles_per_type,  # argument retained, but effectively unused now
            path_length=scribble_length
        )
    else:
        log.debug("--- Skipping Background Scribbles (No BG pixels in mask) ---")

    log.debug(f"(get_scribbles): Final FG points: {len(scribbles['foreground'])}, "
              f"BG points: {len(scribbles['background'])}")
    return scribbles

def get_bounding_boxes(mask_np):
    """Returns list of bounding boxes [(ymin, xmin, ymax, xmax)] for each object."""
    boxes = []
    labels = measure.label(mask_np, connectivity=2)
    props = measure.regionprops(labels)
    props = sorted(props, key=lambda p: p.label)
    for prop in props:
        if prop.label != 0:
            boxes.append(prop.bbox)
    # log.debug(f"(get_bounding_boxes): Found {len(boxes)} boxes.") # Optional debug
    return boxes

def main(args):
    # --- Configure Logging ---
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    # --- End Logging Config ---

    np.random.seed(args.seed)
    random.seed(args.seed)
    log.info(f"Using fixed random seed: {args.seed} for reproducibility.")
    log.info(f"Logging level set to: {args.log_level.upper()}")

    log.info("Generating weak labels...")
    data_dir = args.data_dir
    output_file = args.output_file

    image_dir = os.path.join(data_dir, 'images')
    trimap_dir = os.path.join(data_dir, 'annotations', 'trimaps')
    # Use glob to find images, adjust pattern if needed
    image_files = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))

    num_images = len(image_files)
    if num_images == 0:
        log.critical(f"Error: No images found in {image_dir} matching '*.jpg'")
        return

    # Using a fixed split ratio - adjust if needed
    num_train = int(num_images * 0.7)
    train_image_files = image_files[:num_train]
    log.info(f"Processing {len(train_image_files)} images for training weak labels.")

    all_weak_labels = {}

    for img_path in tqdm(train_image_files, desc="Processing images"):
        img_filename = os.path.basename(img_path)
        img_name_no_ext = os.path.splitext(img_filename)[0]
        trimap_path = os.path.join(trimap_dir, img_name_no_ext + '.png')

        log.debug(f"--- Processing: {img_filename} ---")

        binary_mask = get_binary_mask_from_trimap(trimap_path)

        weak_label_data = {
            'tags': [],
            'point': [],
            'scatter': [],
            'scribbles': {'foreground': [], 'background': []},
            'boxes': []
        }

        if binary_mask is not None and np.any(binary_mask):
            try:
                log.debug("(main): Mask is valid, attempting to generate labels...")
                weak_label_data['tags'] = get_tags(binary_mask)
                weak_label_data['point'] = get_point(binary_mask, num_point_per_obj=args.num_point)
                weak_label_data['scatter'] = get_scatter(binary_mask, num_point_per_obj=args.num_scatter_point)
                weak_label_data['scribbles'] = get_scribbles(
                    binary_mask,
                    num_scribbles_per_type=args.num_scribbles,
                    scribble_length=args.scribble_length,
                    erosion_factor=args.scribble_erosion
                )
                weak_label_data['boxes'] = get_bounding_boxes(binary_mask)
                log.debug(f"(main): Generated Boxes: {len(weak_label_data['boxes'])}, "
                          f"Points: {len(weak_label_data['point'])}, "
                          f"Scatter: {len(weak_label_data['scatter'])}, "
                          f"FG Scribbles: {len(weak_label_data['scribbles']['foreground'])}, "
                          f"BG Scribbles: {len(weak_label_data['scribbles']['background'])}")

            except Exception as e:
                log.error(f"Error processing {img_filename}", exc_info=True)
                weak_label_data = {
                    'tags': [],
                    'point': [],
                    'scatter': [],
                    'scribbles': {'foreground': [], 'background': []},
                    'boxes': []
                }
        elif binary_mask is None:
            log.warning(f"(main): Mask generation failed for {img_filename}.")
        else:
            log.warning(f"(main): Mask generated for {img_filename}, but it contains no foreground pixels "
                        f"(sum={np.sum(binary_mask)}).")

        all_weak_labels[img_filename] = weak_label_data

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(all_weak_labels, f)

    log.info(f"Weak labels saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Weak Labels (including Points, Scatter, Scribbles, Boxes) for Oxford Pets")
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to the root Oxford Pets dataset directory')
    parser.add_argument('--output_file', type=str, default='./weak_labels/weak_labels_train.pkl',
                        help='Path to save the generated weak labels pickle file')
    parser.add_argument('--num_point', type=int, default=DEFAULT_NUM_POINT_PER_OBJ,
                        help="Number of isolated points to generate per object for the 'point' key")
    parser.add_argument('--num_scatter_point', type=int, default=DEFAULT_NUM_SCATTER_POINT_PER_OBJ,
                        help="Number of isolated points to generate per object from eroded regions for the 'scatter' key")
    parser.add_argument('--num_scribbles', type=int, default=DEFAULT_NUM_SCRIBBLES_PER_TYPE,
                        help="Number of connected scribble paths to generate for FOREGROUND and BACKGROUND each")
    parser.add_argument('--scribble_length', type=int, default=DEFAULT_SCRIBBLE_LENGTH,
                        help="Base length (pixels) of each connected scribble path (internally scaled up)")
    parser.add_argument('--scribble_erosion', type=float, default=DEFAULT_SCRIBBLE_EROSION_FACTOR,
                        help="Erosion factor (fraction of min dim) to find scribble start points")
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help='Random seed for deterministic point/scribble generation')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level (default: INFO)')

    args = parser.parse_args()
    main(args)
