import os
import pickle
import numpy as np
import argparse
import glob
import random
import logging

from PIL import Image
from skimage import measure, morphology
from tqdm import tqdm

# --- Configuration ---
DEFAULT_NUM_POINT_PER_OBJ = 30
RANDOM_SEED = 42
# --- End Configuration ---

# --- Logger Setup ---
log = logging.getLogger(__name__)
# --- End Logger Setup ---

# <<< MODIFIED FUNCTION >>>
def get_binary_mask_and_size_from_trimap(trimap_path):
    """
    Loads the trimap, gets its original size, converts to grayscale,
    and creates a binary mask based on the foreground value (assuming 1).
    Returns the binary mask (numpy array) and original size (width, height).
    """
    try:
        trimap = Image.open(trimap_path).convert('L')
        original_size = trimap.size # Get original (width, height)
        trimap_np = np.array(trimap)
        mask = (trimap_np == 1).astype(np.uint8)

        # Check if any foreground pixels exist
        if not np.any(mask):
            log.warning(f"No foreground pixels (value 1) found in trimap: {trimap_path}")
            return None, None # Return None if no foreground

        # Return mask at ORIGINAL size and the original size tuple
        return mask, original_size
    except FileNotFoundError:
        log.error(f"Trimap file not found: {trimap_path}")
        return None, None
    except Exception as e:
        log.error(f"Error processing trimap {trimap_path}: {e}")
        return None, None


def get_point(mask_np, num_point_per_obj=DEFAULT_NUM_POINT_PER_OBJ):
    """Generates points sampled from each object mask."""
    point = []
    # Ensure mask is boolean for labeling
    labels = measure.label(mask_np > 0, connectivity=2)

    for region in measure.regionprops(labels):
        # In labeled masks, background is usually 0.
        # If mask generation ensures only foreground is > 0, this check might be redundant
        # but safe to keep. region.label == 0 should not happen if mask is binary 0/1.
        # However, if using a different foreground value check, ensure labels are correct.
        # if region.label == 0: # This check is technically correct for measure.label output
        #     continue

        coords = region.coords # Get (row, col) or (y, x) coordinates
        if coords.shape[0] == 0:
            continue

        num_to_sample = min(num_point_per_obj, coords.shape[0])
        indices = np.random.choice(coords.shape[0], size=num_to_sample, replace=False)

        # Ensure points are stored as (x, y) if that's the convention expected downstream
        # The region.coords are (row, col) which corresponds to (y, x)
        # Storing as (y, x) is common in image processing libraries (like numpy, skimage)
        # If you need (x, y), swap them: point.extend([(int(x), int(y)) for y, x in coords[indices]])
        point.extend([(int(c[1]), int(c[0])) for c in coords[indices]]) # Storing as (x, y)

    return point

def get_scribbles(mask_np, bg_dilation=40):
    """Generate scribbles using longest skeleton path."""
    def find_longest_path(skeleton):
        # Check if skeleton is empty
        if not np.any(skeleton):
            return []

        visited = np.zeros_like(skeleton, dtype=bool)
        longest_path = []
        start_points = list(zip(*np.where(skeleton)))
        if not start_points:
             return []

        processed_in_stack = np.zeros_like(skeleton, dtype=bool) # Avoid re-adding to stack

        # Instead of iterating through all points, focus on endpoints/junctions?
        # For simplicity, sticking with original approach but adding checks
        for r, c in start_points:
            if visited[r, c]:
                continue

            # Use DFS to find paths from this starting point
            current_longest_path_from_start = []
            path_stack = [(r, c, [(r, c)])] # (row, col, path_list)
            visited_in_dfs = np.zeros_like(skeleton, dtype=bool) # Track visited within this DFS

            while path_stack:
                cy, cx, current_path = path_stack.pop()

                # Mark as visited for this specific DFS traversal
                visited_in_dfs[cy, cx] = True
                # Mark as globally visited to avoid starting new DFS from here
                visited[cy, cx] = True

                is_endpoint = True
                neighbors = []
                # Check 8-connected neighbors
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
                             if skeleton[ny, nx] and not visited_in_dfs[ny, nx]:
                                is_endpoint = False
                                neighbors.append((ny, nx))

                if is_endpoint:
                    # Reached end of a path branch
                    if len(current_path) > len(current_longest_path_from_start):
                         current_longest_path_from_start = current_path
                else:
                    # Continue exploring neighbors
                    for ny, nx in neighbors:
                         new_path = current_path + [(ny, nx)]
                         path_stack.append((ny, nx, new_path))


            # Update overall longest path if the one found from this start point is longer
            if len(current_longest_path_from_start) > len(longest_path):
                 longest_path = current_longest_path_from_start

        return longest_path

    scribbles = {'foreground': [], 'background': []}
    h, w = mask_np.shape

    # Foreground scribbles
    fg_mask = (mask_np == 1) # Assuming 1 is foreground
    if np.any(fg_mask):
        try:
            skeleton_fg = morphology.skeletonize(fg_mask)
            fg_path = find_longest_path(skeleton_fg)
             # Subsample and convert to (x, y)
            scribbles['foreground'] = [(int(x), int(y)) for y, x in fg_path[::2]] # Sample every 2nd point
        except Exception as e:
            log.error(f"Error generating foreground skeleton/path: {e}")
            scribbles['foreground'] = []


    # Background scribbles
    # Create BG mask carefully - outside the dilated FG mask
    # Ensure fg_mask is boolean
    fg_mask_bool = fg_mask.astype(bool)
    try:
        # Dilate the foreground
        dilated_fg = morphology.binary_dilation(fg_mask_bool, morphology.disk(bg_dilation))

        # Define background region *outside* the dilated area
        # Make sure it stays within image bounds
        full_mask = np.ones_like(mask_np, dtype=bool)
        bg_region = full_mask & ~dilated_fg # Pixels outside the dilated foreground

        if np.any(bg_region):
            skeleton_bg = morphology.skeletonize(bg_region)
            bg_path = find_longest_path(skeleton_bg)
             # Subsample, convert to (x, y) and clip coords just in case (shouldn't be needed if logic is correct)
            scribbles['background'] = [
                (int(min(w - 1, max(0, x))), int(min(h - 1, max(0, y))))
                for y, x in bg_path[::3] # Sample every 3rd point
            ]
        else:
             scribbles['background'] = [] # No valid background region found

    except Exception as e:
        log.error(f"Error generating background skeleton/path: {e}")
        scribbles['background'] = []


    return scribbles


def get_bounding_boxes(mask_np):
    """Returns bounding boxes for each object as (xmin, ymin, xmax, ymax)."""
    boxes = []
    # Ensure mask is boolean for labeling
    labels = measure.label(mask_np > 0, connectivity=2)
    regions = measure.regionprops(labels)

    for region in regions:
        # region.bbox provides (min_row, min_col, max_row, max_col) -> (ymin, xmin, ymax, xmax)
        min_row, min_col, max_row, max_col = region.bbox
        # Convert to (xmin, ymin, xmax, ymax) format
        boxes.append((int(min_col), int(min_row), int(max_col), int(max_row)))
    return boxes

def main(args):
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', # Added logger name
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Use specific logger
    log.info(f"Starting weak label generation with seed {args.seed}")

    np.random.seed(args.seed)
    random.seed(args.seed)

    image_files = sorted(glob.glob(os.path.join(args.data_dir, 'images', f'*.{args.image_ext}'))) # Use image_ext arg
    if not image_files:
        log.critical(f"No images found in {os.path.join(args.data_dir, 'images')} with extension {args.image_ext}")
        return

    num_images_to_process = int(len(image_files) * args.data_split) # Use data_split arg
    log.info(f"Found {len(image_files)} images. Processing {num_images_to_process} ({args.data_split*100:.1f}%).")

    weak_labels = {}
    processed_count = 0
    skipped_count = 0

    # Process specified fraction of images
    for img_path in tqdm(image_files[:num_images_to_process], desc="Processing Images"):
        img_name = os.path.basename(img_path)
        # Construct trimap path using trimap_ext arg
        trimap_name = img_name.replace(f'.{args.image_ext}', f'.{args.trimap_ext}')
        trimap_path = os.path.join(args.data_dir, 'annotations', 'trimaps', trimap_name)

        if not os.path.exists(trimap_path):
            log.warning(f"Trimap not found for image {img_name}, skipping.")
            skipped_count += 1
            continue

        # <<< MODIFIED CALL >>> Get mask and original size
        mask, original_size = get_binary_mask_and_size_from_trimap(trimap_path)

        # Handle cases where mask generation failed or trimap was empty
        if mask is None or original_size is None:
            log.warning(f"Skipping {img_name} due to issues processing its trimap.")
            skipped_count += 1
            continue

        # Check if mask is all zeros (might happen if foreground value assumption is wrong)
        if not np.any(mask):
             log.warning(f"Mask for {img_name} is empty (all zeros), skipping annotation generation.")
             skipped_count += 1
             continue


        # <<< Generate annotations using the original size mask >>>
        points = get_point(mask, args.num_point)
        scribbles = get_scribbles(mask, args.bg_dilation)
        boxes = get_bounding_boxes(mask)

        # Optional: Add original image size to the stored data for reference
        weak_labels[img_name] = {
            'points': points,         # Coordinates are in original image space (x, y)
            'scribbles': scribbles,   # Coordinates are in original image space (x, y)
            'boxes': boxes,           # Coordinates are in original image space (xmin, ymin, xmax, ymax)
            'original_size': original_size # Store (width, height)
        }
        processed_count += 1

    log.info(f"Finished processing. Generated labels for {processed_count} images. Skipped {skipped_count} images.")

    # --- Save Results ---
    try:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, 'wb') as f:
            pickle.dump(weak_labels, f)
        log.info(f"Saved weak labels for {len(weak_labels)} images to {args.output_file}")
    except Exception as e:
        log.critical(f"Failed to save output file {args.output_file}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate weak supervision data (points, scribbles, boxes) from trimaps.")
    parser.add_argument('--data_dir', default='./data', help='Dataset root directory (expects subfolders: images, annotations/trimaps)')
    parser.add_argument('--output_file', default='./weak_labels/weak_labels_train.pkl', help='Output file path for the pickled dictionary')
    parser.add_argument('--image_ext', default='jpg', help='Extension of image files (e.g., jpg, png)')
    parser.add_argument('--trimap_ext', default='png', help='Extension of trimap files (e.g., png)')
    parser.add_argument('--num_point', type=int, default=DEFAULT_NUM_POINT_PER_OBJ, help='Max points to sample per distinct object')
    parser.add_argument('--bg_dilation', type=int, default=40, help='Dilation radius in pixels for defining the background scribble region')
    parser.add_argument('--data_split', type=float, default=0.7, help='Fraction of images to process (e.g., 0.7 for 70%%)') # Added split arg
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help='Random seed for reproducibility')
    parser.add_argument('--log_level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Logging level')

    args = parser.parse_args()
    main(args)