import os
import pickle
import numpy as np
import argparse
import glob
import random
import logging
from collections import deque
from scipy import ndimage
from skimage import measure, morphology
from tqdm import tqdm
from PIL import Image

# --- Configuration ---
DEFAULT_NUM_POINT_PER_OBJ = 30
RANDOM_SEED = 42
# --- End Configuration ---

# --- Logger Setup ---
log = logging.getLogger(__name__)
# --- End Logger Setup ---

def get_binary_mask_and_size_from_trimap(trimap_path):
    """Load trimap, create binary mask, and return original size."""
    try:
        trimap = Image.open(trimap_path).convert('L')
        original_size = trimap.size
        trimap_np = np.array(trimap)
        mask = (trimap_np == 1).astype(np.uint8)
        return (mask, original_size) if np.any(mask) else (None, None)
    except Exception as e:
        log.error(f"Error processing {trimap_path}: {e}")
        return None, None

def get_points_from_regions(regions, num_point_per_obj=DEFAULT_NUM_POINT_PER_OBJ):
    """Generate points from precomputed regions."""
    points = []
    for region in regions:
        coords = region.coords
        if not coords.size:
            continue
        n = min(num_point_per_obj, coords.shape[0])
        indices = np.random.choice(coords.shape[0], n, replace=False)
        points.extend([(int(c[1]), int(c[0])) for c in coords[indices]])
    return points

def find_longest_path(skeleton):
    """Find longest path using BFS-based two-pass method."""
    if not np.any(skeleton):
        return []
    
    coords = np.argwhere(skeleton)
    if not coords.size:
        return []
    start = tuple(coords[0])

    def bfs(start_node):
        visited = np.zeros_like(skeleton, dtype=bool)
        queue = deque([start_node])
        visited[start_node] = True
        parent = {start_node: None}
        farthest_node = start_node

        while queue:
            current = queue.popleft()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == dx == 0:
                        continue
                    ny, nx = current[0]+dy, current[1]+dx
                    if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
                        if skeleton[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            parent[(ny, nx)] = current
                            queue.append((ny, nx))
                            farthest_node = (ny, nx)
        return farthest_node, parent

    def reconstruct_path(parent, end_node):
        path = []
        while end_node:
            path.append(end_node)
            end_node = parent.get(end_node)
        return path[::-1]

    u, u_parent = bfs(start)
    v, v_parent = bfs(u)
    return reconstruct_path(v_parent, v)

def get_scribbles(mask_np, bg_dilation=40):
    """Generate scribbles using optimized skeleton processing."""
    scribbles = {'foreground': [], 'background': []}
    h, w = mask_np.shape

    # Foreground processing
    fg_mask = mask_np == 1
    if np.any(fg_mask):
        skeleton_fg = morphology.skeletonize(fg_mask)
        fg_path = find_longest_path(skeleton_fg)
        scribbles['foreground'] = [(x, y) for y, x in fg_path[::2]]

    # Background processing with distance transform
    if np.any(~fg_mask):
        try:
            distance = ndimage.distance_transform_edt(~fg_mask)
            bg_region = distance > bg_dilation
            if np.any(bg_region):
                skeleton_bg = morphology.skeletonize(bg_region)
                bg_path = find_longest_path(skeleton_bg)
                scribbles['background'] = [
                    (min(w-1, max(0, x)), min(h-1, max(0, y)))
                    for y, x in bg_path[::3]
                ]
        except Exception as e:
            log.error(f"Background scribble error: {e}")

    return scribbles

def get_boxes_from_regions(regions):
    """Extract bounding boxes from precomputed regions."""
    return [(int(r.bbox[1]), int(r.bbox[0]), 
             int(r.bbox[3]), int(r.bbox[2])) for r in regions]

def main(args):
    logging.basicConfig(
        level=args.log_level.upper(),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    np.random.seed(args.seed)
    random.seed(args.seed)

    image_files = sorted(glob.glob(os.path.join(args.data_dir, 'images', f'*.{args.image_ext}')))
    num_images = int(len(image_files) * args.data_split)
    weak_labels = {}
    processed = skipped = 0

    for img_path in tqdm(image_files[:num_images], desc="Processing"):
        img_name = os.path.basename(img_path)
        trimap_path = os.path.join(args.data_dir, 'annotations', 'trimaps', 
                                 img_name.replace(f'.{args.image_ext}', f'.{args.trimap_ext}'))
        
        if not os.path.exists(trimap_path):
            skipped +=1
            continue

        mask, size = get_binary_mask_and_size_from_trimap(trimap_path)
        if mask is None or not np.any(mask):
            skipped +=1
            continue

        try:
            labels = measure.label(mask, connectivity=2, optimized=True)
            regions = measure.regionprops(labels)
            scribbles = get_scribbles(mask, args.bg_dilation)
            
            weak_labels[img_name] = {
                'points': get_points_from_regions(regions, args.num_point),
                'scribbles': scribbles,
                'boxes': get_boxes_from_regions(regions),
                'original_size': size
            }
            processed +=1
        except Exception as e:
            log.error(f"Error processing {img_name}: {e}")
            skipped +=1

    log.info(f"Processed {processed}, skipped {skipped}")
    try:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, 'wb') as f:
            pickle.dump(weak_labels, f)
        log.info(f"Saved to {args.output_file}")
    except Exception as e:
        log.critical(f"Save failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimized weak label generator")
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--output_file', default='./weak_labels/weak_labels_efficient.pkl')
    parser.add_argument('--image_ext', default='jpg')
    parser.add_argument('--trimap_ext', default='png')
    parser.add_argument('--num_point', type=int, default=DEFAULT_NUM_POINT_PER_OBJ)
    parser.add_argument('--bg_dilation', type=int, default=40)
    parser.add_argument('--data_split', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--log_level', default='INFO')
    main(parser.parse_args())