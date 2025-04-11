import os
import pickle
from PIL import Image
import numpy as np
from skimage import measure, morphology
from tqdm import tqdm
import argparse
import glob
import random
import logging

# --- Configuration ---
DEFAULT_NUM_POINT_PER_OBJ = 30
RANDOM_SEED = 42
# --- End Configuration ---

# --- Logger Setup ---
log = logging.getLogger(__name__)
# --- End Logger Setup ---

def get_binary_mask_from_trimap(trimap_path, target_size=(256,256)):
    trimap = Image.open(trimap_path).convert('L')
    trimap = trimap.resize(target_size, Image.NEAREST)  # Resize first
    trimap_np = np.array(trimap)
    mask = (trimap_np == 1).astype(np.uint8)
    return mask

def get_point(mask_np, num_point_per_obj=DEFAULT_NUM_POINT_PER_OBJ):
    """Generates points sampled from each object mask."""
    point = []
    labels = measure.label(mask_np, connectivity=2)
    
    for region in measure.regionprops(labels):
        if region.label == 0:
            continue
            
        coords = region.coords
        if len(coords) == 0:
            continue
            
        num_to_sample = min(num_point_per_obj, len(coords))
        indices = np.random.choice(len(coords), size=num_to_sample, replace=False)
        point.extend([(int(y), int(x)) for y, x in coords[indices]])
        
    return point

def get_scribbles(mask_np, bg_dilation=20):
    """Generate scribbles using longest skeleton path."""
    def find_longest_path(skeleton):
        visited = np.zeros_like(skeleton, dtype=bool)
        longest_path = []
        
        for y, x in zip(*np.where(skeleton)):
            if visited[y, x]: 
                continue
                
            current_path = []
            stack = [(y, x)]
            while stack:
                cy, cx = stack.pop()
                if visited[cy, cx]:
                    continue
                visited[cy, cx] = True
                current_path.append((cy, cx))
                
                # Check 8-connected neighbors
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = cy+dy, cx+dx
                        if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
                            if skeleton[ny, nx] and not visited[ny, nx]:
                                stack.append((ny, nx))
                                
            if len(current_path) > len(longest_path):
                longest_path = current_path
                
        return longest_path

    scribbles = {'foreground': [], 'background': []}
    
    # Foreground scribbles
    fg_mask = (mask_np == 1)
    if np.any(fg_mask):
        skeleton = morphology.skeletonize(fg_mask)
        scribbles['foreground'] = find_longest_path(skeleton)[::2]

    # Background scribbles
    bg_mask = (mask_np == 0)
    if np.any(bg_mask):
        dilated_fg = morphology.binary_dilation(fg_mask, morphology.disk(bg_dilation))
        bg_region = dilated_fg ^ fg_mask
        
        if np.any(bg_region):
            skeleton = morphology.skeletonize(bg_region)
            bg_path = find_longest_path(skeleton)
            scribbles['background'] = [
                (max(0, min(y, mask_np.shape[0]-1)), 
                 max(0, min(x, mask_np.shape[1]-1)))
                for y, x in bg_path[::3]
            ]

    return scribbles

def get_bounding_boxes(mask_np):
    """Returns bounding boxes for each object."""
    return [region.bbox for region in measure.regionprops(measure.label(mask_np)) 
            if region.label != 0]

def main(args):
    # --- Configure Logging ---
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # --- Data Processing ---
    image_files = sorted(glob.glob(os.path.join(args.data_dir, 'images', '*.jpg')))
    if not image_files:
        log.critical("No images found")
        return

    weak_labels = {}
    
    for img_path in tqdm(image_files[:int(len(image_files)*0.7)], desc="Processing"):
        img_name = os.path.basename(img_path)
        trimap_path = os.path.join(args.data_dir, 'annotations', 'trimaps', 
                                 img_name.replace('.jpg', '.png'))
        
        mask = get_binary_mask_from_trimap(trimap_path)
        if mask is None:
            continue
            
        weak_labels[img_name] = {
            'points': get_point(mask, args.num_point),
            'scribbles': get_scribbles(mask, args.bg_dilation),
            'boxes': get_bounding_boxes(mask)
        }

    # --- Save Results ---
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'wb') as f:
        pickle.dump(weak_labels, f)
        
    log.info(f"Saved weak labels for {len(weak_labels)} images to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate weak supervision data")
    parser.add_argument('--data_dir', default='./data', help='Dataset root directory')
    parser.add_argument('--output_file', default='./weak_labels.pkl', 
                       help='Output file path')
    parser.add_argument('--num_point', type=int, default=DEFAULT_NUM_POINT_PER_OBJ,
                       help='Points per object')
    parser.add_argument('--bg_dilation', type=int, default=20,
                       help='Background scribble dilation radius')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, 
                       help='Random seed')
    parser.add_argument('--log_level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    
    args = parser.parse_args()
    main(args)