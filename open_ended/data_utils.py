# data_utils.py

from logging import log
import os
import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import pickle

# ImageNet normalization stats
MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])

# Define the ignore index for CrossEntropyLoss
IGNORE_INDEX = 255 # Common value, can be changed if needed

class PetsDataset(Dataset):
    """
    Oxford-IIIT Pet Dataset Loader.
    Handles loading images, masks, and pre-generated weak labels.
    """
    def __init__(self, data_dir, split='train', supervision_mode='full',
                 weak_label_path=None, img_size=(256, 256), augment=False):
        """
        Args:
            data_dir (str): Path to the dataset directory.
            split (str): 'train', 'val', or 'test'.
            supervision_mode (str): Type of supervision ('full', 'tags', 'points',
                                     'scribbles', 'boxes', 'hybrid_tags_points').
            weak_label_path (str, optional): Path to the pre-generated weak labels file
                                             (required for weak supervision modes).
            img_size (tuple): Target image size (height, width).
            augment (bool): Whether to apply data augmentation (only for train split).
        """
        self.data_dir = data_dir
        self.split = split
        self.supervision_mode = supervision_mode
        self.img_size = img_size
        
        self.mask_transform = T.Compose([
            T.Resize(img_size, interpolation=T.InterpolationMode.NEAREST),
        ])
        self.augment = augment and split == 'train'
        # ****************************

        # ***** Print data info being used by dataset *****
        print(f"Initializing dataset: split={split}, mode={supervision_mode}, augment={self.augment}")
        # ****************************************************


        image_dir = os.path.join(data_dir, 'images')
        trimap_dir = os.path.join(data_dir, 'annotations', 'trimaps')

        # Load image paths (use .jpg)
        self.image_files = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))

        # Simple split (adjust if official splits are available/preferred)
        num_images = len(self.image_files)
        num_train = int(num_images * 0.7)
        num_val = int(num_images * 0.15)
        # num_test = num_images - num_train - num_val # Test uses remaining

        if split == 'train':
            self.image_files = self.image_files[:num_train]
            print(f"Using first {len(self.image_files)} images for training.")
        elif split == 'val':
            self.image_files = self.image_files[num_train:num_train + num_val]
            # Corrected print statement for val end index
            print(f"Using images {num_train} to {num_train + num_val - 1} for validation.")
        elif split == 'test':
            self.image_files = self.image_files[num_train + num_val:]
            print(f"Using images from {num_train + num_val} onwards for testing.")
        else:
            raise ValueError(f"Invalid split name: {split}")

        self.trimap_files = [
            os.path.join(trimap_dir, os.path.basename(f).replace('.jpg', '.png'))
            for f in self.image_files
        ]

        # Load weak labels if needed (only for training set)
        self.weak_labels = None
        if split == 'train' and supervision_mode != 'full':
            if weak_label_path and os.path.exists(weak_label_path):
                print(f"Loading weak labels from {weak_label_path}")
                with open(weak_label_path, 'rb') as f:
                    self.weak_labels = pickle.load(f)
            else:
                 raise FileNotFoundError(f"Weak label file not found at {weak_label_path}. "
                                        "Run weak_label_generator.py first.")

        # Define transformations
        self.base_transform = T.Compose([
            T.Resize(img_size),
            T.ToTensor(),
        ])

        self.augmentation_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
        ])

        self.normalize = T.Normalize(mean=MEAN, std=STD)

        # Map supervision mode to required weak label keys
        self.mode_to_key = {
            'points': 'points',
            'scribbles': 'scribbles',
            'boxes': 'boxes', # Special case
            'hybrid_points_scribbles': ['scribbles', 'points'],
            'hybrid_points_boxes': ['points', 'boxes'],
            'hybrid_scribbles_boxes': ['scribbles', 'boxes'],
            'hybrid_points_scribbles_boxes': ['points', 'scribbles', 'boxes']
        }

    def __len__(self):
        return len(self.image_files)

    def _load_and_transform_mask(self, trimap_path):
        # Load trimap: 1=Foreground, 2=Background, 3=Boundary
        trimap = Image.open(trimap_path).convert('L') # Ensure grayscale
        trimap_resized = self.mask_transform(trimap) # Resize first
        trimap_np = np.array(trimap_resized).squeeze() # To numpy array HxW

        # Convert trimap to binary mask: Foreground=1, Background=0, Boundary=IGNORE_INDEX
        mask = np.zeros_like(trimap_np, dtype=np.int64) # Use int64 for CE Loss
        mask[trimap_np == 1] = 1  # Foreground Pet class (INDEX 1)
        mask[trimap_np == 2] = 0  # Background class (INDEX 0)
        mask[trimap_np == 3] = IGNORE_INDEX # Ignore boundary pixels

        return torch.from_numpy(mask) # Return as HxW tensor

    def _get_weak_supervision(self, index):
        """
        Prepares weak supervision mask based on mode.
        If mode is 'points' and labels are missing/empty, uses image center as default.
        Returns a HxW tensor mask.
        """
        img_filename = os.path.basename(self.image_files[index])
        height, width = self.img_size

        # Initialize the mask we will return (all ignored initially)
        supervision_mask = torch.full(self.img_size, IGNORE_INDEX, dtype=torch.int64)
        weak_label_applied = False # Flag to track if we successfully used weak labels

        # Check if weak labels exist for this file at all
        item_labels = None
        if self.weak_labels and img_filename in self.weak_labels:
            item_labels = self.weak_labels[img_filename]
        # else:
            

        # --- Apply specific weak label type based on mode ---

        if self.supervision_mode == 'points':
            coords = []
            if item_labels:
                coords = item_labels.get('points', []) # Get points if key exists

            if coords: # Check if list is not empty
                # log.debug(f"Applying {len(coords)} points for {img_filename}")
                for point_item in coords:
                    # --- Check point format and unpack ---
                    if not isinstance(point_item, (list, tuple)) or len(point_item) != 2:
                        log.warning(f"Skipping invalid point format {point_item} in {img_filename}")
                        continue
                    # Assuming format from weak_label_generator is (x, y)
                    x, y = point_item
                    # --- End Check ---

                    # Clamp coordinates and apply to mask
                    y_c = max(0, min(int(y), height - 1))
                    x_c = max(0, min(int(x), width - 1))
                    supervision_mask[y_c, x_c] = 1  # Pet class (label 1)
                    weak_label_applied = True # Mark as applied
            # Default logic applied later if weak_label_applied is still False

        elif self.supervision_mode == 'scribbles':
            scribbles_dict = {}
            if item_labels:
                scribbles_dict = item_labels.get('scribbles', {})

            fg_scribbles = scribbles_dict.get('foreground', [])
            bg_scribbles = scribbles_dict.get('background', [])

            if fg_scribbles or bg_scribbles: # Check if either list is not empty
                # log.debug(f"Applying {len(fg_scribbles)} FG / {len(bg_scribbles)} BG scribbles for {img_filename}")
                # FG scribbles (class 1)
                for scribble_item in fg_scribbles:
                    if not isinstance(scribble_item, (list, tuple)) or len(scribble_item) != 2: continue
                    x, y = scribble_item # Assuming (x, y)
                    y_c = max(0, min(int(y), height - 1))
                    x_c = max(0, min(int(x), width - 1))
                    supervision_mask[y_c, x_c] = 1
                    weak_label_applied = True
                # BG scribbles (class 0)
                for scribble_item in bg_scribbles:
                     if not isinstance(scribble_item, (list, tuple)) or len(scribble_item) != 2: continue
                     x, y = scribble_item # Assuming (x, y)
                     y_c = max(0, min(int(y), height - 1))
                     x_c = max(0, min(int(x), width - 1))
                     supervision_mask[y_c, x_c] = 0 # Background class (label 0)
                     weak_label_applied = True
            # No default centroid for scribbles, mask remains IGNORE if no labels found

        elif self.supervision_mode == 'boxes':
            boxes = []
            if item_labels:
                boxes = item_labels.get('boxes', [])

            if boxes: # Check if list is not empty
                # log.debug(f"Applying {len(boxes)} boxes for {img_filename}")
                pet_class_index = 1
                for box_item in boxes:
                    if not isinstance(box_item, (list, tuple)) or len(box_item) != 4: continue
                    # Assuming (xmin, ymin, xmax, ymax) from weak_label_generator
                    xmin, ymin, xmax, ymax = box_item
                    ymin_c = max(0, min(int(ymin), height - 1))
                    xmin_c = max(0, min(int(xmin), width - 1))
                    ymax_c = max(0, min(int(ymax), height - 1)) # Use H for ymax
                    xmax_c = max(0, min(int(xmax), width - 1))  # Use W for xmax
                    if ymax_c > ymin_c and xmax_c > xmin_c:
                        supervision_mask[ymin_c:ymax_c, xmin_c:xmax_c] = pet_class_index
                        weak_label_applied = True
            # No default centroid for boxes, mask remains IGNORE if no labels found

        # --- Handle Hybrid Modes (Example - add specific logic as needed) ---
        # elif self.supervision_mode == 'hybrid_points_boxes':
        #    # Apply points logic... set weak_label_applied if points are added
        #    # Apply boxes logic... set weak_label_applied if boxes are added
        #    # Default centroid might apply only if points were expected but missing

        elif self.supervision_mode == "full":
             # This function shouldn't normally be called in 'full' mode,
             # but return ignore mask as a fallback.
             log.warning(f"Called _get_weak_supervision in 'full' mode for {img_filename}.")
             return torch.full(self.img_size, IGNORE_INDEX, dtype=torch.int64)
        
        else:
             log.warning(f"Unsupported supervision mode '{self.supervision_mode}' in _get_weak_supervision for {img_filename}.")
             # Returns the ignore mask by default


        # --- Apply Default Centroid Logic (ONLY for 'points' mode if no labels were applied) ---
        if not weak_label_applied and self.supervision_mode == 'points':
            log.warning(f"Weak label points missing or empty for {img_filename}. Using image center point as default.")
            y_center = height // 2
            x_center = width // 2
            # Ensure center is within bounds (should always be if img_size > 0)
            if 0 <= y_center < height and 0 <= x_center < width:
                supervision_mask[y_center, x_center] = 1 # Pet class (label 1)
            else:
                log.error(f"Calculated center ({y_center}, {x_center}) is out of bounds for image size {self.img_size}. Cannot apply default point.")


        # If weak_label_applied is False for other modes (scribbles, boxes),
        # the mask remains the initial full IGNORE_INDEX mask.
        # if not weak_label_applied and self.supervision_mode != 'points':
        #      log.debug(f"No valid weak labels applied for {img_filename} in mode '{self.supervision_mode}'. Returning ignore mask.")


        return supervision_mask

    def __getitem__(self, index):
        img_path = self.image_files[index]
        trimap_path = self.trimap_files[index]

        image = Image.open(img_path).convert('RGB')
        target_mask = self._load_and_transform_mask(trimap_path)  # HxW GT mask

        # Track augmentation parameters
        aug_params = {}  # Stores flip/rotation info

        # Apply base transform first (resize, ToTensor)
        image_tensor = self.base_transform(image)

        # Apply augmentation if enabled
        if self.augment:
            # Random Horizontal Flip
            if torch.rand(1) < 0.5:
                image_tensor = T.functional.hflip(image_tensor)
                aug_params['hflip'] = True

            # Random Rotation (example: 15 degrees)
            angle = float(torch.randint(-15, 15, (1,)).item())
            image_tensor = T.functional.rotate(image_tensor, angle)
            aug_params['rotation'] = angle

        # Normalize after augmentation
        image_tensor = T.functional.normalize(image_tensor, MEAN, STD)

        # --- Prepare Supervision Target for Loss ---
        supervision_target = None # Initialize
        if self.split == 'train':
            if self.supervision_mode == 'full':
                supervision_target = target_mask # Use GT mask
            else:
                # --- MODIFICATION ---
                # Get weak supervision, check if it returned None (meaning skip)
                weak_supervision_result = self._get_weak_supervision(index)
                if weak_supervision_result is None:
                    # The weak label was missing, signal to skip this item
                    return None
                else:
                    supervision_target = weak_supervision_result
                # --- END MODIFICATION ---
        else: # Validation/Test
            supervision_target = target_mask

        return image_tensor, supervision_target, target_mask
    
    def _augment_weak_labels(self, weak_data, aug_params):
        """Applies geometric augmentations to weak labels."""
        img_size = (self.img_size[0], self.img_size[1])  # (height, width)
        
        if 'hflip' in aug_params and aug_params['hflip']:
            # Flip augmentation
            if 'points' in weak_data:
                weak_data['points'] = self._augment_points(
                    weak_data['points'], img_size, flip='h'
                )
            if 'scribbles' in weak_data:
                weak_data['scribbles'] = self._augment_scribbles(
                    weak_data['scribbles'], img_size, flip='h'
                )
            if 'boxes' in weak_data:
                weak_data['boxes'] = self._augment_boxes(
                    weak_data['boxes'], img_size, flip='h'
                )

        if 'rotation' in aug_params:
            angle = aug_params['rotation']
            if 'points' in weak_data:
                weak_data['points'] = self._augment_points(
                    weak_data['points'], img_size, rotate=angle
                )
            if 'scribbles' in weak_data:
                weak_data['scribbles'] = self._augment_scribbles(
                    weak_data['scribbles'], img_size, rotate=angle
                )
            if 'boxes' in weak_data:
                weak_data['boxes'] = self._augment_boxes(
                    weak_data['boxes'], img_size, rotate=angle
                )

        return weak_data
        
    def _augment_points(self, points, img_size, flip=None, rotate=None):
        """Adjust point coordinates for flips/rotations."""
        h, w = img_size
        new_points = []
        print(points)
        for y, x in points:
            # Horizontal Flip
            if flip == 'h':
                x = w - x - 1  # Mirror x-coordinate

            # Rotation (example: 15 degrees)
            if rotate is not None:
                # Convert to image center coordinates
                cx, cy = w // 2, h // 2
                # Rotate point (simplified example)
                # For precise rotation, use rotation matrix
                theta = np.radians(rotate)
                x_new = (x - cx) * np.cos(theta) - (y - cy) * np.sin(theta) + cx
                y_new = (x - cx) * np.sin(theta) + (y - cy) * np.cos(theta) + cy
                x, y = int(x_new), int(y_new)

            # Clamp to valid coordinates
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            new_points.append((y, x))

        return new_points
    
    def _augment_scribbles(self, scribbles_data, img_size, flip=None, rotate=None):
        """
        Applies geometric transformations to scribble coordinates.
        Args:
            scribbles_data: Dictionary with 'foreground' and 'background' keys
            img_size: (height, width) tuple
            flip: None or 'h' (horizontal flip)
            rotate: Rotation angle in degrees (positive = counter-clockwise)
        Returns:
            Augmented scribbles dictionary
        """
        h, w = img_size
        augmented_scribbles = {'foreground': [], 'background': []}
        
        # Process both foreground and background scribbles
        for category in ['foreground', 'background']:
            if category not in scribbles_data:
                continue
                
            for y, x in scribbles_data[category]:
                # Original coordinates (assuming they're in [0, h-1] x [0, w-1])
                orig_y, orig_x = y, x
                
                # Apply horizontal flip
                if flip == 'h':
                    orig_x = w - orig_x - 1  # Mirror x-coordinate

                # Apply rotation
                if rotate is not None:
                    # Convert to image center coordinates
                    cx, cy = w // 2, h // 2
                    x_centered = orig_x - cx
                    y_centered = orig_y - cy
                    
                    # Convert angle to radians
                    theta = np.radians(rotate)
                    
                    # Rotation matrix
                    new_x = x_centered * np.cos(theta) - y_centered * np.sin(theta)
                    new_y = x_centered * np.sin(theta) + y_centered * np.cos(theta)
                    
                    # Convert back to image coordinates
                    orig_x = new_x + cx
                    orig_y = new_y + cy

                # Clamp coordinates to valid range
                final_x = int(np.clip(orig_x, 0, w-1))
                final_y = int(np.clip(orig_y, 0, h-1))
                
                augmented_scribbles[category].append((final_y, final_x))

        return augmented_scribbles

    def _augment_boxes(self, boxes, img_size, flip=None, rotate=None):
        """
        Applies geometric transformations to bounding boxes.
        Args:
            boxes: List of (ymin, xmin, ymax, xmax) tuples
            img_size: (height, width) tuple
            flip: None or 'h' (horizontal flip)
            rotate: Rotation angle in degrees
        Returns:
            List of augmented bounding boxes
        """
        h, w = img_size
        augmented_boxes = []
        
        for box in boxes:
            ymin, xmin, ymax, xmax = box
            
            # Create four corners of the box
            corners = np.array([
                [ymin, xmin],
                [ymin, xmax],
                [ymax, xmin],
                [ymax, xmax]
            ], dtype=np.float32)

            # Apply horizontal flip
            if flip == 'h':
                corners[:, 1] = w - corners[:, 1] - 1  # Flip x-coordinates

            # Apply rotation
            if rotate is not None:
                # Convert to image center coordinates
                cx, cy = w // 2, h // 2
                corners[:, 1] -= cx  # x coordinates
                corners[:, 0] -= cy  # y coordinates
                
                # Convert angle to radians
                theta = np.radians(rotate)
                
                # Rotation matrix
                rot_matrix = np.array([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]
                ])
                
                # Apply rotation to all corners
                rotated = np.dot(corners[:, [1, 0]], rot_matrix.T)  # Swap x,y for rotation
                
                # Convert back to original coordinate system
                corners[:, 1] = rotated[:, 0] + cx  # x coordinates
                corners[:, 0] = rotated[:, 1] + cy  # y coordinates

            # Find new bounding box coordinates
            new_ymin = np.clip(corners[:, 0].min(), 0, h-1)
            new_ymax = np.clip(corners[:, 0].max(), 0, h-1)
            new_xmin = np.clip(corners[:, 1].min(), 0, w-1)
            new_xmax = np.clip(corners[:, 1].max(), 0, w-1)

            # Only keep valid boxes
            if new_ymax > new_ymin and new_xmax > new_xmin:
                augmented_boxes.append((
                    int(new_ymin), int(new_xmin),
                    int(new_ymax), int(new_xmax)
                ))

        return augmented_boxes