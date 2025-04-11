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
        Returns a dictionary mapping supervision type(s) to HxW tensor masks.
        Returns None if the item should be skipped.
        """
        img_filename = os.path.basename(self.image_files[index])
        height, width = self.img_size
        item_labels = self.weak_labels.get(img_filename) if self.weak_labels else None

        supervision_dict = {} # Initialize empty dictionary
        weak_label_applied = False

        # --- Determine keys based on supervision_mode ---
        required_keys = []
        if self.supervision_mode in self.mode_to_key:
            keys = self.mode_to_key[self.supervision_mode]
            required_keys = keys if isinstance(keys, list) else [keys]
        elif self.supervision_mode == 'full': # Should not happen here ideally
            return torch.full(self.img_size, IGNORE_INDEX, dtype=torch.int64) # Or handle error
        else:
            print(f"Warning: Unsupported supervision mode '{self.supervision_mode}' in _get_weak_supervision")
            return torch.full(self.img_size, IGNORE_INDEX, dtype=torch.int64) # Return ignore mask

        # --- Generate masks for each required key ---
        for key in required_keys:
            key_mask = torch.full(self.img_size, IGNORE_INDEX, dtype=torch.int64)
            applied_for_key = False

            if item_labels:
                if key == 'points':
                    coords = item_labels.get('points', [])
                    if coords:
                        for x, y in coords: # Assuming (x, y)
                            y_c = max(0, min(int(y), height - 1))
                            x_c = max(0, min(int(x), width - 1))
                            key_mask[y_c, x_c] = 1 # Pet class
                            applied_for_key = True
                    # Default center logic (if needed specifically for points key)
                    if not applied_for_key:
                        y_center, x_center = height // 2, width // 2
                        if 0 <= y_center < height and 0 <= x_center < width:
                            key_mask[y_center, x_center] = 1
                            applied_for_key = True # Mark default as applied

                elif key == 'scribbles':
                    scribbles_dict = item_labels.get('scribbles', {})
                    fg = scribbles_dict.get('foreground', [])
                    bg = scribbles_dict.get('background', [])
                    if fg:
                        for x,y in fg:
                            y_c = max(0, min(int(y), height - 1)); x_c = max(0, min(int(x), width - 1))
                            key_mask[y_c, x_c] = 1 # FG
                            applied_for_key = True
                    if bg:
                        for x,y in bg:
                            y_c = max(0, min(int(y), height - 1)); x_c = max(0, min(int(x), width - 1))
                            key_mask[y_c, x_c] = 0 # BG
                            applied_for_key = True # Mark BG scribbles as applied too

                elif key == 'boxes':
                    boxes = item_labels.get('boxes', [])
                    if boxes:
                        for xmin, ymin, xmax, ymax in boxes:
                            ymin_c = max(0, min(int(ymin), height - 1))
                            xmin_c = max(0, min(int(xmin), width - 1))
                            ymax_c = max(0, min(int(ymax), height - 1))
                            xmax_c = max(0, min(int(xmax), width - 1))
                            if ymax_c > ymin_c and xmax_c > xmin_c:
                                key_mask[ymin_c:ymax_c, xmin_c:xmax_c] = 1 # Pet class inside box
                                applied_for_key = True

                # --- Add logic for other weak label types ---

            if applied_for_key:
                supervision_dict[key] = key_mask
                weak_label_applied = True # Mark that at least one type of label was processed

        # Decide whether to return the dict or skip (e.g., if no labels found at all)
        # This logic might need refinement based on your requirements
        if not weak_label_applied and self.supervision_mode != 'full':
            print(f"Warning: No weak labels found or applied for {img_filename} in mode {self.supervision_mode}. Returning empty dict (loss might be zero).")
            # Or you could return None here to have the collate_fn skip it, but need a custom collate_fn.
            # Returning an empty dict is simpler for now, CombinedLoss must handle it.
            return {}
            # Alternative: Return a default ignore mask if no labels are ever found
            # return {'segmentation': torch.full(self.img_size, IGNORE_INDEX, dtype=torch.int64)}

        return supervision_dict

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
                supervision_target = target_mask # Use GT mask (Tensor)
            else:
                # Get weak supervision dictionary
                supervision_target = self._get_weak_supervision(index)
                # Handle potential skip case if _get_weak_supervision returns None or empty dict
                if not supervision_target: # Checks for None or {}
                    # Handle skip - needs custom collate_fn or return dummy data
                    print(f"Skipping item {index} due to missing weak labels.")
                    # Returning None often requires a custom collate_fn in DataLoader
                    # For simplicity, might return dummy data or raise error earlier
                    # Let's assume for now _get_weak_supervision returns at least {}
                    pass # DataLoader will handle the dict collation

        else: # Validation/Test
            supervision_target = target_mask # Tensor

        return image_tensor, supervision_target, target_mask # supervision_target is now Tensor OR Dict
    
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