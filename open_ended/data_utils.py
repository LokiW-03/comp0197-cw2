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
        self.augment = augment and split == 'train'

        print(f"Initializing dataset: split={split}, mode={supervision_mode}, augment={self.augment}")

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
            print(f"Using images {num_train} to {num_train + num_val -1} for validation.")
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
            T.Normalize(mean=MEAN, std=STD)
        ])
        self.mask_transform = T.Compose([
            T.Resize(img_size, interpolation=T.InterpolationMode.NEAREST), # Use NEAREST for masks
        ])
        # Augmentations (optional)
        self.augmentation_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            # Add more augmentations if needed (e.g., ColorJitter)
        ])

        # Map supervision mode to required weak label keys
        self.mode_to_key = {
            'tags': 'tags',
            'points': 'points',
            'scribbles': 'scribbles',
            'boxes': 'boxes',
            'hybrid_tags_points': ['tags', 'points'] # Special case
        }
        # Pet dataset has 1 class (pet), background, boundary. We simplify to binary.
        # If multi-class (breeds) is needed, adjust class mapping.
        self.num_classes = 1 # Binary: Pet vs Background

    def __len__(self):
        return len(self.image_files)

    def _load_and_transform_mask(self, trimap_path):
        # Load trimap: 1=Foreground, 2=Background, 3=Boundary
        trimap = Image.open(trimap_path).convert('L') # Ensure grayscale
        trimap_resized = self.mask_transform(trimap) # Resize first
        trimap_np = np.array(trimap_resized).squeeze() # To numpy array HxW

        # Convert trimap to binary mask: Foreground=1, Background=0, Boundary=IGNORE_INDEX
        mask = np.zeros_like(trimap_np, dtype=np.int64) # Use int64 for CE Loss
        mask[trimap_np == 1] = 1  # Foreground Pet class
        mask[trimap_np == 2] = 0  # Background class
        mask[trimap_np == 3] = IGNORE_INDEX # Ignore boundary pixels

        return torch.from_numpy(mask) # Return as HxW tensor

    def _get_weak_supervision(self, index):
        """ Prepares weak supervision signals based on mode. """
        img_filename = os.path.basename(self.image_files[index])
        if self.weak_labels is None or img_filename not in self.weak_labels:
             # Should not happen if weak labels were generated correctly for train split
             print(f"Warning: No weak label found for {img_filename} in mode {self.supervision_mode}")
             # Return dummy data or handle error appropriately
             if self.supervision_mode == 'tags': return torch.zeros(self.num_classes, dtype=torch.float32)
             if self.supervision_mode == 'points': return torch.zeros(self.img_size, dtype=torch.int64) + IGNORE_INDEX
             if self.supervision_mode == 'scribbles': return torch.zeros(self.img_size, dtype=torch.int64) + IGNORE_INDEX
             if self.supervision_mode == 'boxes': return torch.zeros(self.img_size, dtype=torch.int64) + IGNORE_INDEX
             if self.supervision_mode == 'hybrid_tags_points':
                 return {
                     'tags': torch.zeros(self.num_classes, dtype=torch.float32),
                     'points': torch.zeros(self.img_size, dtype=torch.int64) + IGNORE_INDEX
                 }

        item_labels = self.weak_labels[img_filename]
        weak_data = {}

        required_keys = self.mode_to_key.get(self.supervision_mode, [])
        if not isinstance(required_keys, list):
            required_keys = [required_keys]

        for key in required_keys:
            if key == 'tags':
                 # Assuming tags are list of present classes (0 or 1 for binary)
                 # Convert to multi-hot encoding if needed, here simple binary presence
                 tag_tensor = torch.zeros(self.num_classes, dtype=torch.float32)
                 if 1 in item_labels.get('tags', []): # Check if pet class is present
                     tag_tensor[0] = 1.0 # Binary case: only class 0 (pet) exists
                 weak_data['tags'] = tag_tensor

            elif key in ['points', 'scribbles']:
                 # Expecting list of (y, x) coordinates for the foreground class (1)
                 sparse_mask = torch.full(self.img_size, IGNORE_INDEX, dtype=torch.int64)
                 coords = item_labels.get(key, [])
                 for y, x in coords:
                      # Clamp coordinates to be within image bounds after resize
                      y_clamped = max(0, min(y, self.img_size[0] - 1))
                      x_clamped = max(0, min(x, self.img_size[1] - 1))
                      sparse_mask[y_clamped, x_clamped] = 1 # Label points/scribbles as foreground (class 1)
                 weak_data[key] = sparse_mask # HxW tensor

            elif key == 'boxes':
                 # Expecting list of boxes (ymin, xmin, ymax, xmax)
                 # Generate pseudo-mask: Inside box=1 (Foreground), Outside=0 (Background)
                 box_pseudo_mask = torch.zeros(self.img_size, dtype=torch.int64) # Start with background
                 boxes = item_labels.get('boxes', [])
                 for box in boxes:
                      ymin, xmin, ymax, xmax = box
                      # Clamp coordinates
                      ymin_c = max(0, min(ymin, self.img_size[0] - 1))
                      xmin_c = max(0, min(xmin, self.img_size[1] - 1))
                      ymax_c = max(0, min(ymax, self.img_size[0] - 1))
                      xmax_c = max(0, min(xmax, self.img_size[1] - 1))
                      if ymax_c > ymin_c and xmax_c > xmin_c:
                          box_pseudo_mask[ymin_c:ymax_c, xmin_c:xmax_c] = 1 # Mark inside box as foreground
                 weak_data['boxes'] = box_pseudo_mask # HxW tensor

        # Return appropriate format based on mode
        if self.supervision_mode == 'hybrid_tags_points':
            return weak_data # Return dict with 'tags' and 'points'
        elif required_keys:
            return weak_data[required_keys[0]] # Return the single tensor
        else: # Should only happen for 'full' mode on train split, which is not expected
            return torch.zeros(self.img_size, dtype=torch.int64) + IGNORE_INDEX


    def __getitem__(self, index):
        img_path = self.image_files[index]
        trimap_path = self.trimap_files[index]

        image = Image.open(img_path).convert('RGB')
        target_mask = self._load_and_transform_mask(trimap_path) # HxW GT mask

        # Apply base transform (resize, ToTensor, Normalize)
        image_tensor = self.base_transform(image)

        # Apply augmentation if enabled
        if self.augment:
            # Augmentation needs careful handling for masks/points
            # Simple example: Apply same geometric transforms to image and mask
            # Seed setting might be needed for reproducibility if random transforms differ
            # For points/scribbles, coordinates need transformation too (more complex)
            # Keeping it simple: only flip/rotate image for now. Mask augmentation is harder.
            image_tensor = self.augmentation_transform(image_tensor)
            # NOTE: Augmenting sparse labels (points/scribbles) correctly is non-trivial
            #       and omitted here for simplicity within the 1-week scope.
            #       Full mask augmentation is also omitted but could be added if desired.

        # Prepare supervision target based on mode
        if self.split == 'train':
            if self.supervision_mode == 'full':
                supervision_target = target_mask # Use GT mask directly
            else:
                supervision_target = self._get_weak_supervision(index)
        else: # For val/test, always use the GT mask for evaluation
             supervision_target = target_mask

        return image_tensor, supervision_target, target_mask # Return GT mask always for eval