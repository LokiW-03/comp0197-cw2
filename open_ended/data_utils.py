# data_utils.py

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
        
        # A hardcoded path to the weak labels file
        # This should be replaced with a dynamic path or passed as an argument
        data = './open_ended/weak_labels/weak_labels_train.pkl'

        # Open the file in binary read mode ('rb')
        with open(data, 'rb') as f:
            # Load the data from the file
            loaded_data = pickle.load(f)
        # ****************************
        train_image_files = loaded_data.keys()
        set_train_image_files = set(train_image_files)
        
        
        print(f"Initializing dataset: split={split}, mode={supervision_mode}, augment={self.augment}")
        
        ## NOTE: We load all images from weak label generator, even though this seems counter intutive, however, weak label are gurantted to have 70% of the images, so enough for training data
        ## NOTE: THen we do 50% split for val and test, same with 0.7 for train, 0.15 for val and 0.15 for test

        image_dir = os.path.join(data_dir, 'images')
        trimap_dir = os.path.join(data_dir, 'annotations', 'trimaps')

        # Load image paths (use .jpg)
        self.image_files = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
        all_image_files = set([os.path.basename(f) for f in self.image_files])
        
        
        val_test_image_files = all_image_files - set_train_image_files
        sorted_val_test_image_files = sorted(list(val_test_image_files))
        

        if split == 'train':
            self.image_files = [os.path.join(data_dir, 'images', f) for f in train_image_files]
            
            # print(f"Using first {len(self.image_files)} images for training.")
        elif split == 'val':
            
            self.image_files = sorted_val_test_image_files[:int(len(sorted_val_test_image_files) * 0.5)]
            self.image_files = [os.path.join(data_dir, 'images', f) for f in self.image_files]

        elif split == 'test':
            self.image_files = sorted_val_test_image_files[:int(len(sorted_val_test_image_files) * 0.5)]
            self.image_files = [os.path.join(data_dir, 'images', f) for f in self.image_files]
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
        """ Prepares weak supervision signals based on mode. """
        img_filename = os.path.basename(self.image_files[index])
        if self.weak_labels is None or img_filename not in self.weak_labels:
            raise ValueError(f"Warning: No weak label found for {img_filename} in mode {self.supervision_mode}")

        item_labels = self.weak_labels[img_filename]
        weak_data = {}

        required_keys = self.mode_to_key.get(self.supervision_mode, [])
        if not isinstance(required_keys, list):
            required_keys = [required_keys]

        for key in required_keys:
            if key in ['points', 'scribbles']:
                sparse_mask = torch.full(self.img_size, IGNORE_INDEX, dtype=torch.int64)
                
                if key == 'points':
                    # Handle points (single class)
                    coords = item_labels.get(key, [])
                    for y, x in coords:
                        y = max(0, min(y, self.img_size[0] - 1))
                        x = max(0, min(x, self.img_size[1] - 1))
                        sparse_mask[y, x] = 1  # Pet class

                elif key == 'scribbles':
                    # Handle scribbles (both foreground and background)
                    scribbles = item_labels.get(key, {})
                    # Foreground scribbles (class 1)
                    for y, x in scribbles.get('foreground', []):
                        y = max(0, min(y, self.img_size[0] - 1))
                        x = max(0, min(x, self.img_size[1] - 1))
                        sparse_mask[y, x] = 1
                    # Background scribbles (class 0)
                    for y, x in scribbles.get('background', []):
                        y = max(0, min(y, self.img_size[0] - 1))
                        x = max(0, min(x, self.img_size[1] - 1))
                        sparse_mask[y, x] = 0

                weak_data[key] = sparse_mask

            elif key == 'boxes':
                 # Expecting list of boxes (ymin, xmin, ymax, xmax)
                 # Generate pseudo-mask: Inside box=1 (Foreground), Outside=0 (Background)
                 # Assuming box marks the Pet class (index 1), outside is Background (index 0)
                 pet_class_index = 1
                 bg_class_index = 0
                 box_pseudo_mask = torch.full(self.img_size, bg_class_index, dtype=torch.int64) # Start with background
                 boxes = item_labels.get('boxes', [])
                 for box in boxes:
                      ymin, xmin, ymax, xmax = box
                      ymin_c = max(0, min(ymin, self.img_size[0] - 1))
                      xmin_c = max(0, min(xmin, self.img_size[1] - 1))
                      ymax_c = max(0, min(ymax, self.img_size[0] - 1))
                      xmax_c = max(0, min(xmax, self.img_size[1] - 1))
                      if ymax_c > ymin_c and xmax_c > xmin_c:
                          box_pseudo_mask[ymin_c:ymax_c, xmin_c:xmax_c] = pet_class_index # Mark inside as Pet
                 weak_data['boxes'] = box_pseudo_mask # HxW tensor

        # Return appropriate format based on mode
        if self.supervision_mode == "full":
            return torch.zeros(self.img_size, dtype=torch.int64) + IGNORE_INDEX
        else:
            return weak_data

    def __getitem__(self, index):
        img_path = self.image_files[index]
        trimap_path = self.trimap_files[index]

        image = Image.open(img_path).convert('RGB')
        target_mask = self._load_and_transform_mask(trimap_path) # HxW GT mask

        # Apply base transform (resize, ToTensor, Normalize)
        image_tensor = self.base_transform(image)

        # Apply augmentation if enabled
        if self.augment:
            image_tensor = self.augmentation_transform(image_tensor)
            # NOTE: Augmenting sparse labels/masks correctly requires more complex implementation
            #       if geometric transforms are applied to them. Omitted for simplicity.

        # Prepare supervision target based on mode
        if self.split == 'train':
            if self.supervision_mode == 'full':
                supervision_target = target_mask # Use GT mask directly
            else:
                supervision_target = self._get_weak_supervision(index)
        else: # For val/test, always use the GT mask for evaluation
             supervision_target = target_mask

        return image_tensor, supervision_target, target_mask # Return GT mask always for eval