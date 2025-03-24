

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import functional as TF

# Set random seed for reproducibility
torch.manual_seed(42)

DATA_DIR = './data'
IMAGE_SIZE = 224  # we will resize images to 224x224 for training
# Normalization values (ImageNet mean & std) – common practice if using pretrained backbone
normalize_mean = [0.485, 0.456, 0.406]
normalize_std  = [0.229, 0.224, 0.225]

# Custom dataset class for Oxford IIIT Pet with segmentation maps
class PetSegmentationDataset(Dataset):
    def __init__(self, root_dir=DATA_DIR, split='trainval', transforms=None):
        """
        Args:
            - split: 'trainval' or 'test'
            - transforms: Torch transform object
        """
        self.dataset = OxfordIIITPet(root=root_dir, split=split, target_types='segmentation', download=True)
        self.transforms = transforms
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, mask = self.dataset[idx]

        # Convert PIL to numpy for easier manipulation of mask
        mask = np.array(mask)
        # Mask values are {1, 2, 3} for {pet, background, border} – shift to {0,1,2}
        mask = mask - 1
        
        # Resize to consistent dimensions
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        mask = np.array(Image.fromarray(mask).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST))
        
        # Convert to tensor
        image = TF.to_tensor(image)
        
        # Normalize image
        image = TF.normalize(image, mean=normalize_mean, std=normalize_std)
        
        # Convert mask to tensor and adjust labels
        mask = torch.from_numpy(np.array(mask)).long()
        
        # Apply any additional transforms
        if self.transforms:
            image, mask = self.transforms(image, mask)
            
        return image, mask
