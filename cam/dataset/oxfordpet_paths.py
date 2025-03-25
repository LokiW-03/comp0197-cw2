from torchvision.datasets import OxfordIIITPet
import os

class OxfordIIITPetWithPaths(OxfordIIITPet):
    """Extends dataset class to return image paths"""
    def __init__(self, root="./data", split="trainval", target_types="category", 
                 download=False, transform=None, target_transform=None):
        super().__init__(
            root=root,
            split=split,
            target_types=target_types,
            download=download,
            transform=transform,
            target_transform=target_transform
        )
        
        # Build image path list
        self.image_paths = [
            os.path.join(self._images_folder, name) 
            for name in self._images
        ]

    def __getitem__(self, index):
        # Original data
        image, target = super().__getitem__(index)
        return image, target, self.image_paths[index]