import torch
from crm import IMG_SIZE
import os

class OxfordPetSuperpixels(torch.utils.data.Dataset):
    def __init__(self, base_dataset, superpixel_dir, transform=None):
        self.base_dataset = base_dataset
        self.image_paths = base_dataset.image_paths
        self.labels = base_dataset._labels
        self.superpixel_dir = superpixel_dir
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label, path = self.base_dataset[idx]
        filename = os.path.basename(path).replace(".jpg", ".pt")
        sp_path = os.path.join(self.superpixel_dir, filename)
        sp = torch.load(sp_path)  # (H, W)

        if self.transform:
            image = self.transform(image)
            sp = torch.nn.functional.interpolate(
                sp[None, None].float(), size=(IMG_SIZE, IMG_SIZE), mode='nearest'
            ).long().squeeze(0).squeeze(0)

        return image, label, sp