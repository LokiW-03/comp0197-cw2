import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset

# Set random seed for reproducibility
torch.manual_seed(42)

DATA_DIR = ''
IMAGE_SIZE = 224  # we will resize images to 224x224 for training
# Normalization values (ImageNet mean & std) – common practice if using pretrained backbone
normalize_mean = [0.485, 0.456, 0.406]
normalize_std  = [0.229, 0.224, 0.225]


def tensor_trimap(t):
    x = t * 255
    x = x.to(torch.long)
    x = x - 1
    return x



class ImageTransform:
    common_image_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])

    common_mask_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Lambda(tensor_trimap)
    ])

    cam_train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])

    cam_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])

    oeq_image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])

    oeq_mask_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.NEAREST)
    ])

    oeq_augmentation_image_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
    ])


class OxfordIIITPetWithPaths(datasets.OxfordIIITPet):
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



class OxfordPetWithPseudo(Dataset):
    def __init__(self, pseudo_data):
        """
        Args:
            pseudo_data (dict): Data dictionary loaded via load_pseudo()
        """
        self.pairs = pseudo_data['pairs']

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Returns format consistent with original dataset: (image, target)
        Note: Here the target is actually a pseudo mask
        """
        item = self.pairs[idx]
        return item['image'], item['mask']


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
                sp[None, None].float(), size=(IMAGE_SIZE, IMAGE_SIZE), mode='nearest'
            ).long().squeeze(0).squeeze(0)

        return image, label, sp


trainset = datasets.OxfordIIITPet(
    root='./data',
    split='trainval',
    target_types= 'segmentation',
    download=True,
    transform=ImageTransform.common_image_transform,
    target_transform=ImageTransform.common_mask_transform
)


testset = datasets.OxfordIIITPet(
    root='./data',
    split='test',
    target_types= 'segmentation',
    download=True,
    transform=ImageTransform.common_image_transform,
    target_transform=ImageTransform.common_mask_transform)