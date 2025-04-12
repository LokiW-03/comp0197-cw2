from functools import partial
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from dataset.oxfordpet_paths import OxfordIIITPetWithPaths
from model.data import ImageTransform

IMAGE_SIZE = 224
BATCH_SIZE = 64
WORKERS = 4

def collate_fn_impl(batch, with_paths=False):
    if with_paths:  # (image, label, path)
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        paths = [item[2] for item in batch]
        return images, labels, paths
    else:  # (image, label)
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        return images, labels

def download_pet_dataset(with_paths=False):
    """
    Download and prepare the Oxford Pet dataset.

    Args:
        with_paths (bool): Whether to return a loader with image paths.

    Returns:
        train_loader, test_loader: Data loaders.
    """
    # Choose the dataset class based on the requirement
    dataset_class = OxfordIIITPetWithPaths if with_paths else OxfordIIITPet

    # Create datasets
    train_dataset = dataset_class(
        root="./data",
        split="trainval",
        target_types="category",
        download=True,
        transform=ImageTransform.common_image_transform
    )
    
    test_dataset = dataset_class(
        root="./data",
        split="test",
        target_types="category",
        download=True,
        transform=ImageTransform.common_image_transform
    )

    # Use partial to create a top-level function, replacing lambda
    train_collate = partial(collate_fn_impl, with_paths=with_paths)
    test_collate = partial(collate_fn_impl, with_paths=with_paths)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        pin_memory=True,
        num_workers=WORKERS,
        collate_fn=train_collate
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=WORKERS,
        collate_fn=test_collate
    )
    
    return train_loader, test_loader