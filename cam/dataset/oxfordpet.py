from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from dataset.oxfordpet_paths import OxfordIIITPetWithPaths

IMAGE_SIZE = 224
BATCH_SIZE = 32

def download_pet_dataset(with_paths=False):
    """
    Download and prepare the Oxford Pet dataset
    
    Args:
        with_paths (bool): Whether to return loaders with image paths
        
    Returns:
        train_loader, test_loader: Data loaders
    """
    # Select dataset class
    dataset_class = OxfordIIITPetWithPaths if with_paths else OxfordIIITPet

    # Define preprocessing pipeline
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = dataset_class(
        root="./data",
        split="trainval",
        target_types="category",
        download=True,
        transform=train_transform
    )
    
    test_dataset = dataset_class(
        root="./data",
        split="test",
        target_types="category",
        download=True,
        transform=test_transform
    )

    # Dynamic collate function
    def collate_fn(batch):
        """Automatically handle different return formats"""
        if with_paths:  # (image, label, path)
            images = torch.stack([item[0] for item in batch])
            labels = torch.tensor([item[1] for item in batch])
            paths = [item[2] for item in batch]
            return images, labels, paths
        else:  # (image, label)
            images = torch.stack([item[0] for item in batch])
            labels = torch.tensor([item[1] for item in batch])
            return images, labels

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return train_loader, test_loader