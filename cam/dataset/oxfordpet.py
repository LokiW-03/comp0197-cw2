from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader
from torchvision import transforms
import torch

IMAGE_SIZE = 224
BATCH_SIZE = 32

def download_pet_dataset():
    """Returns data loaders with properly processed tensors"""
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
    
    # Ensure only category labels are returned
    train_dataset = OxfordIIITPet(
        root="./data",
        split="trainval",
        target_types="category",
        download=True,
        transform=train_transform,
        target_transform=None  # Explicitly don't return segmentation masks
    )
    
    test_dataset = OxfordIIITPet(
        root="./data",
        split="test",
        target_types="category",
        download=True,
        transform=test_transform,
        target_transform=None
    )
    
    # Custom collate function to handle possible tuples
    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        return images, labels
    
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

# Usage example
if __name__ == "__main__":
    train_loader, test_loader = download_pet_dataset()
    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of test samples: {len(test_loader.dataset)}")