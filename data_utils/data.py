import torch
import random
from torchvision import datasets, transforms

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