import os
import torch
import logging
from functools import partial
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set random seed for reproducibility
torch.manual_seed(42)

IMAGE_SIZE = 224  # we will resize images to 224x224 for training
# Normalization values (ImageNet mean & std) – common practice if using pretrained backbone
normalize_mean = [0.485, 0.456, 0.406]
normalize_std  = [0.229, 0.224, 0.225]
BATCH_SIZE = 64
WORKERS = 4


def tensor_trimap(t):
    x = t * 255
    x = x.to(torch.long)
    x = x - 1
    return x



class ImageTransform:
    """ Class storing different transform for image and mask"""

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
    """
    Extends the OxfordIIITPet dataset class to include image file paths.

    In addition to the usual image and target pair returned by the parent class,
    this class also returns the path to the image file in the dataset.

    Args:
        root (str): Root directory where the dataset is stored. Default is "./data".
        split (str): The dataset split to use, can be "trainval", "train", or "test". Default is "trainval".
        target_types (str): Type of targets to return, such as "category" or "segmentation". Default is "category".
        download (bool): Whether to download the dataset if not found. Default is False.
        transform (callable, optional): A function/transform that takes in an image and returns a transformed version. Default is None.
        target_transform (callable, optional): A function/transform that takes in a target and returns a transformed version. Default is None.
    """

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
        """
        Override the original `__getitem__` to return the image, target, and image path.

        Args:
            index (int): The index of the image and target to retrieve.

        Returns:
            A tuple containing the image, target, and the image path.
        """

        # Original data
        image, target = super().__getitem__(index)
        return image, target, self.image_paths[index]



class OxfordPetWithPseudo(Dataset):
    """
    Custom dataset class for OxfordPet dataset that uses pseudo-labels for training.

    Args:
        pseudo_data (dict): A dictionary containing the pseudo-label data, with keys like 'pairs'.
    """

    def __init__(self, pseudo_data):
        self.pairs = pseudo_data['pairs']

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            The number of data pairs (image, mask) in the dataset.
        """

        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset, consisting of an image and its corresponding pseudo mask.

        Args:
            idx (int): The index of the data pair to retrieve.

        Returns:
            A tuple containing the image and the pseudo mask (image, mask).
        """

        item = self.pairs[idx]
        return item['image'], item['mask']


class OxfordPetSuperpixels(torch.utils.data.Dataset):
    """
    Custom dataset class for OxfordPet dataset with superpixel annotations.

    This class takes in a base dataset and augment the data by adding superpixel information
    associated with the images.

    Args:
        base_dataset (torch.utils.data.Dataset): The base dataset (such as OxfordIIITPetWithPaths) to use.
        superpixel_dir (str): Directory where the superpixel data (as .pt files) is stored.
        transform (callable, optional): A function/transform that takes in an image and returns a transformed version. Default is None.
    """

    def __init__(self, base_dataset, superpixel_dir, transform=None):
        self.base_dataset = base_dataset
        self.image_paths = base_dataset.image_paths
        self.labels = base_dataset._labels
        self.superpixel_dir = superpixel_dir
        self.transform = transform

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            The number of samples in the dataset.
        """
        return len(self.base_dataset)

    def __getitem__(self, idx):
        """
        Retrieves an image, label, and superpixel mask from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            A tuple containing the image, label, and superpixel mask (image, label, superpixel_mask).
        """

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


def collate_fn_impl(batch, with_paths=False):
    """
    Custom collate function for batching samples from a dataset.

    Depending on the `with_paths` flag, the function returns either the standard
    (image, label) pairs or (image, label, path) pairs.

    Args:
        batch (list): A list of samples (either (image, label) or (image, label, path)).
        with_paths (bool): Flag to indicate whether to include image paths in the output.

    Returns:
        tuple: A tuple containing a batch of images and labels, and optionally paths.
                - If `with_paths=True`: (images, labels, paths)
                - If `with_paths=False`: (images, labels)
    """

    if with_paths:  # (image, label, path)
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        paths = [item[2] for item in batch]
        return images, labels, paths
    else:  # (image, label)
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        return images, labels


def get_cam_pet_dataset(with_paths=False):
    """
    download and prepare pet dataset

    Args:
        with_paths (bool): whether to return loader with path

    Returns:
        cam_train_loader, cam_test_loader: data loaders
    """
    # select dataset
    dataset_class = OxfordIIITPetWithPaths if with_paths else datasets.OxfordIIITPet

    # create datasets
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

    train_collate = partial(collate_fn_impl, with_paths=with_paths)
    test_collate = partial(collate_fn_impl, with_paths=with_paths)

    cam_train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=WORKERS,
        collate_fn=train_collate
    )

    cam_test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=WORKERS,
        collate_fn=test_collate
    )

    return cam_train_loader, cam_test_loader


def download_oxford_pet_oeq(download_root):
    """
    Downloads the Oxford-IIIT Pet dataset using torchvision.
    Uses download_root as the target for the initial download.

    Args:
        download_root (str): The path where torchvision will initially place files.

    Returns:
        bool: True if download/verification was successful or data already exists,
              False otherwise.
    """
    logging.info(f"Checking for Oxford-IIIT Pet dataset in '{os.path.abspath(download_root)}'...")

    try:
        # We instantiate the dataset class primarily to trigger the download=True logic.
        # The actual dataset object isn't used further in this download script.
        # We use download_root here, which will contain the 'oxford-iiit-pet' subdir.
        logging.info("Attempting to download/verify Oxford-IIIT Pet dataset (images and annotations)...")
        logging.info(f"Download target directory: {os.path.abspath(download_root)}")
        logging.info("This may take a while depending on your internet connection.")

        _ = datasets.OxfordIIITPet(root=download_root, split="trainval", target_types="segmentation", download=True)

        logging.info("Dataset download/verification step complete.")
        # Further verification happens implicitly during the restructuring phase
        return True

    except Exception as e:
        logging.error(f"An error occurred during dataset download: {e}", exc_info=True)
        logging.error("Please check your internet connection, disk space, and permissions.")
        return False


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


crm_base_dataset = OxfordIIITPetWithPaths(root='./data', split='trainval', target_types='category', download=True,
                                          transform=None)

crm_testset = datasets.OxfordIIITPet(
        root='./data', split='test', target_types='category',
        download=True, transform=ImageTransform.common_image_transform
    )

crm_trainset = OxfordPetSuperpixels(
        base_dataset=crm_base_dataset,
        superpixel_dir="./superpixels",
        transform=ImageTransform.common_image_transform
    )