# load_pseudo.py
import torch
from torch.utils.data import DataLoader
from data_utils.data import OxfordPetWithPseudo
from cam.common import MODEL_SAVE_PATH

def load_pseudo(save_path, batch_size=32, shuffle=False, device=torch.device('cpu'), collapse_contour=False):
    """
    Load pseudo mask data and create a DataLoader compatible with original format
    
    Args:
        save_path (str): Path to pseudo mask data
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle the data
        device (torch.device): Device to load the data on
    
    Returns:
        DataLoader: Loader with format consistent with original dataset
    """
    dataset = load_pseudo_dataset(save_path, device)
    
    # Define collate_fn to maintain original format
    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        masks = torch.stack([item[1] for item in batch])
        if collapse_contour:
            # Collapse contour class into background
            masks[masks == 2] = 0
        return images, masks
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )
    return dataloader

def load_pseudo_dataset(save_path, device=torch.device('cpu')):
    """
    Load pseudo mask data and create a DataLoader compatible with original dataset
    
    Args:
        save_path (str): Path to pseudo mask data
        device (torch.device): Device to load the data on
    
    Returns:
        DataLoader: Loader with format consistent with original dataset
    """
    # Load original data
    pseudo_data = torch.load(save_path, map_location=device, weights_only=True)
    
    # Create dataset
    dataset = OxfordPetWithPseudo(pseudo_data)
    return dataset

