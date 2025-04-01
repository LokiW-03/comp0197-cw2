# load_pseudo.py
import torch
from torch.utils.data import DataLoader
from dataset.oxfordpet_pseudo import OxfordPetWithPseudo
from common import MODEL_SAVE_PATH

def load_pseudo(save_path, batch_size=32, shuffle=False, device=torch.device('cpu')):
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
    # Load original data
    pseudo_data = torch.load(save_path, map_location=device, weights_only=True)
    
    # Create dataset
    dataset = OxfordPetWithPseudo(pseudo_data)
    
    # Define collate_fn to maintain original format
    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        masks = torch.stack([item[1] for item in batch])
        return images, masks  # (B,3,H,W), (B,1,H,W)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load pseudo data (assuming save path is './pseudo_masks.pt')
    pseudo_loader = load_pseudo(
        f'{MODEL_SAVE_PATH}/resnet50_pet_cam_pseudo.pt',
        batch_size=16,
        shuffle=True,
        device=device
    )
    
    # Verify format compatibility
    print(len(pseudo_loader.dataset))  # Should output number of samples in pseudo dataset
    for images, masks in pseudo_loader:
        print(f"Image shape: {images.shape}")  # Should output torch.Size([16, 3, 224, 224])
        print(f"Mask shape: {masks.shape}")    # Should output torch.Size([16, 1, 224, 224])
        break