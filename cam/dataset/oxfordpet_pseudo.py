from torch.utils.data import Dataset


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