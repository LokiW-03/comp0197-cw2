import torch
import argparse

from data_utils.data import trainset, testset
from torch.utils.data import DataLoader
from cam.load_pseudo import load_pseudo

parser = argparse.ArgumentParser()
parser.add_argument('--pseudo_path', type=str, default='./pseudo_masks.pt', help='Path to pseudo masks')

args = parser.parse_args()

device = torch.device("cpu")

# Load pseudo masks
pseudo_loader = load_pseudo(args.pseudo_path, batch_size=32, shuffle=True, device=device)

trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
testloader = DataLoader(testset, batch_size=32, shuffle=False)


def compute_label_distribution(dataloader):
    """
    Computes the distribution of label values (0, 1, 2) from label tensors with shape (B, 1, H, W)
    provided by the torch DataLoader.
    
    Assumptions:
    - Each batch from the dataloader is a tuple (inputs, labels).
    - 'labels' is a tensor of shape (B, 1, H, W) with pixel values in the set {0, 1, 2}.
    
    Returns:
    - A torch tensor of shape (3,), where each element is the count of occurrences of 0, 1, and 2 respectively.
    """
    # Initialize a counter tensor for classes 0, 1, and 2
    distribution = torch.zeros(3, dtype=torch.int64)
    
    # Iterate over each batch from the dataloader
    for batch in dataloader:
        # Assuming each batch is a tuple: (inputs, labels)
        inputs, labels = batch  # labels shape: (B, 1, H, W)
        
        # Remove the channel dimension to obtain a shape of (B, H, W)
        labels = labels.squeeze(1)
        
        # Flatten the labels to a 1D tensor containing all label values in the batch
        labels_flat = labels.view(-1)
        
        # Compute the counts of 0, 1, and 2 using torch.bincount
        batch_counts = torch.bincount(labels_flat, minlength=3)
        
        # Update the overall distribution by summing up the counts from this batch
        distribution += batch_counts
    
    # display in percentage
    distribution = distribution.float() / distribution.sum()

    return distribution


print("Bit mask distribution for trainset:")
train_distribution = compute_label_distribution(trainloader)
print(train_distribution)

print("Bit mask distribution for testset:")
test_distribution = compute_label_distribution(testloader)
print(test_distribution)

print("Bit mask distribution for pseudo masks:")
pseudo_distribution = compute_label_distribution(pseudo_loader)
print(pseudo_distribution)