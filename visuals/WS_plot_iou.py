#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt

def load_data(file_path):
    """
    Load the JSON data from the specified file.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def plot_iou(data):
    """
    For each trial in the dataset, extract the epoch numbers, training IoU, and testing IoU values.
    The function then plots two subplots:
      - Left: Training IoU vs. Epoch.
      - Right: Testing IoU vs. Epoch.
    """
    # Create a figure with two side-by-side subplots.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for trial in data:
        trial_num = trial.get('trial_number', 'N/A')
        epochs_data = trial.get('epochs', [])
        # Sort epochs based on epoch number (if not already sorted)
        epochs_data.sort(key=lambda x: x.get('epoch', 0))
        
        # Extract epoch numbers, training IoU, and testing IoU for each epoch.
        epoch_numbers = [entry.get('epoch') for entry in epochs_data]
        train_iou = [entry.get('train', {}).get('iou') for entry in epochs_data]
        test_iou = [entry.get('test', {}).get('iou') for entry in epochs_data]

        # Plot training IoU against epoch.
        ax1.plot(epoch_numbers, train_iou, marker='o', label=f'Trial {trial_num}')
        # Plot testing IoU against epoch.
        ax2.plot(epoch_numbers, test_iou, marker='o', label=f'Trial {trial_num}')

    # Set titles, labels, and legends for the training IoU subplot.
    ax1.set_title('Training IoU vs. Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('IoU')
    ax1.legend()
    
    # Set titles, labels, and legends for the testing IoU subplot.
    ax2.set_title('Testing IoU vs. Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('IoU')
    ax2.legend()
    
    # Adjust layout and display the plot.
    plt.tight_layout()
    plt.savefig("WS_segnet_iou.png")

def main():    
    # Load the data from the specified file.
    data = load_data("../ablation/results/best_3_verbose_segnet.txt")
    
    # Plot the IoU values against epoch.
    plot_iou(data)

if __name__ == "__main__":
    main()
