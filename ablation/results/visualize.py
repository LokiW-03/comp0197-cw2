import json
import matplotlib.pyplot as plt

def load_model_data(file_path):
    """
    Load the JSON log data from a file. The file is assumed
    to contain a JSON array; we return the first element.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    # If the file contains a list of experiments, return the first dictionary.
    if isinstance(data, list):
        return data[0]
    return data

def plot_metric(data, metric, model_name):
    """
    Plot the given metric (e.g., loss, accuracy, iou, dice) over epochs for
    all trials contained in the 'data' dictionary. Two curves per trial are plotted:
    one for training and one for testing.
    """
    plt.figure(figsize=(8, 6))
    
    # Loop over trials from the log file.
    for trial in data.get('trials', []):
        epochs = []
        train_vals = []
        test_vals = []
        
        # For each epoch, extract the numeric values for the chosen metric.
        for epoch in trial.get('epochs', []):
            epoch_num = epoch.get('epoch')
            
            # Get metric values for train and test; use None if a value is missing or if there is an error.
            # train_metric = epoch.get('train', {}).get(metric, None)
            test_metric = epoch.get('test', {}).get(metric, None)
            
            epochs.append(epoch_num)
            # Only append if the value is numeric; otherwise append None.
            # train_vals.append(train_metric if isinstance(train_metric, (int, float)) else None)
            test_vals.append(test_metric if isinstance(test_metric, (int, float)) else None)
        
        # Plot the training and test curves for the current trial.
        # plt.plot(epochs, train_vals, marker='o', label=f"Trial {trial.get('trial_number')} Train")
        plt.plot(epochs, test_vals, marker='x', label=f"Trial {trial.get('trial_number')} Test")
    
    plt.xlabel("Epoch")
    plt.ylabel(metric.capitalize())
    plt.title(f"{model_name} – {metric.capitalize()} vs. Epoch")
    plt.legend()
    plt.grid(True)
    plt.show()

# Define model names and the paths to their respective parsed log files.
models = [
    ("Efficient U-Net", "parsed_effunet.txt"),
    ("SegNet", "parsed_segnet.txt"),
    ("SegNeXt", "parsed_segnext.txt"),
    ("U-Net", "parsed_unet.txt"),
]

# Define which metrics to visualize.
# metrics = ["loss", "accuracy", "iou", "dice"]
metrics = ["accuracy", "iou"]

# Loop through each model and produce charts for each metric.
for model_name, file_path in models:
    try:
        model_data = load_model_data(file_path)
    except Exception as e:
        print(f"Error loading {model_name} data from {file_path}: {e}")
        continue

    print(f"Generating plots for {model_name} ...")
    for metric in metrics:
        plot_metric(model_data, metric, model_name)
