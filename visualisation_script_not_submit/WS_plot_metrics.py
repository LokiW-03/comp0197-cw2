import matplotlib.pyplot as plt

# Creating plots for weakly supervised case 
epochs = [2, 4, 6, 8, 10]

SegNet = {
    "accuracy": [0.5385, 0.6371, 0.6341, 0.5461, 0.6246],
    "iou": [0.2987, 0.3834, 0.3733, 0.3149, 0.3316],
    "precision": [0.5303, 0.6067, 0.6133, 0.5917, 0.5620],
    "recall": [0.4583, 0.5324, 0.5261, 0.4889, 0.4716],
    "dice": [0.4208, 0.5159, 0.5045, 0.4440, 0.4505]
}

EffUNet = {
    "accuracy": [0.6110, 0.5694, 0.5443, 0.6017, 0.6311],
    "iou": [0.2249, 0.1909, 0.1815, 0.2308, 0.2208],
    "precision": [0.3815, 0.5195, 0.5330, 0.5952, 0.2458],
    "recall": [0.3387, 0.3342, 0.3334, 0.3553, 0.3340],
    "dice": [0.2930, 0.2441, 0.2351, 0.2974, 0.2744]
}

SegNeXt = {
    "accuracy": [0.2504, 0.3627, 0.3701, 0.2910, 0.2956],
    "iou": [0.0835, 0.1209, 0.1234, 0.0970, 0.0985],
    "precision": [0.7501, 0.7876, 0.7900, 0.7637, 0.7652],
    "recall": [0.3333, 0.3333, 0.3333, 0.3333, 0.3333],
    "dice": [0.1335, 0.1774, 0.1801, 0.1503, 0.1521]
}

UNet = {
    "accuracy": [0.6780, 0.6143, 0.5934, 0.6505, 0.5903],
    "iou": [0.2904, 0.2625, 0.2139, 0.2719, 0.2055],
    "precision": [0.8304, 0.7584, 0.7196, 0.8276, 0.4179],
    "recall": [0.4028, 0.3919, 0.3458, 0.3930, 0.3413],
    "dice": [0.3807, 0.3565, 0.2809, 0.3611, 0.2657]
}

metrics = ['accuracy', 'iou', 'precision', 'recall', 'dice']
models = [SegNet, EffUNet, SegNeXt, UNet]
labels = ['SegNet', 'EffUNet', 'SegNeXt', 'UNet']

for metric in metrics:
    plt.figure(figsize=(8, 6))
    for model, label in zip(models, labels):
        plt.plot(epochs, model[metric], marker='o', label=label)
    plt.title(f'{metric.capitalize()} vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid(True)
    plt.savefig("visuals/WS_" + metric + "_plot.png", bbox_inches='tight')
    print(f"Saved WS_{metric}_plot.png")
    plt.close()
