import matplotlib.pyplot as plt

# Define the data for each model
data = {
    'EffUNet': [
        {'trial': 10, 'IoU': 0.3345284163951874, 'Accuracy': 0.5673018097877502},
        {'trial': 12, 'IoU': 0.29103508591651917, 'Accuracy': 0.5611971020698547},
        {'trial': 16, 'IoU': 0.27115553617477417, 'Accuracy': 0.5953456163406372}
    ],
    'SegNet': [
        {'trial': 20, 'IoU': 0.417961448431015, 'Accuracy': 0.6879932284355164},
        {'trial': 16, 'IoU': 0.41724562644958496, 'Accuracy': 0.7130351066589355},
        {'trial': 21, 'IoU': 0.41365084052085876, 'Accuracy': 0.6805597543716431}
    ],
    'SegNeXt': [
        {'trial': 5, 'IoU': 0.37294554710388184, 'Accuracy': 0.661917507648468},
        {'trial': 21, 'IoU': 0.3514586389064789, 'Accuracy': 0.6383455395698547},
        {'trial': 13, 'IoU': 0.3052079379558563, 'Accuracy': 0.5614440441131592}
    ],
    'UNet': [
        {'trial': 9, 'IoU': 0.36140650510787964, 'Accuracy': 0.695797324180603},
        {'trial': 13, 'IoU': 0.34374257922172546, 'Accuracy': 0.6200038194656372},
        {'trial': 21, 'IoU': 0.3289145231246948, 'Accuracy': 0.6505522131919861}
    ]
}

# Colors for each model
colors = {
    'EffUNet': 'red',
    'SegNet': 'blue',
    'SegNeXt': 'green',
    'UNet': 'orange'
}

plt.figure(figsize=(8, 6))

model_names = list(data.keys())
# Use integer positions for each model group on x-axis
positions = range(len(model_names))  # 0,1,2,3

# For each model, add scatter points for each trial.
# We add a small horizontal offset for each trial so that points don't overlap.
for i, model in enumerate(model_names):
    trials = data[model]
    # Offsets for each trial (e.g., -0.1, 0, 0.1)
    offsets = [-0.1, 0, 0.1]
    for j, trial in enumerate(trials):
        x = i + offsets[j]
        # Plot IoU as a circle
        plt.scatter(x, trial['IoU'], marker='o', color=colors[model],
                    label=f"{model} IoU" if j == 0 else "")
        # Plot Accuracy as a square
        plt.scatter(x, trial['Accuracy'], marker='s', color=colors[model],
                    label=f"{model} Accuracy" if j == 0 else "")

# Set x-axis ticks and labels corresponding to the model names
plt.xticks(positions, model_names)
plt.xlabel('Model')
plt.ylabel('Score')
plt.title('Scatter Plot of Best Trials: IoU (circles) and Accuracy (squares)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("WS_scatter_plot.png")
