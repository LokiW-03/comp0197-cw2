# Heatmap of IoU Across Trials for All Models
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Hardcoded IoU DataFrame
data = [
    [0, 0.23, 0.41, 0.21, 0.22],
    [1, 0.23, 0.40, 0.29, 0.28],
    [2, 0.23, 0.38, 0.12, 0.21],
    [3, 0.25, 0.31, 0.11, 0.21],
    [4, 0.21, 0.40, 0.15, 0.31],
    [5, 0.26, 0.41, 0.37, 0.30],
    [6, 0.21, 0.37, 0.19, 0.22],
    [7, 0.21, 0.38, 0.25, 0.20],
    [8, 0.23, 0.38, 0.12, 0.29],
    [9, 0.23, 0.37, 0.27, 0.36],
    [10, 0.34, 0.29, 0.13, 0.22],
    [11, 0.21, 0.33, 0.13, 0.21],
    [12, 0.29, 0.38, 0.12, 0.11],
    [13, 0.24, 0.34, 0.31, 0.34],
    [14, 0.22, 0.38, 0.21, 0.21],
    [15, 0.20, 0.34, 0.28, 0.21],
    [16, 0.27, 0.42, 0.24, 0.21],
    [17, 0.24, 0.38, 0.30, 0.21],
    [18, 0.23, 0.26, 0.23, 0.22],
    [19, 0.23, 0.32, 0.11, 0.20],
    [20, 0.21, 0.42, 0.12, 0.23],
    [21, 0.26, 0.41, 0.35, 0.33],
    [22, 0.21, 0.39, 0.13, 0.22],
    [23, 0.20, 0.36, 0.12, 0.21],
]

columns = ['trial'] + ['effunet', 'segnet', 'segnext', 'unet']
iou_df = pd.DataFrame(data, columns=columns)
iou_df.set_index('trial', inplace=True)

# Plot and save the IoU heatmap
plt.figure(figsize=(8, 8))
sns.heatmap(iou_df, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={"label": "IoU"})
plt.title("IoU per Trial Across Models")
plt.ylabel("Trial")
plt.xlabel("Model")
plt.tight_layout()
plt.savefig("iou_heatmap.png", dpi=300)
plt.show()
