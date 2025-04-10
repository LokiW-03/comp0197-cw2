import argparse
import torch
import os
import matplotlib.pyplot as plt

def plot_loss(model_name, file_format):
    history_path = f"./graph/{model_name}_loss_history.pt"
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"Loss history file not found: {history_path}")

    loss_history = torch.load(history_path)
    epochs = range(1, len(loss_history["total"]) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_history["cls"], label="CLS Loss")
    plt.plot(epochs, loss_history["rec"], label="REC Loss")
    plt.plot(epochs, loss_history["total"], label="Total Loss", linestyle="--")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curves ({model_name})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(ticks=epochs)

    os.makedirs("./graph", exist_ok=True)

    if file_format == "both":
        for ext in ["png", "eps"]:
            out_path = f"./graph/{model_name}_crm_loss_curve.{ext}"
            plt.savefig(out_path, format=ext)
            print(f"Saved loss curve to {out_path}")
    else:
        out_path = f"./graph/{model_name}_crm_loss_curve.{file_format}"
        plt.savefig(out_path, format=file_format)
        print(f"Saved loss curve to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        choices=["resnet", "resnet_drs", "efficientnet"],
                        help="Which model's loss history to plot")

    parser.add_argument('--format', type=str, default="both",
                        choices=["png", "eps", "both"],
                        help="Output file format: png | eps | both")
    args = parser.parse_args()

    plot_loss(args.model, args.format)
