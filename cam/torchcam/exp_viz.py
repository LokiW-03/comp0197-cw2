from matplotlib import pyplot as plt
from cam.preprocessing import unnormalize
import torch
def vis(test_loader, model, taged=False):
    if taged:
        batch = next(iter(test_loader))
        images, masks = batch["image"], batch["mask"]
    else:
        images, masks = next(iter(test_loader))

    images = unnormalize(images.cpu()).to(masks.device)
    masks = masks.squeeze(1)

    # Switch the model to evaluation mode
    with torch.no_grad():
        model.eval()
        logits = model(images)  # Get raw logits from the model

    # Apply softmax to get class probabilities
    # Shape: [batch_size, num_classes, H, W]

    pr_masks = logits.softmax(dim=1)
    # Convert class probabilities to predicted class labels
    pr_masks = pr_masks.argmax(dim=1)  # Shape: [batch_size, H, W]

    # Visualize a few samples (image, ground truth mask, and predicted mask)
    for idx, (image, gt_mask, pr_mask) in enumerate(zip(images, masks, pr_masks)):
        if idx <= 1:  # Visualize first 5 samples
            plt.figure(figsize=(12, 6))

            # Original Image
            plt.subplot(1, 3, 1)
            plt.imshow(
                image.cpu().numpy().transpose(1, 2, 0)
            )  # Convert CHW to HWC for plotting
            plt.title("Image")
            plt.axis("off")

            # Ground Truth Mask
            plt.subplot(1, 3, 2)
            plt.imshow(gt_mask.cpu().numpy(), cmap="tab20")  # Visualize ground truth mask
            plt.title("Ground truth")
            plt.axis("off")

            # Predicted Mask
            plt.subplot(1, 3, 3)
            plt.imshow(pr_mask.cpu().numpy(), cmap="tab20")  # Visualize predicted mask
            plt.title("Prediction")
            plt.axis("off")

            # Save the figure
            import os
            os.makedirs("cam/torchcam/output", exist_ok=True)
            plt.savefig(f"cam/torchcam/output/image_{idx}.png")
        else:
            break
