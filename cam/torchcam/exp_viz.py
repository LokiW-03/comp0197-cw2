import os
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

def vis_2class(test_loader, model, taged=False):
    device = next(model.parameters()).device
    batch = next(iter(test_loader))

    if taged:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
    else:
        images, masks = batch
        images = images.to(device)
        masks = masks.to(device)

    if masks.ndim == 4:
         masks = masks.squeeze(1)

    images_unnorm = unnormalize(images.cpu())

    model.eval()
    with torch.no_grad():
        logits = model(images)

    # --- Binary Prediction Processing ---
    probs = logits.sigmoid()
    preds = (probs > 0.5).long().squeeze(1)
    # --- End Binary Processing ---

    num_samples_to_show = min(5, images.shape[0])
    output_dir = "cam/torchcam/output" # Consider making this an argument
    os.makedirs(output_dir, exist_ok=True)

    for idx in range(num_samples_to_show):
        image = images_unnorm[idx].cpu()
        gt_mask = masks[idx].cpu()
        pr_mask = preds[idx].cpu()

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(image.numpy().transpose(1, 2, 0))
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(gt_mask.numpy(), cmap="gray", vmin=0, vmax=1)
        plt.title("Ground Truth") # Add "(0=FG)" etc. if needed
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pr_mask.numpy(), cmap="gray", vmin=0, vmax=1)
        plt.title("Prediction") # Add "(0=FG)" etc. if needed
        plt.axis("off")

        plt.savefig(os.path.join(output_dir, f"vis_2class_sample_{idx}.png"))
        plt.close()

    print(f"Saved {num_samples_to_show} 2-class visualization samples to {output_dir}")
