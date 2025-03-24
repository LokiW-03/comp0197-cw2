import torch
from torch import nn
from torch.utils.data import DataLoader
from segnext import SegNeXt
from data import PetSegmentationDataset

if __name__ == '__main__':

    train_dataset = PetSegmentationDataset(split='trainval')
    test_dataset  = PetSegmentationDataset(split='test')

    # Create DataLoaders
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)

    num_batches = len(train_loader)
    print("Num batches", num_batches)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegNeXt(num_classes=3).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # expects raw logits and target class indices
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Optional: learning rate scheduler (poly schedule approximated by step decay here for simplicity)
    # We'll decay the LR by 0.1 every 15 epochs as a rough proxy
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    # ==================== TRAIN ====================

    num_epochs = 50
    model.train()
    for epoch in range(num_epochs):
        print("Epoch", epoch)
        total_loss = 0.0
        batch_no = 0
        for images, masks in train_loader:
            batch_no += 1
            print("Batch", batch_no)
            images = images.to(device)
            masks  = masks.to(device)
            optimizer.zero_grad()
            # Forward pass
            outputs = model(images)  # shape [B, 3, H, W]
            loss = criterion(outputs, masks)
            # Backward and optimize
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        scheduler.step()  # update learning rate
        avg_loss = total_loss / len(train_loader.dataset)
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}")


    # ==================== TEST ====================
    model.eval()
    iou_scores = torch.zeros(3)  # to accumulate IoU numerator (intersection) for each class
    union_scores = torch.zeros(3)  # to accumulate IoU denominator (union) for each class

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)  # [1, 3, H, W]
            preds = outputs.argmax(dim=1)  # [1, H, W], predicted class per pixel
            preds = preds.squeeze(0)  # remove batch dim
            masks = masks.squeeze(0)
            # Compute IoU components for each class
            for cls in range(3):
                # True positives: prediction and ground truth both equal this class
                intersection = ((preds == cls) & (masks == cls)).sum().item()
                # Union: prediction == cls or ground truth == cls (or both)
                union = ((preds == cls) | (masks == cls)).sum().item()
                iou_scores[cls] += intersection
                union_scores[cls] += union

    # Calculate mean IoU
    ious = iou_scores / (union_scores + 1e-6)
    mean_iou = ious.mean().item()
    print(f"IoU per class: Pet={ious[0]:.3f}, Background={ious[1]:.3f}, Border={ious[2]:.3f}")
    print(f"Mean IoU: {mean_iou:.3f}")


# import matplotlib.pyplot as plt
# import numpy as np

# # Function to overlay mask on image for visualization
# def overlay_mask_on_image(image, mask):
#     # image: [3, H, W] tensor, mask: [H, W] tensor of class indices
#     image = image.cpu().numpy().transpose(1,2,0)  # to HxWxC
#     image = (image * std + mean)  # de-normalize for display
#     image = (image * 255).astype(np.uint8)
#     mask = mask.cpu().numpy()
#     # Define colors for classes (BGR or RGB)
#     colors = np.array([
#         [255, 0, 0],    # pet: red
#         [0, 255, 0],    # background: green
#         [0, 0, 255]     # border: blue
#     ], dtype=np.uint8)
#     color_mask = colors[mask]  # shape HxWx3
#     overlay = (0.5 * image + 0.5 * color_mask).astype(np.uint8)
#     return overlay

# # Pick a few test samples to visualize
# model.eval()
# samples = [0, 1, 2]  # indices of test samples to visualize
# for idx in samples:
#     img, mask = test_dataset[idx]
#     img = img.to(device).unsqueeze(0)
#     with torch.no_grad():
#         pred = model(img).argmax(dim=1).squeeze(0)
#     overlay = overlay_mask_on_image(img.squeeze(0), pred)
#     plt.figure(figsize=(6,6))
#     plt.title("SegNeXt Prediction")
#     plt.imshow(overlay)
#     plt.axis('off')
#     plt.show()

