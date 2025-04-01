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
