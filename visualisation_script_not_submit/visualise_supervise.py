# # visualise_supervise.py
# # python -m model.visualise --model_path .\saved_models\segnet_epoch_10.pth --model_type segnet --num_images 10

# # import matplotlib.pyplot as plt
# # import numpy as np

# # # Function to overlay mask on image for visualization
# # def overlay_mask_on_image(image, mask):
# #     # image: [3, H, W] tensor, mask: [H, W] tensor of class indices
# #     image = image.cpu().numpy().transpose(1,2,0)  # to HxWxC
# #     image = (image * std + mean)  # de-normalize for display
# #     image = (image * 255).astype(np.uint8)
# #     mask = mask.cpu().numpy()
# #     # Define colors for classes (BGR or RGB)
# #     colors = np.array([
# #         [255, 0, 0],    # pet: red
# #         [0, 255, 0],    # background: green
# #         [0, 0, 255]     # border: blue
# #     ], dtype=np.uint8)
# #     color_mask = colors[mask]  # shape HxWx3
# #     overlay = (0.5 * image + 0.5 * color_mask).astype(np.uint8)
# #     return overlay

# # # Pick a few test samples to visualize
# # model.eval()
# # samples = [0, 1, 2]  # indices of test samples to visualize
# # for idx in samples:
# #     img, mask = test_dataset[idx]
# #     img = img.to(device).unsqueeze(0)
# #     with torch.no_grad():
# #         pred = model(img).argmax(dim=1).squeeze(0)
# #     overlay = overlay_mask_on_image(img.squeeze(0), pred)
# #     plt.figure(figsize=(6,6))
# #     plt.title("SegNeXt Prediction")
# #     plt.imshow(overlay)
# #     plt.axis('off')
# #     plt.show()

# import torch
# import argparse
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# from torch.utils.data import DataLoader

# from data import testset
# from baseline_segnet import SegNet
# from efficient_unet import EfficientUNet
# from baseline_unet import UNet
# from segnext import SegNeXt

# def create_and_save_collage(image_list, gt_list, pred_list, num_classes, output_path):
#     """
#     Creates a collage of images, ground truths, and predictions and saves it.

#     Args:
#         image_list (list): List of NumPy arrays for original images (H, W, C).
#         gt_list (list): List of NumPy arrays for ground truth masks (H, W).
#         pred_list (list): List of NumPy arrays for predicted masks (H, W).
#         num_classes (int): Number of segmentation classes for color mapping.
#         output_path (str): Path to save the final collage image.
#     """
#     num_images = len(image_list)
#     if num_images == 0:
#         print("No images to create a collage.")
#         return

#     n_rows = num_images
#     n_cols = 3
#     # Adjust figsize for clarity (width, height) - e.g., 5 units per column/row
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))

#     # Handle case where num_images = 1, plt.subplots returns a 1D axes array
#     if n_rows == 1:
#         axes = axes.reshape(1, -1) # Make axes always 2D for consistent indexing

#     for i in range(n_rows):
#         img_np = image_list[i]
#         gt_mask_np = gt_list[i]
#         pred_mask_np = pred_list[i]

#         # --- Plotting on the grid ---
#         ax_img = axes[i, 0]
#         ax_gt = axes[i, 1]
#         ax_pred = axes[i, 2]

#         # Adjust image range if necessary for display (e.g., reverse normalization)
#         # Assuming ToTensor scaled to [0, 1]
#         img_display = np.clip(img_np, 0, 1) if img_np.min() >= 0 and img_np.max() <=1 else img_np

#         # Original Image
#         ax_img.imshow(img_display)
#         ax_img.set_title(f"Image {i+1}")
#         ax_img.axis('off')

#         # Ground Truth Mask
#         # Use consistent color mapping across GT and Prediction
#         cmap = 'viridis'
#         vmin = 0
#         vmax = num_classes - 1
#         ax_gt.imshow(gt_mask_np, cmap=cmap, vmin=vmin, vmax=vmax)
#         ax_gt.set_title(f"Ground Truth {i+1}")
#         ax_gt.axis('off')

#         # Predicted Mask
#         ax_pred.imshow(pred_mask_np, cmap=cmap, vmin=vmin, vmax=vmax)
#         ax_pred.set_title(f"Prediction {i+1}")
#         ax_pred.axis('off')

#     plt.tight_layout()

#     # --- Save the figure ---
#     try:
#         # Ensure output directory exists
#         output_dir = os.path.dirname(output_path)
#         if output_dir:
#             os.makedirs(output_dir, exist_ok=True)

#         plt.savefig(output_path)
#         print(f"Collage saved successfully to {output_path}")
#     except Exception as e:
#         print(f"Error saving collage to {output_path}: {e}")
#     finally:
#         plt.close(fig) # Close the figure to free memory


# def generate_visualisation_data(model, dataloader, device, num_images=10):
#     """
#     Generates data needed for visualization.

#     Args:
#         model (torch.nn.Module): The trained segmentation model.
#         dataloader (torch.utils.data.DataLoader): DataLoader for the test set (batch_size=1 recommended).
#         device: The device to run inference on (cpu, cuda, mps).
#         num_images (int): The number of images to process.

#     Returns:
#         tuple: (list_images, list_gts, list_preds) containing NumPy arrays.
#     """
#     model.eval()
#     count = 0

#     images_to_plot = []
#     gts_to_plot = []
#     preds_to_plot = []

#     print(f"Generating prediction data for {num_images} images...")

#     with torch.no_grad():
#         for i, (image, gt_mask) in enumerate(dataloader):
#             if count >= num_images:
#                 break

#             image, gt_mask = image.to(device), gt_mask.to(device)

#             # Ensure gt_mask is (B, H, W) or (H, W) if B=1
#             if gt_mask.dim() > 3:
#                  gt_mask = gt_mask.squeeze(1)

#             # Get model prediction
#             pred_logits = model(image) # Shape: (B, num_classes, H, W)
#             pred_mask = torch.argmax(pred_logits, dim=1) # Shape: (B, H, W)

#             # Move data to CPU and convert to NumPy for plotting/saving
#             img_np = image.squeeze(0).cpu().numpy().transpose(1, 2, 0) # (H, W, C)
#             gt_mask_np = gt_mask.squeeze(0).cpu().numpy() # (H, W)
#             pred_mask_np = pred_mask.squeeze(0).cpu().numpy() # (H, W)

#             # Append results
#             images_to_plot.append(img_np)
#             gts_to_plot.append(gt_mask_np)
#             preds_to_plot.append(pred_mask_np)

#             count += 1

#     return images_to_plot, gts_to_plot, preds_to_plot


# def main():
#     parser = argparse.ArgumentParser(description="Visualise segmentation model predictions by saving a collage")
#     parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint (.pth file)')
#     parser.add_argument('--model_type', type=str, default='effunet', choices=['segnet', 'segnext', 'effunet', 'unet'], help='Type of segmentation model architecture used for training')
#     parser.add_argument('--num_images', type=int, default=10, help='Number of images to include in the collage')
#     parser.add_argument('--collapse_contour', action='store_true', help='Indicates if contour class (2) was collapsed to background (0) during training/data loading')
#     parser.add_argument('--output_file', type=str, default='output/model_grid.jpg', help='Path to save the output collage image')
    
#     args = parser.parse_args()

#     # --- Device Setup ---
#     device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
#     print(f'Using device: {device}')

#     # --- Determine Number of Classes ---
#     num_classes = 2 if args.collapse_contour else 3
#     print(f"Expecting {num_classes} output classes from the model.")

#     # --- Load Data ---
#     custom_collate_fn = None
#     if args.collapse_contour:
#         print("Using custom collate function to collapse contour class.")
#         def custom_collate_fn_viz(batch):
#             # Assumes batch is list of (image_tensor, mask_tensor)
#             masks = torch.stack([item[1] for item in batch])
#             masks[masks == 2] = 0 # Collapse class 2 -> 0
#             images = torch.stack([item[0] for item in batch])
#             return images, masks
#         custom_collate_fn = custom_collate_fn_viz

#     # Shuffle=False ensures we get the same images each time (usually the first N).
#     test_loader = DataLoader(
#         testset,
#         batch_size=1,
#         shuffle=False,
#         collate_fn=custom_collate_fn
#     )
#     print(f"Test dataset loaded with {len(testset)} samples.")

#     # --- Instantiate Model ---
#     print(f"Instantiating model type: {args.model_type}")
#     if args.model_type == 'segnext':
#         model = SegNeXt(num_classes=num_classes).to(device)
#     elif args.model_type == 'segnet':
#         model = SegNet(n_classes=num_classes).to(device)
#     elif args.model_type == 'unet':
#         model = UNet(n_channels=3, n_classes=num_classes).to(device)
#     elif args.model_type == 'effunet':
#         model = EfficientUNet(num_classes=num_classes).to(device)
#     else:
#         print(f"Warning: Model type '{args.model_type}' not explicitly handled, trying EfficientUNet.")
#         model = EfficientUNet(num_classes=num_classes).to(device)

#     # --- Load Model Weights ---
#     if not os.path.exists(args.model_path):
#         print(f"Error: Model checkpoint not found at {args.model_path}")
#         return

#     print(f"Loading model weights from: {args.model_path}")
#     try:
#         # Use weights_only=False for compatibility with checkpoints saved with optimizer/scheduler state
#         checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)

#         # Attempt to load the state dictionary from the checkpoint structure
#         if isinstance(checkpoint, dict):
#             if 'model' in checkpoint:
#                 # Common case: checkpoint is a dict with 'model' key
#                 state_dict = checkpoint['model']
#             elif 'state_dict' in checkpoint:
#                 # Another common case: key is 'state_dict'
#                 state_dict = checkpoint['state_dict']
#             else:
#                 # Less common: Assume the checkpoint dict *is* the state_dict
#                 state_dict = checkpoint
#         else:
#              # If checkpoint is not a dict, assume it *is* the state_dict directly
#              state_dict = checkpoint

#         # Load the extracted state dictionary into the model
#         model.load_state_dict(state_dict)
#         print("Model weights loaded successfully.")

#     except Exception as e:
#         # Catch any exception during torch.load or model.load_state_dict
#         print(f"\nError loading model checkpoint from {args.model_path}.")
#         print(f"Details: {e}")
#         print("Please ensure the model path is correct, the checkpoint file is not corrupted,")
#         print("and the model architecture matches the one used for saving the checkpoint.")
#         return

#     # --- Generate Data ---
#     img_list, gt_list, pred_list = generate_visualisation_data(
#         model,
#         test_loader,
#         device,
#         num_images=args.num_images
#     )

#     # --- Create and Save Collage ---
#     create_and_save_collage(
#         img_list,
#         gt_list,
#         pred_list,
#         num_classes=num_classes,
#         output_path=args.output_file
#     )

# if __name__ == "__main__":
#     main()
