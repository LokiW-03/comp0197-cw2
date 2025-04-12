import torch
import argparse
import itertools
import torch.nn as nn
from torch.utils.data import DataLoader

from cam.load_pseudo import load_pseudo
from cam.postprocessing import generate_pseudo_masks
from data_utils.data import testset, get_cam_pet_dataset
from model.baseline_segnet import SegNet
from model.efficient_unet import EfficientUNet
from model.segnext import SegNeXt
from model.baseline_unet import UNet
from model.resnet_gradcampp import ResNet50_CAM, GradCAMpp
from supervised.train import train_model, compute_test_metrics_fn
from supervised.loss import DiceLoss
from ablation.search_tools import save_results_to_csv



def search(seg_model_name, # Segmentation model
           model_path, # Path to the classification model
           save_path, # temporary path to save pseudo masks
           batch_size,
           thres_low, # Low threshold for CAM
           thres_high, # High threshold for CAM
           epochs, # Number of epochs for training
           loss_fn, # Loss function
           optimizer_generator: callable, # Optimizer
           scheduler_generator: callable, # Learning rate scheduler
           ):
    """
    Runs a pseudo-mask-based weakly supervised segmentation pipeline using a CAM-based classifier.

    This function performs the following steps:
        1. Loads a pre-trained classification model (ResNet50_CAM).
        2. Generates Class Activation Maps (CAMs) using Grad-CAM++.
        3. Applies thresholding to create pseudo segmentation masks.
        4. Loads the pseudo masks as a training dataset.
        5. Selects and initializes a segmentation model based on `seg_model_name`.
        6. Trains the segmentation model using the pseudo masks.
        7. Evaluates the trained model on the test set and returns test metrics.

    Args:
        seg_model_name (str): Name of the segmentation model to use. Options include
            "segnet", "efficientunet", "segnext", or "unet".
        model_path (str): Path to the saved classification model weights.
        save_path (str): Temporary directory path to save generated pseudo masks.
        batch_size (int): Batch size for training and evaluation.
        thres_low (float): Lower threshold for CAM values to include in the pseudo mask.
        thres_high (float): Upper threshold for CAM values to include in the pseudo mask.
        epochs (int): Number of training epochs for the segmentation model.
        loss_fn (callable): Loss function to optimize during training.
        optimizer_generator (callable): Function that accepts a model and returns an optimizer.
        scheduler_generator (callable): Function that accepts an optimizer and returns a scheduler.

    Returns:
        dict: A dictionary of test evaluation metrics computed after training the segmentation model.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ResNet50_CAM(37)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    cam_generator = lambda model: GradCAMpp(model)

    train_loader, _ = get_cam_pet_dataset(with_paths=True)
    generate_pseudo_masks(train_loader, model, cam_generator, 
                          save_path=save_path,
                          threshold_low=thres_low,
                          threshold_high=thres_high,
                          device=device,
                          # do not generate visualization to make it faster
                          side_effect=False)
    pseudo_loader = load_pseudo(save_path, batch_size=batch_size, shuffle=True, device=device, collapse_contour=False)
    trainval_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    if seg_model_name == "segnet":
        seg_model = SegNet()
    elif seg_model_name == "efficientunet":
        seg_model = EfficientUNet()
    elif seg_model_name == "segnext":
        seg_model = SegNeXt(num_classes=3)
    elif seg_model_name == "unet":
        seg_model = UNet(3, 3)
    else:
        raise ValueError(f"Invalid segmentation model: {seg_model_name}")
    
    seg_model.to(device)
    seg_model.eval()
    optimizer = optimizer_generator(seg_model)
    scheduler = scheduler_generator(optimizer)
    train_model(seg_model, pseudo_loader, trainval_loader, loss_fn, optimizer, epochs, device, compute_test_metrics = True, model_name=seg_model_name, scheduler=scheduler, verbose=False)
    test_metrics = compute_test_metrics_fn(seg_model, test_loader, loss_fn, device, num_classes = 3, num_eval_batches=None)

    return test_metrics

if __name__ == "__main__":
    results = []
    parser = argparse.ArgumentParser(description="Grid search for multiple hyperparameters")
    parser.add_argument('--model', type=str, default='segnet', choices=['segnet', 'segnext', 'effunet', 'unet'], help='Segmentation model')
    parser.add_argument("--model_path", type=str, default="cam/saved_models/resnet_pet_cam.pth")
    parser.add_argument("--result_path", type=str, default="./grid_search_results.csv", help="Path to search results")
    args = parser.parse_args()

    MAX_TRIALS = 40
    EPOCHS = 10
    max_iou = 0
    best_params = {}
    best_metrics = None

    cam_threshold_space = [
        (0.25, 0.325),
        (0.3, 0.7), # Use a common setting to avoid bias
        (0.21, 0.33)
    ]
    loss_fn_space = {
        "ce_mean": nn.CrossEntropyLoss(reduction='mean'),
        "dice_loss": DiceLoss(),
    }
    optimizer_generator_space = {
        "adamw_1e-3_1e-4": lambda model: torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4),
        "adamw_1e-2_1e-4": lambda model: torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    }
    scheduler_generator_space = {
        "steplr_15_0.1": lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1),
    }
    batch_size_space = [16, 64]

    # generate all combinations of parameters
    param_combinations = itertools.product(
        cam_threshold_space,
        loss_fn_space.keys(),
        optimizer_generator_space.keys(),
        scheduler_generator_space.keys(),
        batch_size_space
    )

    for i, (thresholds, loss_fn_key, opt_gen_key, sch_gen_key, batch_size) in enumerate(param_combinations):
        if i == MAX_TRIALS:
            print("Max trials reached.")
            break
        thres_low, thres_high = thresholds
        
        print(f"\nTrial {i}:")
        
        result = {
            "parameters": {
                "thresholds": (thres_low, thres_high),
                "loss_fn": loss_fn_key,
                "optimizer": opt_gen_key,
                "scheduler": sch_gen_key,
                "batch_size": batch_size,
            }
        }

        print(f"Parameters: {result['parameters']}")

        test_metrics = search(
            seg_model_name=args.model,
            model_path=args.model_path,
            save_path=f"cam/saved_models/resnet_gradcampp_trial_pseudo.pt",
            batch_size=batch_size,
            thres_low=thres_low,
            thres_high=thres_high,
            epochs=EPOCHS,
            loss_fn=loss_fn_space[loss_fn_key],
            optimizer_generator=optimizer_generator_space[opt_gen_key],
            scheduler_generator=scheduler_generator_space[sch_gen_key]
        )

        iou = test_metrics["iou"]
        result.update({
            "iou": iou,
            "metrics": test_metrics
        })
        results.append(result)

        if iou > max_iou:
            max_iou = iou
            best_params = result["parameters"]
            best_metrics = test_metrics

    print("\nBest parameters:")
    print(f"Best parameters: {best_params}")
    print(f"Max IoU: {max_iou:.4f}")
    print("Best metrics:", best_metrics)

    save_results_to_csv(results, filename="grid_search_results.csv")
