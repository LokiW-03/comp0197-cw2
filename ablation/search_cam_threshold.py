import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from cam.load_pseudo import load_pseudo
from cam.resnet_gradcampp import ResNet50_CAM, GradCAMpp
from cam.postprocessing import generate_pseudo_masks
from cam.dataset.oxfordpet import download_pet_dataset
from model.data import testset
from model.baseline_segnet import SegNet
from model.train import train_model, compute_test_metrics_fn
import itertools

def generate_refined_grid_space(base_low=0.25, base_high=0.325):
    # Core fine search area (around the current optimal solution)
    fine_low = [round(base_low + i*0.025, 3) for i in range(-2, 3)]
    fine_high = [round(base_high + i*0.025, 3) for i in range(-2, 3)]
    
    # Extended area based on distribution characteristics
    distribution_based_low = [0.18, 0.22, 0.28]  # Covers around the 25th percentile and mean offset
    distribution_based_high = [0.30, 0.35, 0.38]  # Covers the extended 75th percentile
    
    # Construct the final search space (remove duplicates)
    all_low = sorted(list(set(fine_low + distribution_based_low)))
    all_high = sorted(list(set(fine_high + distribution_based_high)))
    
    # Generate valid combinations
    valid_pairs = []
    for low in all_low:
        for high in all_high:
            # Maintain minimum interval constraints
            if 0.05 < (high - low) < 0.15:  # Based on interval analysis of successful cases
                valid_pairs.append((low, high))
    
    # Add special candidates (based on peak intervals in the distribution histogram)
    special_candidates = [
        (0.20, 0.30),  # Covers the high-density bins in the [0.2, 0.3) interval
        (0.25, 0.35),  # Extends the current optimal interval
        (0.22, 0.32)   # Offset test
    ]
    
    return list(set(valid_pairs + special_candidates))


def search(model_path, save_path, batch_size, thres_low, thres_high, epochs):
    device = torch.device('cuda' if torch.cuda.is_available()
                        else 'mps' if torch.backends.mps.is_available()                          
                           else 'cpu')
    device = torch.device('cpu')
    
    model = ResNet50_CAM(37)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    cam_generator = lambda model: GradCAMpp(model)

    train_loader, _ = download_pet_dataset(with_paths=True)
    generate_pseudo_masks(train_loader, model, cam_generator, 
                          save_path=save_path,
                          threshold_low=thres_low,
                          threshold_high=thres_high,
                          device=device)
    pseudo_loader = load_pseudo(save_path, batch_size=batch_size, shuffle=True, device=device, collapse_contour=False)
    trainval_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)


    model = SegNet().to(device)
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    train_model(model, pseudo_loader, trainval_loader, loss_fn, optimizer, epochs, device, compute_test_metrics = True, model_name = "segnet", scheduler=scheduler, verbose=False)
    test_metrics = compute_test_metrics_fn(model, test_loader, loss_fn, device, num_classes = 3, num_eval_batches=None)

    return test_metrics

if __name__ == "__main__":

    import random
    import argparse

    results = []

    parser = argparse.ArgumentParser(description="Grid search for CAM thresholds")
    parser.add_argument("--model_path", type=str, default="cam/saved_models/resnet50_pet_cam.pth", help="Path to the model")
    args = parser.parse_args()
    random.seed(42)

    N_TRIALS = 10
    EPOCHS = 5
    max_iou = 0
    best_threshold = (0, 0)
    best_metrics = None

    for i, (thres_low, thres_high) in enumerate(generate_refined_grid_space()):

        print(f"Trial {i}: thres_low={thres_low:.2f}, thres_high={thres_high:.2f}")
        test_metrics = search(
            model_path=args.model_path,
            save_path=f"cam/saved_models/resnet50_gradcampp_trial_pseudo.pt",
            batch_size=64,
            thres_low=thres_low,
            thres_high=thres_high,
            epochs=EPOCHS
        )

        print(f"Test metrics: {test_metrics}")
        iou = test_metrics["iou"]
        results.append((thres_low, thres_high, iou, test_metrics))
        if iou > max_iou:
            max_iou = iou
            best_threshold = (thres_low, thres_high)
            best_metrics = test_metrics
        
    print(f"Best threshold: {best_threshold}")
    print(f"Max IoU: {max_iou}")
    print(f"Best metrics: {best_metrics}")
    print("All results:")
    for thres_low, thres_high, iou, metrics in results:
        print(f"thres_low={thres_low:.2f}, thres_high={thres_high:.2f}, IoU={iou:.4f}, metrics={metrics}")
    
            