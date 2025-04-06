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

def search(model_path, save_path, batch_size, thres_low, thres_step, epochs):
    device = torch.device('cuda' if torch.cuda.is_available()
                        else 'mps' if torch.backends.mps.is_available()                          
                           else 'cpu')
    
    model = ResNet50_CAM(37)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    cam_generator = lambda model: GradCAMpp(model)

    train_loader, test_loader = download_pet_dataset(with_paths=True)
    generate_pseudo_masks(train_loader, model, cam_generator, 
                          save_path=save_path,
                          threshold_low=thres_low,
                          threshold_high=thres_low + thres_step,
                          device=device)
    pseudo_loader = load_pseudo(save_path, batch_size=batch_size, shuffle=True, device=device, collapse_contour=False)
    trainval_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)


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

    parser = argparse.ArgumentParser(description="Grid search for CAM thresholds")
    parser.add_argument("--model_path", type=str, default="cam/saved_models/resnet50_pet_cam.pth", help="Path to the model")
    args = parser.parse_args()
    random.seed(42)

    N_TRIALS = 10
    EPOCHS = 5

    for i in range(N_TRIALS):
        thres_low = random.uniform(0, 1)
        thres_step = random.uniform(0, 1-thres_low)

        print(f"Trial {i+1}: thres_low={thres_low:.2f}, thres_step={thres_step:.2f}")

        test_metrics = search(
            model_path=args.model_path,
            save_path=f"cam/saved_models/resnet50_gradcampp_trial_pseudo.pt",
            batch_size=64,
            thres_low=thres_low,
            thres_step=thres_step,
            epochs=EPOCHS
        )