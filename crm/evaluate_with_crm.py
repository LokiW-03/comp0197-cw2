import os
import argparse
import torch
from torch.utils.data import DataLoader
from model.resnet_gradcampp import ResNet50_CAM, GradCAMpp
from model.resnet_drs import ResNet50_CAM_DRS
from model.reconstruct_net import ReconstructNet
from crm.visualize import visualize_recon_grid
from crm import CRM_MODEL_SAVE_PATH, NUM_CLASSES
from data_utils.data import crm_testset


def evaluate_crm(model_name='resnet', save_dir='crm_eval_outputs'):
    """
    Evaluate generated crm

    :param model_name: model used to generate crm
    :param save_dir: dir to save evaluation output
    """
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    testset = crm_testset
    test_loader = DataLoader(testset, batch_size=8, shuffle=False)

  
    # Load models
    if model_name == 'resnet':
        cam_model = ResNet50_CAM(NUM_CLASSES).to(device)
        cam_model.load_state_dict(torch.load(f"{CRM_MODEL_SAVE_PATH}/resnet_pet_gradcampp_crm.pth", map_location=device, weights_only=True))
        cam_generator = GradCAMpp(cam_model)
        recon_model = ReconstructNet(NUM_CLASSES).to(device)
        recon_model.load_state_dict(torch.load(f"{CRM_MODEL_SAVE_PATH}/reconstruct_net_resnet.pth", map_location=device, weights_only=True))
    
    elif model_name == 'resnet_drs':
        cam_model = ResNet50_CAM_DRS(NUM_CLASSES).to(device)
        cam_model.load_state_dict(torch.load(f"{CRM_MODEL_SAVE_PATH}/resnet_drs_pet_gradcampp_crm.pth", map_location=device, weights_only=True))
        cam_generator = GradCAMpp(cam_model)
        recon_model = ReconstructNet(NUM_CLASSES).to(device)
        recon_model.load_state_dict(torch.load(f"{CRM_MODEL_SAVE_PATH}/reconstruct_net_resnet_drs.pth", map_location=device, weights_only=True))

    cam_model.eval()
    recon_model.eval()

    correct = 0
    total = 0

    sample_images = []
    sample_recons = []

    counter = 0
    for images, labels in test_loader:
        if (counter + 1) % (len(test_loader) // 10) == 0 or (counter + 1) == len(test_loader):
            percentage = int((counter + 1) / len(test_loader) * 100)
            print(f'Processing: {percentage}%')
        counter += 1

        images = images.to(device)
        labels = labels.to(device)
        cams, logits = cam_generator.generate_cam(images, all_classes=True, resize=False)
        
        with torch.no_grad():
            logits = cam_model(images)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            recon = recon_model(cams)

            # Save 5 samples for visualization
            if len(sample_images) < 5:
                needed = 5 - len(sample_images)
                sample_images.extend(images[:needed])
                sample_recons.extend(recon[:needed])

        

    acc = correct / total

    print("=== Evaluation Results ===")
    print(f"Classification Accuracy (full test set): {acc:.4f}")

    # Visualization
    if sample_images:
        stacked_input = torch.stack(sample_images)
        stacked_recon = torch.stack(sample_recons)
        save_path = f"{save_dir}/recon_comparison.jpg"
        visualize_recon_grid(stacked_input, stacked_recon, save_path, nrow=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'resnet_drs'])
    args = parser.parse_args()

    evaluate_crm(args.model)
