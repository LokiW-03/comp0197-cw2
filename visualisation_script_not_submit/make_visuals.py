# Import packages
import torch
from torch import nn
from torch.utils.data import DataLoader
from metrics import compute_metrics
from baseline_segnet import SegNet
from efficient_unet import EfficientUNet
from baseline_unet import UNet
from segnext import SegNeXt
from data_utils.data import testset
from montage import visualise_fs_segmentation

SAVE_WEIGHTS_FREQUENCY = 2 # save weights to a file every {num} epochs
EPOCHS = 1

def compute_test_metrics_fn(model, testloader, loss_fn, device, num_classes = 3, num_eval_batches = None):
    """
    Evaluate metrics on test set, option to specify number of batches
    
    Args:
        model (torch.nn.Module): The segmentation model.
        testloader (torch.utils.data.DataLoader): DataLoader for test data.
        loss_fn (torch.nn.Module): Loss function.
        device: device to use, cpu or gpu if available
        num_classes (int): Number of segmentation classes
        num_eval_batches (int): Number of test set batches to use for computing metrics.
            If not specified, this is run on the entire test set.

    Return: 
        test_metrics (dict)
    """
    model.eval()
    with torch.no_grad():
        test_loss, test_accuracy, test_precision, test_recall, test_iou, test_dice, test_samples = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        
        for j, (X_test, y_test) in enumerate(testloader):
            if j == num_eval_batches:
                break

            X_test, y_test = X_test.to(device), y_test.to(device)

            y_test_hat = model(X_test)
            y_test = y_test.squeeze(dim=1)
            batch_metrics = compute_metrics(y_test_hat, y_test, loss_fn, num_classes)
            batch_size = X_test.size(0)

            test_loss += batch_metrics["loss"] * batch_size
            test_accuracy += batch_metrics["accuracy"] * batch_size
            test_precision += batch_metrics["precision"] * batch_size
            test_recall += batch_metrics["recall"] * batch_size
            test_iou += batch_metrics["iou"] * batch_size
            test_dice += batch_metrics["dice"] * batch_size
            test_samples += batch_size

        test_metrics = {
            "loss": test_loss / test_samples,
            "accuracy": test_accuracy / test_samples,
            "precision": test_precision / test_samples,
            "recall": test_recall / test_samples,
            "iou": test_iou / test_samples,
            "dice": test_dice / test_samples
        }

        return test_metrics

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='effunet', choices=['segnet', 'segnext', 'effunet', 'unet'], help='Segmentation model')
    parser.add_argument('--pseudo', action="store_true", help='Use pseudo masks')
    parser.add_argument('--pseudo_path', type=str, default='cam/saved_models/resnet50_pet_cam_pseudo.pt', help='Path to pseudo masks')
    parser.add_argument('--verbose', action="store_true", help='Print verbose output')
    parser.add_argument('--collapse_contour', action='store_true')
    args = parser.parse_args()
    # check cuda availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print('Using device:', device)

    # Create DataLoaders for batch processing
    # train_loader = DataLoader(trainset, batch_size=16, shuffle=True) # Using small batch-size as running out of memory
    # trainval_loader = DataLoader(testset, batch_size=16, shuffle=True)
    
    if args.collapse_contour:
        def custom_collate_fn(batch):
            # collapse contour class into foreground
            masks = torch.stack([item[1] for item in batch])
            masks[masks == 2] = 0
            images = torch.stack([item[0] for item in batch])
            return images, masks
        
        # train_loader = DataLoader(trainset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn) # Using small batch-size as running out of memory
        # trainval_loader = DataLoader(testset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)
        test_loader = DataLoader(testset, batch_size=64, shuffle=False, collate_fn=custom_collate_fn)
    else:
        # train_loader = DataLoader(trainset, batch_size=16, shuffle=True) # Using small batch-size as running out of memory
        # trainval_loader = DataLoader(testset, batch_size=16, shuffle=True)
        test_loader = DataLoader(testset, batch_size=64, shuffle=False)

    # Check training input shape
    # X_train_batch, y_train_batch = next(iter(train_loader))
    # X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
    # print(X_train_batch.shape, y_train_batch.shape)

    # initialise model
    model = EfficientUNet().to(device)
    if args.model == 'segnext':
        model = SegNeXt(num_classes=3).to(device)
        print('Using SegNeXt model')
    elif args.model == 'segnet':
        model = SegNet().to(device)
        print('Using SegNet model')
    elif args.model == 'unet':
        print('Using UNet model')
        model = UNet(3, 3).to(device)
    else:
        print('Using EfficientUNet model')

    model.eval()

    # test model giving correct shape
    # output = model(X_train_batch)
    # print(output.shape)

    # initialise optimiser & loss class
    # loss_fn = nn.CrossEntropyLoss(reduction='mean')
    # loss_fn = DiceLoss()
    # loss_fn = CombinedCELDiceLoss()
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    model_name = 'EffUNet'
    epoch = 10
    checkpoint_file = f"{model_name}_epoch_{epoch}.pth"

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_file)
    # Restore states
    model.load_state_dict(checkpoint['model'])
    # optimizer.load_state_dict(checkpoint['optimizer'])

    # # Check if lr_scheduler was saved
    # if 'lr_scheduler' in checkpoint:
    #     scheduler.load_state_dict(checkpoint['lr_scheduler'])

    # compute metrics on entire test set
    test_metrics = compute_test_metrics_fn(model, test_loader, loss_fn, device, num_classes = 3, num_eval_batches=None)
    print(f"Final test metrics:   -> {test_metrics}")

    visualise_fs_segmentation(model, testset, device)

if __name__ == "__main__":
    main()
