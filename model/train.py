# Import files 
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from baseline_segnet import SegNet
from metrics import compute_metrics
from efficient_unet import EfficientUNet
from baseline_unet import UNet
from metrics import compute_metrics
from enum import IntEnum
from PIL import Image

SAVE_WEIGHTS_FREQUENCY = 1 # save weights to a file every {num} epochs

# Transform input
img2t = transforms.ToTensor()

def tensor_trimap(t):
    x = t * 255
    x = x.to(torch.long)
    x = x - 1
    return x

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mask_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Lambda(tensor_trimap)
])

class TrimapClasses(IntEnum):
    PET = 0
    BACKGROUND = 1
    BORDER = 2

def trimap2f(trimap):
    return (img2t(trimap) * 255.0 - 1) / 2

def compute_test_metrics_fn(model, testloader, loss_fn, num_classes = 3, num_eval_batches = None):
    """
    Evaluate metrics on test set, option to specify number of batches
    
    Args:
        model (torch.nn.Module): The segmentation model.
        testloader (torch.utils.data.DataLoader): DataLoader for test data.
        loss_fn (torch.nn.Module): Loss function.
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



# Evaluating training and testing metrics simultaneously to save time 
def train_model(model, trainloader, trainvalloader, loss_fn, optimizer, epochs, num_classes=3, compute_test_metrics=False, model_name: str='example'):
    """
    Trains a segmentation model and computes metrics: loss, accuracy, precision, recall, IoU, and Dice coefficient at each epoch.

    Args:
        model (torch.nn.Module): The segmentation model.
        trainloader (torch.utils.data.DataLoader): DataLoader for training data.
        trainvalloader (torch.utils.data.DataLoader): DataLoader for eval data.
        loss_fn (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        epochs (int): Number of training epochs.
        num_classes (int): Number of segmentation classes
        compute_test_metrics (bool): option to compute test metrics while training
    """
    print("Number of train batches:", len(trainloader))

    for epoch in range(1, epochs + 1):
        print("Start epoch", epoch)
        model.train()
        running_loss, total_accuracy, total_precision, total_recall, total_iou, total_dice, num_samples = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        for id, (X_train, y_train) in enumerate(trainloader, 0):
            print(f'Training batch {id}')

            optimizer.zero_grad()
            y_hat = model(X_train)

            y_train = y_train.squeeze(dim=1)
            loss = loss_fn(y_hat, y_train)

            loss.backward()
            optimizer.step()

            # Compute batch metrics
            batch_metrics = compute_metrics(y_hat, y_train, loss_fn, num_classes)
            batch_size = X_train.size(0)

            # Accumulate metrics
            running_loss += batch_metrics["loss"] * batch_size
            total_accuracy += batch_metrics["accuracy"] * batch_size
            total_precision += batch_metrics["precision"] * batch_size
            total_recall += batch_metrics["recall"] * batch_size
            total_iou += batch_metrics["iou"] * batch_size
            total_dice += batch_metrics["dice"] * batch_size
            num_samples += batch_size
            

        # Compute average metrics for training
        train_metrics = {
            "loss": running_loss / num_samples,
            "accuracy": total_accuracy / num_samples,
            "precision": total_precision / num_samples,
            "recall": total_recall / num_samples,
            "iou": total_iou / num_samples,
            "dice": total_dice / num_samples
        }

        # Print metrics for the current epoch
        print(f"Epoch {epoch}/{epochs}")
        print(f"Train -> {train_metrics}")
        if compute_test_metrics:
            test_metrics = compute_test_metrics_fn(
                model, trainvalloader, loss_fn, num_classes=3, num_eval_batches=1
            )
            print(f"Test   -> {test_metrics}")
        print("-" * 50)

        # Save model weights every 5 epochs
        if epoch % SAVE_WEIGHTS_FREQUENCY == 0:
            torch.save(model.state_dict(), f"{model_name}_epoch_{epoch}.pth")
            print(f"Model weights saved for model {model_name} at epoch {epoch}")


def main():
    # check cuda availability
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print('Using device:', device)


    # prepare data
    trainset = datasets.OxfordIIITPet(root='./data', split='trainval', target_types= 'segmentation',download=True, transform=transform, target_transform=mask_transform)
    testset = datasets.OxfordIIITPet(root='./data', split='test', target_types= 'segmentation', download=True, transform=transform, target_transform=mask_transform)

    # Create DataLoaders for batch processing
    train_loader = DataLoader(trainset, batch_size=16, shuffle=True) # Using small batch-size as running out of memory
    trainval_loader = DataLoader(testset, batch_size=16, shuffle=True)
    test_loader = DataLoader(testset, batch_size=64, shuffle=True)

    X_train_batch, _ = next(iter(train_loader))
    print(X_train_batch.shape)

    # initialise model
    #model = SegNet()
    #model = EfficientUNet()
    model = UNet(3, 3)

    # test model giving correct shape
    model.eval()
    output = model(X_train_batch)
    print(output.shape)

    # # initialise optimiser & loss class
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # train model
    train_model(model, train_loader, trainval_loader, loss_fn, optimizer, 2, compute_test_metrics = True)

    # compute metrics on entire test set (may take a while)
    test_metrics = compute_test_metrics_fn(
        model, test_loader, loss_fn, num_classes = 3,
        num_eval_batches=None) 
    print(f"Final test metrics:   -> {test_metrics}")


if __name__ == "__main__":
    main()