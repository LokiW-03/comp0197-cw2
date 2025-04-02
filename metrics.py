"""Metrics calculations for baseline model

Metrics Explanation
Choose abundance of metrics so we can pick and choose later on depending on preference

1. Pixel Accuracy
Measures how many pixels are correctly classified.
Pro is that it is easy to interpret and compute
However, it doesn't account for imbalances in classes

2. Precision
Number of true positives divided by all positives
Precision = TP/(TP+FP) 

3. Recall (sensitivity)
Number of true positions divided by all values that should have been positives
Recall = TP/ (TP + FN)

4. Mean Intersection over Union (Jaccard Index)
Super common in computer vision
Scores overlap between predicted segmentation and ground truth 
Penalizes under- and over-segmentation more than dice coefficient
IoU = TP/(TP + FP + FN)
Area of overlap between prediction and ground truth divided by area of union


5. Dice Coefficient (F1-score)
Super common in computer vision 
Scores overlap between predicted segmentation and ground truth 
Harmonic mean of precision and recall: 
Dice = (2TP)/(2TP + FP + FN) 
(2 x intersection of prediction and ground truth divided by the total area)
"""

import torch 

# Accuracy 
def accuracy_fn(preds, y):
    """
    Computes accuracy based on the predicted labels and true target labels.

    Args:
        preds (torch.Tensor): The predicted values.
        y (torch.Tensor): The true target labels (same shape as preds).

    Returns:
        torch.Tensor: The computed accuracy as a fraction of correct predictions.
    """
    # Convert predicted probabilities to class labels
    pred_labs = preds.argmax(dim=1)
    amt_correct = (pred_labs == y).to(torch.float32)
    return torch.mean(amt_correct)


def precision_fn(preds, y, num_classes=3, epsilon=1e-6):
    """
    Calculates average precision for a multiclass classification problem.

    Args:
        preds (torch.Tensor): Model predictions.
        y (torch.Tensor): True labels (batch_size).
        num_classes (int): Number of classes.

    Returns:
        torch.Tensor: Average precision across all classes.
    """
    # Convert predicted probabilities to class labels
    pred_labs = preds.argmax(dim=1)
    precision_list = []

    for ii in range(num_classes):
        pred_cls = (pred_labs == ii)
        true_cls = (y == ii)

        intersection = (pred_cls * true_cls).sum().to(torch.float32)
        precision = (intersection + epsilon)/ (
            torch.sum(pred_cls).to(torch.float32)+ epsilon)
        precision_list.append(precision)

    # Return average precision across all classes
    return torch.mean(torch.stack(precision_list))

def recall_fn(preds, y, num_classes=3, epsilon=1e-6):
    """
    Calculates average recall for a multiclass classification problem.

    Args:
        preds (torch.Tensor): Model predictions.
        y (torch.Tensor): True labels (batch_size).
        num_classes (int): Number of classes.

    Returns:
        torch.Tensor: Average precision across all classes.
    """
    # Convert predicted probabilities to class labels
    pred_labs = preds.argmax(dim=1)
    recall_list = []

    for ii in range(num_classes):
        pred_cls = (pred_labs == ii)
        true_cls = (y == ii)

        intersection = (pred_cls * true_cls).sum().to(torch.float32)

        recall =(torch.sum(intersection)+ epsilon)/ (
            torch.sum(true_cls).to(torch.float32)+ epsilon)
        recall_list.append(recall)


    # Return average precision across all classes
    return torch.mean(torch.stack(recall_list))


# Mean Intersection over Union (IoU) (Jaccard Index)
def iou_fn(preds, y, num_classes=3):
    """
    Computes the mean IoU between predicted values and ground truth.

    Args:
        preds (torch.Tensor): Predicted logits or probabilities (B, C, H, W).
        y (torch.Tensor): Ground truth labels (B, H, W).
        num_classes (int): Number of classes.

    Returns:
        torch.Tensor: Mean IoU across all classes.
    """
    # Convert predicted probabilities to class labels
    pred_labs = preds.argmax(dim=1)
    ious = []

    for ii in range(num_classes):
        pred_cls = (pred_labs == ii)
        true_cls = (y == ii)

        intersection = torch.logical_and(pred_cls, true_cls).sum()
        union = torch.logical_or(pred_cls, true_cls).sum()

        if union == 0:  
            ious.append(torch.tensor(1.0)) 
        else:
            ious.append(intersection / union)

    return torch.mean(torch.stack(ious))

# Dice Coefficient
def dice_coefficient_fn(preds, y, num_classes=3, epsilon=1e-6):
    """
    Computes the mean Dice Coefficient between predicted and true masks.

    Args:
        preds (torch.Tensor): Predicted logits or probabilities (B, C, H, W).
        y (torch.Tensor): Ground truth labels (B, H, W).
        num_classes (int): Number of classes.
        epsilon (float): Small value to avoid division by zero.

    Returns:
        torch.Tensor: Mean Dice coefficient across all classes.
    """
    # Convert predicted probabilities to class labels
    pred_labs = preds.argmax(dim=1)
    dice_scores = []

    for ii in range(num_classes):
        pred_cls = (pred_labs == ii)
        true_cls = (y == ii)

        intersection = (pred_cls * true_cls).sum().to(torch.float32)
        denominator = pred_cls.sum() + true_cls.sum().to(torch.float32)

        dice = (2. * intersection + epsilon) / (denominator + epsilon)
        dice_scores.append(dice)

    return torch.mean(torch.stack(dice_scores))


def compute_metrics(preds, targets, loss_fn, num_classes=3):
    """
    Computes segmentation metrics: Loss, Accuracy, IoU, and Dice Coefficient.

    Args:
        preds (torch.Tensor): Model predictions (logits).
        targets (torch.Tensor): Ground truth masks.
        loss_fn (torch.nn.Module): Loss function.
        num_classes (int): Number of segmentation classes.

    Returns:
        dict: A dictionary containing computed metrics.
    """
    loss = loss_fn(preds, targets).item()
    accuracy = accuracy_fn(preds, targets).item()
    precision = precision_fn(preds, targets).item()
    recall = recall_fn(preds, targets).item()
    iou = iou_fn(preds, targets, num_classes).item()
    dice = dice_coefficient_fn(preds, targets, num_classes).item()

    return {"loss": loss, "accuracy": accuracy, "precision": precision, "recall": recall, "iou": iou, "dice": dice}
