"""Metrics calculations 

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
Very Common in computer vision
Scores overlap between predicted segmentation and ground truth 
Penalizes under- and over-segmentation more than dice coefficient
IoU = TP/(TP + FP + FN)
Area of overlap between prediction and ground truth divided by area of union


5. Dice Coefficient (F1-score)
Very Common in computer vision 
Scores overlap between predicted segmentation and ground truth 
Harmonic mean of precision and recall: 
Dice = (2TP)/(2TP + FP + FN) 
(2 x intersection of prediction and ground truth divided by the total area)
"""



import torch

# Accuracy
def accuracy_fn(preds, y, mask=None):
    """
    Computes accuracy

    Args:
        preds (torch.Tensor): Predicted logits (B, C, H, W).
        y (torch.Tensor): Ground truth labels (B, H, W) or soft labels.
        mask (torch.Tensor, optional): Mask indicating valid pixels for evaluation.

    Returns:
        torch.Tensor: Computed accuracy.
    """
    pred_labs = preds.argmax(dim=1)
    
    if mask is not None:
        correct = ((pred_labs == y) * mask).to(torch.float32)
        return correct.sum() / mask.sum()
    
    correct = (pred_labs == y).to(torch.float32)
    return torch.mean(correct)

# Precision
def precision_fn(preds, y, num_classes=3, mask=None, epsilon=1e-6):
    """
    Computes precision 

    Args:
        preds (torch.Tensor): Model predictions.
        y (torch.Tensor): True labels or soft labels.
        num_classes (int): Number of classes.
        mask (torch.Tensor, optional): Mask indicating valid pixels.

    Returns:
        torch.Tensor: Average precision.
    """
    pred_labs = preds.argmax(dim=1)
    precision_list = []

    for ii in range(num_classes):
        pred_cls = (pred_labs == ii)
        true_cls = (y == ii) if y.dtype == torch.long else y[:, ii]

        if mask is not None:
            pred_cls = pred_cls * mask
            true_cls = true_cls * mask

        intersection = (pred_cls * true_cls).sum().float()
        precision = (intersection + epsilon) / (pred_cls.sum().float() + epsilon)
        precision_list.append(precision)

    return torch.mean(torch.stack(precision_list))

# Recall
def recall_fn(preds, y, num_classes=3, mask=None, epsilon=1e-6):
    """
    Computes recall 

    Args:
        preds (torch.Tensor): Model predictions.
        y (torch.Tensor): True labels or soft labels.
        num_classes (int): Number of classes.
        mask (torch.Tensor, optional): Mask indicating valid pixels.

    Returns:
        torch.Tensor: Average recall.
    """
    pred_labs = preds.argmax(dim=1)
    recall_list = []

    for ii in range(num_classes):
        pred_cls = (pred_labs == ii)
        true_cls = (y == ii) if y.dtype == torch.long else y[:, ii]

        if mask is not None:
            pred_cls = pred_cls * mask
            true_cls = true_cls * mask

        intersection = (pred_cls * true_cls).sum().float()
        recall = (intersection + epsilon) / (true_cls.sum().float() + epsilon)
        recall_list.append(recall)

    return torch.mean(torch.stack(recall_list))

# Mean IoU (Jaccard Index)
def iou_fn(preds, y, num_classes=3, mask=None):
    """
    Computes mean IoU 

    Args:
        preds (torch.Tensor): Model predictions.
        y (torch.Tensor): Ground truth labels or soft labels.
        num_classes (int): Number of classes.
        mask (torch.Tensor, optional): Mask indicating valid pixels.

    Returns:
        torch.Tensor: Mean IoU.
    """
    pred_labs = preds.argmax(dim=1)
    ious = []

    for ii in range(num_classes):
        pred_cls = (pred_labs == ii)
        true_cls = (y == ii) if y.dtype == torch.long else y[:, ii]

        if mask is not None:
            pred_cls = pred_cls * mask
            true_cls = true_cls * mask

        intersection = torch.logical_and(pred_cls, true_cls).sum().float()
        union = torch.logical_or(pred_cls, true_cls).sum().float()

        if union == 0:
            ious.append(torch.tensor(1.0))
        else:
            ious.append(intersection / union)

    return torch.mean(torch.stack(ious))

# Dice Coefficient
def dice_coefficient_fn(preds, y, num_classes=3, mask=None, epsilon=1e-6):
    """
    Computes Dice coefficient

    Args:
        preds (torch.Tensor): Model predictions.
        y (torch.Tensor): Ground truth labels or soft labels.
        num_classes (int): Number of classes.
        mask (torch.Tensor, optional): Mask indicating valid pixels.
        epsilon (float): Small value to avoid division by zero.

    Returns:
        torch.Tensor: Mean Dice coefficient.
    """
    pred_labs = preds.argmax(dim=1)
    dice_scores = []

    for ii in range(num_classes):
        pred_cls = (pred_labs == ii)
        true_cls = (y == ii) if y.dtype == torch.long else y[:, ii]

        if mask is not None:
            pred_cls = pred_cls * mask
            true_cls = true_cls * mask

        intersection = (pred_cls * true_cls).sum().float()
        denominator = pred_cls.sum().float() + true_cls.sum().float()

        dice = (2. * intersection + epsilon) / (denominator + epsilon)
        dice_scores.append(dice)

    return torch.mean(torch.stack(dice_scores))

# Compute all metrics
def compute_metrics(preds, targets, loss_fn, num_classes=3, mask=None):
    """
    Computes segmentation metrics 

    Args:
        preds (torch.Tensor): Model predictions.
        targets (torch.Tensor): Ground truth masks or weak labels.
        loss_fn (torch.nn.Module): Loss function.
        num_classes (int): Number of segmentation classes.
        mask (torch.Tensor, optional): Mask for valid pixels.

    Returns:
        dict: Dictionary containing computed metrics.
    """
    loss = loss_fn(preds, targets).item()
    accuracy = accuracy_fn(preds, targets, mask).item()
    precision = precision_fn(preds, targets, num_classes, mask).item()
    recall = recall_fn(preds, targets, num_classes, mask).item()
    iou = iou_fn(preds, targets, num_classes, mask).item()
    dice = dice_coefficient_fn(preds, targets, num_classes, mask).item()

    return {
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "dice": dice
    }
