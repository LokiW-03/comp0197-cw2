import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.datasets import OxfordIIITPet
from PIL import Image

# ---------------------------
# 1. Classification Dataset Wrapper
# ---------------------------
class OxfordPetClsWrapper(Dataset):
    """
    Wraps torchvision.datasets.OxfordIIITPet for classification.
    We request 'category' as the target, ignoring official segmentation.
    This also returns a string 'basename' to link image <-> pseudo-masks.
    """
    def __init__(self, root, split, transform=None):
        super().__init__()
        self.dataset = OxfordIIITPet(
            root=root,
            split=split,
            target_types='category',
            transform=transform,
            download=True
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        path = self.dataset._images[idx]  # single PosixPath object
        basename = path.stem              # or path.name.split('.')[0]
        return image, label, basename



# ---------------------------
# 2. Pseudo-Segmentation Dataset
# ---------------------------

class OxfordPetPseudoSegDataset(Dataset):
    """
    Loads images from torchvision's OxfordIIITPet and 
    the corresponding pseudo-masks from CAM, then upsamples
    them to 224x224 so they match the segmentation network output size.
    """
    def __init__(self, root, split, pseudo_mask_dir, transform=None):
        super().__init__()
        from torchvision.datasets import OxfordIIITPet
        self.dataset = OxfordIIITPet(
            root=root,
            split=split,
            target_types='category',  # we won't use the category label here
            transform=transform,
            download=True
        )
        self.pseudo_mask_dir = pseudo_mask_dir

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 1) Get the image
        image, _ = self.dataset[idx]
        
        # 2) Build the mask filename from the path stem
        path = self.dataset._images[idx]  # A PosixPath
        basename = path.stem
        
        # 3) Load the pseudo-mask (likely 7x7 if from raw CAM)
        mask_path = f"{self.pseudo_mask_dir}/{basename}.pt"
        pseudo_mask = torch.load(mask_path)  # shape [H, W], e.g. [7,7] or [14,14]

        # 4) Upsample it to 224x224
        # Convert [H,W] -> [1,1,H,W] so F.interpolate can process it
        pseudo_mask_4d = pseudo_mask.unsqueeze(0).unsqueeze(0).float()  # [1,1,H,W]
        pseudo_mask_upsampled = F.interpolate(
            pseudo_mask_4d,
            size=(224, 224),
            mode='nearest'
        ).squeeze(0).squeeze(0).long()  # [224,224] integer mask in {0,1,2}

        # Return the image, upsampled mask, plus basename
        return image, pseudo_mask_upsampled, basename


# ---------------------------
# 3. Classification Model & Training
# ---------------------------
def get_classification_model(num_classes=37):
    model = models.resnet50(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_classification(model, train_loader, val_loader, device, epochs=5, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    
    print("Num epochs", epochs)
    for epoch in range(epochs):
        print("Epoch", epoch)
        model.train()
        running_loss = 0.0
        for images, labels, _ in train_loader:
            # images: [B,3,224,224], labels: [B], 
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)  # [B, 37]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)  # [B, 37]
                _, preds = torch.max(outputs, 1)  # [B]
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total
        
        print(f"[CLS] Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    return model


# ---------------------------
# 4. Generate 3-Class Pseudo Masks from CAM
# ---------------------------
def extract_feature_map_and_weights(model):
    feature_blobs = []

    def hook_feature(module, input, output):
        feature_blobs.append(output)
    
    final_layer = model.layer4[-1].conv3
    handle = final_layer.register_forward_hook(hook_feature)
    
    params = list(model.fc.parameters())
    weight_softmax = params[0]  # [num_classes, 2048]
    bias_softmax   = params[1]  # [num_classes]
    
    return feature_blobs, weight_softmax, bias_softmax, handle


def generate_cam(feature_map, class_weights, class_bias, class_idx):
    """
    feature_map: [1, 2048, H, W], class_weights: [num_classes, 2048], class_bias: [num_classes]
    Returns [H, W] in [0..1].
    """
    w = class_weights[class_idx].unsqueeze(0)  # [1, 2048]
    cam = torch.einsum('ic, ichw->hw', w, feature_map)  # => [H, W]
    cam = cam + class_bias[class_idx]
    cam = F.relu(cam)
    if cam.numel() > 0:
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
    return cam


def extract_3class_mask(cam, threshold=0.2):
    """
    cam: [H, W], in [0..1]
    out: [H, W], in {0,1,2}
    """
    fg_mask = (cam > threshold).long()  # => [H, W], {0,1}

    kernel = torch.ones((1,1,3,3), device=fg_mask.device, dtype=torch.float32)
    fg_4d = fg_mask.unsqueeze(0).unsqueeze(0).float()  # => [1,1,H,W]
    eroded = F.conv2d(fg_4d, kernel, padding=1)        # => [1,1,H,W]
    eroded_bool = (eroded == 9.0)
    
    border = (fg_4d.bool() != eroded_bool) & fg_4d.bool()
    
    out_mask = torch.zeros_like(fg_mask)    # [H, W], default 0
    out_mask[eroded_bool.squeeze(0).squeeze(0)] = 2  # interior
    out_mask[border.squeeze(0).squeeze(0)]      = 1  # border
    return out_mask


def create_pseudo_masks_3class(model, loader, device, out_dir, threshold=0.2):
    """
    For each batch in loader, compute CAM & produce pseudo mask => .pt file.
    """
    model.eval()
    feature_blobs, weight_softmax, bias_softmax, hook_handle = extract_feature_map_and_weights(model)
    os.makedirs(out_dir, exist_ok=True)
    
    with torch.no_grad():
        for images, labels, basenames in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            _ = model(images)  # run forward to fill feature_blobs
            if len(feature_blobs) == 0:
                raise ValueError("Feature map hook did not capture output.")
            
            batch_feature_map = feature_blobs[-1]  # [B, 2048, H, W]
            
            w = weight_softmax.to(device)
            b = bias_softmax.to(device)
            
            for i in range(images.size(0)):
                class_idx = labels[i].item()
                fm = batch_feature_map[i:i+1]  # => [1, 2048, H, W]
                
                cam = generate_cam(fm, w, b, class_idx)
                pseudo_mask = extract_3class_mask(cam, threshold=threshold)
                
                out_path = os.path.join(out_dir, basenames[i] + '.pt')
                torch.save(pseudo_mask.cpu(), out_path)
            
            feature_blobs.clear()

    hook_handle.remove()
    print(f"3-class pseudo masks created in {out_dir}")


# ---------------------------
# 5. Simple Segmentation Model (3 channels)
# ---------------------------
class SimpleSegHead(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        backbone = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])  # [B,2048,7,7]
        self.conv1 = nn.Conv2d(2048, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, num_classes, 1)
    
    def forward(self, x):
        x = self.encoder(x)          # => [B,2048,7,7]
        x = F.relu(self.conv1(x))    # => [B,256,7,7]
        x = self.conv2(x)            # => [B,3,7,7]
        x = F.interpolate(x, scale_factor=32, mode='bilinear', align_corners=False)
        return x                     # => [B,3,224,224]


def train_segmentation(model, train_loader, val_loader, device, epochs=5, lr=1e-3):
    criterion = nn.CrossEntropyLoss()  # expects [B,3,H,W] vs. [B,H,W]
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    
    print("Num epochs", epochs)
    for epoch in range(epochs):
        print("Epoch", epoch)
        model.train()
        running_loss = 0.0
        for images, pseudo_masks, _ in train_loader:
            # images => [B,3,224,224], pseudo_masks => [B,224,224]
            images = images.to(device)
            pseudo_masks = pseudo_masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)    # => [B,3,224,224]
            loss = criterion(outputs, pseudo_masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"[SEG] Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")
    
    return model


# ---------------------------
# 6. Evaluate on Test (3 classes)
# ---------------------------
def compute_iou(pred, target, num_classes=3):
    ious = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        if union == 0:
            ious.append(torch.tensor(1.0, device=pred.device) if intersection == 0
                        else torch.tensor(0.0, device=pred.device))
        else:
            ious.append(intersection / union)
    return torch.mean(torch.stack(ious))


def evaluate_segmentation(model, test_loader, device, num_classes=3):
    model.eval()
    iou_scores = []
    total_correct = 0
    total_pixels = 0
    
    with torch.no_grad():
        for images, masks, _ in test_loader:
            images = images.to(device)  # => [B,3,224,224]
            masks = masks.to(device)    # => [B,224,224]
            
            outputs = model(images)     # => [B,3,224,224]
            preds = outputs.argmax(dim=1)  # => [B,224,224]
            
            for b in range(images.size(0)):
                pred = preds[b]
                mask = masks[b]
                
                total_correct += (pred == mask).sum().item()
                total_pixels  += mask.numel()
                
                iou = compute_iou(pred, mask, num_classes=num_classes)
                iou_scores.append(iou.item())
    
    mean_iou = sum(iou_scores)/len(iou_scores) if iou_scores else 0
    pixel_acc = total_correct / total_pixels if total_pixels else 0
    print(f"Test Mean IoU: {mean_iou:.4f}, Pixel Accuracy: {pixel_acc:.4f}")
    return mean_iou, pixel_acc


# ---------------------------
# 7. Main
# ---------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    root_dir = "./oxford_data"  # Where the dataset + pseudo masks will be stored
    
    # Classification transforms
    transform_cls = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Stage 1: Classification
    train_dataset_cls = OxfordPetClsWrapper(root=root_dir, split='trainval', transform=transform_cls)
    test_dataset_cls  = OxfordPetClsWrapper(root=root_dir, split='test', transform=transform_cls)
    
    train_loader_cls = DataLoader(train_dataset_cls, batch_size=16, shuffle=True)
    test_loader_cls  = DataLoader(test_dataset_cls,  batch_size=16, shuffle=False)
    
    print("Train Classification")
    cls_model = get_classification_model(num_classes=37)
    cls_model = train_classification(cls_model, train_loader_cls, test_loader_cls, device, epochs=5, lr=1e-4)
    
    # Generate Pseudo Masks from CAM
    print("Generate Pseudo Masks")
    pseudo_dir = os.path.join(root_dir, "pseudo_masks_3class")
    create_pseudo_masks_3class(cls_model, train_loader_cls, device, pseudo_dir, threshold=0.2)
    create_pseudo_masks_3class(cls_model, test_loader_cls, device, pseudo_dir, threshold=0.2)
    
    # Stage 2: Segmentation
    transform_seg = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    train_dataset_seg = OxfordPetPseudoSegDataset(root=root_dir, split='trainval',
                                                  pseudo_mask_dir=pseudo_dir,
                                                  transform=transform_seg)
    test_dataset_seg  = OxfordPetPseudoSegDataset(root=root_dir, split='test',
                                                  pseudo_mask_dir=pseudo_dir,
                                                  transform=transform_seg)
    
    train_loader_seg = DataLoader(train_dataset_seg, batch_size=8, shuffle=True)
    test_loader_seg  = DataLoader(test_dataset_seg,  batch_size=8, shuffle=False)
    
    print("Train Segmentation")
    seg_model = SimpleSegHead(num_classes=3)
    seg_model = train_segmentation(seg_model, train_loader_seg, test_loader_seg, device, epochs=5, lr=1e-4)
    
    evaluate_segmentation(seg_model, test_loader_seg, device, num_classes=3)


if __name__ == "__main__":
    main()
