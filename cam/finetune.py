#finetune.py

import torch
import torch.nn as nn
import torch.optim as optim

from resnet_gradcampp import ResNet50_CAM
from dataset.oxfordpet import download_pet_dataset
from common import *


# -------------------- Fine-tuning Function --------------------
def fine_tune_model():
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load data
    train_loader, test_loader = download_pet_dataset()
    
    # Initialize model (fix pretrained parameter spelling)
    model = ResNet50_CAM(NUM_CLASSES)
    model = model.to(device)
    
    # Freeze bottom layer parameters, only train last two layers
    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    
    # Training loop
    best_acc = 0.0
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        # Training phase
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"Train Loss: {epoch_loss:.4f} | Test Acc: {epoch_acc:.4f}")
        
        # Save best model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), f"{MODEL_SAVE_PATH}/resnet50_pet_cam.pth")
            print(f"Model saved at {MODEL_SAVE_PATH}/resnet50_pet_cam.pth with acc {best_acc:.4f}")


# -------------------- Execute Training --------------------
if __name__ == "__main__":
    import os
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)
    fine_tune_model()