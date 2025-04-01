# finetune.py

import torch
import torch.nn as nn
import torch.optim as optim

from efficientnet_scorecam import EfficientNetB4_CAM
from dataset.oxfordpet import download_pet_dataset
from cam import *

def fine_tune_model():
    # Check device
    device = torch.device(
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )
    
    # Load data
    train_loader, test_loader = download_pet_dataset()
    
    # Initialize EfficientNet-B4 with Score-CAM
    model = EfficientNetB4_CAM(NUM_CLASSES)
    model = model.to(device)
    
    # Freeze all feature blocks except the last one
    # Assuming model.effnet.features is a nn.Sequential, freeze all blocks except the last
    for i, block in enumerate(model.effnet.features):
        if i < len(model.effnet.features) - 1:
            for param in block.parameters():
                param.requires_grad = False
    # The classifier remains trainable (and the last block in features)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    
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
        
        # Save best model checkpoint
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), f"{MODEL_SAVE_PATH}/efficientnet_pet_scorecam.pth")
            print(f"Model saved at {MODEL_SAVE_PATH}/efficientnet_pet_scorecam.pth with acc {best_acc:.4f}")

if __name__ == "__main__":
    fine_tune_model()
