## Implemented CNN+CAM

1. **ResNet50 with GradCAM++:**  
    - Pretrained ResNet50 with GradCAM++. See ```resnetcam.py```. Load finetuned model from [Google Drive](https://drive.google.com/file/d/1wT_NrUo6PivRcT4vohPDvoGiLiWkPhD_/view?usp=drive_link). (File cam/saved_models/resnet50_pet_cam.pth is 90.3 MB; this is larger than GitHub's recommended maximum file size of 50.00 MB)


2. **EfficientNet with Score-CAM:**  
   - A pretrained EfficientNet model is also available. Follow the same procedure as above to load the fine-tuned model.  
   - Download it from [OneDrive](https://liveuclac-my.sharepoint.com/:u:/g/personal/ucabz68_ucl_ac_uk/EVMx450VpAdMkzGzf_tiJAQBTjf0wQ4KQ_qKk3L69fcrfw?e=i0k0FC).  

# Pseudo mask generation with ResNet50 and Grad-CAM++

## Overview
This folder implements a weakly-supervised segmentation pipeline for the Oxford-IIIT Pet dataset. We leverage ResNet50 as a classifier to generate pseudo segmentation masks using Grad-CAM++ activation maps, eliminating the need for pixel-level annotations during training.

## Methodology

### 1. ResNet50 Fine-tuning
- Pretrained ResNet50 is fine-tuned on the Oxford-IIIT Pet dataset (37 categories of cats and dogs)
- Trained as a classification task using image-level labels only
- Achieves >92% validation accuracy after fine-tuning

### 2. Grad-CAM++ Activation Maps
- **Target Layer**: `layer4.conv3` (final convolutional layer before pooling)
  - Captures high-level semantic features while preserving spatial information
- **Grad-CAM++**:
  - Improved version of Grad-CAM for better localization
  - Computes weighted combination of feature maps using higher-order gradients
  - Produces sharper activation maps compared to basic Grad-CAM

### 3. Pseudo Mask Generation
1. Generate class activation maps (CAMs) for target classes
2. Apply thresholds on cam to generate pseudo masks

