# CRM Component: Reconstruction and Loss Network

Inspired by: [Spatial Structure Constraints for Weakly Supervised Semantic Segmentation](https://arxiv.org/abs/2401.11122)

## Reconstruction Network (`ReconstructNet`)

### Decision:
Use a lightweight CNN-based decoder that reconstructs the original RGB image from CAM feature maps.

### Justification:
- Acts as a regularization mechanism for CAMs
- Encourages CAMs to preserve spatial and semantic information
- Compatible with both CNN (ResNet and EfficientNet) CAM backbones
- Architecture:
  - 1 Conv + LayerNorm Block
  - 1 ResBlock
  - 5 Upsamping Blocks
  - Conv + Tanh 

---

## Loss 

### Decision:
Use a weighted combined loss:
- Classification Loss Cross Entropy  
- L1 Loss for low-level pixel-wise similarity
- VGG Loss based on pretrained VGG-19 for high-level perceptual features
- Alignment Loss based on SLIC superpixels to enforce local consistency
  - Note: pip install required:
  ``` bash
  pip install scikit-image
  ```

  - Or simply download the [superpixels](https://1drv.ms/u/c/2ef0e412637ecc3c/EQy9SXX7x4tGnqJWRpIJa7EBYK9I7c2ipQB07oCzcjAfKQ?e=ksvFWp) from OneDrive and unzip in directory `comp0197-cw2/`

---

## Discriminative Region Suppression (DRS)

### Decision:
- Suppresses these highly activated regions by capping their values which forces the network to look at less discriminative parts.

### Justification:
- Expand CAM concentration

**Note**: See implementation in `cam/resnet_drs.py`

---

## Training

- CAM backbone: 
  - ResNet50 + GradCAM++ 
  - ResNet50 + DRS + GradCAM++

- Freeze all classifier layers except last two
- Train backbone and Reconstruction Network together and use Reconstruction loss as regularization term
- Optimizers:
  - Classifier: Adam (LR=1e-3)
  - Reconstruction Network: Adam (LR=2e-3)

### Trained Models
- Path: `comp0197-cw2/crm_models/...`

- [ResNet50](https://1drv.ms/u/c/2ef0e412637ecc3c/EawGxav3g3BPke8uXA7C5W0Bdf2oIHQSoV6smZgRWXR1NA?e=zlKiYk) + [CRM](https://1drv.ms/u/c/2ef0e412637ecc3c/EdhrCbIkW6dEpXfImbAcRsoBBb_3ceJHz16NxfTiqLPmhg?e=DWot9e) 
  - Classifier test accuracy: 84%
    

- [ResNet50 + DRS + GradCAM++](https://1drv.ms/u/c/2ef0e412637ecc3c/EQU-6ec3hklKhi9hTXwXxDEBWx5czmOywqLiH3gsT0qhAQ?e=SBTBau) + [CRM](https://1drv.ms/u/c/2ef0e412637ecc3c/EesRuHMqxgZAvj6Qc710poYBfyskimMUQtJAFrfC9wmOCw?e=h5RG8g) 
  - Classifier test accuracy: 91%

- See sample CAM heatmap and reconstruction image in `crm/img`

---

**Note**: 
- `train_cam_with_crm.py` will generate superpixel (if not downloaded), store them and then load them for calculating alignment loss
- `evaluate_with_crm.py` will evaluate the model on the test set, and generate 5 sample reconstructed images

- Pipeline:

  ```bash
  python -m crm.train_cam_with_crm --model=resnet_drs
  python -m crm.evaluate_with_crm --model=resnet_drs
  python -m cam.postprocessing --model=resnet_drs
  ```

  - model can be `[resnet_drs, resnet]`

---

### Limitation 
- CRM require high computational cost for reconstruction network, a superpixel alignment loss and vgg loss
  - Even with a lightweight superpixel processing compared to the original paper
  - Note: Our approach required an additional layer of upsampling due to the dimension of the CAM, hence increases the model's complexity

- Little improvement on CAM compare to pure Resnet50, and poor reconstructed images due to limited resources
  - (see CAM in `crm/img`)
  
- Unstable training dynamics
  - The reconstruction loss remains high in early epochs because the classifier's CAM features are not yet meaningful. But further training could risk classifier to overfit
  - (see `graph/resnet_drs_crm_loss_curve.png`)

- Due to the above constraints, we did not perform full grid search for resnet drs + crm.