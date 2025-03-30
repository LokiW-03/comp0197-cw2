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

## Training

- CAM backbone: ResNet50 + GradCAM++ or EfficientNet + ScoreCAM
- Freeze all classifier layers except last two
- Train backbone and Reconstruction Network together and use Reconstruction loss as regularization term
- Optimizers:
  - Classifier: Adam (LR=1e-3)
  - Reconstruction Network: Adam (LR=1e-2)

- [ResNet50](https://1drv.ms/u/c/2ef0e412637ecc3c/EawGxav3g3BPke8uXA7C5W0Bdf2oIHQSoV6smZgRWXR1NA?e=zlKiYk) + [CRM](https://1drv.ms/u/c/2ef0e412637ecc3c/EdhrCbIkW6dEpXfImbAcRsoBBb_3ceJHz16NxfTiqLPmhg?e=DWot9e) can be downloaded on OneDrive (classifier test accuracy: 84%)
    - Path: `comp0197-cw2/crm_models/...`

---

**Note**: 
- `train_cam_with_crm.py` will generate superpixel (if not downloaded), store them and then load them for calculating alignment loss
- `evaluate_with_crm.py` will evaluate the model on the test set, and generate 5 sample reconstructed images
- Pipeline:

  ```bash
  python -m crm.train_cam_with_crm
  python -m crm.evaluate_with_crm
  ```

To generate the CAM and pseudomask, please refer to `cam.postprocessing`
