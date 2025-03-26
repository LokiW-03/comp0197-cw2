# CRM Component: Reconstruction and Loss Network

Inspired by: [Spatial Structure Constraints for Weakly Supervised Semantic Segmentation](https://arxiv.org/abs/2401.11122)

## Reconstruction Network (`ReconstructNet`)

### Decision:
Use a lightweight CNN-based decoder that reconstructs the original RGB image from CAM feature maps.

### Justification:
- Acts as a regularization mechanism for CAMs
- Encourages CAMs to preserve spatial and semantic information
- Compatible with both CNN and transformer-based CAM backbones

---

## Perception Loss Network (`VGGLoss`)

### Decision:
Use a perceptual loss based on pretrained VGG-19 features.

### Justification:
- Measures similarity in feature space, capturing:
  - Texture
  - Structure
  - Object-level semantics
- Encourages reconstructions to match visual content, not just pixels


