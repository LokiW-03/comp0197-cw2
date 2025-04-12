# MPR Formulation 

------


### Type of  Weak Supervision: Image-Level Label
- **Justification**: Minimum annotation effort compared to bounding box - meaning we can add additional datasets easily, when necessary.
- **Scalability**: More scalable to large datasets.

### Problem Definition
- **Objective**: Learning a pixel-wise classification (segmentation) with only image-level supervision.
- **Key Challenge**: Localizing object without direct pixel-level supervision.

#### Approach
1. **Transfer Learning Backbone**
   - Modify an image-classifier / MCTformer such that a pretrained model generates CAM, and image-level label/class token.
   - Train the backbone using dataset image-level label.

2. **CAMs to Pseudo Label**
   - Extract CAMs before entering the GAP layer.
   - Normalize CAMs.
   - Apply post-processing to refine CAMs.

3. **Train Segmentation Network using Pseudo Label**

### Problem Feasibility
- **Can our WSSS model performs better than Random Guessing?**
- **Within time and hardware constraints?**
  - Not sure how to justify this.

### Performance Benchmark
- **Metrics**: IoU / Dice Score
- **Comparison**: Compare to fully-supervised model (trained using pixel-level annotation).

## Suggested Models / Strategy (with References to Papers)

### Label Generation Models
#### **CNN-Based Classifier**
- **Summary**: Modify last fc layer of an image classifier into 1x1 conv layer + GAP layer to extract CAM
- **Advantages**:
  - Less computationally expensive.
  - Simpler to implement.
  - More flexible (more literature on different strategies/adjustments).
  - Can use pretrained model as backbone (e.g. VGG, ResNet, EfficientNet).

- **Disadvantages**:
  - CAM could be less precise.
  - Transfer learning requires modifying backbone layers.

#### **MCTFormer**:  [Multi-class Token Transformer for Weakly Supervised Semantic Segmentation](https://arxiv.org/abs/2203.02891)
- **Summary**: Transformer generates CAM through self-attention map (feature extraction) + **class-specific** attention map (object localization) instead of traditional gradient based CAM

- **Advantages**:
  - Potentially better quality CAMs with more precise localization.
  - Handles complex scenes better (global attention mechanisms).
  - No modification of backbone required.
  - Aligns with breed labels.
  - Can use pretrained models as backbone (e.g., ViT).
  - Can be installed via `pip`, so no need to implement from scratch.
  **Note**: Need to confirm with lecturer to see if this is okay.

- **Disadvantages**:
  - Could require more computational resources.
  - If implementing, needs handling of multi-class tokens (can refer to [GitHub](https://github.com/xulianuwa/MCTformer)).
  - Relies heavily on the backbone.


### Loss Functions
- **Backbone**: BCE / CE
- **Segmentation**: IoU / Dice

### Segmentation Models
- **Options**: UNet / DeepLab

### Other Strategies (Regularization / Optimization)
#### CRM & ASM refinement for CAM: [Spatial Structure Constraints for Weakly Supervised Semantic Segmentation](https://arxiv.org/abs/2401.11122)
- **Summary**: Reconstruction image from CAM and use the loss as a regularization term for CAM, then use ASM to smooth and refine CAM activation at superpixels level

- **Advantages**:
  - Less computational overhead compare to MCTFormer
  - Can be integrated with both CNN-based or Transformer-based model

- **Disadvantages:**
  - Need to train a reconstruction network
  - Need external superpixels segmentation algorithm


## Ablation Study

### Methodology
- **Grid Search? Random Search?**

### Hyperparameters
- **CAM threshold**
- **Backbone choice**
- Other parameters...

------

# OEQ: Granularity of Image-Level Labels (Binary vs Breed)

---

- Compare models trained using different levels of granularity in image-level labels.
- **Does finer granularity improve?**
  - **Localization**: "Did the model highlight the correct object in the right place?"
    - **Metric**: IoU / CAM / Pixel-wise precision.
  - **Generalization**: "Can the model handle new images and different pet breeds?"
    - **Metric**: Test set IoU / performance on unseen breeds.
  - **Consistency**: "Is the model stable across variations of the same object?"
    - **Metric**: Confidence score variation / IoU stability under augmentations.

### Research Question
> *How does the level of granularity in weak supervision (binary labels vs. breed labels) affect weakly-supervised segmentation quality? Does finer granularity improve feature localization, generalization, or classification consistency?*
