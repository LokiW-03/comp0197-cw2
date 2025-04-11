# Experiments

Remember our weakly-supervised architecture:

classifer -> CAM -> pseudo-masks -> segmentation

We conducted experiments on 24 different hyperparameters configurations and compared their performances.

The experiments were a grid search of 4 dimensions:

- pseudo-mask thresholds: (0.25, 0.325), (0.3, 0.7), (0.21, 0.33)
- segmentation model:
  - Learning rate: 1e-3, 1e-2
  - Loss function: Cross entropy loss, Dice loss
  - Batch size: 16, 64

In total: 3 x 2 x 2 x 2 = 24

# Reason we picked these hyperparameters

- pseudo-mask thresholds: we ran an initial grid search using segnet to find well-performing thresholds, then used these across all models to check performance
- Learning rate: 1e-3 is typical, 1e-2 learning faster to see if it helps
- Loss function: CE is typical, dice loss is another commonly used
- Batch size: 16 and 64 are common, 16 is also the same as fully-supervised

# Hyperparameters that were kept consistent (Model architecture)

**Classifier model**

Throughout the experiments, we maintained the same classifier model and its hyperparameters. This is because the model was tuned to have >90% accuracy and was deemed good enough.

@chenge
In particular:
- We used ResNet50 model (why?)
- Epochs: ...
- Loss fn: ...
- Learning rate: ...
- Batch size: ...
- Optimizer: ...

**CAM**

@chenge
- How is the heatmap generated? (if the process is too complicated, just link the paper)

**Pseudo mask generation**

@chenge
- The generation process involves 2 thresholds that determine the split between 3 classes: foreground, contour, background. These thresholds are varied but the rest of the process is the same.
- Any other details I'm missing?

**Segmentation model**

Aside from LR, loss fn, batch size; other segmentation model hyperparameters were kept consistent:
- AdamW optimizer
  - 1e-4 weight_decay
  - LR varies as above
- StepLR scheduler
  - step_size=15
  - gamma=0.1

# Weakly-supervised baseline: Comparison to fully-supervised experiments

The most relatable configuration to the fully-supervised experiments are

- pseudo-mask thresholds: (0.3, 0.7) - common setting in previous work
- Learning rate: 1e-3
- Loss function: Cross entropy loss
- Batch size: 16

This is because the last 3 hyperparameters are exactly those used for training the fully-supervised models, which we consider our baseline.

# Execution commands

To run grid search

```
# make sure classification model is under cam/saved_models
python -m ablation.grid_search
```
