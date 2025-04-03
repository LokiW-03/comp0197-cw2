
# Open-Ended




## Installation
Additional installation
```bash
pip install Pillow numpy scikit-image segmentation-models-pytorch torchmetrics
```

## Expected File Structure
```
open_ended/
├── data/                     
│   ├── images/
│   └── annotations/
│       └── trimaps/
├── weak_labels/              
│   └── weak_labels_train.pkl
├── data_utils.py             
├── weak_label_generator.py   
├── model.py                  
├── losses.py                 
├── train.py                  
├── evaluate.py               
└── README.md                 
```
## Run


```bash
cd open_ended/
```

Generate Weak Labels (4)
```bash
python weak_label_generator.py --data_dir ./data --output_file ./weak_labels/weak_labels_train.pkl
```

Train models
```bash

# Points, trained
python train.py \
  --supervision_mode points \
  --run_name points_run1 \
  --epochs 10 \
  --batch_size 8 \
  --lr 1e-4



# Tags, error with
python train.py \
  --supervision_mode tags \
  --run_name tags_run1 \
  --epochs 50 \
  --batch_size 8 \
  --lr 1e-4

# Scribble
python train.py \
  --supervision_mode scribbles \
  --run_name scribbles_run1 \
  --epochs 50 \
  --batch_size 8 \
  --lr 1e-4

#Boxes
python train.py \
  --supervision_mode boxes \
  --run_name boxes_run1 \
  --epochs 50 \
  --batch_size 8 \
  --lr 1e-4

# Hybrid (Tags + Points)
python train.py --supervision_mode hybrid_tags_points --run_name hybrid_run1 --epochs 50 --batch_size 8 --lr 1e-4 --lambda_seg 1.0


#colab
!python train.py \
    --supervision_mode points \
    --run_name points_run1 \
    --weak_label_path ./weak_labels/weak_labels_train.pkl \
    --data_dir ./data \
    --batch_size 64 \
    --lr 2e-4 \
    --epochs 75 \
    --num_workers 8 \
    --img_size 256 \
    --checkpoint_dir ./checkpoints_a100

# Points + Box
!python train.py \
    --supervision_modes points boxes \
    --lambda_points 1.0 \
    --lambda_boxes 1.0 \
    --run_name points_boxes_run1 \
    --data_dir ./data \
    --weak_label_path ./weak_labels/weak_labels_train.pkl \
    --batch_size 32 \
    --lr 1e-4 \
    --epochs 50 \
    --num_workers 8 \
    --img_size 256 \
    --checkpoint_dir ./checkpoints_hybrid \
    --augment

# Scribble + Box
!python train.py \
    --supervision_modes scribbles boxes \
    --lambda_scribbles 1.0 \
    --lambda_boxes 1.0 \
    --run_name scribbles_boxes_run1 \
    --data_dir ./data \
    --weak_label_path ./weak_labels/weak_labels_train.pkl \
    --batch_size 32 \
    --lr 1e-4 \
    --epochs 50 \
    --num_workers 8 \
    --img_size 256 \
    --checkpoint_dir ./checkpoints_hybrid \
    --augment

#Points + Scribble
!python train.py \
    --supervision_modes points scribbles \
    --lambda_points 1.0 \
    --lambda_scribbles 1.0 \
    --run_name points_scribbles_run1 \
    --data_dir ./data \
    --weak_label_path ./weak_labels/weak_labels_train.pkl \
    --batch_size 32 \
    --lr 1e-4 \
    --epochs 50 \
    --num_workers 8 \
    --img_size 256 \
    --checkpoint_dir ./checkpoints_hybrid \
    --augment

```


## Colab

```
!pip install Pillow numpy scikit-image segmentation-models-pytorch torchmetrics
!python download_data.py
!python train.py \
    --supervision_mode points \
    --run_name points_run1 \
    --weak_label_path ./weak_labels/weak_labels_train.pkl \
    --data_dir ./data \
    --batch_size 64 \
    --lr 2e-4 \
    --epochs 75 \
    --num_workers 8 \
    --img_size 256 \
    --checkpoint_dir ./checkpoints_a100
```

## Plan

**Phase 1: Setup & Data Preparation**


1.  **Weak Label Generation:** 
    *   Image-level tags (list of classes present).
    *   Points (centroid of each object mask).
    *   Scribbles (e.g., skeletonize mask, sample a path, or erode and sample points). Aim for simplicity first.
    *   Bounding boxes (tightest box around mask).

**Phase 2: Model Training**

5.  **Training Framework:** Set up a basic PyTorch training loop (`train.py`). Include standard components: dataloader, model definition, optimizer (AdamW), loss function placeholder, basic metric calculation (e.g., pixel accuracy during training), training/validation steps, checkpoint saving.
6.  **Implement WSSS Losses & Training Logic:**
    *   **Tags:** Add a classification head (e.g., linear layer after global pooling) to EffUnet. Train using MultiLabelSoftMarginLoss or BCEWithLogitsLoss on image tags. *For segmentation:* Use a simple CAM generation technique (e.g., using final conv layer weights and activations) after classification training, or train end-to-end using a CAM-based loss if simpler. *Initial approach:* Treat CAM output as pseudo-mask and train decoder with CE loss.
    *   **Points:** Implement partial CrossEntropyLoss, ignoring unlabeled pixels. Modify dataloader to provide point labels.
    *   **Scribbles:** Implement partial CrossEntropyLoss, ignoring unlabeled pixels. Modify dataloader to provide scribble labels.
    *   **Boxes:** Generate pseudo-masks (inside box = foreground, outside = background). Train using standard CrossEntropyLoss on these pseudo-masks. Modify dataloader to provide box labels/masks.
    *   **Hybrid (Tags + Points):** Combine the classification loss (from Tags) and the partial CE loss (from Points). Use a simple weighted sum: `Loss_total = Loss_classification + lambda * Loss_partial_CE`. Start with `lambda=1.0`.
7.  **Launch Training Runs:** Start training one model for each supervision type (Tags, Points, Scribbles, Boxes, Hybrid Tags+Points) on available GPUs. Use modest epochs initially (e.g., 50) and monitor validation loss/accuracy. Use consistent hyperparameters (learning rate, batch size) across runs where applicable.
8.  **Debugging & Monitoring:** Monitor training progress (loss curves, basic validation metrics). Debug any issues (NaN losses, slow convergence, bugs in loss implementation). Adjust hyperparameters slightly if necessary (e.g., learning rate).

**Phase 3: Evaluation & Analysis**

9.  **Quantitative Evaluation:** Write an evaluation script (`evaluate.py`). Load the best checkpoint (based on validation loss) for each trained model. Run inference on the *test set*. Calculate mean Intersection over Union (mIoU) using the ground truth pixel masks. Record mIoU for each supervision type.
10.  **Qualitative Evaluation:** Select a diverse subset of test images. Generate segmentation predictions from each model for these images. Create comparison figures showing: Input Image | Ground Truth | Tag-Supervised | Point-Supervised | Scribble-Supervised | Box-Supervised | Hybrid-Supervised. Visually analyze common failure modes.
11.  **Cost-Effectiveness Analysis:** Create a simple plot: X-axis = Estimated Relative Annotation Effort (Tags=1, Points=2, Scribbles=5, Boxes=5 - *use consistent estimates*), Y-axis = Achieved mIoU. Plot the five points.
12.  **Analyze Hybrid Result:** Specifically compare the mIoU and qualitative outputs of the Point-only model vs. the Hybrid (Tag+Point) model. Did adding tags help? Where?

> There are many things we can look into here



---

## Expected Results and Interesting Things to See

**Expected Quantitative Results:**

*   We expect a performance ranking roughly following the amount of spatial information provided:
    *   `Tags` likely lowest mIoU.
    *   `Points` likely better than Tags, but potentially still quite low depending on how well the partial CE loss works without propagation.
    *   `Scribbles` and `Boxes` likely significantly better than Points/Tags, potentially achieving similar mIoU scores to each other.
    *   `Hybrid (Tags + Points)`: This is the key experimental result. We expect it to perform *better* than the `Points`-only model. The interesting question is *how much* better? Will it approach the performance of `Scribbles` or `Boxes`? It's unlikely to surpass them, but closing a significant portion of the gap would be noteworthy.
*   The cost-effectiveness plot should visually reinforce this trade-off, potentially showing diminishing returns as annotation effort increases from points/scribbles/boxes. The hybrid point will add an interesting data point to this curve.

**Expected Qualitative Results:**

*   **Tags:** Likely produce coarse, blobby segmentations often failing to capture fine details or separate nearby objects, possibly highlighting the most salient object but missing others. CAM artifacts might be visible.
*   **Points:** Might segment small regions around the click points well but struggle with object extent and boundaries. May fail completely on objects that weren't clicked (if simulating only one click per *image* instead of per *object*). Class accuracy might be reasonable if points are class-labeled.
*   **Scribbles:** Should produce more complete object shapes than points, potentially with somewhat accurate boundaries where the scribble provides guidance, but rough elsewhere. Might struggle with very thin structures not covered by scribbles.
*   **Boxes:** Likely yield fairly complete object masks but with boundaries adhering somewhat to the box shape, potentially including background pixels near corners or failing to separate objects within the same box.
*   **Hybrid (Tags + Points):**
    *   **Interesting Thing 1 (Completeness):** Compared to Points-only, we hope to see *more complete object segmentation*. Does the global tag information help the model "fill in" the object beyond the single point location?
    *   **Interesting Thing 2 (Class Consistency):** Does adding the tag loss reduce class confusion errors compared to Points-only? E.g., if a point is ambiguously placed, does the image-level tag help assign the correct class to the segmented region?
    *   **Interesting Thing 3 (Failure Modes):** Will the hybrid model exhibit failure modes that are a blend of Tag and Point failures, or does it find a genuinely better solution? For instance, does it still struggle with boundaries like the Point model, or does it become blobby like the Tag model?

>  **Overall:** The most interesting outcome relates to the **hybrid model's effectiveness**. Seeing a improvement over points-only with such minimal extra supervision (tags) would underscore the value of even coarse global context in WSSS. Conversely, if the improvement is negligible, it would suggest that for this architecture/task, simple loss combination isn't enough, and more sophisticated fusion or propagation is needed to leverage minimal signals effectively. The qualitative analysis will be crucial to understand *why* the hybrid model performs as it does.