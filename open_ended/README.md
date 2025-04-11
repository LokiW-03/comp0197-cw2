
# Open-Ended


## Research Questions
How does the type and potential combination of sparse weak annotations (specifically bounding boxes, simulated points, and simulated scribbles) influence the segmentation performance and learning characteristics of a weakly-supervised model trained on the Oxford-IIIT Pet Dataset, compared to using only image-level labels or full pixel-level supervision?

## Installation
Additional installation
```bash
pip install torchmetrics
```

## Run

```
!cd comp0197-cw2/ && python open_ended/download_data.py

!cd comp0197-cw2/ && python open_ended/weak_label_generator.py --data_dir ./data --output_file ./weak_labels/weak_labels_train.pkl


!cd comp0197-cw2/  && python -m open_ended.train \
    --supervision_mode points \
    --run_name segnet_points_run1 \
    --data_dir ./data \
    --weak_label_path ./weak_labels/weak_labels_train.pkl \
    --batch_size 64 \
    --lr 2e-4 \
    --epochs 25 \
    --num_workers 8 \
    --img_size 256 \
    --checkpoint_dir ./checkpoints_single \
    --augment

!cd comp0197-cw2/  && python -m open_ended.train \
    --supervision_mode scribbles \
    --run_name segnet_scribbles_run1 \
    --data_dir ./data \
    --weak_label_path ./weak_labels/weak_labels_train.pkl \
    --batch_size 64 \
    --lr 2e-4 \
    --epochs 25 \
    --num_workers 8 \
    --img_size 256 \
    --checkpoint_dir ./checkpoints_single \
    --augment

!cd comp0197-cw2/  && python -m open_ended.train \
    --supervision_mode boxes \
    --run_name segnet_boxes_run1 \
    --data_dir ./data \
    --weak_label_path ./weak_labels/weak_labels_train.pkl \
    --batch_size 64 \
    --lr 2e-4 \
    --epochs 25 \
    --num_workers 8 \
    --img_size 256 \
    --checkpoint_dir ./checkpoints_single \
    --augment


!cd comp0197-cw2/  && python -m open_ended.train \
    --supervision_mode hybrid_points_scribbles \
    --run_name segnet_hybrid_points_scribbles_run1 \
    --data_dir ./data \
    --weak_label_path ./weak_labels/weak_labels_train.pkl \
    --batch_size 64 \
    --lr 2e-4 \
    --epochs 25 \
    --num_workers 8 \
    --img_size 256 \
    --checkpoint_dir ./checkpoints_hybrid \
    --augment

!cd comp0197-cw2/  && python -m open_ended.train \
    --supervision_mode hybrid_points_boxes \
    --run_name segnet_hybrid_points_boxes_run1 \
    --data_dir ./data \
    --weak_label_path ./weak_labels/weak_labels_train.pkl \
    --batch_size 64 \
    --lr 2e-4 \
    --epochs 25 \
    --num_workers 8 \
    --img_size 256 \
    --checkpoint_dir comp0197/checkpoints_hybrid \
    --augment

!cd comp0197-cw2/  && python -m open_ended.train \
    --supervision_mode hybrid_scribbles_boxes \
    --run_name segnet_hybrid_scribbles_boxes_run1 \
    --data_dir ./data \
    --weak_label_path ./weak_labels/weak_labels_train.pkl \
    --batch_size 64 \
    --lr 2e-4 \
    --epochs 25 \
    --num_workers 8 \
    --img_size 256 \
    --checkpoint_dir ./checkpoints_hybrid \
    --augment

!cd comp0197-cw2/  && python -m open_ended.train \
    --supervision_mode hybrid_points_scribbles_boxes \
    --run_name segnet_hybrid_points_scribbles_boxes_run1 \
    --data_dir ./data \
    --weak_label_path ./weak_labels/weak_labels_train.pkl \
    --batch_size 64 \
    --lr 2e-4 \
    --epochs 25 \
    --num_workers 8 \
    --img_size 256 \
    --checkpoint_dir ./checkpoints_hybrid \
    --augment
```

## Visualization


Visualize labels
```
python -m open_ended.visualize_labels    
```

Visualize weights
```
python -m open_ended.weight_visualization
```



## Evaluate

```
!cd comp0197-cw2/  && python evaluate.py \
    --data_dir ./data \
    --model_paths checkpoints_single/segnet_point_run1_best_acc.pth \
                  checkpoints_single/segnet_scatter_run1_best_acc.pth \
                  checkpoints_single/segnet_boxes_run1_best_acc.pth \
                  checkpoints_hybrid/segnet_hybrid_point_scatter_run1_best_acc.pth \
    --batch_size 8 \
    --device cuda
```

## Plan

**Phase 1: Setup & Data Preparation**


1.  **Weak Label Generation:** 
    *   Points (centroid of each object mask).
    *   Scribbles (freehand stroke per object and background)
    *   Bounding boxes (tightest box around mask).
    Fo

**Phase 2: Model Training**

5.  **Training Framework:** Set up a basic PyTorch training loop. Include standard components: dataloader, model definition, optimizer (AdamW), loss function placeholder, basic metric calculation (e.g., pixel accuracy during training), training/validation steps, checkpoint saving.
6.  **Implement WSSS Losses & Training Logic:**
    *   **Points:** Implement partial CrossEntropyLoss, ignoring unlabeled pixels. Modify dataloader to provide point labels.
    *   **Scribbles:** Implement partial CrossEntropyLoss, ignoring unlabeled pixels.
    *   **Boxes:** Generate pseudo-masks (inside box = foreground, outside = background). Train using standard CrossEntropyLoss on these pseudo-masks. Modify dataloader to provide box labels/masks.
7.  **Launch Training Runs:** Start training one model for each supervision type (Points, Scribbles, Boxes, Hybrid) on available GPUs. Use modest epochs initially (e.g., 50) and monitor validation loss/accuracy. Use consistent hyperparameters (learning rate, batch size) across runs where applicable.
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
    *   `Points` likely better than Tags, but potentially still quite low depending on how well the partial CE loss works without propagation.
    *   `Scribbles` and `Boxes` likely significantly better than Points/Tags, potentially achieving similar mIoU scores to each other.
    *   `Hybrid (Tags + Points)`: This is the key experimental result. We expect it to perform *better* than the `Points`-only model. The interesting question is *how much* better? Will it approach the performance of `Scribbles` or `Boxes`? It's unlikely to surpass them, but closing a significant portion of the gap would be noteworthy.
*   The cost-effectiveness plot should visually reinforce this trade-off, potentially showing diminishing returns as annotation effort increases from points/scribbles/boxes. The hybrid point will add an interesting data point to this curve.

**Expected Qualitative Results:**

*   **Points:** Might segment small regions around the click points well but struggle with object extent and boundaries. May fail completely on objects that weren't clicked (if simulating only one click per *image* instead of per *object*). Class accuracy might be reasonable if points are class-labeled.
*   **Scribbles:** Should produce more complete object shapes than points, potentially with somewhat accurate boundaries where the scribble provides guidance, but rough elsewhere. Might struggle with very thin structures not covered by scribbles.
*   **Boxes:** Likely yield fairly complete object masks but with boundaries adhering somewhat to the box shape, potentially including background pixels near corners or failing to separate objects within the same box.



No longer under experiment because of tags is not valid to compare with, but we can use the idea from this for the hybrid experiment.

*   **Hybrid (Tags + Points):**
    *   **Interesting Thing 1 (Completeness):** Compared to Points-only, we hope to see *more complete object segmentation*. Does the global tag information help the model "fill in" the object beyond the single point location?
    *   **Interesting Thing 2 (Class Consistency):** Does adding the tag loss reduce class confusion errors compared to Points-only? E.g., if a point is ambiguously placed, does the image-level tag help assign the correct class to the segmented region?
    *   **Interesting Thing 3 (Failure Modes):** Will the hybrid model exhibit failure modes that are a blend of Tag and Point failures, or does it find a genuinely better solution? For instance, does it still struggle with boundaries like the Point model, or does it become blobby like the Tag model?

>  **Overall:** The most interesting outcome relates to the **hybrid model's effectiveness**. Seeing a improvement over points-only with such minimal extra supervision (tags) would underscore the value of even coarse global context in WSSS. Conversely, if the improvement is negligible, it would suggest that for this architecture/task, simple loss combination isn't enough, and more sophisticated fusion or propagation is needed to leverage minimal signals effectively. The qualitative analysis will be crucial to understand *why* the hybrid model performs as it does.



## Exploring Hybrid Spatial Supervision

After I trained model with individual types of weak labels. I found anther interesting part with mixing them, which seems highly relevant for practical use cases and less explored in an empircal analysis from past papers.

Think about it: if a team is already annotating bounding boxes, we could automatically generate points (like centroids) and maybe even basic scribbles from those boxes. This would give us richer training data for the segmentation model without asking annotators to do more work.

To see how much benefit we actually get from this, I think the below set of experiments are good enough to see how hybrid improve/or not the result.



So, we have experimented with:
  - Points + Scribble
  - Poitns + Bounding box
  - scrible + bounding box
  - points + scribble + bounding box


All the models were trained with same settings, ex. batch size etc.


## Results

### Single Feature
**SegNet**

| SegNet  | Performance  |               |           |               |
|---------|--------------|---------------|-----------|---------------|
| Feature | Best Val IOU | Best Test IOU | Test Loss | Test Accuracy |
| box     | 0.4826       | 0.5338        | 0.5696    | 0.7295        |
| Scribbles | 0.3343       | 0.3307        | 3.0825    | 0.5080        |
| point   | 0.3320       | 0.1522        | 4.5684    | 0.5000        |

### Hybrid Feature

**SegNet**

| SegNet              | Performance  |               |           |               |
|---------------------|--------------|---------------|-----------|---------------|
| Feature             | Best Val IOU | Best Test IOU | Test Loss | Test Accuracy |
| box, point          | 0.4655       | 0.4793        | 0.9793    | 0.7018        |
| box, Scribbles        | 0.4630       | 0.4737        | 1.0722    | 0.7144        |
| point, Scribbles      | 0.3444       | 0.2027        | 0.3929    | 0.5344        |
| box, point, Scribbles | 0.4694       | 0.4705        | 1.2276    | 0.7185        |
