 Open-Ended Weakly-Supervised Semantic Segmentation Project

## 1. Overview

This project investigates the effectiveness of different **weak supervision** strategies for training semantic segmentation models. Specifically, it explores how various types and combinations of sparse, weakly-annotated labels impact model performance compared to traditional approaches like using only image-level labels or full pixel-level supervision.

The core focus is on using the **Oxford-IIIT Pet Dataset** and training a segmentation model (implicitly a SegNet variant, based on run names) using simulated weak annotations:

*   **Bounding Boxes:** Rectangular boxes encompassing the object of interest.
*   **Points:** A small number of points sampled within the object boundaries.
*   **Scribbles:** Short, free-form lines drawn roughly within the object boundaries.

The project examines the performance when using these annotation types individually and in various hybrid combinations.

## 2. Research Question

The central research question guiding this project is:

> How does the type and potential combination of sparse weak annotations (specifically bounding boxes, simulated points, and simulated scribbles) influence the segmentation performance and learning characteristics of a weakly-supervised model trained on the Oxford-IIIT Pet Dataset, compared to using only image-level labels or full pixel-level supervision?

This involves understanding:
*   The relative effectiveness of bounding boxes, points, and scribbles as standalone weak supervision signals.
*   Whether combining different types of weak annotations (hybrid supervision) yields better performance than single types.
*   How models trained with these weak signals compare (implicitly) to benchmarks trained with full masks or potentially just image tags.

## 3. Prerequisites

*   **Python:** Version 3.x recommended.
*   **pip:** Python package installer.
*   **Git:** For cloning the repository (if not already downloaded).
*   **PyTorch:** The core deep learning framework (ensure a version compatible with your system/CUDA is installed).
*   **(Optional but Recommended):** A virtual environment (`venv`, `conda`) to manage dependencies.
*   **(Optional):** CUDA-enabled GPU for significantly faster training and evaluation.

## 4. Installation

1.  **Clone the Repository (if necessary):**
    ```bash
    cd comp0197-cw2
    ```
    *Note: All subsequent commands assume you are running them from within the `comp0197-cw2` directory.*

2.  **(Optional) Create and Activate Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate  # On Windows
    ```

3.  **Install Core Dependencies:**
    Assuming PyTorch is already installed, install the additional required packages:
    ```bash
    pip install torchmetrics pillow matplotlib
    ```
    *   `torchmetrics`: For calculating evaluation metrics.
    *   `pillow`: For image loading and manipulation (PIL fork).
    *   `matplotlib`: For plotting and visualization.
    *   *(Note: Other dependencies like `torch`, `torchvision`, `numpy`, etc., might be required by the project code and should be installed if not already present. Check for a `requirements.txt` file if available).*

## 5. Workflow and Usage

This section details the steps to download data, prepare labels (if needed), train models with different supervision strategies, and evaluate them.

### Step 1: Download Data

This script downloads and prepares the Oxford-IIIT Pet Dataset.

```bash
python open_ended/download_data.py
```
*   This command will download the dataset into a directory (likely `./data` based on subsequent commands). Ensure you have sufficient disk space and an internet connection.

### Step 2: Generate Weak Labels (Optional, Attached in git)

This script generates the sparse weak annotations (points, scribbles, boxes) from the ground-truth segmentation masks provided by the original dataset.

```bash
python open_ended/weak_label_generator.py --data_dir ./data --output_file ./open_ended/weak_labels/weak_labels_train.pkl
```

*   `--data_dir ./data`: Specifies the directory where the Oxford-IIIT Pet dataset was downloaded (input).
*   `--output_file ./open_ended/weak_labels/weak_labels_train.pkl`: Specifies the path to save the generated weak labels as a Python pickle file (output).

**Important Note:** As mentioned in the original README, pre-generated weak labels (`weak_labels_train.pkl`) are already included in the `./open_ended/weak_labels/` directory within the repository. **Therefore, running this step is generally NOT necessary unless you want to regenerate the labels with different parameters or settings.**

### Step 3: Visualize Weak Labels (Optional)

This script allows you to visualize the generated (or pre-generated) weak labels overlaid on the corresponding images. This is useful for sanity checking the labels.

```bash
python -m open_ended.visualize_labels --seed 19 --label_file open_ended/weak_labels/weak_labels_train.pkl
```

*   `--seed 19`: Sets the random seed for reproducibility if the visualization involves random sampling of images.
*   `--label_file open_ended/weak_labels/weak_labels_train.pkl`: Path to the pickle file containing the weak label data.

### Step 4: Train Segmentation Models

This is the core step where the segmentation model is trained using different weak supervision configurations. The script `open_ended/train.py` is used repeatedly with different arguments.

**Common Training Arguments:**

*   `--supervision_mode [mode]`: **Crucial argument.** Specifies the type of weak supervision to use. Examples below use `points`, `scribbles`, `boxes`, and various `hybrid_...` combinations.
*   `--run_name [name]`: A unique name for this specific training run. Used for logging and naming checkpoint files (e.g., `segnet_points_run1`).
*   `--data_dir ./data`: Path to the dataset directory.
*   `--weak_label_path ./open_ended/weak_labels/weak_labels_train.pkl`: Path to the file containing the weak labels.
*   `--batch_size 64`: Number of samples per batch during training. Adjust based on GPU memory.
*   `--lr 2e-4`: Learning rate for the optimizer.
*   `--epochs 25`: Number of training epochs.
*   `--num_workers 8`: Number of worker processes for data loading. Adjust based on your system's CPU cores.
*   `--img_size 256`: Resize input images to this square dimension.
*   `--checkpoint_dir [path]`: Directory where model checkpoints (saved model weights) will be stored.

**Training Examples:**

**(a) Training with Single Weak Supervision Types:**
These commands train separate models, each using only one type of weak annotation. Checkpoints are saved in `./checkpoints_single`.

*   **Using Points:**
    ```bash
    python -m open_ended.train \
        --supervision_mode points \
        --run_name segnet_points_run1 \
        --data_dir ./data \
        --weak_label_path ./open_ended/weak_labels/weak_labels_train.pkl \
        --batch_size 64 \
        --lr 2e-4 \
        --epochs 25 \
        --num_workers 8 \
        --img_size 256 \
        --checkpoint_dir ./checkpoints_single
    ```

*   **Using Scribbles:**
    ```bash
    python -m open_ended.train \
        --supervision_mode scribbles \
        --run_name segnet_scribbles_run1 \
        --data_dir ./data \
        --weak_label_path ./open_ended/weak_labels/weak_labels_train.pkl \
        --batch_size 64 \
        --lr 2e-4 \
        --epochs 25 \
        --num_workers 8 \
        --img_size 256 \
        --checkpoint_dir ./checkpoints_single
    ```

*   **Using Bounding Boxes:**
    ```bash
    python -m open_ended.train \
        --supervision_mode boxes \
        --run_name segnet_boxes_run1 \
        --data_dir ./data \
        --weak_label_path ./open_ended/weak_labels/weak_labels_train.pkl \
        --batch_size 64 \
        --lr 2e-4 \
        --epochs 25 \
        --num_workers 8 \
        --img_size 256 \
        --checkpoint_dir ./checkpoints_single
    ```

**(b) Training with Hybrid Weak Supervision Types:**
These commands train models using combinations of weak annotation types. Checkpoints are saved in `./checkpoints_hybrid`.

*   **Using Points + Scribbles:**
    ```bash
    python -m open_ended.train \
        --supervision_mode hybrid_points_scribbles \
        --run_name segnet_hybrid_points_scribbles_run1 \
        --data_dir ./data \
        --weak_label_path ./open_ended/weak_labels/weak_labels_train.pkl \
        --batch_size 64 \
        --lr 2e-4 \
        --epochs 25 \
        --num_workers 8 \
        --img_size 256 \
        --checkpoint_dir ./checkpoints_hybrid
    ```

*   **Using Points + Boxes:**
    ```bash
    python -m open_ended.train \
        --supervision_mode hybrid_points_boxes \
        --run_name segnet_hybrid_points_boxes_run1 \
        --data_dir ./data \
        --weak_label_path ./open_ended/weak_labels/weak_labels_train.pkl \
        --batch_size 64 \
        --lr 2e-4 \
        --epochs 25 \
        --num_workers 8 \
        --img_size 256 \
        --checkpoint_dir ./checkpoints_hybrid
    ```

*   **Using Scribbles + Boxes:**
    ```bash
    python -m open_ended.train \
        --supervision_mode hybrid_scribbles_boxes \
        --run_name segnet_hybrid_scribbles_boxes_run1 \
        --data_dir ./data \
        --weak_label_path ./open_ended/weak_labels/weak_labels_train.pkl \
        --batch_size 64 \
        --lr 2e-4 \
        --epochs 25 \
        --num_workers 8 \
        --img_size 256 \
        --checkpoint_dir ./checkpoints_hybrid
    ```

*   **Using Points + Scribbles + Boxes:**
    ```bash
    python -m open_ended.train \
        --supervision_mode hybrid_points_scribbles_boxes \
        --run_name segnet_hybrid_points_scribbles_boxes_run1 \
        --data_dir ./data \
        --weak_label_path ./open_ended/weak_labels/weak_labels_train.pkl \
        --batch_size 64 \
        --lr 2e-4 \
        --epochs 25 \
        --num_workers 8 \
        --img_size 256 \
        --checkpoint_dir ./checkpoints_hybrid
    ```

### Step 5: Evaluate Trained Models

After training, use the `open_ended/evaluate.py` script to evaluate the performance of the saved model checkpoints on the test set.

```bash
python -m open_ended.evaluate \
    --data_dir ./data \
    --model_paths checkpoints_single/segnet_boxes_run1_best_acc.pth \
                  checkpoints_single/segnet_points_run1_best_acc.pth \
                  checkpoints_single/segnet_scribbles_run1_best_acc.pth \
                  checkpoints_hybrid/segnet_hybrid_points_boxes_run1_best_acc.pth \
                  checkpoints_hybrid/segnet_hybrid_points_scribbles_boxes_run1_best_acc.pth \
                  checkpoints_hybrid/segnet_hybrid_points_scribbles_run1_best_acc.pth \
                  checkpoints_hybrid/segnet_hybrid_scribbles_boxes_run1_best_acc.pth \
    --batch_size 8 \
    --device cuda
```

*   `--data_dir ./data`: Path to the dataset directory.
*   `--model_paths [path1] [path2] ...`: **Important:** List the paths to the specific checkpoint files (`.pth`) you want to evaluate. These should typically be the checkpoints saved based on the best validation performance during training (e.g., `_best_acc.pth` or similar, as indicated by the filenames). Ensure these paths correctly point to the files generated in Step 4.
*   `--batch_size 8`: Batch size for evaluation. Can often be larger than training batch size depending on GPU memory.
*   `--device cuda`: Specifies the device for evaluation. Use `cuda` for GPU or `cpu` for CPU.

*(Note: The second `evaluate.py` command listed in the original README under a separate "Evaluate" heading seems potentially inconsistent or refers to different models/runs not shown in the main training sequence. This rewritten guide focuses on evaluating the models trained in Step 4 above.)*

## 6. Additional Visualization Tools

Beyond visualizing the input labels, other visualization scripts may be available.

*   **Visualize Labels:** (Covered in Step 3) For inspecting the weak annotations.
    ```bash
    python -m open_ended.visualize_labels --label_file open_ended/weak_labels/weak_labels_train.pkl
    ```

*   **Visualize Weights:** This script likely helps in understanding the learned features of the model. It might visualize convolutional filters, attention maps, or other internal model representations.
    ```bash
    python -m open_ended.weight_visualization
    ```
    *(Note: This script might require specific arguments, such as the path to a trained model checkpoint. Check the script's implementation or add `--help` if available for more details.)*

---

## Plan

**Phase 1: Setup & Data Preparation**


1.  **Weak Label Generation:** 
    *   Points (centroid of each object mask).
    *   Scribbles (freehand stroke per object and background)
    *   Bounding boxes (tightest box around mask).

The process of generating weak labels begins by iterating through a specified subset of image files and locating their corresponding trimap annotations. For each trimap, the code first loads it, records its original dimensions, and converts it into a binary foreground mask by identifying pixels assumed to represent the object (checking for a specific pixel value, like 1 or 255, which needs to match the trimap's convention). If a valid foreground mask is created, several types of weak labels are derived from it using the original image coordinates: 

- Points are generated by finding distinct connected components (objects) within the mask and randomly sampling a predefined number of pixel coordinates (x, y) from within each component. 
- Scribbles are created by first finding the morphological skeleton (a thin line representation) of the foreground mask, identifying the longest path along this skeleton, and sampling points from it for the foreground scribble; background scribbles are similarly generated from the longest skeleton path within the region outside a dilated version of the foreground mask. 
- Bounding Boxes are determined by identifying each separate foreground object and calculating the minimum and maximum x and y coordinates that enclose it (xmin, ymin, xmax, ymax). 

These generated points, scribbles, and bounding boxes, along with the original image size, are then stored together, in a dictionary keyed by the image name, and saved to a file for later use in training weakly supervised models.
    

**Phase 2: Model Training**

*   **Data Preparation:**
    *   The `PetsDataset` is configured based on the chosen `supervision_mode` (e.g., 'points', 'scribbles', 'boxes', 'hybrid\_...').
    *   For each training sample, the dataloader provides:
        1.  The input image.
        2.  The corresponding *weak labels* (e.g., point coordinates, scribble paths, bounding box coordinates, or a mix in hybrid modes).
        3.  The *full ground truth (GT) segmentation mask* (primarily used for evaluation, not the weak training loss itself).

*   **Model Prediction:**
    *   The segmentation model (`SegNetWrapper`) processes the input image batch.
    *   It outputs dense, pixel-wise predictions (logits) representing the probability of each pixel belonging to the foreground (pet) or background class.

*   **Weakly Supervised Loss Calculation:**
    *   This is the key step where weak labels are used for learning.
    *   A specialized loss function (`CombinedLoss` or `PartialCrossEntropyLoss`) is employed.
    *   This loss function compares the model's predictions *only* at the locations specified by the *weak labels*.
        *   For **points/scribbles**: The loss is calculated based on whether the model correctly predicts foreground/background *at those specific pixel coordinates*. Pixels *not* covered by the points/scribbles are typically ignored in the loss calculation (e.g., using `IGNORE_INDEX`).
        *   For **boxes**: The loss might encourage the model to predict foreground *within* the box and potentially background *outside* it, or use other box-based weak supervision techniques integrated into the `CombinedLoss`.
        *   For **hybrid modes**: The `CombinedLoss` intelligently aggregates the loss signals derived from *all available* weak label types (points, scribbles, boxes) for that specific training image.
    *   The goal is to penalize the model for incorrect predictions *at the weakly labeled locations*.

*   **Model Update:**
    *   The calculated loss value, derived *from the weak supervision signal*, is used for backpropagation.
    *   Gradients are computed, and the optimizer (`AdamW`) updates the model's weights to minimize this weak loss over time.

*   **Evaluation and Checkpointing:**
    *   **Crucially:** While training relies on the *weak loss*, the model's actual segmentation quality is measured during validation (`validate_one_epoch`) by comparing its predictions against the *full ground truth masks* using standard metrics like Intersection over Union (IoU) and Accuracy.
    *   The best-performing model checkpoint is saved based on the performance achieved on the *validation set using the GT masks*, not the weak training loss value.



## Results - Hybrid did not improve acc

### Single

| Model Name | Test Accuracy | IoU (Background) | IoU (Pet) | Avg IoU |
| --- | --- | --- | --- | --- |
| Points | 0.5359 | 0.1109 | 0.3394 | 0.2252 |
| Scribbles | 0.7073 | 0.5417 | 0.4473 | 0.4945 |
| Boxes | 0.7151 | 0.5881 | 0.4535 | 0.5208 |


### Hybrid
Equal Loss Weight

- Combined loss, sum over each individual loss with lambda factor = 1

| Model Name | Test Accuracy | IoU (Background) | IoU (Pet) | Avg IoU |
| --- | --- | --- | --- | --- |
| Points + Scribbles | 0.6825 | 0.4912 | 0.4472 | 0.4692 |
| Points + Boxes | 0.6954 | 0.5524 | 0.4346 | 0.4935 |
| Scribbles + Boxes | 0.7104 | 0.5689 | 0.4639 | 0.5164 |
| Points + Scribbles + Boxes | 0.6981 | 0.5573 | 0.4423 | 0.4998 |

As we see that hybrid did not improve the performance, we made an hypothesis is that the equal loss factor we set in loss, so we adapt  Uncertainty based weighting to see if adaptive weight can mitigate some of the performance drop ([https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf](https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf))

*Uncertainty-based weighting*

| Model Name | Test Accuracy | IoU (Background) | IoU (Pet) | Avg IoU |
| --- | --- | --- | --- | --- |
| Points + Scribble | 0.6829 | 0.4866 | 0.4252 | 0.4559 |
| Points + Boxes | 0.6992 | 0.5537 | 0.4373 | 0.4955 |
| Scribbles + Boxes | 0.7103 | 0.5653 | 0.4493 | 0.5073 |
| Points + Scribbles + Boxes | 0.7066 | 0.5599 | 0.4453 | 0.5026 |

We can see that both loss give the same trend of decreasing iou, so in here, we will step in to analysis what did we do right and wrong, and what concluions we can have in here

### Reasons why the experiment is valid

1. Signal Conflict
    - Boxes enforce coarse object boundaries.
    - Scribbles/Points refine interiors but lack boundary information.
    - Conflict: The model receives mixed signals (e.g., "focus on boundaries" vs. "ignore boundaries, focus on interiors"), leading to optimization instability.
2. Class Imbalance & Model Behavior
    - Background dominance: With pets occupying 29.97% of pixels (background: 70.03%), a trivial baseline labeling *everything* as background would achieve:
        - Accuracy: ~70% (matches background prevalence).
        - Avg IoU: ~35% (background IoU: 70%, pet IoU: 0%).
    - Our results:
        - Box-only model: Achieves 71.51% accuracy and 52.08% Avg IoU (Pet IoU: 45.35%, Background IoU: 58.81%).
        - This exceeds the trivial baseline, proving the model does not cheat by labeling everything as background.
        - The non-zero pet IoU (45.35%) confirms meaningful learning of pet regions.
3. Pseudo generated label contain large noise
    1. Since we used the highest form label bounding box, to generate subsequent poitns and scribble, it will result in high noise, not all points fall in the regions of true pet area
    2. Since scribbles/points are derived from boxes, their noise is bounded by box accuracy. The bigger issue is their **sparsity**, not noise.
4. Annotation Coverage Analysis

| **Annotation** | **Coverage (Pixels)** | **Spatial Consistency** |
| --- | --- | --- |
| Boxes | 81% | High |
| Scribbles | 25% | Medium |
| Points | 2% | Low |
- Result: Boxes dominate training due to superior coverage and boundary information. Scribbles/points are too sparse to meaningfully contribute.
1. Loss Dynamics
    - Box-only Loss: Stable optimization (lowest test loss: 0.5696).
    - Hybrid Loss: Higher losses (e.g., 1.2276 for box+point+scribble) indicate conflicting gradients.
