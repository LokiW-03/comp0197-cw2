# Hybrid did not improve acc

Single

| Model Name | Test Accuracy | IoU (Background) | IoU (Pet) | Avg IoU |
| --- | --- | --- | --- | --- |
| Points | 0.5359 | 0.1109 | 0.3394 | 0.2252 |
| Scribbles | 0.7073 | 0.5417 | 0.4473 | 0.4945 |
| Boxes | 0.7151 | 0.5881 | 0.4535 | 0.5208 |

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