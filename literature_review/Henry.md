# Base

## Classifier Model  
- **ResNet-50** or **VGG-16** for classification tasks.  

## Segmentation Model  
- **DeepLabv3** (using the same backbone as the classifier) or **Fully Convolutional Network (FCN)**.  
  - *DeepLabv3* is widely used, easy to implement with PyTorch or TensorFlow, and provides strong performance.  
  - *FCN* is a simpler and more lightweight option.  

## Workflow  
1. **Train the Classifier** → Generate Class Activation Maps (CAMs) → Apply Thresholding → Create Pseudo-Masks.  
2. **Train the Segmentation Model** using these pseudo-masks as labels.  

## Evaluation Metrics  

- **Mean Intersection over Union (mIoU)**: The primary and most reliable metric for segmentation tasks.  
- **Pixel Accuracy**: Measures the percentage of correctly classified pixels, though it offers less detailed insights than mIoU.  
- **Per-Class IoU** (Optional): Useful for identifying which classes are segmented most or least accurately, but may not be necessary for a baseline evaluation.

## Ablation
- CAM Threshold Sensitivity
    - Vary the threshold used to convert CAM scores to binary masks (e.g., 20%, 30%, 50% of the max activation).
    Observe how it affects the final mIoU.
    Helps show if the sweet spot is around a certain threshold or if the approach is robust.
- Backbone Choice (Optional)
    - If resources allow, test a second backbone (e.g., VGG-16 vs. ResNet-50).
    Show how the choice of classifier backbone affects the quality of pseudo-labels and final segmentation.
- Post-Processing or No Post-Processing (Optional)
    - Compare running a CRF (or a simple morphological operation) vs. no post-processing on the thresholded CAMs.
    - This ablation is straightforward: does smoothing the pseudo-labels help or not?


## Paper
- Token Contrast (ToCo) with Vision Transformers (CVPR)
    - ViTs see the image in pieces.
    - Normally, those pieces blur together.
    - **Token Contrast** keeps them nicely separated so the model knows exactly which pieces belong to the cat, and which are background—leading to better segmentation with minimal supervision.
- Expansion-Shrinkage via Deformable Convolution (NEUIRPS)
    - In the Expansion stage, they add a module called an “expansion sampler” using deformable convolution offsets. By applying an inverse supervision signal (maximizing the classification loss), this module pushes the network to discover less discriminative regions of the object, effectively expanding the CAM to cover more of the objec.

    - Next, in the Shrinkage stage, a second deformable conv branch – the “shrinkage sampler” – learns to identify and retract the parts that likely belong to background (false positives introduced during expansion).  

# Novel

1. Does refining the CAM-generated pseudo-labels improve segmentation performance?
    - Idea
        - Integrate a post-processing step using denseCRF (Conditional Random Field) on the CAM-generated masks before they are used to train the segmentation network. In practice, this means after obtaining the raw CAM for each training image (baseline Step 1), apply a dense CRF algorithm to produce a refined mask that has smoother boundaries and removes isolated false positives. 
        - Retrain the segmentation model using these CRF-refined masks as the pseudo ground truth. 

2. How much can a small amount of strong annotation (e.g. one pixel or scribble per object) boost weakly-supervised segmentation?
3. How does the type of weak supervision signal (image-level labels vs. bounding boxes vs. scribbles vs. point clicks) influence segmentation results and annotation effort? 
    - Different weak annotations provide different levels of guidance – for instance, scribble annotations give rough object locations (more information than an image tag) and are more flexible than bounding boxes for irregular shapes .
