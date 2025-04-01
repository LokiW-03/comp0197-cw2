


#### **Model Files**  

The following files contain implementations of different segmentation models:
- `baseline_segnet.py` - Implements the SegNet model.
- `baseline_unet.py` - Implements the U-Net model.
- `efficient_unet.py` - Implements the EfficientUNet model.
- `segnext.py` - Implements the SegNext model.
    
#### **Supporting Files**  

- `data.py` - Downloads the Oxford-IIIT Pet dataset and applies necessary transformations, including resizing, normalization, and label processing for model input.
- `visual.py` - Contains functions for overlaying predicted segmentation masks onto images for visualization.
- `metrics.py` - Defines various evaluation metrics for assessing model performance.
- `metrics_ws.py` - Mirror of metrics.py but includes optional masking argument
- `train.py` - Main script for training models, evaluating performance, and saving model checkpoints.
- `train_ws.py` - Mirror of train.py but includes option to mask out contour class when evaluating pseudo masks for WSS.

#### **Running the project**  
1.  **Dependencies**: The project runs in a Miniconda environment and is configured to work with the same setup as coursework 1 
2.  **Choosing a model**: Open `train.py` and specify the model you want to train by uncommenting the relevant section.
3.  **Configure hyperparameters**: Adjust batch size, learning rate (schedule), optimizer, number of epochs, and other parameters within `train.py`.
4.  **Run the training script**: `python train.py` This will start the training process, print updates when batches are completed, display performance metrics at each epoch, and save model weights (as well as optimiser and learning rate schedule) periodically.
5.  **Visualize results (optional)**: The `visual.py` script allows for overlaying predicted segmentation masks onto input images to assess model predictions.
6.  **Evaluate the model**: The training script automatically computes performance metrics on the test set while training. You can also load saved model weights for further evaluation.


#### **Model Overview**

**SegNet** is a fully convolutional encoder-decoder architecture specifically designed for semantic segmentation. It leverages the VGG16 backbone for its encoder, omitting the fully connected layers.
- **Encoder:**
    - Composed of 13 convolutional layers from the VGG16-BN architecture.
    - Each block includes convolution, batch normalization, ReLU activation, and max pooling (with pooling indices saved for unpooling).
    - Pretrained VGG16 weights are used for initialization.
- **Decoder:**
    - Mirrors the encoder with corresponding decoder layers.
    - Uses max unpooling layers guided by the saved pooling indices to preserve spatial structure.
    - Followed by convolutional layers to densify the sparse feature maps.        
    - Ends with a softmax classifier to assign class probabilities to each pixel.

**UNet** is a fully convolutional architecture known for its U-shaped design, and consists of a symmetric encoder-decoder structure with skip connections to retain spatial information.
- **Encoder (Contracting Path):**
    - Consists of repeated double convolution blocks followed by max pooling to downsample the spatial resolution.
    - The number of feature channels increases at deeper levels to capture complex representations.
- **Decoder (Expansive Path):**
    - Uses transposed convolutions for upsampling.
    - Features are concatenated with corresponding encoder outputs via skip connections to recover fine-grained details.
    - Followed by double convolutions to refine the segmentation map.
- **Final Layer:**
    - A 1x1 convolution to map the feature maps to the desired number of classes.
        
**EfficientUNet** integrates the EfficientNet-B0 backbone into the UNet framework to balance segmentation accuracy with computational efficiency.
- **Encoder:**
    - Uses pretrained EfficientNet-B0 layers to extract hierarchical features.
    - The stem and block layers progressively downsample the image while increasing feature richness.
- **Decoder:**
    - Comprises a series of upsampling blocks (transpose convolutions) with skip connections to intermediate EfficientNet blocks.
    - Each upsampling block reduces channel depth and fuses it with encoder features to refine spatial details.
- **Final Output:**
    - The upsampled output is passed through a final convolutional layer to produce the segmentation map.


**SegNeXt** is a modern encoder-decoder segmentation model that utilizes a Multi-Scale Convolutional Attention Network (MSCAN) backbone with a lightweight decoder.
- **Encoder (MSCAN Backbone):**
    - Composed of four stages, each consisting of:        
        - A downsampling layer (via strided convolution).
        - Several **MSCA blocks**, which combine:
            - **Large Kernel Attention:** depthwise convolutions with large kernels (e.g., 5×5, 7×7, 11×11, 21×21) to capture long-range dependencies.
            - **ConvFeedForward:** 1x1 convolutions to expand and project channels with GELU activations.
        - Residual connections, layer scaling, and stochastic depth for stable training.
- **Decoder:**
    - All features from different stages are upsampled to a common resolution.
    - Multi-scale features are concatenated and fused via a 1x1 convolution.
    - The fused tensor is passed through a classifier convolution to produce pixel-wise logits.
    - Final output is bilinearly upsampled to the original image size.




#### **Training Overview**  

Hyperparameter tuning was conducted individually for each of the four models model to evaluate how different training configurations impacted performance. We explored various combinations of optimizers, learning rate scheduling strategies, and dropout (where applicable). Performance was evaluated across multiple metrics such as accuracy, precision, recall, IoU, and Dice score. The number of training epochs and batch size constant for each run:
- Epochs: 10
- Batch size: 16

**SegNet** 

|                 |   |   |           |        |   |              |   |                 |   |   |                       |                          |                |             |          |           |        |      |      |
|-----------------|---|---|-----------|--------|---|--------------|---|-----------------|---|---|-----------------------|--------------------------|----------------|-------------|----------|-----------|--------|------|------|
| Baseline SegNet |Hyperparameters|   |           |        |   |              |   |                 |   |   |                       |                          |                | Performance |          |           |        |      |      |
| Model Number    |Epochs|Batch Size| Optimizer | LR     |Momentum| Weight Decay |Learning Rate Constant (Y/N)| LR Scheduler    |Relevant Params (1)|Relevant Params (2)| Data Transformations  | Dropout                  | Model saved as | Loss        | Accuracy | Precision | Recall | IOU  | DICE |
| 1               |10|16| AdamW     | 1e-3   |NA| 1e-4   |N| StepLR          |step_size= 15|gamma =0.1| NA                    | NA                       | SegNet         | 0.26        | 0.90     | 0.86      | 0.81   | 0.73 | 0.83 |
| 2               |10|16| AdamW     | 1e-3   |NA| 1e-4   |N| CosineAnnealing |t_max = 50|NA| NA                    | NA                       | SegNet_CA      | 0.24        | 0.91     | 0.86      | 0.85   | 0.76 | 0.85 |
| 3               |10|16| AdamW     | 1e-3   |NA| 1e-4   |N| StepLR          |step_size= 15|gamma =0.1| NA                    | 0.5 at. both conv layers | SegNet_DROP    | 0.27        | 0.90     | 0.84      | 0.84   | 0.74 | 0.84 |
| 4               |10|16| AdamW     | 1e-3   |NA| 1e-4   |N| StepLR          |step_size= 15|gamma =0.1| Rotation of 45 degree | NA                       | SegNet_rotate  | 0.64        | 0.73     | 0.66      | 0.58   | 0.46 | 0.60 |


**UNet**  

|   |   |   |           |      |   |              |   |   |   |   |   |   |   |   |   |   |   |   |   |
|---|---|---|-----------|------|---|--------------|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Baseline Unet|Hyperparameters|   |           |      |   |              |   |   |   |   |   |   |   |Performance|   |   |   |   |   |
|Model Number|Epochs|Batch Size| Optimizer | LR   |Momentum| Weight Decay |Learning Rate Constant (Y/N)|LR Scheduler|Relevant Params (1)|Relevant Params (2)|Data Transformations|Dropout|Model saved as|Loss|Accuracy|Precision|Recall|IOU|DICE|
|1|10|16| AdamW     | 1e-3 |NA| 1e-4         |N|StepLR|step_size= 15|gamma =0.1|NA|NA|UNet1|0.37|0.86|0.8|0.77|0.66|0.78|
|2|10|16| RAdam     | 5e-4 |NA| 1e-5         |N|CosineAnnealing|t_max = 50|NA|NA|NA|UNet2|0.35|0.87|0.81|0.78|0.67|0.79|
|3|10|16| RAdam     | 5e-4 |NA| 1e-5         |N|CosineAnnealing|t_max = 50|NA|NA|0.3 at. both conv layers|UNet3|0.38|0.86|0.84|0.78|0.68|0.79|


**EfficientNet** 

**SegNext**
