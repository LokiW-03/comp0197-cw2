# Weakly segmentation performance

```sh
!python -m cam.finetune
!python -m cam.postprocessing --model=resnet
python -m model.train --model=effunet --pseudo_path=cam/saved_models/resnet50_pet_cam_pseudo.pt --pseudo
# basic model with current settings

Model weights, optimiser, scheduler order saved for model effunet at epoch 10
Final test metrics:   -> {'loss': 12.06634123587225, 'accuracy': 0.305368797270275, 'precision': 0.2807703690344971, 'recall': 0.31186618541592503, 'iou': 0.11991140069112559, 'dice': 0.19646425337273538}
```