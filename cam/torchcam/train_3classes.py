import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

from model.fpn import FPN

# copied from https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/examples/camvid_segmentation_multiclass.ipynb

class PetModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, lr=2e-4, T_max=50, eta_min=1e-5, **kwargs):
        super().__init__()
        #Save input hyperparameters
        self.save_hyperparameters()
        
        if arch.lower() == "fpn":
            use_pretrained = True
            if in_channels != 3 and use_pretrained:
                print(f"Warning: Custom FPN with pretrained=True expects in_channels=3. Received {in_channels}. Forcing pretrained=False.")
                use_pretrained = False
                
            self.model = FPN(
                in_channels=in_channels,
                classes=out_classes,
                pretrained=use_pretrained,
                **kwargs,
            )
            # Set normalization for standard ImageNet pretraining
            print("Setting normalization to ImageNet defaults for custom FPN.")
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        else:
            self.model = smp.create_model(
                arch,
                encoder_name=encoder_name,
                in_channels=in_channels,
                classes=out_classes,
                **kwargs,
            )
            # Preprocessing parameters for image normalization
            params = smp.encoders.get_preprocessing_params(encoder_name)
            self.number_of_classes = out_classes
            self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
            self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # Loss function for multi-class segmentation
        self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)

        # Step metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, image):
        # Normalize image
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image, mask = batch

        # Ensure that image dimensions are correct
        assert image.ndim == 4  # [batch_size, channels, H, W]

        # Ensure the mask is a long (index) tensor
        mask = mask.long()

        # Mask shape
        if mask.ndim == 4:
            mask = mask.squeeze(1)
        assert mask.ndim == 3  # [batch_size, H, W]

        # Predict mask logits
        logits_mask = self.forward(image)

        assert (
            logits_mask.shape[1] == self.number_of_classes
        )  # [batch_size, number_of_classes, H, W]

        # Ensure the logits mask is contiguous
        logits_mask = logits_mask.contiguous()

        # Compute loss using multi-class Dice loss (pass original mask, not one-hot encoded)
        loss = self.loss_fn(logits_mask, mask)

        # Apply softmax to get probabilities for multi-class segmentation
        prob_mask = logits_mask.softmax(dim=1)

        # Convert probabilities to predicted class labels
        pred_mask = prob_mask.argmax(dim=1)

        # Compute true positives, false positives, false negatives, and true negatives
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask, mask, mode="multiclass", num_classes=self.number_of_classes
        )

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # Aggregate step metrics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # Per-image IoU and dataset IoU calculations
        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.T_max,
            eta_min=self.hparams.eta_min,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

if __name__ == "__main__":
    from cam.load_pseudo import load_pseudo_dataset,load_pseudo
    from model.data import testset
    from torch.utils.data import DataLoader

    device = torch.device('cuda' if torch.cuda.is_available()
                        else 'mps' if torch.backends.mps.is_available()                          
                           else 'cpu')

    
    pseudo_loader = load_pseudo("cam/saved_models/resnet50_pet_cam_pseudo.pt",
                                batch_size=64,
                                shuffle=True,
                                device=device)
    test_loader = DataLoader(testset, batch_size=64, shuffle=False)


    EPOCHS = 10
    T_MAX = EPOCHS * len(pseudo_loader)
    OUT_CLASSES = 3
    LR = 2e-4
    ETA_MIN = 1e-5

    # Initialize model
    model = PetModel("FPN", "resnet34", in_channels=3, out_classes=OUT_CLASSES, lr=LR, 
                     T_max=T_MAX, eta_min=ETA_MIN)

    trainer = pl.Trainer(max_epochs=EPOCHS, log_every_n_steps=1)

    trainer.fit(
        model,
        train_dataloaders=pseudo_loader,
        val_dataloaders=test_loader,
    )

    valid_metrics = trainer.validate(model, dataloaders=test_loader, verbose=False)
    print(valid_metrics)

    test_metrics = trainer.test(model, dataloaders=test_loader, verbose=False)
    print(test_metrics)

    from cam.torchcam.exp_viz import vis
    vis(test_loader, model)