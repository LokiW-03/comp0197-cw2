import logging
import torch.nn as nn
import numpy as np
from torchvision import models

import copy

class BaseModel(nn.Module):
    """
    An abstract base class for all neural network models.

    This class extends `torch.nn.Module` and provides common functionality:
    - A logger tied to the class name for consistent logging.
    - A method to print a summary of trainable parameters.
    - A string representation that includes the number of trainable parameters.

    Intended to be subclassed by specific model implementations.
    Subclasses must override the `forward()` method.

    Methods:
        forward():
            Abstract method that must be implemented in the subclass. Raises NotImplementedError.

        summary():
            Logs the total number of trainable parameters in the model.

        __str__():
            Returns a string representation of the model including trainable parameter count.
    """

    def __init__(self):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self):
        raise NotImplementedError

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info(f'Nbr of trainable parameters: {nbr_params}')

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        return super(BaseModel, self).__str__() + f'\nNbr of trainable parameters: {nbr_params}'


# Code adopted from https://github.com/yassouali/pytorch-segmentation/blob/master/models/segnet.py
# SegNet uses the 13 first layers from VGG16 and left out the fully connected layer
class SegNet(BaseModel):
    """
    A semantic segmentation model based on the VGG16 architecture with batch normalization.

    This implementation constructs an encoder-decoder network where:
    - The encoder is derived from the convolutional layers of VGG16-BN (excluding the fully connected layers).
    - The decoder mirrors the encoder structure, using unpooling and transposed convolution operations.
    - Pooling indices are stored during encoding and reused during decoding for spatial reconstruction.

    Args:
        in_channels (int): Number of input channels. Default is 3 (RGB images).
        freeze_batchnormal (bool): If True, all BatchNorm2d layers are frozen (set to eval mode). Default is False.
        output_num_classes (int): Number of output classes for semantic segmentation. Default is 3.
        **_ : Accepts and ignores additional keyword arguments for compatibility.

    Methods:
        forward(x):
            Runs a forward pass of the SegNet model, returning the segmentation map.

        get_decoder_params():
            Returns the parameters of the model (used for optimization or freezing).

        freeze_bn():
            Sets all BatchNorm2d layers to evaluation mode (freezes their running stats and parameters).
    """
    def __init__(self, in_channels=3, freeze_batchnormal=False, output_num_classes=3, **_):
        super(SegNet, self).__init__()
        vgg_bn = models.vgg16_bn(weights='VGG16_BN_Weights.IMAGENET1K_V1')
        encoder = list(vgg_bn.features.children())
        #print(encoder)

        # Adjust the input size
        if in_channels != 3:
            encoder[0] = nn.Conv2d(
                in_channels, 64, kernel_size=3, stride=1, padding=1)

        # Encoder, VGG without fully connected layer
        self.stage1_encoder = nn.Sequential(*encoder[:6])
        self.stage2_encoder = nn.Sequential(*encoder[7:13])
        self.stage3_encoder = nn.Sequential(*encoder[14:23])
        self.stage4_encoder = nn.Sequential(*encoder[24:33])
        self.stage5_encoder = nn.Sequential(*encoder[34:-1])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # Decoder, same as the encoder but reversed
        # List deep copy
        decoder = copy.deepcopy(encoder)

        decoder = [i for i in list(reversed(decoder))
                   if not isinstance(i, nn.MaxPool2d)]
        # Replace the last conv layer
        decoder[-1] = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # When reversing, we also reversed conv->batchN->relu, correct it
        decoder = [item for i in range(0, len(decoder), 3)
                   for item in decoder[i:i+3][::-1]]
        #print(decoder)
        # Replace some conv layers & batchN after them
        for i, module in enumerate(decoder):
            if isinstance(module, nn.Conv2d):
                if module.in_channels != module.out_channels:
                    decoder[i+1] = nn.BatchNorm2d(module.in_channels)
                    decoder[i] = nn.Conv2d(
                        module.out_channels, module.in_channels, kernel_size=3, stride=1, padding=1)

        self.stage1_decoder = nn.Sequential(*decoder[0:9])
        self.stage2_decoder = nn.Sequential(*decoder[9:18])
        self.stage3_decoder = nn.Sequential(*decoder[18:27])
        self.stage4_decoder = nn.Sequential(*decoder[27:33])
        self.stage5_decoder = nn.Sequential(*decoder[33:],
                                            nn.Conv2d(
                                                64, output_num_classes, kernel_size=3, stride=1, padding=1)
                                            )
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        # do we need to initialise weight?
        self._initialize_weights(self.stage1_decoder, self.stage2_decoder, self.stage3_decoder,
                                 self.stage4_decoder, self.stage5_decoder)
        if freeze_batchnormal:
            self.freeze_bn()

    @staticmethod
    def _initialize_weights(*stages):
        for modules in stages:
            for module in modules.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()

    def forward(self, x):
        # Encoder
        x = self.stage1_encoder(x)
        x1_size = x.size()
        x, indices1 = self.pool(x)

        x = self.stage2_encoder(x)
        x2_size = x.size()
        x, indices2 = self.pool(x)

        x = self.stage3_encoder(x)
        x3_size = x.size()
        x, indices3 = self.pool(x)

        x = self.stage4_encoder(x)
        x4_size = x.size()
        x, indices4 = self.pool(x)

        x = self.stage5_encoder(x)
        x5_size = x.size()
        x, indices5 = self.pool(x)

        # Decoder
        x = self.unpool(x, indices=indices5, output_size=x5_size)
        x = self.stage1_decoder(x)

        x = self.unpool(x, indices=indices4, output_size=x4_size)
        x = self.stage2_decoder(x)

        x = self.unpool(x, indices=indices3, output_size=x3_size)
        x = self.stage3_decoder(x)

        x = self.unpool(x, indices=indices2, output_size=x2_size)
        x = self.stage4_decoder(x)

        x = self.unpool(x, indices=indices1, output_size=x1_size)
        x = self.stage5_decoder(x)

        return x

    def get_decoder_params(self):
        return self.parameters()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
