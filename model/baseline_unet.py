import torch
import torch.nn as nn



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_op(x)

class UpSampleDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampleDecoder, self).__init__()
        # Define upsampling layers (ConvTranspose2d)
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.doubleconv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip_connection):
        # Upsample the feature map using ConvTranspose2d
        x = self.upconv(x)  # Upsample the feature map using transpose convolution

        # Concatenate the upsampled output with the skip connection from the encoder
        x = torch.cat([x, skip_connection], dim=1) # Concatenate along the channel dimension

        # Apply convolution after concatenation
        x = self.doubleconv(x)
        return x


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.doubleconv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.doubleconv(x)
        p = self.pool(down)

        return down, p


class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)

        self.bottle_neck = DoubleConv(512, 1024)

        self.up_convolution_1 = UpSampleDecoder(1024, 512)
        self.up_convolution_2 = UpSampleDecoder(512, 256)
        self.up_convolution_3 = UpSampleDecoder(256, 128)
        self.up_convolution_4 = UpSampleDecoder(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)

        b = self.bottle_neck(p4)

        up_1 = self.up_convolution_1(b, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)

        out = self.out(up_4)
        return out
