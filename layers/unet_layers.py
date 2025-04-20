import torch.nn as nn
import torch.nn.functional as F
import torch

# Define the DoubleConv block, which consists of two convolutional layers followed by batch normalization and ReLU activation
class DoubleConv(nn.Module):
    """
    (convolution => [BN] => ReLU) * 2
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
# Define the Down block, which downscales the input using max pooling followed by a DoubleConv block
class Down(nn.Module):
    """
    Downscaling with maxpool then double conv
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    
# Define the bottleneck
class BottleNeck(nn.Model):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()

        self.bottleneck = nn.Sequential(
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.bottleneck(x)
    
# Define the Up block, which upscales the input and concatenates it with the skip connection from the encoder, followed by a DoubleConv block
class Up(nn.Module):
    """
    Upscaling then double conv
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

         # If bilinear interpolation is used, upscale using bilinear interpolation and reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # Otherwise, use transposed convolution for upscaling
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # breakpoint()
        x1 = self.up(x1)
        # Calculate the difference in size between the input and the skip connection
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # Pad the input to match the size of the skip connection
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # Concatenate the skip connection with the upscaled input
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
# Define the final output convolution layer
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
