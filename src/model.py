import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    # two consecutive conv layers each followed by batch norm and relu
    # this is the basic building block repeated throughout the network
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    # one encoder step: conv block followed by max pooling
    # returns both the pooled output and the skip connection
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()

        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)
        pooled = self.pool(skip)
        return pooled, skip


class DecoderBlock(nn.Module):
    # one decoder step: upsample then concatenate skip connection then conv
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        self.upsample = nn.ConvTranspose2d(in_channels, out_channels,
                                            kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for binary brain tumour segmentation.
    Input:  RGB MRI slice  (batch, 3, 256, 256)
    Output: tumour mask    (batch, 1, 256, 256)
    """
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # encoder path — progressively double channels
        ch = in_channels
        for f in features:
            self.encoders.append(EncoderBlock(ch, f))
            ch = f

        # bottleneck — deepest part of the network
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        # decoder path — progressively halve channels
        for f in reversed(features):
            self.decoders.append(DecoderBlock(f * 2, f))

        # final 1x1 conv to produce single channel mask
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []

        # encoder: go down collecting skip connections
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)

        # bottleneck
        x = self.bottleneck(x)

        # decoder: go up using skip connections in reverse order
        skips = skips[::-1]
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skips[i])

        return self.final_conv(x)
