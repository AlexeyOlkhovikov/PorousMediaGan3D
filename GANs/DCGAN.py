import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class PreActResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Parameters
        ----------
        in_channels : int
            number channels in the input image
        out_channels : int
            number of channels in the output image
        Returns
        -------
        torch.Tensor
        """
        super().__init__()

        self.skip_connection = nn.Conv3d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=(1, 1, 1),
                                         bias=False)

        self.basic_block = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.LeakyReLU(),
            spectral_norm(nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding='same', bias=False)),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(),
            spectral_norm(nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding='same', bias=False))
        )

    def forward(self, x):
        return  self.basic_block(x) + self.skip_connection(x)


class Generator(nn.Module):
    def __init__(self, noise_channels, type='wasserstein'):
        super().__init__()

        self.linear_mapping = nn.Sequential(
            spectral_norm(nn.Linear(noise_channels, 512 * 4 * 4 * 4, bias=False)),
            nn.LeakyReLU()
        )

        self.upsample_layer = nn.Upsample(scale_factor=2)
        self.block1 = PreActResBlock(512, 256) # 4x4x4 => 8x8x8
        self.block2 = PreActResBlock(256, 128) # 8x8x8 => 16x16x16
        self.block3 = PreActResBlock(128, 64) # 16x16x16 => 32x32x32
        self.block4 = PreActResBlock(64, 32) # 32x32x32 => 64x64x64

        self.final_layer = nn.Sequential(
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            spectral_norm(nn.Conv3d(32, 1, kernel_size=(1, 1, 1), bias=False)),
            nn.Sigmoid()
        )


    def forward(self, x):
        features = self.linear_mapping(x)
        features = features.view(-1, 512, 4, 4, 4)

        features = self.block1(features)
        features = self.upsample_layer(features)

        features = self.block2(features)
        features = self.upsample_layer(features)

        features = self.block3(features)
        features = self.upsample_layer(features)

        features = self.block4(features)
        features = self.upsample_layer(features)

        return self.final_layer(features)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_layer = nn.Sequential(
            nn.BatchNorm3d(1),
            nn.LeakyReLU(),
            spectral_norm(nn.Conv3d(1, 32, kernel_size=(1, 1, 1), bias=False))
        )

        self.downsample = nn.AvgPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.linear_mapping = nn.Sequential(
            spectral_norm(nn.Linear(in_features=512*4*4*4, out_features=1)),
            # nn.Sigmoid()
        )

        self.block1 = PreActResBlock(32, 64)
        self.block2 = PreActResBlock(64, 128)
        self.block3 = PreActResBlock(128, 256)
        self.block4 = PreActResBlock(256, 512)

    def forward(self, x):
        features = self.first_layer(x)

        features = self.block1(features)
        features = self.downsample(features)

        features = self.block2(features)
        features = self.downsample(features)

        features = self.block3(features)
        features = self.downsample(features)

        features = self.block4(features)
        features = self.downsample(features)

        features = features.view(-1, 512*4*4*4)
        scores = self.linear_mapping(features).view(-1)

        return scores
