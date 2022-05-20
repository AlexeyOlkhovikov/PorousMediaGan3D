import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class AdaptiveBatchNorm(nn.Module):
    def __init__(self, noise_dim, embed_features, num_features):
        super(AdaptiveBatchNorm, self).__init__()
        self.batch_norm = nn.BatchNorm3d(num_features=num_features, affine=False)
        self.gamma = nn.Linear(embed_features + noise_dim, num_features)
        self.bias = nn.Linear(embed_features + noise_dim, num_features)

    def forward(self, g_feautures, noise, embed_features):
        features = torch.cat([noise, embed_features], dim=1)

        gamma = self.gamma(features)  # output num_features
        bias = self.bias(features)  # output num_features

        outputs = self.batch_norm(g_feautures)

        return outputs * gamma[..., None, None, None] + bias[..., None, None, None]


class PreActResBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, noise_dim, embed_features):
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

        self.skip_connection = nn.Sequential(spectral_norm(nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                                                       kernel_size=(1, 1, 1), bias=False)),
                                              nn.BatchNorm3d(out_channels),
                                              nn.LeakyReLU()
                                              )

        self.basic_block = nn.ModuleList([
            AdaptiveBatchNorm(noise_dim, embed_features, in_channels),
            nn.LeakyReLU(),
            spectral_norm(nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding='same', bias=False)),
            AdaptiveBatchNorm(noise_dim, embed_features, out_channels),
            nn.LeakyReLU(),
            spectral_norm(nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding='same', bias=False))
        ])

    def forward(self, x, noise, embeddings):
        skip_x = self.skip_connection(x)

        for i, layer in enumerate(self.basic_block):
            if (i == 0) or (i == 3):
                x = layer(x, noise, embeddings)
            else:
                x = layer(x)

        return x + skip_x


class PreActResBlockDown(nn.Module):
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

        self.skip_connection = nn.Sequential(spectral_norm(nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                                                       kernel_size=(1, 1, 1), bias=False)),
                                              nn.BatchNorm3d(out_channels),
                                              nn.LeakyReLU()
                                              )

        self.basic_block = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.LeakyReLU(),
            spectral_norm(nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding='same', bias=False)),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(),
            spectral_norm(nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding='same', bias=False))
        )

    def forward(self, x):
        skip_x = self.skip_connection(x)
        x = self.basic_block(x)

        return x + skip_x

class Generator(nn.Module):
    def __init__(self, in_channels, noise_channels, embed_channels, type='wasserstein'):
        super().__init__()

        self.linear_mapping = nn.Sequential(
            spectral_norm(nn.Linear(noise_channels, in_channels * 4 * 4 * 4, bias=False)),
            nn.LeakyReLU()
        )

        self.in_channels = in_channels

        self.upsample_layer = nn.Upsample(scale_factor=2)
        self.block1 = PreActResBlockUp(in_channels, in_channels // 2, noise_channels, embed_channels) # 4x4x4 => 8x8x8
        self.block2 = PreActResBlockUp(in_channels // 2, in_channels // 4, noise_channels, embed_channels) # 8x8x8 => 16x16x16
        self.block3 = PreActResBlockUp(in_channels // 4, in_channels // 8, noise_channels, embed_channels) # 16x16x16 => 32x32x32
        self.block4 = PreActResBlockUp(in_channels // 8, in_channels // 16, noise_channels, embed_channels) # 32x32x32 => 64x64x64

        self.final_layer = nn.Sequential(
            nn.BatchNorm3d(in_channels // 16),
            nn.LeakyReLU(),
            spectral_norm(nn.Conv3d(in_channels // 16, 1, kernel_size=(1, 1, 1), bias=False)),
            nn.Tanh(),
            # nn.Sigmoid()
        )


    def forward(self, noise, embed):
        features = self.linear_mapping(noise)
        features = features.view(-1, self.in_channels, 4, 4, 4)

        features = self.block1(features, noise, embed)
        features = self.upsample_layer(features)

        features = self.block2(features, noise, embed)
        features = self.upsample_layer(features)

        features = self.block3(features, noise, embed)
        features = self.upsample_layer(features)

        features = self.block4(features, noise, embed)
        features = self.upsample_layer(features)

        return self.final_layer(features)


class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.first_layer = nn.Sequential(
            nn.BatchNorm3d(1),
            nn.LeakyReLU(),
            spectral_norm(nn.Conv3d(1, in_channels, kernel_size=(1, 1, 1), bias=False))
        )

        self.in_channels = in_channels

        self.downsample = nn.AvgPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.linear_mapping = nn.Sequential(
            spectral_norm(nn.Linear(in_features=16*in_channels*4*4*4, out_features=1)),
        )

        self.block1 = PreActResBlockDown(in_channels, 2 * in_channels)
        self.block2 = PreActResBlockDown(2 * in_channels, 4 * in_channels)
        self.block3 = PreActResBlockDown(4 * in_channels, 8 * in_channels)
        self.block4 = PreActResBlockDown(8 * in_channels, 16 * in_channels)

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
        
        features = features.view(-1, 16*self.in_channels*4*4*4)
        scores = self.linear_mapping(features).view(-1)

        return scores
