import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class AdaptiveBatchNorm(nn.Module):
    def __init__(self, noise_dim, embed_features, num_features):
        super(AdaptiveBatchNorm, self).__init__()
        self.batch_norm = nn.BatchNorm3d(num_features=num_features, affine=False)
        self.gamma = nn.Linear(embed_features+noise_dim, num_features)
        self.bias = nn.Linear(embed_features+noise_dim, num_features)
    
    def forward(self, g_feautures, noise, embed_features):
        features = torch.cat([noise, embed_features], dim=1)

        gamma = self.gamma(features) # output num_features
        bias = self.bias(features) # output num_features

        outputs = self.batch_norm(g_feautures)

        return outputs * gamma[..., None, None, None] + bias[..., None, None, None]


class ResBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, latent_dim, noise_dim):
        super(ResBlockUp, self).__init__()

        self.skip_layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            spectral_norm(nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                    bias=False))
        )

        self.base_layer = nn.ModuleList([
            AdaptiveBatchNorm(noise_dim=noise_dim, embed_features=latent_dim, num_features=in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            spectral_norm(nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1,
                                    bias=False, padding='same')),
            AdaptiveBatchNorm(noise_dim=noise_dim, embed_features=latent_dim, num_features=in_channels),
            nn.ReLU(),
            spectral_norm(nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                    bias=False, padding='same'))
        ])

    def forward(self, x, noise, embeddings):
        skip_features = self.skip_layer(x)

        for i, layer in enumerate(self.base_layer):
            if (i == 0) or (i == 4):
                x = layer(x, noise, embeddings)
            else:
                x = layer(x)

        return x + skip_features



class ResBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlockDown, self).__init__()

        self.skip_layer = nn.Sequential(
            spectral_norm(nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                    bias=False, padding='same')),
            nn.AvgPool3d(kernel_size=3, stride=2, padding=1)
        )

        self.base_layer = nn.Sequential(
            nn.ReLU(),
            spectral_norm(nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1,
                                    bias=False, padding='same')),
            nn.ReLU(),
            spectral_norm(nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                    bias=False, padding='same')),
            nn.AvgPool3d(kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        skip_features = self.skip_layer(x)
        x = self.base_layer(x)

        return skip_features + x


class Generator(nn.Module):
    def __init__(self, in_channels, latent_dim, noise_dim):
        super(Generator, self).__init__()
        self.linear_mapping = nn.Linear(noise_dim, in_channels*4*4*4)

        self.resblock1 = ResBlockUp(in_channels=in_channels, out_channels=in_channels // 2, latent_dim=latent_dim, noise_dim=noise_dim) # 8x8x8
        self.resblock2 = ResBlockUp(in_channels=in_channels // 2, out_channels=in_channels // 4, latent_dim=latent_dim, noise_dim=noise_dim) # 16x16x16
        self.resblock3 = ResBlockUp(in_channels=in_channels // 4, out_channels=in_channels // 8, latent_dim=latent_dim, noise_dim=noise_dim) # 32x32x32
        self.resblock4 = ResBlockUp(in_channels=in_channels // 8, out_channels=1, latent_dim=latent_dim, noise_dim=noise_dim) # 64x64x64

        self.in_channels = in_channels

    def forward(self, noise, embeddings):
        x = self.linear_mapping(noise).view(-1, self.in_channels, 4, 4, 4)
        x = self.resblock1(x, noise, embeddings)
        x = self.resblock2(x, noise, embeddings)
        x = self.resblock3(x, noise, embeddings)
        img = self.resblock4(x, noise, embeddings)

        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        pass

    def forward(self):
        pass