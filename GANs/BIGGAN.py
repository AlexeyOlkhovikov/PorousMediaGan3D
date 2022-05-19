import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import math


class NonLocalBlock(nn.Module):
    def __init__(self, channels):
        super(NonLocalBlock, self).__init__()
        self.inter_channel = channels
        self.conv_phi = nn.Conv3d(in_channels=channels, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                  padding=0, bias=False)
        self.conv_theta = nn.Conv3d(in_channels=channels, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                    padding=0, bias=False)
        self.conv_g = nn.Conv3d(in_channels=channels, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv3d(in_channels=self.inter_channel, out_channels=channels, kernel_size=1, stride=1,
                                   padding=0, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.scale = math.sqrt(self.inter_channel)

    def forward(self, x):
        residual = x

        phi = self.conv_phi(x)
        print('phi', phi.size())

        theta = self.conv_theta(x)

        print('theta', theta.size())

        val_g = self.conv_g(x)

        print('g', val_g.size())

        phi = phi.view(phi.size(0), phi.size(1), -1)

        print('phi', phi.size())

        theta = theta.view(theta.size(0), theta.size(1), -1)
        print('theta', theta.size())

        val_g = val_g.view(val_g.size(0), val_g.size(1), -1)

        print('g', val_g.size())

        sim_map = torch.bmm(phi.transpose(1, 2), theta)

        print('sim map', sim_map.size())
        sim_map = sim_map / self.scale
        #   sim_map = sim_map / self.temperature

        sim_map = self.softmax(sim_map)

        out_sim = torch.bmm(sim_map, val_g.transpose(1, 2))
        print('out_sim', out_sim.size())

        out_sim = out_sim.transpose(1, 2)

        out_sim = out_sim.view(out_sim.size(0), out_sim.size(1), *x.size()[2:])
        out_sim = self.conv_mask(out_sim)

        out_sim = self.gamma * out_sim

        out = out_sim + residual

        return out


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
        self.linear_mapping = nn.Linear(noise_dim, in_channels * 4 * 4 * 4)
        self.non_local_block = NonLocalBlock(channels=in_channels // 8)
        self.resblock1 = ResBlockUp(in_channels=in_channels, out_channels=in_channels // 2, latent_dim=latent_dim,
                                    noise_dim=noise_dim)  # 8x8x8
        self.resblock2 = ResBlockUp(in_channels=in_channels // 2, out_channels=in_channels // 4, latent_dim=latent_dim,
                                    noise_dim=noise_dim)  # 16x16x16
        self.resblock3 = ResBlockUp(in_channels=in_channels // 4, out_channels=in_channels // 8, latent_dim=latent_dim,
                                    noise_dim=noise_dim)  # 32x32x32
        self.resblock4 = ResBlockUp(in_channels=in_channels // 8, out_channels=1, latent_dim=latent_dim,
                                    noise_dim=noise_dim)  # 64x64x64

        self.final_conv = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, )
        self.in_channels = in_channels
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.bnorm = nn.BatchNorm3d(1)

    def forward(self, noise, embeddings):
        x = self.linear_mapping(noise).view(-1, self.in_channels, 4, 4, 4)
        x = self.resblock1(x, noise, embeddings)
        x = self.resblock2(x, noise, embeddings)
        x = self.resblock3(x, noise, embeddings)
        x = self.non_local_block(x)
        x = self.resblock4(x, noise, embeddings)

        x = self.bnorm(self.relu(self.final_conv(x)))
        img = self.tanh(x)

        return img


class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Discriminator, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.relu = nn.ReLU()

        self.non_local_block = NonLocalBlock(channel=self.in_channels * 2)

        self.resblock1 = ResBlockUp(in_channels=self.in_channels, out_channels=self.in_channels * 2)
        self.resblock2 = ResBlockUp(in_channels=self.in_channels * 2, out_channels=self.in_channels * 4)
        self.resblock3 = ResBlockUp(in_channels=self.in_channels * 4, out_channels=self.in_channels * 8)
        self.resblock4 = ResBlockUp(in_channels=self.in_channels * 8, out_channels=self.in_channels * 8)

        #  self.pool = nn.AvgPool3d(in_channels * 8)
        self.out_ = spectral_norm(nn.Linear(self.out_channels, 1))

    def forward(self, x, labels):
        x = self.resblock1(x)
        x = self.non_local_block(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)

        res = torch.sum(self.relu(x, inplace=True), dim=(-1, -2))
        scores = self.out_(res).squeeze(dim=1)
        if self.use_projection_head:
            scores += torch.diag(torch.inner(res, self.embedding_(labels)))

        return scores