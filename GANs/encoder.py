import torch
import torch.nn as nn


class BuildingBlock(nn.Module):
    def __init__(self, in_channels=1, out_channels=32):
        super(BuildingBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding='same',
                      bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, input_size, latent_features):
        """
        Parameters
        ----------
        input_size : int
            size of input 3D cube
        latent_features : int
            Number of features in latent space

        Returns
        -------
        features: torch.Tensor
            Feature tensor
        """

        super(Encoder, self).__init__()


        self.input_block = BuildingBlock(1, 32)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.bottleneck1 = nn.Sequential(
            BuildingBlock(32, 128),
            BuildingBlock(128, 128),
            BuildingBlock(128, 128),
            BuildingBlock(128, 128)
        )

        self.bottleneck2 = nn.Sequential(
            BuildingBlock(160, 256),
            BuildingBlock(256, 256),
            BuildingBlock(256, 256),
            BuildingBlock(256, 256),
        )

        self.bottleneck3 = nn.Sequential(
            BuildingBlock(416, 512),
            BuildingBlock(512, 512),
            BuildingBlock(512, 512),
            BuildingBlock(512, 512),
        )

        self.final_layer = nn.Sequential(
            nn.Conv2d(928,
                      latent_features,
                      kernel_size=(1, 1),
                      stride=(1, 1),
                      padding='same',
                      bias=False),
            nn.BatchNorm2d(latent_features),
            nn.AvgPool2d(kernel_size=(input_size // 8, input_size // 8))
        )

    def forward(self, x):
        features = self.input_block(x)
        skip_1 = features

        features = self.bottleneck1(features)
        features = torch.cat([features, skip_1], dim=1)
        features = self.maxpool(features)

        skip_2 = features

        features = self.bottleneck2(features)
        features = torch.cat([features, skip_2], dim=1)
        features = self.maxpool(features)

        skip_3 = features

        features = self.bottleneck3(features)
        features = torch.cat([features, skip_3], dim=1)
        features = self.maxpool(features)

        features = self.final_layer(features)
        features = features.flatten(1)

        return features
