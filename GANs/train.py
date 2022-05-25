import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision import utils
import numpy as np
import timm

from .encoder import Encoder
from .loss import GanLoss
from .utils import GanDataset


class GAN(pl.LightningModule):
    def __init__(
            self,
            train_data_path: str,
            valid_data_path: str,
            gan_type: str = 'dcgan',
            loss_type: str = 'standard',
            img_size: int = 64,
            batch_size: int = 16,
            lr_gen = 1e-04,
            lr_dis = 1e-04,
            transforms: object = None
    ):
        super(GAN, self).__init__()

        self.gan_type = gan_type

        self.noise_channels = 128
        self.latent_dim_channels = 128

        self.lr_gen = lr_gen
        self.lr_dis = lr_dis

        self.loss = GanLoss(loss_type)

        self.train_dataset = GanDataset(np.load(train_data_path), transforms)
        self.valid_dataset = GanDataset(np.load(valid_data_path), None)

        self.batch_size = batch_size

#         self.encoder = Encoder(img_size, self.latent_dim_channels)
        self.encoder = timm.create_model('efficientnet_b3', in_chans=1, pretrained=True)
        self.encoder.conv_head = nn.Conv2d(384, self.latent_dim_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.encoder.bn2 = nn.BatchNorm2d(self.latent_dim_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.encoder.classifier = nn.Identity()

        if gan_type == 'dcgan':
            from .DCGAN import Generator, Discriminator
            self.gen = Generator(1024,  self.noise_channels, self.latent_dim_channels, loss_type)
            self.dis = Discriminator(16)

        elif gan_type == 'biggan':
            from .BIGGAN import Generator, Discriminator
            self.gen = Generator(512, self.latent_dim_channels, self.noise_channels)
            self.dis = Discriminator(16)

    def forward(self, noise):
        return self.gen(noise)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, slices = batch

        noise = torch.randn(len(slices), self.noise_channels).to(imgs.device)
        latent_slice_features = self.encoder(slices)


        fake_images = self.gen(noise, latent_slice_features)
        real_preds = self.dis(imgs)
        fake_preds = self.dis(fake_images)

        if optimizer_idx == 0:
            loss = self.loss(fake_preds, None, fake_images[:, :, 32, :, :], slices)
        elif optimizer_idx == 1:
            loss = self.loss(fake_preds, real_preds, None, None)

        assert loss.shape == ()

        loss_name = 'loss_gen' if optimizer_idx == 0 else 'loss_dis'
        self.log(loss_name, loss, prog_bar=True)

        return {'loss': loss, 'slices': slices, 'fake_images': fake_images}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        imgs, slices = batch

        noise = torch.randn(len(slices), self.noise_channels).to(imgs.device)
        latent_slice_features = self.encoder(slices)

        fake_images = self.gen(noise, latent_slice_features)

        return {'slices': slices, 'fake_images': fake_images}

    @torch.no_grad()
    def training_epoch_end(self, outputs):
        slices = []
        fake_images = []

        for i in range(len(outputs)):
            slices.extend([x['slices'].cpu() for x in outputs[i]])
            fake_images.extend([x['fake_images'].cpu() for x in outputs[i]])

        slices = torch.cat(slices, dim=0)
        fake_images = torch.cat(fake_images, dim=0)

        slices = slices * 0.12 + 0.52
        fake_images = fake_images * 0.12 + 0.52

        grid_original = utils.make_grid(slices, nrow=8)
        grid_generated = utils.make_grid(fake_images[:, :, 32, :, :], nrow=8)

        self.logger.experiment.add_image('generated_images_train', grid_generated, self.current_epoch)
        self.logger.experiment.add_image('original_images_train', grid_original, self.current_epoch)

    @torch.no_grad()
    def validation_epoch_end(self, outputs):
        slices = torch.cat([x['slices'].cpu() for x in outputs], dim=0)
        fake_images = torch.cat([x['fake_images'].cpu() for x in outputs], dim=0)

        slices = slices * 0.12 + 0.52
        fake_images = fake_images * 0.12 + 0.52

        grid_original = utils.make_grid(slices, nrow=8)
        grid_generated = utils.make_grid(fake_images[:, :, 32, :, :], nrow=8)

        self.logger.experiment.add_image('generated_images_validation', grid_generated, self.current_epoch)
        self.logger.experiment.add_image('original_images_validation', grid_original, self.current_epoch)

    def configure_optimizers(self):
        opt_gen = torch.optim.Adam(self.gen.parameters(), lr=self.lr_gen, betas=(0.0, 0.999))
        opt_dis = torch.optim.Adam(self.dis.parameters(), lr=self.lr_dis, betas=(0.0, 0.9999))
        return [opt_gen, opt_dis], []

    def train_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=8, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, num_workers=8, batch_size=2*self.batch_size, shuffle=False)




