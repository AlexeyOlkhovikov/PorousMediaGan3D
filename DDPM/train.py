import torch
from torch.utils.data import DataLoader
from torchvision import utils
import pytorch_lightning as pl
import numpy as np


class Diffusion(pl.LightningModule):
    def __init__(
            self,
            train_data_path: str,
            valid_data_path: str,
            batch_size: int = 8,
            img_size: int = 64,
            lr: float = 1e-04,
            timesteps: int = 1000,
            loss_type: str = 'l1', # or "l2"
            diffusion_type: str = "2D", # or "3D"
    ):
        super(Diffusion, self).__init__()

        self.lr = lr
        self.diffusion_type = diffusion_type

        self.batch_size = batch_size

        if diffusion_type == "2D":
            from .diffusion2D import Unet, GaussianDiffusion, DiffusionDataset
            self.train_dataset = DiffusionDataset(train_data_path)
            self.valid_dataset = DiffusionDataset(valid_data_path)

            m = Unet(
                dim = img_size,
                channels = 1,
                dim_mults = [1, 2, 4, 8]
            )
            self.model = GaussianDiffusion(
                m,
                channels = 1,
                image_size = img_size,
                timesteps = timesteps,
                loss_type = loss_type
            )
        else:
            from .diffusion3D import Unet, GaussianDiffusion, DiffusionDataset

            self.train_dataset = DiffusionDataset(train_data_path)
            self.valid_dataset = DiffusionDataset(valid_data_path)

            m = Unet(
                img_size,
                dim_mults = (1, 2, 4, 8)
            )
            self.model = GaussianDiffusion(
                m,
                image_size = img_size,
                timesteps = timesteps,
                loss_type = loss_type
            )

    def training_step(self, batch, batch_idx):
        imgs = batch
        loss = self.model(imgs)

        assert loss.shape == ()

        self.log('loss_diffusion_train', loss, prog_bar=True)

        return {'loss': loss}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        imgs = batch
        loss = self.model(imgs)
        self.log('loss_diffusion_validation', loss, prog_bar=True)
        fake_imgs = self.model.sample(batch_size=8)

        return {'fake_images': fake_imgs}

    @torch.no_grad()
    def validation_epoch_end(self, outputs):
        fake_imgs = torch.cat([x['fake_images'].cpu() for x in outputs], dim=0)
        fake_imgs = fake_imgs * 0.12 + 0.52

        if self.diffusion_type == "2D":
            grid_generated = utils.make_grid(fake_imgs[:, :, :, :], nrow=8)
        else:
            grid_generated = utils.make_grid(fake_imgs[:, :, 32, :, :], nrow=8)

        self.logger.experiment.add_image('generated_images_validation', grid_generated, self.current_epoch)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.0, 0.999))
        return [opt], []

    def train_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=8, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, num_workers=8, batch_size=1, shuffle=False)




