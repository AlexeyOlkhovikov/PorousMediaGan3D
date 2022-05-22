import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from tqdm.auto import trange

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Generator(nn.Module):

    def __init__(self, n_inputs, n_outputs):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(n_inputs, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, n_outputs))

    def forward(self, z, y):
        zy = torch.cat((z, y), dim=1)
        return self.net(zy)


class Discriminator(nn.Module):

    def __init__(self, n_inputs):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(n_inputs, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1))

    def forward(self, x, y):
        xy = torch.cat((x, y), dim=1)
        return self.net(xy)


############################################
#   Gradient Penalty Model
############################################
class GeneratorGp(nn.Module):
    def __init__(self, noise_channels, img_channels, features):
        super(GeneratorGp, self).__init__()
        self.net = nn.Sequential(
            self._block(noise_channels, features * 16, 4, 1, 0),  # 1x1 --> 4x4
            self._block(features * 16, features * 8, 4, 2, 1),  # 4x4 --> 8x8
            self._block(features * 8, features * 4, 4, 2, 1),  # 8x8 --> 16x16
            self._block(features * 4, features * 2, 4, 2, 1),  # 16x16 --> 32x32
            nn.ConvTranspose2d(features * 2, img_channels, 4, 2, 1),  # 32x32 --> 64x64
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


# Critic Model Class
class Critic(nn.Module):
    def __init__(self, img_channels, features):
        super(Critic, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(img_channels, features, 4, 2, 1),  # n --> (n-k+s+2p)/s: 64x64 --> 32x32
            nn.LeakyReLU(0.2),  # 32x32 --> 32x32
            self._block(features, features * 2, 4, 2, 1),  # 32x32 --> 16x16
            self._block(features * 2, features * 4, 4, 2, 1),  # 16x16 --> 8x8
            self._block(features * 4, features * 8, 4, 2, 1),  # 4 x 4
            nn.Conv2d(features * 8, 1, 4, 2, 0),  # 1 x 1
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)


class Fitter(object):

    def __init__(self, generator, discriminator, batch_size=32, n_epochs=10, latent_dim=1, lr=0.0001, n_critic=5):

        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.latent_dim = latent_dim
        self.lr = lr
        self.n_critic = n_critic

        self.opt_gen = torch.optim.RMSprop(self.generator.parameters(), lr=self.lr)
        self.opt_disc = torch.optim.RMSprop(self.discriminator.parameters(), lr=self.lr)

        self.generator.to(DEVICE)
        self.discriminator.to(DEVICE)

    def fit(self, X, y):

        # numpy to tensor
        X_real = torch.tensor(X, dtype=torch.float, device=DEVICE)
        y_cond = torch.tensor(y, dtype=torch.float, device=DEVICE)

        # tensor to dataset
        dataset_real = TensorDataset(X_real, y_cond)

        # Turn on training
        self.generator.train(True)
        self.discriminator.train(True)

        self.loss_history = []

        # Fit GAN
        tbar = trange(self.n_epochs, leave=True, desc='?')
        for epoch in tbar:
            for i, (real_batch, cond_batch) in enumerate(
                    DataLoader(dataset_real, batch_size=self.batch_size, shuffle=True)):

                # generate a batch of fake observations
                z_noise = torch.normal(0, 1, (len(real_batch), self.latent_dim)).to(DEVICE)
                fake_batch = self.generator(z_noise, cond_batch)

                # Discriminator
                loss_disc = -torch.mean(self.discriminator(real_batch, cond_batch)) + torch.mean(
                    self.discriminator(fake_batch, cond_batch))

                # optimization step
                self.opt_disc.zero_grad()
                loss_disc.backward(retain_graph=True)
                self.opt_disc.step()

                # Clip weights of discriminator
                for p in self.discriminator.parameters():
                    p.data.clamp_(-0.01, 0.01)

                # Generator
                if i % self.n_critic == 0:
                    # measures generator's ability to fool the discriminator
                    loss_gen = torch.mean(self.discriminator(real_batch, cond_batch)) - torch.mean(
                        self.discriminator(fake_batch, cond_batch))

                    # optimization step
                    self.opt_gen.zero_grad()
                    loss_gen.backward()
                    self.opt_gen.step()

            # calculate and store loss after an epoch
            Z_noise = torch.normal(0, 1, (len(X_real), self.latent_dim)).to(DEVICE)
            X_fake = self.generator(Z_noise, y_cond)
            loss_epoch = torch.mean(self.discriminator(X_real, y_cond)) - torch.mean(self.discriminator(X_fake, y_cond))
            tbar.set_description(f'{loss_epoch.detach().cpu():.3f}')
            tbar.refresh()
            self.loss_history.append(loss_epoch.detach().cpu())

        # Turn off training
        self.generator.train(False)
        self.discriminator.train(False)

    def predict(self, y):
        return generate(self.generator, y, self.latent_dim)


def generate(generator, y, latent_dim):
    Z = torch.normal(0, 1, (len(y), latent_dim)).to(DEVICE)
    y = torch.tensor(y, dtype=torch.float, device=DEVICE)
    X_fake = generator(Z, y).cpu().detach().numpy()
    return X_fake
