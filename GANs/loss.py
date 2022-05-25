import torch
import torch.nn as nn


class GanLoss(nn.Module):
    """
    GAN loss calculator

    Variants:
      - standard
      - wasserstein
    """
    def __init__(self, loss_type):
        super(GanLoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, fake_scores, real_scores=None, fake_slice=None, target_slice=None):
        if real_scores is None:
            # Generator loss
            if self.loss_type == 'standard':
                loss = nn.MSELoss()(fake_slice, target_slice)
                # loss = nn.BCEWithLogitsLoss()(fake_scores, torch.ones_like(fake_scores))
            elif self.loss_type == 'wasserstein':
                loss = nn.MSELoss()(fake_slice, target_slice)

        else:
            # Discriminator loss
            if self.loss_type == 'standard':
                loss = nn.BCEWithLogitsLoss()(real_scores, torch.ones_like(real_scores)) + nn.BCEWithLogitsLoss()(fake_scores, torch.zeros_like(fake_scores))
            elif self.loss_type == 'wasserstein':
                loss = -real_scores.mean() + fake_scores.mean()

        return loss

    def _gradient_penalty(self, critic, real, fake, device='cpu'):
        BATCH_SIZE, C, H, W = real.shape
        alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
        interpolated_images = real * alpha + fake * (1 - alpha)

        # Calculate critic scores
        mixed_scores = critic(interpolated_images)

        # Take the gradient of the scores with respect to the images
        gradient = torch.autograd.grad(
            inputs=interpolated_images,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
        return gradient_penalty
