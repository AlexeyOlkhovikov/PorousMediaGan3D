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
            elif self.loss_type == 'wasserstein':
                loss = nn.MSELoss()(fake_slice, target_slice)

        else:
            # Discriminator loss
            if self.loss_type == 'standard':
                loss = nn.BCEWithLogitsLoss()(real_scores, torch.ones_like(real_scores)) + nn.BCEWithLogitsLoss()(fake_scores, torch.zeros_like(fake_scores))
            elif self.loss_type == 'wasserstein':
                loss = -real_scores.mean() + fake_scores.mean()

        return loss