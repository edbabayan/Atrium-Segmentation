import torch
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from pytorch_lightning.utilities.types import OptimizerLRScheduler

from src.config import CFG
from src.dice_loss import DiceLoss
from src.segmentation_model import UNet


class AtriumSegmentation(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = UNet()
        self.loss = DiceLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=CFG.lr)

    def forward(self, x):
        return torch.sigmoid(self.model(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_true = y.float()
        y_pred = self.model(x)

        loss = self.loss(y_pred, y_true)

        self.log('Training loss:', loss)
        if batch_idx % 50 == 0:
            self.log_images(x.cpu(), y_pred.cpu(), y_true.cpu(), 'Train')
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_true = y.float()
        y_pred = self.model(x)

        loss = self.loss(y_pred, y_true)

        self.log('Validation loss:', loss)
        if batch_idx % 2 == 0:
            self.log_images(x.cpu(), y_pred.cpu(), y_true.cpu(), 'Validation')
        return loss

    def log_images(self, mri, y_pred, y_true, mode):
        pred = y_pred > 0.5
        fig, axes = plt.subplots(1, 2)

        axes[0].imshow(mri[0, 0], cmap='bone')
        mask_ = np.ma.masked_where(y_true[0, 0] == 0, y_true[0, 0])
        axes[0].imshow(mask_, alpha=0.6)

        axes[1].imshow(mri[0, 0], cmap='bone')
        mask_ = np.ma.masked_where(pred[0, 0] == 0, pred[0, 0])
        axes[1].imshow(mask_, alpha=0.6)
        self.logger.experiment.add_figure(f'{mode} example', fig, self.global_step)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return [self.optimizer]
