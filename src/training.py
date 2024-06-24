import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from src.config import CFG
from src.atrium_segmentation_model import AtriumSegmentation
from src.dataloader import CardiacDataset


class SegmentationTrainer:
    def __init__(self):
        self.device = CFG.device

        self.batch_size = CFG.batch_size
        self.num_workers = CFG.num_workers
        self.max_epochs = CFG.max_epochs
        self.max_k = CFG.max_k

        self.train_data_dir = CFG.training_data_dir
        self.val_data_dir = CFG.validation_data_dir
        self.logs_path = CFG.logs_path

        self.model = AtriumSegmentation()

    def train(self):
        train_dataset, val_dataset = self.setup_datasets()
        train_loader, val_loader = self.setup_dataloaders(train_dataset, val_dataset)
        checkpoint_callback = self.initialize_checkpoint_callback()
        logger = self.initialize_logger()
        trainer = self.initialize_trainer(checkpoint_callback, logger)
        trainer.fit(self.model, train_loader, val_loader)

    def setup_datasets(self):
        train_dataset = CardiacDataset(root=self.train_data_dir, apply_augmentation=True)
        val_dataset = CardiacDataset(root=self.val_data_dir, apply_augmentation=False)
        return train_dataset, val_dataset

    def setup_dataloaders(self, train_dataset, val_dataset):
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size,
                                                   num_workers=self.num_workers, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size,
                                                 num_workers=self.num_workers, shuffle=False)
        return train_loader, val_loader

    def initialize_checkpoint_callback(self):
        checkpoint_callback = ModelCheckpoint(monitor='Validation loss:',
                                              save_top_k=self.max_k, mode='min')
        return checkpoint_callback

    def initialize_logger(self):
        logger = TensorBoardLogger(save_dir=self.logs_path)
        return logger

    def initialize_trainer(self, checkpoint_callback, logger):
        trainer = pl.Trainer(
            accelerator=self.device,
            devices=1 if self.device == 'cuda' else 0,
            logger=logger,
            log_every_n_steps=1,
            callbacks=[checkpoint_callback],
            max_epochs=self.max_epochs
        )
        return trainer


if __name__ == "__main__":
    segmentation_trainer = SegmentationTrainer()
    segmentation_trainer.train()
