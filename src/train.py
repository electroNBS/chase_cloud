# src/train.py

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
import os
# Import from our source files
from .config import *
from .data_loader import INSATCloudDataset
from .model import ConditionalUNet, CloudDiffusion

# We need to import the scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

class CloudChaseModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()  # Saves all __init__ args to self.hparams

        self.num_channels = len(CHANNELS)
        self.condition_channels = self.num_channels * INPUT_FRAMES

        self.unet = ConditionalUNet(
            image_size=IMAGE_SIZE,
            in_channels=self.num_channels,  # The noisy image has `num_channels`
            out_channels=self.num_channels,  # The predicted noise also has `num_channels`
            condition_channels=self.condition_channels,
        )

        self.diffusion_model = CloudDiffusion(
            self.unet, timesteps=TIMESTEPS, beta_start=BETA_START, beta_end=BETA_END
        )

    def forward(self, condition, target):
        return self.diffusion_model(target, condition)

    def training_step(self, batch, batch_idx):
        condition, target = batch
        loss = self(condition, target)
        # We log the loss so the scheduler can monitor it
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=LEARNING_RATE)

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',      # Reduce LR when the monitored quantity has stopped decreasing
            factor=0.2,      # New LR = LR * factor (e.g., 0.0001 -> 0.00002)
            patience=5,      # Number of epochs with no improvement after which LR will be reduced
            verbose=True     # Print a message when the LR is updated
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss", # The metric to watch
            },
        }

    def train_dataloader(self):
        dataset = INSATCloudDataset(
            root_dir=DATA_PATH,
            channels=CHANNELS,
            input_frames=INPUT_FRAMES,
            future_frames=FUTURE_FRAMES,
            image_size=IMAGE_SIZE,
        )
        return DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
        )


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    model = CloudChaseModule()

    # Callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH,
        filename="cloud-chase-best-{epoch:02d}-{train_loss:.2f}",
        save_top_k=1,
        verbose=True,
        monitor="train_loss",
        mode="min",
        save_last=True
    )

    resume_checkpoint_path = os.path.join(CHECKPOINT_PATH, "cloud-chase-best-epoch=25-train_loss=0.47.ckpt")
    
    # Check if the file actually exists before trying to resume
    if not os.path.exists(resume_checkpoint_path):
        resume_checkpoint_path = None # Start from scratch if it's not found
        print("Checkpoint not found. Starting training from scratch.")
    else:
        print(f"Attempting to resume training from: {resume_checkpoint_path}")

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback],
        resume_from_checkpoint=resume_checkpoint_path
    )

    trainer.fit(model)
