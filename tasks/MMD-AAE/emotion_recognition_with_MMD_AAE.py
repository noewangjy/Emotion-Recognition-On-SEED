import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import torch
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.loggers
from src.models.MMD_AAE import MMD_AAE_model
from src.utils.dataset import SeedDatasetForMMDAAE
from torch.utils.data import DataLoader


@hydra.main(config_path='conf', config_name='config')
def run(cfg: DictConfig):
    torch.manual_seed(cfg.basic.seed)
    np.random.seed(cfg.basic.seed)

    cfg.basic.data_path = to_absolute_path(cfg.basic.data_path)
    train_set = SeedDatasetForMMDAAE(
        data_path=cfg.basic.data_path,
        target_subject=cfg.basic.target_sub,
        mode='train'
    )
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=cfg.train.workers,
        drop_last=False
    )

    dev_set = SeedDatasetForMMDAAE(
        data_path=cfg.basic.data_path,
        target_subject=cfg.basic.target_sub,
        mode='dev'
    )
    dev_loader = DataLoader(
        dev_set,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=cfg.train.workers,
        drop_last=False
    )

    logger = pl.loggers.TensorBoardLogger(save_dir=cfg.basic.log_dir)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints",
        filename="{epoch}--{step}--{val_epoch_accuracy:.6f}",
        monitor="val_epoch_accuracy",
        save_last=True,
        save_top_k=3,
        mode="max",
        save_weights_only=True,
        save_on_train_epoch_end=True
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=cfg.train.num_epochs,
        callbacks=[checkpoint_callback,
                   pl.callbacks.TQDMProgressBar(refresh_rate=1)],
        enable_checkpointing=True,
        num_sanity_val_steps=0,
        logger=logger
    )
    model = MMD_AAE_model(cfg=cfg)
    trainer.fit(model, train_loader, dev_loader)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    run()





