from typing import List, Dict, Callable
import torch
from torch import Tensor as T
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from src.utils.dataset import DANSample
from src.utils.loss_function import mmd_loss
from omegaconf import DictConfig
import numpy as np


class FeatureExtractor(nn.Module):
    def __init__(
            self,
            cfg: DictConfig
    ):
        super(FeatureExtractor, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(num_features=cfg.model.input_dim),
            nn.Linear(in_features=cfg.model.input_dim, out_features=128),
            nn.PReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.PReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.PReLU()
        )

    def forward(self, x: T):
        return self.model(x)


class LabelPredictor(nn.Module):
    def __init__(
            self,
            cfg: DictConfig
    ):
        super(LabelPredictor, self).__init__()
        self.linear1 = nn.Linear(in_features=64, out_features=64)
        self.linear2 = nn.Linear(in_features=64, out_features=cfg.model.num_classes)
        self.activation = nn.PReLU()

    def forward(self, x: T):
        logits = self.linear2(self.activation(self.linear1(x)))
        return logits


class DAN(pl.LightningModule):
    def __init__(
            self,
            cfg: DictConfig,
    ):
        super(DAN, self).__init__()
        self.cfg: DictConfig = cfg
        self.feature_extractor: nn.Module = FeatureExtractor(cfg)
        self.source_predictor: nn.Module = LabelPredictor(cfg)
        self.target_predictor: nn.Module = LabelPredictor(cfg)
        # self.sigma_list = list(np.exp(np.arange(-8, 8, 0.5)))
        self.sigma_list = [
            1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
            1e3, 1e4, 1e5, 1e6
        ]
        self.mmd_biased: bool = cfg.model.mmd_biased
        self.automatic_optimization: bool = False
        self.epoch_idx: int = 0

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.cfg.train.learning_rate,
            weight_decay=self.cfg.train.weight_decay
        )
        return optimizer

    def forward(self, data: T) -> T:
        feature = self.feature_extractor(data)
        logits = self.target_predictor(feature)
        return logits

    def training_step(self, sample_batch: DANSample, batch_idx: int):
        optimizer = self.optimizers(use_pl_optimizer=True)

        source_data = sample_batch.source_data
        target_data = sample_batch.target_data
        batch_size: int = source_data.size(0)
        source_label = sample_batch.source_label
        target_label = sample_batch.target_label
        data = torch.cat([source_data, target_data], dim=0)

        feature = self.feature_extractor(data)
        source_feature = feature[:batch_size, :]
        target_feature = feature[batch_size:, :]
        source_logits_1 = self.source_predictor.linear1(source_feature)
        target_logits_1 = self.target_predictor.linear1(target_feature)
        source_logits_2 = self.source_predictor.linear2(self.source_predictor.activation(source_logits_1))
        target_logits_2 = self.target_predictor.linear2(self.target_predictor.activation(target_logits_1))

        # mmd_loss_1 = mmd_loss(source_logits_1, target_logits_1, self.sigma_list, self.mmd_biased)
        # mmd_loss_2 = mmd_loss(source_logits_2, target_logits_2, self.sigma_list, self.mmd_biased)
        # mmd_loss_feature = mmd_loss(source_feature, target_feature, self.sigma_list)
        mmd_loss_1 = mmd_loss(source_logits_1, target_logits_1, self.sigma_list)
        mmd_loss_2 = mmd_loss(source_logits_2, target_logits_2, self.sigma_list)
        source_loss = F.nll_loss(F.log_softmax(source_logits_2, dim=1), source_label)
        loss = mmd_loss_1 + 2 * mmd_loss_2 + source_loss

        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

        source_accuracy = (torch.argmax(source_logits_2.detach(), dim=1) == source_label).float().mean()
        target_accuracy = (torch.argmax(target_logits_2.detach(), dim=1) == target_label).float().mean()

        return {
            'train_step_loss': loss.detach(),
            'train_step_source_loss': source_loss.detach(),
            'train_step_source_accuracy': source_accuracy.detach(),
            'train_step_target_accuracy': target_accuracy.detach(),
            # 'train_step_mmd_loss_feature': mmd_loss_feature.detach(),
            'train_step_mmd_loss_1': mmd_loss_1.detach(),
            'train_step_mmd_loss_2': mmd_loss_2.detach(),
        }

    def training_epoch_end(self, outputs: List[Dict]) -> None:
        self.epoch_idx += 1
        epoch_loss = torch.tensor([x['train_step_loss'].mean() for x in outputs]).mean()
        epoch_source_loss = torch.tensor([x['train_step_source_loss'].mean() for x in outputs]).mean()
        source_accuracy = torch.tensor([x['train_step_source_accuracy'].mean() for x in outputs]).mean()
        target_accuracy = torch.tensor([x['train_step_target_accuracy'].mean() for x in outputs]).mean()
        self.log("train_epoch_loss", epoch_loss)
        self.log("train_epoch_source_loss", epoch_source_loss)
        self.log("train_epoch_source_accuracy", source_accuracy)
        self.log("train_epoch_target_accuracy", target_accuracy, prog_bar=True)
        self.log("train_epoch_mmd_loss_1", torch.tensor([x['train_step_mmd_loss_1'].mean() for x in outputs]).mean())
        self.log("train_epoch_mmd_loss_2", torch.tensor([x['train_step_mmd_loss_2'].mean() for x in outputs]).mean())
        # self.log("train_epoch_mmd_loss_feature", torch.tensor([x['train_step_mmd_loss_feature'].mean() for x in outputs]).mean())

    def manual_backward(self, loss: T, *args, **kwargs) -> None:
        loss.backward()
        return

    def validation_step(self, sample_batch: DANSample, batch_idx: int):
        data = sample_batch.target_data
        label = sample_batch.target_label
        target_logits = self.forward(data)
        target_pred = torch.argmax(target_logits, dim=1)
        loss = F.nll_loss(F.log_softmax(target_logits.detach(), dim=1), label)
        accuracy = (target_pred == label).float().mean()

        self.log("validation_step_loss", loss)
        self.log("validation_step_accuracy", accuracy)

        return {
            'validation_step_loss': loss.detach(),
            'validation_step_accuracy': accuracy.detach(),
        }

    def validation_epoch_end(self, outputs: List[Dict]) -> None:
        epoch_loss = torch.tensor([x['validation_step_loss'].mean() for x in outputs]).mean()
        epoch_accuracy = torch.tensor([x['validation_step_accuracy'].mean() for x in outputs]).mean()
        self.log("validation_epoch_loss", epoch_loss)
        self.log("validation_epoch_accuracy", epoch_accuracy, prog_bar=True)
        return
