from typing import List, Dict, Any, Tuple
import torch
from torch import Tensor as T
import pytorch_lightning as pl
import torch.nn.functional as F
from src.utils.dataset import DANNSample
from omegaconf import DictConfig
from src.models.DANN import FeatureExtractor
from src.models.DANN import LabelPredictor
from hydra.utils import to_absolute_path
import os

class BackboneModel(pl.LightningModule):
    def __init__(
            self,
            cfg: DictConfig
    ):
        super(BackboneModel, self).__init__()
        self.cfg: DictConfig = cfg
        self.feature_extractor: FeatureExtractor = FeatureExtractor(cfg)
        self.label_predictor: LabelPredictor = LabelPredictor(cfg)
        self.automatic_optimization: bool = False
        self.epoch_idx: int = 0

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.train.backbone.learning_rate,
            weight_decay=self.cfg.train.backbone.weight_decay,
        )

        return optimizer

    def forward(self, data: T) -> T:
        feature = self.feature_extractor(data)
        class_output = self.label_predictor(feature)
        return class_output

    def training_step(self, sample_batch: DANNSample, batch_idx: int):
        optimizer = self.optimizers(use_pl_optimizer=True)

        data = sample_batch.data
        label = sample_batch.class_label

        output = self.forward(data)
        loss = F.cross_entropy(output, label, reduction='mean')
        accuracy = (torch.argmax(output.detach(), dim=1) == label).sum()/label.size(0)

        self.log('train_step_loss', loss.detach())
        self.log('train_step_acc', accuracy.detach())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {
            'train_step_loss': loss.detach(),
            'train_step_acc': accuracy.detach()
        }

    def training_epoch_end(self, outputs: List[Dict]) -> None:
        self.epoch_idx += 1
        epoch_loss = torch.tensor([x['train_step_loss'].mean() for x in outputs]).mean()
        epoch_acc = torch.tensor([x['train_step_acc'].mean() for x in outputs]).mean()
        self.log('train_epoch_loss', epoch_loss)
        self.log('train_epoch_acc', epoch_acc)

    def validation_step(self, sample_batch: DANNSample, batch_idx: int) -> Dict:
        data = sample_batch.data
        label = sample_batch.class_label
        output = self.forward(data)
        pred = torch.argmax(output.detach(), dim=1)
        loss = F.cross_entropy(output.detach(), label, reduction='mean')
        accuracy = (pred == label).sum()/len(label)

        return {
            'loss': loss.detach(),
            'accuracy': accuracy
        }

    def validation_epoch_end(self, outputs: List[Dict]) -> None:
        epoch_loss = torch.tensor([x['loss'].mean() for x in outputs]).mean()
        epoch_accuracy = torch.tensor([x['accuracy'] for x in outputs]).mean()
        self.log('val_epoch_loss', epoch_loss)
        self.log('val_epoch_accuracy', epoch_accuracy)


class SEEDClassifier(pl.LightningModule):
    def __init__(
            self,
            cfg: DictConfig
    ):
        super(SEEDClassifier, self).__init__()
        self.cfg: DictConfig = cfg
        backbone: BackboneModel = BackboneModel.load_from_checkpoint(
            checkpoint_path=os.path.join(to_absolute_path(cfg.train.classifier.checkpoint_dir), cfg.basic.target_sub+'.ckpt'),
            cfg=cfg
        )
        self.feature_extractor: FeatureExtractor = backbone.feature_extractor
        # del self.backbone
        self.label_predictor: LabelPredictor = LabelPredictor(cfg)
        self.automatic_optimization: bool = False
        self.epoch_idx: int = 0

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.label_predictor.parameters(),
            lr=self.cfg.train.classifier.learning_rate,
            weight_decay=self.cfg.train.classifier.weight_decay,
        )

        return optimizer

    def forward(self, data: T) -> T:
        feature = self.feature_extractor(data)
        class_output = self.label_predictor(feature)
        return class_output

    def training_step(self, sample_batch: DANNSample, batch_idx: int):
        optimizer = self.optimizers(use_pl_optimizer=True)

        data = sample_batch.data
        label = sample_batch.class_label

        output = self.forward(data)
        loss = F.cross_entropy(output, label, reduction='mean')
        accuracy = (torch.argmax(output.detach(), dim=1) == label).sum()/label.size(0)

        self.log('train_step_loss', loss.detach())
        self.log('train_step_acc', accuracy.detach())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {
            'train_step_loss': loss.detach(),
            'train_step_acc': accuracy.detach()
        }

    def training_epoch_end(self, outputs: List[Dict]) -> None:
        self.epoch_idx += 1
        epoch_loss = torch.tensor([x['train_step_loss'].mean() for x in outputs]).mean()
        epoch_acc = torch.tensor([x['train_step_acc'].mean() for x in outputs]).mean()
        self.log('train_epoch_loss', epoch_loss)
        self.log('train_epoch_acc', epoch_acc)

    def validation_step(self, sample_batch: DANNSample, batch_idx: int) -> Dict:
        data = sample_batch.data
        label = sample_batch.class_label
        output = self.forward(data)
        pred = torch.argmax(output.detach(), dim=1)
        loss = F.cross_entropy(output.detach(), label, reduction='mean')
        accuracy = (pred == label).sum()/len(label)

        return {
            'loss': loss.detach(),
            'accuracy': accuracy
        }

    def validation_epoch_end(self, outputs: List[Dict]) -> None:
        epoch_loss = torch.tensor([x['loss'].mean() for x in outputs]).mean()
        epoch_accuracy = torch.tensor([x['accuracy'] for x in outputs]).mean()
        self.log('val_epoch_loss', epoch_loss)
        self.log('val_epoch_accuracy', epoch_accuracy)










