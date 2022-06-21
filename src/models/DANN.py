from typing import List, Dict, Any, Tuple
import torch
from torch import Tensor as T
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from src.utils.dataset import DANNSample
from omegaconf import DictConfig
import numpy as np


class ReverseGradientLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x, adaptation_param: float):
        ctx.adaptation_param = adaptation_param
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output):
        output = grad_output.neg() * ctx.adaptation_param
        return output, None


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
        )

    def forward(self, x: T):
        return self.model(x)


class LabelPredictor(nn.Module):
    def __init__(
            self,
            cfg: DictConfig
    ):
        super(LabelPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=128, out_features=64),
            nn.PReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.PReLU(),
            nn.Linear(in_features=64, out_features=cfg.model.num_classes),
        )

    def forward(self, x: T):
        output = self.model(x)
        return F.log_softmax(output, dim=1)


class DomainDiscriminator(nn.Module):
    def __init__(self):
        super(DomainDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=128, out_features=64),
            nn.PReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.PReLU(),
            nn.Linear(in_features=64, out_features=2),
        )
        self.gradient_reverser = ReverseGradientLayer()

    def forward(self, x: T, alpha: float):
        x = self.gradient_reverser.apply(x, alpha)
        output = self.model(x)
        return F.log_softmax(output, dim=1)


class DANN(pl.LightningModule):
    def __init__(
            self,
            cfg: DictConfig,
    ):
        super().__init__()
        self.cfg: DictConfig = cfg
        self.feature_extractor: FeatureExtractor = FeatureExtractor(cfg)
        self.label_predictor: LabelPredictor = LabelPredictor(cfg)
        self.domain_discriminator: DomainDiscriminator = DomainDiscriminator()
        self.automatic_optimization: bool = False
        self.epoch_idx: int = 0

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.train.learning_rate,
            weight_decay=self.cfg.train.weight_decay,
        )

        return optimizer

    def forward(self, data: T, adaptation_param: float) -> Tuple[T, T]:
        feature = self.feature_extractor(data)
        class_output = self.label_predictor(feature)
        domain_output = self.domain_discriminator(feature, adaptation_param)
        return class_output, domain_output

    def training_step(self, sample_batch: DANNSample, batch_idx: int):
        optimizer = self.optimizers(use_pl_optimizer=True)

        # Calculate the progress of training
        progress: float = self.epoch_idx / self.cfg.train.num_epochs

        # Adjust learning_rate each epoch
        if self.cfg.train.adjust_learning_rate:
            current_lr = self.cfg.train.learning_rate / (1.0 + self.cfg.train.alpha * progress) ** self.cfg.train.beta
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            self.log("lr", current_lr)

        # calculate lambda(adaptation_param) for this step
        if self.cfg.model.adaptation_param == 'auto':
            adaptation_param: float = 2. / (1 + np.exp(-self.cfg.train.gamma * progress)) - 1.
        else:
            adaptation_param = self.cfg.model.adaptation_param
        self.log('adaptation_param', adaptation_param)

        data = sample_batch.data
        class_label = sample_batch.class_label
        domain_label = sample_batch.domain_label
        source_mask = (domain_label == 0)

        class_pred, domain_pred = self.forward(data, adaptation_param)

        class_loss = F.nll_loss(class_pred[source_mask], class_label[source_mask])
        domain_loss = F.nll_loss(domain_pred, domain_label)
        domain_accuracy = (torch.argmax(domain_pred.detach(), dim=1) == domain_label).sum()/domain_label.size(0)
        class_accuracy = (torch.argmax(class_pred.detach()[source_mask], dim=1) == class_label[source_mask]).sum()/class_label[source_mask].size(0)

        loss: T = class_loss + self.cfg.train.domain_loss_weight*domain_loss

        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

        self.log('class_loss', class_loss.detach())
        self.log('domain_loss', domain_loss.detach())
        self.log('train_step_loss', loss.detach())
        self.log('step_class_acc', class_accuracy.detach())
        self.log('step_domain_acc', domain_accuracy.detach())

        return {
                    'train_step_loss': loss.detach(),
                    'domain_step_acc': domain_accuracy.detach(),
                    'class_step_acc': class_accuracy.detach(),
                }

    def training_epoch_end(self, outputs: List[Dict]) -> None:
        self.epoch_idx += 1
        epoch_loss = torch.tensor([x['train_step_loss'].mean() for x in outputs]).mean()
        domain_acc = torch.tensor([x['domain_step_acc'].mean() for x in outputs]).mean()
        class_acc = torch.tensor([x['class_step_acc'].mean() for x in outputs]).mean()
        self.log('train_epoch_loss', epoch_loss)
        self.log('domain_epoch_acc', domain_acc)
        self.log('class_epoch_acc', class_acc)

    def manual_backward(self, loss: T, *args, **kwargs) -> None:
        loss.backward()
        return

    def validation_step(self, sample_batch: DANNSample, batch_idx) -> Dict:
        data = sample_batch.data
        class_label = sample_batch.class_label
        class_output, _ = self.forward(data, self.cfg.model.adaptation_param)
        class_pred = torch.argmax(class_output, dim=1)
        accuracy = (class_pred == class_label).detach().sum()/len(class_label)
        loss = F.nll_loss(class_output, class_label)

        self.log('val_step_loss', loss.detach())
        return {
            'loss': loss.detach(),
            'accuracy': accuracy
        }

    def validation_epoch_end(self, outputs: List[Dict]) -> None:
        epoch_loss = torch.tensor([x['loss'].mean() for x in outputs]).mean()
        epoch_accuracy = torch.tensor([x['accuracy'] for x in outputs]).mean()
        self.log('val_epoch_loss', epoch_loss)
        self.log('val_epoch_accuracy', epoch_accuracy)
        return
