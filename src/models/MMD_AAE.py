from typing import List, Dict, Tuple
import torch
from torch import Tensor as T
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from src.utils.dataset import SEEDSample
from src.utils.loss_function import MMD_AAE_Loss
from omegaconf import DictConfig
from collections import namedtuple
import numpy as np

MMD_AAE_Output = namedtuple(
    'MMD_AAE_Output',
    [
        "encodings",
        "decoded",
        "class_pred",
        "norm_input",
    ]
)


class Encoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            # nn.BatchNorm1d(cfg.model.input_dim),
            nn.Linear(in_features=cfg.model.input_dim, out_features=cfg.model.hidden_dim),
            nn.PReLU(),
            nn.Linear(in_features=cfg.model.hidden_dim, out_features=cfg.model.hidden_dim),
        )

    def forward(self, x: T) -> T:
        output = self.model(x)
        return output


class Decoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(Decoder, self).__init__()
        # self.dropout = nn.Dropout(cfg.model.dropout)
        self.model = nn.Sequential(
            nn.Linear(in_features=cfg.model.hidden_dim, out_features=cfg.model.hidden_dim),
            nn.PReLU(),
            nn.Linear(in_features=cfg.model.hidden_dim, out_features=cfg.model.input_dim),
        )

    def forward(self, x: T) -> T:
        # x = self.dropout(x)
        x = self.model(x)
        return x


class TaskClassifier(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(TaskClassifier, self).__init__()
        # self.dropout = nn.Dropout(cfg.model.dropout)
        self.model = nn.Sequential(
            # nn.BatchNorm1d(num_features=128),
            nn.Linear(in_features=cfg.model.hidden_dim, out_features=64),
            nn.PReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.PReLU(),
            nn.Linear(in_features=64, out_features=cfg.model.num_classes),
        )

    def forward(self, x: T) -> T:
        # x = self.dropout(x)
        x = self.model(x)
        return F.log_softmax(x, dim=1)


class Discriminator(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=cfg.model.hidden_dim, out_features=64),
            nn.PReLU(),
            # nn.Linear(in_features=64, out_features=2),
            nn.Linear(in_features=64, out_features=1),
            nn.Sigmoid(),
        )

    def forward(self, x: T) -> T:
        output = self.model(x)
        return output


class Normalizer(nn.Module):
    def __init__(self, dim: int = 1):
        super(Normalizer, self).__init__()
        self.dim = dim

    def forward(self, x: T, eps: float = 1e-5) -> T:
        output = (x - x.mean(dim=self.dim, keepdim=True)) / (x.var(dim=self.dim, keepdim=True) + eps).sqrt()
        return output


class MMD_AAE_model(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super(MMD_AAE_model, self).__init__()
        self.cfg: DictConfig = cfg
        self.normalizer = nn.LayerNorm(normalized_shape=cfg.model.input_dim)
        # self.normalizer = nn.InstanceNorm1d(num_features=cfg.model.input_dim)
        # self.normalizer = nn.BatchNorm1d(num_features=cfg.model.input_dim)
        self.encoder = Encoder(cfg=cfg)
        self.decoder = Decoder(cfg=cfg)
        self.classifier = TaskClassifier(cfg=cfg)
        self.discriminator = Discriminator(cfg=cfg)
        self.sigma_list = [
            1, 5, 10,
        ]
        self.mmd_loss_function = MMD_AAE_Loss(
            num_sources=cfg.basic.num_subjects,
            sigma_list=self.sigma_list
        )
        # We want the extracted features follow a Laplace Distribution with loc=0, scale-0.1
        self.sampler = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([0.1]))
        self.automatic_optimization: bool = False
        self.epoch_idx: int = 0

    def forward(self, x: T) -> MMD_AAE_Output:
        norm_input = self.normalizer(x)
        encodings = self.encoder(norm_input)
        decoded = self.decoder(encodings)
        class_pred = self.classifier(encodings)
        return MMD_AAE_Output(
            encodings=encodings,
            decoded=decoded,
            class_pred=class_pred,
            norm_input=norm_input,
        )

    def configure_optimizers(self):
        discriminator_optimizer = torch.optim.AdamW(
            params=self.discriminator.parameters(),
            lr=self.cfg.train.adversarial_lr
        )
        generator_optimizer = torch.optim.AdamW(
            [
                {'params': self.normalizer.parameters()},
                {'params': self.encoder.parameters()},
                {'params': self.decoder.parameters()},
                {'params': self.classifier.parameters()}
            ],
            lr=self.cfg.train.model_lr
        )
        return discriminator_optimizer, generator_optimizer

    def training_step(self, sample_batch: SEEDSample, batch_idx: int):
        optimizers = self.optimizers(use_pl_optimizer=True)
        discriminator_optimizer = optimizers[0]
        generator_optimizer = optimizers[1]

        # data.size() = [num_subjects, batch_size, num_features]
        data = sample_batch.data.permute(1, 0, 2)
        # encodings.size() = [num_subjects, batch_size, hidden_dim]
        encodings = torch.zeros(data.size(0), data.size(1), self.cfg.model.hidden_dim).to(data.device)
        # decoded.size() = [num_subjects, batch_size, input_dim]
        decoded = torch.zeros(data.size(0), data.size(1), self.cfg.model.input_dim).to(data.device)
        # class_pred.size() = [num_subjects, batch_size, num_classes]
        class_pred = torch.zeros(data.size(0), data.size(1), self.cfg.model.num_classes).to(data.device)
        # norm_input.size() = [num_subjects, batch_size, input_dim]
        norm_input = torch.zeros_like(data)

        ###########################
        # Train Generator
        ###########################

        for i in range(data.size(0)):
            output = self.forward(data[i])
            encodings[i] = output.encodings
            decoded[i] = output.decoded
            class_pred[i] = output.class_pred
            norm_input[i] = output.norm_input

        source_idx = np.random.choice(data.size(0) - 1)
        source_encoding = encodings[source_idx].unsqueeze(0)
        target_encoding = encodings[-1]
        # source_encoding = encodings[:-1]
        # target_encoding = encodings[-1]

        source_class_pred = class_pred[:-1].flatten(end_dim=1)
        source_labels = torch.cat([sample_batch.label for i in range(data.size(0) - 1)], dim=0)
        decoded = decoded.flatten(end_dim=1)
        norm_input = norm_input.flatten(end_dim=1)

        # fake_encodings = self.sampler.sample(encodings.size()).squeeze(-1).to(encodings.device)
        # fake_encodings = torch.cat([self.sampler.sample((target_encoding.size(1),)) for i in range(target_encoding.size(0))], dim=1).transpose(0,1).to(target_encoding.device)

        source_domain_labels = torch.zeros_like(sample_batch.label)
        target_domain_labels = torch.ones_like(sample_batch.label)
        # real_domain_labels = torch.ones(encodings.size(0), encodings.size(1)).to(encodings.device)
        # fake_domain_labels = torch.zeros(encodings.size(0), encodings.size(1)).to(encodings.device)
        # fake_domain_labels = torch.zeros_like(sample_batch.label)
        # real_domain_labels = torch.ones_like(sample_batch.label)

        all_encodings = torch.cat([source_encoding.mean(dim=0), target_encoding], dim=0)
        real_labels = torch.cat([source_domain_labels, target_domain_labels], dim=0)
        # all_encodings = torch.cat([source_encoding, target_encoding], dim=0)
        # all_encodings = torch.cat([encodings, fake_encodings], dim=0).flatten(end_dim=1) # [num_subjects * batch_size, hidden_dim]
        # all_encodings = torch.cat([target_encoding, fake_encodings], dim=0)  # [2 * batch_size, hidden_dim]
        # real_labels = torch.cat([source_domain_labels, target_domain_labels], dim=0)
        # adv_labels = torch.cat([target_domain_labels, source_domain_labels], dim=0)
        # real_labels = torch.cat([real_domain_labels, fake_domain_labels], dim=0).flatten(end_dim=1)
        # adv_labels = torch.cat([fake_domain_labels, real_domain_labels], dim=0).flatten(end_dim=1)
        # real_labels = torch.cat([real_domain_labels, fake_domain_labels], dim=0)
        # adv_labels = torch.cat([fake_domain_labels, real_domain_labels], dim=0)
        # adv_pred = self.discriminator(all_encodings)
        adv_pred = self.discriminator(target_encoding)

        task_loss = F.nll_loss(source_class_pred, source_labels.long())
        decoder_loss = F.mse_loss(decoded, norm_input)
        adv_loss = F.mse_loss(adv_pred.squeeze(-1), source_domain_labels.float())
        # adv_loss = F.mse_loss(adv_pred.squeeze(-1).square(), adv_labels.float())
        # adv_loss = F.nll_loss(adv_pred, adv_labels.long())
        mmd_loss = self.mmd_loss_function.calculate(encodings)
        model_loss = self.cfg.train.mmd_weight * mmd_loss + \
                     self.cfg.train.adv_weight * adv_loss + \
                     self.cfg.train.task_weight * task_loss + \
                     self.cfg.train.decoder_weight * decoder_loss

        generator_optimizer.optimizer.zero_grad()
        model_loss.backward()
        generator_optimizer.step()

        ########################
        # Train Discriminator
        ########################

        discriminator_pred = self.discriminator(all_encodings.detach())
        discriminator_loss = F.mse_loss(discriminator_pred.squeeze(-1), real_labels.float())
        # discriminator_pred = self.discriminator(all_encodings.detach())
        # discriminator_loss = F.nll_loss(discriminator_pred, real_labels.long())
        # discriminator_loss = F.mse_loss(discriminator_pred.squeeze(-1).square(), real_labels.float())

        discriminator_optimizer.optimizer.zero_grad()
        discriminator_loss.backward()
        discriminator_optimizer.step()

        return {
            'train_step_model_loss': model_loss.detach(),
            'train_step_discriminator_loss': discriminator_loss.detach(),
            'train_step_task_loss': task_loss.detach(),
            'train_step_decoder_loss': decoder_loss.detach(),
            'train_step_mmd_loss': mmd_loss.detach(),
            'train_step_adv_loss': adv_loss.detach(),
            'train_step_task_acc': source_class_pred.argmax(dim=1).eq(source_labels).float().mean(),
            # 'train_step_adv_acc': adv_pred.argmax(dim=1).eq(adv_labels).float().mean(),
            'train_step_adv_acc': adv_pred.squeeze(-1).round().eq(source_domain_labels).float().mean(),
            # 'train_step_discriminator_acc': discriminator_pred.argmax(dim=1).eq(real_labels).float().mean()
            'train_step_discriminator_acc': discriminator_pred.squeeze(-1).round().eq(real_labels).float().mean(),
        }

    def training_epoch_end(self, outputs: List[Dict]) -> None:
        self.epoch_idx += 1
        self.log('train_model_loss', torch.tensor([x['train_step_model_loss'].mean() for x in outputs]).mean())
        self.log('train_discriminator_loss', torch.tensor([x['train_step_discriminator_loss'].mean() for x in outputs]).mean())
        self.log('train_task_loss', torch.tensor([x['train_step_task_loss'].mean() for x in outputs]).mean())
        self.log('train_decoder_loss', torch.tensor([x['train_step_decoder_loss'].mean() for x in outputs]).mean())
        self.log('train_mmd_loss', torch.tensor([x['train_step_mmd_loss'].mean() for x in outputs]).mean(), prog_bar=True)
        self.log('train_adv_loss', torch.tensor([x['train_step_adv_loss'].mean() for x in outputs]).mean())
        self.log('train_task_acc', torch.tensor([x['train_step_task_acc'].mean() for x in outputs]).mean())
        self.log('train_adv_acc', torch.tensor([x['train_step_adv_acc'].mean() for x in outputs]).mean())
        self.log('train_discriminator_acc', torch.tensor([x['train_step_discriminator_acc'].mean() for x in outputs]).mean())

    def validation_step(self, sample_batch: SEEDSample, batch_idx: int):
        data = sample_batch.data
        labels = sample_batch.label
        assert labels.size(0) == data.size(0)

        output = self.forward(data)
        loss = F.nll_loss(output.class_pred, labels)
        acc = output.class_pred.argmax(dim=1).eq(labels).float().mean()

        return {
            'val_step_loss': loss.detach(),
            'val_step_acc': acc.detach(),
        }

    def validation_epoch_end(self, outputs: List[Dict]) -> None:
        self.log('val_epoch_loss', torch.tensor([x['val_step_loss'].mean() for x in outputs]).mean(), prog_bar=True)
        self.log('val_epoch_accuracy', torch.tensor([x['val_step_acc'].mean() for x in outputs]).mean(), prog_bar=True)
        return
