import torch
from torch.autograd import Function


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        # return x
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


torch.manual_seed(0)
x = torch.randn((1,), requires_grad=True)
alpha = 0.1
y = ReverseLayerF.apply(x, alpha)
z = y.exp()
z.backward()
x.grad






import torch.nn as nn


class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 4 * 4, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, 50 * 4 * 4)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output

import torch
from torch import Tensor as T
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import logging
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm


class PassageMatcherPL(pl.LightningModule):
    def __init__(self,
                 cfg: DictConfig,
                 test_loader: DataLoader,
                 global_logger: logging.Logger
                 ):
        super().__init__()
        self.cfg = cfg
        self.test_loader = test_loader
        self.global_logger = global_logger
        self.encoder_config: PretrainedConfig = AutoConfig.from_pretrained(self.cfg.biencoder.model_name)
        self.passage_encoder: PreTrainedModel = AutoModel.from_pretrained(self.cfg.biencoder.model_name,
                                                                          config=self.encoder_config)
        self.classifier: nn.Module = nn.Linear(self.encoder_config.hidden_size, 2)
        self.automatic_optimization = False
        self.epoch_idx = 0

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.train.learning_rate,
                                      weight_decay=self.cfg.train.weight_decay)
        return optimizer

    def forward(
            self,
            query_ids: T,
            query_segments: T,
            query_attention_mask: T,
            context_ids: T,
            context_segments: T,
            context_attention_mask: T
    ):

        encoded_query = self.passage_encoder(
            input_ids=query_ids,
            token_type_ids=query_segments,
            attention_mask=query_attention_mask
        ).last_hidden_state[:, 0, :]

        encoded_context = self.passage_encoder(
            input_ids=context_ids,
            token_type_ids=context_segments,
            attention_mask=context_attention_mask
        ).last_hidden_state[:, 0, :]

        matching_logits = encoded_query * encoded_context
        matching_logits = self.classifier(matching_logits)

        return matching_logits

    def training_step(self, batch: Dict, batch_idx):
        optimizer = self.optimizers(use_pl_optimizer=True)
        optimizer.zero_grad()

        query_passage: Dict = batch["query"]
        context_passage: Dict = batch["context"]
        label: T = batch["label"].to(self.device)
        query_ids: T = query_passage["input_ids"].squeeze(1)
        query_segments: T = query_passage["token_type_ids"].squeeze(1)
        query_attention_mask: T = query_passage["attention_mask"].squeeze(1)
        context_ids: T = context_passage["input_ids"].squeeze(1)
        context_segments: T = context_passage["token_type_ids"].squeeze(1)
        context_attention_mask: T = context_passage["attention_mask"].squeeze(1)

        matching_logits = self.forward(
            query_ids=query_ids,
            query_segments=query_segments,
            query_attention_mask=query_attention_mask,
            context_ids=context_ids,
            context_segments=context_segments,
            context_attention_mask=context_attention_mask
        )

        matching_scores = F.softmax(matching_logits, dim=1)

        loss = F.cross_entropy(matching_scores, label, reduction="mean")
        self.manual_backward(loss)
        optimizer.step()

        self.log("train_step_loss", loss)
        return {"train_step_loss": loss.detach()}

    def manual_backward(self, loss: torch.Tensor, *args, **kwargs) -> None:
        loss.backward()

    def on_train_epoch(self) -> None:
        self.epoch_idx += 1

    def validation_step(self, batch: Dict, batch_idx):
        query_passage: Dict = batch["query"]
        context_passage: Dict = batch["context"]
        label: T = batch["label"]
        query_ids: T = query_passage["input_ids"].squeeze(1)
        query_segments: T = query_passage["token_type_ids"].squeeze(1)
        query_attention_mask: T = query_passage["attention_mask"].squeeze(1)
        context_ids: T = context_passage["input_ids"].squeeze(1)
        context_segments: T = context_passage["token_type_ids"].squeeze(1)
        context_attention_mask: T = context_passage["attention_mask"].squeeze(1)

        matching_logits = self.forward(
            query_ids=query_ids,
            query_segments=query_segments,
            query_attention_mask=query_attention_mask,
            context_ids=context_ids,
            context_segments=context_segments,
            context_attention_mask=context_attention_mask
        )

        matching_scores = F.softmax(matching_logits, dim=1)

        loss = F.cross_entropy(matching_scores, label, reduction="mean")
        self.log("val_step_loss", loss)
        return {"val_step_loss": loss.detach()}

    def validation_epoch_end(self, outputs) -> Dict:
        avg_loss = torch.tensor([x['val_step_loss'].mean() for x in outputs]).mean()
        self.log("val_epoch_loss", avg_loss)
        return {"val_avg_loss": avg_loss}

    def on_validation_end(self) -> None:
        self.eval()
        test_results: List[np.ndarray] = []
        with torch.no_grad():
            with tqdm(range(len(self.test_loader))) as pbar:
                for idx, batch in enumerate(self.test_loader):
                    query_passage: Dict = batch["query"]
                    context_passage: Dict = batch["context"]
                    query_ids: T = query_passage["input_ids"].squeeze(1).to(self.device)
                    query_segments: T = query_passage["token_type_ids"].squeeze(1).to(self.device)
                    query_attention_mask: T = query_passage["attention_mask"].squeeze(1).to(self.device)
                    context_ids: T = context_passage["input_ids"].squeeze(1).to(self.device)
                    context_segments: T = context_passage["token_type_ids"].squeeze(1).to(self.device)
                    context_attention_mask: T = context_passage["attention_mask"].squeeze(1).to(self.device)

                    pred = self.forward(
                        query_ids=query_ids,
                        query_segments=query_segments,
                        query_attention_mask=query_attention_mask,
                        context_ids=context_ids,
                        context_segments=context_segments,
                        context_attention_mask=context_attention_mask
                    )
                    pred = F.softmax(pred, dim=1)[:, 1]
                    test_results.append(pred.detach().cpu().numpy())
                    if idx % 10 == 0:
                        pbar.update(10)
        generate_submission(f'./submissions/epoch_{self.epoch_idx}', np.concatenate(test_results))


import logging

import hydra
import pytorch_lightning.loggers
import torch
import os
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import numpy as np
import pytorch_lightning as pl

from torch.utils.data import Subset, DataLoader
from src.utils import NetworkDatasetPassageMatchingPL
from src.models.passage_matching.biencoder_pl import PassageMatcherPL
from transformers import (
    AutoConfig,
    PretrainedConfig,
    AutoTokenizer,
    PreTrainedTokenizer
)


@hydra.main(config_path="conf_pl", config_name="config")
def run(cfg: DictConfig):
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    tokenizer_config: PretrainedConfig = AutoConfig.from_pretrained(cfg.biencoder.model_name)
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(cfg.biencoder.model_name, config=tokenizer_config)

    cfg.data.data_path = to_absolute_path(cfg.data.data_path)
    train_set = NetworkDatasetPassageMatchingPL(
        dataset_path=os.path.join(cfg.data.data_path, cfg.data.train_file),
        tokenizer=tokenizer,
        max_seq_len=cfg.biencoder.max_seq_len
    )
    if cfg.data.train_size:
        train_set = Subset(train_set, np.arange(cfg.data.train_size) - int(cfg.data.train_size / 2))
    dev_set = NetworkDatasetPassageMatchingPL(
        dataset_path=os.path.join(cfg.data.data_path, cfg.data.dev_file),
        tokenizer=tokenizer,
        max_seq_len=cfg.biencoder.max_seq_len
    )
    if cfg.data.dev_size:
        dev_set = Subset(dev_set, np.arange(cfg.data.dev_size) - int(cfg.data.dev_size / 2))
    test_set = NetworkDatasetPassageMatchingPL(
        dataset_path=os.path.join(cfg.data.data_path, cfg.data.test_file),
        tokenizer=tokenizer,
        max_seq_len=cfg.biencoder.max_seq_len
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=cfg.train.num_workers,
        drop_last=True
    )
    dev_loader = DataLoader(
        dev_set,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=cfg.train.num_workers,
        drop_last=True
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=cfg.train.num_workers,
        drop_last=False
    )
    logger = pl.loggers.TensorBoardLogger(save_dir=cfg.log_dir)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints",
        filename="{epoch}--{step}--{val_epoch_loss:.4f}",
        monitor="val_epoch_loss",
        save_last=True,
        save_top_k=3,
        mode="min",
        save_weights_only=True,
        save_on_train_epoch_end=True
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=cfg.train.epochs,
        progress_bar_refresh_rate=1,
        callbacks=[checkpoint_callback,
                   pl.callbacks.TQDMProgressBar(refresh_rate=1)],
        enable_checkpointing=True,
        num_sanity_val_steps=0,
        logger=logger
    )
    global_logger = logging.getLogger(__name__)
    global_logger.info("Start training")
    model = PassageMatcherPL(
        cfg=cfg,
        test_loader=test_loader,
        global_logger=global_logger
    )
    trainer.fit(model, train_loader, dev_loader)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    run()
