import pickle
from typing import Dict
import torch
from torch import Tensor as T
from torch.utils.data import Dataset
from collections import namedtuple
import numpy as np


SeedSample = namedtuple(
    'SeedSample',
    [
        'data',
        'class_label',
        'domain_label',
    ]
)

class SeedDataset(Dataset):
    data_path: str
    subject: str
    dataset: Dict
    source_data: T
    source_class_labels: T
    source_domain_labels: T
    target_data: T
    target_class_labels: T
    target_domain_labels: T
    all_data: T
    all_class_labels: T
    all_domain_labels: T

    def __init__(
            self,
            data_path: str,
            target_subject: str,
    ):
        super().__init__()
        self.data_path: str = data_path
        with open(self.data_path, 'rb') as f:
            self.dataset = pickle.load(f)
        assert target_subject in self.dataset.keys(), KeyError(f'Target subject NOT in dataset: [{self.dataset.keys()}]')
        self.target_subject: str = target_subject
        self._unpack_data()

    def _unpack_data(self):
        source_data = list()
        source_class_label = list()
        source_domain_label = list()

        for key in self.dataset.keys():
            if key == self.target_subject:
                self.target_data = torch.tensor(self.dataset[key]['data'], dtype=torch.float32)
                self.target_class_labels = torch.tensor(self.dataset[key]['label']+1, dtype=torch.long)
                self.target_domain_labels = torch.ones_like(self.target_class_labels)
            else:
                source_data.append(torch.tensor(self.dataset[key]['data'], dtype=torch.float32))
                source_class_label.append(torch.tensor(self.dataset[key]['label'] + 1, dtype=torch.long))
                source_domain_label.append(torch.zeros_like(source_class_label[-1]))
        self.source_data = torch.cat(source_data, dim=0)
        self.source_class_labels = torch.cat(source_class_label, dim=0)
        self.source_domain_labels = torch.cat(source_domain_label, dim=0)
        self.all_data = torch.cat([self.source_data, self.target_data], dim=0)
        self.all_class_labels = torch.cat([self.source_class_labels, self.target_class_labels], dim=0)
        self.all_domain_labels = torch.cat([self.source_domain_labels, self.target_domain_labels], dim=0)


class SeedDatasetForDANN(SeedDataset):
    def __init__(self, data_path: str, target_subject: str, mode: str):
        super().__init__(data_path, target_subject)
        self.mode = mode

    def __len__(self):
        if self.mode == 'train':
            return len(self.all_domain_labels)
        else:
            return len(self.target_domain_labels)

    def __getitem__(self, item: int):
        sample: SeedSample
        if self.mode == 'train':
            sample = SeedSample(
                data=self.all_data[item, :],
                class_label=self.all_class_labels[item],
                domain_label=self.all_domain_labels[item],
            )
        else:
            sample = SeedSample(
                data=self.target_data[item, :],
                class_label=self.target_class_labels[item],
                domain_label=self.target_domain_labels[item]
            )

        return sample

class SeedDatasetForTLBackbone(SeedDataset):
    def __init__(
            self,
            data_path: str,
            target_subject: str,
            mode: str,
            train_ratio: float
    ):
        super().__init__(data_path, target_subject)
        assert 0.0 < train_ratio < 1.0
        self.mode = mode
        self.shuffled_idx = np.arange(len(self.source_class_labels))
        np.random.seed(0)
        np.random.shuffle(self.shuffled_idx)
        self.train_len = int(len(self.source_class_labels) * train_ratio)
        self.dev_len = len(self.source_class_labels) - self.train_len

    def __len__(self):
        if self.mode == 'train':
            return self.train_len
        else:
            return self.dev_len

    def __getitem__(self, item: int):
        if self.mode == 'train':
            item = self.shuffled_idx[item]
        else:
            item = self.shuffled_idx[self.train_len+item]
        sample = SeedSample(
            data=self.source_data[item, :],
            class_label=self.source_class_labels[item],
            domain_label=self.source_domain_labels[item]
        )
        return sample


class SeedDatasetForTLClassifier(SeedDataset):
    def __init__(
            self,
            data_path: str,
            target_subject: str,
            mode: str,
            train_ratio: float
    ):
        super().__init__(data_path, target_subject)
        assert 0.0 < train_ratio < 1.0
        self.mode = mode
        self.shuffled_idx = np.arange(len(self.target_class_labels))
        np.random.seed(0)
        np.random.shuffle(self.shuffled_idx)
        self.train_len = int(len(self.target_class_labels) * train_ratio)
        self.dev_len = len(self.target_class_labels) - self.train_len

    def __len__(self):
        if self.mode == 'train':
            return self.train_len
        else:
            return self.dev_len

    def __getitem__(self, item: int):
        if self.mode == 'train':
            item = self.shuffled_idx[item]
        else:
            item = self.shuffled_idx[self.train_len+item]

        sample = SeedSample(
            data=self.target_data[item, :],
            class_label=self.target_class_labels[item],
            domain_label=self.target_domain_labels[item]
        )

        return sample


if __name__ == "__main__":
    train_set = SeedDatasetForDANN(
        data_path="data/data.pkl",
        target_subject='sub_1',
        mode='train'
    )
    dev_set = SeedDatasetForDANN(
        data_path="data/data.pkl",
        target_subject='sub_1',
        mode='dev'
    )


