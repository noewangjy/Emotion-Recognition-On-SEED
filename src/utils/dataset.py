import pickle
from typing import Dict, List
import torch
from torch import Tensor as T
from torch.utils.data import Dataset
from collections import namedtuple
import numpy as np


SEEDSample = namedtuple(
    'SEEDSample',
    [
        'data',
        'label',
    ]
)

DANNSample = namedtuple(
    'DANNSample',
    [
        'data',
        'class_label',
        'domain_label',
    ]
)

DANSample = namedtuple(
    'DANSample',
    [
        'source_data',
        'target_data',
        'source_label',
        'target_label',
    ]
)

TransformerPreTrainingSample = namedtuple(
    'TransformerPreTrainingSample',
    [
        'source_data',
        'target_data',
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

    source_data_by_subject: T
    source_labels_by_subject: T
    all_data_by_subject: T
    unified_labels: T

    source_mean: T
    source_std: T
    source_mean_by_subject: T
    source_std_by_subject: T
    target_mean: T
    target_std: T
    all_mean: T
    all_std: T

    train_ratio: float
    all_shuffled_indices: List[int]
    train_indices: List[int]
    dev_indices: List[int]
    train_on_source: bool
    seed: bool

    quantized_source_data: torch.LongTensor
    quantized_target_data: torch.LongTensor
    quantized_all_data: torch.LongTensor

    def __init__(
            self,
            data_path: str,
            target_subject: str,
            train_ratio: float = 0.8,
            train_on_source: bool = True,
            seed: int = 2022,
            do_quantization: bool = False,
            quantization_level: int = 100,
            do_normalization: bool = False,
    ):
        super().__init__()
        self.data_path: str = data_path
        with open(self.data_path, 'rb') as f:
            self.dataset = pickle.load(f)
        assert target_subject in self.dataset.keys(), KeyError(
            f'Target subject NOT in dataset: [{self.dataset.keys()}]')
        self.target_subject: str = target_subject
        self.train_ratio: float = train_ratio
        self.train_on_source: bool = train_on_source
        self.do_quantization: bool = do_quantization
        self.quantization_level: int = quantization_level
        self.seed: int = seed
        self._unpack_data()
        self._train_dev_split()

        if do_normalization:
            self._normalize()

        if self.do_quantization:
            self._quantize(level=self.quantization_level)

    def _unpack_data(self):
        source_data = list()
        source_class_label = list()
        source_domain_label = list()

        for key in self.dataset.keys():
            if key == self.target_subject:
                self.target_data = torch.tensor(self.dataset[key]['data'], dtype=torch.float32)
                self.target_class_labels = torch.tensor(self.dataset[key]['label'] + 1, dtype=torch.long)
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
        self.source_data_by_subject = torch.cat([data.unsqueeze(1) for data in source_data], dim=1)
        self.all_data_by_subject = torch.cat([self.source_data_by_subject, self.target_data.unsqueeze(1)], dim=1)
        self.all_domain_labels_by_subject = torch.stack(source_domain_label + [self.target_domain_labels], dim=-1)
        self.unified_labels = self.target_class_labels

    def _train_dev_split(self):
        self.all_shuffled_indices = list(range(len(self.source_class_labels))) if self.train_on_source else list(range(len(self.all_class_labels)))
        np.random.seed(seed=self.seed)
        np.random.shuffle(self.all_shuffled_indices)
        self.train_indices = self.all_shuffled_indices[:int(self.train_ratio * len(self.all_shuffled_indices))]
        self.dev_indices = self.all_shuffled_indices[int(self.train_ratio * len(self.all_shuffled_indices)):]

    def _normalize(self):
        self.source_mean = self.source_data.mean(dim=0)
        self.source_std = self.source_data.std(dim=0)
        self.source_mean_by_subject = self.source_data_by_subject.mean(dim=0)  # [source_subjects, num_features]
        self.source_std_by_subject = self.source_data_by_subject.std(dim=0)  # [source_subjects, num_features]
        self.target_mean = self.target_data.mean(dim=0)
        self.target_std = self.target_data.std(dim=0)
        self.all_mean = self.all_data.mean(dim=0)
        self.all_std = self.all_data.std(dim=0)
        self.all_mean_by_subject = self.all_data_by_subject.mean(dim=0)
        self.all_std_by_subject = self.all_data_by_subject.std(dim=0)

        self.source_data = (self.source_data - self.source_mean) / self.source_std
        self.target_data = (self.target_data - self.target_mean) / self.target_std  # be careful when you divide by std, std may be a feature
        # self.source_data = self.source_data - self.source_mean
        # self.target_data = self.target_data - self.target_mean
        self.all_data = (self.all_data - self.all_mean) / self.all_std
        self.source_data_by_subject = (self.source_data_by_subject - self.source_mean_by_subject) / self.source_std_by_subject
        self.all_data_by_subject = (self.all_data_by_subject - self.all_mean_by_subject) / self.all_std_by_subject

    def _quantize(self, level: int):
        self.quantized_source_data = ((self.source_data - self.source_data.min()) * level / (self.source_data.max() - self.source_data.min() + 1)).to(torch.long)
        self.quantized_target_data = ((self.target_data - self.target_data.min()) * level / (self.target_data.max() - self.target_data.min() + 1)).to(torch.long)
        self.quantized_all_data = ((self.all_data - self.all_data.min()) * level / (self.all_data.max() - self.all_data.min() + 1)).to(torch.long)


# class SeedDatasetForBaseline(SeedDataset):
#     def __init__(
#             self,
#             data_path: str,
#             target_subject: str,
#             train_ratio: float,
#             seed: int,
#             mode: str = 'train',
#
#             train_on_source: bool = True,
#             do_normalization: bool = True,
#             do_quantization: bool = False,
#             quantization_level: int = 100):
#         super().__init__(
#             data_path=data_path,
#             target_subject=target_subject,
#             train_ratio=train_ratio,
#             seed=seed,
#
#             train_on_source=train_on_source,
#             do_normalization=do_normalization,
#             do_quantization=do_quantization,
#             quantization_level=quantization_level
#         )
#         self.mode = mode
#         self.length = len(self.train_indices) if mode == 'train' else len(self.dev_indices)
#
#     def __len__(self):
#         return self.length
#
#     def __getitem__(self, item: int):
#










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
        sample: DANNSample
        if self.mode == 'train':
            sample = DANNSample(
                data=self.all_data[item, :],
                class_label=self.all_class_labels[item],
                domain_label=self.all_domain_labels[item],
            )

        else:
            sample = DANNSample(
                data=self.target_data[item, :],
                class_label=self.target_class_labels[item],
                domain_label=self.target_domain_labels[item]
            )

        return sample


class SeedDatasetForDAN(SeedDataset):
    def __init__(self, data_path: str, target_subject: str, mode: str):
        super().__init__(data_path, target_subject)
        self.mode = mode

    def __len__(self):
        if self.mode == 'train':
            return len(self.source_domain_labels)
        else:
            return len(self.target_domain_labels)

    def __getitem__(self, item: int):
        sample: DANSample
        if self.mode == 'train':
            target_item: int = int(item % len(self.target_domain_labels))
            sample = DANSample(
                source_data=self.source_data[item, :],
                source_label=self.source_class_labels[item],
                target_data=self.target_data[target_item, :],
                target_label=self.target_class_labels[target_item],
            )

        else:
            sample = DANSample(
                source_data=torch.zeros_like(self.source_data[item, :]),
                source_label=torch.zeros_like(self.source_class_labels[item]),
                target_data=self.target_data[item, :],
                target_label=self.target_class_labels[item],
            )

        return sample


class SeedDatasetForMMDAAE(SeedDataset):
    def __init__(
            self,
            data_path: str,
            target_subject: str,
            mode: str,
    ):
        super().__init__(data_path=data_path, target_subject=target_subject)
        self.mode = mode

    def __len__(self):
        return len(self.unified_labels)

    def __getitem__(self, item: int) -> SEEDSample:
        if self.mode == 'train':
            # train_batch.data.size() = [batch_size, num_train_domain, num_features]
            sample = SEEDSample(
                data=self.all_data_by_subject[item],
                # data=(self.source_data_by_subject[item]-self.source_mean_by_subject)/self.source_std_by_subject,
                # data=(self.all_data_by_subject[item]-self.all_mean)/self.all_std,
                label=self.unified_labels[item],
                )

        else:
            # dev_batch.data.size() = [batch_size, num_features]
            sample = SEEDSample(
                data=self.target_data[item],
                # data=(self.target_data[item]-self.all_mean)/self.all_std,
                # data=(self.target_data[item]-self.all_mean)/self.all_std,
                # data=(self.source_data_by_subject[item, 0, :]-self.source_mean_by_subject[0])/self.source_std_by_subject[0],
                label=self.unified_labels[item]
            )
        return sample


# TODO: implement unsupervised dataset for pre-training
class SeedDatasetForTransformerPreTraining(SeedDataset):
    def __init__(
            self,
            data_path: str,
            target_subject: str,
            mode: str,
            train_ratio: float = 0.8,
            train_on_source: bool = True,
            seed: int = 2022,
            do_quantization: bool = False,
            quantization_level: int = 100,
            do_normalization: bool = False,
    ):
        super().__init__(
            data_path=data_path,
            target_subject=target_subject,
            train_ratio=train_ratio,
            train_on_source=train_on_source,
            seed=seed,
            do_quantization=do_quantization,
            quantization_level=quantization_level,
            do_normalization=do_normalization,
        )
        self.mode = mode
        self.length = len(self.train_indices) if mode == 'train' else len(self.dev_indices)
        self.train_on_source = train_on_source
        self.do_quantization = do_quantization

    def __len__(self):
        return self.length

    def __getitem__(self, item: int) -> TransformerPreTrainingSample:
        sample: TransformerPreTrainingSample
        item = self.train_indices[item] if self.mode == 'train' else self.dev_indices[item]
        target_item: int = int(item % len(self.target_domain_labels))
        if self.train_on_source:
            if self.do_quantization:
                sample = TransformerPreTrainingSample(
                    source_data=self.quantized_source_data[item, :],
                    target_data=self.quantized_target_data[target_item, :],
                )
            else:
                sample = TransformerPreTrainingSample(
                    source_data=self.source_data[item, :],
                    target_data=self.target_data[target_item, :],
                )
        else:
            if self.do_quantization:
                sample = TransformerPreTrainingSample(
                    source_data=self.quantized_all_data[item, :],
                    target_data=self.quantized_target_data[target_item, :],
                )
            else:
                sample = TransformerPreTrainingSample(
                    source_data=self.all_data[item, :],
                    target_data=self.target_data[target_item, :],
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
            item = self.shuffled_idx[self.train_len + item]
        sample = DANNSample(
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
            item = self.shuffled_idx[self.train_len + item]

        sample = DANNSample(
            data=self.target_data[item, :],
            class_label=self.target_class_labels[item],
            domain_label=self.target_domain_labels[item]
        )

        return sample


if __name__ == "__main__":
    # train_set = SeedDatasetForMMDAAE(
    #     data_path="data/de_LDS_data.pkl",
    #     target_subject='sub_1',
    #     mode='train'
    # )
    # dev_set = SeedDatasetForMMDAAE(
    #     data_path="data/data.pkl",
    #     target_subject='sub_1',
    #     mode='dev'
    # )
    train_set = SeedDatasetForTransformerPreTraining(
        data_path="data/de_LDS_data.pkl",
        target_subject='sub_1',
        mode='train',
        train_ratio=0.8,
        train_on_source=False,
        seed=2022,
        do_quantization=True,
        quantization_level=200,

    )
    dev_set = SeedDatasetForTransformerPreTraining(
        data_path="data/de_LDS_data.pkl",
        target_subject='sub_1',
        mode='dev',
        train_ratio=0.8,
        train_on_source=False,
        seed=2022,
        do_quantization=True,
        quantization_level=200,
    )

    print("Test OK!")
