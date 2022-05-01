import pickle
import numpy as np

import torch
from torch import nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class SEEDDateset(Dataset):
    # Leave one subject out
    def __init__(self, target, target_only=False) -> None:
        super().__init__()
        if target not in ('sub_0', 'sub_1', 'sub_2', 'sub_3', 'sub_4'):
            raise RuntimeError
        with open('../../data/data.pkl', 'rb') as f:
            data = pickle.load(f)

        if target_only:
            feats = [data[target]['data'].astype(np.float32)]
            labels = [data[target]['label'].astype(np.int64) + 1]  # -1., 0., 1. -> 0, 1, 2
        else:
            feats, labels = [], []
            for item in list(data.keys()):
                feat = data[item]['data'].astype(np.float32)
                label = data[item]['label'].astype(np.int64) + 1  # -1., 0., 1. -> 0, 1, 2
                # PyTorch requires Long
                if item == target:
                    label[:] = 3  # target
                feats.append(feat)
                labels.append(label)

        feats = np.concatenate(feats)
        labels = np.concatenate(labels)

        self.target = target
        self.feats = feats
        self.labels = labels

    def __len__(self, ):
        return len(self.labels)

    def __getitem__(self, index):
        return self.feats[index], self.labels[index]


class ReverseLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_: float):
        assert lambda_ > 0
        ctx.lambda_ = lambda_
        return x

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out.neg() * ctx.lambda_


class DANN(nn.Module):
    def __init__(self, c_label, c_domain, lambda_=1.0) -> None:
        super().__init__()
        self.c_label = c_label
        self.c_domain = c_domain
        self.lambda_ = lambda_

        self.feature_extractor = nn.Sequential(
            nn.BatchNorm1d(310),
            nn.Linear(310, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.label_predictor = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, c_label),
        )

        self.domain_predictor = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, c_domain),
        )

        self.reverse_layer = ReverseLayer()

    def forward(self, x):
        feats = self.feature_extractor(x)
        pred_label = self.label_predictor(feats)

        pred_domain = self.domain_predictor(self.reverse_layer.apply(feats, self.lambda_))

        return pred_label, pred_domain


def metric(pred_label: torch.Tensor, pred_domain: torch.Tensor, gt_label: torch.Tensor, label_mask: torch.Tensor):
    with torch.no_grad():
        pred_label = pred_label.argmax(-1)
        pred_domain = pred_domain.argmax(-1)

        source_acc = (pred_label == gt_label)[label_mask]
        domain_acc = (pred_domain == label_mask)

        return source_acc, domain_acc


if __name__ == '__main__':
    train_loader = DataLoader(SEEDDateset(target='sub_0'), batch_size=64, shuffle=True)
    test_loader = DataLoader(SEEDDateset(target='sub_0', target_only=True), batch_size=64)

    network = DANN(3, 2, 1.0)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(network.parameters(), 1e-3, 0.9, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)

    writer = SummaryWriter()

    global_step = 0

    for n_epoch in range(100):
        source_accs, domain_accs = [], []
        network.train()

        for feats, gt_label in tqdm(train_loader, ncols=0, postfix='E%d' % n_epoch):
            pred_label, pred_domain = network(feats)

            label_mask = (gt_label != 3)
            loss_label = loss_function(pred_label[label_mask], gt_label[label_mask])

            loss_domain = loss_function(pred_label, label_mask.long())

            loss = loss_label + loss_domain * 0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            source_acc, domain_acc = metric(pred_label, pred_domain, gt_label, label_mask)
            source_accs.append(source_acc)
            domain_accs.append(domain_acc)

            if global_step % 50 == 0:
                writer.add_scalar('loss/loss', loss, global_step=global_step)
                writer.add_scalar('loss/loss_label', loss_label, global_step=global_step)
                writer.add_scalar('loss/loss_domain', loss_domain, global_step=global_step)

            global_step += 1

        scheduler.step()

        source_acc = torch.cat(source_accs).float().mean()
        domain_acc = torch.cat(domain_accs).float().mean()
        print('train', source_acc.item(), domain_acc.item())
        writer.add_scalar('train/source_acc', source_acc, global_step=global_step)
        writer.add_scalar('train/domain_acc', domain_acc, global_step=global_step)

        network.eval()
        test_source_accs, test_domain_accs = [], []
        for feats, gt_label in test_loader:
            pred_label, pred_domain = network(feats)
            source_acc, domain_acc = metric(pred_label, pred_domain, gt_label, gt_label > -100)
            source_accs.append(source_acc)
            domain_accs.append(domain_acc)
        source_acc = torch.cat(source_accs).float().mean()
        domain_acc = torch.cat(domain_accs).float().mean()
        print('test', source_acc.item(), domain_acc.item())
        writer.add_scalar('test/source_acc', source_acc, global_step=global_step)
        writer.add_scalar('test/domain_acc', domain_acc, global_step=global_step)

    writer.close()


#
# import pickle
# import numpy as np
# import torch
# from torch.autograd import Variable
# import torch.nn as nn
#
# from torch.utils.data import TensorDataset
# from torch.utils.data import DataLoader
#
# from torch.autograd import Function
# import torch.nn.functional as Func
#
# def data_normal(origin_data):
#     d_min = origin_data.min()
#     if d_min < 0:
#         origin_data += torch.abs(d_min)
#         d_min = origin_data.min()
#     d_max = origin_data.max()
#     dst = d_max-d_min
#     norm_data = (origin_data-d_min).true_divide(dst)
#     return norm_data
#
#
# class feature_class(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.feature = nn.Sequential(
#             nn.BatchNorm1d(310),
#             nn.Linear(310, 128),
#             nn.ReLU(),
#             nn.Linear(128, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Linear(64, 3),
#             nn.Softmax(dim=1)
#         )
#
#     def forward(self, x):
#         y = self.feature(x)
#         return y
#
#
#
# with open('../../data/data.pkl', 'rb') as f:
#     data = pickle.load(f)
#
# for item in list(data.keys()):
#     label = torch.tensor(data[item]['label'] + 1, dtype=torch.long)
#     data[item]['label'] = label
#     data[item]['data'] = torch.tensor(data[item]['data'], dtype=torch.float32)
#
# x_source = {}
# c_source = {}
# d_source = {}
# x_target = {}
# c_target = {}
# d_target = {}
# key = []
#
# for i in range(5):
#     key.append('set_' + str(i))
#     x_source[key[i]] = torch.ones(310)
#     c_source[key[i]] = torch.ones(1)
#
#     for item in list(data.keys()):
#         if item == 'sub_' + str(i):
#             x_target[key[i]] = data[item]['data']
#             c_target[key[i]] = data[item]['label']
#             d_target[key[i]] = torch.tile(torch.tensor([0, 1], dtype=torch.long), (x_target[key[i]].shape[0], 1))
#         else:
#             x_source[key[i]] = torch.vstack((x_source[key[i]], data[item]['data']))
#             c_source[key[i]] = torch.hstack((c_source[key[i]], data[item]['label']))
#     x_source[key[i]] = x_source[key[i]][1:]
#     c_source[key[i]] = c_source[key[i]][1:]
#     d_source[key[i]] = torch.tile(torch.tensor([1, 0], dtype=torch.long), (x_source[key[i]].shape[0], 1))
#
# torch.manual_seed(2022)
# item = 'set_0'
# batch = 100
#
# source_data_ds = TensorDataset(x_source[item],c_source[item],d_source[item])
# source_data_dl = DataLoader(source_data_ds,batch_size=batch,shuffle=True)
#
# target_data_ds = TensorDataset(x_target[item],d_target[item])
# target_data_dl = DataLoader(target_data_ds, batch_size=batch, shuffle=True)
#
#
# learning_rate = 1e-3
#
#
# cep = nn.CrossEntropyLoss()
# nll = nn.NLLLoss()
#
# FC = feature_class()
# FC_opt = torch.optim.SGD(FC.parameters(), lr=learning_rate, momentum=0.9)
#
# alpha = 0.1
# iters = 1000
#
# FC.train()
#
# for i in range(iters):
#     for x, label, domain in source_data_dl:
#         fc = FC(x)
#         # Lc = nll(fc, label.long())
#         Lc = torch.nn.functional.cross_entropy(fc, label.long(), reduction='mean')
#         FC_opt.zero_grad()
#         Lc.backward()
#
#         FC_opt.step()
#
#     # if i % (iters / 20) == 0:
#     print(Lc)
