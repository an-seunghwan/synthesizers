# %%
import pandas as pd
from sklearn.utils import shuffle
from collections import namedtuple

OutputInfo = namedtuple("OutputInfo", "dim")

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


# %%
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.p = data.size(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx])


# %%
def build_dataset(config):
    df_train = pd.read_csv(
        f'../data/{config["dataset"]}_train.csv', index_col=0
    ).astype(int)
    df_test = pd.read_csv(f'../data/{config["dataset"]}_test.csv', index_col=0).astype(
        int
    )
    colnames = df_train.columns

    print("Unique category numbers...")
    unique_num = []
    for dis in colnames:
        unique_num.append(len(df_train[dis].unique()))
    print(unique_num)

    discrete_dicts = []
    discrete_dicts_reverse = []
    for dis in colnames:
        discrete_dict = {x: i for i, x in enumerate(sorted(df_train[dis].unique()))}
        discrete_dicts_reverse.append(
            {i: x for i, x in enumerate(sorted(df_train[dis].unique()))}
        )
        df_train[dis] = df_train[dis].apply(lambda x: discrete_dict.get(x))
        discrete_dicts.append(discrete_dict)

    df_dummy = []
    for d in colnames:
        df_dummy.append(pd.get_dummies(df_train[d], prefix=d))

    OutputInfo_list = []
    for d, dummy in zip(colnames, df_dummy):
        OutputInfo_list.append(OutputInfo(dummy.shape[1]))

    df_train_ = pd.concat(df_dummy, axis=1)
    dataset_ = MyDataset(torch.from_numpy(df_train_.to_numpy()).to(torch.float32))

    return (
        dataset_,
        df_train,
        df_test,
        OutputInfo_list,
        discrete_dicts,
        discrete_dicts_reverse,
        colnames,
    )


# %%
"""MNIST"""
# dataset_ = datasets.MNIST(
#     root='data/',
#     train=True,
#     transform=transforms.ToTensor(),
#     download=True)
# test_dataset_ = datasets.MNIST(
#     root='data/',
#     train=False,
#     transform=transforms.ToTensor(),
#     download=True)
# %%
