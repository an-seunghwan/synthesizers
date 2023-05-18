#%%
import pandas as pd
from sklearn.utils import shuffle
from collections import namedtuple
OutputInfo = namedtuple('OutputInfo', 'dim')

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
#%%
class MyDataset(Dataset): 
    def __init__(self, data):
        self.data = data
        self.p = data.size(1)
    def __len__(self): 
        return len(self.data)
    def __getitem__(self, idx): 
        return torch.FloatTensor(self.data[idx])
#%%
def build_dataset(config):
    if config["dataset"] == "mnist":
        """MNIST"""
        dataset_ = datasets.MNIST(
            root='data/',
            train=True,
            transform=transforms.ToTensor(),
            download=True)
        test_dataset_ = datasets.MNIST(
            root='data/',
            train=False,
            transform=transforms.ToTensor(),
            download=True)
        return dataset_, test_dataset_
    
    elif config["dataset"] == "census":
        """Census"""
        df = pd.read_csv('../data/census_preprocessed.csv', index_col=0)
        df = shuffle(df, random_state=config["seed"])
        df = df.astype(int).astype(str)
        with open('../assets/census_colnames.txt', 'r') as f:
            colnames = f.read().splitlines() 
        
        test_len = 2000
        train = df.iloc[:-test_len]
        test = df.iloc[-test_len:]
        
        discrete_dicts = []
        discrete_dicts_reverse = []
        for dis in colnames:
            discrete_dict = {x:i for i,x in enumerate(sorted(df[dis].unique()))}
            discrete_dicts_reverse.append({i:x for i,x in enumerate(sorted(df[dis].unique()))})
            df[dis] = df[dis].apply(lambda x: discrete_dict.get(x))
            discrete_dicts.append(discrete_dict)
        
        df_dummy = []
        for d in colnames:
            df_dummy.append(pd.get_dummies(df[d], prefix=d))
        
        OutputInfo_list = []
        for d, dummy in zip(colnames, df_dummy):
            OutputInfo_list.append(OutputInfo(dummy.shape[1]))
        
        df = pd.concat(df_dummy, axis=1)
        dataset_ = MyDataset(torch.from_numpy(df.iloc[:-test_len].to_numpy()).to(torch.float32))
        
        return dataset_, train, test, OutputInfo_list, discrete_dicts, discrete_dicts_reverse, colnames
        
    else:
        raise ValueError('Not supported dataset!')
#%%