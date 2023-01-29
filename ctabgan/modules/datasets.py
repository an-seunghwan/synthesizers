#%%
import tqdm
import os
import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset
#%%
def generate_dataset(config):
    
    if config["dataset"] == 'loan':
        df = pd.read_csv('./data/Bank_Personal_Loan_Modelling.csv')
        df = df.sample(frac=1, random_state=1).reset_index(drop=True)
        df = df.drop(columns=['ID'])
        
        continuous = ['CCAvg', 'Mortgage', 'Income', 'Experience', 'Age']
        df = df[continuous].iloc[:4000]

    elif config["dataset"] == 'adult':
        df = pd.read_csv('./data/adult.csv')
        df = df.sample(frac=1, random_state=1).reset_index(drop=True)
        df = df[(df == '?').sum(axis=1) == 0]
        df['income'] = df['income'].map({'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1})
        
        continuous = ['income', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
        df = df[continuous].iloc[:4000]
        
    elif config["dataset"] == 'covtype':
        df = pd.read_csv('./data/covtype.csv')
        df = df.sample(frac=1, random_state=5).reset_index(drop=True)
        
        continuous = [
            'Horizontal_Distance_To_Hydrology', 
            'Vertical_Distance_To_Hydrology',
            'Horizontal_Distance_To_Roadways',
            'Horizontal_Distance_To_Fire_Points',
            'Elevation', 
            'Aspect', 
            'Slope', 
            'Cover_Type']
        df = df[continuous]
        df = df.dropna(axis=0)
        df = df.iloc[2000:]
        
    else:
        raise ValueError('Not supported dataset!')    

    return df
#%%