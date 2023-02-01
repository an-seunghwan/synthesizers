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

from collections import namedtuple

OutputInfo = namedtuple('OutputInfo', ['dim', 'activation_fn'])
#%%
def generate_dataset(config, device, random_state=0):
    
    if config["dataset"] == 'covtype':
        df = pd.read_csv('../data/covtype.csv')
        df = df.sample(frac=1, random_state=0).reset_index(drop=True)
        df = df.dropna(axis=0)
        df = df.iloc[:50000]
        
        continuous = [
            'Elevation', # target variable
            'Aspect', 
            'Slope',
            'Horizontal_Distance_To_Hydrology', 
            'Vertical_Distance_To_Hydrology',
            'Horizontal_Distance_To_Roadways',
            'Hillshade_9am',
            'Hillshade_Noon',
            'Hillshade_3pm',
            'Horizontal_Distance_To_Fire_Points',
            ]
        discrete = [
            'Cover_Type', # target variable
        ]
        df = df[continuous + discrete]
        
        # one-hot
        df_dummy = []
        for d in discrete:
            df_dummy.append(pd.get_dummies(df[d], prefix=d))
        df = pd.concat([df.drop(columns=discrete)] + df_dummy, axis=1)
        
        train = df.iloc[:45000]
        test = df.iloc[45000:]
        
    elif config["dataset"] == 'credit':
        df = pd.read_csv('../data/application_train.csv')
        df = df.sample(frac=1, random_state=0).reset_index(drop=True)
        
        continuous = [
            'AMT_INCOME_TOTAL', 
            'AMT_CREDIT', # target variable
            'AMT_ANNUITY',
            'AMT_GOODS_PRICE',
            'REGION_POPULATION_RELATIVE', 
            'DAYS_BIRTH', 
            'DAYS_EMPLOYED', 
            'DAYS_REGISTRATION',
            'DAYS_ID_PUBLISH',
            'OWN_CAR_AGE',
        ]
        discrete = [
            'NAME_CONTRACT_TYPE',
            'CODE_GENDER',
            # 'FLAG_OWN_CAR',
            'FLAG_OWN_REALTY',
            'NAME_TYPE_SUITE',
            'NAME_INCOME_TYPE',
            'NAME_EDUCATION_TYPE',
            'NAME_FAMILY_STATUS',
            'NAME_HOUSING_TYPE',
            'TARGET', # target variable
        ]
        df = df[continuous + discrete]
        df = df.dropna(axis=0)
        df = df.iloc[:50000]
        
        # one-hot
        df_dummy = []
        for d in discrete:
            df_dummy.append(pd.get_dummies(df[d], prefix=d))
        df = pd.concat([df.drop(columns=discrete)] + df_dummy, axis=1)
        
        train = df.iloc[:45000]
        test = df.iloc[45000:]
        
    elif config["dataset"] == 'loan':
        df = pd.read_csv('../data/Bank_Personal_Loan_Modelling.csv')
        df = df.sample(frac=1, random_state=0).reset_index(drop=True)
        
        continuous = [
            'Age', # target variable
            'Experience',
            'Income', 
            'CCAvg',
            'Mortgage',
        ]
        discrete = [
            'Family',
            'Personal Loan', # target variable
            'Securities Account',
            'CD Account',
            'Online',
            'CreditCard'
        ]
        df = df[continuous + discrete]
        df = df.dropna()
        
        # one-hot
        df_dummy = []
        for d in discrete:
            df_dummy.append(pd.get_dummies(df[d], prefix=d))
        df = pd.concat([df.drop(columns=discrete)] + df_dummy, axis=1)
        
        train = df.iloc[:4000]
        test = df.iloc[4000:]
        
    elif config["dataset"] == 'adult':
        df = pd.read_csv('../data/adult.csv')
        df = df.sample(frac=1, random_state=0).reset_index(drop=True)
        df = df[(df == '?').sum(axis=1) == 0]
        
        continuous = [
            'age', # target variable
            'educational-num',
            'capital-gain', 
            'capital-loss', 
            'hours-per-week',
        ]
        discrete = [
            'workclass',
            'education',
            'marital-status',
            'occupation',
            'relationship',
            'race',
            'gender',
            'native-country',
            'income', # target variable
        ]
        df = df[continuous + discrete]
        df = df.dropna()
        
        # one-hot
        df_dummy = []
        for d in discrete:
            df_dummy.append(pd.get_dummies(df[d], prefix=d))
        df = pd.concat([df.drop(columns=discrete)] + df_dummy, axis=1)
        
        train = df.iloc[:40000]
        test = df.iloc[40000:]
        
    elif config["dataset"] == 'cabs':
        df = pd.read_csv('../data/sigma_cabs.csv')
        df = df.sample(frac=1, random_state=0).reset_index(drop=True)
        df = df.dropna().reset_index().drop(columns='index')
        
        continuous = [
            'Trip_Distance', # target variable
            'Life_Style_Index', 
            'Customer_Rating', 
            'Var1',
            'Var2',
            'Var3',
        ]
        discrete = [
            'Type_of_Cab',
            'Customer_Since_Months',
            'Confidence_Life_Style_Index',
            'Destination_Type',
            'Cancellation_Last_1Month',
            'Gender',
            'Surge_Pricing_Type', # target variable
        ]
        df = df[continuous + discrete]
        
        # one-hot
        df_dummy = []
        for d in discrete:
            df_dummy.append(pd.get_dummies(df[d], prefix=d))
        df = pd.concat([df.drop(columns=discrete)] + df_dummy, axis=1)
        
        train = df.iloc[:40000]
        test = df.iloc[40000:]
        
    elif config["dataset"] == 'kings':
        df = pd.read_csv('../data/kc_house_data.csv')
        df = df.sample(frac=1, random_state=0).reset_index(drop=True)
        
        continuous = [
            'price', 
            'sqft_living',
            'sqft_lot',
            'sqft_above',
            'sqft_basement',
            'yr_built',
            'yr_renovated',
            'lat',
            'long', # target variable
            'sqft_living15',
            'sqft_lot15',
        ]
        discrete = [
            'bedrooms',
            'bathrooms',
            'floors',
            'waterfront',
            'view',
            'condition', # target variable
            'grade', 
        ]
        df = df[continuous + discrete]
        
        # one-hot
        df_dummy = []
        for d in discrete:
            df_dummy.append(pd.get_dummies(df[d], prefix=d))
        df = pd.concat([df.drop(columns=discrete)] + df_dummy, axis=1)
        
        train = df.iloc[:20000]
        test = df.iloc[20000:]
        
    else:
        raise ValueError('Not supported dataset!')    

    # Output Information
    OutputInfo_list = []
    for c in continuous:
        OutputInfo_list.append(OutputInfo(1, 'MSE'))
    for d, dummy in zip(discrete, df_dummy):
        OutputInfo_list.append(OutputInfo(dummy.shape[1], 'softmax'))

    # standardization
    train_data = train.copy()
    train_data[continuous] -= train_data[continuous].mean(axis=0)
    train_data[continuous] /= train_data[continuous].std(axis=0)
    train_data = train_data.to_numpy().astype('float32')
    
    dataset = TensorDataset(torch.from_numpy(train_data).to(device))
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, drop_last=False)
    
    return OutputInfo_list, dataset, dataloader, train, test, continuous, discrete
#%%