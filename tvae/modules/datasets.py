# %%
import tqdm
import os
import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset

from .data_transformer import DataTransformer


# %%
def generate_dataset(config, device, random_state=0):
    if config["dataset"] == "covtype":
        df = pd.read_csv("../data/covtype.csv")
        df = df.sample(frac=1, random_state=0).reset_index(drop=True)
        df = df.dropna(axis=0)
        df = df.iloc[:50000]

        continuous = [
            "Elevation",  # target variable
            "Aspect",
            "Slope",
            "Horizontal_Distance_To_Hydrology",
            "Vertical_Distance_To_Hydrology",
            "Horizontal_Distance_To_Roadways",
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
            "Horizontal_Distance_To_Fire_Points",
        ]
        discrete = [
            "Cover_Type",  # target variable
        ]
        df = df[continuous + discrete]

        train = df.iloc[:45000]
        test = df.iloc[45000:]

        transformer = DataTransformer()
        transformer.fit(train, discrete_columns=discrete, random_state=random_state)
        train_data = transformer.transform(train)

    elif config["dataset"] == "credit":
        df = pd.read_csv("../data/application_train.csv")
        df = df.sample(frac=1, random_state=0).reset_index(drop=True)

        continuous = [
            "AMT_INCOME_TOTAL",
            "AMT_CREDIT",  # target variable
            "AMT_ANNUITY",
            "AMT_GOODS_PRICE",
            "REGION_POPULATION_RELATIVE",
            "DAYS_BIRTH",
            "DAYS_EMPLOYED",
            "DAYS_REGISTRATION",
            "DAYS_ID_PUBLISH",
            "OWN_CAR_AGE",
        ]
        discrete = [
            "NAME_CONTRACT_TYPE",
            "CODE_GENDER",
            # 'FLAG_OWN_CAR',
            "FLAG_OWN_REALTY",
            "NAME_TYPE_SUITE",
            "NAME_INCOME_TYPE",
            "NAME_EDUCATION_TYPE",
            "NAME_FAMILY_STATUS",
            "NAME_HOUSING_TYPE",
            "TARGET",  # target variable
        ]
        df = df[continuous + discrete]
        df = df.dropna(axis=0)
        df = df.iloc[:50000]

        train = df.iloc[:45000]
        test = df.iloc[45000:]

        transformer = DataTransformer()
        transformer.fit(train, discrete_columns=discrete, random_state=random_state)
        train_data = transformer.transform(train)

    elif config["dataset"] == "loan":
        df = pd.read_csv("../data/Bank_Personal_Loan_Modelling.csv")
        df = df.sample(frac=1, random_state=0).reset_index(drop=True)

        continuous = [
            "Age",  # target variable
            "Experience",
            "Income",
            "CCAvg",
            "Mortgage",
        ]
        discrete = [
            "Family",
            "Personal Loan",  # target variable
            "Securities Account",
            "CD Account",
            "Online",
            "CreditCard",
        ]
        df = df[continuous + discrete]
        df = df.dropna()

        train = df.iloc[:4000]
        test = df.iloc[4000:]

        transformer = DataTransformer()
        transformer.fit(train, discrete_columns=discrete, random_state=random_state)
        train_data = transformer.transform(train)

    elif config["dataset"] == "adult":
        df = pd.read_csv("../data/adult.csv")
        df = df.sample(frac=1, random_state=0).reset_index(drop=True)
        df = df[(df == "?").sum(axis=1) == 0]

        continuous = [
            "age",  # target variable
            "educational-num",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
        ]
        discrete = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "gender",
            "native-country",
            "income",  # target variable
        ]
        df = df[continuous + discrete]
        df = df.dropna()

        train = df.iloc[:40000]
        test = df.iloc[40000:]

        transformer = DataTransformer()
        transformer.fit(train, discrete_columns=discrete, random_state=random_state)
        train_data = transformer.transform(train)

    elif config["dataset"] == "cabs":
        df = pd.read_csv("../data/sigma_cabs.csv")
        df = df.sample(frac=1, random_state=0).reset_index(drop=True)
        df = df.dropna().reset_index().drop(columns="index")

        continuous = [
            "Trip_Distance",  # target variable
            "Life_Style_Index",
            "Customer_Rating",
            "Var1",
            "Var2",
            "Var3",
        ]
        discrete = [
            "Type_of_Cab",
            "Customer_Since_Months",
            "Confidence_Life_Style_Index",
            "Destination_Type",
            "Cancellation_Last_1Month",
            "Gender",
            "Surge_Pricing_Type",  # target variable
        ]
        df = df[continuous + discrete]

        train = df.iloc[:40000]
        test = df.iloc[40000:]

        transformer = DataTransformer()
        transformer.fit(train, discrete_columns=discrete, random_state=random_state)
        train_data = transformer.transform(train)

    elif config["dataset"] == "kings":
        df = pd.read_csv("../data/kc_house_data.csv")
        df = df.sample(frac=1, random_state=0).reset_index(drop=True)

        continuous = [
            "price",
            "sqft_living",
            "sqft_lot",
            "sqft_above",
            "sqft_basement",
            "yr_built",
            "yr_renovated",
            "lat",
            "long",  # target variable
            "sqft_living15",
            "sqft_lot15",
        ]
        discrete = [
            "bedrooms",
            "bathrooms",
            "floors",
            "waterfront",
            "view",
            "condition",  # target variable
            "grade",
        ]
        df = df[continuous + discrete]

        train = df.iloc[:20000]
        test = df.iloc[20000:]

        transformer = DataTransformer()
        transformer.fit(train, discrete_columns=discrete, random_state=random_state)
        train_data = transformer.transform(train)

    else:
        raise ValueError("Not supported dataset!")

    dataset = TensorDataset(torch.from_numpy(train_data.astype("float32")).to(device))
    dataloader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=True, drop_last=False
    )

    return dataset, dataloader, transformer, train, test, continuous, discrete


# %%
