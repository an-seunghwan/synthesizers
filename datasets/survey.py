#%%
"""
- Source:
https://www.kaggle.com/datasets/census/2013-american-community-survey
- Codebook:
https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMSDataDict13.txt
"""
#%%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
#%%%
def main():
    #%%
    print("Data loading...")
    df = pd.read_csv('../data/raw/ss13husa.csv')
    df = df[df["ST"] == 6] # .California/CA
    df.reset_index().drop(columns=['index']).to_csv('../data/raw/survey.csv')
    #%%
    print("Data pre-processing...")
    colnames = [
        "ACR",
        "BATH",
        "BLD",
        "BROADBND",
        "BUS",
        "COMPOTHX",
        "DIALUP",
        "DSL",
        "FIBEROP",
        "FS",
        "HANDHELD",
        "HFL",
        "LAPTOP",
        "MODEM",
        "OTHSVCEX",
        "REFR",
        "RWAT",
        "SATELLITE",
        "SINK",
        "STOV",
        "TEL",
        "TEN",
        "TOIL",
        "VEH",
        "YBL",
        "FES",
        "FPARC",
        "HHL",
        "HHT",
        "HUGCL",
        "HUPAC",
        "HUPAOC",
        "HUPARC",
        "KIT",
        "LNGI",
        "MULTG",
        "MV",
        "NOC",
        "NPF",
        "NPP",
        "NR",
        "NRC",
        "PARTNER",
        "PLM",
        "PSF",
        "R18",
        "R60",
        "R65",
        "RESMODE",
        "SRNT",
        "SVAL",
        "WIF",
        "WKEXREL",
        "WORKSTAT",
    ]
    """
    eliminate column threshold: less than FSMOCP (75779 nan values)
    """
    with open("../data/survey_colnames.txt", "w") as f:
        for s in colnames:
            f.write(str(s) + "\n")
    df[colnames].isna().sum(axis=0).sort_values()
    
    """dropna"""
    df = df[colnames].dropna()
    df = df.reset_index().drop(columns=['index'])
    #%%
    print("Data splitting...")
    np.random.seed(42)
    idx = np.random.choice(
        len(df), 
        len(df), 
        replace=False)
    #%%
    train = df.iloc[idx[:60000]].reset_index(drop=True)
    test = df.iloc[idx[60000:]].reset_index(drop=True)
    #%%
    print("Data saving...")
    train.to_csv("../data/survey_train.csv")
    test.to_csv("../data/survey_test.csv")
    
    print(train.shape)
    print(test.shape)
    #%%
    # for dis in colnames:
    #     if len(df[dis].unique()) == 1:
    #         print(dis)
#%%
if __name__ == "__main__":
    main()
#%%