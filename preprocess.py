#%%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
#%%%
"""Pre-Processing census dataset"""
df = pd.read_csv('./data/ss13husa.csv')
df = df[df["ST"] == 6] # .California/CA
df.reset_index().drop(columns=['index']).to_csv('./data/census.csv')
#%%
# for x in df.columns:
#     print('"{}",'.format(x))
colnames = [
    # "ACCESS", 
    "ACR",
    # "AGS",
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
    # "RWATPR",
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
    # "SSMC",
    "SVAL",
    "WIF",
    "WKEXREL",
    "WORKSTAT",
]
"""
eliminate column threshold: less than FSMOCP (75779 nan values)
"""
with open("./assets/census_colnames.txt", "w") as f:
    for s in colnames:
        f.write(str(s) +"\n")
df[colnames].isna().sum(axis=0).sort_values()
#%%
"""dropna"""
df = df[colnames].dropna()
df = df.reset_index().drop(columns=['index'])
df.to_csv('./data/census_preprocessed.csv')
#%%
# for dis in colnames:
#     if len(df[dis].unique()) == 1:
#         print(dis)
#%%