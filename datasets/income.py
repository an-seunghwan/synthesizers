# %%
"""
Source: http://archive.ics.uci.edu/dataset/117/census+income+kdd
"""
# %%
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np

# %%
import argparse


def get_args(debug):
    parser = argparse.ArgumentParser("parameters")

    parser.add_argument(
        "--seed", type=int, default=42, help="seed for repeatable results"
    )
    parser.add_argument(
        "--train_length", type=int, default=45000, help="length of train dataset"
    )
    parser.add_argument(
        "--test_length", type=int, default=5000, help="length of test dataset"
    )

    if debug:
        return parser.parse_args(args=[])
    else:
        return parser.parse_args()


# %%
def main():
    # %%
    with open("../data/raw/census-income-columns.txt", "r") as f:
        columns = f.readlines()
    columns = [x.split(":")[0] for x in columns]
    #%%
    print("Data loading...")
    config = vars(get_args(debug=True))
    df_raw = pd.read_csv("../data/raw/census-income.data", header=None)
    df_raw.columns = columns
    df_raw = df_raw.drop(columns=[
        # drop continuous columns
        "age",
        "wage per hour",
        "capital gains",
        "capital losses",
        "dividends from stocks",
        "num persons worked for employer",
        "num persons worked for employer",
        "weeks worked in year",
        "instance weight",
        
        # drop columns having a lot of "Not in universe" 
        "class of worker",
        "enroll in edu inst last wk",
        "major industry code",
        "major occupation code",
        "member of a labor union",
        "reason for unemployment",
        "region of previous residence",
        "state of previous residence",
        "migration code-move within reg",
        "own business or self employed",
        ]) 
    
    for col in df_raw.columns:
        df_raw[col] = df_raw[col].apply(lambda x: "Not in universe" if "Not in universe" in str(x) else x)

    for col in df_raw.columns:    
        df_raw = df_raw[df_raw[col] != "Not in universe"]
    #%%
    test_raw = pd.read_csv("../data/raw/census-income.test", header=None)
    test_raw.columns = columns
    test_raw = test_raw.drop(columns=[
        # drop continuous columns
        "age",
        "wage per hour",
        "capital gains",
        "capital losses",
        "dividends from stocks",
        "num persons worked for employer",
        "num persons worked for employer",
        "weeks worked in year",
        "instance weight",
        
        # drop columns having a lot of "Not in universe" 
        "class of worker",
        "enroll in edu inst last wk",
        "major industry code",
        "major occupation code",
        "member of a labor union",
        "reason for unemployment",
        "region of previous residence",
        "state of previous residence",
        "migration code-move within reg",
        "own business or self employed",
        ]) 
    
    for col in test_raw.columns:
        test_raw[col] = test_raw[col].apply(lambda x: "Not in universe" if "Not in universe" in str(x) else x)

    for col in test_raw.columns:    
        test_raw = test_raw[test_raw[col] != "Not in universe"]
    #%%
    for col in df_raw.columns:
        tmp = sorted(df_raw[col].unique())
        tmp = dict(zip(tmp, range(len(tmp))))
        df_raw[col] = df_raw[col].apply(lambda x: tmp.get(x)) # train
        test_raw[col] = test_raw[col].apply(lambda x: tmp.get(x)) # test
    #%%
    print("Data splitting...")
    np.random.seed(config["seed"])
    idx = np.random.choice(len(df_raw), config["train_length"], replace=False)
    train = df_raw.iloc[idx].reset_index(drop=True)
    
    idx = np.random.choice(len(test_raw), config["test_length"], replace=False)
    test = test_raw.iloc[idx].reset_index(drop=True)
    # %%
    print("Data saving...")
    train.to_csv("../data/income_train.csv")
    test.to_csv("../data/income_test.csv")

    print(train.shape)
    print(test.shape)
    

# %%
if __name__ == "__main__":
    main()
# %%
