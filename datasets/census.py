# %%
"""
Source: https://archive.ics.uci.edu/ml/datasets/US+Census+Data+(1990)
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
        "--train_length", type=int, default=30000, help="length of train dataset"
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
    print("Data loading...")
    config = vars(get_args(debug=False))
    df_raw = pd.read_csv("../data/raw/USCensus1990.data.txt")
    df_raw = df_raw.drop(columns="caseid")
    # %%
    print("Data splitting...")
    np.random.seed(config["seed"])
    idx = np.random.choice(
        len(df_raw), config["train_length"] + config["test_length"], replace=False
    )

    train = df_raw.iloc[idx[: config["train_length"]]].reset_index(drop=True)
    test = df_raw.iloc[idx[config["train_length"] :]].reset_index(drop=True)
    # %%
    print("Data saving...")
    train.to_csv("../data/census_train.csv")
    test.to_csv("../data/census_test.csv")

    print(train.shape)
    print(test.shape)


# %%
if __name__ == "__main__":
    main()
# %%
