# %%
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# %%
import numpy as np
import pandas as pd
import tqdm
from PIL import Image
import matplotlib.pyplot as plt

plt.switch_backend("agg")

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset

from evaluation.simulation import set_random_seed
from modules.model import TVAE
from modules.shadow_datasets import generate_dataset
from modules.train import train

# %%
import sys
import subprocess

try:
    import wandb
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    with open("./wandb_api.txt", "r") as f:
        key = f.readlines()
    subprocess.run(["wandb", "login"], input=key[0], encoding="utf-8")
    import wandb

run = wandb.init(
    project="DistVAE",
    entity="anseunghwan",
    tags=["TVAE", "Privacy"],
)
# %%
import argparse
import ast


def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError('Argument "%s" is not a list' % (s))
    return v


def get_args(debug):
    parser = argparse.ArgumentParser("parameters")

    parser.add_argument(
        "--seed", type=int, default=1, help="seed for repeatable results"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="covtype",
        help="Dataset options: covtype, credit, loan, adult, cabs, kings",
    )

    parser.add_argument(
        "--latent_dim", default=2, type=int, help="the dimension of latent variable"
    )

    # optimization options
    parser.add_argument("--epochs", default=200, type=int, help="maximum iteration")
    parser.add_argument("--batch_size", default=256, type=int, help="batch size")
    parser.add_argument("--lr", default=0.005, type=float, help="learning rate")
    parser.add_argument(
        "--weight_decay", default=1e-5, type=float, help="weight decay parameter"
    )
    parser.add_argument(
        "--sigma_range",
        default=[0.1, 1],
        type=arg_as_list,
        help="range of observational noise",
    )

    if debug:
        return parser.parse_args(args=[])
    else:
        return parser.parse_args()


# %%
def main():
    # %%
    config = vars(get_args(debug=False))  # default configuration
    config["cuda"] = torch.cuda.is_available()
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    wandb.config.update(config)

    set_random_seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["cuda"]:
        torch.cuda.manual_seed(config["seed"])
    # %%
    """Load shadow data"""
    K = 1  # the number of shadow models
    shadow_data = []
    for s in range(K):
        df = pd.read_csv(
            f'./privacy/{config["dataset"]}/train_{config["seed"]}_synthetic{s}.csv',
            index_col=0,
        )
        shadow_data.append(df)
    shadow_data_test = []
    for s in range(K):
        df = pd.read_csv(
            f'./privacy/{config["dataset"]}/test_{config["seed"]}_synthetic{s}.csv',
            index_col=0,
        )
        shadow_data_test.append(df)
    # %%
    for k in range(len(shadow_data)):
        print(f"Training {k}th shadow model...\n")

        dataset, dataloader, transformer, _, _, _, _ = generate_dataset(
            config, shadow_data[k], shadow_data_test[k], device, random_state=0
        )

        config["input_dim"] = transformer.output_dimensions

        model = TVAE(config, device).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
        )

        model.train()

        for epoch in range(config["epochs"]):
            logs = train(
                transformer.output_info_list,
                dataset,
                dataloader,
                model,
                config,
                optimizer,
                device,
            )

            print_input = "[epoch {:03d}]".format(epoch + 1)
            print_input += "".join(
                [", {}: {:.4f}".format(x, np.mean(y)) for x, y in logs.items()]
            )
            print(print_input)

            """update log"""
            wandb.log({x: np.mean(y) for x, y in logs.items()})

        """model save"""
        torch.save(
            model.state_dict(), "./assets/shadow_TVAE_{}.pth".format(config["dataset"])
        )
        artifact = wandb.Artifact(
            "shadow_TVAE_{}".format(config["dataset"]), type="model", metadata=config
        )  # description=""
        artifact.add_file("./assets/shadow_TVAE_{}.pth".format(config["dataset"]))
        artifact.add_file("./main.py")
        artifact.add_file("./modules/model.py")
        wandb.log_artifact(artifact)
    # %%
    wandb.run.finish()


# %%
if __name__ == "__main__":
    main()
# %%
