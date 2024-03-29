# %%
import os

# os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# %%
import numpy as np
import pandas as pd
import tqdm
from PIL import Image
import matplotlib.pyplot as plt

# plt.switch_backend('agg')

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset

from evaluation.simulation import set_random_seed
from modules.model import TVAE
from modules.datasets import generate_dataset

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
    tags=["TVAE", "Ablation"],
)
# %%
import argparse


def get_args(debug):
    parser = argparse.ArgumentParser("parameters")

    parser.add_argument("--num", type=int, default=6, help="model version")
    parser.add_argument(
        "--dataset",
        type=str,
        default="credit",
        help="Dataset options: only supports credit dataset!",
    )

    if debug:
        return parser.parse_args(args=[])
    else:
        return parser.parse_args()


# %%
def main():
    # %%
    config = vars(get_args(debug=False))  # default configuration

    """model load"""
    artifact = wandb.use_artifact(
        "anseunghwan/DistVAE/TVAE_{}:v{}".format(config["dataset"], config["num"]),
        type="model",
    )
    for key, item in artifact.metadata.items():
        config[key] = item
    model_dir = artifact.download()

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
    """model"""
    model = TVAE(config, device).to(device)

    if config["cuda"]:
        model_name = [x for x in os.listdir(model_dir) if x.endswith("pth")][0]
        model.load_state_dict(torch.load(model_dir + "/" + model_name))
    else:
        model_name = [x for x in os.listdir(model_dir) if x.endswith("pth")][0]
        model.load_state_dict(
            torch.load(model_dir + "/" + model_name, map_location=torch.device("cpu"))
        )

    model.eval()
    # %%
    """dataset"""
    (
        dataset,
        dataloader,
        transformer,
        train,
        test,
        continuous,
        discrete,
    ) = generate_dataset(config, device, random_state=0)

    config["input_dim"] = transformer.output_dimensions
    # %%
    if config["dataset"] == "credit":
        # %%
        latents = []
        dataloader_ = DataLoader(
            dataset, batch_size=config["batch_size"], shuffle=False
        )
        for (x_batch,) in tqdm.tqdm(iter(dataloader_), desc="inner loop"):
            if config["cuda"]:
                x_batch = x_batch.cuda()

            with torch.no_grad():
                mean, logvar = model.get_posterior(x_batch)
            latents.append(mean)
        latents = torch.cat(latents, dim=0).numpy()
        labels = train["TARGET"].to_numpy()
        # %%
        plt.figure(figsize=(5, 5))
        plt.scatter(
            latents[labels == 0, 0],
            latents[labels == 0, 1],
            s=25,
            c="blue",
            alpha=0.5,
            label="0",
        )
        plt.scatter(
            latents[labels == 1, 0],
            latents[labels == 1, 1],
            s=25,
            c="red",
            alpha=0.5,
            label="1",
        )
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel("$z_1$", fontsize=18)
        plt.ylabel("$z_2$", fontsize=18)
        plt.legend(fontsize=16)

        plt.tight_layout()
        plt.savefig(
            "./assets/{}/{}_latent_space.png".format(
                config["dataset"], config["dataset"]
            )
        )
        plt.show()
        plt.close()
        # %%
        plt.figure(figsize=(5, 5))
        plt.bar([0, 1], [(labels == 0).mean(), (labels == 1).mean()])
        plt.xticks([0, 1], ["0", "1"])
        plt.ylim(0, 1)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel("label", fontsize=18)
        plt.ylabel("proportion", fontsize=18)

        plt.tight_layout()
        plt.savefig(
            "./assets/{}/{}_label_ratio.png".format(
                config["dataset"], config["dataset"]
            )
        )
        plt.show()
        plt.close()
        # %%
    else:
        raise ValueError("Not supported dataset!")
    # %%
    wandb.run.finish()


# %%
if __name__ == "__main__":
    main()
# %%
