# %%
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# %%
import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
import importlib

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F

from module.utils import str2bool, postprocess
from module.evaluation import evaluate, DCR_metric, attribute_disclosure, privacyloss
from module.datasets import MyDataset, build_dataset

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

project = "Synthetic(High)"
entity = "anseunghwan"

run = wandb.init(
    project=project,
    entity=entity,
    tags=["privacy"],
)
# %%
import argparse


def get_args(debug):
    parser = argparse.ArgumentParser("parameters")

    parser.add_argument("--num", type=int, default=0, help="model number")
    parser.add_argument("--model", type=str, default="DAAE")
    parser.add_argument(
        "--dataset", type=str, default="census", help="Dataset options: census, survey, income"
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
    model_name = f'DAAE_{config["dataset"]}'
    artifact = wandb.use_artifact(
        f'{entity}/{project}/{model_name}:v{config["num"]}', type="model"
    )
    for key, item in artifact.metadata.items():
        config[key] = item
    model_dir = artifact.download()

    torch.manual_seed(config["seed"])
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    config["cuda"] = torch.cuda.is_available()
    wandb.config.update(config)
    # %%
    out = build_dataset(config)
    dataset = out[0]

    config["p"] = dataset.p
    # %%
    auto_model_module = importlib.import_module("module.model_auto")
    importlib.reload(auto_model_module)
    autoencoder = getattr(auto_model_module, "AutoEncoder")(
        config, config["hidden_dims"], list(reversed(config["hidden_dims"]))
    ).to(device)
    # %%
    model_module = importlib.import_module("module.model")
    importlib.reload(model_module)

    generator = getattr(model_module, "Generator")(
        config["embedding_dim"],
        config["embedding_dim"],
        hidden_sizes=config["hidden_dims_gen"],
        bn_decay=0.1,
    ).to(device)
    # %%
    try:
        autoencoder.load_state_dict(
            torch.load(
                model_dir
                + "/"
                + [
                    x
                    for x in os.listdir(model_dir)
                    if x.endswith("pth") and x.startswith("autoencoder")
                ][0]
            )
        )
        generator.load_state_dict(
            torch.load(
                model_dir
                + "/"
                + [
                    x
                    for x in os.listdir(model_dir)
                    if x.endswith("pth") and x.startswith("generator")
                ][0]
            )
        )
    except:
        autoencoder.load_state_dict(
            torch.load(
                model_dir
                + "/"
                + [
                    x
                    for x in os.listdir(model_dir)
                    if x.endswith("pth") and x.startswith("autoencoder")
                ][0],
                map_location=torch.device("cpu"),
            )
        )
        generator.load_state_dict(
            torch.load(
                model_dir
                + "/"
                + [
                    x
                    for x in os.listdir(model_dir)
                    if x.endswith("pth") and x.startswith("generator")
                ][0],
                map_location=torch.device("cpu"),
            )
        )

    autoencoder.eval(), generator.eval()
    # %%
    count_parameters = lambda model: sum(p.numel() for p in model.parameters())
    num_params = count_parameters(autoencoder.decoder) + count_parameters(generator)
    print(f"Number of Parameters: {num_params / 1000:.1f}K")
    wandb.log({"Number of Parameters": num_params / 1000})
    # %%
    train = out[1]
    test = out[2]
    OutputInfo_list = out[3]
    discrete_dicts = out[4]
    discrete_dicts_reverse = out[5]
    colnames = out[6]

    n = len(train)
    torch.random.manual_seed(config["seed"])
    # %%
    """Synthetic Data Generation"""
    data = []
    steps = n // config["batch_size"] + 1

    with torch.no_grad():
        for _ in range(steps):
            z = torch.randn(config["batch_size"], config["embedding_dim"]).to(device)
            data.append(autoencoder.decoder(generator(z), training=False))
    data = torch.cat(data, dim=0)
    data = data[:n, :]

    syndata = postprocess(
        data, OutputInfo_list, colnames, discrete_dicts, discrete_dicts_reverse
    )
    # %%
    print("\nDistance to Closest Record...\n")
    
    DCR = DCR_metric(train, syndata)
    
    print('DCR (R&S): {:.3f}'.format(DCR[0]))
    print('DCR (R): {:.3f}'.format(DCR[1]))
    print('DCR (S): {:.3f}'.format(DCR[2]))
    wandb.log({'DCR (R&S)': DCR[0]})
    wandb.log({'DCR (R)': DCR[1]})
    wandb.log({'DCR (S)': DCR[2]})
    #%%
    print("\nAttribute Disclosure...\n")
    
    compromised_idx = np.random.choice(
        range(len(train)), 
        int(len(train) * 0.01), # 1% of data is compromised
        replace=False)
    compromised = train.iloc[compromised_idx]
    
    for attr_num in [5, 10, 15, 20]:
        attr_compromised = train.columns[:attr_num] # randomly chosen
        for K in [1, 10, 100]:
            acc, f1 = attribute_disclosure(
                K, compromised, syndata, attr_compromised)
            print(f'AD F1 (S={attr_num}, K={K}): {f1:.3f}')
            wandb.log({f'AD F1 (S={attr_num}, K={K})': f1})
    #%%
    print("\nNearest Neighbor Adversarial Accuracy...\n")
    
    AA = privacyloss(train, test, syndata)

    print('AA (train): {:.3f}'.format(AA[0]))
    print('AA (test): {:.3f}'.format(AA[1]))
    print('AA (privacy): {:.3f}'.format(AA[2]))
    wandb.log({'AA (train)': AA[0]})
    wandb.log({'AA (test)': AA[1]})
    wandb.log({'AA (privacy)': AA[2]})
    # %%
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()


# %%
if __name__ == "__main__":
    main()
# %%
