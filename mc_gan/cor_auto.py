# %%
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# %%
import numpy as np
import importlib

import torch
from torch.utils.data import DataLoader

from module.utils import str2bool
from module.datasets import build_dataset

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
    # tags=[''],
)
# %%
import ast


def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError('Argument "%s" is not a list' % (s))
    return v


import argparse


def get_args(debug):
    parser = argparse.ArgumentParser("parameters")

    parser.add_argument(
        "--seed", type=int, default=0, help="seed for repeatable results"
    )
    parser.add_argument("--model", type=str, default="cor_auto")
    parser.add_argument(
        "--dataset", type=str, default="census", help="Dataset options: census, survey, income"
    )

    parser.add_argument(
        "--embedding_dim", default=128, type=int, help="the embedding dimension size"
    )
    parser.add_argument(
        "--hidden_dims",
        default=[128],
        type=arg_as_list,  # single layer
        help="hidden dimensions for autoencoder",
    )

    parser.add_argument("--epochs", default=100, type=int, help="the number of epochs")
    parser.add_argument("--batch_size", default=128, type=int, help="batch size")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument(
        "--l2reg", default=0.001, type=float, help="L2 regularization: weight decay"
    )
    parser.add_argument(
        "--mc", default=False, type=str2bool, help="Multi-Categorical setting"
    )

    if debug:
        return parser.parse_args(args=[])
    else:
        return parser.parse_args()


# %%
def main():
    # %%
    config = vars(get_args(debug=False))  # default configuration
    torch.manual_seed(config["seed"])
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    wandb.config.update(config)
    # %%
    out = build_dataset(config)
    dataset = out[0]
    trainloader = DataLoader(
        dataset=dataset, batch_size=config["batch_size"], shuffle=True, drop_last=False
    )

    config["p"] = dataset.p
    # %%
    auto_model_module = importlib.import_module("module.model_cor_auto")
    importlib.reload(auto_model_module)
    autoencoder = getattr(auto_model_module, "AutoEncoder")(
        config,
        config["hidden_dims"],
        list(reversed(config["hidden_dims"])),
        device=device,
    ).to(device)
    autoencoder.train()
    # %%
    count_parameters = lambda model: sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    num_params = count_parameters(autoencoder.decoder)
    print(f"Number of Parameters: {num_params / 1000:.1f}K")
    wandb.log({"Number of Parameters": num_params / 1000})
    # %%
    optimizer = torch.optim.Adam(
        autoencoder.parameters(), lr=config["lr"], weight_decay=config["l2reg"]
    )
    # %%
    train_module = importlib.import_module("module.train_cor_auto")
    importlib.reload(train_module)
    train_function = getattr(train_module, "train_function")  # without MNIST option

    for epoch in range(config["epochs"]):
        logs = train_function(trainloader, autoencoder, optimizer, device)

        print_input = "[epoch {:03d}]".format(epoch + 1)
        print_input += "".join(
            [", {}: {:.4f}".format(x, np.mean(y)) for x, y in logs.items()]
        )
        print(print_input)

        """update log"""
        wandb.log({x: np.mean(y) for x, y in logs.items()})
    # %%
    """model save"""
    model_name = f'dec_corGAN_{config["dataset"]}'
    torch.save(autoencoder.decoder.state_dict(), f"./assets/model/{model_name}.pth")
    artifact = wandb.Artifact(
        model_name, type="model", metadata=config
    )  # description=""
    artifact.add_file(f"./assets/model/{model_name}.pth")
    artifact.add_file("./cor_auto.py")
    artifact.add_file("./module/model_cor_auto.py")
    # %%
    wandb.log_artifact(artifact)
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()


# %%
if __name__ == "__main__":
    main()
# %%
