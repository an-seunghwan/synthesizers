# %%
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# %%
import numpy as np
import importlib

import torch
from torch.utils.data import DataLoader

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
    parser.add_argument("--model", type=str, default="MC-Gumbel")
    parser.add_argument(
        "--dataset", type=str, default="census", help="Dataset options: census, survey, income"
    )

    parser.add_argument(
        "--embedding_dim",
        default=128,
        type=int,  # noise_dim
        help="the embedding dimension size",
    )
    parser.add_argument(
        "--hidden_dims_disc",
        default=[256, 128],
        type=arg_as_list,
        help="hidden dimensions for discriminator",
    )
    parser.add_argument(
        "--hidden_dims_gen",
        default=[256, 128],
        type=arg_as_list,
        help="hidden dimensions for generator",
    )

    parser.add_argument("--epochs", default=1000, type=int, help="the number of epochs")
    parser.add_argument("--batch_size", default=1024, type=int, help="batch size")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument(
        "--l2reg", default=0.001, type=float, help="L2 regularization: weight decay"
    )
    parser.add_argument(
        "--tau", default=0.666, type=float, help="temperature in Gumbel-Softmax"
    )

    parser.add_argument(
        "--mc", default=True, type=bool, help="Multi-Categorical setting"
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
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    OutputInfo_list = out[3]

    config["p"] = dataset.p
    # %%
    model_module = importlib.import_module("module.model")
    importlib.reload(model_module)

    generator = getattr(model_module, "Generator")(
        config["embedding_dim"],
        [x.dim for x in OutputInfo_list],
        hidden_sizes=config["hidden_dims_gen"],
        bn_decay=0.1,
        device=device,
    ).to(device)
    discriminator = getattr(model_module, "Discriminator")(
        config["p"],
        hidden_sizes=config["hidden_dims_disc"],
        bn_decay=0.1,
        critic=False,
        device=device,
    ).to(device)
    generator.train(mode=True), discriminator.train(mode=True)
    # %%
    count_parameters = lambda model: sum(p.numel() for p in model.parameters())
    num_params = count_parameters(generator)
    print(f"Number of Parameters: {num_params / 1000:.1f}K")
    wandb.log({"Number of Parameters": num_params / 1000})
    # %%
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), lr=config["lr"], weight_decay=config["l2reg"]
    )
    optimizer_G = torch.optim.Adam(
        generator.parameters(), lr=config["lr"], weight_decay=config["l2reg"]
    )
    # %%
    train_module = importlib.import_module("module.train")
    importlib.reload(train_module)

    for epoch in range(config["epochs"]):
        logs = train_module.train_Gumbel(
            dataloader,
            discriminator,
            generator,
            config,
            optimizer_D,
            optimizer_G,
            epoch,
            device,
        )

        print_input = "[epoch {:04d}]".format(epoch + 1)
        print_input += "".join(
            [", {}: {:.4f}".format(x, np.mean(y)) for x, y in logs.items()]
        )
        print(print_input)

        """update log"""
        wandb.log({x: np.mean(y) for x, y in logs.items()})
    # %%
    """model save"""
    model_name = f'mc_Gumbel_{config["dataset"]}'
    torch.save(generator.state_dict(), f"./assets/model/generator_{model_name}.pth")
    artifact = wandb.Artifact(
        model_name, type="model", metadata=config
    )  # description=""
    artifact.add_file(f"./assets/model/generator_{model_name}.pth")
    artifact.add_file("./mc_gumbel.py")
    artifact.add_file("./module/model.py")
    # %%
    wandb.log_artifact(artifact)
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()


# %%
if __name__ == "__main__":
    main()
# %%
