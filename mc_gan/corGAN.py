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
    parser.add_argument("--model", type=str, default="corGAN")
    parser.add_argument(
        "--dataset", type=str, default="census", help="Dataset options: census, survey"
    )

    parser.add_argument(
        "--embedding_dim", default=128, type=int, help="the embedding dimension size"
    )
    parser.add_argument(
        "--hidden_dims",
        default=[128],
        type=arg_as_list,
        help="hidden dimensions for autoencoder",
    )
    parser.add_argument(
        "--hidden_dims_disc",
        default=[256, 128],
        type=arg_as_list,
        help="hidden dimensions for discriminator",
    )

    parser.add_argument("--epochs", default=1000, type=int, help="the number of epochs")
    parser.add_argument("--batch_size", default=1024, type=int, help="batch size")
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
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    config["p"] = dataset.p
    # %%
    auto_model_module = importlib.import_module("module.model_cor_auto")
    importlib.reload(auto_model_module)
    auto_model_name = f'dec_corGAN_{config["dataset"]}'
    artifact = wandb.use_artifact(
        f'{entity}/{project}/{auto_model_name}:v{config["seed"]}', type="model"
    )
    model_dir = artifact.download()

    autoencoder = getattr(auto_model_module, "AutoEncoder")(
        config,
        config["hidden_dims"],
        list(reversed(config["hidden_dims"])),
        device=device,
    ).to(device)

    try:
        model_name = [x for x in os.listdir(model_dir) if x.endswith("pth")][0]
        autoencoder.decoder.load_state_dict(torch.load(model_dir + "/" + model_name))
    except:
        model_name = [x for x in os.listdir(model_dir) if x.endswith("pth")][0]
        autoencoder.decoder.load_state_dict(
            torch.load(model_dir + "/" + model_name, map_location=torch.device("cpu"))
        )
    autoencoder.eval()
    # %%
    model_module = importlib.import_module("module.model")
    importlib.reload(model_module)

    discriminator = getattr(model_module, "medGANDiscriminator")(
        config["p"], hidden_sizes=config["hidden_dims_disc"], device=device
    ).to(device)
    generator = getattr(model_module, "medGANGenerator")(
        config["embedding_dim"], device=device
    ).to(device)
    discriminator.train(), generator.train()
    # %%
    count_parameters = lambda model: sum(p.numel() for p in model.parameters())
    num_params = count_parameters(autoencoder.decoder) + count_parameters(generator)
    print(f"Number of Parameters: {num_params / 1000:.1f}K")
    wandb.log({"Number of Parameters": num_params / 1000})
    # %%
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), lr=config["lr"], weight_decay=config["l2reg"]
    )
    optimizer_G = torch.optim.Adam(
        list(generator.parameters()) + list(autoencoder.decoder.parameters()),
        lr=config["lr"],
        weight_decay=config["l2reg"],
    )
    # %%
    train_module = importlib.import_module("module.train")
    importlib.reload(train_module)

    for epoch in range(config["epochs"]):
        logs = train_module.train_medGAN(
            dataloader,
            autoencoder,
            discriminator,
            generator,
            config,
            optimizer_D,
            optimizer_G,
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
    model_name = f'corGAN_{config["dataset"]}'
    torch.save(generator.state_dict(), f"./assets/model/{model_name}.pth")
    artifact = wandb.Artifact(
        model_name, type="model", metadata=config
    )  # description=""
    artifact.add_file(f"./assets/model/{model_name}.pth")
    artifact.add_file("./medgan.py")
    artifact.add_file("./module/model.py")
    # %%
    wandb.log_artifact(artifact)
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()


# %%
if __name__ == "__main__":
    main()
# %%
