# %%
"""
Reference:
[1] https://github.com/sdv-dev/CTGAN/blob/master/ctgan/synthesizers/ctgan.py
"""
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
from modules.model import *
from modules.data_sampler import *
from modules.datasets import generate_dataset
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
    tags=["CTGAN"],
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
    parser.add_argument("--epochs", default=300, type=int, help="maximum iteration")
    parser.add_argument("--batch_size", default=500, type=int, help="batch size")

    parser.add_argument(
        "--generator_lr",
        type=float,
        default=2e-4,
        help="Learning rate for the generator.",
    )
    parser.add_argument(
        "--discriminator_lr",
        type=float,
        default=2e-4,
        help="Learning rate for the discriminator.",
    )

    parser.add_argument(
        "--generator_decay",
        type=float,
        default=1e-6,
        help="Weight decay for the generator.",
    )
    parser.add_argument(
        "--discriminator_decay",
        type=float,
        default=0,
        help="Weight decay for the discriminator.",
    )

    parser.add_argument(
        "--generator_dim",
        type=str,
        default="16,16",
        help="Dimension of each generator layer. "
        "Comma separated integers with no whitespaces.",
    )
    parser.add_argument(
        "--discriminator_dim",
        type=str,
        default="12,12",
        help="Dimension of each discriminator layer. "
        "Comma separated integers with no whitespaces.",
    )

    parser.add_argument(
        "--pac",
        type=int,
        default=10,
        help="Number of samples to group together when applying the discriminator.",
    )
    parser.add_argument(
        "--discriminator_steps",
        type=int,
        default=1,
        help="Number of discriminator updates to do for each generator update."
        "From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper"
        "default is 5. Default used is 1 to match original CTGAN implementation.",
    )
    parser.add_argument(
        "--log_frequency",
        action="store_false",
        help="Whether to use log frequency of categorical levels in conditional sampling."
        "Defaults to True.",
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
    """dataset"""
    (
        train_data,
        dataset,
        dataloader,
        transformer,
        train_,
        _,
        _,
        discrete,
    ) = generate_dataset(config, device, random_state=0)

    assert validate_discrete_columns(train_, discrete) is None
    # %%
    """training-by-sampling"""
    data_sampler = DataSampler(
        train_data, transformer.output_info_list, config["log_frequency"]
    )

    config["data_dim"] = transformer.output_dimensions
    # %%
    """model"""
    generator_dim = [int(x) for x in config["generator_dim"].split(",")]
    discriminator_dim = [int(x) for x in config["discriminator_dim"].split(",")]

    generator = Generator(
        config["latent_dim"] + data_sampler.dim_cond_vec(),
        generator_dim,
        config["data_dim"],
    ).to(device)

    discriminator = Discriminator(
        config["data_dim"] + data_sampler.dim_cond_vec(),
        discriminator_dim,
        pac=config["pac"],
    ).to(device)

    optimizerG = optim.Adam(
        generator.parameters(),
        lr=config["generator_lr"],
        betas=(0.5, 0.9),
        weight_decay=config["generator_decay"],
    )

    optimizerD = optim.Adam(
        discriminator.parameters(),
        lr=config["discriminator_lr"],
        betas=(0.5, 0.9),
        weight_decay=config["discriminator_decay"],
    )

    mean = torch.zeros(config["batch_size"], config["latent_dim"], device=device)
    std = mean + 1

    print(generator.train())
    print(discriminator.train())
    # %%
    """number of parameters"""
    count_parameters = lambda model: sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    num_params = count_parameters(generator) + count_parameters(discriminator)
    print("Number of Parameters:", num_params)
    # %%
    """training"""
    for epoch in range(config["epochs"]):
        logs = train(
            generator,
            discriminator,
            optimizerG,
            optimizerD,
            train_data,
            data_sampler,
            transformer,
            config,
            mean,
            std,
            device,
        )

        print_input = "[epoch {:03d}]".format(epoch + 1)
        print_input += "".join(
            [", {}: {:.4f}".format(x, np.mean(y)) for x, y in logs.items()]
        )
        print(print_input)

        """update log"""
        wandb.log({x: np.mean(y) for x, y in logs.items()})
    # %%
    """model save"""
    torch.save(
        generator.state_dict(), "./assets/CTGAN_{}.pth".format(config["dataset"])
    )
    artifact = wandb.Artifact(
        "CTGAN_{}".format(config["dataset"]), type="model", metadata=config
    )  # description=""
    artifact.add_file("./assets/CTGAN_{}.pth".format(config["dataset"]))
    artifact.add_file("./main.py")
    artifact.add_file("./modules/model.py")
    wandb.log_artifact(artifact)
    # %%
    wandb.run.finish()


# %%
if __name__ == "__main__":
    main()
# %%
