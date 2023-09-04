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

from modules.datasets import generate_dataset
from modules.data_preparation import DataPrep
from modules.transformer import DataTransformer, ImageTransformer
from modules.synthesizer import *

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
    tags=["CTAB-GAN", "Privacy"],
)
# %%
import argparse


def get_args(debug):
    parser = argparse.ArgumentParser("parameters")

    parser.add_argument("--num", type=int, default=0, help="model version")
    parser.add_argument(
        "--dataset",
        type=str,
        default="covtype",
        help="Dataset options: covtype, credit, loan, adult, cabs, kings",
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
        "anseunghwan/DistVAE/CTABGAN_{}:v{}".format(config["dataset"], config["num"]),
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
    """dataset"""
    (
        train_data,
        transformer,
        data_dim,
        target_index,
        data_prep,
        train,
        test,
        continuous,
        discrete,
    ) = generate_dataset(config)
    # %%
    """model setting"""
    # initializing the condvec object to sample conditional vectors during training
    cond_generator = Condvec(train_data, transformer.output_info)

    # obtaining the desired height/width for generating square images from the generator network that can be converted back to tabular domain
    sides = [4, 8, 16, 24, 32]
    col_size_g = data_dim
    for i in sides:
        if i * i >= col_size_g:
            gside = i
            break

    # constructing the generator and discriminator networks
    layers_G = determine_layers_gen(
        gside, config["latent_dim"] + cond_generator.n_opt, config["num_channels"]
    )
    generator = Generator(layers_G).to(device)

    if config["cuda"]:
        model_name = [x for x in os.listdir(model_dir) if x.endswith("pth")][0]
        generator.load_state_dict(torch.load(model_dir + "/" + model_name))
    else:
        model_name = [x for x in os.listdir(model_dir) if x.endswith("pth")][0]
        generator.load_state_dict(
            torch.load(model_dir + "/" + model_name, map_location=torch.device("cpu"))
        )

    generator.eval()
    # %%
    if not os.path.exists("./privacy/{}".format(config["dataset"])):
        os.makedirs("./privacy/{}".format(config["dataset"]))
    # %%
    """Sampling Mechanism"""
    n = len(train)
    m = len(test)

    # column information associated with the transformer fit to the pre-processed training data
    output_info = transformer.output_info
    Gtransformer = ImageTransformer(gside)

    K = 1  # the number of shadow models
    # generating synthetic data in batches accordingly to the total no. required
    steps = (n + m) // config["batch_size"] + 1
    for s in tqdm(range(K), desc="Generating shadow train and test datasets..."):
        torch.manual_seed(s)

        data = []
        for _ in range(steps):
            # generating synthetic data using sampled noise and conditional vectors
            noisez = torch.randn(
                config["batch_size"], config["latent_dim"], device=device
            )
            condvec = cond_generator.sample(config["batch_size"])
            c = condvec
            c = torch.from_numpy(c).to(device)
            noisez = torch.cat([noisez, c], dim=1)
            noisez = noisez.view(
                config["batch_size"], config["latent_dim"] + cond_generator.n_opt, 1, 1
            )
            fake = generator(noisez)
            faket = Gtransformer.inverse_transform(fake)
            fakeact = apply_activate(faket, output_info)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)

        # applying the inverse transform and returning synthetic data in a similar form as the original pre-processed training data
        result = transformer.inverse_transform(data)

        sample_df = data_prep.inverse_prep(result[0 : n + m])

        sample_df.iloc[:n].to_csv(
            f'./privacy/{config["dataset"]}/train_{config["seed"]}_synthetic{s}.csv'
        )
        sample_df.iloc[n:].to_csv(
            f'./privacy/{config["dataset"]}/test_{config["seed"]}_synthetic{s}.csv'
        )
    # %%
    wandb.run.finish()


# %%
if __name__ == "__main__":
    main()
# %%
