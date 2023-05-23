#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F

from module.datasets import MyDataset, build_dataset
#%%
import sys
import subprocess
try:
    import wandb
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    with open("./wandb_api.txt", "r") as f:
        key = f.readlines()
    subprocess.run(["wandb", "login"], input=key[0], encoding='utf-8')
    import wandb

run = wandb.init(
    project="HDistVAE", 
    entity="anseunghwan",
    tags=['medGAN'],
)
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results')
    parser.add_argument('--dataset', type=str, default='survey', 
                        help='Dataset options: mnist, census, survey')
    
    parser.add_argument("--embedding_dim", default=16, type=int,
                        help="the embedding dimension size")
    parser.add_argument("--latent_dim", default=2, type=int,
                        help="the latent dimension size")
    
    parser.add_argument('--epochs', default=200, type=int,
                        help='the number of epochs')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate')
    
    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=False)) # default configuration
    torch.manual_seed(config["seed"])
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    wandb.config.update(config)
    #%%
    out = build_dataset(config)
    dataset = out[0]
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    if config["dataset"] == "mnist": config["p"] = 784
    else: config["p"] = dataset.p

    dec = nn.Sequential(
        nn.Linear(config["embedding_dim"], 32),
        nn.ELU(),
        nn.Linear(32, 128),
        nn.ELU(),
        nn.Linear(128, config["p"]),
        nn.Sigmoid(),
    ).to(device)

    try:
        dec.load_state_dict(torch.load(f'./assets/{config["dataset"]}_dec.pth'))
    except:
        dec.load_state_dict(torch.load(f'./assets/{config["dataset"]}_dec.pth', 
            map_location=torch.device('cpu')))
    #%%
    import importlib
    model_module = importlib.import_module('module.model')
    importlib.reload(model_module)

    model = getattr(model_module, 'GAN')(config, dec, device).to(device)
    model.train()
    #%%
    """Number of Parameters"""
    count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = count_parameters(model)
    print("Number of Parameters:", num_params)
    wandb.log({'Number of Parameters': num_params})
    #%%
    optimizer_D = torch.optim.Adam(
        model.discriminator.parameters(), 
        lr=config["lr"]
    )
    optimizer_G = torch.optim.Adam(
        list(model.generator.parameters()) + list(model.dec.parameters()), 
        lr=config["lr"]
    )

    import importlib
    train_module = importlib.import_module('module.train')
    importlib.reload(train_module)

    for epoch in range(config["epochs"]):
        logs = train_module.train_GAN(dataloader, model, config, optimizer_D, optimizer_G, device)
        
        print_input = "[epoch {:03d}]".format(epoch + 1)
        print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y)) for x, y in logs.items()])
        print(print_input)
        
        """update log"""
        wandb.log({x : np.mean(y) for x, y in logs.items()})
    #%%
    """model save"""
    torch.save(model.state_dict(), f'./assets/{config["dataset"]}_medGAN.pth')
    artifact = wandb.Artifact(
        '{}_medGAN'.format(config["dataset"]), 
        type='model',
        metadata=config) # description=""
    artifact.add_file(f'./assets/{config["dataset"]}_medGAN.pth'.format(config["dataset"]))
    artifact.add_file('./main.py')
    artifact.add_file('./module/model.py')
    #%%
    wandb.log_artifact(artifact)
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%