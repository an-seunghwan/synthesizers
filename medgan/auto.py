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

from module.datasets import build_dataset
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
    parser.add_argument('--dataset', type=str, default='census', 
                        help='Dataset options: mnist, census')
    
    parser.add_argument("--embedding_dim", default=16, type=int,
                        help="the embedding dimension size")
    
    parser.add_argument('--epochs', default=200, type=int,
                        help='the number of epochs')
    parser.add_argument('--batch_size', default=512, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate')
    parser.add_argument('--l2reg', default=0.001, type=float,
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
    trainloader = DataLoader(
        dataset=dataset,
        batch_size=config["batch_size"], 
        shuffle=True,
        drop_last=False)
    #%%
    if config["dataset"] == "mnist": config["p"] = 784
    elif config["dataset"] == "census": config["p"] = dataset.p
    else: raise ValueError('Not supported dataset!')
    #%%
    """AutoEncoder"""
    enc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(config["p"], 128),
        nn.ELU(),
        nn.Linear(128, 32),
        nn.ELU(),
        nn.Linear(32, config["embedding_dim"]),
    ).to(device)

    dec = nn.Sequential(
        nn.Linear(config["embedding_dim"], 32),
        nn.ELU(),
        nn.Linear(32, 128),
        nn.ELU(),
        nn.Linear(128, config["p"]),
        # nn.Sigmoid(), # mnist
    ).to(device)
    #%%
    optimizer = torch.optim.Adam(
        list(enc.parameters()) + list(dec.parameters()), 
        lr=config["lr"],
        weight_decay=config["l2reg"]
    )
    #%%
    import importlib
    train_module = importlib.import_module('module.train_auto')
    importlib.reload(train_module)
    train_function = getattr(train_module, 'train_' + config["dataset"])

    for epoch in range(config["epochs"]):
        logs = train_function(trainloader, enc, dec, out, config, optimizer, device)
        
        print_input = "[epoch {:03d}]".format(epoch + 1)
        print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y)) for x, y in logs.items()])
        print(print_input)
        
        """update log"""
        wandb.log({x : np.mean(y) for x, y in logs.items()})
    #%%
    """model save"""
    torch.save(dec.state_dict(), f'./assets/{config["dataset"]}_dec.pth')
    artifact = wandb.Artifact(
        'medGAN_{}'.format(config["dataset"]), 
        type='model',
        metadata=config) # description=""
    artifact.add_file(f'./assets/{config["dataset"]}_dec.pth'.format(config["dataset"]))
    artifact.add_file('./auto.py')
    #%%
    """embedding dataset"""
    emb = []
    with torch.no_grad():
        try:
            for (x_batch, _) in tqdm.tqdm(iter(trainloader), desc="inner loop"):
                x_batch = x_batch.to(device)
                emb.append(enc(x_batch))
        except:
            for x_batch in tqdm.tqdm(iter(trainloader), desc="inner loop"):
                x_batch = x_batch.to(device)
                emb.append(enc(x_batch))
    emb = torch.cat(emb, dim=0)
    torch.save(emb, f'./assets/{config["dataset"]}_emb.pt')
    artifact.add_file(f'./assets/{config["dataset"]}_emb.pt')
    #%%
    wandb.log_artifact(artifact)
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%