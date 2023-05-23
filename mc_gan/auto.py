#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
import importlib

import torch
from torch.utils.data import DataLoader

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
    tags=['medGAN', 'AE'],
)
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results')
    parser.add_argument('--dataset', type=str, default='census', 
                        help='Dataset options: mnist, census, survey')
    
    parser.add_argument("--embedding_dim", default=16, type=int,
                        help="the embedding dimension size")
    
    parser.add_argument('--epochs', default=100, type=int,
                        help='the number of epochs')
    parser.add_argument('--batch_size', default=512, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=0.002, type=float,
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
    
    if config["dataset"] == "mnist": config["p"] = 784
    else: config["p"] = dataset.p
    #%%
    """AutoEncoder"""
    auto_model_module = importlib.import_module('module.model_auto')
    importlib.reload(auto_model_module)
    autoencoder = getattr(auto_model_module, 'AutoEncoder')(
        config, 
        [128, 32], 
        [32, 128]).to(device)
    autoencoder.train()
    #%%
    optimizer = torch.optim.Adam(
        autoencoder.parameters(), 
        lr=config["lr"],
        weight_decay=config["l2reg"]
    )
    #%%
    train_module = importlib.import_module('module.train_auto')
    importlib.reload(train_module)
    train_function = getattr(train_module, 'train_function') # without MNIST option

    for epoch in range(config["epochs"]):
        logs = train_function(trainloader, autoencoder, optimizer, device)
        
        print_input = "[epoch {:03d}]".format(epoch + 1)
        print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y)) for x, y in logs.items()])
        print(print_input)
        
        """update log"""
        wandb.log({x : np.mean(y) for x, y in logs.items()})
    #%%
    """model save"""
    torch.save(autoencoder.decoder.state_dict(), f'./assets/{config["dataset"]}_dec.pth')
    artifact = wandb.Artifact(
        'dec_medGAN_{}'.format(config["dataset"]), 
        type='model',
        metadata=config) # description=""
    artifact.add_file(f'./assets/{config["dataset"]}_dec.pth')
    artifact.add_file('./auto.py')
    artifact.add_file('./module/model_auto.py')
    #%%
    wandb.log_artifact(artifact)
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%