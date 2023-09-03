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
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F

from module.utils import str2bool, postprocess
from module.evaluation import evaluate
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

project = "Synthetic(High)"
entity = "anseunghwan"

run = wandb.init(
    project=project, 
    entity=entity,
    tags=['inference'],
)
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--num', type=int, default=0, 
                        help='model number')
    parser.add_argument('--model', type=str, default='medGAN')
    parser.add_argument('--dataset', type=str, default='census', 
                        help='Dataset options: census, survey')
    parser.add_argument('--tau', default=0.666, type=float,
                        help='temperature in Gumbel-Softmax')
    
    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=False)) # default configuration
    
    if config["tau"] == 0:
        config["mc"] = False
    else:
        config["mc"] = True
    
    """model load"""
    model_name = lambda x: f'mc_medGAN_{config["dataset"]}' if x else f'medGAN_{config["dataset"]}'
    artifact = wandb.use_artifact(f'{entity}/{project}/{model_name(config["mc"])}:v{config["num"]}', type='model')
    for key, item in artifact.metadata.items():
        config[key] = item
    model_dir = artifact.download()
    
    torch.manual_seed(config["seed"])
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    config["cuda"] = torch.cuda.is_available()
    wandb.config.update(config)
    
    model_module = importlib.import_module('module.model')
    importlib.reload(model_module)

    generator = getattr(model_module, 'medGANGenerator')(
        config["embedding_dim"], device=device).to(device)
    
    try:
        generator.load_state_dict(
            torch.load(
                model_dir + '/' + [x for x in os.listdir(model_dir) if x.endswith('pth')][0]))
    except:
        generator.load_state_dict(
            torch.load(
                model_dir + '/' + [x for x in os.listdir(model_dir) if x.endswith('pth')][0], 
                map_location=torch.device('cpu')))
    generator.eval()
    #%%
    out = build_dataset(config)
    dataset = out[0]

    config["p"] = dataset.p
    
    OutputInfo_list = None
    if config["mc"]:
        OutputInfo_list = out[3]
    #%%
    auto_model_module = importlib.import_module('module.model_auto')
    importlib.reload(auto_model_module)
    auto_model_name = lambda x: f'dec_mc_medGAN_{config["dataset"]}' if x else f'dec_medGAN_{config["dataset"]}'
    artifact = wandb.use_artifact(f'{entity}/{project}/{auto_model_name(config["mc"])}:v{config["seed"]}', type='model')
    model_dir = artifact.download()
    
    autoencoder = getattr(auto_model_module, 'AutoEncoder')(
        config, 
        config["hidden_dims"], 
        list(reversed(config["hidden_dims"])), 
        OutputInfo_list=OutputInfo_list,
        device=device).to(device)
    
    try:
        autoencoder.decoder.load_state_dict(
            torch.load(
                model_dir + '/' + [x for x in os.listdir(model_dir) if x.endswith('pth')][0]))
    except:
        autoencoder.decoder.load_state_dict(
            torch.load(
                model_dir + '/' + [x for x in os.listdir(model_dir) if x.endswith('pth')][0], 
                map_location=torch.device('cpu')))
    autoencoder.eval()
    #%%
    count_parameters = lambda model: sum(p.numel() for p in model.parameters())
    num_params = count_parameters(autoencoder.decoder) + count_parameters(generator)
    print(f"Number of Parameters: {num_params / 1000:.1f}K")
    wandb.log({'Number of Parameters': num_params / 1000})
    #%%
    train = out[1]
    test = out[2]
    OutputInfo_list = out[3]
    discrete_dicts = out[4]
    discrete_dicts_reverse = out[5]
    colnames = out[6]

    n = len(train)
    torch.random.manual_seed(config["seed"])
    #%%
    """Synthetic Data Generation"""
    data = []
    steps = n // config["batch_size"] + 1
    
    with torch.no_grad():
        for _ in range(steps):
            z = torch.randn(config["batch_size"], config["embedding_dim"]).to(device) 
            data.append(autoencoder.decoder(generator(z), training=False))
    data = torch.cat(data, dim=0)
    data = data[:n, :]
    
    syndata = postprocess(data, OutputInfo_list, colnames, discrete_dicts, discrete_dicts_reverse)
    #%%
    metrics = evaluate(syndata, train, test, config, model_name(config["mc"]))
    
    print(f"KL: {metrics.KL:.3f}")
    wandb.log({'KL': metrics.KL})
    print(f"KS: {metrics.KS:.3f}")
    wandb.log({'KS': metrics.KS})
    print(f"Coverage: {metrics.coverage:.3f}")
    wandb.log({'Coverage': metrics.coverage})
    print(f"DimProb: {metrics.mse_dim_prob:.3f}")
    wandb.log({'DimProb': metrics.mse_dim_prob})
    wandb.log({'Proportion': wandb.Image(metrics.Proportion)})
    print(f"PCD(Pearson): {metrics.PCD_Pearson:.3f}")
    wandb.log({'PCD(Pearson)': metrics.PCD_Pearson})
    print(f"PCD(Kendall): {metrics.PCD_Kendall:.3f}")
    wandb.log({'PCD(Kendall)': metrics.PCD_Kendall})
    print(f"logcluster: {metrics.logcluster:.3f}")
    wandb.log({'logcluster': metrics.logcluster})
    print(f"VarPred: {metrics.VarPred:.3f}")
    wandb.log({'VarPred': metrics.VarPred})
    wandb.log({'ACC': wandb.Image(metrics.ACC)})
    #%%
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%