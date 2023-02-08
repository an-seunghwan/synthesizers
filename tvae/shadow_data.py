#%%
import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#%%
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
from modules.model import TVAE
from modules.datasets import generate_dataset
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
    project="DistVAE", 
    entity="anseunghwan",
    tags=['TVAE', 'Privacy'],
)
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--num', type=int, default=0, 
                        help='model version')
    parser.add_argument('--dataset', type=str, default='covtype', 
                        help='Dataset options: covtype, credit, loan, adult, cabs, kings')

    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=False)) # default configuration
    
    """model load"""
    artifact = wandb.use_artifact(
        'anseunghwan/DistVAE/TVAE_{}:v{}'.format(config["dataset"], config["num"]), type='model')
    for key, item in artifact.metadata.items():
        config[key] = item
    model_dir = artifact.download()
    
    config["cuda"] = torch.cuda.is_available()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    wandb.config.update(config)
    
    set_random_seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["cuda"]:
        torch.cuda.manual_seed(config["seed"])
    #%%
    """model"""
    model = TVAE(config, device).to(device)
    
    if config["cuda"]:
        model_name = [x for x in os.listdir(model_dir) if x.endswith('pth')][0]
        model.load_state_dict(
            torch.load(
                model_dir + '/' + model_name))
    else:
        model_name = [x for x in os.listdir(model_dir) if x.endswith('pth')][0]
        model.load_state_dict(
            torch.load(
                model_dir + '/' + model_name, map_location=torch.device('cpu')))
    
    model.eval()
    #%%
    """dataset"""
    _, _, transformer, train, test, continuous, discrete = generate_dataset(config, device, random_state=0)
    
    config["input_dim"] = transformer.output_dimensions
    #%%
    if not os.path.exists('./privacy/{}'.format(config["dataset"])):
        os.makedirs('./privacy/{}'.format(config["dataset"]))
    #%%
    """synthetic dataset"""
    torch.manual_seed(config["seed"])
    n = len(train)
    m = len(test)
    
    K = 1 # the number of shadow models
    steps = (n + m) // config["batch_size"] + 1
    for s in tqdm.tqdm(range(K), desc="Generating shadow train and test datasets..."):
        torch.manual_seed(s)
        
        data = []
        with torch.no_grad():
            for _ in range(steps):
                mean = torch.zeros(config["batch_size"], config["latent_dim"])
                std = mean + 1
                noise = torch.normal(mean=mean, std=std).to(device)
                fake = model.decoder(noise)
                fake = torch.tanh(fake)
                data.append(fake.numpy())
        data = np.concatenate(data, axis=0)
        data = data[:n + m]
        
        # train
        sample_df = transformer.inverse_transform(data[:n], model.sigma.detach().cpu().numpy())
        sample_df.to_csv(f'./privacy/{config["dataset"]}/train_{config["seed"]}_synthetic{s}.csv')
        # test
        sample_df = transformer.inverse_transform(data[n:], model.sigma.detach().cpu().numpy())
        sample_df.to_csv(f'./privacy/{config["dataset"]}/test_{config["seed"]}_synthetic{s}.csv')
    #%%
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%