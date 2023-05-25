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

run = wandb.init(
    project="HDistVAE", 
    entity="anseunghwan",
    tags=['medGAN', 'inference'],
)
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--num', type=int, default=0, 
                        help='model number')
    parser.add_argument('--dataset', type=str, default='survey', 
                        help='Dataset options: mnist, census, survey')
    parser.add_argument('--mc', default=False, type=str2bool,
                        help='Multi-Categorical setting')
    
    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=False)) # default configuration
    
    """model load"""
    model_name = lambda x: f'mc_medGAN_{config["dataset"]}' if x else f'medGAN_{config["dataset"]}'
    artifact = wandb.use_artifact(f'anseunghwan/HDistVAE/{model_name(config["mc"])}:v{config["num"]}', type='model')
    for key, item in artifact.metadata.items():
        config[key] = item
    model_dir = artifact.download()
    
    model_module = importlib.import_module('module.model')
    importlib.reload(model_module)

    generator = getattr(model_module, 'medGANGenerator')(
        config["embedding_dim"]).to(device)
    
    try:
        model_name = [x for x in os.listdir(model_dir) if x.endswith('pth')][0]
        generator.load_state_dict(
            torch.load(
                model_dir + '/' + model_name))
    except:
        model_name = [x for x in os.listdir(model_dir) if x.endswith('pth')][0]
        generator.load_state_dict(
            torch.load(
                model_dir + '/' + model_name, map_location=torch.device('cpu')))
    generator.eval()
    #%%
    torch.manual_seed(config["seed"])
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    config["cuda"] = torch.cuda.is_available()
    wandb.config.update(config)
    #%%
    out = build_dataset(config)
    dataset = out[0]

    if config["dataset"] == "mnist": config["p"] = 784
    else: config["p"] = dataset.p
    
    OutputInfo_list = None
    if config["mc"]:
        OutputInfo_list = out[3]
    #%%
    auto_model_module = importlib.import_module('module.model_auto')
    importlib.reload(auto_model_module)
    auto_model_name = lambda x: f'dec_mc_medGAN_{config["dataset"]}' if x else f'dec_medGAN_{config["dataset"]}'
    artifact = wandb.use_artifact(f'anseunghwan/HDistVAE/{auto_model_name(config["mc"])}:v{config["seed"]}', type='model')
    model_dir = artifact.download()
    
    autoencoder = getattr(auto_model_module, 'AutoEncoder')(
        config, 
        config["hidden_dims"], 
        list(reversed(config["hidden_dims"])), 
        OutputInfo_list=OutputInfo_list).to(device)
    
    try:
        model_name = [x for x in os.listdir(model_dir) if x.endswith('pth')][0]
        autoencoder.decoder.load_state_dict(
            torch.load(
                model_dir + '/' + model_name))
    except:
        model_name = [x for x in os.listdir(model_dir) if x.endswith('pth')][0]
        autoencoder.decoder.load_state_dict(
            torch.load(
                model_dir + '/' + model_name, map_location=torch.device('cpu')))
    autoencoder.eval()
    #%%
    count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = count_parameters(generator) + count_parameters(autoencoder.decoder)
    print("Number of Parameters:", num_params)
    wandb.log({'Number of Parameters': num_params})
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
            if config["mc"]:
                data.append(autoencoder.decoder(generator(z), training=False))
            else:
                data.append(autoencoder.decoder(generator(z)))
    data = torch.cat(data, dim=0)
    data = data[:n, :]
    
    syndata = postprocess(data, OutputInfo_list, colnames, discrete_dicts, discrete_dicts_reverse)
    #%%
    metrics = evaluate(syndata, train, test, config)
    
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
# computational issue -> sampling 5000
# tmp1 = syndata.astype(int).to_numpy()[:5000, :]
# tmp2 = train.astype(int).to_numpy()[:5000, :] # train

# hamming = np.zeros((tmp1.shape[0], tmp2.shape[0]))
# for i in tqdm.tqdm(range(len(tmp1))):
#     hamming[i, :] = (tmp2 - tmp1[[i]] != 0).mean(axis=1)
# #%%
# fig = plt.figure(figsize=(6, 4))
# plt.hist(hamming.flatten(), density=True, bins=20)
# plt.axvline(np.quantile(hamming.flatten(), 0.05), color='red')
# plt.xlabel('Hamming Dist', fontsize=13)
# plt.ylabel('density', fontsize=13)
# plt.savefig('./assets/census_hamming.png')
# # plt.show()
# plt.close()
# wandb.log({'Hamming Distance': wandb.Image(fig)})
# wandb.log({'Hamming': np.quantile(hamming.flatten(), 0.05)})
#%%
# if config["dataset"] == 'mnist':
#     n = 100
#     with torch.no_grad():
#         syndata = model.generate_data(n, seed=1)
#     # syndata = torch.where(syndata > 0.5, 1, 0)
#     syndata = syndata.reshape(-1, 28, 28)
#     fig, ax = plt.subplots(10, 10, figsize=(20, 20))
#     for j in range(n):
#         ax.flatten()[j].imshow(syndata[j], cmap='gray_r')
#         ax.flatten()[j].axis('off')
#     plt.tight_layout()
#     plt.savefig('./assets/mnist.png')
#     # plt.savefig('./assets/cifar10.png')
#     # plt.show()
#     plt.close()
#%%