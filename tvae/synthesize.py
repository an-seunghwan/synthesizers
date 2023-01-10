#%%
import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
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

from modules.simulation import set_random_seed
from modules.model import TVAE
from modules.datasets import generate_dataset
from modules.evaluation import (
    regression_eval,
    classification_eval,
    goodness_of_fit,
    privacy_metrics
)
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
    tags=['TVAE', 'Synthetic'],
)
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--num', type=int, default=2, 
                        help='model version')

    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=False)) # default configuration
    
    dataset = 'covtype'
    # dataset = 'credit'
    
    """model load"""
    artifact = wandb.use_artifact(
        'anseunghwan/DistVAE/TVAE_{}:v{}'.format(dataset, config["num"]), type='model')
    for key, item in artifact.metadata.items():
        config[key] = item
    assert dataset == config["dataset"]
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
    """Number of Parameters"""
    count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = count_parameters(model)
    print("Number of Parameters:", num_params)
    wandb.log({'Number of Parameters': num_params})
    #%%
    """dataset"""
    _, _, transformer, train, test, continuous, discrete = generate_dataset(config, device, random_state=0)
    
    config["input_dim"] = transformer.output_dimensions
    #%%
    # preprocess
    std = train[continuous].std(axis=0)
    mean = train[continuous].mean(axis=0)
    train[continuous] = (train[continuous] - mean) / std
    test[continuous] = (test[continuous] - mean) / std
    
    df = pd.concat([train, test], axis=0)
    df_dummy = []
    for d in discrete:
        df_dummy.append(pd.get_dummies(df[d], prefix=d))
    df = pd.concat([df.drop(columns=discrete)] + df_dummy, axis=1)
    
    train = df.iloc[:45000]
    test = df.iloc[45000:]
    #%%
    """synthetic dataset"""
    torch.manual_seed(config["seed"])
    steps = len(train) // config["batch_size"] + 1
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
    data = data[:len(train)]
    sample_df = transformer.inverse_transform(data, model.sigma.detach().cpu().numpy())
    
    std = sample_df[continuous].std(axis=0)
    mean = sample_df[continuous].mean(axis=0)
    sample_df[continuous] = (sample_df[continuous] - mean) / std
    
    df_dummy = []
    for d in discrete:
        df_dummy.append(pd.get_dummies(sample_df[d], prefix=d))
    sample_df = pd.concat([sample_df.drop(columns=discrete)] + df_dummy, axis=1)
    #%%
    """Regression"""
    if config["dataset"] == "covtype":
        target = 'Elevation'
    elif config["dataset"] == "credit":
        target = 'AMT_INCOME_TOTAL'
    else:
        raise ValueError('Not supported dataset!')
    #%%
    # baseline
    print("\nBaseline: Machine Learning Utility in Regression...\n")
    base_r2result = regression_eval(train, test, target)
    wandb.log({'R^2 (Baseline)': np.mean([x[1] for x in base_r2result])})
    #%%
    # TVAE
    print("\nSynthetic: Machine Learning Utility in Regression...\n")
    r2result = regression_eval(sample_df, test, target)
    wandb.log({'R^2 (TVAE)': np.mean([x[1] for x in r2result])})
    #%%
    # visualization
    fig = plt.figure(figsize=(5, 4))
    plt.plot([x[1] for x in base_r2result], 'o--', label='baseline')
    plt.plot([x[1] for x in r2result], 'o--', label='synthetic')
    plt.ylim(0, 1)
    plt.ylabel('$R^2$', fontsize=13)
    plt.xticks([0, 1, 2], [x[0] for x in base_r2result], fontsize=13)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./assets/{}/{}_MLU_regression.png'.format(config["dataset"], config["dataset"]))
    # plt.show()
    plt.close()
    wandb.log({'ML Utility (Regression)': wandb.Image(fig)})
    #%%
    """Classification"""
    if config["dataset"] == "covtype":
        target = 'Cover_Type'
    elif config["dataset"] == "credit":
        target = 'TARGET'
    else:
        raise ValueError('Not supported dataset!')
    #%%
    # baseline
    print("\nBaseline: Machine Learning Utility in Classification...\n")
    base_f1result = classification_eval(train, test, target)
    wandb.log({'F1 (Baseline)': np.mean([x[1] for x in base_f1result])})
    #%%
    # TVAE
    print("\nSynthetic: Machine Learning Utility in Classification...\n")
    f1result = classification_eval(sample_df, test, target)
    wandb.log({'F1 (TVAE)': np.mean([x[1] for x in f1result])})
    #%%
    # visualization
    fig = plt.figure(figsize=(5, 4))
    plt.plot([x[1] for x in base_f1result], 'o--', label='baseline')
    plt.plot([x[1] for x in f1result], 'o--', label='synthetic')
    plt.ylim(0, 1)
    plt.ylabel('$F_1$', fontsize=13)
    plt.xticks([0, 1, 2], [x[0] for x in base_f1result], fontsize=13)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./assets/{}/{}_MLU_classification.png'.format(config["dataset"], config["dataset"]))
    # plt.show()
    plt.close()
    wandb.log({'ML Utility (Classification)': wandb.Image(fig)})
    #%%
    """Goodness of Fit""" # only continuous
    print("\nGoodness of Fit...\n")
    
    Dn, W1 = goodness_of_fit(len(continuous), train.to_numpy(), sample_df.to_numpy())
    
    print('Goodness of Fit (Kolmogorov): {:.3f}'.format(Dn))
    print('Goodness of Fit (1-Wasserstein): {:.3f}'.format(W1))
    wandb.log({'Goodness of Fit (Kolmogorov)': Dn})
    wandb.log({'Goodness of Fit (1-Wasserstein)': W1})
    #%%
    """Privacy Preservability""" # only continuous
    print("\nPrivacy Preservability...\n")
    
    privacy = privacy_metrics(train[continuous], sample_df[continuous])
    
    DCR = privacy[0, :3]
    print('DCR (R&S): {:.3f}'.format(DCR[0]))
    print('DCR (R): {:.3f}'.format(DCR[1]))
    print('DCR (S): {:.3f}'.format(DCR[2]))
    wandb.log({'DCR (R&S)': DCR[0]})
    wandb.log({'DCR (R)': DCR[1]})
    wandb.log({'DCR (S)': DCR[2]})
    
    NNDR = privacy[0, 3:]
    print('NNDR (R&S): {:.3f}'.format(NNDR[0]))
    print('NNDR (R): {:.3f}'.format(NNDR[1]))
    print('NNDR (S): {:.3f}'.format(NNDR[2]))
    wandb.log({'NNDR (R&S)': NNDR[0]})
    wandb.log({'NNDR (R)': NNDR[1]})
    wandb.log({'NNDR (S)': NNDR[2]})
    #%%
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%