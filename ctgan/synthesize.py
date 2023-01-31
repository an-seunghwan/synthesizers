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
from modules.model import *
from modules.data_sampler import *
from modules.datasets import generate_dataset

from evaluation.evaluation import (
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
    tags=['CTGAN', 'Synthetic'],
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
        'anseunghwan/DistVAE/CTGAN_{}:v{}'.format(config["dataset"], config["num"]), type='model')
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
    """dataset"""
    train_data, dataset, dataloader, transformer, train, test, continuous, discrete = generate_dataset(config, device, random_state=0)
    
    assert validate_discrete_columns(train, discrete) is None
    #%%
    """training-by-sampling"""
    data_sampler = DataSampler(
        train_data,
        transformer.output_info_list,
        config["log_frequency"])

    config["data_dim"] = transformer.output_dimensions
    #%%
    """model"""
    generator_dim = [int(x) for x in config["generator_dim"].split(',')]
    
    generator = Generator(
        config["latent_dim"] + data_sampler.dim_cond_vec(),
        generator_dim,
        config["data_dim"]
    ).to(device)
    
    if config["cuda"]:
        model_name = [x for x in os.listdir(model_dir) if x.endswith('pth')][0]
        generator.load_state_dict(
            torch.load(
                model_dir + '/' + model_name))
    else:
        model_name = [x for x in os.listdir(model_dir) if x.endswith('pth')][0]
        generator.load_state_dict(
            torch.load(
                model_dir + '/' + model_name, map_location=torch.device('cpu')))
    
    generator.eval()
    #%%
    if not os.path.exists('./assets/{}'.format(config["dataset"])):
        os.makedirs('./assets/{}'.format(config["dataset"]))
    #%%
    # preprocess
    train_mean = train[continuous].mean(axis=0)
    train_std = train[continuous].std(axis=0)
    train[continuous] = (train[continuous] - train_mean) / train_std
    test[continuous] = (test[continuous] - train_mean) / train_std
    
    df = pd.concat([train, test], axis=0)
    df_dummy = []
    for d in discrete:
        df_dummy.append(pd.get_dummies(df[d], prefix=d))
    df = pd.concat([df.drop(columns=discrete)] + df_dummy, axis=1)
    
    if config["dataset"] == "covtype":
        train = df.iloc[:45000]
        test = df.iloc[45000:]
    elif config["dataset"] == "credit":
        train = df.iloc[:45000]
        test = df.iloc[45000:]
    elif config["dataset"] == "loan":
        train = df.iloc[:4000]
        test = df.iloc[4000:]
    elif config["dataset"] == "adult":
        train = df.iloc[:40000]
        test = df.iloc[40000:]
    elif config["dataset"] == "cabs":
        train = df.iloc[:40000]
        test = df.iloc[40000:]
    elif config["dataset"] == "kings":
        train = df.iloc[:20000]
        test = df.iloc[20000:]
    else:
        raise ValueError('Not supported dataset!')
    #%%
    """synthetic dataset"""
    torch.manual_seed(config["seed"])
    n = len(train)
    
    steps = n // config["batch_size"] + 1
    data = []
    for i in range(steps):
        mean = torch.zeros(config["batch_size"], config["latent_dim"])
        std = mean + 1
        fakez = torch.normal(mean=mean, std=std).to(device)
        
        condvec = data_sampler.sample_original_condvec(config["batch_size"])
        c1 = condvec
        c1 = torch.from_numpy(c1).to(device)
        fakez = torch.cat([fakez, c1], dim=1)

        fake = generator(fakez)
        fakeact = apply_activate(fake, transformer, generator._gumbel_softmax)
        data.append(fakeact.detach().cpu().numpy())

    data = np.concatenate(data, axis=0)
    data = data[:n]

    sample_df = transformer.inverse_transform(data)
    #%%
    df_dummy = []
    for d in discrete:
        df_dummy.append(pd.get_dummies(sample_df[d], prefix=d))
    sample_df = pd.concat([sample_df.drop(columns=discrete)] + df_dummy, axis=1)
    #%%
    sample_mean = sample_df[continuous].mean(axis=0)
    sample_std = sample_df[continuous].std(axis=0)
    sample_df_scaled = sample_df.copy()
    sample_df_scaled[continuous] = (sample_df_scaled[continuous] - sample_mean) / sample_std
    #%%
    """Goodness of Fit""" # only continuous
    print("\nGoodness of Fit...\n")
    
    Dn, W1 = goodness_of_fit(len(continuous), train.to_numpy(), sample_df_scaled.to_numpy())
    
    print('Goodness of Fit (Kolmogorov): {:.3f}'.format(Dn))
    print('Goodness of Fit (1-Wasserstein): {:.3f}'.format(W1))
    wandb.log({'Goodness of Fit (Kolmogorov)': Dn})
    wandb.log({'Goodness of Fit (1-Wasserstein)': W1})
    #%%
    """Privacy Preservability""" # only continuous
    print("\nPrivacy Preservability...\n")
    
    privacy = privacy_metrics(train[continuous], sample_df_scaled[continuous])
    
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
    """Regression"""
    if config["dataset"] == "covtype":
        target = 'Elevation'
    elif config["dataset"] == "credit":
        target = 'AMT_CREDIT'
    elif config["dataset"] == "loan":
        target = 'Age'
    elif config["dataset"] == "adult":
        target = 'age'
    elif config["dataset"] == "cabs":
        target = 'Trip_Distance'
    elif config["dataset"] == "kings":
        target = 'long'
    else:
        raise ValueError('Not supported dataset!')
    #%%
    # standardization except for target variable
    real_train = train.copy()
    real_test = test.copy()
    real_train[target] = real_train[target] * train_std[target] + train_mean[target]
    real_test[target] = real_test[target] * train_std[target] + train_mean[target]
    
    cont = [x for x in continuous if x not in [target]]
    sample_df_scaled = sample_df.copy()
    sample_df_scaled[cont] = (sample_df_scaled[cont] - sample_mean[cont]) / sample_std[cont]
    #%%
    # baseline
    print("\nBaseline: Machine Learning Utility in Regression...\n")
    base_reg = regression_eval(real_train, real_test, target)
    wandb.log({'MARE (Baseline)': np.mean([x[1] for x in base_reg])})
    # wandb.log({'R^2 (Baseline)': np.mean([x[1] for x in base_reg])})
    #%%
    # CTGAN
    print("\nSynthetic: Machine Learning Utility in Regression...\n")
    reg = regression_eval(sample_df_scaled, real_test, target)
    wandb.log({'MARE': np.mean([x[1] for x in reg])})
    # wandb.log({'R^2': np.mean([x[1] for x in reg])})
    #%%
    # # visualization
    # fig = plt.figure(figsize=(5, 4))
    # plt.plot([x[1] for x in base_reg], 'o--', label='baseline')
    # plt.plot([x[1] for x in reg], 'o--', label='synthetic')
    # plt.ylim(0, 1)
    # plt.ylabel('MARE', fontsize=13)
    # # plt.ylabel('$R^2$', fontsize=13)
    # plt.xticks([0, 1, 2], [x[0] for x in base_reg], fontsize=13)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('./assets/{}/{}_MLU_regression.png'.format(config["dataset"], config["dataset"]))
    # # plt.show()
    # plt.close()
    # wandb.log({'ML Utility (Regression)': wandb.Image(fig)})
    #%%
    """Classification"""
    if config["dataset"] == "covtype":
        target = 'Cover_Type'
    elif config["dataset"] == "credit":
        target = 'TARGET'
    elif config["dataset"] == "loan":
        target = 'Personal Loan'
    elif config["dataset"] == "adult":
        target = 'income'
    elif config["dataset"] == "cabs":
        target = 'Surge_Pricing_Type'
    elif config["dataset"] == "kings":
        target = 'condition'
    else:
        raise ValueError('Not supported dataset!')
    #%%
    # baseline
    print("\nBaseline: Machine Learning Utility in Classification...\n")
    base_clf = classification_eval(train, test, target)
    wandb.log({'F1 (Baseline)': np.mean([x[1] for x in base_clf])})
    #%%
    sample_df_scaled = sample_df.copy()
    sample_df_scaled[continuous] = (sample_df_scaled[continuous] - sample_mean) / sample_std
    
    # CTGAN
    print("\nSynthetic: Machine Learning Utility in Classification...\n")
    clf = classification_eval(sample_df_scaled, test, target)
    wandb.log({'F1': np.mean([x[1] for x in clf])})
    #%%
    # plt.bar(np.arange(7) - 0.17, train.describe().loc['mean'][-7:],
    #         width=0.3, label="train")
    # plt.bar(np.arange(7) + 0.17, sample_df_scaled.describe().loc['mean'][-7:],
    #         width=0.3, label="synthetic")
    # plt.legend()
    #%%
    # # visualization
    # fig = plt.figure(figsize=(5, 4))
    # plt.plot([x[1] for x in base_clf], 'o--', label='baseline')
    # plt.plot([x[1] for x in clf], 'o--', label='synthetic')
    # plt.ylim(0, 1)
    # plt.ylabel('$F_1$', fontsize=13)
    # plt.xticks([0, 1, 2], [x[0] for x in base_clf], fontsize=13)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('./assets/{}/{}_MLU_classification.png'.format(config["dataset"], config["dataset"]))
    # # plt.show()
    # plt.close()
    # wandb.log({'ML Utility (Classification)': wandb.Image(fig)})
    #%%
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%