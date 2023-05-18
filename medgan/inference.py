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
    tags=['medGAN', 'inference'],
)
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--num', type=int, default=0, 
                        help='model number')
    parser.add_argument('--dataset', type=str, default='census', 
                        help='Dataset options: mnist, census')
    
    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=False)) # default configuration
    
    """model load"""
    artifact = wandb.use_artifact('anseunghwan/HDistVAE/medGAN_{}:v{}'.format(
        config["dataset"], config["num"]), type='model')
    for key, item in artifact.metadata.items():
        config[key] = item
    model_dir = artifact.download()
    
    torch.manual_seed(config["seed"])
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    config["cuda"] = torch.cuda.is_available()
    wandb.config.update(config)
    #%%
    out = build_dataset(config)
    dataset = out[0]
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    if config["dataset"] == "mnist": config["p"] = 784
    elif config["dataset"] == "census": config["p"] = dataset.p
    else: raise ValueError('Not supported dataset!')
    
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
    train = out[1]
    test = out[2]
    OutputInfo_list = out[3]
    discrete_dicts = out[4]
    discrete_dicts_reverse = out[5]
    colnames = out[6]

    n = len(train)
    with torch.no_grad():
        syndata = model.generate_data(n, seed=1)

    syndata = model.postprocess(syndata, OutputInfo_list, colnames, discrete_dicts_reverse)
    #%%
    # computational issue -> sampling 5000
    tmp1 = syndata.astype(int).to_numpy()[:5000, :]
    tmp2 = train.astype(int).to_numpy()[:5000, :] # train

    hamming = np.zeros((tmp1.shape[0], tmp2.shape[0]))
    for i in tqdm.tqdm(range(len(tmp1))):
        hamming[i, :] = (tmp2 - tmp1[[i]] != 0).mean(axis=1)
    #%%
    fig = plt.figure(figsize=(6, 4))
    plt.hist(hamming.flatten(), density=True, bins=20)
    plt.axvline(np.quantile(hamming.flatten(), 0.05), color='red')
    plt.xlabel('Hamming Dist', fontsize=13)
    plt.ylabel('density', fontsize=13)
    plt.savefig('./assets/medgan_census_hamming.png')
    # plt.show()
    plt.close()
    wandb.log({'Hamming Distance': wandb.Image(fig)})
    #%%
    """Proportion of one-hot vector"""
    syn_result = pd.get_dummies(syndata.astype(int).astype(str)).mean(axis=0)
    train_result = pd.get_dummies(train.astype(int).astype(str)).mean(axis=0)
    test_result = pd.get_dummies(test.astype(int).astype(str)).mean(axis=0)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    inter_cols = [x for x in train_result.index if x in syn_result]
    # len(inter_cols) / len(syn_result)
    ax[0].scatter(
        syn_result[inter_cols], train_result[inter_cols],
    )
    ax[0].axline((0, 0), slope=1, color='red')
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(0, 1)
    ax[0].set_xlabel('proportion(synthetic)', fontsize=14)
    ax[0].set_ylabel('proportion(train)', fontsize=14)
    inter_cols = [x for x in test_result.index if x in syn_result]
    # len(inter_cols) / len(syn_result)
    ax[1].scatter(
        syn_result[inter_cols], test_result[inter_cols],
    )
    ax[1].axline((0, 0), slope=1, color='red')
    ax[1].set_xlim(0, 1)
    ax[1].set_ylim(0, 1)
    ax[1].set_xlabel('proportion(synthetic)', fontsize=14)
    ax[1].set_ylabel('proportion(test)', fontsize=14)
    plt.savefig('./assets/medgan_census_proportion.png')
    # plt.show()
    plt.close()
    wandb.log({'Proportion': wandb.Image(fig)})
    #%%
    """(MLu) Classification accuracy 
    -> TOO MUCH COMPUTATIONAL COST 
    1) choose subset of covariates
    2) Logistic regression (simple model)"""
    import warnings
    warnings.filterwarnings("ignore")
    # from catboost import CatBoostClassifier

    from sklearn.linear_model import LogisticRegression

    # test.describe().transpose()
    # target = 'MODEM'

    acc_synthetic = []
    acc_real = []
    for target in test.columns:
        # synthetic
        covariates = [x for x in syndata.columns if x != target]
        # clf = CatBoostClassifier(verbose=0, iterations=100, random_seed=0)
        clf = LogisticRegression(
            random_state=0, fit_intercept=True,
            multi_class='ovr', max_iter=100)
        clf.fit(
            syndata[covariates], 
            syndata[target])
            # cat_features=covariates)
        yhat = clf.predict(test[covariates])
        acc1 = (test[target].to_numpy() == yhat.squeeze()).mean()
        print('[{}] ACC(synthetic): {:.3f}'.format(target, acc1))
        
        # real
        covariates = [x for x in out[2].columns if x != target]
        # clf = CatBoostClassifier(verbose=0, iterations=100, random_seed=0)
        clf = LogisticRegression(
            random_state=0, fit_intercept=True,
            multi_class='ovr', max_iter=100)
        clf.fit(
            train[covariates],
            train[target])
            # cat_features=covariates)
        yhat = clf.predict(test[covariates])
        acc2 = (test[target].to_numpy() == yhat.squeeze()).mean()
        print('[{}] ACC(real): {:.3f}'.format(target, acc2))
        
        acc_synthetic.append((target, acc1))
        acc_real.append((target, acc2))
    #%%
    synthetic_acc_mean = np.mean([x[1] for x in acc_synthetic])
    real_acc_mean = np.mean([x[1] for x in acc_real])
    wandb.log({'ACC(synthetic)': synthetic_acc_mean})
    wandb.log({'ACC(real)': real_acc_mean})
    
    fig = plt.figure(figsize=(5, 5))
    # plt.bar(range(len(acc_real)), [x[1] - y[1] for x, y in zip(acc_real, acc_synthetic)])
    plt.scatter(
        [x[1] for x in acc_synthetic],
        [x[1] for x in acc_real],
    )
    plt.axline((0, 0), slope=1, color='red')
    plt.xlabel('ACC(synthetic)', fontsize=14)
    plt.ylabel('ACC(train)', fontsize=14)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig('./assets/medgan_census_acc.png')
    # plt.show()
    plt.close()
    wandb.log({'ACC': wandb.Image(fig)})
    #%%
    if config["dataset"] == 'mnist':
        n = 100
        with torch.no_grad():
            syndata = model.generate_data(n, seed=1)
        # syndata = torch.where(syndata > 0.5, 1, 0)
        syndata = syndata.reshape(-1, 28, 28)
        fig, ax = plt.subplots(10, 10, figsize=(20, 20))
        for j in range(n):
            ax.flatten()[j].imshow(syndata[j], cmap='gray_r')
            ax.flatten()[j].axis('off')
        plt.tight_layout()
        plt.savefig('./assets/medgan_mnist.png')
        # plt.savefig('./assets/medgan_cifar10.png')
        # plt.show()
        plt.close()
    #%%
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%