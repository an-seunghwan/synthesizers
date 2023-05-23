#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

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
    parser.add_argument('--dataset', type=str, default='survey', 
                        help='Dataset options: mnist, census, survey')
    
    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=False)) # default configuration
    
    """model load"""
    artifact = wandb.use_artifact('anseunghwan/HDistVAE/{}_medGAN:v{}'.format(
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

    syndata = model.postprocess(syndata, OutputInfo_list, colnames, discrete_dicts, discrete_dicts_reverse)
    #%%
    """Data utility"""
    """1. KL-Divergence"""
    def KLDivergence(a, b):
        return np.sum(np.where(
            a != 0, 
            a * (np.log(a) - np.log(b)), 
            0))
    
    KL = []
    for col in tqdm.tqdm(syndata.columns, desc="KL-divergence..."):
        prob_df = pd.merge(
            pd.DataFrame(syndata[col].value_counts(normalize=True)).reset_index(),
            pd.DataFrame(train[col].value_counts(normalize=True)).reset_index(),
            how='outer',
            on=col
        )
        prob_df = prob_df.fillna(1e-12)
        KL.append(KLDivergence(prob_df['proportion_x'].to_numpy(), prob_df['proportion_y'].to_numpy()))
    print(f"KL: {np.mean(KL):.3f}")
    wandb.log({'KL': np.mean(KL)})
    #%%
    """2. Kolmogorov-Smirnov test"""
    KS = []
    for col in tqdm.tqdm(syndata.columns, desc="Kolmogorov-Smirnov test..."):
        train_ecdf = ECDF(train[col])
        syn_ecdf = ECDF(syndata[col])
        KS.append(np.abs(train_ecdf(train[col]) - syn_ecdf(train[col])).max())
    print(f"KS: {np.mean(KS):.3f}")
    wandb.log({'KS': np.mean(KS)})
    #%%
    """3. Support (Category) Coverage"""
    coverage = 0
    for col in tqdm.tqdm(syndata.columns, desc="Support (Category) Coverage..."):
        coverage += len(syndata[col].unique()) / len(train[col].unique())
    coverage /= len(syndata.columns)
    print(f"Coverage: {coverage:.3f}")
    wandb.log({'Coverage': coverage})
    #%%
    """4. MSE of dimension-wise probability"""
    syn_dim_prob = pd.get_dummies(syndata.astype(int).astype(str)).mean(axis=0)
    train_dim_prob = pd.get_dummies(train.astype(int).astype(str)).mean(axis=0)
    dim_prob = pd.merge(
        pd.DataFrame(syn_dim_prob).reset_index(),
        pd.DataFrame(train_dim_prob).reset_index(),
        how='outer',
        on='index'
    )
    dim_prob = dim_prob.fillna(0)
    mse_dim_prob = np.linalg.norm(dim_prob.iloc[:, 1].to_numpy() - dim_prob.iloc[:, 2].to_numpy())
    print(f"DimProb: {mse_dim_prob:.3f}")
    wandb.log({'DimProb': mse_dim_prob})
    #%%
    fig = plt.figure(figsize=(5, 5))
    plt.scatter(
        dim_prob.iloc[:, 1].to_numpy(), # synthetic
        dim_prob.iloc[:, 2].to_numpy(), # train
    )
    plt.axline((0, 0), slope=1, color='red')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('synthetic', fontsize=14)
    plt.ylabel('train', fontsize=14)
    plt.savefig('./assets/census_proportion.png')
    # plt.show()
    plt.close()
    wandb.log({'Proportion': wandb.Image(fig)})
    #%%
    """5. Pairwise correlation difference (PCD)"""
    syn_corr = np.corrcoef(syndata.T)
    train_corr = np.corrcoef(train.T)
    pcd = np.linalg.norm(syn_corr - train_corr)
    print(f"PCD(Pearson): {pcd:.3f}")
    wandb.log({'PCD(Pearson)': pcd})
    #%%
    """6. Kendall's tau rank correlation"""
    syn_tau = syndata.corr(method='kendall')
    train_tau = train.corr(method='kendall')
    pcd = np.linalg.norm(syn_tau - train_tau)
    print(f"PCD(Kendall): {pcd:.3f}")
    wandb.log({'PCD(Kendall)': pcd})
    #%%
    """7. log-cluster"""
    from sklearn.cluster import KMeans
    k = 20
    kmeans = KMeans(n_clusters=k, random_state=config["seed"])
    kmeans.fit(pd.concat([train, syndata], axis=0))
    
    logcluster = 0
    for c in range(k):
        n_total = (kmeans.labels_ == c).sum()
        n_train = (kmeans.labels_[:len(train)] == c).sum()
        logcluster += (n_train / n_total - 0.5) ** 2
    logcluster /= k
    logcluster = np.log(logcluster)
    print(f"logcluster: {logcluster:.3f}")
    wandb.log({'logcluster': logcluster})
    #%%
    """8. MSE of variable-wise prediction"""
    from sklearn.ensemble import RandomForestClassifier
    
    acc_synthetic = []
    acc_train = []
    for target in test.columns:
        # synthetic
        covariates = [x for x in syndata.columns if x != target]
        clf = RandomForestClassifier(random_state=0)
        clf.fit(
            syndata[covariates], 
            syndata[target])
        yhat = clf.predict(test[covariates])
        acc1 = (test[target].to_numpy() == yhat.squeeze()).mean()
        print('[{}] ACC(synthetic): {:.3f}'.format(target, acc1))
        
        # train
        covariates = [x for x in train.columns if x != target]
        clf = RandomForestClassifier(random_state=0)
        clf.fit(
            train[covariates],
            train[target])
        yhat = clf.predict(test[covariates])
        acc2 = (test[target].to_numpy() == yhat.squeeze()).mean()
        print('[{}] ACC(train): {:.3f}'.format(target, acc2))
        
        acc_synthetic.append(acc1)
        acc_train.append(acc2)
    
    mse_var_pred = np.linalg.norm(np.array(acc_synthetic) - np.array(acc_train))    
    print(f"VarPred: {mse_var_pred:.3f}")
    wandb.log({'VarPred': mse_var_pred})
    #%%
    fig = plt.figure(figsize=(5, 5))
    plt.scatter(acc_synthetic, acc_train)
    plt.axline((0, 0), slope=1, color='red')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('ACC(synthetic)', fontsize=14)
    plt.ylabel('ACC(train)', fontsize=14)
    plt.savefig('./assets/census_acc.png')
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
        plt.savefig('./assets/mnist.png')
        # plt.savefig('./assets/cifar10.png')
        # plt.show()
        plt.close()
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