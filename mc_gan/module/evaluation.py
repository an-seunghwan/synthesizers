#%%
import pandas as pd
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from collections import namedtuple
Metrics = namedtuple(
    'Metrics', 
    ['KL', 'KS', 'coverage', 'mse_dim_prob', 'Proportion', 'PCD_Pearson', 'PCD_Kendall', 'logcluster', 'VarPred', 'ACC'])
#%%
def evaluate(syndata, train, test, config, show=False):
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
    
    """2. Kolmogorov-Smirnov test"""
    KS = []
    for col in tqdm.tqdm(syndata.columns, desc="Kolmogorov-Smirnov test..."):
        train_ecdf = ECDF(train[col])
        syn_ecdf = ECDF(syndata[col])
        KS.append(np.abs(train_ecdf(train[col]) - syn_ecdf(train[col])).max())
    
    """3. Support (Category) Coverage"""
    coverage = 0
    for col in tqdm.tqdm(syndata.columns, desc="Support (Category) Coverage..."):
        coverage += len(syndata[col].unique()) / len(train[col].unique())
    coverage /= len(syndata.columns)
    
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
    
    fig1 = plt.figure(figsize=(5, 5))
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
    if show: plt.show()
    # plt.show()
    plt.close()
    
    """5. Pairwise correlation difference (PCD)"""
    print("Pairwise correlation difference (PCD)...")
    syn_corr = np.corrcoef(syndata.T)
    train_corr = np.corrcoef(train.T)
    pcd_corr = np.linalg.norm(syn_corr - train_corr)
    
    """6. Kendall's tau rank correlation"""
    print("Kendall's tau rank correlation...")
    syn_tau = syndata.corr(method='kendall')
    train_tau = train.corr(method='kendall')
    pcd_kendall = np.linalg.norm(syn_tau - train_tau)
    
    """7. log-cluster"""
    print("log-cluster...")
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
    
    """8. MSE of variable-wise prediction"""
    print("MSE of variable-wise prediction...")
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

    fig2 = plt.figure(figsize=(5, 5))
    plt.scatter(acc_synthetic, acc_train)
    plt.axline((0, 0), slope=1, color='red')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('ACC(synthetic)', fontsize=14)
    plt.ylabel('ACC(train)', fontsize=14)
    plt.savefig('./assets/census_acc.png')
    if show: plt.show()
    # plt.show()
    plt.close()
    
    return Metrics(np.mean(KL), np.mean(KS), coverage, mse_dim_prob, fig1, pcd_corr, pcd_kendall, logcluster, mse_var_pred, fig2)
#%%