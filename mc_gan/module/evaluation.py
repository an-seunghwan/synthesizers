# %%
import pandas as pd
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from collections import namedtuple
from sklearn.metrics import f1_score

Metrics = namedtuple(
    "Metrics",
    [
        "KL",
        "KS",
        "coverage",
        "mse_dim_prob",
        "Proportion",
        "PCD_Pearson",
        "PCD_Kendall",
        "logcluster",
        "VarPred",
        "ACC",
    ],
)
#%%
def compute_HammingDistance(X, Y):
        return (X[:, None, :] != Y).sum(2)

# %%
def evaluate(syndata, train, test, config, model_name, show=False):
    """Data utility"""
    """1. KL-Divergence"""

    def KLDivergence(a, b):
        return np.sum(np.where(a != 0, a * (np.log(a) - np.log(b)), 0))

    KL = []
    for col in tqdm.tqdm(syndata.columns, desc="KL-divergence..."):
        prob_df = pd.merge(
            pd.DataFrame(syndata[col].value_counts(normalize=True)).reset_index(),
            pd.DataFrame(train[col].value_counts(normalize=True)).reset_index(),
            how="outer",
            on="index",
        )
        prob_df = prob_df.fillna(1e-12)
        KL.append(
            KLDivergence(prob_df[f"{col}_x"].to_numpy(), prob_df[f"{col}_y"].to_numpy())
        )

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
        how="outer",
        on="index",
    )
    dim_prob = dim_prob.fillna(0)
    mse_dim_prob = np.linalg.norm(
        dim_prob.iloc[:, 1].to_numpy() - dim_prob.iloc[:, 2].to_numpy()
    )

    fig1 = plt.figure(figsize=(5, 5))
    plt.scatter(
        dim_prob.iloc[:, 1].to_numpy(),  # synthetic
        dim_prob.iloc[:, 2].to_numpy(),  # train
    )
    plt.axline((0, 0), slope=1, color="red")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("synthetic", fontsize=14)
    plt.ylabel("train", fontsize=14)
    plt.savefig(f"./assets/{model_name}_proportion.png")
    if show:
        plt.show()
    # plt.show()
    plt.close()

    """5. Pairwise correlation difference (PCD)"""
    print("Pairwise correlation difference (PCD)...")
    syn_corr = np.corrcoef(syndata.T)
    train_corr = np.corrcoef(train.T)
    pcd_corr = np.linalg.norm(syn_corr - train_corr)

    """6. Kendall's tau rank correlation"""
    print("Kendall's tau rank correlation...")
    syn_tau = syndata.corr(method="kendall")
    train_tau = train.corr(method="kendall")
    pcd_kendall = np.linalg.norm(syn_tau - train_tau)

    """7. log-cluster"""
    print("log-cluster...")
    k = 20
    kmeans = KMeans(n_clusters=k, random_state=config["seed"])
    kmeans.fit(pd.concat([train, syndata], axis=0))

    logcluster = 0
    for c in range(k):
        n_total = (kmeans.labels_ == c).sum()
        n_train = (kmeans.labels_[: len(train)] == c).sum()
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
        clf.fit(syndata[covariates], syndata[target])
        yhat = clf.predict(test[covariates])
        acc1 = (test[target].to_numpy() == yhat.squeeze()).mean()
        print("[{}] ACC(synthetic): {:.3f}".format(target, acc1))

        # train
        covariates = [x for x in train.columns if x != target]
        clf = RandomForestClassifier(random_state=0)
        clf.fit(train[covariates], train[target])
        yhat = clf.predict(test[covariates])
        acc2 = (test[target].to_numpy() == yhat.squeeze()).mean()
        print("[{}] ACC(train): {:.3f}".format(target, acc2))

        acc_synthetic.append(acc1)
        acc_train.append(acc2)

    mse_var_pred = np.linalg.norm(np.array(acc_synthetic) - np.array(acc_train))

    fig2 = plt.figure(figsize=(5, 5))
    plt.scatter(acc_synthetic, acc_train)
    plt.axline((0, 0), slope=1, color="red")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("ACC(synthetic)", fontsize=14)
    plt.ylabel("ACC(train)", fontsize=14)
    plt.savefig(f"./assets/{model_name}_acc.png")
    if show:
        plt.show()
    # plt.show()
    plt.close()

    return Metrics(
        np.mean(KL),
        np.mean(KS),
        coverage,
        mse_dim_prob,
        fig1,
        pcd_corr,
        pcd_kendall,
        logcluster,
        mse_var_pred,
        fig2,
    )


# %%
def DCR_metric(train, synthetic, data_percent=15):
    
    """
    Reference:
    [1] https://github.com/Team-TUD/CTAB-GAN/blob/main/model/eval/evaluation.py
    
    Returns Distance to Closest Record
    
    Inputs:
    1) train -> real data
    2) synthetic -> corresponding synthetic data
    3) data_percent -> percentage of data to be sampled from real and synthetic datasets for computing Distance to Closest Record
    Outputs:
    1) List containing the 5th percentile distance to closest record (DCR) between real and synthetic as well as within real and synthetic datasets
    along with 5th percentile of nearest neighbour distance ratio (NNDR) between real and synthetic as well as within real and synthetic datasets
    
    """
    
    # Sampling smaller sets of real and synthetic data to reduce the time complexity of the evaluation
    real_sampled = train.sample(n=int(len(train)*(.01*data_percent)), random_state=42).to_numpy()
    fake_sampled = synthetic.sample(n=int(len(synthetic)*(.01*data_percent)), random_state=42).to_numpy()

    # Computing pair-wise distances between real and synthetic 
    dist_rf = compute_HammingDistance(real_sampled, fake_sampled)
    # Computing pair-wise distances within real 
    dist_rr = compute_HammingDistance(real_sampled, real_sampled)
    # Computing pair-wise distances within synthetic
    dist_ff = compute_HammingDistance(fake_sampled, fake_sampled) 
    
    # Removes distances of data points to themselves to avoid 0s within real and synthetic 
    rd_dist_rr = dist_rr[~np.eye(dist_rr.shape[0],dtype=bool)].reshape(dist_rr.shape[0],-1)
    rd_dist_ff = dist_ff[~np.eye(dist_ff.shape[0],dtype=bool)].reshape(dist_ff.shape[0],-1) 
    
    # Computing first and second smallest nearest neighbour distances between real and synthetic
    smallest_two_indexes_rf = [dist_rf[i].argsort()[:2] for i in range(len(dist_rf))]
    smallest_two_rf = [dist_rf[i][smallest_two_indexes_rf[i]] for i in range(len(dist_rf))]       
    # Computing first and second smallest nearest neighbour distances within real
    smallest_two_indexes_rr = [rd_dist_rr[i].argsort()[:2] for i in range(len(rd_dist_rr))]
    smallest_two_rr = [rd_dist_rr[i][smallest_two_indexes_rr[i]] for i in range(len(rd_dist_rr))]
    # Computing first and second smallest nearest neighbour distances within synthetic
    smallest_two_indexes_ff = [rd_dist_ff[i].argsort()[:2] for i in range(len(rd_dist_ff))]
    smallest_two_ff = [rd_dist_ff[i][smallest_two_indexes_ff[i]] for i in range(len(rd_dist_ff))]
    
    # Computing 5th percentiles for DCR and NNDR between and within real and synthetic datasets
    min_dist_rf = np.array([i[0] for i in smallest_two_rf])
    fifth_perc_rf = np.percentile(min_dist_rf,5)
    min_dist_rr = np.array([i[0] for i in smallest_two_rr])
    fifth_perc_rr = np.percentile(min_dist_rr,5)
    min_dist_ff = np.array([i[0] for i in smallest_two_ff])
    fifth_perc_ff = np.percentile(min_dist_ff,5)
    # nn_ratio_rf = np.array([i[0]/i[1] for i in smallest_two_rf])
    # nn_fifth_perc_rf = np.percentile(nn_ratio_rf,5)
    # nn_ratio_rr = np.array([i[0]/i[1] for i in smallest_two_rr])
    # nn_fifth_perc_rr = np.percentile(nn_ratio_rr,5)
    # nn_ratio_ff = np.array([i[0]/i[1] for i in smallest_two_ff])
    # nn_fifth_perc_ff = np.percentile(nn_ratio_ff,5)
    
    return [fifth_perc_rf,fifth_perc_rr,fifth_perc_ff]
    # return np.array([fifth_perc_rf,fifth_perc_rr,fifth_perc_ff,nn_fifth_perc_rf,nn_fifth_perc_rr,nn_fifth_perc_ff]).reshape(1,6) 
#%%
def attribute_disclosure(K, compromised, syndata, attr_compromised):
    dist = compute_HammingDistance(
        compromised[attr_compromised].to_numpy(),
        syndata[attr_compromised].to_numpy())
    K_idx = dist.argsort(axis=1)[:, :K]
    
    def most_common(lst):
        return max(set(lst), key=lst.count)
    
    votes = []
    trues = []
    for i in tqdm.tqdm(range(len(K_idx)), desc="Marjority vote..."):
        true = np.zeros((len(compromised.columns) - len(attr_compromised), ))
        vote = np.zeros((len(compromised.columns) - len(attr_compromised), ))
        for j in range(len(compromised.columns) - len(attr_compromised)):
            true[j] = compromised.to_numpy()[i, len(attr_compromised) + j]
            vote[j] = most_common(list(syndata.to_numpy()[K_idx[i], len(attr_compromised) + j]))
        votes.append(vote)
        trues.append(true)
    votes = np.vstack(votes)
    trues = np.vstack(trues)
    
    acc = 0
    f1 = 0
    for j in range(trues.shape[1]):
        acc += (trues[:, j] == votes[:, j]).mean()
        f1 += f1_score(trues[:, j], votes[:, j], average="macro", zero_division=0)
    acc /= trues.shape[1]
    f1 /= trues.shape[1]

    return acc, f1
#%%
def privacyloss(train, test, syndata, data_percent=15):
    # Sampling smaller sets of real and synthetic data to reduce the time complexity of the evaluation
    train_sampled = test.sample(n=int(len(train)*(.01*data_percent)), random_state=42).to_numpy()
    test_sampled = test.sample(n=int(len(test)*(.01*data_percent)), random_state=42).to_numpy()
    syndata_sampled = syndata.sample(n=int(len(syndata)*(.01*data_percent)), random_state=42).to_numpy()
    
    # train
    dist = compute_HammingDistance(
        train_sampled,
        syndata_sampled)
    dist_TS = dist.min(axis=1, keepdims=True)
    dist_ST = dist.min(axis=0, keepdims=True).T
    
    dist = compute_HammingDistance(
        train_sampled,
        train_sampled)
    dist.sort(axis=1)
    dist_TT = dist[:, [1]] # leave-one-out
    
    dist = compute_HammingDistance(
        syndata_sampled,
        syndata_sampled)
    dist.sort(axis=1)
    dist_SS = dist[:, [1]] # leave-one-out
    
    AA_train = (dist_TS > dist_TT).mean() + (dist_ST > dist_SS).mean()
    AA_train /= 2
    
    # test
    dist = compute_HammingDistance(
        test_sampled,
        syndata_sampled)
    dist_TS = dist.min(axis=1, keepdims=True)
    dist_ST = dist.min(axis=0, keepdims=True).T
    
    dist = compute_HammingDistance(
        test_sampled,
        test_sampled)
    dist.sort(axis=1)
    dist_TT = dist[:, [1]] # leave-one-out
    
    dist = compute_HammingDistance(
        syndata_sampled,
        syndata_sampled)
    dist.sort(axis=1)
    dist_SS = dist[:, [1]] # leave-one-out
    
    AA_test = (dist_TS > dist_TT).mean() + (dist_ST > dist_SS).mean()
    AA_test /= 2
    
    AA = np.abs(AA_train - AA_test)
    
    return AA_train, AA_test, AA
#%%