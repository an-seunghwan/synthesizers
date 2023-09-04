# %%
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# %%
import numpy as np
import pandas as pd
import tqdm
from PIL import Image
import matplotlib.pyplot as plt

plt.switch_backend("agg")

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset

from evaluation.simulation import set_random_seed
from modules.model import TVAE
from modules import shadow_datasets, datasets
from modules.train import train

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# %%
import sys
import subprocess

try:
    import wandb
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    with open("./wandb_api.txt", "r") as f:
        key = f.readlines()
    subprocess.run(["wandb", "login"], input=key[0], encoding="utf-8")
    import wandb

run = wandb.init(
    project="DistVAE",
    entity="anseunghwan",
    tags=["TVAE", "Privacy", "Attack"],
)
# %%
import argparse


def get_args(debug):
    parser = argparse.ArgumentParser("parameters")

    parser.add_argument("--num", type=int, default=0, help="model version")
    parser.add_argument(
        "--dataset",
        type=str,
        default="covtype",
        help="Dataset options: covtype, credit, loan, adult, cabs, kings",
    )

    if debug:
        return parser.parse_args(args=[])
    else:
        return parser.parse_args()


# %%
def main():
    # %%
    config = vars(get_args(debug=False))  # default configuration

    """model load"""
    K = 1  # the number of shadow models
    model_dirs = []
    for n in tqdm.tqdm(range(K), desc="Loading trained shadow models..."):
        num = config["num"] * K + n
        artifact = wandb.use_artifact(
            "anseunghwan/DistVAE/shadow_TVAE_{}:v{}".format(config["dataset"], num),
            type="model",
        )
        for key, item in artifact.metadata.items():
            config[key] = item
        model_dir = artifact.download()
        model_dirs.append(model_dir)

    config["cuda"] = torch.cuda.is_available()
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    wandb.config.update(config)

    set_random_seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["cuda"]:
        torch.cuda.manual_seed(config["seed"])
    # %%
    """Load shadow models"""
    models = []
    for k in range(len(model_dirs)):
        model_dir = model_dirs[k]

        model = TVAE(config, device).to(device)
        if config["cuda"]:
            model_name = [x for x in os.listdir(model_dir) if x.endswith("pth")][0]
            model.load_state_dict(torch.load(model_dir + "/" + model_name))
        else:
            model_name = [x for x in os.listdir(model_dir) if x.endswith("pth")][0]
            model.load_state_dict(
                torch.load(
                    model_dir + "/" + model_name, map_location=torch.device("cpu")
                )
            )

        model.eval()
        models.append(model)
    # %%
    if config["dataset"] == "covtype":
        target = "Cover_Type"
    elif config["dataset"] == "credit":
        target = "TARGET"
    elif config["dataset"] == "loan":
        target = "Personal Loan"
    elif config["dataset"] == "adult":
        target = "income"
    elif config["dataset"] == "cabs":
        target = "Surge_Pricing_Type"
    elif config["dataset"] == "kings":
        target = "condition"
    else:
        raise ValueError("Not supported dataset!")
    # %%
    """real dataset"""
    (
        dataset,
        dataloader,
        transformer,
        train,
        test,
        continuous,
        discrete,
    ) = datasets.generate_dataset(config, device, random_state=0)
    # %%
    """Load shadow data"""
    targets = []
    targets_test = []

    """training latent variables (in)"""
    latents = []
    """test latent variables (out)"""
    latents_test = []

    for k in range(len(model_dirs)):
        df_train = pd.read_csv(
            f'./privacy/{config["dataset"]}/train_{config["seed"]}_synthetic{k}.csv',
            index_col=0,
        )
        df_test = pd.read_csv(
            f'./privacy/{config["dataset"]}/test_{config["seed"]}_synthetic{k}.csv',
            index_col=0,
        )

        # if np.min(df_train[target].to_numpy()) == 0:
        #     targets.append(df_train[target].to_numpy())
        #     targets_test.append(df_test[target].to_numpy())
        # else:
        #     targets.append(df_train[target].to_numpy()-1)
        #     targets_test.append(df_test[target].to_numpy()-1)

        (
            _,
            shadow_dataloader,
            shadow_transformer,
            _,
            _,
            _,
            _,
        ) = shadow_datasets.generate_dataset(
            config, df_train, df_test, device, random_state=0
        )

        zs = []
        for (x_batch,) in tqdm.tqdm(iter(shadow_dataloader), desc="inner loop"):
            if config["cuda"]:
                x_batch = x_batch.cuda()
            with torch.no_grad():
                mean, _ = models[k].get_posterior(x_batch)
            zs.append(mean)
        zs = torch.cat(zs, dim=0)
        latents.append(zs)

        test_data = shadow_transformer.transform(df_test)
        shadow_dataset = TensorDataset(
            torch.from_numpy(test_data.astype("float32")).to(device)
        )
        shadow_dataloader = DataLoader(
            shadow_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            drop_last=False,
        )
        zs = []
        for (x_batch,) in tqdm.tqdm(iter(shadow_dataloader), desc="inner loop"):
            if config["cuda"]:
                x_batch = x_batch.cuda()
            with torch.no_grad():
                mean, _ = models[k].get_posterior(x_batch)
            zs.append(mean)
        zs = torch.cat(zs, dim=0)
        latents_test.append(zs)

        df_dummy = []
        for d in discrete:
            df_dummy.append(pd.get_dummies(df_train[d], prefix=d))
        df_train = pd.concat([df_train.drop(columns=discrete)] + df_dummy, axis=1)
        targets.append(
            df_train[[x for x in df_train.columns if x.startswith(target)]]
            .to_numpy()
            .argmax(axis=1)
        )

        df_dummy = []
        for d in discrete:
            df_dummy.append(pd.get_dummies(df_test[d], prefix=d))
        df_test = pd.concat([df_test.drop(columns=discrete)] + df_dummy, axis=1)
        targets_test.append(
            df_test[[x for x in df_test.columns if x.startswith(target)]]
            .to_numpy()
            .argmax(axis=1)
        )
    # %%
    """attack training records"""
    target_num = train[target].nunique()
    attack_training = {}
    for t in range(target_num):
        tmp1 = []
        for k in range(len(model_dirs)):
            tmp1.append(latents[k].numpy()[[targets[k] == t][0], :])
        tmp1 = np.concatenate(tmp1, axis=0)
        tmp1 = np.concatenate([tmp1, np.ones((len(tmp1), 1))], axis=1)

        tmp2 = []
        for k in range(len(model_dirs)):
            tmp2.append(latents_test[k].numpy()[[targets_test[k] == t][0], :])
        tmp2 = np.concatenate(tmp2, axis=0)
        tmp2 = np.concatenate([tmp2, np.zeros((len(tmp2), 1))], axis=1)

        tmp = np.concatenate([tmp1, tmp2], axis=0)

        attack_training[t] = tmp
    # %%
    """training attack model"""
    # if the number of category of synthetic != real
    agg_attacking_training = np.concatenate(
        [x for x in attack_training.values() if len(x)], axis=0
    )

    from sklearn.ensemble import GradientBoostingClassifier

    attackers = {}
    for t in range(target_num):
        if len(attack_training[t]):
            clf = GradientBoostingClassifier(random_state=0).fit(
                attack_training[t][:, : config["latent_dim"]], attack_training[t][:, -1]
            )
            attackers[t] = clf
        else:
            clf = GradientBoostingClassifier(random_state=0).fit(
                agg_attacking_training[:, : config["latent_dim"]],
                agg_attacking_training[:, -1],
            )
            attackers[t] = clf
    # %%
    """target model"""
    artifact = wandb.use_artifact(
        "anseunghwan/DistVAE/TVAE_{}:v{}".format(config["dataset"], config["num"]),
        type="model",
    )
    for key, item in artifact.metadata.items():
        config[key] = item
    model_dir = artifact.download()

    model = TVAE(config, device).to(device)
    if config["cuda"]:
        model_name = [x for x in os.listdir(model_dir) if x.endswith("pth")][0]
        model.load_state_dict(torch.load(model_dir + "/" + model_name))
    else:
        model_name = [x for x in os.listdir(model_dir) if x.endswith("pth")][0]
        model.load_state_dict(
            torch.load(model_dir + "/" + model_name, map_location=torch.device("cpu"))
        )

    model.eval()

    test_data = transformer.transform(test)
    test_dataset = TensorDataset(
        torch.from_numpy(test_data.astype("float32")).to(device)
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False, drop_last=False
    )
    # %%
    """Ground-truth training latent variables"""
    gt_latents = []
    for (x_batch,) in tqdm.tqdm(iter(dataloader), desc="inner loop"):
        if config["cuda"]:
            x_batch = x_batch.cuda()
        with torch.no_grad():
            mean, _ = model.get_posterior(x_batch)
        gt_latents.append(mean)
    gt_latents = torch.cat(gt_latents, dim=0)
    # %%
    """Ground-truth test latent variables"""
    gt_latents_test = []
    for (x_batch,) in tqdm.tqdm(iter(test_dataloader), desc="inner loop"):
        if config["cuda"]:
            x_batch = x_batch.cuda()
        with torch.no_grad():
            mean, _ = model.get_posterior(x_batch)
        gt_latents_test.append(mean)
    gt_latents_test = torch.cat(gt_latents_test, dim=0)
    # %%
    """attacker accuracy"""
    df_dummy = []
    for d in discrete:
        df_dummy.append(pd.get_dummies(train[d], prefix=d))
    train = pd.concat([train.drop(columns=discrete)] + df_dummy, axis=1)
    gt_targets = (
        train[[x for x in train.columns if x.startswith(target)]]
        .to_numpy()
        .argmax(axis=1)
    )

    df_dummy = []
    for d in discrete:
        df_dummy.append(pd.get_dummies(test[d], prefix=d))
    test = pd.concat([test.drop(columns=discrete)] + df_dummy, axis=1)
    gt_targets_test = (
        test[[x for x in test.columns if x.startswith(target)]]
        .to_numpy()
        .argmax(axis=1)
    )

    # if np.min(train[target].to_numpy()) == 0:
    #     gt_targets = train[target].to_numpy()
    #     gt_targets_test = test[target].to_numpy()
    # else:
    #     gt_targets = train[target].to_numpy()-1
    #     gt_targets_test = test[target].to_numpy()-1

    gt_latents = gt_latents[: len(gt_latents_test), :]
    gt_targets = gt_targets[: len(gt_latents_test)]

    pred = []
    for t in range(target_num):
        pred.append(attackers[t].predict(gt_latents[gt_targets == t]))
    for t in range(target_num):
        pred.append(attackers[t].predict(gt_latents_test[gt_targets_test == t]))
    pred = np.concatenate(pred)

    gt = np.zeros((len(pred),))
    gt[: len(gt_latents)] = 1

    acc = (gt == pred).mean()
    f1 = f1_score(gt, pred)
    auc = roc_auc_score(gt, pred)

    print("MI Accuracy: {:.3f}".format(acc))
    print("MI F1: {:.3f}".format(f1))
    print("MI AUC: {:.3f}".format(auc))
    wandb.log({"MI Accuracy": acc})
    wandb.log({"MI F1": f1})
    wandb.log({"MI AUC": auc})
    # %%
    wandb.run.finish()


# %%
if __name__ == "__main__":
    main()
# %%
