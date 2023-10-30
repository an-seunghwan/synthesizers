# %%
import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


# %%
def train_function(trainloader, autoencoder, optimizer, config, device, epoch):
    logs = {
        "loss": [],
    }

    if config["mc"]:
        """Gumbel-Softmax temperatur annealing"""
        tau = np.maximum(5 * np.exp(-0.025 * epoch), config["tau"])

    # with torch.autograd.set_detect_anomaly(True):
    for x_batch in tqdm.tqdm(iter(trainloader), desc="inner loop"):
        x_batch = x_batch.to(device)

        loss_ = []
        optimizer.zero_grad()

        if config["mc"]:
            _, xhat = autoencoder(x_batch, training=True, temperature=tau, concat=False)
        else:
            _, xhat = autoencoder(x_batch, training=True)

        if config["mc"]:
            st = 0
            loss = 0
            for i, info in enumerate(autoencoder.OutputInfo_list):
                ed = st + info.dim
                batch_target = torch.argmax(x_batch[:, st:ed], dim=1)
                loss += F.cross_entropy(xhat[i], batch_target)
                st = ed
        else:
            loss = (
                F.binary_cross_entropy(xhat, x_batch, reduction="none")
                .sum(axis=1)
                .mean()
            )

        loss_.append(("loss", loss))

        loss.backward()
        optimizer.step()

        for x, y in loss_:
            logs[x] = logs.get(x) + [y.item()]

    return logs


# %%
def phi(s, D):
    return (1 + (4 * s) / (2 * D - 3)) ** (-1 / 2)
#%%
def train_CW2(
    dataloader,
    autoencoder,
    config,
    optimizer,
    device,
):
    logs = {
        "loss": [],
        "recon": [],
        "cw_dist": [],
    }
    
    for x_batch in tqdm.tqdm(iter(dataloader), desc="inner loop"):
        x_batch = x_batch.to(device)

        loss_ = []

        optimizer.zero_grad()

        emb, xhat = autoencoder(x_batch)
        
        gamma = (4 / (3 * x_batch.size(0))) ** (2 / 5)
        
        """1. reconstruction"""
        cw1 = torch.cdist(x_batch, x_batch) ** 2 / (4 * gamma)
        cw2 = torch.cdist(xhat, xhat) ** 2 / (4 * gamma)
        cw3 = torch.cdist(x_batch, xhat) ** 2 / (4 * gamma)
        recon = phi(cw1, D=x_batch.size(1)).sum()
        recon += phi(cw2, D=x_batch.size(1)).sum()
        recon += -2 * phi(cw3, D=x_batch.size(1)).sum()
        recon /= (2 * x_batch.size(0) ** 2 * torch.tensor(torch.pi * gamma).sqrt())
        recon = recon.log()
        loss_.append(("recon", recon))
        
        """2. regularization"""
        cw1 = torch.cdist(emb, emb) ** 2 / (4 * gamma)
        cw2 = x_batch.size(0) ** 2 / np.sqrt(1 + gamma)
        cw3 = torch.linalg.vector_norm(emb, dim=-1).pow(2) / (2 + 4 * gamma)
        cw_dist = phi(cw1, D=x_batch.size(1)).sum() / np.sqrt(gamma)
        cw_dist += cw2
        cw_dist += - phi(cw3, D=x_batch.size(1)).sum() * (2 * x_batch.size(0) / np.sqrt(gamma + .5))
        cw_dist /= (2 * x_batch.size(0) ** 2 * torch.tensor(torch.pi).sqrt())
        cw_dist = cw_dist.log()
        loss_.append(("cw_dist", cw_dist))
    
        loss = recon + config["lambda"] * cw_dist        
        loss_.append(("loss", loss))

        loss.backward()
        optimizer.step()

        """accumulate losses"""
        for x, y in loss_:
            logs[x] = logs.get(x) + [y.item()]

    return logs


# %%
