#%%
import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset
from torch.nn.functional import cross_entropy
#%%
def train(OutputInfo_list, dataloader, model, config, optimizer, device):
    logs = {
        'loss': [], 
        'recon': [],
        'KL': [],
    }
    # for debugging
    logs['activated'] = []
    
    for (x_batch, ) in tqdm.tqdm(iter(dataloader), desc="inner loop"):
        
        if config["cuda"]:
            x_batch = x_batch.cuda()
        
        # with torch.autograd.set_detect_anomaly(True):    
        optimizer.zero_grad()
        
        mean, logvar, latent, xhat = model(x_batch)
        
        loss_ = []
        
        """reconstruction"""
        j = 0
        st = 0
        recon = 0
        for j, info in enumerate(OutputInfo_list):
            if info.activation_fn == "MSE":
                std = model.sigma[j]
                residual = x_batch[:, j] - xhat[:, j]
                recon += (residual ** 2 / 2 / (std ** 2)).mean()
                recon += torch.log(std)
            
            elif info.activation_fn == "softmax":
                ed = st + info.dim
                _, targets = x_batch[:, config["MSE_dim"] + st : config["MSE_dim"] + ed].max(dim=1)
                out = xhat[:, config["MSE_dim"] + st : config["MSE_dim"] + ed]
                recon += nn.CrossEntropyLoss()(out, targets)
                st = ed
        loss_.append(('recon', recon))
        
        """KL-Divergence"""
        KL = torch.pow(mean, 2).sum(axis=1)
        KL -= logvar.sum(axis=1)
        KL += torch.exp(logvar).sum(axis=1)
        KL -= config["latent_dim"]
        KL *= 0.5
        KL = KL.mean()
        loss_.append(('KL', KL))
        
        ### activated: for debugging
        var_ = torch.exp(logvar).mean(axis=0)
        loss_.append(('activated', (var_ < 0.1).sum()))
        
        loss = recon + KL 
        loss_.append(('loss', loss))
        
        loss.backward()
        optimizer.step()
        # model.sigma.data.clamp_(0.01, 0.1)
        model.sigma.data.clamp_(config["sigma_range"][0], config["sigma_range"][1])
            
        """accumulate losses"""
        for x, y in loss_:
            logs[x] = logs.get(x) + [y.item()]
    
    return logs
#%%