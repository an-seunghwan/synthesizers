#%%
import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
#%%
def train_function(trainloader, autoencoder, optimizer, config, device, epoch):
    logs = {
        'loss': [], 
    }
    
    """Gumbel-Softmax temperatur annealing"""
    tau = np.maximum(5 * np.exp(-0.025 * epoch), 2/3)
    
    # with torch.autograd.set_detect_anomaly(True):
    for x_batch in tqdm.tqdm(iter(trainloader), desc="inner loop"):
        
        x_batch = x_batch.to(device)
        
        loss_ = []
        optimizer.zero_grad()
        
        if config["mc"]:
            _, xhat = autoencoder(x_batch, training=True, temperature=tau, concat=False)
        else:
            _, xhat = autoencoder(x_batch)
        
        if config["mc"]:
            st = 0
            loss = 0
            for i, info in enumerate(autoencoder.OutputInfo_list):
                ed = st + info.dim
                batch_target = torch.argmax(x_batch[:, st : ed], dim=1)
                loss += F.cross_entropy(xhat[i], batch_target)
                # loss -= (x_batch[:, st : ed] * torch.log(xhat[i] + 1e-6)).sum(dim=1).mean()
                st = ed
        else:
            loss = F.binary_cross_entropy(xhat, x_batch, reduction='none').sum(axis=1).mean()
                
        loss_.append(('loss', loss))
        
        loss.backward()
        optimizer.step()
            
        for x, y in loss_:
            logs[x] = logs.get(x) + [y.item()]
    
    return logs
#%%