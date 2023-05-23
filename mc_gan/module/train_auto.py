#%%
import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset
#%%
def train_function(trainloader, autoencoder, optimizer, device):
    logs = {
        'loss': [], 
    }
    
    for x_batch in tqdm.tqdm(iter(trainloader), desc="inner loop"):
        
        x_batch = x_batch.to(device)
        
        loss_ = []
        optimizer.zero_grad()
        
        _, xhat = autoencoder(x_batch)
        
        loss = F.binary_cross_entropy(xhat, x_batch, reduction='none').sum(axis=1).mean()
                
        loss_.append(('loss', loss))
        
        loss.backward()
        optimizer.step()
            
        for x, y in loss_:
            logs[x] = logs.get(x) + [y.item()]
    
    return logs
#%%