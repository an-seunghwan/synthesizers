#%%
import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset
#%%
def train_mnist(trainloader, enc, dec, optimizer, device):
    logs = {
        'loss': [], 
    }
    
    for (x_batch, _) in tqdm.tqdm(iter(trainloader), desc="inner loop"):
        
        x_batch = x_batch.to(device)
        
        loss_ = []
        optimizer.zero_grad()
        
        z = enc(x_batch)
        xhat = dec(z)
        
        loss = F.binary_cross_entropy(xhat, nn.Flatten()(x_batch), reduction='none').sum(axis=1).mean()
        loss_.append(('loss', loss))
        
        loss.backward()
        optimizer.step()
            
        for x, y in loss_:
            logs[x] = logs.get(x) + [y.item()]
    
    return logs
#%%
def train_function(trainloader, enc, dec, out, optimizer, device):
    OutputInfo_list = out[3]
    logs = {
        'loss': [], 
    }
    
    for x_batch in tqdm.tqdm(iter(trainloader), desc="inner loop"):
        
        x_batch = x_batch.to(device)
        
        loss_ = []
        optimizer.zero_grad()
        
        z = enc(x_batch)
        logit = dec(z)
        
        st = 0
        loss = 0
        for info in OutputInfo_list:
            ed = st + info.dim
            _, targets = x_batch[:, st : ed].max(dim=1)
            out = logit[:, st : ed]
            loss += nn.CrossEntropyLoss()(out, targets)
            st = ed
                
        loss_.append(('loss', loss))
        
        loss.backward()
        optimizer.step()
            
        for x, y in loss_:
            logs[x] = logs.get(x) + [y.item()]
    
    return logs
#%%