#%%
import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset
#%%
def train_GAN(dataloader, model, config, optimizer_D, optimizer_G, device):
    criterion = nn.BCELoss()
    
    logs = {
        'loss_D': [], 
        'loss_G': [], 
    }
    
    for (x_batch) in tqdm.tqdm(iter(dataloader), desc="inner loop"):
        
        x_batch = x_batch.to(device)
        
        optimizer_D.zero_grad()
        optimizer_G.zero_grad()
        
        loss_ = []
        
        z = model.sampling(x_batch.size(0))
        real_label = torch.ones((x_batch.size(0), 1))
        fake_label = torch.zeros((x_batch.size(0), 1))
        
        # discriminator training
        true = model.discriminator(x_batch)
        loss_D_real = criterion(true, real_label)
        
        _, xhat = model(z) # generator
        fake = model.discriminator(xhat)
        loss_D_fake = criterion(fake, fake_label)
        
        loss_D = loss_D_real + loss_D_fake
        loss_D.backward()
        loss_.append(('loss_D', loss_D))
        optimizer_D.step()
        
        # generator training
        _, xhat = model(z) # generator
        fake = model.discriminator(xhat)
        loss_G = criterion(fake, real_label)
        loss_.append(('loss_G', loss_G))
        loss_G.backward()
        optimizer_G.step()
        
        """accumulate losses"""
        for x, y in loss_:
            logs[x] = logs.get(x) + [y.item()]
    
    return logs
#%%