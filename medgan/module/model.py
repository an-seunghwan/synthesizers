#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
#%%
class GAN(nn.Module):
    def __init__(self, config, dec, device):
        super(GAN, self).__init__()
        
        self.config = config
        self.device = device
        
        """Discriminator"""
        self.discriminator = nn.Sequential(
            nn.Linear(config["p"], 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        ).to(device)
        
        """Generator"""
        self.generator = nn.Sequential(
            nn.Linear(config["latent_dim"], 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, config["embedding_dim"]),
            nn.BatchNorm1d(config["embedding_dim"]),
            nn.ReLU(),
        ).to(device)
        
        self.dec = dec # fine-tuned
    
    def sampling(self, batch_size):
        noise = torch.randn(batch_size, self.config["latent_dim"]).to(self.device) 
        return noise
    
    def forward(self, z):
        xhat = self.dec(self.generator(z))
        return z, xhat
    
    # def gumbel_sampling(self, size, eps = 1e-20):
    #     U = torch.rand(size)
    #     G = (- (U + eps).log() + eps).log()
    #     return G
    
    # def generate_data(self, n, seed):
    #     torch.random.manual_seed(seed)
        
    #     data = []
    #     steps = n // self.config["batch_size"] + 1
        
    #     with torch.no_grad():
    #         for _ in range(steps):
    #             randn = torch.randn(self.config["batch_size"], self.config["latent_dim"]) # prior
    #             gamma, beta = self.quantile_parameter(randn)

    #             samples = []
    #             for j in range(self.config["embedding_dim"]):
    #                 alpha = torch.rand(self.config["batch_size"], 1)
    #                 samples.append(self.quantile_function(alpha, gamma, beta, j))
                        
    #             samples = torch.cat(samples, dim=1)
    #             data.append(samples)
    #     data = torch.cat(data, dim=0)
    #     data = data[:n, :]
    #     data = self.dec(data)
    #     return data
    
    # def postprocess(self, syndata, OutputInfo_list, colnames, discrete_dicts_reverse):
    #     samples = []
    #     st = 0
    #     for j, info in enumerate(OutputInfo_list):
    #         ed = st + info.dim
    #         logit = syndata[:, st : ed]
            
    #         """Gumbel-Max Trick"""
    #         G = self.gumbel_sampling(logit.shape)
    #         _, logit = (nn.LogSoftmax(dim=1)(logit) + G).max(dim=1)
            
    #         samples.append(logit.unsqueeze(1))
    #         st = ed

    #     samples = torch.cat(samples, dim=1)
    #     syndata = pd.DataFrame(samples.numpy(), columns=colnames)

    #     """reverse to original column names"""
    #     for dis, disdict in zip(colnames, discrete_dicts_reverse):
    #         syndata[dis] = syndata[dis].apply(lambda x:disdict.get(x))
    #     return syndata
#%%