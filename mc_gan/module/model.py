#%%
"""
Reference:
https://github.com/rcamino/multi-categorical-gans/tree/master
"""
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
#%%
class medGANDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_sizes):
        super(medGANDiscriminator, self).__init__()

        previous_layer_size = input_dim * 2
        layers = []
        for layer_size in hidden_sizes:
            layers.append(nn.Linear(previous_layer_size, layer_size))
            layers.append(nn.LeakyReLU())
            previous_layer_size = layer_size
        layers.append(nn.Linear(previous_layer_size, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def minibatch_averaging(self, input):
        """
        This method is explained in the MedGAN paper.
        """
        mean_per_feature = input.mean(dim=0)
        mean_per_feature_repeated = mean_per_feature.repeat(len(input), 1)
        return torch.cat((input, mean_per_feature_repeated), dim=1)

    def forward(self, input):
        input = self.minibatch_averaging(input)
        return self.net(input).view(-1)
#%%
class medGANGenerator(nn.Module):
    def __init__(self, embedding_dim, num_hidden_layers=2, bn_decay=0.01):
        super(medGANGenerator, self).__init__()

        self.modules = []
        self.batch_norms = []
        for layer_number in range(num_hidden_layers):
            self.add_generator_module(
                "hidden_{:d}".format(layer_number + 1),
                embedding_dim, 
                nn.ReLU(), 
                bn_decay)
        self.add_generator_module(
            "output", 
            embedding_dim, 
            nn.Tanh(), 
            bn_decay)

    def add_generator_module(self, name, embedding_dim, activation, bn_decay):
        batch_norm = nn.BatchNorm1d(embedding_dim, momentum=(1 - bn_decay))
        module = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim, bias=False),  # bias is not necessary because of the batch normalization
            batch_norm,
            activation
        )
        self.modules.append(module)
        self.add_module(name, module)
        self.batch_norms.append(batch_norm)

    def batch_norm_train(self, mode=True):
        for batch_norm in self.batch_norms:
            batch_norm.train(mode=mode)

    def forward(self, noise):
        outputs = noise
        for module in self.modules:
            # Cannot write "outputs += module(outputs)" because it is an inplace operation (no differentiable)
            outputs = module(outputs) + outputs  # shortcut connection
        return outputs
#%%
# class GAN(nn.Module):
#     def __init__(self, config, dec, device):
#         super(GAN, self).__init__()
        
#         self.config = config
#         self.device = device
        
#         """Discriminator"""
#         self.discriminator = nn.Sequential(
#             nn.Linear(config["p"], 8),
#             nn.ReLU(),
#             nn.Linear(8, 1),
#             nn.Sigmoid()
#         ).to(device)
        
#         """Generator"""
#         self.generator = nn.Sequential(
#             nn.Linear(config["latent_dim"], 8),
#             nn.BatchNorm1d(8),
#             nn.ReLU(),
#             nn.Linear(8, 16),
#             nn.BatchNorm1d(16),
#             nn.ReLU(),
#             nn.Linear(16, 32),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.Linear(32, config["embedding_dim"]),
#             nn.BatchNorm1d(config["embedding_dim"]),
#             nn.ReLU(),
#         ).to(device)
        
#         self.dec = dec # fine-tuned
    
#     def sampling(self, batch_size):
#         noise = torch.randn(batch_size, self.config["latent_dim"]).to(self.device) 
#         return noise
    
#     def forward(self, z):
#         xhat = self.dec(self.generator(z))
#         return z, xhat
    
#     def gumbel_sampling(self, size, eps = 1e-20):
#         U = torch.rand(size)
#         G = (- (U + eps).log() + eps).log()
#         return G
    
#     def generate_data(self, n, seed):
#         torch.random.manual_seed(seed)
        
#         data = []
#         steps = n // self.config["batch_size"] + 1
        
#         with torch.no_grad():
#             for _ in range(steps):
#                 z = self.sampling(self.config["batch_size"])
#                 data.append(self.forward(z)[1])
#         data = torch.cat(data, dim=0)
#         data = data[:n, :]
#         return data
    
#     def postprocess(self, syndata, OutputInfo_list, colnames, discrete_dicts, discrete_dicts_reverse):
#         samples = []
#         st = 0
#         for j, info in enumerate(OutputInfo_list):
#             ed = st + info.dim
#             logit = syndata[:, st : ed]
            
#             """argmax"""
#             _, logit = logit.max(dim=1)
            
#             samples.append(logit.unsqueeze(1))
#             st = ed

#         samples = torch.cat(samples, dim=1)
#         syndata = pd.DataFrame(samples.numpy(), columns=colnames)

#         """reverse to original column names"""
#         for dis, disdict in zip(colnames, discrete_dicts_reverse):
#             syndata[dis] = syndata[dis].apply(lambda x:disdict.get(x))
#         for dis, disdict in zip(colnames, discrete_dicts):
#             syndata[dis] = syndata[dis].apply(lambda x:disdict.get(x))
#         return syndata
# #%%