#%%
"""
Reference:
https://github.com/rcamino/multi-categorical-gans/tree/master

Modification:
Tanh activation -> ELU activation
"""
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.one_hot_categorical import OneHotCategorical
#%%
class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_sizes=[], device=None):
        super(Encoder, self).__init__()

        layer_sizes = list(hidden_sizes) + [embedding_dim]
        layers = []
        for layer_size in layer_sizes:
            layers.append(nn.Conv1d(1, layer_size, kernel_size=input_dim, stride=4))
            input_dim = layer_size
        self.hidden_layers = nn.Sequential(*layers).to(device)

    def forward(self, input):
        h = input
        for layer in self.hidden_layers:
            h = nn.ELU()(layer(h[:, None, :]))
            h = h.squeeze(2)
        return h
#%%
class Decoder(nn.Module):
    def __init__(self, embedding_dim, output_dim, hidden_sizes=[], device=None):
        super(Decoder, self).__init__()
        
        previous_layer_size = embedding_dim
        hidden_layers = []
        for layer_size in hidden_sizes:
            hidden_layers.append(nn.Conv1d(
                1, layer_size, kernel_size=previous_layer_size, stride=4))
            previous_layer_size = layer_size
        hidden_layers.extend([
            nn.Conv1d(1, output_dim, kernel_size=previous_layer_size, stride=4),
        ])
        self.hidden_layers = nn.Sequential(*hidden_layers).to(device)

    def forward(self, input, training=True):
        h = input
        for layer in self.hidden_layers[:-1]:
            h = nn.ELU()(layer(h[:, None, :]))
            h = h.squeeze(2)
        h = nn.Sigmoid()(self.hidden_layers[-1](h[:, None, :]))
        h = h.squeeze(2)
        return h
#%%
class AutoEncoder(nn.Module):
    def __init__(self, config, encoder_hidden_sizes=[], decoder_hidden_sizes=[], device=None):
        super(AutoEncoder, self).__init__()
        self.config = config

        self.encoder = Encoder(
            config["p"], config["embedding_dim"], hidden_sizes=encoder_hidden_sizes).to(device)

        self.decoder = Decoder(
            config["embedding_dim"], config["p"], hidden_sizes=decoder_hidden_sizes).to(device)

    def forward(self, input, training=True):
        emb = self.encoder(input)
        recon = self.decoder(emb, training=training)
        return emb, recon
#%%