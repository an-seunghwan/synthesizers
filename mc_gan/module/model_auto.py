#%%
"""
Reference:
https://github.com/rcamino/multi-categorical-gans/tree/master

Modification:
Tanh activation -> ELU activation
"""
#%%
import torch.nn as nn
#%%
class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_sizes=[]):
        super(Encoder, self).__init__()

        layer_sizes = list(hidden_sizes) + [embedding_dim]
        layers = []
        for layer_size in layer_sizes:
            layers.append(nn.Linear(input_dim, layer_size))
            layers.append(nn.ELU()) # nonlinear activation
            input_dim = layer_size
        self.hidden_layers = nn.Sequential(*layers[:-1])

    def forward(self, input):
        return self.hidden_layers(input)
#%%
class Decoder(nn.Module):
    def __init__(self, embedding_dim, output_dim, hidden_sizes=[]):
        super(Decoder, self).__init__()
        
        previous_layer_size = embedding_dim
        hidden_layers = []
        for layer_size in hidden_sizes:
            hidden_layers.append(nn.Linear(previous_layer_size, layer_size))
            hidden_layers.append(nn.ELU()) # nonlinear activation
            previous_layer_size = layer_size
        hidden_layers.extend(
            [nn.Linear(previous_layer_size, output_dim),
            nn.Sigmoid()]
        )
        self.hidden_layers = nn.Sequential(*hidden_layers)

    def forward(self, input):
        out = self.hidden_layers(input)
        return out
#%%
class AutoEncoder(nn.Module):
    def __init__(self, config, encoder_hidden_sizes=[], decoder_hidden_sizes=[]):
        super(AutoEncoder, self).__init__()
        self.config = config

        self.encoder = Encoder(
            config["p"], config["embedding_dim"], hidden_sizes=encoder_hidden_sizes)

        self.decoder = Decoder(
            config["embedding_dim"], config["p"], hidden_sizes=decoder_hidden_sizes)

    def forward(self, input):
        emb = self.encoder(input)
        recon = self.decoder(emb)
        return emb, recon
#%%