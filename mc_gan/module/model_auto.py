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
    def __init__(self, input_dim, embedding_dim, hidden_sizes=[]):
        super(Encoder, self).__init__()

        layer_sizes = list(hidden_sizes) + [embedding_dim]
        layers = []
        for layer_size in layer_sizes:
            layers.append(nn.Linear(input_dim, layer_size))
            layers.append(nn.ELU()) # nonlinear activation
            input_dim = layer_size
        self.hidden_layers = nn.Sequential(*layers[:-1]) # exclude last nonlinear activation

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
        hidden_layers.extend([
            nn.Linear(previous_layer_size, output_dim),
            nn.Sigmoid()
        ])
        self.hidden_layers = nn.Sequential(*hidden_layers)

    def forward(self, input, training=True):
        return self.hidden_layers(input)
#%%
class CategoricalDecoder(nn.Module):
    def __init__(self, embedding_dim, hidden_sizes=[], OutputInfo_list=None):
        super(CategoricalDecoder, self).__init__()
        
        previous_layer_size = embedding_dim
        hidden_layers = []
        for layer_size in hidden_sizes:
            hidden_layers.append(nn.Linear(previous_layer_size, layer_size))
            hidden_layers.append(nn.ELU()) # nonlinear activation
            previous_layer_size = layer_size
        self.hidden_layers = nn.Sequential(*hidden_layers)
        
        self.output_layers = nn.ModuleList()
        self.output_activations = nn.ModuleList()
        for info in OutputInfo_list:
            self.output_layers.append(nn.Linear(previous_layer_size, info.dim))
            self.output_activations.append(CategoricalActivation())

    def forward(self, input, training=True, temperature=0.666, concat=True):
        hidden = self.hidden_layers(input)
        """Multi-Categorical Setting"""
        outputs = []
        for output_layer, output_activation in zip(self.output_layers, self.output_activations):
            logits = output_layer(hidden)
            output = output_activation(logits, training=training, temperature=temperature)
            outputs.append(output)
        if concat:
            return torch.cat(outputs, dim=1)
        else:
            return outputs
#%%
class CategoricalActivation(nn.Module):
    def __init__(self):
        super(CategoricalActivation, self).__init__()

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape)
        return -Variable(torch.log(-torch.log(U + eps) + eps))

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, hard=True, temperature=None):
        y = self.gumbel_softmax_sample(logits, temperature)
        if not hard:
            return y
        else:
            shape = y.size()
            _, ind = y.max(dim=-1)
            y_hard = torch.zeros_like(y).view(-1, shape[-1])
            y_hard.scatter_(1, ind.view(-1, 1), 1)
            y_hard = y_hard.view(*shape)
            return (y_hard - y).detach() + y
    
    def forward(self, logits, training=True, temperature=None):
        # gumbel-softmax (training and evaluation)
        if temperature is not None:
            return self.gumbel_softmax(logits, hard=not training, temperature=temperature)
        # softmax training
        elif training:
            return F.softmax(logits, dim=1)
        # softmax evaluation
        else:
            return OneHotCategorical(logits=logits).sample()
#%%
class AutoEncoder(nn.Module):
    def __init__(self, config, encoder_hidden_sizes=[], decoder_hidden_sizes=[], OutputInfo_list=None):
        super(AutoEncoder, self).__init__()
        self.config = config
        self.OutputInfo_list = OutputInfo_list

        self.encoder = Encoder(
            config["p"], config["embedding_dim"], hidden_sizes=encoder_hidden_sizes)

        if self.OutputInfo_list is not None:
            self.decoder = CategoricalDecoder(
                config["embedding_dim"], hidden_sizes=decoder_hidden_sizes, OutputInfo_list=OutputInfo_list)
        else:
            self.decoder = Decoder(
                config["embedding_dim"], config["p"], hidden_sizes=decoder_hidden_sizes)

    def forward(self, input, training=True, temperature=0.666, concat=True):
        emb = self.encoder(input)
        if self.OutputInfo_list is not None:
            recon = self.decoder(emb, training=training, temperature=temperature, concat=concat)
        else:
            recon = self.decoder(emb, training=training)
        return emb, recon
#%%