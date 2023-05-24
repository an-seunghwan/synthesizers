#%%
"""
Reference:
https://github.com/rcamino/multi-categorical-gans/tree/master
"""
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.one_hot_categorical import OneHotCategorical
#%%
class medGANDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_sizes):
        super(medGANDiscriminator, self).__init__()

        previous_layer_size = input_dim * 2 # mini-batch averaging
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
class SingleOutput(nn.Module):
    def __init__(self, previous_layer_size, output_size, activation=None):
        super(SingleOutput, self).__init__()
        if activation is None:
            self.model = nn.Linear(previous_layer_size, output_size)
        else:
            self.model = nn.Sequential(nn.Linear(previous_layer_size, output_size), activation)

    def forward(self, hidden, training=False, temperature=None):
        return self.model(hidden)
#%%
class MultiCategorical(nn.Module):
    def __init__(self, input_size, variable_sizes):
        super(MultiCategorical, self).__init__()

        self.output_layers = nn.ModuleList()
        self.output_activations = nn.ModuleList()
        for i, variable_size in enumerate(variable_sizes):
            self.output_layers.append(nn.Linear(input_size, variable_size))
            self.output_activations.append(CategoricalActivation())

    def forward(self, inputs, training=True, temperature=None, concat=True):
        outputs = []
        for output_layer, output_activation in zip(self.output_layers, self.output_activations):
            logits = output_layer(inputs)
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
class Generator(nn.Module):
    def __init__(self, noise_dim, output_dim, hidden_sizes=[], bn_decay=0.01):
        super(Generator, self).__init__()

        previous_layer_size = noise_dim
        hidden_layers = []
        for layer_number, layer_size in enumerate(hidden_sizes):
            hidden_layers.append(nn.Linear(previous_layer_size, layer_size))
            if layer_number > 0 and bn_decay > 0:
                hidden_layers.append(nn.BatchNorm1d(layer_size, momentum=(1 - bn_decay)))
            hidden_layers.append(nn.ReLU())
            previous_layer_size = layer_size

        if len(hidden_layers) > 0:
            self.hidden_layers = nn.Sequential(*hidden_layers)
        else:
            self.hidden_layers = None

        if type(output_dim) is int:
            self.output = SingleOutput(previous_layer_size, output_dim)
        elif type(output_dim) is list:
            self.output = MultiCategorical(previous_layer_size, output_dim)
        else:
            raise Exception("Invalid output size.")

    def forward(self, noise, training=False, temperature=None):
        if self.hidden_layers is None:
            hidden = noise
        else:
            hidden = self.hidden_layers(noise)
        return self.output(hidden, training=training, temperature=temperature) 
#%%
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_sizes=(256, 128), bn_decay=0.01, critic=False):
        super(Discriminator, self).__init__()

        previous_layer_size = input_size
        layers = []

        for layer_number, layer_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(previous_layer_size, layer_size))
            if layer_number > 0 and bn_decay > 0:
                layers.append(nn.BatchNorm1d(layer_size, momentum=(1 - bn_decay)))
            layers.append(nn.LeakyReLU(0.2))
            previous_layer_size = layer_size
        layers.append(nn.Linear(previous_layer_size, 1))

        # the critic has a linear output
        if not critic:
            layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.model(inputs).view(-1)
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