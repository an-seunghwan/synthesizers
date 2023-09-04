# %%
"""
Reference:
[1] https://github.com/Team-TUD/CTAB-GAN/blob/main/model/synthesizer/ctabgan_synthesizer.py
"""
# %%
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torch.optim as optim
from torch.optim import Adam
from torch.nn import functional as F
from torch.nn import (
    Dropout,
    LeakyReLU,
    Linear,
    Module,
    ReLU,
    Sequential,
    Conv2d,
    ConvTranspose2d,
    BatchNorm2d,
    Sigmoid,
    init,
    BCELoss,
    CrossEntropyLoss,
    SmoothL1Loss,
)
from tqdm import tqdm
import random


# %%
def random_choice_prob_index_sampling(probs, col_idx):
    """
    Used to sample a specific category within a chosen one-hot-encoding representation

    Inputs:
    1) probs -> probability mass distribution of categories
    2) col_idx -> index used to identify any given one-hot-encoding

    Outputs:
    1) option_list -> list of chosen categories

    """

    option_list = []
    for i in col_idx:
        # for improved stability
        pp = probs[i] + 1e-6
        pp = pp / sum(pp)
        # sampled based on given probability mass distribution of categories within the given one-hot-encoding
        option_list.append(np.random.choice(np.arange(len(probs[i])), p=pp))

    return np.array(option_list).reshape(col_idx.shape)


# %%
class Condvec(object):

    """
    This class is responsible for sampling conditional vectors to be supplied to the generator

    Variables:
    1) model -> list containing an index of highlighted categories in their corresponding one-hot-encoded represenations
    2) interval -> an array holding the respective one-hot-encoding starting positions and sizes
    3) n_col -> total no. of one-hot-encoding representations
    4) n_opt -> total no. of distinct categories across all one-hot-encoding representations
    5) p_log_sampling -> list containing log of probability mass distribution of categories within their respective one-hot-encoding representations
    6) p_sampling -> list containing probability mass distribution of categories within their respective one-hot-encoding representations

    Methods:
    1) __init__() -> takes transformed input data with respective column information to compute class variables
    2) sample_train() -> used to sample the conditional vector during training of the model
    3) sample() -> used to sample the conditional vector for generating data after training is finished
    """

    def __init__(self, data, output_info):
        self.model = []
        self.interval = []
        self.n_col = 0
        self.n_opt = 0
        self.p_log_sampling = []
        self.p_sampling = []

        # iterating through the transformed input data columns
        st = 0
        for item in output_info:
            # ignoring columns that do not represent one-hot-encodings
            if item[1] == "tanh":
                st += item[0]
                continue
            elif item[1] == "softmax":
                # using starting (st) and ending (ed) position of any given one-hot-encoded representation to obtain relevant information
                ed = st + item[0]
                self.model.append(np.argmax(data[:, st:ed], axis=-1))
                self.interval.append((self.n_opt, item[0]))
                self.n_col += 1
                self.n_opt += item[0]
                freq = np.sum(data[:, st:ed], axis=0)
                log_freq = np.log(freq + 1)
                log_pmf = log_freq / np.sum(log_freq)
                self.p_log_sampling.append(log_pmf)
                pmf = freq / np.sum(freq)
                self.p_sampling.append(pmf)
                st = ed

        self.interval = np.asarray(self.interval)

    def sample_train(self, batch):
        """
        Used to create the conditional vectors for feeding it to the generator during training

        Inputs:
        1) batch -> no. of data records to be generated in a batch

        Outputs:
        1) vec -> a matrix containing a conditional vector for each data point to be generated
        2) mask -> a matrix to identify chosen one-hot-encodings across the batch
        3) idx -> list of chosen one-hot encoding across the batch
        4) opt1prime -> selected categories within chosen one-hot-encodings

        """

        if self.n_col == 0:
            return None
        batch = batch

        # each conditional vector in vec is a one-hot vector used to highlight a specific category across all possible one-hot-encoded representations
        # (i.e., including modes of continuous and mixed columns)
        vec = np.zeros((batch, self.n_opt), dtype="float32")

        # choosing one specific one-hot-encoding from all possible one-hot-encoded representations
        idx = np.random.choice(np.arange(self.n_col), batch)

        # matrix of shape (batch x total no. of one-hot-encoded representations) with 1 in indexes of chosen representations and 0 elsewhere
        mask = np.zeros((batch, self.n_col), dtype="float32")
        mask[np.arange(batch), idx] = 1

        # producing a list of selected categories within each of selected one-hot-encoding representation
        opt1prime = random_choice_prob_index_sampling(self.p_log_sampling, idx)

        # assigning the appropriately chosen category for each corresponding conditional vector
        for i in np.arange(batch):
            vec[i, self.interval[idx[i], 0] + opt1prime[i]] = 1

        return vec, mask, idx, opt1prime

    def sample(self, batch):
        """
        Used to create the conditional vectors for feeding it to the generator after training is finished

        Inputs:
        1) batch -> no. of data records to be generated in a batch

        Outputs:
        1) vec -> an array containing a conditional vector for each data point to be generated
        """

        if self.n_col == 0:
            return None

        batch = batch

        # each conditional vector in vec is a one-hot vector used to highlight a specific category across all possible one-hot-encoded representations
        # (i.e., including modes of continuous and mixed columns)
        vec = np.zeros((batch, self.n_opt), dtype="float32")

        # choosing one specific one-hot-encoding from all possible one-hot-encoded representations
        idx = np.random.choice(np.arange(self.n_col), batch)

        # producing a list of selected categories within each of selected one-hot-encoding representation
        opt1prime = random_choice_prob_index_sampling(self.p_sampling, idx)

        # assigning the appropriately chosen category for each corresponding conditional vector
        for i in np.arange(batch):
            vec[i, self.interval[idx[i], 0] + opt1prime[i]] = 1

        return vec


# %%
def cond_loss(data, output_info, c, m):
    """
    Used to compute the conditional loss for ensuring the generator produces the desired category as specified by the conditional vector

    Inputs:
    1) data -> raw data synthesized by the generator
    2) output_info -> column informtion corresponding to the data transformer
    3) c -> conditional vectors used to synthesize a batch of data
    4) m -> a matrix to identify chosen one-hot-encodings across the batch

    Outputs:
    1) loss -> conditional loss corresponding to the generated batch

    """

    # used to store cross entropy loss between conditional vector and all generated one-hot-encodings
    tmp_loss = []
    # counter to iterate generated data columns
    st = 0
    # counter to iterate conditional vector
    st_c = 0
    # iterating through column information
    for item in output_info:
        # ignoring numeric columns
        if item[1] == "tanh":
            st += item[0]
            continue
        # computing cross entropy loss between generated one-hot-encoding and corresponding encoding of conditional vector
        elif item[1] == "softmax":
            ed = st + item[0]
            ed_c = st_c + item[0]
            tmp = F.cross_entropy(
                data[:, st:ed], torch.argmax(c[:, st_c:ed_c], dim=1), reduction="none"
            )
            tmp_loss.append(tmp)
            st = ed
            st_c = ed_c

    # computing the loss across the batch only and only for the relevant one-hot-encodings by applying the mask
    tmp_loss = torch.stack(tmp_loss, dim=1)
    loss = (tmp_loss * m).sum() / data.size()[0]

    return loss


# %%
class Sampler(object):

    """
    This class is used to sample the transformed real data according to the conditional vector

    Variables:
    1) data -> real transformed input data
    2) model -> stores the index values of data records corresponding to any given selected categories for all columns
    3) n -> size of the input data

    Methods:
    1) __init__() -> initiates the sampler object and stores class variables
    2) sample() -> takes as input the number of rows to be sampled (n), chosen column (col)
                   and category within the column (opt) to sample real records accordingly
    """

    def __init__(self, data, output_info):
        super(Sampler, self).__init__()

        self.data = data
        self.model = []
        self.n = len(data)

        # counter to iterate through columns
        st = 0
        # iterating through column information
        for item in output_info:
            # ignoring numeric columns
            if item[1] == "tanh":
                st += item[0]
                continue
            # storing indices of data records for all categories within one-hot-encoded representations
            elif item[1] == "softmax":
                ed = st + item[0]
                tmp = []
                # iterating through each category within a one-hot-encoding
                for j in range(item[0]):
                    # storing the relevant indices of data records for the given categories
                    tmp.append(np.nonzero(data[:, st + j])[0])
                self.model.append(tmp)
                st = ed

    def sample(self, n, col, opt):
        # if there are no one-hot-encoded representations, we may ignore sampling using a conditional vector
        if col is None:
            idx = np.random.choice(np.arange(self.n), n)
            return self.data[idx]

        # used to store relevant indices of data records based on selected category within a chosen one-hot-encoding
        idx = []

        # sampling a data record index randomly from all possible indices that meet the given criteria of the chosen category and one-hot-encoding
        for c, o in zip(col, opt):
            idx.append(np.random.choice(self.model[c][o]))

        return self.data[idx]


# %%
def get_st_ed(target_col_index, output_info):
    """
    Used to obtain the start and ending positions of the target column as per the transformed data to be used by the classifier

    Inputs:
    1) target_col_index -> column index of the target column used for machine learning tasks (binary/multi-classification) in the raw data
    2) output_info -> column information corresponding to the data after applying the data transformer

    Outputs:
    1) starting (st) and ending (ed) positions of the target column as per the transformed data

    """
    # counter to iterate through columns
    st = 0
    # counter to check if the target column index has been reached
    c = 0
    # counter to iterate through column information
    tc = 0
    # iterating until target index has reached to obtain starting position of the one-hot-encoding used to represent target column in transformed data
    for item in output_info:
        # exiting loop if target index has reached
        if c == target_col_index:
            break
        if item[1] == "tanh":
            st += item[0]
        elif item[1] == "softmax":
            st += item[0]
            c += 1
        tc += 1

    # obtaining the ending position by using the dimension size of the one-hot-encoding used to represent the target column
    ed = st + output_info[tc][0]

    return (st, ed)


# %%
class Classifier(Module):

    """
    This class represents the classifier module used along side the discriminator to train the generator network

    Variables:
    1) dim -> column dimensionality of the transformed input data after removing target column
    2) class_dims -> list of dimensions used for the hidden layers of the classifier network
    3) str_end -> tuple containing the starting and ending positions of the target column in the transformed input data

    Methods:
    1) __init__() -> initializes and builds the layers of the classifier module
    2) forward() -> executes the forward pass of the classifier module on the corresponding input data and
                    outputs the predictions and corresponding true labels for the target column

    """

    def __init__(self, input_dim, class_dims, st_ed):
        super(Classifier, self).__init__()
        # subtracting the target column size from the input dimensionality
        self.dim = input_dim - (st_ed[1] - st_ed[0])
        # storing the starting and ending positons of the target column in the input data
        self.str_end = st_ed

        # building the layers of the network with same hidden layers as discriminator
        seq = []
        tmp_dim = self.dim
        for item in list(class_dims):
            seq += [Linear(tmp_dim, item), LeakyReLU(0.2), Dropout(0.5)]
            tmp_dim = item

        # in case of binary classification the last layer outputs a single numeric value which is squashed to a probability with sigmoid
        if (st_ed[1] - st_ed[0]) == 2:
            seq += [Linear(tmp_dim, 1), Sigmoid()]
        # in case of multi-classs classification, the last layer outputs an array of numeric values associated to each class
        else:
            seq += [Linear(tmp_dim, (st_ed[1] - st_ed[0]))]

        self.seq = Sequential(*seq)

    def forward(self, input):
        # true labels obtained from the input data
        label = torch.argmax(input[:, self.str_end[0] : self.str_end[1]], axis=-1)

        # input to be fed to the classifier module
        new_imp = torch.cat(
            (input[:, : self.str_end[0]], input[:, self.str_end[1] :]), 1
        )

        # returning predictions and true labels for binary/multi-class classification
        if (self.str_end[1] - self.str_end[0]) == 2:
            return self.seq(new_imp).view(-1), label
        else:
            return self.seq(new_imp), label


# %%
class Discriminator(Module):

    """
    This class represents the discriminator network of the model

    Variables:
    1) seq -> layers of the network used for making the final prediction of the discriminator model
    2) seq_info -> layers of the discriminator network used for computing the information loss

    Methods:
    1) __init__() -> initializes and builds the layers of the discriminator model
    2) forward() -> executes a forward pass on the input data to output the final predictions and corresponding
                    feature information associated with the penultimate layer used to compute the information loss

    """

    def __init__(self, layers):
        super(Discriminator, self).__init__()
        self.seq = Sequential(*layers)
        self.seq_info = Sequential(*layers[: len(layers) - 2])

    def forward(self, input):
        return (self.seq(input)), self.seq_info(input)


# %%
class Generator(Module):

    """
    This class represents the discriminator network of the model

    Variables:
    1) seq -> layers of the network used by the generator

    Methods:
    1) __init__() -> initializes and builds the layers of the generator model
    2) forward() -> executes a forward pass using noise as input to generate data

    """

    def __init__(self, layers):
        super(Generator, self).__init__()
        self.seq = Sequential(*layers)

    def forward(self, input):
        return self.seq(input)


# %%
def determine_layers_disc(side, num_channels):
    """
    This function describes the layers of the discriminator network as per DCGAN (https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)

    Inputs:
    1) side -> height/width of the input fed to the discriminator
    2) num_channels -> no. of channels used to decide the size of respective hidden layers

    Outputs:
    1) layers_D -> layers of the discriminator network

    """

    # computing the dimensionality of hidden layers
    layer_dims = [(1, side), (num_channels, side // 2)]

    while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
        # the number of channels increases by a factor of 2 whereas the height/width decreases by the same factor with each layer
        layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))

    # constructing the layers of the discriminator network based on the recommendations mentioned in https://arxiv.org/abs/1511.06434
    layers_D = []
    for prev, curr in zip(layer_dims, layer_dims[1:]):
        layers_D += [
            Conv2d(prev[0], curr[0], 4, 2, 1, bias=False),
            BatchNorm2d(curr[0]),
            LeakyReLU(0.2, inplace=True),
        ]
    # last layer reduces the output to a single numeric value which is squashed to a probabability using sigmoid function
    layers_D += [Conv2d(layer_dims[-1][0], 1, layer_dims[-1][1], 1, 0), Sigmoid()]

    return layers_D


# %%
def determine_layers_gen(side, random_dim, num_channels):
    """
    This function describes the layers of the generator network

    Inputs:
    1) random_dim -> height/width of the noise matrix to be fed for generation
    2) num_channels -> no. of channels used to decide the size of respective hidden layers

    Outputs:
    1) layers_G -> layers of the generator network

    """

    # computing the dimensionality of hidden layers
    layer_dims = [(1, side), (num_channels, side // 2)]

    while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
        layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))

    # similarly constructing the layers of the generator network based on the recommendations mentioned in https://arxiv.org/abs/1511.06434
    # first layer of the generator takes the channel dimension of the noise matrix to the desired maximum channel size of the generator's layers
    layers_G = [
        ConvTranspose2d(
            random_dim,
            layer_dims[-1][0],
            layer_dims[-1][1],
            1,
            0,
            output_padding=0,
            bias=False,
        )
    ]

    # the following layers are then reversed with respect to the discriminator
    # such as the no. of channels reduce by a factor of 2 and height/width of generated image increases by the same factor with each layer
    for prev, curr in zip(reversed(layer_dims), reversed(layer_dims[:-1])):
        layers_G += [
            BatchNorm2d(prev[0]),
            ReLU(True),
            ConvTranspose2d(prev[0], curr[0], 4, 2, 1, output_padding=0, bias=True),
        ]

    return layers_G


# %%
def apply_activate(data, output_info):
    """
    This function applies the final activation corresponding to the column information associated with transformer

    Inputs:
    1) data -> input data generated by the model in the same format as the transformed input data
    2) output_info -> column information associated with the transformed input data

    Outputs:
    1) act_data -> resulting data after applying the respective activations

    """

    data_t = []
    # used to iterate through columns
    st = 0
    # used to iterate through column information
    for item in output_info:
        # for numeric columns a final tanh activation is applied
        if item[1] == "tanh":
            ed = st + item[0]
            data_t.append(torch.tanh(data[:, st:ed]))
            st = ed
        # for one-hot-encoded columns, a final gumbel softmax (https://arxiv.org/pdf/1611.01144.pdf) is used
        # to sample discrete categories while still allowing for back propagation
        elif item[1] == "softmax":
            ed = st + item[0]
            # note that as tau approaches 0, a completely discrete one-hot-vector is obtained
            data_t.append(F.gumbel_softmax(data[:, st:ed], tau=0.2))
            st = ed

    act_data = torch.cat(data_t, dim=1)

    return act_data


# %%
def weights_init(model):
    """
    This function initializes the learnable parameters of the convolutional and batch norm layers

    Inputs:
    1) model->  network for which the parameters need to be initialized

    Outputs:
    1) network with corresponding weights initialized using the normal distribution

    """

    classname = model.__class__.__name__

    if classname.find("Conv") != -1:
        init.normal_(model.weight.data, 0.0, 0.02)

    elif classname.find("BatchNorm") != -1:
        init.normal_(model.weight.data, 1.0, 0.02)
        init.constant_(model.bias.data, 0)


# %%
