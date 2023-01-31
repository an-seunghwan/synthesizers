#%%
"""
Reference:
[1] https://github.com/sdv-dev/CTGAN/blob/master/ctgan/synthesizers/ctgan.py
"""
#%%
import tqdm

import numpy as np
import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset
from torch.nn.functional import cross_entropy

from .model import apply_activate
#%%
def cond_loss(output_info_list, data, c, m):
    """Compute the cross entropy loss on the fixed discrete column."""
    loss = []
    st = 0
    st_c = 0
    for column_info in output_info_list:
        for span_info in column_info:
            if len(column_info) != 1 or span_info.activation_fn != 'softmax':
                # not discrete column
                st += span_info.dim
            else:
                ed = st + span_info.dim
                ed_c = st_c + span_info.dim
                tmp = functional.cross_entropy(
                    data[:, st:ed],
                    torch.argmax(c[:, st_c:ed_c], dim=1),
                    reduction='none'
                )
                loss.append(tmp)
                st = ed
                st_c = ed_c

    loss = torch.stack(loss, dim=1)  # noqa: PD013

    return (loss * m).sum() / data.size()[0]
#%%
def train(generator, discriminator, 
          optimizerG, optimizerD,
          train_data, data_sampler, transformer,
          config, mean, std, 
          device):

    logs = {
        'loss(G)': [], 
        'loss(D)': [], 
        'CE': [], 
    }

    steps_per_epoch = max(len(train_data) // config["batch_size"], 1)
    
    for id_ in tqdm.tqdm(range(steps_per_epoch), desc="inner loop"):
        
        loss_ = []
        
        for n in range(config["discriminator_steps"]):
            """1. Discriminator learning"""
            fakez = torch.normal(mean=mean, std=std)

            condvec = data_sampler.sample_condvec(config["batch_size"])
            if condvec is None:
                c1, m1, col, opt = None, None, None, None
                real = data_sampler.sample_data(config["batch_size"], col, opt)
            else:
                c1, m1, col, opt = condvec
                c1 = torch.from_numpy(c1).to(device)
                m1 = torch.from_numpy(m1).to(device)
                fakez = torch.cat([fakez, c1], dim=1)

                perm = np.arange(config["batch_size"])
                np.random.shuffle(perm)
                real = data_sampler.sample_data(config["batch_size"], col[perm], opt[perm])
                c2 = c1[perm]

            fake = generator(fakez)
            fakeact = apply_activate(fake, transformer, generator._gumbel_softmax)

            real = torch.from_numpy(real.astype('float32')).to(device)

            if c1 is not None:
                fake_cat = torch.cat([fakeact, c1], dim=1)
                real_cat = torch.cat([real, c2], dim=1)
            else:
                real_cat = real
                fake_cat = fakeact

            y_fake = discriminator(fake_cat)
            y_real = discriminator(real_cat)

            pen = discriminator.calc_gradient_penalty(
                real_cat, fake_cat, device, config["pac"])
            loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
            loss_.append(('loss(D)', loss_d))

            optimizerD.zero_grad()
            pen.backward(retain_graph=True)
            loss_d.backward()
            optimizerD.step()

        """2. Generator learning"""
        fakez = torch.normal(mean=mean, std=std)
        condvec = data_sampler.sample_condvec(config["batch_size"])

        if condvec is None:
            c1, m1, col, opt = None, None, None, None
        else:
            c1, m1, col, opt = condvec
            c1 = torch.from_numpy(c1).to(device)
            m1 = torch.from_numpy(m1).to(device)
            fakez = torch.cat([fakez, c1], dim=1)

        fake = generator(fakez)
        fakeact = apply_activate(fake, transformer, generator._gumbel_softmax)

        if c1 is not None:
            y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
        else:
            y_fake = discriminator(fakeact)

        if condvec is None:
            cross_entropy = 0
        else:
            cross_entropy = cond_loss(transformer.output_info_list, fake, c1, m1)

        loss_g = -torch.mean(y_fake) + cross_entropy
        loss_.append(('loss(G)', loss_g))
        loss_.append(('CE', cross_entropy))

        optimizerG.zero_grad()
        loss_g.backward()
        optimizerG.step()

        """accumulate losses"""
        for x, y in loss_:
            logs[x] = logs.get(x) + [y.item()]
    
    return logs
#%%