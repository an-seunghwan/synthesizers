# %%
"""
Reference
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
from .synthesizer import *
from tqdm import tqdm


# %%
def train(
    config,
    train_data,
    device,
    generator,
    discriminator,
    cond_generator,
    data_sampler,
    transformer,
    optimizerG,
    optimizerD,
    Gtransformer,
    Dtransformer,
    target_index,
    st_ed,
    optimizerC,
    classifier,
):
    logs = {
        "loss(D)": [],
        "loss(G)": [],
        "info": [],
        "CLF": [],
    }

    steps_per_epoch = max(1, len(train_data) // config["batch_size"])
    for _ in tqdm(range(steps_per_epoch), desc="inner loop"):
        loss_ = []

        # sampling noise vectors using a standard normal distribution
        noisez = torch.randn(config["batch_size"], config["latent_dim"], device=device)
        # sampling conditional vectors
        condvec = cond_generator.sample_train(config["batch_size"])
        c, m, col, opt = condvec
        c = torch.from_numpy(c).to(device)
        m = torch.from_numpy(m).to(device)
        # concatenating conditional vectors and converting resulting noise vectors into the image domain to be fed to the generator as input
        noisez = torch.cat([noisez, c], dim=1)
        noisez = noisez.view(
            config["batch_size"], config["latent_dim"] + cond_generator.n_opt, 1, 1
        )

        # sampling real data according to the conditional vectors and shuffling it before feeding to discriminator to isolate conditional loss on generator
        perm = np.arange(config["batch_size"])
        np.random.shuffle(perm)
        real = data_sampler.sample(config["batch_size"], col[perm], opt[perm])
        real = torch.from_numpy(real.astype("float32")).to(device)

        # storing shuffled ordering of the conditional vectors
        c_perm = c[perm]
        # generating synthetic data as an image
        fake = generator(noisez)
        # converting it into the tabular domain as per format of the trasformed training data
        faket = Gtransformer.inverse_transform(fake)
        # applying final activation on the generated data (i.e., tanh for numeric and gumbel-softmax for categorical)
        fakeact = apply_activate(faket, transformer.output_info)

        # the generated data is then concatenated with the corresponding condition vectors
        fake_cat = torch.cat([fakeact, c], dim=1)
        # the real data is also similarly concatenated with corresponding conditional vectors
        real_cat = torch.cat([real, c_perm], dim=1)

        # transforming the real and synthetic data into the image domain for feeding it to the discriminator
        real_cat_d = Dtransformer.transform(real_cat)
        fake_cat_d = Dtransformer.transform(fake_cat)

        # executing the gradient update step for the discriminator
        optimizerD.zero_grad()
        # computing the probability of the discriminator to correctly classify real samples hence y_real should ideally be close to 1
        y_real, _ = discriminator(real_cat_d)
        # computing the probability of the discriminator to correctly classify fake samples hence y_fake should ideally be close to 0
        y_fake, _ = discriminator(fake_cat_d)
        # computing the loss to essentially maximize the log likelihood of correctly classifiying real and fake samples as log(D(x))+log(1−D(G(z)))
        # or equivalently minimizing the negative of log(D(x))+log(1−D(G(z))) as done below
        loss_d = -(torch.log(y_real + 1e-4).mean()) - (
            torch.log(1.0 - y_fake + 1e-4).mean()
        )
        # accumulating gradients based on the loss
        loss_d.backward()
        # computing the backward step to update weights of the discriminator
        optimizerD.step()

        loss_.append(("loss(D)", loss_d))

        # similarly sample noise vectors and conditional vectors
        noisez = torch.randn(config["batch_size"], config["latent_dim"], device=device)
        condvec = cond_generator.sample_train(config["batch_size"])
        c, m, col, opt = condvec
        c = torch.from_numpy(c).to(device)
        m = torch.from_numpy(m).to(device)
        noisez = torch.cat([noisez, c], dim=1)
        noisez = noisez.view(
            config["batch_size"], config["latent_dim"] + cond_generator.n_opt, 1, 1
        )

        # executing the gradient update step for the generator
        optimizerG.zero_grad()

        # similarly generating synthetic data and applying final activation
        fake = generator(noisez)
        faket = Gtransformer.inverse_transform(fake)
        fakeact = apply_activate(faket, transformer.output_info)
        # concatenating conditional vectors and converting it to the image domain to be fed to the discriminator
        fake_cat = torch.cat([fakeact, c], dim=1)
        fake_cat = Dtransformer.transform(fake_cat)

        # computing the probability of the discriminator classifiying fake samples as real
        # along with feature representaions of fake data resulting from the penultimate layer
        y_fake, info_fake = discriminator(fake_cat)
        # extracting feature representation of real data from the penultimate layer of the discriminator
        _, info_real = discriminator(real_cat_d)
        # computing the conditional loss to ensure the generator generates data records with the chosen category as per the conditional vector
        cross_entropy = cond_loss(faket, transformer.output_info, c, m)

        # computing the loss to train the generator where we want y_fake to be close to 1 to fool the discriminator
        # and cross_entropy to be close to 0 to ensure generator's output matches the conditional vector
        g = -(torch.log(y_fake + 1e-4).mean()) + cross_entropy
        # in order to backprop the gradient of separate losses w.r.t to the learnable weight of the network independently
        # we may use retain_graph=True in backward() method in the first back-propagated loss
        # to maintain the computation graph to execute the second backward pass efficiently
        g.backward(retain_graph=True)

        loss_.append(("loss(G)", g))

        # computing the information loss by comparing means and stds of real/fake feature representations extracted from discriminator's penultimate layer
        loss_mean = torch.norm(
            torch.mean(info_fake.view(config["batch_size"], -1), dim=0)
            - torch.mean(info_real.view(config["batch_size"], -1), dim=0),
            1,
        )
        loss_std = torch.norm(
            torch.std(info_fake.view(config["batch_size"], -1), dim=0)
            - torch.std(info_real.view(config["batch_size"], -1), dim=0),
            1,
        )
        loss_info = loss_mean + loss_std
        # computing the finally accumulated gradients
        loss_info.backward()
        # executing the backward step to update the weights
        optimizerG.step()

        loss_.append(("info", loss_info))

        # the classifier module is used in case there is a target column associated with ML tasks
        if target_index is not None:
            c_loss = None
            # in case of binary classification, the binary cross entropy loss is used
            if (st_ed[1] - st_ed[0]) == 2:
                c_loss = BCELoss()
            # in case of multi-class classification, the standard cross entropy loss is used
            else:
                c_loss = CrossEntropyLoss()

            # updating the weights of the classifier
            optimizerC.zero_grad()
            # computing classifier's target column predictions on the real data along with returning corresponding true labels
            real_pre, real_label = classifier(real)
            if (st_ed[1] - st_ed[0]) == 2:
                real_label = real_label.type_as(real_pre)
            # computing the loss to train the classifier so that it can perform well on the real data
            loss_cc = c_loss(real_pre, real_label)
            loss_cc.backward()
            optimizerC.step()

            # updating the weights of the generator
            optimizerG.zero_grad()
            # generate synthetic data and apply the final activation
            fake = generator(noisez)
            faket = Gtransformer.inverse_transform(fake)
            fakeact = apply_activate(faket, transformer.output_info)
            # computing classifier's target column predictions on the fake data along with returning corresponding true labels
            fake_pre, fake_label = classifier(fakeact)
            if (st_ed[1] - st_ed[0]) == 2:
                fake_label = fake_label.type_as(fake_pre)
            # computing the loss to train the generator to improve semantic integrity between target column and rest of the data
            loss_cg = c_loss(fake_pre, fake_label)
            loss_cg.backward()
            optimizerG.step()

            loss_.append(("CLF", loss_cg))

        """accumulate losses"""
        for x, y in loss_:
            logs[x] = logs.get(x) + [y.item()]

    return logs


# %%
