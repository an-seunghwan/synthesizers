#%%
"""
Reference:
https://github.com/rcamino/multi-categorical-gans/blob/master/multi_categorical_gans/methods/medgan/trainer.py
"""
#%%
import numpy as np
import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd.variable import Variable
from module.utils import calculate_gradient_penalty
#%%
def train_medGAN(dataloader, autoencoder, discriminator, generator, config, optimizer_D, optimizer_G, device):
    criterion = nn.BCELoss()
    
    logs = {
        'disc_loss': [], 
        'gen_loss': [], 
    }
    
    for (x_batch) in tqdm.tqdm(iter(dataloader), desc="inner loop"):
        
        x_batch = x_batch.to(device)
        
        loss_ = []
        
        """1. train discriminator"""
        optimizer_D.zero_grad()
        generator.batch_norm_train(mode=False)
        
        # using "one sided smooth labels" is one trick to improve GAN training
        label_zeros = Variable(torch.zeros(len(x_batch))).to(device)
        smooth_label_ones = Variable(torch.FloatTensor(len(x_batch)).uniform_(0.9, 1)).to(device)

        # first train the discriminator only with real data
        real_pred = discriminator(x_batch)
        real_loss = criterion(real_pred, smooth_label_ones)
        real_loss.backward()

        # then train the discriminator only with fake data
        noise = Variable(torch.FloatTensor(len(x_batch), config["embedding_dim"]).normal_()).to(device)
        fake_code = generator(noise)
        if config["mc"]:
            fake_features = autoencoder.decoder(fake_code, training=True, temperature=config["tau"], concat=True).detach()
        else:
            fake_features = autoencoder.decoder(fake_code, training=True).detach()
        fake_pred = discriminator(fake_features)
        fake_loss = criterion(fake_pred, label_zeros)
        fake_loss.backward()

        optimizer_D.step()

        disc_loss = real_loss + fake_loss
        loss_.append(('disc_loss', disc_loss))

        """2. train generator"""
        optimizer_G.zero_grad()
        generator.batch_norm_train(mode=True)

        noise = Variable(torch.FloatTensor(len(x_batch), config["embedding_dim"]).normal_()).to(device)
        gen_code = generator(noise)
        if config["mc"]:
            gen_features = autoencoder.decoder(gen_code, training=True, temperature=config["tau"], concat=True)
        else:
            gen_features = autoencoder.decoder(gen_code, training=True)
        gen_pred = discriminator(gen_features)

        smooth_label_ones = Variable(torch.FloatTensor(len(x_batch)).uniform_(0.9, 1)).to(device)
        gen_loss = criterion(gen_pred, smooth_label_ones)
        gen_loss.backward()

        optimizer_G.step()
        loss_.append(('gen_loss', gen_loss))

        """accumulate losses"""
        for x, y in loss_:
            logs[x] = logs.get(x) + [y.item()]
    
    return logs
#%%
def add_noise_to_code(code, noise_radius, device):
    if noise_radius > 0:
        means = torch.zeros_like(code)
        gauss_noise = torch.normal(means, noise_radius)
        return code + Variable(gauss_noise).to(device)
    else:
        return code
#%%
def train_ARAE(dataloader, autoencoder, discriminator, generator, config, optimizer_AE, optimizer_D, optimizer_G, epoch, device):
    logs = {
        'ae_loss': [],
        'disc_loss': [], 
        'gen_loss': [], 
    }
    
    """Gumbel-Softmax temperature annealing"""
    tau = np.maximum(5 * np.exp(-0.025 * epoch), config["tau"])
    """Gaussian noise annealing"""
    ae_noise_radius = config["noise_radius"] * (config["noise_anneal"] ** epoch)
    
    for (x_batch) in tqdm.tqdm(iter(dataloader), desc="inner loop"):
        
        x_batch = x_batch.to(device)
        
        loss_ = []
        
        """1. train autoencoder (mutli-categorical)"""
        optimizer_AE.zero_grad()
        
        code = autoencoder.encoder(x_batch)
        code = add_noise_to_code(code, ae_noise_radius, device)
        xhat = autoencoder.decoder(code, training=True, temperature=tau, concat=False)
        st = 0
        ae_loss = 0
        for i, info in enumerate(autoencoder.OutputInfo_list):
            ed = st + info.dim
            batch_target = torch.argmax(x_batch[:, st : ed], dim=1)
            ae_loss += F.cross_entropy(xhat[i], batch_target)
            st = ed
        loss_.append(('ae_loss', ae_loss))
        ae_loss.backward()
        optimizer_AE.step()
        
        """2. train discriminator (critic)"""
        optimizer_D.zero_grad()
        
        # first train the discriminator only with real data
        real_code = autoencoder.encoder(x_batch)
        real_code = add_noise_to_code(real_code, ae_noise_radius, device).detach() # do not propagate to the autoencoder
        real_pred = discriminator(real_code)
        real_loss = - real_pred.mean(dim=0).view(1)
        real_loss.backward()
        
        # then train the discriminator only with fake data
        noise = Variable(torch.FloatTensor(len(x_batch), config["embedding_dim"]).normal_()).to(device)
        fake_code = generator(noise).detach() # do not propagate to the generator
        fake_pred = discriminator(fake_code)
        fake_loss = fake_pred.mean(dim=0).view(1)
        fake_loss.backward()
        
        # this is the magic from WGAN-GP
        gradient_penalty = calculate_gradient_penalty(discriminator, config["penalty"], real_code, fake_code, device)
        gradient_penalty.backward()

        disc_loss = real_loss + fake_loss + gradient_penalty
        loss_.append(('disc_loss', disc_loss))
        
        optimizer_D.step()
        
        # weight-clipping
        with torch.no_grad():
            for param in discriminator.parameters():
                param.clamp_(-config["clipping"], config["clipping"])
        
        """3. train autoencoder(encoder) & generator"""
        optimizer_AE.zero_grad()
        optimizer_G.zero_grad()

        # train autoencoder(encoder) with real data
        real_code = autoencoder.encoder(x_batch)
        real_code = add_noise_to_code(real_code, ae_noise_radius, device)
        real_pred = discriminator(real_code)
        real_loss = real_pred.mean(dim=0).view(1)
        real_loss.backward()
        
        # train generator with fake data
        noise = Variable(torch.FloatTensor(len(x_batch), config["embedding_dim"]).normal_()).to(device)
        gen_code = generator(noise)
        fake_pred = discriminator(gen_code)
        fake_loss = - fake_pred.mean(dim=0).view(1)
        fake_loss.backward()
        
        gen_loss = real_loss + fake_loss
        loss_.append(('gen_loss', gen_loss))

        optimizer_AE.step()
        optimizer_G.step()

        """accumulate losses"""
        for x, y in loss_:
            logs[x] = logs.get(x) + [y.item()]
    
    return logs
#%%
def train_Gumbel(dataloader, discriminator, generator, config, optimizer_D, optimizer_G, epoch, device):
    criterion = nn.BCELoss()
    logs = {
        'disc_loss': [], 
        'gen_loss': [], 
    }
    
    """Gumbel-Softmax temperature annealing"""
    tau = np.maximum(5 * np.exp(-0.025 * epoch), config["tau"])
    
    for (x_batch) in tqdm.tqdm(iter(dataloader), desc="inner loop"):
        
        x_batch = x_batch.to(device)
        
        loss_ = []
        
        """1. train discriminator"""
        optimizer_D.zero_grad()
        
        # using "one sided smooth labels" is one trick to improve GAN training
        label_zeros = Variable(torch.zeros(len(x_batch))).to(device)
        smooth_label_ones = Variable(torch.FloatTensor(len(x_batch)).uniform_(0.9, 1)).to(device)

        # first train the discriminator only with real data
        real_pred = discriminator(x_batch)
        real_loss = criterion(real_pred, smooth_label_ones)
        real_loss.backward()

        # then train the discriminator only with fake data
        noise = Variable(torch.FloatTensor(len(x_batch), config["embedding_dim"]).normal_()).to(device)
        fake_features = generator(noise, temperature=tau)
        fake_pred = discriminator(fake_features)
        fake_loss = criterion(fake_pred, label_zeros)
        fake_loss.backward()

        optimizer_D.step()

        disc_loss = real_loss + fake_loss
        loss_.append(('disc_loss', disc_loss))

        """2. train generator"""
        optimizer_G.zero_grad()

        noise = Variable(torch.FloatTensor(len(x_batch), config["embedding_dim"]).normal_()).to(device)
        gen_features = generator(noise, temperature=tau)
        gen_pred = discriminator(gen_features)

        smooth_label_ones = Variable(torch.FloatTensor(len(x_batch)).uniform_(0.9, 1)).to(device)
        gen_loss = criterion(gen_pred, smooth_label_ones)
        gen_loss.backward()

        optimizer_G.step()
        loss_.append(('gen_loss', gen_loss))

        """accumulate losses"""
        for x, y in loss_:
            logs[x] = logs.get(x) + [y.item()]
    
    return logs
#%%
def train_WGAN_GP(dataloader, discriminator, generator, config, optimizer_D, optimizer_G, epoch, device):
    logs = {
        'disc_loss': [], 
        'gen_loss': [], 
    }
    
    """Gumbel-Softmax temperature annealing"""
    tau = np.maximum(5 * np.exp(-0.025 * epoch), config["tau"])
    
    for (x_batch) in tqdm.tqdm(iter(dataloader), desc="inner loop"):
        
        x_batch = x_batch.to(device)
        
        loss_ = []
        
        """1. train discriminator"""
        optimizer_D.zero_grad()
        
        # first train the discriminator only with real data
        real_pred = discriminator(x_batch)
        real_loss = - real_pred.mean(dim=0).view(1)
        real_loss.backward()

        # then train the discriminator only with fake data
        noise = Variable(torch.FloatTensor(len(x_batch), config["embedding_dim"]).normal_()).to(device)
        fake_features = generator(noise, temperature=tau).detach() # do not propagate to the generator
        fake_pred = discriminator(fake_features)
        fake_loss = fake_pred.mean(dim=0).view(1)
        fake_loss.backward()
        
        # this is the magic from WGAN-GP
        gradient_penalty = calculate_gradient_penalty(discriminator, config["penalty"], x_batch, fake_features, device)
        gradient_penalty.backward()

        disc_loss = real_loss + fake_loss + gradient_penalty
        loss_.append(('disc_loss', disc_loss))
        
        optimizer_D.step()

        disc_loss = real_loss + fake_loss
        loss_.append(('disc_loss', disc_loss))

        """2. train generator"""
        optimizer_G.zero_grad()

        noise = Variable(torch.FloatTensor(len(x_batch), config["embedding_dim"]).normal_()).to(device)
        gen_features = generator(noise, temperature=tau)
        gen_pred = discriminator(gen_features)
        gen_loss = - gen_pred.mean(dim=0).view(1)
        gen_loss.backward()
        
        optimizer_G.step()
        loss_.append(('gen_loss', gen_loss))

        """accumulate losses"""
        for x, y in loss_:
            logs[x] = logs.get(x) + [y.item()]
    
    return logs
#%%