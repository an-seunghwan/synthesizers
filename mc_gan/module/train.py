#%%
"""
Reference:
https://github.com/rcamino/multi-categorical-gans/blob/master/multi_categorical_gans/methods/medgan/trainer.py
"""
#%%
import tqdm

import torch
from torch import nn
from torch.autograd.variable import Variable
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
        
        # train discriminator
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
        fake_features = autoencoder.decoder(fake_code).detach()
        fake_pred = discriminator(fake_features)
        fake_loss = criterion(fake_pred, label_zeros)
        fake_loss.backward()

        # finally update the discriminator weights
        # using two separated batches is another trick to improve GAN training
        optimizer_D.step()

        disc_loss = real_loss + fake_loss
        loss_.append(('disc_loss', disc_loss))

        del disc_loss
        del fake_loss
        del real_loss
        
        # train generator
        optimizer_G.zero_grad()
        generator.batch_norm_train(mode=True)

        noise = Variable(torch.FloatTensor(len(x_batch), config["embedding_dim"]).normal_()).to(device)
        gen_code = generator(noise)
        gen_features = autoencoder.decoder(gen_code)
        gen_pred = discriminator(gen_features)

        smooth_label_ones = Variable(torch.FloatTensor(len(x_batch)).uniform_(0.9, 1)).to(device)

        gen_loss = criterion(gen_pred, smooth_label_ones)
        gen_loss.backward()

        optimizer_G.step()
        loss_.append(('gen_loss', gen_loss))

        del gen_loss
        
        """accumulate losses"""
        for x, y in loss_:
            logs[x] = logs.get(x) + [y.item()]
    
    return logs
#%%