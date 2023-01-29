#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#%%
import numpy as np
import pandas as pd
import tqdm
from PIL import Image
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset

from evaluation.simulation import set_random_seed
from modules.datasets import generate_dataset
from modules.transformer import ImageTransformer
from modules.synthesizer import *
from modules.train import train
#%%
import sys
import subprocess
try:
    import wandb
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    with open("./wandb_api.txt", "r") as f:
        key = f.readlines()
    subprocess.run(["wandb", "login"], input=key[0], encoding='utf-8')
    import wandb

run = wandb.init(
    project="DistVAE", 
    entity="anseunghwan",
    tags=["CTAB-GAN"],
)
#%%
import argparse
import ast

def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v

def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results')
    parser.add_argument('--dataset', type=str, default='covtype', 
                        help='Dataset options: covtype, credit, loan, adult, cabs, kings')

    parser.add_argument("--latent_dim", default=2, type=int,
                        help="size of the noise vector fed to the generator")
    parser.add_argument("--num_channels", default=4, type=int, 
                        help="no. of channels for deciding respective hidden layers of discriminator and generator networks")
    parser.add_argument('--class_dim', default=[32, 32, 32, 32], type=arg_as_list, 
                        help='list containing dimensionality of hidden layers for the classifier network')
    
    # optimization options
    parser.add_argument('--epochs', default=150, type=int,
                        help='no. of epochs to train the model')
    parser.add_argument('--batch_size', default=500, type=int,
                        help='no. of records to be processed in each mini-batch of training')
    parser.add_argument('--lr', default=2e-4, type=float,
                        help='learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float,
                        help='parameter to decide strength of regularization of the network based on constraining l2 norm of weights')
    
    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=False)) # default configuration
    config["cuda"] = torch.cuda.is_available()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    wandb.config.update(config)

    set_random_seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["cuda"]:
        torch.cuda.manual_seed(config["seed"])
    #%%
    """dataset"""
    train_data, transformer, data_dim, target_index, _, _, _, _ = generate_dataset(config)
    #%%
    """model setting"""
    # initializing the sampler object to execute training-by-sampling 
    data_sampler = Sampler(train_data, transformer.output_info)
    # initializing the condvec object to sample conditional vectors during training
    cond_generator = Condvec(train_data, transformer.output_info)

    # obtaining the desired height/width for converting tabular data records to square images for feeding it to discriminator network 		
    sides = [4, 8, 16, 24, 32]
    # the discriminator takes the transformed training data concatenated by the corresponding conditional vectors as input
    col_size_d = data_dim + cond_generator.n_opt
    for i in sides:
        if i * i >= col_size_d:
            dside = i
            break

    # obtaining the desired height/width for generating square images from the generator network that can be converted back to tabular domain 		
    sides = [4, 8, 16, 24, 32]
    col_size_g = data_dim
    for i in sides:
        if i * i >= col_size_g:
            gside = i
            break

    # constructing the generator and discriminator networks
    layers_G = determine_layers_gen(gside, 
                                    config["latent_dim"] + cond_generator.n_opt, 
                                    config["num_channels"])
    layers_D = determine_layers_disc(dside, 
                                    config["num_channels"])
    generator = Generator(layers_G)
    discriminator = Discriminator(layers_D)

    # assigning the respective optimizers for the generator and discriminator networks
    optimizer_params = dict(lr=config["lr"], 
                            betas=(0.5, 0.9), 
                            eps=1e-3, 
                            weight_decay=config["weight_decay"])
    optimizerG = Adam(generator.parameters(), **optimizer_params)
    optimizerD = Adam(discriminator.parameters(), **optimizer_params)

    st_ed = None
    classifier=None
    optimizerC= None

    if target_index != None:
        # obtaining the one-hot-encoding starting and ending positions of the target column in the transformed data
        st_ed = get_st_ed(target_index, transformer.output_info)
        # configuring the classifier network and it's optimizer accordingly 
        classifier = Classifier(data_dim, config["class_dim"], st_ed)
        optimizerC = optim.Adam(classifier.parameters(),**optimizer_params)

    # initializing learnable parameters of the discrimnator and generator networks  
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # initializing the image transformer objects for the generator and discriminator networks for transitioning between image and tabular domain 
    Gtransformer = ImageTransformer(gside)       
    Dtransformer = ImageTransformer(dside)
    #%%
    count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = count_parameters(generator) + count_parameters(discriminator)
    print("Number of Parameters:", num_params)
    
    num_params = count_parameters(classifier)
    print("Number of Parameters:", num_params)
    #%%
    """training"""
    for epoch in range(config["epochs"]):
        logs = train(
            config,
            train_data,
            device,
            generator, discriminator, 
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
            )
        
        print_input = "[epoch {:03d}]".format(epoch + 1)
        print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y)) for x, y in logs.items()])
        print(print_input)
        
        """update log"""
        wandb.log({x : np.mean(y) for x, y in logs.items()})
    #%%
    """model save"""
    torch.save(generator.state_dict(), './assets/CTABGAN_{}.pth'.format(config["dataset"]))
    artifact = wandb.Artifact('CTABGAN_{}'.format(config["dataset"]), 
                            type='model',
                            metadata=config) # description=""
    artifact.add_file('./assets/CTABGAN_{}.pth'.format(config["dataset"]))
    artifact.add_file('./main.py')
    artifact.add_file('./modules/synthesizer.py')
    wandb.log_artifact(artifact)
    #%%
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%