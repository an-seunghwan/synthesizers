#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
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

from modules.datasets import generate_dataset
from modules.data_preparation import DataPrep
from modules.transformer import DataTransformer, ImageTransformer
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
    project="CausalDisentangled", 
    entity="anseunghwan",
    tags=["Tabular", "CTAB-GAN"],
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
    parser.add_argument('--dataset', type=str, default='loan', 
                        help='Dataset options: loan, adult, covtype')

    parser.add_argument("--latent_dim", default=3, type=int,
                        help="size of the noise vector fed to the generator")
    parser.add_argument("--num_channels", default=64, type=int, 
                        help="no. of channels for deciding respective hidden layers of discriminator and generator networks")
    parser.add_argument('--class_dim', default=[256, 256, 256, 256], type=arg_as_list, 
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
    df = generate_dataset(config)

    if config["dataset"] == 'loan':
        config["latent_dim"] = 3
        target_index = None
        data_prep = DataPrep(raw_df=df, categorical=[], log=[], mixed={}, 
                            integer=['Mortgage', 'Income', 'Experience', 'Age'])
        transformer = DataTransformer(train_data=data_prep.df, 
                                    categorical_list=data_prep.column_types["categorical"], 
                                    mixed_dict=data_prep.column_types["mixed"])

    elif config["dataset"] == 'adult':
        config["latent_dim"] = 3
        target_col = 'income'
        data_prep = DataPrep(raw_df=df, 
                            categorical=['income'], 
                            log=[], 
                            mixed={'capital-loss':[0.0], 'capital-gain':[0.0]},  
                            integer=['age', 'capital-gain', 'capital-loss','hours-per-week'])
        target_index = data_prep.df.columns.get_loc(target_col)
        transformer = DataTransformer(train_data=data_prep.df, 
                                    categorical_list=data_prep.column_types["categorical"], 
                                    mixed_dict=data_prep.column_types["mixed"])

    elif config["dataset"] == 'covtype':
        config["latent_dim"] = 6
        target_col = 'Cover_Type'
        data_prep = DataPrep(raw_df=df, 
                            categorical=['Cover_Type'], 
                            log=[], 
                            mixed={},  
                            integer=['Horizontal_Distance_To_Hydrology', 
                                    'Vertical_Distance_To_Hydrology',
                                    'Horizontal_Distance_To_Roadways',
                                    'Horizontal_Distance_To_Fire_Points',
                                    'Elevation', 
                                    'Aspect', 
                                    'Slope'])
        target_index = data_prep.df.columns.get_loc(target_col)
        transformer = DataTransformer(train_data=data_prep.df, 
                                    categorical_list=data_prep.column_types["categorical"], 
                                    mixed_dict=data_prep.column_types["mixed"])
        
    else:
        raise ValueError('Not supported dataset!')

    transformer.fit() 
    train_data = transformer.transform(data_prep.df.values)
    # storing column size of the transformed training data
    data_dim = transformer.output_dim
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
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%