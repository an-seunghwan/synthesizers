#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import numpy as np
import importlib

import torch
from torch.utils.data import DataLoader

from module.datasets import MyDataset, build_dataset
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
    project="HDistVAE", 
    entity="anseunghwan",
    tags=['DAAE'],
)
#%%
import ast
def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v

import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--seed', type=int, default=0, 
                        help='seed for repeatable results')
    parser.add_argument('--dataset', type=str, default='census', 
                        help='Dataset options: mnist, census, survey')
    
    parser.add_argument("--embedding_dim", default=128, type=int, # noise_dim
                        help="the embedding dimension size")
    parser.add_argument("--hidden_dims", default=[128, 128], type=arg_as_list,
                        help="hidden dimensions for autoencoder")
    parser.add_argument("--hidden_dims_disc", default=[128], type=arg_as_list,
                        help="hidden dimensions for discriminator")
    parser.add_argument("--hidden_dims_gen", default=[128, 128], type=arg_as_list,
                        help="hidden dimensions for generator")
    
    parser.add_argument('--epochs', default=500, type=int,
                        help='the number of epochs')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=0.0001, type=float,
                        help='learning rate')
    parser.add_argument('--penalty', default=0.1, type=float,
                        help='WGAN-GP gradient penalty lambda.')
    parser.add_argument('--mc', default=False, type=bool,
                        help='Multi-Categorical setting')
    
    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=False)) # default configuration
    torch.manual_seed(config["seed"])
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    wandb.config.update(config)
    #%%
    out = build_dataset(config)
    dataset = out[0]
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    if config["dataset"] == "mnist": config["p"] = 784
    else: config["p"] = dataset.p
    #%%
    auto_model_module = importlib.import_module('module.model_auto')
    importlib.reload(auto_model_module)
    autoencoder = getattr(auto_model_module, 'AutoEncoder')(
        config, 
        config["hidden_dims"], 
        list(reversed(config["hidden_dims"]))).to(device)
    autoencoder.train()
    #%%
    model_module = importlib.import_module('module.model')
    importlib.reload(model_module)

    generator = getattr(model_module, 'Generator')(
        config["embedding_dim"], 
        config["embedding_dim"], 
        hidden_sizes=config["hidden_dims_gen"],
        bn_decay=0.1).to(device)
    discriminator_x = getattr(model_module, 'Discriminator')(
        config["p"], 
        hidden_sizes=config["hidden_dims_disc"],
        bn_decay=0,
        critic=True).to(device)
    discriminator_z = getattr(model_module, 'Discriminator')(
        config["embedding_dim"], 
        hidden_sizes=config["hidden_dims_disc"],
        bn_decay=0,
        critic=True).to(device)
    generator.train(), discriminator_x.train(), discriminator_z.train()
    #%%
    count_parameters = lambda model: sum(p.numel() for p in model.parameters())
    num_params = count_parameters(autoencoder) \
        + count_parameters(discriminator_x) + count_parameters(discriminator_z) \
            + count_parameters(generator)
    print("Number of Parameters:", num_params)
    wandb.log({'Number of Parameters': num_params})
    #%%
    optimizer_enc = torch.optim.Adam(
        autoencoder.encoder.parameters(), 
        lr=config["lr"],
    )
    optimizer_dec = torch.optim.Adam(
        autoencoder.decoder.parameters(), 
        lr=config["lr"],
    )
    optimizer_Dx = torch.optim.Adam( # critic
        discriminator_x.parameters(), 
        lr=config["lr"],
    )
    optimizer_Dz = torch.optim.Adam( # critic
        discriminator_z.parameters(), 
        lr=config["lr"],
    )
    optimizer_G = torch.optim.Adam(
        generator.parameters(), 
        lr=config["lr"],
    )
    #%%
    train_module = importlib.import_module('module.train')
    importlib.reload(train_module)

    for epoch in range(config["epochs"]):
        logs = train_module.train_DAAE(
            dataloader, autoencoder, discriminator_x, discriminator_z, generator, 
            config, optimizer_enc, optimizer_dec, optimizer_Dx, optimizer_Dz, optimizer_G, epoch, device)
        
        print_input = "[epoch {:04d}]".format(epoch + 1)
        print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y)) for x, y in logs.items()])
        print(print_input)
        
        """update log"""
        wandb.log({x : np.mean(y) for x, y in logs.items()})
    #%%
    """model save"""
    model_name = f'DAAE_{config["dataset"]}'
    torch.save(generator.state_dict(), f'./assets/model/generator_{model_name}.pth')
    torch.save(autoencoder.state_dict(), f'./assets/model/autoencoder_{model_name}.pth')
    artifact = wandb.Artifact(
        model_name, 
        type='model',
        metadata=config) # description=""
    artifact.add_file(f'./assets/model/generator_{model_name}.pth')
    artifact.add_file(f'./assets/model/autoencoder_{model_name}.pth')
    artifact.add_file('./daae.py')
    artifact.add_file('./module/model.py')
    #%%
    wandb.log_artifact(artifact)
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%