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
    project="Synthetic(High)", 
    entity="anseunghwan",
    # tags=[''],
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
    parser.add_argument('--model', type=str, default='MC-ARAE')
    parser.add_argument('--dataset', type=str, default='census', 
                        help='Dataset options: mnist, census, survey')
    
    parser.add_argument("--embedding_dim", default=128, type=int, # noise_dim
                        help="the embedding dimension size")
    parser.add_argument("--hidden_dims", default=[100, 100], type=arg_as_list,
                        help="hidden dimensions for autoencoder")
    parser.add_argument("--hidden_dims_disc", default=[100], type=arg_as_list,
                        help="hidden dimensions for discriminator")
    parser.add_argument("--hidden_dims_gen", default=[100, 100, 100], type=arg_as_list,
                        help="hidden dimensions for generator")
    
    parser.add_argument('--epochs', default=1000, type=int,
                        help='the number of epochs')
    parser.add_argument('--batch_size', default=1024, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=1e-5, type=float,
                        help='learning rate')
    parser.add_argument('--l2reg', default=0, type=float,
                        help='L2 regularization: weight decay')
    parser.add_argument('--tau', default=0.666, type=float,
                        help='temperature in Gumbel-Softmax')
    
    parser.add_argument('--noise_radius', default=0.2, type=float,
                        help='Gaussian noise standard deviation for the latent code (autoencoder regularization).')
    parser.add_argument('--noise_anneal', default=0.995, type=float,
                        help='Anneal the noise radius by this value after every epoch.')
    parser.add_argument('--penalty', default=0.1, type=float,
                        help='WGAN-GP gradient penalty lambda.')
    parser.add_argument('--clipping', default=0.01, type=float,
                        help='weight-clipping of critic network.')
    parser.add_argument('--mc', default=True, type=bool,
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
    OutputInfo_list = out[3]

    if config["dataset"] == "mnist": config["p"] = 784
    else: config["p"] = dataset.p
    #%%
    auto_model_module = importlib.import_module('module.model_auto')
    importlib.reload(auto_model_module)
    autoencoder = getattr(auto_model_module, 'AutoEncoder')(
        config, 
        config["hidden_dims"], 
        list(reversed(config["hidden_dims"])), 
        OutputInfo_list=OutputInfo_list).to(device)
    autoencoder.train()
    #%%
    model_module = importlib.import_module('module.model')
    importlib.reload(model_module)

    generator = getattr(model_module, 'Generator')(
        config["embedding_dim"], 
        config["embedding_dim"], 
        hidden_sizes=config["hidden_dims_gen"],
        bn_decay=0.1).to(device)
    discriminator = getattr(model_module, 'Discriminator')(
        config["embedding_dim"], 
        hidden_sizes=config["hidden_dims_disc"],
        bn_decay=0,
        critic=True).to(device)
    generator.train(), discriminator.train()
    #%%
    count_parameters = lambda model: sum(p.numel() for p in model.parameters())
    num_params = count_parameters(autoencoder.decoder) + count_parameters(generator)
    print(f"Number of Parameters: {num_params / 1000:.1f}K")
    wandb.log({'Number of Parameters': num_params / 1000})
    #%%
    optimizer_AE = torch.optim.Adam(
        autoencoder.parameters(), 
        lr=config["lr"] * 50,
        weight_decay=config["l2reg"]
    )
    optimizer_D = torch.optim.Adam( # critic
        discriminator.parameters(), 
        lr=config["lr"],
        weight_decay=config["l2reg"]
    )
    optimizer_G = torch.optim.Adam(
        generator.parameters(), 
        lr=config["lr"] * 5,
        weight_decay=config["l2reg"]
    )
    #%%
    train_module = importlib.import_module('module.train')
    importlib.reload(train_module)

    for epoch in range(config["epochs"]):
        logs = train_module.train_ARAE(
            dataloader, autoencoder, discriminator, generator, config, optimizer_AE, optimizer_D, optimizer_G, epoch, device)
        
        print_input = "[epoch {:04d}]".format(epoch + 1)
        print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y)) for x, y in logs.items()])
        print(print_input)
        
        """update log"""
        wandb.log({x : np.mean(y) for x, y in logs.items()})
    #%%
    """model save"""
    model_name = f'mc_ARAE_{config["dataset"]}'
    torch.save(generator.state_dict(), f'./assets/model/generator_{model_name}.pth')
    torch.save(autoencoder.state_dict(), f'./assets/model/autoencoder_{model_name}.pth')
    artifact = wandb.Artifact(
        model_name, 
        type='model',
        metadata=config) # description=""
    artifact.add_file(f'./assets/model/generator_{model_name}.pth')
    artifact.add_file(f'./assets/model/autoencoder_{model_name}.pth')
    artifact.add_file('./mc_arae.py')
    artifact.add_file('./module/model.py')
    #%%
    wandb.log_artifact(artifact)
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%