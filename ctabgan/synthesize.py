#%%
import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import numpy as np
import pandas as pd
import tqdm
from PIL import Image
import matplotlib.pyplot as plt
# plt.switch_backend('agg')

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset

from modules.evaluation import (
    regression_eval,
    classification_eval,
)

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
    tags=["Tabular", "CTAB-GAN", "Inference2"],
)
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--num', type=int, default=10, 
                        help='model version')

    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=False)) # default configuration
    
    dataset = 'loan'
    # dataset = 'adult'
    # dataset = 'covtype'
    
    """model load"""
    artifact = wandb.use_artifact(
        'anseunghwan/CausalDisentangled/CTABGAN_{}:v{}'.format(dataset, config["num"]), type='model')
    for key, item in artifact.metadata.items():
        config[key] = item
    assert dataset == config["dataset"]
    model_dir = artifact.download()
    
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
    generator = Generator(layers_G)
    
    if config["cuda"]:
        model_name = [x for x in os.listdir(model_dir) if x.endswith('pth')][0]
        generator.load_state_dict(
            torch.load(
                model_dir + '/' + model_name))
    else:
        model_name = [x for x in os.listdir(model_dir) if x.endswith('pth')][0]
        generator.load_state_dict(
            torch.load(
                model_dir + '/' + model_name, map_location=torch.device('cpu')))
    
    generator.eval()
    #%%
    if not os.path.exists('./assets/{}'.format(config["dataset"])):
        os.makedirs('./assets/{}'.format(config["dataset"]))
    
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.GraphUtils import GraphUtils
    
    if config["dataset"] == 'loan':
        df = pd.read_csv('./data/Bank_Personal_Loan_Modelling.csv')
        df = df.sample(frac=1, random_state=1).reset_index(drop=True)
        df = df.drop(columns=['ID'])
        continuous = ['CCAvg', 'Mortgage', 'Income', 'Experience', 'Age']
        df = df[continuous]
        
        df_ = (df - df.mean(axis=0)) / df.std(axis=0)
        train = df_.iloc[:4000]
        test = df_.iloc[4000:]
        
        i_test = 'chisq'
        
    elif config["dataset"] == 'adult':
        df = pd.read_csv('./data/adult.csv')
        df = df.sample(frac=1, random_state=1).reset_index(drop=True)
        df = df[(df == '?').sum(axis=1) == 0]
        df['income'] = df['income'].map({'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1})
        continuous = ['income', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
        df = df[continuous]
        
        train = df.iloc[:40000]
        test = df.iloc[40000:]
        
        i_test = 'chisq'
        
    elif config["dataset"] == 'covtype':
        df = pd.read_csv('./data/covtype.csv')
        df = df.sample(frac=1, random_state=5).reset_index(drop=True)
        continuous = [
            'Horizontal_Distance_To_Hydrology', 
            'Vertical_Distance_To_Hydrology',
            'Horizontal_Distance_To_Roadways',
            'Horizontal_Distance_To_Fire_Points',
            'Elevation', 
            'Aspect', 
            'Slope', 
            'Cover_Type']
        df = df[continuous]
        df = df.dropna(axis=0)
        
        train = df.iloc[2000:, ]
        test = df.iloc[:2000, ]
        
        i_test = 'fisherz'
        
    else:
        raise ValueError('Not supported dataset!')
    
    cg = pc(data=train.to_numpy(), 
            alpha=0.05, 
            indep_test=i_test) 
    print(cg.G)
    trainG = cg.G.graph
    
    # visualization
    pdy = GraphUtils.to_pydot(cg.G, labels=df.columns)
    pdy.write_png('./assets/{}/dag_train_{}.png'.format(config["dataset"], config["dataset"]))
    fig = Image.open('./assets/{}/dag_train_{}.png'.format(config["dataset"], config["dataset"]))
    wandb.log({'Baseline DAG (Train)': wandb.Image(fig)})
    #%%
    """Sampling Mechanism"""
    n = len(train)
    
    # column information associated with the transformer fit to the pre-processed training data
    output_info = transformer.output_info
    Gtransformer = ImageTransformer(gside)       
    
    # generating synthetic data in batches accordingly to the total no. required
    steps = n // config["batch_size"] + 1
    data = []
    for _ in range(steps):
        # generating synthetic data using sampled noise and conditional vectors
        noisez = torch.randn(config["batch_size"], config["latent_dim"], device=device)
        condvec = cond_generator.sample(config["batch_size"])
        c = condvec
        c = torch.from_numpy(c).to(device)
        noisez = torch.cat([noisez, c], dim=1)
        noisez =  noisez.view(config["batch_size"], config["latent_dim"]+cond_generator.n_opt,1,1)
        fake = generator(noisez)
        faket = Gtransformer.inverse_transform(fake)
        fakeact = apply_activate(faket,output_info)
        data.append(fakeact.detach().cpu().numpy())

    data = np.concatenate(data, axis=0)
    
    # applying the inverse transform and returning synthetic data in a similar form as the original pre-processed training data
    result = transformer.inverse_transform(data)
    
    sample_df = result[0:n]
    #%%
    """PC algorithm : synthetic dataset"""
    sample_df = pd.DataFrame(sample_df, columns=train.columns)
    cg = pc(data=sample_df.to_numpy(), 
            alpha=0.05, 
            indep_test='fisherz') 
    print(cg.G)
    
    # SHD: https://arxiv.org/pdf/1306.1043.pdf
    sampleSHD = (np.triu(trainG) != np.triu(cg.G.graph)).sum() # unmatch in upper-triangular
    nonzero_idx = np.where(np.triu(cg.G.graph) != 0)
    flag = np.triu(trainG)[nonzero_idx] == np.triu(cg.G.graph)[nonzero_idx]
    nonzero_idx = (nonzero_idx[1][flag], nonzero_idx[0][flag])
    sampleSHD += (np.tril(trainG)[nonzero_idx] != np.tril(cg.G.graph)[nonzero_idx]).sum()
    print('SHD (Sample): {}'.format(sampleSHD))
    wandb.log({'SHD (Sample)': sampleSHD})
    
    # visualization
    pdy = GraphUtils.to_pydot(cg.G, labels=sample_df.columns)
    pdy.write_png('./assets/{}/dag_recon_sample_{}.png'.format(config["dataset"], config["dataset"]))
    fig = Image.open('./assets/{}/dag_recon_sample_{}.png'.format(config["dataset"], config["dataset"]))
    wandb.log({'Reconstructed DAG (Sampled)': wandb.Image(fig)})
    #%%
    """Machine Learning Efficacy"""
    if config["dataset"] == "loan": # regression
        target = 'CCAvg'
        
        # baseline
        print("\nBaseline: Machine Learning Utility in Regression...\n")
        base_r2result = regression_eval(train, test, target)
        wandb.log({'R^2 (Baseline)': np.mean([x[1] for x in base_r2result])})
        
        # Synthetic
        print("\nSynthetic: Machine Learning Utility in Regression...\n")
        r2result = regression_eval(sample_df, test, target)
        wandb.log({'R^2 (Synthetic)': np.mean([x[1] for x in r2result])})
    
    elif config["dataset"] == "adult": # classification
        target = 'income' 
        
        # baseline
        print("\nBaseline: Machine Learning Utility in Classification...\n")
        base_f1result = classification_eval(train, test, target)
        wandb.log({'F1 (Baseline)': np.mean([x[1] for x in base_f1result])})
        
        # Synthetic
        print("\nSynthetic: Machine Learning Utility in Classification...\n")
        f1result = classification_eval(sample_df, test, target)
        wandb.log({'F1 (Synthetic)': np.mean([x[1] for x in f1result])})
        
    elif config["dataset"] == "covtype": # classification
        target = 'Cover_Type'
        
        # baseline
        print("\nBaseline: Machine Learning Utility in Classification...\n")
        base_f1result = classification_eval(train, test, target)
        wandb.log({'F1 (Baseline)': np.mean([x[1] for x in base_f1result])})
        
        # Synthetic
        print("\nSynthetic: Machine Learning Utility in Classification...\n")
        f1result = classification_eval(sample_df, test, target)
        wandb.log({'F1 (Synthetic)': np.mean([x[1] for x in f1result])})
        
    else:
        raise ValueError('Not supported dataset!')
    #%%
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%