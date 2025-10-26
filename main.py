import comet_ml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sklearn
import time
import functools
from sklearn.model_selection import train_test_split
from IPython import display as ipythondisplay
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm
from scipy.io.wavfile import write
import yaml
from src.dataset import Datasets
import random
from src.model import MusicLSTM
from src.train import train
from src.generate import generate
from src.utils import set_seed, save_song_to_abc, abc2wav
import time
# import cuda

set_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Operating in {device} device")

# opening the config file.
with open('music_lstm/config.yaml', 'r') as f:
    config = yaml.safe_load(f)


# Creating the dataset
dataset = Datasets(config["data"]["dataset"],
                   seq_len= config["data"]["seq_len"],
                   percentage= config["data"]["percentage"],
                   device=device)
print(f'Dataset created with {dataset.__len__()} elements')


indices = list(range(dataset.__len__()))

# First split: train/val vs test
train_val_idx, test_idx = train_test_split(
    indices, 
    test_size=config['train']['test_size'], 
    random_state=42, 
    shuffle=True
)

# Second split: train vs val
train_idx, val_idx = train_test_split(
    train_val_idx,
    test_size=config['train']['test_size'],
    random_state=42,
    shuffle=True
)

# Creating Subset
train_set = Subset(dataset, train_idx)
val_set = Subset(dataset, val_idx)
test_set = Subset(dataset, test_idx)


# DataLoader
train_loader = DataLoader(train_set, batch_size=config['train']['batch_size'], shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=config['train']['batch_size'])
test_loader  = DataLoader(test_set,  batch_size=config['train']['batch_size'])

subset_fraction = 0.0001

# --- TRAIN subset --- REMOVE COMMENTS IF YOU WANT FAST CHECKING
#num_train = int(len(train_set) * subset_fraction)
#train_indices = np.random.choice(len(train_set), num_train, replace=False)
#small_train_set = Subset(train_set, train_indices)
#
## --- VALIDATION subset ---
#num_val = int(len(val_set) * subset_fraction)
#val_indices = np.random.choice(len(val_set), num_val, replace=False)
#small_val_set = Subset(val_set, val_indices)
#
## --- TEST subset ---
#num_test = int(len(test_set) * subset_fraction)
#test_indices = np.random.choice(len(test_set), num_test, replace=False)
#small_test_set = Subset(test_set, test_indices)
#
##--- DataLoader per i subset ---
#train_loader = DataLoader(small_train_set, batch_size=config['train']['batch_size'], shuffle=True)
#val_loader   = DataLoader(small_val_set, batch_size=config['train']['batch_size'])
#test_loader  = DataLoader(small_test_set, batch_size=config['train']['batch_size'])


print(f"DataLoaders created: train_size: {train_loader.__len__()}, val_size: {val_loader.__len__()}, test_size: {test_loader.__len__()}")


model = MusicLSTM(vocab_size=len(dataset.vocab), #total number of UNIQUE characters in the dataset 
                  embedding_dim=config['model']['embedding_dim'], # dimension of the word embedding space
                  hidden_dim=config['model']['hidden_dim'], # dimension of the hidden space h_t
                  num_layers=config['model']['num_layers'], # number of LSTM layers
                  dropout=config['model']['dropout'], device=device).to(device=device)

print(f'Model created with {sum(p.numel() for p in model.parameters())} parameters')

if config['comet_ml']['start']:
    COMET_KEY = config['comet_ml']['comet_ml_key'] 
    
    try:
        experiment = comet_ml.Experiment(
        api_key= COMET_KEY,
        project_name= config['comet_ml']['project_name'],
        auto_output_logging= "simple",
        auto_metric_logging= True,
        parse_args= False
    )
        experiment.set_name(config['comet_ml']['experiment_name'])
        experiment.log_parameters({
            "vocab_size": len(dataset.vocab),
            "embedding_dim": 128,
            "hidden_dim": 256,
            "num_layers": 2,
            "dropout": 0.2,
            "batch_size": config['train']['batch_size']
        })
    except Exception as e:
        print(f"Captured error {e}")
        print(f"Type of error: {type(e).__name__}")
        
        
    print('Comet.ml experiment started')
else:
    experiment = None
    print("Comet_ml tracking is disabled")


if config["train"]['flag'] == True:
    print("Training started!")
    
    train(model= model, 
          train_loader= train_loader, 
          val_loader= val_loader, 
          config= config, 
          experiment= experiment if config['comet_ml']['start'] else None,
          device=device)
    
    print('Training finished!')
else:
    print("Training skipped")
    model.load_state_dict(torch.load(f"music_lstm/models/{config['data']['dataset']}_best_model.pth", map_location=device))


generated_sequence = generate(
    model,
    start_sequence=config["eval"]["start_sequence"],
    vocab=dataset.char2idx,
    max_length=config['eval']['max_length'],
    output_path=config['eval']['output_path'],
    experiment=experiment if config['comet_ml']['start'] else None,
    device= device
)
print('Generation finished')

abc_file = save_song_to_abc(
    generated_sequence,
    output_path=config['eval']['output_path'],
    filename=config['saving']['filename']
)

wav_file = abc2wav(abc_file)

if wav_file:
    print(f"Generated audio in: {config['saving']['output_path']}")
else:
    print("Audio has not been generated")

if config['comet_ml']['start']:
    experiment.end()
    print('Experiment ended')  
