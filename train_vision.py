#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from tqdm import tqdm


# In[2]:


import time
import wandb


# In[3]:


# Main Hyperparameters
img_size = 32                           # Dimension of spatial axes of input images
patch_size = 4                          # Patch size
in_channels = 1                         # Dimension of input channels

embed_dim = 128                         # Dimension of embeddings
batch_size = 32                        # Number of batch
epochs = 30                            # Number of epochs
dim_c = 64                             # Dimension of 'code' vector
dim_inter = 192                         # Dimension of intermediate feature vector

ns = 2                                  # Number of 'scripts'
ni = 4                                  # Number of 'function' iterations
nl = 1                                  # Number of LOCs
nf = 5                                  # Number of 'function's
n_cls = 1                               # Number of CLS tokens
n_heads = 4                             # Number of heads per LOC
loc_features = 128                      # Number of features per LOC head

type_inference_depth = 2                # Type Inference MLP depth
type_inference_width = 192              # Type Inference MLP width 
treshold = 0.5                         # Trunctation Parameter
signature_dim = 24                      # Dimension of type_space

attn_prob = 0.0                         # Drop-out probability of ModAttn layer
proj_drop = 0.0                         # Drop-out probability of Projection 
mlp_depth = 4             
number_of_class_mnist = 10                         
# Pretraining Hyperparameters # Dimension of input channels
frozen_function_codes = False           # Required for pretraining
frozen_function_signatures = False      # Required for pretraining

# Optimization Hyperparameters          
beta1 = 0.9                             # Adam Optimizer beta1 parameter
beta2 = 0.999                           # Adam Optimizer beta2 parameter
lr = 1e-3                               # Learning Rate
warmup_steps = 20                       # Scheduler warm up steps


# In[4]:


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)


# # Layers

# In[5]:


from models.basic_layers import NeuralInterpreter, NeuralInterpreter_vision


# # Loader

# In[6]:


from dataset import get_data_loader_mixed, get_data_loader_mnist


# In[7]:


# Parameters for dataset
datasetname = 'digits'
root = '/depo/web490/2022/Cutify/assets/data/'
# root = 'data/'

batch_size = 64
train_loader, valid_loader = get_data_loader_mnist(datasetname, root, batch_size)


# # Train

# In[8]:


# Create Neural Interpreter for vision Task

from models.basic_layers import NeuralInterpreter_vision


model = NeuralInterpreter_vision(ns, ni, nf, embed_dim, dim_c, mlp_depth, n_heads,
                type_inference_width, signature_dim, treshold,  # typematch params
                dim_c, n_classes=10,
                img_size=32, patch_size=4, in_channels=1, n_cls=1,
                attn_prob=0, proj_prob=0, # dropout rate for attention block
              ).to(device)


# In[9]:


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


# In[10]:


print(get_n_params(model))


# In[11]:


from train import *

# Define Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))
criterion = torch.nn.CrossEntropyLoss()
scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps, t_total=epochs)

# log directory => save checkpoints
LOG_DIR = '/depo/web490/2022/Cutify/assets/checkpoints_mnist/'


# In[ ]:


# Initialize wandb

wandb.init(project="Neural-Interpreter", entity="metugan")

# Run train
train(model, train_loader, valid_loader, criterion, optimizer, epochs, scheduler, LOG_DIR, device)


# In[ ]:




