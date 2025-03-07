# .\env\Scripts\activate to activate virtual environment
# pip freeze > requirements.txt You can export a list of all installed packages
# pip install -r requirements.txt to install from a requirements file
# pip list to list all packages
 
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import snntorch as snn
from snntorch import spikegen
from snntorch.functional.stdp_learner import stdp_linear_single_step
import snntorch.spikeplot as splt
import numpy as np
import matplotlib.pyplot as plt
from func import *
from model import *

dim = np.array([6, 6]) 
ppa = np.array([8, 8])
ang = np.ceil(dim / ppa)
ctr = (1/3) * dim[0]
sur = (2/3) * dim[0]

n_train = 100
n_test = 200
r_seed = 2021

split_params = (n_train, n_test, r_seed)
kernel_params = (dim, ppa, ang, ctr, sur)

convON_train, convOFF_train = data_preprocessing(
    'FashionMNIST', 
    'data', 
    split_params, 
    kernel_params)

# Network Architecture

num_input = 28*28
num_output = 10

# Spiking Dynamics 

beta = 0.8
threshold = 1
reset_mechanism = "zero"

# Hyperparameters

num_epochs = 2
num_steps = 255

train_loader = DataLoader(convON_train, batch_size= 1, shuffle=True)

net = Net(num_input, num_output, beta, threshold, reset_mechanism)

for epoch in range(num_epochs):
    iter_counter = 0
    
    for img, label in train_loader:
        flat_img = torch.flatten(img, start_dim=1)
        spk_img = spikegen.latency(flat_img, num_steps = num_steps, normalize = True, linear= True)

        spk_rec, mem_rec = net(spk_img)
        iter_counter += 1
        print(f"Image: {iter_counter}")

    
        



