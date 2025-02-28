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
import snntorch.spikeplot as splt
import numpy as np
import matplotlib.pyplot as plt
from func import *

dim = np.array([6, 6]) 
ppa = np.array([8, 8])
ang = np.ceil(dim / ppa)
ctr = (1/3) * dim[0]
sur = (2/3) * dim[0]

n_train = 1000
n_test = 200
r_seed = 2021

split_params = (n_train, n_test, r_seed)
kernel_params = (dim, ppa, ang, ctr, sur)

convON_train, convOFF_train = data_preprocessing('FashionMNIST', 'data', split_params, kernel_params)

# Set up data loaders

train_loader = DataLoader(convON_train, batch_size = 128, shuffle=True)
data = iter(train_loader)
data_it, targets_it = next(data) #[batch_size = 128, C = 1, H = 28, H = 28]

# Set up a neuron to take in input from 784 neurons 
# Set up a network model with one neuron to take in input from 784 neurons 
# The goal is to compare the output of both these after one run to see how they compare 

# Neuron Model

l1 = snn.Leaky(beta=0.8, threshold=1, reset_mechanism="zero")

num_steps = 255
l1_w = 0.31
l1_cur_in = convr(data_it, num_steps) * 0.31
l1_mem = torch.zeros(1)
l1_spk = torch.zeros(1)
l1_mem_rec = []
l1_spk_rec = []

for step in range(num_steps):
    l1_spk, l1_mem = l1(l1_cur_in[step], l1_mem)
    l1_mem_rec.append(l1_mem)
    l1_spk_rec.append(l1_spk)

l1_mem_rec = torch.stack(l1_mem_rec)
l1_spk_rec = torch.stack(l1_spk_rec)

# Network Model

num_inputs = 784
num_outputs = 1

fc1 = nn.Linear(num_inputs, num_outputs, bias=False)
lif1 = snn.Leaky(beta=0.8, threshold=1, reset_mechanism="zero")

with torch.no_grad():
    fc1.weight.fill_(.31)

mem1 = lif1.init_leaky()

mem1_rec = []
spk1_rec = []

img_spikes = spikegen.latency(data_it[0].squeeze(0), 255, normalize=True, linear=True)
img_spikes = img_spikes.view(255, 1, -1)

for step in range(num_steps):
    cur1 = fc1(img_spikes[step])
    spk1, mem1 = lif1(cur1, mem1)
    
    mem1_rec.append(mem1)
    spk1_rec.append(spk1)

mem1_rec = torch.stack(mem1_rec)
spk1_rec = torch.stack(spk1_rec)

mem1_rec = mem1_rec.squeeze().detach()

plot_cur_mem_spk(l1_cur_in, l1_mem_rec, l1_spk_rec, mem1_rec, thr_line = 1, ylim_max1 = 2, title="snn.Leaky Neuron Model")
