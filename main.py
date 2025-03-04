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

# Network Architecture 
num_inputs = 784
num_outputs = 40

# Temporal Dynamics 
num_steps = 255
beta= 0.95

# Define Network

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Layer Initialization (1 layer)

        self.fc = nn.Linear(num_inputs, num_outputs, biase=False)
        self.lif = snn.Leaky(beta=beta, threshold=1, reset_mechanism="zero")

    def forward(self, x):

        mem = self.lif.init_leaky() # Initialize membrane potentials for all neurons to 0, shape = [1, num_outputs]

        spk_rec = [] # Store spike activity over num_steps (time steps), shape = [num_steps, 1, num_outputs]
        mem_rec = [] # Store membrane potentials over num_steps (time steps), shape = shape = [num_steps, 1, num_outputs]

        for step in range(num_steps):
            cur = self.fc(x) # Computes input current by weighting input across all weights 
            spk, mem = self.lif(cur, mem) # Updates mem using the equation defined by leaky-integrate-and-fire neurons, produces a spike if mem for a neuron is above threshold

            spk_rec.append(spk) 
            mem_rec.append(mem) 

        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)

# Set up data loaders

train_loader = DataLoader(convON_train, batch_size = 128, shuffle=True)
data = iter(train_loader)
data_it, targets_it = next(data) #[batch_size = 128, C = 1, H = 28, H = 28]

# Training Parameters

num_epochs = 1 
counter = 0

# Training Loop



fc1 = snn.Linear()

# Neuron Model

l1 = snn.Leaky(beta=0.8, threshold=1, reset_mechanism="zero")

num_steps = 255
l1_w = 0.31
l1_cur_in = convr(data_it, num_steps) * l1_w
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

splt.traces(mem1_rec, spk=spk1_rec.squeeze(1))
fig = plt.gcf()
fig.set_size_inches(8, 6)
plt.show()

# Multi-Neuron Model





# Now implement the full training loop with multiple neurons. STDP is an unsupervised learning rule so
# you don't need to keep track of accuracy or anything like that, just keep running each iteration till a neuron spikes
# then move on to the next image 
# and continue for a certain amount of time till the network is "trained"