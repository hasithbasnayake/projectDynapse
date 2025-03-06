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

n_train = 10
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

        self.fc = nn.Linear(num_inputs, num_outputs, bias=False)
        self.lif = snn.Leaky(beta=beta, threshold=1, reset_mechanism="zero")

    def forward(self, x):

        mem = self.lif.init_leaky() # Initialize membrane potentials for all neurons to 0, shape = [1, num_outputs]

        spk_rec = [] # Store spike activity over num_steps (time steps), shape = [num_steps, 1, num_outputs]
        mem_rec = [] # Store membrane potentials over num_steps (time steps), shape = shape = [num_steps, 1, num_outputs]

        for step in range(num_steps):
            cur = self.fc(x[step]) # cur represents input current for 40 output neurons across 255 time steps, [255, 1, 40] 
            # print(f"Shape of cur: {cur.shape}")
            spk, mem = self.lif(cur, mem) # mem is the membrane potential, cur is the input current, lif model processes both over time steps using the equation
                                          # mem_new = B*mem_old + cur
                                          # if mem exceeds the threshold, a spike occurs, saved in spk, and the membrane potential is reset
                                          # mem = [1, 40], spk = [1, 40]

            spk_rec.append(spk) 
            mem_rec.append(mem) 

        spk_rec = torch.stack(spk_rec, dim=0)
        mem_rec = torch.stack(mem_rec, dim=0)

        # print(f"Shape of spk_rec: {spk_rec.shape}")
        # print(f"Shape of mem_rec: {mem_rec.shape}")

        return spk_rec, mem_rec

# Training Parameters

train_loader = DataLoader(convON_train, batch_size = 1, shuffle=True)

num_epochs = 1 
counter = 0

# Training Loop

model = Net()
tau_pre = 20.0
tau_post = 20.0
f_pre = lambda x: x
f_post = lambda x: x 

for epoch in range(num_epochs):
    iter_counter = 0
    train_batch = iter(train_loader)

    for data, targets in train_batch: #data.shape = [batch_size = 1, C = 1, H = 28, H = 28]
        # print(f"shape of data: {data.shape}")
        img_spikes = spikegen.latency(data.squeeze(0), num_steps, normalize=True, linear=True)
        img_spikes = img_spikes.view(num_steps, 1, -1)

        # print(f"shape of img_spikes: {img_spikes.shape}")

        spk_rec, mem_rec = model(img_spikes)

        trace_pre = None
        trace_post = None

        for step in range(num_steps):
            in_spike = img_spikes
            out_spike = spk_rec[step]

            trace_pre, trace_post, delta_w = stdp_linear_single_step(
                model.fc,
                in_spike,
                out_spike,
                trace_pre,
                trace_post,
                tau_pre,
                tau_post,
                f_pre,
                f_post,
            )
        
        # # Before weight update: Plot weight matrix as an image
        # plt.imshow(delta_w.squeeze(0).detach().numpy(), cmap='gray', aspect='auto')
        # plt.show()

        model.fc.weight.data += delta_w.squeeze(0)  # Apply weight update

        # # After weight update: Plot weight matrix as an image
        # weight_matrix_after = model.fc.weight.data.numpy()  # Convert to numpy for plotting
        # plt.imshow(weight_matrix_after, cmap='gray', aspect='auto')  # 'gray' for grayscale colormap
        # plt.colorbar()
        # plt.title(f"Model Weights After Update (Epoch {epoch+1})")
        # plt.show()
