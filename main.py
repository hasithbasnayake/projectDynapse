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

n_train = 1000
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
num_output = 20

# Spiking Dynamics 

beta = .7
threshold = 20
reset_mechanism = "zero"

# Hyperparameters

num_epochs = 1
num_steps = 255

train_loader = DataLoader(convON_train, batch_size= 1, shuffle=True)

net = Net(num_input, num_output, beta, threshold, reset_mechanism)
trace_pre = None
trace_post = None

for epoch in range(num_epochs):
    iter_counter = 0

    org_weights = net.fc1.weight.data.cpu().numpy()
    # fig, axes = plt.subplots(2, 5, figsize=(10, 5))

    # weights = net.fc1.weight.data.cpu().numpy()

    # for i, ax in enumerate(axes.flat):
    #     img = weights[i].reshape(28, 28)  # Reshape each row into 28x28
    #     ax.imshow(img, cmap="viridis")
    #     ax.set_title(f"Neuron {i}")
    #     ax.axis("off")

    # plt.suptitle("Receptive Fields of Output Neurons")
    # plt.show()
    

    for img, label in train_loader:
        flat_img = torch.flatten(img, start_dim=1)
        spk_img = spikegen.latency(flat_img, num_steps = num_steps, normalize = True, linear= True)
        
        spk_rec, mem_rec = net(spk_img)

        trace_pre, trace_post, delta_w = stdp_linear_single_step(
            fc = net.fc1,
            in_spike = spk_img.squeeze(1),
            out_spike = spk_rec.squeeze(1),
            trace_pre = trace_pre,
            trace_post = trace_post,
            tau_pre = 20,
            tau_post = 20,
            f_pre = lambda x: 5e-3 * x,
            f_post = lambda x: 3.75e-3 * x,
        )

        # print(f"Spikes Input: {spk_img.sum().item()}, Output: {spk_rec.sum().item()}")
        # print(f"trace_pre: {trace_pre.sum().item()}, trace_post: {trace_post.sum().item()}")
        # print(f"Delta_w Min: {delta_w.min().item()}, Max: {delta_w.max().item()}")
        
        with torch.no_grad():
            net.fc1.weight += delta_w

        # print(f"Shape of spk_rec: {spk_rec.shape}")
        # print(f"Shape of mem_rec: {mem_rec.shape}")

        iter_counter += 1
        print(f"Image: {iter_counter}")

    weights = net.fc1.weight.data.cpu().numpy()

    plt.figure(figsize=(12, 5))
    plt.imshow(weights, cmap="viridis", aspect="auto")
    plt.colorbar(label="Weight Value")
    plt.xlabel("Input Neurons (Flattened Pixels)")
    plt.ylabel("Output Neurons")
    plt.title("STDP Weight Matrix Visualization")
    plt.show()

    dif_weights = org_weights - weights

    plt.figure(figsize=(12, 5))
    plt.imshow(dif_weights, cmap="viridis", aspect="auto")
    plt.colorbar(label="Weight Value")
    plt.xlabel("Input Neurons (Flattened Pixels)")
    plt.ylabel("Output Neurons")
    plt.title("STDP Weight Matrix Visualization")
    plt.show()

    num_neurons = num_output
    rows = int(np.ceil(num_neurons / 10))  # Adjust rows based on num_output
    cols = min(10, num_neurons)  # Limit to 10 columns for better visualization

    fig, axes = plt.subplots(rows, cols, figsize=(10, rows * 2))

    weights = net.fc1.weight.data.cpu().numpy()

    for i, ax in enumerate(axes.flat):
        if i < num_neurons:
            img = weights[i].reshape(28, 28)  # Reshape each row into 28x28
            ax.imshow(img, cmap="viridis")
            ax.set_title(f"Neuron {i}")
            ax.axis("off")
        else:
            ax.axis("off")  # Hide extra subplots

    plt.suptitle("Receptive Fields of Output Neurons")
    plt.show()

        
        



