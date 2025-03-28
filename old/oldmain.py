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
from old.oldfunc import *
from old.oldmodel import *

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
num_output = 10

# Spiking Dynamics 

beta = .9
threshold = 1
reset_mechanism = "zero"

tau_pre = 20
tau_post = 20

f_pre = 5e-3
f_post = 3.75e-3

# Hyperparameters

num_epochs = 1
num_steps = 255

train_loader = DataLoader(convON_train, batch_size= 1, shuffle=True)

net = Net(num_input, num_output, beta, threshold, reset_mechanism)
trace_pre = None
trace_post = None

dir = "Model Two"


for epoch in range(num_epochs):
    iter_counter = 0

    for img, label in train_loader:
        flat_img = torch.flatten(img, start_dim=1)
        spk_img = spikegen.latency(flat_img, num_steps = num_steps, normalize = True, linear= True)
        print(spk_img.shape)
        spk_rec, mem_rec = net(spk_img)

        trace_pre, trace_post, delta_w = stdp_linear_single_step(
            fc = net.fc1,
            in_spike = spk_img.squeeze(1),
            out_spike = spk_rec.squeeze(1),
            trace_pre = trace_pre,
            trace_post = trace_post,
            tau_pre = tau_pre,
            tau_post = tau_post,
            f_pre = lambda x: f_pre * x,
            f_post = lambda x: f_post * x,
        )

        # print(f"Spikes Input: {spk_img.sum().item()}, Output: {spk_rec.sum().item()}")
        # print(f"trace_pre: {trace_pre.sum().item()}, trace_post: {trace_post.sum().item()}")
        # print(f"Delta_w Min: {delta_w.min().item()}, Max: {delta_w.max().item()}")
        
        with torch.no_grad():
            net.fc1.weight += delta_w

        print(f"Min Δw: {delta_w.min().item()}, Max Δw: {delta_w.max().item()}")


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
    weight_matrix_path = os.path.join(dir, f"weight_matrix_epoch_{epoch}.png")
    plt.savefig(weight_matrix_path)
    plt.close()
    plt.show()

    num_neurons = num_output
    rows = int(np.ceil(num_neurons / 10)) 
    cols = min(10, num_neurons)  

    fig, axes = plt.subplots(rows, cols, figsize=(10, rows * 2))

    weights = net.fc1.weight.data.cpu().numpy()

    for i, ax in enumerate(axes.flat):
        if i < num_neurons:
            img = weights[i].reshape(28, 28)  
            ax.imshow(img, cmap="viridis")
            ax.set_title(f"Neuron {i}")
            ax.axis("off")
        else:
            ax.axis("off")  

    plt.suptitle("Receptive Fields of Output Neurons")
    receptive_field_path = os.path.join(dir, f"receptive_fields_epoch_{epoch}.png")
    plt.savefig(receptive_field_path)
    plt.show()

# torch.save(net.state_dict(), f"{dir}/_n_train:{n_train}_num_output:{num_output}.pth")



# torch.save(net.state_dict(), 'model_weights.pth')
   
# test_net = Net(num_input, num_output, beta, threshold, reset_mechanism)
# test_net.load_state_dict(torch.load('model_weights.pth', weights_only=True))
# print(test_net.eval())
