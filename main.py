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


dim = np.array([6, 6])  # kernel dimensions (in pixels)
ppa = np.array([8, 8])  # pixels per arc (scaling factor)
ang = np.ceil(dim / ppa)  # angular size based on pixels per arc
ctr = (1/3) * dim[0]  # center size as a fraction of kernel size
sur = (2/3) * dim[0]  # surround size as a fraction of kernel size

split_params = (1000,200,2021)
kernel_params = (dim, ppa, ang, ctr, sur)

train_set, test_set = data_preprocessing('FashionMNIST','data', split_params, kernel_params)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True) #data loader loads train_loader with 128 image batch sets from train_set and shuffles it at every epoch
data = iter(train_loader)  #iter, makes train_loader a iterator
data_it, targets_it = next(data) #next(data) receives one batch from train_loader, returning two tensors, data_it and targets_it 

#data_it is a tensor in the shape, [128, C, H, W], so it should be [128,1,28,28]

test_img = data_it[0]
test_img = test_img.squeeze(0)
test_img_spike = spikegen.latency(test_img, num_steps=255, normalize=True, linear=True)

test_img_spike = test_img_spike.numpy().astype(int)

# Initialize dictionary with zero-filled lists
pixel_dict = {(i+1, j+1): [0] * 255 for i in range(28) for j in range(28)}

# Iterate through time steps and update pixel_dict
for t, time_step in enumerate(test_img_spike):  # t is the time step index
    for i in range(28):  # Iterate over rows
        for j in range(28):  # Iterate over columns
            if time_step[i, j] == 1:  # Check if a spike occurred at (i, j) at time t
                pixel_dict[(i+1, j+1)][t] = 1  # Update spike time in the dictionary

# Print first 10 key-value pairs for verification
# for key, value in list(pixel_dict.items()):
#     print(f"{key}: {value}\n")

# print(type(test_img_spike))
# print(test_img_spike.size())
# print(test_img_spike.dim())

# test_img = test_img.numpy()
# image_array_int = (test_img * 255).astype(int)

# for row in image_array_int:
#     print(" ".join(f"{pixel:3}" for pixel in row))  # Align numbers neatly

# plt.imshow(test_img, cmap="gray")  # Convert to NumPy and plot
# plt.title(f"Label: {targets_it[0].item()}")  # Display the corresponding label
# plt.axis("off")  # Hide axes
# plt.show()

lif1 = snn.Leaky(beta=0.8, threshold=1)

summed_list = [0] * 784
for value in pixel_dict.values():
    summed_list = [x + y for x, y in zip(summed_list, value)]
summed_tensor = torch.tensor(summed_list)

print(summed_tensor)

w = 1
cur_in = summed_tensor*w
mem = torch.zeros(1)
spk = torch.zeros(1)
mem_rec = []
spk_rec = []

for step in range(255):
    spk, mem = lif1(cur_in[step], mem)
    mem_rec.append(mem)
    spk_rec.append(spk)

mem_rec = torch.stack(mem_rec)
spk_rec = torch.stack(spk_rec)

def plot_cur_mem_spk(cur, mem, spk, thr_line=1, ylim_max1=0.5, title="LIF Neuron Response"):
    fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

    # Plot input current
    axs[0].plot(cur.numpy(), label="Input Current", color='orange')
    axs[0].set_ylabel("Current (A)")
    axs[0].legend()
    axs[0].set_ylim(0, ylim_max1)

    # Plot membrane potential
    axs[1].plot(mem.numpy(), label="Membrane Potential", color='blue')
    axs[1].axhline(thr_line, linestyle="--", color="black", label="Threshold")
    axs[1].set_ylabel("Membrane Voltage (V)")
    axs[1].legend()

    # Plot spikes as vertical lines
    spike_times = spk.numpy().nonzero()[0]  # Get the indices where spikes occur
    axs[2].vlines(spike_times, 0, 1, colors='black', label="Spikes")  # Draw vertical lines at spike times
    axs[2].set_ylabel("Spikes")
    axs[2].set_xlabel("Time Steps")
    axs[2].legend()

    plt.suptitle(title)
    plt.show()


plot_cur_mem_spk(cur_in, mem_rec, spk_rec, thr_line = 1, ylim_max1 = 2, title="snn.Leaky Neuron Model")

# w=0.21
# cur_in = torch.cat((torch.zeros(10), torch.ones(190)*w), 0)
# mem = torch.zeros(1)
# spk = torch.zeros(1)
# mem_rec = []
# spk_rec = []

# # neuron simulation
# for step in range(200):
#   spk, mem = lif1(cur_in[step], mem)
#   mem_rec.append(mem)
#   spk_rec.append(spk)

# # convert lists to tensors
# mem_rec = torch.stack(mem_rec)
# spk_rec = torch.stack(spk_rec)

# plot_cur_mem_spk(cur_in, mem_rec, spk_rec, thr_line=1, ylim_max1=0.5,
#                  title="snn.Leaky Neuron Model")





# Maybe use channels to see if you can incorporate both the DoG off and DoG on images, so [128,2,28,28].
# You now know the size and that the images are being convolved correctly 

# data_it is a tensor

# spike_data = spikegen.latency(data_it, num_steps=100, tau=5, threshold=0.01)

# # The function in SNNTorch convert_to_time(data, tau, threshold) allows us to convert a feature of intensity X_ij [0,1] into a latency coded response 

# a = torch.Tensor([0.02, 0.5, 0.5])

# print(a)

# a_data = spikegen.latency(a, num_steps = 5, normalize=True, linear=True)

# print(a_data)

# print(data_it)
# print(data_it.dim())
# print(data_it.size())