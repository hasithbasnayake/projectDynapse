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

num_input = 28 * 28
num_output = 40
beta = 0.8
threshold = 1
num_steps = 255

# input_image = torch.flatten(torch.randn(24,1,28,28), start_dim=1)
# print(f"Image Shape: {input_image.shape}")

# ex1 = nn.Linear(28*28, 10, bias=False)
# print(f"Shape of Weights: {ex1.weight.shape}")

# output = ex1(input_image)

# print(f"Shape of Output: {output.shape}")

# Input Image Shape, [1,784]
# Weight Shape, [10, 784]
# Output is going to be [1,10] or [10]

img = torch.rand(1, 1, 28, 28)
flat_img = torch.flatten(img, start_dim= 1)
print(f"Shape of Flattened Image: {flat_img.shape}") # [64,784] 64 images, each with 784 pixels
spike_img = spikegen.latency(flat_img, num_steps = 255, normalize=True)
print(f"Shape of Spiking Image: {spike_img.shape}")

fc1 = nn.Linear(28*28, 10, bias=False)
lif = snn.Leaky(beta=0.8, threshold=.6, reset_mechanism="zero")
mem = lif.init_leaky()

spk_rec = []
mem_rec = []

# cur = fc1(spike_img)
# print(f"Shape of cur: {cur.shape}")

for step in range(1):
    cur = fc1(spike_img)
    print(f"Shape of cur: {cur.shape}")

    spk, mem = lif(cur, mem)
    print(f"Shape of spk: {spk.shape}")
    print(f"Shape of mem: {mem.shape}")

    spk_rec.append(spk)
    mem_rec.append(mem)
    print(f"Shape of spk_rec: {len(spk_rec)}")
    print(f"Shape of mem_rec: {len(mem_rec)}")
    

mem_rec_tensor = torch.stack(mem_rec, dim=0)  # Shape: [255, 10]

x = torch.squeeze(mem_rec_tensor)

list_of_neuron_mem = []

for neuron in range(x.shape[1]):  
    neuron_mem = x[:, neuron]
    list_of_neuron_mem.append(neuron_mem)

plt.figure(figsize=(10, 6))

for neuron_idx, neuron_mem in enumerate(list_of_neuron_mem):
    plt.plot(neuron_mem.detach().numpy(), label=f'Neuron {neuron_idx+1}')

plt.axhline(1, linestyle="--", color="black", label="Threshold")

plt.xlabel("Time Step")
plt.ylabel("Membrane Potential (V)")
plt.title("Membrane Potential for All Neurons")
plt.legend()

plt.ylim(0, .6)

plt.show()
