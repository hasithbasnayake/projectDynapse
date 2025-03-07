import torch
import torch.nn as nn
import snntorch as snn
from snntorch import spikegen
import snntorch.spikeplot as splt
import numpy as np
import matplotlib.pyplot as plt
from func import *

class Net(nn.Module):
    def __init__(self,
                 num_input,
                 num_output,
                 beta,
                 threshold,
                 reset_mechanism):
        super().__init__()

        # Initialize layers

        self.fc1 = nn.Linear(in_features= num_input, out_features = num_output, bias = False)
        self.lif = snn.Leaky(beta = beta, threshold = threshold, reset_mechanism = reset_mechanism)

    def forward(self, x):

        mem = self.lif.init_leaky()

        cur = self.fc1(x)
        spk, mem = self.lif(cur, mem)
    
        return spk, mem    

# img = torch.rand(1, 1, 28, 28)
# flat_img = torch.flatten(img, start_dim= 1)
# print(f"Shape of Flattened Image: {flat_img.shape}") # [64,784] 64 images, each with 784 pixels
# spike_img = spikegen.latency(flat_img, num_steps = 255, normalize=True)
# print(f"Shape of Spiking Image: {spike_img.shape}")

# model = Net(784, 1, beta = 0.8, threshold = 1.0, reset_mechanism="zero")

# spk, mem = model(spike_img)

# print(f"{type(spk)}")
# print(f"{type(mem)}")
# print(f"Shape of spk {spk.shape}")
# print(f"Shape of mem {mem.shape}")

# spk_np = spk.detach().numpy().squeeze()
# mem_np = mem.detach().numpy().squeeze()

# # Time axis
# timesteps = range(len(mem_np))

# # Create figure
# fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# # Plot membrane potential
# axs[0].plot(timesteps, mem_np, label="Membrane Potential", color='blue')
# axs[0].axhline(1.0, linestyle="--", color="black", label="Threshold")  # Threshold line
# axs[0].set_ylabel("Membrane Voltage (V)")
# axs[0].legend()
# axs[0].set_title("Membrane Potential Over Time")

# # Plot spikes
# axs[1].plot(timesteps, spk_np, label="Spiking Activity", color='red')
# axs[1].set_ylabel("Spike (0 or 1)")
# axs[1].set_xlabel("Time (ms)")
# axs[1].legend()
# axs[1].set_title("Spike Train Over Time")

# plt.tight_layout()
# plt.show()