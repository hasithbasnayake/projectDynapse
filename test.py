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

# Define a two neuron model 
# Get an image from the convolved dataset, and then print it out as a sanity check
# Pass it into the model and then print out the spike and memory (need to check formatting)

model = Net(num_input=28*28, num_output=10, beta=0.9, threshold=1, reset_mechanism="zero")

data = torch.load("test_sample.pt", weights_only="True")
img, label = data

# convOFF_train.pt is a list of tuples, and each tuple is in the format (img (tensor), label (int) )
# Note that the img tensor is in the dimensions 1 x 28 x 28 where 1 is the amount of channels (C, H, W)
# The image is of a Ankle Boot, label 9 and stored as test_sample.py 
# test_sample.py is a tuple (img (tensor), label (int))

# Create the debugging run 

print(f"Image shape (should be C:1 x H:28 x W:28): {img.shape}")
img = torch.flatten(img, start_dim=1)
print(f"Flattened image shape (should be C:1 x P:784): {img.shape}")
img = spikegen.latency(data=img, num_steps=255, normalize=True, linear=True)
print(f"Spike image shape (should be T:255 x C:1 x P:784): {img.shape}")

for step in range(100):

    # Get spike and membrane recordings
    spk_rec, mem_rec = model(img)

    print(f"spk_rec shape: {spk_rec.shape}")
    print(f"mem_rec shape: {mem_rec.shape}")

    # Iterate over neurons (assuming 10 neurons in output layer)
    for neuron_idx in range(10):  # For each of the 10 neurons

        # Get the spike train for the current neuron across all time steps
        neuron_spk = spk_rec[:, 0, neuron_idx]
        print(f"Spike train of neuron {neuron_idx} across all time steps: {neuron_spk.shape}")
        print(f"Max: {torch.max(neuron_spk)}")
        print(f"Min: {torch.min(neuron_spk)}")

        # Now we check if this neuron spikes at the same time as any other neuron
        for other_neuron_idx in range(neuron_idx + 1, 10):  # Compare with neurons after it to avoid redundant checks
            other_neuron_spk = spk_rec[:, 0, other_neuron_idx]

            # Check if there are time steps where both neurons spike
            # common_spikes = torch.logical_and(neuron_spk, other_neuron_spk)

            main_max = torch.max(neuron_spk)
            sec_max = torch.max(other_neuron_spk)
             
            if main_max == sec_max:
                print(f"ERROR: Neurons {neuron_idx} and {other_neuron_idx} fired at the same time!")
                break

            # if torch.any(common_spikes):
            #     print(f"ERROR: Neurons {neuron_idx} and {other_neuron_idx} fired at the same time!")
            #     print(f"Time steps: {torch.nonzero(common_spikes).squeeze()}")
            #     break
