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

model = Net(num_input=28*28, num_output=2, beta=0.9, threshold=20, reset_mechanism="zero")

data = torch.load("test_sample.pt", weights_only="True")
img, label = data

# convOFF_train.pt is a list of tuples, and each tuple is in the format (img (tensor), label (int) )
# Note that the img tensor is in the dimensions 1 x 28 x 28 where 1 is the amount of channels (C, H, W)
# The image is of a Ankle Boot, label 9 and stored as test_sample.py 
# test_sample.py is a tuple (img (tensor), label (int))

# Create the debugging run 

save_img = img 
print(f"Image shape (should be C:1 x H:28 x W:28): {img.shape}")
img = torch.flatten(img, start_dim=1)
print(f"Flattened image shape (should be C:1 x P:784): {img.shape}")
img = spikegen.latency(data=img, num_steps=255, normalize=True, linear=True)
print(f"Spike image shape (should be T:255 x C:1 x P:784): {img.shape}")

trace_pre = None
trace_post = None

iter_counter = 0

for step in range(200):

    # Get spike and membrane recordings
    spk_rec, mem_rec = model(img)

    print(spk_rec.max().item())
    print(mem_rec.max().item())
    # print(spk_rec[254])

    print(f"spk_rec shape: {spk_rec.shape}")
    print(f"mem_rec shape: {mem_rec.shape}")

    trace_pre, trace_post, delta_w = stdp_linear_single_step(
        fc = model.fc1,
        in_spike = img.squeeze(1),
        out_spike = spk_rec.squeeze(1),
        trace_pre=trace_pre,
        trace_post=trace_post,
        tau_pre = 5,
        tau_post = 5,
        f_pre = lambda x: 5e-3 * x**0.65,
        f_post = lambda x: 3.75e-3 * x**0.05, 
    )

    with torch.no_grad():
        model.fc1.weight += delta_w
        
    print(f"Membrane potential min: {mem_rec.min().item()}, max: {mem_rec.max().item()}")
    print(f"Min Δw: {delta_w.min().item()}, Max Δw: {delta_w.max().item()}")

    # weights = model.fc1.weight.data.cpu().numpy()
    # plt.imshow(weights[0].reshape(28, 28), cmap="viridis")
    # plt.title("Updated Weights for Neuron 0")
    # plt.axis("off")
    # plt.show()


    iter_counter += 1
    print(f"Image: {iter_counter}")

weights = model.fc1.weight.data.cpu().numpy()

num_output_neurons = weights.shape[0]

rows = int(np.ceil(num_output_neurons / 10))  
cols = min(10, num_output_neurons)  

fig, axes = plt.subplots(rows, cols, figsize=(10, rows * 2))

for i, ax in enumerate(axes.flat):
    if i < num_output_neurons:
        img = weights[i].reshape(28, 28)
        ax.imshow(img, cmap="gray") 
        ax.set_title(f"Neuron {i}")
        ax.axis("off") 
    else:
        ax.axis("off") 

# Title for the entire plot
plt.suptitle("Receptive Fields of Output Neurons")
plt.tight_layout()
plt.show()

save_img = save_img.squeeze(0) 
plt.imshow(save_img, cmap="gray") 
plt.title(f"Label: {label}")  
plt.axis("off")  
plt.show()