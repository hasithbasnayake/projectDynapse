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

# Define a two neuron model 
# Get an image from the convolved dataset, and then print it out as a sanity check
# Pass it into the model and then print out the spike and memory (need to check formatting)

model = Net(num_input=28*28, num_output=20, beta=0.9, threshold=20, reset_mechanism="zero")

data = torch.load("test_sample.pt", weights_only="True")
data_2 = torch.load("test_sample_two.pt", weights_only="True")

img, label = data
img_2, label_2 = data_2

# convOFF_train.pt is a list of tuples, and each tuple is in the format (img (tensor), label (int) )
# Note that the img tensor is in the dimensions 1 x 28 x 28 where 1 is the amount of channels (C, H, W)
# The image is of a Ankle Boot, label 9 and stored as test_sample.py 
# test_sample.py is a tuple (img (tensor), label (int))

# Create the debugging run 

save_img = img 
save_img_2 = img_2 



print(f"Image shape (should be C:1 x H:28 x W:28): {img.shape}")
img = torch.flatten(img, start_dim=1)
print(f"Flattened image shape (should be C:1 x P:784): {img.shape}")
img = spikegen.latency(data=img, num_steps=255, normalize=True, linear=True)
print(f"Spike image shape (should be T:255 x C:1 x P:784): {img.shape}")

print(f"Image shape (should be C:1 x H:28 x W:28): {img_2.shape}")
img_2 = torch.flatten(img_2, start_dim=1)
print(f"Flattened image shape (should be C:1 x P:784): {img_2.shape}")
img_2 = spikegen.latency(data=img_2, num_steps=255, normalize=True, linear=True)
print(f"Spike image shape (should be T:255 x C:1 x P:784): {img_2.shape}")

data_list = []  

for iter in range(1):  
    if iter % 2 == 0:  
        data_list.append(img)  
    else:  
        data_list.append(img_2)  

# for cnt, IMAGE in enumerate(data_list):
#     plt.imshow(IMAGE.numpy().squeeze(), cmap="grey")
#     plt.show()

trace_pre = None
trace_post = None

# for step in range(1):
for step, val in enumerate(data_list):

    # Get spike and membrane recordings
    
    spk_rec, mem_rec, random_neuron, stepp = model(val)

    # print(spk_rec.max().item())
    # print(mem_rec.max().item())
    # print(spk_rec[254])

    print(f"spk_rec shape: {spk_rec.shape}")
    print(f"mem_rec shape: {mem_rec.shape}")

    print(f"Shape of in_spike: {val.squeeze(1).shape}")
    print(f"Shape of out_spike: {spk_rec.squeeze(1).shape}")

    # print(f"Shape of weights: {model.fc1.weight.shape}")

    trace_pre, trace_post, delta_w = stdp_linear_single_step(
        fc = model.fc1,
        in_spike = val.squeeze(1),
        out_spike = spk_rec.squeeze(1),
        trace_pre=trace_pre,
        trace_post=trace_post,
        tau_pre = 20,
        tau_post = 20,
        f_pre = lambda x: 5e-3 * x**0.65,
        f_post = lambda x: 3.75e-3 * x**0.05, 
    )

    original_weights = model.fc1.weight.clone()

    print(f"Shape of delta_w: {delta_w.shape}")

    # print(f"Check if its right: {spk_rec.squeeze(1)[:, random_neuron].max().item()}")
    # print(f"Check if its right: {spk_rec.squeeze(1)[step + 1, random_neuron]}")

    # Remove batch dimension to get shape (255, 20)
    spikes = spk_rec.squeeze(1)

    # Loop through each time step and check if the neuron spiked
    for t in range(spikes.shape[0]):  # Iterate over time steps (255)
        if spikes[t, random_neuron] == 1:  # Check if neuron spiked
            print(f"Time step {t}: 1")

    with torch.no_grad():
        model.fc1.weight[random_neuron] += delta_w[random_neuron]
    # with torch.no_grad():
    #     model.fc1.weight += delta_w

    print(f"Random Neuron Min Δw: {delta_w[random_neuron].min().item()}, Max Δw: {delta_w[random_neuron].max().item()}")
    diff =  model.fc1.weight - original_weights
    print(f"Diff Min Δw: {diff.min().item()}, Max Δw: {diff.max().item()}")



    for row in range(model.fc1.weight.shape[0]):
        org_neuron_weight = original_weights[row]
        new_neuron_weight = model.fc1.weight[row]

        if (org_neuron_weight != new_neuron_weight).any():
            print(f"Neuron {row} has updated weights")
            # print(f"Original weights of neuron {row}: {org_neuron_weight[:10]}")
            # print(f"New weights of neuron {row}: {new_neuron_weight[:10]}")

        
    print(f"Membrane potential min: {mem_rec.min().item()}, max: {mem_rec.max().item()}")
    print(f"Min Δw: {delta_w.min().item()}, Max Δw: {delta_w.max().item()}")

    # weights = model.fc1.weight.data.cpu().numpy()
    # plt.imshow(weights[0].reshape(28, 28), cmap="viridis")
    # plt.title("Updated Weights for Neuron 0")
    # plt.axis("off")
    # plt.show()

    print(f"Image: {step}")

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


# plt.imshow(save_img.squeeze(0), cmap="gray") 
# plt.title(f"Label: {label}")  
# plt.axis("off")  
# plt.show()

# plt.imshow(save_img_2.squeeze(0), cmap="gray") 
# plt.title(f"Label: {label_2}")  
# plt.axis("off")  
# plt.show()