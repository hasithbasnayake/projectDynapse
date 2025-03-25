import torch
import math
import torch.nn as nn
import snntorch as snn
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

def stdp(fc, in_spike, out_spike, params):
    '''
    fc: nn.Linear (The model's matrix of weights with the dimensions [# Output Neurons, # Input Neurons]
    in_spike: torch.Tensor (A tensor containing the input spikes with the dimensions [# Timestep, # Input Neurons])
    out_spike: torch.Tensor (A tensor containing the output spikes with the dimension [# Timestep. # Output Neurons])
    params: Python List (A list containing A_plus, A_minus, tau, mu_plus, mu_minus, the hyperparameters of the STDP rule)
    
    Need to finish writing the rest of the parameters here, as well as an example 

    '''

    post_synaptic_neuron_idx = None
    A_plus, A_minus, tau, mu_plus, mu_minus = params

    for time_step in range(out_spike.shape[0]):  # Each row of out_spike represents a timestep, where each element is 0 (no spike) or 1 (spike) for a neuron
        if out_spike[time_step].any():  # If there's a spike for any neuron
            spiking_neuron_indices = torch.where(out_spike[time_step] == 1)[0]  # Get the indices of neurons that spiked
            post_synaptic_neuron_idx = spiking_neuron_indices[torch.randint(0, len(spiking_neuron_indices), (1,)).item()]  # Randomly choose one neuron

            break  # Exit the loop after finding the first spike

    post_synaptic_neuron_spike_train = out_spike[:, post_synaptic_neuron_idx]  # Keep only the spike train of the chosen neuron, discard the rest
    spike_times_of_input_neurons = torch.argmax(in_spike, dim=0)
    # print(f"spike_times_of_input_neurons: {spike_times_of_input_neurons}")
    # DEBUG: Note that if there's no input spike from an input neuron, the column contains just zeros, it will mark it as 0, this might lead to undefined behavior

    # Weight Update

    neuron_weights = fc[post_synaptic_neuron_idx, :] # Get the existing weight matrix for the chosen neuron
    # print(f"neuron_weights: {neuron_weights}")
 
    delta_w = torch.zeros_like(neuron_weights) # Create a tensor to store the weight changes
    # print(f"delta_w: {delta_w}")

    spike_times_of_output_neuron = torch.where(post_synaptic_neuron_spike_train == 1)[0]

    # print(f"spk_indices: {spike_times_of_output_neuron}")

    for synapse, t_pre in enumerate(spike_times_of_input_neurons):
        # synapse is the index, first index is first pixel and so on
        # t_pre is the t of that index (pixel)

        for t_post in spike_times_of_output_neuron:
            delta_t = t_post - t_pre
            w = neuron_weights[synapse]

            if delta_t > 0:
                delta_w[synapse] += A_plus * (1 - w)**mu_plus * math.exp(-abs(delta_t) / tau)

            if delta_t < 0: 
                delta_w[synapse] += -A_minus * w**mu_minus * math.exp(-abs(delta_t) / tau)

    # print(f"delta_w: {delta_w}")

    return post_synaptic_neuron_idx, delta_w

class Net(nn.Module):
    def __init__(self,
                 num_input,
                 num_output,
                 beta,
                 threshold,
                 reset_mechanism):
        super().__init__()

        self.fc1 = nn.Linear(in_features= num_input, out_features = num_output, bias = False)
        self.lif = snn.Leaky(beta = beta, threshold = threshold, reset_mechanism = reset_mechanism)

        torch.nn.init.uniform_(self.fc1.weight, a=0.0, b=1.0)

    def forward(self, x):

        mem = self.lif.init_leaky()
        spk_rec = []
        mem_rec = []

        for step in range(x.shape[0]):  
            cur = self.fc1(x[step]) 
            spk, mem = self.lif(cur, mem)

            spk_rec.append(spk)
            mem_rec.append(mem)

        return torch.stack(spk_rec), torch.stack(mem_rec)

model = Net(num_input=28*28, num_output=2, beta=0.9, threshold=20, reset_mechanism="zero")

data = torch.load("test_sample.pt", weights_only="True")
data_2 = torch.load("test_sample_two.pt", weights_only="True")

img, label = data
img_2, label_2 = data_2

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

for iter in range(100):  
    if iter % 2 == 0:  
        data_list.append(img)  
    else:  
        data_list.append(img_2)  


A_plus = 5e-3
A_minus = 3.75e-3
tau = 200
mu_plus = 0.65
mu_minus = 0.05

params = [A_plus, A_minus, tau, mu_plus, mu_minus]

for step, val in enumerate(data_list):
    
    spk_rec, mem_rec = model(val)

    # print(f"spk_rec shape: {spk_rec.shape}")
    # print(f"mem_rec shape: {mem_rec.shape}")

    # print(f"Shape of in_spike: {val.squeeze(1).shape}")
    # print(f"Shape of out_spike: {spk_rec.squeeze(1).shape}")

    # print(f"Shape of weights: {model.fc1.weight.shape}")

    post_synaptic_neuron_idx, delta_w = stdp(fc=model.fc1.weight, in_spike=val.squeeze(1), out_spike=spk_rec.squeeze(1), params=params)

    with torch.no_grad():
        model.fc1.weight[post_synaptic_neuron_idx] += delta_w
        model.fc1.weight[post_synaptic_neuron_idx].clamp_(0.0, 1.0)


    print(f"Min Δw: {model.fc1.weight.min().item()}, Max Δw: {model.fc1.weight.max().item()}")

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