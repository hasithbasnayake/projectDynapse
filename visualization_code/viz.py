import torch 
import time
import torchvision
import numpy as np
import snntorch as snn
import snntorch.spikeplot as splt
import matplotlib.pyplot as plt 
from IPython.display import HTML
from torchvision import datasets
import torchvision.transforms as transforms

from visualization_code.spikegen import *
from visualization_code.visualization import *
from model.model import * 
from training.stdp_time import *


num_input = 784 # Model
num_output = 5
beta = 0.9
threshold = 20
reset_mechanism = "zero"

A_plus = 5e-3 # STDP
A_minus = 3.75e-3
tau = 200 # Take note of the tau?
mu_plus = 0.65
mu_minus = 0.05

params = [A_plus, A_minus, tau, mu_plus, mu_minus]

model = Net(num_input=num_input, num_output=num_output, beta=beta, threshold=threshold, reset_mechanism=reset_mechanism)
model.load_state_dict(torch.load('newrun/model_5/model_6.pt'))

model_weights = model.fc1.weight
num_neurons = model_weights.shape[0]
cols = int(np.ceil(np.sqrt(num_neurons)))
rows = int(np.ceil(num_neurons / cols))

fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
axes = np.array(axes)  

for i in range(num_neurons):
    neuron_weights = model_weights[i].reshape(28, 28).detach().numpy()
    ax = axes[i // cols, i % cols]
    im = ax.imshow(neuron_weights, cmap="gray")
    ax.set_title(f"Neuron {i}", fontsize=8)
    ax.axis("off")

plt.tight_layout()
plt.show()

transform = transforms.Compose( # Data
    [transforms.ToTensor()]
)

training_set = datasets.MNIST('MNIST', train=True, transform=transform, download=True)

training_subset = torch.utils.data.Subset(training_set, range(1000)) # print(len(training_subset))

torch.save(training_subset, "training_subset.pt")

training_loader = torch.utils.data.DataLoader(training_subset, batch_size=1, shuffle=True) # training_loader = torch.utils.data.DataLoader(training_set, batch_size=1, shuffle=True) # print('Training set has {} images'.format(len(training_set)))

dataiter = iter(training_loader)

max_list = []
min_list = []

for step in range(1000):

    image, label = next(dataiter)

    image = image.squeeze()

    image = torch.flatten(image, start_dim=0) 

    spike_image = spikegen(image=image, num_steps=255) # visualize_spikegen(spike_image)

    spk_rec, mem_rec = model(spike_image) # Pass the [timestep x features] tensor into the model

    out_neuron, delta_w = stdp_time(weight_matrix=model.fc1.weight, in_spike=spike_image, out_spike=spk_rec, params=params)   

    # print(f"Min Δw: {delta_w.min().item()}, Max Δw: {delta_w.max().item()}")
        
    with torch.no_grad():
        model.fc1.weight[out_neuron] += delta_w
        model.fc1.weight[out_neuron].clamp_(0.0, 1.0) 

    max_list.append(delta_w.max().item())
    min_list.append(delta_w.min().item())

print(np.mean(max_list))
print(np.mean(min_list))