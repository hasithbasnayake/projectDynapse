import torch 
import torchvision
import numpy as np
import snntorch as snn
import snntorch.spikeplot as splt
import matplotlib.pyplot as plt 
from IPython.display import HTML
from torchvision import datasets
import torchvision.transforms as transforms

from spikegen import *
from visualization import *
from model import * 
from stdp_time import *

# Hyperparameters

num_input = 784 # Model
num_output = 10
beta = 0.9
threshold = 20
reset_mechanism = "zero"

iterations = 10000 # Training
num_steps = 255

A_plus = 5e-3 # STDP
A_minus = 3.75e-3
tau = 200 # Take note of the tau?
mu_plus = 0.65
mu_minus = 0.05

params = [A_plus, A_minus, tau, mu_plus, mu_minus]


transform = transforms.Compose( # Data
    [transforms.ToTensor()]
)

# Dataloading 

training_set = datasets.MNIST('MNIST', train=True, transform=transform, download=True)

training_loader = torch.utils.data.DataLoader(training_set, batch_size=1, shuffle=True) # print('Training set has {} images'.format(len(training_set)))

dataiter = iter(training_loader) # sample_images(5, training_loader, False)

# Model Instantiation

model = Net(num_input=num_input, num_output=num_output, beta=beta, threshold=threshold, reset_mechanism=reset_mechanism)

avg_max_min = [0, 0]

# Training Loop 

for step in range(iterations):

    print(f"Progress: {step / iterations}%", end='\r', flush=True)

    image, label = next(dataiter)

    image = image.squeeze()

    image = torch.flatten(image, start_dim=0) 

    spike_image = spikegen(image=image, num_steps=num_steps) # visualize_spikegen(spike_image)

    spk_rec, mem_rec = model(spike_image) # Pass the [timestep x features] tensor into the model

    out_neuron, delta_w = stdp_time(weight_matrix=model.fc1.weight, in_spike=spike_image, out_spike=spk_rec, params=params)   

    # print(f"Min Δw: {delta_w.min().item()}, Max Δw: {delta_w.max().item()}")

    avg_max_min[0] += delta_w.max().item()
    avg_max_min[1] += delta_w.min().item()

    with torch.no_grad():
        model.fc1.weight[out_neuron] += delta_w
        model.fc1.weight[out_neuron].clamp_(0.0, 1.0) 

    if step % 1000 == 0:

        print(f"Max: {avg_max_min[0] / 1000}, Min: {avg_max_min[1] / 1000}")

        avg_max_min[0] = 0
        avg_max_min[1] = 0

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

    #     plt.figure()

    #     for neuron in range(mem_rec.shape[1]):
    #         neuron_mem_rec = mem_rec[:, neuron]
    #         plt.plot(neuron_mem_rec.detach().numpy(), label=f"Neuron {neuron}")
        
    #     plt.xlabel("Time Step")
    #     plt.ylabel("Membrane Potential")
    #     plt.title(f"Membrane Recording at Step: {step}")
    #     plt.ylim(0, threshold+5)
    #     plt.legend()
    #     plt.show()


