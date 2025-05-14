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

# Hyperparameters

num_input = 784 # Model
num_output = [200,500,750]
beta = 0.9
threshold = 20
reset_mechanism = "zero"

# iterations = 20000 # Training
training_set_size = 1000
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

training_subset = torch.utils.data.Subset(training_set, range(training_set_size)) # print(len(training_subset))

torch.save(training_subset, "training_subset.pt")

training_loader = torch.utils.data.DataLoader(training_subset, batch_size=1, shuffle=True) # training_loader = torch.utils.data.DataLoader(training_set, batch_size=1, shuffle=True) # print('Training set has {} images'.format(len(training_set)))

# Training Loop 

for index, N_output in enumerate(num_output):

    increase = 0
    epochs = 0

    if N_output == 750:
        model = Net(num_input=num_input, num_output=N_output, beta=beta, threshold=threshold, reset_mechanism=reset_mechanism)
        model.load_state_dict(torch.load('newrun/model_750/model_190.pt'))
        epochs = 750
        increase = 191

    print(f"Starting training for model_{N_output}")
    print(f"Training for epochs: {epochs}")
    print(f"Starting from epoch: {increase}")

    start_time = time.time()

    max_every_epoch = []
    min_every_epoch = []

    for epoch in range(epochs):

        sum_max_delta = 0.0
        sum_min_delta = 0.0

        dataiter = iter(training_loader) 

        for step in range(training_set_size):

            if step % 100 == 0:
                print(f"Epoch {epoch}: {step} / {training_set_size}", end='\r', flush=True)

            image, label = next(dataiter)

            image = image.squeeze()

            image = torch.flatten(image, start_dim=0) 

            spike_image = spikegen(image=image, num_steps=num_steps) 

            spk_rec, mem_rec = model(spike_image) 

            out_neuron, delta_w = stdp_time(weight_matrix=model.fc1.weight, in_spike=spike_image, out_spike=spk_rec, params=params)   

            # print(f"Min Δw: {delta_w.min().item()}, Max Δw: {delta_w.max().item()}")

            sum_max_delta += delta_w.max().item()
            sum_min_delta += delta_w.min().item()
            
            with torch.no_grad():
                model.fc1.weight[out_neuron] += delta_w
                model.fc1.weight[out_neuron].clamp_(0.0, 1.0) 

        sum_max_delta = sum_max_delta / training_set_size
        sum_min_delta = sum_min_delta / training_set_size

        max_every_epoch.append(sum_max_delta)
        min_every_epoch.append(sum_min_delta)

        torch.save(model.state_dict(), f"newrun/model_{N_output}/model_{epoch + increase}.pt")

        # model_weights = model.fc1.weight
        # num_neurons = model_weights.shape[0]
        # cols = int(np.ceil(np.sqrt(num_neurons)))
        # rows = int(np.ceil(num_neurons / cols))
        
        # fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        # axes = np.array(axes)  

        # for i in range(num_neurons):
        #     neuron_weights = model_weights[i].reshape(28, 28).detach().numpy()
        #     ax = axes[i // cols, i % cols]
        #     im = ax.imshow(neuron_weights, cmap="gray")
        #     ax.set_title(f"Neuron {i}", fontsize=8)
        #     ax.axis("off")

        # plt.tight_layout()
        # plt.savefig(f"newrun/model_{N_output}/modelfig_{epoch}.png", dpi=25)
        # plt.close()

    end_time = time.time()
    print(f"Training completed for model {N_output} in {end_time - start_time:.2f} seconds")

    torch.save(max_every_epoch, f"newrun/model_{N_output}/max_every_epoch_model_{N_output}.pt")
    torch.save(min_every_epoch, f"newrun/model_{N_output}/min_every_epoch_model_{N_output}.pt")
    