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

# Hyperparameters

num_input = 784 # Model
num_output = 10
beta = 0.9
threshold = 20
reset_mechanism = "zero"

iterations = 1 # Training
num_steps = 255

transform = transforms.Compose( # Data
    [transforms.ToTensor()]
)

# Dataloading 

training_set = datasets.MNIST('MNIST', train=True, transform=transform, download=True)

training_loader = torch.utils.data.DataLoader(training_set, batch_size=1, shuffle=True) # print('Training set has {} images'.format(len(training_set)))

dataiter = iter(training_loader) # sample_images(5, training_loader, False)

# Model Instantiation

model = Net(num_input=num_input, num_output=num_output, beta=beta, threshold=threshold, reset_mechanism=reset_mechanism)

# Training Loop 

for step in range(iterations):

    image, label = next(dataiter)

    image = image.squeeze()

    image = torch.flatten(image, start_dim=0) 

    spike_image = spikegen(image=image, num_steps=num_steps) # visualize_spikegen(spike_image)

    spk_rec, mem_rec = model(spike_image) # Pass the [timestep x features] tensor into the model




    # spike_image = spikegen.latency(data = image, num_steps = timesteps, normalize = True, linear= True) # print(spike_image.shape)


    # Write the spikegen function, pass into the model, and then start testing STDP

    


